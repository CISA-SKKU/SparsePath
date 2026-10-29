#!/usr/bin/env python3
"""
Distributed Operation Engine for DiskANN graph index (Colocation-Native).

Performs insert/delete/consolidate/search across multiple DPU nodes using RDMA.
Each node owns one partition and uses RDMA WRITE for cross-partition
reverse edges and RDMA READ for cross-partition neighbor/vector access.

All data access goes through the colocation layout — there is no Python
GraphEngine. The C++ ColocationStore (mmap'd .coloc file) is the single
source of truth for vectors, neighbor lists, and VectorIDs.

Architecture:
  - UpdateContext: shared state + primitive operations (extracted from old
    DistributedUpdateWorker). Holds rdma_store, slot_manager, tag_to_local,
    and all data access / graph mutation methods.
  - Operation: base class for discrete operations (InsertOp, DeleteOp,
    ConsolidateOp, SearchOp, CheckpointOp).
  - OperationEngine: dispatches operations from a workload definition,
    managing barriers and statistics.

Used by node_worker.py as the 'distributed_engine' stage handler.
(Previously named distributed_engine.py; renamed to reflect broader scope.)
"""

import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from pipeline.danns_types import OperationHook
from pipeline.engine_context import UpdateContext
from pipeline.engine_report import generate_engine_report
from pipeline.slot_manager import INVALID_LOC
from pipeline.workload import parse_workload_specs
from pipeline.operation_base import Operation
from pipeline.ops.checkpoint import CheckpointOp
from pipeline.ops.concurrent import ConcurrentOp
from pipeline.ops.consolidate import ConsolidateOp
from pipeline.ops.delete import DeleteOp
from pipeline.ops.insert import InsertOp
from pipeline.ops.search import SearchOp


# ======================================================================
# Operation Registry
# ======================================================================

_OPERATION_REGISTRY: Dict[str, type] = {
    'insert': InsertOp,
    'delete': DeleteOp,
    'consolidate': ConsolidateOp,
    'concurrent': ConcurrentOp,
    'search': SearchOp,
    'checkpoint': CheckpointOp,
}


# ======================================================================
# OperationEngine — main entry point
# ======================================================================

class OperationEngine:
    """Dispatches operations from a workload definition.

    Replaces the old DistributedUpdateWorker.run() loop. Each operation
    gets exactly 2 barriers (start + end) — no per-batch barriers.

    Usage from node_worker.py:
        engine = OperationEngine(node_worker, config)
        engine.load()
        engine.execute_workload()
        engine.save()
    """

    def __init__(self, node_worker, config: dict,
                 hooks: Optional[List[OperationHook]] = None,
                 timestamp: Optional[str] = None):
        self.ctx = UpdateContext(node_worker, config, hooks=hooks)
        self.workload: List[Operation] = []
        self.op_history: List[dict] = []

        # Resolve run directory. Split-layout configs write to
        # results/<dataset>/<experiment>_<timestamp>; legacy configs keep their
        # historical output.base_dir layout.
        from pipeline.experiment_paths import run_dir as build_run_dir
        self.run_dir = build_run_dir(
            config, "distributed_engine", timestamp or "latest")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ctx.run_dir = self.run_dir

    def load(self):
        """Load DiskANN index and tag_to_local mappings."""
        self.ctx.log("OperationEngine: loading index...")
        self.ctx._load_diskann_index()
        self.ctx._load_tag_to_local()

        # Initialize sliding window cursor from current max tag in tag_to_local.
        # If graph already contains prior update checkpoints, max_tag may
        # exceed initial_size. Using initial_size would overlap.
        max_tag = 0
        for p, t2l in self.ctx.tag_to_local.items():
            if len(t2l) > 0:
                max_tag = max(max_tag, len(t2l) - 1)
        self.ctx._next_insert_gid = max_tag  # next gid = max_tag (tag = max_tag + 1)
        total_vecs = len(self.ctx.vectors) if self.ctx.vectors is not None else getattr(self.ctx, 'total_vectors', 0)
        self.ctx.log(f"OperationEngine: insert cursor initialized at gid={max_tag} "
                     f"(max_tag={max_tag}, total_vectors={total_vecs})")

        self.ctx.sync.barrier("engine-loaded")

        # Parse workload after context is ready
        self.workload = self._parse_workload()
        self.ctx.log(f"OperationEngine: {len(self.workload)} operations in workload")

    def execute_workload(self):
        """Execute all operations in the workload sequentially."""
        self.ctx.log(f"OperationEngine: starting {len(self.workload)} operations...")

        for seq, op in enumerate(self.workload):
            self.ctx.log(f"--- Operation {seq}: {op.name} ---")
            self._dispatch(seq, op)

        self.ctx.log("OperationEngine: workload complete")

    def _dispatch(self, seq: int, op: Operation):
        """Dispatch a single operation with start/end barriers."""
        op.stats.op_name = op.name
        op.stats.seq = seq
        self.ctx._current_op_seq = seq  # expose seq to ctx for trace filenames

        # Start barrier
        self.ctx.sync.barrier(f"op-{seq}-start")

        # Prepare + broadcast (if needed)
        payload = None
        if op.needs_broadcast:
            if self.ctx.is_coordinator:
                payload = op.prepare(self.ctx)
                data = pickle.dumps(payload)
                self.ctx.sync.broadcast_bytes(data)
            else:
                data = self.ctx.sync.recv_bytes()
                payload = pickle.loads(data)

        # Reset RDMA counters before execution
        self.ctx.rdma_counters.reset()

        # Execute
        t0 = time.time()
        try:
            op.execute(self.ctx, payload)
            op.stats.status = "success"
        except Exception as e:
            op.stats.status = "failed"
            op.stats.error = str(e)
            self.ctx.log(f"[ERROR] Operation {op.name}#{seq} failed: {e}")
            import traceback
            traceback.print_exc()

        op.stats.wall_time = time.time() - t0

        # Snapshot RDMA counters
        op.stats.custom['rdma_counts'] = self.ctx.rdma_counters.to_dict()

        # End barrier (with status to avoid deadlock on failure)
        all_ok = self.ctx.sync.barrier_with_status(
            f"op-{seq}-end", op.stats.status == "success")

        if not all_ok:
            self.ctx.log(f"[WARN] Operation {op.name}#{seq}: some nodes reported failure")
            if op.stats.status == "success":
                op.stats.status = "partial_failure"

        # Each node writes its own stats file (no sync needed)
        node_stats = op.stats.to_dict()
        node_stats['node_id'] = self.ctx.node_id
        node_stats['partition'] = self.ctx.my_partition
        if self.ctx.slot_manager is not None:
            node_stats['active_points'] = self.ctx.slot_manager.active_count

        stats_dir = self.run_dir / "node_stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        stats_file = stats_dir / f"node{self.ctx.node_id}_seq{seq}.json"
        with open(stats_file, 'w') as f:
            json.dump(node_stats, f, default=str)

        # Still gather to coordinator for backward compat (op_history.json)
        stats_bytes = pickle.dumps(node_stats)
        if self.ctx.is_coordinator:
            gathered = self.ctx.sync.gather_bytes()
            per_node = {self.ctx.node_id: node_stats}
            for peer_id, peer_data in gathered.items():
                per_node[peer_id] = pickle.loads(peer_data)
            op.stats.custom['per_node'] = per_node
        else:
            self.ctx.sync.send_to_coordinator(stats_bytes)

        # Sync after insert/delete: lightweight incremental sync only.
        # Delete: all nodes already received full delete assignments via broadcast
        #   in prepare() — tag_to_local is unchanged. No sync needed.
        # Insert: only sync new (partition, local_loc, tag) mappings.
        if op.name in ('insert', 'delete', 'consolidate', 'concurrent'):
            t_sync = time.time()
            my_part = self.ctx.my_partition

            if op.name == 'delete':
                # Delete tags already broadcast in prepare() — all nodes have them.
                # tag_to_local unchanged by delete. No TCP sync needed.
                pass

            elif op.name == 'insert':
                # Only sync new local_to_tag mappings (incremental, lightweight).
                my_new_l2t = getattr(self.ctx, '_new_local_to_tag', [])
                sync_payload = pickle.dumps(my_new_l2t)

                if self.ctx.is_coordinator:
                    gathered = self.ctx.sync.gather_bytes()
                    all_new_l2t = list(my_new_l2t)
                    for peer_id, peer_data in gathered.items():
                        all_new_l2t.extend(pickle.loads(peer_data))
                    # Apply to coordinator's tag_to_local. Bucket by partition
                    # and pre-extend once per partition to avoid O(N*L) concat.
                    _by_part: Dict[int, List[Tuple[int, int]]] = {}
                    for part, ll, tag in all_new_l2t:
                        _by_part.setdefault(part, []).append((tag, ll))
                    for part, pairs in _by_part.items():
                        t2l = self.ctx.tag_to_local.get(part)
                        if t2l is None or not pairs:
                            continue
                        max_tag = max(t for t, _ in pairs)
                        if max_tag >= len(t2l):
                            ext = np.full(max_tag + 1 - len(t2l),
                                          INVALID_LOC, dtype=np.uint32)
                            t2l = np.concatenate([t2l, ext])
                            self.ctx.tag_to_local[part] = t2l
                        tags_arr = np.fromiter((t for t, _ in pairs),
                                               dtype=np.int64, count=len(pairs))
                        locs_arr = np.fromiter((l for _, l in pairs),
                                               dtype=np.uint32, count=len(pairs))
                        t2l[tags_arr] = locs_arr
                    # Broadcast consolidated new_l2t to all nodes
                    bcast = pickle.dumps(all_new_l2t)
                    self.ctx.sync.broadcast_bytes(bcast)
                    # Update C++ local_to_tag for remote partitions
                    for part, ll, tag in all_new_l2t:
                        if part != my_part:
                            self.ctx.rdma_store.update_local_to_tag_entry(part, ll, tag)
                else:
                    self.ctx.sync.send_to_coordinator(sync_payload)
                    bcast = self.ctx.sync.recv_bytes()
                    all_new_l2t = pickle.loads(bcast)
                    # Bucket by partition + pre-extend once per partition.
                    _by_part: Dict[int, List[Tuple[int, int]]] = {}
                    for part, ll, tag in all_new_l2t:
                        _by_part.setdefault(part, []).append((tag, ll))
                    for part, pairs in _by_part.items():
                        t2l = self.ctx.tag_to_local.get(part)
                        if t2l is None or not pairs:
                            continue
                        max_tag = max(t for t, _ in pairs)
                        if max_tag >= len(t2l):
                            ext = np.full(max_tag + 1 - len(t2l),
                                          INVALID_LOC, dtype=np.uint32)
                            t2l = np.concatenate([t2l, ext])
                            self.ctx.tag_to_local[part] = t2l
                        tags_arr = np.fromiter((t for t, _ in pairs),
                                               dtype=np.int64, count=len(pairs))
                        locs_arr = np.fromiter((l for _, l in pairs),
                                               dtype=np.uint32, count=len(pairs))
                        t2l[tags_arr] = locs_arr
                    # C++ side: only remote partitions (mine already updated by InsertOp)
                    for part, ll, tag in all_new_l2t:
                        if part != my_part:
                            self.ctx.rdma_store.update_local_to_tag_entry(part, ll, tag)

            elif op.name in ('consolidate', 'concurrent'):
                # Reverse of insert sync: zero tag_to_local[part][tag] for tags
                # released by every node's consolidation, so the coordinator's
                # tag_to_local (and remote C++ local_to_tag) no longer list
                # consolidated tags as active. Without this the next delete
                # re-selects ghost tags and active_global_ids is inflated (C1).
                # Symmetric: every node does exactly one gather/broadcast (coord)
                # or one send/recv (worker) — never branch on node id here.
                my_released = getattr(self.ctx, '_released_local_to_tag', [])
                sync_payload = pickle.dumps(my_released)

                def _apply_released(rel):
                    by_part = {}
                    for part, ll, tag in rel:
                        by_part.setdefault(part, []).append(tag)
                    for part, tags_list in by_part.items():
                        t2l = self.ctx.tag_to_local.get(part)
                        if t2l is None or not tags_list:
                            continue
                        tags_arr = np.asarray(tags_list, dtype=np.int64)
                        in_range = tags_arr < len(t2l)
                        t2l[tags_arr[in_range]] = INVALID_LOC
                    for part, ll, tag in rel:
                        if part != my_part:
                            self.ctx.rdma_store.update_local_to_tag_entry(part, ll, 0)

                if self.ctx.is_coordinator:
                    gathered = self.ctx.sync.gather_bytes()
                    all_released = list(my_released)
                    for peer_id, peer_data in gathered.items():
                        all_released.extend(pickle.loads(peer_data))
                    _apply_released(all_released)
                    self.ctx.sync.broadcast_bytes(pickle.dumps(all_released))
                else:
                    self.ctx.sync.send_to_coordinator(sync_payload)
                    all_released = pickle.loads(self.ctx.sync.recv_bytes())
                    _apply_released(all_released)
                self.ctx._released_local_to_tag = []

            self.ctx._new_local_to_tag = []
            sync_ms = (time.time() - t_sync) * 1000.0
            op.stats.custom['tag_sync_ms'] = round(sync_ms, 3)

        # Track active points: per-partition and total
        if self.ctx.is_coordinator:
            pn = op.stats.custom.get('per_node', {})
            per_part_active = {}
            total_active = 0
            for nid, ns in sorted(pn.items(), key=lambda x: int(x[0])):
                ap = ns.get('active_points', 0)
                per_part_active[nid] = ap
                total_active += ap
            op.stats.custom['active_points'] = total_active
            op.stats.custom['active_per_node'] = per_part_active
        elif self.ctx.slot_manager is not None:
            op.stats.custom['active_points'] = self.ctx.slot_manager.active_count
        self.op_history.append(op.stats.to_dict())

        # Fail fast if configured
        if op.stats.status == "failed":
            fail_fast = self.ctx.config.get('distributed_engine', {}).get('fail_fast', True)
            if fail_fast:
                raise RuntimeError(
                    f"Operation {op.name}#{seq} failed: {op.stats.error}")

    def save(self):
        """Save operation history as JSON + human-readable report."""
        # Barrier-fence before the merge: stats files are written at the end
        # of each _dispatch (after the op-end barrier but with no further
        # sync), and on NFS the report-writer can sometimes start scanning
        # before remote workers' writes are visible — symptom is missing
        # per_node entries (e.g. N2 absent from consolidate's per_node).
        self.ctx.sync.barrier("save-stats-flushed")

        # When orchestrator is external (not a worker), the lowest-partition
        # worker takes responsibility for writing the shared report files.
        is_report_writer = self.ctx.is_coordinator or (
            not self.ctx.coordinator_is_worker and self.ctx.my_partition == 0)
        if is_report_writer:
            # Merge per-node stats from node_stats/ files into op_history
            stats_dir = self.run_dir / "node_stats"
            if stats_dir.exists():
                for op in self.op_history:
                    seq = op.get('seq', None)
                    if seq is None or op.get('per_node'):
                        continue
                    per_node = {}
                    for sf in sorted(stats_dir.glob(f"node*_seq{seq}.json")):
                        try:
                            with open(sf) as _f:
                                nd_stats = json.load(_f)
                            nid = nd_stats.get('node_id', sf.stem)
                            per_node[nid] = nd_stats
                        except Exception:
                            pass
                    if per_node:
                        op['per_node'] = per_node
                        # Recompute aggregate active_points from per-node data.
                        # Without this, the report writer's local active count
                        # (one partition only) leaks into the timeline as if
                        # it were the cluster total.
                        ap_sum = 0
                        ap_per_node = {}
                        for _nid, _ns in per_node.items():
                            _ap = _ns.get('active_points', 0)
                            ap_per_node[_nid] = _ap
                            ap_sum += _ap
                        op['active_points'] = ap_sum
                        op['active_per_node'] = ap_per_node

            # Build full report with workload config embedded
            update_cfg = self.ctx.config.get('distributed_engine', {})
            report_data = {
                "experiment_name": self.ctx.config.get('experiment_name', 'unnamed'),
                "workload_config": update_cfg.get('workload', []),
                "num_nodes": self.ctx.num_partitions,
                "node_ids": sorted(self.ctx.cluster.nodes.keys()),
                "operations": self.op_history,
            }
            report_file = self.run_dir / "op_history.json"

            class _NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if isinstance(obj, (np.integer,)):
                        return int(obj)
                    if isinstance(obj, (np.floating,)):
                        return float(obj)
                    return super().default(obj)

            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2, cls=_NumpyEncoder)
            self.ctx.log(f"Operation history saved: {report_file}")

            txt_report = generate_engine_report(self)
            txt_file = self.run_dir / "engine_report.txt"
            with open(txt_file, 'w') as f:
                f.write(txt_report)
            self.ctx.log(f"Engine report saved: {txt_file}")

        self.ctx.sync.barrier("engine-done")
        self.ctx.log("OperationEngine: save complete")

    def _parse_workload(self) -> List[Operation]:
        """Parse workload from config."""
        return [self._make_op(spec.as_dict())
                for spec in parse_workload_specs(self.ctx.config)]

    def _make_op(self, spec: dict) -> Operation:
        """Create an Operation from a workload spec dict."""
        op_name = spec.get('op', '')
        op_class = _OPERATION_REGISTRY.get(op_name)
        if op_class is None:
            raise ValueError(f"Unknown operation: {op_name!r}. "
                             f"Available: {list(_OPERATION_REGISTRY.keys())}")
        # Pass all keys except 'op' as params
        params = {k: v for k, v in spec.items() if k != 'op'}
        return op_class(params)



# ======================================================================
# Backward compatibility alias
# ======================================================================

class DistributedUpdateWorker:
    """Backward-compatible wrapper. Delegates to OperationEngine.

    Preserves the old interface:
        worker = DistributedUpdateWorker(node_worker, config)
        worker.run(update_cfg)
    """

    def __init__(self, node_worker, config: dict,
                 hooks: Optional[List[OperationHook]] = None):
        self._engine = OperationEngine(node_worker, config, hooks=hooks)

    def run(self, update_cfg: dict):
        """Main entry point (backward compatible)."""
        self._engine.load()
        self._engine.execute_workload()
        self._engine.save()
