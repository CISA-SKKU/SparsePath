#!/usr/bin/env python3
"""
Orchestrator for DANNS distributed pipeline.

Runs on orchestrator-host (no RDMA, no GPU). Coordinates DPU worker nodes via TCP.
Workers handle all RDMA search/insert/consolidate operations.

Usage:
    python3 orchestrator.py \
        --config configs/deep1M_e2e.yaml \
        --nodes 1,2,3,4 \
        --stage distributed_engine

Architecture:
    orchestrator-host (orchestrator)           workers (workers)
    ──────────────────              ─────────────────
    load config + NFS data          load colocation + RDMA
    parse workload                  wait for payload
    for each op:                    for each op:
      prepare() → payload             recv payload
      broadcast to workers             execute (RDMA ops)
      wait for completion              send stats
      gather stats                     barrier
      update local state
    generate report
"""

import argparse
import json
import numpy as np
import os
import pickle
import sys
import time
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Path setup
_this_dir = Path(__file__).resolve().parent
_project_root = _this_dir.parent
_scripts_dir = _project_root / "scripts"
for _p in [str(_this_dir), str(_project_root), str(_scripts_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline.config import ClusterConfig, ClusterNode, apply_config_overrides
from pipeline.sync import TCPSync
from pipeline.workload import parse_workload_specs
from pipeline.danns_types import (
    NearestCentroidPolicy, PartitionAssignmentPolicy,
    INVALID_VID, encode_vector_id, decode_vector_id,
)
from pipeline.slot_manager import ColocationSlotManager, INVALID_LOC
from common.data_loader import load_bin_data, get_bin_metadata, get_dataset_info
from common.partition_paths import (
    get_insert_strategy, partition_dir as build_partition_dir,
    colocation_dir as build_colocation_dir,
    colocation_dir_candidates,
)
from pipeline.experiment_paths import (
    artifact_colocation_dir, artifact_experiment_dir, artifact_dir_candidates, run_dir as build_run_dir,
)


_VEC_DTYPE_MAP = {'float32': np.float32, 'int8': np.int8, 'uint8': np.uint8}


def _resolve_vec_dtype(config: dict):
    """Resolve numpy dtype from config['dataset']['data_type']."""
    dt = config.get('dataset', {}).get('data_type', 'float32')
    return _VEC_DTYPE_MAP.get(dt, np.float32)


# ============================================================================
# OrchestratorContext — lightweight state (no RDMA, NFS only)
# ============================================================================

class OrchestratorContext:
    """Coordinator state without RDMA. Reads data from NFS."""

    def __init__(self, config: dict, cluster: ClusterConfig,
                 sync: TCPSync, num_partitions: int):
        self.config = config
        self.cluster = cluster
        self.sync = sync
        self.num_partitions = num_partitions
        self.is_coordinator = True

        # Data (loaded from NFS)
        self.vectors: Optional[np.ndarray] = None
        self.queries: Optional[np.ndarray] = None
        self.centroids: Optional[np.ndarray] = None
        self.partition_ids: Optional[np.ndarray] = None
        self.total_vectors: int = 0
        self.base_file: Optional[str] = None

        # Partition policy
        self.partition_policy: PartitionAssignmentPolicy = NearestCentroidPolicy()

        # Tag/loc mappings (per partition, synced from workers)
        self.tag_to_local: Dict[int, np.ndarray] = {}

        # Build params
        build_cfg = config.get('build', {})
        self.R = build_cfg.get('R', 64)
        self.L = build_cfg.get('L', 100)
        self.alpha = build_cfg.get('alpha', 1.2)
        self.maxc = build_cfg.get('maxc', 750)
        self.slack_R = int(np.ceil(1.3 * self.R))

        # Insert cursor
        self._next_insert_gid: int = 0
        # Delete tracking
        self.delete_tags: Set[int] = set()
        self._global_delete_tags: Dict[int, Set[int]] = {}
        self._delete_seed_counter: int = 0

    def log(self, msg: str):
        print(f"[Orchestrator] {msg}", flush=True)

    def _read_vectors_range(self, start: int, count: int) -> np.ndarray:
        """Read a range of vectors from the base file (NFS)."""
        return load_bin_data(self.base_file, dtype=_resolve_vec_dtype(self.config),
                             offset_points=start, num_points=count)

    def _get_local_loc(self, partition: int, tag: int) -> int:
        t2l = self.tag_to_local.get(partition)
        if t2l is None or tag >= len(t2l):
            return -1
        loc = t2l[tag]
        return -1 if loc == INVALID_LOC else int(loc)

    # ---- prepare methods (same logic as UpdateContext, no RDMA) ----

    def prepare_insert(self, num_inserts: int, seed: Optional[int] = None) -> dict:
        start_gid = self._next_insert_gid
        end_gid = min(start_gid + num_inserts, self.total_vectors)
        actual_count = end_gid - start_gid

        if actual_count <= 0:
            self.log(f"[WARN] No more vectors to insert (cursor={start_gid})")
            result = {p: [] for p in range(self.num_partitions)}
            result['_meta'] = {'start_gid': start_gid, 'end_gid': start_gid}
            return result

        points = self._read_vectors_range(start_gid, actual_count)
        partition_ids = self.partition_policy.assign_batch(points, self.centroids)

        assignments: Dict = {p: [] for p in range(self.num_partitions)}
        for idx in range(actual_count):
            gid = start_gid + idx
            p = int(partition_ids[idx])
            assignments[p].append(gid)

        self._next_insert_gid = end_gid

        self.log(f"  Insert window: gid [{start_gid}, {end_gid}) = {actual_count} points")
        for p in range(self.num_partitions):
            self.log(f"  Partition {p}: {len(assignments[p])} inserts")

        assignments['_meta'] = {'start_gid': start_gid, 'end_gid': end_gid}
        return assignments

    def prepare_delete(self, num_deletes: int, seed: Optional[int] = None) -> dict:
        active_tags_by_partition: Dict[int, List[int]] = {}
        for p, t2l in self.tag_to_local.items():
            active_tags = np.where(t2l != INVALID_LOC)[0]
            del_set = self.delete_tags | self._global_delete_tags.get(p, set())
            if del_set:
                del_arr = np.array(list(del_set), dtype=np.int64)
                mask = ~np.isin(active_tags, del_arr)
                active_tags = active_tags[mask]
            active_tags_by_partition[p] = active_tags.tolist()

        all_active = []
        for p, tags in active_tags_by_partition.items():
            for tag in tags:
                all_active.append((p, tag))

        # Sort by tag (= global_id + 1) so lowest IDs are deleted first (streaming order)
        all_active.sort(key=lambda x: x[1])

        if len(all_active) > num_deletes:
            selected = all_active[:num_deletes]
        else:
            selected = all_active

        delete_tags: Dict[int, list] = {p: [] for p in range(self.num_partitions)}
        delete_vids: Dict[int, list] = {p: [] for p in range(self.num_partitions)}
        for p, tag in selected:
            delete_tags[p].append(tag)
            ll = self._get_local_loc(p, tag)
            if ll >= 0:
                delete_vids[p].append(encode_vector_id(p, ll))

        # Accumulate into _global_delete_tags so subsequent prepare_delete
        # calls don't re-pick the same lowest tags. Workers track their own
        # delete state inside DeleteOp; orchestrator was missing the mirror.
        for p, tag in selected:
            self._global_delete_tags.setdefault(p, set()).add(tag)

        for p in range(self.num_partitions):
            self.log(f"  Partition {p}: {len(delete_tags[p])} deletes")

        self._delete_seed_counter += 1
        return {"tags": delete_tags, "vids": delete_vids}

    def prepare_search(self, params: dict) -> dict:
        num_queries = params.get('num_queries', 100)
        query_offset = int(params.get('query_offset', 0))
        K = params.get('K', 10)
        L_values = params.get('L_values', None)
        vfct_label = params.get('vfct_label', None)
        if L_values is None:
            search_cfg = self.config.get('search', {})
            L_values = search_cfg.get('L_values', None)
        if L_values is None:
            L_values = [params.get('L', 100)]
        L_values = sorted(L_values)

        if self.queries is None:
            self.log("[WARN] No queries available")
            return {"K": K, "L_values": L_values, "num_queries": 0,
                    "query_offset": 0, "vfct_label": vfct_label}

        pool_size = len(self.queries)
        if query_offset >= pool_size:
            self.log(f"[WARN] query_offset={query_offset} >= pool_size={pool_size}, "
                     "returning 0 queries")
            return {"K": K, "L_values": L_values, "num_queries": 0,
                    "query_offset": query_offset, "vfct_label": vfct_label}
        actual_count = min(num_queries, pool_size - query_offset)
        query_vectors = self.queries[query_offset:query_offset + actual_count]
        query_partitions = self.partition_policy.assign_batch(
            query_vectors, self.centroids)
        for p in range(self.num_partitions):
            count = int(np.sum(query_partitions == p))
            self.log(f"  Search routing[{query_offset}:{query_offset+actual_count}]: "
                     f"partition {p} gets {count} queries")

        return {"K": K, "L_values": L_values, "num_queries": actual_count,
                "query_offset": query_offset, "vfct_label": vfct_label}


# ============================================================================
# Orchestrator — main loop
# ============================================================================

class Orchestrator:
    """Coordinates DPU workers from orchestrator-host without RDMA."""

    def __init__(self, config_path: str, active_nodes: List[int],
                 stage: str = "distributed_engine", timestamp: str = None,
                 project_root: str = None, overrides: List[str] = None):
        self.stage = stage
        self.timestamp = timestamp or time.strftime("%Y%m%d_%H%M%S")

        # Project root for resolving relative paths
        self.project_root = Path(project_root).resolve() if project_root else Path("/workspace")

        # Load config + resolve paths
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        apply_config_overrides(self.config, overrides or [], log=True)
        from pipeline.config import resolve_config_paths
        resolve_config_paths(self.config, self.project_root)

        # Cluster setup: build_full injects orchestrator as node 0
        config_dir = str(Path(config_path).parent)
        cluster_cfg = self.config.get('cluster', {})
        self.cluster, rdma_cfg = ClusterConfig.build_full(
            cluster_cfg, config_dir, coordinator_node=0,
            active_nodes=active_nodes)

        self.active_worker_nodes = active_nodes
        self.num_partitions = len(active_nodes)

        # Sync
        self.sync = TCPSync(self.cluster, my_id=0)

        # Context
        self.ctx = OrchestratorContext(
            self.config, self.cluster, self.sync, self.num_partitions)

        # Output
        self.run_dir = build_run_dir(self.config, stage, self.timestamp)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Backup config to run dir
        import shutil
        shutil.copy2(config_path, self.run_dir / Path(config_path).name)

        self.op_history: List[dict] = []

    def run(self):
        print("=" * 60)
        print(f"ORCHESTRATOR — stage: {self.stage}")
        print(f"  workers: {self.active_worker_nodes}")
        print(f"  output: {self.run_dir}")
        print("=" * 60)

        self.sync.setup()
        self._load_data()

        # Broadcast timestamp to workers so they use same run dir
        self.sync.broadcast_value("timestamp", self.timestamp)
        self.sync.barrier("data-loaded")

        # Relay worker RDMA setup barriers (orchestrator has no RDMA)
        self._relay_rdma_barriers()

        if self.stage == "distributed_engine":
            self._run_distributed_engine()
        elif self.stage == "search":
            self._run_search()
        else:
            raise ValueError(f"Unknown stage: {self.stage}")

        self._save_report()
        self.sync.close()
        print("Orchestrator done.")

    def _relay_rdma_barriers(self):
        """Pass through RDMA setup barriers that workers exchange.

        The orchestrator has no RDMA but acts as the TCPSync relay.
        Must match the barrier sequence in node_worker._setup_rdma()
        and _setup_colocation().
        """
        self.ctx.log("Relaying worker RDMA setup barriers...")

        self.sync.barrier("rdma-init")

        # QP connect barriers: one per DPU node pair (orchestrator excluded)
        dpu_ids = sorted(self.active_worker_nodes)
        for i in range(len(dpu_ids)):
            for j in range(i + 1, len(dpu_ids)):
                self.sync.barrier(f"qp-connect-{dpu_ids[i]}-{dpu_ids[j]}")

        # Live-redirect alloc channel bootstrap (mirrors node_worker
        # Phase 1b; gating drift = hang).
        lb_cfg = self.config.get('load_balancer') or {}
        if lb_cfg.get('enabled', False) and lb_cfg.get('redirect', False):
            for i in range(len(dpu_ids)):
                for j in range(i + 1, len(dpu_ids)):
                    self.sync.barrier(f"redirect-qp-{dpu_ids[i]}-{dpu_ids[j]}")
            self.sync.barrier("redirect-server-ready")

        self.sync.barrier("rdma-qp-ready")
        self.sync.barrier("rdma-connected")

        # Colocation setup barriers (from _setup_colocation)
        self.sync.barrier("coloc-loaded")

        # Colocation MR exchange: one per DPU node pair
        for i in range(len(dpu_ids)):
            for j in range(i + 1, len(dpu_ids)):
                self.sync.barrier(f"coloc-mr-{dpu_ids[i]}-{dpu_ids[j]}")

        self.sync.barrier("coloc-ready")

        self.ctx.log("Worker RDMA setup complete.")

    def _load_data(self):
        """Load dataset metadata and lightweight arrays from NFS."""
        ctx = self.ctx
        dataset_cfg = self.config.get('dataset', {})
        ctx.base_file = dataset_cfg.get('base_file', '')
        query_file = dataset_cfg.get('query_file', '')

        # Vector count + dim from header (no full load)
        if ctx.base_file and os.path.exists(ctx.base_file):
            npts, dim = get_bin_metadata(ctx.base_file)
            ctx.total_vectors = npts
            ctx.log(f"Base file: {npts} vectors, dim={dim}")

        # Queries (small, load fully) — use dataset dtype so uint8/int8 work
        if query_file and os.path.exists(query_file):
            ctx.queries = load_bin_data(
                query_file, dtype=_resolve_vec_dtype(self.config))
            ctx.log(f"Queries: {ctx.queries.shape} ({ctx.queries.dtype})")

        # Partition assignments
        coloc_cfg = self.config.get('colocation', {})
        coloc_dir = coloc_cfg.get('output_dir', '')
        if not coloc_dir:
            experiment_name = self.config.get('experiment_name', 'unnamed')
            num_partitions = self.config.get('partition', {}).get('num_partitions', 4)
            strategy_name = get_insert_strategy(self.config)
            # Match worker's coloc_subdir resolution: prefer build.graph_variant,
            # fall back to iteration0. Split artifact configs are strict and do
            # not probe legacy roots.
            graph_variant = self.config.get('build', {}).get('graph_variant', None)
            coloc_subdir = graph_variant if graph_variant else "iteration0"
            # Orchestrator runs on the coordination host (NFS-only, no local SSD).
            # local_dir is for worker payload loading; orchestrator always reads
            # metadata from the NFS artifacts path.
            candidates = [build_colocation_dir(
                artifact_experiment_dir(self.config), num_partitions,
                strategy_name, variant=coloc_subdir)]
            for candidate in candidates:
                if (candidate / "colocation_meta.json").exists():
                    coloc_dir = str(candidate)
                    break

        # tag_to_local is only consumed by insert/delete prep + post-op sync.
        # Skip loading (can be several GB) for search-only workloads.
        needs_tag_to_local = any(
            op in ("insert", "delete") for op, _ in self._parse_workload())

        meta_file = os.path.join(coloc_dir, "colocation_meta.json") if coloc_dir else ""
        if needs_tag_to_local and (not meta_file or not os.path.exists(meta_file)):
            raise FileNotFoundError(
                "Required colocation metadata not found. "
                f"Checked local/artifact candidates for {coloc_subdir}: "
                f"{meta_file or coloc_dir}"
            )
        if os.path.exists(meta_file):
            with open(meta_file) as f:
                coloc_meta = json.load(f)
            ctx.log(f"Colocation meta loaded: {len(coloc_meta.get('partitions', {}))} partitions")

            if needs_tag_to_local:
                for p_str, p_meta in coloc_meta.get('partitions', {}).items():
                    p = int(p_str)
                    t2l_file = p_meta.get('tag_to_local_file', '')
                    if not t2l_file:
                        t2l_file = os.path.join(coloc_dir, f"tag_to_local_p{p}.bin")
                    if os.path.exists(t2l_file):
                        raw = np.fromfile(t2l_file, dtype=np.uint32)
                        if len(raw) > 1:
                            ctx.tag_to_local[p] = raw[1:]  # skip header (max_tag)
                        ctx.log(f"  Partition {p}: tag_to_local loaded ({len(raw)-1} entries)")
            else:
                ctx.log("  tag_to_local skipped (no insert/delete in workload)")

        # Centroids — stored under partition dir, not colocation dir
        centroids_file = dataset_cfg.get('centroids_file', '')
        if not centroids_file:
            num_partitions = self.config.get('partition', {}).get('num_partitions', 4)
            strategy_name = get_insert_strategy(self.config)
            for artifact_dir in artifact_dir_candidates(self.config):
                partition_dir = build_partition_dir(
                    artifact_dir, num_partitions, strategy_name)
                candidate = partition_dir / "centroids.fbin"
                if candidate.exists():
                    centroids_file = str(candidate)
                    break
        if centroids_file and os.path.exists(centroids_file):
            ctx.centroids = load_bin_data(centroids_file, dtype=np.float32)
            ctx.log(f"Centroids: {ctx.centroids.shape}")
        else:
            ctx.log(f"[WARN] Centroids not found: {centroids_file}")

        # Init insert cursor from tag_to_local
        max_tag = 0
        for p, t2l in ctx.tag_to_local.items():
            if len(t2l) > 0:
                valid_mask = (t2l != INVALID_LOC)
                if valid_mask.any():
                    max_tag = max(max_tag, int(np.where(valid_mask)[0][-1]) + 1)
        # Fallback: use build.initial_size if tag_to_local was empty
        if max_tag == 0:
            max_tag = self.config.get('build', {}).get('initial_size', 0)
        ctx._next_insert_gid = max_tag
        ctx.log(f"Insert cursor: gid={max_tag}")

    def _run_distributed_engine(self):
        """Run distributed update workload (insert/delete/consolidate/search)."""
        workload = self._parse_workload()
        self.ctx.log(f"Workload: {len(workload)} operations")

        # Workers expect dupdate_timestamp before engine-loaded barrier
        self.sync.broadcast_value("dupdate_timestamp", self.timestamp)
        self.sync.barrier("engine-loaded")

        for seq, (op_name, params) in enumerate(workload):
            self.ctx.log(f"--- Op {seq}: {op_name} ---")
            self._dispatch_op(seq, op_name, params)

        self.ctx.log("Workload complete")
        # Mirror engine.py:save() barrier — fences NFS visibility of
        # node_stats files before the report-writer starts merging them.
        self.sync.barrier("save-stats-flushed")
        self.sync.barrier("engine-done")

    def _run_search(self):
        """Run search-only stage."""
        search_cfg = self.config.get('search', {})
        params = {
            'num_queries': search_cfg.get('num_queries', 100),
            'K': search_cfg.get('K', 10),
            'L_values': search_cfg.get('L_values', [100]),
        }
        self.sync.broadcast_value("dupdate_timestamp", self.timestamp)
        self.sync.barrier("engine-loaded")
        self._dispatch_op(0, "search", params)

    def _dispatch_op(self, seq: int, op_name: str, params: dict):
        """Dispatch one operation: prepare → broadcast → wait → gather."""
        t0 = time.time()

        # Start barrier
        self.sync.barrier(f"op-{seq}-start")

        # Prepare + broadcast
        needs_broadcast = op_name in ("insert", "delete", "search")
        if needs_broadcast:
            payload = self._prepare(op_name, params)
            data = pickle.dumps(payload)
            self.sync.broadcast_bytes(data)
            self.ctx.log(f"  Broadcast payload: {len(data)} bytes")
            # Stash delete payloads to disk so coverage analysis (post-run)
            # can recover the deleted_set without re-replaying the workload.
            if op_name == "delete" and isinstance(payload, dict):
                try:
                    import numpy as _np
                    run_dir = getattr(self, 'run_dir', None)
                    if run_dir is not None:
                        out = Path(run_dir) / "delete_history"
                        out.mkdir(parents=True, exist_ok=True)
                        # Flatten per-partition vid lists into one array.
                        all_vids = []
                        for p, vs in (payload.get("vids") or {}).items():
                            all_vids.extend(int(v) for v in vs)
                        _np.save(out / f"delete_seq{seq}_vids.npy",
                                 _np.array(all_vids, dtype=_np.uint64))
                        all_tags = []
                        for p, ts in (payload.get("tags") or {}).items():
                            all_tags.extend(int(t) for t in ts)
                        _np.save(out / f"delete_seq{seq}_tags.npy",
                                 _np.array(all_tags, dtype=_np.int64))
                except Exception as _e:
                    self.ctx.log(f"  [WARN] delete history dump failed: {_e}")

        # Relay mid-op barriers that workers fire during execution.
        # Gating MUST match engine.py's worker-side conditions exactly.
        self._relay_mid_op_barriers(op_name, params)

        # Workers execute (orchestrator waits at end barrier)
        # End barrier — workers finished executing
        success = self.sync.barrier_with_status(f"op-{seq}-end", True)
        elapsed = time.time() - t0

        if not success:
            self.ctx.log(f"  [WARN] Some workers failed for op {seq}")

        # Gather stats from workers
        stats_data = self.sync.gather_bytes()
        worker_stats = {}
        for node_id, raw in stats_data.items():
            try:
                worker_stats[node_id] = pickle.loads(raw)
            except Exception:
                worker_stats[node_id] = {}

        # Post-op state sync (matching OperationEngine._dispatch protocol).
        # 'concurrent' embeds a consolidation → same released-tags sync.
        if op_name in ("insert", "delete", "consolidate", "concurrent"):
            self._sync_tag_to_local(
                'consolidate' if op_name == 'concurrent' else op_name)

        self.op_history.append({
            'seq': seq,
            'op': op_name,
            'params': params,
            'elapsed_sec': elapsed,
            'worker_stats': worker_stats,
            'success': success,
        })

        self.ctx.log(f"  Op {seq} ({op_name}) done in {elapsed:.2f}s")

    def _relay_mid_op_barriers(self, op_name: str, params: dict):
        """Relay barriers that workers fire inside op execution.

        Must exactly mirror the gating in pipeline/engine.py:
          - SearchOp (L2065, L2320, L2326): search-multi-qp-setup,
            search-L{L}-done (per L), search-multi-qp-teardown
            Gated by: search_num_threads > 1 AND search.use_multi_qp
          - ConsolidateOp (L1172, L1311): consol-multi-qp-setup/teardown
            Gated by: consolidate.num_threads > 1 AND consolidate.use_multi_qp
          - InsertOp (pipeline/ops/insert.py): lb-counters-pushed
            Gated by: load_balancer.enabled (fires on every worker,
            independent of batch size / peer count)
          - InsertOp parallel insert: insert-multi-qp-setup/teardown —
            gated by params num_threads>1 AND use_multi_qp AND
            num_partitions>1 (redirect-on included).
            (num_partitions>1 ⟺ workers' num_peers>0 only for 1:1
            partition:node deployments — the only supported topology today).
          - InsertOp live redirect (load_balancer.redirect):
            redirect-batches-done — fires on every worker when redirect is
            on, even on mid-op failure (errors carried past the fence).
        Any gating drift between worker and orchestrator = deadlock.
        """
        # Single-worker (no peers) deployments skip multi-QP entirely. Must
        # match the worker-side gate in engine.py (num_peers > 0). Drift here
        # would cause the orchestrator to wait on a barrier the worker never
        # enters.
        num_peers = max(0, len(self.active_worker_nodes) - 1)
        if op_name == "search":
            search_cfg = self.config.get('search', {}) or {}
            num_threads = params.get('num_threads', search_cfg.get('num_threads', 1))
            use_mqp = search_cfg.get('use_multi_qp', True)
            if num_threads > 1 and use_mqp and num_peers > 0:
                # Per-pair QP-setup barriers — mirror the worker's
                # EngineContext.setup_multi_qp_barriered loop (one barrier per
                # DPU node pair, same sorted order). Same shape as the init
                # qp-connect relay (_relay_rdma_barriers). Drift = deadlock.
                dpu_ids = sorted(self.active_worker_nodes)
                self.sync.barrier("search-multiqp-begin")  # [16p fix 2026-08-28] mirror engine_context begin fence
                for i in range(len(dpu_ids)):
                    for j in range(i + 1, len(dpu_ids)):
                        self.sync.barrier(f"search-multiqp-{dpu_ids[i]}-{dpu_ids[j]}")
                self.sync.barrier("search-multi-qp-setup")
                # Replicate worker's L_values resolution (engine.py L1994-2001)
                L_values = params.get('L_values')
                if L_values is None:
                    L_values = search_cfg.get('L_values')
                if L_values is None:
                    L_values = [params.get('L', 100)]
                L_values = sorted(L_values)
                for L in L_values:
                    self.sync.barrier(f"search-L{L}-done")
                self.sync.barrier("search-multi-qp-teardown")
        elif op_name == "consolidate":
            # NOTE: config key is 'consolidation' (matches engine.py:1170);
            # earlier 'consolidate' here was a typo that caused this branch
            # to silently no-op and workers to hang at consol-multi-qp-setup.
            consol_cfg = self.config.get('consolidation', {}) or {}
            num_threads = consol_cfg.get('num_threads', 1)
            use_mqp = consol_cfg.get('use_multi_qp', False) and num_threads > 1
            if use_mqp and num_peers > 0:
                # Per-pair QP-setup barriers — mirror the worker's
                # EngineContext.setup_multi_qp_barriered loop (consol prefix).
                dpu_ids = sorted(self.active_worker_nodes)
                self.sync.barrier("consol-multiqp-begin")  # [16p fix 2026-08-28] mirror engine_context begin fence
                for i in range(len(dpu_ids)):
                    for j in range(i + 1, len(dpu_ids)):
                        self.sync.barrier(f"consol-multiqp-{dpu_ids[i]}-{dpu_ids[j]}")
                self.sync.barrier("consol-multi-qp-setup")
                self.sync.barrier("consol-multi-qp-teardown")
        elif op_name == "concurrent":
            # ConcurrentOp (pipeline/ops/concurrent.py): one shared multi-QP
            # pool (consol+search thread slices) set up/torn down serially
            # around the concurrent window. Gate mirrors the worker exactly:
            # consolidation.use_multi_qp AND consolidation.num_threads > 1.
            consol_cfg = self.config.get('consolidation', {}) or {}
            num_threads = consol_cfg.get('num_threads', 1)
            use_mqp = consol_cfg.get('use_multi_qp', False) and num_threads > 1
            if use_mqp and num_peers > 0:
                dpu_ids = sorted(self.active_worker_nodes)
                for i in range(len(dpu_ids)):
                    for j in range(i + 1, len(dpu_ids)):
                        self.sync.barrier(
                            f"concurrent-multiqp-{dpu_ids[i]}-{dpu_ids[j]}")
                self.sync.barrier("concurrent-multi-qp-setup")
                self.sync.barrier("concurrent-multi-qp-teardown")
        elif op_name == "insert":
            # LoadBalancer counter push (pipeline/ops/insert.py): every
            # worker fires this when load_balancer.enabled, regardless of
            # how many points it received.
            lb_cfg = self.config.get('load_balancer', {}) or {}
            if lb_cfg.get('enabled', False):
                self.sync.barrier("lb-counters-pushed")
            # Parallel-insert multi-QP bracket. MUST mirror ins_multi_qp in
            # pipeline/ops/insert.py exactly: workload num_threads>1,
            # use_multi_qp (default true), and >1 data nodes (workers'
            # num_peers>0 ⟺ num_partitions>1). Redirect-on included.
            ins_threads = int(params.get('num_threads', 1))
            ins_use_mqp = bool(params.get('use_multi_qp', True))
            if (ins_threads > 1 and ins_use_mqp
                    and self.num_partitions > 1):
                self.sync.barrier("insert-multi-qp-setup")
                self.sync.barrier("insert-multi-qp-teardown")
            if lb_cfg.get('enabled', False) and lb_cfg.get('redirect', False):
                # Live redirect: workers close their redirect window
                # only after every batch finished — relay that fence.
                self.sync.barrier("redirect-batches-done")

    def _prepare(self, op_name: str, params: dict) -> Any:
        if op_name == "insert":
            return self.ctx.prepare_insert(
                params.get('count', 0), seed=params.get('seed'))
        elif op_name == "delete":
            return self.ctx.prepare_delete(
                params.get('count', 0), seed=params.get('seed'))
        elif op_name == "search":
            return self.ctx.prepare_search(params)
        else:
            return {}

    def _sync_tag_to_local(self, op_name: str):
        """Post-op sync matching OperationEngine._dispatch protocol.

        insert: gather diffs from workers → merge → broadcast consolidated back
        delete: no sync needed (delete tags already broadcast in prepare)
        """
        if op_name == 'delete':
            # Workers skip sync for delete (tags already broadcast).
            return

        if op_name == 'insert':
            # Gather new (partition, local_loc, tag) tuples from workers.
            # Bucket by partition and apply vectorized to avoid per-tag
            # np.concatenate on the full tag_to_local array (O(N*L)).
            sync_data = self.sync.gather_bytes()
            all_updates = []
            per_partition_tags: Dict[int, List[int]] = {}
            per_partition_locs: Dict[int, List[int]] = {}
            for node_id, raw in sync_data.items():
                try:
                    updates = pickle.loads(raw)
                except Exception as e:
                    self.ctx.log(f"  [WARN] Failed to sync tag_to_local from node {node_id}: {e}")
                    continue
                all_updates.extend(updates)
                for partition, local_loc, tag in updates:
                    per_partition_tags.setdefault(partition, []).append(tag)
                    per_partition_locs.setdefault(partition, []).append(local_loc)

            for partition, tags_list in per_partition_tags.items():
                t2l = self.ctx.tag_to_local.get(partition)
                if t2l is None or not tags_list:
                    continue
                tags_arr = np.asarray(tags_list, dtype=np.int64)
                locs_arr = np.asarray(per_partition_locs[partition], dtype=np.uint32)
                max_tag = int(tags_arr.max())
                if max_tag >= len(t2l):
                    ext = np.full(max_tag + 1 - len(t2l), INVALID_LOC, dtype=np.uint32)
                    t2l = np.concatenate([t2l, ext])
                    self.ctx.tag_to_local[partition] = t2l
                t2l[tags_arr] = locs_arr

            # Broadcast consolidated updates back to all workers
            bcast = pickle.dumps(all_updates)
            self.sync.broadcast_bytes(bcast)

        if op_name == 'consolidate':
            # Reverse of insert: gather released (partition, local_loc, tag) from
            # workers, zero the orchestrator's tag_to_local[part][tag], broadcast
            # back so workers zero theirs + remote C++ local_to_tag (C1). The
            # orchestrator has no rdma_store, so it only updates Python state.
            # MUST mirror the worker engine.py consolidate branch (one gather +
            # one broadcast) or the cluster hangs (code_review §8.5 class).
            sync_data = self.sync.gather_bytes()
            all_released: List[tuple] = []
            per_partition_tags: Dict[int, List[int]] = {}
            for node_id, raw in sync_data.items():
                try:
                    rel = pickle.loads(raw)
                except Exception as e:
                    self.ctx.log(f"  [WARN] Failed to sync consolidate from node {node_id}: {e}")
                    continue
                all_released.extend(rel)
                for partition, local_loc, tag in rel:
                    per_partition_tags.setdefault(partition, []).append(tag)

            for partition, tags_list in per_partition_tags.items():
                t2l = self.ctx.tag_to_local.get(partition)
                if t2l is None or not tags_list:
                    continue
                tags_arr = np.asarray(tags_list, dtype=np.int64)
                in_range = tags_arr < len(t2l)
                t2l[tags_arr[in_range]] = INVALID_LOC

            self.sync.broadcast_bytes(pickle.dumps(all_released))

    def _parse_workload(self) -> List[tuple]:
        """Parse workload from config."""
        return [spec.as_tuple() for spec in parse_workload_specs(self.config)]

    def _save_report(self):
        """Save orchestrator report."""
        report = {
            'stage': self.stage,
            'timestamp': self.timestamp,
            'workers': self.active_worker_nodes,
            'num_partitions': self.num_partitions,
            'operations': self.op_history,
            'config': self.config,
        }
        report_file = self.run_dir / "orchestrator_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        self.ctx.log(f"Report saved: {report_file}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="DANNS Orchestrator (orchestrator-host)")
    parser.add_argument("--config", required=True, help="Pipeline YAML config")
    parser.add_argument("--nodes", required=True,
                        help="Comma-separated DPU node IDs (e.g., 1,2,3,4)")
    parser.add_argument("--stage", default="distributed_engine",
                        choices=["distributed_engine", "search"])
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--project-root", default=None,
                        help="Project root for resolving relative paths in config")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
        help="Override a config key via dotted path before execution",
    )
    args = parser.parse_args()

    active_nodes = [int(n) for n in args.nodes.split(',')]

    orch = Orchestrator(
        config_path=args.config,
        active_nodes=active_nodes,
        stage=args.stage,
        timestamp=args.timestamp,
        project_root=args.project_root,
        overrides=args.overrides,
    )
    orch.run()


if __name__ == "__main__":
    main()
