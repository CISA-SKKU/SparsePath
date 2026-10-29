#!/usr/bin/env python3
"""Shared distributed engine context and primitive graph operations."""

from __future__ import annotations

import os
import re as _re
import struct
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from pipeline.danns_types import (
    GRAPH_SLACK_FACTOR,
    OperationHook,
    NearestCentroidPolicy,
    INVALID_VID,
    encode_vector_id,
    decode_vector_id,
)
from pipeline.engine_types import RDMACounters
from pipeline.experiment_paths import (
    artifact_colocation_dir, artifact_dir_candidates,
)
from pipeline.slot_manager import ColocationSlotManager, INVALID_LOC


def _extract_iteration_from_graph(graph_file: str) -> int:
    """Extract iteration number from graph checkpoint filename."""
    if not graph_file:
        return 0
    m = _re.match(r'^iteration(\d+)$', Path(graph_file).name)
    return int(m.group(1)) if m else 0



# ======================================================================
# UpdateContext — shared state + primitive operations
# ======================================================================

class UpdateContext:
    """Shared state and primitive operations for distributed update.

    Extracted from the old DistributedUpdateWorker. Holds rdma_store,
    slot_manager, tag_to_local, and all data access / graph mutation
    methods. Operation subclasses use this context to perform work.

    Requires:
        - node_worker: NodeWorker instance (has rdma_store, sync, cluster, etc.)
        - config: full pipeline config dict
    """

    def __init__(self, node_worker, config: dict,
                 hooks: Optional[List[OperationHook]] = None):
        self.nw = node_worker
        self.config = config
        self.hooks: List[OperationHook] = list(hooks) if hooks else []
        self.rdma_store = node_worker.rdma_store_wrapper.rdma_store
        self.sync = node_worker.sync
        self.cluster = node_worker.cluster
        self.node_id = node_worker.node_id
        self.is_coordinator = node_worker.is_coordinator
        # External orchestrator mode: coordinator (node 0) doesn't run operations.
        # Workers must not send internal op data (e.g., SearchOp results) via sync,
        # because orchestrator has no matching gather for those messages.
        active = getattr(node_worker, 'active_nodes', [])
        self.coordinator_is_worker = self.cluster.coordinator_node in active

        # Partition mapping (exclude coordinator — no RDMA/data)
        sorted_nodes = sorted(n for n in self.cluster.nodes.keys()
                              if n != self.cluster.coordinator_node)
        self.my_partition = sorted_nodes.index(self.node_id)
        self.num_partitions = len(sorted_nodes)
        self.partition_to_node = {i: sorted_nodes[i] for i in range(self.num_partitions)}
        self.node_to_partition = {sorted_nodes[i]: i for i in range(self.num_partitions)}

        # Partition assignment policy
        update_cfg = config.get('distributed_engine', {})
        policy_name = update_cfg.get('partition_policy', 'nearest_centroid')
        if policy_name == 'nearest_centroid':
            self.partition_policy = NearestCentroidPolicy()
        else:
            self.partition_policy = NearestCentroidPolicy()

        # Co-location metadata
        self.colocation_meta = node_worker.colocation_meta
        self.tag_to_local: Dict[int, Dict[int, int]] = {}  # partition -> {tag: local_loc}
        self.slot_manager: Optional[ColocationSlotManager] = None  # for my partition

        # Centroids for partition assignment
        self.centroids = node_worker.centroids

        # All vectors (needed for insert point lookup and distance computation)
        # In colocation-only mode, vectors is None (not loaded for memory optimization)
        self.vectors = node_worker.vectors
        self.total_vectors = getattr(node_worker, 'total_vectors', 0)

        # Queries (for search operations)
        self.queries = getattr(node_worker, 'queries', None)

        # Tags and partition_ids arrays (lightweight, for search result conversion)
        self.tags: Optional[np.ndarray] = None
        self.partition_ids_array: Optional[np.ndarray] = None

        # Build parameters
        build_cfg = config.get('build', {})
        self.R = build_cfg.get('R', 64)
        self.L = build_cfg.get('L', 100)
        self.alpha = build_cfg.get('alpha', 1.2)
        self.slack_R = int(GRAPH_SLACK_FACTOR * self.R)
        self.maxc = 750  # max candidate pool size for pruning

        # Delete tracking (tag-based, no GraphEngine delete_set)
        self.delete_tags: Set[int] = set()
        # Global delete tags from all partitions (populated by DeleteOp)
        self._global_delete_tags: Dict[int, Set[int]] = {}

        # Iteration number (set in _load_diskann_index)
        self.graph_iteration: int = 0

        # Sliding window cursor: next global_id to insert from base vectors.
        # Initialized in OperationEngine.load() after tag_to_local is populated.
        self._next_insert_gid: int = 0
        # Delete seed counter
        self._delete_seed_counter: int = 0

        # RDMA call counters (reset per operation by _dispatch)
        self.rdma_counters = RDMACounters()

        # Remote cache (set via set_remote_cache())
        self._remote_cache = None
        self._configure_consolidation_cache()

    def _resolve_cache_numa_env(self):
        """DANNS_CACHE_NUMA_NODE=nic|far -> concrete node id, resolved per
        host BEFORE the C++ arena allocation reads the env (the NIC's socket
        differs across hosts: hosts A/B mlx5_0 vs hosts C/D mlx5_1). Numeric
        values pass through untouched; unresolvable symbolic values are unset
        so the arena falls back to legacy first-touch placement."""
        val = os.environ.get('DANNS_CACHE_NUMA_NODE', '').strip().lower()
        if not val or val.lstrip('-').isdigit():
            return
        if val in ('interleave', 'il'):
            # Handled C++-side (MPOL_INTERLEAVE over all online nodes); there
            # is no single node id to resolve. Normalise and pass through.
            os.environ['DANNS_CACHE_NUMA_NODE'] = 'interleave'
            self.log("Cache arena NUMA: MPOL_INTERLEAVE across all online nodes")
            return
        try:
            if val not in ('nic', 'far'):
                raise ValueError(f"unrecognized value {val!r}")
            dev = getattr(self.nw.rdma_store_wrapper, 'rdma_device', '') or ''
            nic = int(open(f'/sys/class/infiniband/{dev}/device/numa_node')
                      .read())
            if nic < 0:
                raise ValueError(f'{dev} reports numa_node=-1')
            if val == 'nic':
                node = nic
            else:
                nodes = sorted(
                    int(d[4:]) for d in os.listdir('/sys/devices/system/node')
                    if d.startswith('node') and d[4:].isdigit())
                node = next((n for n in nodes if n != nic), None)
                if node is None:
                    raise ValueError('single NUMA node')
            os.environ['DANNS_CACHE_NUMA_NODE'] = str(node)
            self.log(f"Cache arena NUMA bind: '{val}' -> node {node} "
                     f"(NIC socket={nic})")
        except Exception as exc:
            os.environ.pop('DANNS_CACHE_NUMA_NODE', None)
            self.log(f"Cache arena NUMA bind: cannot resolve '{val}' ({exc})"
                     " — unset, legacy first-touch")

    def _configure_consolidation_cache(self):
        cache_cfg = self.config.get('consolidation_cache', {}) or {}
        if not hasattr(self.rdma_store, 'configure_consolidation_cache'):
            return
        self._resolve_cache_numa_env()
        nrec = int(cache_cfg.get('neighbor_records_per_thread', 0) or 0)
        gvbytes = int(cache_cfg.get('global_vector_bytes', 0) or 0)
        num_partitions = int(
            self.config.get('partition', {}).get(
                'num_partitions', self.num_partitions) or self.num_partitions or 1)
        partitions_meta = (self.colocation_meta or {}).get('partitions', {})
        max_local_locs = 0
        if partitions_meta:
            max_local_locs = max(
                int(info.get('num_entries', 0) or 0)
                for info in partitions_meta.values())
        if max_local_locs <= 0:
            build_cfg = self.config.get('build', {}) or {}
            max_local_locs = int(
                build_cfg.get('max_capacity', 0) or self.total_vectors or 0)
        dataset_cfg = self.config.get('dataset', {}) or {}
        dim = int(dataset_cfg.get('dimension', 0) or 0)
        dtype = dataset_cfg.get('data_type', 'float32')
        elem_size = {'float32': 4, 'uint8': 1, 'int8': 1}.get(dtype, 4)
        vector_bytes = dim * elem_size if dim else 0
        self.rdma_store.configure_consolidation_cache(
            neighbor_records_per_thread=nrec,
            global_vector_bytes=gvbytes,
            max_local_locs=max_local_locs,
            num_partitions=num_partitions,
            vector_bytes=vector_bytes)
        if nrec or gvbytes:
            slots = (gvbytes // vector_bytes) if vector_bytes else 0
            self.log("Consolidation cache caps: "
                     f"neighbor_records/thread={nrec or 'default'}, "
                     f"global_vector_bytes={gvbytes // 1024 // 1024 // 1024:.1f} GiB "
                     f"({slots // 1_000_000}M slots), "
                     f"dedup_bitset={num_partitions}x"
                     f"{max_local_locs // 1_000_000}M local_locs "
                     f"({num_partitions * max_local_locs // 8 // 1024 // 1024} MiB)")

    def set_remote_cache(self, cache):
        """Attach a RemoteCache instance for caching remote fetches."""
        self._remote_cache = cache

    def _call_hooks(self, method_name: str, *args, **kwargs):
        """Dispatch hook calls directly (no engine delegation)."""
        result = None
        for hook in self.hooks:
            r = getattr(hook, method_name)(*args, **kwargs)
            if r is not None:
                result = r
        return result

    def _coloc_dir(self) -> Path:
        """Colocation directory for the current graph iteration."""
        from common.partition_paths import (
            get_insert_strategy, colocation_dir as build_colocation_dir,
        )
        from pipeline.experiment_paths import artifact_experiment_dir
        coloc_cfg = self.config.get('colocation', {})
        # output_dir: fully resolved absolute path (legacy --set override)
        if coloc_cfg.get('output_dir'):
            return Path(coloc_cfg['output_dir'])
        # Derive from artifacts root: artifact_experiment_dir is the base that
        # build_colocation_dir appends colocation/<strategy>/<variant> to.
        num_partitions = self.config.get('partition', {}).get('num_partitions', 4)
        strategy_name = get_insert_strategy(self.config)
        return build_colocation_dir(
            artifact_experiment_dir(self.config), num_partitions, strategy_name,
            variant=f"iteration{self.graph_iteration}",
        )

    def log(self, msg: str):
        print(f"[DUpdate Node {self.node_id}] {msg}", flush=True)

    def setup_multi_qp_barriered(self, num_threads, base_port, barrier_prefix):
        """Per-pair, barrier-fenced multi-QP setup.

        Mirrors node_worker.connect_to_peer (the proven single-QP pattern): walk
        node pairs in a globally-consistent SORTED order, set up ONE peer at a
        time, and fence every pair with a cluster-wide barrier so the server is
        guaranteed to be listening before the client connects — by construction,
        not by timing margin. Replaces the single setup_multi_qp() call whose
        unordered per-pair iteration raced the accept/retry windows under load
        (worse under the consolidation-cache arena, which slows ibv_reg_mr).

        The orchestrator (coordinator) mirrors these per-pair barriers in
        orchestrator._relay_mid_op_barriers — barrier_prefix must match there.
        """
        all_node_ids = sorted(n for n in self.cluster.nodes.keys()
                              if n != self.cluster.coordinator_node)
        self.rdma_store.setup_multi_qp_begin()
        # [16p fix 2026-08-28] fence begin(): under the consolidation-cache arena,
        # ibv_reg_mr in begin() can exceed the client 100s connect ceiling of the
        # FIRST pair (observed: node1 begin() > 100s -> node2 refused x200). One
        # barrier guarantees every node finished begin() before any pair starts.
        self.sync.barrier(f"{barrier_prefix}-begin")
        for i in range(len(all_node_ids)):
            for j in range(i + 1, len(all_node_ids)):
                node_a, node_b = all_node_ids[i], all_node_ids[j]
                if self.node_id == node_a or self.node_id == node_b:
                    peer_id = node_b if self.node_id == node_a else node_a
                    self.rdma_store.setup_multi_qp_pair(
                        num_threads, base_port, peer_id)
                self.sync.barrier(f"{barrier_prefix}-{node_a}-{node_b}")
        self.rdma_store.setup_multi_qp_finalize(num_threads)

    # ==================================================================
    # Index loading (C++ DiskANN + lightweight metadata)
    # ==================================================================

    def _load_diskann_index(self):
        """Load DiskANN index into C++ search engine + lightweight metadata arrays.

        Does NOT create a Python GraphEngine. The C++ index is used only for
        search_colocation_pure() beam search. All graph data access goes through
        the colocation layout through C++ operation bindings.
        """
        experiment_dir = artifact_dir_candidates(self.config)[0]

        # Find latest graph checkpoint
        graph_file = tags_file = None
        for candidate_dir in artifact_dir_candidates(self.config):
            try:
                graph_file, tags_file = self.nw._find_graph_checkpoint(candidate_dir)
                experiment_dir = candidate_dir
                break
            except FileNotFoundError:
                continue
        if graph_file is None or tags_file is None:
            graph_file, tags_file = self.nw._find_graph_checkpoint(experiment_dir)
        self.graph_iteration = _extract_iteration_from_graph(graph_file)
        self.log(f"Loading index: {graph_file} (iteration {self.graph_iteration})")
        self.log(f"Loading tags:  {tags_file}")

        # Load tags as lightweight numpy array (for VID conversion)
        import node_worker as _node_worker
        self.tags = _node_worker.load_tags_file(tags_file)

        # Load C++ graph + distance function
        num_partitions = self.config.get('partition', {}).get('num_partitions', 4)
        coloc_cfg = self.config.get('colocation', {})
        colocation_only = coloc_cfg.get('enabled', False)
        data_type = getattr(self.nw, 'data_type', 'float32')

        import time as _time
        t0 = _time.perf_counter()
        if colocation_only:
            self.rdma_store.load_tags(tags_file)
            self.rdma_store.init_distance_only(
                getattr(self.nw, 'vector_dim', 96), data_type)
            self.log(f"Distance fn initialized in {_time.perf_counter() - t0:.2f}s "
                     f"(colocation-only, graph not loaded)")
        else:
            raise NotImplementedError(
                "Non-colocation mode is not supported in current C++ bindings. "
                "load_graph() and load_diskann_index() are not bound. "
                "Set colocation_only=true in config.")

        # Set partition entry points (virtual centroids)
        self.nw._set_partition_entry_points(self.rdma_store, experiment_dir)

        self.log(f"Index metadata loaded: {len(self.tags)} tags")

    def _load_tag_to_local(self):
        """Load tag_to_local mappings from colocation metadata.

        Loads as numpy arrays (not Python dicts) for fast initialization.
        tag_to_local[p] is a uint32 array where table[tag] = local_loc,
        INVALID_LOC (0xFFFFFFFF) means tag is absent.
        """
        if not self.colocation_meta:
            self.log("[WARN] No colocation metadata — tag_to_local unavailable")
            return

        coloc_dir = self._coloc_dir()

        t2l_files = self.colocation_meta.get('tag_to_local_files', {})
        for p_str, filename in t2l_files.items():
            p = int(p_str)
            filepath = coloc_dir / filename
            if filepath.exists():
                # Binary format: [max_tag: uint32, table: uint32[max_tag+1]]
                # table[tag] = local_loc (or UINT32_MAX if absent)
                raw = np.fromfile(str(filepath), dtype=np.uint32)
                max_tag_header = int(raw[0])
                table = raw[1:]  # skip header, keep as numpy array
                n_active = int(np.count_nonzero(table != INVALID_LOC))
                self.tag_to_local[p] = table
                self.log(f"  tag_to_local[{p}]: {n_active} entries "
                         f"(max_tag={max_tag_header})")

        # Initialize ColocationSlotManager for my partition
        # Use data_capacity (excludes centroid/frozen slots) so those are never
        # allocated to inserts. Centroid and frozen entries sit BEYOND data_capacity.
        partitions_meta = self.colocation_meta.get('partitions', {})
        my_part_meta = partitions_meta.get(str(self.my_partition), {})
        capacity = my_part_meta.get('data_capacity',
                                    my_part_meta.get('num_entries', 0))
        active = my_part_meta.get('num_active', capacity)  # backward compat
        my_t2l = self.tag_to_local.get(self.my_partition,
                                       np.array([], dtype=np.uint32))

        if capacity > 0:
            self.slot_manager = ColocationSlotManager(capacity, active, my_t2l)
            self.log(f"  ColocationSlotManager: capacity={capacity}, "
                     f"active={active}, free={self.slot_manager.num_free}")

    # ==================================================================
    # Helper: VectorID / partition / addressing
    # ==================================================================

    def _tag_to_vid(self, tag: int, partition: int) -> int:
        """Convert (tag, partition) to graph edge VID (partition|local_loc)."""
        local_loc = self._get_local_loc(partition, tag)
        if local_loc < 0:
            return INVALID_VID
        return encode_vector_id(partition, local_loc)

    def _vid_to_local_loc_partition(self, vid: int) -> Tuple[int, int]:
        """Decode VectorID into (partition, local_loc)."""
        return decode_vector_id(vid)

    def _get_local_loc(self, partition: int, tag: int) -> int:
        """Get co-location local_loc for a given (partition, tag)."""
        t2l = self.tag_to_local.get(partition)
        if t2l is not None and tag < len(t2l) and t2l[tag] != INVALID_LOC:
            return int(t2l[tag])
        return -1

    def _is_local(self, partition: int) -> bool:
        """Check if a partition is local to this node."""
        return partition == self.my_partition

    # ==================================================================
    # Insert primitives
    # ==================================================================

    def _read_vectors_range(self, start: int, count: int) -> np.ndarray:
        """Read a range of vectors from the base file without loading everything."""
        from common.data_loader import load_bin_data
        dtype_map = {'float32': np.float32, 'int8': np.int8,
                     'uint8': np.uint8, 'float16': np.float16}
        dtype = dtype_map.get(getattr(self.nw, 'data_type', 'float32'), np.float32)
        return load_bin_data(self.nw.base_file, dtype=dtype,
                             offset_points=start, num_points=count)

    def _prepare_insert_assignments(self, num_inserts: int,
                                    seed: Optional[int] = None) -> dict:
        """Coordinator: select next `num_inserts` points from the base vector
        set using a sliding window cursor (_next_insert_gid).

        The cursor advances monotonically — no overlap with previously
        inserted ranges, no dependency on tag_to_local synchronization.

        Returns lightweight assignment: {partition_id: [global_id, ...]}
        Vectors are NOT included — each node reads its own from the shared
        base file in InsertOp.execute() to avoid 1.5GB+ broadcast.
        Also returns '_meta': {start_gid, end_gid} for vector reads.
        """
        total_vectors = self.total_vectors
        start_gid = self._next_insert_gid
        end_gid = min(start_gid + num_inserts, total_vectors)
        actual_count = end_gid - start_gid

        if actual_count <= 0:
            self.log(f"[WARN] No more vectors to insert (cursor={start_gid}, "
                     f"total={total_vectors})")
            result = {p: [] for p in range(self.num_partitions)}
            result['_meta'] = {'start_gid': start_gid, 'end_gid': start_gid}
            return result

        if actual_count < num_inserts:
            self.log(f"[WARN] Only {actual_count} vectors remaining "
                     f"(requested {num_inserts})")

        # Read vectors for partition assignment only (coordinator)
        points = self._read_vectors_range(start_gid, actual_count)
        partition_ids = self.partition_policy.assign_batch(points, self.centroids)

        # Group gids by partition (vectors NOT included in broadcast)
        assignments: Dict = {p: [] for p in range(self.num_partitions)}
        for idx in range(actual_count):
            gid = start_gid + idx
            p = int(partition_ids[idx])
            assignments[p].append(gid)

        self._next_insert_gid = end_gid

        self.log(f"  Insert window: gid [{start_gid}, {end_gid}) = {actual_count} points")
        for p in range(self.num_partitions):
            self.log(f"  Insert assignment: partition {p} gets {len(assignments[p])} points")

        assignments['_meta'] = {'start_gid': start_gid, 'end_gid': end_gid}
        return assignments

    def _vid_to_gid(self, vid: int) -> int:
        """Convert VID (partition|local_loc) to global_id (tag - 1).
        Returns -1 if mapping not found."""
        part, ll = decode_vector_id(vid)
        # Build reverse index (numpy arrays) on first call
        if not hasattr(self, '_local_to_tag_rev'):
            self._local_to_tag_rev = {}
            for p, t2l in self.tag_to_local.items():
                active_tags = np.where(t2l != INVALID_LOC)[0]
                active_locs = t2l[active_tags]
                if len(active_locs) == 0:
                    self._local_to_tag_rev[p] = np.array([], dtype=np.uint32)
                    continue
                max_loc = int(active_locs.max())
                rev = np.full(max_loc + 1, INVALID_LOC, dtype=np.uint32)
                rev[active_locs] = active_tags.astype(np.uint32)
                self._local_to_tag_rev[p] = rev
        rev = self._local_to_tag_rev.get(part)
        if rev is not None and ll < len(rev):
            tag = rev[ll]
            if tag != INVALID_LOC and tag > 0:
                return int(tag) - 1
        # Check slot_manager for newly inserted points
        if part == self.my_partition and self.slot_manager:
            if ll < len(self.slot_manager.local_to_tag):
                tag = self.slot_manager.local_to_tag[ll]
                if tag != INVALID_LOC and tag > 0:
                    return int(tag) - 1
        return -1

    # ==================================================================
    # Delete primitives
    # ==================================================================

    def _prepare_delete_tags(self, num_deletes: int,
                             seed: Optional[int] = None) -> dict:
        """Coordinator: select tags to delete, grouped by partition."""
        # Collect all active tags across partitions from tag_to_local,
        # excluding tags that are already lazy-deleted (pending consolidation).
        active_tags_by_partition: Dict[int, List[int]] = {}
        for p, t2l in self.tag_to_local.items():
            active_tags = np.where(t2l != INVALID_LOC)[0]
            # Filter out already-deleted tags
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

        # Group by partition, also pre-compute VIDs for cross-node sync
        delete_tags: Dict[int, list] = {p: [] for p in range(self.num_partitions)}
        delete_vids: Dict[int, list] = {p: [] for p in range(self.num_partitions)}
        for p, tag in selected:
            delete_tags[p].append(tag)
            ll = self._get_local_loc(p, tag)
            if ll >= 0:
                delete_vids[p].append(encode_vector_id(p, ll))

        for p in range(self.num_partitions):
            self.log(f"  Delete assignment: partition {p} gets {len(delete_tags[p])} tags")

        self._delete_seed_counter += 1
        return {"tags": delete_tags, "vids": delete_vids}

    # ==================================================================
    # Consolidation primitives
    # ==================================================================

    def _consolidate_local(self, defer_cleanup: bool = False) -> dict:
        """Consolidate deletes for this partition.

        Delegates to C++ consolidate_local() for performance:
        - Zero-copy local vector access
        - AVX2 SIMD distance computation
        - Batch RDMA vector prefetch
        - No Python→C++ boundary crossing in hot loop

        defer_cleanup=True (ConcurrentOp): skip the post-consolidation cleanup
        (slot release, local_to_tag zeroing, deleted-set clears) — a co-running
        search still reads those structures. The caller must invoke
        finish_deferred_consol_cleanup() after all concurrent readers stop.

        Returns stats dict.
        """
        # Released (my_partition, local_loc, tag) tuples from THIS consolidation,
        # read by engine.py / orchestrator.py post-op consolidate sync to zero
        # tag_to_local on all nodes (mirror of insert's _new_local_to_tag).
        # Always defined so the sync gather/broadcast stays symmetric even on
        # the nothing-to-consolidate early return.
        self._released_local_to_tag = []
        if not self.delete_tags and not self._global_delete_tags:
            self.log("Consolidate: nothing to consolidate")
            return {'modified': 0, 'deleted_count': 0}

        # Collect all deleted VIDs across partitions (using partition|local_loc)
        global_delete_vids: Set[int] = set()
        unresolved_tags = 0
        for p, tags in self._global_delete_tags.items():
            for tag in tags:
                ll = self._get_local_loc(p, tag)
                if ll >= 0:
                    global_delete_vids.add(encode_vector_id(p, ll))
                else:
                    unresolved_tags += 1

        total_delete_tags = sum(len(t) for t in self._global_delete_tags.values())
        self.log(f"Consolidate: {len(global_delete_vids)} deleted VIDs from "
                 f"{total_delete_tags} tags ({unresolved_tags} unresolved)")

        if not global_delete_vids:
            self.log("Consolidate: no deleted VIDs")
            return {'modified': 0, 'deleted_count': 0}

        old_delete_tags = set(self.delete_tags)
        self._call_hooks('on_pre_consolidation', None, old_delete_tags)

        if self.slot_manager is None:
            self.log("[WARN] No slot manager — cannot consolidate")
            return {'modified': 0, 'deleted_count': 0}

        # Prepare arrays for C++ call — extract active entries from numpy arrays
        l2t = self.slot_manager.local_to_tag
        active_mask = l2t != INVALID_LOC
        local_locs_arr = np.where(active_mask)[0].astype(np.uint32)
        tags_arr = l2t[local_locs_arr].astype(np.uint32)
        locs_arr = local_locs_arr
        del_vids_arr = np.array(list(global_delete_vids), dtype=np.uint64)
        del_tags_arr = np.array(list(old_delete_tags), dtype=np.uint32)

        # Consolidation trace configuration
        trace_cfg = self.config.get('trace', {})
        ct_trace_path = ""
        if trace_cfg.get('enabled', False):
            ct_trace_dir = trace_cfg.get('output_dir', '') or str(self.run_dir / "traces")
            Path(ct_trace_dir).mkdir(parents=True, exist_ok=True)
            ct_trace_path = str(Path(ct_trace_dir) / f"consol_trace_N{self.node_id}_S{getattr(self, '_current_op_seq', 0)}.bin")
            self.log(f"[CONSOLIDATE] Trace output: {ct_trace_path}")

        # Per-vector fetch-count dump (lightweight, dense per-partition counters).
        # Independent toggle from full trace; emits one binary file per node.
        vfct_dump_path = ""
        vfct_cfg = self.config.get('vec_fetch_count', {})
        if vfct_cfg.get('enabled', False):
            vfct_dir = vfct_cfg.get('output_dir', '') or str(self.run_dir / "vec_fetch_count")
            Path(vfct_dir).mkdir(parents=True, exist_ok=True)
            vfct_dump_path = str(Path(vfct_dir)
                / f"vfct_N{self.node_id}_S{getattr(self, '_current_op_seq', 0)}.bin")
            self.log(f"[CONSOLIDATE] Per-vector fetch-count dump: {vfct_dump_path}")

        # C++ consolidation: 2-hop expansion, batch RDMA, RobustPrune, write
        consol_cfg = self.config.get('consolidation', {})
        enable_vec_cache = consol_cfg.get('enable_vec_cache', False)
        num_threads = consol_cfg.get('num_threads', 1)
        pass2_num_threads = consol_cfg.get('pass2_num_threads', 0)
        # De-overlapped baseline: serialize the OFF path's remote vector reads.
        # Only passed when enabled, so a pre-patch .so (which lacks the kwarg)
        # keeps working; the compiled default is false.
        sync_fetch = bool(consol_cfg.get('sync_fetch', False))
        _sync_kw = {'sync_fetch': True} if sync_fetch else {}
        # Fetch in-flight window depth K (supersedes sync_fetch: K=1 IS the
        # sync_fetch code path). Same only-when-set idiom, so a pre-patch .so
        # that lacks the kwarg keeps loading. Both kwargs are forwarded when
        # both are configured, so the C++ side can LOG the precedence decision
        # instead of one being silently dropped Python-side.
        fetch_window = max(0, int(consol_cfg.get('fetch_window', 0) or 0))
        if fetch_window > 0:
            _sync_kw['fetch_window'] = fetch_window
        # Buffer-reuse arm: per-thread staging arena for fetched remote vectors
        # instead of one heap block per vector. Same only-when-enabled idiom, so
        # a pre-patch .so keeps loading. Independent of sync_fetch — the two
        # kwarg dicts are merged at the call sites, all four combos are valid.
        reuse_fetch_buffers = bool(consol_cfg.get('reuse_fetch_buffers', False))
        _reuse_kw = {'reuse_fetch_buffers': True} if reuse_fetch_buffers else {}
        cache_cfg = self.config.get('consolidation_cache', {}) or {}
        cache_neighbor_records_per_thread = int(cache_cfg.get('neighbor_records_per_thread', 0) or 0)
        cache_global_vector_bytes = int(cache_cfg.get('global_vector_bytes', 0) or 0)
        enable_lits_filter = self.config.get('lits', {}).get('enabled', False)
        # LITS-Aware-Update tau rule: deleted neighbors join the tau sample
        # (default ON; False = old survivors-only tau, kept as an A/B toggle)
        lits_tau_include_deleted = bool(
            self.config.get('lits', {}).get('tau_include_deleted', True))

        # Multi-QP setup: per-thread QPs for lock-free RDMA in consolidation.
        # Setup/teardown is amortizable startup cost (~tens of seconds for
        # 32 threads × N peers); track separately so it doesn't pollute the
        # consolidation-body wall metric reported in engine_report.
        # Skip when no peers exist (single-node mode, num_partitions=1) — the
        # underlying setup_multi_qp errors out with "No peer connections".
        num_peers = sum(1 for nid in self.partition_to_node.values()
                        if nid != self.node_id)
        # ConcurrentOp pre-arranges one shared QP pool (consol+search slices)
        # and owns its setup/teardown barriers — skip both here in that mode.
        external_qp = getattr(self, '_external_multi_qp', False)
        use_multi_qp = (consol_cfg.get('use_multi_qp', False)
                        and num_threads > 1
                        and num_peers > 0
                        and not external_qp)
        _t_qp_setup = time.time() if use_multi_qp else 0.0
        if use_multi_qp:
            my_node = self.cluster.get_node(self.node_id)
            self.log(f"[CONSOLIDATE] Setting up {num_threads} per-thread QPs...")
            self.setup_multi_qp_barriered(
                num_threads, my_node.rdma_port, "consol-multiqp")
            self.sync.barrier("consol-multi-qp-setup")
            self.log(f"[CONSOLIDATE] Multi-QP ready")
        qp_setup_sec = (time.time() - _t_qp_setup) if use_multi_qp else 0.0
        # Mark the start of the consolidation BODY (post-setup, includes C++
        # call + Python diagnostics + cleanup). qp_teardown_sec is captured
        # separately below so engine_report can subtract both ends.
        _t_body_start = time.time()

        # Sequential consolidation mode: one node at a time (for debugging)
        sequential_consol = consol_cfg.get('sequential', False)
        if sequential_consol:
            for nid in sorted(self.partition_to_node.values()):
                if nid == self.node_id:
                    self.log(f"[CONSOLIDATE] Sequential mode: node {nid} running")
                self.sync.barrier(f"consol-seq-{nid}")
                if nid != self.node_id:
                    continue
                stats = self.rdma_store.consolidate_local(
                    locs_arr, tags_arr, del_vids_arr, del_tags_arr,
                    self.R, self.alpha, self.maxc,
                    trace_path=ct_trace_path,
                    enable_vec_cache=enable_vec_cache,
                    num_threads=num_threads,
                    enable_lits_filter=enable_lits_filter,
                    lits_tau_include_deleted=lits_tau_include_deleted,
                    vec_fetch_count_dump_path=vfct_dump_path,
                    cache_neighbor_records_per_thread=cache_neighbor_records_per_thread,
                    cache_global_vector_bytes=cache_global_vector_bytes,
                    **_sync_kw, **_reuse_kw,
                )
            self.sync.barrier("consol-seq-done")
        else:
            stats = self.rdma_store.consolidate_local(
                locs_arr, tags_arr, del_vids_arr, del_tags_arr,
                self.R, self.alpha, self.maxc,
                trace_path=ct_trace_path,
                enable_vec_cache=enable_vec_cache,
                num_threads=num_threads,
                enable_lits_filter=enable_lits_filter,
                lits_tau_include_deleted=lits_tau_include_deleted,
                pass2_num_threads=pass2_num_threads,
                vec_fetch_count_dump_path=vfct_dump_path,
                cache_neighbor_records_per_thread=cache_neighbor_records_per_thread,
                cache_global_vector_bytes=cache_global_vector_bytes,
                **_sync_kw, **_reuse_kw,
            )

        # Reflect C++ RDMA stats into Python-side counters for report.
        remote_vec = stats.get('remote_vec_accesses', 0)
        remote_nbr = stats.get('remote_nbr_accesses', 0)
        self.rdma_counters.vector_access_local += stats.get('local_vec_accesses', 0)
        self.rdma_counters.vector_access_remote += remote_vec
        self.rdma_counters.neighbor_access_local += stats.get('local_nbr_accesses', 0)
        self.rdma_counters.neighbor_access_remote += remote_nbr
        self.rdma_counters.fetch_entry_calls += remote_vec + remote_nbr
        self.rdma_counters.fetch_entry_items += remote_vec + remote_nbr

        # Diagnostic logging is gated — set consolidation.diagnostics=true to
        # re-enable the per-op breakdown / profile / contention / LITS / prefetch /
        # degree-histogram lines. Raw counters still flow through stats dict
        # so engine_report sections still populate.
        consol_diag = consol_cfg.get('diagnostics', False)
        modified = stats.get('modified', 0)
        if consol_diag and modified > 0:
            self.log(f"[CONSOLIDATE DIAG] modified={modified} "
                     f"path_empty={stats.get('diag_path_empty', 0)} "
                     f"path_no_prune={stats.get('diag_path_no_prune', 0)} "
                     f"path_prune={stats.get('diag_path_prune', 0)} "
                     f"expanded_avg={stats.get('diag_avg_expanded', 0):.1f}")
            loop_ns = stats.get('loop_only_sec', 0) * 1e9
            def _pct(ns_val):
                return ns_val / loop_ns * 100 if loop_ns > 0 else 0
            rdma_ns = (stats.get('twohop_rdma_ns', 0) + stats.get('vecfetch_rdma_ns', 0))
            local_ns = (stats.get('twohop_local_ns', 0) + stats.get('vecfetch_local_ns', 0))
            dist_ns = stats.get('dist_compute_ns', 0)
            prune_ns_val = stats.get('prune_ns', 0)
            self.log(f"[CONSOLIDATE PROFILE] "
                     f"rdma={rdma_ns/1e9:.2f}s({_pct(rdma_ns):.0f}%) "
                     f"local={local_ns/1e9:.2f}s({_pct(local_ns):.0f}%) "
                     f"dist={dist_ns/1e9:.2f}s({_pct(dist_ns):.0f}%) "
                     f"prune={prune_ns_val/1e9:.2f}s({_pct(prune_ns_val):.0f}%)")

        # Body-window ends here (before teardown). Teardown time is recorded
        # separately so it can be excluded from the consolidate-body metric.
        consol_body_sec = time.time() - _t_body_start

        # Teardown multi-QP after consolidation
        _t_qp_teardown = time.time() if use_multi_qp else 0.0
        if use_multi_qp:
            self.sync.barrier("consol-multi-qp-teardown")  # Wait for ALL nodes to finish
            self.rdma_store.teardown_multi_qp()
            self.log(f"[CONSOLIDATE] Multi-QP torn down")
        qp_teardown_sec = (time.time() - _t_qp_teardown) if use_multi_qp else 0.0

        # Stash QP setup/teardown breakdown into stats so callers can split
        # multi-QP setup cost from actual consolidation work.
        stats['qp_setup_sec'] = round(qp_setup_sec, 3)
        stats['qp_teardown_sec'] = round(qp_teardown_sec, 3)
        stats['consol_body_sec'] = round(consol_body_sec, 3)
        self.log(f"[CONSOLIDATE TIME] qp_setup={qp_setup_sec:.2f}s "
                 f"body={consol_body_sec:.2f}s qp_teardown={qp_teardown_sec:.2f}s")

        if defer_cleanup:
            # Concurrent search may still be reading deleted_vids_/deleted_tags_
            # and local_to_tag — caller runs the cleanup after joining readers.
            self._pending_consol_cleanup = (
                old_delete_tags, global_delete_vids, consol_diag, stats)
            return stats

        self._consolidate_post_cleanup(
            old_delete_tags, global_delete_vids, consol_diag, stats)
        return stats

    def finish_deferred_consol_cleanup(self):
        """Run the cleanup a defer_cleanup=True consolidation left pending.

        No-op when nothing is pending (e.g. early-returned consolidation)."""
        pending = getattr(self, '_pending_consol_cleanup', None)
        if pending is None:
            return
        self._pending_consol_cleanup = None
        self._consolidate_post_cleanup(*pending)

    def _consolidate_post_cleanup(self, old_delete_tags, global_delete_vids,
                                  consol_diag, stats):
        """Post-consolidation cleanup — slot release, local_to_tag zeroing,
        deleted-set clears, hooks. Extracted from _consolidate_local so
        ConcurrentOp can fence it behind the search join (R2/R3 races)."""
        # Optional sample-verification (gated). Disabled by default — adds
        # 20 C++ calls + numpy mask but otherwise non-essential.
        if consol_diag:
            l2t_after_v = self.slot_manager.local_to_tag
            _eligible = np.where((l2t_after_v != INVALID_LOC) & (l2t_after_v != 0))[0]
            sample_n = min(20, len(_eligible))
            if sample_n > 0:
                _idx = np.random.default_rng().choice(
                    len(_eligible), size=sample_n, replace=False)
                stale = 0
                for _ll in _eligible[_idx]:
                    nbrs = self.rdma_store.local_get_neighbors(int(_ll))
                    for vid in nbrs:
                        vid = int(vid)
                        if vid == INVALID_VID: break
                        if vid in global_delete_vids: stale += 1
                self.log(f"[CONSOLIDATE DIAG] sample stale={stale}/{sample_n}")

        # Post-consolidation cleanup — fundamental, not diagnostic.
        # Releases slots to free list (so future inserts can reuse) and
        # zeroes local_to_tag (so straggler edges resolve to UINT32_MAX).
        # Vectorize the slot-manager + ctx.tag_to_local updates; only the
        # per-entry C++ update_local_to_tag_entry remains serial because
        # there's no bulk binding for it yet.
        if old_delete_tags:
            tags_arr = np.fromiter(old_delete_tags, dtype=np.uint32,
                                   count=len(old_delete_tags))
            sm = self.slot_manager
            if sm is not None:
                in_range = tags_arr < len(sm.tag_to_local)
                tags_in = tags_arr[in_range]
                live = sm.tag_to_local[tags_in] != INVALID_LOC
                live_tags = tags_in[live]
                live_locs = sm.tag_to_local[live_tags].astype(np.uint32)
                sm.tag_to_local[live_tags] = INVALID_LOC
                sm.local_to_tag[live_locs] = INVALID_LOC
                sm.free_slots.extend(live_locs.tolist())
                sm.active_count -= int(len(live_tags))
                # Expose released (partition, local_loc, tag) for cross-node sync
                # (engine.py / orchestrator.py zero tag_to_local[part][tag] on all
                # nodes + C++ local_to_tag[part][ll] on remote partitions).
                self._released_local_to_tag = [
                    (self.my_partition, int(_ll), int(_tag))
                    for _ll, _tag in zip(live_locs.tolist(), live_tags.tolist())
                ]
                # Per-loc C++ update for MY partition — only call needed locally
                # (no bulk binding). Remote partitions handled by the post-op sync.
                for _ll in live_locs:
                    self.rdma_store.update_local_to_tag_entry(
                        self.my_partition, int(_ll), 0)
            t2l = self.tag_to_local.get(self.my_partition)
            if t2l is not None:
                in_t2l = tags_arr[tags_arr < len(t2l)]
                t2l[in_t2l] = INVALID_LOC

        # PQ GC no longer needed: flat array with soft delete + tag reuse

        # Clear delete tracking after consolidation
        self.delete_tags.clear()
        self._global_delete_tags.clear()

        # Clear C++ deleted_vids_ — local_to_tag is zeroed for deleted
        # entries, so stale edges resolve to UINT32_MAX (filtered out).
        self.rdma_store.clear_deleted_vids()
        self.rdma_store.clear_deleted_tags()
        self.log(f"[CONSOLIDATE] Cleared deleted_vids + deleted_tags, zeroed local_to_tag for deleted entries")

        self._call_hooks('on_post_consolidation', None, old_delete_tags, stats)

    # ==================================================================
    # DiskANN binary checkpoint (colocation → DiskANN format)
    # ==================================================================

    def _save_checkpoint(self, run_dir: Path):
        """Save colocation data as a DiskANN-format checkpoint.

        Reads all entries from local ColocationStore and writes them in
        DiskANN binary graph + tags format. This is the reverse direction:
        colocation (source of truth) → DiskANN binary (for compatibility).
        """
        self.log("Saving checkpoint (colocation → DiskANN)...")
        t0 = time.time()

        ckpt_dir = run_dir / "checkpoint"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        if self.slot_manager is None:
            self.log("[WARN] No slot manager — cannot save checkpoint")
            return

        # Collect all active entries from colocation
        # Build tag → sequential DiskANN location mapping
        active_entries = []  # [(tag, local_loc), ...]
        t2l_sm = self.slot_manager.tag_to_local
        active_tags = np.where(t2l_sm != INVALID_LOC)[0]
        active_tags_sorted = np.sort(active_tags)
        for tag in active_tags_sorted:
            active_entries.append((int(tag), int(t2l_sm[tag])))

        total_nodes = len(active_entries)
        # tag → DiskANN location
        tag_to_diskann_loc: Dict[int, int] = {}
        for diskann_loc, (tag, _) in enumerate(active_entries):
            tag_to_diskann_loc[tag] = diskann_loc

        # Build DiskANN graph: neighbor VIDs → DiskANN locations
        graph: List[List[int]] = []
        tags_array = np.zeros(total_nodes, dtype=np.uint32)

        for diskann_loc, (tag, local_loc) in enumerate(active_entries):
            tags_array[diskann_loc] = tag

            # Read neighbors from colocation
            nbr_vids = self.rdma_store.local_get_neighbors(local_loc)
            diskann_nbrs = []
            for vid in nbr_vids:
                vid = int(vid)
                if vid == INVALID_VID:
                    break
                _, nbr_tag = decode_vector_id(vid)
                nbr_dloc = tag_to_diskann_loc.get(nbr_tag)
                if nbr_dloc is not None:
                    diskann_nbrs.append(nbr_dloc)
            graph.append(diskann_nbrs)

        # Write graph binary
        graph_file = ckpt_dir / f"graph_dengine_node{self.node_id}.bin"
        self._write_graph_binary(str(graph_file), graph)

        # Write tags binary
        tags_file = ckpt_dir / f"tags_dengine_node{self.node_id}.bin"
        self._write_tags_binary(str(tags_file), tags_array)

        elapsed = time.time() - t0
        self.log(f"Checkpoint saved in {elapsed:.2f}s: {graph_file} "
                 f"({total_nodes} nodes)")

    def _write_graph_binary(self, filepath: str, graph: List[List[int]]):
        """Write graph in DiskANN binary format.

        Format: [expected_file_size: uint64]
                for each node: [degree: uint32][neighbor_ids: uint32[degree]]
        """
        total_nodes = len(graph)

        # Compute total size
        total_size = 8  # file size header
        for nbrs in graph:
            total_size += 4 + len(nbrs) * 4

        with open(filepath, 'wb') as f:
            f.write(struct.pack('<Q', total_size))
            for nbrs in graph:
                f.write(struct.pack('<I', len(nbrs)))
                for nbr in nbrs:
                    f.write(struct.pack('<I', nbr))

    def _write_tags_binary(self, filepath: str, tags: np.ndarray):
        """Write tags in DiskANN binary format.

        Format: [npts: uint32][dim=1: uint32][tags: uint32[npts]]
        """
        total = len(tags)
        with open(filepath, 'wb') as f:
            f.write(struct.pack('<II', total, 1))
            tags.tofile(f)
