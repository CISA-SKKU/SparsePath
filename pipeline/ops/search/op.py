"""Search operation for the distributed engine."""

from __future__ import annotations

import numpy as np

from pipeline.operation_base import Operation
from .diagnostics import SearchTraceConfig, SearchVfctSession
from .options import resolve_search_runtime_options
from .query_routing import balance_assignment, squared_distances_to_centroids
from .recall import compute_recall
from .runtime import run_l_sweep
from .results import (
    active_global_ids, make_result_payload, merge_gathered_results,
    save_search_results,
)


class SearchOp(Operation):
    """Run search queries for recall measurement during update.

    Supports L sweep: runs search for each L value and reports recall@K per L.

    params:
        num_queries: int — number of queries to run (default: 100)
        K: int — top-K results (default: 10)
        L: int — single search beam width (legacy, used if L_values absent)
        L_values: list[int] — multiple L values for sweep (preferred)
    """
    name = "search"
    needs_broadcast = True

    def prepare(self, ctx: UpdateContext) -> dict:
        num_queries = self.params.get('num_queries', 100)
        K = self.params.get('K', 10)

        # L sweep: prefer L_values list, fall back to single L
        L_values = self.params.get('L_values', None)
        if L_values is None:
            # Check search config as fallback
            search_cfg = ctx.config.get('search', {})
            L_values = search_cfg.get('L_values', None)
        if L_values is None:
            L_values = [self.params.get('L', 100)]
        L_values = sorted(L_values)

        if ctx.queries is None:
            ctx.log("[WARN] SearchOp: no queries available")
            return {"K": K, "L_values": L_values, "num_queries": 0}

        pool_size = len(ctx.queries)
        actual_count = num_queries
        if actual_count > pool_size:
            reps = actual_count // pool_size + 1
            ctx.log(f"  SearchOp: looping queries ({num_queries} requested, pool={pool_size}, reps={reps})")
        ctx.log(f"  SearchOp L sweep: {L_values}")

        return {"K": K, "L_values": L_values, "num_queries": actual_count}

    def execute(self, ctx: UpdateContext, payload: dict):
        K = payload.get("K", 10)
        L_values = payload.get("L_values", [100])
        num_queries = payload.get("num_queries", 0)
        query_offset = int(payload.get("query_offset", 0))
        # vfct_label may be supplied via params (legacy) or payload (broadcast
        # from orchestrator). Payload wins so all nodes agree on the label.
        if payload.get("vfct_label", None) is not None:
            self.params["vfct_label"] = payload.get("vfct_label")

        # Workers route queries locally — every node runs the same
        # deterministic nearest-centroid assignment over the same query pool,
        # so broadcasting the routing table is unnecessary.
        # search.broadcast=true: every node processes every query (scatter-gather
        # baseline); useful for sharded-graph topologies where intra-partition
        # search alone cannot reach high recall.
        broadcast = bool(ctx.config.get('search', {}).get('broadcast', False))
        # search.balance_query_routing=true: cap each partition's query count at
        # ceil(Q/P) so cluster wall-clock is not set by an overloaded node. Off
        # by default so the routing stays A/B-able against the paper's numbers.
        balance_query_routing = bool(self.params.get(
            'balance_query_routing',
            ctx.config.get('search', {}).get('balance_query_routing', False)))
        # search.query_entry_partition=true: seed the beam from each query's
        # NEAREST partition centroid instead of this node's own, pulling the
        # centroid's neighbor list over RDMA when it is remote. Decouples
        # entry-point quality from which node runs the query -- the coupling
        # that forces query routing to stay static. Recall-affecting (different
        # seeds -> different trajectory), so default off.
        query_entry_partition = bool(self.params.get(
            'query_entry_partition',
            ctx.config.get('search', {}).get('query_entry_partition', False)))
        # search.query_loop_repeats=R runs the same query slice R times to widen
        # the measurement window. Per-node slices finish in 0.2-0.9s at
        # Q=10000/4 nodes, so a single 0.3s hiccup swings a per-L wall-clock
        # number by 2-3x (2026-07-27: one sift L=60 point moved 25,381->14,144
        # while every other L held +34%). Repeats get fresh qids, and
        # compute_recall_posthoc skips qids past the end of GT, so recall is
        # still measured over exactly one pass and GT stays at num_queries --
        # unlike the legacy num_queries>pool tiling, which would demand GT for
        # the whole (possibly larger) pool.
        loop_repeats = int(self.params.get(
            'query_loop_repeats',
            ctx.config.get('search', {}).get('query_loop_repeats', 1)))
        # search.work_stealing=true: drop the static query->node split entirely
        # and let every node pull ids from one RDMA-atomic counter. Supersedes
        # balance_query_routing (which only equalises COUNTS; work per query
        # still varies ~1.5x across nodes and cannot be predicted beforehand).
        work_stealing = bool(self.params.get(
            'work_stealing',
            ctx.config.get('search', {}).get('work_stealing', False)))
        my_entry_parts = None   # (N,) uint32 per-query entry partition, or None
        if num_queries == 0 or ctx.queries is None:
            my_queries = []
        else:
            pool = ctx.queries
            pool_size = len(pool)
            if query_offset >= pool_size:
                my_queries = []
            else:
                end = min(query_offset + num_queries, pool_size)
                if num_queries > pool_size:
                    # Tile-fallback for legacy callers that ask for more queries
                    # than the pool. query_offset must be 0 in this case.
                    reps = num_queries // pool_size + 1
                    query_vectors = np.tile(pool, (reps, 1))[:num_queries]
                    qid_base = 0
                else:
                    query_vectors = pool[query_offset:end]
                    qid_base = query_offset
                if loop_repeats > 1:
                    if query_offset != 0:
                        ctx.log(f"[WARN] SearchOp: query_loop_repeats ignored "
                                f"(query_offset={query_offset} != 0; repeat qids "
                                f"would collide with another op's range)")
                    else:
                        n_base = len(query_vectors)
                        query_vectors = np.tile(query_vectors, (loop_repeats, 1))
                        ctx.log(f"  Query loop: {n_base} x {loop_repeats} = "
                                f"{len(query_vectors)} queries "
                                f"(recall still measured over the first {n_base})")
                if broadcast or work_stealing:
                    # Stealing: hand every node the whole array; the shared
                    # counter decides who actually runs what.
                    my_local_ids = np.arange(len(query_vectors), dtype=np.int64)
                else:
                    if balance_query_routing:
                        # Nearest-centroid routing skews per-node query counts
                        # (measured 16.5-39.8% shares on spacev100M/4p), and
                        # cluster wall-clock is set by the busiest node. Cap
                        # every partition at ceil(Q/P) and relocate the
                        # boundary queries. Every node runs this over the same
                        # pool and the same centroids, so the assignment agrees
                        # without any communication.
                        n_parts = len(ctx.centroids)
                        _d = squared_distances_to_centroids(
                            query_vectors, ctx.centroids)
                        query_partitions = balance_assignment(
                            np.argmin(_d, axis=1).astype(np.int32), _d, n_parts)
                    else:
                        query_partitions = ctx.partition_policy.assign_batch(
                            query_vectors, ctx.centroids)
                    my_local_ids = np.where(query_partitions == ctx.my_partition)[0]
                if query_entry_partition and not broadcast:
                    # Nearest centroid per query. Deliberately independent of the
                    # routing decision above: routing picks WHICH node runs the
                    # query, this picks WHERE its beam starts. Decoupling them is
                    # the point.
                    _dn = squared_distances_to_centroids(query_vectors, ctx.centroids)
                    my_entry_parts = np.ascontiguousarray(
                        np.argmin(_dn, axis=1).astype(np.uint32)[my_local_ids])
                my_queries = [
                    (int(qid_base + lid), query_vectors[lid])
                    for lid in my_local_ids
                ]

        ctx.log(f"SearchOp: {len(my_queries)} queries × {len(L_values)} L values "
                f"for partition {ctx.my_partition}")

        trace_config = SearchTraceConfig.from_context(ctx, self.stats.seq)
        vfct_session = SearchVfctSession.start(ctx, self.params)

        # Search controls. The optimization path is convergence-aware batched
        # fetching (phase-aware batching plus PQ filtering) in C++; compute_skip
        # is a benchmark/probe knob, not an optimization.
        search_options = resolve_search_runtime_options(ctx, self.params)
        batch_size = search_options.batch_size
        conv_window = search_options.conv_window
        conv_threshold_pct = search_options.conv_threshold_pct
        compute_skip = search_options.compute_skip
        pq_alpha = search_options.pq_alpha
        search_num_threads = search_options.num_threads
        search_use_multi_qp = search_options.use_multi_qp

        if batch_size > 0:
            ctx.log(f"  Batch prefetch: batch_size={batch_size} conv_window={conv_window} "
                    f"conv_threshold_pct={conv_threshold_pct}")
        if compute_skip:
            ctx.log(f"  [PROBE MODE] compute_skip=true — distance compute bypassed; recall is meaningless")
        if pq_alpha is not None and hasattr(ctx.rdma_store, 'set_pq_alpha'):
            ctx.rdma_store.set_pq_alpha(pq_alpha)
            ctx.log(f"  PQ alpha override: {pq_alpha}")

        # Setup per-thread QPs for multi-threaded search (avoids shared QP + mutex)
        if search_use_multi_qp:
            my_node = ctx.cluster.get_node(ctx.node_id)
            ctx.log(f"  Setting up {search_num_threads} per-thread QPs for search...")
            ctx.setup_multi_qp_barriered(
                search_num_threads, my_node.rdma_port, "search-multiqp")
            ctx.sync.barrier("search-multi-qp-setup")
            ctx.log(f"  Multi-QP ready")

        # FrozenSeedCache op boundary. The cache holds copies of the frozen
        # point's remote neighbor vectors keyed by VID = (partition|local_loc);
        # a delete / insert / consolidate between ops can hand that local_loc to
        # a different vector, which would make the copy silently wrong. Dropping
        # it here bounds its lifetime to this op, so a cached seed can never
        # span a mutation and the search path stays fully decoupled from the
        # update path. Cost is one re-warm (~R reads, once) per op.
        #
        # Placed AFTER multi-QP setup because the warm uses the per-thread QPs.
        #
        # NOTE: this boundary does not exist for workloads that consolidate
        # CONCURRENTLY with search (ops/concurrent.py) — disable the cache there
        # with `distributed_search.frozen_seed_cache: false`.
        if search_options.frozen_seed_cache and hasattr(
                ctx.rdma_store, 'reset_frozen_seed_cache'):
            try:
                _dropped = ctx.rdma_store.reset_frozen_seed_cache()
                if _dropped:
                    ctx.log(f"  FrozenSeedCache: dropped {_dropped} stale entries "
                            f"(re-warms on the first query of this op)")
            except Exception as _e:
                ctx.log(f"  [WARN] reset_frozen_seed_cache failed: {_e}")

        results_by_L, _per_L_stats, _per_L_latency, agg_stats = run_l_sweep(
            ctx=ctx,
            my_queries=my_queries,
            L_values=L_values,
            K=K,
            broadcast=broadcast,
            options=search_options,
            trace_config=trace_config,
            stats_seq=self.stats.seq,
            entry_parts=my_entry_parts,
            work_stealing=work_stealing,
        )

        self.stats.items_processed = len(my_queries)

        # Teardown per-thread QPs after search
        # Order: barrier FIRST so all peers finish RDMA before MRs are torn down
        if search_use_multi_qp:
            ctx.sync.barrier("search-multi-qp-teardown")
            ctx.rdma_store.teardown_multi_qp()

        # Counter survives this op so the next search op can accumulate on top.
        vfct_session.dump(ctx)

        # Consolidation-cache memory snapshot at this checkpoint (occupancy
        # curve over query count). get_consolidation_cache_stats() is const /
        # reads live atomics, so it is non-destructive and safe mid-workload.
        # One file per node per vfct_label, alongside search_vfct/.
        _cache_label = self.params.get("vfct_label")
        if _cache_label and hasattr(ctx.rdma_store, "get_consolidation_cache_stats"):
            try:
                import json as _json
                _cs = dict(ctx.rdma_store.get_consolidation_cache_stats())
                _cs["label"] = _cache_label
                _cs["node_id"] = ctx.node_id
                _csdir = ctx.run_dir / "cache_stats"
                _csdir.mkdir(parents=True, exist_ok=True)
                with open(_csdir / f"cache_N{ctx.node_id}_{_cache_label}.json", "w") as _f:
                    _json.dump(_cs, _f, indent=2)
                ctx.log(f"  [cache_stats] {_cache_label}: "
                        f"vec_records={_cs.get('vector_records')} "
                        f"vec_bytes={_cs.get('compacted_vector_bytes')} "
                        f"dropped_vec={_cs.get('dropped_vector_records')}")
            except Exception as _e:
                ctx.log(f"  [cache_stats] dump failed: {_e}")

        # Store per-L search stats and latency
        self.stats.custom['search_stats_by_L'] = _per_L_stats
        if _per_L_latency:
            self.stats.custom['latency_by_L'] = _per_L_latency

        # Store aggregate stats for backward compat
        if L_values and results_by_L:
            self.stats.custom['search_pure_stats'] = dict(agg_stats)  # last L

        # Gather/search result handling is intentionally outside the C++ hot path.
        send_payload = make_result_payload(results_by_L, ctx.my_partition)

        if ctx.is_coordinator:
            all_results_by_L = merge_gathered_results(ctx, results_by_L, L_values)
            all_active_gids = active_global_ids(ctx)

            total_gathered = sum(len(r) for r in all_results_by_L.values())
            self.stats.custom['total_queries_gathered'] = total_gathered
            ctx.log(f"SearchOp: gathered {total_gathered} total results "
                    f"({len(L_values)} L values × queries)")

            save_search_results(ctx, all_results_by_L, K, self.stats.seq)

            search_cfg = ctx.config.get('search', {}) or {}
            recall_flag = search_cfg.get('in_run_recall', False)
            if isinstance(recall_flag, str):
                in_run_recall = recall_flag.strip().lower() in ('1', 'true', 'yes', 'on')
            else:
                in_run_recall = bool(recall_flag)
            if in_run_recall and any(len(r) > 0 for r in all_results_by_L.values()):
                recall_stats = compute_recall(
                    ctx, all_results_by_L, all_active_gids, K, self.stats.seq)
                if recall_stats is not None:
                    self.stats.custom['recall'] = recall_stats
            elif not in_run_recall:
                ctx.log("[INFO] SearchOp: recall not requested; skipping "
                        "in-run recall (raw results saved)")
        elif ctx.coordinator_is_worker:
            # Only send if a worker (not external orchestrator) will gather.
            # In external orchestrator mode, sending here would pollute the
            # TCP stream and desync the dispatch-level protocol.
            ctx.sync.send_to_coordinator(send_payload)
        else:
            save_search_results(
                ctx, {L: list(results_by_L[L]) for L in L_values}, K, self.stats.seq)
