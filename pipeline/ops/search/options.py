"""Search runtime option resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class SearchRuntimeOptions:
    batch_size: int
    conv_window: int
    conv_threshold_pct: float
    compute_skip: bool
    pq_alpha: Optional[float]
    num_threads: int
    use_multi_qp: bool
    interleave_factor: int
    skip_timing: bool
    batch_interleave_completion: bool
    batch_remote_first: bool
    frozen_seed_cache: bool


def resolve_search_runtime_options(ctx: Any, params: Mapping[str, Any]) -> SearchRuntimeOptions:
    """Resolve per-op search controls from workload params and config.

    The actual search optimization is convergence-aware batched fetching
    (phase-aware batching plus PQ filtering) in the C++ search path.
    compute_skip is a benchmark/probe knob, not an optimization. This helper
    keeps Python responsible only for option resolution.
    """
    config = ctx.config
    cap_cfg = config.get('convergence_aware_prefetch', {}) or {}
    bf_cfg = cap_cfg.get('batched_fetching', {}) or {}
    batch_cfg = config.get('batch_prefetch', {}) or {}
    search_cfg = config.get('search', {}) or {}
    probe_cfg = config.get('benchmark_probes', {}) or {}

    cfg_batch_size = bf_cfg.get('batch_size', batch_cfg.get('batch_size', 0))
    cap_enabled = bool(cap_cfg.get('enabled', int(cfg_batch_size or 0) > 0))
    bf_enabled = bool(bf_cfg.get('enabled', int(cfg_batch_size or 0) > 0))
    if not (cap_enabled and bf_enabled):
        cfg_batch_size = 0
    batch_size = int(params.get('batch_size', cfg_batch_size))
    conv_window = int(bf_cfg.get(
        'convergence_window', bf_cfg.get(
            'conv_window', batch_cfg.get('conv_window', 3))))
    conv_threshold_pct = float(bf_cfg.get(
        'convergence_threshold_pct', bf_cfg.get(
            'conv_threshold_pct', batch_cfg.get('conv_threshold_pct', 1.0))))
    compute_skip = bool(params.get(
        'compute_skip', probe_cfg.get(
            'compute_skip', search_cfg.get('compute_skip', False))))

    pq_alpha = params.get('pq_alpha', None)
    pq_alpha = float(pq_alpha) if pq_alpha is not None else None

    num_threads = int(params.get('num_threads', search_cfg.get('num_threads', 1)))
    num_peers = sum(1 for nid in ctx.partition_to_node.values()
                    if nid != ctx.node_id)
    use_multi_qp = (bool(search_cfg.get('use_multi_qp', True))
                    and num_threads > 1
                    and num_peers > 0)

    # Hop-interleaved multi-query-per-thread factor (1 = disabled). Honored by
    # the C++ batch search only on the multi-QP + separated path with all
    # optimizations off; otherwise it falls back to the per-query path.
    interleave_factor = int(params.get(
        'interleave_factor', search_cfg.get('interleave_factor', 1)))

    # Elide all per-query timer_start/timer_ns (hrc::now) calls in the C++
    # search hot path. The finest-grained counters wrap *every* neighbor edge
    # (visited insert, resolve_vid), so leaving them on costs 44-63% of
    # throughput -- measured on the 07-27 ab3 timing-on/off pair (spacev CAP
    # T16: 39,167 -> 27,157 QPS at L=50; 18,466 -> 11,308 at L=200). Default
    # True so a benchmark can never silently pay that; set
    # search.skip_timing=false to get the CPU breakdown back, and read the
    # resulting percentages knowing they carry that overhead.
    skip_timing = bool(params.get(
        'skip_timing', search_cfg.get(
            'skip_timing', probe_cfg.get('skip_timing', True))))

    # Batch (CAP) path: owner-interleaved CQ completion processing instead of
    # the bulk-sync drain barrier (P1 item 6). Order-nondeterministic across
    # owners -> recall-equivalent, not byte-identical. Default off.
    batch_interleave_completion = bool(params.get(
        'batch_interleave_completion',
        search_cfg.get('batch_interleave_completion', False)))

    # Batch (CAP) path: post remote fetches before local distance compute
    # (P1b). Discovery order otherwise puts the local owner first, exposing the
    # full RTT. Recall-equivalent, not byte-identical (looser PQ threshold).
    batch_remote_first = bool(params.get(
        'batch_remote_first',
        search_cfg.get('batch_remote_first', False)))

    # FrozenSeedCache: serve the frozen point's remote neighbor seeds from a
    # process-local copy warmed once per search op, instead of one blocking
    # RDMA read per seed per query. Recall-neutral (identical bytes, identical
    # distances). Default on.
    #
    # The cache is dropped and re-warmed at every search-op boundary
    # (reset_frozen_seed_cache below), so it can never span a delete / insert /
    # consolidate. A workload that runs consolidation CONCURRENTLY with search
    # (ops/concurrent.py) has no such boundary -- set
    # `distributed_search.frozen_seed_cache: false` for those.
    ds_cfg = config.get('distributed_search', {}) or {}
    frozen_seed_cache = bool(params.get(
        'frozen_seed_cache',
        ds_cfg.get('frozen_seed_cache',
                   search_cfg.get('frozen_seed_cache', True))))

    return SearchRuntimeOptions(
        batch_size=batch_size,
        conv_window=conv_window,
        conv_threshold_pct=conv_threshold_pct,
        compute_skip=compute_skip,
        pq_alpha=pq_alpha,
        num_threads=num_threads,
        use_multi_qp=use_multi_qp,
        interleave_factor=interleave_factor,
        skip_timing=skip_timing,
        batch_interleave_completion=batch_interleave_completion,
        batch_remote_first=batch_remote_first,
        frozen_seed_cache=frozen_seed_cache,
    )
