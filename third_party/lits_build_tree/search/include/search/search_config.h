#pragma once
// search/search_config.h — Search configuration parameters

#include <cstdint>

namespace danns {

class ConsolidationCacheWarmer;

struct SearchConfig {
    uint32_t K = 10;               // Number of results
    uint32_t L = 100;              // Search beam width
    uint32_t entry_mode = 0;       // 0=CENTROID_AND_FROZEN, 1=FROZEN_ONLY
    uint32_t query_partition = 0;  // Partition to start search from
    uint32_t batch_size = 0;       // Batch prefetch size (0=disabled)
    uint32_t conv_window = 3;      // Convergence detector window
    float conv_threshold_pct = 0.1f; // Convergence threshold %
    bool return_pool = false;
    // Resolve top-K candidates to global tags (result.indices) and build the
    // edge_to_entry_vid map that feeds it. Search needs this to return tags to
    // the user; insert discards result.indices and uses only result.pool, so
    // it sets this false — skipping the per-candidate map build (and its l2t /
    // colocation reads) and the final top-K conversion. Pool is unaffected.
    bool need_result_tags = true;
    bool lits_filter = false;
    uint32_t lits_my_partition = 0;
    bool skip_timing = false;
    // Trace
    uint32_t trace_query_id = 0;
    uint16_t trace_level = 0;
    // Benchmark/probe mode: skip every dist_fn->compare and substitute a
    // deterministic vid-derived fake value. Keeps the RDMA fetch / queue /
    // visited path intact so distance-compute overhead is suppressed and
    // network/RDMA overhead is easier to isolate. NOT for production search;
    // recall is meaningless.
    bool compute_skip = false;

    // Hop-interleaved multi-query-per-thread factor. 1 = disabled (one query per
    // thread, sequential hops — the legacy path). >1 = each thread drives Q
    // queries concurrently, interleaving their per-hop RDMA so up to Q reads are
    // outstanding per thread instead of 1. Only honored on the multi-QP +
    // SEPARATED fetch path with all per-query optimizations off (no PQ filter,
    // no batch prefetch, no trace); the binding falls back to the per-query path
    // otherwise. Recall is identical to the per-query path because interleaving
    // only reorders *when* each query waits, never any query's computation.
    uint32_t interleave_factor = 1;

    // Optional search-warming hook for the consolidation cache. Default null
    // keeps search behavior and hot-path overhead unchanged.
    ConsolidationCacheWarmer* consolidation_cache_warmer = nullptr;
};

} // namespace danns
