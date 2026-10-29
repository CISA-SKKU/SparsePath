#pragma once
// search/search_config.h — Search configuration parameters

#include <cstdint>

namespace danns {

class ConsolidationCacheWarmer;

struct SearchConfig {
    uint32_t K = 10;               // Number of results
    uint32_t L = 100;              // Search beam width
    uint32_t entry_mode = 0;       // 0=CENTROID_AND_FROZEN, 1=FROZEN_ONLY
    // Partition whose centroid seeds this query (entry_mode 0).
    //
    // Historically declared but never read: the seed always came from
    // node_store->partition_id(), i.e. whichever node happened to run the
    // query. That couples entry-point quality to the execution node, which is
    // what forces query routing to stay static -- relocating a query for load
    // balance also degrades where its search starts. RED-ANNS decouples the two
    // with a global learn graph and CoTra by replicating the top 1% index; we
    // can do it for free because any vertex is addressable by VID over RDMA.
    //
    // Only honored when use_query_entry_partition is set, and only for the
    // centroid seed -- the frozen-point seed stays local (it is replicated in
    // every partition, so it is already node-independent).
    uint32_t query_partition = 0;
    // Read query_partition instead of the local partition for the centroid
    // seed. Changes which vertices seed the beam -> different search
    // trajectory, so this is recall-affecting (not recall-neutral like the
    // Python-side query router). Default off pending paired recall runs.
    bool use_query_entry_partition = false;
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
    // P1 item 6: in the batch (CAP) path, process CQ completions per owner in
    // arrival order interleaved with distance compute (port of the sequential
    // Pass-3 round-robin), instead of the bulk-sync drain barrier. Absorbs the
    // CQ busy-poll spin into useful compute. Queue-insertion order across
    // owners then follows completion arrival, so results are order-
    // nondeterministic (recall-equivalent, NOT byte-identical). Default off.
    bool batch_interleave_completion = false;
    // P1b: in the batch (CAP) path the owner loop follows neighbor-discovery
    // order, and with ~94% of neighbors local the local owner is almost always
    // first -- so local distance compute finishes BEFORE any remote fetch is
    // posted and the RTT is fully exposed. This moves the local owner last, so
    // the hop becomes post-remote -> compute-local -> drain-remote, which is
    // what the overlap design intended. Caveat: the batch PQ gate thresholds on
    // the current queue worst, which the local compute would otherwise have
    // tightened first, so the filter runs slightly looser -> recall-equivalent,
    // NOT byte-identical. Default off until measured.
    bool batch_remote_first = false;
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

    // FrozenSeedCache: serve the frozen point's REMOTE neighbor seeds from a
    // process-local copy instead of re-issuing one blocking post_read + 5ms-
    // timeout wait per neighbor per query.
    //
    // The frozen point is the one vertex updates never touch, and it is
    // replicated in every partition, so every query seeds from the SAME ~R
    // neighbor VIDs and re-fetches the SAME immutable vectors. Under
    // entry_mode=1 (FROZEN_ONLY, the current search convention) that is the
    // entire seed phase: ~65 serialized RTTs before the first hop.
    //
    // Warmed once per process (std::call_once) from the first query that
    // reaches the frozen block; afterwards the map is read-only and lock-free.
    // A neighbor whose warm fetch failed is simply absent -> that VID falls
    // through to the unmodified RDMA path, so a partial warm is safe.
    //
    // Recall-neutral: identical bytes, identical distances, identical queue
    // insertion order. Only WHERE the seed vector is read from changes.
    //
    // Caveat: the cached copy is not invalidated. Correct for search-only
    // workloads (the vertices are immutable there); if consolidation compacts
    // the store and reuses a cached neighbor's local_loc, the cached vector and
    // its cached tag go stale. The update path pins this to false for that
    // reason (update/src/insertion.cpp).
    bool frozen_seed_cache = true;

    // Optional search-warming hook for the consolidation cache. Default null
    // keeps search behavior and hot-path overhead unchanged.
    ConsolidationCacheWarmer* consolidation_cache_warmer = nullptr;
};

} // namespace danns
