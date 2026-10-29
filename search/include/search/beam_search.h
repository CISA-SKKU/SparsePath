#pragma once
// search/beam_search.h — ColocationBeamSearch interface
// Extracted from: distributed_bindings.cpp L3672-5861
//
// The core beam search operates in VectorID space over colocation entries.
// It depends on: NodeStore fetchers, LatencyStats, SearchTraceWriter,
// PQFilter, PartitionMapper, and a Distance<T> function.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "search/search_config.h"
#include "search/search_result.h"
#include "store/colocation_store.h"
#include "core/priority_queue.h"
#include "core/latency_stats.h"
#include "core/vector_id.h"
#include "trace/search_trace.h"

#include "distance.h"  // DiskANN Distance<T> full definition

namespace danns {

class NodeStore;
class PartitionMapper;
class PQFilter;

class ColocationBeamSearch {
public:
    ColocationBeamSearch() = default;

    // Single-query beam search (thread-safe with per-thread resources)
    template <typename T>
    SearchResult<T> search(
        const T* aligned_query,
        size_t aligned_dim,
        size_t cmp_dim,
        diskann::Distance<T>* dist_fn,
        const SearchConfig& config,
        // Resources (per-thread or shared)
        NodeStore* node_store,
        PartitionMapper* mapper,
        PQFilter* pq_filter,
        std::map<uint32_t, std::mutex>* fetcher_mutexes,
        // Multi-QP state
        int num_threads,
        int thread_id);

    // Batch search (multi-threaded via OpenMP)
    template <typename T>
    std::vector<SearchResult<T>> search_batch(
        const T* queries,
        const uint32_t* query_ids,
        uint32_t num_queries,
        size_t vector_dim,
        size_t aligned_dim,
        size_t cmp_dim,
        diskann::Distance<T>* dist_fn,
        const SearchConfig& config,
        NodeStore* node_store,
        PartitionMapper* mapper,
        PQFilter* pq_filter,
        SearchTraceWriter* trace_writer,
        int num_threads);

    // Hop-interleaved multi-query search for ONE group of queries on ONE thread.
    //
    // Drives `group_size` queries concurrently, interleaving their per-hop RDMA
    // so up to `group_size` reads are outstanding per thread. Each query's beam
    // search is computed identically to search() — only the *timing* of RDMA
    // waits is reordered — so per-query results are bit-identical to the
    // per-query path. Returns one SearchResult per query (in group order).
    //
    // Supports ONLY the multi-QP + SEPARATED fetch path with all per-query
    // optimizations off (no PQ filter, no batch prefetch, no trace, no
    // consolidation-cache warming). Callers must verify feasibility (see
    // interleave_feasible) before invoking.
    template <typename T>
    std::vector<SearchResult<T>> search_interleaved(
        const T* const* aligned_queries,   // group_size pointers, each aligned_dim long
        const uint32_t* query_ids,         // group_size query ids (for stats only)
        uint32_t group_size,
        size_t aligned_dim,
        size_t cmp_dim,
        diskann::Distance<T>* dist_fn,
        const SearchConfig& config,
        NodeStore* node_store,
        PartitionMapper* mapper,
        int num_threads,
        int thread_id);

    // True if the interleaved path can serve this configuration. Checks fetch
    // mode, multi-QP availability, optimization flags, and that the per-thread
    // fetch buffer is large enough to slice into `interleave_factor` regions of
    // at least slack_R entries each. When false, callers use the per-query path.
    static bool interleave_feasible(
        const SearchConfig& config,
        NodeStore* node_store,
        PartitionMapper* mapper,
        PQFilter* pq_filter,
        int num_threads);

    // ---- FrozenSeedCache diagnostics (see SearchConfig::frozen_seed_cache) ----
    // Number of remote frozen neighbors that were successfully warmed. 0 means
    // the cache is cold, disabled, or every frozen neighbor is local.
    size_t frozen_seed_cache_entries() const { return frozen_seed_cache_.slot.size(); }
    // Process-wide tally of seeds served from the cache (one relaxed add per
    // query, not per hit — the per-query count rides in SearchResult).
    uint64_t frozen_seed_cache_hits() const {
        return frozen_seed_cache_.hits.load(std::memory_order_relaxed);
    }

    // Drop the cache and re-arm the one-shot warm, so the next query that
    // reaches the frozen block re-reads every remote frozen neighbor.
    // Returns how many entries were dropped (diagnostic).
    //
    // This is the staleness containment boundary. The cached bytes are keyed by
    // VID = (partition | local_loc); delete + insert + consolidate can hand a
    // local_loc to a different vector (and local_to_tag a different tag), which
    // would make the cached copy silently wrong. Calling this at the start of
    // every search op bounds the cache's lifetime to one op, so it can never
    // span a mutation — the search path needs zero coupling to the update path.
    // Cost is one re-warm per op (~R blocking reads, once), not per query.
    //
    // PRECONDITION: no search may be in flight. Reads of the cache are lock-free
    // by design; this is the one writer, and it is only safe at an op boundary
    // (which is where pipeline/ops/search/op.py calls it, before run_l_sweep).
    // A workload that runs consolidation CONCURRENTLY with search (ops/
    // concurrent.py) has no such boundary — disable the cache there via
    // `distributed_search.frozen_seed_cache: false`.
    size_t reset_frozen_seed_cache();

private:
    // ------------------------------------------------------------------
    // FrozenSeedCache — process-local copy of the frozen point's REMOTE
    // neighbor vectors.
    //
    // Lifetime: ONE SEARCH OP. Warmed once (std::call_once) by the first query
    // that reaches the frozen seeding block, then strictly read-only until
    // reset_frozen_seed_cache() re-arms it at the next op boundary. Reads need
    // no lock; the happens-before edge comes from call_once itself, and callers
    // must only consult the cache AFTER their own call_once returns (see the
    // fsc_ready flag in beam_search.cpp).
    //
    // The flag is held by pointer because std::once_flag is neither resettable
    // nor movable: re-arming = installing a fresh one.
    // ------------------------------------------------------------------
    struct FrozenSeedCache {
        std::unique_ptr<std::once_flag> warm_flag = std::make_unique<std::once_flag>();
        // Backing storage, over-allocated by one alignment quantum so `base`
        // can be pushed to a 64B boundary (entry_size is not necessarily a
        // multiple of 64, and the distance kernels prefer aligned loads).
        std::vector<char> storage;
        char* base = nullptr;
        size_t stride = 0;                                  // bytes per slot
        std::unordered_map<uint64_t, uint32_t> slot;        // edge VID -> slot
        std::unordered_map<uint64_t, uint64_t> entry_vid;   // edge VID -> (partition|tag)
        std::atomic<uint64_t> hits{0};
        bool active = false;   // published by the warming thread, last write
    };
    FrozenSeedCache frozen_seed_cache_;

    // One-shot warm. Type-independent: copies raw vector bytes, so it is a
    // plain (non-template) member. Individual failures are skipped, leaving
    // that VID on the unmodified RDMA path.
    void warm_frozen_seed_cache(
        const uint64_t* nbr_vids,
        uint32_t degree,
        NodeStore* node_store,
        PartitionMapper* mapper,
        bool use_multi_qp,
        int thread_id);

    // Resolve VID to (owner_node, local_loc, valid)
    static std::tuple<uint32_t, uint32_t, bool> resolve_vid(
        uint64_t vid,
        NodeStore* node_store,
        PartitionMapper* mapper);

    // LITS filter for expanded candidate pool
    template <typename T>
    static void filter_lits_expanded_pool(
        std::vector<std::pair<uint64_t, float>>& pool,
        uint32_t my_partition);
};

} // namespace danns
