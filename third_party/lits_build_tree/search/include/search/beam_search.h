#pragma once
// search/beam_search.h — ColocationBeamSearch interface
// Extracted from: distributed_bindings.cpp L3672-5861
//
// The core beam search operates in VectorID space over colocation entries.
// It depends on: NodeStore fetchers, LatencyStats, SearchTraceWriter,
// PQFilter, PartitionMapper, and a Distance<T> function.

#include <cstdint>
#include <map>
#include <mutex>
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

private:
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
