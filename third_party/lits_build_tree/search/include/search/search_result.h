#pragma once
// search/search_result.h — Search result structure
// Extracted from: distributed_bindings.cpp L3632-3668

#include <cstdint>
#include <vector>
#include <utility>
#include "trace/trace_types.h"

namespace danns {

template <typename T>
struct SearchResult {
    std::vector<uint32_t> indices;
    std::vector<float> distances;
    uint64_t n_hops = 0, n_cmps = 0;
    uint64_t local_vector_accesses = 0, remote_vector_accesses = 0;
    uint64_t local_neighbor_accesses = 0, remote_neighbor_accesses = 0;
    uint64_t resolve_fail = 0, visited_count = 0;
    long long rdma_vector_ns = 0, rdma_neighbor_ns = 0;
    uint64_t rdma_entry_calls = 0, rdma_entry_items = 0;
    uint64_t rdma_vec_fetch_calls = 0, rdma_nbr_expand_calls = 0;
    uint64_t rdma_full_entry_fetches = 0, rdma_vec_only_fetches = 0;
    uint64_t convergence_hop = 0, n_batch_rounds = 0;
    uint64_t n_batch_candidates_total = 0, fallback_count = 0;
    uint64_t pq_filtered = 0;
    uint64_t unresolved_vids = 0;
    // CPU breakdown (nanoseconds)
    long long cpu_visited_ns = 0, cpu_resolve_ns = 0;
    long long cpu_distance_ns = 0, cpu_queue_ns = 0;
    long long cpu_nbr_loop_ns = 0, cpu_map_ns = 0, cpu_pq_ns = 0;
    // PQ LUT build (M * 256 * chunk_size FP ops, runs once per query on convergence)
    long long lut_build_ns = 0;
    // Candidate-fetch phase wall-clock: Pass 1 (remote post/drain) through Pass 3 (remote dist)
    long long candidate_fetch_phase_ns = 0;
    // Per-pass wall-clock (subset of candidate_fetch_phase_ns):
    //   pass1 = remote owner async post_read_batch_async
    //   pass2 = local owner distance compute
    //   pass3 = CQ drain check + remote distance compute
    long long pass1_ns = 0, pass2_ns = 0, pass3_ns = 0;
    // Pass 3 internal breakdown:
    //   pass3_poll_ns    = time spent inside poll_read_nonblocking calls
    //                      (usually small after the bulk-sync drain)
    //   pass3_compute_ns = time spent in per-entry distance + map
    long long pass3_poll_ns = 0, pass3_compute_ns = 0;
    // Converge batch phase async polling (multi-QP path):
    //   batch_nbr_poll_ns = poll wait for deferred neighbor-list fetches
    //   batch_vec_poll_ns = poll wait for deferred vector fetches
    long long batch_nbr_poll_ns = 0, batch_vec_poll_ns = 0;
    // Trace data
    std::vector<HopRecord> trace_hops;
    std::vector<std::vector<FetchedEntry>> trace_fetched;
    std::vector<ResultEntry> trace_results;
    uint16_t result_count = 0;
    // Full candidate pool
    std::vector<std::pair<uint64_t, float>> pool;
};

} // namespace danns
