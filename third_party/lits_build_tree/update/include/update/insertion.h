#pragma once
// update/insertion.h — Distributed insertion engine
//
// Implements DiskANN insert_point() algorithm adapted for distributed
// colocation layout with RDMA. Uses the same RobustPrune as consolidation
// (via prune_utils.h) and ColocationBeamSearch for neighbor discovery.
//
// Algorithm per vector (matching DiskANN index.cpp insert_point L5523-5683):
//   1. Register tag/loc mapping (PartitionMapper)
//   2. Write vector + entry_vid to local colocation entry
//   3. Beam search from frozen point → candidate pool
//   4. RobustPrune → forward edges
//   5. Write forward edges to colocation entry
//   6. distributed_inter_insert → reverse edges (local + RDMA)

#include <cstdint>
#include <iterator>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/vector_id.h"

#include "distance.h"  // DiskANN Distance<T>

namespace danns {

// Forward declarations
class NodeStore;
class PartitionMapper;
class ColocationBeamSearch;
class PQFilter;
class RedirectAlloc;

// ========================================================================
// InsertionConfig
// ========================================================================

struct InsertionConfig {
    uint32_t R = 64;               // Max degree
    uint32_t L = 100;              // Beam width for neighbor search
    float alpha = 1.2f;            // RobustPrune alpha
    uint32_t maxc = 750;           // Max candidate pool size
    uint32_t slack_R = 84;         // ceil(GRAPH_SLACK_FACTOR * R), padded neighbor capacity
    uint32_t entry_mode = 1;       // FROZEN_ONLY (DiskANN insert parity)
    uint32_t num_threads = 1;      // OpenMP threads for batch insert
    std::string trace_path;        // Empty = no trace
    bool lits_filter = false;      // Apply LITS (long∧inter) filter on forward pool
    uint32_t lits_my_partition = 0;// Partition used for the inter judgment
    // LoadBalancer (Stage B, measure-only): compute m_assign from the search
    // pool via deterministic argmin over peer vector counters, but never
    // redirect — only record redirect_considered/redirect_would_happen.
    bool load_balance = false;
    double lb_threshold = 0.7;     // candidate cut: v_i >= lb_threshold * v_max
    // Live redirect: act on the verdict in-batch — request a final slot from
    // m_assign's alloc server, prune on m_search, RDMA-write the finished
    // entry to the final home (one search per point). false = Stage B
    // measure-only behavior exactly.
    bool redirect = false;
    uint32_t redirect_timeout_ms = 500;  // alloc request timeout (drillable)
};

// ========================================================================
// InsertionResult — per-vector result
// ========================================================================

struct InsertionResult {
    uint32_t local_loc = 0;
    uint32_t tag = 0;
    uint32_t forward_degree = 0;         // Number of forward edges created
    uint32_t reverse_appends = 0;        // Reverse edges added by simple append
    uint32_t reverse_prunes = 0;         // Reverse edges that triggered re-prune
    // LITS filter counters (per-vector, forward pool)
    uint64_t lits_checked = 0;           // Candidates examined by LITS filter
    uint64_t lits_filtered = 0;          // Candidates removed (long∧inter)
    // LoadBalancer verdict (per-vector, measure-only)
    bool redirect_considered = false;    // >= 2 candidate owners had counters
    bool redirect_would_happen = false;  // owner(m_assign) != my node
    uint32_t lb_m_assign = 0;            // chosen partition (= my_partition unless redirect)
    bool redirected = false;             // live redirect happened (final home = m_assign)
    bool redirect_fallback = false;      // alloc FAIL/timeout — stayed on m_search
    uint32_t redirect_final_loc = 0;     // final local_loc on m_assign (when redirected)
    // RDMA access counters (per-vector)
    uint64_t local_vec_reads = 0;
    uint64_t remote_vec_reads = 0;
    uint64_t local_nbr_reads = 0;
    uint64_t remote_nbr_reads = 0;
    uint64_t local_nbr_writes = 0;
    uint64_t remote_nbr_writes = 0;
    // Timing (nanoseconds)
    uint64_t search_ns = 0;
    uint64_t prune_ns = 0;
    uint64_t write_ns = 0;
    uint64_t reverse_edge_ns = 0;
};

// ========================================================================
// InsertionStats — batch-level statistics
// ========================================================================

struct InsertionStats {
    uint32_t inserted = 0;
    uint32_t skipped = 0;
    uint32_t failed = 0;
    // LITS filter counters (batch totals, forward pool)
    uint64_t lits_checked = 0;
    uint64_t lits_filtered = 0;
    // LoadBalancer (batch totals, measure-only)
    uint64_t redirect_considered = 0;
    uint64_t redirect_would_happen = 0;
    // Live redirect ledger (origin side): every point this batch moved.
    // Python settlement releases old_loc and retracts the provisional
    // mapping; the final mapping is published by the TARGET (adopt).
    struct RedirectOut {
        uint32_t tag;
        uint32_t old_loc;     // provisional slot on m_search (to reclaim)
        uint32_t fpart;       // m_assign
        uint32_t floc;        // final slot on m_assign
    };
    std::vector<RedirectOut> redirected;
    uint64_t redirect_fallback = 0;      // alloc FAIL/timeouts (stayed local)
    // Counter snapshot read at batch start: node_id -> live vector count
    // (std::map for deterministic report order)
    std::map<uint32_t, uint64_t> lb_counter_snapshot;
    // RDMA access counters
    uint64_t local_vec_reads = 0;
    uint64_t remote_vec_reads = 0;
    uint64_t local_nbr_reads = 0;
    uint64_t remote_nbr_reads = 0;
    uint64_t local_nbr_writes = 0;
    uint64_t remote_nbr_writes = 0;
    // Timing breakdown (seconds)
    double search_sec = 0;
    double prune_sec = 0;
    double write_sec = 0;
    double reverse_sec = 0;
    double elapsed_sec = 0;
    // Per-vector detail (optional, populated when trace enabled)
    std::vector<InsertionResult> per_vector;

    // Combine a thread-local partial into this one (parallel insert_batch):
    // scalars +=, vectors append. NOT merged: lb_counter_snapshot (read once
    // at batch start, set on the final stats outside the parallel region)
    // and elapsed_sec (whole-batch wall time, set after the region).
    void merge(InsertionStats&& o) {
        inserted += o.inserted;
        skipped += o.skipped;
        failed += o.failed;
        lits_checked += o.lits_checked;
        lits_filtered += o.lits_filtered;
        redirect_considered += o.redirect_considered;
        redirect_would_happen += o.redirect_would_happen;
        redirect_fallback += o.redirect_fallback;
        local_vec_reads += o.local_vec_reads;
        remote_vec_reads += o.remote_vec_reads;
        local_nbr_reads += o.local_nbr_reads;
        remote_nbr_reads += o.remote_nbr_reads;
        local_nbr_writes += o.local_nbr_writes;
        remote_nbr_writes += o.remote_nbr_writes;
        search_sec += o.search_sec;
        prune_sec += o.prune_sec;
        write_sec += o.write_sec;
        reverse_sec += o.reverse_sec;
        redirected.insert(redirected.end(),
                          std::make_move_iterator(o.redirected.begin()),
                          std::make_move_iterator(o.redirected.end()));
        per_vector.insert(per_vector.end(),
                          std::make_move_iterator(o.per_vector.begin()),
                          std::make_move_iterator(o.per_vector.end()));
    }
};

// ========================================================================
// Insertion — main insertion engine
// ========================================================================

class Insertion {
public:
    Insertion() = default;

    /// Batch insert: DiskANN insert_point() algorithm, distributed version.
    ///
    /// Caller must pre-allocate slots (tags + local_locs) and update
    /// PartitionMapper tag/loc mappings BEFORE calling this.
    ///
    /// @param config         Insertion parameters
    /// @param node_store     Local colocation store + peer fetchers
    /// @param mapper         Partition/node mapping + deletion tracking
    /// @param beam_search    Beam search engine (reused from search module)
    /// @param pq_filter      PQ pre-filter (may be nullptr if disabled)
    /// @param dist_fn        Distance function (L2, cosine, etc.)
    /// @param vector_dim     Original vector dimension
    /// @param aligned_dim    Aligned dimension for SIMD (>= vector_dim)
    /// @param vectors        Contiguous array of N vectors (N x vector_dim)
    /// @param tags           Array of N tags (1-based, matching DiskANN convention)
    /// @param local_locs     Array of N pre-allocated local locations
    /// @param count          Number of vectors to insert
    /// @param my_partition   This node's partition ID
    /// @param special_vids   VIDs to exclude from edges (frozen + centroid entries)
    /// @param redirect_alloc Live-redirect alloc client (nullptr = no redirect)
    template <typename T>
    InsertionStats insert_batch(
        const InsertionConfig& config,
        NodeStore* node_store,
        PartitionMapper* mapper,
        ColocationBeamSearch* beam_search,
        PQFilter* pq_filter,
        diskann::Distance<T>* dist_fn,
        uint32_t vector_dim,
        uint32_t aligned_dim,
        const T* vectors,
        const uint32_t* tags,
        const uint32_t* local_locs,
        uint32_t count,
        uint32_t my_partition,
        const std::unordered_set<uint64_t>& special_vids,
        RedirectAlloc* redirect_alloc = nullptr);

private:
    /// Single vector insert (DiskANN insert_point equivalent).
    template <typename T>
    InsertionResult insert_single(
        const InsertionConfig& config,
        NodeStore* node_store,
        PartitionMapper* mapper,
        ColocationBeamSearch* beam_search,
        PQFilter* pq_filter,
        diskann::Distance<T>* dist_fn,
        uint32_t vector_dim,
        uint32_t aligned_dim,
        const T* vector,
        uint32_t tag,
        uint32_t local_loc,
        uint32_t my_partition,
        const std::unordered_set<uint64_t>& special_vids,
        std::map<uint32_t, std::mutex>& fetcher_mutexes,
        std::vector<std::mutex>& local_locks,
        const std::unordered_map<uint32_t, uint64_t>* lb_counters,
        RedirectAlloc* redirect_alloc,
        int thread_id = 0);

    /// Distributed inter-insert: add reverse edges to each forward neighbor.
    /// Equivalent to DiskANN inter_insert() but works across partitions via RDMA.
    ///
    /// For each neighbor VID in pruned_vids:
    ///   - If local: read neighbors from mmap, append or prune, write back
    ///     (under the per-stripe local_locks shard lock)
    ///   - If remote: RDMA read neighbors, append or prune, RDMA write back
    template <typename T>
    void distributed_inter_insert(
        uint64_t new_vid,
        const T* new_vec,
        const std::vector<uint64_t>& pruned_vids,
        const InsertionConfig& config,
        NodeStore* node_store,
        PartitionMapper* mapper,
        diskann::Distance<T>* dist_fn,
        size_t cmp_dim,
        uint32_t my_partition,
        const std::unordered_set<uint64_t>& special_vids,
        std::map<uint32_t, std::mutex>& fetcher_mutexes,
        std::vector<std::mutex>& local_locks,
        InsertionResult& result);
};

} // namespace danns
