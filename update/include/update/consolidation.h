#pragma once
// update/consolidation.h — Consolidation engine
// Extracted from: distributed_bindings.cpp L7418-8509
//
// 2-hop expansion + RobustPrune + multi-threaded via OpenMP.
// Manages per-deleted-node cache state and RDMA fetchers.

#include <cstdint>
#include <utility>
#include <vector>
#include <unordered_set>
#include <string>

#include "core/vector_id.h"
#include "trace/consolidation_trace.h"
#include "update/lits_policy.h"

#include "distance.h"  // DiskANN Distance<T> — full definition needed for ->compare()

namespace danns {

class NodeStore;
class PartitionMapper;
class ConsolidationCache;

struct ConsolidationConfig {
    uint32_t R = 64;               // Maximum degree
    float alpha = 1.2f;            // RobustPrune alpha
    uint32_t maxc = 750;           // Max candidates in expanded pool
    uint32_t prefetch_batch = 8;
    float forward_threshold = 0.0f;
    uint32_t num_threads = 1;
    std::string trace_path;        // Empty = no trace
    bool lits_filter = false;
    uint32_t lits_my_partition = 0;
    // LITS-Aware-Update tau rule: include the point's DELETED original
    // neighbors in the tau (mean-distance threshold) sample. With a large
    // deletion buffer a survivors-only tau rests on a tiny sample (and the
    // filter shuts off entirely when every neighbor is deleted); the full
    // original list keeps the sample at ~R regardless of deletion ratio.
    // false reproduces the old survivors-only tau (A/B toggle).
    bool lits_tau_include_deleted = true;
    // Per-deleted-node cache: group cache by deleted Di, store its neighbor
    // list + vectors contiguously (CSR). Avoids the global unique-vid merge
    // bottleneck and gives sequential prune access. Per-call lifetime.
    bool enable_vec_cache = false;
    // De-overlapped ("sync fetch") baseline for the OFF path.
    //
    // The remote VECTOR fetch normally goes through post_read_batch(), which is
    // blocking at the CALL boundary but internally a pipeline: it keeps up to
    // DEFAULT_SQ_DEPTH/2 = 256 RDMA READ WRs in flight and interleaves posting
    // new WRs with polling completions, so per-read RTT is almost entirely
    // hidden behind other reads. With 32 threads x 256 WRs a node can have
    // thousands of reads outstanding, which is why a multi-TB fetch volume
    // exposes only a few hundred seconds of wall clock.
    //
    // sync_fetch=true replaces that batch with one read at a time, each drained
    // to completion before the next is posted: every RTT is fully exposed and
    // no fetch overlaps any other fetch or any compute within a thread. Results
    // are bit-identical (same bytes, same buffer offsets, same memcpy loop) —
    // only the wall clock changes. This is for presenting a serial-fetch
    // baseline, not for production.
    //
    // Applies to BOTH arms (2026-08-24; was OFF-path-only before that).
    // With the cache fetch interceptor active (enable_vec_cache), a cache HIT
    // never reaches a fetch site — it is served from the arena — so the only
    // reads this can serialize there are the interceptor's MISSES. That is the
    // point: it makes a controlled "naive vs naive+cache" pair possible, where
    // both arms issue their remaining reads unpipelined and the wall ratio
    // reflects the cache rather than a batching difference between the arms.
    // Since misses are ~0.7% of the OFF arm's reads, expect the ON arm's wall
    // to move very little. The per-Di path (DANNS_PERDI_CACHE) is untouched.
    //
    // SUPERSEDED by fetch_window (below), which subsumes it: sync_fetch=true is
    // exactly fetch_window=1. Kept for backward compatibility; behaviour and
    // log lines are unchanged when it is used on its own.
    bool sync_fetch = false;
    // Fetch in-flight window depth (sliding window K) for the remote VECTOR
    // fetch. Generalises sync_fetch from a two-state switch into a knob that
    // sweeps the whole "pipelining depth" axis between its two extremes
    // (deep100M/8p measured: naive 4,662 s <-> batched 1,175 s, a 3.97x cliff).
    //
    //   K = 0  : unlimited -> the historical batched post (up to 256 WRs in
    //            flight, posting interleaved with polling). Default; identical
    //            to sync_fetch=false in every byte and every log line.
    //   K = 1  : one read outstanding at a time, drained before the next is
    //            posted. IDENTICAL to sync_fetch=true -- it dispatches to the
    //            very same post_read / wait_send_completion code (same wr_id,
    //            same 10 s per-read budget, same rdma_wait_fail counter, same
    //            throw), so the measured naive baseline IS the K=1 point of the
    //            curve rather than a near-neighbour of it.
    //   K >= 2 : strict sliding window -- post until K are in flight, then post
    //            exactly one more per completion reaped. Runs the SAME
    //            post_read_batch loop as K=0 with its flow-control cap lowered
    //            from 256 to K, so out-of-order completion handling, error
    //            handling and timeout semantics are the batched path's own, not
    //            a reimplementation. Clamped to the SQ cap (256), i.e. K >= 256
    //            converges to the K=0 behaviour.
    //
    // Results are bit-identical for every K: each WR's destination address is
    // fixed by the SGL at POST time, so the window changes only WHEN a read is
    // issued, never WHERE its bytes land, and the fetch site still returns only
    // after every read of the chunk has completed. Only wall clock moves.
    //
    // Precedence: fetch_window > 0 wins and sync_fetch is ignored (logged
    // explicitly). fetch_window == 0 with sync_fetch=true maps to K = 1.
    //
    // Scope matches sync_fetch exactly: the remote VECTOR fetch on BOTH arms
    // (cache OFF: all reads; cache ON: the interceptor's MISS reads, since a
    // hit never reaches a fetch site). The Di 2-hop NEIGHBOR fetch is already
    // fully synchronous on both branches and is untouched.
    int fetch_window = 0;
    // Per-thread buffer arena for staged remote vectors (OFF path).
    //
    // Default (false) keeps the historical allocation shape: every fetched
    // remote vector gets its OWN heap block
    // (remote_vec_storage.emplace_back(cmp_dim) => one operator new + one
    // value-init zero-fill + one operator delete, per vector). On deep100M/8p
    // with a 10M-delete workload that is ~4.5e9 malloc/free pairs and ~1.7 TB
    // of zero-stores that are immediately overwritten by the memcpy. None of it
    // is inside any consolidation phase timer, so it shows up only as wall
    // clock -- which makes the cache ON/OFF wall ratio partly an allocator
    // measurement rather than a network measurement.
    //
    // true: each consolidation thread keeps ONE growable arena of fixed-stride
    // slots and hands out slot pointers instead. Steady state is zero heap
    // traffic per fetch; the arena grows geometrically and never shrinks.
    // Results are bit-identical (same bytes copied, same vec_store contents,
    // same prune input) -- only the wall clock changes.
    //
    // OFF-path only: ignored when the cache fetch interceptor is active
    // (enable_vec_cache), so the cache ON arm keeps its historical allocation
    // behaviour and this rung measures the cache rather than the allocator.
    // Deliberately unlike sync_fetch, which DOES reach the ON arm's miss
    // fetches. Otherwise orthogonal to it; all four combinations are valid.
    bool reuse_fetch_buffers = false;
    // Pass 2 (prune) thread count override for static-cache path. 0 = use
    // num_threads. Smaller values trade single-thread memory bandwidth
    // (less DRAM channel contention) for fewer parallel cores. The Pass 2
    // hot loop is purely memory-bound (random reads on multi-GB
    // vec_storage); 32T saturates memory channels → per-thread effective
    // BW drops ~9x. Sweeping this lets us pick the sweet spot.
    uint32_t pass2_num_threads = 0;
    // Per-vector fetch-count dump path (binary). Empty = disabled.
    // When set, the OFF (no-cache) consolidation path tracks how many times
    // each vector is accessed (local + remote) during this consolidation
    // and writes a dense per-partition counter array at the end. Format:
    //   magic[4]="VFCT", num_partitions:u32,
    //   then per partition: partition_id:u32, n_locs:u32, counts:u32[n_locs]
    std::string vec_fetch_count_dump_path;
    // Optional search-warmed deleted-node cache. When set, the per-Di
    // consolidation path first probes this cache for Di neighbor lists and
    // Di-scoped neighbor vector payloads, then falls back to normal fetches.
    ConsolidationCache* consolidation_cache = nullptr;
};

// Test-only: exercises the consolidation LITS filter end-to-end, including
// the tau-include-deleted contribution (tau_extra_sum/tau_extra_count). Wraps
// LITSFilter::filter_consolidation_candidates so the Python oracle-parity test
// can validate the exact filter implementation. Returns {checked, filtered}.
std::pair<uint64_t, uint64_t> filter_lits_consolidation_candidates_for_test(
    std::vector<std::pair<uint64_t, float>>& candidates,
    const std::unordered_set<uint64_t>& orig_nbr_vids,
    uint32_t my_partition, double tau_extra_sum, uint32_t tau_extra_count);

struct ConsolidationStats {
    uint64_t entries_processed = 0;
    uint64_t entries_modified = 0;
    uint64_t entries_pruned = 0;
    uint64_t total_deleted_nbrs = 0;
    uint64_t total_expanded = 0;
    uint64_t local_vec_accesses = 0;
    uint64_t remote_vec_accesses = 0;
    uint64_t local_nbr_accesses = 0;
    uint64_t remote_nbr_accesses = 0;
    uint64_t cache_hits = 0;
    uint64_t cache_misses = 0;
    uint64_t cache_neighbor_hits = 0;
    uint64_t cache_neighbor_misses = 0;
    uint64_t cache_vector_hits = 0;
    uint64_t cache_vector_misses = 0;
    // OFF-path interceptor: fetched vector misses pushed back into the warm
    // store (attempts; includes records deduped/dropped inside the store).
    uint64_t cache_writebacks = 0;
    // Fetched Di neighbor-list misses published back into the nbr store
    // (successful publishes only; drops/duplicates are not counted).
    uint64_t cache_nbr_writebacks = 0;
    uint64_t cache_entries = 0;
    uint64_t cache_warm_expansion_events = 0;
    uint64_t cache_warm_neighbor_records = 0;
    uint64_t cache_warm_vector_records = 0;
    uint64_t cache_warm_bytes = 0;
    uint64_t cache_buffer_neighbor_bytes = 0;
    uint64_t cache_buffer_vector_bytes = 0;
    uint64_t cache_compacted_neighbor_bytes = 0;
    uint64_t cache_compacted_vector_bytes = 0;
    uint64_t cache_compacted_index_bytes = 0;
    uint64_t cache_dropped_neighbor_records = 0;
    uint64_t cache_dropped_vector_records = 0;
    // Per-Di cache stats (non-zero only when enable_vec_cache=true).
    // unique_vec_fetches: #distinct deleted vertices cached as Di entries.
    // vec_bytes_fetched: actual RDMA bytes brought onto this node.
    // vec_bytes_saved_by_cache: (total_requests - unique) * dim * elem_size.
    // vec_buffer_peak_bytes: total static buffer size (held for the whole
    //                        consolidation; peak == steady-state size).
    uint64_t unique_vec_fetches = 0;
    uint64_t vec_bytes_fetched = 0;
    uint64_t vec_bytes_saved_by_cache = 0;
    uint64_t vec_buffer_peak_bytes = 0;
    // Lock stats (non-zero only on mutex-fallback path)
    uint64_t twohop_lock_wait_ns = 0;
    uint64_t vecfetch_lock_wait_ns = 0;
    uint64_t twohop_lock_contentions = 0;
    uint64_t vecfetch_lock_contentions = 0;
    bool use_multi_qp = false;
    double elapsed_sec = 0;
    // Wall time of just the parallel for body that does the actual prune
    // work — excludes scan, populate, vec_storage setup, vid_to_idx build.
    // For OFF path this equals the main loop wall (no separate setup).
    // For ON (cache) path this excludes the amortizable setup phase.
    // Use this metric for fair "algorithm cost" comparison vs DiskANN baseline.
    double loop_only_sec = 0;
    // Phase offsets relative to consolidate_start (seconds). Lets callers /
    // plotters slice the NIC BW timeline by phase. Only populated by the ON
    // (cache) path; OFF leaves them at 0 (single-phase body covers the wall).
    double scan_end_offset_sec = 0;
    double pop_start_offset_sec = 0;
    double pop_end_offset_sec = 0;
    double pass2_loop_start_offset_sec = 0;
    double pass2_loop_end_offset_sec = 0;
    // Degree histogram (degree -> count)
    std::vector<uint32_t> degree_histogram;
};

class Consolidation {
public:
    Consolidation() = default;

    // Run consolidation on local partition
    // TODO: Extract from distributed_bindings.cpp consolidate_local_typed_ (L7418-8509)
    template <typename T>
    ConsolidationStats run(
        const ConsolidationConfig& config,
        NodeStore* node_store,
        PartitionMapper* mapper,
        diskann::Distance<T>* dist_fn,
        const std::unordered_set<uint64_t>& deleted_vids,
        uint32_t vector_dim,
        uint32_t aligned_dim);
};

} // namespace danns
