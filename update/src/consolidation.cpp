// update/src/consolidation.cpp
// Extracted from: distributed_bindings.cpp L7291-8562
//
// Consolidation engine: 2-hop expansion + RobustPrune + multi-threaded via OpenMP.
// Manages per-deleted-node cache state and RDMA fetchers.

#include "update/consolidation.h"
#include "update/prune_utils.h"
#include "update/lits_policy.h"
#include "store/node_store.h"
#include "store/consolidation_cache.h"
#include "cluster/partition_mapper.h"
#include "core/vector_id.h"
#include "trace/consolidation_trace.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <thread>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <tsl/robin_set.h>
#include <tsl/robin_map.h>

#include <omp.h>
#include <parallel/algorithm>  // __gnu_parallel::sort
#include <sys/mman.h>

namespace danns {

// Test-only wrapper so the oracle-parity debug binding can exercise the exact
// consolidation LITS filter (including the tau-include-deleted contribution).
// Delegates to LITSFilter::filter_consolidation_candidates and re-packs its
// LITSFilterStats into the {checked, filtered} pair the Python test expects.
std::pair<uint64_t, uint64_t> filter_lits_consolidation_candidates_for_test(
    std::vector<std::pair<uint64_t, float>>& candidates,
    const std::unordered_set<uint64_t>& orig_nbr_vids,
    uint32_t my_partition, double tau_extra_sum, uint32_t tau_extra_count) {
    LITSFilterStats stats = LITSFilter::filter_consolidation_candidates(
        candidates, orig_nbr_vids, my_partition,
        tau_extra_sum, tau_extra_count);
    return {stats.checked, stats.filtered};
}

// write_neighbors_internal and robust_prune_internal are now in
// update/include/update/prune_utils.h — shared between consolidation and insertion.

// ========================================================================
// DeletedVidIndex — dense bitset replacement for `deleted_set.count()`
//
// RESULT-INVARIANT (2026-08-25). Pure membership-test acceleration: same
// answer for every VID, no algorithmic change, no reordering.
//
// WHY. The consolidation loop probes deleted-membership ~6.6e9 times per
// node on deep100M/8p 10%-delete:
//   * 1 per entry (self-deleted check)                        ~1.1e7
//   * 2 per 1-hop neighbor (has_deleted pre-scan + expand)    ~1.4e9
//   * 1 per 2-hop neighbor in the expansion inner loop        ~5.2e9
// The set handed down from the binding is a std::unordered_set<uint64_t>
// holding EVERY globally deleted VID (10M at 10% of 100M). That is ~300 MB
// of scattered nodes + an 80 MB bucket array, and libstdc++ resolves a
// bucket with a runtime `% prime` (a 64-bit hardware divide) followed by
// two DEPENDENT pointer chases. Under 64 threads all of it misses to DRAM,
// so a probe costs ~0.6 us and the aggregate lands in the ~5,000 thread-s
// that sit OUTSIDE every [CONSOLIDATE TIMING] scope.
//
// WHAT. VIDs are (partition[16b] | local_loc[48b]) drawn from a dense
// domain, so membership fits a flat bitset keyed by
//     partition * stride + local_loc,   stride = max_local_loc + 1
// sized from the set's own contents (no config coupling, no assumptions
// about slot counts). deep/laion 8p land at 8 * 15e6 bits = ~15 MB — small
// enough to live in LLC, and one non-dependent load per probe.
//
// EXACTNESS. Every deleted VID is inserted, so a set member always reads
// back 1. A VID outside the observed (partition, local_loc) envelope cannot
// be in the set, so the early-out returns the same `false` the hash lookup
// would (INVALID_VID decodes to partition 0xFFFF and takes that path).
// If sizing ever overflows the guard below, `bits_` stays null and every
// probe falls through to the original unordered_set — identical behaviour,
// no fast path.
// ========================================================================
class DeletedVidIndex {
public:
    // 2^33 bits = 1 GiB. deep/laion 8p need ~1.2e8; a domain past this cap
    // means the VID space is not dense and the bitset would waste more than
    // it saves, so we keep the hash set.
    static constexpr uint64_t kMaxBits = 1ull << 33;

    // bits_ points into storage_, so copying would dangle. The one instance
    // is a function-local const shared into the OMP region; never copied.
    DeletedVidIndex(const DeletedVidIndex&) = delete;
    DeletedVidIndex& operator=(const DeletedVidIndex&) = delete;

    DeletedVidIndex(const std::unordered_set<uint64_t>& s, int num_threads)
        : fallback_(&s) {
        if (s.empty()) return;

        // One serial walk of the hash set (its node chain is a pointer
        // chase either way); everything after this is parallel.
        std::vector<uint64_t> vids;
        vids.reserve(s.size());
        for (uint64_t v : s) vids.push_back(v);

        uint64_t max_part = 0, max_ll = 0;
        #pragma omp parallel for schedule(static) if(num_threads > 1) \
            reduction(max:max_part, max_ll)
        for (size_t i = 0; i < vids.size(); i++) {
            const uint64_t p = decode_partition(vids[i]);
            const uint64_t l = decode_local_loc(vids[i]);
            if (p > max_part) max_part = p;
            if (l > max_ll) max_ll = l;
        }

        stride_ = max_ll + 1;
        max_part_ = max_part;
        max_ll_ = max_ll;
        const uint64_t n_keys = (max_part + 1) * stride_;
        if (stride_ == 0 || n_keys / stride_ != (max_part + 1) ||
            n_keys > kMaxBits) {
            return;  // overflow or too sparse -> keep the hash-set path
        }

        words_ = (n_keys + 63) / 64;
        storage_.assign(words_, 0ull);
        uint64_t* w = storage_.data();
        #pragma omp parallel for schedule(static) if(num_threads > 1)
        for (size_t i = 0; i < vids.size(); i++) {
            const uint64_t k = decode_partition(vids[i]) * stride_ +
                               decode_local_loc(vids[i]);
            __atomic_fetch_or(&w[k >> 6], 1ull << (k & 63), __ATOMIC_RELAXED);
        }
        bits_ = w;
    }

    inline bool contains(uint64_t vid) const noexcept {
        if (__builtin_expect(bits_ == nullptr, 0))
            return fallback_->count(vid) != 0;
        const uint64_t p = decode_partition(vid);
        const uint64_t l = decode_local_loc(vid);
        if (p > max_part_ || l > max_ll_) return false;
        const uint64_t k = p * stride_ + l;
        return (bits_[k >> 6] >> (k & 63)) & 1ull;
    }

    bool dense() const noexcept { return bits_ != nullptr; }
    uint64_t bytes() const noexcept { return words_ * sizeof(uint64_t); }

private:
    const std::unordered_set<uint64_t>* fallback_;
    std::vector<uint64_t> storage_;
    const uint64_t* bits_ = nullptr;
    uint64_t words_ = 0;
    uint64_t stride_ = 0;
    uint64_t max_part_ = 0;
    uint64_t max_ll_ = 0;
};

// Forward decl for the per-deleted-node cache path. Group cache by Di →
// its R neighbors (vid + vec).
//
// DEPRECATED (2026-06-18): see the deprecation banner at the definition below.
// The consolidation cache is being re-implemented as a transparent fetch
// interceptor on the OFF/legacy inline path (hit -> skip RDMA, miss -> RDMA),
// so cache-on differs from cache-off ONLY in RDMA traffic, not as this separate
// per-Di restructured algorithm. Do not extend this path.
template <typename T>
static ConsolidationStats run_with_per_di_impl(
    const ConsolidationConfig& config,
    NodeStore* node_store,
    PartitionMapper* mapper,
    diskann::Distance<T>* dist_fn,
    const std::unordered_set<uint64_t>& deleted_set,
    uint32_t vector_dim,
    uint32_t aligned_dim);

// ========================================================================
// consolidate_local_typed: main consolidation loop
// Extracted from distributed_bindings.cpp L7418-8509
//
// Adapted from member function to free function taking NodeStore*,
// PartitionMapper*, etc. as parameters.
// ========================================================================
template <typename T>
ConsolidationStats Consolidation::run(
    const ConsolidationConfig& config,
    NodeStore* node_store,
    PartitionMapper* mapper,
    diskann::Distance<T>* dist_fn,
    const std::unordered_set<uint64_t>& deleted_set,
    uint32_t vector_dim,
    uint32_t aligned_dim)
{
    // --- Validate inputs ---
    if (!node_store || !node_store->local_store()) {
        throw std::runtime_error("consolidate: NodeStore or local_store not initialized");
    }
    if (!dist_fn) {
        throw std::runtime_error("consolidate: no distance function provided. "
                                 "Call load_diskann_index() or init_distance_only() first.");
    }

    // Dispatch: per-deleted-node vec cache path.
    // DEPRECATED (2026-06-18): this branch runs a SEPARATE per-Di algorithm
    // (prefetch every remote vector into one big vec_storage, then prune) — it is
    // NOT "the OFF algorithm minus RDMA". It conflates the cache effect with a
    // per-Di restructuring and materializes a multi-GB vec_storage (OOM on
    // high-remote layouts; net SLOWER than OFF despite high hit rate). Superseded
    // by cache-as-fetch-interceptor on the OFF path below. Kept for reference;
    // do not extend.
    // enable_vec_cache now routes HERE (not to the deprecated per-Di path): it
    // turns on the cache FETCH INTERCEPTOR in the OFF/inline path below. On a
    // hit, the vector is served from the search-warmed cache arena (no RDMA);
    // misses fall through to RDMA. So cache-on is the SAME algorithm as
    // cache-off and differs ONLY in RDMA traffic.
    // (To A/B against the old prefetch path, temporarily restore the
    //  run_with_per_di_impl<T>(...) call here; see its deprecation banner.)
    const bool cache_intercept =
        config.enable_vec_cache && (config.consolidation_cache != nullptr);

    // De-overlapped baseline switch. See ConsolidationConfig::sync_fetch.
    //
    // Scope: the remote VECTOR fetch only. The Di 2-hop NEIGHBOR fetch is
    // already fully synchronous on both branches (multi-QP does post_read +
    // wait_send_completion per Di; the shared-fetcher fallback does one
    // blocking rdma_read), so there is nothing there to de-overlap.
    //
    // Applies on BOTH arms (2026-08-24). With the cache interceptor on, the
    // hits never reach a fetch site at all — they are served from the arena and
    // `continue` — so the only thing this can touch there is the MISS fetch,
    // i.e. exactly the remote vector reads the cache failed to serve. That is
    // what makes a "naive vs naive+cache" control pair possible: both arms then
    // issue their remaining reads the same (unpipelined) way, and the ratio
    // reflects the cache instead of a batching difference between the arms.
    // The Di 2-hop NEIGHBOR fetch is already fully synchronous on both branches
    // (multi-QP does post_read + wait_send_completion per Di; the shared-fetcher
    // fallback does one blocking rdma_read), so there is nothing there to
    // de-overlap, on either arm.
    //
    // 2026-08-25: generalised to a sliding WINDOW of depth K
    // (ConsolidationConfig::fetch_window). sync_fetch is now the K=1 special
    // case and keeps its exact code path and log lines, so the already-measured
    // naive point is reusable verbatim as K=1.
    //
    //   K = 0   -> post_read_batch as before (256 in flight)   [default]
    //   K = 1   -> the sync_fetch loop below, byte-for-byte unchanged
    //   K >= 2  -> post_read_batch with its flow-control cap set to K
    //
    // Precedence: fetch_window wins over sync_fetch when both are set; the
    // legacy flag alone maps to K=1.
    int fetch_window = config.fetch_window;
    if (fetch_window < 0) fetch_window = 0;
    if (config.fetch_window > 0 && config.sync_fetch) {
        rdma::Logger::info("[CONSOLIDATE] both fetch_window=%d and sync_fetch=true set; "
                           "fetch_window takes precedence (sync_fetch IGNORED; it is "
                           "the K=1 special case of the same knob)",
                           config.fetch_window);
    } else if (config.fetch_window <= 0 && config.sync_fetch) {
        fetch_window = 1;
        rdma::Logger::info("[CONSOLIDATE] sync_fetch=true mapped to fetch_window=1 "
                           "(legacy alias; identical code path)");
    }
    // K=1 keeps the historical per-read post+drain implementation verbatim.
    const bool sync_fetch = (fetch_window == 1);
    // K>=2 reuses post_read_batch's own post/poll interleave with a shallower
    // in-flight cap. 0 there means "unlimited" (its historical 256).
    const uint32_t fetch_window_wr =
        (fetch_window >= 2) ? static_cast<uint32_t>(fetch_window) : 0u;
    rdma::Logger::info("[CONSOLIDATE] fetch_window=%d%s", fetch_window,
                       fetch_window == 0
                           ? " (unlimited: batched post, up to 256 reads in flight)"
                       : fetch_window == 1
                           ? " (one read outstanding; == legacy sync_fetch)"
                       : fetch_window >= 256
                           ? " (>= SQ cap 256: clamped, converges to unlimited)"
                           : " (sliding window: K in flight, 1 posted per completion)");
    if (sync_fetch) {
        if (cache_intercept) {
            rdma::Logger::info("[CONSOLIDATE] sync_fetch=ON (applies to cache-miss "
                               "fetches): cache hits are served from the arena "
                               "unchanged; every remote vector read the interceptor "
                               "could not serve is issued one at a time and drained "
                               "to completion");
        } else {
            rdma::Logger::info("[CONSOLIDATE] sync_fetch=ON (de-overlapped baseline): "
                               "remote vector reads are issued one at a time and "
                               "drained to completion; RDMA pipelining disabled");
        }
    } else if (fetch_window >= 2) {
        if (cache_intercept) {
            rdma::Logger::info("[CONSOLIDATE] fetch_window=%d (applies to cache-miss "
                               "fetches): cache hits are served from the arena "
                               "unchanged; the interceptor's misses are fetched with "
                               "at most %d reads in flight",
                               fetch_window, fetch_window);
        } else {
            rdma::Logger::info("[CONSOLIDATE] fetch_window=%d (throttled pipeline): "
                               "remote vector reads keep at most %d in flight; one "
                               "new read is posted per completion reaped",
                               fetch_window, fetch_window);
        }
    }

    // Buffer-reuse switch. See ConsolidationConfig::reuse_fetch_buffers.
    //
    // Scope: the staging buffer that each fetched REMOTE vector is memcpy'd
    // into before the prune reads it. Default is one fresh heap block per
    // vector; true serves them from a per-thread arena instead.
    //
    // Excluded when the cache interceptor is on: cache-ON must keep its
    // historical allocation behaviour so the ladder rung this switch exists for
    // ("OFF minus the allocator waste" vs an unmodified ON) measures the cache
    // and not the allocator. NOTE this is deliberately UNLIKE sync_fetch, which
    // does apply to the ON arm's miss fetches — the two knobs answer different
    // questions and are otherwise fully orthogonal (they touch different lines,
    // so all four combinations are valid).
    const bool reuse_fetch_buffers =
        config.reuse_fetch_buffers && !cache_intercept;
    if (config.reuse_fetch_buffers) {
        if (cache_intercept) {
            rdma::Logger::info("[CONSOLIDATE] reuse_fetch_buffers requested but IGNORED "
                               "(cache interceptor active; OFF-path-only switch)");
        } else {
            rdma::Logger::info("[CONSOLIDATE] reuse_fetch_buffers=ON: staged remote "
                               "vectors come from a per-thread arena; per-fetch "
                               "malloc/free eliminated (results bit-identical)");
        }
    }

    // EXPERIMENTAL (2026-06-29): per-Di "cache" = ZERO-COPY per-Di path.
    // Keeps per-Di's compact contiguous prune layout (fast loop ~215) but a
    // cache HIT is read at prune time straight from the search-warmed arena via
    // lookup_vector_ptr (no copy into vec_storage); ONLY misses are staged.
    // This fixes per-Di's two defects (serial hit-populate + multi-GB
    // vec_storage OOM) so it becomes a legitimate low-memory, cache-driven path.
    // Gated by env DANNS_PERDI_CACHE so the default cache-on path stays the
    // inline fetch interceptor below.
    if (cache_intercept && std::getenv("DANNS_PERDI_CACHE") != nullptr) {
        return run_with_per_di_impl<T>(config, node_store, mapper, dist_fn,
                                       deleted_set, vector_dim, aligned_dim);
    }
    (void)aligned_dim;  // now consumed only by the per-Di cache path

    // Compact the search-warmed Di neighbor lists into the lookup table the
    // 2-hop expand below probes. Without this the warmed lists are invisible
    // to this path and every remote Di adjacency goes to RDMA once per
    // referencing repair task.
    if (cache_intercept) {
        // Experiment infra: DANNS_CACHE_LOAD replaces the warm phase with a
        // state file dumped by a previous run (local SSD path — per-host);
        // DANNS_CACHE_DUMP persists the freshly compacted state for reuse.
        const char* cache_load = getenv("DANNS_CACHE_LOAD");
        bool state_loaded = false;
        if (cache_load && *cache_load) {
            auto t0 = std::chrono::high_resolution_clock::now();
            state_loaded = config.consolidation_cache->load_state(cache_load);
            rdma::Logger::info("[CONSOLIDATE] cache state %s: %s (%.1fs)",
                state_loaded ? "LOADED" : "LOAD FAILED (falling back to warm compaction)",
                cache_load,
                std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now() - t0).count());
        }
        if (!state_loaded) {
            config.consolidation_cache->prepare_for_consolidation();
            const char* cache_dump = getenv("DANNS_CACHE_DUMP");
            if (cache_dump && *cache_dump) {
                auto t0 = std::chrono::high_resolution_clock::now();
                bool ok = config.consolidation_cache->save_state(cache_dump);
                rdma::Logger::info("[CONSOLIDATE] cache state %s: %s (%.1fs)",
                    ok ? "DUMPED" : "DUMP FAILED", cache_dump,
                    std::chrono::duration<double>(
                        std::chrono::high_resolution_clock::now() - t0).count());
            }
        }
        const auto cs = config.consolidation_cache->stats();
        rdma::Logger::info(
            "[CONSOLIDATE] warm cache compacted: nbr_entries=%lu vec_records=%lu "
            "dropped_nbr=%lu dropped_vec=%lu",
            cs.entries, cs.vector_records, cs.dropped_neighbor_records,
            cs.dropped_vector_records);
    }

    rdma::ColocationStore* coloc_store = node_store->local_store();
    const rdma::ColocationConfig& coloc_config = node_store->config();
    const uint32_t my_partition = node_store->partition_id();
    const uint32_t my_node = node_store->node_id();
    // Compare/copy over real vector_dim, not aligned_dim — the colocation entry
    // vector field is vector_dim*element_size wide; aligned_dim over-reads into
    // degree bytes when dim%8!=0 (spacev 100 vs 104). RDMA fetch width already
    // uses coloc_config.dim, so this also stops copying garbage tail bytes.
    // deep/sift/laion (dim%8==0) byte-identical.
    const size_t cmp_dim = static_cast<size_t>(vector_dim);
    const uint32_t R = config.R;
    const float alpha = config.alpha;
    const uint32_t maxc = config.maxc;
    const int num_threads = static_cast<int>(config.num_threads);
    const bool enable_lits_filter = config.lits_filter;
    const bool enable_lits_tau_deleted =
        config.lits_filter && config.lits_tau_include_deleted;

    // Consolidation trace writer
    ConsolidationTraceWriter ctw;
    ctw.open(config.trace_path, my_node, vector_dim);

    auto t_start = std::chrono::high_resolution_clock::now();

    // C2: capture exceptions escaping the RDMA-issuing OMP loops below. An
    // RDMAException unwinding out of a parallel region calls std::terminate ->
    // SIGABRT (no Python exception, no FAILED marking). First error wins; it is
    // re-thrown serially right after the offending loop so pybind converts it to
    // a Python exception. Declared at function scope so every phase can see it.
    std::atomic<bool> err_flag{false};
    std::string err_msg;
    std::mutex err_mtx;

    // ---- Build entry list from local store ----
    // We iterate over all local entries; the caller should pass only active ones
    // via the ConsolidationConfig or a separate list.  Here we iterate the full
    // store and skip entries that are themselves deleted.
    const uint32_t n_entries = coloc_config.num_entries;

    // --- Diagnostic / profiling counters (OpenMP reduction) ---
    uint64_t modified = 0;
    uint64_t diag_path_empty = 0;
    uint64_t diag_path_no_prune = 0;
    uint64_t diag_path_prune = 0;
    uint64_t diag_fetch_none = 0;
    uint64_t local_vec_fetches = 0;
    uint64_t remote_vec_fetches = 0;
    uint64_t local_nbr_fetches = 0;
    uint64_t remote_nbr_fetches = 0;
    // Remote Di 2-hop expansions skipped because the RDMA send completion timed
    // out or errored (peer death/congestion). Atomic: shared across the OMP loop.
    std::atomic<uint64_t> rdma_wait_fail{0};
    uint64_t total_dist_comps = 0;

    double degree_before_sum = 0;
    double degree_after_sum = 0;
    uint64_t degree_count = 0;
    double expanded_sum = 0;
    uint64_t low_degree = 0;
    uint64_t zero_degree = 0;

    // Per-phase timing accumulators (nanoseconds, OpenMP reduction)
    uint64_t twohop_rdma_ns = 0, twohop_local_ns = 0;
    uint64_t vecfetch_rdma_ns = 0, vecfetch_local_ns = 0;
    uint64_t dist_compute_ns = 0, prune_ns = 0;

    // Lock wait accumulators (only non-zero when using fetcher_mutex fallback)
    uint64_t twohop_lock_wait_ns = 0, vecfetch_lock_wait_ns = 0;
    uint64_t twohop_lock_contentions = 0, vecfetch_lock_contentions = 0;

    // LITS filter counters
    uint64_t total_lits_checked = 0, total_lits_filtered = 0;
    uint64_t total_lits_no_survivor = 0;

    // Degree histogram buckets (bucket_start -> count)
    std::map<uint32_t, uint64_t> degree_histogram;

    // Fetch-interceptor cache counters (config.enable_vec_cache):
    // cache_hits/misses count remote VECTOR probes, nbr_cache_hits/misses
    // count remote Di NEIGHBOR-LIST probes in the 2-hop expand, and
    // cache_writebacks counts fetched vector misses pushed back into the
    // warm store (includes attempts deduped or dropped inside the store).
    uint64_t cache_hits = 0;
    uint64_t cache_misses = 0;
    uint64_t nbr_cache_hits = 0;
    uint64_t nbr_cache_misses = 0;
    uint64_t cache_writebacks = 0;
    uint64_t nbr_writebacks = 0;

    // Thread-local trace buffers for MT
    struct LocalTraceEntry {
        ConsolidationEntryHeader header;
        std::vector<AccessRecord> hop_acc;
        std::vector<AccessRecord> vec_acc;
    };
    int actual_threads = (num_threads > 1) ? num_threads : 1;
    std::vector<std::vector<LocalTraceEntry>> thread_traces(actual_threads);

    // Per-thread profiling arrays
    std::vector<uint64_t> per_thread_wall_ns(actual_threads, 0);
    std::vector<uint64_t> per_thread_entries(actual_threads, 0);

    // Thread-local scratch reused across repair nodes A: clear() keeps capacity
    // (no per-A malloc/free/rehash), flat open-addressing (tsl::robin) gives
    // cache-friendly insert/find. Replaces the per-iteration
    // std::unordered_set/map that churned ~22M times. Indexed by tid (loop runs
    // with omp_set_num_threads(num_threads) so tid < actual_threads).
    std::vector<tsl::robin_set<uint64_t>> thr_expanded(actual_threads);
    std::vector<tsl::robin_map<uint64_t, const T*>> thr_vec_store(actual_threads);

    // Resolved candidate list for step 3b, reused across repair nodes (same
    // clear()-keeps-capacity idea as above). Splitting "resolve the pointer"
    // from "read the vector" lets the second pass run a lookahead prefetch
    // over a CONTIGUOUS pointer array instead of interleaving two random
    // streams (hash probe + cold 384 B vector) in one dependent chain. Order
    // and contents are identical to the fused loop, so `candidates` — and
    // therefore the prune input and the written edges — are bit-identical.
    std::vector<std::vector<std::pair<uint64_t, const T*>>> thr_cand_ptr(
        actual_threads);
    // Prune input buffer, likewise reused (was a fresh ~11 KB vector per
    // repair node = ~11 M malloc/free pairs per consolidation).
    std::vector<std::vector<PruneCandidate<T>>> thr_candidates(actual_threads);

    // Lookahead depth for the candidate-vector prefetch, and how many cache
    // lines one vector spans. deep: 96 f32 = 384 B = 6 lines; 8 * 6 = 48 line
    // requests (3 KB) in flight per thread, which fits the L2 superqueue.
    // Depth 8 measured 1.30x over the fused loop on an idle Ice Lake 6338 at
    // 64 threads over a 16 GiB arena (depth 2/4/16 gave 1.14/1.19/1.23 in the
    // same harness, so 8 is on the plateau). Override with
    // DANNS_CONSOL_PREFETCH; 0 disables the prefetch but KEEPS the two-pass
    // split, so an A/B isolates the prefetch itself.
    const int prefetch_depth = [] {
        const char* e = getenv("DANNS_CONSOL_PREFETCH");
        return (e && *e) ? atoi(e) : 8;
    }();
    const size_t vec_lines = (cmp_dim * sizeof(T) + 63) / 64;

    // Per-thread staging arena for fetched remote vectors (reuse_fetch_buffers).
    // Same "reused across repair nodes A" idea as thr_expanded/thr_vec_store,
    // one level down: instead of reusing the CONTAINER we reuse the VECTOR
    // STORAGE itself, so the ~4.5e9 per-fetch operator new/delete pairs on the
    // OFF path collapse to a handful of geometric growths per thread.
    //
    // Slot stride is padded to a 64 B multiple so consecutive staged vectors
    // never share a cache line. Padding is safe to read past cmp_dim only
    // because nothing ever does: DistanceL2Float::compare consumes
    // floor(size/8)*8 elements, l2_sqr_avx512 consumes exactly `size`, and
    // ConsolidationCache::record_neighbor_vector copies exactly
    // cmp_dim*sizeof(T) bytes. So the staged bytes a consumer sees are
    // byte-for-byte what the per-vector heap block held.
    struct VecArena {
        std::vector<T> buf;   // slots * arena_stride elements
        size_t slots = 0;     // capacity, in whole vectors
    };
    const size_t arena_stride =
        ((cmp_dim * sizeof(T) + 63) / 64) * 64 / sizeof(T);
    std::vector<VecArena> thr_vec_arena(
        reuse_fetch_buffers ? actual_threads : 0);

    // Detect multi-QP availability: Python orchestrator should have called
    // setup_multi_qp() before consolidate_local() when use_multi_qp: true in config.
    // Each peer gets `num_thread_qps` per-thread RDMAConnections + per-thread buffers.
    const uint32_t num_qps = node_store->num_thread_qps();
    const bool use_multi_qp = (num_threads > 1) &&
                              (num_qps >= static_cast<uint32_t>(num_threads));
    if (num_threads > 1) {
        rdma::Logger::info("[CONSOLIDATE] multi_qp=%s (num_thread_qps=%u, num_threads=%d)",
            use_multi_qp ? "true" : "false", num_qps, num_threads);
    }

    // Fetcher mutexes for thread-safe remote access (fallback path only).
    // Pre-populate for all known peers BEFORE entering the parallel region —
    // concurrent std::map insert from the hot loop is a data race. This is a
    // no-op overhead when use_multi_qp=true, since the mutexes are never touched.
    std::map<uint32_t, std::mutex> fetcher_mutexes;
    if (num_threads > 1 && !use_multi_qp) {
        for (uint32_t pid : node_store->peer_ids()) {
            (void)fetcher_mutexes[pid];  // force-insert
        }
    }

    // Track per-deleted-VID access count (quantify redundant 2-hop fetches)
    std::unordered_map<uint64_t, std::atomic<uint64_t>> deleted_access_count;
    for (uint64_t dv : deleted_set) {
        deleted_access_count.emplace(std::piecewise_construct,
            std::forward_as_tuple(dv), std::forward_as_tuple(0));
    }

    // Per-vector fetch-count tracker: dense [partition][local_loc] → atomic<uint32_t>.
    // Pre-sized to the per-partition local_to_tag arrays so concurrent
    // fetch_add() needs no allocation. Active iff caller passed a dump path.
    const bool dump_vec_fetch =
        !config.vec_fetch_count_dump_path.empty();
    std::vector<std::vector<std::atomic<uint32_t>>> vec_fetch_count;
    if (dump_vec_fetch) {
        const auto& l2t = node_store->local_to_tag();
        vec_fetch_count.resize(l2t.size());
        for (size_t p = 0; p < l2t.size(); p++) {
            vec_fetch_count[p] = std::vector<std::atomic<uint32_t>>(l2t[p].size());
        }
    }

    const bool ct_active = ctw.active();

    if (num_threads > 1) {
        omp_set_num_threads(num_threads);
    }

    // Dense deleted-membership index (see DeletedVidIndex banner). Built
    // INSIDE the t_start..t_end window so its cost is charged honestly to
    // consolidation. Every deleted_set.count() below goes through this.
    const DeletedVidIndex del_idx(deleted_set, num_threads);
    {
        auto t_di = std::chrono::high_resolution_clock::now();
        rdma::Logger::info(
            "[CONSOLIDATE] deleted index: %s (%lu vids, %.1f MB, built by %.2fs mark)",
            del_idx.dense() ? "dense bitset" : "HASH FALLBACK (domain too sparse)",
            static_cast<unsigned long>(deleted_set.size()),
            del_idx.bytes() / 1e6,
            std::chrono::duration<double>(t_di - t_start).count());
    }

    // P-E: which RobustPrune inner-loop kernel is live (see prune_utils.h).
    // Result-invariant either way; logged so an A/B cell can be identified
    // from the worker log alone.
    {
        const bool f32 = std::is_same<T, float>::value;
#ifdef DANNS_PRUNE_SIMD_F32
        const bool prune_fast = f32 && prune_opt_enabled();
#else
        const bool prune_fast = false;
#endif
        rdma::Logger::info(
            "[CONSOLIDATE] prune kernel: %s%s",
            prune_fast ? "P-E batched f32" : "baseline",
            prune_fast ? ((cmp_dim * sizeof(T) >= 2048) ? " (8-lane)" : " (4-lane)")
                       : (f32 ? " (DANNS_CONSOL_PRUNE_OPT=0)" : " (non-float dtype)"));
        // P-F: same gate, step-3b distance loop. Different lane split (768 B).
#ifdef DANNS_PRUNE_SIMD_F32
        rdma::Logger::info(
            "[CONSOLIDATE] dist kernel: %s%s",
            prune_fast ? "P-F batched f32" : "baseline",
            prune_fast ? (dist_opt_use_8_lanes(cmp_dim * sizeof(T))
                              ? " (8-lane)" : " (2-lane)")
                       : (f32 ? " (DANNS_CONSOL_PRUNE_OPT=0)" : " (non-float dtype)"));
#else
        rdma::Logger::info("[CONSOLIDATE] dist kernel: baseline (no AVX2)");
#endif
    }

    // Software bandwidth pacing (DANNS_CONSOL_PACE_GBPS=<total node Gb/s>):
    // per-thread token bucket that delays after each RDMA read so the node's
    // aggregate consolidation ingress converges to the target — emulates a
    // slower link for BW-sensitivity experiments. NIC packet pacing is
    // RAW_PACKET-only on this ConnectX-6/driver, so RC traffic is paced at
    // issue time instead. env unset → zero overhead. NOTE: accounting starts
    // at each thread's first paced read and assumes one consolidate phase per
    // process (our experiment harness); wall-clock validated against the 1ms
    // NIC monitor.
    const double pace_gbps = [] {
        const char* e = getenv("DANNS_CONSOL_PACE_GBPS");
        return (e && *e) ? atof(e) : 0.0;
    }();
    const double pace_bytes_per_s_per_thread =
        pace_gbps > 0 ? pace_gbps * 1e9 / 8.0 / std::max(num_threads, 1) : 0.0;
    if (pace_gbps > 0)
        rdma::Logger::info("[CONSOLIDATE] SW pacing active: %.1f Gb/s node target "
                           "(%.0f MB/s per thread)", pace_gbps,
                           pace_bytes_per_s_per_thread / 1e6);
    struct PaceState { double start_s = -1.0; uint64_t bytes = 0; };
    auto pace_throttle = [&](uint64_t add_bytes) {
        if (pace_bytes_per_s_per_thread <= 0) return;
        thread_local PaceState ps;
        auto now_s = [] {
            return std::chrono::duration<double>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
        };
        if (ps.start_s < 0) ps.start_s = now_s();
        ps.bytes += add_bytes;
        const double due = ps.start_s + ps.bytes / pace_bytes_per_s_per_thread;
        double t = now_s();
        while (t < due) {
            std::this_thread::sleep_for(std::chrono::microseconds(
                static_cast<int64_t>(std::min(200.0, (due - t) * 1e6)) + 1));
            t = now_s();
        }
    };

    // Di-grouped task order (DANNS_CONSOL_DI_ORDER=1): sort repair tasks by
    // their first deleted neighbor so the ~in-degree(Di) tasks that expand the
    // same Di run adjacently — Di's candidate vectors then stay LLC-warm
    // across the group instead of being re-read cold ~40× spread over the
    // whole run. Pure reordering: tasks are independent, results identical.
    std::vector<uint32_t> di_order;
    {
        const char* e = getenv("DANNS_CONSOL_DI_ORDER");
        if (e && *e == '1') {
            auto t_ord = std::chrono::high_resolution_clock::now();
            std::vector<uint64_t> order_key(n_entries);
            #pragma omp parallel for schedule(static) if(num_threads > 1)
            for (uint32_t i = 0; i < n_entries; i++) {
                uint64_t k = std::numeric_limits<uint64_t>::max();
                uint32_t deg = coloc_store->get_degree(i);
                if (deg > coloc_config.slack_R) deg = coloc_config.slack_R;
                const uint64_t* nb = coloc_store->get_neighbors(i);
                for (uint32_t j = 0; j < deg; j++) {
                    uint64_t vid = nb[j];
                    if (vid == INVALID_VID) break;
                    if (del_idx.contains(vid)) { k = vid; break; }
                }
                order_key[i] = k;
            }
            di_order.resize(n_entries);
            for (uint32_t i = 0; i < n_entries; i++) di_order[i] = i;
            std::sort(di_order.begin(), di_order.end(),
                      [&order_key](uint32_t a, uint32_t b) {
                          return order_key[a] < order_key[b];
                      });
            rdma::Logger::info("[CONSOLIDATE] di-order: keyed+sorted %u tasks in %.2fs",
                n_entries,
                std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now() - t_ord).count());
        }
    }

    auto t_loop_start = std::chrono::high_resolution_clock::now();
    // 5% intervals so long consolidations don't look stuck for minutes.
    const size_t progress_interval = std::max(static_cast<size_t>(n_entries) / 20, size_t(1));

    // ========================================================================
    // MAIN PARALLEL LOOP: OpenMP parallel for over all local entries
    // ========================================================================
    #pragma omp parallel for schedule(dynamic, 64) if(num_threads > 1) \
        reduction(+:modified, diag_path_empty, diag_path_no_prune, diag_path_prune, \
                  diag_fetch_none, local_vec_fetches, remote_vec_fetches, \
                  local_nbr_fetches, remote_nbr_fetches, total_dist_comps, \
                  cache_hits, cache_misses, nbr_cache_hits, nbr_cache_misses, \
                  cache_writebacks, nbr_writebacks, \
                  degree_before_sum, degree_after_sum, degree_count, \
                  expanded_sum, low_degree, zero_degree, \
                  twohop_rdma_ns, twohop_local_ns, \
                  vecfetch_rdma_ns, vecfetch_local_ns, \
                  dist_compute_ns, prune_ns, \
                  twohop_lock_wait_ns, vecfetch_lock_wait_ns, \
                  twohop_lock_contentions, vecfetch_lock_contentions, \
                  total_lits_checked, total_lits_filtered, \
                  total_lits_no_survivor)
    for (uint32_t idx = 0; idx < n_entries; idx++) {
        if (err_flag.load(std::memory_order_relaxed)) continue;
        try {
        // Progress log: thread 0 only, ~10% intervals
        if (idx % progress_interval == 0 && idx > 0) {
            int _tid = (num_threads > 1) ? omp_get_thread_num() : 0;
            if (_tid == 0) {
                auto t_now = std::chrono::high_resolution_clock::now();
                double elapsed_s = std::chrono::duration<double>(t_now - t_loop_start).count();
                rdma::Logger::info("consolidate progress: idx=%u/%u (%.0f%%) %.1fs elapsed modified=%zu",
                    idx, n_entries, 100.0 * idx / n_entries, elapsed_s,
                    static_cast<size_t>(modified));
            }
        }

        auto t_iter_start = std::chrono::high_resolution_clock::now();
        int tid = (num_threads > 1) ? omp_get_thread_num() : 0;

        uint32_t local_loc = di_order.empty() ? idx : di_order[idx];

        // Check if this entry is itself deleted
        uint64_t entry_vid = danns::encode_vector_id(my_partition, local_loc);
        if (del_idx.contains(entry_vid)) {
            per_thread_wall_ns[tid] += std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now() - t_iter_start).count();
            per_thread_entries[tid]++;
            continue;
        }

        // Read current neighbors (zero-copy from local store)
        uint32_t degree = coloc_store->get_degree(local_loc);
        const uint64_t* nbr_ptr = coloc_store->get_neighbors(local_loc);

        // Check if any neighbor is deleted + count deleted
        bool has_deleted = false;
        uint16_t n_deleted_nbrs = 0;
        for (uint32_t i = 0; i < degree; i++) {
            if (nbr_ptr[i] == INVALID_VID) break;
            if (del_idx.contains(nbr_ptr[i])) {
                has_deleted = true;
                n_deleted_nbrs++;
            }
        }
        if (!has_deleted) {
            per_thread_wall_ns[tid] += std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now() - t_iter_start).count();
            per_thread_entries[tid]++;
            continue;
        }

        // Trace accumulators
        std::vector<AccessRecord> ct_hop_acc, ct_vec_acc;

        // Count deleted neighbors for diagnostics
        degree_before_sum += degree;
        degree_count++;

        // ---- 2-hop expansion ----
        uint64_t my_vid = danns::encode_vector_id(my_partition, local_loc);
        auto& expanded = thr_expanded[tid];
        expanded.clear();
        // This point's deleted original neighbors — tau sample contribution
        // (LITS-Aware-Update); never candidates themselves.
        std::vector<uint64_t> deleted_nbrs;

        // Cap degree to slack_R for safety (prevents buffer overread if data corrupted)
        uint32_t safe_degree = std::min(degree, coloc_config.slack_R);

        for (uint32_t i = 0; i < safe_degree; i++) {
            uint64_t vid = nbr_ptr[i];
            if (vid == INVALID_VID) break;
            if (!del_idx.contains(vid)) {
                expanded.insert(vid);
            } else {
                // 2-hop: fetch deleted neighbor's neighbors
                if (enable_lits_tau_deleted) deleted_nbrs.push_back(vid);
                deleted_access_count[vid].fetch_add(1, std::memory_order_relaxed);
                uint32_t del_part = decode_partition(vid);
                uint32_t del_ll = static_cast<uint32_t>(decode_local_loc(vid));
                uint32_t del_owner = mapper->partition_id_to_node_id(del_part);

                const uint64_t* nbr2_ptr = nullptr;
                uint32_t deg2 = 0;
                std::vector<uint64_t> nbr2_copy; // MT: owns copied data
                bool is_remote_2hop = (del_owner != my_node);

                if (!is_remote_2hop) {
                    // Local: zero-copy read
                    auto t_local = std::chrono::high_resolution_clock::now();
                    deg2 = coloc_store->get_degree(del_ll);
                    if (deg2 > coloc_config.slack_R) {
                        rdma::Logger::error("deg2 overflow (local): del_ll=%u deg2=%u slack_R=%u",
                            del_ll, deg2, coloc_config.slack_R);
                        deg2 = coloc_config.slack_R;
                    }
                    nbr2_ptr = coloc_store->get_neighbors(del_ll);
                    twohop_local_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::high_resolution_clock::now() - t_local).count();
                    local_nbr_fetches++;
                } else if (const char* lb = node_store->local_data(del_owner)) {
                    // LOCAL_MMAP: peer partition is mmap'd locally (room-at-the-top).
                    // Read degree+nbr list straight from the local pointer — no RDMA,
                    // no NIC, no lock. Counts as remote_nbr (semantic = peer data).
                    auto t_local2 = std::chrono::high_resolution_clock::now();
                    uint64_t off = static_cast<uint64_t>(del_ll) *
                        coloc_config.entry_size + coloc_config.degree_offset;
                    std::memcpy(&deg2, lb + off, sizeof(uint32_t));
                    if (deg2 > coloc_config.slack_R) {
                        rdma::Logger::error("deg2 overflow (local_mmap): del_ll=%u deg2=%u slack_R=%u",
                            del_ll, deg2, coloc_config.slack_R);
                        deg2 = coloc_config.slack_R;
                    }
                    const uint64_t* raw = reinterpret_cast<const uint64_t*>(
                        lb + off + sizeof(uint32_t));
                    nbr2_copy.assign(raw, raw + deg2);
                    nbr2_ptr = nbr2_copy.data();
                    twohop_rdma_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::high_resolution_clock::now() - t_local2).count();
                    remote_nbr_fetches++;
                } else if (const uint64_t* cnbrs = cache_intercept
                               ? config.consolidation_cache->lookup_neighbors_ptr(vid, deg2)
                               : nullptr) {
                    // Warmed or written-back Di adjacency: serve the 2-hop
                    // expand straight from the nbr arena (lock-free, zero
                    // copy). Lists were clamped to slack_R at record time
                    // (configure passes slack_R as max_degree), so the
                    // defensive clamp below rarely fires.
                    if (deg2 > coloc_config.slack_R) deg2 = coloc_config.slack_R;
                    nbr2_ptr = cnbrs;
                    nbr_cache_hits++;
                } else if (use_multi_qp) {
                    if (cache_intercept) nbr_cache_misses++;
                    // Remote via per-thread QP — NO lock
                    auto& ts_map = node_store->thread_states_mut();
                    auto ts_it = ts_map.find(del_owner);
                    const auto* mr_ptr = node_store->remote_mr(del_owner);
                    if (ts_it == ts_map.end() || !mr_ptr) {
                        rdma::Logger::error("No thread_state/mr for peer %u (2-hop)", del_owner);
                        continue;
                    }
                    auto& state = ts_it->second.at(tid);
                    uint32_t nbr_read_sz = sizeof(uint32_t) +
                        coloc_config.slack_R * sizeof(uint64_t);
                    uint64_t remote_offset = static_cast<uint64_t>(del_ll) *
                        coloc_config.entry_size + coloc_config.degree_offset;
                    rdma::RemoteMemoryRegion target = *mr_ptr;
                    target.addr += remote_offset;
                    target.size = nbr_read_sz;
                    char* buf = state.neighbor_fetch_buffer.data();
                    auto t_rdma = std::chrono::high_resolution_clock::now();
                    state.conn->post_read(buf, nbr_read_sz, state.neighbor_fetch_mr, target, 0);
                    if (!state.conn->wait_send_completion(0, 5000)) {
                        rdma_wait_fail.fetch_add(1, std::memory_order_relaxed);
                        continue;  // skip this Di; do not read the stale buffer
                    }
                    std::memcpy(&deg2, buf, sizeof(uint32_t));
                    if (deg2 > coloc_config.slack_R) {
                        rdma::Logger::error("deg2 overflow (multi-qp): del_ll=%u deg2=%u slack_R=%u",
                            del_ll, deg2, coloc_config.slack_R);
                        deg2 = coloc_config.slack_R;
                    }
                    const uint64_t* raw = reinterpret_cast<const uint64_t*>(
                        buf + sizeof(uint32_t));
                    nbr2_copy.assign(raw, raw + deg2);
                    nbr2_ptr = nbr2_copy.data();
                    twohop_rdma_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::high_resolution_clock::now() - t_rdma).count();
                    remote_nbr_fetches++;
                    pace_throttle(nbr_read_sz);
                    // Write the fetched Di adjacency back so later repair
                    // tasks hit the cache instead of re-fetching (Di lists
                    // are immutable during the loop; repairs only rewrite
                    // local live nodes).
                    if (cache_intercept && deg2 > 0 &&
                        config.consolidation_cache->record_neighbors_writeback(
                            vid, nbr2_ptr, deg2))
                        nbr_writebacks++;
                } else {
                    // Remote via shared fetcher + mutex (fallback path)
                    if (cache_intercept) nbr_cache_misses++;
                    rdma::ColocationFetcher* fetcher = node_store->fetcher(del_owner);
                    if (!fetcher) {
                        rdma::Logger::error("No fetcher for peer %u (2-hop del_ll=%u)", del_owner, del_ll);
                        continue;
                    }
                    bool contended = false;
                    auto t_lock = std::chrono::high_resolution_clock::now();
                    if (num_threads > 1) {
                        if (!fetcher_mutexes[del_owner].try_lock()) {
                            contended = true;
                            fetcher_mutexes[del_owner].lock();
                        }
                    }
                    uint64_t lock_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::high_resolution_clock::now() - t_lock).count();
                    if (num_threads > 1) {
                        twohop_lock_wait_ns += lock_ns;
                        if (contended) twohop_lock_contentions++;
                    }
                    auto t_rdma = std::chrono::high_resolution_clock::now();
                    const uint64_t* raw = fetcher->fetch_neighbors_only(del_ll);
                    deg2 = fetcher->fetched_degree();
                    if (deg2 > coloc_config.slack_R) {
                        rdma::Logger::error("deg2 overflow (remote): del_ll=%u deg2=%u slack_R=%u",
                            del_ll, deg2, coloc_config.slack_R);
                        deg2 = coloc_config.slack_R;
                    }
                    nbr2_copy.assign(raw, raw + deg2);
                    nbr2_ptr = nbr2_copy.data();
                    twohop_rdma_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::high_resolution_clock::now() - t_rdma).count();
                    if (num_threads > 1) {
                        fetcher_mutexes[del_owner].unlock();
                    }
                    remote_nbr_fetches++;
                    pace_throttle(sizeof(uint32_t) +
                                  coloc_config.slack_R * sizeof(uint64_t));
                    if (cache_intercept && deg2 > 0 &&
                        config.consolidation_cache->record_neighbors_writeback(
                            vid, nbr2_ptr, deg2))
                        nbr_writebacks++;
                }

                if (ct_active) {
                    AccessRecord ar{};
                    ar.vid = vid;
                    ar.is_remote = is_remote_2hop ? 1 : 0;
                    ct_hop_acc.push_back(ar);
                }

                for (uint32_t j = 0; j < deg2; j++) {
                    uint64_t nn_vid = nbr2_ptr[j];
                    if (nn_vid == INVALID_VID) break;
                    if (nn_vid != my_vid && !del_idx.contains(nn_vid)) {
                        expanded.insert(nn_vid);
                    }
                }
            }
        }
        expanded.erase(my_vid);
        expanded_sum += expanded.size();

        // Helper: update degree histogram (thread-safe)
        auto update_histogram = [&](uint32_t deg) {
            uint32_t bucket = (deg / 8) * 8;
            if (num_threads > 1) {
                #pragma omp critical(histogram)
                { degree_histogram[bucket]++; }
            } else {
                degree_histogram[bucket]++;
            }
        };

        // Helper lambda: buffer trace entry (thread-local)
        auto write_ct_entry = [&](uint16_t deg_after, uint8_t path) {
            if (!ct_active) return;
            ConsolidationEntryHeader eh{};
            eh.local_loc = local_loc;
            eh.tag = local_loc; // In modular design, tag == local_loc unless remapped
            eh.degree_before = static_cast<uint16_t>(degree);
            eh.degree_after = deg_after;
            eh.n_deleted_nbrs = n_deleted_nbrs;
            eh.n_expanded = static_cast<uint16_t>(
                std::min(expanded.size(), static_cast<size_t>(UINT16_MAX)));
            eh.path = path;
            eh.n_2hop_accesses = static_cast<uint16_t>(ct_hop_acc.size());
            eh.n_vec_accesses = static_cast<uint16_t>(ct_vec_acc.size());
            int t = (num_threads > 1) ? omp_get_thread_num() : 0;
            thread_traces[t].push_back({eh, std::move(ct_hop_acc), std::move(ct_vec_acc)});
        };

        // ---- Path 1: empty expanded set ----
        if (expanded.empty()) {
            write_neighbors_internal(coloc_store, coloc_config, local_loc, nullptr, 0);
            modified++;
            diag_path_empty++;
            degree_after_sum += 0;
            zero_degree++;
            update_histogram(0);
            write_ct_entry(0, 1);
            per_thread_wall_ns[tid] += std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now() - t_iter_start).count();
            per_thread_entries[tid]++;
            continue;
        }

        // ---- Path 2: no prune needed (expanded fits within R) ----
        if (expanded.size() <= R) {
            std::vector<uint64_t> new_nbrs(expanded.begin(), expanded.end());
            uint32_t new_deg = static_cast<uint32_t>(new_nbrs.size());
            write_neighbors_internal(coloc_store, coloc_config, local_loc,
                                     new_nbrs.data(), new_deg);
            modified++;
            diag_path_no_prune++;
            degree_after_sum += new_deg;
            if (new_deg < R / 2) low_degree++;
            if (new_deg == 0) zero_degree++;
            update_histogram(new_deg);
            write_ct_entry(static_cast<uint16_t>(new_deg), 2);
            per_thread_wall_ns[tid] += std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now() - t_iter_start).count();
            per_thread_entries[tid]++;
            continue;
        }

        // ---- Path 3: need distance computation + RobustPrune ----

        // Step 3a: Fetch all candidate vectors
        const T* node_vec = static_cast<const T*>(coloc_store->get_vector(local_loc));

        // vec_store: vid -> T* (vector pointer). Thread-local, reused (clear
        // keeps capacity); robin map for cache-friendly insert/find.
        auto& vec_store = thr_vec_store[tid];
        vec_store.clear();

        // Partition expanded VIDs by owner for batch RDMA
        std::unordered_map<uint32_t, std::vector<uint32_t>> peer_locs;
        std::unordered_map<uint32_t, std::vector<uint64_t>> peer_vids;
        std::vector<uint64_t> local_vids;

        for (uint64_t vid : expanded) {
            uint32_t part = decode_partition(vid);
            uint32_t ll = static_cast<uint32_t>(decode_local_loc(vid));
            uint32_t owner = mapper->partition_id_to_node_id(part);
            if (owner == my_node) {
                local_vids.push_back(vid);
            } else {
                peer_locs[owner].push_back(ll);
                peer_vids[owner].push_back(vid);
            }
        }
        // Deleted original neighbors join the SAME batched vector fetch so
        // d(p, Di) can feed tau. They are not in `expanded`, so they never
        // become prune candidates; vec_store is keyed by vid and the prune
        // only looks up candidate vids. Because they ride the same per-peer
        // batch, their vectors are memcpy'd out of the fetcher buffer INSIDE
        // the per-peer fetcher_mutexes lock below — no extra unlocked copy.
        for (uint64_t vid : deleted_nbrs) {
            uint32_t part = decode_partition(vid);
            uint32_t ll = static_cast<uint32_t>(decode_local_loc(vid));
            uint32_t owner = mapper->partition_id_to_node_id(part);
            if (owner == my_node) {
                local_vids.push_back(vid);
            } else {
                peer_locs[owner].push_back(ll);
                peer_vids[owner].push_back(vid);
            }
        }

        // Own memory for remote vectors (fetcher buffer is reused)
        //
        // EXACT upper bound on staging slots this iteration can consume.
        // The two loops above push every vid of `expanded` and every vid of
        // `deleted_nbrs` into exactly one of local_vids (zero-copy pointer
        // into the local store, no slot) or peer_locs (one slot), so the
        // number of slots is at most this, and exactly this when the cache
        // interceptor is off. Both terms are bounded by the expansion's own
        // caps: |deleted_nbrs| <= safe_degree <= slack_R, and |expanded| <=
        // safe_degree + safe_degree*slack_R, so
        //     n_remote_bound <= slack_R * (slack_R + 2)
        // (slack_R=83 -> ~7.1k slots -> ~2.7 MB/thread at deep's 384 B; the
        // measured mean is avg_expanded ~462 -> ~178 KB/thread). The arena
        // keeps the high-water mark and never shrinks, so worst-case resident
        // extra memory is num_threads * that bound.
        const size_t n_remote_bound = expanded.size() + deleted_nbrs.size()
                                      - local_vids.size();
        std::vector<std::vector<T>> remote_vec_storage;  // legacy per-fetch heap

        // Arena state for this iteration (reuse_fetch_buffers only).
        //
        // LIFETIME ARGUMENT for resetting the cursor every iteration:
        // the staged pointers escape only into vec_store[vid] and, through it,
        // into PruneCandidate::vec. Every reader — the dist_compute loop, the
        // LITS tau loop, LITSFilter::filter_consolidation_candidates,
        // robust_prune_internal, and the cache-ON miss writeback — runs inside
        // THIS iteration, and vec_store is cleared at the top of the next one
        // (see `vec_store.clear()` above). Nothing staged here is read after
        // the iteration ends, so reusing the same slots next iteration cannot
        // alias a live pointer.
        //
        // NO-REALLOC-WHILE-LIVE: the arena is grown to n_remote_bound HERE,
        // before the first slot of this iteration is handed out. Inside the
        // fetch loops the arena is never resized, so no slot pointer can be
        // invalidated by a growth. Growth is geometric, so the steady state
        // after the first few repair nodes is zero allocations per fetch.
        T* arena_base = nullptr;
        size_t arena_slots = 0;
        size_t arena_used = 0;
        if (reuse_fetch_buffers) {
            auto& ar = thr_vec_arena[tid];
            if (n_remote_bound > ar.slots) {
                const size_t want = std::max<size_t>(
                    n_remote_bound, ar.slots ? ar.slots * 2 : size_t(64));
                ar.buf.resize(want * arena_stride);
                ar.slots = want;
            }
            arena_base = ar.buf.data();
            arena_slots = ar.slots;
        } else {
            remote_vec_storage.reserve(n_remote_bound);
        }

        // Storage for ONE fetched remote vector. Reuse path: the next arena
        // slot, no allocation. Legacy path: a fresh heap block, byte-for-byte
        // the previous behaviour.
        auto stage_vec = [&]() -> T* {
            if (reuse_fetch_buffers) {
                if (arena_used >= arena_slots) {
                    // Unreachable unless the n_remote_bound derivation above is
                    // wrong. Fail loud rather than write past the arena; the
                    // OMP try/catch turns this into a Python exception.
                    throw std::runtime_error(
                        "consolidate reuse_fetch_buffers: staging arena overflow");
                }
                T* p = arena_base + arena_used * arena_stride;
                arena_used++;
                return p;
            }
            remote_vec_storage.emplace_back(cmp_dim);
            return remote_vec_storage.back().data();
        };

        // Batch fetch remote vectors per peer
        for (auto& [owner, r_locs] : peer_locs) {
            auto& r_vids = peer_vids[owner];
            uint32_t vec_bytes = coloc_config.dim * coloc_config.element_size;

            // --- Consolidation-cache fetch interceptor (OFF/inline path) ---
            // Zero-copy hit: point vec_store DIRECTLY at the vector inside the
            // search-warmed cache arena (no per-hit alloc/memcpy, ONE shard
            // lookup). cache-on == cache-off except hit -> skip RDMA. Misses are
            // compacted in place to the front of r_locs/r_vids and fetched by
            // the branches below. Skipped for LOCAL_MMAP owners (already a local
            // read, no RDMA to save).
            // Thread-safe: lookup_vector_ptr takes a shared lock; the arena is
            // immutable during consolidation so the returned pointer stays valid
            // (slots never move). Alignment: arena slot stride must suit the
            // distance fn's loads — holds for deep 384B (this run); copy for
            // unaligned dims (see lookup_vector_ptr note).
            if (cache_intercept && vec_bytes > 0 &&
                !node_store->local_data(owner)) {
                size_t w = 0;
                for (size_t i = 0; i < r_locs.size(); i++) {
                    const void* p =
                        config.consolidation_cache->lookup_vector_ptr(r_vids[i]);
                    if (p) {
                        vec_store[r_vids[i]] = reinterpret_cast<const T*>(p);
                        cache_hits++;
                    } else {
                        r_locs[w] = r_locs[i];
                        r_vids[w] = r_vids[i];
                        ++w;
                    }
                }
                r_locs.resize(w);
                r_vids.resize(w);
                if (r_locs.empty()) continue;  // all served from cache
            }

            if (const char* lb = node_store->local_data(owner)) {
                // LOCAL_MMAP: copy vectors out of locally-mmap'd peer partition.
                auto t_vlocal = std::chrono::high_resolution_clock::now();
                for (size_t i = 0; i < r_locs.size(); i++) {
                    const void* raw = lb +
                        static_cast<size_t>(r_locs[i]) * coloc_config.entry_size +
                        coloc_config.vector_offset;
                    T* dst = stage_vec();
                    std::memcpy(dst, raw, cmp_dim * sizeof(T));
                    vec_store[r_vids[i]] = dst;
                }
                vecfetch_rdma_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::high_resolution_clock::now() - t_vlocal).count();
            } else if (use_multi_qp) {
                // Per-thread QP path — NO lock. Issue post_read_batch directly
                // into per-thread buffer, then copy into our own storage.
                auto& ts_map = node_store->thread_states_mut();
                auto ts_it = ts_map.find(owner);
                const auto* mr_ptr = node_store->remote_mr(owner);
                if (ts_it == ts_map.end() || !mr_ptr) {
                    rdma::Logger::error("No thread_state/mr for peer %u (vecfetch)", owner);
                    continue;
                }
                auto& state = ts_it->second.at(tid);
                char* buf = reinterpret_cast<char*>(state.fetch_buffer.data());
                // Chunk to the per-thread fetch_buffer capacity. The buffer is
                // registered for FETCH_BUF_ENTRIES entries (setup_multi_qp); a
                // single post_read_batch larger than that writes past the MR and
                // trips LOCAL_PROTECTION_ERROR (seen at 10% delete where a hub Di
                // bucket exceeds the cap). Mirrors run_with_per_di_impl's
                // buf_entries_cap chunking.
                const size_t fbuf_bytes = state.fetch_buffer.size() * sizeof(float);
                const size_t es = std::max<size_t>(coloc_config.entry_size, 1);
                const size_t cap = std::max<size_t>(1, fbuf_bytes / es);
                auto t_vrdma = std::chrono::high_resolution_clock::now();
                for (size_t base = 0; base < r_locs.size(); base += cap) {
                    const size_t chunk = std::min(cap, r_locs.size() - base);
                    std::vector<rdma::SGLEntry> sgl(chunk);
                    for (size_t i = 0; i < chunk; i++) {
                        sgl[i].local_addr = buf +
                            i * coloc_config.entry_size + coloc_config.vector_offset;
                        sgl[i].length = vec_bytes;
                        sgl[i].remote_offset =
                            static_cast<uint64_t>(r_locs[base + i]) * coloc_config.entry_size +
                            coloc_config.vector_offset;
                    }
                    if (sync_fetch) {
                        // fetch_window == 1 (== legacy sync_fetch). Post ONE
                        // read, wait for its completion, then post the next.
                        // post_read_batch would instead keep 256 WRs in flight
                        // and interleave posting with polling, hiding nearly all
                        // of the RTT.
                        //
                        // This block is deliberately kept BYTE-FOR-BYTE as the
                        // sync_fetch patch left it rather than folded into the
                        // windowed call below, so that the already-measured
                        // naive baseline is literally the K=1 point of the
                        // fetch_window curve (same wr_id, same 10 s per-read
                        // budget, same rdma_wait_fail counter, same throw).
                        //
                        // With the cache interceptor on, r_locs has already been
                        // compacted to the MISSES above, so this serializes
                        // exactly the reads the cache could not serve.
                        //
                        // Same destination addresses, same lengths, same remote
                        // offsets as the SGL above -> the bytes that land in
                        // `buf` are identical, and the memcpy loop below is
                        // untouched. Only the wall clock differs.
                        for (size_t i = 0; i < chunk; i++) {
                            rdma::RemoteMemoryRegion tgt = *mr_ptr;
                            tgt.addr += sgl[i].remote_offset;
                            tgt.size = sgl[i].length;
                            state.conn->post_read(sgl[i].local_addr, sgl[i].length,
                                                  state.fetch_buffer_mr, tgt, 0);
                            // Mirror post_read_batch's fail-loud contract: it
                            // throws on timeout, so a silent `continue` here
                            // would let the stale buffer be memcpy'd out and
                            // break bit-identity.
                            if (!state.conn->wait_send_completion(0, 10000)) {
                                rdma_wait_fail.fetch_add(1, std::memory_order_relaxed);
                                throw std::runtime_error(
                                    "consolidate sync_fetch: vector read timed out");
                            }
                        }
                    } else if (fetch_window_wr) {
                        // fetch_window >= 2: the SAME batched call, with its
                        // in-flight cap lowered from 256 to K. post_read_batch
                        // already implements exactly the requested sliding
                        // window -- it posts while (posted - completed) < cap
                        // and polls in between -- so nothing about completion
                        // reaping is reimplemented here.
                        //
                        // Out-of-order completions are a non-issue by
                        // construction: entry i's destination is fixed at post
                        // time (sgl[i].local_addr), completions are only
                        // COUNTED (never mapped back to a slot), and the call
                        // still returns after all `chunk` completions, so the
                        // memcpy loop below sees a fully populated buffer in
                        // the identical layout as every other K.
                        state.conn->post_read_batch(sgl, *mr_ptr, state.fetch_buffer_mr,
                                                    0, 10000, fetch_window_wr);
                    } else {
                        state.conn->post_read_batch(sgl, *mr_ptr, state.fetch_buffer_mr, 0);
                    }
                    pace_throttle(static_cast<uint64_t>(chunk) * vec_bytes);
                    for (size_t i = 0; i < chunk; i++) {
                        const void* raw = buf +
                            i * coloc_config.entry_size + coloc_config.vector_offset;
                        T* dst = stage_vec();
                        std::memcpy(dst, raw, cmp_dim * sizeof(T));
                        vec_store[r_vids[base + i]] = dst;
                    }
                }
                vecfetch_rdma_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::high_resolution_clock::now() - t_vrdma).count();
            } else {
                // Shared fetcher + mutex path (fallback)
                rdma::ColocationFetcher* fetcher = node_store->fetcher(owner);
                if (!fetcher) {
                    rdma::Logger::error("No fetcher for peer %u (vecfetch)", owner);
                    continue;
                }
                bool contended = false;
                auto t_lock = std::chrono::high_resolution_clock::now();
                if (num_threads > 1) {
                    if (!fetcher_mutexes[owner].try_lock()) {
                        contended = true;
                        fetcher_mutexes[owner].lock();
                    }
                }
                uint64_t lock_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::high_resolution_clock::now() - t_lock).count();
                if (num_threads > 1) {
                    vecfetch_lock_wait_ns += lock_ns;
                    if (contended) vecfetch_lock_contentions++;
                }
                auto t_vrdma = std::chrono::high_resolution_clock::now();
                if (sync_fetch) {
                    // fetch_window == 1 (== legacy sync_fetch), shared-fetcher
                    // branch (num_threads == 1, or fewer QPs than threads).
                    // fetch_vectors_batch() funnels into the same 256-deep
                    // post_read_batch pipeline, so issue one location per call
                    // instead. ensure_fetch_buffer only ever grows, so the
                    // 1-entry calls cannot shrink the registered buffer. Each
                    // vector is copied out immediately, exactly as the batch
                    // loop does -> bit-identical.
                    //
                    // Kept verbatim (not folded into the windowed call below)
                    // for the same reason as the multi-QP branch: K=1 must be
                    // the measured naive point, not an approximation of it.
                    std::vector<uint32_t> one(1);
                    for (size_t i = 0; i < r_locs.size(); i++) {
                        one[0] = r_locs[i];
                        fetcher->fetch_vectors_batch(one);
                        const void* raw = fetcher->get_batch_vector(0);
                        T* dst = stage_vec();
                        std::memcpy(dst, raw, cmp_dim * sizeof(T));
                        vec_store[r_vids[i]] = dst;
                    }
                } else {
                // fetch_window_wr == 0 -> historical unlimited batch; >= 2 ->
                // same call, same SGL, same fetch-buffer slot layout (entry i
                // -> get_batch_vector(i)), only the in-flight cap is lowered to
                // K inside post_read_batch. The copy-out loop is shared, so the
                // window cannot perturb which bytes end up in vec_store.
                fetcher->fetch_vectors_batch(r_locs, fetch_window_wr);
                for (size_t i = 0; i < r_locs.size(); i++) {
                    const void* raw = fetcher->get_batch_vector(static_cast<uint32_t>(i));
                    T* dst = stage_vec();
                    std::memcpy(dst, raw, cmp_dim * sizeof(T));
                    vec_store[r_vids[i]] = dst;
                }
                }
                vecfetch_rdma_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::high_resolution_clock::now() - t_vrdma).count();
                if (num_threads > 1) {
                    fetcher_mutexes[owner].unlock();
                }
            }

            remote_vec_fetches += r_locs.size();
            if (cache_intercept && !node_store->local_data(owner)) {
                // Misses = vids the interceptor actually probed and failed;
                // LOCAL_MMAP owners are never probed so they don't count.
                cache_misses += r_locs.size();
                // Write fetched misses back into the search-warmed store so
                // later repair tasks hit instead of re-fetching — each remote
                // vector is re-requested ~40x per consolidation on average.
                // record_neighbor_vector dedups by vid under the shard lock
                // and saturates once the arena is full.
                for (size_t i = 0; i < r_locs.size(); i++) {
                    auto it = vec_store.find(r_vids[i]);
                    if (it != vec_store.end()) {
                        config.consolidation_cache->record_neighbor_vector(
                            0, r_vids[i], it->second, vec_bytes);
                        cache_writebacks++;
                    }
                }
            }

            if (ct_active) {
                for (size_t i = 0; i < r_locs.size(); i++) {
                    AccessRecord ar{};
                    ar.vid = r_vids[i];
                    ar.is_remote = 1;
                    ct_vec_acc.push_back(ar);
                }
            }
            if (dump_vec_fetch) {
                for (size_t i = 0; i < r_vids.size(); i++) {
                    uint64_t vid = r_vids[i];
                    uint32_t part = decode_partition(vid);
                    uint32_t loc = static_cast<uint32_t>(decode_local_loc(vid));
                    if (part < vec_fetch_count.size() &&
                        loc < vec_fetch_count[part].size()) {
                        vec_fetch_count[part][loc].fetch_add(
                            1, std::memory_order_relaxed);
                    }
                }
            }
        }

        // Local vectors: zero-copy pointers
        {
            auto t_lvec = std::chrono::high_resolution_clock::now();
            for (uint64_t vid : local_vids) {
                uint32_t ll = static_cast<uint32_t>(decode_local_loc(vid));
                vec_store[vid] = static_cast<const T*>(coloc_store->get_vector(ll));
                local_vec_fetches++;
                if (ct_active) {
                    AccessRecord ar{};
                    ar.vid = vid;
                    ar.is_remote = 0;
                    ct_vec_acc.push_back(ar);
                }
                if (dump_vec_fetch) {
                    uint32_t part = decode_partition(vid);
                    if (part < vec_fetch_count.size() &&
                        ll < vec_fetch_count[part].size()) {
                        vec_fetch_count[part][ll].fetch_add(
                            1, std::memory_order_relaxed);
                    }
                }
            }
            vecfetch_local_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now() - t_lvec).count();
        }

        // Step 3b: Compute distances
        {
            auto t_dist = std::chrono::high_resolution_clock::now();
            // Carry each candidate's vector pointer inline (PruneCandidate)
            // instead of {vid,dist}. The pointer is already in hand from this
            // vec_store.find; keeping it lets step 3b.5 (LITS) and the O(n^2)
            // robust_prune read candidates[i].vec directly, eliminating the
            // per-pair vec_store hashmap lookup that dominated the prune loop
            // on compute-bound (e.g. lits) consolidation. Cache-independent.
            auto& candidates = thr_candidates[tid];
            candidates.clear();
            candidates.reserve(expanded.size());
            // Pass A — resolve vid -> vector pointer. Touches only the
            // thread-local robin_map (a few tens of KB, L1/L2 resident).
            auto& cand_ptr = thr_cand_ptr[tid];
            cand_ptr.clear();
            cand_ptr.reserve(expanded.size());
            for (uint64_t vid : expanded) {
                auto it = vec_store.find(vid);
                if (it == vec_store.end()) {
                    diag_fetch_none++;
                    continue;
                }
                cand_ptr.emplace_back(vid, it->second);
            }
            // Pass B — the actual DRAM work. Each candidate vector is a cold
            // random read (cache arena / local store / staging arena), and on
            // deep100M/8p this step alone measures ~650 ns per candidate:
            // latency-bound, not bandwidth-bound (~38 GB/s node-wide). The
            // lookahead prefetch overlaps `prefetch_depth` of those misses.
            // Same order, same inputs, same distances as the fused loop.
            const size_t n_cand = cand_ptr.size();
            // P-F: devirtualised + B-lane batched kernel (float only, gated by
            // DANNS_CONSOL_PRUNE_OPT together with P-E). Same candidates, same
            // order, bit-identical distances, same prefetch set — see the
            // banner over dist_compute_batched_f32() in prune_utils.h.
            bool dist_done = false;
#ifdef DANNS_PRUNE_SIMD_F32
            if constexpr (std::is_same<T, float>::value) {
                if (prune_opt_enabled()) {
                    if (dist_opt_use_8_lanes(cmp_dim * sizeof(float))) {
                        dist_compute_batched_f32<8>(
                            node_vec, cand_ptr.data(), n_cand,
                            static_cast<uint32_t>(cmp_dim), prefetch_depth,
                            vec_lines, candidates);
                    } else {
                        dist_compute_batched_f32<2>(
                            node_vec, cand_ptr.data(), n_cand,
                            static_cast<uint32_t>(cmp_dim), prefetch_depth,
                            vec_lines, candidates);
                    }
                    total_dist_comps += n_cand;   // == n_cand increments
                    dist_done = true;
                }
            }
#endif
            if (!dist_done) {
              for (size_t ci = 0; ci < n_cand; ci++) {
                if (prefetch_depth > 0) {
                    const size_t pi = ci + static_cast<size_t>(prefetch_depth);
                    if (pi < n_cand) {
                        const char* pf =
                            reinterpret_cast<const char*>(cand_ptr[pi].second);
                        for (size_t b = 0; b < vec_lines; b++)
                            __builtin_prefetch(pf + b * 64, 0, 3);
                    }
                }
                float dist = dist_fn->compare(node_vec, cand_ptr[ci].second,
                                              cmp_dim);
                candidates.push_back(
                    {cand_ptr[ci].first, dist, cand_ptr[ci].second});
                total_dist_comps++;
              }
            }
            dist_compute_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now() - t_dist).count();

            // Step 3b.5: LITS filter -- remove long inter-partition 2-hop candidates
            uint64_t lits_checked_local = 0, lits_filtered_local = 0;
            if (enable_lits_filter) {
                // Build set of original (non-deleted) neighbor VIDs
                std::unordered_set<uint64_t> orig_nbr_vids;
                for (uint32_t i = 0; i < safe_degree; i++) {
                    uint64_t vid = nbr_ptr[i];
                    if (vid == INVALID_VID) break;
                    if (!del_idx.contains(vid)) {
                        orig_nbr_vids.insert(vid);
                    }
                }
                if (orig_nbr_vids.empty()) total_lits_no_survivor++;

                // tau sample from DELETED original neighbors (their vectors
                // joined the batch fetch above)
                double tau_del_sum = 0.0;
                uint32_t tau_del_cnt = 0;
                if (enable_lits_tau_deleted) {
                    for (uint64_t dvid : deleted_nbrs) {
                        auto it = vec_store.find(dvid);
                        if (it == vec_store.end()) continue;
                        tau_del_sum += dist_fn->compare(node_vec, it->second,
                                                        cmp_dim);
                        tau_del_cnt++;
                        total_dist_comps++;
                    }
                }
                auto stats = LITSFilter::filter_consolidation_candidates(
                    candidates, orig_nbr_vids, my_partition,
                    tau_del_sum, tau_del_cnt);
                lits_checked_local = stats.checked;
                lits_filtered_local = stats.filtered;
            }
            total_lits_checked += lits_checked_local;
            total_lits_filtered += lits_filtered_local;

            // Step 3c: Sort + RobustPrune
            auto t_prune = std::chrono::high_resolution_clock::now();
            std::sort(candidates.begin(), candidates.end(),
                      [](const auto& a, const auto& b) { return a.dist < b.dist; });

            std::vector<uint64_t> pruned = robust_prune_internal<T>(
                node_vec, candidates, R, alpha, maxc, dist_fn, cmp_dim);
            total_dist_comps += candidates.size() * candidates.size() / 2; // approximate O(n^2)
            prune_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now() - t_prune).count();

            // Step 3d: Write pruned neighbors
            uint32_t pruned_deg = static_cast<uint32_t>(pruned.size());
            write_neighbors_internal(coloc_store, coloc_config, local_loc,
                                     pruned.data(), pruned_deg);
            modified++;
            diag_path_prune++;
            degree_after_sum += pruned_deg;
            if (pruned_deg < R / 2) low_degree++;
            if (pruned_deg == 0) zero_degree++;
            update_histogram(pruned_deg);
            write_ct_entry(static_cast<uint16_t>(pruned_deg), 3);
        } // end of distance+prune scope

        per_thread_wall_ns[tid] += std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now() - t_iter_start).count();
        per_thread_entries[tid]++;
        } catch (const std::exception& e) {
            bool expected = false;
            if (err_flag.compare_exchange_strong(expected, true)) {
                std::lock_guard<std::mutex> lk(err_mtx);
                err_msg = e.what();
            }
        }
    }
    // ========================================================================
    // END MAIN PARALLEL LOOP
    // ========================================================================

    // C2: re-raise serially if a worker threw (peer death/congestion).
    if (err_flag.load()) throw std::runtime_error(err_msg);

    // Flush thread-local trace buffers (sorted by local_loc for deterministic output)
    if (ct_active) {
        std::vector<LocalTraceEntry> all_trace_entries;
        for (auto& tv : thread_traces) {
            all_trace_entries.insert(all_trace_entries.end(),
                std::make_move_iterator(tv.begin()),
                std::make_move_iterator(tv.end()));
        }
        std::sort(all_trace_entries.begin(), all_trace_entries.end(),
            [](const auto& a, const auto& b) {
                return a.header.local_loc < b.header.local_loc;
            });
        for (auto& te : all_trace_entries) {
            ctw.write_entry(te.header, te.hop_acc, te.vec_acc);
        }
    }
    ctw.close();

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_s = std::chrono::duration<double>(t_end - t_start).count();
    double loop_elapsed_s = std::chrono::duration<double>(t_end - t_loop_start).count();

    // --- Build stats ---
    ConsolidationStats stats;
    stats.entries_processed = n_entries;
    stats.entries_modified = modified;
    stats.entries_pruned = diag_path_prune;
    stats.total_deleted_nbrs = 0; // sum up from all entries
    stats.total_expanded = static_cast<uint64_t>(expanded_sum);
    stats.local_vec_accesses = local_vec_fetches;
    stats.remote_vec_accesses = remote_vec_fetches;
    stats.local_nbr_accesses = local_nbr_fetches;
    stats.remote_nbr_accesses = remote_nbr_fetches;
    if (rdma_wait_fail.load())
        rdma::Logger::warn("consolidation: skipped %lu remote Di 2-hop expansion(s) — RDMA send completion timeout/error",
                           static_cast<unsigned long>(rdma_wait_fail.load()));
    stats.cache_hits = cache_hits + nbr_cache_hits;
    stats.cache_misses = cache_misses + nbr_cache_misses;
    // OFF-path fetch interceptor: a vector hit is a remote vector served from
    // the search-warmed arena instead of RDMA; a neighbor hit is a remote Di
    // adjacency served from the compacted warm entries in the 2-hop expand.
    stats.cache_vector_hits = cache_hits;
    stats.cache_vector_misses = cache_misses;
    stats.cache_neighbor_hits = nbr_cache_hits;
    stats.cache_neighbor_misses = nbr_cache_misses;
    stats.cache_writebacks = cache_writebacks;
    stats.cache_nbr_writebacks = nbr_writebacks;
    stats.vec_bytes_saved_by_cache =
        static_cast<uint64_t>(cache_hits) * cmp_dim * sizeof(T) +
        static_cast<uint64_t>(nbr_cache_hits) *
            (sizeof(uint32_t) + coloc_config.slack_R * sizeof(uint64_t));
    // Record cache occupancy (how much of the configured arena search actually
    // warmed) so OFF-path runs report fill, not just hits. cache_entries also
    // reflects the compacted Di neighbor lists now that this path runs
    // prepare_for_consolidation().
    if (config.consolidation_cache) {
        auto cs = config.consolidation_cache->stats();
        stats.cache_entries = cs.entries;
        stats.cache_warm_vector_records = cs.vector_records;
        stats.cache_warm_neighbor_records = cs.neighbor_records;
        stats.cache_buffer_vector_bytes = cs.buffer_vector_bytes;
        stats.cache_compacted_vector_bytes = cs.compacted_vector_bytes;
        stats.cache_warm_bytes = cs.bytes;
        stats.cache_dropped_vector_records = cs.dropped_vector_records;
        stats.cache_dropped_neighbor_records = cs.dropped_neighbor_records;
    }
    stats.twohop_lock_wait_ns = twohop_lock_wait_ns;
    stats.vecfetch_lock_wait_ns = vecfetch_lock_wait_ns;
    stats.twohop_lock_contentions = twohop_lock_contentions;
    stats.vecfetch_lock_contentions = vecfetch_lock_contentions;
    stats.use_multi_qp = use_multi_qp;
    stats.elapsed_sec = elapsed_s;
    stats.loop_only_sec = loop_elapsed_s;  // OFF: main parallel loop wall
    // Phase offsets (OFF path): scan+work+prune fused into one parallel for.
    // Collapse scan/setup/populate to 0 so only "loop" gets emitted (the
    // ~70 ms of pre-parallel-for prep is not a meaningful "scan" phase —
    // labeling it scattered the plot's coverage). The single loop phase
    // covers the entire C++ wall.
    stats.scan_end_offset_sec = 0.0;
    stats.pop_start_offset_sec = 0.0;
    stats.pop_end_offset_sec = 0.0;
    stats.pass2_loop_start_offset_sec = 0.0;
    stats.pass2_loop_end_offset_sec = elapsed_s;

    // Degree histogram: flatten into vector
    if (!degree_histogram.empty()) {
        uint32_t max_bucket = degree_histogram.rbegin()->first;
        stats.degree_histogram.resize(max_bucket / 8 + 1, 0);
        for (auto& [bucket, count] : degree_histogram) {
            uint32_t idx_h = bucket / 8;
            if (idx_h < stats.degree_histogram.size()) {
                stats.degree_histogram[idx_h] = static_cast<uint32_t>(count);
            }
        }
    }

    // Log summary
    double avg_before = degree_count > 0 ? degree_before_sum / degree_count : 0.0;
    double avg_after = degree_count > 0 ? degree_after_sum / degree_count : 0.0;
    double avg_expanded = degree_count > 0 ? expanded_sum / degree_count : 0.0;

    rdma::Logger::info(
        "[CONSOLIDATE] done: modified=%lu path_empty=%lu path_no_prune=%lu path_prune=%lu "
        "avg_deg_before=%.1f avg_deg_after=%.1f avg_expanded=%.1f "
        "local_vec=%lu remote_vec=%lu local_nbr=%lu remote_nbr=%lu "
        "dist_comps=%lu low_deg=%lu zero_deg=%lu "
        "elapsed=%.3fs loop=%.3fs threads=%d "
        "lits_checked=%lu lits_filtered=%lu lits_no_survivor=%lu",
        modified, diag_path_empty, diag_path_no_prune, diag_path_prune,
        avg_before, avg_after, avg_expanded,
        local_vec_fetches, remote_vec_fetches, local_nbr_fetches, remote_nbr_fetches,
        total_dist_comps, low_degree, zero_degree,
        elapsed_s, loop_elapsed_s, num_threads,
        total_lits_checked, total_lits_filtered, total_lits_no_survivor);

    // Log per-thread work distribution
    for (int i = 0; i < actual_threads; i++) {
        rdma::Logger::info("[CONSOLIDATE] thread %d: entries=%lu wall_ns=%lu",
            i, per_thread_entries[i], per_thread_wall_ns[i]);
    }

    if (cache_intercept) {
        rdma::Logger::info(
            "[CONSOLIDATE CACHE] vec_hits=%lu vec_misses=%lu nbr_hits=%lu "
            "nbr_misses=%lu writebacks=%lu nbr_writebacks=%lu",
            cache_hits, cache_misses, nbr_cache_hits, nbr_cache_misses,
            cache_writebacks, nbr_writebacks);
    }

    // Log timing breakdown
    rdma::Logger::info(
        "[CONSOLIDATE TIMING] twohop_local=%lu twohop_rdma=%lu "
        "vecfetch_local=%lu vecfetch_rdma=%lu "
        "dist_compute=%lu prune=%lu (all ns)",
        twohop_local_ns, twohop_rdma_ns,
        vecfetch_local_ns, vecfetch_rdma_ns,
        dist_compute_ns, prune_ns);

    // Deleted node access frequency
    {
        uint64_t unique_deleted_accessed = 0;
        uint64_t total_deleted_accesses = 0;
        uint64_t max_deleted_accesses = 0;
        for (auto& [dv, cnt] : deleted_access_count) {
            uint64_t c = cnt.load();
            if (c > 0) {
                unique_deleted_accessed++;
                total_deleted_accesses += c;
                max_deleted_accesses = std::max(max_deleted_accesses, c);
            }
        }
        rdma::Logger::info(
            "[CONSOLIDATE DELETED ACCESS] unique_accessed=%lu total_accesses=%lu "
            "avg=%.1f max=%lu",
            unique_deleted_accessed, total_deleted_accesses,
            unique_deleted_accessed > 0
                ? static_cast<double>(total_deleted_accesses) / unique_deleted_accessed : 0.0,
            max_deleted_accesses);
    }

    // ---- Dump per-vector fetch count (binary VFCT format) -------------
    // Layout:
    //   char     magic[4]      = "VFCT"
    //   uint32_t version       = 1
    //   uint32_t num_partitions
    //   per partition:
    //     uint32_t partition_id
    //     uint32_t n_locs
    //     uint32_t counts[n_locs]
    if (dump_vec_fetch) {
        std::ofstream f(config.vec_fetch_count_dump_path,
                        std::ios::binary | std::ios::trunc);
        if (!f) {
            rdma::Logger::error(
                "[CONSOLIDATE VFCT] failed to open dump path: %s",
                config.vec_fetch_count_dump_path.c_str());
        } else {
            const char magic[4] = {'V', 'F', 'C', 'T'};
            uint32_t version = 1;
            uint32_t num_parts = static_cast<uint32_t>(vec_fetch_count.size());
            f.write(magic, 4);
            f.write(reinterpret_cast<const char*>(&version), sizeof(version));
            f.write(reinterpret_cast<const char*>(&num_parts), sizeof(num_parts));
            uint64_t total_fetches = 0;
            uint64_t unique_fetched = 0;
            uint32_t max_fetches = 0;
            for (uint32_t p = 0; p < num_parts; p++) {
                uint32_t n_locs = static_cast<uint32_t>(vec_fetch_count[p].size());
                f.write(reinterpret_cast<const char*>(&p), sizeof(p));
                f.write(reinterpret_cast<const char*>(&n_locs), sizeof(n_locs));
                std::vector<uint32_t> snapshot(n_locs);
                for (uint32_t i = 0; i < n_locs; i++) {
                    uint32_t c = vec_fetch_count[p][i].load(std::memory_order_relaxed);
                    snapshot[i] = c;
                    total_fetches += c;
                    if (c > 0) unique_fetched++;
                    if (c > max_fetches) max_fetches = c;
                }
                f.write(reinterpret_cast<const char*>(snapshot.data()),
                        n_locs * sizeof(uint32_t));
            }
            f.close();
            rdma::Logger::info(
                "[CONSOLIDATE VFCT] dumped %s: parts=%u "
                "total_fetches=%lu unique_fetched=%lu avg=%.1f max=%u",
                config.vec_fetch_count_dump_path.c_str(), num_parts,
                total_fetches, unique_fetched,
                unique_fetched > 0
                    ? static_cast<double>(total_fetches) / unique_fetched : 0.0,
                max_fetches);
        }
    }

    return stats;
}

// ========================================================================
// run_with_per_di_impl: per-deleted-node cache layout.
//
// !!! DEPRECATED (2026-06-18) — DO NOT EXTEND !!!
// This path makes cache-on a DIFFERENT algorithm than cache-off: it globally
// dedups Di's and PREFETCHES every remote (Di,N) vector into one contiguous
// vec_storage (= n_remote_total * vec_bytes) before pruning. Consequences the
// intended design never wanted:
//   - vec_storage materialization (71 GB on lits node4; OOM on round_robin),
//     stored REDUNDANTLY alongside the 64 GiB search-warmed cache arena;
//   - a serial "populate" phase (memcpy every cache hit into vec_storage) that
//     dominated wall-clock and made cache-on SLOWER than cache-off despite a
//     97.5% hit rate (deep100M 10% del: 325 s off -> 677 s on).
// The cache is being moved to a transparent fetch interceptor on the OFF/legacy
// inline path: cache-on == cache-off except hit -> skip RDMA, miss -> RDMA. That
// keeps the algorithm identical and isolates the pure cache (RDMA-saved) effect.
// See project memory: project_consol_cache_walltime_exp.md.
//
// Differences vs static_cache_impl:
//   - Phase 1 (scan): no Di-neighbor RDMA. Each work_item just records its
//     deleted-neighbor list (Di's). thread_di_set collects the union.
//   - Phase 1.5: merge thread_di_set → di_list (~1M entries).
//   - Phase 1b: fetch each unique Di's neighbor list ONCE (vs the per-vid
//     path's 30x redundant Di-fetches inside scan). Stored in CSR layout.
//   - Phase 2: fetch each (Di, N) pair's vector. Cross-Di duplicates are
//     not deduped — at typical workloads this only adds ~1.3x BW vs the
//     unique-vid path, in exchange for eliminating the merge cost.
//   - Pass 2 (prune): per-A iterates A.deleted_nbrs → di_buffer[Di] in CSR
//     order. Within-A dedup uses a small per-iteration set.
// ========================================================================
template <typename T>
static ConsolidationStats run_with_per_di_impl(
    const ConsolidationConfig& config,
    NodeStore* node_store,
    PartitionMapper* mapper,
    diskann::Distance<T>* dist_fn,
    const std::unordered_set<uint64_t>& deleted_set,
    uint32_t /*vector_dim*/,
    uint32_t aligned_dim)
{
    rdma::ColocationStore* coloc_store = node_store->local_store();
    const rdma::ColocationConfig& coloc_config = node_store->config();
    const uint32_t my_partition = node_store->partition_id();
    const uint32_t my_node = node_store->node_id();
    // Compare/copy over real vector_dim (= coloc_config.dim here, since the
    // vector_dim param is unused in this overload), not aligned_dim — the entry
    // vector field is vector_dim wide; aligned_dim over-reads degree bytes when
    // dim%8!=0 (spacev). cmp_dim drives RDMA fetch len + memcpy + vec_storage
    // here, so this keeps them all consistent. deep/sift/laion byte-identical.
    const size_t cmp_dim = static_cast<size_t>(coloc_config.dim);
    const uint32_t R = config.R;
    const float alpha = config.alpha;
    const uint32_t maxc = config.maxc;
    const int num_threads = static_cast<int>(config.num_threads);
    const bool enable_lits_filter = config.lits_filter;
    const uint32_t n_entries = coloc_config.num_entries;

    auto t_start = std::chrono::high_resolution_clock::now();

    const uint32_t num_qps = node_store->num_thread_qps();
    const bool use_multi_qp = (num_threads > 1) &&
                              (num_qps >= static_cast<uint32_t>(num_threads));
    rdma::Logger::info("[CONSOLIDATE-PERDI][ZEROCOPY] per-Di cache path: hits read "
        "the search-warmed arena via lookup_vector_ptr (no vec_storage copy); "
        "only misses are staged. Compact prune layout + cache-driven, low memory.");
    rdma::Logger::info("[CONSOLIDATE-PERDI] multi_qp=%s num_threads=%d",
        use_multi_qp ? "true" : "false", num_threads);
    if (num_threads > 1) omp_set_num_threads(num_threads);

    // C2: capture exceptions escaping the RDMA-issuing OMP loops below (an
    // RDMAException unwinding out of a parallel region -> std::terminate ->
    // SIGABRT). First error wins; re-thrown serially right after the offending
    // loop so pybind converts it to a Python exception.
    std::atomic<bool> err_flag{false};
    std::string err_msg;
    std::mutex err_mtx;

    // Counters
    uint64_t modified = 0;
    uint64_t diag_path_empty = 0, diag_path_no_prune = 0, diag_path_prune = 0;
    uint64_t local_vec_fetches = 0, remote_vec_fetches = 0;
    uint64_t local_nbr_fetches = 0, remote_nbr_fetches = 0;
    // Remote Di neighbor fetches skipped on RDMA send completion timeout/error.
    std::atomic<uint64_t> rdma_wait_fail{0};
    uint64_t total_dist_comps = 0;
    double degree_before_sum = 0, degree_after_sum = 0;
    uint64_t degree_count = 0;
    double expanded_sum = 0;
    uint64_t low_degree = 0, zero_degree = 0;
    std::map<uint32_t, uint64_t> degree_histogram;
    uint64_t total_lits_checked = 0, total_lits_filtered = 0;

    const size_t vec_bytes_per = cmp_dim * sizeof(T);
    const uint32_t max_R = coloc_config.slack_R;

    // ------------------------------------------------------------------
    // Phase 1: lightweight scan. Per A, just collect deleted_nbr list.
    // ------------------------------------------------------------------
    struct WorkItemLite {
        uint32_t local_loc;
        uint32_t degree;
        std::vector<uint64_t> deleted_nbrs;     // Di vids that A points to
        std::vector<uint32_t> deleted_nbr_idx;  // Resolved Di → di_list index
                                                // (filled after Phase 1.5 builds di_idx_map)
    };

    int actual_threads = (num_threads > 1) ? num_threads : 1;
    std::vector<std::vector<WorkItemLite>> thread_work(actual_threads);
    std::vector<std::unordered_set<uint64_t>> thread_di_set(actual_threads);

    auto t_scan_start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for schedule(dynamic, 64) if(num_threads > 1) \
        reduction(+:degree_before_sum, degree_count)
    for (uint32_t idx = 0; idx < n_entries; idx++) {
        int tid = (num_threads > 1) ? omp_get_thread_num() : 0;
        uint64_t entry_vid = danns::encode_vector_id(my_partition, idx);
        if (deleted_set.count(entry_vid)) continue;
        uint32_t degree = coloc_store->get_degree(idx);
        const uint64_t* nbrs = coloc_store->get_neighbors(idx);
        WorkItemLite wi;
        wi.local_loc = idx;
        wi.degree = degree;
        uint32_t safe_deg = std::min(degree, max_R);
        for (uint32_t i = 0; i < safe_deg; i++) {
            uint64_t v = nbrs[i];
            if (v == INVALID_VID) break;
            if (deleted_set.count(v)) {
                wi.deleted_nbrs.push_back(v);
                thread_di_set[tid].insert(v);
            }
        }
        if (wi.deleted_nbrs.empty()) continue;
        degree_before_sum += degree;
        degree_count++;
        thread_work[tid].push_back(std::move(wi));
    }
    auto t_scan_end = std::chrono::high_resolution_clock::now();

    // Merge work_items
    std::vector<WorkItemLite> work_items;
    {
        size_t total = 0;
        std::vector<size_t> off(actual_threads + 1, 0);
        for (int t = 0; t < actual_threads; t++) off[t + 1] = off[t] + thread_work[t].size();
        total = off[actual_threads];
        work_items.resize(total);
        #pragma omp parallel for schedule(static) if(num_threads > 1)
        for (int t = 0; t < actual_threads; t++) {
            std::move(thread_work[t].begin(), thread_work[t].end(),
                      work_items.begin() + off[t]);
        }
        thread_work.clear();
        thread_work.shrink_to_fit();
    }

    // ------------------------------------------------------------------
    // Phase 1.5: merge thread_di_set → di_list (sorted, unique)
    // ------------------------------------------------------------------
    std::vector<uint64_t> di_list;
    {
        size_t total = 0;
        for (auto& s : thread_di_set) total += s.size();
        di_list.resize(total);
        std::vector<size_t> off(thread_di_set.size() + 1, 0);
        for (size_t i = 0; i < thread_di_set.size(); i++)
            off[i + 1] = off[i] + thread_di_set[i].size();
        #pragma omp parallel for schedule(static) if(num_threads > 1)
        for (size_t i = 0; i < thread_di_set.size(); i++) {
            size_t cur = off[i];
            for (uint64_t v : thread_di_set[i]) di_list[cur++] = v;
        }
        // Parallel free (heap node destruction)
        #pragma omp parallel for schedule(static) if(num_threads > 1)
        for (size_t i = 0; i < thread_di_set.size(); i++) {
            std::unordered_set<uint64_t> tmp;
            tmp.swap(thread_di_set[i]);
        }
        thread_di_set.clear();
        thread_di_set.shrink_to_fit();
        __gnu_parallel::sort(di_list.begin(), di_list.end());
        auto last = std::unique(di_list.begin(), di_list.end());
        di_list.erase(last, di_list.end());
        di_list.shrink_to_fit();
    }
    const size_t N_di = di_list.size();

    std::unordered_map<uint64_t, uint32_t> di_idx_map;
    di_idx_map.reserve(N_di);
    for (uint32_t i = 0; i < N_di; i++) di_idx_map[di_list[i]] = i;

    // Pre-resolve each work_item's deleted_nbrs → di_list indices, so
    // pass2 can iterate by index instead of doing per-A hashmap lookups.
    #pragma omp parallel for schedule(static) if(num_threads > 1)
    for (size_t k = 0; k < work_items.size(); k++) {
        auto& w = work_items[k];
        w.deleted_nbr_idx.reserve(w.deleted_nbrs.size());
        for (uint64_t Di : w.deleted_nbrs) {
            auto it = di_idx_map.find(Di);
            if (it != di_idx_map.end()) {
                w.deleted_nbr_idx.push_back(it->second);
            }
        }
    }
    auto t_di_merge_end = std::chrono::high_resolution_clock::now();

    // ------------------------------------------------------------------
    // Phase 1b: fetch each Di's neighbor list (vid IDs).
    //   CSR layout with fixed stride max_R for simplicity.
    //   di_neighbor_vids[i * max_R + j] = j-th neighbor of di_list[i]
    //   di_degree[i] = actual neighbor count (≤ max_R)
    // ------------------------------------------------------------------
    std::vector<uint32_t> di_degree(N_di, 0);
    std::vector<uint64_t> di_neighbor_vids(N_di * max_R, INVALID_VID);
    std::vector<ConsolidationCacheEntry> warmed_entries(N_di);
    std::vector<uint8_t> warmed_neighbor_hit(N_di, 0);
    uint64_t cache_neighbor_hits = 0;
    uint64_t cache_neighbor_misses = 0;
    uint64_t cache_vector_hits = 0;
    uint64_t cache_vector_misses = 0;
    uint64_t cache_entries_seen = 0;
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

    if (config.consolidation_cache) {
        config.consolidation_cache->prepare_for_consolidation();
        auto cs = config.consolidation_cache->stats();
        cache_entries_seen = cs.entries;
        cache_warm_expansion_events = cs.expansion_events;
        cache_warm_neighbor_records = cs.neighbor_records;
        cache_warm_vector_records = cs.vector_records;
        cache_warm_bytes = cs.bytes;
        cache_buffer_neighbor_bytes = cs.buffer_neighbor_bytes;
        cache_buffer_vector_bytes = cs.buffer_vector_bytes;
        cache_compacted_neighbor_bytes = cs.compacted_neighbor_bytes;
        cache_compacted_vector_bytes = cs.compacted_vector_bytes;
        cache_compacted_index_bytes = cs.compacted_index_bytes;
        cache_dropped_neighbor_records = cs.dropped_neighbor_records;
        cache_dropped_vector_records = cs.dropped_vector_records;
        for (uint32_t i = 0; i < N_di; i++) {
            ConsolidationCacheEntry entry;
            if (!config.consolidation_cache->lookup(di_list[i], entry) ||
                entry.neighbors.empty()) {
                cache_neighbor_misses++;
                continue;
            }
            uint32_t deg = std::min<uint32_t>(entry.degree, max_R);
            deg = std::min<uint32_t>(deg, static_cast<uint32_t>(entry.neighbors.size()));
            size_t base = static_cast<size_t>(i) * max_R;
            for (uint32_t j = 0; j < deg; j++) {
                di_neighbor_vids[base + j] = entry.neighbors[j];
            }
            di_degree[i] = deg;
            warmed_entries[i] = std::move(entry);
            warmed_neighbor_hit[i] = 1;
            cache_neighbor_hits++;
        }
    }

    // Bucket Di's by owner for parallel cross-peer fetch.
    auto& ts_map_phase1b = node_store->thread_states_mut();
    struct DiFetch {
        uint32_t di_idx;
        uint32_t local_loc;
        uint32_t owner;
    };
    std::vector<DiFetch> remote_di;
    remote_di.reserve(N_di);
    std::vector<DiFetch> local_di;
    local_di.reserve(N_di / 4);
    for (uint32_t i = 0; i < N_di; i++) {
        if (warmed_neighbor_hit[i]) continue;
        uint64_t Di = di_list[i];
        uint32_t part = decode_partition(Di);
        uint32_t ll = static_cast<uint32_t>(decode_local_loc(Di));
        uint32_t owner = mapper->partition_id_to_node_id(part);
        DiFetch f{i, ll, owner};
        if (owner == my_node) local_di.push_back(f);
        else remote_di.push_back(f);
    }

    // Local Di's: parallel direct read.
    #pragma omp parallel for schedule(static) if(num_threads > 1) \
        reduction(+:local_nbr_fetches)
    for (size_t k = 0; k < local_di.size(); k++) {
        const DiFetch& f = local_di[k];
        uint32_t deg = coloc_store->get_degree(f.local_loc);
        if (deg > max_R) deg = max_R;
        const uint64_t* ptr = coloc_store->get_neighbors(f.local_loc);
        for (uint32_t j = 0; j < deg; j++) {
            di_neighbor_vids[static_cast<size_t>(f.di_idx) * max_R + j] = ptr[j];
        }
        di_degree[f.di_idx] = deg;
        local_nbr_fetches++;
    }

    // Remote Di's: per-thread parallel RDMA, one Di at a time per thread.
    // LOCAL_MMAP fast-path takes precedence when peer partition is mmap'd
    // locally (room-at-the-top): read degree+nbrs from local pointer, no RDMA.
    #pragma omp parallel for schedule(dynamic) if(num_threads > 1) \
        reduction(+:remote_nbr_fetches)
    for (size_t k = 0; k < remote_di.size(); k++) {
        if (err_flag.load(std::memory_order_relaxed)) continue;
        try {
        int tid = (num_threads > 1) ? omp_get_thread_num() : 0;
        const DiFetch& f = remote_di[k];
        if (const char* lb = node_store->local_data(f.owner)) {
            uint64_t off = static_cast<uint64_t>(f.local_loc) *
                coloc_config.entry_size + coloc_config.degree_offset;
            uint32_t deg = 0;
            std::memcpy(&deg, lb + off, sizeof(uint32_t));
            if (deg > max_R) deg = max_R;
            const uint64_t* raw = reinterpret_cast<const uint64_t*>(
                lb + off + sizeof(uint32_t));
            size_t base = static_cast<size_t>(f.di_idx) * max_R;
            for (uint32_t j = 0; j < deg; j++) di_neighbor_vids[base + j] = raw[j];
            di_degree[f.di_idx] = deg;
            remote_nbr_fetches++;
            continue;
        }
        auto ts_it = ts_map_phase1b.find(f.owner);
        const auto* mr_ptr = node_store->remote_mr(f.owner);
        if (ts_it == ts_map_phase1b.end() || !mr_ptr) continue;
        auto& state = ts_it->second.at(tid);
        uint32_t nbr_read_sz = sizeof(uint32_t) +
            max_R * sizeof(uint64_t);
        uint64_t remote_offset = static_cast<uint64_t>(f.local_loc) *
            coloc_config.entry_size + coloc_config.degree_offset;
        rdma::RemoteMemoryRegion target = *mr_ptr;
        target.addr += remote_offset;
        target.size = nbr_read_sz;
        char* buf = state.neighbor_fetch_buffer.data();
        state.conn->post_read(buf, nbr_read_sz, state.neighbor_fetch_mr, target, 0);
        if (!state.conn->wait_send_completion(0, 5000)) {
            rdma_wait_fail.fetch_add(1, std::memory_order_relaxed);
            continue;  // skip this Di; do not read the stale buffer
        }
        uint32_t deg = 0;
        std::memcpy(&deg, buf, sizeof(uint32_t));
        if (deg > max_R) deg = max_R;
        const uint64_t* raw = reinterpret_cast<const uint64_t*>(
            buf + sizeof(uint32_t));
        size_t base = static_cast<size_t>(f.di_idx) * max_R;
        for (uint32_t j = 0; j < deg; j++) di_neighbor_vids[base + j] = raw[j];
        di_degree[f.di_idx] = deg;
        remote_nbr_fetches++;
        } catch (const std::exception& e) {
            bool expected = false;
            if (err_flag.compare_exchange_strong(expected, true)) {
                std::lock_guard<std::mutex> lk(err_mtx);
                err_msg = e.what();
            }
        }
    }
    if (err_flag.load()) throw std::runtime_error(err_msg);

    // Phase 1c: mark deleted N's as INVALID_VID so the inner pass2 loop
    // can skip them via a single array compare (no deleted_set hashmap
    // lookup per candidate).
    #pragma omp parallel for schedule(static) if(num_threads > 1)
    for (size_t i = 0; i < N_di; i++) {
        uint32_t deg = di_degree[i];
        size_t base = i * max_R;
        for (uint32_t j = 0; j < deg; j++) {
            uint64_t N = di_neighbor_vids[base + j];
            if (N == INVALID_VID) continue;
            if (deleted_set.count(N)) {
                di_neighbor_vids[base + j] = INVALID_VID;
            }
        }
    }
    auto t_di_nbr_end = std::chrono::high_resolution_clock::now();

    // ------------------------------------------------------------------
    // Phase 2: fetch each REMOTE (Di, N) pair's vector.
    //   LOCAL (Di, N): skipped — prune reads coloc_store directly (zero-copy).
    //   REMOTE (Di, N): assigned a compact slot in vec_storage; fetched via RDMA.
    //
    // remote_idx[base+j] encodes the per-pair lookup state:
    //     -1  → either skip (INVALID/deleted) OR local — handle in prune via
    //           coloc_store lookup (the actual vid still lives in
    //           di_neighbor_vids; partition can be decoded directly)
    //     ≥0  → index into vec_storage (compact, contiguous over remote pairs)
    //
    // Effect vs prior layout (full N_di × max_R × vec_bytes_per buffer):
    //     vec_storage shrinks by (1 - remote_fraction). For LITS ~10% remote
    //     this is a ~90% memory reduction; for RR ~75% remote ~25%.
    // ------------------------------------------------------------------
    // remote_idx[base+j] encodes per-pair vector source for Pass 2:
    //   -1 (REMOTE_IDX_LOCAL)     → N is local to this node, coloc zero-copy
    //   -2 (REMOTE_IDX_CACHE_HIT) → cache hit, read arena via lookup_vector_ptr
    //   ≥0                        → miss, index into compact vec_storage (RDMA)
    constexpr int32_t REMOTE_IDX_CACHE_HIT = -2;
    std::vector<int32_t> remote_idx(static_cast<size_t>(N_di) * max_R, -1);

    // Bucket REMOTE-MISS (di_idx, j) pairs by N's owner. Local pairs and cache
    // hits get no vec_storage slot (hits read the arena at prune time).
    std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, uint32_t>>> n_by_owner;
    n_by_owner.reserve(8);
    size_t n_remote_total = 0;
    for (uint32_t i = 0; i < N_di; i++) {
        uint32_t deg = di_degree[i];
        size_t base = static_cast<size_t>(i) * max_R;
        for (uint32_t j = 0; j < deg; j++) {
            uint64_t N = di_neighbor_vids[base + j];
            if (N == INVALID_VID) continue;  // also covers deleted N's (Phase 1c)
            uint32_t part = decode_partition(N);
            uint32_t owner = mapper->partition_id_to_node_id(part);
            if (owner == my_node) {
                // Local — coloc_store handles read, no vec_storage slot.
                continue;
            }
            // ZERO-COPY: a cache HIT is served at prune time from the arena
            // (marker -2, no slot, no copy). Only MISSES get a compact
            // vec_storage slot + RDMA/local-mmap fetch.
            bool warmed_vec = false;
            if (config.consolidation_cache)
                warmed_vec = config.consolidation_cache->has_vector(N);
            if (warmed_vec) {
                remote_idx[base + j] = REMOTE_IDX_CACHE_HIT;
                cache_vector_hits++;
            } else {
                remote_idx[base + j] = static_cast<int32_t>(n_remote_total++);
                n_by_owner[owner].push_back({i, j});
                if (config.consolidation_cache) cache_vector_misses++;
            }
        }
    }

    // Allocate vec_storage compactly — only n_remote_total slots.
    const size_t vec_storage_bytes = n_remote_total * vec_bytes_per;
    uint8_t* vec_storage_ptr = nullptr;
    if (vec_storage_bytes > 0) {
        void* p = ::mmap(nullptr, vec_storage_bytes,
                         PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) throw std::runtime_error("perdi vec_storage mmap failed");
        vec_storage_ptr = static_cast<uint8_t*>(p);
        ::madvise(vec_storage_ptr, vec_storage_bytes, MADV_HUGEPAGE);
        #pragma omp parallel for schedule(static) if(num_threads > 1)
        for (size_t off = 0; off < vec_storage_bytes; off += 4096) {
            const size_t end = std::min(off + 4096, vec_storage_bytes);
            std::memset(vec_storage_ptr + off, 0, end - off);
        }
    }
    struct VecGuard { uint8_t* p; size_t n; ~VecGuard() { if (p) ::munmap(p, n); } }
        _vg{vec_storage_ptr, vec_storage_bytes};

    auto t_pop_start = std::chrono::high_resolution_clock::now();

    // (ZERO-COPY: the serial hit-populate loop is gone. Cache hits are NOT
    // copied into vec_storage — they are read directly from the arena via
    // lookup_vector_ptr in Pass 2. vec_storage now holds ONLY misses, so this
    // "populate" phase is just the parallel miss fetch below. This removes the
    // single-thread memcpy bottleneck and the double-storage OOM.)
    // (Local memcpy phase removed: local (Di, N) pairs are read directly
    // from coloc_store at prune time. Their vec_storage slot does not exist.)

    // Remote per peer (cross-peer parallel via flat chunk list).
    auto& ts_map_phase2 = node_store->thread_states_mut();
    struct PeerCtx2 {
        std::vector<NodeStore::ThreadRDMAState>* states = nullptr;
        const rdma::RemoteMemoryRegion* mr = nullptr;
        size_t buf_entries_cap = 1;
        bool used = false;
    };
    std::unordered_map<uint32_t, PeerCtx2> peer_ctx2;
    struct ChunkIdx { uint32_t owner; size_t off; };
    std::vector<ChunkIdx> chunks2;
    for (auto& [owner, pairs] : n_by_owner) {
        if (owner == my_node) continue;
        // LOCAL_MMAP peers handled in a separate fast-path below; skip the
        // RDMA chunk setup so the parallel post_read_batch loop never runs
        // for them.
        if (node_store->local_data(owner)) continue;
        auto ts_it = ts_map_phase2.find(owner);
        const auto* mr_ptr = node_store->remote_mr(owner);
        if (!use_multi_qp || ts_it == ts_map_phase2.end() || !mr_ptr) continue;
        const auto& sample = ts_it->second.at(0);
        const size_t buf_bytes = sample.fetch_buffer.size() * sizeof(float);
        size_t cap = (coloc_config.entry_size == 0) ? 1 :
            (buf_bytes / coloc_config.entry_size);
        if (cap == 0) cap = 1;
        PeerCtx2 c{};
        c.states = &ts_it->second;
        c.mr = mr_ptr;
        c.buf_entries_cap = cap;
        c.used = true;
        peer_ctx2[owner] = c;
        for (size_t off = 0; off < pairs.size(); off += cap) {
            chunks2.push_back({owner, off});
        }
    }

    // LOCAL_MMAP fast-path for peers whose partitions are mmap'd locally.
    // memcpy each (Di, N) pair's vector from the local pointer into
    // vec_storage. Parallelized over (di_idx, j) pairs across all such peers.
    {
        struct LocalPair { const char* lb; uint32_t di_idx; uint32_t j; uint64_t N; };
        std::vector<LocalPair> local_pairs;
        size_t local_total_est = 0;
        for (auto& [owner, pairs] : n_by_owner) {
            const char* lb = node_store->local_data(owner);
            if (!lb || owner == my_node) continue;
            local_total_est += pairs.size();
        }
        local_pairs.reserve(local_total_est);
        for (auto& [owner, pairs] : n_by_owner) {
            const char* lb = node_store->local_data(owner);
            if (!lb || owner == my_node) continue;
            for (auto& p : pairs) {
                uint32_t di_idx = p.first;
                uint32_t j = p.second;
                uint64_t N = di_neighbor_vids[
                    static_cast<size_t>(di_idx) * max_R + j];
                local_pairs.push_back({lb, di_idx, j, N});
            }
        }
        const size_t LP = local_pairs.size();
        #pragma omp parallel for schedule(static) if(num_threads > 1) \
            reduction(+:remote_vec_fetches)
        for (size_t i = 0; i < LP; i++) {
            const LocalPair& lp = local_pairs[i];
            uint32_t ll = static_cast<uint32_t>(decode_local_loc(lp.N));
            const void* raw = lp.lb +
                static_cast<size_t>(ll) * coloc_config.entry_size +
                coloc_config.vector_offset;
            int32_t ridx = remote_idx[
                static_cast<size_t>(lp.di_idx) * max_R + lp.j];
            std::memcpy(vec_storage_ptr +
                        static_cast<size_t>(ridx) * vec_bytes_per,
                        raw, vec_bytes_per);
            remote_vec_fetches++;
        }
    }

    #pragma omp parallel for schedule(dynamic) if(num_threads > 1) \
        reduction(+:remote_vec_fetches)
    for (size_t ci = 0; ci < chunks2.size(); ci++) {
        if (err_flag.load(std::memory_order_relaxed)) continue;
        try {
        int tid = (num_threads > 1) ? omp_get_thread_num() : 0;
        ChunkIdx ck = chunks2[ci];
        auto& ctx = peer_ctx2[ck.owner];
        auto& pairs = n_by_owner[ck.owner];
        size_t total = pairs.size();
        size_t chunk = std::min(ctx.buf_entries_cap, total - ck.off);
        auto& state = ctx.states->at(tid);
        char* buf = reinterpret_cast<char*>(state.fetch_buffer.data());

        std::vector<rdma::SGLEntry> sgl(chunk);
        for (size_t i = 0; i < chunk; i++) {
            uint32_t di_idx = pairs[ck.off + i].first;
            uint32_t j = pairs[ck.off + i].second;
            uint64_t N = di_neighbor_vids[static_cast<size_t>(di_idx) * max_R + j];
            uint32_t ll = static_cast<uint32_t>(decode_local_loc(N));
            sgl[i].local_addr = buf +
                static_cast<size_t>(i) * coloc_config.entry_size +
                coloc_config.vector_offset;
            sgl[i].length = vec_bytes_per;
            sgl[i].remote_offset = static_cast<uint64_t>(ll) *
                coloc_config.entry_size + coloc_config.vector_offset;
        }
        state.conn->post_read_batch(sgl, *ctx.mr, state.fetch_buffer_mr, 0);
        for (size_t i = 0; i < chunk; i++) {
            uint32_t di_idx = pairs[ck.off + i].first;
            uint32_t j = pairs[ck.off + i].second;
            const void* raw = buf + i * coloc_config.entry_size +
                              coloc_config.vector_offset;
            int32_t ridx = remote_idx[
                static_cast<size_t>(di_idx) * max_R + j];
            // ridx must be ≥0 here — only remote pairs are scheduled.
            std::memcpy(vec_storage_ptr +
                        static_cast<size_t>(ridx) * vec_bytes_per,
                        raw, vec_bytes_per);
        }
        remote_vec_fetches += chunk;
        } catch (const std::exception& e) {
            bool expected = false;
            if (err_flag.compare_exchange_strong(expected, true)) {
                std::lock_guard<std::mutex> lk(err_mtx);
                err_msg = e.what();
            }
        }
    }
    if (err_flag.load()) throw std::runtime_error(err_msg);
    auto t_pop_end = std::chrono::high_resolution_clock::now();

    // ------------------------------------------------------------------
    // Phase 3 (Pass 2): per-A prune. Iterate Di's, expand via di_buffer,
    // dedup-within-A, distance-compute, sort, prune.
    // ------------------------------------------------------------------
    const int pass2_threads = (config.pass2_num_threads > 0)
        ? std::min<int>(static_cast<int>(config.pass2_num_threads), num_threads)
        : num_threads;
    rdma::Logger::info("[CONSOLIDATE-PERDI] pass2_num_threads=%d (parent=%d) "
                       "N_di=%zu work_items=%zu",
                       pass2_threads, num_threads, N_di, work_items.size());

    auto& ts_map_phase3 = node_store->thread_states_mut();

    auto t_pass2_loop_start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for schedule(dynamic, 64) num_threads(pass2_threads) \
        if(pass2_threads > 1) \
        reduction(+:modified, diag_path_empty, diag_path_no_prune, diag_path_prune, \
                  total_dist_comps, degree_after_sum, low_degree, zero_degree, \
                  total_lits_checked, total_lits_filtered, expanded_sum, \
                  local_vec_fetches, remote_vec_fetches)
    for (size_t wi_idx = 0; wi_idx < work_items.size(); wi_idx++) {
        if (err_flag.load(std::memory_order_relaxed)) continue;
        try {
        auto& w = work_items[wi_idx];
        int tid = (pass2_threads > 1) ? omp_get_thread_num() : 0;
        const uint32_t local_loc = w.local_loc;
        const uint32_t degree = w.degree;
        uint64_t my_vid = danns::encode_vector_id(my_partition, local_loc);
        const T* node_vec = static_cast<const T*>(coloc_store->get_vector(local_loc));

        std::unordered_set<uint64_t> seen;
        seen.reserve(static_cast<size_t>(w.deleted_nbrs.size()) * max_R + degree);
        std::vector<PruneCandidate<T>> candidates;
        candidates.reserve(static_cast<size_t>(w.deleted_nbrs.size()) * max_R + degree);

        // ==================================================================
        // Part 1: A's surviving (non-deleted) neighbors as candidates.
        //   Local → zero-copy from coloc_store.
        //   Remote → batched RDMA per peer (one post_read_batch per peer).
        //            Vectors copied into a per-A scratch buffer so candidate
        //            vec pointers stay valid after fetch_buffer reuse.
        // ==================================================================
        const uint64_t* nbr_ptr = coloc_store->get_neighbors(local_loc);
        const uint32_t safe_deg = std::min(degree, max_R);

        // Bucket A's non-deleted, not-yet-seen neighbors by owner.
        std::unordered_map<uint32_t, std::vector<uint64_t>> a_nbrs_by_owner;
        std::vector<uint64_t> a_local_nbrs;
        for (uint32_t i = 0; i < safe_deg; i++) {
            uint64_t N = nbr_ptr[i];
            if (N == INVALID_VID) break;
            if (deleted_set.count(N)) continue;
            if (N == my_vid) continue;
            if (!seen.insert(N).second) continue;
            uint32_t N_part = decode_partition(N);
            uint32_t owner = mapper->partition_id_to_node_id(N_part);
            if (owner == my_node) {
                a_local_nbrs.push_back(N);
            } else {
                a_nbrs_by_owner[owner].push_back(N);
            }
        }

        // 1a: local A-nbrs — zero-copy.
        for (uint64_t N : a_local_nbrs) {
            uint32_t ll = static_cast<uint32_t>(decode_local_loc(N));
            const T* v = static_cast<const T*>(coloc_store->get_vector(ll));
            float dist = dist_fn->compare(node_vec, v, cmp_dim);
            candidates.push_back({N, dist, v});
            local_vec_fetches++;
        }

        // 1b: remote A-nbrs — batched on-demand RDMA per peer.
        // Per-A scratch holds the fetched vectors so candidate.vec stays valid.
        size_t total_remote = 0;
        for (auto& [_, v] : a_nbrs_by_owner) total_remote += v.size();
        std::vector<T> a_remote_vecs;
        a_remote_vecs.resize(total_remote * cmp_dim);
        size_t buf_off = 0;
        for (auto& [owner, owner_nbrs] : a_nbrs_by_owner) {
            // LOCAL_MMAP fast-path: peer partition mmap'd locally — copy
            // vectors via memcpy from the local pointer instead of RDMA.
            if (const char* lb = node_store->local_data(owner)) {
                for (uint64_t N : owner_nbrs) {
                    uint32_t ll = static_cast<uint32_t>(decode_local_loc(N));
                    const void* raw = lb +
                        static_cast<size_t>(ll) * coloc_config.entry_size +
                        coloc_config.vector_offset;
                    T* dst = a_remote_vecs.data() + buf_off * cmp_dim;
                    std::memcpy(dst, raw, vec_bytes_per);
                    float dist = dist_fn->compare(node_vec, dst, cmp_dim);
                    candidates.push_back({N, dist, dst});
                    buf_off++;
                    remote_vec_fetches++;
                }
                continue;
            }
            auto ts_it = ts_map_phase3.find(owner);
            const auto* mr_ptr = node_store->remote_mr(owner);
            if (!use_multi_qp || ts_it == ts_map_phase3.end() || !mr_ptr) {
                // Fallback: skip — would require fetcher + lock; rare path.
                continue;
            }
            auto& state = ts_it->second.at(tid);
            const size_t buf_bytes = state.fetch_buffer.size() * sizeof(float);
            const size_t cap = (coloc_config.entry_size == 0) ? 1 :
                (buf_bytes / coloc_config.entry_size);
            const size_t total = owner_nbrs.size();
            for (size_t off = 0; off < total; off += cap) {
                size_t chunk = std::min(cap, total - off);
                char* fbuf = reinterpret_cast<char*>(state.fetch_buffer.data());
                std::vector<rdma::SGLEntry> sgl(chunk);
                for (size_t k = 0; k < chunk; k++) {
                    uint64_t N = owner_nbrs[off + k];
                    uint32_t ll = static_cast<uint32_t>(decode_local_loc(N));
                    sgl[k].local_addr = fbuf +
                        k * coloc_config.entry_size + coloc_config.vector_offset;
                    sgl[k].length = vec_bytes_per;
                    sgl[k].remote_offset =
                        static_cast<uint64_t>(ll) * coloc_config.entry_size +
                        coloc_config.vector_offset;
                }
                state.conn->post_read_batch(sgl, *mr_ptr, state.fetch_buffer_mr, 0);
                for (size_t k = 0; k < chunk; k++) {
                    uint64_t N = owner_nbrs[off + k];
                    const void* raw = fbuf + k * coloc_config.entry_size +
                                      coloc_config.vector_offset;
                    T* dst = a_remote_vecs.data() + buf_off * cmp_dim;
                    std::memcpy(dst, raw, vec_bytes_per);
                    float dist = dist_fn->compare(node_vec, dst, cmp_dim);
                    candidates.push_back({N, dist, dst});
                    buf_off++;
                    remote_vec_fetches++;
                }
            }
        }

        // ==================================================================
        // Part 2: Di-expansion candidates from pre-fetched per-Di CSR cache.
        //   - Iterates pre-resolved deleted_nbr_idx (no di_idx_map.find).
        //   - INVALID_VID covers both end-of-list AND deleted N (Phase 1c).
        //   - vec ptr bound directly into candidate (no vec_store map).
        // ==================================================================
        for (uint32_t i : w.deleted_nbr_idx) {
            uint32_t deg = di_degree[i];
            size_t base = static_cast<size_t>(i) * max_R;
            for (uint32_t j = 0; j < deg; j++) {
                uint64_t N = di_neighbor_vids[base + j];
                if (N == INVALID_VID) continue;  // end OR deleted (Phase 1c)
                if (N == my_vid) continue;
                if (!seen.insert(N).second) continue;
                // Vector source (zero-copy per-Di):
                //   -2 → cache hit: read the search-warmed arena directly.
                //   ≥0 → miss: index into compact vec_storage (RDMA-fetched).
                //   -1 → N is local: zero-copy from coloc_store.
                int32_t ridx = remote_idx[base + j];
                const T* N_vec = nullptr;
                if (ridx == REMOTE_IDX_CACHE_HIT) {
                    N_vec = static_cast<const T*>(
                        config.consolidation_cache->lookup_vector_ptr(N));
                    // null only if the arena entry vanished — no staged slot
                    // exists, so the candidate is skipped below (rare).
                } else if (ridx >= 0) {
                    N_vec = reinterpret_cast<const T*>(
                        vec_storage_ptr +
                        static_cast<size_t>(ridx) * vec_bytes_per);
                } else {
                    // Local N: zero-copy from local colocation store.
                    const uint32_t cand_ll =
                        static_cast<uint32_t>(decode_local_loc(N));
                    N_vec = static_cast<const T*>(
                        coloc_store->get_vector(cand_ll));
                    local_vec_fetches++;
                }
                if (!N_vec) continue;
                float dist = dist_fn->compare(node_vec, N_vec, cmp_dim);
                candidates.push_back({N, dist, N_vec});
            }
        }
        total_dist_comps += candidates.size();
        expanded_sum += candidates.size();

        if (candidates.empty()) {
            write_neighbors_internal(coloc_store, coloc_config, local_loc, nullptr, 0);
            modified++;
            diag_path_empty++;
            zero_degree++;
            #pragma omp critical(perdi_hist)
            { degree_histogram[0]++; }
            continue;
        }

        if (candidates.size() <= R) {
            std::vector<uint64_t> just_vids;
            just_vids.reserve(candidates.size());
            for (auto& c : candidates) just_vids.push_back(c.vid);
            const uint32_t new_deg = static_cast<uint32_t>(just_vids.size());
            write_neighbors_internal(coloc_store, coloc_config, local_loc,
                                     just_vids.data(), new_deg);
            modified++;
            diag_path_no_prune++;
            degree_after_sum += new_deg;
            if (new_deg < R / 2) low_degree++;
            #pragma omp critical(perdi_hist)
            { degree_histogram[(new_deg/8)*8]++; }
            continue;
        }

        if (enable_lits_filter) {
            std::unordered_set<uint64_t> orig_nbr_vids;
            const uint64_t* nbr_ptr = coloc_store->get_neighbors(local_loc);
            const uint32_t safe_degree = std::min(degree, coloc_config.slack_R);
            for (uint32_t j = 0; j < safe_degree; j++) {
                uint64_t v = nbr_ptr[j];
                if (v == INVALID_VID) break;
                if (!deleted_set.count(v)) orig_nbr_vids.insert(v);
            }
            auto stats = LITSFilter::filter_consolidation_candidates(
                candidates, orig_nbr_vids, my_partition);
            total_lits_checked += stats.checked;
            total_lits_filtered += stats.filtered;
        }

        std::sort(candidates.begin(), candidates.end(),
                  [](const auto& a, const auto& b) { return a.dist < b.dist; });
        std::vector<uint64_t> pruned = robust_prune_internal<T>(
            node_vec, candidates, R, alpha, maxc, dist_fn, cmp_dim);
        const uint32_t pruned_deg = static_cast<uint32_t>(pruned.size());
        write_neighbors_internal(coloc_store, coloc_config, local_loc,
                                 pruned.data(), pruned_deg);
        modified++;
        diag_path_prune++;
        degree_after_sum += pruned_deg;
        if (pruned_deg < R / 2) low_degree++;
        if (pruned_deg == 0) zero_degree++;
        #pragma omp critical(perdi_hist)
        { degree_histogram[(pruned_deg/8)*8]++; }
        } catch (const std::exception& e) {
            bool expected = false;
            if (err_flag.compare_exchange_strong(expected, true)) {
                std::lock_guard<std::mutex> lk(err_mtx);
                err_msg = e.what();
            }
        }
    }
    if (err_flag.load()) throw std::runtime_error(err_msg);
    auto t_pass2_loop_end = std::chrono::high_resolution_clock::now();

    // ------------------------------------------------------------------
    // Stats
    // ------------------------------------------------------------------
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_s = std::chrono::duration<double>(t_end - t_start).count();
    double scan_s = std::chrono::duration<double>(t_scan_end - t_scan_start).count();
    double di_merge_s = std::chrono::duration<double>(t_di_merge_end - t_scan_end).count();
    double di_nbr_s = std::chrono::duration<double>(t_di_nbr_end - t_di_merge_end).count();
    double pop_s = std::chrono::duration<double>(t_pop_end - t_pop_start).count();
    double pass2_loop_wall_s =
        std::chrono::duration<double>(t_pass2_loop_end - t_pass2_loop_start).count();

    rdma::Logger::info("[CONSOLIDATE-PERDI] timing: elapsed=%.3fs scan=%.3fs "
                       "di_merge=%.3fs di_nbr_fetch=%.3fs vec_fetch=%.3fs "
                       "pass2_loop=%.3fs N_di=%zu",
                       elapsed_s, scan_s, di_merge_s, di_nbr_s, pop_s,
                       pass2_loop_wall_s, N_di);

    ConsolidationStats stats;
    stats.entries_processed = n_entries;
    stats.entries_modified = modified;
    stats.entries_pruned = diag_path_prune;
    stats.total_expanded = static_cast<uint64_t>(expanded_sum);
    stats.local_vec_accesses = local_vec_fetches;
    stats.remote_vec_accesses = remote_vec_fetches;
    stats.local_nbr_accesses = local_nbr_fetches;
    stats.remote_nbr_accesses = remote_nbr_fetches;
    if (rdma_wait_fail.load())
        rdma::Logger::warn("consolidation: skipped %lu remote Di neighbor fetch(es) — RDMA send completion timeout/error",
                           static_cast<unsigned long>(rdma_wait_fail.load()));
    stats.cache_neighbor_hits = cache_neighbor_hits;
    stats.cache_neighbor_misses = cache_neighbor_misses;
    stats.cache_vector_hits = cache_vector_hits;
    stats.cache_vector_misses = cache_vector_misses;
    stats.cache_entries = cache_entries_seen;
    stats.cache_warm_expansion_events = cache_warm_expansion_events;
    stats.cache_warm_neighbor_records = cache_warm_neighbor_records;
    stats.cache_warm_vector_records = cache_warm_vector_records;
    stats.cache_warm_bytes = cache_warm_bytes;
    stats.cache_buffer_neighbor_bytes = cache_buffer_neighbor_bytes;
    stats.cache_buffer_vector_bytes = cache_buffer_vector_bytes;
    stats.cache_compacted_neighbor_bytes = cache_compacted_neighbor_bytes;
    stats.cache_compacted_vector_bytes = cache_compacted_vector_bytes;
    stats.cache_compacted_index_bytes = cache_compacted_index_bytes;
    stats.cache_dropped_neighbor_records = cache_dropped_neighbor_records;
    stats.cache_dropped_vector_records = cache_dropped_vector_records;
    stats.cache_hits = cache_neighbor_hits + cache_vector_hits;
    stats.cache_misses = cache_neighbor_misses + cache_vector_misses;
    // unique_vec_fetches semantic per-Di = N_di (number of distinct deleted
    // vertices the per-Di CSR caches).
    stats.unique_vec_fetches = N_di;
    stats.vec_bytes_fetched = remote_vec_fetches * vec_bytes_per;
    stats.vec_bytes_saved_by_cache = cache_vector_hits * vec_bytes_per;
    stats.vec_buffer_peak_bytes = vec_storage_bytes;
    stats.use_multi_qp = use_multi_qp;
    stats.elapsed_sec = elapsed_s;
    stats.loop_only_sec = pass2_loop_wall_s;
    // Phase offsets (relative to t_start). Map perdi phases onto the same
    // slots used by per-vid for the engine_report layout:
    //   scan      → Phase 1
    //   setup     → di_merge + di_nbr_fetch
    //   populate  → vec_fetch
    //   loop      → pass2 prune
    stats.scan_end_offset_sec = scan_s;
    stats.pop_start_offset_sec = scan_s + di_merge_s + di_nbr_s;
    stats.pop_end_offset_sec = stats.pop_start_offset_sec + pop_s;
    stats.pass2_loop_start_offset_sec =
        std::chrono::duration<double>(t_pass2_loop_start - t_start).count();
    stats.pass2_loop_end_offset_sec =
        std::chrono::duration<double>(t_pass2_loop_end - t_start).count();

    if (!degree_histogram.empty()) {
        uint32_t max_bucket = degree_histogram.rbegin()->first;
        stats.degree_histogram.resize(max_bucket / 8 + 1, 0);
        for (auto& [bucket, count] : degree_histogram) {
            uint32_t idx_h = bucket / 8;
            if (idx_h < stats.degree_histogram.size())
                stats.degree_histogram[idx_h] = static_cast<uint32_t>(count);
        }
    }

    double avg_before = degree_count > 0 ? degree_before_sum / degree_count : 0.0;
    double avg_after = degree_count > 0 ? degree_after_sum / degree_count : 0.0;
    double avg_expanded = degree_count > 0 ? expanded_sum / degree_count : 0.0;
    rdma::Logger::info(
        "[CONSOLIDATE-PERDI] done: modified=%lu path_empty=%lu path_no_prune=%lu "
        "path_prune=%lu avg_deg_before=%.1f avg_deg_after=%.1f avg_expanded=%.1f "
        "N_di=%zu remote_vec_fetches=%lu local_vec_fetches=%lu threads=%d",
        modified, diag_path_empty, diag_path_no_prune, diag_path_prune,
        avg_before, avg_after, avg_expanded,
        N_di, remote_vec_fetches, local_vec_fetches, num_threads);

    return stats;
}

// ========================================================================
// Explicit template instantiations
// ========================================================================
template ConsolidationStats Consolidation::run<float>(
    const ConsolidationConfig& config,
    NodeStore* node_store,
    PartitionMapper* mapper,
    diskann::Distance<float>* dist_fn,
    const std::unordered_set<uint64_t>& deleted_vids,
    uint32_t vector_dim,
    uint32_t aligned_dim);

template ConsolidationStats Consolidation::run<int8_t>(
    const ConsolidationConfig& config,
    NodeStore* node_store,
    PartitionMapper* mapper,
    diskann::Distance<int8_t>* dist_fn,
    const std::unordered_set<uint64_t>& deleted_vids,
    uint32_t vector_dim,
    uint32_t aligned_dim);

template ConsolidationStats Consolidation::run<uint8_t>(
    const ConsolidationConfig& config,
    NodeStore* node_store,
    PartitionMapper* mapper,
    diskann::Distance<uint8_t>* dist_fn,
    const std::unordered_set<uint64_t>& deleted_vids,
    uint32_t vector_dim,
    uint32_t aligned_dim);

} // namespace danns
