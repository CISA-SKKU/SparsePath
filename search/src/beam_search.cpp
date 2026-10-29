// search/src/beam_search.cpp — ColocationBeamSearch implementation
// Extracted from: distributed_bindings.cpp L3672-5808
//   - search_colocation_pure_inner_  -> ColocationBeamSearch::search()
//   - search_colocation_pure_typed   -> (inlined into search() w/ aligned query handling)
//   - search_colocation_pure_batch_typed -> ColocationBeamSearch::search_batch()
//
// All member variable accesses have been converted to explicit parameters:
//   coloc_store_          -> node_store->local_store()
//   coloc_fetchers_[id]   -> node_store->fetcher(id)
//   coloc_config_         -> node_store->config()
//   my_partition_id_      -> node_store->partition_id()
//   node_id_              -> node_store->node_id()
//   coloc_fetch_mode_     -> node_store->fetch_mode()
//   thread_states_        -> node_store->thread_states()
//   coloc_remote_mrs_     -> node_store->remote_mr(peer_id)
//   local_to_tag_         -> node_store->local_to_tag()
//   centroid_local_locs_  -> mapper->centroid_local_locs()
//   frozen_local_loc_     -> mapper->frozen_local_loc()
//   frozen_canonical_vid_ -> mapper->frozen_canonical_vid()
//   deleted_vids_         -> mapper->deleted_vids()
//   pq_filter_enabled_    -> pq_filter->enabled()
//   pq_M_, pq_centroids_  -> pq_filter members
//   resolve_vid()         -> standalone static method using mapper

#include "search/beam_search.h"
#include "search/epoch_hash.h"
#include "search/search_config.h"
#include "search/search_result.h"
#include "search/pq_filter.h"
#include "store/node_store.h"
#include "store/consolidation_cache.h"
#include "cluster/partition_mapper.h"
#include "core/vector_id.h"
#include "core/priority_queue.h"
#include "core/latency_stats.h"
#include "trace/search_trace.h"
#include "trace/trace_types.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <tsl/robin_map.h>
#include <tsl/robin_set.h>

#include <omp.h>

namespace danns {

// ============================================================================
// resolve_vid — Convert VID to (owner_node, local_loc, valid)
// ============================================================================
std::tuple<uint32_t, uint32_t, bool> ColocationBeamSearch::resolve_vid(
    uint64_t vid,
    NodeStore* node_store,
    PartitionMapper* mapper)
{
    // Canonical frozen VID -> redirect to local frozen (same data, avoids RDMA)
    if (vid == mapper->frozen_canonical_vid() &&
        mapper->frozen_local_loc() != UINT32_MAX) {
        return {node_store->node_id(), mapper->frozen_local_loc(), true};
    }
    uint32_t partition = danns::decode_partition(vid);
    uint64_t local_loc64 = danns::decode_local_loc(vid);
    // Bounds guard (concurrent consolidation): a torn/garbage VID whose
    // local_loc lands past the store would index past the local mmap, or —
    // worse — issue an RDMA read past the peer's MR and error the QP.
    // Both checks are predicted-not-taken; no-ops while bounds are unset (0).
    const uint32_t nparts = mapper->num_partitions();
    const uint64_t max_locs = mapper->max_local_locs();
    if ((nparts != 0 && partition >= nparts) ||
        (max_locs != 0 && local_loc64 >= max_locs)) {
        return {0, 0, false};
    }
    uint32_t local_loc = static_cast<uint32_t>(local_loc64);
    uint32_t owner_node = mapper->partition_id_to_node_id(partition);
    return {owner_node, local_loc, true};
}

// ============================================================================
// filter_lits_expanded_pool — LITS filter for candidate pool
// ============================================================================
template <typename T>
void ColocationBeamSearch::filter_lits_expanded_pool(
    std::vector<std::pair<uint64_t, float>>& pool,
    uint32_t my_partition)
{
    if (pool.empty()) return;
    double sum_dist = 0.0;
    for (auto& [vid, dist] : pool) sum_dist += dist;
    float avg_dist = static_cast<float>(sum_dist / pool.size());
    pool.erase(
        std::remove_if(pool.begin(), pool.end(),
            [avg_dist, my_partition](const std::pair<uint64_t, float>& p) {
                return (danns::decode_partition(p.first) != my_partition) &&
                       (p.second > avg_dist);
            }),
        pool.end()
    );
}

// ============================================================================
// warm_frozen_seed_cache — one-shot copy of the frozen point's REMOTE
// neighbor vectors into process-local DRAM.
//
// Called under std::call_once from the frozen seeding block, so exactly one
// thread runs this and every other thread blocks until it returns (a one-time
// cost on the first batch). After it returns the cache is immutable, and
// call_once supplies the happens-before edge that makes the lock-free reads in
// seed_vid() well defined.
//
// Failure policy: a neighbor whose fetch fails is left OUT of the map. The
// lookup then misses and that VID takes the unmodified RDMA path, so a partial
// (or empty) warm degrades to exactly the pre-patch behavior.
// ============================================================================
void ColocationBeamSearch::warm_frozen_seed_cache(
    const uint64_t* nbr_vids,
    uint32_t degree,
    NodeStore* node_store,
    PartitionMapper* mapper,
    bool use_multi_qp,
    int thread_id)
{
    FrozenSeedCache& fsc = frozen_seed_cache_;
    const auto& coloc_config = node_store->config();
    const auto& thread_states = node_store->thread_states();
    const auto& local_to_tag = node_store->local_to_tag();
    const uint32_t my_node_id = node_store->node_id();

    const size_t vec_bytes =
        static_cast<size_t>(coloc_config.dim) * coloc_config.element_size;
    if (vec_bytes == 0 || degree == 0 || nbr_vids == nullptr) {
        fsc.active = true;   // nothing to cache; every lookup will miss
        return;
    }

    // Pass 1: keep only the remote, resolvable neighbors. Local ones are
    // already zero-copy reads out of the local store — caching them would only
    // duplicate memory.
    struct RemoteSeed { uint64_t vid; uint32_t owner; uint32_t local_loc; };
    std::vector<RemoteSeed> remote;
    remote.reserve(degree);
    for (uint32_t i = 0; i < degree; i++) {
        const uint64_t vid = nbr_vids[i];
        if (vid == UINT64_MAX) continue;
        auto [owner, local_loc, valid] = resolve_vid(vid, node_store, mapper);
        if (!valid || owner == my_node_id) continue;
        remote.push_back(RemoteSeed{vid, owner, local_loc});
    }
    if (remote.empty()) {
        fsc.active = true;
        return;
    }

    constexpr size_t kAlign = 64;
    const size_t stride = (vec_bytes + kAlign - 1) & ~(kAlign - 1);
    fsc.storage.assign(stride * remote.size() + kAlign, 0);
    const uintptr_t raw = reinterpret_cast<uintptr_t>(fsc.storage.data());
    const uintptr_t aligned = (raw + (kAlign - 1)) & ~static_cast<uintptr_t>(kAlign - 1);
    fsc.base = reinterpret_cast<char*>(aligned);
    fsc.stride = stride;
    fsc.slot.reserve(remote.size() * 2);
    fsc.entry_vid.reserve(remote.size() * 2);

    // Pass 2: fetch each remote seed once, through the SAME path seed_vid()
    // would have used, and copy it out.
    uint32_t next_slot = 0;
    for (const auto& rs : remote) {
        const void* src = nullptr;
        if (use_multi_qp) {
            auto ts_it = thread_states.find(rs.owner);
            const auto* mr_ptr = node_store->remote_mr(rs.owner);
            if (ts_it == thread_states.end() || !mr_ptr ||
                thread_id >= static_cast<int>(ts_it->second.size())) continue;
            auto& state = const_cast<NodeStore::ThreadRDMAState&>(
                ts_it->second[thread_id]);

            rdma::RemoteMemoryRegion target = *mr_ptr;
            target.addr += static_cast<uint64_t>(rs.local_loc) * coloc_config.entry_size
                         + coloc_config.vector_offset;
            target.size = static_cast<uint32_t>(vec_bytes);

            char* buf = reinterpret_cast<char*>(state.fetch_buffer.data());
            state.conn->post_read(buf, static_cast<uint32_t>(vec_bytes),
                                  state.fetch_buffer_mr, target, 0);
            if (!state.conn->wait_send_completion(0, 5000)) continue;
            src = buf;
        } else {
            auto* fetcher = node_store->fetcher(rs.owner);
            if (!fetcher) continue;
            src = fetcher->fetch_vector_only(rs.local_loc);
        }
        if (!src) continue;

        std::memcpy(fsc.base + static_cast<size_t>(next_slot) * stride, src, vec_bytes);
        fsc.slot[rs.vid] = next_slot++;

        // Tag resolution for the need_result_tags path. local_to_tag is a
        // local flat array (no RDMA), and this reproduces exactly what
        // seed_vid()'s remote branch writes into edge_to_entry_vid.
        const uint32_t part = danns::decode_partition(rs.vid);
        if (part < local_to_tag.size() && rs.local_loc < local_to_tag[part].size()) {
            const uint32_t tag = local_to_tag[part][rs.local_loc];
            fsc.entry_vid[rs.vid] = danns::encode_vector_id(part, tag);
        }
    }

    fsc.active = true;   // last write: everything above is visible with it
}

// ============================================================================
// reset_frozen_seed_cache — drop the cache and re-arm the one-shot warm.
//
// Called at search-op boundaries (pipeline/ops/search/op.py) so a cache entry
// can never outlive an op and therefore never straddle a delete / insert /
// consolidate that reassigns a local_loc. See the header for the precondition:
// no search may be in flight.
// ============================================================================
size_t ColocationBeamSearch::reset_frozen_seed_cache()
{
    FrozenSeedCache& fsc = frozen_seed_cache_;
    const size_t dropped = fsc.slot.size();
    fsc.slot.clear();
    fsc.entry_vid.clear();
    std::vector<char>().swap(fsc.storage);
    fsc.base = nullptr;
    fsc.stride = 0;
    fsc.active = false;
    // std::once_flag is neither resettable nor movable -- install a fresh one.
    fsc.warm_flag = std::make_unique<std::once_flag>();
    return dropped;
}

// ============================================================================
// Per-thread search scratch (P1 hot-path: container reuse across queries)
//
// Every container that used to be constructed per query / per hop / per batch
// round now lives here and is reused via clear() (capacity preserved). This
// eliminates the per-hop heap-allocation churn and the per-query robin_map /
// robin_set rebuild+rehash cost identified in the 2026-07-26 CPU-sink
// profiling. Reuse only changes WHERE the memory lives, never the values
// written into it — every container is cleared (or fully overwritten) before
// use, so search results are byte-identical to the previous per-query
// allocation scheme.
// ============================================================================
namespace {

constexpr uint32_t MAX_OWNERS_SCRATCH = danns::kMaxOwnerNodes;  // == search()'s MAX_OWNERS (node ids are 1-based: 16 nodes -> 17)

// Unified pending-neighbor record (sequential path's PendingNbr and the batch
// path's BatchPendingNbr had identical layout — merged so the per-owner
// scratch buffers can be shared between the two mutually-exclusive paths).
struct PendingNbr {
    uint64_t vid;
    uint32_t local_loc;
    float pq_dist = 0.0f;
    uint64_t source_deleted_vid = danns::INVALID_VID;
};
struct OrderedNbr {
    uint64_t vid;
    uint32_t owner;
    uint32_t local_loc;
};
struct ComputedResult {
    float distance;
    float pq_distance = 0.0f;
    bool is_remote = false;
};
struct CandNeighbors {
    uint32_t degree = 0;
    std::vector<uint64_t> nbr_vids;
};
// Async fetch descriptors: the payload vectors now live in SearchScratch
// (keyed by owner), so these only carry bookkeeping.
struct AsyncRemoteFetch {
    uint32_t owner;
    uint32_t pending_count;
    danns::NodeStore::ThreadRDMAState* state;  // multi-QP path
    rdma::ColocationFetcher* fetcher;          // single-thread path
    bool full_entry;
};
struct AsyncNbrFetch {
    uint32_t owner;
    uint32_t pending_count;
    danns::NodeStore::ThreadRDMAState* state;
    bool nbr_full_entry;
};
struct AsyncVecFetch {
    uint32_t owner;
    uint32_t pending_count;
    danns::NodeStore::ThreadRDMAState* state;
    bool batch_full_entry;
    uint32_t read_sz;
};

struct SearchScratch {
    // ---- per-query hash containers (cleared at query start; buckets kept) ----
    // Audit item 3 (2026-08-05): epoch-stamped open addressing, same set
    // semantics as the tsl::robin_set it replaces (see epoch_hash.h).
    search_detail::EpochKeySet visited;
    // Audit item 2 (2026-08-05): epoch-stamped open addressing. Same
    // key->value relation as the tsl::robin_map it replaces; the per-query
    // reset becomes an epoch bump instead of an O(bucket) clear + a full
    // tsl rehash_impl() reallocation. See search/include/search/epoch_hash.h.
    search_detail::EpochKeyMap edge_to_entry_vid;
    tsl::robin_map<uint64_t, ComputedResult> computed;

    // ---- sequential-path per-hop containers ----
    std::vector<PendingNbr> pending_per_owner[MAX_OWNERS_SCRATCH];
    std::vector<OrderedNbr> ordered_nbrs;
    std::vector<uint64_t> nbr_local_copy;
    std::vector<rdma::SGLEntry> sgl_entries;
    // Pass-1 (seq) / vector-fetch (batch) uncached lists, keyed by owner.
    // The two paths are mutually exclusive within a hop, so they share slots.
    std::vector<PendingNbr> uncached_per_owner[MAX_OWNERS_SCRATCH];
    std::vector<PendingNbr> pq_passed;  // PQ-filter double buffer (swap)
    std::vector<AsyncRemoteFetch> async_remote_fetches;

    // ---- batch-path per-round containers ----
    std::vector<danns::VIDNeighbor> batch_cands;
    std::vector<CandNeighbors> all_cand_neighbors;
    std::vector<std::tuple<uint32_t, uint32_t, bool>> cand_resolved;
    std::array<std::vector<size_t>, MAX_OWNERS_SCRATCH> cand_by_owner;
    std::vector<size_t> nbr_uncached_ci[MAX_OWNERS_SCRATCH];
    std::vector<uint32_t> nbr_uncached_locs[MAX_OWNERS_SCRATCH];
    std::array<std::vector<PendingNbr>, MAX_OWNERS_SCRATCH> all_nbrs_by_owner;
    std::vector<uint32_t> batch_locs;
    std::vector<rdma::SGLEntry> batch_sgl_entries;
    std::vector<AsyncNbrFetch> async_nbr_fetches;
    std::vector<AsyncVecFetch> async_vec_fetches;

    uint32_t reserved_R = 0;

    // Capacity blow-up guard: if any reusable buffer ballooned past
    // ~4*L*slack_R elements (a pathological query / L transition), release it.
    // Heuristic only — affects capacity, never contents.
    template <typename V>
    static void shrink_vec(V& v, size_t cap) {
        if (v.capacity() > cap) { V().swap(v); }
    }
    template <typename H>
    static void shrink_hash(H& h, size_t cap) {
        if (h.bucket_count() > 2 * cap) { h.clear(); h.rehash(0); }
    }
    // Same heuristic for the epoch-stamped containers (capacity only).
    template <typename E>
    static void shrink_epoch(E& e, size_t cap) {
        if (e.slot_count() > 2 * cap) e.reset_to_minimum();
    }

    void prepare(uint32_t L, uint32_t R) {
        const uint32_t r = R ? R : 64u;
        const size_t cap = static_cast<size_t>(4) * L * r;

        shrink_epoch(visited, cap);
        shrink_epoch(edge_to_entry_vid, cap);
        shrink_hash(computed, cap);
        for (auto& v : pending_per_owner) shrink_vec(v, cap);
        for (auto& v : uncached_per_owner) shrink_vec(v, cap);
        for (auto& v : all_nbrs_by_owner) shrink_vec(v, cap);
        shrink_vec(ordered_nbrs, cap);
        shrink_vec(pq_passed, cap);
        shrink_vec(batch_locs, cap);
        shrink_vec(batch_sgl_entries, cap);

        if (reserved_R < r) {
            for (auto& v : pending_per_owner) v.reserve(r);
            for (auto& v : uncached_per_owner) v.reserve(r);
            ordered_nbrs.reserve(r);
            nbr_local_copy.reserve(r);
            sgl_entries.reserve(r);
            reserved_R = r;
        }

        // Per-query reset. clear() keeps buckets/capacity -> no rehash or
        // allocation on the queries after the first (steady state).
        // O(1) epoch bump; reserve() is a no-op once the table is large
        // enough (the old guard truncated max_load_factor() to 0 and so
        // reallocated the bucket array on every query -- see the item 2 commit).
        visited.clear();
        visited.reserve(L * 20u);
        // O(1) epoch bump. The old guard was dead: max_load_factor() is a
        // float (0.5f) and static_cast<size_t> truncated it to 0, so the
        // product was always 0 < L*32 and tsl's reserve() -> rehash_impl()
        // reallocated the whole bucket array on *every* query.
        edge_to_entry_vid.clear();
        edge_to_entry_vid.reserve(L * 32u);  // no-op once large enough
        computed.clear();
        if (computed.bucket_count() < r) computed.reserve(r);

        // Defensive: per-hop containers are cleared at hop end by contract,
        // but reset here too so a query never observes a previous query's
        // leftovers under any early-exit path.
        for (auto& v : pending_per_owner) v.clear();
        for (auto& v : uncached_per_owner) v.clear();
        for (auto& v : all_nbrs_by_owner) v.clear();
        for (auto& v : cand_by_owner) v.clear();
        for (auto& v : nbr_uncached_ci) v.clear();
        for (auto& v : nbr_uncached_locs) v.clear();
        ordered_nbrs.clear();
        async_remote_fetches.clear();
        async_nbr_fetches.clear();
        async_vec_fetches.clear();
    }
};

SearchScratch& thread_scratch() {
    static thread_local SearchScratch s;
    return s;
}

} // anonymous namespace

// ============================================================================
// search() — Core single-query beam search
// Extracted from: search_colocation_pure_inner_ (L3672-5221)
// ============================================================================
template <typename T>
SearchResult<T> ColocationBeamSearch::search(
    const T* aligned_query,
    size_t aligned_dim,
    size_t cmp_dim,
    diskann::Distance<T>* dist_fn,
    const SearchConfig& config,
    NodeStore* node_store,
    PartitionMapper* mapper,
    PQFilter* pq_filter,
    std::map<uint32_t, std::mutex>* fetcher_mutexes,
    int num_threads,
    int thread_id)
{
    // Unpack config
    const uint32_t K = config.K;
    const uint32_t L = config.L;
    const uint32_t entry_mode = config.entry_mode;
    const bool use_query_entry_partition = config.use_query_entry_partition;
    const uint32_t batch_size = config.batch_size;
    const uint32_t conv_window = config.conv_window;
    const float conv_threshold_pct = config.conv_threshold_pct;
    const bool return_pool = config.return_pool;
    const bool need_result_tags = config.need_result_tags;
    const bool lits_filter = config.lits_filter;
    const uint32_t lits_my_partition = config.lits_my_partition;
    const bool skip_timing = config.skip_timing;
    // P1 item 6: owner-interleaved completion processing in the batch (CAP)
    // path. Default off — bulk-sync barrier semantics preserved.
    const bool batch_interleave = config.batch_interleave_completion;
    const bool batch_remote_first = config.batch_remote_first;
    const uint32_t trace_query_id = config.trace_query_id;
    const uint16_t trace_level = config.trace_level;
    // Probe mode: bypass dist_fn->compare and substitute a vid-derived fake
    // value. Keeps OMP scheduling, RDMA fetch path, queue mechanics intact —
    // isolates the cost of compute on NIC pipeline utilization. Recall is
    // garbage in this mode; only use for BW measurement runs.
    const bool compute_skip = config.compute_skip;
    auto fake_dist = [](uint64_t vid) -> float {
        return static_cast<float>((vid * 2654435761ull) & 0xFFFFFF);
    };

    // Access node_store members
    auto* coloc_store = node_store->local_store();
    const auto& coloc_config = node_store->config();
    const uint32_t my_partition_id = node_store->partition_id();
    const uint32_t my_node_id = node_store->node_id();
    const auto& local_to_tag = node_store->local_to_tag();
    const auto& thread_states = node_store->thread_states();
    const uint32_t num_thread_qps = node_store->num_thread_qps();
    const rdma::FetchMode fetch_mode = node_store->fetch_mode();

    // Access mapper members
    const auto& centroid_local_locs = mapper->centroid_local_locs();
    const uint32_t frozen_local_loc = mapper->frozen_local_loc();
    const uint64_t frozen_canonical_vid = mapper->frozen_canonical_vid();
    const auto& deleted_vids = mapper->deleted_vids();
    const auto& deleted_tags = mapper->deleted_tags();
    auto* consolidation_cache_warmer = config.consolidation_cache_warmer;

    const size_t cache_vector_bytes = cmp_dim * sizeof(T);
    auto record_deleted_expansion = [&](uint64_t vid, uint32_t owner_node,
                                        uint32_t local_loc, bool batch_mode) {
        if (!consolidation_cache_warmer) return;
        if (!deleted_vids.count(vid)) return;
        DeletedNodeExpansion event{};
        event.vid = vid;
        event.owner_node = owner_node;
        event.local_loc = local_loc;
        event.query_id = trace_query_id;
        event.search_node_id = my_node_id;
        event.thread_id = static_cast<uint32_t>(thread_id);
        event.batch_mode = batch_mode;
        consolidation_cache_warmer->record_deleted_expansion(event);
    };
    auto record_deleted_neighbors = [&](uint64_t deleted_vid, const uint64_t* nbrs,
                                        uint32_t degree) {
        if (!consolidation_cache_warmer) return;
        if (!deleted_vids.count(deleted_vid)) return;
        consolidation_cache_warmer->record_deleted_neighbors(deleted_vid, nbrs, degree);
    };
    auto record_deleted_vector = [&](uint64_t deleted_vid, uint64_t nbr_vid,
                                     const void* vec) {
        if (!consolidation_cache_warmer) return;
        if (deleted_vid == INVALID_VID) return;
        consolidation_cache_warmer->record_neighbor_vector(
            deleted_vid, nbr_vid, vec, cache_vector_bytes);
    };

    // PQ filter state
    const bool pq_filter_enabled = pq_filter && pq_filter->enabled();

    // Use per-thread QP when available (avoids shared fetcher + mutex serialization)
    const bool use_multi_qp_search = (num_threads > 1 && !thread_states.empty() &&
                                       num_thread_qps > 0);

    // FrozenSeedCache gate. The stride guard is defensive: the cache stores
    // dim*element_size bytes per seed, which must cover the cmp_dim elements of
    // T the distance kernel reads. They are equal for every dataset we build
    // (element_size == sizeof(T), cmp_dim == dim), so this never trips — but a
    // mismatch would be an out-of-bounds read, not a wrong answer.
    const size_t fsc_vec_bytes =
        static_cast<size_t>(coloc_config.dim) * coloc_config.element_size;
    const bool fsc_enabled = config.frozen_seed_cache &&
                             fsc_vec_bytes >= cmp_dim * sizeof(T);
    // Set only AFTER this query's call_once returns, so the lock-free reads in
    // seed_vid() are ordered behind the warm. Guards the centroid seeding block
    // too, which runs BEFORE the frozen block and would otherwise race the
    // warming thread.
    bool fsc_ready = false;
    uint64_t frozen_seed_cache_hits = 0;
    using hrc = std::chrono::high_resolution_clock;
    const bool do_timing = !skip_timing;
    // Conditional timing helpers -- when skip_timing=true, hrc::now() calls are elided
    auto timer_start = [do_timing]() -> hrc::time_point {
        return do_timing ? hrc::now() : hrc::time_point{};
    };
    auto timer_ns = [do_timing](hrc::time_point t0) -> long long {
        return do_timing ? std::chrono::duration_cast<std::chrono::nanoseconds>(
            hrc::now() - t0).count() : 0;
    };
    SearchResult<T> result;

    // Stats accumulators
    uint64_t n_hops = 0, n_cmps = 0;
    uint64_t local_vector_accesses = 0, remote_vector_accesses = 0;
    uint64_t local_neighbor_accesses = 0, remote_neighbor_accesses = 0;
    uint64_t resolve_fail = 0;
    long long rdma_vector_ns = 0, rdma_neighbor_ns = 0;
    // CPU breakdown accumulators
    long long cpu_visited_ns = 0, cpu_resolve_ns = 0;
    long long cpu_distance_ns = 0, cpu_queue_ns = 0;
    long long cpu_nbr_loop_ns = 0, cpu_map_ns = 0, cpu_pq_ns = 0;
    long long lut_build_ns = 0;  // PQ LUT construction (once per query, on convergence)
    long long candidate_fetch_phase_ns = 0;  // wall-clock for Pass 1-3 candidate vector fetch/compute
    // Per-pass wall-clock timing for search diagnostics
    long long pass1_ns = 0;  // remote owner async post_read_batch_async
    long long pass2_ns = 0;  // local owner distance compute
    long long pass3_ns = 0;  // poll completions + remote distance compute (wall clock)
    // Pass 3 internal breakdown: poll = CQ drain check, compute = remote distance work
    long long pass3_poll_ns = 0;
    long long pass3_compute_ns = 0;
    // Converge batch async-poll wait time (multi-QP path)
    long long batch_nbr_poll_ns = 0;
    long long batch_vec_poll_ns = 0;
    uint64_t rdma_entry_calls = 0, rdma_entry_items = 0;
    uint64_t rdma_vec_fetch_calls = 0, rdma_nbr_expand_calls = 0;
    uint64_t rdma_full_entry_fetches = 0, rdma_vec_only_fetches = 0;
    uint64_t pq_filtered = 0;

    // Trace buffers (only allocated if tracing active)
    const uint16_t tl = trace_level;
    std::vector<HopRecord> trace_hops;
    std::vector<std::vector<FetchedEntry>> trace_fetched;

    // ---- Convergence detector for batch prefetch ----
    struct ConvergenceDetector {
        float prev_best = std::numeric_limits<float>::infinity();
        uint32_t stable_count = 0;
        uint32_t window;
        float threshold_pct;
        ConvergenceDetector(uint32_t w, float t) : window(w), threshold_pct(t) {}
        bool update(float best_distance) {
            if (prev_best > 0 && prev_best < std::numeric_limits<float>::infinity()) {
                float delta_pct = std::abs(best_distance - prev_best) / prev_best * 100.0f;
                if (delta_pct < threshold_pct) {
                    stable_count++;
                } else {
                    stable_count = 0;
                }
            }
            prev_best = best_distance;
            return stable_count >= window;
        }
        void reset() { stable_count = 0; }
    };
    ConvergenceDetector conv_detector(conv_window, conv_threshold_pct);
    // `converged` is the CONTROL flag: it gates the batch expansion path and
    // the PQ filter, and the batch path clears it again on a >5% improvement.
    // `converged_observed` is the STAT: sticky, set the first time the detector
    // fires, never cleared. They are separate because the control flag is only
    // ever raised when batch/PQ is enabled, which would make the convergence
    // stats identically zero in every plain (non-CAP, non-PQ) run.
    bool converged = false;
    bool converged_observed = false;
    uint64_t convergence_hop = 0;
    uint64_t n_batch_rounds = 0;
    uint64_t n_batch_candidates_total = 0;
    uint64_t fallback_count = 0;

    // ---- Initialize beam search ----
    VIDNeighborPriorityQueue queue(L);
    if (return_pool) queue.enable_pool_recording();
    // Per-thread scratch: visited / edge map / per-hop containers are reused
    // across queries (clear-and-reuse). prepare() clears all per-query state.
    SearchScratch& scratch = thread_scratch();
    scratch.prepare(L, coloc_config.slack_R);
    auto& visited = scratch.visited;

    // Cache: edge VID (partition|local_loc) -> entry VID (partition|tag)
    auto& edge_to_entry_vid = scratch.edge_to_entry_vid;

    // Helper lambda: seed a VID into the beam search queue.
    // Handles both local (zero-copy) and remote (RDMA fetch) vectors
    // to match DiskANN's behavior of using ALL neighbors as seeds.
    auto seed_vid = [&](uint64_t nbr_vid) {
        if (nbr_vid == UINT64_MAX) return;
        if (visited.count(nbr_vid)) return;

        auto [nbr_owner, nbr_local_loc, nbr_valid] = resolve_vid(nbr_vid, node_store, mapper);
        if (!nbr_valid) return;
        visited.insert(nbr_vid);

        if (nbr_owner == my_node_id) {
            // Local: zero-copy access
            const T* vec = static_cast<const T*>(coloc_store->get_vector(nbr_local_loc));
            float dist = compute_skip ? fake_dist(nbr_vid)
                                      : dist_fn->compare(aligned_query, vec, cmp_dim);
            queue.insert(VIDNeighbor(nbr_vid, dist));
            n_cmps++;
            local_vector_accesses++;
            if (need_result_tags)
                edge_to_entry_vid[nbr_vid] = coloc_store->get_vector_id(nbr_local_loc);
        } else {
            // FrozenSeedCache: the frozen point is immutable within an op, so
            // its remote neighbors' vectors are byte-identical for every query
            // in that op. Serve them from the local copy warmed by the first
            // query instead of a blocking post_read + wait each time.
            //
            // fsc_ready is false until this query's call_once has returned, so
            // the centroid block (which runs first) never reads a map another
            // thread is still building.
            if (fsc_ready) {
                auto slot_it = frozen_seed_cache_.slot.find(nbr_vid);
                if (slot_it != frozen_seed_cache_.slot.end()) {
                    const T* cvec = reinterpret_cast<const T*>(
                        frozen_seed_cache_.base +
                        static_cast<size_t>(slot_it->second) * frozen_seed_cache_.stride);
                    float dist = compute_skip ? fake_dist(nbr_vid)
                                              : dist_fn->compare(aligned_query, cvec, cmp_dim);
                    queue.insert(VIDNeighbor(nbr_vid, dist));
                    n_cmps++;
                    // Deliberately NOT remote_vector_accesses / rdma_entry_calls:
                    // those two must drop by exactly this count vs. baseline.
                    frozen_seed_cache_hits++;
                    if (need_result_tags) {
                        auto ev_it = frozen_seed_cache_.entry_vid.find(nbr_vid);
                        if (ev_it != frozen_seed_cache_.entry_vid.end())
                            edge_to_entry_vid[nbr_vid] = ev_it->second;
                    }
                    return;
                }
            }

            // Remote: RDMA fetch vector for seed
            const T* vec = nullptr;

            if (use_multi_qp_search) {
                // Per-thread QP -- thread-safe, no lock needed
                auto ts_it = thread_states.find(nbr_owner);
                const auto* mr_ptr = node_store->remote_mr(nbr_owner);
                if (ts_it == thread_states.end() || !mr_ptr ||
                    thread_id >= static_cast<int>(ts_it->second.size())) return;
                auto& state = const_cast<NodeStore::ThreadRDMAState&>(ts_it->second[thread_id]);
                auto remote_mr = *mr_ptr;

                uint64_t remote_offset = static_cast<uint64_t>(nbr_local_loc) * coloc_config.entry_size
                                        + coloc_config.vector_offset;
                uint32_t vec_bytes = coloc_config.dim * coloc_config.element_size;

                rdma::RemoteMemoryRegion target = remote_mr;
                target.addr += remote_offset;
                target.size = vec_bytes;

                char* buf = reinterpret_cast<char*>(state.fetch_buffer.data());
                state.conn->post_read(buf, vec_bytes, state.fetch_buffer_mr, target, 0);
                if (!state.conn->wait_send_completion(0, 5000)) { resolve_fail++; return; }
                vec = reinterpret_cast<const T*>(buf);
            } else {
                // Shared fetcher (single-threaded or mutex-protected)
                auto* fetcher = node_store->fetcher(nbr_owner);
                if (!fetcher) return;
                vec = static_cast<const T*>(fetcher->fetch_vector_only(nbr_local_loc));
            }

            if (!vec) return;
            float dist = compute_skip ? fake_dist(nbr_vid)
                                      : dist_fn->compare(aligned_query, vec, cmp_dim);
            queue.insert(VIDNeighbor(nbr_vid, dist));
            n_cmps++;
            remote_vector_accesses++;
            rdma_entry_calls++;
            rdma_entry_items++;
            // Resolve entry_vid from local_to_tag (flat array); search only
            if (need_result_tags) {
                uint32_t nbr_part = danns::decode_partition(nbr_vid);
                if (nbr_part < local_to_tag.size() && nbr_local_loc < local_to_tag[nbr_part].size()) {
                    uint32_t tag = local_to_tag[nbr_part][nbr_local_loc];
                    edge_to_entry_vid[nbr_vid] = danns::encode_vector_id(nbr_part, tag);
                }
            }
        }
    };

    // Seed from a partition centroid's NEIGHBORS (entry_mode 0 only).
    //
    // Default: the local partition's centroid (historical behavior). With
    // use_query_entry_partition, the *query's* nearest centroid instead --
    // possibly owned by a peer, in which case its neighbor list is pulled with
    // one RDMA read. centroid_local_locs already holds an entry for every
    // partition (node_worker.py fills it with `for p in range(num_partitions)`),
    // and the centroid is a materialized vertex, so it is addressable as
    // encode_vector_id(part, centroid_loc). seed_vid() already resolves remote
    // neighbor VIDs, so only the centroid's own neighbor list needs fetching.
    if (entry_mode == 0) {
        const uint32_t seed_part = use_query_entry_partition
            ? config.query_partition : my_partition_id;
        auto cl_it = centroid_local_locs.find(seed_part);
        if (cl_it != centroid_local_locs.end()) {
            uint32_t centroid_loc = cl_it->second;
            uint32_t owner = mapper->partition_id_to_node_id(seed_part);

            if (owner == my_node_id) {
                uint32_t centroid_deg = coloc_store->get_degree(centroid_loc);
                const uint64_t* centroid_nbrs = coloc_store->get_neighbors(centroid_loc);
                for (uint32_t i = 0; i < centroid_deg; i++) {
                    seed_vid(centroid_nbrs[i]);
                }
            } else if (use_multi_qp_search && owner < MAX_OWNERS_SCRATCH) {
                // Remote centroid: one full-entry read, then seed its neighbors.
                // Mirrors the expansion path's single-entry fetch (:665-700).
                auto ts_it = thread_states.find(owner);
                const auto* mr_ptr = node_store->remote_mr(owner);
                if (ts_it != thread_states.end() && mr_ptr &&
                    thread_id < static_cast<int>(ts_it->second.size())) {
                    auto& state = const_cast<NodeStore::ThreadRDMAState&>(
                        ts_it->second.at(thread_id));
                    char* buf = reinterpret_cast<char*>(state.fetch_buffer.data());

                    const bool full_entry = (fetch_mode != rdma::FetchMode::SEPARATED);
                    uint32_t read_sz = full_entry
                        ? coloc_config.entry_size
                        : (sizeof(uint32_t) + coloc_config.slack_R * sizeof(uint64_t));
                    uint32_t read_off = full_entry ? 0 : coloc_config.degree_offset;

                    rdma::RemoteMemoryRegion target = *mr_ptr;
                    target.addr += static_cast<uint64_t>(centroid_loc)
                                 * coloc_config.entry_size + read_off;
                    target.size = read_sz;

                    auto t0 = timer_start();
                    state.conn->post_read(buf, read_sz, state.fetch_buffer_mr, target, 0);
                    if (state.conn->wait_send_completion(0, 5000)) {
                        rdma_entry_calls++; rdma_entry_items++;
                        rdma_nbr_expand_calls++;
                        remote_neighbor_accesses++;
                        // Degree/neighbors sit at degree_offset within the entry;
                        // the SEPARATED read already starts there.
                        const char* nbr_base = full_entry
                            ? (buf + coloc_config.degree_offset) : buf;
                        uint32_t centroid_deg = 0;
                        std::memcpy(&centroid_deg, nbr_base, sizeof(uint32_t));
                        if (centroid_deg > coloc_config.slack_R)
                            centroid_deg = coloc_config.slack_R;
                        const uint64_t* centroid_nbrs =
                            reinterpret_cast<const uint64_t*>(nbr_base + sizeof(uint32_t));
                        // Copy out: seed_vid() reuses the same fetch buffer for
                        // its own remote vector reads and would clobber this.
                        auto& cnbrs = scratch.nbr_local_copy;
                        cnbrs.assign(centroid_nbrs, centroid_nbrs + centroid_deg);
                        rdma_neighbor_ns += timer_ns(t0);
                        for (uint32_t i = 0; i < centroid_deg; i++) seed_vid(cnbrs[i]);
                    } else {
                        resolve_fail++;
                    }
                }
            }
        }
    }

    // Seed from frozen point itself + its NEIGHBORS (always, all entry modes)
    if (frozen_local_loc != UINT32_MAX) {
        // Use canonical VID (partition 0) so visited set deduplicates
        // frozen across partitions.  Physical read is still local.
        uint64_t frozen_vid = (frozen_canonical_vid != UINT64_MAX)
            ? frozen_canonical_vid
            : danns::encode_vector_id(my_partition_id, frozen_local_loc);
        if (!visited.count(frozen_vid)) {
            visited.insert(frozen_vid);
            const T* frozen_vec = static_cast<const T*>(coloc_store->get_vector(frozen_local_loc));
            float frozen_dist = compute_skip ? fake_dist(frozen_vid)
                                             : dist_fn->compare(aligned_query, frozen_vec, cmp_dim);
            queue.insert(VIDNeighbor(frozen_vid, frozen_dist));
            n_cmps++;
            local_vector_accesses++;
            if (need_result_tags)
                edge_to_entry_vid[frozen_vid] = coloc_store->get_vector_id(frozen_local_loc);
        }

        uint32_t frozen_deg = coloc_store->get_degree(frozen_local_loc);
        const uint64_t* frozen_nbrs = coloc_store->get_neighbors(frozen_local_loc);

        // FrozenSeedCache warm. Exactly one thread per op runs this; the rest
        // block here for the duration of ~R serialized reads (a one-time cost
        // paid by the first batch of the op, then amortized over every query).
        // The op-boundary reset in reset_frozen_seed_cache() is what keeps the
        // cached bytes from ever spanning a graph mutation.
        if (fsc_enabled) {
            std::call_once(*frozen_seed_cache_.warm_flag, [&]() {
                warm_frozen_seed_cache(frozen_nbrs, frozen_deg, node_store,
                                       mapper, use_multi_qp_search, thread_id);
            });
            // Ordered behind call_once -> the map is fully published here.
            fsc_ready = frozen_seed_cache_.active;
        }

        for (uint32_t i = 0; i < frozen_deg; i++) {
            seed_vid(frozen_nbrs[i]);
        }
    }

    // ---- PQ distance table (built eagerly at query entry) ----
    // LUT = M * 256 entries of L[m][k] = ||q_m - c_{m,k}||^2.
    // Built once per query here so it is ready by the time convergence is detected
    // and PQ filter activates. Overlaps with seed/approach-phase RDMA naturally
    // because we compute it on CPU before entering the main hop loop that issues RDMA.
    //
    // Cost: M * 256 * chunk_size FP ops (~3-25us depending on M, d).
    // TODO(future): move to a dedicated LUT thread that starts even earlier
    //               (before seed processing) and publishes pq_dt atomically.
    std::vector<float> pq_dt;
    if (pq_filter_enabled) {
        auto t_lut = timer_start();
        uint32_t ksub = 256;
        uint32_t pq_M_lut = pq_filter->M();
        const auto& pq_chunk_offsets = pq_filter->chunk_offsets();
        const auto& pq_centroid_offsets = pq_filter->centroid_offsets();
        const auto& pq_centroids = pq_filter->centroids();
        pq_dt.resize(pq_M_lut * ksub);
        for (uint32_t m = 0; m < pq_M_lut; m++) {
            uint32_t chunk_start = pq_chunk_offsets[m];
            uint32_t chunk_size = pq_chunk_offsets[m + 1] - chunk_start;
            const float* cent_m = pq_centroids.data() + pq_centroid_offsets[m];
            for (uint32_t k = 0; k < ksub; k++) {
                float d = 0.0f;
                for (uint32_t di = 0; di < chunk_size; di++) {
                    float diff = static_cast<float>(aligned_query[chunk_start + di])
                                 - cent_m[k * chunk_size + di];
                    d += diff * diff;
                }
                pq_dt[m * ksub + k] = d;
            }
        }
        lut_build_ns += timer_ns(t_lut);
    }

    // PQ parameters (captured for use in inner lambdas)
    const uint32_t pq_M = pq_filter_enabled ? pq_filter->M() : 0;
    const float pq_alpha = pq_filter_enabled ? pq_filter->alpha() : 0.0f;

    // ---- Pre-allocated per-hop containers (scratch: reused across hops AND
    //      queries; struct definitions hoisted to file scope for SearchScratch) ----
    static constexpr uint32_t MAX_OWNERS = MAX_OWNERS_SCRATCH;
    auto& pending_per_owner = scratch.pending_per_owner;
    uint32_t active_owners[MAX_OWNERS];
    uint32_t n_active_owners = 0;

    auto& ordered_nbrs = scratch.ordered_nbrs;
    auto& computed = scratch.computed;
    auto& nbr_local_copy = scratch.nbr_local_copy;
    auto& sgl_entries = scratch.sgl_entries;

    // ---- Main beam search loop ----
    while (queue.has_unexpanded_node()) {

      if (!converged || batch_size == 0) {
        // ---- Sequential hop ----
        VIDNeighbor cur = queue.closest_unexpanded();
        n_hops++;

        uint16_t hop_local_vec = 0, hop_remote_vec = 0;
        uint16_t hop_inserted = 0, hop_visited_skip = 0;
        long long hop_rdma_ns = 0;
        std::vector<FetchedEntry> hop_fetched;

        const bool use_full_entry = (fetch_mode == rdma::FetchMode::COLOCATION);

        auto [owner_node, local_loc, valid] = resolve_vid(cur.vid, node_store, mapper);
        if (!valid) {
            resolve_fail++;
            continue;
        }
        record_deleted_expansion(cur.vid, owner_node, local_loc, false);

        uint32_t degree = 0;
        const uint64_t* nbr_vids = nullptr;
        nbr_local_copy.clear();

        if (owner_node == my_node_id) {
            degree = coloc_store->get_degree(local_loc);
            nbr_vids = coloc_store->get_neighbors(local_loc);
            local_neighbor_accesses++;
        } else {
            {
                auto* fetcher = node_store->fetcher(owner_node);
                if (!fetcher) {
                    resolve_fail++;
                    continue;
                }

                auto t0 = timer_start();
                if (use_multi_qp_search) {
                    // Per-thread QP -- no lock needed
                    auto ts_it = thread_states.find(owner_node);
                    const auto* mr_ptr = node_store->remote_mr(owner_node);
                    if (ts_it == thread_states.end() || !mr_ptr) {
                        resolve_fail++;
                        continue;
                    }
                    auto& state = const_cast<NodeStore::ThreadRDMAState&>(ts_it->second.at(thread_id));
                    auto remote_mr = *mr_ptr;
                    char* buf = reinterpret_cast<char*>(state.fetch_buffer.data());

                    if (fetch_mode != rdma::FetchMode::SEPARATED) {
                    // --- Full entry (colocation) ---
                    uint64_t remote_offset = static_cast<uint64_t>(local_loc) * coloc_config.entry_size;
                    uint32_t read_bytes = coloc_config.entry_size;

                    rdma::RemoteMemoryRegion target = remote_mr;
                    target.addr += remote_offset;
                    target.size = read_bytes;

                    state.conn->post_read(buf, read_bytes, state.fetch_buffer_mr, target, 0);
                    if (!state.conn->wait_send_completion(0, 5000)) { resolve_fail++; continue; }
                    rdma_entry_calls++; rdma_entry_items++;
                    rdma_nbr_expand_calls++;

                    const char* entry = buf;
                    std::memcpy(&degree, entry + coloc_config.degree_offset, sizeof(uint32_t));
                    if (degree > coloc_config.slack_R) degree = coloc_config.slack_R;
                    const uint64_t* raw_nbrs = reinterpret_cast<const uint64_t*>(
                        entry + coloc_config.degree_offset + sizeof(uint32_t));
                    nbr_local_copy.assign(raw_nbrs, raw_nbrs + degree);
                    nbr_vids = nbr_local_copy.data();

                    } else {
                    // --- Separated: fetch neighbor list only (degree + neighbors) ---
                    uint32_t nbr_read_sz = sizeof(uint32_t) + coloc_config.slack_R * sizeof(uint64_t);
                    uint64_t remote_offset = static_cast<uint64_t>(local_loc) * coloc_config.entry_size
                                           + coloc_config.degree_offset;

                    rdma::RemoteMemoryRegion target = remote_mr;
                    target.addr += remote_offset;
                    target.size = nbr_read_sz;

                    state.conn->post_read(buf, nbr_read_sz, state.fetch_buffer_mr, target, 0);
                    if (!state.conn->wait_send_completion(0, 5000)) { resolve_fail++; continue; }
                    rdma_entry_calls++; rdma_entry_items++;
                    rdma_nbr_expand_calls++;

                    // Buffer starts with degree, then neighbor VIDs
                    std::memcpy(&degree, buf, sizeof(uint32_t));
                    if (degree > coloc_config.slack_R) degree = coloc_config.slack_R;
                    const uint64_t* raw_nbrs = reinterpret_cast<const uint64_t*>(buf + sizeof(uint32_t));
                    nbr_local_copy.assign(raw_nbrs, raw_nbrs + degree);
                    nbr_vids = nbr_local_copy.data();

                    }
                } else if (num_threads > 1 && fetcher_mutexes) {
                    std::lock_guard<std::mutex> lk((*fetcher_mutexes)[owner_node]);
                    if (fetch_mode != rdma::FetchMode::SEPARATED) {
                        const char* entry = fetcher->fetch_entry(local_loc);
                        rdma_entry_calls++; rdma_entry_items++;
                        rdma_nbr_expand_calls++;
                        degree = fetcher->entry_get_degree(entry);
                        const uint64_t* raw_nbrs = fetcher->entry_get_neighbors(entry);
                        nbr_local_copy.assign(raw_nbrs, raw_nbrs + degree);
                        nbr_vids = nbr_local_copy.data();
                    } else {
                        const uint64_t* raw_nbrs = fetcher->fetch_neighbors_only(local_loc);
                        rdma_entry_calls++; rdma_entry_items++;
                        rdma_nbr_expand_calls++;
                        degree = fetcher->fetched_degree();
                        nbr_local_copy.assign(raw_nbrs, raw_nbrs + degree);
                        nbr_vids = nbr_local_copy.data();
                    }
                } else {
                    if (fetch_mode != rdma::FetchMode::SEPARATED) {
                        const char* entry = fetcher->fetch_entry(local_loc);
                        rdma_entry_calls++; rdma_entry_items++;
                        rdma_nbr_expand_calls++;
                        degree = fetcher->entry_get_degree(entry);
                        nbr_vids = fetcher->entry_get_neighbors(entry);

                    } else {
                        nbr_vids = fetcher->fetch_neighbors_only(local_loc);
                        rdma_entry_calls++; rdma_entry_items++;
                        rdma_nbr_expand_calls++;
                        degree = fetcher->fetched_degree();

                    }
                }
                auto ns = timer_ns(t0);
                rdma_neighbor_ns += ns;
                hop_rdma_ns += ns;

            }
            remote_neighbor_accesses++;
        }

        if (degree > coloc_config.slack_R) degree = coloc_config.slack_R;
        const uint64_t cur_source_deleted = deleted_vids.count(cur.vid) ? cur.vid : INVALID_VID;
        if (cur_source_deleted != INVALID_VID) {
            record_deleted_neighbors(cur_source_deleted, nbr_vids, degree);
        }

        // Clear reusable per-hop containers (capacity preserved)
        n_active_owners = 0;
        ordered_nbrs.clear();
        computed.clear();

        {
        auto t_nbr_loop = timer_start();
        for (uint32_t i = 0; i < degree; i++) {
            uint64_t nbr_vid = nbr_vids[i];
            if (nbr_vid == UINT64_MAX) continue;

            auto t_vis = timer_start();
            bool inserted = visited.insert(nbr_vid);
            cpu_visited_ns += timer_ns(t_vis);
            if (!inserted) {
                hop_visited_skip++;
                continue;
            }

            auto t_res = timer_start();
            auto [nbr_owner, nbr_local_loc, nbr_valid] = resolve_vid(nbr_vid, node_store, mapper);
            cpu_resolve_ns += timer_ns(t_res);
            if (!nbr_valid) {
                resolve_fail++;
                continue;
            }
            ordered_nbrs.push_back({nbr_vid, nbr_owner, nbr_local_loc});
            if (nbr_owner < MAX_OWNERS) {
                if (pending_per_owner[nbr_owner].empty()) {
                    active_owners[n_active_owners++] = nbr_owner;
                }
                pending_per_owner[nbr_owner].push_back({nbr_vid, nbr_local_loc, 0.0f, cur_source_deleted});
            }
        }
        cpu_nbr_loop_ns += timer_ns(t_nbr_loop);
        }

        // Phase 1: Compute distances grouped by owner (RDMA efficiency)
        // Results stored in computed map; queue.insert deferred to Phase 2
        //
        // Candidate vector processing:
        //   Pass 1: remote owners -> PQ filter + RDMA post
        //   Bulk sync: drain remote reads before local/remote distance compute
        //   Pass 2: local owner distance compute
        //   Pass 3: remote distance compute from completed buffers
        // Non-multi-QP paths use blocking fetch within Pass 1.

        // AsyncRemoteFetch descriptors live in scratch; the per-owner uncached
        // payload lives in scratch.uncached_per_owner[owner] (reused buffers).
        auto& async_remote_fetches = scratch.async_remote_fetches;
        async_remote_fetches.clear();

        // ---- Pass 1: Remote owners — PQ filter + RDMA post ----
        auto t_candidate_fetch = timer_start();
        auto t_pass1 = timer_start();
        for (uint32_t oi = 0; oi < n_active_owners; oi++) {
            uint32_t owner = active_owners[oi];
            if (owner == my_node_id) continue;  // local handled in Pass 2
            auto& pending = pending_per_owner[owner];

            auto& uncached = scratch.uncached_per_owner[owner];
            uncached.clear();
            for (auto& p : pending) {

                uncached.push_back(p);
            }

            // ---- PQ filter (converge-only: gated by converged flag) ----
            if (converged && pq_filter_enabled && !uncached.empty() && !pq_dt.empty()) {
                auto t_pq = timer_start();
                float worst = queue.size() > 0 ? queue[queue.size()-1].distance : std::numeric_limits<float>::infinity();
                bool beam_full = (queue.size() >= L);
                float threshold = worst * (1.0f + pq_alpha);
                auto& pq_passed = scratch.pq_passed;
                pq_passed.clear();
                const size_t N_u = uncached.size();
                // Prefetch next candidate's pq_code one iter ahead to hide L3 miss.
                for (size_t ui = 0; ui < N_u; ui++) {
                    auto& u = uncached[ui];
                    if (ui + 1 < N_u) {
                        uint32_t npart = danns::decode_partition(uncached[ui+1].vid);
                        if (npart < local_to_tag.size() && uncached[ui+1].local_loc < local_to_tag[npart].size()) {
                            uint32_t ntag = local_to_tag[npart][uncached[ui+1].local_loc];
                            if (ntag > 0) {
                                uint32_t ngid = ntag - 1;
                                const uint8_t* ncode = pq_filter->get_code(ngid);
                                if (ncode) __builtin_prefetch(ncode, 0, 0);
                            }
                        }
                    }
                    if (beam_full) {
                        uint32_t u_part = danns::decode_partition(u.vid);
                        if (u_part < local_to_tag.size() && u.local_loc < local_to_tag[u_part].size()) {
                            uint32_t tag = local_to_tag[u_part][u.local_loc];
                            if (tag > 0) {
                                uint32_t gid = tag - 1;
                                const uint8_t* pq_code = pq_filter->get_code(gid);
                                if (pq_code) {
                                    // Early-abort accumulator: break once partial sum
                                    // already exceeds threshold — saves M/2 chunks on
                                    // average when filter rate is high.
                                    float pq_dist = 0.0f;
                                    bool aborted = false;
                                    for (uint32_t m = 0; m < pq_M; m++) {
                                        pq_dist += pq_dt[m * 256 + pq_code[m]];
                                        if (pq_dist > threshold) { aborted = true; break; }
                                    }
                                    u.pq_dist = pq_dist;
                                    if (aborted) {
                                        pq_filtered++;
                                        continue;
                                    }
                                }
                            }
                        }
                    }
                    pq_passed.push_back(u);
                }
                std::swap(uncached, pq_passed);  // uncached <- filtered (buffers reused)
                cpu_pq_ns += timer_ns(t_pq);
            }

            if (uncached.empty()) continue;

            auto* fetcher = node_store->fetcher(owner);
            if (!fetcher) {
                resolve_fail += uncached.size();
                continue;
            }

            if (use_multi_qp_search) {
                // Per-thread QP: async post, distance compute deferred to Pass 3
                auto ts_it = thread_states.find(owner);
                const auto* mr_ptr = node_store->remote_mr(owner);
                if (ts_it == thread_states.end() || !mr_ptr) {
                    resolve_fail += uncached.size();
                    continue;
                }
                auto& state = const_cast<NodeStore::ThreadRDMAState&>(ts_it->second.at(thread_id));
                auto remote_mr = *mr_ptr;
                uint32_t entry_sz = coloc_config.entry_size;
                uint32_t count = static_cast<uint32_t>(uncached.size());
                char* buf = reinterpret_cast<char*>(state.fetch_buffer.data());

                if (use_full_entry) {
                    // --- Full entry async fetch ---
                    sgl_entries.resize(count);
                    for (uint32_t j = 0; j < count; j++) {
                        sgl_entries[j].local_addr = buf + static_cast<size_t>(j) * entry_sz;
                        sgl_entries[j].length = entry_sz;
                        sgl_entries[j].remote_offset =
                            static_cast<uint64_t>(uncached[j].local_loc) * entry_sz;
                    }
                    auto t0 = timer_start();
                    uint32_t pending_count = state.conn->post_read_batch_async(
                        sgl_entries, remote_mr, state.fetch_buffer_mr, 0);
                    rdma_entry_calls++; rdma_entry_items += count;
                    rdma_vec_fetch_calls++; rdma_full_entry_fetches += count;
                    auto ns = timer_ns(t0);
                    rdma_vector_ns += ns;
                    hop_rdma_ns += ns;
                    async_remote_fetches.push_back({owner, pending_count,
                                                    &state, nullptr, true});
                } else {
                    // --- Vector-only async fetch ---
                    uint32_t vec_read_sz = coloc_config.degree_offset;
                    sgl_entries.resize(count);
                    for (uint32_t j = 0; j < count; j++) {
                        sgl_entries[j].local_addr = buf + static_cast<size_t>(j) * vec_read_sz;
                        sgl_entries[j].length = vec_read_sz;
                        sgl_entries[j].remote_offset =
                            static_cast<uint64_t>(uncached[j].local_loc) * entry_sz;
                    }
                    auto t0 = timer_start();
                    uint32_t pending_count = state.conn->post_read_batch_async(
                        sgl_entries, remote_mr, state.fetch_buffer_mr, 0);
                    rdma_entry_calls++; rdma_entry_items += count;
                    rdma_vec_fetch_calls++; rdma_vec_only_fetches += count;
                    auto ns = timer_ns(t0);
                    rdma_vector_ns += ns;
                    hop_rdma_ns += ns;
                    async_remote_fetches.push_back({owner, pending_count,
                                                    &state, nullptr, false});
                }
            } else if (num_threads > 1 && fetcher_mutexes) {
                std::lock_guard<std::mutex> lk((*fetcher_mutexes)[owner]);
                if (use_full_entry) {
                    auto& batch_locs = scratch.batch_locs;
                    batch_locs.clear();
                    for (auto& u : uncached) batch_locs.push_back(u.local_loc);

                    auto t0 = timer_start();
                    fetcher->fetch_entries_batch(batch_locs);
                    rdma_entry_calls++; rdma_entry_items += uncached.size();
                    rdma_vec_fetch_calls++; rdma_full_entry_fetches += uncached.size();
                    auto ns = timer_ns(t0);
                    rdma_vector_ns += ns;
                    hop_rdma_ns += ns;

                    for (size_t j = 0; j < uncached.size(); j++) {
                        const char* entry = fetcher->get_batch_entry(j);
                        const T* vec = static_cast<const T*>(fetcher->entry_get_vector(entry));
                        auto t_dist = timer_start();
                        float dist = compute_skip ? fake_dist(uncached[j].vid)
                                                  : dist_fn->compare(aligned_query, vec, cmp_dim);
                        cpu_distance_ns += timer_ns(t_dist);
                        auto t_map = timer_start();
                        computed[uncached[j].vid] = {dist, uncached[j].pq_dist, true};
                        record_deleted_vector(uncached[j].source_deleted_vid, uncached[j].vid, vec);
                        n_cmps++;
                        remote_vector_accesses++;
                        hop_remote_vec++;

                        if (need_result_tags) {
                            uint64_t entry_vid;
                            std::memcpy(&entry_vid, entry + coloc_config.vector_id_offset, sizeof(uint64_t));
                            edge_to_entry_vid[uncached[j].vid] = entry_vid;
                        }
                        cpu_map_ns += timer_ns(t_map);

                    }
                } else {
                    auto& batch_locs = scratch.batch_locs;
                    batch_locs.clear();
                    for (auto& u : uncached) batch_locs.push_back(u.local_loc);

                    auto t0 = timer_start();
                    fetcher->fetch_vectors_batch(batch_locs);
                    rdma_entry_calls++; rdma_entry_items += uncached.size();
                    rdma_vec_fetch_calls++; rdma_vec_only_fetches += uncached.size();
                    auto ns = timer_ns(t0);
                    rdma_vector_ns += ns;
                    hop_rdma_ns += ns;

                    for (size_t j = 0; j < uncached.size(); j++) {
                        const T* vec = static_cast<const T*>(fetcher->get_batch_vector(j));
                        auto t_dist = timer_start();
                        float dist = compute_skip ? fake_dist(uncached[j].vid)
                                                  : dist_fn->compare(aligned_query, vec, cmp_dim);
                        cpu_distance_ns += timer_ns(t_dist);
                        auto t_map = timer_start();
                        computed[uncached[j].vid] = {dist, uncached[j].pq_dist, true};
                        record_deleted_vector(uncached[j].source_deleted_vid, uncached[j].vid, vec);
                        n_cmps++;
                        remote_vector_accesses++;
                        hop_remote_vec++;
                        if (need_result_tags) {
                            uint32_t u_part = danns::decode_partition(uncached[j].vid);
                            if (u_part < local_to_tag.size() && uncached[j].local_loc < local_to_tag[u_part].size()) {
                                uint32_t tag = local_to_tag[u_part][uncached[j].local_loc];
                                edge_to_entry_vid[uncached[j].vid] = danns::encode_vector_id(u_part, tag);
                            }
                        }

                            cpu_map_ns += timer_ns(t_map);
                    }
                }
            } else {
                // Single-thread: async post, defer distance compute to Pass 3
                auto& batch_locs = scratch.batch_locs;
                batch_locs.clear();
                for (auto& u : uncached) batch_locs.push_back(u.local_loc);

                if (use_full_entry) {
                    auto t0 = timer_start();
                    uint32_t pending_count = fetcher->fetch_entries_batch_async(batch_locs);
                    rdma_entry_calls++; rdma_entry_items += uncached.size();
                    rdma_vec_fetch_calls++; rdma_full_entry_fetches += uncached.size();
                    auto ns = timer_ns(t0);
                    rdma_vector_ns += ns;
                    hop_rdma_ns += ns;
                    async_remote_fetches.push_back({owner, pending_count,
                                                    nullptr, fetcher, true});
                } else {
                    auto t0 = timer_start();
                    uint32_t pending_count = fetcher->fetch_vectors_batch_async(batch_locs);
                    rdma_entry_calls++; rdma_entry_items += uncached.size();
                    rdma_vec_fetch_calls++; rdma_vec_only_fetches += uncached.size();
                    auto ns = timer_ns(t0);
                    rdma_vector_ns += ns;
                    hop_rdma_ns += ns;
                    async_remote_fetches.push_back({owner, pending_count,
                                                    nullptr, fetcher, false});
                }
            }
        }

        pass1_ns += timer_ns(t_pass1);

        // ---- Bulk-sync barrier ----
        // Drains every async RDMA before local compute starts.
        // The downstream Pass 2 + Pass 3 code is unchanged; Pass 3 polls
        // return pre-drained completions once we arrive.
        {
            auto t_wait = timer_start();
            for (auto& af : async_remote_fetches) {
                uint32_t remaining = af.pending_count;
                while (remaining > 0) {
                    uint32_t polled = af.state ? af.state->conn->poll_read_nonblocking()
                                               : af.fetcher->connection()->poll_read_nonblocking();
                    if (polled > remaining) polled = remaining;
                    remaining -= polled;
                }
                af.pending_count = 0;  // drained; Pass 3 will see all completions ready
            }
            long long wait_ns = timer_ns(t_wait);
            rdma_vector_ns += wait_ns;
            hop_rdma_ns += wait_ns;
        }

        // ---- Pass 2: Local owner distance compute ----
        auto t_pass2 = timer_start();
        {
            auto& local_pending = pending_per_owner[my_node_id];
            // Software prefetch: the pending list gives every local vector's
            // address up front, so pull candidate pi+4 into cache while pi is
            // being compared (upstream DiskANN does the same in its optimized
            // search loop; this colocation path never had it). 2 lines cover
            // the first 128B; the HW prefetcher streams the rest mid-compare.
            constexpr size_t PF_AHEAD = 4;
            for (size_t pi = 0; pi < local_pending.size(); pi++) {
                if (pi + PF_AHEAD < local_pending.size()) {
                    const char* pf = static_cast<const char*>(
                        coloc_store->get_vector(local_pending[pi + PF_AHEAD].local_loc));
                    __builtin_prefetch(pf, 0, 3);
                    __builtin_prefetch(pf + 64, 0, 3);
                }
                auto& p = local_pending[pi];
                auto t_dist = timer_start();
                const T* vec = static_cast<const T*>(coloc_store->get_vector(p.local_loc));
                float dist = compute_skip ? fake_dist(p.vid)
                                          : dist_fn->compare(aligned_query, vec, cmp_dim);
                cpu_distance_ns += timer_ns(t_dist);
                auto t_map = timer_start();
                computed[p.vid] = {dist, 0.0f, false};
                // No cache record for local vids: consolidation reads them
                // zero-copy from coloc_store and never probes the warm store,
                // so recording them only displaces remote arena entries.
                n_cmps++;
                local_vector_accesses++;
                hop_local_vec++;
                if (need_result_tags)
                    edge_to_entry_vid[p.vid] = coloc_store->get_vector_id(p.local_loc);
                cpu_map_ns += timer_ns(t_map);
            }
        }

        pass2_ns += timer_ns(t_pass2);

        // ---- Pass 3: Non-blocking poll + interleaved distance compute ----
        // Round-robin across owners' CQs. Each (peer, thread) has its own CQ,
        // so completions arrive independently per owner. Process entries as
        // soon as their RDMA completes — CPU never idle waiting on one owner.
        auto t_pass3 = timer_start();
        {
        const uint32_t n_fetches = static_cast<uint32_t>(async_remote_fetches.size());

        // Per-owner progress tracking
        struct OwnerProgress {
            uint32_t completions = 0;   // CQ completions received
            uint32_t processed = 0;     // entries whose distances are computed
        };
        OwnerProgress owner_progress[MAX_OWNERS];  // indexed by fetch idx

        // post_read_batch_async drains completions internally when the SQ
        // fills (SQ_DEPTH/2 cap). Those drained WRs already have their data
        // in the buffer, but we only see (pending_count = uncached.size() -
        // drained) completions from poll. Initialize `completions` to the
        // pre-drained count so the loop's termination condition stays in
        // sync with the actual work remaining.
        uint32_t total_remaining = 0;
        for (uint32_t fi = 0; fi < n_fetches; fi++) {
            auto& af = async_remote_fetches[fi];
            uint32_t n_entries = static_cast<uint32_t>(
                scratch.uncached_per_owner[af.owner].size());
            owner_progress[fi].completions = n_entries - af.pending_count;
            total_remaining += n_entries;
        }

        // Lambda: process one completed entry for a given fetch
        auto process_entry = [&](AsyncRemoteFetch& af, uint32_t j) {
            char* buf = af.state ? reinterpret_cast<char*>(af.state->fetch_buffer.data())
                                 : af.fetcher->buffer();
            uint32_t entry_sz = coloc_config.entry_size;
            auto& unc = scratch.uncached_per_owner[af.owner];

            if (af.full_entry) {
                const char* entry = buf + static_cast<size_t>(j) * entry_sz;
                const T* vec = reinterpret_cast<const T*>(entry + coloc_config.vector_offset);
                auto t_dist = timer_start();
                float dist = compute_skip ? fake_dist(unc[j].vid)
                                          : dist_fn->compare(aligned_query, vec, cmp_dim);
                cpu_distance_ns += timer_ns(t_dist);
                auto t_map = timer_start();
                computed[unc[j].vid] = {dist, unc[j].pq_dist, true};
                record_deleted_vector(unc[j].source_deleted_vid, unc[j].vid, vec);
                n_cmps++;
                remote_vector_accesses++;
                hop_remote_vec++;

                if (need_result_tags) {
                    uint64_t entry_vid;
                    std::memcpy(&entry_vid, entry + coloc_config.vector_id_offset, sizeof(uint64_t));
                    edge_to_entry_vid[unc[j].vid] = entry_vid;
                }
                cpu_map_ns += timer_ns(t_map);


            } else {
                uint32_t vec_stride = af.state ? coloc_config.degree_offset : entry_sz;
                const char* entry = buf + static_cast<size_t>(j) * vec_stride;
                const T* vec = reinterpret_cast<const T*>(entry + coloc_config.vector_offset);
                auto t_dist = timer_start();
                float dist = compute_skip ? fake_dist(unc[j].vid)
                                          : dist_fn->compare(aligned_query, vec, cmp_dim);
                cpu_distance_ns += timer_ns(t_dist);
                auto t_map = timer_start();
                computed[unc[j].vid] = {dist, unc[j].pq_dist, true};
                record_deleted_vector(unc[j].source_deleted_vid, unc[j].vid, vec);
                n_cmps++;
                remote_vector_accesses++;
                hop_remote_vec++;

                if (need_result_tags) {
                    if (af.state) {
                        uint64_t entry_vid;
                        std::memcpy(&entry_vid, entry + coloc_config.vector_id_offset, sizeof(uint64_t));
                        edge_to_entry_vid[unc[j].vid] = entry_vid;
                    } else {
                        uint32_t u_part = danns::decode_partition(unc[j].vid);
                        if (u_part < local_to_tag.size() && unc[j].local_loc < local_to_tag[u_part].size()) {
                            uint32_t tag = local_to_tag[u_part][unc[j].local_loc];
                            edge_to_entry_vid[unc[j].vid] = danns::encode_vector_id(u_part, tag);
                        }
                    }
                }
                cpu_map_ns += timer_ns(t_map);


            }
        };

        // Round-robin non-blocking poll loop
        long long hop_pass3_poll_ns = 0;
        long long hop_pass3_compute_ns = 0;
        while (total_remaining > 0) {
            for (uint32_t fi = 0; fi < n_fetches; fi++) {
                auto& af = async_remote_fetches[fi];
                auto& op = owner_progress[fi];
                uint32_t n_entries = static_cast<uint32_t>(
                    scratch.uncached_per_owner[af.owner].size());
                if (op.processed >= n_entries) continue;

                // Non-blocking poll this owner's CQ (timed)
                auto t_poll = timer_start();
                uint32_t polled = 0;
                if (af.state) {
                    polled = af.state->conn->poll_read_nonblocking();
                } else {
                    polled = af.fetcher->connection()->poll_read_nonblocking();
                }
                hop_pass3_poll_ns += timer_ns(t_poll);
                op.completions += polled;

                // IB single-QP ordering: entries [0..completions-1] are complete
                if (op.processed < op.completions) {
                    auto t_comp = timer_start();
                    while (op.processed < op.completions && op.processed < n_entries) {
                        process_entry(af, op.processed);
                        op.processed++;
                        total_remaining--;
                    }
                    hop_pass3_compute_ns += timer_ns(t_comp);
                }
            }
        }

        pass3_poll_ns += hop_pass3_poll_ns;
        pass3_compute_ns += hop_pass3_compute_ns;
        // hop_rdma_ns: account exposed poll time (proxy for RDMA wait not hidden by compute)
        hop_rdma_ns += hop_pass3_poll_ns;
        rdma_vector_ns += hop_pass3_poll_ns;
        }
        pass3_ns += timer_ns(t_pass3);
        candidate_fetch_phase_ns += timer_ns(t_candidate_fetch);

        // Phase 2: Insert into queue in original neighbor order (matches DiskANN)
        {
        auto t_queue = timer_start();
        for (auto& on : ordered_nbrs) {
            auto it = computed.find(on.vid);
            if (it == computed.end()) continue;  // PQ filtered or fetch failed

            bool entered = queue.insert(VIDNeighbor(on.vid, it->second.distance));
            if (entered) hop_inserted++;

            if (tl >= 2) {
                FetchedEntry fe{};
                fe.vid = on.vid;
                fe.distance = it->second.distance;
                fe.pq_distance = it->second.pq_distance;
                fe.is_remote = it->second.is_remote ? 1 : 0;
                fe.entered_beam = entered ? 1 : 0;
                fe.phase = converged ? 2 : 1;
                hop_fetched.push_back(fe);
            }
        }
        cpu_queue_ns += timer_ns(t_queue);
        }

        // Clear active pending_per_owner vectors for next hop
        for (uint32_t oi = 0; oi < n_active_owners; oi++) {
            pending_per_owner[active_owners[oi]].clear();
        }

        // Record hop trace
        if (tl >= 1) {
            HopRecord hr{};
            hr.hop_number = static_cast<uint16_t>(n_hops);
            hr.is_remote_expand = (owner_node != my_node_id) ? 1 : 0;
            hr.degree = degree;
            hr.expanded_vid = cur.vid;
            hr.expanded_distance = cur.distance;
            hr.n_local_vec_fetch = hop_local_vec;
            hr.n_remote_vec_fetch = hop_remote_vec;
            hr.n_inserted = hop_inserted;
            hr.n_visited_skip = hop_visited_skip;
            hr.best_distance = queue.size() > 0 ? queue[0].distance : std::numeric_limits<float>::infinity();
            hr.worst_distance = queue.size() > 0 ? queue[queue.size()-1].distance : std::numeric_limits<float>::infinity();
            hr.rdma_us = static_cast<float>(hop_rdma_ns / 1000.0);
            trace_hops.push_back(hr);
            if (tl >= 2)
                trace_fetched.push_back(std::move(hop_fetched));
        }

        // ---- Convergence detection ----
        // The detector is a pure observer of queue[0].distance, so it now runs
        // on every hop regardless of configuration: that is what populates
        // convergence_hop / converged_observed for plain runs (they used to be
        // identically zero unless batch or PQ was on). The CONTROL flag stays
        // gated exactly as before -- batch expansion and the PQ filter see no
        // behavior change.
        if (queue.size() > 0) {
            bool just_converged = conv_detector.update(queue[0].distance);
            if (just_converged) {
                if (!converged_observed) {
                    converged_observed = true;
                    if (convergence_hop == 0) convergence_hop = n_hops;
                }
                if ((batch_size > 0 || pq_filter_enabled) && !converged) {
                    converged = true;
                    // LUT was built eagerly at query entry — nothing to do here.
                }
            }
        }

      } else {
        // ================================================================
        // ---- Batch expansion path (convergence detected, batch_size > 0)
        // ================================================================
        float pre_batch_best = queue.size() > 0 ? queue[0].distance : std::numeric_limits<float>::infinity();

        auto& batch_cands = scratch.batch_cands;
        batch_cands.clear();
        while (batch_cands.size() < batch_size && queue.has_unexpanded_node()) {
            batch_cands.push_back(queue.closest_unexpanded());
        }
        n_batch_rounds++;
        n_batch_candidates_total += batch_cands.size();

        uint16_t batch_local_vec = 0, batch_remote_vec = 0;
        uint16_t batch_inserted = 0, batch_visited_skip = 0;
        long long batch_rdma_ns = 0;
        uint32_t batch_total_degree = 0;
        uint8_t batch_has_remote = 0;
        std::vector<FetchedEntry> batch_fetched;

        // Scratch: reused across rounds/queries. degree is reset for every
        // slot below because a resolve-fail candidate leaves its slot unfilled
        // (a freshly-constructed vector used to guarantee degree == 0).
        auto& all_cand_neighbors = scratch.all_cand_neighbors;
        all_cand_neighbors.resize(batch_cands.size());
        for (auto& cn : all_cand_neighbors) cn.degree = 0;

        // Flat array keyed by dense owner_id (peer node_id 0..MAX_OWNERS-1) +
        // parallel active-owner tracking. Avoids std::map tree ops per candidate.
        auto& cand_by_owner = scratch.cand_by_owner;  // empty by end-of-round contract
        uint32_t cand_active_owners[MAX_OWNERS];
        uint32_t n_cand_active = 0;
        auto& cand_resolved = scratch.cand_resolved;
        cand_resolved.resize(batch_cands.size());
        {
        auto t_nbr_loop = timer_start();
        for (size_t ci = 0; ci < batch_cands.size(); ci++) {
            auto t_res = timer_start();
            cand_resolved[ci] = resolve_vid(batch_cands[ci].vid, node_store, mapper);
            cpu_resolve_ns += timer_ns(t_res);
            auto [owner_node_r, local_loc_r, valid_r] = cand_resolved[ci];
            if (!valid_r || owner_node_r >= MAX_OWNERS) { resolve_fail++; continue; }
            record_deleted_expansion(batch_cands[ci].vid, owner_node_r, local_loc_r, true);
            if (cand_by_owner[owner_node_r].empty()) {
                cand_active_owners[n_cand_active++] = owner_node_r;
            }
            cand_by_owner[owner_node_r].push_back(ci);
        }
        cpu_nbr_loop_ns += timer_ns(t_nbr_loop);
        }

        // Async neighbor fetch tracking (multi-QP path defers processing).
        // Descriptors + per-owner index/loc payloads live in scratch.
        auto& async_nbr_fetches = scratch.async_nbr_fetches;
        async_nbr_fetches.clear();

        for (uint32_t oi = 0; oi < n_cand_active; oi++) {
            uint32_t owner = cand_active_owners[oi];
            auto& cand_indices = cand_by_owner[owner];
            if (owner == my_node_id) {
                auto t_nbr_loop = timer_start();
                for (size_t ci : cand_indices) {
                    auto [ow, local_loc_c, valid_c] = cand_resolved[ci];
                    uint32_t deg = coloc_store->get_degree(local_loc_c);
                    const uint64_t* nbrs = coloc_store->get_neighbors(local_loc_c);
                    if (deg > coloc_config.slack_R) deg = coloc_config.slack_R;
                    all_cand_neighbors[ci].degree = deg;
                    all_cand_neighbors[ci].nbr_vids.assign(nbrs, nbrs + deg);
                    local_neighbor_accesses++;
                    batch_total_degree += deg;
                }
                cpu_nbr_loop_ns += timer_ns(t_nbr_loop);
            } else {
                batch_has_remote = 1;
                auto& uncached_ci = scratch.nbr_uncached_ci[owner];
                auto& uncached_locs = scratch.nbr_uncached_locs[owner];
                uncached_ci.clear();
                uncached_locs.clear();

                for (size_t ci : cand_indices) {
                    auto [ow, local_loc_c, valid_c] = cand_resolved[ci];

                    uncached_ci.push_back(ci);
                    uncached_locs.push_back(local_loc_c);
                    remote_neighbor_accesses++;
                }

                if (!uncached_locs.empty()) {
                    auto* fetcher = node_store->fetcher(owner);
                    if (!fetcher) {
                        resolve_fail += uncached_locs.size();
                        continue;
                    }

                    // Helper lambda for batch neighbor fetch (with or without lock)
                    auto do_batch_nbr_fetch = [&]() {
                        if (fetch_mode == rdma::FetchMode::COLOCATION) {
                            auto t0 = timer_start();
                            fetcher->fetch_entries_batch(uncached_locs);
                            rdma_entry_calls++; rdma_entry_items += uncached_locs.size();
                            auto ns = timer_ns(t0);
                            rdma_neighbor_ns += ns;
                            batch_rdma_ns += ns;

                            auto t_nbr_loop = timer_start();
                            for (size_t j = 0; j < uncached_ci.size(); j++) {
                                size_t ci = uncached_ci[j];
                                const char* entry = fetcher->get_batch_entry(j);
                                uint32_t deg = fetcher->entry_get_degree(entry);
                                const uint64_t* nbrs = fetcher->entry_get_neighbors(entry);
                                if (deg > coloc_config.slack_R) deg = coloc_config.slack_R;
                                all_cand_neighbors[ci].degree = deg;
                                all_cand_neighbors[ci].nbr_vids.assign(nbrs, nbrs + deg);
                                batch_total_degree += deg;


                            }
                            cpu_nbr_loop_ns += timer_ns(t_nbr_loop);
                        } else {
                            // Separated/local_mmap: batch fetch neighbor lists
                            auto t0 = timer_start();
                            fetcher->fetch_neighbors_batch(uncached_locs);
                            rdma_entry_calls++; rdma_entry_items += uncached_locs.size();
                            rdma_nbr_expand_calls++;
                            auto ns = timer_ns(t0);
                            rdma_neighbor_ns += ns;
                            batch_rdma_ns += ns;

                            auto t_nbr_loop = timer_start();
                            for (size_t j = 0; j < uncached_ci.size(); j++) {
                                size_t ci = uncached_ci[j];
                                uint32_t deg = fetcher->get_batch_degree(j);
                                if (deg > coloc_config.slack_R) deg = coloc_config.slack_R;
                                const uint64_t* nbrs = fetcher->get_batch_neighbors(j);
                                all_cand_neighbors[ci].degree = deg;
                                all_cand_neighbors[ci].nbr_vids.assign(nbrs, nbrs + deg);
                                batch_total_degree += deg;


                            }
                            cpu_nbr_loop_ns += timer_ns(t_nbr_loop);
                        }
                    };

                    if (use_multi_qp_search) {
                        // Per-thread QP: async post, defer processing to poll loop below
                        auto ts_it = thread_states.find(owner);
                        const auto* mr_ptr = node_store->remote_mr(owner);
                        if (ts_it == thread_states.end() || !mr_ptr) {
                            resolve_fail += uncached_locs.size();
                            continue;
                        }
                        auto& state = const_cast<NodeStore::ThreadRDMAState&>(ts_it->second.at(thread_id));
                        auto remote_mr = *mr_ptr;
                        uint32_t entry_sz = coloc_config.entry_size;
                        uint32_t count = static_cast<uint32_t>(uncached_locs.size());
                        char* buf = reinterpret_cast<char*>(state.fetch_buffer.data());

                        bool nbr_full_entry = (fetch_mode == rdma::FetchMode::COLOCATION);
                        uint32_t nbr_read_sz = nbr_full_entry ? entry_sz
                            : (sizeof(uint32_t) + coloc_config.slack_R * sizeof(uint64_t));
                        uint32_t nbr_read_offset = nbr_full_entry ? 0 : coloc_config.degree_offset;

                        auto& batch_sgl_entries = scratch.batch_sgl_entries;
                        batch_sgl_entries.resize(count);
                        for (uint32_t j = 0; j < count; j++) {
                            batch_sgl_entries[j].local_addr = buf + static_cast<size_t>(j) * entry_sz
                                + nbr_read_offset;
                            batch_sgl_entries[j].length = nbr_read_sz;
                            batch_sgl_entries[j].remote_offset =
                                static_cast<uint64_t>(uncached_locs[j]) * entry_sz
                                + nbr_read_offset;
                        }

                        auto t0 = timer_start();
                        uint32_t pending_count = state.conn->post_read_batch_async(
                            batch_sgl_entries, remote_mr, state.fetch_buffer_mr, 0);
                        rdma_entry_calls++; rdma_entry_items += count;
                        rdma_nbr_expand_calls++;
                        auto ns = timer_ns(t0);
                        rdma_neighbor_ns += ns;
                        batch_rdma_ns += ns;

                        // Defer processing — payloads stay in scratch (keyed by owner)
                        async_nbr_fetches.push_back({owner, pending_count, &state,
                            nbr_full_entry});
                    } else if (num_threads > 1 && fetcher_mutexes) {
                        std::lock_guard<std::mutex> lk((*fetcher_mutexes)[owner]);
                        do_batch_nbr_fetch();
                    } else {
                        do_batch_nbr_fetch();
                    }
                }
            }
        }

        if (!batch_interleave) {
        // Bulk-sync barrier: drain every owner's neighbor fetch before any compute.
        {
            auto t_wait = timer_start();
            for (auto& anf : async_nbr_fetches) {
                uint32_t completions = 0;
                while (completions < anf.pending_count) {
                    completions += anf.state->conn->poll_read_nonblocking();
                }
                anf.pending_count = 0;
            }
            long long wait_ns = timer_ns(t_wait);
            batch_nbr_poll_ns += wait_ns;
            batch_rdma_ns += wait_ns;
        }

        // Non-blocking poll loop for deferred async neighbor fetches (multi-QP)
        for (auto& anf : async_nbr_fetches) {
            uint32_t completions = 0;
            const auto& uncached_ci = scratch.nbr_uncached_ci[anf.owner];
            uint32_t n_entries = static_cast<uint32_t>(uncached_ci.size());
            auto t_poll = timer_start();
            while (completions < anf.pending_count) {
                completions += anf.state->conn->poll_read_nonblocking();
            }
            batch_nbr_poll_ns += timer_ns(t_poll);
            // Process all fetched neighbor lists
            char* buf = reinterpret_cast<char*>(anf.state->fetch_buffer.data());
            uint32_t entry_sz = coloc_config.entry_size;
            auto t_nbr_loop = timer_start();
            for (uint32_t j = 0; j < n_entries; j++) {
                size_t ci = uncached_ci[j];
                const char* entry = buf + static_cast<size_t>(j) * entry_sz;
                uint32_t deg;
                std::memcpy(&deg, entry + coloc_config.degree_offset, sizeof(uint32_t));
                if (deg > coloc_config.slack_R) deg = coloc_config.slack_R;
                const uint64_t* nbrs = reinterpret_cast<const uint64_t*>(
                    entry + coloc_config.degree_offset + sizeof(uint32_t));
                all_cand_neighbors[ci].degree = deg;
                all_cand_neighbors[ci].nbr_vids.assign(nbrs, nbrs + deg);
                batch_total_degree += deg;


            }
            cpu_nbr_loop_ns += timer_ns(t_nbr_loop);
        }
        } else {
        // ==== batch_interleave_completion: owner-interleaved neighbor-list
        // completion processing (port of sequential Pass-3 round-robin).
        // Parsing fills slot-keyed all_cand_neighbors[ci]; cross-owner arrival
        // order is immaterial, so this stage is byte-identical to the barrier
        // path even when the flag is on.
        uint32_t nf_completions[MAX_OWNERS] = {0};
        uint32_t nf_processed[MAX_OWNERS] = {0};
        const uint32_t n_nf = static_cast<uint32_t>(async_nbr_fetches.size());
        uint32_t nbr_total_remaining = 0;
        for (uint32_t fi = 0; fi < n_nf; fi++) {
            auto& anf = async_nbr_fetches[fi];
            uint32_t n_entries = static_cast<uint32_t>(
                scratch.nbr_uncached_ci[anf.owner].size());
            // post_read_batch_async may have pre-drained some completions
            nf_completions[fi] = n_entries - anf.pending_count;
            nbr_total_remaining += n_entries;
        }
        long long nbr_poll_acc_ns = 0;
        while (nbr_total_remaining > 0) {
            for (uint32_t fi = 0; fi < n_nf; fi++) {
                auto& anf = async_nbr_fetches[fi];
                const auto& uncached_ci = scratch.nbr_uncached_ci[anf.owner];
                uint32_t n_entries = static_cast<uint32_t>(uncached_ci.size());
                if (nf_processed[fi] >= n_entries) continue;

                auto t_poll = timer_start();
                nf_completions[fi] += anf.state->conn->poll_read_nonblocking();
                nbr_poll_acc_ns += timer_ns(t_poll);

                if (nf_processed[fi] < nf_completions[fi]) {
                    char* buf = reinterpret_cast<char*>(anf.state->fetch_buffer.data());
                    uint32_t entry_sz = coloc_config.entry_size;
                    auto t_nbr_loop = timer_start();
                    while (nf_processed[fi] < nf_completions[fi] &&
                           nf_processed[fi] < n_entries) {
                        uint32_t j = nf_processed[fi];
                        size_t ci = uncached_ci[j];
                        const char* entry = buf + static_cast<size_t>(j) * entry_sz;
                        uint32_t deg;
                        std::memcpy(&deg, entry + coloc_config.degree_offset, sizeof(uint32_t));
                        if (deg > coloc_config.slack_R) deg = coloc_config.slack_R;
                        const uint64_t* nbrs = reinterpret_cast<const uint64_t*>(
                            entry + coloc_config.degree_offset + sizeof(uint32_t));
                        all_cand_neighbors[ci].degree = deg;
                        all_cand_neighbors[ci].nbr_vids.assign(nbrs, nbrs + deg);
                        batch_total_degree += deg;
                        nf_processed[fi]++;
                        nbr_total_remaining--;
                    }
                    cpu_nbr_loop_ns += timer_ns(t_nbr_loop);
                }
            }
        }
        batch_nbr_poll_ns += nbr_poll_acc_ns;
        batch_rdma_ns += nbr_poll_acc_ns;
        }

        // Audit item 1 (same probe class): skip the whole sweep when the index
        // has no deletions -- count() is then 0 for every key, so the loop body
        // never ran anyway.
        if (!deleted_vids.empty()) {
        for (size_t ci = 0; ci < batch_cands.size(); ci++) {
            if (!deleted_vids.count(batch_cands[ci].vid)) continue;
            auto& cn = all_cand_neighbors[ci];
            record_deleted_neighbors(batch_cands[ci].vid, cn.nbr_vids.data(), cn.degree);
        }
        }

        // Phase 2: Collect all unvisited neighbors from all B candidates
        // (BatchPendingNbr unified into file-scope PendingNbr; per-owner lists
        //  live in scratch — empty at round start by end-of-round contract)
        auto& all_nbrs_by_owner = scratch.all_nbrs_by_owner;
        uint32_t nbr_active_owners[MAX_OWNERS];
        uint32_t n_nbr_active = 0;

        {
        auto t_nbr_loop = timer_start();
        // Hot-loop audit item 1 (2026-08-05). The deleted_vids probe below keys
        // on batch_cands[ci].vid -- invariant across the inner (i) loop -- yet
        // ran once per *neighbor*: O(B*slack_R) std::unordered_set lookups
        // (libstdc++ -> hash + integer modulo by a prime bucket count, and the
        // set is a cold cache line on a delete-free run) where O(B) suffice.
        // Hoisted to the ci loop; have_deleted drops it to zero probes when the
        // index carries no deletions. Neither deleted_vids (a const& to the
        // mapper's set) nor batch_cands is written anywhere inside this loop
        // nest, so the hoisted value equals the per-neighbor one on every
        // iteration -> byte-identical source_deleted for every pushed neighbor.
        const bool have_deleted = !deleted_vids.empty();
        for (size_t ci = 0; ci < batch_cands.size(); ci++) {
            auto& cn = all_cand_neighbors[ci];
            const uint64_t cand_vid = batch_cands[ci].vid;
            const uint64_t source_deleted =
                (have_deleted && deleted_vids.count(cand_vid)) ? cand_vid
                                                               : INVALID_VID;
            for (uint32_t i = 0; i < cn.degree; i++) {
                uint64_t nbr_vid = cn.nbr_vids[i];
                if (nbr_vid == UINT64_MAX) continue;
                auto t_vis = timer_start();
                bool inserted = visited.insert(nbr_vid);
                cpu_visited_ns += timer_ns(t_vis);
                if (!inserted) {
                    batch_visited_skip++;
                    continue;
                }
                auto t_res = timer_start();
                auto [nbr_owner, nbr_local_loc, nbr_valid] = resolve_vid(nbr_vid, node_store, mapper);
                cpu_resolve_ns += timer_ns(t_res);
                if (!nbr_valid || nbr_owner >= MAX_OWNERS) { resolve_fail++; continue; }
                if (all_nbrs_by_owner[nbr_owner].empty()) {
                    nbr_active_owners[n_nbr_active++] = nbr_owner;
                }
                all_nbrs_by_owner[nbr_owner].push_back({nbr_vid, nbr_local_loc, 0.0f, source_deleted});
            }
        }
        cpu_nbr_loop_ns += timer_ns(t_nbr_loop);
        }

        // P1b: hoist the local owner to the end so the owner loop below posts
        // every remote fetch before spending time on local distance compute
        // (discovery order otherwise puts the local owner first ~always).
        if (batch_remote_first && n_nbr_active > 1) {
            for (uint32_t oi = 0; oi + 1 < n_nbr_active; oi++) {
                if (nbr_active_owners[oi] == my_node_id) {
                    std::swap(nbr_active_owners[oi],
                              nbr_active_owners[n_nbr_active - 1]);
                    break;
                }
            }
        }

        // Async vector fetch tracking (multi-QP path defers processing).
        // Descriptors + per-owner uncached payloads live in scratch.
        auto& async_vec_fetches = scratch.async_vec_fetches;
        async_vec_fetches.clear();

        // Fetch vectors and compute distances per owner group
        for (uint32_t oi = 0; oi < n_nbr_active; oi++) {
            uint32_t owner = nbr_active_owners[oi];
            auto& pending = all_nbrs_by_owner[owner];
            if (owner == my_node_id) {
                // Software prefetch (port of the sequential Pass-2 pattern):
                // pull candidate pi+4's vector into cache while pi is compared.
                constexpr size_t PF_AHEAD = 4;
                for (size_t pi = 0; pi < pending.size(); pi++) {
                    if (pi + PF_AHEAD < pending.size()) {
                        const char* pf = static_cast<const char*>(
                            coloc_store->get_vector(pending[pi + PF_AHEAD].local_loc));
                        __builtin_prefetch(pf, 0, 3);
                        __builtin_prefetch(pf + 64, 0, 3);
                    }
                    auto& p = pending[pi];
                    auto t_dist = timer_start();
                    const T* vec = static_cast<const T*>(coloc_store->get_vector(p.local_loc));
                    float dist = compute_skip ? fake_dist(p.vid)
                                              : dist_fn->compare(aligned_query, vec, cmp_dim);
                    cpu_distance_ns += timer_ns(t_dist);
                    auto t_queue = timer_start();
                    bool entered = queue.insert(VIDNeighbor(p.vid, dist));
                    cpu_queue_ns += timer_ns(t_queue);
                    // No cache record for local vids (see the seq-path local
                    // compute loop): the warm store is probed for remote
                    // vectors only.
                    n_cmps++;
                    local_vector_accesses++;
                    batch_local_vec++;
                    if (entered) batch_inserted++;
                    if (need_result_tags) {
                        auto t_map = timer_start();
                        edge_to_entry_vid[p.vid] = coloc_store->get_vector_id(p.local_loc);
                        cpu_map_ns += timer_ns(t_map);
                    }
                    if (tl >= 2) {
                        FetchedEntry fe{}; fe.vid = p.vid; fe.distance = dist; fe.pq_distance = 0.0f;
                        fe.is_remote = 0; fe.entered_beam = entered ? 1 : 0;
                        fe.phase = 2;  // batch path = converge
                        batch_fetched.push_back(fe);
                    }
                }
            } else {
                // Late-hop remote skip: skip remote vector fetch after N hops
                auto& uncached = scratch.uncached_per_owner[owner];
                uncached.clear();
                for (auto& p : pending) {

                    uncached.push_back(p);
                }

                // ---- PQ filter (batch path): remove uncached neighbors exceeding PQ threshold ----
                if (pq_filter_enabled && !uncached.empty() && !pq_dt.empty()) {
                    auto t_pq = timer_start();
                    float worst = queue.size() > 0 ? queue[queue.size()-1].distance : std::numeric_limits<float>::infinity();
                    bool beam_full = (queue.size() >= L);
                    float threshold = worst * (1.0f + pq_alpha);
                    auto& pq_passed = scratch.pq_passed;
                    pq_passed.clear();
                    const size_t N_u = uncached.size();
                    for (size_t ui = 0; ui < N_u; ui++) {
                        auto& u = uncached[ui];
                        if (ui + 1 < N_u) {
                            uint32_t npart = danns::decode_partition(uncached[ui+1].vid);
                            if (npart < local_to_tag.size() && uncached[ui+1].local_loc < local_to_tag[npart].size()) {
                                uint32_t ntag = local_to_tag[npart][uncached[ui+1].local_loc];
                                if (ntag > 0) {
                                    uint32_t ngid = ntag - 1;
                                    const uint8_t* ncode = pq_filter->get_code(ngid);
                                    if (ncode) __builtin_prefetch(ncode, 0, 0);
                                }
                            }
                        }
                        if (beam_full) {
                            uint32_t u_part = danns::decode_partition(u.vid);
                            if (u_part < local_to_tag.size() && u.local_loc < local_to_tag[u_part].size()) {
                                uint32_t tag = local_to_tag[u_part][u.local_loc];
                                if (tag > 0) {
                                    uint32_t gid = tag - 1;
                                    const uint8_t* pq_code = pq_filter->get_code(gid);
                                    if (pq_code) {
                                        float pq_dist = 0.0f;
                                        bool aborted = false;
                                        for (uint32_t m = 0; m < pq_M; m++) {
                                            pq_dist += pq_dt[m * 256 + pq_code[m]];
                                            if (pq_dist > threshold) { aborted = true; break; }
                                        }
                                        u.pq_dist = pq_dist;
                                        if (aborted) {
                                            pq_filtered++;
                                            continue;
                                        }
                                    }
                                }
                            }
                        }
                        pq_passed.push_back(u);
                    }
                    std::swap(uncached, pq_passed);  // uncached <- filtered
                    cpu_pq_ns += timer_ns(t_pq);
                }

                if (!uncached.empty()) {
                    auto* fetcher = node_store->fetcher(owner);
                    if (!fetcher) {
                        resolve_fail += uncached.size();
                        continue;
                    }

                    // Helper lambda for batch vector fetch
                    auto do_batch_vec_fetch = [&]() {
                        if (fetch_mode == rdma::FetchMode::COLOCATION) {
                            auto& batch_locs = scratch.batch_locs;
                            batch_locs.clear();
                            for (auto& u : uncached) batch_locs.push_back(u.local_loc);

                            auto t0 = timer_start();
                            fetcher->fetch_entries_batch(batch_locs);
                            rdma_entry_calls++; rdma_entry_items += uncached.size();
                            rdma_vec_fetch_calls++; rdma_full_entry_fetches += uncached.size();
                            auto ns = timer_ns(t0);
                            rdma_vector_ns += ns;
                            batch_rdma_ns += ns;

                            for (size_t j = 0; j < uncached.size(); j++) {
                                auto t_dist = timer_start();
                                const char* entry = fetcher->get_batch_entry(j);
                                const T* vec = static_cast<const T*>(fetcher->entry_get_vector(entry));
                                float dist = compute_skip ? fake_dist(uncached[j].vid)
                                                          : dist_fn->compare(aligned_query, vec, cmp_dim);
                                cpu_distance_ns += timer_ns(t_dist);
                                auto t_queue = timer_start();
                                bool entered = queue.insert(VIDNeighbor(uncached[j].vid, dist));
                                cpu_queue_ns += timer_ns(t_queue);
                                record_deleted_vector(uncached[j].source_deleted_vid, uncached[j].vid, vec);
                                n_cmps++;
                                remote_vector_accesses++;
                                batch_remote_vec++;
                                if (entered) batch_inserted++;

                                if (need_result_tags) {
                                    auto t_map = timer_start();
                                    uint64_t entry_vid;
                                    std::memcpy(&entry_vid, entry + coloc_config.vector_id_offset, sizeof(uint64_t));
                                    edge_to_entry_vid[uncached[j].vid] = entry_vid;
                                    cpu_map_ns += timer_ns(t_map);
                                }

                                if (tl >= 2) {
                                    FetchedEntry fe{}; fe.vid = uncached[j].vid; fe.distance = dist;
                                    fe.pq_distance = uncached[j].pq_dist;
                                    fe.is_remote = 1; fe.entered_beam = entered ? 1 : 0;
                                    fe.phase = 2;  // batch path = converge
                                    batch_fetched.push_back(fe);
                                }


                            }
                        } else {
                            auto& batch_locs = scratch.batch_locs;
                            batch_locs.clear();
                            for (auto& u : uncached) batch_locs.push_back(u.local_loc);

                            auto t0 = timer_start();
                            fetcher->fetch_vectors_batch(batch_locs);
                            rdma_entry_calls++; rdma_entry_items += uncached.size();
                            rdma_vec_fetch_calls++; rdma_vec_only_fetches += uncached.size();
                            auto ns = timer_ns(t0);
                            rdma_vector_ns += ns;
                            batch_rdma_ns += ns;

                            for (size_t j = 0; j < uncached.size(); j++) {
                                auto t_dist = timer_start();
                                const T* vec = static_cast<const T*>(fetcher->get_batch_vector(j));
                                float dist = compute_skip ? fake_dist(uncached[j].vid)
                                                          : dist_fn->compare(aligned_query, vec, cmp_dim);
                                cpu_distance_ns += timer_ns(t_dist);
                                auto t_queue = timer_start();
                                bool entered = queue.insert(VIDNeighbor(uncached[j].vid, dist));
                                cpu_queue_ns += timer_ns(t_queue);
                                record_deleted_vector(uncached[j].source_deleted_vid, uncached[j].vid, vec);
                                n_cmps++;
                                remote_vector_accesses++;
                                batch_remote_vec++;
                                if (entered) batch_inserted++;
                                if (need_result_tags) {
                                    auto t_map = timer_start();
                                    uint32_t u_part = danns::decode_partition(uncached[j].vid);
                                    if (u_part < local_to_tag.size() && uncached[j].local_loc < local_to_tag[u_part].size()) {
                                        uint32_t tag = local_to_tag[u_part][uncached[j].local_loc];
                                        edge_to_entry_vid[uncached[j].vid] = danns::encode_vector_id(u_part, tag);
                                    }
                                    cpu_map_ns += timer_ns(t_map);
                                }

                                if (tl >= 2) {
                                    FetchedEntry fe{}; fe.vid = uncached[j].vid; fe.distance = dist;
                                    fe.pq_distance = uncached[j].pq_dist;
                                    fe.is_remote = 1; fe.entered_beam = entered ? 1 : 0;
                                    fe.phase = 2;  // batch path = converge
                                    batch_fetched.push_back(fe);
                                }
                            }
                        }
                    };

                    if (use_multi_qp_search) {
                        // Per-thread QP: async post, defer processing to poll loop
                        auto ts_it = thread_states.find(owner);
                        const auto* mr_ptr = node_store->remote_mr(owner);
                        if (ts_it == thread_states.end() || !mr_ptr) {
                            resolve_fail += uncached.size();
                            continue;
                        }
                        auto& state = const_cast<NodeStore::ThreadRDMAState&>(ts_it->second.at(thread_id));
                        auto remote_mr = *mr_ptr;
                        uint32_t entry_sz = coloc_config.entry_size;
                        uint32_t count = static_cast<uint32_t>(uncached.size());
                        char* buf = reinterpret_cast<char*>(state.fetch_buffer.data());

                        const bool batch_full_entry =
                            (fetch_mode == rdma::FetchMode::COLOCATION);
                        uint32_t read_sz = batch_full_entry ? entry_sz : coloc_config.degree_offset;

                        auto& batch_sgl_entries = scratch.batch_sgl_entries;
                        batch_sgl_entries.resize(count);
                        for (uint32_t j = 0; j < count; j++) {
                            batch_sgl_entries[j].local_addr = buf + static_cast<size_t>(j) * read_sz;
                            batch_sgl_entries[j].length = read_sz;
                            batch_sgl_entries[j].remote_offset =
                                static_cast<uint64_t>(uncached[j].local_loc) * entry_sz;
                        }

                        auto t0 = timer_start();
                        uint32_t pending_count = state.conn->post_read_batch_async(
                            batch_sgl_entries, remote_mr, state.fetch_buffer_mr, 0);
                        rdma_entry_calls++; rdma_entry_items += count;
                        rdma_vec_fetch_calls++;
                        if (batch_full_entry) rdma_full_entry_fetches += count;
                        else rdma_vec_only_fetches += count;
                        auto ns = timer_ns(t0);
                        rdma_vector_ns += ns;
                        batch_rdma_ns += ns;

                        // Payload stays in scratch.uncached_per_owner[owner]
                        async_vec_fetches.push_back({owner,
                            pending_count, &state, batch_full_entry, read_sz});
                    } else if (num_threads > 1 && fetcher_mutexes) {
                        std::lock_guard<std::mutex> lk((*fetcher_mutexes)[owner]);
                        do_batch_vec_fetch();
                    } else {
                        do_batch_vec_fetch();
                    }
                }
            }
        }

        // Lambda: process one completed remote vector entry (shared by the
        // bulk-sync path and the owner-interleaved path — identical math).
        auto process_vec_entry = [&](AsyncVecFetch& avf, uint32_t j) {
            auto& unc = scratch.uncached_per_owner[avf.owner];
            char* buf = reinterpret_cast<char*>(avf.state->fetch_buffer.data());
            const char* entry = buf + static_cast<size_t>(j) * avf.read_sz;
            const T* vec = reinterpret_cast<const T*>(entry + coloc_config.vector_offset);
            auto t_dist = timer_start();
            float dist = compute_skip ? fake_dist(unc[j].vid)
                                      : dist_fn->compare(aligned_query, vec, cmp_dim);
            cpu_distance_ns += timer_ns(t_dist);
            auto t_queue = timer_start();
            bool entered = queue.insert(VIDNeighbor(unc[j].vid, dist));
            cpu_queue_ns += timer_ns(t_queue);
            record_deleted_vector(unc[j].source_deleted_vid, unc[j].vid, vec);
            n_cmps++;
            remote_vector_accesses++;
            batch_remote_vec++;
            if (entered) batch_inserted++;

            if (need_result_tags) {
                auto t_map = timer_start();
                uint64_t entry_vid;
                std::memcpy(&entry_vid, entry + coloc_config.vector_id_offset, sizeof(uint64_t));
                edge_to_entry_vid[unc[j].vid] = entry_vid;
                cpu_map_ns += timer_ns(t_map);
            }

            if (tl >= 2) {
                FetchedEntry fe{}; fe.vid = unc[j].vid; fe.distance = dist;
                fe.pq_distance = unc[j].pq_dist;
                fe.is_remote = 1; fe.entered_beam = entered ? 1 : 0;
                fe.phase = 2;
                batch_fetched.push_back(fe);
            }
        };

        if (!batch_interleave) {
        // Bulk-sync barrier: drain every owner's CQ before vector distance compute.
        {
            auto t_wait = timer_start();
            for (auto& avf : async_vec_fetches) {
                uint32_t completions = 0;
                while (completions < avf.pending_count) {
                    completions += avf.state->conn->poll_read_nonblocking();
                }
                avf.pending_count = 0;  // drained; subsequent poll returns 0
            }
            long long wait_ns = timer_ns(t_wait);
            batch_vec_poll_ns += wait_ns;
            batch_rdma_ns += wait_ns;
        }

        // Non-blocking poll loop for deferred async vector fetches (multi-QP)
        for (auto& avf : async_vec_fetches) {
            uint32_t completions = 0;
            auto t_poll = timer_start();
            while (completions < avf.pending_count) {
                completions += avf.state->conn->poll_read_nonblocking();
            }
            batch_vec_poll_ns += timer_ns(t_poll);
            uint32_t count = static_cast<uint32_t>(
                scratch.uncached_per_owner[avf.owner].size());
            for (uint32_t j = 0; j < count; j++) {
                process_vec_entry(avf, j);
            }
        }
        } else {
        // ==== batch_interleave_completion: owner-interleaved vector completion
        // processing (port of the sequential Pass-3 round-robin). Completed
        // entries are processed in ARRIVAL order across owners, so the CQ-drain
        // spin is absorbed into distance compute. NOTE: queue insertion order
        // across owners follows arrival — results are order-nondeterministic
        // (recall-equivalent, not byte-identical). Default OFF.
        uint32_t vf_completions[MAX_OWNERS] = {0};
        uint32_t vf_processed[MAX_OWNERS] = {0};
        const uint32_t n_vf = static_cast<uint32_t>(async_vec_fetches.size());
        uint32_t vec_total_remaining = 0;
        for (uint32_t fi = 0; fi < n_vf; fi++) {
            auto& avf = async_vec_fetches[fi];
            uint32_t n_entries = static_cast<uint32_t>(
                scratch.uncached_per_owner[avf.owner].size());
            // post_read_batch_async may have pre-drained some completions
            vf_completions[fi] = n_entries - avf.pending_count;
            vec_total_remaining += n_entries;
        }
        long long vec_poll_acc_ns = 0;
        while (vec_total_remaining > 0) {
            for (uint32_t fi = 0; fi < n_vf; fi++) {
                auto& avf = async_vec_fetches[fi];
                uint32_t n_entries = static_cast<uint32_t>(
                    scratch.uncached_per_owner[avf.owner].size());
                if (vf_processed[fi] >= n_entries) continue;

                auto t_poll = timer_start();
                vf_completions[fi] += avf.state->conn->poll_read_nonblocking();
                vec_poll_acc_ns += timer_ns(t_poll);

                // IB single-QP ordering: entries [0..completions) are complete
                while (vf_processed[fi] < vf_completions[fi] &&
                       vf_processed[fi] < n_entries) {
                    process_vec_entry(avf, vf_processed[fi]);
                    vf_processed[fi]++;
                    vec_total_remaining--;
                }
            }
        }
        batch_vec_poll_ns += vec_poll_acc_ns;
        batch_rdma_ns += vec_poll_acc_ns;
        }

        n_hops += batch_cands.size();

        // Clear active scratch slots for the next round (empty-at-start contract)
        for (uint32_t oi = 0; oi < n_cand_active; oi++) {
            cand_by_owner[cand_active_owners[oi]].clear();
        }
        for (uint32_t oi = 0; oi < n_nbr_active; oi++) {
            all_nbrs_by_owner[nbr_active_owners[oi]].clear();
        }

        float post_batch_best = queue.size() > 0 ? queue[0].distance : std::numeric_limits<float>::infinity();
        if (pre_batch_best > 0 && pre_batch_best < std::numeric_limits<float>::infinity()) {
            float improvement_pct = (pre_batch_best - post_batch_best) / pre_batch_best * 100.0f;
            if (improvement_pct > 5.0f) {
                converged = false;
                conv_detector.reset();
                fallback_count++;
            }
        }

        // Record batch hop trace
        if (tl >= 1) {
            HopRecord hr{};
            hr.hop_number = static_cast<uint16_t>(n_hops);
            hr.is_remote_expand = batch_has_remote;
            hr.degree = batch_total_degree;
            hr.expanded_vid = batch_cands[0].vid;
            hr.expanded_distance = batch_cands[0].distance;
            hr.n_local_vec_fetch = batch_local_vec;
            hr.n_remote_vec_fetch = batch_remote_vec;
            hr.n_inserted = batch_inserted;
            hr.n_visited_skip = batch_visited_skip;
            hr.best_distance = queue.size() > 0 ? queue[0].distance : std::numeric_limits<float>::infinity();
            hr.worst_distance = queue.size() > 0 ? queue[queue.size()-1].distance : std::numeric_limits<float>::infinity();
            hr.rdma_us = static_cast<float>(batch_rdma_ns / 1000.0);
            hr.reserved = static_cast<uint64_t>(batch_cands.size());
            trace_hops.push_back(hr);
            if (tl >= 2)
                trace_fetched.push_back(std::move(batch_fetched));
        }

      } // end if sequential/batch
    } // end while

    // ---- Extract top-K results ----
    // Only when the caller needs global tags (search). Insert discards
    // result.indices/result_count and reads result.pool, so it skips this
    // entire block (and never built edge_to_entry_vid). pos stays 0 → the
    // trace block below produces nothing (trace is off on the insert path).
    size_t pos = 0;
    if (need_result_tags) {
        result.indices.resize(K);
        result.distances.resize(K);

        uint64_t unresolved_vid_count = 0;
        for (size_t i = 0; i < queue.size() && pos < K; i++) {
            uint64_t vid = queue[i].vid;
            if (deleted_vids.count(vid)) continue;
            // TODO: also filter centroid VIDs — they have tag=0 in colocation
            //       entry_vid and can reach top-K via neighbor traversal.
            //       Consider deprecating centroid entries from colocation layout
            //       entirely (frozen point already handles seeding).
            if (vid == frozen_canonical_vid) continue;  // routing node, not data

            uint32_t resolved_tag = 0;
            const uint64_t* evi = edge_to_entry_vid.find(vid);
            if (evi) {
                resolved_tag = static_cast<uint32_t>(danns::decode_local_loc(*evi));
            } else {
                uint32_t part = danns::decode_partition(vid);
                uint32_t ll = static_cast<uint32_t>(danns::decode_local_loc(vid));
                if (part < local_to_tag.size() && ll < local_to_tag[part].size() && local_to_tag[part][ll] > 0) {
                    resolved_tag = local_to_tag[part][ll];
                }
            }

            // Tag-based deletion filter (catches VID mismatches)
            if (resolved_tag > 0 && deleted_tags.count(resolved_tag)) continue;

            result.indices[pos] = (resolved_tag > 0) ? (resolved_tag - 1) : UINT32_MAX;
            if (resolved_tag == 0) unresolved_vid_count++;
            result.distances[pos] = queue[i].distance;
            pos++;
        }
        for (size_t i = pos; i < K; i++) {
            result.indices[i] = UINT32_MAX;
            result.distances[i] = std::numeric_limits<float>::max();
        }
        result.result_count = static_cast<uint16_t>(pos);
        result.unresolved_vids = unresolved_vid_count;
    }

    // ---- Return full candidate pool if requested (for insert) ----
    if (return_pool) {
        result.pool = queue.get_pool();  // All candidates with computed distance
        // LITS filter on pool if requested
        if (lits_filter && !result.pool.empty()) {
            filter_lits_expanded_pool<T>(result.pool, lits_my_partition);
        }
    }

    // ---- Store trace data in result ----
    if (tl >= 1) {
        result.trace_hops = std::move(trace_hops);
        result.trace_fetched = std::move(trace_fetched);
        result.trace_results.resize(pos);
        for (size_t i = 0, ri = 0; i < queue.size() && ri < pos; i++) {
            if (deleted_vids.count(queue[i].vid)) continue;
            result.trace_results[ri].vid = queue[i].vid;
            result.trace_results[ri].distance = queue[i].distance;
            result.trace_results[ri].global_id = result.indices[ri];
            ri++;
        }
    }

    // Fill stats
    result.n_hops = n_hops;
    result.n_cmps = n_cmps;
    result.local_vector_accesses = local_vector_accesses;
    result.remote_vector_accesses = remote_vector_accesses;
    result.local_neighbor_accesses = local_neighbor_accesses;
    result.remote_neighbor_accesses = remote_neighbor_accesses;
    result.resolve_fail = resolve_fail;
    result.visited_count = static_cast<uint64_t>(visited.size());
    result.rdma_vector_ns = rdma_vector_ns;
    result.rdma_neighbor_ns = rdma_neighbor_ns;
    result.rdma_entry_calls = rdma_entry_calls;
    result.rdma_entry_items = rdma_entry_items;
    result.rdma_vec_fetch_calls = rdma_vec_fetch_calls;
    result.rdma_nbr_expand_calls = rdma_nbr_expand_calls;
    result.rdma_full_entry_fetches = rdma_full_entry_fetches;
    result.rdma_vec_only_fetches = rdma_vec_only_fetches;
    result.convergence_hop = convergence_hop;
    result.converged = converged_observed;
    result.n_batch_rounds = n_batch_rounds;
    result.frozen_seed_cache_hits = frozen_seed_cache_hits;
    if (frozen_seed_cache_hits)
        frozen_seed_cache_.hits.fetch_add(frozen_seed_cache_hits,
                                          std::memory_order_relaxed);
    result.n_batch_candidates_total = n_batch_candidates_total;
    result.fallback_count = fallback_count;
    result.pq_filtered = pq_filtered;
    result.cpu_visited_ns = cpu_visited_ns;
    result.cpu_resolve_ns = cpu_resolve_ns;
    result.cpu_distance_ns = cpu_distance_ns;
    result.cpu_queue_ns = cpu_queue_ns;
    result.cpu_nbr_loop_ns = cpu_nbr_loop_ns;
    result.cpu_map_ns = cpu_map_ns;
    result.cpu_pq_ns = cpu_pq_ns;
    result.lut_build_ns = lut_build_ns;
    result.candidate_fetch_phase_ns = candidate_fetch_phase_ns;
    result.pass1_ns = pass1_ns;
    result.pass2_ns = pass2_ns;
    result.pass3_ns = pass3_ns;
    result.pass3_poll_ns = pass3_poll_ns;
    result.pass3_compute_ns = pass3_compute_ns;
    result.batch_nbr_poll_ns = batch_nbr_poll_ns;
    result.batch_vec_poll_ns = batch_vec_poll_ns;

    // Unpin all beam candidates at query end


    return result;
}

// ============================================================================
// search_batch() — Multi-threaded batch search via OpenMP
// Extracted from: search_colocation_pure_batch_typed (L5407-5808)
//
// This is the TOP PRIORITY function for multi-threading and recall measurement.
// Uses OpenMP #pragma omp parallel for with per-thread resources.
// ============================================================================
template <typename T>
std::vector<SearchResult<T>> ColocationBeamSearch::search_batch(
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
    int num_threads)
{
    const ssize_t N = static_cast<ssize_t>(num_queries);
    const uint32_t K = config.K;
    uint32_t L = config.L;
    if (L < K) L = K;

    const auto& coloc_config = node_store->config();
    const rdma::FetchMode fetch_mode = node_store->fetch_mode();

    // Pre-allocate output
    std::vector<SearchResult<T>> results(N);

    // Create fetcher mutexes if multi-threaded
    std::unique_ptr<std::map<uint32_t, std::mutex>> fetcher_mutexes;
    if (num_threads > 1) {
        fetcher_mutexes = std::make_unique<std::map<uint32_t, std::mutex>>();
        // Create mutex entries for all known peers
        // We iterate through known remote_mr peers via trial fetcher lookups
        // The mutexes are created lazily when accessed via operator[]
    }

    // Per-thread trace buffers
    const uint16_t tl = trace_writer ? trace_writer->level() : config.trace_level;
    struct TraceEntry {
        ssize_t query_idx;
        uint32_t query_id;
        uint16_t K_val, L_val;
        SearchResult<T> sr;
    };
    std::vector<std::vector<TraceEntry>> thread_traces;
    if (tl >= 1) {
        thread_traces.resize(num_threads > 1 ? num_threads : 1);
    }

    // Reduction variables for aggregate stats
    uint64_t total_n_hops = 0, total_n_cmps = 0;
    uint64_t total_local_vec = 0, total_remote_vec = 0;
    uint64_t total_local_nbr = 0, total_remote_nbr = 0;
    uint64_t total_resolve_fail = 0, total_visited = 0;
    long long total_rdma_vec_ns = 0, total_rdma_nbr_ns = 0;
    uint64_t total_rdma_entry_calls = 0, total_rdma_entry_items = 0;
    uint64_t total_rdma_vec_fetch_calls = 0, total_rdma_nbr_expand_calls = 0;
    uint64_t total_rdma_full_entry_fetches = 0, total_rdma_vec_only_fetches = 0;
    uint64_t total_convergence_hop = 0, total_batch_rounds = 0;
    uint64_t total_batch_cands = 0, total_fallback = 0;
    uint64_t total_pq_filtered = 0;
    uint64_t total_unresolved_vids = 0;
    long long total_cpu_visited_ns = 0, total_cpu_resolve_ns = 0;
    long long total_cpu_distance_ns = 0, total_cpu_queue_ns = 0;
    long long total_cpu_nbr_loop_ns = 0, total_cpu_map_ns = 0, total_cpu_pq_ns = 0;
    long long total_lut_build_ns = 0;
    long long total_candidate_fetch_phase_ns = 0;
    long long total_pass1_ns = 0, total_pass2_ns = 0, total_pass3_ns = 0;
    long long total_pass3_poll_ns = 0, total_pass3_compute_ns = 0;
    long long total_batch_nbr_poll_ns = 0, total_batch_vec_poll_ns = 0;

    // Per-query latency recording (microseconds)
    std::vector<double> per_query_latency_us(N, 0.0);

    using hrc = std::chrono::high_resolution_clock;

    // Log search mode
    const auto& thread_states = node_store->thread_states();
    const uint32_t num_thread_qps = node_store->num_thread_qps();
    bool search_uses_multi_qp = (num_threads > 1 && !thread_states.empty() && num_thread_qps > 0);
    if (search_uses_multi_qp) {
        rdma::Logger::info("search_batch: using per-thread QP (threads=%d, qps=%u)",
                          num_threads, num_thread_qps);
    } else if (num_threads > 1) {
        rdma::Logger::info("search_batch: using shared fetcher + mutex (threads=%d)",
                          num_threads);
    }

    if (num_threads > 1) {
        omp_set_num_threads(num_threads);
    }

    // Build a per-query SearchConfig (shared base, per-query trace_query_id)
    SearchConfig per_query_config = config;
    per_query_config.L = L;
    per_query_config.trace_level = tl;

    #pragma omp parallel if(num_threads > 1)
    {
        // Thread-local aligned query buffer (allocated once per thread)
        size_t aq_bytes = aligned_dim * sizeof(T);
        aq_bytes = ((aq_bytes + 31) / 32) * 32;
        void* aq_mem = std::aligned_alloc(32, aq_bytes);
        T* aq = aq_mem ? static_cast<T*>(aq_mem) : nullptr;

        int tid = 0;
        if (num_threads > 1) {
            tid = omp_get_thread_num();
        }

        #pragma omp for schedule(dynamic, 4) \
            reduction(+:total_n_hops, total_n_cmps, total_local_vec, total_remote_vec, \
                      total_local_nbr, total_remote_nbr, total_resolve_fail, total_visited, \
                      total_rdma_vec_ns, total_rdma_nbr_ns, \
                      total_rdma_entry_calls, total_rdma_entry_items, \
                      total_rdma_vec_fetch_calls, total_rdma_nbr_expand_calls, \
                      total_rdma_full_entry_fetches, total_rdma_vec_only_fetches, \
                      total_convergence_hop, \
                      total_batch_rounds, total_batch_cands, total_fallback, \
                      total_pq_filtered, \
                      total_cpu_visited_ns, total_cpu_resolve_ns, \
                      total_cpu_distance_ns, total_cpu_queue_ns, total_cpu_nbr_loop_ns, \
                      total_cpu_map_ns, total_lut_build_ns, total_candidate_fetch_phase_ns, \
                      total_pass1_ns, total_pass2_ns, total_pass3_ns, \
                      total_pass3_poll_ns, total_pass3_compute_ns, \
                      total_batch_nbr_poll_ns, total_batch_vec_poll_ns)
        for (ssize_t qi = 0; qi < N; qi++) {
            if (!aq) continue;  // skip on alloc failure

            // Fill aligned query from input
            std::memcpy(aq, queries + qi * vector_dim, vector_dim * sizeof(T));
            if (aligned_dim > vector_dim) {
                std::memset(aq + vector_dim, 0, (aligned_dim - vector_dim) * sizeof(T));
            }

            // Cache lifecycle: clear on per_query


            // Per-query config with this query's ID
            SearchConfig qi_config = per_query_config;
            qi_config.trace_query_id = query_ids[qi];

            auto t_query_start = hrc::now();
            auto sr = search(
                aq, aligned_dim, cmp_dim, dist_fn,
                qi_config,
                node_store, mapper, pq_filter,
                fetcher_mutexes.get(),
                num_threads,
                tid);
            auto t_query_end = hrc::now();
            per_query_latency_us[qi] = std::chrono::duration_cast<std::chrono::nanoseconds>(
                t_query_end - t_query_start).count() / 1000.0;

            // Accumulate stats
            total_n_hops += sr.n_hops;
            total_n_cmps += sr.n_cmps;
            total_local_vec += sr.local_vector_accesses;
            total_remote_vec += sr.remote_vector_accesses;
            total_local_nbr += sr.local_neighbor_accesses;
            total_remote_nbr += sr.remote_neighbor_accesses;
            total_resolve_fail += sr.resolve_fail;
            total_visited += sr.visited_count;
            total_rdma_vec_ns += sr.rdma_vector_ns;
            total_rdma_nbr_ns += sr.rdma_neighbor_ns;
            total_rdma_entry_calls += sr.rdma_entry_calls;
            total_rdma_entry_items += sr.rdma_entry_items;
            total_rdma_vec_fetch_calls += sr.rdma_vec_fetch_calls;
            total_rdma_nbr_expand_calls += sr.rdma_nbr_expand_calls;
            total_rdma_full_entry_fetches += sr.rdma_full_entry_fetches;
            total_rdma_vec_only_fetches += sr.rdma_vec_only_fetches;
            total_convergence_hop += sr.convergence_hop;
            total_batch_rounds += sr.n_batch_rounds;
            total_batch_cands += sr.n_batch_candidates_total;
            total_fallback += sr.fallback_count;
            total_pq_filtered += sr.pq_filtered;
            total_unresolved_vids += sr.unresolved_vids;
            total_cpu_visited_ns += sr.cpu_visited_ns;
            total_cpu_resolve_ns += sr.cpu_resolve_ns;
            total_cpu_distance_ns += sr.cpu_distance_ns;
            total_cpu_queue_ns += sr.cpu_queue_ns;
            total_cpu_nbr_loop_ns += sr.cpu_nbr_loop_ns;
            total_cpu_map_ns += sr.cpu_map_ns;
            total_cpu_pq_ns += sr.cpu_pq_ns;
            total_lut_build_ns += sr.lut_build_ns;
            total_candidate_fetch_phase_ns += sr.candidate_fetch_phase_ns;
            total_pass1_ns += sr.pass1_ns;
            total_pass2_ns += sr.pass2_ns;
            total_pass3_ns += sr.pass3_ns;
            total_pass3_poll_ns += sr.pass3_poll_ns;
            total_pass3_compute_ns += sr.pass3_compute_ns;
            total_batch_nbr_poll_ns += sr.batch_nbr_poll_ns;
            total_batch_vec_poll_ns += sr.batch_vec_poll_ns;

            // Store trace if active
            if (tl >= 1) {
                int trace_tid = (num_threads > 1) ? tid : 0;
                TraceEntry te;
                te.query_idx = qi;
                te.query_id = query_ids[qi];
                te.K_val = static_cast<uint16_t>(K);
                te.L_val = static_cast<uint16_t>(L);
                te.sr = std::move(sr);
                thread_traces[trace_tid].push_back(std::move(te));
            } else {
                // Store result directly
                results[qi] = std::move(sr);
            }
        } // end omp for

        std::free(aq_mem);
    } // end omp parallel

    // Merge and write traces sorted by query index
    if (tl >= 1 && trace_writer) {
        std::vector<TraceEntry*> all_traces;
        for (auto& tt : thread_traces) {
            for (auto& te : tt) {
                all_traces.push_back(&te);
            }
        }
        std::sort(all_traces.begin(), all_traces.end(),
                  [](const TraceEntry* a, const TraceEntry* b) {
                      return a->query_idx < b->query_idx;
                  });
        for (auto* te : all_traces) {
            QueryHeader qh{};
            qh.query_id = te->query_id;
            qh.K = te->K_val;
            qh.L = te->L_val;
            qh.n_hops = static_cast<uint16_t>(te->sr.trace_hops.size());
            qh.n_results = te->sr.result_count;
            trace_writer->write_query(qh, te->sr.trace_results.data(),
                                             te->sr.result_count,
                                             te->sr.trace_hops, te->sr.trace_fetched);
            // Move results out of trace entries into results vector
            results[te->query_idx] = std::move(te->sr);
        }
    }

    // Store per-query latency in each result
    for (ssize_t qi = 0; qi < N; qi++) {
        // Latency stored as a stat field (callers can access per_query_latency_us separately)
        // The aggregate stats are available through the returned SearchResults
    }

    return results;
}

// ============================================================================
// interleave_feasible — gate for the hop-interleaved multi-query path
// ============================================================================
bool ColocationBeamSearch::interleave_feasible(
    const SearchConfig& config,
    NodeStore* node_store,
    PartitionMapper* mapper,
    PQFilter* pq_filter,
    int num_threads)
{
    if (config.interleave_factor <= 1) return false;
    if (num_threads <= 1) return false;
    // Only the multi-QP + SEPARATED fetch path is supported.
    if (node_store->fetch_mode() != rdma::FetchMode::SEPARATED) return false;
    const auto& ts = node_store->thread_states();
    if (ts.empty() || node_store->num_thread_qps() == 0) return false;
    // Per-query optimizations the interleaved core does not replicate.
    if (pq_filter && pq_filter->enabled()) return false;
    if (config.batch_size > 1) return false;   // batch-prefetch path unsupported
    if (config.trace_level >= 1) return false;
    if (config.return_pool) return false;
    // Interleaved core skips consolidation-cache warming and mid-search deletion
    // bookkeeping; only correct when there are no active deletions.
    if (mapper && (!mapper->deleted_vids().empty() ||
                   !mapper->deleted_tags().empty())) return false;

    // The per-thread fetch buffer must slice into interleave_factor regions each
    // big enough for one hop's worst case (slack_R vectors, or one neighbor list).
    const auto& cc = node_store->config();

    // SQ-capacity guard: each interleaved query issues up to (slack_R neighbor
    // reads + 1 vector/entry read) WRs per hop; with interleave_factor queries
    // running concurrently this can exceed the 128-deep SQ. B1 turns overflow
    // into waiting rather than a crash, but admitting a config that cannot fit
    // would just serialize on the SQ, so keep the interleaved path off then.
    // With cross-call SQ flow control (B1), demand beyond the SQ depth is no
    // longer a crash — post_read_batch_async waits for space — so exceeding
    // the SQ only partially serializes posting. Admit and log instead of
    // rejecting (the original hard reject also mis-rejected factor=2:
    // 2*(64+1)=130 > 128). Keep a sanity cap so absurd factors fall back.
    uint32_t per_query_wrs = std::min<uint32_t>(cc.slack_R, 64u) + 1u;
    uint64_t worst_demand = static_cast<uint64_t>(config.interleave_factor) * per_query_wrs;
    if (worst_demand > static_cast<uint64_t>(rdma::DEFAULT_SQ_DEPTH) * 8u) {
        rdma::Logger::info(
            "interleave_feasible: false — interleave_factor=%u demand %lu WRs > 8x SQ_DEPTH",
            config.interleave_factor, worst_demand);
        return false;
    }
    if (worst_demand > rdma::DEFAULT_SQ_DEPTH) {
        rdma::Logger::info(
            "interleave: factor=%u worst-case demand %lu WRs > SQ_DEPTH=%u — "
            "posting will partially serialize on SQ space (safe, B1 flow control)",
            config.interleave_factor, worst_demand, rdma::DEFAULT_SQ_DEPTH);
    }

    size_t buf_bytes = 0;
    for (const auto& kv : ts) {
        if (config.interleave_factor == 0) break;
        if (!kv.second.empty()) { buf_bytes = kv.second[0].fetch_buffer.size() * sizeof(float); break; }
    }
    if (buf_bytes == 0) return false;
    size_t slice_bytes = buf_bytes / config.interleave_factor;
    size_t vec_read_sz = cc.degree_offset;  // [8B id + vector]
    size_t nbr_read_sz = sizeof(uint32_t) + static_cast<size_t>(cc.slack_R) * sizeof(uint64_t);
    if (slice_bytes < static_cast<size_t>(cc.slack_R) * vec_read_sz) return false;
    if (slice_bytes < nbr_read_sz) return false;
    return true;
}

// ============================================================================
// search_interleaved — hop-interleaved multi-query-per-thread beam search
//
// Drives `group_size` queries concurrently on one thread. Each query is an
// independent beam-search state machine; the driver issues a query's per-hop
// RDMA, then moves on to other queries instead of spinning on its completion,
// so up to `group_size` reads are outstanding per thread. Completions are routed
// by FIFO order on each per-(owner,thread) CQ (each owner's reads complete in
// post order), and each (owner, query) gets a private slice of the owner's
// fetch buffer so concurrent reads never clobber one another.
//
// Because interleaving only changes *when* a query waits — never the sequence of
// operations on its own queue/visited set — each query's result is bit-identical
// to the per-query search() path. Seeding is done blocking/sequentially up front
// (a small fraction of total hops) to reuse the exact seeding semantics.
// ============================================================================
template <typename T>
std::vector<SearchResult<T>> ColocationBeamSearch::search_interleaved(
    const T* const* aligned_queries,
    const uint32_t* query_ids,
    uint32_t group_size,
    size_t aligned_dim,
    size_t cmp_dim,
    diskann::Distance<T>* dist_fn,
    const SearchConfig& config,
    NodeStore* node_store,
    PartitionMapper* mapper,
    int num_threads,
    int thread_id)
{
    (void)query_ids; (void)aligned_dim; (void)num_threads;
    const uint32_t K = config.K;
    const uint32_t L = config.L;
    const uint32_t entry_mode = config.entry_mode;
    const bool compute_skip = config.compute_skip;
    auto fake_dist = [](uint64_t vid) -> float {
        return static_cast<float>((vid * 2654435761ull) & 0xFFFFFF);
    };

    auto* coloc_store = node_store->local_store();
    const auto& coloc_config = node_store->config();
    const uint32_t my_node_id = node_store->node_id();
    const uint32_t my_partition_id = node_store->partition_id();
    const auto& local_to_tag = node_store->local_to_tag();
    const auto& thread_states = node_store->thread_states();

    const auto& centroid_local_locs = mapper->centroid_local_locs();
    const uint32_t frozen_local_loc = mapper->frozen_local_loc();
    const uint64_t frozen_canonical_vid = mapper->frozen_canonical_vid();
    const auto& deleted_vids = mapper->deleted_vids();
    const auto& deleted_tags = mapper->deleted_tags();

    static constexpr uint32_t MAX_OWNERS = danns::kMaxOwnerNodes;

    const uint32_t entry_size = coloc_config.entry_size;
    const uint32_t degree_offset = static_cast<uint32_t>(coloc_config.degree_offset);
    const uint32_t vector_offset = static_cast<uint32_t>(coloc_config.vector_offset);
    const uint32_t vector_id_offset = static_cast<uint32_t>(coloc_config.vector_id_offset);
    const uint32_t slack_R = coloc_config.slack_R;
    const uint32_t vec_bytes = coloc_config.dim * coloc_config.element_size;
    const uint32_t vec_read_sz = degree_offset;                     // [8B id + vector]
    const uint32_t nbr_read_sz = sizeof(uint32_t) + slack_R * sizeof(uint64_t);

    // Resolve the per-thread RDMA state for an owner (nullptr if unavailable).
    auto owner_state = [&](uint32_t owner) -> NodeStore::ThreadRDMAState* {
        auto it = thread_states.find(owner);
        if (it == thread_states.end()) return nullptr;
        if (thread_id >= static_cast<int>(it->second.size())) return nullptr;
        return const_cast<NodeStore::ThreadRDMAState*>(&it->second[thread_id]);
    };

    // Per-owner fetch-buffer slice size (bytes). All owners share identical
    // buffer sizing, so one value suffices. Feasibility guarantees it fits.
    size_t buf_bytes = 0;
    for (const auto& kv : thread_states) {
        if (thread_id < static_cast<int>(kv.second.size())) {
            buf_bytes = kv.second[thread_id].fetch_buffer.size() * sizeof(float);
            break;
        }
    }
    const size_t slice_bytes = group_size ? (buf_bytes / group_size) : buf_bytes;

    enum class Phase : uint8_t { NEED_EXPAND, EXPAND_WAIT, VECTOR_WAIT, DONE };
    struct OrderedNbr { uint64_t vid; uint32_t owner; uint32_t local_loc; };
    struct VecFetch  { uint64_t vid; uint32_t local_loc; };

    struct QState {
        uint32_t idx = 0;
        const T* query = nullptr;
        VIDNeighborPriorityQueue queue;
        tsl::robin_set<uint64_t> visited;
        tsl::robin_map<uint64_t, uint64_t> edge_to_entry_vid;
        Phase phase = Phase::NEED_EXPAND;
        uint32_t pending_segments = 0;
        uint32_t expand_owner = 0;
        std::vector<OrderedNbr> ordered;
        std::vector<VecFetch> vec_per_owner[MAX_OWNERS];
        uint32_t vec_active_owners[MAX_OWNERS];
        uint32_t n_vec_active = 0;
        tsl::robin_map<uint64_t, float> computed;
        // stats
        uint64_t n_hops = 0, n_cmps = 0;
        uint64_t local_vec = 0, remote_vec = 0, local_nbr = 0, remote_nbr = 0;
        uint64_t resolve_fail = 0;
        uint64_t rdma_entry_calls = 0, rdma_entry_items = 0;
        uint64_t rdma_vec_fetch_calls = 0, rdma_nbr_expand_calls = 0, rdma_vec_only_fetches = 0;
    };

    std::vector<QState> states(group_size);
    for (uint32_t q = 0; q < group_size; q++) {
        states[q].idx = q;
        states[q].query = aligned_queries[q];
        states[q].queue = VIDNeighborPriorityQueue(L);
        states[q].visited.reserve(L * 20);
    }

    // Per-owner FIFO segment queues: each segment is (query, remaining WRs).
    std::array<std::deque<std::pair<uint32_t, uint32_t>>, MAX_OWNERS> owner_deque;

    std::vector<rdma::SGLEntry> sgl;
    sgl.reserve(slack_R);

    // ------------------------------------------------------------------
    // Seeding — exact replica of search()'s seed_vid + centroid/frozen seed,
    // done blocking and sequentially per query.
    // ------------------------------------------------------------------
    auto seed_vid = [&](QState& s, uint64_t nbr_vid) {
        if (nbr_vid == UINT64_MAX) return;
        if (s.visited.count(nbr_vid)) return;
        auto [nbr_owner, nbr_local_loc, nbr_valid] = resolve_vid(nbr_vid, node_store, mapper);
        if (!nbr_valid) return;
        s.visited.insert(nbr_vid);
        if (nbr_owner == my_node_id) {
            const T* vec = static_cast<const T*>(coloc_store->get_vector(nbr_local_loc));
            float dist = compute_skip ? fake_dist(nbr_vid)
                                      : dist_fn->compare(s.query, vec, cmp_dim);
            s.queue.insert(VIDNeighbor(nbr_vid, dist));
            s.n_cmps++; s.local_vec++;
            s.edge_to_entry_vid[nbr_vid] = coloc_store->get_vector_id(nbr_local_loc);
        } else {
            auto* st = owner_state(nbr_owner);
            const auto* mr_ptr = node_store->remote_mr(nbr_owner);
            if (!st || !mr_ptr) return;
            rdma::RemoteMemoryRegion target = *mr_ptr;
            target.addr += static_cast<uint64_t>(nbr_local_loc) * entry_size + vector_offset;
            target.size = vec_bytes;
            char* buf = reinterpret_cast<char*>(st->fetch_buffer.data());
            st->conn->post_read(buf, vec_bytes, st->fetch_buffer_mr, target, 0);
            if (!st->conn->wait_send_completion(0, 5000)) { s.resolve_fail++; return; }
            const T* vec = reinterpret_cast<const T*>(buf);
            float dist = compute_skip ? fake_dist(nbr_vid)
                                      : dist_fn->compare(s.query, vec, cmp_dim);
            s.queue.insert(VIDNeighbor(nbr_vid, dist));
            s.n_cmps++; s.remote_vec++;
            s.rdma_entry_calls++; s.rdma_entry_items++;
            uint32_t nbr_part = danns::decode_partition(nbr_vid);
            if (nbr_part < local_to_tag.size() && nbr_local_loc < local_to_tag[nbr_part].size()) {
                uint32_t tag = local_to_tag[nbr_part][nbr_local_loc];
                s.edge_to_entry_vid[nbr_vid] = danns::encode_vector_id(nbr_part, tag);
            }
        }
    };

    for (uint32_t q = 0; q < group_size; q++) {
        QState& s = states[q];
        if (entry_mode == 0) {
            auto cl_it = centroid_local_locs.find(my_partition_id);
            if (cl_it != centroid_local_locs.end()) {
                uint32_t centroid_loc = cl_it->second;
                uint32_t centroid_deg = coloc_store->get_degree(centroid_loc);
                const uint64_t* centroid_nbrs = coloc_store->get_neighbors(centroid_loc);
                for (uint32_t i = 0; i < centroid_deg; i++) seed_vid(s, centroid_nbrs[i]);
            }
        }
        if (frozen_local_loc != UINT32_MAX) {
            uint64_t frozen_vid = (frozen_canonical_vid != UINT64_MAX)
                ? frozen_canonical_vid
                : danns::encode_vector_id(my_partition_id, frozen_local_loc);
            if (!s.visited.count(frozen_vid)) {
                s.visited.insert(frozen_vid);
                const T* fvec = static_cast<const T*>(coloc_store->get_vector(frozen_local_loc));
                float fd = compute_skip ? fake_dist(frozen_vid)
                                        : dist_fn->compare(s.query, fvec, cmp_dim);
                s.queue.insert(VIDNeighbor(frozen_vid, fd));
                s.n_cmps++; s.local_vec++;
                s.edge_to_entry_vid[frozen_vid] = coloc_store->get_vector_id(frozen_local_loc);
            }
            uint32_t fdeg = coloc_store->get_degree(frozen_local_loc);
            const uint64_t* fnbrs = coloc_store->get_neighbors(frozen_local_loc);
            for (uint32_t i = 0; i < fdeg; i++) seed_vid(s, fnbrs[i]);
        }
    }

    // ------------------------------------------------------------------
    // Hop helpers
    // ------------------------------------------------------------------
    // From a node's neighbor list: build the per-hop ordered/owner-grouped work,
    // compute local-owned vectors immediately, and post async batches for remote
    // owners. Transitions the query to VECTOR_WAIT (remote reads outstanding) or
    // straight to integrate (all-local / all-drained).
    std::function<void(QState&)> integrate;
    // Scratch for build_and_post's two-pass local compute (per call; this
    // function body runs on one thread, calls are sequential).
    std::vector<std::pair<uint64_t, uint64_t>> bp_local;
    bp_local.reserve(slack_R);
    auto build_and_post = [&](QState& s, uint32_t degree, const uint64_t* nbrs) {
        s.ordered.clear();
        s.computed.clear();
        s.n_vec_active = 0;
        if (degree > slack_R) degree = slack_R;
        for (uint32_t i = 0; i < degree; i++) {
            uint64_t nbr_vid = nbrs[i];
            if (nbr_vid == UINT64_MAX) continue;
            if (!s.visited.insert(nbr_vid).second) continue;
            auto [no, nll, nv] = resolve_vid(nbr_vid, node_store, mapper);
            if (!nv) { s.resolve_fail++; continue; }
            s.ordered.push_back({nbr_vid, no, nll});
            if (no == my_node_id) {
                // Two-pass local compute: prefetch every local vector as it is
                // resolved (addresses are known from the neighbor list), then
                // compute distances in a second pass once the loads are in
                // flight. Mirrors upstream DiskANN's prefetched search loop.
                const char* pf = static_cast<const char*>(coloc_store->get_vector(nll));
                __builtin_prefetch(pf, 0, 3);
                __builtin_prefetch(pf + 64, 0, 3);
                bp_local.push_back({nbr_vid, nll});
            } else if (no < MAX_OWNERS) {
                if (s.vec_per_owner[no].empty()) s.vec_active_owners[s.n_vec_active++] = no;
                s.vec_per_owner[no].push_back({nbr_vid, nll});
            }
        }
        for (const auto& lv : bp_local) {
            const T* vec = static_cast<const T*>(coloc_store->get_vector(lv.second));
            float d = compute_skip ? fake_dist(lv.first)
                                   : dist_fn->compare(s.query, vec, cmp_dim);
            s.computed[lv.first] = d;
            s.n_cmps++; s.local_vec++;
            s.edge_to_entry_vid[lv.first] = coloc_store->get_vector_id(lv.second);
        }
        bp_local.clear();

        s.pending_segments = 0;
        for (uint32_t k = 0; k < s.n_vec_active; k++) {
            uint32_t owner = s.vec_active_owners[k];
            auto& vf = s.vec_per_owner[owner];
            auto* st = owner_state(owner);
            const auto* mr_ptr = node_store->remote_mr(owner);
            if (!st || !mr_ptr) { s.resolve_fail += vf.size(); vf.clear(); continue; }
            char* slice = reinterpret_cast<char*>(st->fetch_buffer.data())
                          + static_cast<size_t>(s.idx) * slice_bytes;
            sgl.resize(vf.size());
            for (size_t j = 0; j < vf.size(); j++) {
                sgl[j].local_addr = slice + j * vec_read_sz;
                sgl[j].length = vec_read_sz;
                sgl[j].remote_offset = static_cast<uint64_t>(vf[j].local_loc) * entry_size;
            }
            uint32_t pend = st->conn->post_read_batch_async(sgl, *mr_ptr, st->fetch_buffer_mr, 0);
            s.rdma_entry_calls++; s.rdma_entry_items += static_cast<uint64_t>(vf.size());
            s.rdma_vec_fetch_calls++; s.rdma_vec_only_fetches += static_cast<uint64_t>(vf.size());
            if (pend > 0) {
                owner_deque[owner].push_back({s.idx, pend});
                s.pending_segments++;
            }
        }

        if (s.pending_segments == 0) {
            integrate(s);
            s.phase = Phase::NEED_EXPAND;
        } else {
            s.phase = Phase::VECTOR_WAIT;
        }
    };

    // Compute remote-owned vectors (now in slices) and insert all neighbors into
    // the beam queue in original neighbor order, matching search() semantics.
    integrate = [&](QState& s) {
        for (uint32_t k = 0; k < s.n_vec_active; k++) {
            uint32_t owner = s.vec_active_owners[k];
            auto& vf = s.vec_per_owner[owner];
            auto* st = owner_state(owner);
            if (st && !vf.empty()) {
                char* slice = reinterpret_cast<char*>(st->fetch_buffer.data())
                              + static_cast<size_t>(s.idx) * slice_bytes;
                for (size_t j = 0; j < vf.size(); j++) {
                    const char* ent = slice + j * vec_read_sz;
                    const T* vec = reinterpret_cast<const T*>(ent + vector_offset);
                    float d = compute_skip ? fake_dist(vf[j].vid)
                                           : dist_fn->compare(s.query, vec, cmp_dim);
                    s.computed[vf[j].vid] = d;
                    s.n_cmps++; s.remote_vec++;
                    uint64_t entry_vid;
                    std::memcpy(&entry_vid, ent + vector_id_offset, sizeof(uint64_t));
                    s.edge_to_entry_vid[vf[j].vid] = entry_vid;
                }
            }
            vf.clear();
        }
        s.n_vec_active = 0;
        for (auto& on : s.ordered) {
            auto it = s.computed.find(on.vid);
            if (it == s.computed.end()) continue;
            s.queue.insert(VIDNeighbor(on.vid, it->second));
        }
        s.ordered.clear();
        s.computed.clear();
    };

    // Parse the just-completed neighbor-list read and launch the hop's vectors.
    auto finish_expand = [&](QState& s) {
        auto* st = owner_state(s.expand_owner);
        if (!st) { s.phase = Phase::NEED_EXPAND; return; }
        char* slice = reinterpret_cast<char*>(st->fetch_buffer.data())
                      + static_cast<size_t>(s.idx) * slice_bytes;
        uint32_t degree = 0;
        std::memcpy(&degree, slice, sizeof(uint32_t));
        if (degree > slack_R) degree = slack_R;
        // Copy neighbor VIDs out before vector batches reuse this slice.
        const uint64_t* raw = reinterpret_cast<const uint64_t*>(slice + sizeof(uint32_t));
        static thread_local std::vector<uint64_t> nbr_copy;
        nbr_copy.assign(raw, raw + degree);
        build_and_post(s, degree, nbr_copy.data());
    };

    // Issue one expand step for a query in NEED_EXPAND. Local-owned nodes resolve
    // synchronously (no RDMA); remote nodes post a 1-WR neighbor-list read.
    auto issue_expand = [&](QState& s) {
        if (!s.queue.has_unexpanded_node()) { s.phase = Phase::DONE; return; }
        VIDNeighbor cur = s.queue.closest_unexpanded();
        s.n_hops++;
        auto [owner, local_loc, valid] = resolve_vid(cur.vid, node_store, mapper);
        if (!valid) { s.resolve_fail++; return; }  // stays NEED_EXPAND; next node next round
        s.expand_owner = owner;
        if (owner == my_node_id) {
            s.local_nbr++;
            uint32_t degree = coloc_store->get_degree(local_loc);
            const uint64_t* nbrs = coloc_store->get_neighbors(local_loc);
            build_and_post(s, degree, nbrs);
        } else {
            auto* st = owner_state(owner);
            const auto* mr_ptr = node_store->remote_mr(owner);
            if (!st || !mr_ptr) { s.resolve_fail++; return; }
            char* slice = reinterpret_cast<char*>(st->fetch_buffer.data())
                          + static_cast<size_t>(s.idx) * slice_bytes;
            sgl.resize(1);
            sgl[0].local_addr = slice;
            sgl[0].length = nbr_read_sz;
            sgl[0].remote_offset = static_cast<uint64_t>(local_loc) * entry_size + degree_offset;
            uint32_t pend = st->conn->post_read_batch_async(sgl, *mr_ptr, st->fetch_buffer_mr, 0);
            s.remote_nbr++;
            s.rdma_entry_calls++; s.rdma_entry_items++; s.rdma_nbr_expand_calls++;
            if (pend == 0) {
                finish_expand(s);   // drained during post — data already present
            } else {
                owner_deque[owner].push_back({s.idx, pend});
                s.pending_segments = 1;
                s.phase = Phase::EXPAND_WAIT;
            }
        }
    };

    // ------------------------------------------------------------------
    // Drive loop
    // ------------------------------------------------------------------
    uint32_t active = 0;
    for (uint32_t q = 0; q < group_size; q++) {
        if (states[q].queue.has_unexpanded_node()) active++;
        else states[q].phase = Phase::DONE;
    }

    while (active > 0) {
        // ISSUE: advance every query that can post without blocking.
        for (uint32_t q = 0; q < group_size; q++) {
            QState& s = states[q];
            while (s.phase == Phase::NEED_EXPAND) {
                issue_expand(s);
                if (s.phase == Phase::DONE) { active--; break; }
                if (s.phase == Phase::NEED_EXPAND) continue;  // resolve-fail / all-local hop
                break;  // EXPAND_WAIT or VECTOR_WAIT
            }
        }

        // POLL: drain each owner's CQ and retire FIFO segments.
        for (uint32_t o = 0; o < MAX_OWNERS; o++) {
            if (owner_deque[o].empty()) continue;
            auto* st = owner_state(o);
            if (!st) { owner_deque[o].clear(); continue; }
            uint32_t n = st->conn->poll_read_nonblocking(16);
            while (n > 0 && !owner_deque[o].empty()) {
                auto& seg = owner_deque[o].front();
                uint32_t take = (n < seg.second) ? n : seg.second;
                seg.second -= take;
                n -= take;
                if (seg.second == 0) {
                    uint32_t qi = seg.first;
                    owner_deque[o].pop_front();
                    if (states[qi].pending_segments > 0) states[qi].pending_segments--;
                }
            }
        }

        // COMPLETE: advance queries whose outstanding reads have all arrived.
        for (uint32_t q = 0; q < group_size; q++) {
            QState& s = states[q];
            if (s.phase == Phase::EXPAND_WAIT && s.pending_segments == 0) {
                finish_expand(s);
            } else if (s.phase == Phase::VECTOR_WAIT && s.pending_segments == 0) {
                integrate(s);
                s.phase = Phase::NEED_EXPAND;
            }
        }
    }

    // ------------------------------------------------------------------
    // Extract per-query top-K (identical to search())
    // ------------------------------------------------------------------
    std::vector<SearchResult<T>> results(group_size);
    for (uint32_t q = 0; q < group_size; q++) {
        QState& s = states[q];
        SearchResult<T>& result = results[q];
        result.indices.resize(K);
        result.distances.resize(K);
        size_t pos = 0;
        uint64_t unresolved_vid_count = 0;
        for (size_t i = 0; i < s.queue.size() && pos < K; i++) {
            uint64_t vid = s.queue[i].vid;
            if (deleted_vids.count(vid)) continue;
            if (vid == frozen_canonical_vid) continue;
            uint32_t resolved_tag = 0;
            auto evi = s.edge_to_entry_vid.find(vid);
            if (evi != s.edge_to_entry_vid.end()) {
                resolved_tag = static_cast<uint32_t>(danns::decode_local_loc(evi->second));
            } else {
                uint32_t part = danns::decode_partition(vid);
                uint32_t ll = static_cast<uint32_t>(danns::decode_local_loc(vid));
                if (part < local_to_tag.size() && ll < local_to_tag[part].size() &&
                    local_to_tag[part][ll] > 0) {
                    resolved_tag = local_to_tag[part][ll];
                }
            }
            if (resolved_tag > 0 && deleted_tags.count(resolved_tag)) continue;
            result.indices[pos] = (resolved_tag > 0) ? (resolved_tag - 1) : UINT32_MAX;
            if (resolved_tag == 0) unresolved_vid_count++;
            result.distances[pos] = s.queue[i].distance;
            pos++;
        }
        for (size_t i = pos; i < K; i++) {
            result.indices[i] = UINT32_MAX;
            result.distances[i] = std::numeric_limits<float>::max();
        }
        result.result_count = static_cast<uint16_t>(pos);
        result.unresolved_vids = unresolved_vid_count;

        result.n_hops = s.n_hops;
        result.n_cmps = s.n_cmps;
        result.local_vector_accesses = s.local_vec;
        result.remote_vector_accesses = s.remote_vec;
        result.local_neighbor_accesses = s.local_nbr;
        result.remote_neighbor_accesses = s.remote_nbr;
        result.resolve_fail = s.resolve_fail;
        result.visited_count = static_cast<uint64_t>(s.visited.size());
        result.rdma_entry_calls = s.rdma_entry_calls;
        result.rdma_entry_items = s.rdma_entry_items;
        result.rdma_vec_fetch_calls = s.rdma_vec_fetch_calls;
        result.rdma_nbr_expand_calls = s.rdma_nbr_expand_calls;
        result.rdma_vec_only_fetches = s.rdma_vec_only_fetches;
    }

    return results;
}

// ============================================================================
// Explicit template instantiations
// ============================================================================
template SearchResult<float> ColocationBeamSearch::search<float>(
    const float*, size_t, size_t, diskann::Distance<float>*,
    const SearchConfig&, NodeStore*, PartitionMapper*, PQFilter*,
    std::map<uint32_t, std::mutex>*, int, int);

template SearchResult<int8_t> ColocationBeamSearch::search<int8_t>(
    const int8_t*, size_t, size_t, diskann::Distance<int8_t>*,
    const SearchConfig&, NodeStore*, PartitionMapper*, PQFilter*,
    std::map<uint32_t, std::mutex>*, int, int);

template SearchResult<uint8_t> ColocationBeamSearch::search<uint8_t>(
    const uint8_t*, size_t, size_t, diskann::Distance<uint8_t>*,
    const SearchConfig&, NodeStore*, PartitionMapper*, PQFilter*,
    std::map<uint32_t, std::mutex>*, int, int);

template std::vector<SearchResult<float>> ColocationBeamSearch::search_batch<float>(
    const float*, const uint32_t*, uint32_t, size_t, size_t, size_t,
    diskann::Distance<float>*, const SearchConfig&,
    NodeStore*, PartitionMapper*, PQFilter*, SearchTraceWriter*, int);

template std::vector<SearchResult<int8_t>> ColocationBeamSearch::search_batch<int8_t>(
    const int8_t*, const uint32_t*, uint32_t, size_t, size_t, size_t,
    diskann::Distance<int8_t>*, const SearchConfig&,
    NodeStore*, PartitionMapper*, PQFilter*, SearchTraceWriter*, int);

template std::vector<SearchResult<uint8_t>> ColocationBeamSearch::search_batch<uint8_t>(
    const uint8_t*, const uint32_t*, uint32_t, size_t, size_t, size_t,
    diskann::Distance<uint8_t>*, const SearchConfig&,
    NodeStore*, PartitionMapper*, PQFilter*, SearchTraceWriter*, int);

// search_interleaved explicit instantiations
template std::vector<SearchResult<float>> ColocationBeamSearch::search_interleaved<float>(
    const float* const*, const uint32_t*, uint32_t, size_t, size_t,
    diskann::Distance<float>*, const SearchConfig&,
    NodeStore*, PartitionMapper*, int, int);

template std::vector<SearchResult<int8_t>> ColocationBeamSearch::search_interleaved<int8_t>(
    const int8_t* const*, const uint32_t*, uint32_t, size_t, size_t,
    diskann::Distance<int8_t>*, const SearchConfig&,
    NodeStore*, PartitionMapper*, int, int);

template std::vector<SearchResult<uint8_t>> ColocationBeamSearch::search_interleaved<uint8_t>(
    const uint8_t* const*, const uint32_t*, uint32_t, size_t, size_t,
    diskann::Distance<uint8_t>*, const SearchConfig&,
    NodeStore*, PartitionMapper*, int, int);

// LITS filter explicit instantiations
template void ColocationBeamSearch::filter_lits_expanded_pool<float>(
    std::vector<std::pair<uint64_t, float>>&, uint32_t);
template void ColocationBeamSearch::filter_lits_expanded_pool<int8_t>(
    std::vector<std::pair<uint64_t, float>>&, uint32_t);
template void ColocationBeamSearch::filter_lits_expanded_pool<uint8_t>(
    std::vector<std::pair<uint64_t, float>>&, uint32_t);

} // namespace danns
