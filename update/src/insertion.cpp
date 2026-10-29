// update/src/insertion.cpp — Distributed insertion engine
//
// Implements DiskANN insert_point() algorithm adapted for distributed
// colocation layout with RDMA.
//
// Reference: third_party/diskann/src/index.cpp
//   insert_point:   L5523-5683
//   occlude_list:   L2109-2202  (= robust_prune_internal in prune_utils.h)
//   inter_insert:   L2613-2697  (= distributed_inter_insert below)

#include "update/insertion.h"
#include "update/load_balancer.h"
#include "update/prune_utils.h"
#include "store/node_store.h"
#include "store/redirect_alloc.h"
#include "cluster/partition_mapper.h"
#include "search/beam_search.h"
#include "search/search_config.h"
#include "search/search_result.h"
#include "search/pq_filter.h"
#include "core/vector_id.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <map>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace danns {

// ========================================================================
// insert_single: DiskANN insert_point() distributed version
//
// Algorithm:
//   1. Write vector + entry_vid to local colocation entry
//   2. Beam search from frozen point → candidate pool
//   3. Remove self-loop and special VIDs from pool
//   4. Fetch candidate vectors (local=mmap, remote=RDMA)
//   5. RobustPrune → forward edges
//   6. Write forward edges
//   7. distributed_inter_insert → reverse edges
// ========================================================================
template <typename T>
InsertionResult Insertion::insert_single(
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
    int thread_id)
{
    InsertionResult result;
    result.local_loc = local_loc;
    result.tag = tag;
    result.lb_m_assign = my_partition;

    auto* coloc_store = node_store->local_store();
    const auto& coloc_config = node_store->config();
    // Compare over the real vector_dim, not aligned_dim. The colocation entry's
    // vector field is exactly vector_dim*element_size wide (degree immediately
    // after), so aligned_dim (= ROUND_UP(dim,8)) reads past it into degree bytes
    // when dim%8!=0 (spacev: 100 vs 104). Search already uses vector_dim_; this
    // makes insert consistent. deep/sift/laion (dim%8==0) are byte-identical.
    const size_t cmp_dim = static_cast<size_t>(vector_dim);
    const uint64_t my_vid = encode_vector_id(my_partition, local_loc);

    // ------------------------------------------------------------------
    // Step 1: Write vector + entry_vid to local colocation entry
    // ------------------------------------------------------------------
    auto t_write_start = std::chrono::high_resolution_clock::now();
    {
        char* entry = coloc_store->get_entry_mut(local_loc);

        // Write entry_vid at offset 0 (partition|tag encoding for identity)
        uint64_t entry_vid = encode_vector_id(my_partition,
                                              static_cast<uint64_t>(tag));
        std::memcpy(entry + coloc_config.vector_id_offset,
                    &entry_vid, sizeof(uint64_t));

        // Write vector data
        std::memcpy(entry + coloc_config.vector_offset,
                    vector, vector_dim * coloc_config.element_size);

        // Initialize degree = 0 and pad neighbors with INVALID_VID
        uint32_t zero_degree = 0;
        std::memcpy(entry + coloc_config.degree_offset,
                    &zero_degree, sizeof(uint32_t));
        for (uint32_t i = 0; i < coloc_config.slack_R; i++) {
            std::memcpy(entry + coloc_config.neighbor_offset + i * sizeof(uint64_t),
                        &INVALID_VID, sizeof(uint64_t));
        }
    }
    auto t_write_end = std::chrono::high_resolution_clock::now();
    result.write_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        t_write_end - t_write_start).count();

    // ------------------------------------------------------------------
    // Step 2: Beam search from frozen point → candidate pool
    //
    // DiskANN: search_for_point_and_prune → iterate_to_fixed_point
    // We use the existing ColocationBeamSearch with return_pool=true.
    // ------------------------------------------------------------------
    auto t_search_start = std::chrono::high_resolution_clock::now();

    SearchConfig search_cfg;
    search_cfg.K = config.L;          // Use L as K to get enough candidates
    search_cfg.L = config.L;
    search_cfg.entry_mode = config.entry_mode;  // FROZEN_ONLY = 1
    search_cfg.query_partition = my_partition;
    search_cfg.return_pool = true;    // Return full candidate pool for pruning
    search_cfg.skip_timing = true;    // Skip per-query timing overhead
    search_cfg.need_result_tags = false;  // insert uses only .pool; skip the
                                          // top-K tag conversion + its map build
    // FrozenSeedCache is a SEARCH-side optimization only. Its containment rule
    // is "one cache per search op, dropped at the op boundary"; the insert path
    // has no such boundary — it IS the mutation that can hand a cached
    // local_loc to a different vector. Pinned off so insert keeps zero coupling
    // to the cache and its neighbor discovery stays bit-identical to pre-patch.
    search_cfg.frozen_seed_cache = false;

    // Use the inserted vector as query
    SearchResult<T> search_result = beam_search->search<T>(
        vector, aligned_dim, cmp_dim, dist_fn,
        search_cfg, node_store, mapper, pq_filter,
        &fetcher_mutexes,
        static_cast<int>(config.num_threads),  // effective batch thread count
        thread_id);

    auto t_search_end = std::chrono::high_resolution_clock::now();
    result.search_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        t_search_end - t_search_start).count();

    // ------------------------------------------------------------------
    // Step 3: Filter pool — remove self, special VIDs, deleted VIDs
    // ------------------------------------------------------------------
    auto& pool = search_result.pool;

    // Sort by distance (beam search pool may not be fully sorted)
    std::sort(pool.begin(), pool.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; });

    // Remove self-loop, special VIDs (frozen/centroid), deleted VIDs
    pool.erase(
        std::remove_if(pool.begin(), pool.end(),
            [&](const std::pair<uint64_t, float>& p) {
                return p.first == my_vid ||
                       p.first == INVALID_VID ||
                       special_vids.count(p.first) ||
                       mapper->is_vid_deleted(p.first);
            }),
        pool.end());

    // ------------------------------------------------------------------
    // Step 3a: LoadBalancer. Histogram runs BEFORE the LITS erase (LITS
    // removes long∧inter members that still carry locality signal).
    //   measure-only      : verdict recorded, insert proceeds on m_search;
    //                       counters come from the batch-start snapshot.
    //   live redirect     : counters are read LIVE (my atomic / peer 8-byte
    //                       RDMA READ per distinct candidate owner) and a
    //                       would-redirect point acquires its final slot on
    //                       m_assign NOW via the alloc server. The rest of
    //                       this insert then targets that home (LITS basis,
    //                       entry placement, backlinks) while the prune
    //                       still runs here on m_search where the candidate
    //                       vectors are cheapest to fetch (spec §6).
    // ------------------------------------------------------------------
    bool redirected = false;
    uint32_t final_partition = my_partition;
    uint32_t final_loc = local_loc;
    uint64_t final_vid = my_vid;
    if (config.load_balance) {
        const uint32_t my_node = node_store->node_id();
        auto counter_of = [&](uint32_t n) -> std::optional<uint64_t> {
            if (config.redirect) {
                if (n == my_node) return node_store->local_vector_count();
                // LIVE read for in-batch decisions. The counter QP/CQ is the
                // SAME endpoint as this peer's ColocationFetcher — all
                // shared-endpoint users must hold the same per-peer lock
                // (fetcher_mutexes), or completions get cross-consumed.
                // Lock order fetcher→slot is acyclic (slot.mtx is a leaf).
                try {
                    std::lock_guard<std::mutex> lk(fetcher_mutexes[n]);
                    return node_store->read_peer_vector_count(n);
                } catch (const std::exception&) {
                    return std::nullopt;  // unobservable — cannot bid
                }
            }
            if (!lb_counters) return std::nullopt;
            auto it = lb_counters->find(n);
            if (it == lb_counters->end()) return std::nullopt;
            return it->second;
        };
        auto lb = compute_load_balance_assignment(
            pool, my_partition, config.lb_threshold,
            [mapper](uint32_t p) { return mapper->partition_id_to_node_id(p); },
            counter_of);
        result.lb_m_assign = lb.m_assign;
        result.redirect_considered = lb.considered;
        result.redirect_would_happen = lb.would_redirect;

        if (config.redirect && lb.would_redirect && redirect_alloc) {
            const uint32_t owner_node =
                mapper->partition_id_to_node_id(lb.m_assign);
            uint32_t floc = RedirectAlloc::ALLOC_FAIL;
            if (node_store->fetcher(owner_node) &&
                redirect_alloc->has_peer(owner_node)) {
                floc = redirect_alloc->request_slot(
                    owner_node, tag, static_cast<int>(config.redirect_timeout_ms));
            }
            if (floc != RedirectAlloc::ALLOC_FAIL) {
                redirected = true;
                final_partition = lb.m_assign;
                final_loc = floc;
                final_vid = encode_vector_id(final_partition, floc);
                // Local FAA: the alloc server thread may be incrementing
                // this same counter for grants-in concurrently.
                node_store->counter_fetch_sub(1);
                // Own replicated tables: retract the provisional mapping,
                // record the final one. Arrays pre-sized at window open —
                // in-place stores only (bounds checks are defensive).
                auto& t2l = node_store->tag_to_local_mut();
                auto& l2t = node_store->local_to_tag_mut();
                if (tag < t2l[my_partition].size())
                    t2l[my_partition][tag] = UINT32_MAX;
                if (local_loc < l2t[my_partition].size())
                    l2t[my_partition][local_loc] = 0;
                if (tag < t2l[final_partition].size())
                    t2l[final_partition][tag] = floc;
                if (final_partition < l2t.size() &&
                    floc < l2t[final_partition].size())
                    l2t[final_partition][floc] = tag;
                result.redirected = true;
                result.lb_m_assign = final_partition;
                result.redirect_final_loc = floc;
            } else {
                result.redirect_fallback = true;
            }
        }
    }

    // If anything between slot acquisition and backlink completion throws
    // (RDMA fetch/write errors), revert the local redirect bookkeeping so
    // settlement sees a consistent non-redirected point. For throws BEFORE
    // the full-entry write (Step 6) the orphaned grant is reclaimed by the
    // adopt-side entry_vid verification. KNOWN GAP (failure-path only): a
    // throw BETWEEN the Step 6 write and Step 7 completion leaves a valid
    // entry_vid on m_assign — the target adopts it while this revert
    // re-publishes the provisional mapping, i.e. a mid-op insert failure
    // can surface as multi_home_tags > 0. Such a failure already fails the
    // run's gates; narrowing the window needs redirected-state reporting
    // from the catch path (deferred — see the 2026-06-12 redirect-results
    // notes).
    struct RedirectRevert {
        bool armed;
        NodeStore* ns;
        uint32_t my_part, tag, old_loc, fpart, floc;
        ~RedirectRevert() {
            if (!armed) return;
            ns->counter_fetch_add(1);
            auto& t2l = ns->tag_to_local_mut();
            auto& l2t = ns->local_to_tag_mut();
            if (tag < t2l[my_part].size()) t2l[my_part][tag] = old_loc;
            if (old_loc < l2t[my_part].size()) l2t[my_part][old_loc] = tag;
            if (tag < t2l[fpart].size()) t2l[fpart][tag] = UINT32_MAX;
            if (fpart < l2t.size() && floc < l2t[fpart].size())
                l2t[fpart][floc] = 0;
        }
    } redirect_revert{redirected, node_store, my_partition, tag,
                      local_loc, final_partition, final_loc};

    // ------------------------------------------------------------------
    // Step 3b: LITS filter — drop long ∧ inter candidates before prune.
    //
    // ρ_p = mean pool distance (the pool is the expanded "visited set" the
    // beam search returned and that robust_prune consumes below — same mean
    // as the reference DiskANN insert long threshold). Runs BEFORE the
    // vector-fetch loop (Step 4) so filtered cross-partition candidates are
    // never fetched. inter is judged against the FINAL owner (spec D-1):
    // m_assign when this point was live-redirected, else config.lits_my_partition.
    // ------------------------------------------------------------------
    if (config.lits_filter) {
        auto lits_counts = filter_lits_pool_inplace(
            pool, redirected ? final_partition : config.lits_my_partition);
        result.lits_checked = lits_counts.first;
        result.lits_filtered = lits_counts.second;
    }

    if (pool.empty() && !redirected) {
        // No candidates found — entry has no forward edges
        return result;
    }
    // (a redirected point falls through even with an empty pool: its final
    // entry on m_assign must still be written, else the published mapping
    // would point at a never-written slot)

    // ------------------------------------------------------------------
    // Step 4: Fetch candidate vectors for RobustPrune
    //
    // Build vec_store: VID → vector pointer
    //   - Local: zero-copy from mmap (coloc_store->get_vector)
    //   - Remote: RDMA read via fetcher->fetch_vector_only
    // ------------------------------------------------------------------
    auto t_prune_start = std::chrono::high_resolution_clock::now();

    std::unordered_map<uint64_t, const T*> vec_store;
    // Temporary buffer for remote vectors (keep alive until prune completes)
    std::vector<std::vector<T>> remote_vec_buffers;

    for (const auto& [vid, dist] : pool) {
        uint32_t owner_partition = decode_partition(vid);
        uint32_t nbr_local_loc = static_cast<uint32_t>(decode_local_loc(vid));
        uint32_t owner_node = mapper->partition_id_to_node_id(owner_partition);

        if (owner_node == node_store->node_id()) {
            // Local: zero-copy pointer
            vec_store[vid] = static_cast<const T*>(
                coloc_store->get_vector(nbr_local_loc));
            result.local_vec_reads++;
        } else {
            // Remote: RDMA fetch
            auto* fetcher = node_store->fetcher(owner_node);
            if (!fetcher) continue;

            // Copy to owned buffer (fetcher buffer may be reused). The
            // memcpy must stay INSIDE the lock: another thread's fetch to
            // the same peer would overwrite the shared fetcher buffer.
            remote_vec_buffers.emplace_back(aligned_dim);
            {
                std::lock_guard<std::mutex> lock(fetcher_mutexes[owner_node]);
                const void* remote_vec =
                    fetcher->fetch_vector_only(nbr_local_loc);
                std::memcpy(remote_vec_buffers.back().data(), remote_vec,
                            vector_dim * coloc_config.element_size);
            }
            vec_store[vid] = remote_vec_buffers.back().data();
            result.remote_vec_reads++;
        }
    }

    // ------------------------------------------------------------------
    // Step 5: RobustPrune → forward edges
    //
    // DiskANN: prune_neighbors → occlude_list
    // Uses shared robust_prune_internal from prune_utils.h
    // ------------------------------------------------------------------
    std::vector<uint64_t> pruned_vids = robust_prune_internal<T>(
        vector, pool, config.R, config.alpha, config.maxc,
        dist_fn, cmp_dim, vec_store);

    auto t_prune_end = std::chrono::high_resolution_clock::now();
    result.prune_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        t_prune_end - t_prune_start).count();

    // ------------------------------------------------------------------
    // Step 6: Write forward edges to the colocation entry.
    //   normal       : in-place into my local provisional entry.
    //   live redirect: assemble the COMPLETE entry (identity vid = encode
    //                  (partition, tag), vector, degree, padded neighbors)
    //                  and place it in its final home on m_assign with one
    //                  RDMA WRITE. The local provisional entry stays as
    //                  unreachable residue (settlement reclaims the slot).
    // ------------------------------------------------------------------
    result.forward_degree = static_cast<uint32_t>(pruned_vids.size());
    if (!redirected) {
        write_neighbors_internal(coloc_store, coloc_config, local_loc,
                                 pruned_vids.data(), result.forward_degree);
        result.local_nbr_writes++;
    } else {
        std::vector<char> entry(coloc_config.entry_size, 0);
        const uint64_t entry_vid =
            encode_vector_id(final_partition, static_cast<uint64_t>(tag));
        std::memcpy(entry.data() + coloc_config.vector_id_offset,
                    &entry_vid, sizeof(uint64_t));
        std::memcpy(entry.data() + coloc_config.vector_offset,
                    vector, vector_dim * coloc_config.element_size);
        std::memcpy(entry.data() + coloc_config.degree_offset,
                    &result.forward_degree, sizeof(uint32_t));
        char* nbr_base = entry.data() + coloc_config.neighbor_offset;
        std::memcpy(nbr_base, pruned_vids.data(),
                    result.forward_degree * sizeof(uint64_t));
        for (uint32_t i = result.forward_degree; i < coloc_config.slack_R; i++) {
            std::memcpy(nbr_base + i * sizeof(uint64_t),
                        &INVALID_VID, sizeof(uint64_t));
        }

        const uint32_t owner_node =
            mapper->partition_id_to_node_id(final_partition);
        auto* fetcher = node_store->fetcher(owner_node);  // non-null: gated above
        std::lock_guard<std::mutex> lock(fetcher_mutexes[owner_node]);
        fetcher->write_full_entry(final_loc, entry.data());
        result.remote_nbr_writes++;
    }

    // ------------------------------------------------------------------
    // Step 7: Distributed inter-insert → reverse edges
    //
    // DiskANN: inter_insert(location, pruned_list, scratch)
    // Backlinks point at the FINAL home (final_vid = m_assign when redirected).
    // ------------------------------------------------------------------
    auto t_reverse_start = std::chrono::high_resolution_clock::now();

    distributed_inter_insert<T>(
        final_vid, vector, pruned_vids, config,
        node_store, mapper, dist_fn, cmp_dim, my_partition,
        special_vids, fetcher_mutexes, local_locks, result);

    auto t_reverse_end = std::chrono::high_resolution_clock::now();
    result.reverse_edge_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        t_reverse_end - t_reverse_start).count();

    redirect_revert.armed = false;  // redirect fully committed
    return result;
}

// ========================================================================
// distributed_inter_insert: DiskANN inter_insert() distributed version
//
// For each forward neighbor:
//   - Check if new_vid already in neighbor's list → skip
//   - If room (degree < slack_R): append new_vid → simple write
//   - If full: fetch all neighbor vectors, build expanded pool,
//     RobustPrune → write back pruned list
//
// DiskANN reference: index.cpp inter_insert L2613-2697
//   Slack check: des_pool.size() < GRAPH_SLACK_FACTOR * range
//   Our version: degree < slack_R (slack_R = ceil(1.3 * R))
// ========================================================================
template <typename T>
void Insertion::distributed_inter_insert(
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
    InsertionResult& result)
{
    auto* coloc_store = node_store->local_store();
    const auto& coloc_config = node_store->config();
    const uint32_t my_node = node_store->node_id();

    for (uint64_t nbr_vid : pruned_vids) {
        // Skip special entries (frozen/centroid) — they are read-only
        if (special_vids.count(nbr_vid)) continue;
        // Skip deleted
        if (mapper->is_vid_deleted(nbr_vid)) continue;

        uint32_t nbr_partition = decode_partition(nbr_vid);
        uint32_t nbr_local_loc = static_cast<uint32_t>(decode_local_loc(nbr_vid));
        uint32_t nbr_owner_node = mapper->partition_id_to_node_id(nbr_partition);

        bool is_local = (nbr_owner_node == my_node);

        // --- Read-modify-write neighbor list ---
        // Local neighbors: hold shard lock for the full RMW to prevent concurrent
        // threads from clobbering each other's reverse-edge writes (same pattern as
        // DiskANN's per-node _locks[]). Remote neighbors: fetcher_mutex serializes
        // the RDMA sequence; cross-thread remote conflicts are accepted (same as
        // DiskANN parallel build — one reverse edge may be lost, graph quality OK).
        const size_t kLockStripes = local_locks.size();

        if (is_local) {
            std::lock_guard<std::mutex> node_lk(
                local_locks[nbr_local_loc % kLockStripes]);

            uint32_t nbr_degree = coloc_store->get_degree(nbr_local_loc);
            const uint64_t* nbrs_raw = coloc_store->get_neighbors(nbr_local_loc);
            std::vector<uint64_t> nbr_list(nbrs_raw, nbrs_raw + nbr_degree);
            result.local_nbr_reads++;

            bool already_linked = false;
            for (uint32_t i = 0; i < nbr_degree; i++) {
                if (nbr_list[i] == new_vid) { already_linked = true; break; }
            }
            if (already_linked) continue;

            if (nbr_degree < config.slack_R) {
                nbr_list.push_back(new_vid);
                nbr_degree++;
                write_neighbors_internal(coloc_store, coloc_config,
                                         nbr_local_loc,
                                         nbr_list.data(), nbr_degree);
                result.local_nbr_writes++;
                result.reverse_appends++;
            } else {
                // Full → need RobustPrune
                const T* nbr_vec = static_cast<const T*>(
                    coloc_store->get_vector(nbr_local_loc));
                result.local_vec_reads++;

                std::vector<std::pair<uint64_t, float>> candidates;
                candidates.reserve(nbr_degree + 1);
                float new_dist = dist_fn->compare(nbr_vec, new_vec, cmp_dim);
                candidates.push_back({new_vid, new_dist});

                std::unordered_map<uint64_t, const T*> prune_vec_store;
                std::vector<std::vector<T>> remote_bufs;
                prune_vec_store[new_vid] = new_vec;

                for (uint32_t i = 0; i < nbr_degree; i++) {
                    uint64_t existing_vid = nbr_list[i];
                    if (existing_vid == INVALID_VID) continue;
                    if (mapper->is_vid_deleted(existing_vid)) continue;
                    uint32_t ex_partition = decode_partition(existing_vid);
                    uint32_t ex_local_loc = static_cast<uint32_t>(decode_local_loc(existing_vid));
                    uint32_t ex_owner = mapper->partition_id_to_node_id(ex_partition);
                    const T* ex_vec = nullptr;
                    if (ex_owner == my_node) {
                        ex_vec = static_cast<const T*>(coloc_store->get_vector(ex_local_loc));
                        result.local_vec_reads++;
                    } else {
                        auto* fetcher = node_store->fetcher(ex_owner);
                        if (!fetcher) continue;
                        remote_bufs.emplace_back(cmp_dim);
                        {
                            std::lock_guard<std::mutex> lk(fetcher_mutexes[ex_owner]);
                            const void* rv = fetcher->fetch_vector_only(ex_local_loc);
                            std::memcpy(remote_bufs.back().data(), rv,
                                        coloc_config.dim * coloc_config.element_size);
                        }
                        ex_vec = remote_bufs.back().data();
                        result.remote_vec_reads++;
                    }
                    float ex_dist = dist_fn->compare(nbr_vec, ex_vec, cmp_dim);
                    prune_vec_store[existing_vid] = ex_vec;
                    candidates.push_back({existing_vid, ex_dist});
                }

                std::sort(candidates.begin(), candidates.end(),
                    [](const auto& a, const auto& b) { return a.second < b.second; });
                std::vector<uint64_t> new_nbrs = robust_prune_internal<T>(
                    nbr_vec, candidates, config.R, config.alpha, config.maxc,
                    dist_fn, cmp_dim, prune_vec_store);
                uint32_t new_degree = static_cast<uint32_t>(new_nbrs.size());
                write_neighbors_internal(coloc_store, coloc_config,
                                         nbr_local_loc,
                                         new_nbrs.data(), new_degree);
                result.local_nbr_writes++;
                result.reverse_prunes++;
            }
            continue;  // handled local path fully
        }

        // --- Remote neighbor path ---
        uint32_t nbr_degree = 0;
        std::vector<uint64_t> nbr_list;
        {
            auto* fetcher = node_store->fetcher(nbr_owner_node);
            if (!fetcher) continue;

            std::lock_guard<std::mutex> lock(fetcher_mutexes[nbr_owner_node]);
            const uint64_t* nbrs = fetcher->fetch_neighbors_only(nbr_local_loc);
            nbr_degree = fetcher->fetched_degree();
            nbr_list.assign(nbrs, nbrs + nbr_degree);
            result.remote_nbr_reads++;
        }

        // --- Check if already connected ---
        bool already_linked = false;
        for (uint32_t i = 0; i < nbr_degree; i++) {
            if (nbr_list[i] == new_vid) {
                already_linked = true;
                break;
            }
        }
        if (already_linked) continue;

        // --- Decide: append or prune ---
        if (nbr_degree < config.slack_R) {
            // Room available → simple append
            nbr_list.push_back(new_vid);
            nbr_degree++;

            {
                auto* fetcher = node_store->fetcher(nbr_owner_node);
                std::lock_guard<std::mutex> lock(fetcher_mutexes[nbr_owner_node]);
                fetcher->write_neighbors(nbr_local_loc, nbr_degree,
                                         nbr_list.data());
                result.remote_nbr_writes++;
            }
            result.reverse_appends++;
        } else {
            // Full → need RobustPrune on remote neighbor
            std::vector<T> nbr_vec_buf(cmp_dim);
            const T* nbr_vec = nullptr;
            {
                auto* fetcher = node_store->fetcher(nbr_owner_node);
                std::lock_guard<std::mutex> lock(fetcher_mutexes[nbr_owner_node]);
                const void* rv = fetcher->fetch_vector_only(nbr_local_loc);
                std::memcpy(nbr_vec_buf.data(), rv,
                            coloc_config.dim * coloc_config.element_size);
            }
            nbr_vec = nbr_vec_buf.data();
            result.remote_vec_reads++;

            std::vector<std::pair<uint64_t, float>> candidates;
            candidates.reserve(nbr_degree + 1);
            float new_dist = dist_fn->compare(nbr_vec, new_vec, cmp_dim);
            candidates.push_back({new_vid, new_dist});

            std::unordered_map<uint64_t, const T*> prune_vec_store;
            std::vector<std::vector<T>> remote_bufs;
            prune_vec_store[new_vid] = new_vec;

            for (uint32_t i = 0; i < nbr_degree; i++) {
                uint64_t existing_vid = nbr_list[i];
                if (existing_vid == INVALID_VID) continue;
                if (mapper->is_vid_deleted(existing_vid)) continue;
                uint32_t ex_partition = decode_partition(existing_vid);
                uint32_t ex_local_loc = static_cast<uint32_t>(decode_local_loc(existing_vid));
                uint32_t ex_owner = mapper->partition_id_to_node_id(ex_partition);
                const T* ex_vec = nullptr;
                if (ex_owner == my_node) {
                    ex_vec = static_cast<const T*>(coloc_store->get_vector(ex_local_loc));
                    result.local_vec_reads++;
                } else {
                    auto* fetcher = node_store->fetcher(ex_owner);
                    if (!fetcher) continue;
                    remote_bufs.emplace_back(cmp_dim);
                    {
                        std::lock_guard<std::mutex> lock(fetcher_mutexes[ex_owner]);
                        const void* rv = fetcher->fetch_vector_only(ex_local_loc);
                        std::memcpy(remote_bufs.back().data(), rv,
                                    coloc_config.dim * coloc_config.element_size);
                    }
                    ex_vec = remote_bufs.back().data();
                    result.remote_vec_reads++;
                }
                float ex_dist = dist_fn->compare(nbr_vec, ex_vec, cmp_dim);
                prune_vec_store[existing_vid] = ex_vec;
                candidates.push_back({existing_vid, ex_dist});
            }

            std::sort(candidates.begin(), candidates.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; });
            std::vector<uint64_t> new_nbrs = robust_prune_internal<T>(
                nbr_vec, candidates, config.R, config.alpha, config.maxc,
                dist_fn, cmp_dim, prune_vec_store);

            uint32_t new_degree = static_cast<uint32_t>(new_nbrs.size());
            {
                auto* fetcher = node_store->fetcher(nbr_owner_node);
                std::lock_guard<std::mutex> lock(fetcher_mutexes[nbr_owner_node]);
                fetcher->write_neighbors(nbr_local_loc, new_degree,
                                         new_nbrs.data());
                result.remote_nbr_writes++;
            }

            result.reverse_prunes++;
        }
    }
}

// ========================================================================
// insert_batch: main entry point
//
// Iterates over N vectors, calling insert_single for each.
// Tag/loc mappings must be registered in PartitionMapper before calling.
// ========================================================================
template <typename T>
InsertionStats Insertion::insert_batch(
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
    RedirectAlloc* redirect_alloc)
{
    InsertionStats stats;
    if (count == 0) return stats;

    // Validate
    if (!node_store || !node_store->local_store()) {
        throw std::runtime_error("insert_batch: NodeStore not initialized");
    }
    if (!dist_fn) {
        throw std::runtime_error("insert_batch: no distance function");
    }
    if (!beam_search) {
        throw std::runtime_error("insert_batch: no beam search engine");
    }

    auto t_start = std::chrono::high_resolution_clock::now();

    const uint32_t num_threads = std::max(1u, config.num_threads);

    // Pre-populate per-peer fetcher mutexes before the parallel region.
    // Parallel threads share this map (read-only structure after init);
    // each mutex serializes RDMA ops to its peer. The self key is added so
    // the live-redirect counter_of lambda can index fetcher_mutexes[n]
    // uniformly (std::map::operator[] is a structural mutation — never run
    // it inside the parallel region).
    std::map<uint32_t, std::mutex> fetcher_mutexes;
    fetcher_mutexes[node_store->node_id()];  // default-construct mutex
    for (uint32_t peer_id : node_store->peer_ids()) {
        fetcher_mutexes[peer_id];  // default-construct mutex
    }

    // Shard locks for local graph node write protection (one per stripe bucket).
    // Same role as DiskANN's _locks[]: prevents two threads from clobbering the
    // same local neighbor list during distributed_inter_insert.
    static constexpr uint32_t kLockStripes = 8192;
    std::vector<std::mutex> local_locks(kLockStripes);

    // LoadBalancer counter snapshot at batch start. In measure-only mode it
    // is also the decision source (counters are static during the batch);
    // in live-redirect mode decisions read live counters per insert and this
    // snapshot is kept for reporting only.
    std::unordered_map<uint32_t, uint64_t> lb_counters;
    if (config.load_balance) {
        lb_counters[node_store->node_id()] = node_store->local_vector_count();
        for (uint32_t peer : node_store->counter_peer_ids()) {
            try {
                lb_counters[peer] = node_store->read_peer_vector_count(peer);
            } catch (const std::exception& e) {
                // Peer drops out of LB candidacy (cannot bid without a
                // counter); the insert itself is unaffected.
                fprintf(stderr,
                        "[insert_batch] counter read from peer %u failed: %s\n",
                        peer, e.what());
            }
        }
        stats.lb_counter_snapshot.insert(lb_counters.begin(), lb_counters.end());
    }

#ifdef _OPENMP
    if (num_threads > 1) omp_set_num_threads(static_cast<int>(num_threads));
#endif

    // Per-thread private stats, merged once after the region (uniform for
    // scalars + vectors, no shared writes, no false sharing).
    std::vector<InsertionStats> local(num_threads);
    for (auto& ls : local) {
        ls.per_vector.reserve(count / num_threads + 1);
    }

    #pragma omp parallel if(num_threads > 1) num_threads(static_cast<int>(num_threads))
    {
        #pragma omp for schedule(dynamic, 1)
        for (uint32_t i = 0; i < count; i++) {
            const int tid = (num_threads > 1) ? omp_get_thread_num() : 0;
            InsertionStats& ls = local[tid];

            // Caller packs the input as (count, vector_dim) — stride by
            // vector_dim, not aligned_dim. With aligned_dim>vector_dim (spacev
            // 104>100), row i>=1 was mis-offset by +4*i elements -> wrong
            // vectors inserted. deep/sift/laion (aligned==vector) identical.
            const T* vec = vectors + static_cast<size_t>(i) * vector_dim;
            try {
                InsertionResult r = insert_single<T>(
                    config, node_store, mapper, beam_search, pq_filter,
                    dist_fn, vector_dim, aligned_dim,
                    vec, tags[i], local_locs[i], my_partition,
                    special_vids, fetcher_mutexes, local_locks,
                    config.load_balance ? &lb_counters : nullptr,
                    redirect_alloc, tid);

                ls.inserted++;
                if (r.redirected) {
                    ls.redirected.push_back(InsertionStats::RedirectOut{
                        r.tag, local_locs[i],
                        r.lb_m_assign, r.redirect_final_loc});
                }
                if (r.redirect_fallback) ls.redirect_fallback++;
                ls.search_sec  += r.search_ns / 1e9;
                ls.prune_sec   += r.prune_ns / 1e9;
                ls.write_sec   += r.write_ns / 1e9;
                ls.reverse_sec += r.reverse_edge_ns / 1e9;
                ls.local_vec_reads  += r.local_vec_reads;
                ls.remote_vec_reads += r.remote_vec_reads;
                ls.local_nbr_reads  += r.local_nbr_reads;
                ls.remote_nbr_reads += r.remote_nbr_reads;
                ls.local_nbr_writes += r.local_nbr_writes;
                ls.remote_nbr_writes += r.remote_nbr_writes;
                ls.lits_checked  += r.lits_checked;
                ls.lits_filtered += r.lits_filtered;
                ls.redirect_considered   += r.redirect_considered ? 1 : 0;
                ls.redirect_would_happen += r.redirect_would_happen ? 1 : 0;
                ls.per_vector.push_back(std::move(r));
            } catch (const std::exception& e) {
                // Exceptions must not escape an OpenMP region — caught per
                // iteration (same policy as the sequential loop).
                ls.failed++;
                fprintf(stderr, "[insert_batch] vector %u (tag=%u) failed: %s\n",
                        i, tags[i], e.what());
            }
        }
    }

    for (auto& ls : local) {
        stats.merge(std::move(ls));
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    stats.elapsed_sec = std::chrono::duration<double>(t_end - t_start).count();

    return stats;
}

// ========================================================================
// Explicit template instantiations
// ========================================================================
template InsertionStats Insertion::insert_batch<float>(
    const InsertionConfig&, NodeStore*, PartitionMapper*,
    ColocationBeamSearch*, PQFilter*, diskann::Distance<float>*,
    uint32_t, uint32_t, const float*, const uint32_t*, const uint32_t*,
    uint32_t, uint32_t, const std::unordered_set<uint64_t>&, RedirectAlloc*);

template InsertionStats Insertion::insert_batch<int8_t>(
    const InsertionConfig&, NodeStore*, PartitionMapper*,
    ColocationBeamSearch*, PQFilter*, diskann::Distance<int8_t>*,
    uint32_t, uint32_t, const int8_t*, const uint32_t*, const uint32_t*,
    uint32_t, uint32_t, const std::unordered_set<uint64_t>&, RedirectAlloc*);

template InsertionStats Insertion::insert_batch<uint8_t>(
    const InsertionConfig&, NodeStore*, PartitionMapper*,
    ColocationBeamSearch*, PQFilter*, diskann::Distance<uint8_t>*,
    uint32_t, uint32_t, const uint8_t*, const uint32_t*, const uint32_t*,
    uint32_t, uint32_t, const std::unordered_set<uint64_t>&, RedirectAlloc*);

} // namespace danns
