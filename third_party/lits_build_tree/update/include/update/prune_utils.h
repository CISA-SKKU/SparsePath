#pragma once
// update/prune_utils.h — Shared graph pruning utilities
//
// Extracted from consolidation.cpp to be reused by both
// Consolidation and Insertion modules.
//
// Functions:
//   robust_prune_internal<T>()  — DiskANN occlude_list algorithm
//   write_neighbors_internal()  — Write neighbor list to colocation entry

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/vector_id.h"
#include "store/colocation_store.h"

#include "distance.h"  // DiskANN Distance<T>

namespace danns {

// ========================================================================
// write_neighbors_internal: write neighbor VIDs + degree into local store
//
// Writes directly into the mmap'd colocation entry at local_loc.
// Pads unused slots with INVALID_VID.
//
// Write order matters for concurrent readers (search co-running with
// consolidation): neighbors + padding land first, degree is published
// LAST with release semantics. A reader that still sees the old (larger)
// degree walks new VIDs + INVALID_VID padding, which search skips —
// never a new degree over a half-written list.
// ========================================================================
inline void write_neighbors_internal(
    rdma::ColocationStore* store,
    const rdma::ColocationConfig& config,
    uint32_t local_loc,
    const uint64_t* vids,
    uint32_t degree)
{
    char* entry_base = const_cast<char*>(store->get_entry(local_loc));

    // Write valid neighbor VIDs
    uint32_t copy_count = std::min(degree, config.slack_R);
    if (copy_count > 0) {
        std::memcpy(entry_base + config.neighbor_offset,
                    vids, copy_count * sizeof(uint64_t));
    }

    // Pad remaining slots with INVALID_VID
    for (uint32_t i = copy_count; i < config.slack_R; i++) {
        std::memcpy(entry_base + config.neighbor_offset + i * sizeof(uint64_t),
                    &INVALID_VID, sizeof(uint64_t));
    }

    // Publish degree last (release: neighbor bytes visible before degree)
    char* deg_ptr = entry_base + config.degree_offset;
    if ((reinterpret_cast<uintptr_t>(deg_ptr) & 3) == 0) {
        __atomic_store_n(reinterpret_cast<uint32_t*>(deg_ptr), degree,
                         __ATOMIC_RELEASE);
    } else {
        // degree_offset = 8 + dim*element_size is 4-aligned for every current
        // dataset; unaligned fallback keeps single-writer correctness only.
        std::memcpy(deg_ptr, &degree, sizeof(uint32_t));
    }
}

// ========================================================================
// filter_lits_pool_inplace: LITS filter for the insert forward pool.
//
// Removes "long ∧ inter" candidates from the pool before RobustPrune:
//   long  : dist(p, v) > ρ_p, where ρ_p = mean distance over the pool.
//           The insert pool is the expanded "visited set" the beam search
//           returns (return_pool=true) and that robust_prune then consumes,
//           so this mean matches the reference DiskANN insert long threshold
//           (DiskANN src/index.cpp:1255-1262, "mean distance of pool (visited
//           set)").
//   inter : decode_partition(v) != my_partition (a cross-partition edge).
// to-start is intentionally omitted (long ∧ inter only), matching the
// existing search/consolidation LITS filters.
//
// Mirrors ColocationBeamSearch::filter_lits_expanded_pool (beam_search.cpp).
// Returns {checked, filtered}: checked = candidates examined (whole pool),
// filtered = candidates removed.
// ========================================================================
inline std::pair<uint64_t, uint64_t> filter_lits_pool_inplace(
    std::vector<std::pair<uint64_t, float>>& pool,
    uint32_t my_partition)
{
    if (pool.empty()) return {0, 0};

    double sum_dist = 0.0;
    for (const auto& vd : pool) sum_dist += vd.second;
    const float avg_dist = static_cast<float>(sum_dist / pool.size());

    const uint64_t checked = static_cast<uint64_t>(pool.size());
    uint64_t filtered = 0;
    pool.erase(
        std::remove_if(pool.begin(), pool.end(),
            [avg_dist, my_partition, &filtered](const std::pair<uint64_t, float>& p) {
                if ((decode_partition(p.first) != my_partition) &&
                    (p.second > avg_dist)) {
                    filtered++;
                    return true;
                }
                return false;
            }),
        pool.end());
    return {checked, filtered};
}

// ========================================================================
// robust_prune_internal: DiskANN RobustPrune (occlude_list) algorithm
//
// Selects up to R diverse neighbors from candidates using alpha-based
// occlusion filtering with geometric alpha progression (1.0 → *1.2 each round).
//
// Matches DiskANN index.cpp occlude_list() exactly:
//   occlude_factor[j] = max(occlude_factor[j], dist_j / dist_ij)
//   Selected when occlude_factor[i] <= cur_alpha
//
// Parameters:
//   node_vec   — query/insert point vector
//   candidates — (VID, distance) pairs, sorted by distance (modified in-place)
//   R          — max output degree
//   alpha      — max occlusion threshold (typically 1.2)
//   maxc       — max candidates to consider (truncate if larger)
//   dist_fn    — distance function (L2, cosine, etc.)
//   cmp_dim    — aligned dimension for SIMD distance
//   vec_store  — VID → vector pointer map (local=mmap ptr, remote=fetched buffer)
//
// Returns: vector of selected neighbor VIDs (size <= R)
// ========================================================================
template <typename T>
std::vector<uint64_t> robust_prune_internal(
    const T* node_vec,
    std::vector<std::pair<uint64_t, float>>& candidates,
    uint32_t R, float alpha, uint32_t maxc,
    diskann::Distance<T>* dist_fn, size_t cmp_dim,
    const std::unordered_map<uint64_t, const T*>& vec_store)
{
    if (candidates.size() > maxc)
        candidates.resize(maxc);

    size_t n = candidates.size();
    std::vector<float> occlude_factor(n, 0.0f);
    std::vector<bool> selected(n, false);
    std::vector<uint64_t> result;
    result.reserve(R);

    float cur_alpha = 1.0f;
    while (cur_alpha <= alpha + 1e-6f && result.size() < R) {
        for (size_t i = 0; i < n; i++) {
            if (result.size() >= R) break;
            if (selected[i]) continue;
            if (occlude_factor[i] > cur_alpha) continue;

            selected[i] = true;
            result.push_back(candidates[i].first);

            auto it_i = vec_store.find(candidates[i].first);
            if (it_i == vec_store.end()) continue;
            const T* vec_i = it_i->second;

            for (size_t j = i + 1; j < n; j++) {
                if (selected[j] || occlude_factor[j] > alpha) continue;
                auto it_j = vec_store.find(candidates[j].first);
                if (it_j == vec_store.end()) continue;
                const T* vec_j = it_j->second;

                float dist_ij = dist_fn->compare(vec_i, vec_j, cmp_dim);
                float dist_j = candidates[j].second;

                if (dist_ij == 0.0f) {
                    occlude_factor[j] = std::numeric_limits<float>::infinity();
                } else {
                    float factor = dist_j / dist_ij;
                    occlude_factor[j] = std::max(occlude_factor[j], factor);
                }
            }
        }
        cur_alpha *= 1.2f;
    }
    return result;
}

// ========================================================================
// PruneCandidate: candidate carrying its vec pointer inline.
// Avoids the per-pair vec_store hashmap lookup inside prune.
// ========================================================================
template <typename T>
struct PruneCandidate {
    uint64_t vid;
    float dist;
    const T* vec;
};

// ========================================================================
// robust_prune_internal — overload that takes PruneCandidate<T> with bound
// vec pointers. No vec_store map needed; prune reads candidates[i].vec
// directly.
// ========================================================================
template <typename T>
std::vector<uint64_t> robust_prune_internal(
    const T* node_vec,
    std::vector<PruneCandidate<T>>& candidates,
    uint32_t R, float alpha, uint32_t maxc,
    diskann::Distance<T>* dist_fn, size_t cmp_dim)
{
    if (candidates.size() > maxc)
        candidates.resize(maxc);

    size_t n = candidates.size();
    std::vector<float> occlude_factor(n, 0.0f);
    std::vector<bool> selected(n, false);
    std::vector<uint64_t> result;
    result.reserve(R);

    float cur_alpha = 1.0f;
    while (cur_alpha <= alpha + 1e-6f && result.size() < R) {
        for (size_t i = 0; i < n; i++) {
            if (result.size() >= R) break;
            if (selected[i]) continue;
            if (occlude_factor[i] > cur_alpha) continue;

            selected[i] = true;
            result.push_back(candidates[i].vid);

            const T* vec_i = candidates[i].vec;
            if (!vec_i) continue;

            for (size_t j = i + 1; j < n; j++) {
                if (selected[j] || occlude_factor[j] > alpha) continue;
                const T* vec_j = candidates[j].vec;
                if (!vec_j) continue;

                float dist_ij = dist_fn->compare(vec_i, vec_j, cmp_dim);
                float dist_j = candidates[j].dist;

                if (dist_ij == 0.0f) {
                    occlude_factor[j] = std::numeric_limits<float>::infinity();
                } else {
                    float factor = dist_j / dist_ij;
                    occlude_factor[j] = std::max(occlude_factor[j], factor);
                }
            }
        }
        cur_alpha *= 1.2f;
    }
    return result;
}

} // namespace danns
