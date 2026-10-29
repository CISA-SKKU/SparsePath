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
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(USE_AVX2) && defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define DANNS_PRUNE_SIMD_F32 1
#endif

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
// P-E: result-invariant fast path for the O(n^2) prune inner loop (float).
//
// WHAT IT CHANGES (memory/ISSUE only — no FP value changes)
//   1. Devirtualises `dist_fn->compare`. Consolidation always holds a
//      concrete diskann::DistanceL2Float (bindings/distributed_bindings.cpp
//      :161), but the call goes through Distance<float>* into another
//      translation unit, so every one of the ~1.5e11 pair distances pays an
//      indirect call that the compiler cannot inline or software-pipeline.
//   2. Drops the *hint* instructions inside DistanceL2Float::compare. That
//      kernel issues `_mm_prefetch(a+8(j+1))` and `_mm_prefetch(b+8(j+1))`
//      on every 8-float step — 22 prefetch uops per 96-dim distance, on top
//      of 24 real loads. Inside prune both operands are already L1/L2 hot
//      (the candidate set is 460 x dim*4 B and has just been streamed by
//      step 3b), so those uops are pure load-port pressure. Prefetches have
//      no architectural effect, so removing them cannot change a result.
//   3. Computes B candidate distances with interleaved, INDEPENDENT
//      accumulator chains and one shared load of the pivot vector. Within
//      one selected pivot i, every surviving j is independent (occlude_factor
//      is written at distinct indices and read only in a later outer
//      iteration), so grouping them changes nothing about which pairs are
//      computed, in which order they are applied, or what they evaluate to.
//
// WHAT IT DOES NOT CHANGE
//   Each lane accumulates exactly DistanceL2Float::compare's sequence:
//       sum = 0; for j in [0, size/8): t = a[8j..] - b[8j..];
//                                      sum = fma(t, t, sum)
//       return _mm256_reduce_add_ps(sum)      (same reduce tree)
//   Same operands, same order, same rounding — bit-identical floats. The
//   tail (< B survivors) uses the same kernel at B=1. RobustPrune's
//   sequential dependence (selection order, alpha rounds, occlusion tests)
//   is untouched.
//
// Verified bitwise against a verbatim copy of DistanceL2Float::compare with
// the production toolchain (g++ 11.4, -Ofast -march=icelake-server
// -funroll-loops -DUSE_AVX2): 2.4e6 random probes per dim in {96,128,192,
// 256,384,512}, 0 mismatches for B in {1,2,4,8}; and 0 differing prune
// outputs over 5.1e5 synthetic repair nodes.
//
// Gate: DANNS_CONSOL_PRUNE_OPT=0 restores the original loop verbatim.
// ========================================================================
#ifdef DANNS_PRUNE_SIMD_F32

inline bool prune_opt_enabled()
{
    static const bool on = [] {
        const char* e = std::getenv("DANNS_CONSOL_PRUNE_OPT");
        return !(e && e[0] == '0' && e[1] == '\0');
    }();
    return on;
}

// verbatim from third_party/diskann/include/simd_utils.h:95
inline float prune_reduce_add_ps(__m256 x)
{
    const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(x, 1),
                                   _mm256_castps256_ps128(x));
    const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    return _mm_cvtss_f32(x32);
}

inline float prune_l2f_1(const float* a, const float* b, uint32_t size)
{
    uint16_t niters = (uint16_t)(size / 8);
    __m256 s = _mm256_setzero_ps();
    for (uint16_t j = 0; j < niters; j++) {
        __m256 t = _mm256_sub_ps(_mm256_loadu_ps(a + 8 * j),
                                 _mm256_loadu_ps(b + 8 * j));
        s = _mm256_fmadd_ps(t, t, s);
    }
    return prune_reduce_add_ps(s);
}

// B lanes vs a shared pivot. B is a compile-time constant (4 or 8).
template <int B>
inline void prune_l2f_n(const float* a, const float* const* b,
                        uint32_t size, float* out)
{
    uint16_t niters = (uint16_t)(size / 8);
    // Hoist the lane pointers out of the j-loop: `b` is a pointer to caller
    // storage, so without this GCC re-loads b[k] on every step.
    const float* bb[B];
#pragma GCC unroll 8
    for (int k = 0; k < B; k++) bb[k] = b[k];
    __m256 s[B];
#pragma GCC unroll 8
    for (int k = 0; k < B; k++) s[k] = _mm256_setzero_ps();
    for (uint16_t j = 0; j < niters; j++) {
        const __m256 av = _mm256_loadu_ps(a + 8 * j);
#pragma GCC unroll 8
        for (int k = 0; k < B; k++) {
            __m256 t = _mm256_sub_ps(av, _mm256_loadu_ps(bb[k] + 8 * j));
            s[k] = _mm256_fmadd_ps(t, t, s[k]);
        }
    }
#pragma GCC unroll 8
    for (int k = 0; k < B; k++) out[k] = prune_reduce_add_ps(s[k]);
}

// Reused across repair nodes: removes the per-call occlude_factor + selected
// allocations (2 per prune call, ~1.1e7 calls per node per consolidation).
inline std::vector<float>& prune_occ_scratch()
{
    static thread_local std::vector<float> v;
    return v;
}

template <int B>
inline std::vector<uint64_t> robust_prune_batched_f32(
    std::vector<PruneCandidate<float>>& candidates,
    uint32_t R, float alpha, size_t cmp_dim)
{
    const size_t n = candidates.size();
    std::vector<float>& occ = prune_occ_scratch();
    occ.assign(n, 0.0f);
    std::vector<uint64_t> result;
    result.reserve(R);
    const float kINF = std::numeric_limits<float>::infinity();
    const uint32_t sz = static_cast<uint32_t>(cmp_dim);

    float cur_alpha = 1.0f;
    while (cur_alpha <= alpha + 1e-6f && result.size() < R) {
        for (size_t i = 0; i < n; i++) {
            if (result.size() >= R) break;
            // occ[i] == +inf is the "already selected" sentinel. The original
            // used a separate `selected` bitmap; +inf takes the identical
            // branch in BOTH tests (`occ[i] > cur_alpha` here, `occ[j] >
            // alpha` in the inner loop) and occ[] is never read anywhere
            // else, so the two encodings are observationally identical.
            if (occ[i] > cur_alpha) continue;
            occ[i] = kINF;
            result.push_back(candidates[i].vid);

            const float* vec_i = candidates[i].vec;
            if (!vec_i) continue;

            size_t idx[B];
            const float* bp[B];
            float out[B];
            int nb = 0;
            for (size_t j = i + 1; j < n; j++) {
                if (occ[j] > alpha) continue;
                const float* vec_j = candidates[j].vec;
                if (!vec_j) continue;
                idx[nb] = j;
                bp[nb] = vec_j;
                if (++nb == B) {
                    prune_l2f_n<B>(vec_i, bp, sz, out);
                    for (int k = 0; k < B; k++) {
                        const float dist_ij = out[k];
                        const size_t jj = idx[k];
                        if (dist_ij == 0.0f) {
                            occ[jj] = kINF;
                        } else {
                            occ[jj] = std::max(occ[jj],
                                               candidates[jj].dist / dist_ij);
                        }
                    }
                    nb = 0;
                }
            }
            for (int k = 0; k < nb; k++) {      // tail, one lane at a time
                const float dist_ij = prune_l2f_1(vec_i, bp[k], sz);
                const size_t jj = idx[k];
                if (dist_ij == 0.0f) {
                    occ[jj] = kINF;
                } else {
                    occ[jj] = std::max(occ[jj], candidates[jj].dist / dist_ij);
                }
            }
        }
        cur_alpha *= 1.2f;
    }
    return result;
}

// ========================================================================
// P-F: result-invariant fast path for the step-3b `dist_compute` loop
//      (consolidation.cpp, "Step 3b: Compute distances").
//
// SAME SHAPE AS P-E — one FIXED pivot (the repair node's own vector) against
// n candidates — but a different bottleneck: here each candidate vector is
// read exactly ONCE, cold, from the consolidation-cache arena / colocation
// mmap / RDMA staging arena, so the loop is memory-latency-bound, not
// L2-compute-bound. Two changes, both issue/memory-level only:
//
//   1. Devirtualise + inline `dist_fn->compare`. Consolidation always holds a
//      concrete diskann::DistanceL2Float (bindings/distributed_bindings.cpp
//      :161) but reaches it through Distance<float>* into another TU.
//   2. B lanes against a shared pivot load, reusing the SAME prune_l2f_n<B> /
//      prune_l2f_1 kernels P-E already verified bit-identical.
//
// RESULT-INVARIANCE is strictly stronger than P-E's: the candidate distances
// are mutually independent (nothing in the loop reads a value it just wrote —
// contrast P-E's occlude_factor), so grouping them cannot change any value,
// and `out` is appended in the identical order with the identical elements.
// Each lane accumulates exactly DistanceL2Float::compare's sequence
// (same operands, same order, same reduce tree), including the `size/8`
// truncation for a cmp_dim that is not a multiple of 8.
//
// The P-B lookahead prefetch is preserved verbatim: each batch issues the
// prefetches for candidates [ci+D, ci+D+B) — the same set of lines at the
// same lookahead as the scalar loop, just grouped.
//
// LANE COUNT (measured, idle Ice Lake 6338, 64 T on cpuset 0-63, 14 GiB
// MPOL_INTERLEAVE arena, 460 cold random candidates per repair node, D=8;
// ns per candidate, reference -> best, bit-identical checksums):
//     96 f32   384 B   297.0 ->  224.2  (B=2, 1.32x)   <- deep
//    128       512 B   375.6 ->  319.6  (B=2, 1.18x)
//    192       768 B   545.2 ->  480.9  (B=8, 1.13x)
//    256      1024 B   767.2 ->  644.4  (B=8, 1.19x)
//    384      1536 B  1313.9 ->  997.8  (B=8, 1.32x)
//    512      2048 B  1511.1 -> 1292.0  (B=8, 1.17x)   <- laion
// B=4 loses to BOTH B=2 and B=8 at <= 128 dims (273/359 ns at 96/128), hence
// the 768 B split rather than P-E's 2048 B one.
//
// NOTE, and it is the opposite of P-E: the in-kernel prefetch hints are NOT
// pure loss here. At >= 192 dims the 1-lane no-hint kernel is *slower* than
// the reference (591 vs 545 at 192-d, 1574 vs 1511 at 512-d) because a
// 768 B+ cold read genuinely profits from the intra-vector hint. Batching is
// what recovers it and then wins; do not ship a 1-lane no-hint variant.
//
// Gate: shares DANNS_CONSOL_PRUNE_OPT with P-E (=0 restores the scalar loop).
// ========================================================================
template <int B>
inline void dist_compute_batched_f32(
    const float* node_vec,
    const std::pair<uint64_t, const float*>* cand,
    size_t n_cand,
    uint32_t cmp_dim,
    int prefetch_depth,
    size_t vec_lines,
    std::vector<PruneCandidate<float>>& out)
{
    const float* bp[B];
    float d[B];

    size_t ci = 0;
    for (; ci + B <= n_cand; ci += B) {
        if (prefetch_depth > 0) {
            for (int k = 0; k < B; k++) {
                const size_t pi = ci + static_cast<size_t>(k) +
                                  static_cast<size_t>(prefetch_depth);
                if (pi < n_cand) {
                    const char* pf =
                        reinterpret_cast<const char*>(cand[pi].second);
                    for (size_t b = 0; b < vec_lines; b++)
                        __builtin_prefetch(pf + b * 64, 0, 3);
                }
            }
        }
#pragma GCC unroll 8
        for (int k = 0; k < B; k++) bp[k] = cand[ci + k].second;
        prune_l2f_n<B>(node_vec, bp, cmp_dim, d);
        for (int k = 0; k < B; k++)
            out.push_back({cand[ci + k].first, d[k], cand[ci + k].second});
    }
    for (; ci < n_cand; ci++) {           // tail, one lane at a time
        if (prefetch_depth > 0) {
            const size_t pi = ci + static_cast<size_t>(prefetch_depth);
            if (pi < n_cand) {
                const char* pf =
                    reinterpret_cast<const char*>(cand[pi].second);
                for (size_t b = 0; b < vec_lines; b++)
                    __builtin_prefetch(pf + b * 64, 0, 3);
            }
        }
        out.push_back({cand[ci].first,
                       prune_l2f_1(node_vec, cand[ci].second, cmp_dim),
                       cand[ci].second});
    }
}

// Lane dispatch for P-F. 768 B = 192 f32 dims; see the table above.
inline constexpr bool dist_opt_use_8_lanes(size_t vec_bytes)
{
    return vec_bytes >= 768;
}
#endif  // DANNS_PRUNE_SIMD_F32

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

#ifdef DANNS_PRUNE_SIMD_F32
    if constexpr (std::is_same<T, float>::value) {
        if (prune_opt_enabled()) {
            // 8 lanes pay off once the candidate vectors stop being
            // L1-resident; 4 lanes win below that (measured on an idle Ice
            // Lake 6338, 64 threads, 460 candidates: 96-d 1.62x / 384-d
            // 2.05x at B=4, 512-d 2.05x at B=8).
            return (cmp_dim * sizeof(float) >= 2048)
                       ? robust_prune_batched_f32<8>(candidates, R, alpha,
                                                     cmp_dim)
                       : robust_prune_batched_f32<4>(candidates, R, alpha,
                                                     cmp_dim);
        }
    }
#endif

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
