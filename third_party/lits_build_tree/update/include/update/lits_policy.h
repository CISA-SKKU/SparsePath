#pragma once
// update/lits_policy.h -- LITS-aware update policy boundary
//
// This header is the extension point for the paper-level LITS-Aware Update
// path. The current implementation preserves the existing consolidation filter
// behavior while giving build/insert/consolidation a shared place to hook into.

#include <algorithm>
#include <cstdint>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/vector_id.h"
namespace danns {

template <typename T>
struct PruneCandidate;

struct LITSFilterStats {
    uint64_t checked = 0;
    uint64_t filtered = 0;
};

class LITSFilter {
public:
    static void filter_expanded_pool(
        std::vector<std::pair<uint64_t, float>>& pool,
        uint32_t my_partition,
        uint32_t max_keep);

    static LITSFilterStats filter_consolidation_candidates(
        std::vector<std::pair<uint64_t, float>>& candidates,
        const std::unordered_set<uint64_t>& original_neighbor_vids,
        uint32_t my_partition,
        // tau contribution from DELETED original neighbors (LITS-Aware-Update):
        // their distances keep tau stable when most/all survivors are gone.
        double tau_extra_sum = 0.0,
        uint32_t tau_extra_count = 0)
    {
        if (candidates.empty()) return {};

        double sum_orig = tau_extra_sum;
        uint32_t count_orig = tau_extra_count;
        for (const auto& [vid, dist] : candidates) {
            if (original_neighbor_vids.count(vid)) {
                sum_orig += dist;
                count_orig++;
            }
        }
        if (count_orig == 0) return {};

        const float tau = static_cast<float>(sum_orig / count_orig);
        LITSFilterStats stats;
        candidates.erase(
            std::remove_if(candidates.begin(), candidates.end(),
                [&](const std::pair<uint64_t, float>& p) {
                    if (original_neighbor_vids.count(p.first)) return false;
                    stats.checked++;
                    if ((decode_partition(p.first) != my_partition) &&
                        (p.second > tau)) {
                        stats.filtered++;
                        return true;
                    }
                    return false;
                }),
            candidates.end());
        return stats;
    }

    template <typename T>
    static LITSFilterStats filter_consolidation_candidates(
        std::vector<PruneCandidate<T>>& candidates,
        const std::unordered_set<uint64_t>& original_neighbor_vids,
        uint32_t my_partition,
        // tau contribution from DELETED original neighbors (LITS-Aware-Update):
        // their distances keep tau stable when most/all survivors are gone.
        double tau_extra_sum = 0.0,
        uint32_t tau_extra_count = 0)
    {
        if (candidates.empty()) return {};

        double sum_orig = tau_extra_sum;
        uint32_t count_orig = tau_extra_count;
        for (const auto& c : candidates) {
            if (original_neighbor_vids.count(c.vid)) {
                sum_orig += c.dist;
                count_orig++;
            }
        }
        if (count_orig == 0) return {};

        const float tau = static_cast<float>(sum_orig / count_orig);
        LITSFilterStats stats;
        candidates.erase(
            std::remove_if(candidates.begin(), candidates.end(),
                [&](const PruneCandidate<T>& c) {
                    if (original_neighbor_vids.count(c.vid)) return false;
                    stats.checked++;
                    if ((decode_partition(c.vid) != my_partition) &&
                        (c.dist > tau)) {
                        stats.filtered++;
                        return true;
                    }
                    return false;
                }),
            candidates.end());
        return stats;
    }
};

} // namespace danns
