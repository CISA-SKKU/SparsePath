// update/src/lits_policy.cpp -- LITS-aware update policy boundary

#include "update/lits_policy.h"

namespace danns {

void LITSFilter::filter_expanded_pool(
    std::vector<std::pair<uint64_t, float>>& pool,
    uint32_t my_partition,
    uint32_t /*max_keep*/)
{
    if (pool.empty()) return;

    double sum_dist = 0.0;
    for (const auto& [vid, dist] : pool) {
        (void)vid;
        sum_dist += dist;
    }
    const float avg_dist = static_cast<float>(sum_dist / pool.size());

    pool.erase(
        std::remove_if(pool.begin(), pool.end(),
            [avg_dist, my_partition](const std::pair<uint64_t, float>& p) {
                return (decode_partition(p.first) != my_partition) &&
                       (p.second > avg_dist);
            }),
        pool.end());
}

} // namespace danns
