#pragma once
// update/load_balancer.h — Stage B LoadBalancer selection rule (measure-only)
//
// Computes the would-be assignment partition m_assign for an inserted point
// from its insert-search pool V. Pure logic: no RDMA, no state — the counter
// snapshot is taken once per insert_batch and passed in.
//
// This function is mirrored 1:1 by the host-side oracle used for TDD
// (see docs/features/load_balancer_stage_b_design.md §4); the lb_compute
// debug binding runs the same test vectors against this implementation.
//
// Rule (design §2):
//   1. v[part] = histogram of decode_partition over the pool
//   2. candidates = { part : v[part] >= threshold * v_max }   (inclusive,
//      double arithmetic — matches Python float comparison exactly)
//   3. group candidates by owner node; owners without a counter snapshot
//      cannot bid and are dropped. Pool empty or <= 1 distinct bidder
//      -> (my_partition, considered=false, would_redirect=false)
//   4. deterministic argmin over bidders' counters.
//      Ties: my node first, then lowest node id. Among the winning owner's
//      candidate partitions (> 1 only if partition->node is not 1:1):
//      highest v, then lowest partition id.
//   5. would_redirect = (winning node != my node). Measure-only: the caller
//      records the verdict and ALWAYS proceeds on my_partition.

#include <algorithm>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/vector_id.h"

namespace danns {

struct LoadBalanceDecision {
    uint32_t m_assign;       // partition the LoadBalancer would assign
    bool considered;         // >= 2 distinct candidate owners had counters
    bool would_redirect;     // owner(m_assign) != my node
};

// counter_of(node) -> std::optional<uint64_t>: the owner's live vector count,
// or nullopt if unobservable (that owner cannot bid). Probed at most once per
// distinct candidate owner — in live-redirect mode each probe is an 8-byte
// RDMA READ, in measure-only mode a snapshot-map lookup.
template <typename PartToNodeFn, typename CounterFn>
inline LoadBalanceDecision compute_load_balance_assignment(
    const std::vector<std::pair<uint64_t, float>>& pool,  // (vid, dist) = V
    uint32_t my_partition,
    double threshold,
    PartToNodeFn&& part_to_node,
    CounterFn&& counter_of)
{
    LoadBalanceDecision d{my_partition, false, false};
    if (pool.empty()) return d;

    const uint32_t my_node = part_to_node(my_partition);

    // 1. partition histogram over the pool
    std::unordered_map<uint32_t, uint32_t> v;
    for (const auto& vd : pool)
        v[static_cast<uint32_t>(decode_partition(vd.first))]++;

    uint32_t v_max = 0;
    for (const auto& kv : v) v_max = std::max(v_max, kv.second);

    // 2+3. candidate partitions grouped by owner; counterless owners dropped.
    // Probe each distinct owner's counter exactly once.
    std::unordered_map<uint32_t, std::vector<uint32_t>> owners;
    std::unordered_map<uint32_t, std::optional<uint64_t>> probed;
    for (const auto& kv : v) {
        if (static_cast<double>(kv.second) <
            threshold * static_cast<double>(v_max)) continue;
        const uint32_t owner = part_to_node(kv.first);
        auto it = probed.find(owner);
        if (it == probed.end())
            it = probed.emplace(owner, counter_of(owner)).first;
        if (!it->second.has_value()) continue;
        owners[owner].push_back(kv.first);
    }
    if (owners.size() <= 1) return d;

    // 4. deterministic argmin over bidders' counters
    //    (explicit comparisons — unordered_map iteration order must not
    //    influence the result). Tie: my node first, then lowest node id.
    bool have_best = false;
    uint32_t best_node = 0;
    uint64_t best_cnt = 0;
    for (const auto& kv : owners) {
        const uint32_t n = kv.first;
        const uint64_t cnt = *probed.at(n);
        bool take = false;
        if (!have_best || cnt < best_cnt) {
            take = true;
        } else if (cnt == best_cnt && best_node != my_node) {
            take = (n == my_node) || (n < best_node);
        }
        if (take) {
            have_best = true;
            best_node = n;
            best_cnt = cnt;
        }
    }

    // Winning owner's partition: highest v, then lowest partition id
    const auto& parts = owners[best_node];
    uint32_t m = parts[0];
    for (const uint32_t p : parts) {
        if (v[p] > v[m] || (v[p] == v[m] && p < m)) m = p;
    }

    d.m_assign = m;
    d.considered = true;
    d.would_redirect = (best_node != my_node);
    return d;
}

} // namespace danns
