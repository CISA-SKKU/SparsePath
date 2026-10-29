// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <sstream>
#include <typeinfo>
#include <unordered_map>

#include "omp.h"
#include "defaults.h"

namespace diskann
{

// Pruning strategy for graph construction
enum class PruningStrategy
{
    ROBUST_PRUNE,           // Original RobustPrune with alpha (diversity-aware)
    GREEDY_NEAREST,         // Greedy: select K nearest neighbors (no occlusion, fastest)
    RANDOM_PRUNE,           // Robust prune + random reduction to reduce degree
    HYBRID_PRUNE,           // Future: combination strategies
    PARTITION_AWARE_PRUNE,  // Partition-aware: prioritize intra-partition edges
    MEDIAN_EVICT            // Evict median-distance edges, keep short-range + long-range
};

class IndexWriteParameters

{
  public:
    const uint32_t search_list_size; // L
    const uint32_t max_degree;       // R
    const bool saturate_graph;
    const uint32_t max_occlusion_size; // C
    const float alpha;
    const uint32_t num_threads;
    const uint32_t filter_list_size; // Lf
    const PruningStrategy pruning_strategy;  // Pruning algorithm to use
    const float random_prune_ratio;          // For RANDOM_PRUNE: target_degree = max_degree * ratio
    const float delete_degree_ratio;         // For DELETE: target_degree = max_degree * ratio (default 1.0)
    const bool enable_pruning_trace;         // Enable detailed pruning trace recording (default false)
    const bool enable_locality_aware_pruning; // Enable locality-aware pruning during delete consolidation (default false)
    const float short_range_ratio;           // Ratio of deleted neighbors to treat as short-range (default 0.5)
    const std::string partition_file;        // For PARTITION_AWARE_PRUNE: path to partition file (optional)

    IndexWriteParameters(const uint32_t search_list_size, const uint32_t max_degree, const bool saturate_graph,
                         const uint32_t max_occlusion_size, const float alpha, const uint32_t num_threads,
                         const uint32_t filter_list_size, const PruningStrategy pruning_strategy = PruningStrategy::ROBUST_PRUNE,
                         const float random_prune_ratio = 1.0f, const float delete_degree_ratio = 1.0f,
                         const bool enable_pruning_trace = false, const bool enable_locality_aware_pruning = false,
                         const float short_range_ratio = 0.5f, const std::string& partition_file = "")
        : search_list_size(search_list_size), max_degree(max_degree), saturate_graph(saturate_graph),
          max_occlusion_size(max_occlusion_size), alpha(alpha), num_threads(num_threads),
          filter_list_size(filter_list_size), pruning_strategy(pruning_strategy), random_prune_ratio(random_prune_ratio),
          delete_degree_ratio(delete_degree_ratio), enable_pruning_trace(enable_pruning_trace),
          enable_locality_aware_pruning(enable_locality_aware_pruning), short_range_ratio(short_range_ratio),
          partition_file(partition_file)
    {
    }

    friend class IndexWriteParametersBuilder;
};

class IndexSearchParams
{
  public:
    IndexSearchParams(const uint32_t initial_search_list_size, const uint32_t num_search_threads)
        : initial_search_list_size(initial_search_list_size), num_search_threads(num_search_threads)
    {
    }
    const uint32_t initial_search_list_size; // search L
    const uint32_t num_search_threads;       // search threads
};

class IndexWriteParametersBuilder
{
    /**
     * Fluent builder pattern to keep track of the 7 non-default properties
     * and their order. The basic ctor was getting unwieldy.
     */
  public:
    IndexWriteParametersBuilder(const uint32_t search_list_size, // L
                                const uint32_t max_degree        // R
                                )
        : _search_list_size(search_list_size), _max_degree(max_degree)
    {
    }

    IndexWriteParametersBuilder &with_max_occlusion_size(const uint32_t max_occlusion_size)
    {
        _max_occlusion_size = max_occlusion_size;
        return *this;
    }

    IndexWriteParametersBuilder &with_saturate_graph(const bool saturate_graph)
    {
        _saturate_graph = saturate_graph;
        return *this;
    }

    IndexWriteParametersBuilder &with_alpha(const float alpha)
    {
        _alpha = alpha;
        return *this;
    }

    IndexWriteParametersBuilder &with_num_threads(const uint32_t num_threads)
    {
        _num_threads = num_threads == 0 ? omp_get_num_procs() : num_threads;
        return *this;
    }

    IndexWriteParametersBuilder &with_filter_list_size(const uint32_t filter_list_size)
    {
        _filter_list_size = filter_list_size == 0 ? _search_list_size : filter_list_size;
        return *this;
    }

    IndexWriteParametersBuilder &with_pruning_strategy(const PruningStrategy pruning_strategy)
    {
        _pruning_strategy = pruning_strategy;
        return *this;
    }

    IndexWriteParametersBuilder &with_random_prune_ratio(const float random_prune_ratio)
    {
        _random_prune_ratio = random_prune_ratio;
        return *this;
    }

    IndexWriteParametersBuilder &with_delete_degree_ratio(const float delete_degree_ratio)
    {
        _delete_degree_ratio = delete_degree_ratio;
        return *this;
    }

    IndexWriteParametersBuilder &with_pruning_trace(const bool enable_pruning_trace)
    {
        _enable_pruning_trace = enable_pruning_trace;
        return *this;
    }

    IndexWriteParametersBuilder &with_locality_aware_pruning(const bool enable_locality_aware_pruning)
    {
        _enable_locality_aware_pruning = enable_locality_aware_pruning;
        return *this;
    }

    IndexWriteParametersBuilder &with_short_range_ratio(const float short_range_ratio)
    {
        _short_range_ratio = short_range_ratio;
        return *this;
    }

    IndexWriteParametersBuilder &with_partition_file(const std::string& partition_file)
    {
        _partition_file = partition_file;
        return *this;
    }

    IndexWriteParameters build() const
    {
        return IndexWriteParameters(_search_list_size, _max_degree, _saturate_graph, _max_occlusion_size, _alpha,
                                    _num_threads, _filter_list_size, _pruning_strategy, _random_prune_ratio,
                                    _delete_degree_ratio, _enable_pruning_trace, _enable_locality_aware_pruning,
                                    _short_range_ratio, _partition_file);
    }

    IndexWriteParametersBuilder(const IndexWriteParameters &wp)
        : _search_list_size(wp.search_list_size), _max_degree(wp.max_degree),
          _max_occlusion_size(wp.max_occlusion_size), _saturate_graph(wp.saturate_graph), _alpha(wp.alpha),
          _filter_list_size(wp.filter_list_size), _pruning_strategy(wp.pruning_strategy),
          _random_prune_ratio(wp.random_prune_ratio), _delete_degree_ratio(wp.delete_degree_ratio),
          _enable_pruning_trace(wp.enable_pruning_trace), _enable_locality_aware_pruning(wp.enable_locality_aware_pruning),
          _short_range_ratio(wp.short_range_ratio), _partition_file(wp.partition_file)
    {
    }
    IndexWriteParametersBuilder(const IndexWriteParametersBuilder &) = delete;
    IndexWriteParametersBuilder &operator=(const IndexWriteParametersBuilder &) = delete;

  private:
    uint32_t _search_list_size{};
    uint32_t _max_degree{};
    uint32_t _max_occlusion_size{defaults::MAX_OCCLUSION_SIZE};
    bool _saturate_graph{defaults::SATURATE_GRAPH};
    float _alpha{defaults::ALPHA};
    uint32_t _num_threads{defaults::NUM_THREADS};
    uint32_t _filter_list_size{defaults::FILTER_LIST_SIZE};
    PruningStrategy _pruning_strategy{PruningStrategy::ROBUST_PRUNE};
    float _random_prune_ratio{1.0f};
    float _delete_degree_ratio{1.0f};
    bool _enable_pruning_trace{false};
    bool _enable_locality_aware_pruning{false};
    float _short_range_ratio{0.5f};
    std::string _partition_file{};
};

} // namespace diskann
