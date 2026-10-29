// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#ifdef _WINDOWS
#include <numeric>
#endif
#include <string>
#include <vector>

#include "distance.h"
#include "parameters.h"

namespace diskann
{
struct QueryStats
{
    float total_us = 0; // total time to process query in micros
    float io_us = 0;    // total time spent in IO
    float cpu_us = 0;   // total time spent in CPU

    unsigned n_4k = 0;         // # of 4kB reads
    unsigned n_8k = 0;         // # of 8kB reads
    unsigned n_12k = 0;        // # of 12kB reads
    unsigned n_ios = 0;        // total # of IOs issued
    unsigned read_size = 0;    // total # of bytes read
    unsigned n_cmps_saved = 0; // # cmps saved
    unsigned n_cmps = 0;       // # cmps
    unsigned n_cache_hits = 0; // # cache_hits
    unsigned n_hops = 0;       // # search hops

    // === Search breakdown (for latency profiling) ===
    float dist_us = 0;      // Distance computation time
    float fetch_us = 0;     // Neighbor fetch time (graph traversal)
    float queue_us = 0;     // Priority queue operations (insert + pop)
    float visited_us = 0;   // Visited set operations (lookup + insert)

    // === Partition-aware breakdown (for distributed latency profiling) ===
    // These are only populated when partition_ids is provided
    // Using double for nanosecond-derived precision (ns / 1000.0)
    double dist_local_us = 0;     // Distance computation for local partition
    double dist_remote_us = 0;    // Distance computation for remote partitions
    double fetch_local_us = 0;    // Neighbor fetch from local partition
    double fetch_remote_us = 0;   // Neighbor fetch from remote partitions
    unsigned local_vector_accesses = 0;   // Number of local vector accesses
    unsigned remote_vector_accesses = 0;  // Number of remote vector accesses
    unsigned local_neighbor_accesses = 0; // Number of local neighbor list accesses
    unsigned remote_neighbor_accesses = 0;// Number of remote neighbor list accesses
    unsigned rdma_round_trips = 0;        // Number of RDMA round trips (per-partition-per-hop)
    double rdma_neighbor_fetch_ns = 0;    // Cumulative RDMA time for remote neighbor fetches (ns)
};

// === Insert operation breakdown ===
struct InsertStats
{
    float total_us = 0;         // Total insert time
    float lock_us = 0;          // Lock acquisition time
    float reserve_us = 0;       // Location reservation time
    float data_store_us = 0;    // Data store write time
    float search_prune_us = 0;  // Search and prune time
    float link_us = 0;          // Graph linking time (set_neighbours + inter_insert)
};

// === Delete operation breakdown ===
struct DeleteStats
{
    float total_us = 0;       // Total delete time
    float lock_us = 0;        // Lock acquisition time
    float tag_lookup_us = 0;  // Tag to location lookup time
    float mark_us = 0;        // Delete marking time
};

// === Consolidate operation breakdown ===
struct ConsolidateStats
{
    float total_us = 0;        // Total consolidate time
    float compaction_us = 0;   // Data compaction time
    float relink_us = 0;       // Graph relinking time
    unsigned points_removed = 0;
    unsigned points_remaining = 0;
};

template <typename T>
inline T get_percentile_stats(QueryStats *stats, uint64_t len, float percentile,
                              const std::function<T(const QueryStats &)> &member_fn)
{
    std::vector<T> vals(len);
    for (uint64_t i = 0; i < len; i++)
    {
        vals[i] = member_fn(stats[i]);
    }

    std::sort(vals.begin(), vals.end(), [](const T &left, const T &right) { return left < right; });

    auto retval = vals[(uint64_t)(percentile * len)];
    vals.clear();
    return retval;
}

template <typename T>
inline double get_mean_stats(QueryStats *stats, uint64_t len, const std::function<T(const QueryStats &)> &member_fn)
{
    double avg = 0;
    for (uint64_t i = 0; i < len; i++)
    {
        avg += (double)member_fn(stats[i]);
    }
    return avg / len;
}
} // namespace diskann
