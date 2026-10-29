// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
//
// Custom trace types for DiskANN analysis
// Author: Wonseok Oh

#pragma once

#include <cstdint>
#include <vector>

namespace diskann
{
namespace trace
{

// Search trace: captures hop-by-hop search progression
struct SearchTraceInfo
{
    uint32_t node_id;
    uint32_t partition_id;
    uint32_t hop;
    uint32_t L;  // Search list size

    uint32_t intra_partition_comps;  // Comparisons within same partition
    uint32_t inter_partition_comps;  // Comparisons across partitions

    // Candidate set at this hop (IDs only, distances computed at analysis time)
    std::vector<uint32_t> candidate_ids;
};

// Pruning trace: captures edge changes during insert/delete operations
struct PruningTraceEvent
{
    uint32_t node_id;
    std::vector<uint32_t> edges_before;    // Original neighbors before operation
    std::vector<uint32_t> edges_expanded;  // Candidate set after expansion (before pruning)
    std::vector<uint32_t> edges_after;     // Final neighbors after pruning

    // For locality-aware analysis
    std::vector<uint32_t> short_range_neighbors;  // Expanded (kept)
    std::vector<uint32_t> long_range_neighbors;   // Dropped
};

// Fine-grained trace for delete process
struct DeletePruningDetailEvent
{
    uint32_t source_node_id;                     // Node being processed
    std::vector<uint32_t> deleted_neighbor_ids;  // Deleted neighbors that triggered expansion
    size_t expanded_candidate_count;             // Size of expanded candidate set
};

// Info about a deleted neighbor and its connections
struct DeletedNeighborInfo
{
    uint32_t deleted_node_tag;        // Tag of the deleted neighbor
    std::vector<uint32_t> neighbors;  // Tags of the deleted neighbor's neighbors
};

// Locality-aware pruning trace event (all IDs are tags/dataset indices)
struct LocalityAwareTraceEvent
{
    uint32_t source_node_tag;                              // Tag of the node being processed
    std::vector<uint32_t> original_neighbors;              // Tags of original neighbors
    std::vector<uint32_t> neighbors_after_pruning;         // Tags of neighbors after pruning
    std::vector<DeletedNeighborInfo> short_range_deleted;  // Short-range (expanded)
    std::vector<DeletedNeighborInfo> long_range_dropped;   // Long-range (dropped)
};

}  // namespace trace
}  // namespace diskann

// Backward compatibility aliases (for existing code)
using TraceInfo = diskann::trace::SearchTraceInfo;
using PruningTraceEvent = diskann::trace::PruningTraceEvent;
using DeletePruningDetailEvent = diskann::trace::DeletePruningDetailEvent;
using DeletedNeighborInfo = diskann::trace::DeletedNeighborInfo;
using LocalityAwareTraceEvent = diskann::trace::LocalityAwareTraceEvent;
