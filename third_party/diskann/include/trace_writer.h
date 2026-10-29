// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
//
// Trace writer utilities for DiskANN analysis
// Author: Wonseok Oh

#pragma once

#include <string>
#include <vector>
#include "trace_types.h"

namespace diskann
{
namespace trace
{

// Save inter-insert pruning trace to CSV file
// Format: node_id,before_degree,after_degree,edges_before,edges_after
void save_inter_insert_trace_csv(const std::string &filepath,
                                  const std::vector<std::vector<PruningTraceEvent>> &trace_buffers);

// Save delete-consolidate pruning trace to CSV file
// Format: node_id,before_degree,expanded_degree,after_degree,edges_before,edges_expanded,edges_after
void save_delete_consolidate_trace_csv(const std::string &filepath,
                                        const std::vector<std::vector<PruningTraceEvent>> &trace_buffers);

// Save delete detail trace to JSON file
// Contains detailed locality-aware information with tags
void save_delete_detail_trace_json(const std::string &filepath,
                                    const std::vector<std::vector<LocalityAwareTraceEvent>> &trace_buffers);

// Save search trace to CSV file
// Format: query_id,hop,node_id,partition_id,L,intra_comps,inter_comps,candidate_count
void save_search_trace_csv(const std::string &filepath, uint32_t query_id,
                           const std::vector<SearchTraceInfo> &trace_path);

}  // namespace trace
}  // namespace diskann
