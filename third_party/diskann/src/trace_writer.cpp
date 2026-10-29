// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
//
// Trace writer implementation for DiskANN analysis
// Author: Wonseok Oh

#include "trace_writer.h"
#include <fstream>
#include <iostream>

namespace diskann
{
namespace trace
{

void save_inter_insert_trace_csv(const std::string &filepath,
                                  const std::vector<std::vector<PruningTraceEvent>> &trace_buffers)
{
    std::ofstream out(filepath);
    if (!out.is_open())
    {
        std::cerr << "Warning: Could not open inter-insert trace file: " << filepath << std::endl;
        return;
    }

    out << "node_id,before_degree,after_degree,edges_before,edges_after\n";

    for (const auto &buffer : trace_buffers)
    {
        for (const auto &event : buffer)
        {
            out << event.node_id << "," << event.edges_before.size() << "," << event.edges_after.size() << ",\"";

            for (size_t i = 0; i < event.edges_before.size(); i++)
            {
                if (i > 0)
                    out << " ";
                out << event.edges_before[i];
            }
            out << "\",\"";

            for (size_t i = 0; i < event.edges_after.size(); i++)
            {
                if (i > 0)
                    out << " ";
                out << event.edges_after[i];
            }
            out << "\"\n";
        }
    }

    out.close();
    std::cout << "Inter-insert pruning trace saved to: " << filepath << std::endl;
}

void save_delete_consolidate_trace_csv(const std::string &filepath,
                                        const std::vector<std::vector<PruningTraceEvent>> &trace_buffers)
{
    std::ofstream out(filepath);
    if (!out.is_open())
    {
        std::cerr << "Warning: Could not open delete-consolidate trace file: " << filepath << std::endl;
        return;
    }

    out << "node_id,before_degree,expanded_degree,after_degree,edges_before,edges_expanded,edges_after\n";

    for (const auto &buffer : trace_buffers)
    {
        for (const auto &event : buffer)
        {
            out << event.node_id << "," << event.edges_before.size() << "," << event.edges_expanded.size() << ","
                << event.edges_after.size() << ",\"";

            for (size_t i = 0; i < event.edges_before.size(); i++)
            {
                if (i > 0)
                    out << " ";
                out << event.edges_before[i];
            }
            out << "\",\"";

            for (size_t i = 0; i < event.edges_expanded.size(); i++)
            {
                if (i > 0)
                    out << " ";
                out << event.edges_expanded[i];
            }
            out << "\",\"";

            for (size_t i = 0; i < event.edges_after.size(); i++)
            {
                if (i > 0)
                    out << " ";
                out << event.edges_after[i];
            }
            out << "\"\n";
        }
    }

    out.close();
    std::cout << "Delete-consolidate pruning trace saved to: " << filepath << std::endl;
}

void save_delete_detail_trace_json(const std::string &filepath,
                                    const std::vector<std::vector<LocalityAwareTraceEvent>> &trace_buffers)
{
    std::ofstream out(filepath);
    if (!out.is_open())
    {
        std::cerr << "Failed to open file for delete detail trace: " << filepath << std::endl;
        return;
    }

    // Collect all events from all thread buffers
    std::vector<LocalityAwareTraceEvent> all_events;
    for (const auto &buffer : trace_buffers)
    {
        for (const auto &event : buffer)
        {
            all_events.push_back(event);
        }
    }

    // Write JSON
    out << "{\n";
    out << "  \"events\": [\n";

    for (size_t i = 0; i < all_events.size(); i++)
    {
        const auto &event = all_events[i];
        out << "    {\n";
        out << "      \"source_node_tag\": " << event.source_node_tag << ",\n";

        // Original neighbors (before pruning)
        out << "      \"original_neighbors\": [";
        for (size_t j = 0; j < event.original_neighbors.size(); j++)
        {
            out << event.original_neighbors[j];
            if (j < event.original_neighbors.size() - 1)
                out << ", ";
        }
        out << "],\n";

        // Neighbors after pruning
        out << "      \"neighbors_after_pruning\": [";
        for (size_t j = 0; j < event.neighbors_after_pruning.size(); j++)
        {
            out << event.neighbors_after_pruning[j];
            if (j < event.neighbors_after_pruning.size() - 1)
                out << ", ";
        }
        out << "],\n";

        // Short-range deleted
        out << "      \"short_range_deleted\": [\n";
        for (size_t j = 0; j < event.short_range_deleted.size(); j++)
        {
            const auto &info = event.short_range_deleted[j];
            out << "        {\n";
            out << "          \"deleted_node_tag\": " << info.deleted_node_tag << ",\n";
            out << "          \"neighbors\": [";
            for (size_t k = 0; k < info.neighbors.size(); k++)
            {
                out << info.neighbors[k];
                if (k < info.neighbors.size() - 1)
                    out << ", ";
            }
            out << "]\n";
            out << "        }";
            if (j < event.short_range_deleted.size() - 1)
                out << ",";
            out << "\n";
        }
        out << "      ],\n";

        // Long-range dropped
        out << "      \"long_range_dropped\": [\n";
        for (size_t j = 0; j < event.long_range_dropped.size(); j++)
        {
            const auto &info = event.long_range_dropped[j];
            out << "        {\n";
            out << "          \"dropped_node_tag\": " << info.deleted_node_tag << ",\n";
            out << "          \"neighbors\": [";
            for (size_t k = 0; k < info.neighbors.size(); k++)
            {
                out << info.neighbors[k];
                if (k < info.neighbors.size() - 1)
                    out << ", ";
            }
            out << "]\n";
            out << "        }";
            if (j < event.long_range_dropped.size() - 1)
                out << ",";
            out << "\n";
        }
        out << "      ]\n";

        out << "    }";
        if (i < all_events.size() - 1)
            out << ",";
        out << "\n";
    }

    out << "  ]\n";
    out << "}\n";

    out.close();
    std::cout << "Delete detail trace saved to: " << filepath << " (" << all_events.size() << " events)" << std::endl;
}

void save_search_trace_csv(const std::string &filepath, uint32_t query_id,
                           const std::vector<SearchTraceInfo> &trace_path)
{
    std::ofstream out(filepath, std::ios::app);  // Append mode
    if (!out.is_open())
    {
        std::cerr << "Warning: Could not open search trace file: " << filepath << std::endl;
        return;
    }

    // Check if file is empty to write header
    out.seekp(0, std::ios::end);
    if (out.tellp() == 0)
    {
        out << "query_id,hop,node_id,partition_id,L,intra_comps,inter_comps,candidate_count\n";
    }

    for (const auto &info : trace_path)
    {
        out << query_id << "," << info.hop << "," << info.node_id << "," << info.partition_id << "," << info.L << ","
            << info.intra_partition_comps << "," << info.inter_partition_comps << "," << info.candidate_ids.size()
            << "\n";
    }

    out.close();
}

}  // namespace trace
}  // namespace diskann
