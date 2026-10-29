#pragma once
// trace/search_trace.h — Search trace writer
// Extracted from: distributed_bindings.cpp L143-187

#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include "trace/trace_types.h"

namespace danns {

class SearchTraceWriter {
public:
    SearchTraceWriter() : level_(0) {}

    bool open(const std::string& path, uint16_t level, uint32_t node_id, uint32_t dim) {
        if (level == 0) return true;
        level_ = level;
        fs_.open(path, std::ios::binary);
        if (!fs_) return false;
        TraceFileHeader hdr{};
        std::memcpy(hdr.magic, "SRCT", 4);
        hdr.version = 2;
        hdr.trace_level = level;
        hdr.node_id = node_id;
        hdr.vector_dim = dim;
        fs_.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
        return true;
    }

    void write_query(const QueryHeader& qh,
                     const ResultEntry* results, uint16_t n_results,
                     const std::vector<HopRecord>& hops,
                     const std::vector<std::vector<FetchedEntry>>& fetched_per_hop) {
        if (!level_) return;
        fs_.write(reinterpret_cast<const char*>(&qh), sizeof(qh));
        fs_.write(reinterpret_cast<const char*>(results), n_results * sizeof(ResultEntry));
        for (size_t h = 0; h < hops.size(); h++) {
            fs_.write(reinterpret_cast<const char*>(&hops[h]), sizeof(HopRecord));
            if (level_ >= 2 && h < fetched_per_hop.size()) {
                uint16_t n = static_cast<uint16_t>(fetched_per_hop[h].size());
                fs_.write(reinterpret_cast<const char*>(&n), sizeof(n));
                if (n > 0)
                    fs_.write(reinterpret_cast<const char*>(fetched_per_hop[h].data()),
                              n * sizeof(FetchedEntry));
            }
        }
    }

    void close() { if (fs_.is_open()) fs_.close(); level_ = 0; }
    uint16_t level() const { return level_; }

private:
    std::ofstream fs_;
    uint16_t level_;
};

} // namespace danns
