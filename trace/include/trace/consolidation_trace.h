#pragma once
// trace/consolidation_trace.h — Consolidation trace writer
// Extracted from: distributed_bindings.cpp L190-228

#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include "trace/trace_types.h"

namespace danns {

class ConsolidationTraceWriter {
public:
    ConsolidationTraceWriter() : active_(false) {}

    bool open(const std::string& path, uint32_t node_id, uint32_t dim) {
        if (path.empty()) return true;
        fs_.open(path, std::ios::binary);
        if (!fs_) return false;
        active_ = true;
        TraceFileHeader hdr{};
        std::memcpy(hdr.magic, "CONT", 4);
        hdr.version = 1;
        hdr.trace_level = 1;
        hdr.node_id = node_id;
        hdr.vector_dim = dim;
        fs_.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
        return true;
    }

    void write_entry(const ConsolidationEntryHeader& eh,
                     const std::vector<AccessRecord>& hop_accesses,
                     const std::vector<AccessRecord>& vec_accesses) {
        if (!active_) return;
        fs_.write(reinterpret_cast<const char*>(&eh), sizeof(eh));
        if (eh.n_2hop_accesses > 0)
            fs_.write(reinterpret_cast<const char*>(hop_accesses.data()),
                      eh.n_2hop_accesses * sizeof(AccessRecord));
        if (eh.n_vec_accesses > 0)
            fs_.write(reinterpret_cast<const char*>(vec_accesses.data()),
                      eh.n_vec_accesses * sizeof(AccessRecord));
    }

    void close() { if (fs_.is_open()) fs_.close(); active_ = false; }
    bool active() const { return active_; }

private:
    std::ofstream fs_;
    bool active_;
};

} // namespace danns
