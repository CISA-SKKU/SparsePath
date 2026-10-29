#pragma once
// core/latency_stats.h — Latency statistics with percentile computation
// Extracted from: distributed_bindings.cpp L328-460
// NOTE: py::dict/py::array conversion moved to bindings/

#include <cstdint>
#include <vector>
#include <algorithm>

namespace danns {

struct LatencyStats {
    // Raw latency samples (nanoseconds)
    std::vector<uint64_t> local_latencies_ns;
    std::vector<uint64_t> remote_latencies_ns;

    // Counters
    uint64_t local_count = 0;
    uint64_t remote_count = 0;
    uint64_t total_local_ns = 0;
    uint64_t total_remote_ns = 0;

    // Bandwidth tracking
    uint64_t bytes_transferred = 0;

    // Configuration
    bool detailed_sampling = false;
    uint32_t sample_rate = 100;
    uint32_t sample_counter = 0;

    void record_local(uint64_t ns) {
        local_count++;
        total_local_ns += ns;
        if (detailed_sampling || (++sample_counter % sample_rate == 0)) {
            local_latencies_ns.push_back(ns);
        }
    }

    void record_remote(uint64_t ns, uint64_t bytes = 0) {
        remote_count++;
        total_remote_ns += ns;
        bytes_transferred += bytes;
        if (detailed_sampling || (sample_counter % sample_rate == 0)) {
            remote_latencies_ns.push_back(ns);
        }
    }

    void reset() {
        local_latencies_ns.clear();
        remote_latencies_ns.clear();
        local_count = remote_count = 0;
        total_local_ns = total_remote_ns = 0;
        bytes_transferred = 0;
        sample_counter = 0;
    }

    static double percentile(std::vector<uint64_t>& sorted_data, double p) {
        if (sorted_data.empty()) return 0.0;
        size_t idx = static_cast<size_t>(p * (sorted_data.size() - 1));
        return static_cast<double>(sorted_data[idx]);
    }

    struct Stats {
        uint64_t local_count = 0;
        uint64_t remote_count = 0;
        double avg_local_us = 0;
        double avg_remote_us = 0;
        double total_local_ms = 0;
        double total_remote_ms = 0;
        uint64_t bytes_transferred = 0;
        double bandwidth_mbps = 0;
        // Local percentiles
        double local_min_us = 0, local_max_us = 0;
        double local_p50_us = 0, local_p90_us = 0, local_p99_us = 0, local_p999_us = 0;
        size_t local_samples = 0;
        // Remote percentiles
        double remote_min_us = 0, remote_max_us = 0;
        double remote_p50_us = 0, remote_p90_us = 0, remote_p99_us = 0, remote_p999_us = 0;
        size_t remote_samples = 0;
        // Overhead ratio
        double remote_overhead_ratio = 0;
    };

    Stats get_stats() {
        Stats s;
        s.local_count = local_count;
        s.remote_count = remote_count;
        s.avg_local_us = local_count > 0 ? static_cast<double>(total_local_ns) / local_count / 1000.0 : 0.0;
        s.avg_remote_us = remote_count > 0 ? static_cast<double>(total_remote_ns) / remote_count / 1000.0 : 0.0;
        s.total_local_ms = total_local_ns / 1e6;
        s.total_remote_ms = total_remote_ns / 1e6;
        s.bytes_transferred = bytes_transferred;
        double total_time_s = (total_local_ns + total_remote_ns) / 1e9;
        s.bandwidth_mbps = total_time_s > 0 ? (bytes_transferred / 1e6) / total_time_s : 0.0;

        if (!local_latencies_ns.empty()) {
            std::vector<uint64_t> sorted_local = local_latencies_ns;
            std::sort(sorted_local.begin(), sorted_local.end());
            s.local_min_us = sorted_local.front() / 1000.0;
            s.local_max_us = sorted_local.back() / 1000.0;
            s.local_p50_us = percentile(sorted_local, 0.50) / 1000.0;
            s.local_p90_us = percentile(sorted_local, 0.90) / 1000.0;
            s.local_p99_us = percentile(sorted_local, 0.99) / 1000.0;
            s.local_p999_us = percentile(sorted_local, 0.999) / 1000.0;
            s.local_samples = local_latencies_ns.size();
        }

        if (!remote_latencies_ns.empty()) {
            std::vector<uint64_t> sorted_remote = remote_latencies_ns;
            std::sort(sorted_remote.begin(), sorted_remote.end());
            s.remote_min_us = sorted_remote.front() / 1000.0;
            s.remote_max_us = sorted_remote.back() / 1000.0;
            s.remote_p50_us = percentile(sorted_remote, 0.50) / 1000.0;
            s.remote_p90_us = percentile(sorted_remote, 0.90) / 1000.0;
            s.remote_p99_us = percentile(sorted_remote, 0.99) / 1000.0;
            s.remote_p999_us = percentile(sorted_remote, 0.999) / 1000.0;
            s.remote_samples = remote_latencies_ns.size();
        }

        if (local_count > 0 && remote_count > 0) {
            double avg_local = static_cast<double>(total_local_ns) / local_count;
            double avg_remote = static_cast<double>(total_remote_ns) / remote_count;
            s.remote_overhead_ratio = avg_remote / avg_local;
        }

        return s;
    }
};

} // namespace danns
