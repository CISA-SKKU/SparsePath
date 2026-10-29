// search/src/pq_filter.cpp
// PQ distance pre-filter implementation (flat array version).
// Replaces ShardedPqCache with direct gid-indexed flat array.

#include "search/pq_filter.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <limits>

namespace danns {

// ---------------------------------------------------------------------------
// PQFilter::load
// ---------------------------------------------------------------------------
void PQFilter::load(const std::string& codebook_path,
                    const std::string& codes_path,
                    uint32_t /*M_hint*/,
                    uint32_t /*dim_hint*/,
                    size_t max_capacity) {
    // ---- 1. Load codebook: [M:u32, chunk_offsets:u32[M+1], ksub:u32,
    //          centroids: float[sum(256 * chunk_size_m)]] ----
    {
        std::ifstream f(codebook_path, std::ios::binary);
        if (!f)
            throw std::runtime_error("Cannot open PQ codebook: " + codebook_path);

        uint32_t M;
        f.read(reinterpret_cast<char*>(&M), 4);
        M_ = M;

        if (M > 128)
            throw std::runtime_error("PQ M=" + std::to_string(M) + " exceeds max 128");

        chunk_offsets_.resize(M + 1);
        f.read(reinterpret_cast<char*>(chunk_offsets_.data()), (M + 1) * 4);

        uint32_t ksub;
        f.read(reinterpret_cast<char*>(&ksub), 4);
        if (ksub != 256)
            throw std::runtime_error("PQ ksub must be 256, got " +
                                     std::to_string(ksub));

        // Compute centroid offsets and total size
        centroid_offsets_.resize(M);
        size_t total_floats = 0;
        for (uint32_t m = 0; m < M; m++) {
            centroid_offsets_[m] = total_floats;
            uint32_t chunk_size = chunk_offsets_[m + 1] - chunk_offsets_[m];
            total_floats += static_cast<size_t>(256) * chunk_size;
        }

        centroids_.resize(total_floats);
        f.read(reinterpret_cast<char*>(centroids_.data()),
               total_floats * sizeof(float));

        uint32_t dim = chunk_offsets_[M];
        std::cout << "[PQFilter] codebook loaded: M=" << M
                  << " dim=" << dim << " chunk_sizes=[";
        for (uint32_t m = 0; m < M; m++) {
            if (m > 0) std::cout << ",";
            std::cout << (chunk_offsets_[m + 1] - chunk_offsets_[m]);
        }
        std::cout << "] (" << total_floats * 4 / 1024 << " KB)" << std::endl;
    }

    // ---- 2. Load all_codes into flat array ----
    {
        std::ifstream f(codes_path, std::ios::binary);
        if (!f)
            throw std::runtime_error("Cannot open PQ all_codes: " + codes_path);

        uint32_t N, file_M;
        f.read(reinterpret_cast<char*>(&N), 4);
        f.read(reinterpret_cast<char*>(&file_M), 4);
        if (file_M != M_) {
            throw std::runtime_error(
                "PQ M mismatch: codebook=" + std::to_string(M_) +
                " all_codes=" + std::to_string(file_M));
        }

        // Allocate flat array with max_capacity (>= N)
        max_capacity_ = (max_capacity >= N) ? max_capacity : N;
        codes_.resize(max_capacity_ * M_, 0);

        // Read directly into flat array (first N entries)
        f.read(reinterpret_cast<char*>(codes_.data()),
               static_cast<size_t>(N) * M_);

        size_t mem_mb = (max_capacity_ * M_) / (1024 * 1024);
        std::cout << "[PQFilter] flat array: loaded " << N << " codes"
                  << ", max_capacity=" << max_capacity_
                  << " (" << mem_mb << " MB)" << std::endl;
    }

    enabled_ = true;
    std::cout << "[PQFilter] enabled: alpha=" << alpha_ << std::endl;
}

// ---------------------------------------------------------------------------
// PQFilter::compute_distance
// ---------------------------------------------------------------------------
float PQFilter::compute_distance(const float* query_table, uint32_t gid) const {
    const uint8_t* code = get_code(gid);
    if (!code)
        return -1.0f;

    float dist = 0.0f;
    for (uint32_t m = 0; m < M_; m++) {
        dist += query_table[m * 256 + code[m]];
    }
    return dist;
}

// ---------------------------------------------------------------------------
// PQFilter::encode_and_update
// ---------------------------------------------------------------------------
void PQFilter::encode_and_update(const uint32_t* gids, const float* vectors,
                                 uint32_t count, uint32_t dim) {
    if (!enabled_ || M_ == 0)
        return;

    for (uint32_t i = 0; i < count; i++) {
        const float* vec = vectors + static_cast<size_t>(i) * dim;
        uint8_t code[128];   // PQ max M raised 64->128 (laion dim/4=128); all other PQ buffers are dynamic

        for (uint32_t m = 0; m < M_; m++) {
            uint32_t chunk_start = chunk_offsets_[m];
            uint32_t chunk_size  = chunk_offsets_[m + 1] - chunk_start;
            const float* cent_m  = centroids_.data() + centroid_offsets_[m];
            float best_dist = std::numeric_limits<float>::max();
            uint8_t best_k  = 0;

            for (uint32_t k = 0; k < 256; k++) {
                float d = 0.0f;
                for (uint32_t di = 0; di < chunk_size; di++) {
                    float diff = vec[chunk_start + di] -
                                 cent_m[k * chunk_size + di];
                    d += diff * diff;
                }
                if (d < best_dist) {
                    best_dist = d;
                    best_k = static_cast<uint8_t>(k);
                }
            }
            code[m] = best_k;
        }
        set_code(gids[i], code);
    }
}

} // namespace danns
