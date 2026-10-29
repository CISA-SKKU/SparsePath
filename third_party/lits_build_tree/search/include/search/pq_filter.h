#pragma once
// search/pq_filter.h — PQ distance pre-filter (flat array version)
//
// Provides approximate distance filtering using product quantization
// to skip unlikely remote neighbors before RDMA fetch.
//
// codes_ is a flat array indexed by gid: codes_[gid * M .. gid * M + M-1]
// Lock-free reads (search) and disjoint writes (insert new gids).

#include <cstdint>
#include <vector>
#include <string>
#include <cstring>

namespace danns {

class PQFilter {
public:
    PQFilter() = default;

    // Load PQ codebook and all codes into flat array.
    // max_capacity: upper bound on gid range (typically build.max_capacity).
    void load(const std::string& codebook_path,
              const std::string& codes_path,
              uint32_t M_hint, uint32_t dim_hint,
              size_t max_capacity);

    // Direct code access — O(1), lock-free.
    // Returns pointer into flat array, or nullptr if gid out of range.
    const uint8_t* get_code(uint32_t gid) const {
        if (gid >= max_capacity_) return nullptr;
        return codes_.data() + static_cast<size_t>(gid) * M_;
    }

    // Write code for gid (used by encode_and_update and insert path).
    void set_code(uint32_t gid, const uint8_t* code) {
        if (gid < max_capacity_)
            std::memcpy(codes_.data() + static_cast<size_t>(gid) * M_, code, M_);
    }

    // PQ distance computation using precomputed query table.
    // Returns negative if gid out of range.
    float compute_distance(const float* query_table, uint32_t gid) const;

    // Encode vectors with codebook and store codes in flat array.
    void encode_and_update(const uint32_t* gids, const float* vectors,
                           uint32_t count, uint32_t dim);

    // Alpha threshold
    void set_alpha(float alpha) { alpha_ = alpha; }
    float alpha() const { return alpha_; }
    bool enabled() const { return enabled_; }
    uint32_t M() const { return M_; }
    size_t max_capacity() const { return max_capacity_; }

    // PQ codebook accessors (needed by beam search for distance table precomputation)
    const std::vector<float>& centroids() const { return centroids_; }
    const std::vector<uint32_t>& chunk_offsets() const { return chunk_offsets_; }
    const std::vector<uint32_t>& centroid_offsets() const { return centroid_offsets_; }

private:
    bool enabled_ = false;
    float alpha_ = 1.5f;
    uint32_t M_ = 0;
    size_t max_capacity_ = 0;

    // Flat PQ code array: size = max_capacity * M
    std::vector<uint8_t> codes_;

    // PQ codebook
    std::vector<float> centroids_;
    std::vector<uint32_t> chunk_offsets_;
    std::vector<uint32_t> centroid_offsets_;
};

} // namespace danns
