#pragma once
// core/vector_id.h — VectorID encode/decode (partition[16b] + local_loc[48b])
// Extracted from: rdma_colocation_store.h encode_vector_id/decode_partition/decode_local_loc

#include <cstdint>

namespace danns {

static constexpr uint64_t INVALID_VID = 0xFFFFFFFFFFFFFFFFULL;

inline uint64_t encode_vector_id(uint32_t partition, uint64_t local_loc) {
    return (static_cast<uint64_t>(partition) << 48) | (local_loc & 0x0000FFFFFFFFFFFFULL);
}

inline uint32_t decode_partition(uint64_t vid) {
    return static_cast<uint32_t>(vid >> 48);
}

inline uint64_t decode_local_loc(uint64_t vid) {
    return vid & 0x0000FFFFFFFFFFFFULL;
}

} // namespace danns
