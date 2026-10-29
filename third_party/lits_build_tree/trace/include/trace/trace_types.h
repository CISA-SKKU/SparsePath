#pragma once
// trace/trace_types.h — Binary trace structures
// Extracted from: distributed_bindings.cpp L67-138

#include <cstdint>
#include <cstring>

namespace danns {

#pragma pack(push, 1)

struct TraceFileHeader {       // 24B
    char     magic[4];         // "SRCT" (search) or "CONT" (consolidation)
    uint16_t version;          // 1
    uint16_t trace_level;      // 1=summary, 2=detailed
    uint32_t node_id;
    uint32_t vector_dim;
    uint64_t reserved;
};

struct HopRecord {             // 48B — per-hop summary (level >= 1)
    uint16_t hop_number;
    uint8_t  is_remote_expand; // expanded node is on remote partition
    uint8_t  pad;
    uint32_t degree;           // neighbor list length of expanded node
    uint64_t expanded_vid;
    float    expanded_distance;
    uint16_t n_local_vec_fetch;
    uint16_t n_remote_vec_fetch;
    uint16_t n_inserted;       // new candidates inserted into queue
    uint16_t n_visited_skip;   // neighbors skipped (already visited)
    float    best_distance;    // queue[0].distance after this hop
    float    worst_distance;   // queue.back().distance after this hop
    float    rdma_us;          // RDMA time in this hop (neighbor + vector)
    uint64_t reserved;
};

struct FetchedEntry {          // 24B — per fetched vector (level >= 2)
    uint64_t vid;
    float    distance;         // actual distance
    float    pq_distance;      // PQ approx distance (0.0 if N/A)
    uint8_t  is_remote;
    uint8_t  entered_beam;     // 1=inserted into beam, 0=rejected
    uint8_t  phase;            // 1=approach, 2=converge
    uint8_t  pad[5];
};

struct QueryHeader {           // 16B — per query
    uint32_t query_id;
    uint16_t K;
    uint16_t L;
    uint16_t n_hops;
    uint16_t n_results;
    uint32_t pad;
};

struct ResultEntry {           // 16B — per result in top-K
    uint64_t vid;
    float    distance;
    uint32_t global_id;
};

struct ConsolidationEntryHeader { // 24B — per modified entry
    uint32_t local_loc;
    uint32_t tag;
    uint16_t degree_before;
    uint16_t degree_after;
    uint16_t n_deleted_nbrs;
    uint16_t n_expanded;
    uint8_t  path;             // 1=empty, 2=no_prune, 3=prune
    uint8_t  pad[3];
    uint16_t n_2hop_accesses;
    uint16_t n_vec_accesses;
};

struct AccessRecord {          // 16B — per VID access in consolidation
    uint64_t vid;
    uint8_t  is_remote;
    uint8_t  is_cache_hit;    // 1=cache hit (no RDMA), 0=fetched
    uint8_t  pad[6];
};

#pragma pack(pop)

} // namespace danns
