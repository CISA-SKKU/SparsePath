#pragma once
// cluster/partition_mapper.h — Partition/node ID mapping and tag management
// Extracted from: distributed_bindings.cpp L578-735, L6938-7046

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>

namespace danns {

// Upper bound (exclusive) on worker node ids the search hot path can address.
// beam_search.cpp sizes every per-owner scratch/dispatch array with this
// constant, and *silently drops* neighbors whose owner id is >= the bound
// (`if (nbr_owner < MAX_OWNERS)` — no counter, recall just decays; see
// beam_search.cpp:858). Node ids are 1-based (nodes.yaml), so a 16-node
// cluster (ids 1..16) needs 17. set_partition_node_mapping() throws when a
// configured node id exceeds this bound — that guard turns the silent recall
// loss into a hard startup error. (16p socket-split fix, 2026-08-11)
inline constexpr uint32_t kMaxOwnerNodes = 17;

class PartitionMapper {
public:
    PartitionMapper() = default;

    // Set total vector count
    void set_total_vectors(uint32_t total);

    // Set partition assignments for all vectors
    void set_partition_assignments(const std::vector<uint32_t>& assignments);

    // Set partition_id <-> node_id mapping
    void set_partition_node_mapping(const std::vector<uint32_t>& sorted_node_ids);

    // Load tags file for global_id <-> local_loc mapping
    void load_tags(const std::string& tags_file);

    // Update individual tag/local entries
    void update_tag_to_local(uint32_t partition, uint32_t tag, uint32_t local_loc);
    void update_local_to_tag(uint32_t partition, uint32_t local_loc, uint32_t tag);

    // Deletion management (VID-based, for traversal filtering)
    void mark_vid_deleted(uint64_t vid);
    void unmark_vid_deleted(uint64_t vid);
    void clear_deleted_vids();
    bool is_vid_deleted(uint64_t vid) const;
    size_t deleted_vids_count() const;
    const std::unordered_set<uint64_t>& deleted_vids() const { return deleted_vids_; }

    // Deletion management (tag-based, for result filtering)
    void mark_tag_deleted(uint32_t tag) { deleted_tags_.insert(tag); }
    void unmark_tag_deleted(uint32_t tag) { deleted_tags_.erase(tag); }
    void clear_deleted_tags() { deleted_tags_.clear(); }
    bool is_tag_deleted(uint32_t tag) const { return deleted_tags_.count(tag) > 0; }
    size_t deleted_tags_count() const { return deleted_tags_.size(); }
    const std::unordered_set<uint32_t>& deleted_tags() const { return deleted_tags_; }

    // Lookups
    // Hot path (called per-neighbor in beam_search): inline + flat array.
    // partition_to_node_arr_ is dense 0..N-1 (built in set_partition_node_mapping);
    // out-of-range fallback returns partition_id (single-node case).
    inline uint32_t partition_id_to_node_id(uint32_t partition_id) const {
        return partition_id < partition_to_node_arr_.size()
            ? partition_to_node_arr_[partition_id]
            : partition_id;
    }
    // Bounds for VID sanity checks in the search hot path (0 = unset/skip).
    // A torn/garbage VID decoded past these bounds would index past the
    // local mmap or a peer's MR (RDMA read -> QP error).
    inline uint32_t num_partitions() const {
        return static_cast<uint32_t>(partition_to_node_arr_.size());
    }
    inline uint64_t max_local_locs() const { return max_local_locs_; }
    void set_max_local_locs(uint64_t v) { max_local_locs_ = v; }
    uint32_t node_id_to_partition_id(uint32_t node_id) const;
    uint32_t get_owner(uint32_t global_id) const;
    uint32_t get_owner_node(uint32_t global_id) const;
    bool is_local(uint32_t global_id, uint32_t my_partition) const;

    // Entry points
    void set_centroid_local_locs(const std::unordered_map<uint32_t, uint32_t>& locs, uint32_t max_points);
    void set_frozen_local_loc(uint32_t loc);
    void set_frozen_canonical_vid(uint64_t vid);

    const std::unordered_map<uint32_t, uint32_t>& centroid_local_locs() const { return centroid_local_locs_; }
    uint32_t frozen_local_loc() const { return frozen_local_loc_; }
    uint64_t frozen_canonical_vid() const { return frozen_canonical_vid_; }
    uint32_t max_points() const { return max_points_; }
    uint32_t total_vectors() const { return total_vectors_; }

private:
    uint32_t total_vectors_ = 0;
    std::vector<uint32_t> partition_assignments_;
    std::vector<uint32_t> sorted_node_ids_;
    std::unordered_map<uint32_t, uint32_t> node_to_partition_;
    // Flat array for hot path partition_id_to_node_id lookup.
    // Indices are dense partition_ids (0..N_partitions-1).
    std::vector<uint32_t> partition_to_node_arr_;

    std::unordered_set<uint64_t> deleted_vids_;
    std::unordered_set<uint32_t> deleted_tags_;

    std::unordered_map<uint32_t, uint32_t> centroid_local_locs_;
    uint32_t frozen_local_loc_ = UINT32_MAX;
    uint64_t frozen_canonical_vid_ = 0;
    uint32_t max_points_ = 0;
    uint64_t max_local_locs_ = 0;
};

} // namespace danns
