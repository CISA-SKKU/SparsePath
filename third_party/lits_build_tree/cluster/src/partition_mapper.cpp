// cluster/src/partition_mapper.cpp
// TODO: Extract full implementation from distributed_bindings.cpp
//   - set_partition_assignments (L685-712)
//   - set_partition_node_mapping (L717-735)
//   - load_tags_file (L6938-6998)
//   - mark/unmark/clear_deleted_vids (L7024-7046)
#include "cluster/partition_mapper.h"
#include <stdexcept>

namespace danns {

void PartitionMapper::set_total_vectors(uint32_t total) {
    total_vectors_ = total;
}

void PartitionMapper::set_partition_assignments(const std::vector<uint32_t>& assignments) {
    partition_assignments_ = assignments;
}

void PartitionMapper::set_partition_node_mapping(const std::vector<uint32_t>& sorted_node_ids) {
    sorted_node_ids_ = sorted_node_ids;
    node_to_partition_.clear();
    partition_to_node_arr_.assign(sorted_node_ids.size(), 0);
    for (uint32_t i = 0; i < sorted_node_ids.size(); i++) {
        node_to_partition_[sorted_node_ids[i]] = i;
        partition_to_node_arr_[i] = sorted_node_ids[i];
    }
}

void PartitionMapper::load_tags(const std::string& /*tags_file*/) {
    // TODO: Extract from distributed_bindings.cpp L6938-6998
}

void PartitionMapper::update_tag_to_local(uint32_t /*partition*/, uint32_t /*tag*/, uint32_t /*local_loc*/) {
    // TODO: Extract
}

void PartitionMapper::update_local_to_tag(uint32_t /*partition*/, uint32_t /*local_loc*/, uint32_t /*tag*/) {
    // TODO: Extract
}

void PartitionMapper::mark_vid_deleted(uint64_t vid) { deleted_vids_.insert(vid); }
void PartitionMapper::unmark_vid_deleted(uint64_t vid) { deleted_vids_.erase(vid); }
void PartitionMapper::clear_deleted_vids() { deleted_vids_.clear(); }
bool PartitionMapper::is_vid_deleted(uint64_t vid) const { return deleted_vids_.count(vid) > 0; }
size_t PartitionMapper::deleted_vids_count() const { return deleted_vids_.size(); }

uint32_t PartitionMapper::node_id_to_partition_id(uint32_t node_id) const {
    auto it = node_to_partition_.find(node_id);
    return it != node_to_partition_.end() ? it->second : node_id;
}

uint32_t PartitionMapper::get_owner(uint32_t global_id) const {
    if (global_id < partition_assignments_.size())
        return partition_assignments_[global_id];
    return 0;
}

uint32_t PartitionMapper::get_owner_node(uint32_t global_id) const {
    return partition_id_to_node_id(get_owner(global_id));
}

bool PartitionMapper::is_local(uint32_t global_id, uint32_t my_partition) const {
    return get_owner(global_id) == my_partition;
}


void PartitionMapper::set_centroid_local_locs(const std::unordered_map<uint32_t, uint32_t>& locs, uint32_t max_points) {
    centroid_local_locs_ = locs;
    max_points_ = max_points;
}

void PartitionMapper::set_frozen_local_loc(uint32_t loc) { frozen_local_loc_ = loc; }
void PartitionMapper::set_frozen_canonical_vid(uint64_t vid) { frozen_canonical_vid_ = vid; }

} // namespace danns
