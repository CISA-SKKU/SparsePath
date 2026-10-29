#pragma once
// cluster/cluster_setup.h — RDMA connection bootstrap
// Extracted from: distributed_bindings.cpp L738-866 (connect_to_peer, init_rdma)

#include <cstdint>
#include <string>
#include <memory>
#include <unordered_map>
#include <mutex>

#include "rdma/connection.h"

namespace danns {

class ClusterSetup {
public:
    ClusterSetup(uint32_t node_id, uint32_t total_nodes,
                 const std::string& rdma_device = "", int gid_index = 1);

    // Initialize RDMA context
    void init();

    // Connect to peer node (TCP bootstrap + QP setup)
    void connect_to_peer(uint32_t peer_id, const std::string& peer_host, uint16_t peer_port);

    // Get endpoint for a connected peer
    std::shared_ptr<rdma::RDMAEndpoint> endpoint(uint32_t peer_id);

    // Get all endpoints
    const std::unordered_map<uint32_t, std::shared_ptr<rdma::RDMAEndpoint>>& endpoints() const {
        return endpoints_;
    }

    uint32_t node_id() const { return node_id_; }
    uint32_t total_nodes() const { return total_nodes_; }

    void cleanup();

private:
    uint32_t node_id_;
    uint32_t total_nodes_;
    std::string rdma_device_;
    int gid_index_;

    std::unordered_map<uint32_t, std::shared_ptr<rdma::RDMAEndpoint>> endpoints_;
    std::mutex connect_mutex_;
};

} // namespace danns
