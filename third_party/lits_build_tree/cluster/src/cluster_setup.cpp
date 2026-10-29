// cluster/src/cluster_setup.cpp
// TODO: Extract from distributed_bindings.cpp
//   - init_rdma()          (L738-742)
//   - connect_to_peer()    (L747-793)
//   - exchange_mr_with_peer() (L797-866)
#include "cluster/cluster_setup.h"
#include "rdma/common.h"

namespace danns {

ClusterSetup::ClusterSetup(uint32_t node_id, uint32_t total_nodes,
                           const std::string& rdma_device, int gid_index)
    : node_id_(node_id), total_nodes_(total_nodes),
      rdma_device_(rdma_device), gid_index_(gid_index) {}

void ClusterSetup::init() {
    rdma::Logger::info("ClusterSetup: node=%u/%u, device=%s",
                       node_id_, total_nodes_, rdma_device_.c_str());
}

void ClusterSetup::connect_to_peer(uint32_t peer_id, const std::string& peer_host, uint16_t peer_port) {
    auto ep = std::make_shared<rdma::RDMAEndpoint>(rdma_device_, gid_index_);

    if (node_id_ < peer_id) {
        ep->listen_and_connect(peer_port);
    } else {
        ep->connect_to(peer_host, peer_port);
    }

    {
        std::lock_guard<std::mutex> lock(connect_mutex_);
        endpoints_[peer_id] = ep;
    }

    rdma::Logger::info("Connected to peer %u at %s:%u", peer_id, peer_host.c_str(), peer_port);
}

std::shared_ptr<rdma::RDMAEndpoint> ClusterSetup::endpoint(uint32_t peer_id) {
    std::lock_guard<std::mutex> lock(connect_mutex_);
    auto it = endpoints_.find(peer_id);
    return it != endpoints_.end() ? it->second : nullptr;
}

void ClusterSetup::cleanup() {
    endpoints_.clear();
}

} // namespace danns
