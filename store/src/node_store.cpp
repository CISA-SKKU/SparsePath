// store/src/node_store.cpp — NodeStore implementation
// Extracted from: distributed_bindings.cpp load_colocation(), exchange_coloc_mr_with_peer()
#include "store/node_store.h"
#include "rdma/common.h"
#include <fstream>
#include <stdexcept>

namespace danns {

void NodeStore::load(const std::string& coloc_file,
                     const std::string& global_to_vectorid_file,
                     const std::unordered_map<uint32_t, std::string>& tag_to_local_files,
                     const std::unordered_map<uint32_t, std::string>& local_to_tag_files,
                     uint32_t partition_id,
                     const rdma::ColocationConfig& config,
                     uint32_t total_vectors) {
    partition_id_ = partition_id;
    config_ = config;
    total_vectors_ = total_vectors;

    // Load colocation binary
    coloc_store_ = std::make_unique<rdma::ColocationStore>(config_);
    coloc_store_->load(coloc_file);

    // Load global_to_vectorid mapping
    {
        std::ifstream f(global_to_vectorid_file, std::ios::binary);
        if (!f) throw std::runtime_error("Failed to open: " + global_to_vectorid_file);
        uint32_t file_total;
        f.read(reinterpret_cast<char*>(&file_total), sizeof(uint32_t));
        global_to_vectorid_.resize(file_total);
        f.read(reinterpret_cast<char*>(global_to_vectorid_.data()),
               file_total * sizeof(uint64_t));
        if (!f) throw std::runtime_error("Failed to read global_to_vectorid");
    }

    // Load tag_to_local lookup tables (per partition)
    for (auto& [p, filepath] : tag_to_local_files) {
        std::ifstream f(filepath, std::ios::binary);
        if (!f) throw std::runtime_error("Failed to open tag_to_local: " + filepath);
        uint32_t max_tag;
        f.read(reinterpret_cast<char*>(&max_tag), sizeof(uint32_t));
        tag_to_local_[p].resize(max_tag + 1, UINT32_MAX);
        f.read(reinterpret_cast<char*>(tag_to_local_[p].data()),
               (max_tag + 1) * sizeof(uint32_t));
        if (!f) throw std::runtime_error("Failed to read tag_to_local for partition " + std::to_string(p));
        rdma::Logger::info("Loaded tag_to_local for partition %u: max_tag=%u", p, max_tag);
    }

    // Load local_to_tag lookup tables (per partition).
    // Flat vector<vector> indexed by partition_id (dense 0..N-1).
    for (auto& [p, filepath] : local_to_tag_files) {
        std::ifstream f(filepath, std::ios::binary);
        if (!f) throw std::runtime_error("Failed to open local_to_tag: " + filepath);
        uint32_t capacity;
        f.read(reinterpret_cast<char*>(&capacity), sizeof(uint32_t));
        if (p >= local_to_tag_.size()) local_to_tag_.resize(p + 1);
        local_to_tag_[p].resize(capacity, 0);
        f.read(reinterpret_cast<char*>(local_to_tag_[p].data()),
               capacity * sizeof(uint32_t));
        if (!f) throw std::runtime_error("Failed to read local_to_tag for partition " + std::to_string(p));
        rdma::Logger::info("Loaded local_to_tag for partition %u: capacity=%u", p, capacity);
    }

    rdma::Logger::info("NodeStore loaded: partition=%u, entries=%u, total_vectors=%u",
                       partition_id, config_.num_entries, total_vectors);
}

void NodeStore::exchange_mr_with_peer(uint32_t peer_id,
                                       uint32_t my_node_id,
                                       std::shared_ptr<rdma::RDMAEndpoint> endpoint,
                                       rdma::FetchMode fetch_mode,
                                       bool exchange_counter,
                                       bool exchange_claim) {
    node_id_ = my_node_id;  // Store node ID for beam search
    if (!coloc_store_)
        throw std::runtime_error("Colocation not loaded. Call load() first.");

    // Register local colocation data as RDMA MR
    ibv_mr* local_mr = endpoint->register_memory(
        const_cast<char*>(coloc_store_->get_entry(0)),
        coloc_store_->total_size(),
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE);

    rdma::RemoteMemoryRegion local_mr_info(
        reinterpret_cast<uint64_t>(coloc_store_->get_entry(0)),
        local_mr->rkey,
        coloc_store_->total_size());

    // Exchange MR info (ordered by node_id to avoid deadlock)
    rdma::RemoteMemoryRegion remote_mr;
    if (my_node_id < peer_id) {
        endpoint->send_mr_info(local_mr_info);
        remote_mr = endpoint->recv_mr_info();
    } else {
        remote_mr = endpoint->recv_mr_info();
        endpoint->send_mr_info(local_mr_info);
    }

    auto fetcher = std::make_unique<rdma::ColocationFetcher>(endpoint, config_);
    fetcher->set_remote_mr(remote_mr);
    fetcher->set_fetch_mode(fetch_mode);

    {
        std::lock_guard<std::mutex> lock(mutex_);
        // Re-exchanging with the same peer: deregister the stale MR before
        // overwriting, otherwise its pinned registration leaks.
        auto existing = local_mrs_.find(peer_id);
        if (existing != local_mrs_.end() && existing->second) {
            ibv_dereg_mr(existing->second);
        }
        local_mrs_[peer_id] = local_mr;
        remote_mrs_[peer_id] = remote_mr;
        fetchers_[peer_id] = std::move(fetcher);
    }

    rdma::Logger::info("MR exchanged with peer %u", peer_id);

    // Vector-counter MR exchange (LoadBalancer, Stage B): a second send/recv
    // pair appended in the SAME my_node_id < peer_id order as the colocation
    // exchange above — both sides must take this branch symmetrically
    // (driven by the shared load_balancer.enabled config) or the handshake
    // deadlocks. The counter is remote-READ-only: peers may read it, only
    // this node writes it (no remote write, no atomics).
    if (exchange_counter) {
        ibv_mr* counter_mr = endpoint->register_memory(
            &vector_count_, sizeof(uint64_t),
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ);
        auto scratch = std::make_unique<CounterScratch>();
        ibv_mr* scratch_mr = endpoint->register_memory(
            &scratch->buf, sizeof(uint64_t),
            IBV_ACCESS_LOCAL_WRITE);

        rdma::RemoteMemoryRegion counter_info(
            reinterpret_cast<uint64_t>(&vector_count_),
            counter_mr->rkey,
            sizeof(uint64_t));

        rdma::RemoteMemoryRegion remote_counter;
        if (my_node_id < peer_id) {
            endpoint->send_mr_info(counter_info);
            remote_counter = endpoint->recv_mr_info();
        } else {
            remote_counter = endpoint->recv_mr_info();
            endpoint->send_mr_info(counter_info);
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            counter_local_mrs_[peer_id] = counter_mr;
            counter_scratch_mrs_[peer_id] = scratch_mr;
            counter_remote_mrs_[peer_id] = remote_counter;
            counter_endpoints_[peer_id] = endpoint;
            counter_scratch_[peer_id] = std::move(scratch);
        }

        rdma::Logger::info("Counter MR exchanged with peer %u (remote addr=0x%lx)",
                           peer_id, remote_counter.addr);
    }

    // Claim-counter exchange (work stealing). Node 0 hosts the counter; every
    // node registers a slot so the handshake stays symmetric (both sides must
    // take this branch or the ordered send/recv below deadlocks -- same rule as
    // the counter exchange above). Only the host's region is ever incremented.
    if (exchange_claim) {
        if (!claim_counter_) {
            void* mem = nullptr;
            if (posix_memalign(&mem, 64, 64) != 0 || !mem)
                throw std::runtime_error("claim counter alloc failed");
            claim_counter_ = static_cast<uint64_t*>(mem);
            *claim_counter_ = 0;
        }
        // Register the SAME counter once per peer endpoint. An rkey is only
        // valid on the PD that minted it, and each endpoint carries its own
        // context/PD, so handing every peer the first endpoint's rkey makes
        // every other peer's atomic fail with REMOTE_ACCESS_ERROR. Multiple MRs
        // over one buffer are legal and this is 8 bytes.
        claim_counter_mr_ = endpoint->register_memory(
            claim_counter_, sizeof(uint64_t),
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
            IBV_ACCESS_REMOTE_ATOMIC);
        claim_local_mrs_[peer_id] = claim_counter_mr_;
        rdma::RemoteMemoryRegion mine(
            reinterpret_cast<uint64_t>(claim_counter_),
            claim_counter_mr_->rkey, sizeof(uint64_t));

        rdma::RemoteMemoryRegion theirs;
        if (my_node_id < peer_id) {
            endpoint->send_mr_info(mine);
            theirs = endpoint->recv_mr_info();
        } else {
            theirs = endpoint->recv_mr_info();
            endpoint->send_mr_info(mine);
        }

        // Host = lowest node id in the cluster. Node ids start at 1, so a node
        // that never meets a lower peer is the host and uses its own counter.
        {
            std::lock_guard<std::mutex> lock(mutex_);
            claim_endpoints_[peer_id] = endpoint;
            claim_peer_regions_[peer_id] = theirs;
            if (!claim_host_valid_ || peer_id < claim_host_ ||
                my_node_id < claim_host_) {
                if (peer_id < my_node_id) { claim_host_ = peer_id; claim_remote_ = theirs; }
                else if (!claim_host_valid_ || my_node_id < claim_host_) {
                    claim_host_ = my_node_id;
                    claim_remote_ = mine;
                }
                claim_host_valid_ = true;
            }
        }
        rdma::Logger::info("Claim MR exchanged with peer %u (host=%u)",
                           peer_id, claim_host_);
    }
}

void NodeStore::init_claim_slots(int num_threads) {
    if (num_threads <= 0) return;
    if (static_cast<int>(claim_slots_.size()) >= num_threads) return;
    if (claim_slots_mr_) {
        // Re-registering means the vector may move; drop the old MR first, via
        // the same endpoint that created it (PD ownership).
        auto oit = claim_endpoints_.find(claim_host_);
        if (oit != claim_endpoints_.end() && oit->second)
            oit->second->get_context()->deregister_memory(claim_slots_mr_);
        claim_slots_mr_ = nullptr;
    }
    claim_slots_.assign(static_cast<size_t>(num_threads), 0);
    // The host increments its own counter locally, so it needs no landing slots.
    if (claim_host_ == node_id_) return;
    // Register with the HOST's endpoint specifically. An MR belongs to the PD of
    // the context that registered it, and each endpoint carries its own context,
    // so an lkey minted on one peer's endpoint is rejected on another's QP with
    // LOCAL_PROTECTION_ERROR -- which is exactly what registering against an
    // arbitrary endpoint produced.
    auto it = claim_endpoints_.find(claim_host_);
    if (it == claim_endpoints_.end() || !it->second) return;
    claim_slots_mr_ = it->second->register_memory(
        claim_slots_.data(), claim_slots_.size() * sizeof(uint64_t),
        IBV_ACCESS_LOCAL_WRITE);
}

void NodeStore::reset_claim_counter(uint64_t value) {
    if (claim_counter_) *claim_counter_ = value;
}

uint64_t NodeStore::claim_next(int thread_id, uint64_t n) {
    if (!claim_host_valid_ || !claim_counter_) return UINT64_MAX;
    // Host increments locally: an RDMA atomic to self would still be correct but
    // costs a round trip for what is a single uncontended cache line here.
    if (claim_host_ == node_id_) {
        return __atomic_fetch_add(claim_counter_, n, __ATOMIC_RELAXED);
    }
    auto it = claim_endpoints_.find(claim_host_);
    if (it == claim_endpoints_.end()) return UINT64_MAX;
    // One 8-byte slot per thread, 8 apart, inside the registered slot array.
    if (!claim_slots_mr_ || claim_slots_.empty() || thread_id < 0 ||
        static_cast<size_t>(thread_id) >= claim_slots_.size())
        return UINT64_MAX;
    uint64_t* slot = &claim_slots_[static_cast<size_t>(thread_id)];
    return it->second->get_connection()->atomic_fetch_add(
        slot, claim_slots_mr_, claim_remote_, n);
}

// Parallel insert: the per-peer slot mutex protects the landing buffer
// and serializes counter reads against EACH OTHER. The QP/CQ is the same
// endpoint as this peer's ColocationFetcher — callers inside parallel
// regions must also hold the same per-peer endpoint lock as fetcher
// users (insert_batch's fetcher_mutexes), or completions cross-consume.
// The maps are setup-time-written (counter MR exchange, under mutex_)
// and runtime-read-only, so concurrent find() is safe.
uint64_t NodeStore::read_peer_vector_count(uint32_t peer_id) {
    auto ep_it = counter_endpoints_.find(peer_id);
    auto mr_it = counter_remote_mrs_.find(peer_id);
    auto sc_it = counter_scratch_mrs_.find(peer_id);
    auto slot_it = counter_scratch_.find(peer_id);
    if (ep_it == counter_endpoints_.end() || mr_it == counter_remote_mrs_.end() ||
        sc_it == counter_scratch_mrs_.end() || slot_it == counter_scratch_.end()) {
        throw std::runtime_error("No counter MR exchanged with peer " +
                                 std::to_string(peer_id));
    }
    auto& slot = *slot_it->second;
    std::lock_guard<std::mutex> lock(slot.mtx);
    ep_it->second->rdma_read(&slot.buf, sizeof(uint64_t),
                             sc_it->second, mr_it->second, /*blocking=*/true);
    return slot.buf;
}

void NodeStore::set_fetch_mode(rdma::FetchMode mode) {
    fetch_mode_ = mode;
    for (auto& [_, fetcher] : fetchers_) {
        fetcher->set_fetch_mode(mode);
    }
}

rdma::ColocationFetcher* NodeStore::fetcher(uint32_t peer_id) {
    auto it = fetchers_.find(peer_id);
    return it != fetchers_.end() ? it->second.get() : nullptr;
}

void NodeStore::cleanup() {
    // Deregister the colocation MRs while the endpoint/context (held alive by
    // the fetchers) still exists, and BEFORE freeing the backing memory in
    // coloc_store_.reset(). Otherwise a late remote READ reads reused heap, and
    // clearing fetchers first could destroy the PD before dereg.
    for (auto& [peer_id, mr] : local_mrs_) {
        (void)peer_id;
        if (mr) ibv_dereg_mr(mr);
    }
    local_mrs_.clear();
    fetchers_.clear();
    remote_mrs_.clear();
    counter_local_mrs_.clear();
    counter_scratch_mrs_.clear();
    counter_remote_mrs_.clear();
    counter_endpoints_.clear();
    counter_scratch_.clear();
    coloc_store_.reset();
}

} // namespace danns
