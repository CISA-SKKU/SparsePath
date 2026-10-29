#pragma once
// store/node_store.h — Local colocation store + peer fetcher management
// Extracted from: distributed_bindings.cpp (MultiPeerGraphStore colocation methods)
//
// NodeStore owns:
//   - Local ColocationStore (my partition's data)
//   - Per-peer ColocationFetcher
//   - Per-peer RDMA MR registrations
//   - Global-to-VectorID mapping
//   - Tag/local_loc lookup tables

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>

#include "store/colocation_store.h"
#include "rdma/connection.h"

namespace danns {

class NodeStore {
public:
    NodeStore() = default;

    // Load local colocation binary + metadata
    void load(const std::string& coloc_file,
              const std::string& global_to_vectorid_file,
              const std::unordered_map<uint32_t, std::string>& tag_to_local_files,
              const std::unordered_map<uint32_t, std::string>& local_to_tag_files,
              uint32_t partition_id,
              const rdma::ColocationConfig& config,
              uint32_t total_vectors);

    // Exchange colocation MR with peer for RDMA access.
    // exchange_counter additionally exchanges the 8-byte vector-count MR
    // (LoadBalancer, Stage B) inside the same ordered handshake — both
    // sides MUST pass the same value (driven by load_balancer.enabled).
    void exchange_mr_with_peer(uint32_t peer_id,
                               uint32_t my_node_id,
                               std::shared_ptr<rdma::RDMAEndpoint> endpoint,
                               rdma::FetchMode fetch_mode,
                               bool exchange_counter = false,
                               bool exchange_claim = false);

    // --- Distributed work-stealing claim counter ---
    //
    // One 64-bit counter, hosted by a single node, that every worker thread on
    // every node increments with an RDMA atomic to take the next query id.
    // Balancing query COUNTS is not enough: with 12,500 queries each, one node
    // still needs 5,228 comparisons per query while another needs 3,456
    // (measured 2026-07-28), and cluster wall-clock is set by the slowest, so
    // ~18% of throughput is lost to work skew that cannot be predicted before
    // the search runs. RED-ANNS solves this the same way (rdma_fetch_add on a
    // shared counter); its ranks finish within 0.0001 s of each other.
    //
    // Registered separately from the colocation MR because that one is not
    // mapped with IBV_ACCESS_REMOTE_ATOMIC, and because writing a counter into
    // the data region would corrupt it.
    void reset_claim_counter(uint64_t value = 0);
    uint64_t claim_counter_value() const {
        return claim_counter_ ? *claim_counter_ : 0;
    }
    bool claim_ready() const { return claim_host_valid_; }
    uint32_t claim_host() const { return claim_host_; }
    // Take the next query id. Returns UINT64_MAX when the counter is not set up.
    uint64_t claim_next(int thread_id, uint64_t n = 1);

    // Set fetch mode for all fetchers
    void set_fetch_mode(rdma::FetchMode mode);

    // --- Local store access ---
    rdma::ColocationStore* local_store() { return coloc_store_.get(); }
    const rdma::ColocationStore* local_store() const { return coloc_store_.get(); }

    // --- Peer fetcher access ---
    rdma::ColocationFetcher* fetcher(uint32_t peer_id);

    // Enumerate registered peer IDs (used to pre-populate per-peer data
    // structures before entering parallel regions — avoids lazy-insert races).
    std::vector<uint32_t> peer_ids() const {
        std::vector<uint32_t> ids;
        ids.reserve(fetchers_.size());
        for (const auto& kv : fetchers_) ids.push_back(kv.first);
        return ids;
    }

    // --- Metadata ---
    uint32_t partition_id() const { return partition_id_; }
    uint32_t node_id() const { return node_id_; }
    // Explicit setter for single-node / no-peer scenarios where
    // exchange_mr_with_peer() is never called and would otherwise leave
    // node_id_ at the default 0, breaking consolidation's local/remote
    // dispatch (del_owner == my_node check).
    void set_node_id(uint32_t nid) { node_id_ = nid; }
    uint32_t total_vectors() const { return total_vectors_; }
    const rdma::ColocationConfig& config() const { return config_; }
    rdma::FetchMode fetch_mode() const { return fetch_mode_; }
    const std::vector<uint64_t>& global_to_vectorid() const { return global_to_vectorid_; }

    // Tag/local_loc lookup tables
    // local_to_tag_ uses flat vector<vector> (indexed by dense partition_id)
    // for hash-free hot-path access in beam_search. tag_to_local_ kept as map
    // (only used in cold paths).
    const std::unordered_map<uint32_t, std::vector<uint32_t>>& tag_to_local() const { return tag_to_local_; }
    const std::vector<std::vector<uint32_t>>& local_to_tag() const { return local_to_tag_; }
    std::unordered_map<uint32_t, std::vector<uint32_t>>& tag_to_local_mut() { return tag_to_local_; }
    std::vector<std::vector<uint32_t>>& local_to_tag_mut() { return local_to_tag_; }

    // --- RDMA remote MR access (for per-thread QP) ---
    const rdma::RemoteMemoryRegion* remote_mr(uint32_t peer_id) const {
        auto it = remote_mrs_.find(peer_id);
        return it != remote_mrs_.end() ? &it->second : nullptr;
    }

    // --- Per-node vector counter MR (LoadBalancer / live redirect) ---
    // 8-byte live vector count of THIS node, exposed to peers for one-sided
    // READ. Python pushes slot_manager.active_count at insert-batch
    // boundaries; DURING a live-redirect batch two local threads mutate it
    // (alloc server fetch_add on grants-in, insert threads fetch_sub on
    // redirects-out) — hence std::atomic with local FAA on both sides. No
    // cross-machine atomics: peers only ever READ (aligned 8B, torn-free).
    void set_local_vector_count(uint64_t n) {
        vector_count_.store(n, std::memory_order_relaxed);
    }
    uint64_t local_vector_count() const {
        return vector_count_.load(std::memory_order_relaxed);
    }
    void counter_fetch_add(uint64_t n) {
        vector_count_.fetch_add(n, std::memory_order_relaxed);
    }
    void counter_fetch_sub(uint64_t n) {
        vector_count_.fetch_sub(n, std::memory_order_relaxed);
    }

    // Pre-size the tag/loc tables for a live-redirect window so every
    // in-window update (alloc server thread on my partition, insert threads on
    // replicated remote partitions) is an in-place aligned store — no
    // resize/rehash may run concurrently with readers. Main thread only,
    // before the lb-counters barrier.
    void presize_tables_for_redirect(uint32_t max_tag, uint32_t num_partitions) {
        if (local_to_tag_.size() < num_partitions)
            local_to_tag_.resize(num_partitions);
        for (uint32_t p = 0; p < num_partitions; p++) {
            auto& t2l = tag_to_local_[p];  // creates key if absent (pre-window)
            if (t2l.size() < static_cast<size_t>(max_tag) + 1)
                t2l.resize(static_cast<size_t>(max_tag) + 1, UINT32_MAX);
        }
        // Own partition's l2t must cover the full colocation capacity (any
        // reserve slot can be granted); replicas grow lazily on the main
        // thread only.
        auto& my_l2t = local_to_tag_[partition_id_];
        if (my_l2t.size() < config_.num_entries)
            my_l2t.resize(config_.num_entries, 0);
    }
    // Peers whose counter MR was exchanged (counter reads possible).
    std::vector<uint32_t> counter_peer_ids() const {
        std::vector<uint32_t> ids;
        ids.reserve(counter_remote_mrs_.size());
        for (const auto& kv : counter_remote_mrs_) ids.push_back(kv.first);
        return ids;
    }
    // One-sided 8-byte RDMA READ of a peer's counter (blocking).
    // Throws if the counter MR was not exchanged with this peer.
    uint64_t read_peer_vector_count(uint32_t peer_id);

    // --- LOCAL_MMAP peer data (room-at-the-top experiment) ---
    // When peer partitions are cross-deployed and mmap'd locally, register the
    // base pointer here. consolidate fetch sites check this before falling
    // back to RDMA paths — non-null pointer means "use memcpy from local RAM".
    void set_local_partition_data(uint32_t peer_id, const char* data) {
        local_partition_data_[peer_id] = data;
    }
    const char* local_data(uint32_t peer_id) const {
        auto it = local_partition_data_.find(peer_id);
        return it != local_partition_data_.end() ? it->second : nullptr;
    }

    // Per-thread RDMA state for multi-QP search
    struct ThreadRDMAState {
        std::shared_ptr<rdma::RDMAConnection> conn;
        ibv_cq* send_cq = nullptr;
        ibv_cq* recv_cq = nullptr;
        std::vector<float> fetch_buffer;
        ibv_mr* fetch_buffer_mr = nullptr;
        std::vector<char> neighbor_fetch_buffer;
        ibv_mr* neighbor_fetch_mr = nullptr;
    };

    // Thread states: peer_id -> vector of per-thread states
    using ThreadStateMap = std::unordered_map<uint32_t, std::vector<ThreadRDMAState>>;
    const ThreadStateMap& thread_states() const { return thread_states_; }
    ThreadStateMap& thread_states_mut() { return thread_states_; }
    uint32_t num_thread_qps() const { return num_thread_qps_; }
    void set_num_thread_qps(uint32_t n) { num_thread_qps_ = n; }

    // Cleanup (explicit destruction order for RDMA resources)
    void cleanup();

private:
    uint32_t partition_id_ = 0;
    uint32_t node_id_ = 0;
    uint32_t total_vectors_ = 0;
    rdma::ColocationConfig config_;

    std::unique_ptr<rdma::ColocationStore> coloc_store_;

    // Per-peer state
    std::unordered_map<uint32_t, std::unique_ptr<rdma::ColocationFetcher>> fetchers_;
    std::unordered_map<uint32_t, ibv_mr*> local_mrs_;
    std::unordered_map<uint32_t, rdma::RemoteMemoryRegion> remote_mrs_;

    // Per-node vector counter MR state (LoadBalancer, Stage B).
    // Each peer endpoint owns its own PD, so the count buffer is registered
    // once per peer; the read-landing slot is per peer.
    // counter_endpoints_ keeps the endpoint alive/reachable for read_peer_vector_count.
    std::atomic<uint64_t> vector_count_{0};  // my live count (remote READ target)
    // Per-peer landing slot for counter READs (parallel insert): own
    // 8-byte buffer + mutex per peer. The slot mutex protects the buffer
    // and serializes counter reads against each other ONLY — the QP/CQ is
    // shared with the peer's ColocationFetcher, so parallel-region callers
    // must also hold the fetcher-side per-peer lock (see the note on
    // read_peer_vector_count). Slots are created at counter-MR-exchange
    // time (setup, single-threaded); runtime only find()s — no structural
    // race.
    struct CounterScratch {
        uint64_t buf = 0;
        std::mutex mtx;
    };
    std::unordered_map<uint32_t, std::unique_ptr<CounterScratch>> counter_scratch_;
    std::unordered_map<uint32_t, ibv_mr*> counter_local_mrs_;
    std::unordered_map<uint32_t, ibv_mr*> counter_scratch_mrs_;
    std::unordered_map<uint32_t, rdma::RemoteMemoryRegion> counter_remote_mrs_;
    std::unordered_map<uint32_t, std::shared_ptr<rdma::RDMAEndpoint>> counter_endpoints_;

    // Claim counter (work stealing). claim_counter_ is 64-byte aligned so the
    // hosting node's local increments do not false-share with anything else.
    uint64_t* claim_counter_ = nullptr;
    ibv_mr* claim_counter_mr_ = nullptr;
    rdma::RemoteMemoryRegion claim_remote_;
    uint32_t claim_host_ = 0;
    bool claim_host_valid_ = false;
    // Per-thread 8-byte landing slots for the atomic reply, one cache line each.
    std::vector<uint64_t> claim_slots_;
    ibv_mr* claim_slots_mr_ = nullptr;
    std::unordered_map<uint32_t, std::shared_ptr<rdma::RDMAEndpoint>> claim_endpoints_;
    std::unordered_map<uint32_t, rdma::RemoteMemoryRegion> claim_peer_regions_;
    // One MR per peer endpoint over the same counter (rkey is PD-scoped).
    std::unordered_map<uint32_t, ibv_mr*> claim_local_mrs_;

public:
    // Allocate one 8-byte landing slot per search thread for the atomic reply.
    // Called once after the MR exchange, before the search region starts.
    void init_claim_slots(int num_threads);

private:

    // peer_id -> base pointer of locally-mmap'd peer partition .coloc.
    // Populated only when fetch_mode_ == LOCAL_MMAP. Empty in normal RDMA runs.
    std::unordered_map<uint32_t, const char*> local_partition_data_;

    // Metadata
    std::vector<uint64_t> global_to_vectorid_;
    std::unordered_map<uint32_t, std::vector<uint32_t>> tag_to_local_;
    // Flat array indexed by partition_id (dense 0..N-1).
    // local_to_tag_[part][local_loc] = tag (0 = empty/deleted).
    std::vector<std::vector<uint32_t>> local_to_tag_;

    rdma::FetchMode fetch_mode_ = rdma::FetchMode::COLOCATION;
    uint32_t num_thread_qps_ = 0;
    ThreadStateMap thread_states_;
    std::mutex mutex_;
};

} // namespace danns
