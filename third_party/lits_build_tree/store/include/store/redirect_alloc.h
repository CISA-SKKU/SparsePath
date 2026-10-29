#pragma once
// store/redirect_alloc.h — live-redirect slot allocation channel
//
// Implements the spec's "alloc server" (lits_aware_insert_SPEC §4.3): during
// an insert batch, a point whose LoadBalancer verdict says m_assign != m_search
// asks m_assign's node for a final slot — a two-sided request served from the
// reserve Python lent at open_redirect_window().
//
// Channel layout (per node pair): TWO dedicated endpoints, role-split so each
// QP/CQ is owned by exactly one thread:
//   - the endpoint where I am SERVER: my alloc server thread posts recvs,
//     polls the recv CQ, and sends replies (its send CQ too).
//   - the endpoint where I am CLIENT: insert threads send requests
//     and poll for replies. Serialized per-channel by cli_mutex.
// Sharing one QP would interleave both threads' completions on one CQ.
// Role convention (deterministic, symmetric): on the LOW-port channel the
// lower node id is the server; on the HIGH-port channel the higher id is.
// Bootstrap mirrors ClusterSetup (lower id listens).
//
// Concurrency: the server thread is the SOLE mutator of the reserve and the
// granted ledger; t2l/l2t entries are written by both the server thread
// (grants) and insert threads (their own tags' retraction/revert) — always
// element-disjoint (tables pre-sized by NodeStore::presize_tables_for_redirect
// — in-place aligned stores only). The shared vector counter uses local FAA
// (NodeStore::counter_fetch_add here, counter_fetch_sub on the insert threads).

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "rdma/connection.h"

namespace danns {

class NodeStore;

// Wire format (12 bytes each way). The reply ECHOES the request tag so a
// stale reply (from a request the client already timed out on) can never be
// mistaken for the current one; magic guards against stray completions.
struct RedirectSlotMsg {
    uint32_t tag;     // request: tag asked for; reply: tag echoed back
    uint32_t value;   // reply: floc or UINT32_MAX (FAIL); request: unused
    uint32_t magic;   // MSG_MAGIC
};

class RedirectAlloc {
public:
    static constexpr uint32_t MSG_MAGIC = 0x52414C43;  // "RALC"
    static constexpr uint32_t ALLOC_FAIL = UINT32_MAX;

    struct Ledger {
        std::vector<std::pair<uint32_t, uint32_t>> granted;  // (tag, floc)
        std::vector<uint32_t> reserve_unused;
    };

    explicit RedirectAlloc(NodeStore* node_store) : node_store_(node_store) {}
    ~RedirectAlloc() { stop(); }

    // Register the two endpoints for a peer (registers the 8B buffers on
    // each). server_ep = the channel where THIS node serves; client_ep =
    // where this node requests. Call during setup (before start()).
    void add_peer(uint32_t peer_id,
                  std::shared_ptr<rdma::RDMAEndpoint> server_ep,
                  std::shared_ptr<rdma::RDMAEndpoint> client_ep);

    // Start/stop the alloc server thread (idempotent). Started once after
    // all peers are added; serves across ops (requests outside an open
    // window get ALLOC_FAIL).
    void start();
    void stop();

    // --- window (main thread; barrier-fenced by the caller) ---
    // open: take ownership of the lent reserve, pre-size tables via
    // NodeStore, then begin granting.
    void open_window(std::vector<uint32_t> reserve, uint32_t max_tag,
                     uint32_t num_partitions);
    // close: stop granting and hand the ledger back. Caller guarantees all
    // peers' batches are done (redirect-batches-done barrier) — a grant
    // after close would be lost.
    Ledger close_window();

    // --- client side (insert threads; serialized per-channel by cli_mutex) ---
    // Blocking slot request to peer_node. Returns floc or ALLOC_FAIL on
    // FAIL-reply/timeout (caller falls back to the m_search path). A
    // timeout POISONS the channel for the rest of the window (a late grant
    // may still be travelling — further requests could consume its stale
    // reply); poisoned channels are revived at the next open_window, and
    // their orphaned grants are reclaimed by the adopt-side entry_vid
    // verification.
    uint32_t request_slot(uint32_t peer_node, uint32_t tag,
                          int timeout_ms = 500);

    bool has_peer(uint32_t peer_id) const {
        return peers_.count(peer_id) > 0;
    }

private:
    struct PeerChannel {
        std::shared_ptr<rdma::RDMAEndpoint> server_ep;  // I serve here
        std::shared_ptr<rdma::RDMAEndpoint> client_ep;  // I request here
        // server-side buffers (server thread only)
        RedirectSlotMsg srv_req{};
        RedirectSlotMsg srv_resp{};
        ibv_mr* srv_req_mr = nullptr;
        ibv_mr* srv_resp_mr = nullptr;
        bool dead = false;       // server thread: post failed — stop serving
        // client-side buffers (insert threads, under cli_mutex)
        RedirectSlotMsg cli_req{};
        RedirectSlotMsg cli_resp{};
        ibv_mr* cli_req_mr = nullptr;
        ibv_mr* cli_resp_mr = nullptr;
        bool poisoned = false;   // insert threads (under cli_mutex): timed
                                 // out — skip this window; revived by
                                 // open_window (barrier-fenced)
        uint64_t cli_wr_seq = 0; // monotonically increasing client wr_ids
        // Serializes request_slot per channel (parallel insert): preserves
        // the one-outstanding-request invariant that the single reply
        // buffer and tag-echo matching depend on. open/close_window need
        // no lock — they run in barrier-fenced phases (before
        // lb-counters-pushed / after redirect-batches-done) with no insert
        // threads alive.
        std::mutex cli_mutex;
    };

    void server_loop();
    uint32_t handle_request(uint32_t tag);  // server thread only

    NodeStore* node_store_;
    std::map<uint32_t, std::unique_ptr<PeerChannel>> peers_;

    std::thread server_thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> serving_{false};

    // window state. grant_mutex_ serializes handle_request (server thread)
    // against open_window/close_window (main thread): a request that was
    // still queued in a server CQ when the batches-done barrier passed
    // (possible after a client timeout) must not mutate these vectors while
    // close_window moves them. Uncontended in steady state.
    std::mutex grant_mutex_;
    std::vector<uint32_t> reserve_;
    std::vector<std::pair<uint32_t, uint32_t>> granted_;
    uint32_t my_partition_ = 0;
};

} // namespace danns
