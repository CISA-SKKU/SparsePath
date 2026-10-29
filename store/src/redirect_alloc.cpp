// store/src/redirect_alloc.cpp — live-redirect alloc server/client
#include "store/redirect_alloc.h"
#include "store/node_store.h"
#include "rdma/common.h"

#include <chrono>

namespace danns {

void RedirectAlloc::add_peer(uint32_t peer_id,
                             std::shared_ptr<rdma::RDMAEndpoint> server_ep,
                             std::shared_ptr<rdma::RDMAEndpoint> client_ep) {
    auto ch = std::make_unique<PeerChannel>();
    ch->server_ep = server_ep;
    ch->client_ep = client_ep;
    ch->srv_req_mr = server_ep->register_memory(
        &ch->srv_req, sizeof(RedirectSlotMsg), IBV_ACCESS_LOCAL_WRITE);
    ch->srv_resp_mr = server_ep->register_memory(
        &ch->srv_resp, sizeof(RedirectSlotMsg), IBV_ACCESS_LOCAL_WRITE);
    ch->cli_req_mr = client_ep->register_memory(
        &ch->cli_req, sizeof(RedirectSlotMsg), IBV_ACCESS_LOCAL_WRITE);
    ch->cli_resp_mr = client_ep->register_memory(
        &ch->cli_resp, sizeof(RedirectSlotMsg), IBV_ACCESS_LOCAL_WRITE);
    peers_[peer_id] = std::move(ch);
    rdma::Logger::info("RedirectAlloc: peer %u channels registered", peer_id);
}

void RedirectAlloc::start() {
    if (running_.exchange(true)) return;
    server_thread_ = std::thread([this] { server_loop(); });
    rdma::Logger::info("RedirectAlloc: server thread started (%zu peers)",
                       peers_.size());
}

void RedirectAlloc::stop() {
    if (!running_.exchange(false)) return;
    if (server_thread_.joinable()) server_thread_.join();
}

void RedirectAlloc::open_window(std::vector<uint32_t> reserve,
                                uint32_t max_tag, uint32_t num_partitions) {
    node_store_->presize_tables_for_redirect(max_tag, num_partitions);
    {
        std::lock_guard<std::mutex> lock(grant_mutex_);
        if (serving_.load(std::memory_order_relaxed)) {
            throw std::runtime_error("RedirectAlloc: window already open");
        }
        my_partition_ = node_store_->partition_id();
        reserve_ = std::move(reserve);
        granted_.clear();
        serving_.store(true, std::memory_order_release);
    }
    // Revive channels poisoned by timeouts in a previous window (their
    // orphaned grants were reclaimed by that window's adopt verification).
    for (auto& [peer, ch] : peers_) ch->poisoned = false;
    rdma::Logger::info("RedirectAlloc: window open (reserve=%zu, max_tag=%u)",
                       reserve_.size(), max_tag);
}

RedirectAlloc::Ledger RedirectAlloc::close_window() {
    // The redirect-batches-done barrier means no peer will SEND new
    // requests, but a request that was already queued when its client
    // timed out can still be sitting in a server CQ — the mutex makes
    // any such late handle_request and this close atomic w.r.t. each
    // other (the loser of the race either grants before the move or is
    // refused by serving_=false).
    std::lock_guard<std::mutex> lock(grant_mutex_);
    serving_.store(false, std::memory_order_release);
    Ledger ledger;
    ledger.granted = std::move(granted_);
    ledger.reserve_unused = std::move(reserve_);
    granted_.clear();
    reserve_.clear();
    rdma::Logger::info("RedirectAlloc: window closed (granted=%zu, unused=%zu)",
                       ledger.granted.size(), ledger.reserve_unused.size());
    return ledger;
}

uint32_t RedirectAlloc::handle_request(uint32_t tag) {
    std::lock_guard<std::mutex> lock(grant_mutex_);
    if (!serving_.load(std::memory_order_relaxed)) return ALLOC_FAIL;
    if (reserve_.empty()) return ALLOC_FAIL;

    const uint32_t floc = reserve_.back();

    // In-place table updates (pre-sized at open — never resize here; the
    // insert threads may be reading these arrays concurrently).
    auto& t2l = node_store_->tag_to_local_mut()[my_partition_];
    auto& l2t = node_store_->local_to_tag_mut()[my_partition_];
    if (tag >= t2l.size() || floc >= l2t.size()) {
        rdma::Logger::error("RedirectAlloc: tag %u / floc %u out of pre-sized "
                            "range (t2l=%zu, l2t=%zu) — replying FAIL",
                            tag, floc, t2l.size(), l2t.size());
        return ALLOC_FAIL;
    }
    reserve_.pop_back();
    node_store_->counter_fetch_add(1);
    t2l[tag] = floc;
    l2t[floc] = tag;
    granted_.emplace_back(tag, floc);
    return floc;
}

void RedirectAlloc::server_loop() {
    // Post the initial recv on every server channel.
    for (auto& [peer, ch] : peers_) {
        try {
            ch->server_ep->get_connection()->post_recv(
                &ch->srv_req, sizeof(RedirectSlotMsg), ch->srv_req_mr,
                /*wr_id=*/peer);
        } catch (const std::exception& e) {
            rdma::Logger::error("RedirectAlloc: initial post_recv for peer %u "
                                "failed (%s) — channel dead", peer, e.what());
            ch->dead = true;
        }
    }

    ibv_wc wc;
    while (running_.load(std::memory_order_relaxed)) {
        bool any = false;
        for (auto& [peer, ch] : peers_) {
            if (ch->dead) continue;
            try {
                auto conn = ch->server_ep->get_connection();
                const int n = conn->poll_recv_cq(&wc, 1);
                if (n <= 0) continue;
                any = true;
                if (wc.status != IBV_WC_SUCCESS) {
                    rdma::Logger::error("RedirectAlloc: recv error from peer "
                                        "%u (status=%d) — channel dead",
                                        peer, wc.status);
                    ch->dead = true;  // QP likely in error; stop serving it
                    continue;
                }

                const uint32_t req_tag = ch->srv_req.tag;
                uint32_t reply = ALLOC_FAIL;
                if (ch->srv_req.magic == MSG_MAGIC) {
                    reply = handle_request(req_tag);
                } else {
                    rdma::Logger::error("RedirectAlloc: bad magic from peer %u",
                                        peer);
                }

                // Repost the recv BEFORE replying so the peer's next request
                // never races an empty recv queue.
                conn->post_recv(&ch->srv_req, sizeof(RedirectSlotMsg),
                                ch->srv_req_mr, peer);

                ch->srv_resp.tag = req_tag;      // echo — client validates
                ch->srv_resp.value = reply;
                ch->srv_resp.magic = MSG_MAGIC;
                conn->post_send(&ch->srv_resp, sizeof(RedirectSlotMsg),
                                ch->srv_resp_mr, /*wr_id=*/peer);
                if (!conn->wait_send_completion(peer, /*timeout_ms=*/200)) {
                    rdma::Logger::error("RedirectAlloc: reply send to peer %u "
                                        "timed out", peer);
                }
            } catch (const std::exception& e) {
                // Exception escaping a std::thread is std::terminate — never
                // let one broken channel kill the node.
                rdma::Logger::error("RedirectAlloc: channel to peer %u failed "
                                    "(%s) — channel dead", peer, e.what());
                ch->dead = true;
            }
        }
        if (!any) {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    }
}

uint32_t RedirectAlloc::request_slot(uint32_t peer_node, uint32_t tag,
                                     int timeout_ms) {
    auto it = peers_.find(peer_node);
    if (it == peers_.end()) return ALLOC_FAIL;
    auto& ch = *it->second;
    // Whole request/reply cycle under the channel lock (poisoned check →
    // drain → post → wait → echo check → poison on failure): per channel
    // the execution is structurally identical to the single-threaded
    // sequence, so the protocol invariants hold unchanged. A thread queued
    // behind a timing-out request sees poisoned==true on entry and fails
    // fast into the redirect_fallback path (after blocking on cli_mutex for
    // up to the first holder's send+recv timeouts — bounded, once per
    // window).
    std::lock_guard<std::mutex> lock(ch.cli_mutex);
    if (ch.poisoned) return ALLOC_FAIL;

    try {
        auto conn = ch.client_ep->get_connection();

        // Drain any stale completion (a reply that arrived after we timed
        // out on a previous request and re-posted nothing).
        ibv_wc wc;
        while (conn->poll_recv_cq(&wc, 1) > 0) { /* discard */ }

        const uint64_t recv_id = ++ch.cli_wr_seq;
        const uint64_t send_id = ++ch.cli_wr_seq;
        // Post the reply recv BEFORE sending the request (the reply is a
        // SEND from the server; RNR retry covers the residual race).
        conn->post_recv(&ch.cli_resp, sizeof(RedirectSlotMsg), ch.cli_resp_mr,
                        recv_id);
        ch.cli_req.tag = tag;
        ch.cli_req.value = 0;
        ch.cli_req.magic = MSG_MAGIC;
        conn->post_send(&ch.cli_req, sizeof(RedirectSlotMsg), ch.cli_req_mr,
                        send_id);
        if (!conn->wait_send_completion(send_id, timeout_ms)) {
            rdma::Logger::error("RedirectAlloc: request send to node %u timed "
                                "out — channel poisoned for this window",
                                peer_node);
            ch.poisoned = true;
            return ALLOC_FAIL;
        }
        if (!conn->wait_recv_completion(recv_id, timeout_ms)) {
            rdma::Logger::error("RedirectAlloc: no reply from node %u within "
                                "%dms — channel poisoned for this window",
                                peer_node, timeout_ms);
            ch.poisoned = true;
            return ALLOC_FAIL;
        }
        // Tag echo: a stale reply (for a request we previously timed out
        // on) can never be attributed to this tag.
        if (ch.cli_resp.magic != MSG_MAGIC || ch.cli_resp.tag != tag) {
            rdma::Logger::error("RedirectAlloc: reply mismatch from node %u "
                                "(echo tag=%u want=%u) — discarded, channel "
                                "poisoned for this window",
                                peer_node, ch.cli_resp.tag, tag);
            ch.poisoned = true;
            return ALLOC_FAIL;
        }
        return ch.cli_resp.value;
    } catch (const std::exception& e) {
        rdma::Logger::error("RedirectAlloc: request to node %u failed (%s) — "
                            "channel poisoned for this window",
                            peer_node, e.what());
        ch.poisoned = true;
        return ALLOC_FAIL;
    }
}

} // namespace danns
