// rdma_connection.h
// RDMA Connection Manager using libibverbs
// Supports RoCE v2 with RDMA one-sided READ/WRITE operations
#pragma once

#include "rdma/common.h"
#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <unordered_map>

namespace rdma {

// Forward declarations
class RDMAConnection;

// Connection info exchanged via TCP during setup
struct ConnectionInfo {
    uint16_t lid;           // Local ID (not used in RoCE)
    uint32_t qp_num;        // Queue Pair number
    uint32_t psn;           // Packet Sequence Number
    union ibv_gid gid;      // GID for RoCE

    void print() const {
        Logger::info("ConnectionInfo: qp_num=%u, psn=%u, gid=%s",
                     qp_num, psn, gid_to_string(gid).c_str());
    }
};

// RDMA Context - manages device, PD, CQ
class RDMAContext {
public:
    RDMAContext(const std::string& device_name = "", int gid_index = DEFAULT_GID_INDEX);
    ~RDMAContext();

    // Non-copyable
    RDMAContext(const RDMAContext&) = delete;
    RDMAContext& operator=(const RDMAContext&) = delete;

    // Getters
    ibv_context* get_context() const { return ctx_; }
    ibv_pd* get_pd() const { return pd_; }
    ibv_cq* get_send_cq() const { return send_cq_; }
    ibv_cq* get_recv_cq() const { return recv_cq_; }
    int get_gid_index() const { return gid_index_; }
    const ibv_gid& get_gid() const { return gid_; }
    uint8_t get_port_num() const { return port_num_; }

    // Device capabilities
    const DeviceCapabilities& get_device_caps() const { return device_caps_; }
    uint32_t get_max_sge() const { return device_caps_.max_sge; }

    // Create standalone CQ pair (for per-thread QPs)
    std::pair<ibv_cq*, ibv_cq*> create_cq(int cq_size = DEFAULT_CQ_SIZE);
    void destroy_cq(ibv_cq* cq);

    // Memory registration
    ibv_mr* register_memory(void* addr, size_t length, int access_flags);
    void deregister_memory(ibv_mr* mr);

private:
    ibv_context* ctx_ = nullptr;
    ibv_pd* pd_ = nullptr;
    ibv_cq* send_cq_ = nullptr;
    ibv_cq* recv_cq_ = nullptr;
    ibv_gid gid_;
    int gid_index_;
    uint8_t port_num_ = 1;
    DeviceCapabilities device_caps_;

    void init_device(const std::string& device_name);
    void init_pd();
    void init_cq();
    void init_gid();
    void query_device_capabilities();
};

// RDMA Connection - manages a single QP connection to a remote peer
class RDMAConnection {
public:
    RDMAConnection(std::shared_ptr<RDMAContext> ctx);
    // Per-thread constructor: uses dedicated CQs instead of context's shared CQs
    RDMAConnection(std::shared_ptr<RDMAContext> ctx, ibv_cq* send_cq, ibv_cq* recv_cq);
    ~RDMAConnection();

    // Non-copyable
    RDMAConnection(const RDMAConnection&) = delete;
    RDMAConnection& operator=(const RDMAConnection&) = delete;

    // Connection setup
    ConnectionInfo get_local_info() const;
    void connect(const ConnectionInfo& remote_info);
    bool is_connected() const { return connected_; }

    // Memory registration for this connection
    ibv_mr* register_memory(void* addr, size_t length, int access_flags =
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE);

    // Post receive buffer (for two-sided ops)
    void post_recv(void* buffer, size_t length, ibv_mr* mr, uint64_t wr_id = 0);

    // Post send (two-sided)
    void post_send(void* buffer, size_t length, ibv_mr* mr, uint64_t wr_id = 0);

    // RDMA READ (one-sided) - reads from remote into local buffer
    void post_read(void* local_buffer, size_t length, ibv_mr* local_mr,
                   const RemoteMemoryRegion& remote, uint64_t wr_id = 0);

    // RDMA WRITE (one-sided) - writes from local buffer to remote
    void post_write(void* local_buffer, size_t length, ibv_mr* local_mr,
                    const RemoteMemoryRegion& remote, uint64_t wr_id = 0);

    // ===== Multi-SGE / Batched operations =====

    // Local scatter: Read contiguous remote region into multiple local buffers
    // - Reads (sum of sgl lengths) bytes from remote.addr
    // - Scatters to multiple local buffers specified by sgl
    void post_read_scatter(const std::vector<SGLEntry>& sgl,
                           const RemoteMemoryRegion& remote,
                           ibv_mr* local_mr,
                           uint64_t wr_id = 0);

    // Batched READ: Post multiple independent READs as separate WRs
    // - Each entry specifies local buffer and remote offset
    // - Posts N work requests and waits for all completions (blocking)
    // - Uses internal flow control to avoid SQ overflow
    void post_read_batch(const std::vector<SGLEntry>& entries,
                         const RemoteMemoryRegion& remote_base,
                         ibv_mr* local_mr,
                         uint64_t base_wr_id = 0,
                         int timeout_ms = 10000);

    // Async batched READ: Post WRs without waiting for completion.
    // Returns the number of WRs posted. Caller must later call
    // poll_read_batch() on this same connection to drain completions.
    // Uses SQ flow control internally (drains some completions if SQ is full).
    uint32_t post_read_batch_async(const std::vector<SGLEntry>& entries,
                                   const RemoteMemoryRegion& remote_base,
                                   ibv_mr* local_mr,
                                   uint64_t base_wr_id = 0);

    // Poll for N completions from a prior post_read_batch_async call.
    // Spins until expected_count completions are received or timeout.
    // Returns the number of completions polled.
    uint32_t poll_read_batch(uint32_t expected_count, int timeout_ms = 5000);

    // Non-blocking poll: drains up to max_poll completions from CQ.
    // Returns immediately with 0 if nothing ready. Throws on CQ error.
    uint32_t poll_read_nonblocking(uint32_t max_poll = 16);

    // Poll for completions
    int poll_send_cq(ibv_wc* wc_array, int max_entries);
    int poll_recv_cq(ibv_wc* wc_array, int max_entries);

    // Blocking wait for completion
    bool wait_send_completion(uint64_t wr_id, int timeout_ms = 5000);
    bool wait_recv_completion(uint64_t wr_id, int timeout_ms = 5000);

    // Wait for multiple completions (for batched operations)
    bool wait_completions(uint64_t base_wr_id, uint32_t count, int timeout_ms = 5000);

    // Getters
    ibv_qp* get_qp() const { return qp_; }
    std::shared_ptr<RDMAContext> get_context() const { return ctx_; }
    uint32_t get_max_sge() const { return actual_max_sge_; }

private:
    std::shared_ptr<RDMAContext> ctx_;
    ibv_qp* qp_ = nullptr;
    uint32_t psn_;  // Packet sequence number
    bool connected_ = false;
    uint32_t actual_max_sge_ = 1;  // Actual max SGE negotiated with device

    // Send-queue WRs (RDMA READ/WRITE/SEND) posted but whose CQE has not yet
    // been consumed. Tracks SQ occupancy ACROSS calls so flow control sees
    // prior in-flight WRs — a single post_read_batch_async could otherwise
    // overflow the 128-deep SQ and throw ENOMEM. Recv-queue WRs use a separate
    // CQ and are NOT counted here. Per-thread QP: no atomic needed; the shared
    // single-QP fetcher path touches this only under the fetcher's mutex.
    uint32_t outstanding_wrs_ = 0;
    // Underflow-safe decrement when CQEs are consumed (n in WR units).
    void consume_send_wrs(uint32_t n) {
        outstanding_wrs_ = (outstanding_wrs_ >= n) ? outstanding_wrs_ - n : 0;
    }
    // Selective signaling (post_read_batch_async): only the last WR of each
    // doorbell chunk is signaled; RC in-order completion means its CQE implies
    // every prior unsignaled WR of the chunk also completed. The chunk size
    // (# WRs that CQE retires) is encoded in the signaled WR's wr_id top 16
    // bits; the low 48 bits keep base_wr_id+i (chunk<=512, ids well under 2^48).
    // Per-WR-signaled paths (blocking) leave the top bits 0 -> run_length 1.
    static constexpr uint64_t kWrRunlenShift = 48;
    static constexpr uint64_t kWrIdLowMask = (1ULL << kWrRunlenShift) - 1;
    static uint32_t wr_run_length(uint64_t wr_id) {
        uint32_t r = static_cast<uint32_t>(wr_id >> kWrRunlenShift);
        return r ? r : 1u;
    }
    static uint64_t wr_id_low(uint64_t wr_id) { return wr_id & kWrIdLowMask; }
    // CQEs consumed inside post_read_batch_async's SQ-space drain that belong
    // to EARLIER batches on this QP (hop-interleaved search keeps several
    // batches in flight, all with base_wr_id=0). They are banked here and
    // handed to the next poll_read_nonblocking/poll_read_batch so the
    // caller-side completion totals (FIFO segments) stay conserved.
    uint32_t foreign_credits_ = 0;
    uint32_t take_foreign_credits() {
        uint32_t c = foreign_credits_;
        foreign_credits_ = 0;
        return c;
    }

    // Per-thread CQs (nullptr = use context's shared CQs)
    ibv_cq* per_thread_send_cq_ = nullptr;
    ibv_cq* per_thread_recv_cq_ = nullptr;

    void create_qp();
    void create_qp_with_cq(ibv_cq* send_cq, ibv_cq* recv_cq);
    void modify_qp_to_init();
    void modify_qp_to_rtr(const ConnectionInfo& remote_info);
    void modify_qp_to_rts();
};

// TCP-based connection exchanger (for bootstrapping RDMA connections)
class ConnectionExchanger {
public:
    // Server side: listen and exchange info
    static ConnectionInfo exchange_as_server(
        uint16_t port,
        const ConnectionInfo& local_info,
        int timeout_sec = 30);

    // Client side: connect and exchange info
    static ConnectionInfo exchange_as_client(
        const std::string& server_addr,
        uint16_t port,
        const ConnectionInfo& local_info,
        int timeout_sec = 10);

    // Multi-QP exchange: exchange N ConnectionInfos over a single TCP socket
    static std::vector<ConnectionInfo> exchange_multi_as_server(
        uint16_t port,
        const std::vector<ConnectionInfo>& local_infos,
        int timeout_sec = 30);

    static std::vector<ConnectionInfo> exchange_multi_as_client(
        const std::string& server_addr,
        uint16_t port,
        const std::vector<ConnectionInfo>& local_infos,
        int timeout_sec = 30,
        int max_retries = 60,
        int retry_interval_ms = 500);

private:
    static void send_info(int sockfd, const ConnectionInfo& info);
    static ConnectionInfo recv_info(int sockfd);
};

// High-level RDMA Endpoint combining context and connection
class RDMAEndpoint {
public:
    RDMAEndpoint(const std::string& device_name = "", int gid_index = DEFAULT_GID_INDEX);
    ~RDMAEndpoint();

    // Setup as server (waits for client connection)
    void listen_and_connect(uint16_t port);

    // Setup as client (connects to server)
    void connect_to(const std::string& server_addr, uint16_t port);

    // Memory operations
    ibv_mr* register_memory(void* addr, size_t length, int access_flags =
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE);

    // Exchange memory region info with peer
    void send_mr_info(const RemoteMemoryRegion& local_mr_info);
    RemoteMemoryRegion recv_mr_info();

    // RDMA operations
    void rdma_read(void* local_buffer, size_t length, ibv_mr* local_mr,
                   const RemoteMemoryRegion& remote, bool blocking = true);

    void rdma_write(void* local_buffer, size_t length, ibv_mr* local_mr,
                    const RemoteMemoryRegion& remote, bool blocking = true);

    // Two-sided messaging
    void send(void* buffer, size_t length, ibv_mr* mr, bool blocking = true);
    void recv(void* buffer, size_t length, ibv_mr* mr, bool blocking = true);

    // Getters
    bool is_connected() const { return conn_ && conn_->is_connected(); }
    std::shared_ptr<RDMAContext> get_context() const { return ctx_; }
    std::shared_ptr<RDMAConnection> get_connection() const { return conn_; }

private:
    std::shared_ptr<RDMAContext> ctx_;
    std::shared_ptr<RDMAConnection> conn_;
    std::atomic<uint64_t> wr_id_counter_{0};

    // Message buffer for protocol messages
    std::unique_ptr<uint8_t[]> msg_buffer_;
    ibv_mr* msg_mr_ = nullptr;
    static constexpr size_t MSG_BUFFER_SIZE = 4096;

    void setup_msg_buffer();
};

} // namespace rdma
