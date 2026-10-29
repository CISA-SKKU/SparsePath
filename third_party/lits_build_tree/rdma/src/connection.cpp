// rdma_connection.cpp
// RDMA Connection Manager implementation

#include "rdma/connection.h"
#include <cerrno>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <poll.h>
#include <random>
#include <chrono>

namespace rdma {

//=============================================================================
// RDMAContext Implementation
//=============================================================================

RDMAContext::RDMAContext(const std::string& device_name, int gid_index)
    : gid_index_(gid_index) {
    init_device(device_name);
    init_pd();
    init_cq();
    init_gid();
    query_device_capabilities();

    Logger::info("RDMAContext initialized: device=%s, gid_index=%d, gid=%s",
                 ibv_get_device_name(ctx_->device), gid_index_,
                 gid_to_string(gid_).c_str());
    device_caps_.print();
}

RDMAContext::~RDMAContext() {
    // Destructor must not throw; surface teardown failures as warnings. A
    // non-zero ibv_dealloc_pd usually means an MR/QP is still registered.
    int ret;
    if (recv_cq_ && (ret = ibv_destroy_cq(recv_cq_)) != 0)
        Logger::warn("ibv_destroy_cq(recv_cq) failed: %d", ret);
    if (send_cq_ && (ret = ibv_destroy_cq(send_cq_)) != 0)
        Logger::warn("ibv_destroy_cq(send_cq) failed: %d", ret);
    if (pd_ && (ret = ibv_dealloc_pd(pd_)) != 0)
        Logger::warn("ibv_dealloc_pd failed: %d (MR/QP still registered?)", ret);
    if (ctx_ && (ret = ibv_close_device(ctx_)) != 0)
        Logger::warn("ibv_close_device failed: %d", ret);
}

void RDMAContext::init_device(const std::string& device_name) {
    int num_devices = 0;
    ibv_device** device_list = ibv_get_device_list(&num_devices);
    RDMA_CHECK_NULL(device_list, "Failed to get device list");
    RDMA_CHECK(num_devices > 0, "No RDMA devices found");

    ibv_device* device = nullptr;
    std::string target_device = device_name;

    // Default to mlx5_1 if no device specified
    if (target_device.empty()) {
        target_device = "mlx5_1";
    }

    // Find specified device
    for (int i = 0; i < num_devices; i++) {
        if (target_device == ibv_get_device_name(device_list[i])) {
            device = device_list[i];
            break;
        }
    }

    if (!device) {
        if (!device_name.empty()) {
            // Caller explicitly named a device that does not exist. Fail loudly
            // with the available device names: on multi-NIC hosts a silent
            // fallback otherwise surfaces as mysterious low bandwidth /
            // RETRY_EXCEEDED (hosts A/B=mlx5_0, hosts C/D=mlx5_1).
            std::string available;
            for (int i = 0; i < num_devices; i++) {
                if (i) available += ", ";
                available += ibv_get_device_name(device_list[i]);
            }
            ibv_free_device_list(device_list);
            throw RDMAException("RDMA device '" + device_name +
                                "' not found. Available devices: " + available);
        }
        // No device requested: fall back to the first available device.
        Logger::warn("Device %s not found, using first available device", target_device.c_str());
        device = device_list[0];
    }

    ctx_ = ibv_open_device(device);
    ibv_free_device_list(device_list);

    RDMA_CHECK_NULL(ctx_, "Failed to open device");
}

void RDMAContext::init_pd() {
    pd_ = ibv_alloc_pd(ctx_);
    RDMA_CHECK_NULL(pd_, "Failed to allocate PD");
}

void RDMAContext::init_cq() {
    send_cq_ = ibv_create_cq(ctx_, DEFAULT_CQ_SIZE, nullptr, nullptr, 0);
    RDMA_CHECK_NULL(send_cq_, "Failed to create send CQ");

    recv_cq_ = ibv_create_cq(ctx_, DEFAULT_CQ_SIZE, nullptr, nullptr, 0);
    RDMA_CHECK_NULL(recv_cq_, "Failed to create recv CQ");
}

void RDMAContext::init_gid() {
    memset(&gid_, 0, sizeof(gid_));
    int ret = ibv_query_gid(ctx_, port_num_, gid_index_, &gid_);
    if (ret != 0) {
        Logger::error("ibv_query_gid failed for port=%d, gid_index=%d", port_num_, gid_index_);
        RDMA_CHECK(ret == 0, "Failed to query GID");
    }

    // Check if GID is valid (not all zeros)
    bool is_zero = true;
    for (int i = 0; i < 16; i++) {
        if (gid_.raw[i] != 0) {
            is_zero = false;
            break;
        }
    }
    if (is_zero) {
        Logger::warn("GID at index %d is all zeros - this may cause connection failures", gid_index_);
        Logger::warn("Try a different gid_index. Use 'show_gids' to list available GIDs.");
    }
}

ibv_mr* RDMAContext::register_memory(void* addr, size_t length, int access_flags) {
    ibv_mr* mr = ibv_reg_mr(pd_, addr, length, access_flags);
    RDMA_CHECK_NULL(mr, "Failed to register memory region");
    return mr;
}

void RDMAContext::deregister_memory(ibv_mr* mr) {
    if (mr) {
        int ret = ibv_dereg_mr(mr);
        if (ret) {
            Logger::warn("Failed to deregister MR: %d", ret);
        }
    }
}

std::pair<ibv_cq*, ibv_cq*> RDMAContext::create_cq(int cq_size) {
    ibv_cq* send_cq = ibv_create_cq(ctx_, cq_size, nullptr, nullptr, 0);
    RDMA_CHECK_NULL(send_cq, "Failed to create per-thread send CQ");

    ibv_cq* recv_cq = ibv_create_cq(ctx_, cq_size, nullptr, nullptr, 0);
    if (!recv_cq) {
        ibv_destroy_cq(send_cq);
        throw RDMAException("Failed to create per-thread recv CQ");
    }

    return {send_cq, recv_cq};
}

void RDMAContext::destroy_cq(ibv_cq* cq) {
    if (cq) {
        ibv_destroy_cq(cq);
    }
}

void RDMAContext::query_device_capabilities() {
    ibv_device_attr attr;
    int ret = ibv_query_device(ctx_, &attr);
    if (ret != 0) {
        Logger::warn("Failed to query device attributes, using defaults");
        return;
    }

    device_caps_.max_sge = std::min(static_cast<uint32_t>(attr.max_sge), DEFAULT_MAX_SGE);
    device_caps_.max_qp_wr = attr.max_qp_wr;
    device_caps_.max_mr_size = attr.max_mr_size;

    // Query extended device attributes for ODP support
    ibv_query_device_ex_input input = {};
    ibv_device_attr_ex attr_ex;
    memset(&attr_ex, 0, sizeof(attr_ex));

    if (ibv_query_device_ex(ctx_, &input, &attr_ex) == 0) {
        // Check ODP support from extended attributes
        // ODP is indicated by non-zero odp_caps
        uint64_t odp_general = attr_ex.odp_caps.general_caps;
        uint32_t odp_rc_caps = attr_ex.odp_caps.per_transport_caps.rc_odp_caps;

        // ODP is supported if we have any ODP capabilities
        device_caps_.odp_supported = (odp_general != 0) || (odp_rc_caps != 0);

#ifdef IBV_ODP_SUPPORT_IMPLICIT
        device_caps_.odp_implicit = (odp_general & IBV_ODP_SUPPORT_IMPLICIT) != 0;
#endif

        Logger::debug("ODP caps: general=0x%lx, rc_caps=0x%x",
                      odp_general, odp_rc_caps);
    }

    // UMR support detection (mlx5 specific)
    // This will be properly detected when mlx5dv is available
    device_caps_.umr_supported = false;  // Will be set by UMR module if available
}

//=============================================================================
// RDMAConnection Implementation
//=============================================================================

RDMAConnection::RDMAConnection(std::shared_ptr<RDMAContext> ctx)
    : ctx_(std::move(ctx)) {
    // Generate random PSN
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dis(0, 0xFFFFFF);
    psn_ = dis(gen);

    create_qp();
    modify_qp_to_init();

    Logger::debug("RDMAConnection created: qp_num=%u, psn=%u", qp_->qp_num, psn_);
}

RDMAConnection::RDMAConnection(std::shared_ptr<RDMAContext> ctx,
                               ibv_cq* send_cq, ibv_cq* recv_cq)
    : ctx_(std::move(ctx)),
      per_thread_send_cq_(send_cq),
      per_thread_recv_cq_(recv_cq) {
    // Generate random PSN
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dis(0, 0xFFFFFF);
    psn_ = dis(gen);

    create_qp_with_cq(send_cq, recv_cq);
    modify_qp_to_init();

    Logger::debug("RDMAConnection created (per-thread CQ): qp_num=%u, psn=%u",
                  qp_->qp_num, psn_);
}

RDMAConnection::~RDMAConnection() {
    if (qp_) {
        ibv_destroy_qp(qp_);
    }
}

void RDMAConnection::create_qp() {
    // Use device's actual max SGE capability
    uint32_t requested_sge = ctx_->get_max_sge();

    ibv_qp_init_attr init_attr = {};
    init_attr.send_cq = ctx_->get_send_cq();
    init_attr.recv_cq = ctx_->get_recv_cq();
    init_attr.qp_type = IBV_QPT_RC;  // Reliable Connection
    init_attr.cap.max_send_wr = DEFAULT_SQ_DEPTH;
    init_attr.cap.max_recv_wr = DEFAULT_RQ_DEPTH;
    init_attr.cap.max_send_sge = requested_sge;
    init_attr.cap.max_recv_sge = requested_sge;
    init_attr.cap.max_inline_data = DEFAULT_MAX_INLINE;

    qp_ = ibv_create_qp(ctx_->get_pd(), &init_attr);
    RDMA_CHECK_NULL(qp_, "Failed to create QP");

    // Store the actual negotiated max SGE (may be different from requested)
    actual_max_sge_ = init_attr.cap.max_send_sge;
    Logger::debug("QP created: requested max_sge=%u, actual max_sge=%u",
                  requested_sge, actual_max_sge_);
}

void RDMAConnection::create_qp_with_cq(ibv_cq* send_cq, ibv_cq* recv_cq) {
    uint32_t requested_sge = ctx_->get_max_sge();

    ibv_qp_init_attr init_attr = {};
    init_attr.send_cq = send_cq;
    init_attr.recv_cq = recv_cq;
    init_attr.qp_type = IBV_QPT_RC;
    init_attr.cap.max_send_wr = DEFAULT_SQ_DEPTH;
    init_attr.cap.max_recv_wr = DEFAULT_RQ_DEPTH;
    init_attr.cap.max_send_sge = requested_sge;
    init_attr.cap.max_recv_sge = requested_sge;
    init_attr.cap.max_inline_data = DEFAULT_MAX_INLINE;

    qp_ = ibv_create_qp(ctx_->get_pd(), &init_attr);
    RDMA_CHECK_NULL(qp_, "Failed to create per-thread QP");

    actual_max_sge_ = init_attr.cap.max_send_sge;
    Logger::debug("Per-thread QP created: requested max_sge=%u, actual max_sge=%u",
                  requested_sge, actual_max_sge_);
}

void RDMAConnection::modify_qp_to_init() {
    ibv_qp_attr attr = {};
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = ctx_->get_port_num();
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE |
                           IBV_ACCESS_REMOTE_READ |
                           IBV_ACCESS_REMOTE_WRITE |
                           IBV_ACCESS_REMOTE_ATOMIC;

    int flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
    int ret = ibv_modify_qp(qp_, &attr, flags);
    RDMA_CHECK(ret == 0, "Failed to modify QP to INIT");
}

void RDMAConnection::modify_qp_to_rtr(const ConnectionInfo& remote_info) {
    ibv_qp_attr attr = {};
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = IBV_MTU_1024;
    attr.dest_qp_num = remote_info.qp_num;
    attr.rq_psn = remote_info.psn;
    attr.max_dest_rd_atomic = 16;
    attr.min_rnr_timer = 12;

    // Address handle for RoCE
    attr.ah_attr.is_global = 1;
    attr.ah_attr.grh.dgid = remote_info.gid;
    attr.ah_attr.grh.sgid_index = ctx_->get_gid_index();
    attr.ah_attr.grh.hop_limit = 64;
    attr.ah_attr.grh.traffic_class = 0;
    attr.ah_attr.grh.flow_label = 0;
    attr.ah_attr.dlid = remote_info.lid;
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = ctx_->get_port_num();

    int flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;

    int ret = ibv_modify_qp(qp_, &attr, flags);
    RDMA_CHECK(ret == 0, "Failed to modify QP to RTR");
}

void RDMAConnection::modify_qp_to_rts() {
    ibv_qp_attr attr = {};
    attr.qp_state = IBV_QPS_RTS;
    attr.sq_psn = psn_;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.max_rd_atomic = 16;

    int flags = IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT |
                IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC;

    int ret = ibv_modify_qp(qp_, &attr, flags);
    RDMA_CHECK(ret == 0, "Failed to modify QP to RTS");
}

ConnectionInfo RDMAConnection::get_local_info() const {
    ConnectionInfo info;
    info.qp_num = qp_->qp_num;
    info.psn = psn_;
    info.gid = ctx_->get_gid();
    info.lid = 0;  // Not used in RoCE
    return info;
}

void RDMAConnection::connect(const ConnectionInfo& remote_info) {
    Logger::info("Connecting to remote peer...");
    remote_info.print();

    modify_qp_to_rtr(remote_info);
    modify_qp_to_rts();

    connected_ = true;
    Logger::info("Connection established, QP state=RTS");
}

ibv_mr* RDMAConnection::register_memory(void* addr, size_t length, int access_flags) {
    return ctx_->register_memory(addr, length, access_flags);
}

void RDMAConnection::post_recv(void* buffer, size_t length, ibv_mr* mr, uint64_t wr_id) {
    ibv_sge sge = {};
    sge.addr = reinterpret_cast<uint64_t>(buffer);
    sge.length = static_cast<uint32_t>(length);
    sge.lkey = mr->lkey;

    ibv_recv_wr wr = {};
    wr.wr_id = wr_id;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.next = nullptr;

    ibv_recv_wr* bad_wr = nullptr;
    int ret = ibv_post_recv(qp_, &wr, &bad_wr);
    RDMA_CHECK(ret == 0, "Failed to post recv");
}

void RDMAConnection::post_send(void* buffer, size_t length, ibv_mr* mr, uint64_t wr_id) {
    ibv_sge sge = {};
    sge.addr = reinterpret_cast<uint64_t>(buffer);
    sge.length = static_cast<uint32_t>(length);
    sge.lkey = mr->lkey;

    ibv_send_wr wr = {};
    wr.wr_id = wr_id;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_SEND;
    wr.send_flags = IBV_SEND_SIGNALED;
    if (length <= DEFAULT_MAX_INLINE) {
        wr.send_flags |= IBV_SEND_INLINE;
    }
    wr.next = nullptr;

    ibv_send_wr* bad_wr = nullptr;
    int ret = ibv_post_send(qp_, &wr, &bad_wr);
    RDMA_CHECK(ret == 0, "Failed to post send");
    outstanding_wrs_++;
}

void RDMAConnection::post_read(void* local_buffer, size_t length, ibv_mr* local_mr,
                                const RemoteMemoryRegion& remote, uint64_t wr_id) {
    ibv_sge sge = {};
    sge.addr = reinterpret_cast<uint64_t>(local_buffer);
    sge.length = static_cast<uint32_t>(length);
    sge.lkey = local_mr->lkey;

    ibv_send_wr wr = {};
    wr.wr_id = wr_id;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_READ;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = remote.addr;
    wr.wr.rdma.rkey = remote.rkey;
    wr.next = nullptr;

    ibv_send_wr* bad_wr = nullptr;
    int ret = ibv_post_send(qp_, &wr, &bad_wr);
    RDMA_CHECK(ret == 0, "Failed to post RDMA read");
    outstanding_wrs_++;
}

void RDMAConnection::post_write(void* local_buffer, size_t length, ibv_mr* local_mr,
                                 const RemoteMemoryRegion& remote, uint64_t wr_id) {
    ibv_sge sge = {};
    sge.addr = reinterpret_cast<uint64_t>(local_buffer);
    sge.length = static_cast<uint32_t>(length);
    sge.lkey = local_mr->lkey;

    ibv_send_wr wr = {};
    wr.wr_id = wr_id;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.send_flags = IBV_SEND_SIGNALED;
    if (length <= DEFAULT_MAX_INLINE) {
        wr.send_flags |= IBV_SEND_INLINE;
    }
    wr.wr.rdma.remote_addr = remote.addr;
    wr.wr.rdma.rkey = remote.rkey;
    wr.next = nullptr;

    ibv_send_wr* bad_wr = nullptr;
    int ret = ibv_post_send(qp_, &wr, &bad_wr);
    RDMA_CHECK(ret == 0, "Failed to post RDMA write");
    outstanding_wrs_++;
}

void RDMAConnection::post_read_scatter(const std::vector<SGLEntry>& sgl,
                                        const RemoteMemoryRegion& remote,
                                        ibv_mr* local_mr,
                                        uint64_t wr_id) {
    if (sgl.empty()) {
        throw RDMAException("Empty SGL provided");
    }
    if (sgl.size() > actual_max_sge_) {
        throw RDMAException("SGL size " + std::to_string(sgl.size()) +
                           " exceeds max_sge " + std::to_string(actual_max_sge_));
    }

    // Build SGE array for scatter
    std::vector<ibv_sge> sge_list(sgl.size());
    uint32_t total_length = 0;

    for (size_t i = 0; i < sgl.size(); i++) {
        sge_list[i].addr = reinterpret_cast<uint64_t>(sgl[i].local_addr);
        sge_list[i].length = sgl[i].length;
        sge_list[i].lkey = local_mr->lkey;
        total_length += sgl[i].length;
    }

    // Single work request with multiple SGEs
    // Reads contiguous remote region and scatters to local buffers
    ibv_send_wr wr = {};
    wr.wr_id = wr_id;
    wr.sg_list = sge_list.data();
    wr.num_sge = static_cast<int>(sge_list.size());
    wr.opcode = IBV_WR_RDMA_READ;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = remote.addr;
    wr.wr.rdma.rkey = remote.rkey;
    wr.next = nullptr;

    ibv_send_wr* bad_wr = nullptr;
    int ret = ibv_post_send(qp_, &wr, &bad_wr);
    RDMA_CHECK(ret == 0, "Failed to post scatter RDMA read");

    Logger::debug("Posted scatter READ: %zu SGEs, total %u bytes from remote 0x%lx",
                  sgl.size(), total_length, remote.addr);
}

// TODO(overlap): Split post_read_batch into post_read_batch_async + poll_read_batch.
//   Currently post+poll are fused in one while loop, making this call fully blocking.
//   Separating them enables:
//     1) Posting RDMA to multiple remote owners simultaneously (parallel NIC utilization)
//     2) Overlapping local compute (distance, PQ LUT) with in-flight RDMA
//   New API sketch:
//     uint32_t post_read_batch_async(entries, remote_base, local_mr, base_wr_id);
//       -> posts all WRs with SQ flow control, returns count posted
//     uint32_t poll_read_batch(expected_count, timeout_ms);
//       -> spins ibv_poll_cq until expected_count completions
//   Note: multi-QP path uses per-owner conn with dedicated CQ, so poll must be per-owner.
void RDMAConnection::post_read_batch(const std::vector<SGLEntry>& entries,
                                      const RemoteMemoryRegion& remote_base,
                                      ibv_mr* local_mr,
                                      uint64_t base_wr_id,
                                      int timeout_ms) {
    if (entries.empty()) {
        return;
    }

    const size_t total = entries.size();

    // Flow control: limit outstanding WRs to avoid SQ overflow
    const uint32_t max_outstanding = DEFAULT_SQ_DEPTH / 2;
    uint32_t posted = 0;
    uint32_t completed = 0;

    // Post all WRs with inline flow control.
    // Timeout guard: a lost completion would otherwise spin here at 100% CPU
    // forever (undiagnosable hang). Mirrors poll_read_batch's timeout pattern.
    auto start = std::chrono::steady_clock::now();
    while (completed < total) {
        // Post as many as we can without exceeding max_outstanding
        while (posted < total && (posted - completed) < max_outstanding) {
            size_t i = posted;

            ibv_sge sge = {};
            sge.addr = reinterpret_cast<uint64_t>(entries[i].local_addr);
            sge.length = entries[i].length;
            sge.lkey = local_mr->lkey;

            ibv_send_wr wr = {};
            wr.wr_id = base_wr_id + i;
            wr.sg_list = &sge;
            wr.num_sge = 1;
            wr.opcode = IBV_WR_RDMA_READ;
            wr.send_flags = IBV_SEND_SIGNALED;
            wr.wr.rdma.remote_addr = remote_base.addr + entries[i].remote_offset;
            wr.wr.rdma.rkey = remote_base.rkey;
            wr.next = nullptr;

            ibv_send_wr* bad_wr = nullptr;
            int ret = ibv_post_send(qp_, &wr, &bad_wr);
            RDMA_CHECK(ret == 0, "Failed to post batch RDMA read");
            posted++;
            outstanding_wrs_++;
        }

        // Poll for completions (use per-thread CQ if available)
        ibv_wc wc[16];
        ibv_cq* poll_cq = per_thread_send_cq_ ? per_thread_send_cq_ : ctx_->get_send_cq();
        int n = ibv_poll_cq(poll_cq, 16, wc);
        if (n > 0) {
            for (int j = 0; j < n; j++) {
                if (wc[j].status != IBV_WC_SUCCESS) {
                    // Query QP state for debugging
                    ibv_qp_attr qp_attr;
                    ibv_qp_init_attr init_attr;
                    ibv_query_qp(qp_, &qp_attr, IBV_QP_STATE, &init_attr);

                    Logger::error("Batch READ completion error: %s (wr_id=%lu, opcode=%d)",
                                 wc_status_to_string(wc[j].status), wc[j].wr_id, wc[j].opcode);
                    Logger::error("  QP state: %s, posted=%u, completed=%u",
                                 qp_state_to_string(qp_attr.qp_state), posted, completed);
                    Logger::error("  vendor_err=0x%x, byte_len=%u",
                                 wc[j].vendor_err, wc[j].byte_len);

                    throw RDMAException("Batch READ failed: " +
                                       std::string(wc_status_to_string(wc[j].status)));
                }
                // Free SQ slots — a foreign CQE from a selectively-signaled
                // async batch on the same (shared) CQ carries run_length in its
                // top bits and retires that many WRs; this path's own WRs are
                // per-WR signaled (run_length 1).
                consume_send_wrs(wr_run_length(wc[j].wr_id));
                // Only count completions that belong to THIS batch. Compare on
                // the LOW 48 bits so a foreign async CQE (top bits = run_length)
                // is not mistaken for an in-range wr_id. A stale CQE left by an
                // earlier timed-out batch must not prematurely finish this one.
                // NOTE: base_wr_id=0 callers get an upper-bound check only.
                const uint64_t wid = wr_id_low(wc[j].wr_id);
                if (wid >= base_wr_id &&
                    wid < base_wr_id + static_cast<uint64_t>(total)) {
                    completed++;
                } else {
                    Logger::error("foreign completion wr_id=%lu not in [%lu,%lu) during batch READ",
                                  wc[j].wr_id, base_wr_id,
                                  base_wr_id + static_cast<uint64_t>(total));
#ifndef NDEBUG
                    throw RDMAException("foreign completion in post_read_batch");
#endif
                }
            }
        } else if (n < 0) {
            throw RDMAException("Poll CQ error during batch READ");
        }

        if (timeout_ms > 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now() - start).count();
            if (elapsed >= timeout_ms) {
                ibv_qp_attr qp_attr;
                ibv_qp_init_attr qp_init;
                ibv_query_qp(qp_, &qp_attr, IBV_QP_STATE, &qp_init);
                throw RDMAException("post_read_batch timeout: posted=" + std::to_string(posted) +
                                    " completed=" + std::to_string(completed) +
                                    "/" + std::to_string(total) +
                                    " outstanding_wrs=" + std::to_string(outstanding_wrs_) +
                                    " qp_state=" + qp_state_to_string(qp_attr.qp_state));
            }
        }
    }

    Logger::debug("Batch READ completed: %zu WRs", total);
}

uint32_t RDMAConnection::post_read_batch_async(const std::vector<SGLEntry>& entries,
                                                const RemoteMemoryRegion& remote_base,
                                                ibv_mr* local_mr,
                                                uint64_t base_wr_id) {
    if (entries.empty()) {
        return 0;
    }

    const size_t total = entries.size();
    uint32_t posted = 0;
    uint32_t drained = 0;  // completions drained internally; caller polls the rest

    // Doorbell-batched chunk: chain up to a full SQ worth of WRs via wr.next,
    // then call ibv_post_send ONCE per chunk. This amortizes the doorbell MMIO
    // (PCIe write to NIC BAR) across N WRs: N posts → 1 doorbell.
    // Per-chunk storage is thread_local so repeat calls reuse capacity (no realloc).
    thread_local std::vector<ibv_send_wr> wr_chain;
    thread_local std::vector<ibv_sge> sge_chain;
    if (wr_chain.capacity() < DEFAULT_SQ_DEPTH) {
        wr_chain.reserve(DEFAULT_SQ_DEPTH);
        sge_chain.reserve(DEFAULT_SQ_DEPTH);
    }

    while (posted < total) {
        // Flow control across ALL calls on this QP: the SQ may already hold
        // in-flight WRs from a previous (e.g. hop-interleaved) batch, so cap by
        // the real remaining SQ capacity rather than this call's local view.
        uint32_t available = (outstanding_wrs_ < DEFAULT_SQ_DEPTH)
                             ? DEFAULT_SQ_DEPTH - outstanding_wrs_ : 0;
        uint32_t remaining = static_cast<uint32_t>(total) - posted;
        uint32_t chunk = (available < remaining) ? available : remaining;

        if (chunk > 0) {
            wr_chain.resize(chunk);
            sge_chain.resize(chunk);

            for (uint32_t k = 0; k < chunk; k++) {
                size_t i = posted + k;

                sge_chain[k] = {};
                sge_chain[k].addr = reinterpret_cast<uint64_t>(entries[i].local_addr);
                sge_chain[k].length = entries[i].length;
                sge_chain[k].lkey = local_mr->lkey;

                const bool last = (k + 1 == chunk);
                wr_chain[k] = {};
                // Selective signaling: signal only the chunk's last WR. Its CQE
                // implies all prior unsignaled WRs in this chunk completed (RC
                // in-order). Encode the chunk size in the signaled wr_id's top
                // 16 bits so pollers credit `chunk` WR-completions for that one
                // CQE. Unsignaled WRs keep a plain low-bits wr_id (never polled).
                wr_chain[k].wr_id = last
                    ? ((static_cast<uint64_t>(chunk) << kWrRunlenShift) |
                       ((base_wr_id + i) & kWrIdLowMask))
                    : (base_wr_id + i);
                wr_chain[k].sg_list = &sge_chain[k];
                wr_chain[k].num_sge = 1;
                wr_chain[k].opcode = IBV_WR_RDMA_READ;
                wr_chain[k].send_flags = last ? IBV_SEND_SIGNALED : 0;
                wr_chain[k].wr.rdma.remote_addr = remote_base.addr + entries[i].remote_offset;
                wr_chain[k].wr.rdma.rkey = remote_base.rkey;
                wr_chain[k].next = (k + 1 < chunk) ? &wr_chain[k + 1] : nullptr;
            }

            ibv_send_wr* bad_wr = nullptr;
            int ret = ibv_post_send(qp_, &wr_chain[0], &bad_wr);
            RDMA_CHECK(ret == 0, "Failed to post async batch RDMA read (chained)");
            posted += chunk;
            outstanding_wrs_ += chunk;
        }

        // If SQ is full and we still have entries to post, drain some completions
        if (posted < total) {
            ibv_wc wc[16];
            ibv_cq* poll_cq = per_thread_send_cq_ ? per_thread_send_cq_ : ctx_->get_send_cq();
            int n = ibv_poll_cq(poll_cq, 16, wc);
            if (n > 0) {
                for (int j = 0; j < n; j++) {
                    if (wc[j].status != IBV_WC_SUCCESS) {
                        Logger::error("Async batch READ completion error: %s (wr_id=%lu)",
                                     wc_status_to_string(wc[j].status), wc[j].wr_id);
                        throw RDMAException("Async batch READ failed: " +
                                           std::string(wc_status_to_string(wc[j].status)));
                    }
                    // One signaled CQE retires `run_length` WRs (selective
                    // signaling). Free that many SQ slots and credit in WR units.
                    const uint32_t rl = wr_run_length(wc[j].wr_id);
                    consume_send_wrs(rl);
                    // Credit up to `total` WRs to this batch's drained count;
                    // bank the surplus WRs (from earlier in-flight base_wr_id=0
                    // batches) so the next poll_* hands them back. Keeps the
                    // returned pending (total-drained) in WR units, never
                    // underflowing, and conserves totals for the FIFO consumers.
                    const uint32_t room = static_cast<uint32_t>(total) - drained;
                    const uint32_t take = (rl < room) ? rl : room;
                    drained += take;
                    foreign_credits_ += (rl - take);
                }
            } else if (n < 0) {
                throw RDMAException("Poll CQ error during async batch READ post");
            }
        }
    }

    // Return total posted; caller must poll for (total - drained) remaining completions
    Logger::debug("Async batch READ posted: %zu WRs in chained doorbells, %u drained",
                  total, drained);
    return static_cast<uint32_t>(total) - drained;
}

uint32_t RDMAConnection::poll_read_batch(uint32_t expected_count, int timeout_ms) {
    if (expected_count == 0) {
        return 0;
    }

    // Absorb completions banked by post_read_batch_async's SQ drain so we
    // don't wait for CQEs that were already consumed there.
    uint32_t completed = take_foreign_credits();
    if (completed > expected_count) completed = expected_count;
    ibv_cq* poll_cq = per_thread_send_cq_ ? per_thread_send_cq_ : ctx_->get_send_cq();

    auto start = std::chrono::steady_clock::now();
    while (completed < expected_count) {
        ibv_wc wc[16];
        int n = ibv_poll_cq(poll_cq, 16, wc);
        if (n > 0) {
            uint32_t wrs = 0;
            for (int j = 0; j < n; j++) {
                if (wc[j].status != IBV_WC_SUCCESS) {
                    Logger::error("poll_read_batch completion error: %s (wr_id=%lu)",
                                 wc_status_to_string(wc[j].status), wc[j].wr_id);
                    throw RDMAException("poll_read_batch failed: " +
                                       std::string(wc_status_to_string(wc[j].status)));
                }
                // Each signaled CQE retires run_length WRs (selective signaling).
                wrs += wr_run_length(wc[j].wr_id);
            }
            completed += wrs;
            consume_send_wrs(wrs);
        } else if (n < 0) {
            throw RDMAException("Poll CQ error during poll_read_batch");
        }

        if (timeout_ms > 0) {
            auto elapsed = std::chrono::steady_clock::now() - start;
            if (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() >= timeout_ms) {
                Logger::error("poll_read_batch timeout: %u/%u completed", completed, expected_count);
                throw RDMAException("poll_read_batch timeout");
            }
        }
    }

    Logger::debug("poll_read_batch completed: %u WRs", completed);
    return completed;
}

uint32_t RDMAConnection::poll_read_nonblocking(uint32_t max_poll) {
    ibv_cq* poll_cq = per_thread_send_cq_ ? per_thread_send_cq_ : ctx_->get_send_cq();
    ibv_wc wc[16];
    uint32_t batch = (max_poll < 16) ? max_poll : 16;
    int n = ibv_poll_cq(poll_cq, batch, wc);
    if (n > 0) {
        uint32_t wrs = 0;
        for (int j = 0; j < n; j++) {
            if (wc[j].status != IBV_WC_SUCCESS) {
                Logger::error("poll_read_nonblocking error: %s (wr_id=%lu)",
                             wc_status_to_string(wc[j].status), wc[j].wr_id);
                throw RDMAException("poll_read_nonblocking failed: " +
                                   std::string(wc_status_to_string(wc[j].status)));
            }
            // Each signaled CQE retires run_length WRs (selective signaling).
            // Returning WR-completion counts keeps every caller (FIFO retire,
            // bulk-sync barriers, Pass-3) correct in WR units with no change.
            wrs += wr_run_length(wc[j].wr_id);
        }
        consume_send_wrs(wrs);
        return wrs + take_foreign_credits();
    } else if (n < 0) {
        throw RDMAException("Poll CQ error during poll_read_nonblocking");
    }
    return take_foreign_credits();
}

int RDMAConnection::poll_send_cq(ibv_wc* wc_array, int max_entries) {
    ibv_cq* cq = per_thread_send_cq_ ? per_thread_send_cq_ : ctx_->get_send_cq();
    return ibv_poll_cq(cq, max_entries, wc_array);
}

int RDMAConnection::poll_recv_cq(ibv_wc* wc_array, int max_entries) {
    ibv_cq* cq = per_thread_recv_cq_ ? per_thread_recv_cq_ : ctx_->get_recv_cq();
    return ibv_poll_cq(cq, max_entries, wc_array);
}

bool RDMAConnection::wait_send_completion(uint64_t wr_id, int timeout_ms) {
    auto start = std::chrono::steady_clock::now();
    ibv_wc wc;

    while (true) {
        int n = poll_send_cq(&wc, 1);
        if (n > 0) {
            if (wc.status != IBV_WC_SUCCESS) {
                Logger::error("Send completion error: %s (wr_id=%lu)",
                             wc_status_to_string(wc.status), wc.wr_id);
                return false;
            }
            // Every send CQE frees SQ slots — a non-matching one from a
            // selectively-signaled async batch on this shared CQ retires
            // run_length WRs (its top bits); a plain per-WR CQE retires 1.
            consume_send_wrs(wr_run_length(wc.wr_id));
            if (wc.wr_id == wr_id) {
                return true;
            }
        } else if (n < 0) {
            Logger::error("Poll CQ error");
            return false;
        }

        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
        if (elapsed >= timeout_ms) {
            Logger::error("Send completion timeout (wr_id=%lu)", wr_id);
            return false;
        }

        // Busy-poll for low latency (no sleep)
    }
}

bool RDMAConnection::wait_recv_completion(uint64_t wr_id, int timeout_ms) {
    // Recv completions come from the separate recv CQ and correspond to RQ WRs
    // (post_recv), which are not tracked by outstanding_wrs_ (send-only). So we
    // intentionally do NOT touch the counter here.
    auto start = std::chrono::steady_clock::now();
    ibv_wc wc;

    while (true) {
        int n = poll_recv_cq(&wc, 1);
        if (n > 0) {
            if (wc.status != IBV_WC_SUCCESS) {
                Logger::error("Recv completion error: %s (wr_id=%lu)",
                             wc_status_to_string(wc.status), wc.wr_id);
                return false;
            }
            if (wc.wr_id == wr_id) {
                return true;
            }
        } else if (n < 0) {
            Logger::error("Poll CQ error");
            return false;
        }

        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
        if (elapsed >= timeout_ms) {
            Logger::error("Recv completion timeout (wr_id=%lu)", wr_id);
            return false;
        }

        // Busy-poll for low latency (no sleep)
    }
}

bool RDMAConnection::wait_completions(uint64_t base_wr_id, uint32_t count, int timeout_ms) {
    if (count == 0) return true;

    auto start = std::chrono::steady_clock::now();
    std::vector<bool> completed(count, false);
    uint32_t completed_count = 0;

    // Poll for completions in batches for efficiency
    constexpr int POLL_BATCH_SIZE = 16;
    ibv_wc wc_array[POLL_BATCH_SIZE];

    while (completed_count < count) {
        int n = poll_send_cq(wc_array, POLL_BATCH_SIZE);

        if (n > 0) {
            for (int i = 0; i < n; i++) {
                if (wc_array[i].status != IBV_WC_SUCCESS) {
                    Logger::error("Batch completion error: %s (wr_id=%lu)",
                                 wc_status_to_string(wc_array[i].status), wc_array[i].wr_id);
                    return false;
                }

                // Check if this completion is in our range
                if (wc_array[i].wr_id >= base_wr_id &&
                    wc_array[i].wr_id < base_wr_id + count) {
                    uint32_t idx = static_cast<uint32_t>(wc_array[i].wr_id - base_wr_id);
                    if (!completed[idx]) {
                        completed[idx] = true;
                        completed_count++;
                    }
                }
            }
            // All n CQEs were successful here (errors return false above); each
            // frees an SQ slot regardless of whether it fell in [base, base+count).
            consume_send_wrs(static_cast<uint32_t>(n));
        } else if (n < 0) {
            Logger::error("Poll CQ error in wait_completions");
            return false;
        }

        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
        if (elapsed >= timeout_ms) {
            Logger::error("Batch completion timeout: completed %u/%u", completed_count, count);
            return false;
        }

        // Busy-poll for low latency
    }

    return true;
}

//=============================================================================
// ConnectionExchanger Implementation
//=============================================================================

// TCP does not guarantee that a single read()/write() transfers the full
// requested length, especially when streaming many small structs back-to-back.
// These helpers loop until the exact byte count is transferred, retrying on
// EINTR. A return of 0 from read()/write() means the peer closed the socket.
// Returns false on peer close or unrecoverable error.
static bool read_fully(int fd, void* buf, size_t len) {
    uint8_t* p = static_cast<uint8_t*>(buf);
    size_t total = 0;
    while (total < len) {
        ssize_t n = read(fd, p + total, len - total);
        if (n > 0) {
            total += static_cast<size_t>(n);
        } else if (n == 0) {
            return false;  // peer closed connection
        } else {
            if (errno == EINTR) continue;
            return false;
        }
    }
    return true;
}

static bool write_fully(int fd, const void* buf, size_t len) {
    const uint8_t* p = static_cast<const uint8_t*>(buf);
    size_t total = 0;
    while (total < len) {
        ssize_t n = write(fd, p + total, len - total);
        if (n > 0) {
            total += static_cast<size_t>(n);
        } else if (n == 0) {
            return false;
        } else {
            if (errno == EINTR) continue;
            return false;
        }
    }
    return true;
}

void ConnectionExchanger::send_info(int sockfd, const ConnectionInfo& info) {
    RDMA_CHECK(write_fully(sockfd, &info, sizeof(info)),
               "Failed to send connection info: peer closed during exchange");
}

ConnectionInfo ConnectionExchanger::recv_info(int sockfd) {
    ConnectionInfo info;
    RDMA_CHECK(read_fully(sockfd, &info, sizeof(info)),
               "Failed to receive connection info: peer closed during exchange");
    return info;
}

ConnectionInfo ConnectionExchanger::exchange_as_server(
    uint16_t port,
    const ConnectionInfo& local_info,
    int timeout_sec) {

    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    RDMA_CHECK(listen_fd >= 0, "Failed to create socket");

    int opt = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = INADDR_ANY;

    int ret = bind(listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
    if (ret < 0) {
        close(listen_fd);
        throw RDMAException("Failed to bind", errno);
    }

    ret = listen(listen_fd, 1);
    if (ret < 0) {
        close(listen_fd);
        throw RDMAException("Failed to listen", errno);
    }

    Logger::info("Waiting for client connection on port %u...", port);

    // Poll for incoming connection with timeout
    pollfd pfd = {listen_fd, POLLIN, 0};
    ret = poll(&pfd, 1, timeout_sec * 1000);
    if (ret <= 0) {
        close(listen_fd);
        throw RDMAException("Accept timeout or error");
    }

    sockaddr_in client_addr = {};
    socklen_t client_len = sizeof(client_addr);
    int client_fd = accept(listen_fd, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
    close(listen_fd);

    if (client_fd < 0) {
        throw RDMAException("Failed to accept", errno);
    }

    // Bound blocking handshake I/O so a stalled client cannot block the setup
    // thread indefinitely; on expiry read_fully/write_fully return false and we
    // throw cleanly. Reuse the existing accept timeout value.
    timeval tv = {timeout_sec, 0};
    setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    Logger::info("Client connected from %s:%u",
                 inet_ntoa(client_addr.sin_addr), ntohs(client_addr.sin_port));

    // Exchange info
    send_info(client_fd, local_info);
    ConnectionInfo remote_info = recv_info(client_fd);
    close(client_fd);

    return remote_info;
}

ConnectionInfo ConnectionExchanger::exchange_as_client(
    const std::string& server_addr,
    uint16_t port,
    const ConnectionInfo& local_info,
    int timeout_sec) {

    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    RDMA_CHECK(sockfd >= 0, "Failed to create socket");

    // Set connection timeout
    timeval tv = {timeout_sec, 0};
    setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(sockfd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (inet_pton(AF_INET, server_addr.c_str(), &addr.sin_addr) != 1) {
        throw RDMAException("invalid peer address: " + server_addr);
    }

    Logger::info("Connecting to server %s:%u...", server_addr.c_str(), port);

    const int max_retries = 120;
    int ret = -1;
    for (int attempt = 0; attempt < max_retries; attempt++) {
        ret = connect(sockfd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
        if (ret == 0) break;
        if (errno != ECONNREFUSED && errno != ETIMEDOUT) break;
        if (attempt < max_retries - 1) {
            Logger::info("Connection refused, retrying in 1s (%d/%d)...", attempt + 1, max_retries);
            close(sockfd);
            sleep(1);
            sockfd = socket(AF_INET, SOCK_STREAM, 0);
            RDMA_CHECK(sockfd >= 0, "Failed to create socket");
            setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
            setsockopt(sockfd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
        }
    }
    if (ret < 0) {
        close(sockfd);
        throw RDMAException("Failed to connect to " + server_addr, errno);
    }

    Logger::info("Connected to server");

    // Exchange info (client sends first after receiving server's info)
    ConnectionInfo remote_info = recv_info(sockfd);
    send_info(sockfd, local_info);
    close(sockfd);

    return remote_info;
}

std::vector<ConnectionInfo> ConnectionExchanger::exchange_multi_as_server(
    uint16_t port,
    const std::vector<ConnectionInfo>& local_infos,
    int timeout_sec) {

    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    RDMA_CHECK(listen_fd >= 0, "Failed to create socket");

    int opt = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = INADDR_ANY;

    int ret = bind(listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
    if (ret < 0) {
        close(listen_fd);
        throw RDMAException("Failed to bind multi-QP exchange port", errno);
    }

    ret = listen(listen_fd, 1);
    if (ret < 0) {
        close(listen_fd);
        throw RDMAException("Failed to listen", errno);
    }

    Logger::info("Multi-QP server: waiting on port %u for %zu QPs...", port, local_infos.size());

    pollfd pfd = {listen_fd, POLLIN, 0};
    ret = poll(&pfd, 1, timeout_sec * 1000);
    if (ret <= 0) {
        close(listen_fd);
        throw RDMAException("Multi-QP accept timeout");
    }

    sockaddr_in client_addr = {};
    socklen_t client_len = sizeof(client_addr);
    int client_fd = accept(listen_fd, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
    close(listen_fd);

    if (client_fd < 0) {
        throw RDMAException("Failed to accept", errno);
    }

    // Bound blocking handshake I/O (see exchange_as_server). Reuse timeout_sec.
    timeval tv = {timeout_sec, 0};
    setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    // Disable Nagle for bulk exchange
    int flag = 1;
    setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

    // Exchange count
    uint32_t count = static_cast<uint32_t>(local_infos.size());
    RDMA_CHECK(write_fully(client_fd, &count, sizeof(count)),
               "Failed to send QP count: peer closed during exchange");

    uint32_t remote_count = 0;
    RDMA_CHECK(read_fully(client_fd, &remote_count, sizeof(remote_count)),
               "Failed to recv QP count: peer closed during exchange");
    RDMA_CHECK(remote_count == count, "QP count mismatch");

    // Send all local infos, then receive all remote infos
    for (uint32_t i = 0; i < count; i++) {
        send_info(client_fd, local_infos[i]);
    }

    std::vector<ConnectionInfo> remote_infos(count);
    for (uint32_t i = 0; i < count; i++) {
        remote_infos[i] = recv_info(client_fd);
    }

    close(client_fd);
    Logger::info("Multi-QP server: exchanged %u QPs", count);
    return remote_infos;
}

std::vector<ConnectionInfo> ConnectionExchanger::exchange_multi_as_client(
    const std::string& server_addr,
    uint16_t port,
    const std::vector<ConnectionInfo>& local_infos,
    int timeout_sec,
    int max_retries,
    int retry_interval_ms) {

    int sockfd = -1;
    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (inet_pton(AF_INET, server_addr.c_str(), &addr.sin_addr) != 1) {
        throw RDMAException("invalid peer address: " + server_addr);
    }

    // Retry loop for connection (server may not be listening yet)
    for (int attempt = 0; attempt < max_retries; attempt++) {
        sockfd = socket(AF_INET, SOCK_STREAM, 0);
        RDMA_CHECK(sockfd >= 0, "Failed to create socket");

        timeval tv = {timeout_sec, 0};
        setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        setsockopt(sockfd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

        int ret = connect(sockfd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
        if (ret == 0) {
            Logger::info("Multi-QP client: connected to %s:%u (attempt %d/%d)",
                        server_addr.c_str(), port, attempt + 1, max_retries);
            break;
        }

        close(sockfd);
        sockfd = -1;

        if (errno == ECONNREFUSED || errno == ETIMEDOUT) {
            if (attempt < max_retries - 1) {
                Logger::info("Multi-QP client: retry %d/%d (server not ready, waiting %dms)",
                            attempt + 1, max_retries, retry_interval_ms);
                std::this_thread::sleep_for(std::chrono::milliseconds(retry_interval_ms));
                continue;
            }
        }

        throw RDMAException("Failed to connect to " + server_addr + ":" +
                           std::to_string(port) + " after " +
                           std::to_string(max_retries) + " attempts", errno);
    }

    if (sockfd < 0) {
        throw RDMAException("Failed to connect after retries");
    }

    // Disable Nagle
    int flag = 1;
    setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

    // Exchange count
    uint32_t count = static_cast<uint32_t>(local_infos.size());
    uint32_t remote_count = 0;

    RDMA_CHECK(read_fully(sockfd, &remote_count, sizeof(remote_count)),
               "Failed to recv QP count: peer closed during exchange");

    RDMA_CHECK(write_fully(sockfd, &count, sizeof(count)),
               "Failed to send QP count: peer closed during exchange");

    RDMA_CHECK(remote_count == count, "QP count mismatch");

    // Receive all remote infos, then send all local infos
    std::vector<ConnectionInfo> remote_infos(count);
    for (uint32_t i = 0; i < count; i++) {
        remote_infos[i] = recv_info(sockfd);
    }

    for (uint32_t i = 0; i < count; i++) {
        send_info(sockfd, local_infos[i]);
    }

    close(sockfd);
    Logger::info("Multi-QP client: exchanged %u QPs", count);
    return remote_infos;
}

//=============================================================================
// RDMAEndpoint Implementation
//=============================================================================

RDMAEndpoint::RDMAEndpoint(const std::string& device_name, int gid_index)
    : ctx_(std::make_shared<RDMAContext>(device_name, gid_index)) {
    setup_msg_buffer();
}

RDMAEndpoint::~RDMAEndpoint() {
    // Deregister the protocol message MR before its backing buffer is freed.
    // Otherwise the pinned MR leaks and ibv_dealloc_pd later fails with EBUSY,
    // leaking the PD per endpoint.
    if (msg_mr_) {
        ibv_dereg_mr(msg_mr_);
        msg_mr_ = nullptr;
    }
    msg_buffer_.reset();
}

void RDMAEndpoint::setup_msg_buffer() {
    msg_buffer_ = std::make_unique<uint8_t[]>(MSG_BUFFER_SIZE);
    msg_mr_ = ctx_->register_memory(msg_buffer_.get(), MSG_BUFFER_SIZE,
                                    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
}

void RDMAEndpoint::listen_and_connect(uint16_t port) {
    conn_ = std::make_shared<RDMAConnection>(ctx_);

    ConnectionInfo local_info = conn_->get_local_info();
    Logger::info("Local connection info:");
    local_info.print();

    // Use 120s timeout for multi-node setups where peer initialization may be slow
    ConnectionInfo remote_info = ConnectionExchanger::exchange_as_server(port, local_info, 120);
    conn_->connect(remote_info);
}

void RDMAEndpoint::connect_to(const std::string& server_addr, uint16_t port) {
    conn_ = std::make_shared<RDMAConnection>(ctx_);

    ConnectionInfo local_info = conn_->get_local_info();
    Logger::info("Local connection info:");
    local_info.print();

    // Use 120s timeout and 60 retries for multi-node setups
    ConnectionInfo remote_info = ConnectionExchanger::exchange_as_client(server_addr, port, local_info, 120);
    conn_->connect(remote_info);
}

ibv_mr* RDMAEndpoint::register_memory(void* addr, size_t length, int access_flags) {
    return ctx_->register_memory(addr, length, access_flags);
}

void RDMAEndpoint::send_mr_info(const RemoteMemoryRegion& local_mr_info) {
    // Copy MR info to message buffer
    memcpy(msg_buffer_.get(), &local_mr_info, sizeof(local_mr_info));

    uint64_t wr_id = wr_id_counter_++;
    conn_->post_send(msg_buffer_.get(), sizeof(local_mr_info), msg_mr_, wr_id);

    if (!conn_->wait_send_completion(wr_id)) {
        throw RDMAException("Failed to send MR info");
    }
}

RemoteMemoryRegion RDMAEndpoint::recv_mr_info() {
    uint64_t wr_id = wr_id_counter_++;
    conn_->post_recv(msg_buffer_.get(), sizeof(RemoteMemoryRegion), msg_mr_, wr_id);

    if (!conn_->wait_recv_completion(wr_id)) {
        throw RDMAException("Failed to receive MR info");
    }

    RemoteMemoryRegion remote_mr;
    memcpy(&remote_mr, msg_buffer_.get(), sizeof(remote_mr));
    return remote_mr;
}

void RDMAEndpoint::rdma_read(void* local_buffer, size_t length, ibv_mr* local_mr,
                              const RemoteMemoryRegion& remote, bool blocking) {
    uint64_t wr_id = wr_id_counter_++;
    conn_->post_read(local_buffer, length, local_mr, remote, wr_id);

    if (blocking) {
        if (!conn_->wait_send_completion(wr_id)) {
            throw RDMAException("RDMA read failed");
        }
    }
}

void RDMAEndpoint::rdma_write(void* local_buffer, size_t length, ibv_mr* local_mr,
                               const RemoteMemoryRegion& remote, bool blocking) {
    uint64_t wr_id = wr_id_counter_++;
    conn_->post_write(local_buffer, length, local_mr, remote, wr_id);

    if (blocking) {
        if (!conn_->wait_send_completion(wr_id)) {
            throw RDMAException("RDMA write failed");
        }
    }
}

void RDMAEndpoint::send(void* buffer, size_t length, ibv_mr* mr, bool blocking) {
    uint64_t wr_id = wr_id_counter_++;
    conn_->post_send(buffer, length, mr, wr_id);

    if (blocking) {
        if (!conn_->wait_send_completion(wr_id)) {
            throw RDMAException("Send failed");
        }
    }
}

void RDMAEndpoint::recv(void* buffer, size_t length, ibv_mr* mr, bool blocking) {
    uint64_t wr_id = wr_id_counter_++;
    conn_->post_recv(buffer, length, mr, wr_id);

    if (blocking) {
        if (!conn_->wait_recv_completion(wr_id)) {
            throw RDMAException("Recv failed");
        }
    }
}

} // namespace rdma
