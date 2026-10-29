// rdma_odp.cpp
// On-Demand Paging (ODP) implementation

#include "rdma/odp.h"
#include <cstring>

namespace rdma {

//=============================================================================
// ODPMemoryRegion Implementation
//=============================================================================

ODPMemoryRegion::ODPMemoryRegion(std::shared_ptr<RDMAContext> ctx,
                                   void* addr,
                                   size_t length,
                                   bool implicit)
    : ctx_(std::move(ctx)), addr_(addr), length_(length), implicit_(implicit) {

    const auto& caps = ctx_->get_device_caps();

    // Try ODP if supported
    if (caps.odp_supported) {
        if (implicit && caps.odp_implicit) {
            // Use implicit ODP (entire address space)
            register_odp_mr();
        } else if (!implicit) {
            // Use explicit ODP for this region
            register_odp_mr();
        } else {
            // Implicit requested but not supported, fall back
            Logger::warn("Implicit ODP not supported, using regular MR");
            register_regular_mr();
        }
    } else {
        Logger::info("ODP not supported, using regular MR");
        register_regular_mr();
    }
}

ODPMemoryRegion::~ODPMemoryRegion() {
    if (mr_) {
        ctx_->deregister_memory(mr_);
    }
}

void ODPMemoryRegion::register_odp_mr() {
    // IBV_ACCESS_ON_DEMAND is an enum value (1<<6), not a macro
    // So we use it directly - it's always available in modern libibverbs
    int access_flags = IBV_ACCESS_LOCAL_WRITE |
                       IBV_ACCESS_REMOTE_READ |
                       IBV_ACCESS_REMOTE_WRITE |
                       IBV_ACCESS_ON_DEMAND;

    mr_ = ibv_reg_mr(ctx_->get_pd(), addr_, length_, access_flags);

    if (mr_) {
        odp_enabled_ = true;
        Logger::info("ODP MR registered: addr=%p, len=%zu, implicit=%s",
                     addr_, length_, implicit_ ? "yes" : "no");
    } else {
        Logger::warn("ODP MR registration failed (errno=%d: %s), falling back to regular MR",
                     errno, strerror(errno));
        register_regular_mr();
    }
}

void ODPMemoryRegion::register_regular_mr() {
    int access_flags = IBV_ACCESS_LOCAL_WRITE |
                       IBV_ACCESS_REMOTE_READ |
                       IBV_ACCESS_REMOTE_WRITE;

    mr_ = ibv_reg_mr(ctx_->get_pd(), addr_, length_, access_flags);
    RDMA_CHECK_NULL(mr_, "Failed to register regular MR");

    odp_enabled_ = false;
    Logger::debug("Regular MR registered: addr=%p, len=%zu", addr_, length_);
}

bool ODPMemoryRegion::prefetch(uint64_t offset, size_t length) {
    if (!odp_enabled_ || !mr_) {
        return false;
    }

    // IBV_ADVISE_MR_ADVICE_PREFETCH is a macro defined in verbs_api.h
    // It wraps IB_UVERBS_ADVISE_MR_ADVICE_PREFETCH from ib_user_ioctl_verbs.h
    struct ibv_sge sg_list;
    sg_list.addr = reinterpret_cast<uint64_t>(addr_) + offset;
    sg_list.length = static_cast<uint32_t>(length);
    sg_list.lkey = mr_->lkey;

    int ret = ibv_advise_mr(ctx_->get_pd(),
                            IBV_ADVISE_MR_ADVICE_PREFETCH,
                            IBV_ADVISE_MR_FLAG_FLUSH,
                            &sg_list, 1);

    if (ret != 0) {
        Logger::debug("Prefetch hint failed: %s", strerror(errno));
        return false;
    }

    Logger::debug("Prefetch issued: offset=%lu, len=%zu", offset, length);
    return true;
}

bool ODPMemoryRegion::prefetch_ranges(const std::vector<std::pair<uint64_t, size_t>>& ranges) {
    if (!odp_enabled_ || !mr_ || ranges.empty()) {
        return false;
    }

    std::vector<ibv_sge> sg_list(ranges.size());

    for (size_t i = 0; i < ranges.size(); i++) {
        sg_list[i].addr = reinterpret_cast<uint64_t>(addr_) + ranges[i].first;
        sg_list[i].length = static_cast<uint32_t>(ranges[i].second);
        sg_list[i].lkey = mr_->lkey;
    }

    int ret = ibv_advise_mr(ctx_->get_pd(),
                            IBV_ADVISE_MR_ADVICE_PREFETCH,
                            IBV_ADVISE_MR_FLAG_FLUSH,
                            sg_list.data(),
                            static_cast<uint32_t>(sg_list.size()));

    if (ret != 0) {
        Logger::debug("Batch prefetch hint failed: %s", strerror(errno));
        return false;
    }

    Logger::debug("Batch prefetch issued: %zu ranges", ranges.size());
    return true;
}

bool ODPMemoryRegion::invalidate(uint64_t offset, size_t length) {
    if (!odp_enabled_ || !mr_) {
        return false;
    }

    // Invalidate is not directly supported via ibv_advise_mr
    // PREFETCH_NO_FAULT doesn't invalidate, it just doesn't fault on missing pages
    (void)offset;
    (void)length;
    Logger::debug("Invalidate not directly supported");
    return false;
}

RemoteMemoryRegion ODPMemoryRegion::get_remote_info() const {
    RemoteMemoryRegion info;
    if (mr_) {
        info.addr = reinterpret_cast<uint64_t>(addr_);
        info.rkey = mr_->rkey;
        info.size = length_;
    }
    return info;
}

//=============================================================================
// ODPVectorFetcher Implementation
//=============================================================================

ODPVectorFetcher::ODPVectorFetcher(std::shared_ptr<RDMAEndpoint> endpoint,
                                     uint32_t vector_dim,
                                     uint32_t max_batch_size)
    : endpoint_(std::move(endpoint)),
      vector_dim_(vector_dim),
      max_batch_size_(max_batch_size) {

    // Allocate buffer for max batch
    size_t buffer_elements = static_cast<size_t>(max_batch_size) * vector_dim;
    buffer_.resize(buffer_elements);

    // Register with ODP if available
    odp_mr_ = std::make_unique<ODPMemoryRegion>(
        endpoint_->get_context(),
        buffer_.data(),
        buffer_.size() * sizeof(float),
        false  // explicit ODP
    );

    Logger::info("ODPVectorFetcher created: dim=%u, max_batch=%u, ODP=%s",
                 vector_dim_, max_batch_size_,
                 odp_mr_->is_odp_enabled() ? "yes" : "no");
}

ODPVectorFetcher::~ODPVectorFetcher() {
    // odp_mr_ cleanup handled by unique_ptr
}

void ODPVectorFetcher::set_remote_mr(const RemoteMemoryRegion& remote_mr) {
    remote_mr_ = remote_mr;
    remote_mr_set_ = true;
}

void ODPVectorFetcher::prefetch_remote_vectors(const std::vector<uint32_t>& vector_ids,
                                                 size_t vector_size) {
    if (!odp_mr_ || !odp_mr_->is_odp_enabled()) {
        return;
    }

    // Build prefetch ranges for local buffer regions that will receive data
    std::vector<std::pair<uint64_t, size_t>> ranges;
    ranges.reserve(vector_ids.size());

    for (size_t i = 0; i < vector_ids.size() && i < max_batch_size_; i++) {
        uint64_t local_offset = i * vector_size;
        ranges.emplace_back(local_offset, vector_size);
    }

    odp_mr_->prefetch_ranges(ranges);
}

float* ODPVectorFetcher::fetch_vectors(const std::vector<uint32_t>& vector_ids,
                                         size_t vector_size) {
    if (!remote_mr_set_) {
        throw RDMAException("Remote MR not set");
    }

    if (vector_ids.size() > max_batch_size_) {
        throw RDMAException("Batch size exceeds maximum");
    }

    // Issue prefetch hint for local buffer
    prefetch_remote_vectors(vector_ids, vector_size);

    // Build SGL entries for batched read
    std::vector<SGLEntry> entries(vector_ids.size());
    for (size_t i = 0; i < vector_ids.size(); i++) {
        entries[i].local_addr = buffer_.data() + i * vector_dim_;
        entries[i].length = static_cast<uint32_t>(vector_size);
        entries[i].remote_offset = static_cast<uint64_t>(vector_ids[i]) * vector_size;
    }

    // Post batched reads
    uint64_t base_wr_id = wr_id_counter_.fetch_add(vector_ids.size());
    auto conn = endpoint_->get_connection();

    conn->post_read_batch(entries, remote_mr_, odp_mr_->get_mr(), base_wr_id);
    conn->wait_completions(base_wr_id, static_cast<uint32_t>(vector_ids.size()));

    return buffer_.data();
}

uint64_t ODPVectorFetcher::fetch_vectors_async(const std::vector<uint32_t>& vector_ids,
                                                 size_t vector_size) {
    if (!remote_mr_set_) {
        throw RDMAException("Remote MR not set");
    }

    if (vector_ids.size() > max_batch_size_) {
        throw RDMAException("Batch size exceeds maximum");
    }

    // Issue prefetch hint
    prefetch_remote_vectors(vector_ids, vector_size);

    // Build SGL entries
    std::vector<SGLEntry> entries(vector_ids.size());
    for (size_t i = 0; i < vector_ids.size(); i++) {
        entries[i].local_addr = buffer_.data() + i * vector_dim_;
        entries[i].length = static_cast<uint32_t>(vector_size);
        entries[i].remote_offset = static_cast<uint64_t>(vector_ids[i]) * vector_size;
    }

    // Post batched reads
    uint64_t base_wr_id = wr_id_counter_.fetch_add(vector_ids.size());
    auto conn = endpoint_->get_connection();

    conn->post_read_batch(entries, remote_mr_, odp_mr_->get_mr(), base_wr_id);

    // Return base_wr_id - caller should wait for (base_wr_id, count) completions
    return base_wr_id;
}

bool ODPVectorFetcher::wait_fetch(uint64_t wr_id, int timeout_ms) {
    // Note: This waits for a single completion, but fetch_vectors_async
    // posts multiple WRs. Caller should use connection's wait_completions instead.
    auto conn = endpoint_->get_connection();
    return conn->wait_send_completion(wr_id, timeout_ms);
}

} // namespace rdma
