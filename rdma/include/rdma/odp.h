// rdma_odp.h
// On-Demand Paging (ODP) support for RDMA operations
// ODP allows memory regions to be registered without pinning all pages upfront.
// Combined with prefetch hints, this can optimize access patterns.

#pragma once

#include "rdma/common.h"
#include "rdma/connection.h"
#include <vector>

namespace rdma {

// ODP Memory Region wrapper
// Provides on-demand paging with optional prefetch hints
class ODPMemoryRegion {
public:
    // Create ODP-enabled memory region
    // If ODP is not supported, falls back to regular MR
    ODPMemoryRegion(std::shared_ptr<RDMAContext> ctx,
                    void* addr,
                    size_t length,
                    bool implicit = false);
    ~ODPMemoryRegion();

    // Non-copyable
    ODPMemoryRegion(const ODPMemoryRegion&) = delete;
    ODPMemoryRegion& operator=(const ODPMemoryRegion&) = delete;

    // Prefetch hint - advise kernel to page in memory regions
    // This can reduce page fault latency during RDMA operations
    // Returns true if prefetch was issued (may not be supported on all systems)
    bool prefetch(uint64_t offset, size_t length);

    // Prefetch multiple non-contiguous regions
    bool prefetch_ranges(const std::vector<std::pair<uint64_t, size_t>>& ranges);

    // Invalidate hint - advise that memory region is no longer needed
    // (optional, for cache management)
    bool invalidate(uint64_t offset, size_t length);

    // Get underlying MR for RDMA operations
    ibv_mr* get_mr() const { return mr_; }
    uint32_t get_lkey() const { return mr_ ? mr_->lkey : 0; }
    uint32_t get_rkey() const { return mr_ ? mr_->rkey : 0; }

    // Get remote MR info for sharing with peers
    RemoteMemoryRegion get_remote_info() const;

    // Check if ODP is actually being used (vs fallback)
    bool is_odp_enabled() const { return odp_enabled_; }
    bool is_implicit() const { return implicit_; }

    // Get base address and size
    void* get_addr() const { return addr_; }
    size_t get_length() const { return length_; }

private:
    std::shared_ptr<RDMAContext> ctx_;
    void* addr_;
    size_t length_;
    ibv_mr* mr_ = nullptr;
    bool odp_enabled_ = false;
    bool implicit_ = false;

    void register_odp_mr();
    void register_regular_mr();
};

// ODP-aware vector fetcher
// Uses ODP memory registration with prefetch hints for optimized access
class ODPVectorFetcher {
public:
    ODPVectorFetcher(std::shared_ptr<RDMAEndpoint> endpoint,
                     uint32_t vector_dim,
                     uint32_t max_batch_size = 32);
    ~ODPVectorFetcher();

    // Set remote memory region info
    void set_remote_mr(const RemoteMemoryRegion& remote_mr);

    // Prefetch vectors from remote - issue prefetch hints before actual read
    // vector_ids: IDs of vectors to prefetch
    // vector_size: size of each vector in bytes
    void prefetch_remote_vectors(const std::vector<uint32_t>& vector_ids,
                                  size_t vector_size);

    // Fetch vectors using ODP-registered local buffer
    // Returns pointer to fetched data in internal buffer
    float* fetch_vectors(const std::vector<uint32_t>& vector_ids,
                         size_t vector_size);

    // Async fetch with ODP
    uint64_t fetch_vectors_async(const std::vector<uint32_t>& vector_ids,
                                  size_t vector_size);
    bool wait_fetch(uint64_t wr_id, int timeout_ms = 5000);

    // Get internal buffer
    float* get_buffer() { return buffer_.data(); }
    size_t get_buffer_size() const { return buffer_.size() * sizeof(float); }

    // Check ODP status
    bool is_odp_enabled() const { return odp_mr_ && odp_mr_->is_odp_enabled(); }

private:
    std::shared_ptr<RDMAEndpoint> endpoint_;
    uint32_t vector_dim_;
    uint32_t max_batch_size_;

    std::vector<float> buffer_;
    std::unique_ptr<ODPMemoryRegion> odp_mr_;

    RemoteMemoryRegion remote_mr_;
    bool remote_mr_set_ = false;

    std::atomic<uint64_t> wr_id_counter_{0};
};

} // namespace rdma
