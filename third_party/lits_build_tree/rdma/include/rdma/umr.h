// rdma_umr.h
// User Memory Region (UMR) / Indirect MKey support for NVIDIA RDMA NICs
// UMR allows mapping multiple non-contiguous memory regions into a single
// virtual contiguous memory key, enabling efficient scatter-gather from
// non-contiguous remote memory.
//
// This is the key technology for optimizing graph-based ANNS neighbor fetching
// where vectors are stored at non-contiguous locations in remote memory.

#pragma once

#include "rdma/common.h"
#include "rdma/connection.h"

// Check for mlx5 Direct Verbs availability
#if __has_include(<infiniband/mlx5dv.h>)
#define HAVE_MLX5_DV 1
#include <infiniband/mlx5dv.h>
#else
#define HAVE_MLX5_DV 0
#endif

#include <vector>
#include <memory>

namespace rdma {

// UMR capabilities from device query
struct UMRCapabilities {
    bool supported;           // UMR feature available
    uint32_t max_klm_entries; // Maximum KLM (Key-Length-Memory) entries per MKey
    uint32_t max_send_wqe_inline_klms;  // Max KLMs that can be inlined in WQE

    UMRCapabilities() : supported(false), max_klm_entries(0), max_send_wqe_inline_klms(0) {}

    void print() const {
        Logger::info("UMR Capabilities:");
        Logger::info("  supported: %s", supported ? "yes" : "no");
        if (supported) {
            Logger::info("  max_klm_entries: %u", max_klm_entries);
            Logger::info("  max_send_wqe_inline_klms: %u", max_send_wqe_inline_klms);
        }
    }
};

// KLM entry - describes one segment in the indirect MKey
struct KLMEntry {
    uint64_t addr;      // Virtual address
    uint32_t length;    // Length in bytes
    uint32_t lkey;      // Local key of the memory region

    KLMEntry() : addr(0), length(0), lkey(0) {}
    KLMEntry(uint64_t a, uint32_t len, uint32_t key)
        : addr(a), length(len), lkey(key) {}
};

#if HAVE_MLX5_DV

// UMR Manager - creates and manages indirect MKeys
class UMRManager {
public:
    UMRManager(std::shared_ptr<RDMAContext> ctx);
    ~UMRManager();

    // Non-copyable
    UMRManager(const UMRManager&) = delete;
    UMRManager& operator=(const UMRManager&) = delete;

    // Query UMR capabilities
    static UMRCapabilities query_capabilities(ibv_context* ctx);

    // Check if UMR is supported
    bool is_supported() const { return caps_.supported; }
    const UMRCapabilities& get_capabilities() const { return caps_; }

    // Create an indirect MKey that maps multiple non-contiguous regions
    // Returns the mkey value (rkey) for the indirect key
    // klm_entries: list of (address, length, lkey) tuples
    uint32_t create_indirect_mkey(const std::vector<KLMEntry>& klm_entries);

    // Update an existing indirect MKey with new mappings
    // This is faster than creating a new mkey for repeated operations
    bool update_indirect_mkey(uint32_t mkey, const std::vector<KLMEntry>& klm_entries);

    // Destroy an indirect MKey
    void destroy_indirect_mkey(uint32_t mkey);

    // Get the mlx5dv context (for advanced operations)
    mlx5dv_context* get_dv_context() const { return dv_ctx_; }

private:
    std::shared_ptr<RDMAContext> ctx_;
    mlx5dv_context* dv_ctx_ = nullptr;
    UMRCapabilities caps_;

    // Map of mkey -> mlx5dv_mkey* for cleanup
    std::unordered_map<uint32_t, mlx5dv_mkey*> mkey_map_;

    void init_dv_context();
    void query_caps();
};

// UMR-based vector fetcher for graph-based ANNS
// Uses indirect MKey to fetch multiple non-contiguous vectors in a single READ
class UMRVectorFetcher {
public:
    UMRVectorFetcher(std::shared_ptr<RDMAEndpoint> endpoint,
                     uint32_t vector_dim,
                     uint32_t max_batch_size = 32);
    ~UMRVectorFetcher();

    // Set remote memory region info (the vector store on remote node)
    void set_remote_mr(const RemoteMemoryRegion& remote_mr);

    // Fetch multiple non-contiguous vectors using UMR
    // This creates an indirect MKey mapping the remote vectors, then does
    // a single RDMA READ of the virtual contiguous region.
    //
    // vector_ids: list of vector IDs to fetch
    // vector_size: size of each vector in bytes
    // Returns: pointer to fetched data in internal buffer
    float* fetch_vectors_umr(const std::vector<uint32_t>& vector_ids,
                              size_t vector_size);

    // Check if UMR is available
    bool is_umr_available() const;

    // Get buffer for direct access
    float* get_buffer() { return buffer_.data(); }
    size_t get_buffer_size() const { return buffer_.size() * sizeof(float); }

private:
    std::shared_ptr<RDMAEndpoint> endpoint_;
    uint32_t vector_dim_;
    uint32_t max_batch_size_;

    std::vector<float> buffer_;
    ibv_mr* buffer_mr_ = nullptr;

    std::unique_ptr<UMRManager> umr_manager_;

    RemoteMemoryRegion remote_mr_;
    bool remote_mr_set_ = false;

    std::atomic<uint64_t> wr_id_counter_{0};

    // Cached indirect mkey for repeated fetches
    uint32_t cached_mkey_ = 0;
    std::vector<uint32_t> cached_vector_ids_;
};

#else  // !HAVE_MLX5_DV

// Stub implementations when mlx5dv is not available
class UMRManager {
public:
    UMRManager(std::shared_ptr<RDMAContext>) {
        Logger::warn("UMR not available: mlx5dv headers not found at compile time");
    }
    ~UMRManager() = default;

    static UMRCapabilities query_capabilities(ibv_context*) {
        return UMRCapabilities();
    }

    bool is_supported() const { return false; }
    UMRCapabilities get_capabilities() const { return UMRCapabilities(); }

    uint32_t create_indirect_mkey(const std::vector<KLMEntry>&) {
        throw RDMAException("UMR not available");
    }
    bool update_indirect_mkey(uint32_t, const std::vector<KLMEntry>&) { return false; }
    void destroy_indirect_mkey(uint32_t) {}
};

class UMRVectorFetcher {
public:
    UMRVectorFetcher(std::shared_ptr<RDMAEndpoint>, uint32_t, uint32_t = 32) {
        Logger::warn("UMRVectorFetcher not available: mlx5dv not found");
    }
    ~UMRVectorFetcher() = default;

    void set_remote_mr(const RemoteMemoryRegion&) {}
    float* fetch_vectors_umr(const std::vector<uint32_t>&, size_t) {
        throw RDMAException("UMR not available");
    }
    bool is_umr_available() const { return false; }
    float* get_buffer() { return nullptr; }
    size_t get_buffer_size() const { return 0; }
};

#endif  // HAVE_MLX5_DV

} // namespace rdma
