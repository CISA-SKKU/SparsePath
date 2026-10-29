// rdma_colocation_store.h
// Co-location array store for distributed RDMA vector search.
//
// Entry layout (fixed-size, 64B-aligned):
//   [VectorID: 8B uint64]  partition[16bit] | local_loc[48bit]
//   [RawVector: dim*4B float32]
//   [Degree:    4B uint32]
//   [NeighborList: slack_R*8B uint64]  (VectorID format)
//   [Padding:   to alignment]
//
// Remote address: base_addr + local_location * entry_size

#pragma once

#include "rdma/connection.h"
#include <vector>
#include <memory>
#include <cstdlib>
#include <cstring>
#include <string>
#include <fstream>

#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace rdma {

// ==========================================================================
// VectorID encoding/decoding
// ==========================================================================

static constexpr uint32_t PARTITION_BITS = 16;
static constexpr uint32_t LOCAL_BITS = 48;
static constexpr uint64_t LOCAL_MASK = (1ULL << LOCAL_BITS) - 1;

inline uint64_t encode_vector_id(uint32_t partition, uint64_t local_loc) {
    return (static_cast<uint64_t>(partition) << LOCAL_BITS) | (local_loc & LOCAL_MASK);
}

inline uint32_t decode_partition(uint64_t vector_id) {
    return static_cast<uint32_t>(vector_id >> LOCAL_BITS);
}

inline uint64_t decode_local_loc(uint64_t vector_id) {
    return vector_id & LOCAL_MASK;
}

// ==========================================================================
// ColocationConfig
// ==========================================================================

struct ColocationConfig {
    uint32_t dim = 0;
    uint32_t R = 0;
    uint32_t slack_R = 0;
    uint32_t entry_size = 0;
    uint32_t alignment = 64;
    uint32_t num_entries = 0;
    uint32_t element_size = 4;  // bytes per vector element (4=float32, 1=int8/uint8)

    // Byte offsets within entry (loaded from metadata)
    size_t vector_id_offset = 0;
    size_t vector_offset = 8;
    size_t degree_offset = 0;    // computed: 8 + dim * element_size
    size_t neighbor_offset = 0;  // computed: degree_offset + 4

    void compute_offsets() {
        vector_id_offset = 0;
        vector_offset = 8;
        degree_offset = vector_offset + dim * element_size;
        neighbor_offset = degree_offset + sizeof(uint32_t);
    }
};

// ==========================================================================
// ColocationStore — local partition data with RDMA MR registration
// ==========================================================================

class ColocationStore {
public:
    explicit ColocationStore(const ColocationConfig& config)
        : config_(config), data_(nullptr), data_size_(0), mr_(nullptr) {}

    ~ColocationStore() {
        if (mr_ && ctx_) {
            ctx_->deregister_memory(mr_);
        }
        release_data();
    }

    // Non-copyable
    ColocationStore(const ColocationStore&) = delete;
    ColocationStore& operator=(const ColocationStore&) = delete;

    // Load .coloc binary file into aligned memory
    void load(const std::string& filepath) {
        std::ifstream in(filepath, std::ios::binary | std::ios::ate);
        if (!in) {
            throw std::runtime_error("Failed to open colocation file: " + filepath);
        }

        data_size_ = static_cast<size_t>(in.tellg());
        in.seekg(0, std::ios::beg);

        // Verify size matches config
        size_t expected = static_cast<size_t>(config_.num_entries) * config_.entry_size;
        if (data_size_ != expected) {
            throw std::runtime_error(
                "Colocation file size mismatch: got " + std::to_string(data_size_) +
                ", expected " + std::to_string(expected) +
                " (" + std::to_string(config_.num_entries) + " entries × " +
                std::to_string(config_.entry_size) + " bytes)");
        }

        // DANNS_STORE_NUMA_NODE=<n>: allocate the store as an anonymous mmap
        // with an mbind(MPOL_BIND) policy so its DRAM home is deterministic
        // (the ifstream read below is single-threaded, so the legacy
        // posix_memalign path leaves placement to first-touch scheduling).
        // Symbolic values (nic/far) are resolved by node_worker before load.
        // mmap is 4 KiB-aligned, satisfying the 64 B entry alignment.
        int bind_node = -1;
        if (const char* e = std::getenv("DANNS_STORE_NUMA_NODE")) {
            char* end = nullptr;
            const long v = std::strtol(e, &end, 10);
            if (end != e && *end == '\0' && v >= 0 && v < 64)
                bind_node = static_cast<int>(v);
        }
        if (bind_node >= 0) {
            void* p = mmap(nullptr, data_size_, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (p != MAP_FAILED) {
                unsigned long nodemask = 1UL << bind_node;
                if (syscall(SYS_mbind, p, data_size_, 2 /*MPOL_BIND*/,
                            &nodemask, sizeof(nodemask) * 8, 0UL) == 0) {
                    Logger::info("ColocationStore: data bound to NUMA node %d (mbind)",
                                 bind_node);
                } else {
                    Logger::info("ColocationStore: mbind REFUSED — first-touch placement");
                }
                data_ = static_cast<char*>(p);
                data_mmapped_ = true;
            }
        }
        if (!data_mmapped_) {
            // Allocate 64B-aligned memory (legacy first-touch path)
            if (posix_memalign(reinterpret_cast<void**>(&data_),
                               config_.alignment, data_size_) != 0) {
                throw std::runtime_error("Failed to allocate aligned memory for colocation data");
            }
        }

        in.read(data_, data_size_);
        if (!in) {
            release_data();
            throw std::runtime_error("Failed to read colocation file: " + filepath);
        }

        Logger::info("ColocationStore loaded: %u entries, %zu bytes (%.2f MB)",
                     config_.num_entries, data_size_,
                     data_size_ / (1024.0 * 1024.0));
    }

    // Register the entire co-location array as one RDMA MR
    void register_rdma(std::shared_ptr<RDMAContext> ctx) {
        ctx_ = ctx;
        if (data_ && data_size_ > 0) {
            mr_ = ctx_->register_memory(data_, data_size_,
                IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE);
            Logger::info("ColocationStore RDMA MR registered: %.2f MB",
                        data_size_ / (1024.0 * 1024.0));
        }
    }

    // ---- Accessors ----

    const void* get_vector(uint32_t local_loc) const {
        return static_cast<const void*>(
            data_ + static_cast<size_t>(local_loc) * config_.entry_size
            + config_.vector_offset);
    }

    uint32_t get_degree(uint32_t local_loc) const {
        uint32_t degree;
        std::memcpy(&degree,
                    data_ + static_cast<size_t>(local_loc) * config_.entry_size
                    + config_.degree_offset,
                    sizeof(uint32_t));
        return degree;
    }

    const uint64_t* get_neighbors(uint32_t local_loc) const {
        return reinterpret_cast<const uint64_t*>(
            data_ + static_cast<size_t>(local_loc) * config_.entry_size
            + config_.neighbor_offset);
    }

    uint64_t get_vector_id(uint32_t local_loc) const {
        uint64_t vid;
        std::memcpy(&vid,
                    data_ + static_cast<size_t>(local_loc) * config_.entry_size
                    + config_.vector_id_offset,
                    sizeof(uint64_t));
        return vid;
    }

    // Raw entry pointer
    const char* get_entry(uint32_t local_loc) const {
        return data_ + static_cast<size_t>(local_loc) * config_.entry_size;
    }

    char* get_entry_mut(uint32_t local_loc) {
        return data_ + static_cast<size_t>(local_loc) * config_.entry_size;
    }

    // ---- Local write operations ----

    void write_neighbors(uint32_t local_loc, const uint64_t* neighbor_vids,
                         uint32_t new_degree) {
        char* entry = get_entry_mut(local_loc);
        std::memcpy(entry + config_.degree_offset, &new_degree, sizeof(uint32_t));
        std::memcpy(entry + config_.neighbor_offset, neighbor_vids,
                    new_degree * sizeof(uint64_t));
    }

    void write_entry(uint32_t local_loc, const float* vector,
                     const uint64_t* neighbor_vids, uint32_t new_degree) {
        char* entry = get_entry_mut(local_loc);
        std::memcpy(entry + config_.vector_offset, vector,
                    config_.dim * config_.element_size);
        std::memcpy(entry + config_.degree_offset, &new_degree, sizeof(uint32_t));
        std::memcpy(entry + config_.neighbor_offset, neighbor_vids,
                    new_degree * sizeof(uint64_t));
    }

    // RDMA MR info for remote access
    RemoteMemoryRegion get_mr_info() const {
        if (!mr_) {
            throw std::runtime_error("RDMA MR not registered");
        }
        return RemoteMemoryRegion(
            reinterpret_cast<uint64_t>(data_),
            mr_->rkey,
            data_size_
        );
    }

    ibv_mr* get_mr() const { return mr_; }
    size_t total_size() const { return data_size_; }
    uint32_t num_entries() const { return config_.num_entries; }
    const ColocationConfig& config() const { return config_; }

private:
    void release_data() {
        if (!data_) return;
        if (data_mmapped_) munmap(data_, data_size_);
        else free(data_);
        data_ = nullptr;
        data_mmapped_ = false;
    }

    ColocationConfig config_;
    char* data_;
    size_t data_size_;
    bool data_mmapped_ = false;
    std::shared_ptr<RDMAContext> ctx_;
    ibv_mr* mr_;
};

// ==========================================================================
// FetchMode — how to read remote co-location entries
//
// NOTE: Only SEPARATED is actively used. COLOCATION reads the full entry
// (vector + neighbors) in a single RDMA READ, but the extra bandwidth cost
// outweighs the saved round-trip in practice — deprecated.
//
// With SEPARATED as the baseline, vector fetch volume is further reduced
// by PQ filter rejections and batched prefetch.
// ==========================================================================

enum class FetchMode {
    COLOCATION,   // [deprecated] 1 RDMA READ for full entry (vector + neighbor list)
    SEPARATED,    // Separate RDMA READs for vector and neighbor list — the active mode
    LOCAL_MMAP    // Cross-deployed: read peer partitions from locally-mmap'd .coloc
                  // (MAP_POPULATE pre-faulted) instead of RDMA. Used for room-at-the-top
                  // measurements where we want zero network overhead.
};

// ==========================================================================
// ColocationFetcher — fetch entries from remote co-location array via RDMA
// ==========================================================================

class ColocationFetcher {
public:
    ColocationFetcher(std::shared_ptr<RDMAEndpoint> endpoint,
                      const ColocationConfig& remote_config)
        : endpoint_(std::move(endpoint)),
          config_(remote_config),
          fetch_mode_(FetchMode::COLOCATION),
          fetch_buffer_(nullptr),
          fetch_buffer_mr_(nullptr),
          fetch_buffer_entries_(0),
          write_buffer_(nullptr),
          write_buffer_mr_(nullptr),
          write_buffer_size_(0) {}

    ~ColocationFetcher() {
        if (fetch_buffer_mr_ && endpoint_) {
            endpoint_->get_context()->deregister_memory(fetch_buffer_mr_);
        }
        if (fetch_buffer_) {
            free(fetch_buffer_);
        }
        if (write_buffer_mr_ && endpoint_) {
            endpoint_->get_context()->deregister_memory(write_buffer_mr_);
        }
        if (write_buffer_) {
            free(write_buffer_);
        }
    }

    // Non-copyable
    ColocationFetcher(const ColocationFetcher&) = delete;
    ColocationFetcher& operator=(const ColocationFetcher&) = delete;

    void set_remote_mr(const RemoteMemoryRegion& mr) {
        remote_mr_ = mr;
        remote_mr_set_ = true;
    }

    void set_fetch_mode(FetchMode mode) { fetch_mode_ = mode; }

    // ---- Full entry fetch (co-location mode) ----

    // Fetch a single complete entry (vector + neighbor list)
    const char* fetch_entry(uint32_t remote_local_loc) {
        ensure_fetch_buffer(1);
        uint64_t remote_offset = static_cast<uint64_t>(remote_local_loc) * config_.entry_size;

        RemoteMemoryRegion target = remote_mr_;
        target.addr += remote_offset;
        target.size = config_.entry_size;

        endpoint_->rdma_read(fetch_buffer_, config_.entry_size,
                            fetch_buffer_mr_, target, true);
        return fetch_buffer_;
    }

    // Extract vector from fetched entry buffer
    const void* entry_get_vector(const char* entry) const {
        return static_cast<const void*>(entry + config_.vector_offset);
    }

    // Extract degree from fetched entry buffer
    uint32_t entry_get_degree(const char* entry) const {
        uint32_t degree;
        std::memcpy(&degree, entry + config_.degree_offset, sizeof(uint32_t));
        return degree;
    }

    // Extract neighbor list from fetched entry buffer
    const uint64_t* entry_get_neighbors(const char* entry) const {
        return reinterpret_cast<const uint64_t*>(entry + config_.neighbor_offset);
    }

    // ---- Separated mode: vector only ----

    const void* fetch_vector_only(uint32_t remote_local_loc) {
        ensure_fetch_buffer(1);
        uint64_t remote_offset = static_cast<uint64_t>(remote_local_loc) * config_.entry_size
                                + config_.vector_offset;
        uint32_t vec_bytes = config_.dim * config_.element_size;

        RemoteMemoryRegion target = remote_mr_;
        target.addr += remote_offset;
        target.size = vec_bytes;

        endpoint_->rdma_read(fetch_buffer_ + config_.vector_offset,
                            vec_bytes, fetch_buffer_mr_, target, true);
        return static_cast<const void*>(fetch_buffer_ + config_.vector_offset);
    }

    // ---- Separated mode: neighbor list only ----

    const uint64_t* fetch_neighbors_only(uint32_t remote_local_loc) {
        ensure_fetch_buffer(1);
        // Fetch degree + neighbor list together (more efficient than 2 reads)
        uint64_t remote_offset = static_cast<uint64_t>(remote_local_loc) * config_.entry_size
                                + config_.degree_offset;
        uint32_t nbr_bytes = sizeof(uint32_t) + config_.slack_R * sizeof(uint64_t);

        RemoteMemoryRegion target = remote_mr_;
        target.addr += remote_offset;
        target.size = nbr_bytes;

        endpoint_->rdma_read(fetch_buffer_ + config_.degree_offset,
                            nbr_bytes, fetch_buffer_mr_, target, true);
        return reinterpret_cast<const uint64_t*>(fetch_buffer_ + config_.neighbor_offset);
    }

    uint32_t fetched_degree() const {
        uint32_t degree;
        std::memcpy(&degree, fetch_buffer_ + config_.degree_offset, sizeof(uint32_t));
        return degree;
    }

    // ---- Batch operations ----

    // Batch fetch: multiple complete entries
    void fetch_entries_batch(const std::vector<uint32_t>& remote_local_locs) {
        if (remote_local_locs.empty()) return;
        uint32_t count = static_cast<uint32_t>(remote_local_locs.size());
        ensure_fetch_buffer(count);

        std::vector<SGLEntry> entries(count);
        for (uint32_t i = 0; i < count; i++) {
            entries[i].local_addr = fetch_buffer_ +
                static_cast<size_t>(i) * config_.entry_size;
            entries[i].length = config_.entry_size;
            entries[i].remote_offset =
                static_cast<uint64_t>(remote_local_locs[i]) * config_.entry_size;
        }

        auto conn = endpoint_->get_connection();
        uint64_t base_wr_id = wr_id_counter_.fetch_add(count);
        conn->post_read_batch(entries, remote_mr_, fetch_buffer_mr_, base_wr_id);
    }

    // Batch fetch: multiple vectors only (separated mode)
    void fetch_vectors_batch(const std::vector<uint32_t>& remote_local_locs) {
        if (remote_local_locs.empty()) return;
        uint32_t count = static_cast<uint32_t>(remote_local_locs.size());
        uint32_t vec_bytes = config_.dim * config_.element_size;
        ensure_fetch_buffer(count);

        std::vector<SGLEntry> entries(count);
        for (uint32_t i = 0; i < count; i++) {
            entries[i].local_addr = fetch_buffer_ +
                static_cast<size_t>(i) * config_.entry_size + config_.vector_offset;
            entries[i].length = vec_bytes;
            entries[i].remote_offset =
                static_cast<uint64_t>(remote_local_locs[i]) * config_.entry_size
                + config_.vector_offset;
        }

        auto conn = endpoint_->get_connection();
        uint64_t base_wr_id = wr_id_counter_.fetch_add(count);
        conn->post_read_batch(entries, remote_mr_, fetch_buffer_mr_, base_wr_id);
    }

    // Async batch fetch: post without waiting. Returns pending completion count.
    // Caller must call poll_batch() later to drain completions.
    uint32_t fetch_entries_batch_async(const std::vector<uint32_t>& remote_local_locs) {
        if (remote_local_locs.empty()) return 0;
        uint32_t count = static_cast<uint32_t>(remote_local_locs.size());
        ensure_fetch_buffer(count);

        std::vector<SGLEntry> entries(count);
        for (uint32_t i = 0; i < count; i++) {
            entries[i].local_addr = fetch_buffer_ +
                static_cast<size_t>(i) * config_.entry_size;
            entries[i].length = config_.entry_size;
            entries[i].remote_offset =
                static_cast<uint64_t>(remote_local_locs[i]) * config_.entry_size;
        }

        auto conn = endpoint_->get_connection();
        uint64_t base_wr_id = wr_id_counter_.fetch_add(count);
        return conn->post_read_batch_async(entries, remote_mr_, fetch_buffer_mr_, base_wr_id);
    }

    uint32_t fetch_vectors_batch_async(const std::vector<uint32_t>& remote_local_locs) {
        if (remote_local_locs.empty()) return 0;
        uint32_t count = static_cast<uint32_t>(remote_local_locs.size());
        uint32_t vec_bytes = config_.dim * config_.element_size;
        ensure_fetch_buffer(count);

        std::vector<SGLEntry> entries(count);
        for (uint32_t i = 0; i < count; i++) {
            entries[i].local_addr = fetch_buffer_ +
                static_cast<size_t>(i) * config_.entry_size + config_.vector_offset;
            entries[i].length = vec_bytes;
            entries[i].remote_offset =
                static_cast<uint64_t>(remote_local_locs[i]) * config_.entry_size
                + config_.vector_offset;
        }

        auto conn = endpoint_->get_connection();
        uint64_t base_wr_id = wr_id_counter_.fetch_add(count);
        return conn->post_read_batch_async(entries, remote_mr_, fetch_buffer_mr_, base_wr_id);
    }

    void poll_batch(uint32_t expected_count, int timeout_ms = 5000) {
        if (expected_count == 0) return;
        auto conn = endpoint_->get_connection();
        conn->poll_read_batch(expected_count, timeout_ms);
    }

    // Batch fetch neighbor lists only (degree + neighbor VIDs) for multiple entries
    void fetch_neighbors_batch(const std::vector<uint32_t>& remote_local_locs) {
        if (remote_local_locs.empty()) return;
        uint32_t count = static_cast<uint32_t>(remote_local_locs.size());
        uint32_t nbr_bytes = sizeof(uint32_t) + config_.slack_R * sizeof(uint64_t);
        ensure_fetch_buffer(count);

        std::vector<SGLEntry> entries(count);
        for (uint32_t i = 0; i < count; i++) {
            entries[i].local_addr = fetch_buffer_ +
                static_cast<size_t>(i) * config_.entry_size + config_.degree_offset;
            entries[i].length = nbr_bytes;
            entries[i].remote_offset =
                static_cast<uint64_t>(remote_local_locs[i]) * config_.entry_size
                + config_.degree_offset;
        }

        auto conn = endpoint_->get_connection();
        uint64_t base_wr_id = wr_id_counter_.fetch_add(count);
        conn->post_read_batch(entries, remote_mr_, fetch_buffer_mr_, base_wr_id);
    }

    // Access batch results — neighbor list for a given batch index
    uint32_t get_batch_degree(uint32_t batch_idx) const {
        uint32_t deg;
        std::memcpy(&deg, fetch_buffer_ + static_cast<size_t>(batch_idx) * config_.entry_size
                    + config_.degree_offset, sizeof(uint32_t));
        return deg;
    }

    const uint64_t* get_batch_neighbors(uint32_t batch_idx) const {
        return reinterpret_cast<const uint64_t*>(
            fetch_buffer_ + static_cast<size_t>(batch_idx) * config_.entry_size
            + config_.neighbor_offset);
    }

    // Access batch results
    const char* get_batch_entry(uint32_t batch_idx) const {
        return fetch_buffer_ + static_cast<size_t>(batch_idx) * config_.entry_size;
    }

    const void* get_batch_vector(uint32_t batch_idx) const {
        return static_cast<const void*>(
            fetch_buffer_ + static_cast<size_t>(batch_idx) * config_.entry_size
            + config_.vector_offset);
    }

    // ---- RDMA WRITE operations (for distributed update) ----

    // Write neighbor list (degree + neighbor VIDs) to a remote entry
    void write_neighbors(uint32_t remote_local_loc, uint32_t new_degree,
                         const uint64_t* neighbor_vids) {
        uint32_t nbr_bytes = sizeof(uint32_t) + config_.slack_R * sizeof(uint64_t);
        ensure_write_buffer(nbr_bytes);

        // Pack: [degree:4B][neighbor_vids: slack_R * 8B]
        std::memcpy(write_buffer_, &new_degree, sizeof(uint32_t));
        std::memcpy(write_buffer_ + sizeof(uint32_t), neighbor_vids,
                    new_degree * sizeof(uint64_t));
        // Fill remaining slots with INVALID sentinel
        static constexpr uint64_t INVALID_VID = 0xFFFFFFFFFFFFFFFFULL;
        for (uint32_t i = new_degree; i < config_.slack_R; i++) {
            std::memcpy(write_buffer_ + sizeof(uint32_t) + i * sizeof(uint64_t),
                        &INVALID_VID, sizeof(uint64_t));
        }

        uint64_t remote_offset = static_cast<uint64_t>(remote_local_loc) * config_.entry_size
                                + config_.degree_offset;
        RemoteMemoryRegion target = remote_mr_;
        target.addr += remote_offset;
        target.size = nbr_bytes;

        endpoint_->rdma_write(write_buffer_, nbr_bytes,
                             write_buffer_mr_, target, true);
    }

    // Write a complete co-location entry to a remote location
    void write_full_entry(uint32_t remote_local_loc, const char* entry_data) {
        ensure_write_buffer(config_.entry_size);
        std::memcpy(write_buffer_, entry_data, config_.entry_size);

        uint64_t remote_offset = static_cast<uint64_t>(remote_local_loc) * config_.entry_size;
        RemoteMemoryRegion target = remote_mr_;
        target.addr += remote_offset;
        target.size = config_.entry_size;

        endpoint_->rdma_write(write_buffer_, config_.entry_size,
                             write_buffer_mr_, target, true);
    }

    // Getters
    const ColocationConfig& config() const { return config_; }
    FetchMode fetch_mode() const { return fetch_mode_; }
    char* buffer() { return fetch_buffer_; }
    std::shared_ptr<RDMAConnection> connection() { return endpoint_->get_connection(); }

private:
    std::shared_ptr<RDMAEndpoint> endpoint_;
    ColocationConfig config_;
    RemoteMemoryRegion remote_mr_;
    bool remote_mr_set_ = false;
    FetchMode fetch_mode_;

    char* fetch_buffer_;
    ibv_mr* fetch_buffer_mr_;
    uint32_t fetch_buffer_entries_;

    char* write_buffer_;
    ibv_mr* write_buffer_mr_;
    uint32_t write_buffer_size_;

    std::atomic<uint64_t> wr_id_counter_{0};

    void ensure_fetch_buffer(uint32_t num_entries) {
        if (num_entries <= fetch_buffer_entries_) return;

        // Deregister old
        if (fetch_buffer_mr_ && endpoint_) {
            endpoint_->get_context()->deregister_memory(fetch_buffer_mr_);
            fetch_buffer_mr_ = nullptr;
        }
        if (fetch_buffer_) {
            free(fetch_buffer_);
            fetch_buffer_ = nullptr;
        }

        fetch_buffer_entries_ = num_entries;
        size_t buf_size = static_cast<size_t>(num_entries) * config_.entry_size;

        if (posix_memalign(reinterpret_cast<void**>(&fetch_buffer_),
                           config_.alignment, buf_size) != 0) {
            throw std::runtime_error("Failed to allocate fetch buffer");
        }
        std::memset(fetch_buffer_, 0, buf_size);

        fetch_buffer_mr_ = endpoint_->register_memory(
            fetch_buffer_, buf_size);

        Logger::debug("ColocationFetcher: buffer resized to %u entries", num_entries);
    }

    void ensure_write_buffer(uint32_t needed_size) {
        if (needed_size <= write_buffer_size_) return;

        // Deregister old
        if (write_buffer_mr_ && endpoint_) {
            endpoint_->get_context()->deregister_memory(write_buffer_mr_);
            write_buffer_mr_ = nullptr;
        }
        if (write_buffer_) {
            free(write_buffer_);
            write_buffer_ = nullptr;
        }

        // Allocate at least one full entry size for generality
        uint32_t alloc_size = std::max(needed_size, config_.entry_size);
        if (posix_memalign(reinterpret_cast<void**>(&write_buffer_),
                           config_.alignment, alloc_size) != 0) {
            throw std::runtime_error("Failed to allocate write buffer");
        }
        std::memset(write_buffer_, 0, alloc_size);

        write_buffer_mr_ = endpoint_->register_memory(write_buffer_, alloc_size);
        write_buffer_size_ = alloc_size;

        Logger::debug("ColocationFetcher: write buffer allocated %u bytes", alloc_size);
    }
};

} // namespace rdma
