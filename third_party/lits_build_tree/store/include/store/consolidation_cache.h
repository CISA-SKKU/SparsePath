#pragma once
// store/consolidation_cache.h - Search-warmed consolidation cache boundary
//
// Deleted-node cache entries are keyed by deleted VID (Di). Each entry holds
// Di's neighbor list. Neighbor vectors are stored in a global VID→vector map
// so the same active-node vector is stored exactly once regardless of how many
// Di's share it as a neighbor.

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace danns {

struct DeletedNodeExpansion {
    uint64_t vid = 0;
    uint32_t owner_node = 0;
    uint32_t local_loc = 0;
    uint32_t query_id = 0;
    uint32_t search_node_id = 0;
    uint32_t thread_id = 0;
    bool batch_mode = false;
};

struct ConsolidationCacheEntry {
    uint64_t deleted_vid = 0;
    uint32_t degree = 0;
    std::vector<uint64_t> neighbors;
    // Vectors for neighbors are looked up via ConsolidationCache::lookup_vector()
    // keyed by neighbor VID — not stored per-entry.
};

struct ConsolidationCacheStats {
    uint64_t expansion_events = 0;
    uint64_t neighbor_records = 0;
    uint64_t vector_records = 0;          // global vec store slots used
    uint64_t entries = 0;
    uint64_t bytes = 0;
    uint64_t buffer_neighbor_bytes = 0;
    uint64_t buffer_vector_bytes = 0;     // = vec arena used bytes
    uint64_t compacted_neighbor_bytes = 0;
    uint64_t compacted_vector_bytes = 0;  // = vec arena used bytes
    uint64_t compacted_index_bytes = 0;   // estimated shard map memory
    uint64_t dropped_neighbor_records = 0;
    uint64_t dropped_vector_records = 0;  // vectors dropped when global arena full
};

class ConsolidationCacheWarmer {
public:
    virtual ~ConsolidationCacheWarmer() = default;

    virtual void record_deleted_expansion(const DeletedNodeExpansion& event) noexcept = 0;

    virtual void record_deleted_neighbors(uint64_t deleted_vid,
                                          const uint64_t* neighbors,
                                          uint32_t degree) noexcept = 0;

    // neighbor_vid's vector observed while expanding deleted_vid.
    // deleted_vid is informational only; the vector is stored globally by neighbor_vid.
    virtual void record_neighbor_vector(uint64_t deleted_vid,
                                        uint64_t neighbor_vid,
                                        const void* vector_data,
                                        size_t vector_bytes) noexcept = 0;
};

class NoopConsolidationCacheWarmer final : public ConsolidationCacheWarmer {
public:
    void record_deleted_expansion(const DeletedNodeExpansion&) noexcept override {}
    void record_deleted_neighbors(uint64_t, const uint64_t*, uint32_t) noexcept override {}
    void record_neighbor_vector(uint64_t, uint64_t, const void*, size_t) noexcept override {}
};

// NUMA-bindable flat arena buffer.
//
// Default (bind_node < 0): a plain heap vector — pages are first-touched by
// whichever thread resizes, and engine init runs on an unpinned thread, so
// the arena's DRAM home is up to the scheduler. With a bind node the buffer
// becomes an anonymous mmap carrying an mbind(MPOL_BIND) policy: every page
// faults onto that node no matter which thread touches it, so consolidation
// threads pinned to that socket stream the arena without crossing UPI or the
// search socket's memory controllers. MPOL_BIND is strict — a fault OOMs if
// the bound node's DRAM is exhausted rather than spilling to the other
// socket (loud failure over silent cross-socket placement). mbind goes
// through syscall(2) to avoid a libnuma link dependency. The lazy-faulting
// mapping also skips the eager 64 GiB zero-fill resize() paid, so arena RSS
// now grows with use.
template <typename T>
class NumaBindableBuffer {
public:
    NumaBindableBuffer() = default;
    NumaBindableBuffer(const NumaBindableBuffer&) = delete;
    NumaBindableBuffer& operator=(const NumaBindableBuffer&) = delete;
    ~NumaBindableBuffer() { drop_mapping(); }

    // (Re)allocate n zero-filled elements. Returns the effective placement:
    // the bound node id, -1 for the heap path, -2 when the mapping exists
    // but mbind was refused (placement degrades to first-touch).
    int resize_zero(size_t n, int bind_node) {
        const size_t bytes = n * sizeof(T);
        if (bind_node < 0 || bind_node >= 64 || bytes == 0) {
            drop_mapping();
            heap_.resize(n, T());
            return -1;
        }
        heap_.clear();
        heap_.shrink_to_fit();
        if (map_ && map_bytes_ == bytes && bound_node_ == bind_node)
            return mbind_ok_ ? bind_node : -2;  // keep pages + policy
        drop_mapping();
        void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) {  // fall back to the legacy heap path
            heap_.resize(n, T());
            return -1;
        }
        map_ = static_cast<T*>(p);
        map_bytes_ = bytes;
        bound_node_ = bind_node;
        unsigned long nodemask = 1UL << bind_node;
        mbind_ok_ = syscall(SYS_mbind, p, bytes, kMpolBind, &nodemask,
                            sizeof(nodemask) * 8, 0UL) == 0;
        return mbind_ok_ ? bind_node : -2;
    }

    void reset() {
        drop_mapping();
        heap_.clear();
        heap_.shrink_to_fit();
    }

    T* data() noexcept { return map_ ? map_ : heap_.data(); }
    const T* data() const noexcept { return map_ ? map_ : heap_.data(); }

private:
    static constexpr int kMpolBind = 2;  // MPOL_BIND (stable kernel ABI)

    void drop_mapping() {
        if (map_) {
            munmap(map_, map_bytes_);
            map_ = nullptr;
            map_bytes_ = 0;
            bound_node_ = -1;
            mbind_ok_ = false;
        }
    }

    std::vector<T> heap_;
    T* map_ = nullptr;
    size_t map_bytes_ = 0;
    int bound_node_ = -1;
    bool mbind_ok_ = false;
};

class ConsolidationCache final : public ConsolidationCacheWarmer {
public:
    void configure(size_t vector_bytes, uint32_t max_degree) {
        if (vector_bytes_ != 0 && vector_bytes_ != vector_bytes) {
            clear();
        }
        vector_bytes_ = vector_bytes;
        max_degree_ = max_degree;
        ensure_thread_buffers();
    }

    // neighbor_records_per_thread: per-thread cap for Di neighbor lists
    // global_vector_bytes: total bytes for the global VID→vector arena
    // max_local_locs: per-partition local_loc bound used for Di dedup.
    // num_partitions: partition count for dense encoded-VID dedup indexing.
    // vector_bytes: element size (dim * sizeof); must be provided at init
    //   time so the arena is allocated before colocation data is loaded.
    void configure_limits(size_t neighbor_records_per_thread,
                          size_t global_vector_bytes,
                          size_t max_local_locs = 0,
                          size_t num_partitions = 1,
                          size_t vector_bytes = 0,
                          size_t global_neighbor_bytes = 0) {
        if (vector_bytes > 0) vector_bytes_ = vector_bytes;
        neighbor_records_limit_ = neighbor_records_per_thread;
        global_vector_bytes_limit_ = global_vector_bytes;
        global_neighbor_bytes_limit_ = global_neighbor_bytes;
        if (max_local_locs > 0) configure_dedup_domain(max_local_locs, num_partitions);
        ensure_thread_buffers();
        setup_global_stores();
    }

    // Effective arena placement from the last setup_global_stores(): the
    // bound NUMA node id, -1 legacy heap/first-touch, -2 mmap'd but mbind
    // refused (placement degraded to first-touch).
    int arena_numa_mode() const noexcept { return arena_numa_mode_; }

    // Size the global dedup bitset. Must be called before search begins.
    void configure_dedup_domain(size_t max_local_locs, size_t num_partitions = 1) {
        max_local_locs_ = max_local_locs;
        num_partitions_ = std::max<size_t>(num_partitions, 1);
        const size_t total_keys = max_local_locs_ * num_partitions_;
        recorded_di_words_ = (total_keys + 63) / 64;
        recorded_di_bits_.reset(new std::atomic<uint64_t>[recorded_di_words_]);
        for (size_t i = 0; i < recorded_di_words_; i++)
            recorded_di_bits_[i].store(0, std::memory_order_relaxed);
    }

    void clear() {
        stats_ = {};
        for (auto& b : thread_buffers_) b.clear();
        if (recorded_di_words_ > 0) {
            for (size_t i = 0; i < recorded_di_words_; i++)
                recorded_di_bits_[i].store(0, std::memory_order_relaxed);
        }
        // Reset global vec store
        vec_next_slot_.store(0, std::memory_order_relaxed);
        vec_dropped_records_.store(0, std::memory_order_relaxed);
        if (vec_slot_of_) {
            for (size_t i = 0; i < dense_keys_; i++)
                vec_slot_of_[i].store(0, std::memory_order_relaxed);
        }
        reset_nbr_store();
    }

    void record_deleted_expansion(const DeletedNodeExpansion&) noexcept override {
        WarmThreadBuffer* b = current_buffer();
        if (!b) return;
        b->expansion_events++;
    }

    void record_deleted_neighbors(uint64_t deleted_vid,
                                  const uint64_t* neighbors,
                                  uint32_t degree) noexcept override {
        if (!neighbors || degree == 0) return;
        WarmThreadBuffer* b = current_buffer();
        if (!b) return;
        const uint32_t safe_degree = max_degree_ > 0 ? std::min(degree, max_degree_) : degree;
        if (safe_degree == 0) return;
        if (b->neighbor_records.size() >= max_neighbor_records_per_thread_) {
            b->dropped_neighbor_records++;
            return;
        }
        if (b->neighbor_arena.size() + safe_degree >
            max_neighbor_records_per_thread_ * static_cast<size_t>(std::max<uint32_t>(max_degree_, 1))) {
            b->dropped_neighbor_records++;
            return;
        }
        if (!try_claim_di(deleted_vid)) return;
        const uint32_t offset = static_cast<uint32_t>(b->neighbor_arena.size());
        try {
            b->neighbor_arena.insert(b->neighbor_arena.end(), neighbors, neighbors + safe_degree);
            b->neighbor_records.push_back({deleted_vid, offset, safe_degree});
        } catch (...) {
            b->neighbor_arena.resize(offset);
            b->dropped_neighbor_records++;
        }
    }

    // Store neighbor_vid's vector in the global VID→vector map.
    // deleted_vid is unused; storage is keyed by neighbor_vid only.
    void record_neighbor_vector(uint64_t /*deleted_vid*/,
                                uint64_t neighbor_vid,
                                const void* vector_data,
                                size_t vector_bytes) noexcept override {
        if (!vector_data || vector_bytes == 0) return;
        if (vector_bytes_ != 0 && vector_bytes != vector_bytes_) return;
        if (vec_max_slots_ == 0 || vector_bytes_ == 0) return;

        // Saturate before claiming a slot: consolidation write-backs multiply
        // record attempts, and an arena-full run would otherwise keep
        // advancing vec_next_slot_ toward uint32 wrap (slot re-use would
        // overwrite live vectors whose pointers lookup_vector_ptr callers hold).
        if (vec_next_slot_.load(std::memory_order_relaxed) >= vec_max_slots_) {
            vec_dropped_records_.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        size_t key = 0;
        if (!dedup_key(neighbor_vid, key)) return;
        auto& cell = vec_slot_of_[key];
        if (cell.load(std::memory_order_acquire) != 0) return;  // already stored

        // Claim a slot
        const uint32_t slot = vec_next_slot_.fetch_add(1, std::memory_order_relaxed);
        if (slot >= vec_max_slots_) {
            vec_dropped_records_.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        // Write the payload BEFORE publishing the index entry: readers that
        // observe the release-store below must see the completed memcpy.
        const size_t off = static_cast<size_t>(slot) * vector_bytes_;
        std::memcpy(vec_arena_.data() + off,
                    static_cast<const uint8_t*>(vector_data), vector_bytes_);

        // Publish (slot+1; 0 = empty). First writer wins; a concurrent
        // duplicate recorder leaks its claimed slot but wrote identical data.
        uint32_t expected = 0;
        cell.compare_exchange_strong(expected, slot + 1,
                                     std::memory_order_release,
                                     std::memory_order_relaxed);
    }

    // Called by consolidation after search warmup. Compacts the per-thread
    // warm buffers into the dense lock-free nbr store. Resets any nbr entries
    // left over from a previous consolidation phase (write-backs are a
    // fetch-time snapshot valid only within one phase). Global vec store is
    // already ready; no per-Di vec merging needed.
    void prepare_for_consolidation() {
        stats_ = {};
        reset_nbr_store();

        for (auto& b : thread_buffers_) {
            stats_.expansion_events += b.expansion_events;
            stats_.neighbor_records += b.neighbor_records.size();
            stats_.dropped_neighbor_records += b.dropped_neighbor_records;

            for (const auto& r : b.neighbor_records)
                record_neighbors_writeback(
                    r.deleted_vid, b.neighbor_arena.data() + r.neighbor_offset,
                    r.degree);
        }

        stats_.compacted_neighbor_bytes =
            std::min(nbr_next_elem_.load(std::memory_order_relaxed),
                     nbr_max_elems_) * sizeof(uint64_t);

        for (auto& b : thread_buffers_) b.clear();

        // Reset dedup bitset for the next cycle
        if (recorded_di_words_ > 0) {
            for (size_t i = 0; i < recorded_di_words_; i++)
                recorded_di_bits_[i].store(0, std::memory_order_relaxed);
        }
    }

    // Look up Di's neighbor list (copying variant, used by the per-Di path).
    bool lookup(uint64_t deleted_vid, ConsolidationCacheEntry& out) const {
        uint32_t deg = 0;
        const uint64_t* p = lookup_neighbors_ptr(deleted_vid, deg);
        if (!p) return false;
        out.deleted_vid = deleted_vid;
        out.degree = deg;
        out.neighbors.assign(p, p + deg);
        return true;
    }

    // Zero-copy Di adjacency lookup: pointer into the nbr arena + degree, or
    // nullptr. Lock-free: one acquire-load on the dense index. The arena is
    // append-only and entries are never relocated, so the pointer stays valid
    // until the next prepare_for_consolidation()/clear().
    const uint64_t* lookup_neighbors_ptr(uint64_t deleted_vid,
                                         uint32_t& deg_out) const noexcept {
        if (nbr_max_elems_ == 0) return nullptr;
        size_t key = 0;
        if (!dedup_key(deleted_vid, key)) return nullptr;
        const uint64_t packed = nbr_entry_of_[key].load(std::memory_order_acquire);
        if (packed == 0) return nullptr;
        deg_out = static_cast<uint32_t>(packed & 0xFFFFu);
        return nbr_arena_.data() + (packed >> 16);
    }

    // Consolidation-loop nbr write-back (lock-free, any thread): store Di's
    // freshly fetched adjacency so later repair tasks skip the RDMA re-fetch.
    // Packing: (arena_elem_offset << 16) | degree; degree <= slack_R << 2^16,
    // 0 = empty (degree 0 is never recorded). Returns true iff this call
    // published the entry.
    bool record_neighbors_writeback(uint64_t deleted_vid,
                                    const uint64_t* neighbors,
                                    uint32_t degree) noexcept {
        if (!neighbors || degree == 0 || nbr_max_elems_ == 0) return false;
        const uint32_t d = max_degree_ > 0 ? std::min(degree, max_degree_) : degree;
        size_t key = 0;
        if (!dedup_key(deleted_vid, key)) return false;
        auto& cell = nbr_entry_of_[key];
        if (cell.load(std::memory_order_acquire) != 0) return false;  // cached
        // Saturation pre-check keeps a full arena from advancing the offset
        // unboundedly (mirrors the vec-store guard).
        if (nbr_next_elem_.load(std::memory_order_relaxed) + d > nbr_max_elems_) {
            nbr_dropped_records_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        const uint64_t off = nbr_next_elem_.fetch_add(d, std::memory_order_relaxed);
        if (off + d > nbr_max_elems_) {  // lost the capacity race
            nbr_dropped_records_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        // Payload before publish (release/acquire pairing with the lookup).
        std::memcpy(nbr_arena_.data() + off, neighbors,
                    static_cast<size_t>(d) * sizeof(uint64_t));
        uint64_t expected = 0;
        if (cell.compare_exchange_strong(expected, (off << 16) | d,
                                         std::memory_order_release,
                                         std::memory_order_relaxed)) {
            nbr_entries_count_.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
        return false;  // concurrent duplicate won; our range leaks (same data)
    }

    // Look up a neighbor vector by VID. Returns true and fills out_buf if found.
    // out_buf must point to at least vector_bytes_ bytes.
    bool lookup_vector(uint64_t vid, void* out_buf) const noexcept {
        const void* p = lookup_vector_ptr(vid);
        if (!out_buf || !p) return false;
        std::memcpy(out_buf, p, vector_bytes_);
        return true;
    }

    // Zero-copy lookup: returns a pointer to vid's vector INSIDE the arena, or
    // nullptr if absent — lock-free, one acquire-load on the dense index plus
    // the arena read. Slots are claimed monotonically and payloads are never
    // relocated, so the returned pointer stays valid for the cache lifetime.
    // ALIGNMENT: the slot is packed at vector_bytes_ stride (no padding). The
    // arena base is page-aligned, so callers needing aligned SIMD loads must
    // ensure vector_bytes_ is a multiple of that alignment (holds for deep
    // 384B / laion 2048B / sift 128B; NOT for spacev 100B — copy there).
    const void* lookup_vector_ptr(uint64_t vid) const noexcept {
        if (vector_bytes_ == 0 || vec_max_slots_ == 0) return nullptr;
        size_t key = 0;
        if (!dedup_key(vid, key)) return nullptr;
        const uint32_t slot1 = vec_slot_of_[key].load(std::memory_order_acquire);
        if (slot1 == 0) return nullptr;
        return vec_arena_.data() +
               static_cast<size_t>(slot1 - 1) * vector_bytes_;
    }

    // Non-destructive check: is vid's vector in the global store?
    bool has_vector(uint64_t vid) const noexcept {
        if (vector_bytes_ == 0 || vec_max_slots_ == 0) return false;
        size_t key = 0;
        if (!dedup_key(vid, key)) return false;
        return vec_slot_of_[key].load(std::memory_order_acquire) != 0;
    }

    // ---- Cache state dump/load (experiment infra) ----
    // Persists the compacted warm state (dense indexes + used arena prefixes)
    // so later runs can skip the warm-search phase entirely. Write to LOCAL
    // SSD only (laion state ≈ 52 GB/node; per-host path like
    // /data/ssd0/... naturally gives each node its own file). The header pins
    // the layout; load refuses on any mismatch. Loading into a LARGER arena
    // than the one dumped is allowed (used prefix must fit).
    struct CacheStateHeader {
        uint64_t magic = 0;
        uint64_t vector_bytes = 0;
        uint64_t dense_keys = 0;
        uint64_t vec_max_slots = 0;
        uint64_t vec_used = 0;
        uint64_t nbr_max_elems = 0;
        uint64_t nbr_used = 0;
        uint64_t nbr_entries = 0;
    };
    static constexpr uint64_t kStateMagic = 0xDA25CAC4E57A7E01ull;
    static_assert(sizeof(std::atomic<uint32_t>) == sizeof(uint32_t),
                  "dense index dump assumes plain-layout atomics");
    static_assert(sizeof(std::atomic<uint64_t>) == sizeof(uint64_t),
                  "dense index dump assumes plain-layout atomics");

    bool save_state(const std::string& path) const {
        if (vec_max_slots_ == 0 || dense_keys_ == 0) return false;
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) return false;
        CacheStateHeader h;
        h.magic = kStateMagic;
        h.vector_bytes = vector_bytes_;
        h.dense_keys = dense_keys_;
        h.vec_max_slots = vec_max_slots_;
        h.vec_used = std::min<uint64_t>(
            vec_next_slot_.load(std::memory_order_relaxed), vec_max_slots_);
        h.nbr_max_elems = nbr_max_elems_;
        h.nbr_used = std::min<uint64_t>(
            nbr_next_elem_.load(std::memory_order_relaxed), nbr_max_elems_);
        h.nbr_entries = nbr_entries_count_.load(std::memory_order_relaxed);
        bool ok = std::fwrite(&h, sizeof h, 1, f) == 1;
        ok = ok && std::fwrite(vec_slot_of_.get(), sizeof(uint32_t),
                               dense_keys_, f) == dense_keys_;
        ok = ok && std::fwrite(nbr_entry_of_.get(), sizeof(uint64_t),
                               dense_keys_, f) == dense_keys_;
        ok = ok && std::fwrite(vec_arena_.data(), 1,
                               h.vec_used * vector_bytes_, f) ==
                       h.vec_used * vector_bytes_;
        ok = ok && std::fwrite(nbr_arena_.data(), sizeof(uint64_t),
                               h.nbr_used, f) == h.nbr_used;
        std::fclose(f);
        return ok;
    }

    // Replaces the warm state (thread buffers are discarded, dedup bitset
    // reset) — call INSTEAD of prepare_for_consolidation().
    bool load_state(const std::string& path) {
        if (vec_max_slots_ == 0 || dense_keys_ == 0) return false;
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) return false;
        CacheStateHeader h;
        bool ok = std::fread(&h, sizeof h, 1, f) == 1 &&
                  h.magic == kStateMagic &&
                  h.vector_bytes == vector_bytes_ &&
                  h.dense_keys == dense_keys_ &&
                  h.vec_used <= vec_max_slots_ &&
                  h.nbr_used <= nbr_max_elems_;
        ok = ok && std::fread(vec_slot_of_.get(), sizeof(uint32_t),
                              dense_keys_, f) == dense_keys_;
        ok = ok && std::fread(nbr_entry_of_.get(), sizeof(uint64_t),
                              dense_keys_, f) == dense_keys_;
        ok = ok && std::fread(vec_arena_.data(), 1,
                              h.vec_used * vector_bytes_, f) ==
                       h.vec_used * vector_bytes_;
        ok = ok && std::fread(nbr_arena_.data(), sizeof(uint64_t),
                              h.nbr_used, f) == h.nbr_used;
        std::fclose(f);
        if (!ok) {
            // Partial read could have clobbered state — reset to empty.
            clear();
            return false;
        }
        vec_next_slot_.store(static_cast<uint32_t>(h.vec_used),
                             std::memory_order_relaxed);
        vec_dropped_records_.store(0, std::memory_order_relaxed);
        nbr_next_elem_.store(h.nbr_used, std::memory_order_relaxed);
        nbr_dropped_records_.store(0, std::memory_order_relaxed);
        nbr_entries_count_.store(h.nbr_entries, std::memory_order_relaxed);
        stats_ = {};
        stats_.neighbor_records = h.nbr_entries;
        for (auto& b : thread_buffers_) b.clear();
        if (recorded_di_words_ > 0) {
            for (size_t i = 0; i < recorded_di_words_; i++)
                recorded_di_bits_[i].store(0, std::memory_order_relaxed);
        }
        return true;
    }

    ConsolidationCacheStats stats() const {
        ConsolidationCacheStats s = stats_;
        s.entries = nbr_entries_count_.load(std::memory_order_relaxed);
        s.dropped_neighbor_records +=
            nbr_dropped_records_.load(std::memory_order_relaxed);
        s.compacted_neighbor_bytes =
            std::min(nbr_next_elem_.load(std::memory_order_relaxed),
                     nbr_max_elems_) * sizeof(uint64_t);

        // Per-thread neighbor buffer stats (accumulated before prepare finishes)
        for (const auto& b : thread_buffers_) {
            s.expansion_events += b.expansion_events;
            s.neighbor_records += b.neighbor_records.size();
            s.dropped_neighbor_records += b.dropped_neighbor_records;
            s.buffer_neighbor_bytes +=
                static_cast<uint64_t>(b.neighbor_records.size() * sizeof(WarmNeighborRecord) +
                                      b.neighbor_arena.size() * sizeof(uint64_t));
        }

        // Global vec store
        const uint32_t slots_used = vec_next_slot_.load(std::memory_order_relaxed);
        const uint32_t clamped = std::min(slots_used, vec_max_slots_);
        s.vector_records = clamped;
        s.dropped_vector_records = vec_dropped_records_.load(std::memory_order_relaxed);
        const uint64_t vec_bytes_used = static_cast<uint64_t>(clamped) * vector_bytes_;
        s.buffer_vector_bytes = vec_bytes_used;
        s.compacted_vector_bytes = vec_bytes_used;
        // Dense index memory (allocated up front): 4 B vec + 8 B nbr per key
        s.compacted_index_bytes = static_cast<uint64_t>(dense_keys_) *
            (sizeof(std::atomic<uint32_t>) + sizeof(std::atomic<uint64_t>));

        s.bytes = s.buffer_neighbor_bytes + s.compacted_neighbor_bytes +
                  s.buffer_vector_bytes + s.compacted_index_bytes;
        return s;
    }

private:
    struct WarmNeighborRecord {
        uint64_t deleted_vid = 0;
        uint32_t neighbor_offset = 0;
        uint32_t degree = 0;
    };

    struct WarmThreadBuffer {
        uint64_t expansion_events = 0;
        uint64_t dropped_neighbor_records = 0;
        std::vector<WarmNeighborRecord> neighbor_records;
        std::vector<uint64_t> neighbor_arena;

        void reserve(size_t neighbor_cap) {
            if (neighbor_records.capacity() < neighbor_cap)
                neighbor_records.reserve(neighbor_cap);
        }

        void clear() {
            expansion_events = 0;
            dropped_neighbor_records = 0;
            neighbor_records.clear();
            neighbor_arena.clear();
        }
    };

    // ---- Global VID → vector store (dense lock-free index) ----
    // Both stores index by the dense dedup_key domain (partition × local_loc)
    // with one atomic cell per key: append-only arenas + release-publish,
    // acquire-load — no mutexes anywhere on the record or lookup paths.
    NumaBindableBuffer<uint8_t> vec_arena_;
    std::unique_ptr<std::atomic<uint32_t>[]> vec_slot_of_;  // slot+1; 0 = empty
    std::atomic<uint32_t> vec_next_slot_{0};
    std::atomic<uint64_t> vec_dropped_records_{0};
    uint32_t vec_max_slots_ = 0;
    size_t global_vector_bytes_limit_ = 0;

    // ---- Global Di → neighbor-list store (dense lock-free index) ----
    NumaBindableBuffer<uint64_t> nbr_arena_;
    std::unique_ptr<std::atomic<uint64_t>[]> nbr_entry_of_;  // (off<<16)|deg; 0 = empty
    std::atomic<uint64_t> nbr_next_elem_{0};
    std::atomic<uint64_t> nbr_dropped_records_{0};
    std::atomic<uint64_t> nbr_entries_count_{0};
    uint64_t nbr_max_elems_ = 0;
    size_t global_neighbor_bytes_limit_ = 0;
    size_t dense_keys_ = 0;
    int arena_numa_mode_ = -1;

    void setup_global_stores() {
        dense_keys_ = (max_local_locs_ > 0) ? num_partitions_ * max_local_locs_ : 0;
        if (global_vector_bytes_limit_ == 0 || vector_bytes_ == 0 ||
            dense_keys_ == 0) {
            vec_max_slots_ = 0;
            vec_arena_.reset();
            vec_slot_of_.reset();
            nbr_max_elems_ = 0;
            nbr_arena_.reset();
            nbr_entry_of_.reset();
            arena_numa_mode_ = -1;
            return;
        }
        // DANNS_CACHE_NUMA_NODE=<n>: mbind both arenas to NUMA node n so
        // their DRAM home is deterministic instead of first-touch-by-init-
        // thread. Symbolic values ("far"/"nic") are resolved to a node id by
        // the Python launcher before this runs; non-numeric text is ignored.
        int bind_node = -1;
        if (const char* e = std::getenv("DANNS_CACHE_NUMA_NODE")) {
            char* end = nullptr;
            const long v = std::strtol(e, &end, 10);
            if (end != e && *end == '\0' && v >= 0 && v < 64)
                bind_node = static_cast<int>(v);
        }
        vec_max_slots_ = static_cast<uint32_t>(
            std::min<size_t>(global_vector_bytes_limit_ / vector_bytes_,
                             static_cast<size_t>(std::numeric_limits<uint32_t>::max()) - 1));
        const int vec_mode = vec_arena_.resize_zero(
            static_cast<size_t>(vec_max_slots_) * vector_bytes_, bind_node);
        vec_slot_of_.reset(new std::atomic<uint32_t>[dense_keys_]());
        vec_next_slot_.store(0, std::memory_order_relaxed);
        vec_dropped_records_.store(0, std::memory_order_relaxed);

        // nbr arena default: 1/8 of the vector arena budget (deep 64 GiB → 8
        // GiB ≈ 10M full slack_R=80 lists; a 10% delete needs ~6.4 GiB worst
        // case with every Di list cached on every node).
        const size_t nbytes = global_neighbor_bytes_limit_ > 0
            ? global_neighbor_bytes_limit_
            : std::max<size_t>(global_vector_bytes_limit_ / 8, size_t(1) << 30);
        nbr_max_elems_ = nbytes / sizeof(uint64_t);
        const int nbr_mode = nbr_arena_.resize_zero(nbr_max_elems_, bind_node);
        arena_numa_mode_ = std::min(vec_mode, nbr_mode);
        nbr_entry_of_.reset(new std::atomic<uint64_t>[dense_keys_]());
        nbr_next_elem_.store(0, std::memory_order_relaxed);
        nbr_dropped_records_.store(0, std::memory_order_relaxed);
        nbr_entries_count_.store(0, std::memory_order_relaxed);
    }

    void reset_nbr_store() {
        if (nbr_entry_of_) {
            for (size_t i = 0; i < dense_keys_; i++)
                nbr_entry_of_[i].store(0, std::memory_order_relaxed);
        }
        nbr_next_elem_.store(0, std::memory_order_relaxed);
        nbr_dropped_records_.store(0, std::memory_order_relaxed);
        nbr_entries_count_.store(0, std::memory_order_relaxed);
    }

    static constexpr size_t kDefaultNeighborRecordsPerThread = 4096;

    static size_t runtime_thread_count() noexcept {
#ifdef _OPENMP
        int n = omp_get_max_threads();
        return n > 0 ? static_cast<size_t>(n) : 1;
#else
        return 1;
#endif
    }

    static size_t current_thread_index() noexcept {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
        return tid >= 0 ? static_cast<size_t>(tid) : 0;
#else
        return 0;
#endif
    }

    void ensure_thread_buffers() {
        const size_t n = runtime_thread_count();
        if (thread_buffers_.size() < n) thread_buffers_.resize(n);
        max_neighbor_records_per_thread_ = neighbor_records_limit_ > 0
            ? neighbor_records_limit_ : kDefaultNeighborRecordsPerThread;
        for (auto& b : thread_buffers_)
            b.reserve(max_neighbor_records_per_thread_);
    }

    WarmThreadBuffer* current_buffer() noexcept {
        if (thread_buffers_.empty()) return nullptr;
        const size_t tid = current_thread_index();
        if (tid >= thread_buffers_.size()) return &thread_buffers_[0];
        return &thread_buffers_[tid];
    }

    bool try_claim_di(uint64_t vid) noexcept {
        size_t key = 0;
        if (!dedup_key(vid, key)) return true;
        const size_t idx = key / 64;
        const uint64_t mask = 1ULL << (key % 64);
        const uint64_t old = recorded_di_bits_[idx].fetch_or(mask, std::memory_order_relaxed);
        return !(old & mask);
    }

    bool di_is_recorded(uint64_t vid) noexcept {
        size_t key = 0;
        if (!dedup_key(vid, key)) return true;
        const size_t idx = key / 64;
        const uint64_t mask = 1ULL << (key % 64);
        return (recorded_di_bits_[idx].load(std::memory_order_relaxed) & mask) != 0;
    }

    bool dedup_key(uint64_t vid, size_t& key) const noexcept {
        if (max_local_locs_ == 0 || recorded_di_words_ == 0) return false;
        const size_t part = static_cast<size_t>(vid >> 48);
        const size_t local = static_cast<size_t>(vid & ((1ULL << 48) - 1));
        if (part >= num_partitions_ || local >= max_local_locs_) return false;
        key = part * max_local_locs_ + local;
        return true;
    }

    size_t vector_bytes_ = 0;
    uint32_t max_degree_ = 0;
    size_t neighbor_records_limit_ = 0;
    size_t max_neighbor_records_per_thread_ = kDefaultNeighborRecordsPerThread;
    std::vector<WarmThreadBuffer> thread_buffers_;
    ConsolidationCacheStats stats_;

    size_t max_local_locs_ = 0;
    size_t num_partitions_ = 1;
    size_t recorded_di_words_ = 0;
    std::unique_ptr<std::atomic<uint64_t>[]> recorded_di_bits_;
};

} // namespace danns
