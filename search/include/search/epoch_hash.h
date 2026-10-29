#pragma once
// ============================================================================
// Epoch-stamped open-addressing hash containers for the beam-search hot path.
//
// Why
// ---
// search() resets two per-query containers (`visited`, `edge_to_entry_vid`) at
// every query. With tsl::robin_{set,map} that reset costs:
//
//   1. clear()   -> O(bucket_count) write over the whole bucket array (each
//                   bucket's dist-from-ideal marker is stamped EMPTY), i.e. a
//                   ~64 KB + ~192 KB memset per query at L=100; and
//   2. reserve() -> tsl's reserve() forwards to rehash(), and rehash_impl()
//                   *unconditionally constructs a fresh table* and swaps --
//                   it never short-circuits when the current bucket_count is
//                   already large enough. The guard that was supposed to
//                   suppress the call,
//                     bucket_count() * static_cast<size_t>(max_load_factor())
//                   truncates the 0.5f load factor to 0, so the product is
//                   always 0 and the reserve fired on *every* query: a full
//                   alloc + free of the bucket array per query per container.
//
// These containers replace both with an O(1) logical clear: every slot carries
// a 32-bit epoch stamp and a slot is live iff stamp == the table's current
// epoch. clear() bumps the epoch. reserve() only reallocates when the table
// genuinely cannot hold the requested element count.
//
// Semantics
// ---------
// Pure key -> {presence | value} containers over uint64_t keys, with the exact
// operation set beam_search uses:
//
//   EpochKeySet:  insert(k) -> true iff newly inserted;  count(k) -> bool;
//                 size();  clear();  reserve(n)
//   EpochKeyMap:  operator[](k) -> uint64_t&  (inserts value-initialized to 0
//                 on a miss, last write wins);  find(k) -> const uint64_t*
//                 (nullptr on miss);  size();  clear();  reserve(n)
//
// No iteration, no ordering, no reference stability across insert is exposed,
// and none is used by the caller -- so results are a pure function of the
// (key -> value) relation, identical to the tsl containers they replace.
// Occupancy is tracked by the epoch stamp, never by a sentinel key value, so
// every uint64_t (including 0 and UINT64_MAX) is a legal key.
//
// Invariants
// ----------
//   - slot_count() is a power of two and >= 16, so mask arithmetic is valid.
//   - size() * 2 <= slot_count() is restored after every insert (max load
//     0.5), so a probe always terminates on an empty slot.
//   - epoch_ is never 0. A fresh/rehashed slot array is zero-stamped, so
//     "stamp == epoch_" can never be satisfied by untouched memory. On the
//     4-billionth clear() the epoch would wrap to 0; that case hard-zeroes the
//     stamps and restarts at 1.
// ============================================================================

#include <cstddef>
#include <cstdint>
#include <vector>

namespace danns {
namespace search_detail {

// splitmix64 finalizer. VIDs are encode_vector_id(partition, local_loc), i.e.
// dense in the low bits and near-constant in the high bits -- the identity
// hash that libstdc++/tsl use for uint64_t would cluster badly under linear
// probing, so mix before masking.
inline uint64_t epoch_mix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

inline size_t epoch_pow2_ceil(size_t n) {
    size_t c = 16;
    while (c < n) c <<= 1;
    return c;
}

constexpr size_t EPOCH_MIN_SLOTS = 16;

// ---------------------------------------------------------------------------
// EpochKeySet -- drop-in for tsl::robin_set<uint64_t> (insert / count / size)
// ---------------------------------------------------------------------------
class EpochKeySet {
  public:
    EpochKeySet() { reset_to_minimum(); }

    // O(1) logical clear: bump the epoch, every slot becomes unreachable.
    void clear() {
        size_ = 0;
        if (++epoch_ == 0) {
            for (auto& s : slots_) s.stamp = 0;
            epoch_ = 1;
        }
    }

    size_t size() const { return size_; }
    size_t slot_count() const { return slots_.size(); }
    // Elements the table holds before it must grow (max load factor 0.5).
    size_t max_elements() const { return slots_.size() >> 1; }

    void reserve(size_t n) {
        if (n <= max_elements()) return;
        rehash_to(epoch_pow2_ceil(n << 1));
    }

    void reset_to_minimum() {
        slots_.assign(EPOCH_MIN_SLOTS, Slot{0, 0});
        std::vector<Slot>(slots_).swap(slots_);  // drop excess capacity
        size_ = 0;
        epoch_ = 1;
    }

    // Test hook (epoch_hash_fuzz.cpp): jump the epoch counter so the
    // wraparound branch in clear() is reachable without 2^32 clears. Hard-zeros
    // the stamps first, so no stale slot can alias the new epoch.
    void set_epoch_for_test(uint32_t e) {
        for (auto& s : slots_) s.stamp = 0;
        size_ = 0;
        epoch_ = e ? e : 1u;
    }

    bool count(uint64_t key) const {
        const size_t mask = slots_.size() - 1;
        size_t i = static_cast<size_t>(epoch_mix64(key)) & mask;
        for (;;) {
            const Slot& s = slots_[i];
            if (s.stamp != epoch_) return false;   // empty slot -> not present
            if (s.key == key) return true;
            i = (i + 1) & mask;
        }
    }

    // Returns true iff the key was not already present.
    bool insert(uint64_t key) {
        const size_t mask = slots_.size() - 1;
        size_t i = static_cast<size_t>(epoch_mix64(key)) & mask;
        for (;;) {
            Slot& s = slots_[i];
            if (s.stamp != epoch_) {
                s.stamp = epoch_;
                s.key = key;
                ++size_;
                if ((size_ << 1) > slots_.size()) rehash_to(slots_.size() << 1);
                return true;
            }
            if (s.key == key) return false;
            i = (i + 1) & mask;
        }
    }

  private:
    struct Slot {
        uint64_t key;
        uint32_t stamp;
    };

    void rehash_to(size_t new_slots) {
        std::vector<Slot> old;
        old.swap(slots_);
        slots_.assign(new_slots, Slot{0, 0});
        const size_t mask = new_slots - 1;
        for (const Slot& s : old) {
            if (s.stamp != epoch_) continue;
            size_t i = static_cast<size_t>(epoch_mix64(s.key)) & mask;
            while (slots_[i].stamp == epoch_) i = (i + 1) & mask;
            slots_[i].key = s.key;
            slots_[i].stamp = epoch_;
        }
    }

    std::vector<Slot> slots_;
    size_t size_ = 0;
    uint32_t epoch_ = 1;
};

// ---------------------------------------------------------------------------
// EpochKeyMap -- drop-in for tsl::robin_map<uint64_t, uint64_t>
//                (operator[] write / find read / size)
// ---------------------------------------------------------------------------
class EpochKeyMap {
  public:
    EpochKeyMap() { reset_to_minimum(); }

    void clear() {
        size_ = 0;
        if (++epoch_ == 0) {
            for (auto& s : slots_) s.stamp = 0;
            epoch_ = 1;
        }
    }

    size_t size() const { return size_; }
    size_t slot_count() const { return slots_.size(); }
    size_t max_elements() const { return slots_.size() >> 1; }

    void reserve(size_t n) {
        if (n <= max_elements()) return;
        rehash_to(epoch_pow2_ceil(n << 1));
    }

    void reset_to_minimum() {
        slots_.assign(EPOCH_MIN_SLOTS, Slot{0, 0, 0});
        std::vector<Slot>(slots_).swap(slots_);
        size_ = 0;
        epoch_ = 1;
    }

    // Test hook (epoch_hash_fuzz.cpp): jump the epoch counter so the
    // wraparound branch in clear() is reachable without 2^32 clears. Hard-zeros
    // the stamps first, so no stale slot can alias the new epoch.
    void set_epoch_for_test(uint32_t e) {
        for (auto& s : slots_) s.stamp = 0;
        size_ = 0;
        epoch_ = e ? e : 1u;
    }

    // nullptr on miss. Valid until the next insert.
    const uint64_t* find(uint64_t key) const {
        const size_t mask = slots_.size() - 1;
        size_t i = static_cast<size_t>(epoch_mix64(key)) & mask;
        for (;;) {
            const Slot& s = slots_[i];
            if (s.stamp != epoch_) return nullptr;
            if (s.key == key) return &s.val;
            i = (i + 1) & mask;
        }
    }

    // Inserts (value 0) on a miss and returns a reference, exactly like
    // std::/tsl:: operator[]. Growth is done *before* the slot is claimed, so
    // the returned reference stays valid.
    uint64_t& operator[](uint64_t key) {
        if (((size_ + 1) << 1) > slots_.size()) rehash_to(slots_.size() << 1);
        const size_t mask = slots_.size() - 1;
        size_t i = static_cast<size_t>(epoch_mix64(key)) & mask;
        for (;;) {
            Slot& s = slots_[i];
            if (s.stamp != epoch_) {
                s.stamp = epoch_;
                s.key = key;
                s.val = 0;
                ++size_;
                return s.val;
            }
            if (s.key == key) return s.val;
            i = (i + 1) & mask;
        }
    }

  private:
    struct Slot {
        uint64_t key;
        uint64_t val;
        uint32_t stamp;
    };

    void rehash_to(size_t new_slots) {
        std::vector<Slot> old;
        old.swap(slots_);
        slots_.assign(new_slots, Slot{0, 0, 0});
        const size_t mask = new_slots - 1;
        for (const Slot& s : old) {
            if (s.stamp != epoch_) continue;
            size_t i = static_cast<size_t>(epoch_mix64(s.key)) & mask;
            while (slots_[i].stamp == epoch_) i = (i + 1) & mask;
            slots_[i].key = s.key;
            slots_[i].val = s.val;
            slots_[i].stamp = epoch_;
        }
    }

    std::vector<Slot> slots_;
    size_t size_ = 0;
    uint32_t epoch_ = 1;
};

}  // namespace search_detail
}  // namespace danns
