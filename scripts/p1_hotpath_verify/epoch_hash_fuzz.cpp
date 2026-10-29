// Differential fuzz test: danns::search_detail::EpochKeySet / EpochKeyMap vs
// the tsl::robin_set / tsl::robin_map they replace in beam_search.cpp.
//
// Every operation beam_search performs is mirrored on both containers and the
// observable results (insert() "was new" flag, count(), find() hit + value,
// size()) are compared after each step. Covers: adversarial key distributions
// (dense low bits like encode_vector_id, sentinel keys 0 / UINT64_MAX, heavy
// duplicates), growth past the reserve, per-query clear(), reserve() at
// arbitrary points, and the epoch-wraparound branch.
//
// Build + run (inside the danns container, from the repo root):
//   g++ -O2 -std=c++17 -Isearch/include -Ithird_party/diskann/include \
//       scripts/p1_hotpath_verify/epoch_hash_fuzz.cpp -o /tmp/epoch_hash_fuzz \
//     && /tmp/epoch_hash_fuzz
#include "search/epoch_hash.h"

#include <tsl/robin_map.h>
#include <tsl/robin_set.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using danns::search_detail::EpochKeyMap;
using danns::search_detail::EpochKeySet;

static long g_checks = 0;

#define CHECK(cond, ...)                                                   \
    do {                                                                   \
        ++g_checks;                                                        \
        if (!(cond)) {                                                     \
            std::fprintf(stderr, "FAIL %s:%d: %s | ", __FILE__, __LINE__,  \
                         #cond);                                           \
            std::fprintf(stderr, __VA_ARGS__);                             \
            std::fprintf(stderr, "\n");                                    \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

// Key generator mimicking the VID space: encode_vector_id(partition,
// local_loc) == (partition << 40) | local_loc, plus the two sentinels and a
// few full-entropy keys.
static uint64_t gen_key(std::mt19937_64& rng, uint32_t nparts, uint64_t nlocs) {
    uint32_t r = static_cast<uint32_t>(rng() % 100);
    if (r == 0) return 0ull;
    if (r == 1) return UINT64_MAX;
    if (r < 6) return rng();
    uint64_t part = rng() % nparts;
    uint64_t loc = rng() % nlocs;
    return (part << 40) | loc;
}

static void fuzz_set(uint64_t seed, uint32_t nparts, uint64_t nlocs,
                     int rounds, int ops_per_round, bool wrap_epoch) {
    std::mt19937_64 rng(seed);
    EpochKeySet mine;
    tsl::robin_set<uint64_t> ref;

    if (wrap_epoch) mine.set_epoch_for_test(0xFFFFFFFEu);

    for (int r = 0; r < rounds; r++) {
        // per-query reset, exactly as SearchScratch::prepare() does it
        mine.clear();
        ref.clear();
        CHECK(mine.size() == 0, "size after clear = %zu", mine.size());
        if (r % 3 == 0) {
            size_t want = 1 + (rng() % 4096);
            mine.reserve(want);
            ref.reserve(want);
            CHECK(mine.size() == 0, "reserve must not change size");
        }
        std::vector<uint64_t> seen;
        seen.reserve(ops_per_round);
        for (int i = 0; i < ops_per_round; i++) {
            uint64_t k = gen_key(rng, nparts, nlocs);
            if (rng() % 4 == 0) {
                bool a = mine.count(k);
                bool b = ref.count(k) > 0;
                CHECK(a == b, "count(%llu) mine=%d ref=%d round=%d",
                      (unsigned long long)k, (int)a, (int)b, r);
            } else {
                bool a = mine.insert(k);
                bool b = ref.insert(k).second;
                CHECK(a == b, "insert(%llu) mine=%d ref=%d round=%d",
                      (unsigned long long)k, (int)a, (int)b, r);
                seen.push_back(k);
            }
            CHECK(mine.size() == ref.size(), "size mine=%zu ref=%zu round=%d",
                  mine.size(), ref.size(), r);
        }
        // full sweep: every key inserted this round must still be present
        for (uint64_t k : seen)
            CHECK(mine.count(k), "lost key %llu in round %d",
                  (unsigned long long)k, r);
        // and nothing survives the next clear
        if (r + 1 == rounds) {
            mine.clear();
            for (uint64_t k : seen)
                CHECK(!mine.count(k), "key %llu survived clear()",
                      (unsigned long long)k);
        }
    }
}

static void fuzz_map(uint64_t seed, uint32_t nparts, uint64_t nlocs,
                     int rounds, int ops_per_round, bool wrap_epoch) {
    std::mt19937_64 rng(seed);
    EpochKeyMap mine;
    tsl::robin_map<uint64_t, uint64_t> ref;

    if (wrap_epoch) mine.set_epoch_for_test(0xFFFFFFFEu);

    for (int r = 0; r < rounds; r++) {
        mine.clear();
        ref.clear();
        CHECK(mine.size() == 0, "size after clear = %zu", mine.size());
        if (r % 3 == 0) {
            size_t want = 1 + (rng() % 4096);
            mine.reserve(want);
            ref.reserve(want);
        }
        std::vector<uint64_t> seen;
        for (int i = 0; i < ops_per_round; i++) {
            uint64_t k = gen_key(rng, nparts, nlocs);
            if (rng() % 3 == 0) {
                const uint64_t* a = mine.find(k);
                auto b = ref.find(k);
                bool a_hit = (a != nullptr);
                bool b_hit = (b != ref.end());
                CHECK(a_hit == b_hit, "find(%llu) mine=%d ref=%d round=%d",
                      (unsigned long long)k, (int)a_hit, (int)b_hit, r);
                if (a_hit)
                    CHECK(*a == b->second,
                          "find(%llu) value mine=%llu ref=%llu",
                          (unsigned long long)k, (unsigned long long)*a,
                          (unsigned long long)b->second);
            } else {
                // last-write-wins overwrite, as every beam_search write site does
                uint64_t v = rng();
                mine[k] = v;
                ref[k] = v;
                seen.push_back(k);
            }
            CHECK(mine.size() == ref.size(), "size mine=%zu ref=%zu round=%d",
                  mine.size(), ref.size(), r);
        }
        for (uint64_t k : seen) {
            const uint64_t* a = mine.find(k);
            auto b = ref.find(k);
            CHECK(a != nullptr, "lost key %llu round %d",
                  (unsigned long long)k, r);
            CHECK(b != ref.end(), "ref lost key %llu round %d",
                  (unsigned long long)k, r);
            CHECK(*a == b->second, "value mismatch key %llu mine=%llu ref=%llu",
                  (unsigned long long)k, (unsigned long long)*a,
                  (unsigned long long)b->second);
        }
        if (r + 1 == rounds) {
            mine.clear();
            for (uint64_t k : seen)
                CHECK(mine.find(k) == nullptr, "key %llu survived clear()",
                      (unsigned long long)k);
        }
    }
}

int main() {
    // (a) dense VID-like keys, high duplicate rate -> exercises probe chains
    fuzz_set(1, 8, 4096, 200, 400, false);
    fuzz_map(2, 8, 4096, 200, 400, false);
    // (b) wide key space -> exercises growth past the reserve
    fuzz_set(3, 64, 1u << 20, 60, 3000, false);
    fuzz_map(4, 64, 1u << 20, 60, 3000, false);
    // (c) tiny key space -> near-100% duplicates, worst-case clustering
    fuzz_set(5, 2, 64, 300, 300, false);
    fuzz_map(6, 2, 64, 300, 300, false);
    // (d) epoch wraparound: starts at 0xFFFFFFFE, so clear() rolls over on
    //     round 2 and must hard-zero the stamps
    fuzz_set(7, 8, 4096, 40, 500, true);
    fuzz_map(8, 8, 4096, 40, 500, true);

    std::printf("epoch_hash_fuzz: OK (%ld assertions)\n", g_checks);
    return 0;
}
