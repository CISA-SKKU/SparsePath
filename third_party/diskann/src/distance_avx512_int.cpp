// AVX-512 squared-L2 kernels for the uint8/int8 datasets (P1b, 2026-07-27).
//
// Why this file exists: the portable uint8/int8 kernels in distance.cpp are
// plain scalar loops with an `omp simd` hint (their AVX2 intrinsic variants are
// walled off behind `#ifdef _WINDOWS`), and the library is compiled -mavx2, so
// on sift/spacev the distance kernel is whatever the auto-vectorizer produces
// at 256-bit width. The float path, by contrast, has a hand-written AVX2 FMA
// kernel -- which is why deep was competitive and the integer datasets were not.
//
// This TU is compiled with AVX-512 (see CMakeLists) and is therefore the ONLY
// place allowed to emit AVX-512 instructions; selection happens at runtime in
// get_distance_function() so a non-AVX-512 host keeps the portable kernel.
//
// Numerics: widen to int16, subtract, then madd_epi16 accumulates d*d into
// int32 lanes. Integer arithmetic is exact and reassociation-free, so the
// result is BIT-IDENTICAL to the scalar kernel (no recall re-validation
// needed, unlike a float SIMD rewrite). Range check: uint8 differences span
// [-255, 255] and int8 [-255, 255], both inside int16; per-lane d*d <= 65025
// and dim <= 4096 keeps the int32 accumulator far from overflow.

// distance.h uses uint32_t/int8_t without including <cstdint> itself, so it
// only compiles when the includer pulled those in first (distance.cpp does so
// transitively). Keep <cstdint> above it.
#include <cstdint>
#include <immintrin.h>

#include "distance.h"

namespace diskann
{

namespace
{

// Horizontal sum of 16 int32 lanes.
inline int32_t hsum_epi32(__m512i v)
{
    return _mm512_reduce_add_epi32(v);
}

// 32 elements per iteration: load 32 bytes, zero/sign-extend to 32x int16,
// subtract, square-accumulate via madd_epi16 (pairs into 16x int32).
template <bool SIGNED> inline float l2_sqr_avx512(const void *pa, const void *pb, uint32_t size)
{
    const char *a = static_cast<const char *>(pa);
    const char *b = static_cast<const char *>(pb);

    __m512i acc = _mm512_setzero_si512();
    uint32_t i = 0;
    for (; i + 32 <= size; i += 32)
    {
        __m256i av = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i));
        __m256i bv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + i));
        __m512i aw, bw;
        if (SIGNED)
        {
            aw = _mm512_cvtepi8_epi16(av);
            bw = _mm512_cvtepi8_epi16(bv);
        }
        else
        {
            aw = _mm512_cvtepu8_epi16(av);
            bw = _mm512_cvtepu8_epi16(bv);
        }
        __m512i d = _mm512_sub_epi16(aw, bw);
        acc = _mm512_add_epi32(acc, _mm512_madd_epi16(d, d));
    }

    int32_t result = hsum_epi32(acc);

    // Tail (e.g. spacev dim=100 leaves 4). Same integer math as the main loop.
    for (; i < size; i++)
    {
        int32_t d = SIGNED ? (int32_t)(int16_t)((const int8_t *)a)[i] - (int32_t)(int16_t)((const int8_t *)b)[i]
                           : (int32_t)(int16_t)((const uint8_t *)a)[i] - (int32_t)(int16_t)((const uint8_t *)b)[i];
        result += d * d;
    }
    return (float)result;
}

} // namespace

float AVX512DistanceL2UInt8::compare(const uint8_t *a, const uint8_t *b, uint32_t size) const
{
    return l2_sqr_avx512<false>(a, b, size);
}

float AVX512DistanceL2Int8::compare(const int8_t *a, const int8_t *b, uint32_t size) const
{
    return l2_sqr_avx512<true>(a, b, size);
}

} // namespace diskann
