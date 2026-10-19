#pragma once
#include <cstdint>

// NIBBLE_SIMD=OFF (cmake) leaves NIBBLE_SIMD_ENABLED undefined: scalar paths only.
#if defined(NIBBLE_SIMD_ENABLED) && defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
#define NIBBLE_HAVE_AVX2 1
#include <immintrin.h>
#if defined(__AVXVNNI__)
#define NIBBLE_HAVE_AVXVNNI 1
#endif
#endif

namespace nibble {
struct Tensor;
}

namespace nibble::simd {

inline const char* kernel_set() {
#if defined(NIBBLE_HAVE_AVXVNNI)
  return "avx2+fma+f16c+avxvnni";
#elif defined(NIBBLE_HAVE_AVX2)
  return "avx2+fma+f16c";
#else
  return "scalar";
#endif
}

#ifdef NIBBLE_HAVE_AVX2
static inline float hsum_ps(__m256 v) {
  __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  s = _mm_add_ps(s, _mm_movehl_ps(s, s));
  s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
  return _mm_cvtss_f32(s);
}

static inline int32_t hsum_epi32(__m256i v) {
  __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
  return _mm_cvtsi128_si32(s);
}

static inline __m256 load_f16x8(const uint16_t* p) {
  return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
}

// Same contract as cpu::matmul_rows for F32/F16 weights (matmul_f.cpp); sums are reordered, so results
// match the scalar oracle to ~1e-6 relative, not bit-exactly.
void matmul_rows_f(const Tensor& w, const float* x, float* y, int64_t n, int64_t row_begin, int64_t row_end);
// out[k] = fp16_to_fp32(row[k]) for k < d.
void embed_row_f16(const uint16_t* row, float* out, int64_t d);
#endif

}  // namespace nibble::simd
