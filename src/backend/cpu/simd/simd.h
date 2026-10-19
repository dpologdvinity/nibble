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
struct BlockQ4_0;
struct BlockQ8_0;
}

namespace nibble::simd {

// One 32-wide Q8_0 block of 8 tokens, interleaved for the tiled GEMM: qs[g][j] holds token j's quants
// 4g..4g+3, d[j] its scale as float, sum[j] the sum of its 32 quants. Tokens past n are zero.
struct alignas(32) BlockQ8_0x8 {
  int8_t qs[8][8][4];
  float d[8];
  int32_t sum[8];
};

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

// int8 dot per group of 4 consecutive bytes: lane k = sum of x[i]*y[i] for i in [4k, 4k+4), as 8 int32 lanes.
// Sign trick after llama.cpp's mul_sum_i8_pairs_float (ggml-cpu/arch/x86/quants.c, MIT).
// Exact unless some y[i] == -128 where x[i] < 0 (sign(-128) wraps). Q4_0 passes the activation as x (y in
// [-8,7]: always exact); Q8_0 passes it as y, which is safe because quantize_row_q8_0 never emits -128 when d = amax/127 is a normal float (amax >~ 1.5e-36).
static inline __m256i mul_sum_i8_maddubs(__m256i x, __m256i y) {
  // maddubs pair sums are at most 2*128*127 = 32512 in magnitude: no int16 saturation.
  return _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(x, x), _mm256_sign_epi8(y, x)),
                           _mm256_set1_epi16(1));
}
static inline __m256i mul_sum_i8(__m256i x, __m256i y) {
#if defined(NIBBLE_HAVE_AVXVNNI)
  return _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), _mm256_sign_epi8(x, x), _mm256_sign_epi8(y, x));
#else
  return mul_sum_i8_maddubs(x, y);
#endif
}

// roundf on each lane (round half away from zero), exact for every finite float: trunc(v) is exact,
// v - trunc(v) is exact (same binade or t == 0), so adding sign(v) where |v - trunc(v)| >= 0.5 is roundf
// by definition. _mm256_round_ps(_MM_FROUND_TO_NEAREST_INT) alone would round ties to even.
// Callers passing a product must build with -ffp-contract=off: a fused x*id - t breaks the exactness.
static inline __m256 round_half_away_ps(__m256 v) {
  const __m256 sign = _mm256_set1_ps(-0.0f);
  const __m256 t = _mm256_round_ps(v, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
  const __m256 frac = _mm256_andnot_ps(sign, _mm256_sub_ps(v, t));
  const __m256 up = _mm256_and_ps(_mm256_cmp_ps(frac, _mm256_set1_ps(0.5f), _CMP_GE_OQ), _mm256_set1_ps(1.0f));
  return _mm256_add_ps(t, _mm256_or_ps(up, _mm256_and_ps(v, sign)));
}

// Bit-identical to nibble::quantize_row_q8_0 for finite x whose block d = amax/127 is a normal float
// (amax >~ 1.5e-36). Below that, rounding of tiny d and id can differ from scalar, and below ~3.7e-37 id
// overflows and scalar is UB (roundf(inf) cast). Such blocks contribute 0 to dots: their fp16 scale is 0.
// Throws on bad n.
void quantize_row_q8_0(const float* x, BlockQ8_0* y, int64_t n);
// Bit-identical to the scalar dots in quant.cpp: exact integer block sums, then the scalar's per-block
// float products accumulated in the scalar's order. Blocks may be unaligned. For Q8_0, a must not contain
// -128 (quantize_row_q8_0 output never does).
float dot_q4_0_q8_0(const BlockQ4_0* w, const BlockQ8_0* a, int64_t n_blocks);
float dot_q8_0_q8_0(const BlockQ8_0* w, const BlockQ8_0* a, int64_t n_blocks);
// Same contract as cpu::matmul_rows for Q4_0/Q8_0 weights (xq = activations from quantize_row_q8_0),
// bit-identical to it; 4 weight rows share each activation block.
void matmul_rows_q(const Tensor& w, const BlockQ8_0* xq, float* y, int64_t n, int64_t row_begin, int64_t row_end);

// Same contract as cpu::matmul_rows for F32/F16 weights (matmul_f.cpp); sums are reordered, so results
// match the scalar oracle to ~1e-6 relative, not bit-exactly.
void matmul_rows_f(const Tensor& w, const float* x, float* y, int64_t n, int64_t row_begin, int64_t row_end);
// Tiled GEMM for prefill (gemm.cpp): same contracts as matmul_rows_f / matmul_rows_q and bit-identical
// to them for any n (each output keeps the row kernel's operation order); faster once n >= 4.
void gemm_f(const Tensor& w, const float* x, float* y, int64_t n, int64_t row_begin, int64_t row_end);
// xp: the n activation rows (nb blocks each) packed by pack_q8_0_x8, (n+7)/8 * nb blocks.
void pack_q8_0_x8(const BlockQ8_0* xq, int64_t n, int64_t nb, BlockQ8_0x8* xp);
void gemm_q(const Tensor& w, const BlockQ8_0x8* xp, float* y, int64_t n, int64_t row_begin, int64_t row_end);
// out[k] = fp16_to_fp32(row[k]) for k < d.
void embed_row_f16(const uint16_t* row, float* out, int64_t d);
#endif

}  // namespace nibble::simd
