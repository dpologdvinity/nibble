#pragma once
// Declarations of the AVX2 (+FMA, +F16C, optionally AVX-VNNI) kernels and the small intrinsics helpers they
// share. Everything real is guarded by NIBBLE_HAVE_AVX2, so with SIMD off (or on a CPU target without these
// extensions) this header declares nothing and the scalar paths in ref_ops.cpp are used.
//
// Contract for every kernel: it matches the scalar twin named in its comment, either bit for bit (the
// quantized dots and the tiled GEMM) or within the stated float tolerance (anything whose summation order
// differs). The translation units that must reproduce scalar rounding are compiled with -ffp-contract=off so
// the compiler cannot fuse multiply-adds behind our back (see CMakeLists.txt).
#include <cstdint>

// NIBBLE_SIMD=OFF (cmake) leaves NIBBLE_SIMD_ENABLED undefined: scalar paths only.
#if defined(NIBBLE_SIMD_ENABLED) && defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
#define NIBBLE_HAVE_AVX2 1
#include <immintrin.h>

#include "fp16.h"
#if defined(__AVXVNNI__)
#define NIBBLE_HAVE_AVXVNNI 1
#endif
#endif

namespace nibble {
struct Tensor;
struct BlockQ4_0;
struct BlockQ8_0;
struct BlockQ8_1;
struct BlockQ5_0;
struct BlockQ5_1;
struct BlockQ4_K;
struct BlockQ5_K;
struct BlockQ6_K;
struct BlockQ8_K;
struct QuantActs;
}  // namespace nibble

namespace nibble::simd {

// One 32-wide Q8_0 block of 8 tokens, interleaved for the tiled GEMM: qs[g][j] holds token j's quants
// 4g..4g+3, d[j] its scale as float, sum[j] the sum of its 32 quants. Tokens past n are zero.
struct alignas(32) BlockQ8_0x8 {
  int8_t qs[8][8][4];
  float d[8];
  int32_t sum[8];
};

// Name of the instruction set the build compiled in, shown by `nibble info` and recorded in bench rows.
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
// Horizontal sum of the 8 float lanes (fixed pairing order, so results are reproducible).
static inline float hsum_ps(__m256 v) {
  __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  s = _mm_add_ps(s, _mm_movehl_ps(s, s));
  s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
  return _mm_cvtss_f32(s);
}

// Horizontal sum of the 8 int32 lanes (exact).
static inline int32_t hsum_epi32(__m256i v) {
  __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
  return _mm_cvtsi128_si32(s);
}

// 8 consecutive half-precision values widened to 8 floats (unaligned load).
static inline __m256 load_f16x8(const uint16_t* p) {
  return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
}

// F32/F16 weight access for the matmul kernels (matmul_f.cpp, gemm.cpp): 8 lanes or one element as float.
struct LoadF32 {
  using T = float;
  static __m256 load(const float* p) { return _mm256_loadu_ps(p); }
  static float one(float v) { return v; }
};
struct LoadF16 {
  using T = uint16_t;
  static __m256 load(const uint16_t* p) { return load_f16x8(p); }
  static float one(uint16_t v) { return fp16_to_fp32(v); }
};

// int8 dot per group of 4 consecutive bytes: lane k = sum of x[i]*y[i] for i in [4k, 4k+4), as 8 int32 lanes.
// Sign trick after llama.cpp's mul_sum_i8_pairs_float (ggml-cpu/arch/x86/quants.c, MIT).
// Exact unless some y[i] == -128 where x[i] < 0 (sign(-128) wraps). Q4_0 passes the activation as x (y in
// [-8,7]: always exact); Q8_0 passes it as y, which is safe because quantize_row_q8_0 never emits -128 when d =
// amax/127 is a normal float (amax >~ 1.5e-36).
static inline __m256i mul_sum_i8_maddubs(__m256i x, __m256i y) {
  // maddubs pair sums are at most 2*128*127 = 32512 in magnitude: no int16 saturation.
  return _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(x, x), _mm256_sign_epi8(y, x)), _mm256_set1_epi16(1));
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

// e^x per lane: max relative error vs exp about 1e-7 over [-88, 88] (measured in test_ops.cpp), one
// rounding into the subnormal range, 0 below about -104, inf above ln(FLT_MAX); NaN stays NaN.
// x = n*ln2 + r (Cody-Waite, |r| <= ~0.35), e^r - 1 by degree-7 Taylor, then fma(t, 2^n, 2^n) rounds
// once even when 2^n is subnormal (n >= -149 is exact as a float). Its last op is a mul, so a translation unit
// that adds exp_ps results (a softmax sum) must build with -ffp-contract=off or g++ may fuse the two.
static inline __m256 exp_ps(__m256 x) {
  x = _mm256_max_ps(_mm256_set1_ps(-104.0f), _mm256_min_ps(_mm256_set1_ps(89.0f), x));  // NaN in x passes through
  const __m256 fn =
      _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256 r = _mm256_fnmadd_ps(fn, _mm256_set1_ps(0.693359375f), x);  // fn * ln2_hi is exact
  r = _mm256_fnmadd_ps(fn, _mm256_set1_ps(-2.12194440e-4f), r);
  __m256 q = _mm256_set1_ps(1.0f / 5040);
  q = _mm256_fmadd_ps(q, r, _mm256_set1_ps(1.0f / 720));
  q = _mm256_fmadd_ps(q, r, _mm256_set1_ps(1.0f / 120));
  q = _mm256_fmadd_ps(q, r, _mm256_set1_ps(1.0f / 24));
  q = _mm256_fmadd_ps(q, r, _mm256_set1_ps(1.0f / 6));
  q = _mm256_fmadd_ps(q, r, _mm256_set1_ps(0.5f));
  q = _mm256_fmadd_ps(q, r, _mm256_set1_ps(1.0f));
  const __m256 t = _mm256_mul_ps(q, r);  // e^r - 1
  // s = 2^min(n, 127): normal exponent bits for n >= -126, else the subnormal bit 1 << (n + 149) (0 below -149).
  const __m256i n = _mm256_cvtps_epi32(fn), nc = _mm256_min_epi32(n, _mm256_set1_epi32(127));
  const __m256i norm = _mm256_slli_epi32(_mm256_add_epi32(nc, _mm256_set1_epi32(127)), 23);
  const __m256i sub = _mm256_sllv_epi32(_mm256_set1_epi32(1), _mm256_add_epi32(nc, _mm256_set1_epi32(149)));
  const __m256 s = _mm256_castsi256_ps(_mm256_blendv_epi8(sub, norm, _mm256_cmpgt_epi32(nc, _mm256_set1_epi32(-127))));
  // n = 128: one more exact doubling, which overflows to inf only when e^x does.
  const __m256 s2 =
      _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(_mm256_sub_epi32(n, nc), _mm256_set1_epi32(127)), 23));
  return _mm256_mul_ps(_mm256_fmadd_ps(t, s, s), s2);
}

// Bit-identical to nibble::quantize_row_q8_0 for finite x whose block d = amax/127 is a normal float
// (amax >~ 1.5e-36). Below that, rounding of tiny d and id can differ from scalar, and below ~3.7e-37 id
// overflows (scalar saturates qs to +-127; this may emit -128). Such blocks contribute 0 to dots: their fp16
// scale is 0. Inf input gives unspecified qs here (scalar defines them); a block with a NaN gets a NaN scale and
// zero quants, like the scalar. Throws on bad n.
void quantize_row_q8_0(const float* x, BlockQ8_0* y, int64_t n);
// Bit-identical to the scalar dots in quant.cpp: exact integer block sums, then the scalar's per-block
// float products accumulated in the scalar's order. Blocks may be unaligned. For Q8_0, a must not contain
// -128 (quantize_row_q8_0 output never does).
float dot_q4_0_q8_0(const BlockQ4_0* w, const BlockQ8_0* a, int64_t n_blocks);
float dot_q8_0_q8_0(const BlockQ8_0* w, const BlockQ8_0* a, int64_t n_blocks);
// Q5 dots (quant_avx2.cpp) and k-quant dots (kquant_avx2.cpp): bit-identical to the scalar twins in quant.h
// for finite inputs (NaN/Inf scales give NaN where the scalar does). Blocks may be unaligned. K-quant
// activations must come from quantize_row_q8_K (qs never -128); Q5_0 and Q5_1 accept any activation bytes.
float dot_q5_0_q8_0(const BlockQ5_0* w, const BlockQ8_0* a, int64_t n_blocks);
float dot_q5_1_q8_1(const BlockQ5_1* w, const BlockQ8_1* a, int64_t n_blocks);
float dot_q4_K_q8_K(const BlockQ4_K* w, const BlockQ8_K* a, int64_t n_blocks);
float dot_q5_K_q8_K(const BlockQ5_K* w, const BlockQ8_K* a, int64_t n_blocks);
float dot_q6_K_q8_K(const BlockQ6_K* w, const BlockQ8_K* a, int64_t n_blocks);
// matmul_rows_q's k-quant half: weights of type Q4_K, Q5_K or Q6_K against xq, 4 weight rows per activation block.
void matmul_rows_k(const Tensor& w, const BlockQ8_K* xq, float* y, int64_t n, int64_t row_begin, int64_t row_end);
// Same contract as cpu::matmul_rows for quantized weights (xq from the quantizers in quant.h), bit-identical
// to it. Every quantized type runs an AVX2 kernel, 4 weight rows sharing each activation block, then a 1-row
// remainder.
void matmul_rows_q(const Tensor& w, const QuantActs& xq, float* y, int64_t n, int64_t row_begin, int64_t row_end);

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

// Same contracts as the cpu:: ops in ref_ops.h (ops.cpp). add is bit-identical; rmsnorm keeps the scalar's
// double math and differs only in sum order; silu_mul and attention use exp_ps and reordered float sums.
// Every element is computed the same way wherever it falls in a call (tails go through the vector path or
// depend only on the row width), so results do not depend on n, chunking or threads.
void add(float* x, const float* y, int64_t count);
void rmsnorm(const float* x, const Tensor& weight, float* y, int64_t n, float eps);
void silu_mul(float* gate, const float* up, int64_t count);
void attention_heads(const float* q, const float* k, const float* v, float* out, int64_t n, int64_t pos0,
                     int64_t n_heads, int64_t n_kv_heads, int64_t head_dim, int64_t head_begin, int64_t head_end,
                     float* scratch);
#endif

}  // namespace nibble::simd
