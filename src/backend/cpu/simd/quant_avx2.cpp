// AVX2 Q8_0 activation quantizer and Q4_0/Q8_0 x Q8_0 dots. Scalar oracle: quant.cpp.
// Integer technique after llama.cpp's ggml-cpu/arch/x86/quants.c (MIT): nibble unpack as in
// bytes_from_nibbles_32, sign trick as in mul_sum_i8_pairs_float. Unlike llama.cpp, results are
// bit-identical to the scalar code: the block integer sums are exact, the per-block float products use the
// scalar's operation order, and they are accumulated sequentially per row (compiled with -ffp-contract=off).
// The 4-row kernel reduces four rows' block sums with one hadd tree, which keeps that order at no cost.
#include "backend/cpu/simd/simd.h"

#ifdef NIBBLE_HAVE_AVX2
#include <cstring>

#include "backend/cpu/ref_ops.h"
#include "fp16.h"
#include "quant.h"
#include "tensor.h"
#include <nibble/error.h>

namespace nibble::simd {
namespace {

using u8 = unsigned char;

uint16_t load_d(const u8* blk) {
  uint16_t d;
  std::memcpy(&d, blk, 2);
  return d;
}

// Weight block layout and its integer dot with one activation block (8 int32 partial sums).
struct Q4 {
  static constexpr size_t kBytes = sizeof(BlockQ4_0);
  static __m256i isum(const u8* w, __m256i a) {
    const __m128i t = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + 2));
    const __m256i nib = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(t, 4), t), _mm256_set1_epi8(0x0F));
    // The activation is the sign source: w - 8 is in [-8,7], so mul_sum_i8 is exact even for a == -128.
    return mul_sum_i8(a, _mm256_sub_epi8(nib, _mm256_set1_epi8(8)));
  }
  // quant.cpp: float(s) * dw * da.
  static __m128 scale(__m128 s, __m128 dw, float da) { return _mm_mul_ps(_mm_mul_ps(s, dw), _mm_set1_ps(da)); }
  static float scale1(int32_t s, float dw, float da) { return static_cast<float>(s) * dw * da; }
};
struct Q8 {
  static constexpr size_t kBytes = sizeof(BlockQ8_0);
  static __m256i isum(const u8* w, __m256i a) {
    return mul_sum_i8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 2)), a);
  }
  // quant.cpp: float(s) * (dw * da).
  static __m128 scale(__m128 s, __m128 dw, float da) { return _mm_mul_ps(s, _mm_mul_ps(dw, _mm_set1_ps(da))); }
  static float scale1(int32_t s, float dw, float da) { return static_cast<float>(s) * (dw * da); }
};

__m256i load_act(const u8* a) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + 2)); }

template <class W>
float dot1(const u8* w, const u8* a, int64_t nb) {
  float sum = 0.f;
  for (int64_t b = 0; b < nb; ++b, w += W::kBytes, a += sizeof(BlockQ8_0))
    sum += W::scale1(hsum_epi32(W::isum(w, load_act(a))), _cvtsh_ss(load_d(w)), _cvtsh_ss(load_d(a)));
  return sum;
}

// Four consecutive weight rows (stride rb bytes) against one activation row; out[0..3].
template <class W>
void dot4(const u8* w, size_t rb, const u8* a, int64_t nb, float* out) {
  float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
  for (int64_t b = 0; b < nb; ++b, w += W::kBytes, a += sizeof(BlockQ8_0)) {
    const __m256i av = load_act(a);
    const __m256i p0 = W::isum(w, av), p1 = W::isum(w + rb, av), p2 = W::isum(w + 2 * rb, av),
                  p3 = W::isum(w + 3 * rb, av);
    // Per 128-bit lane: [p0, p1, p2, p3] partial sums; adding the two lanes gives the 4 exact block sums.
    const __m256i h = _mm256_hadd_epi32(_mm256_hadd_epi32(p0, p1), _mm256_hadd_epi32(p2, p3));
    const __m128i s = _mm_add_epi32(_mm256_castsi256_si128(h), _mm256_extracti128_si256(h, 1));
    const __m128i dw = _mm_setr_epi16(static_cast<short>(load_d(w)), static_cast<short>(load_d(w + rb)),
                                      static_cast<short>(load_d(w + 2 * rb)),
                                      static_cast<short>(load_d(w + 3 * rb)), 0, 0, 0, 0);
    alignas(16) float p[4];
    _mm_store_ps(p, W::scale(_mm_cvtepi32_ps(s), _mm_cvtph_ps(dw), _cvtsh_ss(load_d(a))));
    s0 += p[0];
    s1 += p[1];
    s2 += p[2];
    s3 += p[3];
  }
  out[0] = s0;
  out[1] = s1;
  out[2] = s2;
  out[3] = s3;
}

template <class W>
void matmul_t(const u8* w, int64_t nb, int64_t out, const BlockQ8_0* xq, float* y, int64_t n, int64_t rb,
              int64_t re) {
  const size_t row = static_cast<size_t>(nb) * W::kBytes;
  for (int64_t t = 0; t < n; ++t) {
    const u8* a = reinterpret_cast<const u8*>(xq + t * nb);
    float* yt = y + t * out;
    int64_t r = rb;
    for (; r + 4 <= re; r += 4) dot4<W>(w + static_cast<size_t>(r) * row, row, a, nb, yt + r);
    for (; r < re; ++r) yt[r] = dot1<W>(w + static_cast<size_t>(r) * row, a, nb);
  }
}

}  // namespace

void quantize_row_q8_0(const float* x, BlockQ8_0* y, int64_t n) {
  if (n < 0 || n % kQK != 0) throw Error("quant: length must be a multiple of 32");
  const __m256 sign = _mm256_set1_ps(-0.0f);
  u8* out = reinterpret_cast<u8*>(y);
  for (int64_t b = 0; b < n / kQK; ++b, x += kQK, out += sizeof(BlockQ8_0)) {
    __m256 v0 = _mm256_loadu_ps(x), v1 = _mm256_loadu_ps(x + 8), v2 = _mm256_loadu_ps(x + 16),
           v3 = _mm256_loadu_ps(x + 24);
    // max is order-independent for non-NaN input, so this equals the scalar fmax chain.
    __m256 m = _mm256_max_ps(_mm256_max_ps(_mm256_andnot_ps(sign, v0), _mm256_andnot_ps(sign, v1)),
                             _mm256_max_ps(_mm256_andnot_ps(sign, v2), _mm256_andnot_ps(sign, v3)));
    __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
    m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
    m4 = _mm_max_ss(m4, _mm_movehdup_ps(m4));
    const float amax = _mm_cvtss_f32(m4);
    const float d = amax / 127.f;
    const float id = d ? 1.0f / d : 0.0f;  // scalar formula (llama.cpp's AVX2 path uses 127/amax)
    const uint16_t dh = fp32_to_fp16(d);
    std::memcpy(out, &dh, 2);
    const __m256 vid = _mm256_set1_ps(id);
    const __m256i i0 = _mm256_cvtps_epi32(round_half_away_ps(_mm256_mul_ps(v0, vid)));
    const __m256i i1 = _mm256_cvtps_epi32(round_half_away_ps(_mm256_mul_ps(v1, vid)));
    const __m256i i2 = _mm256_cvtps_epi32(round_half_away_ps(_mm256_mul_ps(v2, vid)));
    const __m256i i3 = _mm256_cvtps_epi32(round_half_away_ps(_mm256_mul_ps(v3, vid)));
    // Values are integers in [-127,127]: the saturating packs are exact; the permute undoes their lane interleave.
    const __m256i q = _mm256_packs_epi16(_mm256_packs_epi32(i0, i1), _mm256_packs_epi32(i2, i3));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + 2),
                        _mm256_permutevar8x32_epi32(q, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7)));
  }
}

float dot_q4_0_q8_0(const BlockQ4_0* w, const BlockQ8_0* a, int64_t n_blocks) {
  return dot1<Q4>(reinterpret_cast<const u8*>(w), reinterpret_cast<const u8*>(a), n_blocks);
}

float dot_q8_0_q8_0(const BlockQ8_0* w, const BlockQ8_0* a, int64_t n_blocks) {
  return dot1<Q8>(reinterpret_cast<const u8*>(w), reinterpret_cast<const u8*>(a), n_blocks);
}

void matmul_rows_q(const Tensor& w, const QuantActs& xq, float* y, int64_t n, int64_t row_begin, int64_t row_end) {
  if (w.type != DType::Q4_0 && w.type != DType::Q8_0) {  // no AVX2 kernel yet: the scalar oracle
    cpu::matmul_rows(w, nullptr, y, n, row_begin, row_end, &xq);
    return;
  }
  const auto* wd = static_cast<const u8*>(w.data);
  const int64_t nb = w.ne[0] / kQK, out = w.rows();
  if (w.type == DType::Q4_0)
    matmul_t<Q4>(wd, nb, out, xq.q8_0, y, n, row_begin, row_end);
  else
    matmul_t<Q8>(wd, nb, out, xq.q8_0, y, n, row_begin, row_end);
}

}  // namespace nibble::simd
#endif
