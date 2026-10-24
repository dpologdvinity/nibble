// AVX2 Q8_0 activation quantizer and the legacy-quant (Q4_0, Q5_0, Q5_1, Q8_0) row dots. Scalar oracle:
// quant.cpp. The k-quant dots live in kquant_avx2.cpp.
// Integer technique after llama.cpp's ggml-cpu/arch/x86/quants.c (MIT): nibble unpack as in
// bytes_from_nibbles_32, sign trick as in mul_sum_i8_pairs_float. Unlike llama.cpp, results are
// bit-identical to the scalar code: the block integer sums are exact, the per-block float products use the
// scalar's operation order, and they are accumulated sequentially per row (compiled with -ffp-contract=off).
// The 4-row kernel reduces four rows' block sums with one hadd tree, which keeps that order at no cost.
#include "backend/cpu/simd/simd.h"

#ifdef NIBBLE_HAVE_AVX2
#include <cstring>
#include <limits>

#include "fp16.h"
#include "quant.h"
#include "tensor.h"
#include <nibble/error.h>

namespace nibble::simd {
namespace {

using u8 = unsigned char;

// Raw fp16 scale at the start of a block (unaligned-safe).
uint16_t load_d(const u8* blk) {
  uint16_t d;
  std::memcpy(&d, blk, 2);
  return d;
}

// The fp16 value at p as a float (F16C).
float load_h(const u8* p) { return _cvtsh_ss(load_d(p)); }

// Four fp16 values at p, p + rb, p + 2 rb, p + 3 rb as floats.
__m128 load_h4(const u8* p, size_t rb) {
  return _mm_cvtph_ps(_mm_setr_epi16(static_cast<short>(load_d(p)), static_cast<short>(load_d(p + rb)),
                                     static_cast<short>(load_d(p + 2 * rb)), static_cast<short>(load_d(p + 3 * rb)), 0,
                                     0, 0, 0));
}

// Weight block layout and its integer dot with one activation block (8 int32 partial sums). kAct is the
// activation block size and kQs its qs offset (Q8_0: 34 / 2, Q8_1: 36 / 4). term1/term4 turn the exact
// block sum(s) into the float block term(s) in the scalar's operation order (quant.cpp); the caller adds
// each term to a running per-row sum, as the scalar does.
struct Q4 {
  static constexpr size_t kBytes = sizeof(BlockQ4_0), kAct = sizeof(BlockQ8_0), kQs = 2;
  static __m256i isum(const u8* w, __m256i a) {
    const __m128i t = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + 2));
    const __m256i nib = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(t, 4), t), _mm256_set1_epi8(0x0F));
    // The activation is the sign source: w - 8 is in [-8,7], so mul_sum_i8 is exact even for a == -128.
    return mul_sum_i8(a, _mm256_sub_epi8(nib, _mm256_set1_epi8(8)));
  }
  // quant.cpp: float(s) * dw * da.
  static float term1(int32_t s, const u8* w, const u8* a) { return static_cast<float>(s) * load_h(w) * load_h(a); }
  static __m128 term4(__m128 s, const u8* w, size_t rb, const u8* a) {
    return _mm_mul_ps(_mm_mul_ps(s, load_h4(w, rb)), _mm_set1_ps(load_h(a)));
  }
};
struct Q8 {
  static constexpr size_t kBytes = sizeof(BlockQ8_0), kAct = sizeof(BlockQ8_0), kQs = 2;
  static __m256i isum(const u8* w, __m256i a) {
    return mul_sum_i8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 2)), a);
  }
  // quant.cpp: float(s) * (dw * da).
  static float term1(int32_t s, const u8* w, const u8* a) { return static_cast<float>(s) * (load_h(w) * load_h(a)); }
  static __m128 term4(__m128 s, const u8* w, size_t rb, const u8* a) {
    return _mm_mul_ps(s, _mm_mul_ps(load_h4(w, rb), _mm_set1_ps(load_h(a))));
  }
};
// Q5_0: the 32 weights are 4 low/high nibbles plus the high bit of each element in qh (bit e for element e),
// minus 16: in [-16, 15], so mul_sum_i8 is exact for any activation (a == -128 included), as for Q4_0.
struct Q5_0 {
  static constexpr size_t kBytes = sizeof(BlockQ5_0), kAct = sizeof(BlockQ8_0), kQs = 2;
  // 32 bytes, byte e = 0x10 if bit e of the 32-bit qh at w + 2 is set.
  static __m256i high_bits(const u8* qh) {
    uint32_t v;
    std::memcpy(&v, qh, 4);
    const __m256i bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(static_cast<int>(v)),
                                              _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2,
                                                               2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3));
    const __m256i bit = _mm256_set1_epi64x(static_cast<long long>(0x8040201008040201ull));
    return _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(bytes, bit), bit), _mm256_set1_epi8(0x10));
  }
  static __m256i isum(const u8* w, __m256i a) {
    const __m128i t = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + 6));
    const __m256i nib = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(t, 4), t), _mm256_set1_epi8(0x0F));
    return mul_sum_i8(a, _mm256_sub_epi8(_mm256_or_si256(nib, high_bits(w + 2)), _mm256_set1_epi8(16)));
  }
  // quant.cpp: (dw * da) * float(s), the same product as Q8's.
  static float term1(int32_t s, const u8* w, const u8* a) { return Q8::term1(s, w, a); }
  static __m128 term4(__m128 s, const u8* w, size_t rb, const u8* a) { return Q8::term4(s, w, rb, a); }
};
// Q5_1 against Q8_1 activations: the weights are unsigned [0, 31], so maddubs cannot saturate
// (2 * 31 * 127 < 32767); the activation is the signed operand, any byte value.
struct Q5_1 {
  static constexpr size_t kBytes = sizeof(BlockQ5_1), kAct = sizeof(BlockQ8_1), kQs = 4;
  static __m256i isum(const u8* w, __m256i a) {
    const __m128i t = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + 8));
    const __m256i nib = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(t, 4), t), _mm256_set1_epi8(0x0F));
    const __m256i q = _mm256_or_si256(nib, Q5_0::high_bits(w + 4));
    return _mm256_madd_epi16(_mm256_maddubs_epi16(q, a), _mm256_set1_epi16(1));
  }
  // quant.cpp: (dw * da) * float(s) + mw * sa.
  static float term1(int32_t s, const u8* w, const u8* a) {
    return (load_h(w) * load_h(a)) * static_cast<float>(s) + load_h(w + 2) * load_h(a + 2);
  }
  static __m128 term4(__m128 s, const u8* w, size_t rb, const u8* a) {
    const __m128 prod = _mm_mul_ps(_mm_mul_ps(load_h4(w, rb), _mm_set1_ps(load_h(a))), s);
    return _mm_add_ps(prod, _mm_mul_ps(load_h4(w + 2, rb), _mm_set1_ps(load_h(a + 2))));
  }
};

// The 32 activation quants of a Q8_0/Q8_1 block, as one vector.
template <class W>
__m256i load_act(const u8* a) {
  return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + W::kQs));
}

// One weight row against one activation row: per block an exact integer sum, then the scalar's float term,
// accumulated in block order.
template <class W>
float dot1(const u8* w, const u8* a, int64_t nb) {
  float sum = 0.f;
  for (int64_t b = 0; b < nb; ++b, w += W::kBytes, a += W::kAct)
    sum += W::term1(hsum_epi32(W::isum(w, load_act<W>(a))), w, a);
  return sum;
}

// Four consecutive weight rows (stride rb bytes) against one activation row; out[0..3].
template <class W>
void dot4(const u8* w, size_t rb, const u8* a, int64_t nb, float* out) {
  float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
  for (int64_t b = 0; b < nb; ++b, w += W::kBytes, a += W::kAct) {
    const __m256i av = load_act<W>(a);
    const __m256i p0 = W::isum(w, av), p1 = W::isum(w + rb, av), p2 = W::isum(w + 2 * rb, av),
                  p3 = W::isum(w + 3 * rb, av);
    // Per 128-bit lane: [p0, p1, p2, p3] partial sums; adding the two lanes gives the 4 exact block sums.
    const __m256i h = _mm256_hadd_epi32(_mm256_hadd_epi32(p0, p1), _mm256_hadd_epi32(p2, p3));
    const __m128i s = _mm_add_epi32(_mm256_castsi256_si128(h), _mm256_extracti128_si256(h, 1));
    alignas(16) float p[4];
    _mm_store_ps(p, W::term4(_mm_cvtepi32_ps(s), w, rb, a));
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

// y[t * out + r] for rows [rb, re) and n activation rows, four rows at a time with a one-row remainder.
template <class W, class A>
void matmul_t(const u8* w, int64_t nb, int64_t out, const A* xq, float* y, int64_t n, int64_t rb, int64_t re) {
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

// Vectorized: per 32-element block, find max|x| with vector max, then scale, round half away from zero
// (round_half_away_ps) and pack to int8 with saturating packs. A block containing a NaN gets a NaN scale and
// all-zero quants, as in the scalar quantizer.
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
    // A NaN anywhere in the block makes the scale NaN (like the scalar quantizer) and the quants 0, so every dot
    // against the block is NaN. The max above cannot be relied on for this: its NaN handling depends on operand order.
    const __m256 unord =
        _mm256_or_ps(_mm256_or_ps(_mm256_cmp_ps(v0, v0, _CMP_UNORD_Q), _mm256_cmp_ps(v1, v1, _CMP_UNORD_Q)),
                     _mm256_or_ps(_mm256_cmp_ps(v2, v2, _CMP_UNORD_Q), _mm256_cmp_ps(v3, v3, _CMP_UNORD_Q)));
    if (_mm256_movemask_ps(unord) != 0) {
      const uint16_t nan_h = fp32_to_fp16(std::numeric_limits<float>::quiet_NaN());
      std::memcpy(out, &nan_h, 2);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + 2), _mm256_setzero_si256());
      continue;
    }
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

float dot_q5_0_q8_0(const BlockQ5_0* w, const BlockQ8_0* a, int64_t n_blocks) {
  return dot1<Q5_0>(reinterpret_cast<const u8*>(w), reinterpret_cast<const u8*>(a), n_blocks);
}

float dot_q5_1_q8_1(const BlockQ5_1* w, const BlockQ8_1* a, int64_t n_blocks) {
  return dot1<Q5_1>(reinterpret_cast<const u8*>(w), reinterpret_cast<const u8*>(a), n_blocks);
}

// Dispatch on the weight type to the right row kernel; the k-quants live in kquant_avx2.cpp.
void matmul_rows_q(const Tensor& w, const QuantActs& xq, float* y, int64_t n, int64_t row_begin, int64_t row_end) {
  const auto* wd = static_cast<const u8*>(w.data);
  const int64_t out = w.rows(), nb = w.ne[0] / dtype_info(w.type).block_size;
  switch (w.type) {
    case DType::Q4_0: return matmul_t<Q4>(wd, nb, out, xq.q8_0, y, n, row_begin, row_end);
    case DType::Q8_0: return matmul_t<Q8>(wd, nb, out, xq.q8_0, y, n, row_begin, row_end);
    case DType::Q5_0: return matmul_t<Q5_0>(wd, nb, out, xq.q8_0, y, n, row_begin, row_end);
    case DType::Q5_1: return matmul_t<Q5_1>(wd, nb, out, xq.q8_1, y, n, row_begin, row_end);
    case DType::Q4_K:
    case DType::Q5_K:
    case DType::Q6_K: return matmul_rows_k(w, xq.q8_K, y, n, row_begin, row_end);
    default: throw Error("simd::matmul_rows_q: not a quantized weight type");
  }
}

}  // namespace nibble::simd
#endif
