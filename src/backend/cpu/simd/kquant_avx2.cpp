// AVX2 Q4_K/Q5_K/Q6_K x Q8_K dots, bit-identical to the scalar dots in quant.cpp (and so to ggml's generic
// ggml_vec_dot_q{4,5,6}_K_q8_K_generic, ggml-cpu/quants.c:696, :771, :851).
//
// Lane rule. The generic code sums, per super-block, aux32[l] = scale-weighted products of every element
// e with e % 8 == l into eight int32 lanes, converts each lane to float and adds d * lane into its own
// float accumulator sums[l]. Float addition is not associative, so the integer lanes must hold exactly
// those sums. Plain maddubs/madd on the natural byte order would add elements 4i..4i+3 into lane i, which
// is a different grouping. So both the weight bytes and the activation bytes of every 32-element group go
// through one fixed permutation (lane_perm) that puts elements {l, l+8, l+16, l+24} into bytes 4l..4l+3.
// maddubs then adds (l, l+8) and (l+16, l+24) into int16 pairs, and madd_epi16 against the sub-block
// scale adds the pair and applies the scale: int32 lane l = scale * (those four products), exactly the
// generic aux32[l] contribution. Q6_K has a scale per 16 elements: l and l+8 share scale 2g and l+16, l+24
// share 2g+1, so its scale vector alternates (s0, s1) per int16 pair.
// Saturation: maddubs pair sums are at most 2*15*127 (Q4_K), 2*31*127 (Q5_K), 2*32*127 (Q6_K), all
// < 32767. The float part uses mul then add (no FMA; this TU builds with -ffp-contract=off) and the final
// eight lanes are added to sumf one at a time, in order, like the scalar.
#include "backend/cpu/simd/simd.h"

#ifdef NIBBLE_HAVE_AVX2
#include <cstring>
#include <type_traits>

#include "fp16.h"
#include "quant.h"
#include "tensor.h"
#include <nibble/error.h>

namespace nibble::simd {
namespace {

using u8 = unsigned char;

float load_h(const u8* p) {
  uint16_t h;
  std::memcpy(&h, p, 2);
  return _cvtsh_ss(h);
}

// Byte 4l..4l+3 of the result = bytes l, l+8, l+16, l+24 of v (see the lane rule above).
inline __m256i lane_perm(__m256i v) {
  const __m128i pair = _mm_setr_epi8(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);
  const __m256i sh = _mm256_shuffle_epi8(v, _mm256_broadcastsi128_si256(pair));  // 16-bit unit l = (e_l, e_l+8)
  const __m128i p = _mm256_castsi256_si128(sh), r = _mm256_extracti128_si256(sh, 1);
  return _mm256_set_m128i(_mm_unpackhi_epi16(p, r), _mm_unpacklo_epi16(p, r));
}

// ggml get_scale_min_k4 for all eight sub-blocks of a Q4_K/Q5_K super-block (12 packed bytes).
inline void unpack_scales(const u8* q, uint8_t sc[8], uint8_t mn[8]) {
  for (int j = 0; j < 4; ++j) {
    sc[j] = q[j] & 63;
    mn[j] = q[j + 4] & 63;
    sc[j + 4] = (q[j + 8] & 0xF) | ((q[j] >> 6) << 4);
    mn[j + 4] = (q[j + 8] >> 4) | ((q[j + 4] >> 6) << 4);
  }
}

// Weights of the eight 32-element groups of a super-block as bytes in natural element order.
struct Q4K {
  using Blk = BlockQ4_K;
  static void groups(const u8* w, __m256i (&wv)[8]) {
    const __m256i m = _mm256_set1_epi8(0x0F);
    for (int c = 0; c < 4; ++c) {
      const __m256i q = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 16 + 32 * c));
      wv[2 * c] = _mm256_and_si256(q, m);
      wv[2 * c + 1] = _mm256_and_si256(_mm256_srli_epi16(q, 4), m);
    }
  }
};
struct Q5K {
  using Blk = BlockQ5_K;
  static void groups(const u8* w, __m256i (&wv)[8]) {
    const __m256i m = _mm256_set1_epi8(0x0F), one = _mm256_set1_epi8(1);
    const __m256i qh = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 16));
    for (int c = 0; c < 4; ++c) {
      const __m256i q = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 48 + 32 * c));
      // High bit of group 2c is bit 2c of qh[l], of group 2c+1 bit 2c+1.
      const __m256i h0 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 2 * c), one), 4);
      const __m256i h1 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 2 * c + 1), one), 4);
      wv[2 * c] = _mm256_or_si256(_mm256_and_si256(q, m), h0);
      wv[2 * c + 1] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q, 4), m), h1);
    }
  }
};
struct Q6K {
  using Blk = BlockQ6_K;
  // Signed weights, (ql nibble | 2 bits of qh << 4) - 32, in [-32, 31].
  static void groups(const u8* w, __m256i (&wv)[8]) {
    const __m256i m = _mm256_set1_epi8(0x0F), three = _mm256_set1_epi8(3), bias = _mm256_set1_epi8(32);
    for (int h = 0; h < 2; ++h) {
      const __m256i qh = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 128 + 32 * h));
      const __m256i ql0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 64 * h));
      const __m256i ql1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 64 * h + 32));
      const __m256i lo[4] = {_mm256_and_si256(ql0, m), _mm256_and_si256(ql1, m),
                             _mm256_and_si256(_mm256_srli_epi16(ql0, 4), m),
                             _mm256_and_si256(_mm256_srli_epi16(ql1, 4), m)};
      for (int k = 0; k < 4; ++k) {
        const __m256i hi = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 2 * k), three), 4);
        wv[4 * h + k] = _mm256_sub_epi8(_mm256_or_si256(lo[k], hi), bias);
      }
    }
  }
};

struct RowAcc {
  __m256 sums8 = _mm256_setzero_ps();
  float sumf = 0.f;
};

inline void activation_groups(const u8* a, __m256i (&ap)[8]) {
  for (int g = 0; g < 8; ++g) ap[g] = lane_perm(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + 4 + 32 * g)));
}

inline float act_d(const u8* a) {
  float d;
  std::memcpy(&d, a, 4);
  return d;
}

// A weight super-block unpacked once: the lane-permuted weight groups, the per-group scale operands for
// madd_epi16 and the float scales. prep_* does the (comparatively expensive) bit unpacking; apply_* is the
// per-activation-row arithmetic. Splitting them lets the prefill tile (tile_k) unpack each weight block once
// for several tokens while every (row, token) result keeps exactly the single-dot operation order.
struct Prep {
  __m256i wp[8];  // lane_perm of the weight groups (Q4_K/Q5_K: unsigned 0..31, Q6_K: signed -32..31)
  __m256i sc[8];  // madd operand: the group's scale in every int16 (Q4_K/Q5_K) or alternating pairs (Q6_K)
  uint8_t mn[8];  // Q4_K/Q5_K sub-block minimums (unused for Q6_K)
  float d, dmin;  // fp16 super-block scales as float (dmin unused for Q6_K)
};

// Q4_K/Q5_K super-block at w.
template <class W>
inline void prep45(const u8* w, Prep& p) {
  uint8_t sc[8];
  unpack_scales(w + 4, sc, p.mn);
  __m256i wv[8];
  W::groups(w, wv);
  for (int g = 0; g < 8; ++g) {
    p.wp[g] = lane_perm(wv[g]);
    p.sc[g] = _mm256_set1_epi16(static_cast<short>(sc[g]));
  }
  p.d = load_h(w);
  p.dmin = load_h(w + 2);
}

inline void prep6(const u8* w, Prep& p) {
  __m256i wv[8];
  Q6K::groups(w, wv);
  const auto* sc = reinterpret_cast<const int8_t*>(w + 192);
  for (int g = 0; g < 8; ++g) {
    p.wp[g] = lane_perm(wv[g]);
    const uint32_t pair = static_cast<uint16_t>(static_cast<int16_t>(sc[2 * g])) |
                          (static_cast<uint32_t>(static_cast<uint16_t>(static_cast<int16_t>(sc[2 * g + 1]))) << 16);
    p.sc[g] = _mm256_set1_epi32(static_cast<int>(pair));
  }
  p.d = load_h(w + 208);
}

// One prepared Q4_K/Q5_K super-block against the permuted activation groups ap of activation block a.
inline void apply45(const Prep& p, const u8* a, const __m256i (&ap)[8], RowAcc& r) {
  __m256i acc = _mm256_setzero_si256();
  for (int g = 0; g < 8; ++g) {
    const __m256i prod = _mm256_maddubs_epi16(p.wp[g], ap[g]);  // unsigned weights x signed activations
    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(prod, p.sc[g]));
  }
  int sumi = 0;
  for (int k = 0; k < 8; ++k) {
    int16_t b0, b1;
    std::memcpy(&b0, a + 260 + 4 * k, 2);
    std::memcpy(&b1, a + 262 + 4 * k, 2);
    sumi += (b0 + b1) * p.mn[k];
  }
  const float ad = act_d(a);
  const float d = p.d * ad;
  r.sums8 = _mm256_add_ps(r.sums8, _mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(acc)));
  const float dmin = p.dmin * ad;
  r.sumf -= dmin * static_cast<float>(sumi);
}

inline void apply6(const Prep& p, const u8* a, const __m256i (&ap)[8], RowAcc& r) {
  __m256i acc = _mm256_setzero_si256();
  for (int g = 0; g < 8; ++g) {
    const __m256i wp = p.wp[g];
    // |w| as the unsigned operand, a negated where w < 0: exact because a != -128 (Q8_K output is in [-127, 127]).
    const __m256i prod = _mm256_maddubs_epi16(_mm256_sign_epi8(wp, wp), _mm256_sign_epi8(ap[g], wp));
    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(prod, p.sc[g]));
  }
  const float d = p.d * act_d(a);
  r.sums8 = _mm256_add_ps(r.sums8, _mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(acc)));
}

template <class W>
inline void prep(const u8* w, Prep& p) {
  if constexpr (std::is_same_v<W, Q6K>)
    prep6(w, p);
  else
    prep45<W>(w, p);
}

template <class W>
inline void apply(const Prep& p, const u8* a, const __m256i (&ap)[8], RowAcc& r) {
  if constexpr (std::is_same_v<W, Q6K>)
    apply6(p, a, ap, r);
  else
    apply45(p, a, ap, r);
}

inline float finish(const RowAcc& r) {
  alignas(32) float s[8];
  _mm256_store_ps(s, r.sums8);
  float sumf = r.sumf;
  for (int l = 0; l < 8; ++l) sumf += s[l];
  return sumf;
}

// R consecutive weight rows (stride rb bytes) against T consecutive activation rows (stride nb blocks):
// out[t * ostride + i] for token t < T, row i < R. Each weight super-block is unpacked once and applied to all
// T tokens; the arithmetic per (row, token) is the same sequence for any R and T, so a tile equals R*T single
// dots bit for bit. T = 1 is the row-at-a-time kernel used for decode.
template <class W, int R, int T>
void tile_k(const u8* w, size_t rb, const u8* a, int64_t nb, float* out, int64_t ostride) {
  RowAcc acc[T][R];
  for (int64_t b = 0; b < nb; ++b, w += sizeof(typename W::Blk)) {
    Prep p[R];
    for (int i = 0; i < R; ++i) prep<W>(w + i * rb, p[i]);
    for (int t = 0; t < T; ++t) {
      const u8* at = a + (t * nb + b) * sizeof(BlockQ8_K);
      __m256i ap[8];
      activation_groups(at, ap);
      for (int i = 0; i < R; ++i) apply<W>(p[i], at, ap, acc[t][i]);
    }
  }
  for (int t = 0; t < T; ++t)
    for (int i = 0; i < R; ++i) out[t * ostride + i] = finish(acc[t][i]);
}

template <class W>
void matmul_k(const u8* w, int64_t nb, int64_t out, const BlockQ8_K* xq, float* y, int64_t n, int64_t rb, int64_t re) {
  const size_t row = static_cast<size_t>(nb) * sizeof(typename W::Blk);
  for (int64_t t = 0; t < n; ++t) {
    const u8* a = reinterpret_cast<const u8*>(xq + t * nb);
    float* yt = y + t * out;
    int64_t r = rb;
    for (; r + 4 <= re; r += 4) tile_k<W, 4, 1>(w + static_cast<size_t>(r) * row, row, a, nb, yt + r, out);
    for (; r < re; ++r) tile_k<W, 1, 1>(w + static_cast<size_t>(r) * row, 0, a, nb, yt + r, out);
  }
}

// Prefill: [rb, re) x [0, n) in tiles of 4 weight rows x 4 tokens; leftover tokens use 1-token tiles and
// leftover rows 1-row tiles.
template <class W>
void gemm_kt(const u8* w, int64_t nb, int64_t out, const BlockQ8_K* xq, float* y, int64_t n, int64_t rb, int64_t re) {
  const size_t row = static_cast<size_t>(nb) * sizeof(typename W::Blk);
  const int64_t re_full = rb + (re - rb) / 4 * 4, n_full = n / 4 * 4;
  auto rows = [&]<int T>(int64_t t) {
    const u8* a = reinterpret_cast<const u8*>(xq + t * nb);
    float* yt = y + t * out;
    int64_t r = rb;
    for (; r < re_full; r += 4) tile_k<W, 4, T>(w + static_cast<size_t>(r) * row, row, a, nb, yt + r, out);
    for (; r < re; ++r) tile_k<W, 1, T>(w + static_cast<size_t>(r) * row, 0, a, nb, yt + r, out);
  };
  int64_t t = 0;
  for (; t < n_full; t += 4) rows.template operator()<4>(t);
  for (; t < n; ++t) rows.template operator()<1>(t);
}

template <class W>
float dot1(const void* w, const BlockQ8_K* a, int64_t nb) {
  float out;
  tile_k<W, 1, 1>(static_cast<const u8*>(w), 0, reinterpret_cast<const u8*>(a), nb, &out, 0);
  return out;
}

}  // namespace

float dot_q4_K_q8_K(const BlockQ4_K* w, const BlockQ8_K* a, int64_t n_blocks) { return dot1<Q4K>(w, a, n_blocks); }
float dot_q5_K_q8_K(const BlockQ5_K* w, const BlockQ8_K* a, int64_t n_blocks) { return dot1<Q5K>(w, a, n_blocks); }
float dot_q6_K_q8_K(const BlockQ6_K* w, const BlockQ8_K* a, int64_t n_blocks) { return dot1<Q6K>(w, a, n_blocks); }

void matmul_rows_k(const Tensor& w, const BlockQ8_K* xq, float* y, int64_t n, int64_t row_begin, int64_t row_end) {
  const auto* wd = static_cast<const u8*>(w.data);
  const int64_t out = w.rows(), nb = w.ne[0] / kQK_K;
  switch (w.type) {
    case DType::Q4_K: return matmul_k<Q4K>(wd, nb, out, xq, y, n, row_begin, row_end);
    case DType::Q5_K: return matmul_k<Q5K>(wd, nb, out, xq, y, n, row_begin, row_end);
    case DType::Q6_K: return matmul_k<Q6K>(wd, nb, out, xq, y, n, row_begin, row_end);
    default: throw Error("simd::matmul_rows_k: not a k-quant weight type");
  }
}

void gemm_k(const Tensor& w, const BlockQ8_K* xq, float* y, int64_t n, int64_t row_begin, int64_t row_end) {
  const auto* wd = static_cast<const u8*>(w.data);
  const int64_t out = w.rows(), nb = w.ne[0] / kQK_K;
  switch (w.type) {
    case DType::Q4_K: return gemm_kt<Q4K>(wd, nb, out, xq, y, n, row_begin, row_end);
    case DType::Q5_K: return gemm_kt<Q5K>(wd, nb, out, xq, y, n, row_begin, row_end);
    case DType::Q6_K: return gemm_kt<Q6K>(wd, nb, out, xq, y, n, row_begin, row_end);
    default: throw Error("simd::gemm_k: not a k-quant weight type");
  }
}

}  // namespace nibble::simd
#endif
