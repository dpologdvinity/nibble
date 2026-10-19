// Tiled prefill GEMM: an R-row x T-token output tile per microkernel call, so each weight load (and F16
// convert / Q4_0 unpack) is shared by T tokens and each activation load by R rows.
// Q4_0/Q8_0 activations are first interleaved 8 tokens per vector (pack_q8_0_x8).
// Every output element goes through the same operation sequence as in the row-at-a-time kernels
// (matmul_f.cpp, quant_avx2.cpp), which depends only on its row and K, never on n or the tile shape:
// - F32/F16: 8 FMA lanes over k in steps of 8, hsum_ps, then the scalar tail.
// - Q4_0/Q8_0: exact integer block sums, the scalar's per-block float products, sequential float adds.
// So results are bit-identical to the row path. F32/F16 partial tiles are delegated to it; Q tiles pad
// tokens to a multiple of 8 and use 1-row tiles for leftover rows.
// Compiled with -ffp-contract=off, like the row kernels.
#include "backend/cpu/simd/simd.h"

#ifdef NIBBLE_HAVE_AVX2
#include <cstring>

#include "fp16.h"
#include "quant.h"
#include "tensor.h"

namespace nibble::simd {
namespace {

using u8 = unsigned char;

// ---- F32 / F16 ----

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

constexpr int kRf = 4, kTf = 3;  // 12 accumulators + 3 activation vectors + 1 weight vector = 16 ymm

// y[t*out + r] for R rows from w and T tokens from x.
template <class L, int R, int T>
void tile_f(const typename L::T* w, int64_t in, const float* x, float* y, int64_t out) {
  const int64_t vec = in & ~int64_t{7};
  __m256 acc[R][T];
  for (int r = 0; r < R; ++r)
    for (int t = 0; t < T; ++t) acc[r][t] = _mm256_setzero_ps();
  for (int64_t k = 0; k < vec; k += 8) {
    __m256 xv[T];
    for (int t = 0; t < T; ++t) xv[t] = _mm256_loadu_ps(x + t * in + k);
    for (int r = 0; r < R; ++r) {
      const __m256 wv = L::load(w + r * in + k);
      for (int t = 0; t < T; ++t) acc[r][t] = _mm256_fmadd_ps(xv[t], wv, acc[r][t]);
    }
  }
  for (int r = 0; r < R; ++r)
    for (int t = 0; t < T; ++t) {
      const float* xt = x + t * in;
      const typename L::T* wr = w + r * in;
      float s = hsum_ps(acc[r][t]);
      for (int64_t k = vec; k < in; ++k) s += xt[k] * L::one(wr[k]);
      y[t * out + r] = s;
    }
}

template <class L>
void gemm_ft(const Tensor& wt, const float* x, float* y, int64_t n, int64_t rb, int64_t re) {
  const auto* w = static_cast<const typename L::T*>(wt.data);
  const int64_t in = wt.ne[0], out = wt.rows();
  const int64_t re_full = rb + (re - rb) / kRf * kRf, n_full = n / kTf * kTf;
  for (int64_t t = 0; t < n_full; t += kTf)
    for (int64_t r = rb; r < re_full; r += kRf)
      tile_f<L, kRf, kTf>(w + r * in, in, x + t * in, y + t * out + r, out);
  if (n_full < n) matmul_rows_f(wt, x + n_full * in, y + n_full * out, n - n_full, rb, re_full);
  if (re_full < re) matmul_rows_f(wt, x, y, n, re_full, re);
}

// ---- Q4_0 / Q8_0 weights x interleaved Q8_0 activations ----
// Lanes are 8 tokens (BlockQ8_0x8): each 4-byte weight group is broadcast and multiplied against the same
// group of 8 tokens, so a block's lanes end up holding each token's exact integer block sum without any
// horizontal reduction. dot() writes those sums for TV vectors of 8 tokens.

uint16_t load_d(const u8* blk) {
  uint16_t d;
  std::memcpy(&d, blk, 2);
  return d;
}
__m256i bcast_group(const u8* blk, int g) {  // weight quants 4g..4g+3 of a Q8_0 block in every lane
  int32_t v;
  std::memcpy(&v, blk + 2 + 4 * g, 4);
  return _mm256_set1_epi32(v);
}
__m256i load_group(const BlockQ8_0x8* a, int g) { return _mm256_load_si256(reinterpret_cast<const __m256i*>(a->qs[g])); }
__m256i load_sum(const BlockQ8_0x8* a) { return _mm256_load_si256(reinterpret_cast<const __m256i*>(a->sum)); }

// Q4_0 nibbles (0..15 = w + 8): qs[4g'..4g'+3] holds elements 4g'..4g'+3 (low nibbles) and 16+4g'.. (high),
// i.e. groups g' and g' + 4.
// dot() handles R rows (stride rowb bytes) at once so the R*TV accumulation chains interleave.
struct Q4 {
  static constexpr size_t kBytes = sizeof(BlockQ4_0);
  template <int R, int TV>
  static void dot(const u8* w, size_t rowb, const BlockQ8_0x8* const* a, __m256i (*s)[TV]) {
    const __m256i m = _mm256_set1_epi8(0x0F);
#if defined(NIBBLE_HAVE_AVXVNNI)
    // sum((w + 8) * a) - 8 * sum(a); vpdpbusd does not saturate.
    for (int v = 0; v < TV; ++v) {
      const __m256i bias = _mm256_sub_epi32(_mm256_setzero_si256(), _mm256_slli_epi32(load_sum(a[v]), 3));
      for (int r = 0; r < R; ++r) s[r][v] = bias;
    }
    for (int g = 0; g < 4; ++g)
      for (int r = 0; r < R; ++r) {
        const __m256i x = bcast_group(w + r * rowb, g);
        const __m256i lo = _mm256_and_si256(x, m), hi = _mm256_and_si256(_mm256_srli_epi16(x, 4), m);
        for (int v = 0; v < TV; ++v) {
          s[r][v] = _mm256_dpbusd_avx_epi32(s[r][v], lo, load_group(a[v], g));
          s[r][v] = _mm256_dpbusd_avx_epi32(s[r][v], hi, load_group(a[v], g + 4));
        }
      }
#else
    // maddubs pairs are at most 2*15*128 = 3840, so 8 of them per int16 lane cannot overflow.
    for (int r = 0; r < R; ++r)
      for (int v = 0; v < TV; ++v) s[r][v] = _mm256_setzero_si256();
    for (int g = 0; g < 4; ++g)
      for (int r = 0; r < R; ++r) {
        const __m256i x = bcast_group(w + r * rowb, g);
        const __m256i lo = _mm256_and_si256(x, m), hi = _mm256_and_si256(_mm256_srli_epi16(x, 4), m);
        for (int v = 0; v < TV; ++v) {
          s[r][v] = _mm256_add_epi16(s[r][v], _mm256_maddubs_epi16(lo, load_group(a[v], g)));
          s[r][v] = _mm256_add_epi16(s[r][v], _mm256_maddubs_epi16(hi, load_group(a[v], g + 4)));
        }
      }
    for (int v = 0; v < TV; ++v) {
      const __m256i bias = _mm256_slli_epi32(load_sum(a[v]), 3);
      for (int r = 0; r < R; ++r) s[r][v] = _mm256_sub_epi32(_mm256_madd_epi16(s[r][v], _mm256_set1_epi16(1)), bias);
    }
#endif
  }
  // quant.cpp: float(s) * dw * da.
  static __m256 scale(__m256 s, __m256 dw, __m256 da) { return _mm256_mul_ps(_mm256_mul_ps(s, dw), da); }
};
struct Q8 {
  static constexpr size_t kBytes = sizeof(BlockQ8_0);
  template <int R, int TV>
  static void dot(const u8* w, size_t rowb, const BlockQ8_0x8* const* a, __m256i (*s)[TV]) {
#if defined(NIBBLE_HAVE_AVXVNNI)
    // sum((w + 128) * a) - 128 * sum(a): exact for any quants, including -128.
    for (int v = 0; v < TV; ++v) {
      const __m256i bias = _mm256_sub_epi32(_mm256_setzero_si256(), _mm256_slli_epi32(load_sum(a[v]), 7));
      for (int r = 0; r < R; ++r) s[r][v] = bias;
    }
    for (int g = 0; g < 8; ++g)
      for (int r = 0; r < R; ++r) {
        const __m256i x = _mm256_xor_si256(bcast_group(w + r * rowb, g), _mm256_set1_epi8(char(0x80)));
        for (int v = 0; v < TV; ++v) s[r][v] = _mm256_dpbusd_avx_epi32(s[r][v], x, load_group(a[v], g));
      }
#else
    // Weight is the sign source (mul_sum_i8_maddubs); activations never hold -128.
    for (int r = 0; r < R; ++r)
      for (int v = 0; v < TV; ++v) s[r][v] = _mm256_setzero_si256();
    for (int g = 0; g < 8; ++g)
      for (int r = 0; r < R; ++r) {
        const __m256i x = bcast_group(w + r * rowb, g), ax = _mm256_sign_epi8(x, x);
        for (int v = 0; v < TV; ++v)
          s[r][v] = _mm256_add_epi32(
              s[r][v], _mm256_madd_epi16(_mm256_maddubs_epi16(ax, _mm256_sign_epi8(load_group(a[v], g), x)),
                                         _mm256_set1_epi16(1)));
      }
#endif
  }
  // quant.cpp: float(s) * (dw * da).
  static __m256 scale(__m256 s, __m256 dw, __m256 da) { return _mm256_mul_ps(s, _mm256_mul_ps(dw, da)); }
};

// R weight rows (stride rowb bytes) x TV tiles of 8 tokens (tile stride nb blocks); y[t*out + r] for the
// first nt tokens of the tiles (the rest are padding).
template <class W, int R, int TV>
void tile_q(const u8* w, size_t rowb, const BlockQ8_0x8* a, int64_t nb, float* y, int64_t out, int64_t nt) {
  __m256 acc[R][TV];
  for (int r = 0; r < R; ++r)
    for (int v = 0; v < TV; ++v) acc[r][v] = _mm256_setzero_ps();
  for (int64_t b = 0; b < nb; ++b) {
    const BlockQ8_0x8* ab[TV];
    for (int v = 0; v < TV; ++v) ab[v] = a + v * nb + b;
    const u8* wb = w + static_cast<size_t>(b) * W::kBytes;
    __m256i s[R][TV];
    W::template dot<R, TV>(wb, rowb, ab, s);
    for (int r = 0; r < R; ++r) {
      const __m256 dw = _mm256_set1_ps(_cvtsh_ss(load_d(wb + r * rowb)));
      for (int v = 0; v < TV; ++v)
        acc[r][v] = _mm256_add_ps(acc[r][v], W::scale(_mm256_cvtepi32_ps(s[r][v]), dw, _mm256_load_ps(ab[v]->d)));
    }
  }
  for (int r = 0; r < R; ++r)
    for (int v = 0; v < TV; ++v) {
      alignas(32) float o[8];
      _mm256_store_ps(o, acc[r][v]);
      for (int64_t j = 0; j < 8 && 8 * v + j < nt; ++j) y[(8 * v + j) * out + r] = o[j];
    }
}

constexpr int kRq = 4, kTVq = 2;  // 4 rows x 16 tokens

template <class W>
void gemm_qt(const Tensor& wt, const BlockQ8_0x8* xp, float* y, int64_t n, int64_t rb, int64_t re) {
  const auto* w = static_cast<const u8*>(wt.data);
  const int64_t nb = wt.ne[0] / kQK, out = wt.rows(), tiles = (n + 7) / 8;
  const size_t row = static_cast<size_t>(nb) * W::kBytes;
  const int64_t re_full = rb + (re - rb) / kRq * kRq;
  auto rows = [&]<int TV>(int64_t tile) {
    const BlockQ8_0x8* a = xp + tile * nb;
    float* yt = y + tile * 8 * out;
    const int64_t nt = n - tile * 8;
    int64_t r = rb;
    for (; r < re_full; r += kRq) tile_q<W, kRq, TV>(w + static_cast<size_t>(r) * row, row, a, nb, yt + r, out, nt);
    for (; r < re; ++r) tile_q<W, 1, TV>(w + static_cast<size_t>(r) * row, row, a, nb, yt + r, out, nt);
  };
  int64_t tile = 0;
  for (; tile + kTVq <= tiles; tile += kTVq) rows.template operator()<kTVq>(tile);
  for (; tile < tiles; ++tile) rows.template operator()<1>(tile);
}

}  // namespace

void gemm_f(const Tensor& w, const float* x, float* y, int64_t n, int64_t row_begin, int64_t row_end) {
  if (w.type == DType::F32)
    gemm_ft<LoadF32>(w, x, y, n, row_begin, row_end);
  else
    gemm_ft<LoadF16>(w, x, y, n, row_begin, row_end);
}

void pack_q8_0_x8(const BlockQ8_0* xq, int64_t n, int64_t nb, BlockQ8_0x8* xp) {
  const int64_t tiles = (n + 7) / 8;
  std::memset(static_cast<void*>(xp), 0, static_cast<size_t>(tiles * nb) * sizeof(BlockQ8_0x8));
  for (int64_t t = 0; t < n; ++t)
    for (int64_t b = 0; b < nb; ++b) {
      const BlockQ8_0& src = xq[t * nb + b];
      BlockQ8_0x8& dst = xp[(t / 8) * nb + b];
      const int64_t j = t % 8;
      for (int g = 0; g < 8; ++g) std::memcpy(dst.qs[g][j], src.qs + 4 * g, 4);
      dst.d[j] = fp16_to_fp32(src.d);
      int32_t sum = 0;
      for (int8_t q : src.qs) sum += q;
      dst.sum[j] = sum;
    }
}

void gemm_q(const Tensor& w, const BlockQ8_0x8* xp, float* y, int64_t n, int64_t row_begin, int64_t row_end) {
  if (w.type == DType::Q4_0)
    gemm_qt<Q4>(w, xp, y, n, row_begin, row_end);
  else
    gemm_qt<Q8>(w, xp, y, n, row_begin, row_end);
}

}  // namespace nibble::simd
#endif
