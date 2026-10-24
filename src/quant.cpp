// Scalar reference implementation of the quantized formats declared in quant.h: dequantization, the Q8_0 /
// Q8_1 / Q8_K activation quantizers and every block dot product.
//
// This file is the oracle for the whole project. Each operation mirrors the corresponding function in
// llama.cpp's generic (non-SIMD) ggml code, including the *order* of floating-point operations, because
// float addition is not associative and the results are compared bit for bit: against golden vectors
// produced by ggml itself (tests/test_quant.cpp) and against the AVX2 kernels (tests/test_simd.cpp). When
// changing anything here, keep the evaluation order and re-run both.
//
// Robustness: blocks come from untrusted files, so every function must tolerate arbitrary bytes (all-ones
// scales, NaN/Inf fp16 scales). Blocks are read with memcpy, never through a typed pointer, because they
// may sit at any byte offset.
#include "quant.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <string>

#include "fp16.h"
#include <nibble/error.h>

namespace nibble {

namespace {
// Throws unless n is a non-negative multiple of `block` (the element count of one block of the type).
void check_n(int64_t n, int64_t block = kQK) {
  if (n < 0 || n % block != 0) throw Error("quant: length must be a multiple of " + std::to_string(block));
}

// Shared Q8_0/Q8_1 block core (ggml quantize_row_q8_{0,1}_ref): writes 32 qs, returns the pre-fp16 d.
// roundf rounds half away from zero (Q8_K's nearest_int is half to even, not this). NaN maps to 0 and results
// saturate to [-127, 127], so the int8 cast is never UB; only non-finite rows or blocks with id = Inf
// (amax < ~3.7e-37) reach either case, and finite normal blocks stay bit-exact.
// A NaN anywhere in the block makes d NaN (ggml's result for NaN input depends on element order and is not
// meaningful), so every dot against the block is NaN instead of a plausible-looking wrong number.
float quantize_block_q8(const float* x, int8_t* qs) {
  float amax = 0.f;
  bool has_nan = false;
  for (int j = 0; j < kQK; ++j) {
    amax = std::fmax(amax, std::fabs(x[j]));  // fmax skips NaN; has_nan accounts for it below
    has_nan |= x[j] != x[j];
  }
  const float d = has_nan ? std::numeric_limits<float>::quiet_NaN() : amax / 127.f;
  const float id = d ? 1.0f / d : 0.0f;
  for (int j = 0; j < kQK; ++j) {
    const float r = std::roundf(x[j] * id);
    qs[j] = r != r ? 0 : static_cast<int8_t>(std::fmin(127.f, std::fmax(-127.f, r)));
  }
  return d;
}

// ggml get_scale_min_k4: the j-th 6-bit scale and min of a Q4_K/Q5_K super-block.
void scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

// Unpacks a K super-block's 256 weights to integers, in element order (group k of 32 = elements 32k..32k+31):
// Q4_K/Q5_K to 0..15 / 0..31, Q6_K to -32..31. Each 32-group has one scale (Q6_K: each 16-group).
// The three overloads differ only in how each format stores its bits; callers pick one by block type.
void expand_k(const BlockQ4_K& b, int8_t a[256]) {
  for (int c = 0; c < 4; ++c)
    for (int l = 0; l < 32; ++l) {
      a[64 * c + l] = static_cast<int8_t>(b.qs[32 * c + l] & 0xF);
      a[64 * c + 32 + l] = static_cast<int8_t>(b.qs[32 * c + l] >> 4);
    }
}
void expand_k(const BlockQ5_K& b, int8_t a[256]) {
  for (int c = 0; c < 4; ++c)
    for (int l = 0; l < 32; ++l) {
      a[64 * c + l] = static_cast<int8_t>((b.qs[32 * c + l] & 0xF) + (b.qh[l] & (1 << (2 * c)) ? 16 : 0));
      a[64 * c + 32 + l] = static_cast<int8_t>((b.qs[32 * c + l] >> 4) + (b.qh[l] & (2 << (2 * c)) ? 16 : 0));
    }
}
// Q6_K: a 128-element half is 4 groups of 32 whose 6-bit values combine ql nibbles (group 0: ql[0..31]
// low nibble, group 1: ql[32..63] low, group 2: ql[0..31] high, group 3: ql[32..63] high) with 2 bits each
// from qh[0..31] (bits 0-1, 2-3, 4-5, 6-7 respectively). The -32 bias makes the values signed.
void expand_k(const BlockQ6_K& b, int8_t a[256]) {
  for (int h = 0; h < 2; ++h)
    for (int l = 0; l < 32; ++l) {
      const uint8_t* ql = b.ql + 64 * h;
      const uint8_t qh = b.qh[32 * h + l];
      a[128 * h + l] = static_cast<int8_t>(((ql[l] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
      a[128 * h + 32 + l] = static_cast<int8_t>(((ql[32 + l] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
      a[128 * h + 64 + l] = static_cast<int8_t>(((ql[l] >> 4) | (((qh >> 4) & 3) << 4)) - 32);
      a[128 * h + 96 + l] = static_cast<int8_t>(((ql[32 + l] >> 4) | (((qh >> 6) & 3) << 4)) - 32);
    }
}

// Dequantizes nb Q4_K or Q5_K super-blocks starting at p: x = d * sc[k] * q - dmin * m[k] per 32-element
// sub-block k, in ggml's operation order (multiply d by the scale first, then by the quant, then subtract).
template <class Blk>
void dequantize_k45(const unsigned char* p, float* dst, int64_t nb) {
  for (int64_t b = 0; b < nb; ++b) {
    Blk blk;
    std::memcpy(&blk, p + b * sizeof blk, sizeof blk);
    int8_t a[256];
    expand_k(blk, a);
    const float d = fp16_to_fp32(blk.d), min = fp16_to_fp32(blk.dmin);
    for (int k = 0; k < 8; ++k) {
      uint8_t sc, m;
      scale_min_k4(k, blk.scales, &sc, &m);
      const float d1 = d * sc, m1 = min * m;
      for (int l = 0; l < 32; ++l) dst[b * kQK_K + 32 * k + l] = d1 * static_cast<float>(a[32 * k + l]) - m1;
    }
  }
}

// ggml generic Q4_K/Q5_K dot (ggml-cpu/quants.c:696, :771): eight int32 lanes (element e feeds lane e % 8), one
// float partial per lane, the min term subtracted into sumf per block, the lanes added to sumf last.
template <class Blk>
float dot_k45(const Blk* w, const BlockQ8_K* act, int64_t n_blocks) {
  const auto* pw = reinterpret_cast<const unsigned char*>(w);
  const auto* pa = reinterpret_cast<const unsigned char*>(act);
  float sums[8] = {};
  float sumf = 0;
  for (int64_t b = 0; b < n_blocks; ++b) {
    Blk bw;
    BlockQ8_K ba;
    std::memcpy(&bw, pw + b * sizeof bw, sizeof bw);
    std::memcpy(&ba, pa + b * sizeof ba, sizeof ba);
    int8_t a[256];
    expand_k(bw, a);
    int32_t aux32[8] = {};
    int sumi = 0;
    for (int k = 0; k < 8; ++k) {
      uint8_t sc, m;
      scale_min_k4(k, bw.scales, &sc, &m);
      sumi += (ba.bsums[2 * k] + ba.bsums[2 * k + 1]) * m;
      for (int e = 0; e < 32; ++e) aux32[e & 7] += sc * (ba.qs[32 * k + e] * a[32 * k + e]);
    }
    const float d = fp16_to_fp32(bw.d) * ba.d;
    for (int l = 0; l < 8; ++l) sums[l] += d * static_cast<float>(aux32[l]);
    const float dmin = fp16_to_fp32(bw.dmin) * ba.d;
    sumf -= dmin * static_cast<float>(sumi);
  }
  for (int l = 0; l < 8; ++l) sumf += sums[l];
  return sumf;
}

// ggml nearest_int (ggml-quants.c:621): round half to even through the float mantissa. |fval| < 2^22.
int nearest_int(float fval) {
  const float val = fval + 12582912.f;
  int32_t i;
  std::memcpy(&i, &val, sizeof i);
  return (i & 0x007fffff) - 0x00400000;
}
}  // namespace

// Dispatches on t. The legacy formats are decoded in place; the k-quants go through the helpers above.
// Each branch checks the length with the block size of its own type before reading anything.
void dequantize_row(DType t, const void* src, float* dst, int64_t n) {
  const auto* p = static_cast<const unsigned char*>(src);
  switch (t) {
    case DType::F32: std::memcpy(dst, src, static_cast<size_t>(n) * sizeof(float)); return;
    case DType::F16:
      for (int64_t i = 0; i < n; ++i) {
        uint16_t h;
        std::memcpy(&h, p + i * 2, 2);
        dst[i] = fp16_to_fp32(h);
      }
      return;
    case DType::Q4_0:
      check_n(n);
      for (int64_t b = 0; b < n / kQK; ++b) {
        BlockQ4_0 blk;
        std::memcpy(&blk, p + b * sizeof(BlockQ4_0), sizeof blk);
        const float d = fp16_to_fp32(blk.d);
        for (int j = 0; j < 16; ++j) {
          dst[b * kQK + j] = static_cast<float>((blk.qs[j] & 0x0F) - 8) * d;
          dst[b * kQK + j + 16] = static_cast<float>((blk.qs[j] >> 4) - 8) * d;
        }
      }
      return;
    case DType::Q8_0:
      check_n(n);
      for (int64_t b = 0; b < n / kQK; ++b) {
        BlockQ8_0 blk;
        std::memcpy(&blk, p + b * sizeof(BlockQ8_0), sizeof blk);
        const float d = fp16_to_fp32(blk.d);
        for (int j = 0; j < 32; ++j) dst[b * kQK + j] = static_cast<float>(blk.qs[j]) * d;
      }
      return;
    case DType::Q5_0:
      check_n(n);
      for (int64_t b = 0; b < n / kQK; ++b) {
        BlockQ5_0 blk;
        std::memcpy(&blk, p + b * sizeof(BlockQ5_0), sizeof blk);
        const float d = fp16_to_fp32(blk.d);
        uint32_t qh;
        std::memcpy(&qh, blk.qh, 4);
        for (int j = 0; j < 16; ++j) {
          const int x0 = ((blk.qs[j] & 0x0F) | (((qh >> j) << 4) & 0x10)) - 16;
          const int x1 = ((blk.qs[j] >> 4) | ((qh >> (j + 12)) & 0x10)) - 16;
          dst[b * kQK + j] = static_cast<float>(x0) * d;
          dst[b * kQK + j + 16] = static_cast<float>(x1) * d;
        }
      }
      return;
    case DType::Q5_1:
      check_n(n);
      for (int64_t b = 0; b < n / kQK; ++b) {
        BlockQ5_1 blk;
        std::memcpy(&blk, p + b * sizeof(BlockQ5_1), sizeof blk);
        const float d = fp16_to_fp32(blk.d), m = fp16_to_fp32(blk.m);
        uint32_t qh;
        std::memcpy(&qh, blk.qh, 4);
        for (int j = 0; j < 16; ++j) {
          const int x0 = (blk.qs[j] & 0x0F) | (((qh >> j) << 4) & 0x10);
          const int x1 = (blk.qs[j] >> 4) | ((qh >> (j + 12)) & 0x10);
          dst[b * kQK + j] = static_cast<float>(x0) * d + m;
          dst[b * kQK + j + 16] = static_cast<float>(x1) * d + m;
        }
      }
      return;
    case DType::Q4_K:
      check_n(n, kQK_K);
      dequantize_k45<BlockQ4_K>(p, dst, n / kQK_K);
      return;
    case DType::Q5_K:
      check_n(n, kQK_K);
      dequantize_k45<BlockQ5_K>(p, dst, n / kQK_K);
      return;
    case DType::Q6_K:
      check_n(n, kQK_K);
      for (int64_t b = 0; b < n / kQK_K; ++b) {
        BlockQ6_K blk;
        std::memcpy(&blk, p + b * sizeof blk, sizeof blk);
        int8_t a[256];
        expand_k(blk, a);
        const float d = fp16_to_fp32(blk.d);
        for (int i = 0; i < 256; ++i)
          dst[b * kQK_K + i] = d * static_cast<float>(blk.scales[i / 16]) * static_cast<float>(a[i]);
      }
      return;
  }
  throw Error("dequantize_row: unsupported dtype");
}

// One block at a time through quantize_block_q8; the finished block is memcpy'd out because y may be
// unaligned.
void quantize_row_q8_0(const float* x, BlockQ8_0* y, int64_t n) {
  check_n(n);
  for (int64_t b = 0; b < n / kQK; ++b) {
    BlockQ8_0 blk;
    blk.d = fp32_to_fp16(quantize_block_q8(x + b * kQK, blk.qs));
    std::memcpy(reinterpret_cast<unsigned char*>(y) + b * sizeof blk, &blk, sizeof blk);
  }
}

// Q8_0's d and qs, plus s = fp16(sum(qs) * d) from the unrounded d.
void quantize_row_q8_1(const float* x, BlockQ8_1* y, int64_t n) {
  check_n(n);
  for (int64_t b = 0; b < n / kQK; ++b) {
    BlockQ8_1 blk;
    const float d = quantize_block_q8(x + b * kQK, blk.qs);
    blk.d = fp32_to_fp16(d);
    int sum = 0;
    for (const int8_t q : blk.qs) sum += q;
    blk.s = fp32_to_fp16(static_cast<float>(sum) * d);
    std::memcpy(reinterpret_cast<unsigned char*>(y) + b * sizeof blk, &blk, sizeof blk);
  }
}

// ggml quantize_row_q8_K_ref, with Ruling P1: the scaled value is clamped to [-127, 127] (NaN to 0) before
// nearest_int, so non-finite input and an infinite iscale (max subnormal) give defined qs, never -128. For
// finite iscale the clamp cannot change a result. An all-zero block also zeroes bsums (ggml leaves them).
void quantize_row_q8_K(const float* x, BlockQ8_K* y, int64_t n) {
  check_n(n, kQK_K);
  for (int64_t b = 0; b < n / kQK_K; ++b, x += kQK_K) {
    BlockQ8_K blk{};
    float max = 0, amax = 0;
    bool has_nan = false;
    for (int j = 0; j < kQK_K; ++j) {
      const float ax = std::fabs(x[j]);
      if (ax > amax) {
        amax = ax;
        max = x[j];
      }
      has_nan |= x[j] != x[j];
    }
    if (amax) {
      const float iscale = -127.f / max;
      for (int j = 0; j < kQK_K; ++j) {
        const float v = iscale * x[j];
        blk.qs[j] = static_cast<int8_t>(v != v ? 0 : nearest_int(std::fmin(127.f, std::fmax(-127.f, v))));
      }
      for (int g = 0; g < 16; ++g) {
        int sum = 0;
        for (int i = 0; i < 16; ++i) sum += blk.qs[g * 16 + i];
        blk.bsums[g] = static_cast<int16_t>(sum);
      }
      blk.d = 1 / iscale;
    }
    if (has_nan) blk.d = std::numeric_limits<float>::quiet_NaN();  // see quantize_block_q8
    std::memcpy(reinterpret_cast<unsigned char*>(y) + b * sizeof blk, &blk, sizeof blk);
  }
}

// Q4_0 x Q8_0: per block, integer sum of (nibble - 8) * activation, then * d_w * d_a. Left-to-right float
// multiplication order is ggml's (sumi * d_w) * d_a.
float dot_q4_0_q8_0(const BlockQ4_0* w, const BlockQ8_0* a, int64_t n_blocks) {
  const auto* pw = reinterpret_cast<const unsigned char*>(w);
  const auto* pa = reinterpret_cast<const unsigned char*>(a);
  float sum = 0.f;
  for (int64_t b = 0; b < n_blocks; ++b) {
    BlockQ4_0 bw;
    BlockQ8_0 ba;
    std::memcpy(&bw, pw + b * sizeof bw, sizeof bw);
    std::memcpy(&ba, pa + b * sizeof ba, sizeof ba);
    int s0 = 0, s1 = 0;
    for (int j = 0; j < 16; ++j) {
      s0 += ((bw.qs[j] & 0x0F) - 8) * ba.qs[j];
      s1 += ((bw.qs[j] >> 4) - 8) * ba.qs[j + 16];
    }
    sum += static_cast<float>(s0 + s1) * fp16_to_fp32(bw.d) * fp16_to_fp32(ba.d);
  }
  return sum;
}

// Q8_0 x Q8_0: per block, the integer dot of the 32 pairs, then sumi * (d_w * d_a) (ggml groups the two
// scales first here, unlike the Q4_0 dot).
float dot_q8_0_q8_0(const BlockQ8_0* w, const BlockQ8_0* a, int64_t n_blocks) {
  const auto* pw = reinterpret_cast<const unsigned char*>(w);
  const auto* pa = reinterpret_cast<const unsigned char*>(a);
  float sum = 0.f;
  for (int64_t b = 0; b < n_blocks; ++b) {
    BlockQ8_0 bw, ba;
    std::memcpy(&bw, pw + b * sizeof bw, sizeof bw);
    std::memcpy(&ba, pa + b * sizeof ba, sizeof ba);
    int s = 0;
    for (int j = 0; j < 32; ++j) s += bw.qs[j] * ba.qs[j];
    sum += static_cast<float>(s) * (fp16_to_fp32(bw.d) * fp16_to_fp32(ba.d));  // ggml generic: sumi*(d_x*d_y)
  }
  return sum;
}

// Q5_0 x Q8_0: as Q4_0 with the fifth bit restored from qh and a -16 bias; scales grouped as (d_w * d_a) *
// sumi.
float dot_q5_0_q8_0(const BlockQ5_0* w, const BlockQ8_0* a, int64_t n_blocks) {
  const auto* pw = reinterpret_cast<const unsigned char*>(w);
  const auto* pa = reinterpret_cast<const unsigned char*>(a);
  float sum = 0.f;
  for (int64_t b = 0; b < n_blocks; ++b) {
    BlockQ5_0 bw;
    BlockQ8_0 ba;
    std::memcpy(&bw, pw + b * sizeof bw, sizeof bw);
    std::memcpy(&ba, pa + b * sizeof ba, sizeof ba);
    uint32_t qh;
    std::memcpy(&qh, bw.qh, 4);
    int s0 = 0, s1 = 0;
    for (int j = 0; j < 16; ++j) {
      s0 += (((bw.qs[j] & 0x0F) | (((qh >> j) << 4) & 0x10)) - 16) * ba.qs[j];
      s1 += (((bw.qs[j] >> 4) | ((qh >> (j + 12)) & 0x10)) - 16) * ba.qs[j + 16];
    }
    sum += (fp16_to_fp32(bw.d) * fp16_to_fp32(ba.d)) * static_cast<float>(s0 + s1);  // ggml generic: (d_x*d_y)*sumi
  }
  return sum;
}

// Q5_1 x Q8_1: unsigned 5-bit values, so the weight's minimum contributes m_w * s_a (s_a is the activation
// block's value sum). The block term (d_w * d_a) * sumi + m_w * s_a is formed first, then added to the total.
float dot_q5_1_q8_1(const BlockQ5_1* w, const BlockQ8_1* a, int64_t n_blocks) {
  const auto* pw = reinterpret_cast<const unsigned char*>(w);
  const auto* pa = reinterpret_cast<const unsigned char*>(a);
  float sum = 0.f;
  for (int64_t b = 0; b < n_blocks; ++b) {
    BlockQ5_1 bw;
    BlockQ8_1 ba;
    std::memcpy(&bw, pw + b * sizeof bw, sizeof bw);
    std::memcpy(&ba, pa + b * sizeof ba, sizeof ba);
    uint32_t qh;
    std::memcpy(&qh, bw.qh, 4);
    int s0 = 0, s1 = 0;
    for (int j = 0; j < 16; ++j) {
      s0 += ((bw.qs[j] & 0x0F) | (((qh >> j) << 4) & 0x10)) * ba.qs[j];
      s1 += ((bw.qs[j] >> 4) | ((qh >> (j + 12)) & 0x10)) * ba.qs[j + 16];
    }
    // ggml generic: sumf += ((d_x*d_y)*sumi + m_x*s_y), the block term summed before accumulating.
    sum += (fp16_to_fp32(bw.d) * fp16_to_fp32(ba.d)) * static_cast<float>(s0 + s1) +
           fp16_to_fp32(bw.m) * fp16_to_fp32(ba.s);
  }
  return sum;
}

float dot_q4_K_q8_K(const BlockQ4_K* w, const BlockQ8_K* a, int64_t n_blocks) { return dot_k45(w, a, n_blocks); }
float dot_q5_K_q8_K(const BlockQ5_K* w, const BlockQ8_K* a, int64_t n_blocks) { return dot_k45(w, a, n_blocks); }

// ggml generic Q6_K dot (ggml-cpu/quants.c:851): the lanes and block order of dot_k45, with one scale per 16
// elements and no min term.
float dot_q6_K_q8_K(const BlockQ6_K* w, const BlockQ8_K* act, int64_t n_blocks) {
  const auto* pw = reinterpret_cast<const unsigned char*>(w);
  const auto* pa = reinterpret_cast<const unsigned char*>(act);
  float sums[8] = {};
  for (int64_t b = 0; b < n_blocks; ++b) {
    BlockQ6_K bw;
    BlockQ8_K ba;
    std::memcpy(&bw, pw + b * sizeof bw, sizeof bw);
    std::memcpy(&ba, pa + b * sizeof ba, sizeof ba);
    int8_t a[256];
    expand_k(bw, a);
    int32_t aux32[8] = {};
    for (int e = 0; e < 256; ++e) aux32[e & 7] += bw.scales[e / 16] * (ba.qs[e] * a[e]);
    const float d = fp16_to_fp32(bw.d) * ba.d;
    for (int l = 0; l < 8; ++l) sums[l] += d * static_cast<float>(aux32[l]);
  }
  float sumf = 0;
  for (int l = 0; l < 8; ++l) sumf += sums[l];
  return sumf;
}

// Mirrors ggml's vec_dot_type table for the supported weight types.
ActType act_type(DType w) {
  switch (w) {
    case DType::Q4_0:
    case DType::Q5_0:
    case DType::Q8_0: return ActType::Q8_0;
    case DType::Q5_1: return ActType::Q8_1;
    case DType::Q4_K:
    case DType::Q5_K:
    case DType::Q6_K: return ActType::Q8_K;
    case DType::F32:
    case DType::F16: return ActType::None;
  }
  throw Error("act_type: unsupported dtype");
}

}  // namespace nibble
