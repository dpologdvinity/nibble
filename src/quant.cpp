#include "quant.h"

#include <cmath>
#include <cstring>

#include "fp16.h"
#include <nibble/error.h>

namespace nibble {

namespace {
void check_n(int64_t n) {
  if (n < 0 || n % kQK != 0) throw Error("quant: length must be a multiple of 32");
}

// Shared Q8_0/Q8_1 block core (ggml quantize_row_q8_{0,1}_ref): writes 32 qs, returns the pre-fp16 d.
// roundf rounds half away from zero (Q8_K's nearest_int is half to even, not this). NaN maps to 0 and results
// saturate to [-127, 127], so the int8 cast is never UB; only non-finite rows or blocks with id = Inf
// (amax < ~3.7e-37) reach either case, and finite normal blocks stay bit-exact.
float quantize_block_q8(const float* x, int8_t* qs) {
  float amax = 0.f;
  // Differs from ggml's MAX(amax, fabsf(v)) only on NaN: fmax skips it, MAX yields it until a later element.
  for (int j = 0; j < kQK; ++j) amax = std::fmax(amax, std::fabs(x[j]));
  const float d = amax / 127.f;
  const float id = d ? 1.0f / d : 0.0f;
  for (int j = 0; j < kQK; ++j) {
    const float r = std::roundf(x[j] * id);
    qs[j] = r != r ? 0 : static_cast<int8_t>(std::fmin(127.f, std::fmax(-127.f, r)));
  }
  return d;
}
}  // namespace

void dequantize_row(DType t, const void* src, float* dst, int64_t n) {
  const auto* p = static_cast<const unsigned char*>(src);
  switch (t) {
    case DType::F32:
      std::memcpy(dst, src, static_cast<size_t>(n) * sizeof(float));
      return;
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
    case DType::Q4_K: case DType::Q5_K: case DType::Q6_K:
      break;  // dequantizers arrive with their kernels
  }
  throw Error("dequantize_row: unsupported dtype");
}

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

}  // namespace nibble
