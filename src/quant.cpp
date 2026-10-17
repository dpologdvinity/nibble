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
  }
  throw Error("dequantize_row: unsupported dtype");
}

void quantize_row_q8_0(const float* x, BlockQ8_0* y, int64_t n) {
  check_n(n);
  for (int64_t b = 0; b < n / kQK; ++b) {
    float amax = 0.f;
    for (int j = 0; j < kQK; ++j) amax = std::fmax(amax, std::fabs(x[b * kQK + j]));
    const float d = amax / 127.f;
    const float id = d ? 1.0f / d : 0.0f;
    BlockQ8_0 blk;
    blk.d = fp32_to_fp16(d);
    for (int j = 0; j < kQK; ++j) blk.qs[j] = static_cast<int8_t>(std::roundf(x[b * kQK + j] * id));
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
    sum += static_cast<float>(s) * fp16_to_fp32(bw.d) * fp16_to_fp32(ba.d);
  }
  return sum;
}

}  // namespace nibble
