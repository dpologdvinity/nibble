#pragma once
#include <cstdint>

#include "tensor.h"

namespace nibble {

struct BlockQ4_0 {
  uint16_t d;
  uint8_t qs[16];
};
struct BlockQ8_0 {
  uint16_t d;
  int8_t qs[32];
};
static_assert(sizeof(BlockQ4_0) == 18);
static_assert(sizeof(BlockQ8_0) == 34);

inline constexpr int64_t kQK = 32;

// n % 32 == 0 for Q types. src may be unaligned. Throws Error on bad dtype/n.
void dequantize_row(DType t, const void* src, float* dst, int64_t n);
void quantize_row_q8_0(const float* x, BlockQ8_0* y, int64_t n);
float dot_q4_0_q8_0(const BlockQ4_0* w, const BlockQ8_0* a, int64_t n_blocks);
float dot_q8_0_q8_0(const BlockQ8_0* w, const BlockQ8_0* a, int64_t n_blocks);

}  // namespace nibble
