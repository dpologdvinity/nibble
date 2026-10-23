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
struct BlockQ5_0 {
  uint16_t d;
  uint8_t qh[4];  // bit j is the high bit of element j
  uint8_t qs[16];
};
struct BlockQ5_1 {
  uint16_t d, m;
  uint8_t qh[4];
  uint8_t qs[16];
};
struct BlockQ8_1 {
  uint16_t d, s;  // s = fp16(d * sum(qs)) with the pre-fp16 d
  int8_t qs[32];
};
static_assert(sizeof(BlockQ4_0) == 18);
static_assert(sizeof(BlockQ8_0) == 34);
static_assert(sizeof(BlockQ5_0) == 22);
static_assert(sizeof(BlockQ5_1) == 24);
static_assert(sizeof(BlockQ8_1) == 36);

inline constexpr int64_t kQK = 32;

// n % 32 == 0 for Q types. src may be unaligned. Throws Error on bad dtype/n.
void dequantize_row(DType t, const void* src, float* dst, int64_t n);
void quantize_row_q8_0(const float* x, BlockQ8_0* y, int64_t n);
void quantize_row_q8_1(const float* x, BlockQ8_1* y, int64_t n);
float dot_q4_0_q8_0(const BlockQ4_0* w, const BlockQ8_0* a, int64_t n_blocks);
float dot_q8_0_q8_0(const BlockQ8_0* w, const BlockQ8_0* a, int64_t n_blocks);
float dot_q5_0_q8_0(const BlockQ5_0* w, const BlockQ8_0* a, int64_t n_blocks);
float dot_q5_1_q8_1(const BlockQ5_1* w, const BlockQ8_1* a, int64_t n_blocks);

}  // namespace nibble
