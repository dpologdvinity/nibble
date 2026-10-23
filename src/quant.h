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
inline constexpr int64_t kQK_K = 256;
struct BlockQ4_K {
  uint16_t d, dmin;
  uint8_t scales[12];  // eight 6-bit scales and eight 6-bit mins, packed
  uint8_t qs[128];
};
struct BlockQ5_K {
  uint16_t d, dmin;
  uint8_t scales[12];
  uint8_t qh[32];  // bit k of qh[l] is the high bit of element 32k + l
  uint8_t qs[128];
};
struct BlockQ6_K {
  uint8_t ql[128];
  uint8_t qh[64];
  int8_t scales[16];
  uint16_t d;
};
struct BlockQ8_K {
  float d;
  int8_t qs[256];
  int16_t bsums[16];  // sums of each 16 qs
};
static_assert(sizeof(BlockQ4_0) == 18);
static_assert(sizeof(BlockQ8_0) == 34);
static_assert(sizeof(BlockQ5_0) == 22);
static_assert(sizeof(BlockQ5_1) == 24);
static_assert(sizeof(BlockQ8_1) == 36);
static_assert(sizeof(BlockQ4_K) == 144);
static_assert(sizeof(BlockQ5_K) == 176);
static_assert(sizeof(BlockQ6_K) == 210);
static_assert(sizeof(BlockQ8_K) == 292);

inline constexpr int64_t kQK = 32;

// n must be a multiple of the type's block size (32; 256 for K types). src may be unaligned. Throws Error on bad dtype/n.
void dequantize_row(DType t, const void* src, float* dst, int64_t n);
void quantize_row_q8_0(const float* x, BlockQ8_0* y, int64_t n);
void quantize_row_q8_1(const float* x, BlockQ8_1* y, int64_t n);
void quantize_row_q8_K(const float* x, BlockQ8_K* y, int64_t n);
float dot_q4_0_q8_0(const BlockQ4_0* w, const BlockQ8_0* a, int64_t n_blocks);
float dot_q8_0_q8_0(const BlockQ8_0* w, const BlockQ8_0* a, int64_t n_blocks);
float dot_q5_0_q8_0(const BlockQ5_0* w, const BlockQ8_0* a, int64_t n_blocks);
float dot_q5_1_q8_1(const BlockQ5_1* w, const BlockQ8_1* a, int64_t n_blocks);

float dot_q4_K_q8_K(const BlockQ4_K* w, const BlockQ8_K* a, int64_t n_blocks);
float dot_q5_K_q8_K(const BlockQ5_K* w, const BlockQ8_K* a, int64_t n_blocks);
float dot_q6_K_q8_K(const BlockQ6_K* w, const BlockQ8_K* a, int64_t n_blocks);

// The activation quantization a weight type is multiplied against (ggml's vec_dot_type). None: float weights.
enum class ActType { None, Q8_0, Q8_1, Q8_K };
ActType act_type(DType w);

}  // namespace nibble
