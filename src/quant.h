#pragma once
// Quantized block formats and the scalar reference kernels that work on them.
//
// Weights live in the GGUF file as arrays of fixed-size blocks (the formats are llama.cpp's, byte for
// byte). A matmul never dequantizes the weights: it quantizes the *activations* to a companion 8-bit
// format (ActType), takes integer dot products block by block, and scales each block's integer sum by the
// two blocks' scales. The pairing follows ggml's `vec_dot_type`:
//
//     weight type           activation block       dot function
//     Q4_0, Q5_0, Q8_0      BlockQ8_0              dot_q4_0_q8_0, dot_q5_0_q8_0, dot_q8_0_q8_0
//     Q5_1                  BlockQ8_1              dot_q5_1_q8_1
//     Q4_K, Q5_K, Q6_K      BlockQ8_K              dot_q4_K_q8_K, dot_q5_K_q8_K, dot_q6_K_q8_K
//
// The functions here are the *scalar oracle*: simple loops whose floating-point operation order was copied
// from ggml's generic (non-SIMD) code, so results are bit-identical to llama.cpp's generic build for finite
// inputs (pinned by golden vectors in tests/data/quant_golden.inc). The AVX2 kernels in
// backend/cpu/simd reproduce these results bit for bit, and tests compare them against this file.
//
// All scales are IEEE half precision stored as raw uint16_t bit patterns (see fp16.h) unless noted. Every
// function reads blocks through memcpy, so block arrays may be unaligned (they point into a mmap'ed file at
// arbitrary byte offsets). Blocks from a file are untrusted: any byte pattern, including NaN/Inf scales,
// must be safe to dequantize or dot (the result is then non-finite, never undefined behaviour).
#include <cstdint>

#include "tensor.h"

namespace nibble {

// ---- Weight and activation block layouts. Element order inside a block is the order of the dequantized
// ---- values; "low/high nibble" refers to the 4-bit halves of a byte.

// Q4_0 weights, 32 elements: x[j] = (low nibble of qs[j] - 8) * d and x[j + 16] = (high nibble - 8) * d.
struct BlockQ4_0 {
  uint16_t d;       // scale (fp16)
  uint8_t qs[16];   // 32 4-bit values, two per byte
};
// Q8_0 weights or activations, 32 elements: x[j] = qs[j] * d. Activations quantized by quantize_row_q8_0
// never contain -128 (qs is in [-127, 127]).
struct BlockQ8_0 {
  uint16_t d;       // scale (fp16)
  int8_t qs[32];    // signed 8-bit values
};
// Q5_0 weights, 32 elements: like Q4_0 with a fifth bit per element, x[e] = (q5 - 16) * d where q5 is the
// element's nibble plus 16 if its bit in qh is set.
struct BlockQ5_0 {
  uint16_t d;       // scale (fp16)
  uint8_t qh[4];    // bit e of this little-endian 32-bit word is the high bit of element e
  uint8_t qs[16];   // low 4 bits of each value: element j in the low nibble of qs[j], element j + 16 in the high
};
// Q5_1 weights, 32 elements: asymmetric, x[e] = q5 * d + m with q5 built as in Q5_0 (no -16 offset).
struct BlockQ5_1 {
  uint16_t d, m;    // scale and minimum (fp16)
  uint8_t qh[4];    // high bits, as in BlockQ5_0
  uint8_t qs[16];   // low nibbles, as in BlockQ5_0
};
// Q8_1 activations, 32 elements: Q8_0 plus s, the block's value sum times d, which the Q5_1 dot needs for
// its minimum term m * s.
struct BlockQ8_1 {
  uint16_t d, s;    // scale (fp16) and s = fp16(d * sum(qs)) computed from the scale *before* it was rounded to fp16
  int8_t qs[32];
};

// Elements in a k-quant super-block.
inline constexpr int64_t kQK_K = 256;

// Q4_K weights, 256 elements = 8 sub-blocks of 32. x = d * sc[k] * q - dmin * m[k] for sub-block k, with a
// 4-bit q. sc[k] and m[k] are 6-bit and packed into `scales` (ggml's get_scale_min_k4 layout): bytes 0-3
// hold the low 6 bits of sc[0..3] and bytes 4-7 those of m[0..3]. For k = 4..7 the low 4 bits of sc[k] are
// the low nibble of byte k + 4 and the low 4 bits of m[k] its high nibble; their top 2 bits are bits 6-7 of
// byte k - 4 (for sc[k]) and of byte k (for m[k]).
struct BlockQ4_K {
  uint16_t d, dmin;     // super-block scale for the scales, and for the minimums (fp16)
  uint8_t scales[12];   // eight 6-bit scales and eight 6-bit mins, packed
  uint8_t qs[128];      // 256 4-bit values: for each 64-element chunk c, 32 bytes whose low nibbles are
                        // elements 64c..64c+31 (sub-block 2c) and high nibbles elements 64c+32..64c+63
};
// Q5_K weights: Q4_K plus a fifth bit per element; q = nibble + 16 if the element's bit in qh is set.
struct BlockQ5_K {
  uint16_t d, dmin;
  uint8_t scales[12];   // packed as in BlockQ4_K
  uint8_t qh[32];       // bit k of qh[l] is the high bit of element 32k + l (sub-block k, position l)
  uint8_t qs[128];      // low nibbles, laid out as in BlockQ4_K
};
// Q6_K weights, 256 elements = 16 sub-blocks of 16: x = d * scales[k] * (q6 - 32), q6 in [0, 63]. There
// is no minimum. The 6 bits are the low 4 bits from ql and the high 2 from qh, interleaved in ggml's
// 128-element halves (see the unpacking in quant.cpp / the AVX2 kernel).
struct BlockQ6_K {
  uint8_t ql[128];      // low 4 bits of the 6-bit values
  uint8_t qh[64];       // high 2 bits
  int8_t scales[16];    // signed 8-bit scale per 16-element sub-block (full int8 range, -128 included)
  uint16_t d;           // super-block scale (fp16); note: last in the struct, unlike Q4_K/Q5_K
};
// Q8_K activations, 256 elements, paired with all three k-quant weight types. The scale is a float (not
// fp16) and `bsums` caches sums of each 16 values so the Q4_K/Q5_K minimum term costs no extra pass.
// quantize_row_q8_K never produces -128 (qs is in [-127, 127]); the AVX2 Q6_K kernel depends on that.
struct BlockQ8_K {
  float d;              // scale
  int8_t qs[256];
  int16_t bsums[16];    // bsums[g] = sum of qs[16 g .. 16 g + 15]
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

// Elements in a legacy (non-k) quant block: Q4_0, Q5_0, Q5_1, Q8_0, Q8_1.
inline constexpr int64_t kQK = 32;

// ---- Dequantization and activation quantization. In all of them n counts elements and must be a multiple
// ---- of the block size involved; otherwise nibble::Error is thrown.

// Converts n elements of a row of type t (F32, F16 or any quantized type) to float. src may be unaligned.
// n must be a multiple of the type's block size (32; 256 for the k-quants). Throws Error on a bad dtype or n.
void dequantize_row(DType t, const void* src, float* dst, int64_t n);
// Quantizes n floats to Q8_0 blocks (ggml quantize_row_q8_0_ref): per block, d = max|x| / 127 and
// qs = round(x / d), rounding half away from zero. Defined for NaN/Inf input too (NaN quantizes to 0 and
// values saturate to +-127), though only finite normal blocks match ggml exactly.
void quantize_row_q8_0(const float* x, BlockQ8_0* y, int64_t n);
// As quantize_row_q8_0, plus the block sum s that Q5_1's dot needs.
void quantize_row_q8_1(const float* x, BlockQ8_1* y, int64_t n);
// Quantizes n floats (n a multiple of 256) to Q8_K blocks (ggml quantize_row_q8_K_ref): the scale comes from
// the element of largest magnitude (the first one on ties), values round half to even, and the result lies
// in [-127, 127]. An all-zero block gets d = 0 and zero bsums.
void quantize_row_q8_K(const float* x, BlockQ8_K* y, int64_t n);

// ---- Block dot products: sum over n_blocks consecutive blocks of the weight row w against the activation
// ---- row a (same number of blocks). Each returns sum_b d_w * d_a * (integer dot of block b) in ggml's
// ---- exact evaluation order. Both pointers may be unaligned.
float dot_q4_0_q8_0(const BlockQ4_0* w, const BlockQ8_0* a, int64_t n_blocks);
float dot_q8_0_q8_0(const BlockQ8_0* w, const BlockQ8_0* a, int64_t n_blocks);
float dot_q5_0_q8_0(const BlockQ5_0* w, const BlockQ8_0* a, int64_t n_blocks);
// Q5_1 adds the minimum term: per block, d_w * d_a * (integer dot) + m_w * s_a.
float dot_q5_1_q8_1(const BlockQ5_1* w, const BlockQ8_1* a, int64_t n_blocks);

// K-quant dots (n_blocks counts 256-element super-blocks). They accumulate in eight float lanes the way
// ggml's generic code does (element e belongs to lane e % 8); Q4_K and Q5_K subtract the minimum term
// dmin * (sum over sub-blocks of m[k] * bsums) from a separate running total.
float dot_q4_K_q8_K(const BlockQ4_K* w, const BlockQ8_K* a, int64_t n_blocks);
float dot_q5_K_q8_K(const BlockQ5_K* w, const BlockQ8_K* a, int64_t n_blocks);
float dot_q6_K_q8_K(const BlockQ6_K* w, const BlockQ8_K* a, int64_t n_blocks);

// The activation quantization a weight type is multiplied against (ggml's vec_dot_type). None: float weights.
enum class ActType { None, Q8_0, Q8_1, Q8_K };
// Activation type for weights of type w: Q4_0/Q5_0/Q8_0 -> Q8_0, Q5_1 -> Q8_1, K-quants -> Q8_K,
// F32/F16 -> None. Throws Error for a value that is not a DType enumerator.
ActType act_type(DType w);

// The activation rows a quantized matmul reads: one pointer per ActType, each holding n rows of
// ne0 / block_size blocks. Only the pointers the weights' act_type()s need are set.
struct QuantActs {
  const BlockQ8_0* q8_0 = nullptr;
  const BlockQ8_1* q8_1 = nullptr;
  const BlockQ8_K* q8_K = nullptr;
};

}  // namespace nibble
