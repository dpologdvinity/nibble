#pragma once
// Plain-C++ reference implementations of every backend operation. They are deliberately simple and slow: the
// correctness oracle that the AVX2 kernels (and any future GPU backend) are tested against, and the fallback
// when SIMD is disabled or unavailable. Where a SIMD kernel is bit-identical to its twin here, tests compare
// bytes; elsewhere they compare within a tolerance.
#include <cstdint>
#include <span>

#include "backend/backend.h"
#include "quant.h"
#include "tensor.h"

// Scalar reference kernels: the correctness oracle for all SIMD/GPU kernels.
// Weight dtypes: F32, F16, Q4_0, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K, Q6_K; others throw nibble::Error.
namespace nibble::cpu {

// Numerically stable softmax over x[0..n) in place (max-subtracted, double-precision sum).
void softmax_inplace(float* x, int64_t n);
// Embedding lookup for any supported table type; throws Error for an out-of-range token id.
void embed(const Tensor& table, std::span<const int32_t> ids, float* out);
// Quantizes n rows of `width` floats (width % 32 == 0) into n * width/32 blocks.
void quantize_rows_q8_0(const float* x, BlockQ8_0* y, int64_t n, int64_t width);
// xq: x quantized to the weight's act_type (rows of w.ne[0] / block_size blocks); required for quantized
// weights, ignored for F32/F16.
void matmul_rows(const Tensor& w, const float* x, float* y, int64_t n, int64_t row_begin, int64_t row_end,
                 const QuantActs* xq = nullptr);
// y = x / sqrt(mean(x^2) + eps) * weight, per row of width weight.ne[0]; the sum is accumulated in double.
void rmsnorm(const float* x, const Tensor& weight, float* y, int64_t n, float eps);
// x[i] += y[i] for i < count.
void add(float* x, const float* y, int64_t count);
// Adds the F32 bias vector to each of the n rows of x.
void add_bias(float* x, const Tensor& bias, int64_t n);
// Rotary position embedding of n tokens (positions pos0..), n_heads heads of head_dim, in place.
void rope(float* x, int64_t n, int64_t n_heads, int64_t head_dim, int64_t pos0, const RopeParams& p);
// Causal attention for heads [head_begin, head_end): token t attends to positions 0..pos0+t, with
// grouped-query head sharing (n_heads / n_kv_heads query heads per K/V head). q and out are
// [n, n_heads, head_dim], k and v are [n_ctx, n_kv_heads, head_dim].
// scratch: caller-provided, at least pos0 + n floats (no allocation inside).
void attention_heads(const float* q, const float* k, const float* v, float* out, int64_t n, int64_t pos0,
                     int64_t n_heads, int64_t n_kv_heads, int64_t head_dim, int64_t head_begin, int64_t head_end,
                     float* scratch);
// gate[i] = silu(gate[i]) * up[i], with silu(g) = g / (1 + e^-g): the SwiGLU activation.
void silu_mul(float* gate, const float* up, int64_t count);

}  // namespace nibble::cpu
