#pragma once
#include <cstdint>
#include <span>

#include "backend/backend.h"
#include "tensor.h"

// Scalar reference kernels: the correctness oracle for all SIMD/GPU kernels.
// Weight dtypes: F32 and F16 only; others throw nibble::Error.
namespace nibble::cpu {

void softmax_inplace(float* x, int64_t n);
void embed(const Tensor& table, std::span<const int32_t> ids, float* out);
void matmul_rows(const Tensor& w, const float* x, float* y, int64_t n, int64_t row_begin, int64_t row_end);
void rmsnorm(const float* x, const Tensor& weight, float* y, int64_t n, float eps);
void add(float* x, const float* y, int64_t count);
void add_bias(float* x, const Tensor& bias, int64_t n);
void rope(float* x, int64_t n, int64_t n_heads, int64_t head_dim, int64_t pos0, const RopeParams& p);
// scratch: caller-provided, at least pos0 + n floats (no allocation inside).
void attention_heads(const float* q, const float* k, const float* v, float* out, int64_t n, int64_t pos0,
                     int64_t n_heads, int64_t n_kv_heads, int64_t head_dim, int64_t head_begin,
                     int64_t head_end, float* scratch);
void silu_mul(float* gate, const float* up, int64_t count);

}  // namespace nibble::cpu
