#pragma once
// The CPU implementation of Backend: scalar reference kernels (ref_ops.h) or AVX2 kernels (simd/) chosen at
// compile time, with matmuls and attention parallelized over a ThreadPool.
//
// Memory: activation buffers come from alloc() (64-byte aligned, zeroed, tracked for cleanup), and the
// scratch used inside ops (attention scores, quantized copies of the activations) is sized once by reserve()
// so the decode loop allocates nothing.
#include <memory>
#include <vector>

#include "backend/backend.h"
#include "backend/cpu/simd/simd.h"
#include "quant.h"
#include "thread_pool.h"

namespace nibble::cpu {

class CpuBackend final : public Backend {
 public:
  // `pool` must outlive the backend; every parallel op runs on it.
  explicit CpuBackend(ThreadPool& pool);
  ~CpuBackend() override;
  CpuBackend(const CpuBackend&) = delete;
  CpuBackend& operator=(const CpuBackend&) = delete;

  // F32, F16, Q4_0, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K, Q6_K.
  bool supports(DType t) const override {
    switch (t) {
      case DType::F32: case DType::F16: case DType::Q4_0: case DType::Q5_0: case DType::Q5_1: case DType::Q8_0:
      case DType::Q4_K: case DType::Q5_K: case DType::Q6_K:
        return true;
    }
    return false;
  }
  // Sizes the attention scratch and the quantized-activation scratch for the given limits. Throws Error if a
  // size overflows or cannot be allocated.
  void reserve(const ReserveParams&) override;
  size_t scratch_capacity() const { return scratch_.size(); }  // test hook
  size_t quant_scratch_capacity() const { return xq_.size() + xq1_.size() + xqk_.size(); }  // test hook (blocks)
  // Test hook: activation-quantization passes so far, one per activation type per matmul/matmul_multi call.
  uint64_t quantize_calls() const { return quantize_calls_; }
  float* alloc(int64_t n_floats) override;  // 64-byte aligned, zeroed
  void free(float* p) override;
  size_t live_allocations() const { return allocs_.size(); }  // test hook
  void read(float* host_dst, const float* src, int64_t n) override;
  void embed(const Tensor& table, std::span<const int32_t> ids, float* out) override;
  // A single-weight matmul_multi.
  void matmul(const Tensor& w, const float* x, float* y, int64_t n) override;
  // Quantizes (and packs) x once per activation type the weights need and runs all weights in one
  // parallel_for over their combined rows; differing input widths fall back to one matmul per weight.
  void matmul_multi(std::span<const Tensor* const> ws, const float* x, std::span<float* const> ys, int64_t n) override;
  // The remaining ops forward to the SIMD or scalar kernel for this build.
  void rmsnorm(const float* x, const Tensor& weight, float* y, int64_t n, float eps) override;
  void add(float* x, const float* y, int64_t count) override;
  void add_bias(float* x, const Tensor& bias, int64_t n) override;
  void rope(float* x, int64_t n, int64_t n_heads, int64_t head_dim, int64_t pos0, const RopeParams& p) override;
  void copy(float* dst, const float* src, int64_t count) override;
  void attention(const float* q, const float* k, const float* v, float* out, int64_t n, int64_t pos0,
                 int64_t n_heads, int64_t n_kv_heads, int64_t head_dim) override;
  void silu_mul(float* gate, const float* up, int64_t count) override;

 private:
  ThreadPool& pool_;
  std::vector<void*> allocs_;
  // Attention scores: one slot of (pos0+n) floats per head (parallel_for chunk = 1 head,
  // so slots are never shared). Sized by reserve(); attention() only grows it if called
  // beyond the reserved size, which is outside the no-allocation guarantee.
  std::vector<float> scratch_;
  // Q8_0-quantized activations for Q-weight matmul, written once by the calling thread; workers
  // only read it. Sized by reserve() to max_rows * max_width / 32 blocks; matmul() only grows it
  // if called beyond that, which is outside the no-allocation guarantee (like attention).
  std::vector<BlockQ8_0> xq_;
  // The same for Q5_1 (Q8_1 blocks, max_rows * max_width / 32) and the K quants (Q8_K blocks,
  // max_rows * ceil(max_width / 256)).
  std::vector<BlockQ8_1> xq1_;
  std::vector<BlockQ8_K> xqk_;
  uint64_t quantize_calls_ = 0;
  // xq_ interleaved 8 tokens per block for the prefill GEMM (n >= 4, SIMD builds); sized by reserve() to
  // ceil(max_rows / 8) * max_width / 32 blocks, same growth rule as xq_.
  std::vector<simd::BlockQ8_0x8> xp_;
};

}  // namespace nibble::cpu
