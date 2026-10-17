#pragma once
#include <memory>
#include <vector>

#include "backend/backend.h"
#include "thread_pool.h"

namespace nibble::cpu {

class CpuBackend final : public Backend {
 public:
  explicit CpuBackend(ThreadPool& pool);
  ~CpuBackend() override;
  CpuBackend(const CpuBackend&) = delete;
  CpuBackend& operator=(const CpuBackend&) = delete;

  bool supports(DType t) const override { return t == DType::F32 || t == DType::F16; }
  void reserve(int64_t n_ctx, int64_t n_heads) override;
  size_t scratch_capacity() const { return scratch_.size(); }  // test hook
  float* alloc(int64_t n_floats) override;
  void read(float* host_dst, const float* src, int64_t n) override;
  void embed(const Tensor& table, std::span<const int32_t> ids, float* out) override;
  void matmul(const Tensor& w, const float* x, float* y, int64_t n) override;
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
};

}  // namespace nibble::cpu
