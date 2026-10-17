#include "backend/cpu/cpu_backend.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <cstring>
#include <new>

#include "backend/cpu/ref_ops.h"
#include "nibble/error.h"

namespace nibble::cpu {

CpuBackend::CpuBackend(ThreadPool& pool) : pool_(pool) {}

void CpuBackend::reserve(int64_t n_ctx, int64_t n_heads) {
  uint64_t need;
  if (n_ctx < 0 || n_heads < 0 || __builtin_mul_overflow(static_cast<uint64_t>(n_ctx), static_cast<uint64_t>(n_heads), &need) ||
      need > std::numeric_limits<size_t>::max() / sizeof(float))
    throw Error("reserve: attention scratch for n_ctx=" + std::to_string(n_ctx) + " n_heads=" + std::to_string(n_heads) +
                " is too large");
  if (scratch_.size() < need) {
    try {
      scratch_.resize(static_cast<size_t>(need));
    } catch (const std::exception&) {
      throw Error("reserve: cannot allocate " + std::to_string(need * sizeof(float)) + " bytes of attention scratch");
    }
  }
}

CpuBackend::~CpuBackend() {
  for (void* p : allocs_) std::free(p);
}

float* CpuBackend::alloc(int64_t n_floats) {
  if (n_floats < 0) throw Error("alloc: negative size " + std::to_string(n_floats) + " floats");
  uint64_t bytes;
  if (__builtin_mul_overflow(static_cast<uint64_t>(n_floats), uint64_t{sizeof(float)}, &bytes) ||
      __builtin_add_overflow(bytes, uint64_t{63}, &bytes))
    throw Error("alloc: size of " + std::to_string(n_floats) + " floats overflows");
  bytes &= ~uint64_t{63};
  if (bytes == 0) bytes = 64;
  void* p = std::aligned_alloc(64, static_cast<size_t>(bytes));
  if (!p) throw Error("alloc: cannot allocate " + std::to_string(bytes) + " bytes");
  std::memset(p, 0, static_cast<size_t>(bytes));
  try {
    allocs_.push_back(p);
  } catch (const std::exception&) {
    std::free(p);
    throw Error("alloc: out of memory tracking allocation of " + std::to_string(bytes) + " bytes");
  }
  return static_cast<float*>(p);
}

void CpuBackend::free(float* p) {
  if (!p) return;
  auto it = std::find(allocs_.begin(), allocs_.end(), static_cast<void*>(p));
  if (it == allocs_.end()) return;  // precondition violation; ignore rather than double free
  std::free(*it);
  allocs_.erase(it);
}

void CpuBackend::read(float* host_dst, const float* src, int64_t n) {
  std::memcpy(host_dst, src, static_cast<size_t>(n) * sizeof(float));
}

void CpuBackend::embed(const Tensor& table, std::span<const int32_t> ids, float* out) {
  cpu::embed(table, ids, out);
}

void CpuBackend::matmul(const Tensor& w, const float* x, float* y, int64_t n) {
  // Unsupported dtype throws here on the calling thread (pool tasks must not throw);
  // Task 9 gates on supports() at load time so the hot path never sees one.
  if (!supports(w.type)) cpu::matmul_rows(w, x, y, 0, 0, 0);  // n=0: only the dtype check runs
  pool_.parallel_for(w.rows(), 16, [&](int64_t b, int64_t e) { cpu::matmul_rows(w, x, y, n, b, e); });
}

void CpuBackend::rmsnorm(const float* x, const Tensor& weight, float* y, int64_t n, float eps) {
  cpu::rmsnorm(x, weight, y, n, eps);
}
void CpuBackend::add(float* x, const float* y, int64_t count) { cpu::add(x, y, count); }
void CpuBackend::add_bias(float* x, const Tensor& bias, int64_t n) { cpu::add_bias(x, bias, n); }
void CpuBackend::rope(float* x, int64_t n, int64_t n_heads, int64_t head_dim, int64_t pos0, const RopeParams& p) {
  cpu::rope(x, n, n_heads, head_dim, pos0, p);
}
void CpuBackend::copy(float* dst, const float* src, int64_t count) {
  std::memmove(dst, src, static_cast<size_t>(count) * sizeof(float));
}
void CpuBackend::silu_mul(float* gate, const float* up, int64_t count) { cpu::silu_mul(gate, up, count); }

void CpuBackend::attention(const float* q, const float* k, const float* v, float* out, int64_t n, int64_t pos0,
                           int64_t n_heads, int64_t n_kv_heads, int64_t head_dim) {
  const int64_t ctx = pos0 + n;
  const size_t need = static_cast<size_t>(ctx * n_heads);
  if (scratch_.size() < need) scratch_.resize(need);
  float* scratch = scratch_.data();
  pool_.parallel_for(n_heads, 1, [&](int64_t b, int64_t e) {
    for (int64_t h = b; h < e; ++h)
      cpu::attention_heads(q, k, v, out, n, pos0, n_heads, n_kv_heads, head_dim, h, h + 1, scratch + h * ctx);
  });
}

}  // namespace nibble::cpu
