#include "backend/cpu/cpu_backend.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <cstring>
#include <new>

#include "backend/cpu/ref_ops.h"
#include "backend/cpu/simd/simd.h"
#include "debug_check.h"
#include "nibble/error.h"

namespace nibble::cpu {

CpuBackend::CpuBackend(ThreadPool& pool) : pool_(pool) {}

void CpuBackend::reserve(const ReserveParams& rp) {
  const int64_t n_ctx = rp.n_ctx, n_heads = rp.n_heads;
  uint64_t need, xq_need;
  if (rp.max_rows < 0 || rp.max_width < 0 ||
      __builtin_mul_overflow(static_cast<uint64_t>(rp.max_rows), static_cast<uint64_t>(rp.max_width) / kQK, &xq_need) ||
      xq_need > std::numeric_limits<size_t>::max() / sizeof(BlockQ8_0))
    throw Error("reserve: activation scratch for max_rows=" + std::to_string(rp.max_rows) +
                " max_width=" + std::to_string(rp.max_width) + " is too large");
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
#ifdef NIBBLE_HAVE_AVX2
  const uint64_t xp_need = (static_cast<uint64_t>(rp.max_rows) + 7) / 8 * (static_cast<uint64_t>(rp.max_width) / kQK);
  if (xp_.size() < xp_need) {
    try {
      xp_.resize(static_cast<size_t>(xp_need));
    } catch (const std::exception&) {
      throw Error("reserve: cannot allocate " + std::to_string(xp_need * sizeof(simd::BlockQ8_0x8)) +
                  " bytes of packed activation scratch");
    }
  }
#endif
  if (xq_.size() < xq_need) {
    try {
      xq_.resize(static_cast<size_t>(xq_need));
    } catch (const std::exception&) {
      throw Error("reserve: cannot allocate " + std::to_string(xq_need * sizeof(BlockQ8_0)) +
                  " bytes of activation scratch");
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
  NIBBLE_DCHECK(host_dst && src && n >= 0, "read: dst=%p src=%p n=%lld", static_cast<void*>(host_dst),
                static_cast<const void*>(src), static_cast<long long>(n));
  std::memcpy(host_dst, src, static_cast<size_t>(n) * sizeof(float));
}

void CpuBackend::embed(const Tensor& table, std::span<const int32_t> ids, float* out) {
  NIBBLE_DCHECK(out && table.data && !ids.empty() && supports(table.type), "embed: out=%p table=%p n=%zu dtype %u",
                static_cast<void*>(out), table.data, ids.size(), static_cast<unsigned>(table.type));
#ifndef NDEBUG
  for (size_t i = 0; i < ids.size(); ++i)
    NIBBLE_DCHECK(ids[i] >= 0 && ids[i] < table.ne[1], "embed: ids[%zu]=%d out of range [0, %lld)", i,
                  static_cast<int>(ids[i]), static_cast<long long>(table.ne[1]));
#endif
#ifdef NIBBLE_HAVE_AVX2
  if (table.type == DType::F16) {
    const int64_t d = table.ne[0];
    for (size_t t = 0; t < ids.size(); ++t) {
      if (ids[t] < 0 || ids[t] >= table.rows()) throw Error("embed: token id " + std::to_string(ids[t]) + " out of range (n_vocab " + std::to_string(table.rows()) + ")");
      simd::embed_row_f16(static_cast<const uint16_t*>(table.data) + ids[t] * d, out + static_cast<int64_t>(t) * d, d);
    }
    return;
  }
#endif
  cpu::embed(table, ids, out);
}

void CpuBackend::matmul(const Tensor& w, const float* x, float* y, int64_t n) {
  NIBBLE_DCHECK(x && y && w.data && n > 0, "matmul: x=%p y=%p w=%p n=%lld", static_cast<const void*>(x),
                static_cast<void*>(y), w.data, static_cast<long long>(n));
  // Unsupported dtype throws here on the calling thread (pool tasks must not throw);
  // Task 9 gates on supports() at load time so the hot path never sees one.
  if (!supports(w.type)) cpu::matmul_rows(w, x, y, 0, 0, 0, nullptr);  // n=0: only the dtype check runs
  const BlockQ8_0* xq = nullptr;
  if (w.type == DType::Q8_0 || w.type == DType::Q4_0) {
    NIBBLE_DCHECK(w.ne[0] % kQK == 0, "matmul: quantized weight ne0=%lld not a multiple of 32",
                  static_cast<long long>(w.ne[0]));
    const size_t need = static_cast<size_t>(n * (w.ne[0] / kQK));
    if (xq_.size() < need) xq_.resize(need);  // beyond reserve(): outside the no-allocation guarantee
#ifdef NIBBLE_HAVE_AVX2
    const int64_t nb = w.ne[0] / kQK;
    for (int64_t t = 0; t < n; ++t) simd::quantize_row_q8_0(x + t * w.ne[0], xq_.data() + t * nb, w.ne[0]);
#else
    cpu::quantize_rows_q8_0(x, xq_.data(), n, w.ne[0]);
#endif
    xq = xq_.data();
  }
#ifdef NIBBLE_HAVE_AVX2
  if (w.type == DType::F32 || w.type == DType::F16) {
    pool_.parallel_for(w.rows(), 16, [&](int64_t b, int64_t e) {
      n >= 4 ? simd::gemm_f(w, x, y, n, b, e) : simd::matmul_rows_f(w, x, y, n, b, e);
    });
    return;
  }
  if (xq && n >= 4) {
    const int64_t nb = w.ne[0] / kQK;
    const size_t need = static_cast<size_t>((n + 7) / 8 * nb);
    if (xp_.size() < need) xp_.resize(need);  // beyond reserve(), like xq_
    simd::pack_q8_0_x8(xq, n, nb, xp_.data());
    const simd::BlockQ8_0x8* xp = xp_.data();
    pool_.parallel_for(w.rows(), 16, [&](int64_t b, int64_t e) { simd::gemm_q(w, xp, y, n, b, e); });
    return;
  }
  if (xq) {
    pool_.parallel_for(w.rows(), 16, [&](int64_t b, int64_t e) { simd::matmul_rows_q(w, xq, y, n, b, e); });
    return;
  }
#endif
  pool_.parallel_for(w.rows(), 16, [&](int64_t b, int64_t e) { cpu::matmul_rows(w, x, y, n, b, e, xq); });
}

void CpuBackend::rmsnorm(const float* x, const Tensor& weight, float* y, int64_t n, float eps) {
  NIBBLE_DCHECK(x && y && weight.data && n > 0 && weight.type == DType::F32,
                "rmsnorm: x=%p y=%p weight=%p n=%lld weight dtype %u (need F32)", static_cast<const void*>(x),
                static_cast<void*>(y), weight.data, static_cast<long long>(n), static_cast<unsigned>(weight.type));
  cpu::rmsnorm(x, weight, y, n, eps);
}
void CpuBackend::add(float* x, const float* y, int64_t count) {
  NIBBLE_DCHECK(x && y && count > 0, "add: x=%p y=%p count=%lld", static_cast<void*>(x),
                static_cast<const void*>(y), static_cast<long long>(count));
  cpu::add(x, y, count);
}
void CpuBackend::add_bias(float* x, const Tensor& bias, int64_t n) {
  NIBBLE_DCHECK(x && bias.data && n > 0 && bias.type == DType::F32,
                "add_bias: x=%p bias=%p n=%lld bias dtype %u (need F32)", static_cast<void*>(x), bias.data,
                static_cast<long long>(n), static_cast<unsigned>(bias.type));
  cpu::add_bias(x, bias, n);
}
void CpuBackend::rope(float* x, int64_t n, int64_t n_heads, int64_t head_dim, int64_t pos0, const RopeParams& p) {
  NIBBLE_DCHECK(x && n > 0 && n_heads > 0 && head_dim > 0 && head_dim % 2 == 0 && pos0 >= 0,
                "rope: x=%p n=%lld n_heads=%lld head_dim=%lld (must be even) pos0=%lld", static_cast<void*>(x),
                static_cast<long long>(n), static_cast<long long>(n_heads), static_cast<long long>(head_dim),
                static_cast<long long>(pos0));
  cpu::rope(x, n, n_heads, head_dim, pos0, p);
}
void CpuBackend::copy(float* dst, const float* src, int64_t count) {
  NIBBLE_DCHECK(dst && src && count >= 0, "copy: dst=%p src=%p count=%lld", static_cast<void*>(dst),
                static_cast<const void*>(src), static_cast<long long>(count));
  std::memmove(dst, src, static_cast<size_t>(count) * sizeof(float));
}
void CpuBackend::silu_mul(float* gate, const float* up, int64_t count) {
  NIBBLE_DCHECK(gate && up && count > 0, "silu_mul: gate=%p up=%p count=%lld", static_cast<void*>(gate),
                static_cast<const void*>(up), static_cast<long long>(count));
  cpu::silu_mul(gate, up, count);
}

void CpuBackend::attention(const float* q, const float* k, const float* v, float* out, int64_t n, int64_t pos0,
                           int64_t n_heads, int64_t n_kv_heads, int64_t head_dim) {
  NIBBLE_DCHECK(q && k && v && out && n > 0 && pos0 >= 0 && n_heads > 0 && n_kv_heads > 0 &&
                    n_heads % n_kv_heads == 0 && head_dim > 0,
                "attention: q=%p k=%p v=%p out=%p n=%lld pos0=%lld n_heads=%lld n_kv_heads=%lld head_dim=%lld "
                "(n_heads must be a multiple of n_kv_heads)",
                static_cast<const void*>(q), static_cast<const void*>(k), static_cast<const void*>(v),
                static_cast<void*>(out), static_cast<long long>(n), static_cast<long long>(pos0),
                static_cast<long long>(n_heads), static_cast<long long>(n_kv_heads), static_cast<long long>(head_dim));
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
