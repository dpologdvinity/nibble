#pragma once
#include <cstdint>
#include <span>

#include "tensor.h"

namespace nibble {

struct RopeParams {
  float freq_base;
  bool neox;
  const float* freq_factors;  // nullptr or head_dim/2 floats (host)
};

// Activations are allocated and touched only through the backend (a CUDA backend
// will hand out device pointers), so model code never dereferences them.
//
// Raw activation pointers carry no shape. Backend ops can only debug-check what
// their arguments describe (tensor dtypes, counts, head ratios, non-null pointers,
// embed ids in range); activation sizes and weight-vs-activation shape agreement
// are debug-checked at the call site (Transformer), which knows the buffer widths.
// Debug checks (NIBBLE_DCHECK, debug_check.h) abort; they compile out in Release.
class Backend {
 public:
  virtual ~Backend() = default;
  virtual bool supports(DType weight_type) const = 0;
  // Called once by the model before generation. Afterwards, ops with pos0 + n <= n_ctx and
  // n_heads <= the reserved head count must not allocate.
  virtual void reserve(int64_t n_ctx, int64_t n_heads) = 0;
  // Throws nibble::Error if n_floats < 0 or the allocation fails. Memory is zeroed.
  virtual float* alloc(int64_t n_floats) = 0;
  // Releases a block from alloc(). free(nullptr) is a no-op; freeing a pointer this
  // backend did not allocate (or freeing twice) is a precondition violation. Blocks not
  // freed are released when the backend is destroyed.
  virtual void free(float* p) = 0;
  virtual void read(float* host_dst, const float* src, int64_t n) = 0;
  virtual void embed(const Tensor& table, std::span<const int32_t> ids, float* out) = 0;  // out[n, ne0]
  virtual void matmul(const Tensor& w, const float* x, float* y, int64_t n) = 0;  // y[n, w.ne1] = x[n, w.ne0] . W^T
  virtual void rmsnorm(const float* x, const Tensor& weight, float* y, int64_t n, float eps) = 0;  // weight F32 [dim]
  virtual void add(float* x, const float* y, int64_t count) = 0;                   // x += y
  virtual void add_bias(float* x, const Tensor& bias, int64_t n) = 0;              // each of n rows += bias (F32)
  virtual void rope(float* x, int64_t n, int64_t n_heads, int64_t head_dim, int64_t pos0, const RopeParams&) = 0;
  virtual void copy(float* dst, const float* src, int64_t count) = 0;
  // q,out: [n, n_heads, head_dim]; k,v: [n_ctx, n_kv_heads, head_dim]; row t has position pos0+t and attends to 0..pos0+t
  virtual void attention(const float* q, const float* k, const float* v, float* out, int64_t n, int64_t pos0,
                         int64_t n_heads, int64_t n_kv_heads, int64_t head_dim) = 0;
  virtual void silu_mul(float* gate, const float* up, int64_t count) = 0;  // gate = silu(gate) * up
};

}  // namespace nibble
