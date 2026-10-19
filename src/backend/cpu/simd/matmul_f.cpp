// AVX2+FMA+F16C F32/F16 matmul and F16 embed. Scalar oracle: cpu::matmul_rows in ref_ops.cpp.
#include "backend/cpu/simd/simd.h"

#ifdef NIBBLE_HAVE_AVX2
#include "fp16.h"
#include "tensor.h"

namespace nibble::simd {
namespace {

struct LoadF32 {
  using T = float;
  static __m256 load(const float* p) { return _mm256_loadu_ps(p); }
  static float one(float v) { return v; }
};
struct LoadF16 {
  using T = uint16_t;
  static __m256 load(const uint16_t* p) { return load_f16x8(p); }
  static float one(uint16_t v) { return fp16_to_fp32(v); }
};

// y[t*out + r] for r in [rb, re): 4 rows share each x load; 8-wide FMA accumulators per row.
template <class L>
void matmul_t(const typename L::T* w, int64_t in, int64_t out, const float* x, float* y, int64_t n, int64_t rb,
              int64_t re) {
  const int64_t vec = in & ~int64_t{7};
  for (int64_t t = 0; t < n; ++t) {
    const float* xt = x + t * in;
    float* yt = y + t * out;
    int64_t r = rb;
    for (; r + 4 <= re; r += 4) {
      const typename L::T* w0 = w + r * in;
      const typename L::T* w1 = w0 + in;
      const typename L::T* w2 = w1 + in;
      const typename L::T* w3 = w2 + in;
      __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
      int64_t k = 0;
      for (; k < vec; k += 8) {
        const __m256 xv = _mm256_loadu_ps(xt + k);
        a0 = _mm256_fmadd_ps(xv, L::load(w0 + k), a0);
        a1 = _mm256_fmadd_ps(xv, L::load(w1 + k), a1);
        a2 = _mm256_fmadd_ps(xv, L::load(w2 + k), a2);
        a3 = _mm256_fmadd_ps(xv, L::load(w3 + k), a3);
      }
      float s0 = hsum_ps(a0), s1 = hsum_ps(a1), s2 = hsum_ps(a2), s3 = hsum_ps(a3);
      for (; k < in; ++k) {
        s0 += xt[k] * L::one(w0[k]);
        s1 += xt[k] * L::one(w1[k]);
        s2 += xt[k] * L::one(w2[k]);
        s3 += xt[k] * L::one(w3[k]);
      }
      yt[r] = s0; yt[r + 1] = s1; yt[r + 2] = s2; yt[r + 3] = s3;
    }
    for (; r < re; ++r) {
      const typename L::T* w0 = w + r * in;
      __m256 a = _mm256_setzero_ps();
      int64_t k = 0;
      for (; k < vec; k += 8) a = _mm256_fmadd_ps(_mm256_loadu_ps(xt + k), L::load(w0 + k), a);
      float s = hsum_ps(a);
      for (; k < in; ++k) s += xt[k] * L::one(w0[k]);
      yt[r] = s;
    }
  }
}

}  // namespace

void matmul_rows_f(const Tensor& w, const float* x, float* y, int64_t n, int64_t row_begin, int64_t row_end) {
  const int64_t in = w.ne[0], out = w.rows();
  if (w.type == DType::F32)
    matmul_t<LoadF32>(static_cast<const float*>(w.data), in, out, x, y, n, row_begin, row_end);
  else
    matmul_t<LoadF16>(static_cast<const uint16_t*>(w.data), in, out, x, y, n, row_begin, row_end);
}

void embed_row_f16(const uint16_t* row, float* out, int64_t d) {
  int64_t k = 0;
  for (; k + 8 <= d; k += 8) _mm256_storeu_ps(out + k, load_f16x8(row + k));
  for (; k < d; ++k) out[k] = fp16_to_fp32(row[k]);
}

}  // namespace nibble::simd
#endif
