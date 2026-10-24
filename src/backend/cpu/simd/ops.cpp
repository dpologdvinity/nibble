// AVX2 elementwise ops and attention. Scalar oracles: cpu::add, rmsnorm, silu_mul, attention_heads (ref_ops.cpp).
#include "backend/cpu/simd/simd.h"

#ifdef NIBBLE_HAVE_AVX2
#include <algorithm>
#include <cmath>
#include <limits>

#include "tensor.h"

namespace nibble::simd {
namespace {

// Horizontal sum of the 4 double lanes.
inline double hsum_pd(__m256d v) {
  __m128d s = _mm_add_pd(_mm256_castpd256_pd128(v), _mm256_extractf128_pd(v, 1));
  return _mm_cvtsd_f64(_mm_add_sd(s, _mm_unpackhi_pd(s, s)));
}

// x[i] * scale * w[i] in double, then rounded to float: the scalar rmsnorm's per-element math.
inline __m128 scale_pd(__m128 x, double scale, __m128 w) {
  return _mm256_cvtpd_ps(_mm256_mul_pd(_mm256_mul_pd(_mm256_cvtps_pd(x), _mm256_set1_pd(scale)), _mm256_cvtps_pd(w)));
}

// silu(g) * u for 8 lanes, using exp_ps for e^-g.
inline __m256 silu_mul8(__m256 g, __m256 u) {
  const __m256 e = exp_ps(_mm256_xor_ps(g, _mm256_set1_ps(-0.0f)));
  return _mm256_mul_ps(_mm256_div_ps(g, _mm256_add_ps(_mm256_set1_ps(1.0f), e)), u);
}

// x[i] = e^(x[i] - m), returns the sum; the last partial vector is padded with -inf (e^-inf = 0).
float exp_sub_sum(float* x, int64_t len, float m) {
  const __m256 mv = _mm256_set1_ps(m);
  __m256 acc = _mm256_setzero_ps();
  int64_t j = 0;
  for (; j + 8 <= len; j += 8) {
    const __m256 e = exp_ps(_mm256_sub_ps(_mm256_loadu_ps(x + j), mv));
    _mm256_storeu_ps(x + j, e);
    acc = _mm256_add_ps(acc, e);
  }
  if (j < len) {
    alignas(32) float buf[8];
    std::fill_n(buf, 8, -std::numeric_limits<float>::infinity());
    std::copy(x + j, x + len, buf);
    const __m256 e = exp_ps(_mm256_sub_ps(_mm256_load_ps(buf), mv));
    _mm256_store_ps(buf, e);
    std::copy(buf, buf + (len - j), x + j);
    acc = _mm256_add_ps(acc, e);
  }
  return hsum_ps(acc);
}

}  // namespace

// Exact elementwise add (bit-identical to the scalar).
void add(float* x, const float* y, int64_t count) {
  int64_t i = 0;
  for (; i + 8 <= count; i += 8) _mm256_storeu_ps(x + i, _mm256_add_ps(_mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i)));
  for (; i < count; ++i) x[i] += y[i];
}

// Sum of squares in vectorized double precision (only the summation order differs from the scalar), then
// per-element scaling in double exactly as the scalar does.
void rmsnorm(const float* x, const Tensor& weight, float* y, int64_t n, float eps) {
  const int64_t d = weight.ne[0];
  const float* w = static_cast<const float*>(weight.data);
  for (int64_t t = 0; t < n; ++t) {
    const float* xt = x + t * d;
    float* yt = y + t * d;
    // float * float is exact in double, so only the sum order differs from the scalar.
    __m256d a0 = _mm256_setzero_pd(), a1 = a0;
    int64_t i = 0;
    for (; i + 8 <= d; i += 8) {
      const __m256 v = _mm256_loadu_ps(xt + i);
      const __m256d lo = _mm256_cvtps_pd(_mm256_castps256_ps128(v)), hi = _mm256_cvtps_pd(_mm256_extractf128_ps(v, 1));
      a0 = _mm256_add_pd(a0, _mm256_mul_pd(lo, lo));
      a1 = _mm256_add_pd(a1, _mm256_mul_pd(hi, hi));
    }
    double ss = hsum_pd(_mm256_add_pd(a0, a1));
    for (; i < d; ++i) ss += static_cast<double>(xt[i]) * xt[i];
    const double scale = 1.0 / std::sqrt(ss / static_cast<double>(d) + static_cast<double>(eps));
    i = 0;
    for (; i + 4 <= d; i += 4) _mm_storeu_ps(yt + i, scale_pd(_mm_loadu_ps(xt + i), scale, _mm_loadu_ps(w + i)));
    for (; i < d; ++i) yt[i] = static_cast<float>(xt[i] * scale * w[i]);
  }
}

// Vector body plus a zero-padded final vector so tail elements take the same path as the rest.
void silu_mul(float* gate, const float* up, int64_t count) {
  int64_t i = 0;
  for (; i + 8 <= count; i += 8) _mm256_storeu_ps(gate + i, silu_mul8(_mm256_loadu_ps(gate + i), _mm256_loadu_ps(up + i)));
  if (i < count) {
    alignas(32) float g[8] = {}, u[8] = {};
    std::copy(gate + i, gate + count, g);
    std::copy(up + i, up + count, u);
    _mm256_store_ps(g, silu_mul8(_mm256_load_ps(g), _mm256_load_ps(u)));
    std::copy(g, g + (count - i), gate + i);
  }
}

// Same structure as the scalar attention: scores into scratch (vector dot per key), max, vectorized exp-sum
// and normalization, then the weighted sum of V rows with FMA.
void attention_heads(const float* q, const float* k, const float* v, float* out, int64_t n, int64_t pos0,
                     int64_t n_heads, int64_t n_kv_heads, int64_t head_dim, int64_t head_begin, int64_t head_end,
                     float* scratch) {
  const int64_t group = n_heads / n_kv_heads, vec = head_dim & ~int64_t{7};
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  for (int64_t h = head_begin; h < head_end; ++h) {
    const int64_t kvh = h / group;
    for (int64_t t = 0; t < n; ++t) {
      const float* qh = q + (t * n_heads + h) * head_dim;
      const int64_t len = pos0 + t + 1;
      float m = -std::numeric_limits<float>::infinity();
      for (int64_t j = 0; j < len; ++j) {
        const float* kj = k + (j * n_kv_heads + kvh) * head_dim;
        __m256 acc = _mm256_setzero_ps();
        int64_t d = 0;
        for (; d < vec; d += 8) acc = _mm256_fmadd_ps(_mm256_loadu_ps(qh + d), _mm256_loadu_ps(kj + d), acc);
        float dot = hsum_ps(acc);
        for (; d < head_dim; ++d) dot += qh[d] * kj[d];
        scratch[j] = dot * scale;
        m = std::max(m, scratch[j]);
      }
      const float inv = 1.0f / exp_sub_sum(scratch, len, m);
      float* o = out + (t * n_heads + h) * head_dim;
      std::fill_n(o, head_dim, 0.0f);
      for (int64_t j = 0; j < len; ++j) {
        const float* vj = v + (j * n_kv_heads + kvh) * head_dim;
        const float pj = scratch[j] * inv;
        const __m256 p = _mm256_set1_ps(pj);
        int64_t d = 0;
        for (; d < vec; d += 8) _mm256_storeu_ps(o + d, _mm256_fmadd_ps(p, _mm256_loadu_ps(vj + d), _mm256_loadu_ps(o + d)));
        for (; d < head_dim; ++d) o[d] += pj * vj[d];
      }
    }
  }
}

}  // namespace nibble::simd
#endif
