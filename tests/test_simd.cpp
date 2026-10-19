#include <algorithm>
#include <cmath>
#include <cstring>

#include "backend/cpu/simd/simd.h"
#include "fp16.h"
#include "simd_test_util.h"

using namespace nibble;

TEST_CASE("kernel_set names the active macros", "[simd]") {
#if defined(NIBBLE_HAVE_AVXVNNI)
  CHECK(std::strcmp(simd::kernel_set(), "avx2+fma+f16c+avxvnni") == 0);
#elif defined(NIBBLE_HAVE_AVX2)
  CHECK(std::strcmp(simd::kernel_set(), "avx2+fma+f16c") == 0);
#else
  CHECK(std::strcmp(simd::kernel_set(), "scalar") == 0);
#endif
}

TEST_CASE("compare_kernels detects mismatches and accepts equal kernels", "[simd]") {
  auto ident = [](const float* in, float* out, size_t n) { for (size_t i = 0; i < n; ++i) out[i] = in[i]; };
  test::compare_kernels(ident, ident, {0, 1, 7, 100}, {1, 2}, 0.f, 0.f);
  auto off = [](const float* in, float* out, size_t n) { for (size_t i = 0; i < n; ++i) out[i] = in[i] + 1e-3f; };
  test::compare_kernels(ident, off, {16}, {1}, 0.f, 1e-2f);  // within atol passes
}

#ifdef NIBBLE_HAVE_AVX2
TEST_CASE("hsum_ps and hsum_epi32 are exact on integers", "[simd]") {
  alignas(32) float f[8];
  alignas(32) int32_t v[8];
  for (int i = 0; i < 8; ++i) { f[i] = float(i * 1000 + 1); v[i] = -i * 100000 + 7; }
  float ef = 0; int32_t ev = 0;
  for (int i = 0; i < 8; ++i) { ef += f[i]; ev += v[i]; }
  CHECK(simd::hsum_ps(_mm256_load_ps(f)) == ef);
  CHECK(simd::hsum_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(v))) == ev);
}

TEST_CASE("load_f16x8 equals fp16_to_fp32 for all 65536 inputs", "[simd]") {
  for (uint32_t base = 0; base < 65536; base += 8) {
    uint16_t h[8];
    for (int i = 0; i < 8; ++i) h[i] = uint16_t(base + i);
    alignas(32) float out[8];
    _mm256_store_ps(out, simd::load_f16x8(h));
    for (int i = 0; i < 8; ++i) {
      float ref = fp16_to_fp32(h[i]);
      if (std::isnan(ref)) CHECK(std::isnan(out[i]));
      else CHECK(std::memcmp(&ref, &out[i], 4) == 0);
    }
  }
}
#endif

#ifdef NIBBLE_HAVE_AVX2
#include <vector>

#include "backend/cpu/ref_ops.h"
#include "tensor.h"

namespace {
// Compares simd::matmul_rows_f with the scalar oracle over [row_begin,row_end) of a rows x in weight.
// compare_kernels' buffers hold n = nt*max(in,rows) floats: x is the first nt*in, y the first nt*rows.
// x and W are deliberately placed at 4-byte (F32) / 2-byte (F16) offsets from allocation start.
void check_matmul(DType type, int64_t in, int64_t rows, int64_t nt, int64_t rb, int64_t re, float scale) {
  std::mt19937 rng(uint32_t(in * 131 + rows));
  std::uniform_real_distribution<float> d(-1.f, 1.f);
  std::vector<float> wf(size_t(rows * in) + 1);
  std::vector<uint16_t> wh(size_t(rows * in) + 1);
  for (size_t i = 0; i < wf.size(); ++i) { wf[i] = d(rng) * scale; wh[i] = fp32_to_fp16(wf[i]); }
  Tensor w;
  w.type = type;
  w.n_dims = 2;
  w.ne = {in, rows, 1, 1};
  w.data = type == DType::F32 ? static_cast<const void*>(wf.data() + 1) : static_cast<const void*>(wh.data() + 1);
  const size_t n = size_t(nt * std::max(in, rows));
  std::vector<float> xs(n + 1), y(size_t(nt * rows));
  auto run = [&](bool simd_path) {
    return [&, simd_path](const float* x, float* out, size_t) {
      std::copy_n(x, size_t(nt * in), xs.begin() + 1);
      std::fill(out, out + n, -7.f);
      std::fill(y.begin(), y.end(), -7.f);
      if (simd_path) simd::matmul_rows_f(w, xs.data() + 1, y.data(), nt, rb, re);
      else cpu::matmul_rows(w, xs.data() + 1, y.data(), nt, rb, re);
      std::copy(y.begin(), y.end(), out);
    };
  };
  test::compare_kernels(run(false), run(true), {n}, {1, 2}, 1e-5f,
                       1e-6f * std::sqrt(float(in)) * scale);  // atol: the oracle's own sequential-sum error under cancellation
}
}  // namespace

TEST_CASE("simd matmul_rows_f matches scalar for F32 and F16", "[simd]") {
  for (DType t : {DType::F32, DType::F16})
    for (int64_t in : {8, 13, 896, 4864})
      for (int64_t rows : {1, 3, 4, 37})
        for (int64_t nt : {1, 5}) {
          INFO("type=" << int(t) << " in=" << in << " rows=" << rows << " n=" << nt);
          check_matmul(t, in, rows, nt, 0, rows, 1.f);
        }
}

TEST_CASE("simd matmul_rows_f handles sub-ranges and extreme magnitudes", "[simd]") {
  for (DType t : {DType::F32, DType::F16}) {
    check_matmul(t, 70, 37, 2, 5, 22, 1.f);   // row range not 4-aligned
    check_matmul(t, 70, 37, 2, 36, 37, 1.f);  // single trailing row
    check_matmul(t, 70, 37, 1, 9, 9, 1.f);    // empty range
    check_matmul(t, 301, 9, 3, 0, 9, 300.f);  // large (F16 stays finite)
    check_matmul(t, 301, 9, 3, 0, 9, 1e-3f);  // small
  }
}

TEST_CASE("embed_row_f16 equals scalar conversion", "[simd]") {
  std::vector<uint16_t> h(41);
  for (size_t i = 0; i < h.size(); ++i) h[i] = uint16_t(i * 1597 + 3);
  for (int64_t d : {0, 1, 7, 8, 9, 40}) {
    std::vector<float> o(41, -1.f);
    simd::embed_row_f16(h.data() + 1, o.data(), d);
    for (int64_t k = 0; k < 40; ++k) {
      float ref = k < d ? fp16_to_fp32(h[size_t(k + 1)]) : -1.f;
      if (std::isnan(ref)) CHECK(std::isnan(o[size_t(k)]));
      else CHECK(o[size_t(k)] == ref);
    }
  }
}
#endif
