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
