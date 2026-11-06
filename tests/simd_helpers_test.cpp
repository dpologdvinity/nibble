// Tests for the intrinsics helpers in src/backend/cpu/simd/simd.h (horizontal sums, F16C loads, the int8
// multiply-accumulate and the vector exp). They take and return __m256 values, so unlike the kernel tests
// they cannot go through the Kernels table: CMake compiles this file once per SIMD tier, with the tier's
// instruction-set flags and NIBBLE_SIMD_TIER set, into one executable per tier (nibble_simd_helpers_<tier>),
// and simd_helpers_main.cpp skips the run when the CPU lacks the tier.
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>

#include "backend/cpu/simd/simd.h"
#include "fp16.h"

using namespace nibble;
namespace st = nibble::simd::NIBBLE_SIMD_TIER;

// The horizontal sums are exact for integer-valued inputs.
TEST_CASE("hsum_ps and hsum_epi32 are exact on integers", "[simd]") {
  alignas(32) float f[8];
  alignas(32) int32_t v[8];
  for (int i = 0; i < 8; ++i) {
    f[i] = float(i * 1000 + 1);
    v[i] = -i * 100000 + 7;
  }
  float ef = 0;
  int32_t ev = 0;
  for (int i = 0; i < 8; ++i) {
    ef += f[i];
    ev += v[i];
  }
  CHECK(st::hsum_ps(_mm256_load_ps(f)) == ef);
  CHECK(st::hsum_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(v))) == ev);
}

// The F16C conversion agrees with the software conversion for every half bit pattern.
TEST_CASE("load_f16x8 equals fp16_to_fp32 for all 65536 inputs", "[simd]") {
  for (uint32_t base = 0; base < 65536; base += 8) {
    uint16_t h[8];
    for (int i = 0; i < 8; ++i) h[i] = uint16_t(base + i);
    alignas(32) float out[8];
    _mm256_store_ps(out, st::load_f16x8(h));
    for (int i = 0; i < 8; ++i) {
      float ref = fp16_to_fp32(h[i]);
      if (std::isnan(ref))
        CHECK(std::isnan(out[i]));
      else
        CHECK(std::memcmp(&ref, &out[i], 4) == 0);
    }
  }
}

// The int8 multiply-accumulate primitive (and its AVX-VNNI variant) equals a plain integer dot, in the
// documented exact range (activation never -128 where the weight is negative).
TEST_CASE("mul_sum_i8 maddubs and dispatch variants agree with integer dot", "[simd][quant]") {
  std::mt19937 g(5);
  for (int rep = 0; rep < 2000; ++rep) {
    alignas(32) int8_t x[32], y[32];
    for (int i = 0; i < 32; ++i) {
      x[i] = int8_t(g());
      y[i] = int8_t(g());
      if (rep % 4 == 0) x[i] = i % 2 ? -128 : 127;  // weight extremes with y in [-127,127]
      if (rep % 4 == 1) y[i] = i % 3 ? 127 : -127;
      if (y[i] == -128 && x[i] < 0) y[i] = -127;  // outside the documented exact range
    }
    const __m256i vx = _mm256_load_si256(reinterpret_cast<const __m256i*>(x));
    const __m256i vy = _mm256_load_si256(reinterpret_cast<const __m256i*>(y));
    alignas(32) int32_t a[8], b[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(a), st::mul_sum_i8(vx, vy));
    _mm256_store_si256(reinterpret_cast<__m256i*>(b), st::mul_sum_i8_maddubs(vx, vy));
    for (int k = 0; k < 8; ++k) {
      int32_t want = 0;
      for (int i = 4 * k; i < 4 * k + 4; ++i) want += int32_t(x[i]) * y[i];
      REQUIRE(a[k] == want);
      REQUIRE(b[k] == want);
    }
  }
}

namespace {
// e^x of one value through the 8-lane vector function.
float exp1(float x) {
  alignas(32) float v[8];
  _mm256_store_ps(v, st::exp_ps(_mm256_set1_ps(x)));
  return v[0];
}
}  // namespace

// Dense sweep (2^24 evenly spaced points) plus every float in the bands where 2^n scaling is special:
// e^x subnormal (x < -87.34) and n = 127/128 up to ln(FLT_MAX) (x > 87.68; n = 128 from 88.38).
TEST_CASE("simd exp_ps max relative error vs std::exp <= 2e-7 over [-88, 88]", "[simd][ops]") {
  double worst = 0;
  float worst_x = 0;
  alignas(32) float xs[8], ys[8];
  int fill = 0;
  auto flush = [&] {
    _mm256_store_ps(ys, st::exp_ps(_mm256_load_ps(xs)));
    for (int i = 0; i < fill; ++i) {
      const double e = std::exp(double(xs[i]));
      const double rel = std::fabs(double(ys[i]) - e) / e;
      if (!(rel <= worst)) worst = rel, worst_x = xs[i];
    }
    fill = 0;
  };
  auto push = [&](float x) {
    xs[fill++] = x;
    if (fill == 8) flush();
  };
  const int64_t steps = int64_t{1} << 24;
  for (int64_t i = 0; i <= steps; ++i) push(float(-88.0 + 176.0 * double(i) / double(steps)));
  for (float x = -88.f; x <= -87.f; x = std::nextafter(x, 0.f)) push(x);
  for (float x = 87.f; x <= 88.72f; x = std::nextafter(x, 100.f)) push(x);
  flush();
  INFO("worst relative error " << worst << " at x=" << worst_x);
  CHECK(worst <= 2e-7);
}

// Special values: e^0 = 1, e^-inf = 0, e^inf = inf, NaN stays NaN, overflow gives inf, and underflow lands in
// the subnormal range within one ulp (never negative).
TEST_CASE("simd exp_ps edge inputs: underflow to 0 or subnormal, overflow to inf, no NaN", "[simd][ops]") {
  const float inf = std::numeric_limits<float>::infinity();
  CHECK(exp1(0.f) == 1.f);
  CHECK(exp1(-inf) == 0.f);
  CHECK(exp1(inf) == inf);
  CHECK(std::isnan(exp1(std::numeric_limits<float>::quiet_NaN())));
  CHECK(exp1(89.f) == inf);
  CHECK(exp1(1e30f) == inf);
  CHECK(std::isfinite(exp1(88.72f)));
  for (float x : {-87.5f, -88.5f, -95.f, -103.f, -104.f, -105.f, -200.f, -1e30f}) {
    const float y = exp1(x);
    INFO("x=" << x << " y=" << y);
    CHECK(y >= 0.f);
    CHECK(y < 1.2e-38f);
    CHECK(std::fabs(double(y) - std::exp(double(x))) <= 1.5e-45);  // within one subnormal ulp
  }
}
