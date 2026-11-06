#pragma once
// Comparison harness for the elementwise SIMD kernel tests: runs a reference kernel and a vector kernel over
// a grid of sizes and seeds and fails with the first mismatching index, and for_each_tier, which runs a kernel
// test once per SIMD tier the CPU supports.
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "backend/cpu/simd/dispatch.h"

namespace nibble::test {

// Runs ref_fn and simd_fn(in, out, n) over every size x seed with uniform [-1,1) input; each writes n floats to out.
// Fails on the first |a-b| > atol + rtol*|a|, reporting index, both values and size.
template <class RefFn, class SimdFn>
void compare_kernels(RefFn ref_fn, SimdFn simd_fn, const std::vector<size_t>& sizes, const std::vector<uint32_t>& seeds,
                     float rtol, float atol) {
  for (size_t n : sizes)
    for (uint32_t seed : seeds) {
      std::mt19937 rng(seed);
      std::uniform_real_distribution<float> d(-1.f, 1.f);
      std::vector<float> in(n), a(n), b(n);
      for (float& x : in) x = d(rng);
      ref_fn(in.data(), a.data(), n);
      simd_fn(in.data(), b.data(), n);
      for (size_t i = 0; i < n; ++i)
        if (!(std::fabs(a[i] - b[i]) <= atol + rtol * std::fabs(a[i]))) {
          FAIL("mismatch at i=" << i << " ref=" << a[i] << " simd=" << b[i] << " n=" << n << " seed=" << seed);
        }
    }
}

// Runs fn(k) once for every SIMD kernel table this CPU can run (simd::available(), weakest first), with the
// tier's name in the failure context. Skips the test case when there is none (no tier built, or no AVX2).
template <class Fn>
void for_each_tier(Fn fn) {
  const std::vector<const simd::Kernels*> tiers = simd::available();
  if (tiers.empty()) SKIP("no SIMD tier on this CPU/build");
  for (const simd::Kernels* k : tiers) {
    INFO("tier " << k->name);
    fn(*k);
  }
}

}  // namespace nibble::test
