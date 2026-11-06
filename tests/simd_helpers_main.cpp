// Entry point of nibble_simd_helpers_<tier> (simd_helpers_test.cpp compiled for one SIMD tier). Compiled for
// the baseline target, so a CPU without the tier never reaches tier code: main checks the CPU first and exits
// with 77, which CMake registers as SKIP_RETURN_CODE, before Catch runs any test.
#include <catch2/catch_session.hpp>
#include <cstring>

#include "backend/cpu/simd/dispatch.h"

#ifndef NIBBLE_HELPERS_TIER
#error "NIBBLE_HELPERS_TIER must name the tier (avx2 or vnni)"
#endif

int main(int argc, char* argv[]) {
  const nibble::simd::CpuFeatures f = nibble::simd::cpu_features();
  const bool avx2 = f.avx2 && f.fma && f.f16c;
  const bool vnni = std::strcmp(NIBBLE_HELPERS_TIER, "vnni") == 0;
  if (!avx2 || (vnni && !f.avxvnni)) return 77;  // this CPU cannot run the tier: skip
  return Catch::Session().run(argc, argv);
}
