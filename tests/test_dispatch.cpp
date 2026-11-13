// Tests for runtime kernel selection (src/backend/cpu/simd/dispatch.cpp): which tier select_for() picks for a
// given set of CPU features, how NIBBLE_KERNELS overrides it, and the errors for values that cannot be honoured.
// select_for() takes the features as a parameter, so CPUs this machine is not can be covered too.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <string>

#include "backend/cpu/simd/dispatch.h"

using namespace nibble::simd;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::StartsWith;

namespace {
const CpuFeatures kNone{}, kAvx2{true, true, true, false}, kVnni{true, true, true, true};
// True when this build compiled the SIMD tiers (x86-64 with NIBBLE_SIMD=ON).
bool tiers_built() { return select_for(nullptr, kVnni) != nullptr; }
// True when this build also compiled the AVX-VNNI tier (CMakeLists.txt: NIBBLE_SIMD_TIER_VNNI).
constexpr bool kVnniBuilt =
#ifdef NIBBLE_SIMD_TIER_VNNI
    true;
#else
    false;
#endif
std::string name_of(const Kernels* k) { return k ? k->name : "scalar"; }
}  // namespace

TEST_CASE("select_for picks the strongest tier the CPU allows", "[dispatch]") {
  CHECK(select_for(nullptr, kNone) == nullptr);
  if (!tiers_built()) SKIP("no SIMD tiers in this build");
  CHECK(name_of(select_for(nullptr, kAvx2)) == "avx2+fma+f16c");
  CHECK(name_of(select_for("", kVnni)) == (kVnniBuilt ? "avx2+fma+f16c+avxvnni" : "avx2+fma+f16c"));
  CHECK(select_for(nullptr, CpuFeatures{true, false, true, false}) == nullptr);  // AVX2 without FMA
  CHECK(select_for(nullptr, CpuFeatures{true, true, false, false}) == nullptr);  // AVX2 without F16C
  CHECK(select_for(nullptr, CpuFeatures{false, true, true, true}) == nullptr);   // VNNI without AVX2
}

TEST_CASE("NIBBLE_KERNELS forces a tier by short or full name", "[dispatch]") {
  CHECK(select_for("scalar", kVnni) == nullptr);
  if (!tiers_built()) SKIP("no SIMD tiers in this build");
  CHECK(name_of(select_for("avx2", kVnni)) == "avx2+fma+f16c");
  CHECK(name_of(select_for("avx2+fma+f16c", kVnni)) == "avx2+fma+f16c");
}

// Its own case, so a build without the vnni tier reports only this one as skipped, not the avx2 half above.
TEST_CASE("NIBBLE_KERNELS forces the vnni tier by short or full name", "[dispatch]") {
  if (!kVnniBuilt) SKIP("no AVX-VNNI tier in this build");
  CHECK(name_of(select_for("avx2+vnni", kVnni)) == "avx2+fma+f16c+avxvnni");
  CHECK(name_of(select_for("avx2+fma+f16c+avxvnni", kVnni)) == "avx2+fma+f16c+avxvnni");
}

TEST_CASE("bad NIBBLE_KERNELS values throw with the valid choices", "[dispatch]") {
  CHECK_THROWS_WITH(select_for("avx512", kVnni),
                    StartsWith("NIBBLE_KERNELS=\"avx512\" is not one of: scalar, avx2, avx2+vnni"));
  CHECK_THROWS_WITH(select_for(" AVX2", kVnni), StartsWith("NIBBLE_KERNELS=\" AVX2\" is not one of"));
  if (!tiers_built()) {
    CHECK_THROWS_WITH(select_for("avx2", kVnni), ContainsSubstring("this build has no SIMD kernels"));
    // The list of choices says which of them this build cannot honour.
    CHECK_THROWS_WITH(select_for("avx512", kVnni), ContainsSubstring("(this build has no SIMD kernels)"));
    return;
  }
  if (kVnniBuilt) {
    CHECK_THROWS_WITH(select_for("avx512", kVnni), !ContainsSubstring("this build has no"));
  } else {
    CHECK_THROWS_WITH(select_for("avx512", kVnni), ContainsSubstring("(this build has no AVX-VNNI kernels)"));
  }
  CHECK_THROWS_WITH(select_for("avx2", kNone),
                    ContainsSubstring("needs a CPU with AVX2, FMA and F16C; this CPU supports: scalar"));
  if (!kVnniBuilt) return;
  CHECK_THROWS_WITH(select_for("avx2+vnni", kAvx2),
                    ContainsSubstring("needs a CPU with AVX-VNNI; this CPU supports: scalar, avx2"));
}

TEST_CASE("kernel_set reports the selected tier", "[dispatch]") {
  CHECK(std::string(kernel_set()) == name_of(select()));
  for (const Kernels* k : available()) CHECK(k != nullptr);
}

TEST_CASE("AVX-VNNI needs CPUID leaf 7 and bit 4 of leaf 7.1 EAX", "[dispatch]") {
  CHECK(avxvnni_from_cpuid(7, 1u << 4));
  CHECK(avxvnni_from_cpuid(0x20, 0xFFFFFFFFu));
  CHECK_FALSE(avxvnni_from_cpuid(6, 0xFFFFFFFFu));  // max leaf below 7: leaf 7 data is meaningless
  CHECK_FALSE(avxvnni_from_cpuid(0, 1u << 4));
  CHECK_FALSE(avxvnni_from_cpuid(7, ~(1u << 4)));
}

TEST_CASE("builds without the vnni tier never select it", "[dispatch]") {
#if defined(NIBBLE_SIMD_TIER_AVX2) && !defined(NIBBLE_SIMD_TIER_VNNI)
  CHECK(name_of(select_for(nullptr, kVnni)) == "avx2+fma+f16c");
  CHECK_THROWS_WITH(select_for("avx2+vnni", kVnni), ContainsSubstring("this build has no AVX-VNNI kernels"));
  for (const Kernels* k : available()) CHECK(std::string(k->name) == "avx2+fma+f16c");
#else
  SKIP("this build has the vnni tier (or no tiers)");
#endif
}
