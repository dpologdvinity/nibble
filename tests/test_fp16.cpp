#include <catch2/catch_test_macros.hpp>
#include <cmath>

#include "fp16.h"

using nibble::fp16_to_fp32;
using nibble::fp32_to_fp16;

TEST_CASE("fp16 known values", "[fp16]") {
  REQUIRE(fp16_to_fp32(0x3C00) == 1.0f);
  REQUIRE(fp16_to_fp32(0xC000) == -2.0f);
  REQUIRE(fp16_to_fp32(0x7BFF) == 65504.0f);
  REQUIRE(fp16_to_fp32(0x0001) == 0x1p-24f);
  REQUIRE(std::isinf(fp16_to_fp32(0x7C00)));
  REQUIRE(fp16_to_fp32(0x7C00) > 0);
  REQUIRE(std::isnan(fp16_to_fp32(0x7E00)));
}

TEST_CASE("fp16 roundtrip all non-NaN", "[fp16]") {
  for (uint32_t h = 0; h < 65536; ++h) {
    float f = fp16_to_fp32(static_cast<uint16_t>(h));
    if (std::isnan(f)) {
      REQUIRE(std::isnan(fp16_to_fp32(fp32_to_fp16(f))));
      continue;
    }
    REQUIRE(fp32_to_fp16(f) == h);
  }
}

TEST_CASE("fp32_to_fp16 rounding", "[fp16]") {
  REQUIRE(fp32_to_fp16(1.0f + 0x1p-11f) == 0x3C00);  // tie -> even
  REQUIRE(fp32_to_fp16(1.0f + 3 * 0x1p-11f) == 0x3C02);  // tie -> even (up)
  REQUIRE(fp32_to_fp16(1e10f) == 0x7C00);
  REQUIRE(fp32_to_fp16(-1e-10f) == 0x8000);
}
