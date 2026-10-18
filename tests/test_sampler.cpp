#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "nibble/error.h"
#include "sampler.h"

using namespace nibble;

namespace {
std::vector<float> logits_of(std::initializer_list<double> p) {
  std::vector<float> v;
  for (double x : p) v.push_back(static_cast<float>(std::log(x)));
  return v;
}
}  // namespace

TEST_CASE("sampler: temperature 0 is argmax", "[sampler]") {
  Sampler s({0.0f, 0, 1.0f, 1}, 4);
  std::vector<float> l{1, 5, 5, 2};
  for (int i = 0; i < 20; ++i) REQUIRE(s.sample(l) == 1);  // lowest index on ties
}

TEST_CASE("sampler: top_k 1 is argmax for any temperature", "[sampler]") {
  for (float t : {0.1f, 1.0f, 5.0f}) {
    Sampler s({t, 1, 1.0f, 7}, 4);
    std::vector<float> l{1, 2, 9, 3};
    for (int i = 0; i < 50; ++i) REQUIRE(s.sample(l) == 2);
  }
}

TEST_CASE("sampler: same seed same sequence", "[sampler]") {
  std::vector<float> l{0, 1, 2, 3};
  Sampler a({1.0f, 0, 1.0f, 42}, 4), b({1.0f, 0, 1.0f, 42}, 4), c({1.0f, 0, 1.0f, 43}, 4);
  bool differs = false;
  for (int i = 0; i < 1000; ++i) {
    const int32_t x = a.sample(l);
    REQUIRE(x == b.sample(l));
    if (x != c.sample(l)) differs = true;
  }
  REQUIRE(differs);
}

TEST_CASE("sampler: distribution", "[sampler]") {
  std::vector<float> l{0.0f, static_cast<float>(std::log(2.0))};
  Sampler s({1.0f, 0, 1.0f, 12345}, 2);
  const int n = 200000;
  int ones = 0;
  for (int i = 0; i < n; ++i) ones += s.sample(l) == 1;
  const double f = static_cast<double>(ones) / n;
  INFO("freq token 1 = " << f);
  REQUIRE(std::abs(f - 2.0 / 3.0) < 0.005);
}

TEST_CASE("sampler: top_p cut", "[sampler]") {
  auto l = logits_of({0.6, 0.3, 0.1});
  {
    Sampler s({1.0f, 0, 0.5f, 3}, 3);
    for (int i = 0; i < 2000; ++i) REQUIRE(s.sample(l) == 0);
  }
  {
    Sampler s({1.0f, 0, 0.85f, 3}, 3);
    bool seen1 = false;
    for (int i = 0; i < 2000; ++i) {
      const int32_t t = s.sample(l);
      REQUIRE(t != 2);
      seen1 |= t == 1;
    }
    REQUIRE(seen1);
  }
}

TEST_CASE("sampler: parameter edges", "[sampler]") {
  auto l = logits_of({0.6, 0.3, 0.1});
  {
    Sampler big({1.0f, 1000000, 1.0f, 9}, 3), all({1.0f, 0, 1.0f, 9}, 3);
    for (int i = 0; i < 500; ++i) REQUIRE(big.sample(l) == all.sample(l));
  }
  {
    Sampler s({1.0f, 0, 0.0f, 9}, 3);
    for (int i = 0; i < 200; ++i) REQUIRE(s.sample(l) == 0);
  }
  REQUIRE_THROWS_WITH(Sampler({-0.5f, 0, 1.0f, 1}, 3),
                      Catch::Matchers::ContainsSubstring("-0.5"));
  REQUIRE_THROWS_AS(Sampler({std::nanf(""), 0, 1.0f, 1}, 3), Error);
  REQUIRE_THROWS_AS(Sampler({1.0f, 0, std::nanf(""), 1}, 3), Error);
  REQUIRE_THROWS_WITH(Sampler({1.0f, 0, -0.25f, 1}, 3),
                      Catch::Matchers::ContainsSubstring("-0.25"));
}
