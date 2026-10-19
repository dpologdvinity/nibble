#include <cmath>
#include <cstdint>
#include <limits>
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

TEST_CASE("sampler: NaN and inf logits", "[sampler]") {
  const float nan = std::nanf(""), inf = std::numeric_limits<float>::infinity();
  std::vector<float> m{nan, 1, nan, 3, 2, nan};
  Sampler g({0.0f, 0, 1.0f, 1}, 6), s({0.8f, 0, 1.0f, 1}, 6), k({0.8f, 2, 0.9f, 1}, 6);
  for (int i = 0; i < 2000; ++i) {
    REQUIRE(g.sample(m) == 3);
    const int32_t a = s.sample(m), b = k.sample(m);
    REQUIRE((a == 1 || a == 3 || a == 4));
    REQUIRE((b == 3 || b == 4));
  }
  std::vector<float> allnan(6, nan), allneg(6, -inf), one{0, 1, inf, 2, inf, 0};
  for (float t : {0.0f, 0.8f}) {
    Sampler x({t, 0, 1.0f, 1}, 6);
    REQUIRE_THROWS_WITH(x.sample(allnan), Catch::Matchers::ContainsSubstring("all logits are NaN"));
    REQUIRE_THROWS_WITH(x.sample(allneg), Catch::Matchers::ContainsSubstring("no finite logits"));
    for (int i = 0; i < 100; ++i) REQUIRE(x.sample(one) == 2);  // lowest-index +inf
  }
  std::vector<float> single{0, 1, 2, inf, 0, 1};
  Sampler y({1.0f, 3, 0.5f, 1}, 6);
  for (int i = 0; i < 100; ++i) REQUIRE(y.sample(single) == 3);
}

TEST_CASE("sampler: validation messages use shortest float form", "[sampler]") {
  using Catch::Matchers::Equals;
  CHECK_THROWS_WITH(Sampler({-0.5f, 0, 1.0f, 1}, 3), Equals("sampler: invalid temperature -0.5 (must be >= 0)"));
  CHECK_THROWS_WITH(Sampler({1.0f, 0, -0.25f, 1}, 3), Equals("sampler: invalid top_p -0.25 (must be >= 0)"));
  CHECK_THROWS_WITH(Sampler({std::nanf(""), 0, 1.0f, 1}, 3), Equals("sampler: invalid temperature nan (must be >= 0)"));
}

TEST_CASE("sampler: top-k tie goes to the lower index", "[sampler]") {
  // Four equal logits, k = 2: only indices 0 and 1 may ever be drawn.
  Sampler s({1.0f, 2, 1.0f, 5}, 4);
  std::vector<float> l{3, 3, 3, 3};
  for (int i = 0; i < 300; ++i) REQUIRE(s.sample(l) < 2);
}

TEST_CASE("sampler: logits size mismatch throws", "[sampler]") {
  Sampler s({1.0f, 0, 1.0f, 1}, 4);
  std::vector<float> l{1, 2, 3};
  CHECK_THROWS_WITH(s.sample(l), Catch::Matchers::Equals("sampler: got 3 logits, expected 4"));
}

TEST_CASE("sampler: top_k and top_p combine", "[sampler]") {
  // p = .5 .3 .1 .1; top_k 3 keeps .5 .3 .1 (renormalised to .556 .333 .111), then
  // top_p .8 keeps the first two. Index 2 and 3 are never drawn, 0 and 1 both are.
  auto l = logits_of({0.5, 0.3, 0.1, 0.1});
  Sampler s({1.0f, 3, 0.8f, 11}, 4);
  int seen[4] = {};
  for (int i = 0; i < 2000; ++i) ++seen[s.sample(l)];
  CHECK(seen[0] > 0);
  CHECK(seen[1] > 0);
  CHECK(seen[2] == 0);
  CHECK(seen[3] == 0);
}

TEST_CASE("sampler: pick_index falls back to the last token with probability", "[sampler]") {
  const double p[] = {0.5, 0.5, 0.0, 0.0};
  CHECK(pick_index(p, 4, 0.25) == 0);
  CHECK(pick_index(p, 4, 0.75) == 1);
  CHECK(pick_index(p, 4, 1.0) == 1);  // rounding: cumulative sum never exceeds u
  const double q[] = {0.0};
  CHECK(pick_index(q, 1, 1.0) == 0);
}
