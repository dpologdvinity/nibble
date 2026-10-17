#include <algorithm>
#include <filesystem>

#include <catch2/catch_test_macros.hpp>

#include "golden_file.h"

TEST_CASE("golden_file reader: capital.bin is self-consistent", "[e2e][golden_file]") {
  const std::string path = std::string(NIBBLE_SOURCE_DIR) + "/models/golden/qwen2.5-0.5b-f16/capital.bin";
  if (!std::filesystem::exists(path)) SKIP("golden data missing: " + path);
  Golden g = Golden::load(path);
  REQUIRE(g.n_vocab == 151936);
  REQUIRE(g.n_gen == 64);
  REQUIRE(g.tokens.size() == size_t(g.n_prompt) + g.n_gen);
  for (size_t i = g.n_prompt - 1; i + 1 < g.tokens.size(); ++i) {
    auto r = g.row(i);
    auto it = std::max_element(r.begin(), r.end());
    REQUIRE(g.tokens[i + 1] == int32_t(it - r.begin()));
  }
}

TEST_CASE("golden_file corpus splitting rule", "[golden_file]") {
  auto v = split_corpus("a\n====\nb\n\n====\nc\n");
  REQUIRE(v.size() == 3);
  REQUIRE(v[0] == "a");
  REQUIRE(v[1] == "b\n");
  REQUIRE(v[2] == "c\n");
}
