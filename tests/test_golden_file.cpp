// Tests for the golden-file reader and corpus splitter that the end-to-end tests are built on
// (tests/golden_file.h, tests/corpus.h).
#include <algorithm>
#include <filesystem>

#include <catch2/catch_test_macros.hpp>

#include "golden_file.h"

// Loads a real llama.cpp golden file (skipped if the models are not downloaded) and checks that its header is
// sane and that the recorded generation is greedy: each token is the argmax of the previous logits row.
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

// The corpus separator rule: entries are divided by a line that is exactly "===="; the newline before the
// separator belongs to the separator, and the last entry runs verbatim to the end of the file.
TEST_CASE("golden_file corpus splitting rule", "[golden_file]") {
  auto v = split_corpus("a\n====\nb\n\n====\nc\n");
  REQUIRE(v.size() == 3);
  REQUIRE(v[0] == "a");
  REQUIRE(v[1] == "b\n");
  REQUIRE(v[2] == "c\n");
}
