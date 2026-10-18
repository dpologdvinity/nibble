#include <catch2/catch_test_macros.hpp>
#include <string>
#include <vector>

#include "corpus.h"
#include "nibble/error.h"
#include "tokenizer/pretokenize.h"

using nibble::tok::pretokenize;
using V = std::vector<std::string>;

static V split(const std::string& s, int n = 1) {
  V out;
  for (auto sv : pretokenize(s, n)) out.emplace_back(sv);
  return out;
}

TEST_CASE("pretokenize rules", "[pretokenize]") {
  CHECK(split("Hello world") == V{"Hello", " world"});
  CHECK(split("I'm 12345") == V{"I", "'m", " ", "1", "2", "3", "4", "5"});
  CHECK(split("I'm 12345", 3) == V{"I", "'m", " ", "123", "45"});
  CHECK(split("12345", 3) == V{"123", "45"});
  CHECK(split("1234567 x", 3) == V{"123", "456", "7", " x"});
  CHECK(split("DON'T") == V{"DON", "'T"});
  CHECK(split("a\n\n  b") == V{"a", "\n\n", " ", " b"});
  CHECK(split("trailing   ") == V{"trailing", "   "});
  CHECK(split("x!!\n") == V{"x", "!!\n"});
  CHECK(split("  hi") == V{" ", " hi"});
  CHECK(split("(abc)") == V{"(abc", ")"});
  CHECK(split("").empty());
}

TEST_CASE("pretokenize concatenates to input on corpus", "[pretokenize]") {
  auto entries = read_corpus(std::string(NIBBLE_SOURCE_DIR) + "/tests/tokenizer/corpus.txt");
  REQUIRE(!entries.empty());
  for (const auto& e : entries) {
    for (int n : {1, 3}) {
      std::string joined;
      for (auto sv : pretokenize(e, n)) {
        CHECK(!sv.empty());
        joined.append(sv);
      }
      CHECK(joined == e);
    }
  }
}

TEST_CASE("pretokenize rejects bad digit run", "[pretokenize]") {
  CHECK_THROWS_AS(pretokenize("1", 0), nibble::Error);
}
