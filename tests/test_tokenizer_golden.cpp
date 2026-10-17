#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "corpus.h"
#include "gguf/gguf.h"
#include "tokenizer/tokenizer.h"

static std::string join(const std::vector<int32_t>& v) {
  std::string s;
  for (auto x : v) s += std::to_string(x) + " ";
  return s;
}

TEST_CASE("tokenizer matches llama.cpp golden", "[e2e][tokenizer]") {
  const std::string root = NIBBLE_SOURCE_DIR;
  const std::string model = root + "/models/qwen2.5-0.5b-instruct-fp16.gguf";
  const std::string golden = root + "/models/golden/qwen2.5-0.5b-f16/tokenize.txt";
  if (!std::filesystem::exists(model) || !std::filesystem::exists(golden)) SKIP("model or golden missing");
  auto file = nibble::gguf::File::open(model);
  auto tok = nibble::tok::Tokenizer::from_gguf(file);
  auto corpus = read_corpus(root + "/tests/tokenizer/corpus.txt");
  std::ifstream gf(golden);
  std::string line;
  size_t n = 0;
  for (size_t i = 0; i < corpus.size(); ++i) {
    REQUIRE(std::getline(gf, line));
    std::vector<int32_t> want;
    std::istringstream ls(line);
    for (int32_t x; ls >> x;) want.push_back(x);
    auto got = tok.encode(corpus[i], false);
    std::string pieces, cat;
    for (auto id : got) {
      pieces += "[" + tok.decode_token(id) + "]";
      cat += tok.decode_token(id);
    }
    INFO("entry " << i << "\nwant: " << join(want) << "\ngot:  " << join(got) << "\npieces: " << pieces);
    CHECK(got == want);
    CHECK(cat == corpus[i]);
    ++n;
  }
  CHECK(n == 37);
}
