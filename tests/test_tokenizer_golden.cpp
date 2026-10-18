#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>

#include "corpus.h"
#include "gguf/gguf.h"
#include "nibble/error.h"
#include "tokenizer/tokenizer.h"

static std::string join(const std::vector<int32_t>& v) {
  std::string s;
  for (auto x : v) s += std::to_string(x) + " ";
  return s;
}

static void check_golden(const std::string& model_file, const std::string& golden_dir) {
  const std::string root = NIBBLE_SOURCE_DIR;
  const std::string model = root + "/models/" + model_file;
  const std::string golden = root + "/models/golden/" + golden_dir + "/tokenize.txt";
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
    // SPM adds one leading space (add_space_prefix) that decode cannot tell from text.
    const bool prefixed = tok.add_space_prefix() && !corpus[i].empty();
    CHECK(cat == (prefixed ? " " + corpus[i] : corpus[i]));
    ++n;
  }
  CHECK(n == 37);
}

TEST_CASE("tokenizer matches llama.cpp golden", "[e2e][tokenizer]") {
  check_golden("qwen2.5-0.5b-instruct-fp16.gguf", "qwen2.5-0.5b-f16");
}

TEST_CASE("llama 3.2 tokenizer matches llama.cpp golden", "[e2e][tokenizer]") {
  check_golden("llama-3.2-1b-instruct-f16.gguf", "llama32-1b-f16");
}

TEST_CASE("tinyllama spm tokenizer matches llama.cpp golden", "[e2e][tokenizer]") {
  check_golden("tinyllama-1.1b-chat-f16.gguf", "tinyllama-1.1b-f16");
}

TEST_CASE("is_eog matches llama.cpp eog golden", "[e2e][tokenizer][eog]") {
  struct M { const char *file, *dir; };
  const M models[] = {{"qwen2.5-0.5b-instruct-fp16.gguf", "qwen2.5-0.5b-f16"},
                      {"qwen2.5-0.5b-instruct-q8_0.gguf", "qwen2.5-0.5b-q8_0"},
                      {"qwen2.5-0.5b-instruct-q4_0.gguf", "qwen2.5-0.5b-q4_0"},
                      {"llama-3.2-1b-instruct-f16.gguf", "llama32-1b-f16"},
                      {"tinyllama-1.1b-chat-f16.gguf", "tinyllama-1.1b-f16"}};
  const std::string root = NIBBLE_SOURCE_DIR;
  int ran = 0;
  for (const M& m : models) {
    const std::string gguf = root + "/models/" + m.file, gold = root + "/models/golden/" + m.dir + "/eog.txt";
    if (!std::filesystem::exists(gguf) || !std::filesystem::exists(gold)) continue;
    auto file = nibble::gguf::File::open(gguf);
    std::optional<nibble::tok::Tokenizer> tok;
    tok.emplace(nibble::tok::Tokenizer::from_gguf(file));  // a throwing tokenizer fails the test
    std::ifstream gf(gold);
    std::vector<int32_t> want, got;
    for (int32_t x; gf >> x;) want.push_back(x);
    for (int32_t id = 0; id < tok->n_vocab(); ++id) if (tok->is_eog(id)) got.push_back(id);
    std::string msg;
    for (auto id : want) msg += " " + std::to_string(id) + "='" + tok->decode_token(id) + "'";
    std::string msg2;
    for (auto id : got) msg2 += " " + std::to_string(id) + "='" + tok->decode_token(id) + "'";
    INFO(m.dir << "\nllama.cpp:" << msg << "\nnibble:  " << msg2);
    CHECK(got == want);
    ++ran;
  }
  int expected = 0;  // every model with both files present must have been compared
  for (const M& m : models)
    expected += std::filesystem::exists(root + "/models/" + m.file) &&
                std::filesystem::exists(root + "/models/golden/" + m.dir + "/eog.txt");
  CHECK(ran == expected);
  if (ran == 0) SKIP("no model with eog golden");
}
