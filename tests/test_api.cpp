#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <filesystem>
#include <fstream>

#include "nibble/nibble.h"
#include "tiny_model.h"
#include "tokenizer/unicode.h"

using namespace nibble;

namespace {
struct TmpModel {
  std::filesystem::path path;
  explicit TmpModel(const std::vector<uint8_t>& bytes) {
    static std::atomic<int> n{0};
    path = std::filesystem::temp_directory_path() /
           ("nibble_api_" + std::to_string(::getpid()) + "_" + std::to_string(n++) + ".gguf");
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(bytes.data()),
                                                static_cast<std::streamsize>(bytes.size()));
  }
  ~TmpModel() { std::filesystem::remove(path); }
};
const char* kPrompt = "abcdefghij";
}  // namespace

TEST_CASE("api: max tokens", "[api]") {
  TmpModel f(tiny_model_gguf(1));
  Model m = Model::load(f.path.string(), {.n_threads = 2});
  Session s(m);
  CHECK(s.generate(kPrompt, {.max_tokens = 3}, [](std::string_view) { return true; }) == StopReason::MaxTokens);
  CHECK(s.n_generated() == 3);
  CHECK(s.n_prompt_tokens() == 10);
  CHECK(m.architecture() == "qwen2");
}

TEST_CASE("api: cancel", "[api]") {
  TmpModel f(tiny_model_gguf(1));
  Model m = Model::load(f.path.string(), {.n_threads = 2});
  Session s(m);
  int calls = 0;
  auto r = s.generate(kPrompt, {}, [&](std::string_view) { ++calls; return false; });
  CHECK(r == StopReason::Cancelled);
  CHECK(calls == 1);
}

TEST_CASE("api: context full", "[api]") {
  TmpModel f(tiny_model_gguf(1));
  Model m = Model::load(f.path.string(), {.n_ctx = 8, .n_threads = 2});
  Session s(m);
  CHECK(s.generate("abc", {.max_tokens = 100}, [](std::string_view) { return true; }) == StopReason::ContextFull);
  CHECK(s.n_prompt_tokens() + s.n_generated() <= 8);
}

TEST_CASE("api: errors", "[api]") {
  TmpModel f(tiny_model_gguf(1));
  Model m = Model::load(f.path.string(), {.n_ctx = 4, .n_threads = 2});
  Session s(m);
  auto cb = [](std::string_view) { return true; };
  CHECK_THROWS_AS(s.generate(kPrompt, {}, cb), Error);
  CHECK_THROWS_AS(s.generate("", {}, cb), Error);
  CHECK_THROWS_AS(Model::load("/nonexistent/x.gguf"), Error);
}

TEST_CASE("api: eos", "[api]") {
  TmpModel f(tiny_model_gguf(1, DType::F32, true, 0));
  Model m = Model::load(f.path.string(), {.n_threads = 2});
  Session s(m);
  int calls = 0;
  CHECK(s.generate(kPrompt, {}, [&](std::string_view) { ++calls; return true; }) == StopReason::Eos);
  CHECK(calls == 0);
  CHECK(s.n_generated() == 0);
}

TEST_CASE("api: deterministic and valid utf8", "[api]") {
  TmpModel f(tiny_model_gguf(1));
  Model m = Model::load(f.path.string(), {.n_threads = 2});
  Session s(m);
  auto run = [&] {
    std::string out;
    s.generate(kPrompt, {.max_tokens = 20}, [&](std::string_view p) {
      CHECK_NOTHROW(tok::utf8_decode(p));
      out += p;
      return true;
    });
    return out;
  };
  CHECK(run() == run());
}

TEST_CASE("api: model is movable", "[api]") {
  TmpModel f(tiny_model_gguf(1));
  Model a = Model::load(f.path.string(), {.n_threads = 2});
  Model b = std::move(a);
  Session s(b);
  CHECK(s.generate("abc", {.max_tokens = 2}, [](std::string_view) { return true; }) == StopReason::MaxTokens);
}

TEST_CASE("api: vocab size mismatch rejected at load", "[api]") {
  for (int64_t d : {int64_t{1}, int64_t{-1}}) {
    TinyModelOpts mo;
    mo.vocab_delta = d;
    TmpModel f(tiny_model_gguf(1, nibble::DType::F32, false, 259, mo));
    CHECK_THROWS_WITH(Model::load(f.path.string(), {.n_threads = 1}),
                      Catch::Matchers::ContainsSubstring("tokenizer has 261 tokens but token_embd.weight has " +
                                                         std::to_string(261 + d) + " rows"));
  }
}

TEST_CASE("api: huge n_ctx throws Error", "[api]") {
#ifdef __SANITIZE_ADDRESS__  // ASan aborts on multi-TB allocations instead of failing them
  SKIP("multi-TB allocation aborts under ASan");
#endif
  TmpModel f(tiny_model_gguf(1));
  Model m = Model::load(f.path.string(), {.n_ctx = 1000000000000LL, .n_threads = 1});
  CHECK_THROWS_WITH(Session(m), Catch::Matchers::ContainsSubstring("n_ctx"));
}
