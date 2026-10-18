#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <filesystem>
#include <fstream>

#include "nibble/nibble.h"
#include "tiny_model.h"
#include "tokenizer/unicode.h"

using namespace nibble;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::MessageMatches;

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
  CHECK(s.generate(kPrompt, {.max_tokens = 3, .temperature = 0}, [](std::string_view) { return true; }) == StopReason::MaxTokens);
  CHECK(s.n_generated() == 3);
  CHECK(s.n_prompt_tokens() == 10);
  CHECK(m.architecture() == "qwen2");
}

TEST_CASE("api: cancel", "[api]") {
  TmpModel f(tiny_model_gguf(1));
  Model m = Model::load(f.path.string(), {.n_threads = 2});
  Session s(m);
  int calls = 0;
  auto r = s.generate(kPrompt, {.temperature = 0}, [&](std::string_view) { ++calls; return false; });
  CHECK(r == StopReason::Cancelled);
  CHECK(calls == 1);
}

TEST_CASE("api: context full", "[api]") {
  TmpModel f(tiny_model_gguf(1));
  Model m = Model::load(f.path.string(), {.n_ctx = 8, .n_threads = 2});
  Session s(m);
  CHECK(s.generate("abc", {.max_tokens = 100, .temperature = 0}, [](std::string_view) { return true; }) == StopReason::ContextFull);
  CHECK(s.n_prompt_tokens() + s.n_generated() == 8);  // ContextFull fires exactly at n_ctx
}

TEST_CASE("api: errors", "[api]") {
  TmpModel f(tiny_model_gguf(1));
  Model m = Model::load(f.path.string(), {.n_ctx = 4, .n_threads = 2});
  Session s(m);
  auto cb = [](std::string_view) { return true; };
  CHECK_THROWS_AS(s.generate(kPrompt, {}, cb), Error);
  CHECK_THROWS_MATCHES(s.generate("", {}, cb), Error, MessageMatches(ContainsSubstring("empty prompt")));
  CHECK_THROWS_AS(Model::load("/nonexistent/x.gguf"), Error);
}

TEST_CASE("api: eos", "[api]") {
  TmpModel f(tiny_model_gguf(1, DType::F32, true, 0));
  Model m = Model::load(f.path.string(), {.n_threads = 2});
  Session s(m);
  int calls = 0;
  CHECK(s.generate(kPrompt, {.temperature = 0}, [&](std::string_view) { ++calls; return true; }) == StopReason::Eos);
  CHECK(calls == 0);
  CHECK(s.n_generated() == 0);
}

TEST_CASE("api: deterministic and valid utf8", "[api]") {
  TmpModel f(tiny_model_gguf(1));
  Model m = Model::load(f.path.string(), {.n_threads = 2});
  Session s(m);
  auto run = [&] {
    std::string out;
    int pieces = 0;
    s.generate(kPrompt, {.max_tokens = 20, .temperature = 0}, [&](std::string_view p) {
      CHECK_NOTHROW(tok::utf8_decode(p));
      out += p;
      ++pieces;
      return true;
    });
    CHECK(pieces >= 1);
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

TEST_CASE("api: moved-from model throws", "[api]") {
  TmpModel f(tiny_model_gguf(1));
  Model a = Model::load(f.path.string(), {.n_threads = 1});
  Model b = std::move(a);
  const auto moved = MessageMatches(ContainsSubstring("use of moved-from Model"));
  CHECK_THROWS_MATCHES(a.architecture(), Error, moved);
  CHECK_THROWS_MATCHES(a.n_ctx(), Error, moved);
  CHECK_THROWS_MATCHES(Session(a), Error, moved);
  CHECK(b.architecture() == "qwen2");
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
  CHECK_THROWS_WITH(Model::load(f.path.string(), {.n_ctx = 1000000000000LL, .n_threads = 1}),
                    ContainsSubstring("n_ctx"));
}

namespace {
auto keep_going = [](std::string_view) { return true; };

std::string run_text(Session& s, std::span<const PromptPart> parts, const GenParams& gp, StopReason* r = nullptr) {
  std::string out;
  StopReason got = s.generate(parts, gp, [&](std::string_view p) { out += p; return true; });
  if (r) *r = got;
  return out;
}
}  // namespace

TEST_CASE("api: greedy with temperature 0 ignores the seed", "[api]") {
  TmpModel f(tiny_model_gguf(1));
  Model m = Model::load(f.path.string(), {.n_threads = 2});
  Session s(m);
  const PromptPart p[] = {{kPrompt, true}};
  const std::string a = run_text(s, p, {.max_tokens = 12, .temperature = 0, .seed = 1});
  const std::string b = run_text(s, p, {.max_tokens = 12, .temperature = 0, .seed = 2});
  const std::string c = run_text(s, p, {.max_tokens = 12, .temperature = 0});
  CHECK(a == b);
  CHECK(a == c);
}

TEST_CASE("api: seeded sampling is reproducible", "[api]") {
  TmpModel f(tiny_model_gguf(1));
  Model m = Model::load(f.path.string(), {.n_threads = 2});
  Session s(m);
  const PromptPart p[] = {{kPrompt, true}};
  const GenParams gp{.max_tokens = 24, .temperature = 5.0f, .top_k = 0, .top_p = 1.0f, .seed = 42};
  const std::string a = run_text(s, p, gp);
  CHECK(a == run_text(s, p, gp));
  GenParams other = gp;
  other.seed = 43;
  CHECK(a != run_text(s, p, other));
  // A random seed runs (output not checked).
  GenParams rnd = gp;
  rnd.seed.reset();
  CHECK_NOTHROW(run_text(s, p, rnd));
}

TEST_CASE("api: default sampling params", "[api]") {
  GenParams gp;
  CHECK(gp.temperature == 0.8f);
  CHECK(gp.top_k == 40);
  CHECK(gp.top_p == 0.95f);
  CHECK_FALSE(gp.seed.has_value());
  CHECK_FALSE(gp.append);
}

TEST_CASE("api: append continues", "[api]") {
  TmpModel f(tiny_model_gguf(1));
  Model m = Model::load(f.path.string(), {.n_threads = 2});
  const GenParams one{.max_tokens = 1, .temperature = 0};
  const GenParams more{.max_tokens = 6, .temperature = 0};
  GenParams more_append = more;
  more_append.append = true;

  Session fresh(m);
  const PromptPart abcd[] = {{"abcd", true}};
  StopReason r_fresh;
  const std::string want = run_text(fresh, abcd, more, &r_fresh);
  const int64_t want_past = fresh.n_past();
  const int64_t want_gen = fresh.n_generated();
  fresh.generate(abcd, {.max_tokens = 0}, keep_going);  // max_tokens 0 leaves the cache alone
  CHECK(fresh.n_past() == want_past);

  Session s(m);
  const PromptPart ab[] = {{"ab", true}};
  const PromptPart cd[] = {{"cd", true}};
  run_text(s, ab, one);
  CHECK(s.n_past() == 2);  // the token that ended the call is not in the cache
  StopReason r;
  const std::string got = run_text(s, cd, more_append, &r);
  CHECK(s.n_prompt_tokens() == 2);
  CHECK(s.n_past() == want_past);
  CHECK(s.n_generated() == want_gen);
  CHECK(r == r_fresh);
  CHECK(got == want);
}

TEST_CASE("api: append with no new tokens continues from the cache", "[api]") {
  TmpModel f(tiny_model_gguf(1));
  Model m = Model::load(f.path.string(), {.n_threads = 2});
  Session s(m);
  const PromptPart abc[] = {{"abc", true}};
  const PromptPart empty[] = {{"", true}};
  const std::string first = run_text(s, abc, {.max_tokens = 1, .temperature = 0});
  const std::string again = run_text(s, std::span<const PromptPart>(), {.max_tokens = 1, .temperature = 0, .append = true});
  CHECK(again == first);  // same logits, greedy: same token again
  CHECK(s.n_past() == 3);
  CHECK(run_text(s, empty, {.max_tokens = 1, .temperature = 0, .append = true}) == first);
  CHECK_THROWS_WITH(s.generate(empty, {.temperature = 0}, keep_going), ContainsSubstring("empty prompt"));
  CHECK_THROWS_WITH(s.generate(std::span<const PromptPart>(), {.temperature = 0}, keep_going),
                    ContainsSubstring("empty prompt"));
  Session s2(m);  // nothing cached: appending nothing is still an empty prompt
  CHECK_THROWS_WITH(s2.generate(empty, {.temperature = 0, .append = true}, keep_going),
                    ContainsSubstring("empty prompt"));
}

TEST_CASE("api: turn longer than remaining context", "[api]") {
  TmpModel f(tiny_model_gguf(1));
  Model m = Model::load(f.path.string(), {.n_ctx = 8, .n_threads = 2});
  Session s(m);
  const PromptPart abc[] = {{"abc", true}};
  run_text(s, abc, {.max_tokens = 1, .temperature = 0});
  REQUIRE(s.n_past() == 3);
  int calls = 0;
  auto count = [&](std::string_view) { ++calls; return true; };
  const PromptPart five[] = {{"abcde", true}};  // 3 + 5 >= 8
  CHECK(s.generate(five, {.temperature = 0, .append = true}, count) == StopReason::ContextFull);
  CHECK(s.n_past() == 3);
  CHECK(calls == 0);
  CHECK(s.n_prompt_tokens() == 0);
  CHECK(s.n_generated() == 0);
  const PromptPart four[] = {{"abcd", true}};  // 3 + 4 < 8 fits
  const StopReason r4 = s.generate(four, {.max_tokens = 5, .temperature = 0, .append = true}, keep_going);
  CHECK(s.n_prompt_tokens() == 4);
  CHECK(s.n_past() == 7);
  if (r4 != StopReason::Eos) CHECK(r4 == StopReason::ContextFull);  // 7 cached + 1 sampled == n_ctx
  // Fresh parts prompt with n_ctx tokens: ContextFull, no throw, cache untouched.
  const PromptPart eight[] = {{"abcdefgh", true}};
  CHECK(s.generate(eight, {.temperature = 0}, count) == StopReason::ContextFull);
  CHECK(s.n_past() == 7);
  CHECK(calls == 0);
  // The string overload keeps throwing on a fresh prompt that does not fit.
  CHECK_THROWS_AS(s.generate("abcdefgh", {.temperature = 0}, count), Error);
}

TEST_CASE("api: cancel does not flush", "[api]") {
  // Sample near-uniformly over many seeds; cancelling on the first piece must
  // produce exactly one callback even when the UTF-8 streamer holds bytes. The
  // same seed run to max_tokens = n_generated shows whether bytes were held
  // (the MaxTokens flush then adds a second callback).
  TmpModel f(tiny_model_gguf(1));
  Model m = Model::load(f.path.string(), {.n_threads = 1});
  Session s(m);
  const PromptPart p[] = {{"a", true}};
  int held = 0;
  for (uint64_t seed = 0; seed < 400; ++seed) {
    GenParams gp{.max_tokens = 50, .temperature = 50.0f, .top_k = 0, .top_p = 1.0f, .seed = seed};
    int calls = 0;
    const StopReason r = s.generate(p, gp, [&](std::string_view) { ++calls; return false; });
    if (r != StopReason::Cancelled) continue;
    CHECK(calls == 1);
    gp.max_tokens = static_cast<int>(s.n_generated());
    int calls2 = 0;
    s.generate(p, gp, [&](std::string_view) { ++calls2; return true; });
    if (calls2 == 2) ++held;
  }
  CHECK(held > 0);
}

TEST_CASE("api: max_tokens 0 skips prefill", "[api]") {
  TmpModel f(tiny_model_gguf(1));
  Model m = Model::load(f.path.string(), {.n_threads = 2});
  Session s(m);
  int calls = 0;
  const PromptPart p[] = {{kPrompt, true}};
  CHECK(s.generate(p, {.max_tokens = 0}, [&](std::string_view) { ++calls; return true; }) == StopReason::MaxTokens);
  CHECK(s.generate(kPrompt, {.max_tokens = -1}, [&](std::string_view) { ++calls; return true; }) ==
        StopReason::MaxTokens);
  CHECK(s.n_past() == 0);
  CHECK(s.n_generated() == 0);
  CHECK(calls == 0);
}

TEST_CASE("api: prompt parts control special-token parsing", "[api]") {
  TmpModel f(tiny_model_gguf(1));
  Model m = Model::load(f.path.string(), {.n_threads = 2});
  Session s(m);
  const PromptPart special[] = {{"<|ctl|>", true}};
  const PromptPart plain[] = {{"<|ctl|>", false}};
  s.generate(special, {.max_tokens = 1, .temperature = 0}, keep_going);
  CHECK(s.n_prompt_tokens() == 1);  // token 259
  s.generate(plain, {.max_tokens = 1, .temperature = 0}, keep_going);
  CHECK(s.n_prompt_tokens() == 7);  // one byte token per character
  const PromptPart mixed[] = {{"ab", false}, {"<|ctl|>", true}, {"<|ctl|>", false}};
  s.generate(mixed, {.max_tokens = 1, .temperature = 0}, keep_going);
  CHECK(s.n_prompt_tokens() == 2 + 1 + 7);
  s.generate("<|ctl|>", {.max_tokens = 1, .temperature = 0}, keep_going);  // string overload: special
  CHECK(s.n_prompt_tokens() == 1);
}

TEST_CASE("api: session reuse across calls", "[api]") {
  TmpModel f(tiny_model_gguf(1));
  Model m = Model::load(f.path.string(), {.n_threads = 2});
  const GenParams gp{.max_tokens = 8, .temperature = 0};
  Session reused(m);
  const PromptPart a[] = {{"hello world", true}};
  const PromptPart b[] = {{kPrompt, true}};
  run_text(reused, a, gp);
  const std::string got = run_text(reused, b, gp);
  const int64_t past = reused.n_past();
  Session fresh(m);
  CHECK(run_text(fresh, b, gp) == got);
  CHECK(fresh.n_past() == past);
}

TEST_CASE("api: model params", "[api]") {
  TmpModel f(tiny_model_gguf(1));
  CHECK(Model::load(f.path.string(), {.n_threads = 1}).n_ctx() == 64);  // min(4096, context_length 64)
  CHECK(Model::load(f.path.string(), {.n_ctx = 64, .n_threads = 1}).n_ctx() == 64);
  CHECK_THROWS_MATCHES(Model::load(f.path.string(), {.n_ctx = 65, .n_threads = 1}), Error,
                       MessageMatches(ContainsSubstring("n_ctx 65 exceeds the model's context_length 64")));
  // Rejected before the file is touched: a missing path still reports the bad param.
  CHECK_THROWS_WITH(Model::load("/nonexistent/x.gguf", {.n_ctx = -1}), ContainsSubstring("n_ctx"));
  CHECK_THROWS_WITH(Model::load("/nonexistent/x.gguf", {.n_threads = -1}), ContainsSubstring("n_threads"));
}

TEST_CASE("api: vocab check runs before the tokenizer is built", "[api]") {
  TinyModelOpts mo;
  mo.vocab_delta = 1;
  mo.tok.merges = {"bad"};  // would fail tokenizer construction
  TmpModel f(tiny_model_gguf(1, nibble::DType::F32, false, 259, mo));
  CHECK_THROWS_WITH(Model::load(f.path.string(), {.n_threads = 1}), ContainsSubstring("tokenizer has 261 tokens"));
}

TEST_CASE("api: load errors escape control bytes", "[api]") {
  auto msg_of = [](const TinyModelOpts& mo) {
    TmpModel f(tiny_model_gguf(1, nibble::DType::F32, false, 259, mo));
    try {
      (void)Model::load(f.path.string(), {.n_threads = 1});
    } catch (const Error& e) {
      return std::string(e.what());
    }
    return std::string();
  };
  TinyModelOpts arch;
  arch.arch = "q\x1b[31m";
  TinyModelOpts tokm;
  tokm.tok.model = "g\x1b[31m";
  TinyModelOpts merge;
  merge.tok.merges = {"\x1b[31m"};
  for (const auto& mo : {arch, tokm, merge}) {
    const std::string msg = msg_of(mo);
    CHECK(msg.find("\\x1b") != std::string::npos);
    CHECK(msg.find('\x1b') == std::string::npos);
  }
}
