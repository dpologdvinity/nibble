#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <thread>

#include "backend/cpu/cpu_backend.h"
#include "golden_file.h"
#include "gguf/gguf.h"
#include "model/transformer.h"
#include "thread_pool.h"
#include "tokenizer/tokenizer.h"

using namespace nibble;

namespace {
struct Engine {
  ThreadPool pool{int(std::max(1u, std::thread::hardware_concurrency()))};
  cpu::CpuBackend be{pool};
  gguf::File file;
  tok::Tokenizer tokenizer;
  model::LlamaConfig cfg;
  model::Weights w;
  explicit Engine(const std::string& path)
      : file(gguf::File::open(path)),
        tokenizer(tok::Tokenizer::from_gguf(file)),
        cfg(model::LlamaConfig::from_gguf(file)),
        w(model::Weights::bind(file, cfg, be)) {}
};

// One engine at a time: switching models frees the previous one first.
Engine& engine(const std::string& path) {
  static std::unique_ptr<Engine> e;
  static std::string loaded;
  if (!e || loaded != path) {
    e.reset();
    e = std::make_unique<Engine>(path);
    loaded = path;
  }
  return *e;
}

double cosine(std::span<const float> a, const float* b) {
  double ab = 0, aa = 0, bb = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    ab += double(a[i]) * b[i];
    aa += double(a[i]) * a[i];
    bb += double(b[i]) * b[i];
  }
  return ab / (std::sqrt(aa) * std::sqrt(bb));
}

size_t argmax(std::span<const float> r) {
  size_t m = 0;
  for (size_t i = 1; i < r.size(); ++i)
    if (r[i] > r[m]) m = i;
  return m;
}

// Near-tie rule (spec 8.3, amended by user decision 2026-10-17): a position is a near-tie
// if the golden top-2 gap OR nibble's own top-2 gap is < 1e-3. llama.cpp rounds activations
// to F16, which can push a true near-tie over the limit: at capital pos 50 the naive double
// oracle agrees with nibble (gap -0.00088) while golden shows +0.00264.
// gap between top-1 and top-2 logits
double top_gap(std::span<const float> r) {
  size_t m = argmax(r);
  double second = -1e300;
  for (size_t i = 0; i < r.size(); ++i)
    if (i != m) second = std::max(second, double(r[i]));
  return double(r[m]) - second;
}

// Per-model gate. F16 is held to the strict gate. Q8_0/Q4_0 use a looser one by user decision
// (2026-10-17): activation requantization amplifies rounding-level differences, and the measured
// three-way noise floor (nibble vs generic llama.cpp vs an fp64 oracle with Q8_0 requantization)
// is about 0.9966 cosine, so 0.999 per position and exact argmax are not attainable. Quantized
// goldens come from the generic llama.cpp build with an F32 KV cache (see tools/make_golden.sh).
struct Gate {
  double min_cos;
  double min_agree;     // required fraction of positions whose argmax agrees (or near-tie)
  bool strict_argmax;   // every non-tie position must agree
  bool assert_greedy;   // greedy divergence must be a near-tie; else only WARN
};
Gate gate_for(const std::string& model) {
  if (model == "f16") return {0.999, 1.0, true, true};
  return {0.995, 0.90, false, false};
}

void run_prompt(const std::string& model, const std::string& name) {
  const Gate gate = gate_for(model);
  const std::string root = NIBBLE_SOURCE_DIR;
  const std::string model_path = root + "/models/qwen2.5-0.5b-instruct-" + (model == "f16" ? "fp16" : model) + ".gguf";
  const std::string golden_path = root + "/models/golden/qwen2.5-0.5b-" + model + "/" + name + ".bin";
  if (!std::filesystem::exists(model_path) || !std::filesystem::exists(golden_path)) SKIP("model or golden missing");

  Golden g = Golden::load(golden_path);
  REQUIRE(g.n_gen == 64);
  REQUIRE(g.tokens.size() == g.n_prompt + g.n_gen);
  Engine& e = engine(model_path);
  REQUIRE(g.n_vocab == uint32_t(e.cfg.n_vocab));
  const size_t total = g.tokens.size(), np = g.n_prompt;
  std::vector<float> out(g.n_vocab);

  // 0. our tokenizer (parse_special=true, as the API uses) reproduces llama.cpp's prompt tokens
  {
    std::ifstream in(root + "/tests/e2e/prompts/" + name + ".txt", std::ios::binary);
    REQUIRE(in);
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const std::vector<int32_t> want(g.tokens.begin(), g.tokens.begin() + std::ptrdiff_t(np));
    REQUIRE(e.tokenizer.encode(text, /*parse_special=*/true) == want);
  }

  // 1. incremental, one token at a time
  {
    model::Transformer t(e.cfg, e.w, e.be, 256);
    double min_cos = 2;
    size_t agree = 0;
    for (size_t i = 0; i < total; ++i) {
      t.forward(std::span(g.tokens).subspan(i, 1), int64_t(i), out.data());
      double c = cosine(g.row(i), out.data());
      min_cos = std::min(min_cos, c);
      size_t ga = argmax(g.row(i)), na = argmax(out);
      bool tie = top_gap(g.row(i)) < 1e-3 || top_gap(out) < 1e-3;
      INFO(name << " pos " << i << " cos " << c << " golden argmax " << ga << " (" << g.row(i)[ga] << ") nibble argmax " << na
                << " (" << out[na] << ")");
      REQUIRE(c >= gate.min_cos);
      if (gate.strict_argmax && !tie) REQUIRE(ga == na);
      if (tie || ga == na) ++agree;
    }
    WARN(name << " [" << model << "]: min cosine over " << total << " positions = " << min_cos << ", top-1 agreement " << agree << "/"
                << total);
    REQUIRE(double(agree) >= gate.min_agree * double(total));
  }

  // 2. batch prefill
  model::Transformer t(e.cfg, e.w, e.be, 256);
  t.forward(std::span(g.tokens).first(np), 0, out.data());
  {
    double c = cosine(g.row(np - 1), out.data());
    WARN(name << ": prefill last-row cosine = " << c);
    REQUIRE(c >= gate.min_cos);
  }

  // 3. greedy from prefilled state
  size_t matched = 0;
  for (size_t k = 0; k < g.n_gen; ++k) {
    size_t tok = argmax(out);
    size_t gpos = np - 1 + k;  // golden row whose argmax is tokens[np+k]
    if (int32_t(tok) != g.tokens[np + k]) {
      double gap = std::min(top_gap(g.row(gpos)), top_gap(out));
      WARN(name << ": greedy diverged at gen " << k << " gap " << gap);
      if (gate.assert_greedy) REQUIRE(gap < 1e-3);
      break;
    }
    ++matched;
    if (k + 1 < g.n_gen) {
      int32_t t32 = int32_t(tok);
      t.forward(std::span(&t32, 1), int64_t(np + k), out.data());
    }
  }
  WARN(name << ": greedy matched " << matched << "/" << g.n_gen);
}
}  // namespace

#define E2E_CASE(MODEL, PROMPT) \
  TEST_CASE("e2e golden: " PROMPT " [" MODEL "]", "[e2e]") { run_prompt(MODEL, PROMPT); }
E2E_CASE("f16", "capital")
E2E_CASE("f16", "code")
E2E_CASE("f16", "chat")
E2E_CASE("q8_0", "capital")
E2E_CASE("q8_0", "code")
E2E_CASE("q8_0", "chat")
E2E_CASE("q4_0", "capital")
E2E_CASE("q4_0", "code")
E2E_CASE("q4_0", "chat")
