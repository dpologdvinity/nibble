#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>

#include "backend/cpu/cpu_backend.h"
#include "golden_file.h"
#include "gguf/gguf.h"
#include "model/transformer.h"
#include "thread_pool.h"

using namespace nibble;

namespace {
struct Engine {
  ThreadPool pool{int(std::max(1u, std::thread::hardware_concurrency()))};
  cpu::CpuBackend be{pool};
  gguf::File file;
  model::LlamaConfig cfg;
  model::Weights w;
  explicit Engine(const std::string& path)
      : file(gguf::File::open(path)), cfg(model::LlamaConfig::from_gguf(file)), w(model::Weights::bind(file, cfg, be)) {}
};

Engine& engine(const std::string& path) {
  static std::unique_ptr<Engine> e;
  if (!e) e = std::make_unique<Engine>(path);
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

void run_prompt(const std::string& name) {
  const std::string root = NIBBLE_SOURCE_DIR;
  const std::string model_path = root + "/models/qwen2.5-0.5b-instruct-fp16.gguf";
  const std::string golden_path = root + "/models/golden/qwen2.5-0.5b-f16/" + name + ".bin";
  if (!std::filesystem::exists(model_path) || !std::filesystem::exists(golden_path)) SKIP("model or golden missing");

  Golden g = Golden::load(golden_path);
  Engine& e = engine(model_path);
  REQUIRE(g.n_vocab == uint32_t(e.cfg.n_vocab));
  const size_t total = g.tokens.size(), np = g.n_prompt;
  std::vector<float> out(g.n_vocab);

  // 1. incremental, one token at a time
  {
    model::Transformer t(e.cfg, e.w, e.be, 256);
    double min_cos = 2;
    for (size_t i = 0; i < total; ++i) {
      t.forward(std::span(g.tokens).subspan(i, 1), int64_t(i), out.data());
      double c = cosine(g.row(i), out.data());
      min_cos = std::min(min_cos, c);
      size_t ga = argmax(g.row(i)), na = argmax(out);
      bool tie = top_gap(g.row(i)) < 1e-3 || top_gap(out) < 1e-3;
      INFO(name << " pos " << i << " cos " << c << " golden argmax " << ga << " (" << g.row(i)[ga] << ") nibble argmax " << na
                << " (" << out[na] << ")");
      REQUIRE(c >= 0.999);
      if (!tie) REQUIRE(ga == na);
    }
    WARN(name << ": min cosine over " << total << " positions = " << min_cos);
  }

  // 2. batch prefill
  model::Transformer t(e.cfg, e.w, e.be, 256);
  t.forward(std::span(g.tokens).first(np), 0, out.data());
  {
    double c = cosine(g.row(np - 1), out.data());
    WARN(name << ": prefill last-row cosine = " << c);
    REQUIRE(c >= 0.999);
  }

  // 3. greedy from prefilled state
  size_t matched = 0;
  for (size_t k = 0; k < g.n_gen; ++k) {
    size_t tok = argmax(out);
    size_t gpos = np - 1 + k;  // golden row whose argmax is tokens[np+k]
    if (int32_t(tok) != g.tokens[np + k]) {
      double gap = std::min(top_gap(g.row(gpos)), top_gap(out));
      WARN(name << ": greedy diverged at gen " << k << " gap " << gap);
      REQUIRE(gap < 1e-3);
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

TEST_CASE("e2e golden: capital", "[e2e]") { run_prompt("capital"); }
TEST_CASE("e2e golden: code", "[e2e]") { run_prompt("code"); }
TEST_CASE("e2e golden: chat", "[e2e]") { run_prompt("chat"); }
