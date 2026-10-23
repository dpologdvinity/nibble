#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
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
  if (model.ends_with("f16")) return {0.999, 1.0, true, true};
  return {0.995, 0.90, false, false};
}

// One row per model file under test: id used in the case names, gguf and golden names, and the chat prompt.
struct E2EModel {
  const char* id;
  const char* gguf;
  const char* golden;
  const char* chat;
};
const E2EModel kModels[] = {
    {"f16", "qwen2.5-0.5b-instruct-fp16", "qwen2.5-0.5b-f16", "chat"},
    {"llama32-f16", "llama-3.2-1b-instruct-f16", "llama32-1b-f16", "chat_llama3"},
    {"tinyllama-f16", "tinyllama-1.1b-chat-f16", "tinyllama-1.1b-f16", "chat_zephyr"},
    {"q8_0", "qwen2.5-0.5b-instruct-q8_0", "qwen2.5-0.5b-q8_0", "chat"},
    {"q4_0", "qwen2.5-0.5b-instruct-q4_0", "qwen2.5-0.5b-q4_0", "chat"},
    {"q4_k_m", "qwen2.5-0.5b-instruct-q4_k_m", "qwen2.5-0.5b-q4_k_m", "chat"},
    {"q5_k_m", "qwen2.5-0.5b-instruct-q5_k_m", "qwen2.5-0.5b-q5_k_m", "chat"},
    {"q6_k", "qwen2.5-0.5b-instruct-q6_k", "qwen2.5-0.5b-q6_k", "chat"},
    {"llama32-q4_k_m", "llama-3.2-1b-instruct-q4_k_m", "llama32-1b-q4_k_m", "chat_llama3"},
    {"llama32-q5_k_m", "llama-3.2-1b-instruct-q5_k_m", "llama32-1b-q5_k_m", "chat_llama3"},
    {"llama32-q6_k", "llama-3.2-1b-instruct-q6_k", "llama32-1b-q6_k", "chat_llama3"},
    {"tinyllama-q4_k_m", "tinyllama-1.1b-chat-q4_k_m", "tinyllama-1.1b-q4_k_m", "chat_zephyr"},
    {"tinyllama-q5_k_m", "tinyllama-1.1b-chat-q5_k_m", "tinyllama-1.1b-q5_k_m", "chat_zephyr"},
    {"tinyllama-q6_k", "tinyllama-1.1b-chat-q6_k", "tinyllama-1.1b-q6_k", "chat_zephyr"},
};
const E2EModel& model_by_id(const std::string& id) {
  for (const E2EModel& m : kModels)
    if (id == m.id) return m;
  FAIL("unknown e2e model id " << id);
  return kModels[0];
}

// Prompt file for `name`: the chat prompt differs per model family.
void check_prompt_tokens(const tok::Tokenizer& t, const Golden& g, const std::string& name, const std::string& file = "") {
  const std::string root = NIBBLE_SOURCE_DIR;
  std::ifstream in(root + "/tests/e2e/prompts/" + (file.empty() ? name : file) + ".txt", std::ios::binary);
  REQUIRE(in);
  const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  const std::vector<int32_t> want(g.tokens.begin(), g.tokens.begin() + std::ptrdiff_t(g.n_prompt));
  const tok::Part part{text, true};
  INFO(name << " (" << file << ")");
  REQUIRE(t.encode_parts(std::span(&part, 1), /*add_bos=*/true) == want);
}

void run_prompt(const std::string& model, const std::string& name) {
  const Gate gate = gate_for(model);
  const std::string root = NIBBLE_SOURCE_DIR;
  const E2EModel& em = model_by_id(model);
  const std::string gguf = em.gguf, golden = em.golden, chat = em.chat;
  const std::string model_path = root + "/models/" + gguf + ".gguf";
  const std::string golden_path = root + "/models/golden/" + golden + "/" + name + ".bin";
  if (!std::filesystem::exists(model_path) || !std::filesystem::exists(golden_path)) SKIP("model or golden missing");

  Golden g = Golden::load(golden_path);
  REQUIRE(g.n_gen == 64);
  REQUIRE(g.tokens.size() == g.n_prompt + g.n_gen);
  Engine& e = engine(model_path);
  REQUIRE(g.n_vocab == uint32_t(e.cfg.n_vocab));
  const size_t total = g.tokens.size(), np = g.n_prompt;
  std::vector<float> out(g.n_vocab);

  // 0. our tokenizer (special parsing and automatic bos, as the API uses) reproduces llama.cpp's prompt tokens
  check_prompt_tokens(e.tokenizer, g, name, name == "chat" ? chat : name);

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
#define E2E_MODEL(MODEL) E2E_CASE(MODEL, "capital") E2E_CASE(MODEL, "code") E2E_CASE(MODEL, "chat")
// Ordered by model file so engine() reloads once per file.
E2E_MODEL("f16")
E2E_MODEL("llama32-f16")
E2E_MODEL("tinyllama-f16")
E2E_MODEL("q8_0")
E2E_MODEL("q4_0")
E2E_MODEL("q4_k_m")
E2E_MODEL("q5_k_m")
E2E_MODEL("q6_k")
E2E_MODEL("llama32-q4_k_m")
E2E_MODEL("llama32-q5_k_m")
E2E_MODEL("llama32-q6_k")
E2E_MODEL("tinyllama-q4_k_m")
E2E_MODEL("tinyllama-q5_k_m")
E2E_MODEL("tinyllama-q6_k")

// Prompt tokens only (no forward pass) for every model that has goldens.
TEST_CASE("e2e prompt tokens match llama.cpp", "[e2e]") {
  const std::string root = NIBBLE_SOURCE_DIR;
  int checked = 0, expected = 0;
  for (const E2EModel& m : kModels) {
    const std::string gguf = root + "/models/" + m.gguf + ".gguf";
    if (!std::filesystem::exists(gguf) || !std::filesystem::is_directory(root + "/models/golden/" + m.golden)) continue;
    expected += 3;
    const auto t = tok::Tokenizer::from_gguf(gguf::File::open(gguf));
    for (const char* p : {"capital", "code", "chat"}) {
      const std::string bin = root + "/models/golden/" + m.golden + "/" + p + ".bin";
      INFO(m.golden << " " << p);
      REQUIRE(std::filesystem::exists(bin));
      check_prompt_tokens(t, Golden::load(bin), p, std::string(p) == "chat" ? m.chat : p);
      ++checked;
    }
  }
  REQUIRE(checked == expected);
}

// The k-quant files must be llama-quantize's default mixes (fallback types next to the K types), not --pure
// regenerations: tensor counts per type, from the mixes the M7 research recorded.
TEST_CASE("k-quant files have the llama-quantize mix", "[e2e]") {
  using Counts = std::map<std::string, int>;
  const struct {
    const char* id;
    Counts want;
  } mixes[] = {
      {"q4_k_m", {{"F32", 121}, {"Q5_0", 133}, {"Q8_0", 13}, {"Q4_K", 12}, {"Q6_K", 12}}},
      {"q5_k_m", {{"F32", 121}, {"Q5_1", 133}, {"Q8_0", 13}, {"Q5_K", 12}, {"Q6_K", 12}}},
      {"q6_k", {{"F32", 121}, {"Q8_0", 146}, {"Q6_K", 24}}},
      {"llama32-q4_k_m", {{"F32", 34}, {"Q4_K", 96}, {"Q6_K", 17}}},
      {"llama32-q5_k_m", {{"F32", 34}, {"Q5_K", 96}, {"Q6_K", 17}}},
      {"llama32-q6_k", {{"F32", 34}, {"Q6_K", 113}}},
      {"tinyllama-q4_k_m", {{"F32", 45}, {"Q4_K", 135}, {"Q6_K", 21}}},
      {"tinyllama-q5_k_m", {{"F32", 45}, {"Q5_K", 135}, {"Q6_K", 21}}},
      {"tinyllama-q6_k", {{"F32", 45}, {"Q6_K", 156}}},
  };
  const std::string root = NIBBLE_SOURCE_DIR;
  int checked = 0;
  for (const auto& mix : mixes) {
    const std::string path = root + "/models/" + model_by_id(mix.id).gguf + ".gguf";
    if (!std::filesystem::exists(path)) continue;
    Counts got;
    const gguf::File file = gguf::File::open(path);
    for (const auto& t : file.tensors()) ++got[dtype_info(t.tensor.type).name];
    INFO(mix.id);
    CHECK(got == mix.want);
    ++checked;
  }
  if (checked == 0) SKIP("no k-quant model files");
}
