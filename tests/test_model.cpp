// Tests for model loading and the transformer forward pass (src/model/*) on tiny synthetic models built by
// tests/tiny_model.h: the logits must match an independent naive double-precision implementation
// (tests/naive_llama.h), chunked/incremental evaluation must not change a bit, and bad model files must be
// rejected with a specific message.
#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cmath>
#include <csignal>
#include <cstring>
#include <string>

#include <sys/wait.h>
#include <unistd.h>

#include "backend/cpu/cpu_backend.h"
#include "model/transformer.h"
#include "naive_llama.h"
#include "nibble/error.h"
#include "tiny_model.h"

// Sanitizer detection (GCC defines macros, Clang uses __has_feature): a few tests skip behaviour the
// sanitizers cannot support.
#ifdef __has_feature
#define NIBBLE_HAS_FEATURE(x) __has_feature(x)
#else
#define NIBBLE_HAS_FEATURE(x) 0
#endif
#if defined(__SANITIZE_ADDRESS__) || NIBBLE_HAS_FEATURE(address_sanitizer)
#define NIBBLE_UNDER_ASAN 1
#else
#define NIBBLE_UNDER_ASAN 0
#endif
#if defined(__SANITIZE_THREAD__) || NIBBLE_HAS_FEATURE(thread_sanitizer)
#define NIBBLE_UNDER_TSAN 1
#else
#define NIBBLE_UNDER_TSAN 0
#endif

using namespace nibble;
using Catch::Matchers::ContainsSubstring;

namespace {
// A fixed 5-token prompt used by most tests (includes the BOS-like id 1 and a merged token 257).
const std::vector<int32_t> kToks = {1, 50, 257, 3, 99};

// Largest absolute difference between two logits vectors.
double maxdiff(const std::vector<float>& a, const std::vector<float>& b) {
  REQUIRE(a.size() == b.size());
  double m = 0;
  for (size_t i = 0; i < a.size(); ++i) m = std::max(m, double(std::fabs(a[i] - b[i])));
  return m;
}

// Exact comparison of two float vectors by bytes.
bool bitwise_equal(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

// A parsed tiny model: owns the file bytes, the File, the config and the bound Weights (declared in
// dependency order so destruction is safe).
struct Loaded {
  std::vector<uint8_t> bytes;
  gguf::File file;
  model::LlamaConfig cfg;
  model::Weights w;
  Loaded(std::vector<uint8_t> b, const Backend& be)
      : bytes(std::move(b)), file(gguf::File::parse(bytes)), cfg(model::LlamaConfig::from_gguf(file)),
        w(model::Weights::bind(file, cfg, be)) {}
};

// Runs the tokens through a fresh Transformer and returns the last position's logits, either as one batch or
// one token at a time, with the given chunk size (max_batch).
std::vector<float> run(Loaded& m, Backend& be, std::vector<int32_t> toks, int64_t max_batch, bool one_by_one) {
  model::Transformer t(m.cfg, m.w, be, 64, max_batch);
  std::vector<float> out(static_cast<size_t>(m.cfg.n_vocab));
  if (one_by_one) {
    for (size_t i = 0; i < toks.size(); ++i) t.forward(std::span(toks).subspan(i, 1), static_cast<int64_t>(i), out.data());
  } else {
    t.forward(toks, 0, out.data());
  }
  return out;
}
}  // namespace

// The config read from a qwen2-style tiny file: dimensions, derived head_dim, rope base, and the qwen2 flags
// (NeoX rope, QKV bias).
TEST_CASE("model config of tiny model", "[model]") {
  ThreadPool pool(2);
  cpu::CpuBackend be(pool);
  Loaded m(tiny_model_gguf(1), be);
  CHECK(m.cfg.arch == "qwen2");
  CHECK(m.cfg.n_layers == 2);
  CHECK(m.cfg.n_embd == 16);
  CHECK(m.cfg.n_heads == 4);
  CHECK(m.cfg.n_kv_heads == 2);
  CHECK(m.cfg.n_ff == 32);
  CHECK(m.cfg.head_dim == 4);
  CHECK(m.cfg.n_vocab == 261);
  CHECK(m.cfg.n_ctx_train == 64);
  CHECK(m.cfg.rope_base == 10000.f);
  CHECK(m.cfg.rope_neox);
  CHECK(m.cfg.qkv_bias);
}

// F32 and F16 logits agree with the naive reference to 1e-4.
TEST_CASE("model matches naive reference", "[model]") {
  ThreadPool pool(2);
  cpu::CpuBackend be(pool);
  for (DType ty : {DType::F32, DType::F16}) {
    Loaded m(tiny_model_gguf(7, ty), be);
    auto ref = naive::naive_logits(m.file, kToks);
    CHECK(maxdiff(run(m, be, kToks, 512, false), ref) <= 1e-4);
  }
}

// The llama architecture (no bias, adjacent-pair rope, optional rope_freqs tensor) also matches the naive
// reference.
TEST_CASE("llama arch matches naive", "[model][llama]") {
  ThreadPool pool(2);
  cpu::CpuBackend be(pool);
  for (DType ty : {DType::F32, DType::F16})
    for (bool rf : {false, true}) {
      TinyModelOpts o;
      o.arch = "llama";
      o.rope_freqs = rf;
      Loaded m(tiny_model_gguf(7, ty, false, 259, o), be);
      CHECK(m.cfg.arch == "llama");
      CHECK(!m.cfg.rope_neox);
      CHECK(!m.cfg.qkv_bias);
      CHECK(m.w.rope_freqs.has_value() == rf);
      CHECK(!m.w.layers[0].bq);
      CHECK(maxdiff(run(m, be, kToks, 512, false), naive::naive_logits(m.file, kToks)) <= 1e-4);
    }
}

// Batch, one-token-at-a-time and 2-token-chunk evaluation produce identical bits.
TEST_CASE("llama chunked prefill equals incremental", "[model][llama]") {
  ThreadPool pool(2);
  cpu::CpuBackend be(pool);
  TinyModelOpts o;
  o.arch = "llama";
  o.rope_freqs = true;
  Loaded m(tiny_model_gguf(3, DType::F32, false, 259, o), be);
  auto batch = run(m, be, kToks, 512, false);
  CHECK(bitwise_equal(run(m, be, kToks, 512, true), batch));
  CHECK(bitwise_equal(run(m, be, kToks, 2, false), batch));
}

// A rope_freqs tensor of the wrong length or type, and rope.dimension_count / attention.key_length that
// disagree with embedding_length / head_count, are rejected naming the key.
TEST_CASE("llama bad rope_freqs and head_dim throw", "[model][llama]") {
  ThreadPool pool(2);
  cpu::CpuBackend be(pool);
  auto load = [&](TinyModelOpts o) { Loaded m(tiny_model_gguf(1, DType::F32, false, 259, o), be); };
  TinyModelOpts o;
  o.arch = "llama";
  o.rope_freqs = true;
  for (int bad : {1, 2}) {
    o.bad_rope_freqs = bad;
    CHECK_THROWS_WITH(load(o), ContainsSubstring("rope_freqs.weight"));
  }
  o = {};
  o.arch = "llama";
  o.rope_dim = 8;  // head_dim is 4
  CHECK_THROWS_WITH(load(o), ContainsSubstring("llama.rope.dimension_count") && ContainsSubstring("8") &&
                                 ContainsSubstring("4"));
  o.rope_dim = 4;
  CHECK_NOTHROW(load(o));
  o.key_length = 6;
  CHECK_THROWS_WITH(load(o), ContainsSubstring("llama.attention.key_length"));
  CHECK_THROWS_WITH(load(o), ContainsSubstring("invalid model"));
}

// A llama-architecture file carrying attention biases is rejected rather than silently ignoring them.
TEST_CASE("llama file with qkv bias throws", "[model][llama]") {
  ThreadPool pool(2);
  cpu::CpuBackend be(pool);
  TinyModelOpts o;
  o.arch = "llama";
  o.llama_bias = true;
  CHECK_THROWS_WITH((Loaded(tiny_model_gguf(1, DType::F32, false, 259, o), be)),
                    ContainsSubstring("unsupported tensor blk.0.attn_q.bias for architecture llama"));
}

// The same head-dimension consistency checks apply to qwen2 files.
TEST_CASE("qwen2 head_dim keys are checked", "[model]") {
  ThreadPool pool(2);
  cpu::CpuBackend be(pool);
  TinyModelOpts o;
  o.rope_dim = 4;
  o.key_length = 4;
  CHECK_NOTHROW((Loaded(tiny_model_gguf(1, DType::F32, false, 259, o), be)));
  o.rope_dim = 8;
  CHECK_THROWS_WITH((Loaded(tiny_model_gguf(1, DType::F32, false, 259, o), be)),
                    ContainsSubstring("qwen2.rope.dimension_count"));
  o.rope_dim = 4;
  o.key_length = 8;
  CHECK_THROWS_WITH((Loaded(tiny_model_gguf(1, DType::F32, false, 259, o), be)),
                    ContainsSubstring("qwen2.attention.key_length"));
}

// Q8_0 and Q4_0 tiny models: logits have cosine >= 0.999 with the naive model run on the dequantized weights,
// the same argmax, and batch equals incremental bit for bit.
TEST_CASE("quantized model matches naive on dequantized weights", "[model][quant]") {
  ThreadPool pool(2);
  cpu::CpuBackend be(pool);
  TinyModelOpts o;
  o.quant_dims = true;
  for (DType ty : {DType::Q8_0, DType::Q4_0}) {
    Loaded m(tiny_model_gguf(7, ty, false, 259, o), be);
    auto ref = naive::naive_logits(m.file, kToks);
    auto got = run(m, be, kToks, 512, false);
    double dot = 0, na = 0, nb = 0;
    size_t ag = 0, ar = 0;
    for (size_t i = 0; i < got.size(); ++i) {
      dot += double(got[i]) * ref[i];
      na += double(got[i]) * got[i];
      nb += double(ref[i]) * ref[i];
      if (got[i] > got[ag]) ag = i;
      if (ref[i] > ref[ar]) ar = i;
    }
    const double cosv = dot / std::sqrt(na * nb);
    UNSCOPED_INFO("cosine " << cosv);
    WARN("tiny " << (ty == DType::Q8_0 ? "Q8_0" : "Q4_0") << " cosine vs naive = " << cosv);
    CHECK(cosv >= 0.999);
    CHECK(ag == ar);
    auto batch = run(m, be, kToks, 512, false);
    CHECK(bitwise_equal(run(m, be, kToks, 512, true), batch));
  }
}

// With 1 to 8 threads (and so different row splits and the packed GEMM path) the logits are bit-identical.
TEST_CASE("quantized prefill logits do not depend on the thread count", "[model][quant]") {
  TinyModelOpts o;
  o.quant_dims = true;
  for (DType ty : {DType::Q8_0, DType::Q4_0}) {
    std::vector<std::vector<float>> got;
    for (int t : {1, 2, 3, 4, 8}) {
      ThreadPool pool(t);
      cpu::CpuBackend be(pool);
      Loaded m(tiny_model_gguf(7, ty, false, 259, o), be);
      got.push_back(run(m, be, kToks, 512, false));  // n = 5 >= 4: the packed GEMM path
    }
    for (size_t i = 1; i < got.size(); ++i) CHECK(bitwise_equal(got[i], got[0]));
  }
}

// Kernels use a fixed accumulation order per row, so how tokens are batched cannot change a single bit.
TEST_CASE("incremental and chunked prefill equal batch", "[model]") {
  ThreadPool pool(2);
  cpu::CpuBackend be(pool);
  Loaded m(tiny_model_gguf(3), be);
  auto batch = run(m, be, kToks, 512, false);
  // Kernels are per-row with a fixed accumulation order, so batching must not change a single bit.
  CHECK(bitwise_equal(run(m, be, kToks, 512, true), batch));
  CHECK(bitwise_equal(run(m, be, kToks, 2, false), batch));
}

// Files without output.weight tie the output to the embedding; an explicit output matrix is used when present
// and gives different logits.
TEST_CASE("model tied and explicit output", "[model]") {
  ThreadPool pool(2);
  cpu::CpuBackend be(pool);
  Loaded tied(tiny_model_gguf(5), be);
  CHECK(tied.file.find_tensor("output.weight") == nullptr);
  TinyModelOpts o;
  o.explicit_output = true;
  Loaded expl(tiny_model_gguf(5, DType::F32, false, 259, o), be);
  CHECK(expl.file.find_tensor("output.weight") != nullptr);
  CHECK(maxdiff(run(expl, be, kToks, 512, false), naive::naive_logits(expl.file, kToks)) <= 1e-4);
  // output.weight comes from its own RNG, so all shared tensors are byte-identical and
  // the logit difference below can only come from the output projection.
  for (const char* name : {"token_embd.weight", "blk.0.attn_q.weight", "blk.1.ffn_down.weight"}) {
    const auto* a = tied.file.find_tensor(name);
    const auto* b = expl.file.find_tensor(name);
    REQUIRE(a->nbytes == b->nbytes);
    CHECK(std::memcmp(a->tensor.data, b->tensor.data, a->nbytes) == 0);
  }
  CHECK(maxdiff(run(tied, be, kToks, 512, false), run(expl, be, kToks, 512, false)) > 1e-3);
}

// Missing tensor, wrong tensor shape, unsupported architecture and unsupported tensor type are each rejected
// with a message naming the culprit.
TEST_CASE("model rejects bad files", "[model]") {
  ThreadPool pool(2);
  cpu::CpuBackend be(pool);
  auto load = [&](TinyModelOpts o) { Loaded m(tiny_model_gguf(1, DType::F32, false, 259, o), be); };
  SECTION("missing tensor") {
    TinyModelOpts o;
    o.drop_tensor = "blk.1.ffn_up.weight";
    CHECK_THROWS_WITH(load(o), ContainsSubstring("blk.1.ffn_up.weight"));
  }
  SECTION("shape mismatch") {
    TinyModelOpts o;
    o.bad_shape = "blk.0.attn_k.weight";
    CHECK_THROWS_WITH(load(o), ContainsSubstring("expected"));
    CHECK_THROWS_WITH(load(o), ContainsSubstring("blk.0.attn_k.weight"));
  }
  SECTION("unsupported architecture") {
    TinyModelOpts o;
    o.arch = "phi3";
    CHECK_THROWS_WITH(load(o), ContainsSubstring("phi3"));
  }
  SECTION("unsupported dtype") {
    TinyModelOpts o;
    o.raw_type_tensor = "blk.0.ffn_down.weight";
    CHECK_THROWS_WITH(load(o), ContainsSubstring("Q3_K"));
    CHECK_THROWS_WITH(load(o), ContainsSubstring("blk.0.ffn_down.weight"));
  }
}

// Feeding tokens past the end of the context is an Error.
TEST_CASE("forward bounds and head divisibility", "[model]") {
  ThreadPool pool(2);
  cpu::CpuBackend be(pool);
  Loaded m(tiny_model_gguf(1), be);
  model::Transformer t(m.cfg, m.w, be, 4);
  std::vector<float> out(261);
  CHECK_THROWS_AS(t.forward(kToks, 0, out.data()), Error);
  CHECK_THROWS_AS(t.forward(std::span(kToks).subspan(0, 2), 3, out.data()), Error);
  CHECK_NOTHROW(t.forward(std::span(kToks).subspan(0, 2), 2, out.data()));
}

// Head counts that do not divide the embedding width, or an odd head dimension (rope needs pairs), are
// rejected.
TEST_CASE("config rejects indivisible head counts", "[model]") {
  auto cfg = [](int64_t heads, int64_t kv) {
    TinyModelOpts o;
    o.heads_kv = heads;
    o.kv_heads_kv = kv;
    auto bytes = tiny_model_gguf(1, DType::F32, false, 259, o);
    gguf::File f = gguf::File::parse(bytes);
    (void)model::LlamaConfig::from_gguf(f);
  };
  CHECK_THROWS_WITH(cfg(3, 3), ContainsSubstring("embedding_length 16 is not divisible by"));
  CHECK_THROWS_WITH(cfg(4, 3), ContainsSubstring("head_count 4 is not divisible by"));
  CHECK_THROWS_WITH(cfg(16, 16), ContainsSubstring("head_dim 1 must be even"));
  CHECK_NOTHROW(cfg(4, 2));
}

// A Weights object with fewer layers than the config says is rejected at construction.
TEST_CASE("transformer rejects weights with the wrong layer count", "[model]") {
  ThreadPool pool(1);
  cpu::CpuBackend be(pool);
  Loaded m(tiny_model_gguf(1), be);
  m.w.layers.pop_back();
  CHECK_THROWS_WITH(model::Transformer(m.cfg, m.w, be, 8), ContainsSubstring("layers"));
}

// Constructing and destroying Transformers repeatedly leaves no live backend allocations.
TEST_CASE("transformer destroy frees all backend allocations", "[model]") {
  ThreadPool pool(1);
  cpu::CpuBackend be(pool);
  Loaded m(tiny_model_gguf(1), be);
  const size_t base = be.live_allocations();
  for (int i = 0; i < 5; ++i) {
    model::Transformer t(m.cfg, m.w, be, 64, 8);
    CHECK(be.live_allocations() > base);
  }
  CHECK(be.live_allocations() == base);
}

// Absurd context sizes fail with an Error mentioning n_ctx (overflow paths and, outside sanitizers, a real
// huge allocation) and leak nothing.
TEST_CASE("transformer huge n_ctx throws Error naming n_ctx", "[model]") {
  ThreadPool pool(1);
  cpu::CpuBackend be(pool);
  Loaded m(tiny_model_gguf(1), be);
  const size_t base = be.live_allocations();
  std::vector<int64_t> sizes = {INT64_MAX / 4, INT64_MAX};  // overflow paths
#if !NIBBLE_UNDER_ASAN && !NIBBLE_UNDER_TSAN  // ASan/TSan abort on multi-TB allocations instead of failing them
  sizes.push_back(int64_t{1000000000000});
#endif
  for (int64_t n : sizes) {
    CHECK_THROWS_WITH(model::Transformer(m.cfg, m.w, be, n, 8), ContainsSubstring("n_ctx=" + std::to_string(n)));
  }
  CHECK(be.live_allocations() == base);
}

#ifndef NDEBUG
// Spec 6.4: Debug builds check shape agreement at every op call. A valid model must
// pass every check in all prefill modes.
// In debug builds, a valid model passes every NIBBLE_DCHECK in all evaluation modes.
TEST_CASE("debug checks pass on a valid tiny-model forward", "[model][dcheck]") {
  ThreadPool pool(2);
  cpu::CpuBackend be(pool);
  for (DType ty : {DType::F32, DType::F16}) {
    Loaded m(tiny_model_gguf(11, ty), be);
    auto batch = run(m, be, kToks, 512, false);
    CHECK(bitwise_equal(run(m, be, kToks, 512, true), batch));
    CHECK(bitwise_equal(run(m, be, kToks, 2, false), batch));
  }
  TinyModelOpts o;
  o.explicit_output = true;
  Loaded expl(tiny_model_gguf(5, DType::F32, false, 259, o), be);
  CHECK(run(expl, be, kToks, 512, false).size() == size_t(expl.cfg.n_vocab));
}

// A deliberately wrong argument (an F16 RMSNorm weight) makes a forked child abort with SIGABRT and a message
// naming the check, proving the checks are live.
TEST_CASE("debug check failure aborts naming the op", "[model][dcheck]") {
  int fds[2];
  REQUIRE(::pipe(fds) == 0);
  const pid_t pid = ::fork();
  REQUIRE(pid >= 0);
  if (pid == 0) {
    ::dup2(fds[1], 2);
    ThreadPool pool(1);
    cpu::CpuBackend be(pool);
    std::vector<uint16_t> w16(4, 0);
    Tensor w;
    w.type = DType::F16;  // rmsnorm weights must be F32
    w.n_dims = 1;
    w.ne = {4, 1, 1, 1};
    w.data = w16.data();
    std::vector<float> x(4, 1.f), y(4);
    be.rmsnorm(x.data(), w, y.data(), 1, 1e-6f);
    ::_exit(0);  // not reached when the check fires
  }
  ::close(fds[1]);
  std::string err;
  char buf[512];
  for (ssize_t r; (r = ::read(fds[0], buf, sizeof buf)) > 0;) err.append(buf, size_t(r));
  ::close(fds[0]);
  int status = 0;
  REQUIRE(::waitpid(pid, &status, 0) == pid);
  INFO("child stderr: " << err);
  REQUIRE(WIFSIGNALED(status));
  CHECK(WTERMSIG(status) == SIGABRT);
  CHECK_THAT(err, ContainsSubstring("NIBBLE_DCHECK") && ContainsSubstring("rmsnorm") &&
                      ContainsSubstring("need F32"));
}
#endif
