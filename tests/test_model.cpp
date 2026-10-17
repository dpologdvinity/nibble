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

using namespace nibble;
using Catch::Matchers::ContainsSubstring;

namespace {
const std::vector<int32_t> kToks = {1, 50, 257, 3, 99};

double maxdiff(const std::vector<float>& a, const std::vector<float>& b) {
  REQUIRE(a.size() == b.size());
  double m = 0;
  for (size_t i = 0; i < a.size(); ++i) m = std::max(m, double(std::fabs(a[i] - b[i])));
  return m;
}

bool bitwise_equal(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

struct Loaded {
  std::vector<uint8_t> bytes;
  gguf::File file;
  model::LlamaConfig cfg;
  model::Weights w;
  Loaded(std::vector<uint8_t> b, const Backend& be)
      : bytes(std::move(b)), file(gguf::File::parse(bytes)), cfg(model::LlamaConfig::from_gguf(file)),
        w(model::Weights::bind(file, cfg, be)) {}
};

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

TEST_CASE("model matches naive reference", "[model]") {
  ThreadPool pool(2);
  cpu::CpuBackend be(pool);
  for (DType ty : {DType::F32, DType::F16}) {
    Loaded m(tiny_model_gguf(7, ty), be);
    auto ref = naive::naive_logits(m.file, kToks);
    CHECK(maxdiff(run(m, be, kToks, 512, false), ref) <= 1e-4);
  }
}

TEST_CASE("incremental and chunked prefill equal batch", "[model]") {
  ThreadPool pool(2);
  cpu::CpuBackend be(pool);
  Loaded m(tiny_model_gguf(3), be);
  auto batch = run(m, be, kToks, 512, false);
  // Kernels are per-row with a fixed accumulation order, so batching must not change a single bit.
  CHECK(bitwise_equal(run(m, be, kToks, 512, true), batch));
  CHECK(bitwise_equal(run(m, be, kToks, 2, false), batch));
}

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
  CHECK(maxdiff(run(tied, be, kToks, 512, false), run(expl, be, kToks, 512, false)) > 1e-3);
}

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
    o.q4_tensor = "blk.0.ffn_down.weight";
    CHECK_THROWS_WITH(load(o), ContainsSubstring("Q4_0"));
    CHECK_THROWS_WITH(load(o), ContainsSubstring("blk.0.ffn_down.weight"));
  }
}

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

TEST_CASE("transformer huge n_ctx throws Error naming n_ctx", "[model]") {
  ThreadPool pool(1);
  cpu::CpuBackend be(pool);
  Loaded m(tiny_model_gguf(1), be);
  const size_t base = be.live_allocations();
  std::vector<int64_t> sizes = {INT64_MAX / 4, INT64_MAX};  // overflow paths
#ifndef __SANITIZE_ADDRESS__  // ASan aborts on multi-TB allocations instead of failing them
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
