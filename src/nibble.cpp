#include "nibble/nibble.h"

#include <algorithm>
#include <new>
#include <span>
#include <thread>
#include <vector>

#include "backend/cpu/cpu_backend.h"
#include "gguf/gguf.h"
#include "model/transformer.h"
#include "thread_pool.h"
#include "tokenizer/tokenizer.h"
#include "tokenizer/unicode.h"

namespace nibble {

// Members are in dependency order; destruction runs in reverse
// (Weights before File, backend before pool).
struct Model::Impl {
  gguf::File file;
  tok::Tokenizer tokenizer;
  model::LlamaConfig cfg;
  ThreadPool pool;
  cpu::CpuBackend backend;
  model::Weights weights;
  int64_t n_ctx;

  Impl(const std::string& path, const ModelParams& p)
      : file(gguf::File::open(path)),
        tokenizer(tok::Tokenizer::from_gguf(file)),
        cfg(model::LlamaConfig::from_gguf(file)),
        pool(p.n_threads > 0 ? p.n_threads
                             : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()))),
        backend(pool),
        weights(model::Weights::bind(file, cfg, backend)),
        n_ctx(p.n_ctx > 0 ? p.n_ctx : std::min<int64_t>(4096, cfg.n_ctx_train)) {
    if (tokenizer.n_vocab() != cfg.n_vocab)
      throw Error("tokenizer has " + std::to_string(tokenizer.n_vocab()) + " tokens but token_embd.weight has " +
                  std::to_string(cfg.n_vocab) + " rows");
    if (p.n_ctx < 0) throw Error("n_ctx must be non-negative, got " + std::to_string(p.n_ctx));
    if (n_ctx <= 0) throw Error("invalid context length " + std::to_string(n_ctx));
  }
};

Model::Model() = default;
Model::Model(Model&&) noexcept = default;
Model& Model::operator=(Model&&) noexcept = default;
Model::~Model() = default;

Model Model::load(const std::string& path, ModelParams p) {
  Model m;
  try {
    m.impl_ = std::make_unique<Impl>(path, p);
  } catch (const std::bad_alloc&) {
    throw Error("out of memory loading " + path);
  }
  return m;
}

const std::string& Model::architecture() const { return impl_->cfg.arch; }
int64_t Model::n_ctx() const { return impl_->n_ctx; }

struct Session::Impl {
  Model::Impl& m;
  model::Transformer tf;
  std::vector<float> logits;  // preallocated; forward path must not allocate
  int64_t n_prompt = 0, n_gen = 0;

  explicit Impl(Model& model)
      : m(*model.impl_),
        tf(m.cfg, m.weights, m.backend, m.n_ctx),
        logits(static_cast<size_t>(m.cfg.n_vocab)) {}

  int32_t argmax() const {
    return static_cast<int32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
  }
};

Session::Session(Model& m) : impl_(std::make_unique<Impl>(m)) {}
Session::~Session() = default;
int64_t Session::n_prompt_tokens() const { return impl_->n_prompt; }
int64_t Session::n_generated() const { return impl_->n_gen; }

StopReason Session::generate(std::string_view prompt, const GenParams& gp,
                             const std::function<bool(std::string_view)>& on_token) {
  Impl& s = *impl_;
  s.n_prompt = s.n_gen = 0;
  const int64_t n_ctx = s.m.n_ctx;
  std::vector<int32_t> toks = s.m.tokenizer.encode(prompt, true);
  if (toks.empty()) throw Error("empty prompt");
  if (static_cast<int64_t>(toks.size()) >= n_ctx)
    throw Error("prompt has " + std::to_string(toks.size()) + " tokens but context is " + std::to_string(n_ctx));
  s.n_prompt = static_cast<int64_t>(toks.size());

  tok::Utf8Streamer streamer;
  auto finish = [&](StopReason r) {
    std::string tail = streamer.flush();
    if (!tail.empty()) on_token(tail);  // already stopping: ignore return value
    return r;
  };

  s.tf.forward(toks, 0, s.logits.data());
  int64_t pos = s.n_prompt;
  while (true) {
    if (gp.max_tokens <= 0) return finish(StopReason::MaxTokens);
    const int32_t next = s.argmax();
    if (s.m.tokenizer.is_eog(next)) return finish(StopReason::Eos);
    std::string piece = streamer.push(s.m.tokenizer.decode_token(next));
    ++s.n_gen;
    if (!piece.empty() && !on_token(piece)) return finish(StopReason::Cancelled);
    if (s.n_gen >= gp.max_tokens) return finish(StopReason::MaxTokens);
    // Total tokens (prompt + generated) may not exceed n_ctx.
    if (s.n_prompt + s.n_gen >= n_ctx) return finish(StopReason::ContextFull);
    const int32_t one[1] = {next};
    s.tf.forward(std::span<const int32_t>(one, 1), pos, s.logits.data());
    ++pos;
  }
}

}  // namespace nibble
