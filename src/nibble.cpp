#include "nibble/nibble.h"

#include <algorithm>
#include <new>
#include <random>
#include <span>
#include <system_error>
#include <thread>
#include <variant>
#include <vector>

#include "backend/cpu/cpu_backend.h"
#include "gguf/gguf.h"
#include "model/transformer.h"
#include "sampler.h"
#include "thread_pool.h"
#include "tokenizer/tokenizer.h"
#include "tokenizer/unicode.h"

namespace nibble {

namespace {

// Compares the tokenizer vocab size with token_embd rows straight from the GGUF
// metadata, before Tokenizer::from_gguf builds its tables, so a file with a huge
// token array cannot make us build them first. Missing or mistyped entries are left
// to the tokenizer/config loaders, which report them.
const gguf::File& check_vocab_matches_embd(const gguf::File& f) {
  const gguf::Value* v = f.find("tokenizer.ggml.tokens");
  const auto* toks = v ? std::get_if<gguf::Array>(v) : nullptr;
  const auto* te = f.find_tensor("token_embd.weight");
  if (toks && toks->is_string() && te && static_cast<int64_t>(toks->size()) != te->tensor.ne[1])
    throw Error("tokenizer has " + std::to_string(toks->size()) + " tokens but token_embd.weight has " +
                std::to_string(te->tensor.ne[1]) + " rows");
  return f;
}

// Builds the config and checks the requested n_ctx against context_length before
// the tokenizer, weights and backend are built.
model::LlamaConfig load_config(const gguf::File& f, const ModelParams& p) {
  model::LlamaConfig c = model::LlamaConfig::from_gguf(f);
  if (p.n_ctx > c.n_ctx_train)
    throw Error("n_ctx " + std::to_string(p.n_ctx) + " exceeds the model's context_length " +
                std::to_string(c.n_ctx_train));
  return c;
}

}  // namespace

// Members are in dependency order; destruction runs in reverse
// (Weights before File, backend before pool).
struct Model::Impl {
  gguf::File file;
  model::LlamaConfig cfg;
  tok::Tokenizer tokenizer;
  ThreadPool pool;
  cpu::CpuBackend backend;
  model::Weights weights;
  int64_t n_ctx;

  Impl(const std::string& path, const ModelParams& p)
      : file(gguf::File::open(path)),
        cfg(load_config(file, p)),
        tokenizer(tok::Tokenizer::from_gguf(check_vocab_matches_embd(file))),
        pool(p.n_threads > 0 ? p.n_threads
                             : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()))),
        backend(pool),
        weights(model::Weights::bind(file, cfg, backend)),
        n_ctx(p.n_ctx > 0 ? p.n_ctx : std::min<int64_t>(4096, cfg.n_ctx_train)) {
    if (n_ctx <= 0) throw Error("invalid context length " + std::to_string(n_ctx));
  }
};

Model::Model() = default;
Model::Model(Model&&) noexcept = default;
Model& Model::operator=(Model&&) noexcept = default;
Model::~Model() = default;

Model Model::load(const std::string& path, ModelParams p) {
  if (p.n_ctx < 0) throw Error("n_ctx must be non-negative, got " + std::to_string(p.n_ctx));
  if (p.n_threads < 0) throw Error("n_threads must be non-negative, got " + std::to_string(p.n_threads));
  Model m;
  try {
    m.impl_ = std::make_unique<Impl>(path, p);
  } catch (const std::bad_alloc&) {
    throw Error("out of memory loading " + path);
  }
  return m;
}

namespace {
[[noreturn]] void throw_moved_from() { throw Error("use of moved-from Model"); }
}  // namespace

const std::string& Model::architecture() const {
  if (!impl_) throw_moved_from();
  return impl_->cfg.arch;
}
int64_t Model::n_ctx() const {
  if (!impl_) throw_moved_from();
  return impl_->n_ctx;
}

struct Session::Impl {
  Model::Impl& m;
  model::Transformer tf;
  std::vector<float> logits;  // preallocated; forward path must not allocate. Valid when n_past > 0.
  int64_t n_prompt = 0, n_gen = 0;
  int64_t n_past = 0;  // tokens in the KV cache
  // generate calls made on this Session, counted on entry, so also those that
  // throw (even before validation). Seeds the per-call RNG; Chat's desync guard.
  uint64_t n_calls = 0;

  explicit Impl(Model& model)
      : m(model.impl_ ? *model.impl_ : throw Error("use of moved-from Model")),
        tf(m.cfg, m.weights, m.backend, m.n_ctx),
        logits(static_cast<size_t>(m.cfg.n_vocab)) {}

  StopReason run(const std::vector<int32_t>& toks, const GenParams& gp,
                 const std::function<bool(std::string_view)>& on_token);
};

Session::Session(Model& m) : impl_(std::make_unique<Impl>(m)) {}
Session::~Session() = default;
int64_t Session::n_prompt_tokens() const { return impl_->n_prompt; }
int64_t Session::n_generated() const { return impl_->n_gen; }
int64_t Session::n_past() const { return impl_->n_past; }
const std::string& Session::architecture() const { return impl_->m.cfg.arch; }
uint64_t Session::generations() const { return impl_->n_calls; }
std::string Session::chat_template() const {
  const gguf::File& f = impl_->m.file;
  return f.find("tokenizer.chat_template") ? f.get_string("tokenizer.chat_template") : std::string();
}

StopReason Session::Impl::run(const std::vector<int32_t>& toks, const GenParams& gp,
                              const std::function<bool(std::string_view)>& on_token) {
  n_prompt = n_gen = 0;
  const int64_t n_new = static_cast<int64_t>(toks.size());
  const int64_t start = gp.append ? n_past : 0;
  if (n_new == 0 && start == 0) throw Error("empty prompt");
  // Built before any state changes, so invalid params throw with the cache intact.
  // Allocates its n_vocab buffers here, outside the per-token loop.
  uint64_t seed = 0;
  if (gp.seed) {
    uint64_t x = *gp.seed ^ (n_calls - 1);  // this call's index
    seed = splitmix64(x);
  }
  if (!gp.seed && gp.temperature > 0.0f) {  // the seed only matters when sampling
    try {
      std::random_device rd;
      seed = (static_cast<uint64_t>(rd()) << 32) ^ rd();
    } catch (const std::system_error& e) {
      throw Error(std::string("cannot seed the sampler: random_device failed: ") + e.what());
    }
  }
  Sampler sampler({gp.temperature, gp.top_k, gp.top_p, seed}, m.cfg.n_vocab);
  if (!gp.append) n_past = 0;
  if (gp.max_tokens <= 0) return StopReason::MaxTokens;
  if (start + n_new >= m.n_ctx) return StopReason::ContextFull;

  // n_past is 0 while a forward is in flight, so if it throws the cache reads as
  // empty rather than out of step with `logits`.
  n_past = 0;
  if (n_new > 0) tf.forward(toks, start, logits.data());
  n_past = start + n_new;
  n_prompt = n_new;

  tok::Utf8Streamer streamer;
  auto finish = [&](StopReason r) {
    std::string tail = streamer.flush();
    if (!tail.empty()) on_token(tail);  // already stopping: ignore return value
    return r;
  };
  // Invariant: n_past < n_ctx whenever a token is sampled, so every sampled
  // non-EOS token is forwarded (cached) before the next stop check.
  while (true) {
    const int32_t next = sampler.sample(logits);
    if (m.tokenizer.is_eog(next)) return finish(StopReason::Eos);  // EOS is never forwarded
    std::string piece = streamer.push(m.tokenizer.decode_token(next));
    ++n_gen;
    const bool cancel = !piece.empty() && !on_token(piece);
    const int32_t one[1] = {next};
    const int64_t pos = n_past;
    n_past = 0;
    tf.forward(std::span<const int32_t>(one, 1), pos, logits.data());
    n_past = pos + 1;
    if (cancel) {
      (void)streamer.flush();  // drop held bytes: the caller asked to stop
      return StopReason::Cancelled;
    }
    if (n_gen >= gp.max_tokens) return finish(StopReason::MaxTokens);
    if (n_past >= m.n_ctx) return finish(StopReason::ContextFull);  // total tokens == n_ctx
  }
}

StopReason Session::generate(std::span<const PromptPart> parts, const GenParams& gp,
                             const std::function<bool(std::string_view)>& on_token) {
  ++impl_->n_calls;
  std::vector<tok::Part> tp;
  bool any_text = false;  // an all-empty prompt stays empty (no lone bos) so "empty prompt" still throws
  for (const PromptPart& p : parts) {
    tp.push_back({p.text, p.special});
    any_text |= !p.text.empty();
  }
  return impl_->run(impl_->m.tokenizer.encode_parts(tp, !gp.append && any_text), gp, on_token);
}

StopReason Session::generate(std::string_view prompt, const GenParams& gp,
                             const std::function<bool(std::string_view)>& on_token) {
  ++impl_->n_calls;
  const tok::Part part{prompt, true};
  std::vector<int32_t> toks = impl_->m.tokenizer.encode_parts(std::span(&part, 1), !gp.append && !prompt.empty());
  if (!gp.append) {  // M0-M3 contract: a fresh prompt that cannot be used throws
    if (toks.empty()) throw Error("empty prompt");
    if (static_cast<int64_t>(toks.size()) >= impl_->m.n_ctx)
      throw Error("prompt has " + std::to_string(toks.size()) + " tokens but context is " +
                  std::to_string(impl_->m.n_ctx));
  }
  return impl_->run(toks, gp, on_token);
}

}  // namespace nibble
