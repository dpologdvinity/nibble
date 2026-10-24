#pragma once
// Public API for nibble: load a GGUF model, open a Session over it and generate text, optionally through a
// Chat that applies the model's chat template.
//
//   nibble::Model model = nibble::Model::load("qwen2.5-0.5b-instruct-q4_k_m.gguf");
//   nibble::Session session(model);
//   nibble::Chat chat(session);
//   chat.send("What is the capital of France?", {}, [](std::string_view piece) { std::cout << piece; return true; });
//
// Everything is synchronous and single-threaded from the caller's point of view (the engine uses its own
// worker threads internally). Failures that depend on input or on a file throw nibble::Error.
// Include only this header (plus nibble/error.h, pulled in here).
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "nibble/error.h"

namespace nibble {

// Options for Model::load.
// Model::load throws Error for negative values (before opening the file) and for
// n_ctx above the model's context_length.
struct ModelParams {
  int64_t n_ctx = 0;  // 0 = auto: min(4096, model context length)
  int n_threads = 0;  // 0 = auto: the fastest CPUs if hybrid, else 2/3 of hardware_concurrency.
                      // The hybrid count is of logical CPUs, so SMT siblings are included.
};

// Sampling and stopping options for one generate/send call.
struct GenParams {
  int max_tokens = 256;  // <= 0: return MaxTokens at once without prefilling (append = false still resets n_past to 0)
  // Sampling order: top-k, then temperature-scaled softmax, then top-p. llama.cpp's
  // default chain is top-k, top-p, min-p (0.05), temperature, so the same values
  // select a different nucleus than in llama.cpp.
  float temperature = 0.8f;  // 0 = greedy (argmax, lowest id on ties)
  int top_k = 40;            // <= 0 or >= n_vocab: disabled
  float top_p = 0.95f;       // >= 1: disabled
  // nullopt = random (std::random_device). With a seed, call number i on a Session
  // (counting every generate call from 0, including ones that throw) samples from
  // splitmix64(seed ^ i). So a fresh Session given the same seed and the same call
  // sequence reproduces its output, while successive calls, such as the turns of a
  // seeded Chat, draw different streams.
  std::optional<uint64_t> seed = std::nullopt;
  bool append = false;  // continue after the previous call's context instead of resetting
};

// One piece of a prompt. special = true parses control-token text (e.g.
// "<|im_start|>") as the control token; false tokenizes it as plain text.
struct PromptPart {
  std::string_view text;
  bool special;
};

// Why a generate/send call returned.
//   Eos         the model produced an end-of-generation token (not delivered to the callback)
//   MaxTokens   GenParams::max_tokens tokens were produced
//   ContextFull the context window (n_ctx) is full or the new tokens do not fit
//   Cancelled   the on_token callback returned false
enum class StopReason { Eos, MaxTokens, ContextFull, Cancelled };

// A loaded model: the memory-mapped weights, tokenizer, hyperparameters and compute backend. Heavy to
// create, cheap to keep; build one per model file and share it with one Session at a time.
// Move-only. Throws nibble::Error on any load failure.
class Model {
 public:
  // Opens and validates the GGUF file at path, maps its weights and builds the tokenizer. Throws Error with a
  // message naming the problem (unreadable file, unsupported architecture or tensor type, missing tensor,
  // inconsistent shapes, out of memory).
  static Model load(const std::string& path, ModelParams = {});
  Model(Model&&) noexcept;  // leaves the source in a moved-from state
  Model& operator=(Model&&) noexcept;
  ~Model();
  Model(const Model&) = delete;  // a model owns a file mapping and thread pool: not copyable
  Model& operator=(const Model&) = delete;

  // Throw Error("use of moved-from Model") on a moved-from Model (as does Session(Model&)).
  // GGUF `general.architecture` of the model, e.g. "qwen2" or "llama".
  const std::string& architecture() const;
  // Context window in tokens that Sessions over this model use (ModelParams::n_ctx after auto-sizing).
  int64_t n_ctx() const;

 private:
  Model();
  struct Impl;
  std::unique_ptr<Impl> impl_;
  friend class Session;
};

// A Session must not outlive its Model. v1 allows ONE active Session per Model
// at a time (the backend's attention scratch is shared).
class Chat;

// One conversation state over a Model: the KV cache plus the sampler. generate() runs prefill and decoding.
class Session {
 public:
  // Allocates the KV cache for model.n_ctx() tokens. Throws Error if that does not fit in memory or if the
  // model was moved from.
  explicit Session(Model&);
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  // Encodes the parts (in order, as one string: adjacent plain text is tokenized together, so
  // "a" + "b" equals "ab"; the exception is that a special token's text never spans
  // parts, so Control text split across two parts stays plain) and samples with
  // GenParams. append = false starts at position 0 (n_past() becomes 0); append = true
  // prefills the new tokens at n_past(). Every sampled token except EOS is forwarded
  // into the KV cache, so after any return (not a throw)
  //   n_past() == start + n_prompt_tokens() + n_generated().
  // Only EOS is never forwarded (it is not delivered either): callers re-supply any
  // closing markup. on_token receives valid UTF-8 pieces; return false to cancel (held
  // partial UTF-8 bytes are then dropped, not delivered; their token is cached).
  //
  // Returns without prefilling or sampling when
  //  - max_tokens <= 0 (MaxTokens), or
  //  - the new tokens do not fit: start + n_new >= n_ctx(), where start is n_past()
  //    if append else 0 (ContextFull; no throw, in chat this is expected).
  // In both cases n_past() is unchanged for append = true and 0 for append = false.
  // Otherwise ContextFull is returned once start + n_prompt_tokens() + n_generated()
  // == n_ctx() (the cache is then full).
  //
  // With append = false, the BOS token is prepended when the model defines one and wants it
  // (TinyLlama, a SentencePiece model, and Llama 3, a byte-level BPE one), unless the prompt
  // already starts with it; append = true adds none. Qwen has no BOS. An all-empty prompt
  // gets no BOS.
  //
  // Empty input: zero new tokens (an empty parts list, or parts that encode to
  // nothing) with append = false throws Error("empty prompt"). With append = true it
  // continues generating from the cached context, unless n_past() == 0, which also
  // throws Error("empty prompt").
  // Invalid sampling params throw Error before any state changes. Every validation
  // throw (empty prompt and bad params alike) leaves n_past() unchanged, including
  // with append = false.
  StopReason generate(std::span<const PromptPart> parts, const GenParams&,
                      const std::function<bool(std::string_view piece)>& on_token);
  // One part with special = true. Unlike the overload above, with append = false
  // this keeps the M0-M3 contract: throws Error on an empty prompt or on a prompt
  // with n_ctx() or more tokens (instead of returning ContextFull).
  StopReason generate(std::string_view prompt, const GenParams&,
                      const std::function<bool(std::string_view piece)>& on_token);
  int64_t n_prompt_tokens() const;  // new tokens prefilled by the last generate call
  int64_t n_generated() const;      // tokens sampled by the last generate call (excl. EOS)
  int64_t n_past() const;           // tokens currently in the KV cache

 private:
  friend class Chat;
  const std::string& architecture() const;  // of the Session's Model
  std::string chat_template() const;        // tokenizer.chat_template, empty if absent
  uint64_t generations() const;             // generate calls so far, including ones that threw
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Multi-turn chat over a Session, formatted with the model's chat template
// (ChatML, Llama 3 or Zephyr, chosen from the model's tokenizer.chat_template). The KV cache is
// kept between turns, so each send prefills only the new turn's tokens. The Chat must not
// outlive its Session. Not thread-safe; the Session's one-active-Session-per-Model rule
// applies. Do not use the Session directly while a Chat drives it: the next send then
// throws Error("chat context lost"). To start over (e.g. after ContextFull), construct a
// new Chat. Move-only.
class Chat {
 public:
  // Throws Error("no chat template for architecture <a>") if the model has no
  // known chat format. An empty system prompt selects the model's default
  // (Qwen: "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
  // Llama 3: empty; Zephyr: no system block).
  explicit Chat(Session&, std::string system_prompt = {});
  Chat(Chat&&) noexcept;
  Chat& operator=(Chat&&) noexcept;
  ~Chat();
  Chat(const Chat&) = delete;
  Chat& operator=(const Chat&) = delete;

  // Sends one user message and streams the assistant reply. User and system text
  // are tokenized as plain text, so they cannot inject control tokens. The first
  // turn (including the system prompt) starts the Session fresh; later turns append.
  // GenParams::append is ignored. Once a send returns ContextFull (the turn did not
  // fit, or the reply filled the context), every later send returns ContextFull at
  // once without calling on_token. Throws Error("chat context lost") if the
  // Session's cache no longer matches the conversation (it was used directly, or a
  // forward threw). An exception from on_token propagates; the prompt and the
  // earlier reply tokens stay cached (the token whose delivery threw does not)
  // and the conversation continues.
  // Throws Error on a moved-from Chat.
  StopReason send(std::string_view user, const GenParams&, const std::function<bool(std::string_view piece)>& on_token);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace nibble
