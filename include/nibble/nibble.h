#pragma once
// Public API for nibble. Include only this header (plus nibble/error.h, pulled in here).
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "nibble/error.h"

namespace nibble {

// Model::load throws Error for negative values (before opening the file) and for
// n_ctx above the model's context_length.
struct ModelParams {
  int64_t n_ctx = 0;   // 0 = auto: min(4096, model context length)
  int n_threads = 0;   // 0 = auto: hardware_concurrency
};

struct GenParams {
  int max_tokens = 256;  // <= 0: return MaxTokens at once without prefilling (append = false still resets n_past to 0)
  float temperature = 0.8f;  // 0 = greedy (argmax, lowest id on ties)
  int top_k = 40;            // <= 0 or >= n_vocab: disabled
  float top_p = 0.95f;       // >= 1: disabled
  std::optional<uint64_t> seed = std::nullopt;  // nullopt = random (std::random_device)
  bool append = false;  // continue after the previous call's context instead of resetting
};

// One piece of a prompt. special = true parses control-token text (e.g.
// "<|im_start|>") as the control token; false tokenizes it as plain text.
struct PromptPart {
  std::string_view text;
  bool special;
};

enum class StopReason { Eos, MaxTokens, ContextFull, Cancelled };

// Move-only. Throws nibble::Error on any load failure.
class Model {
 public:
  static Model load(const std::string& path, ModelParams = {});
  Model(Model&&) noexcept;
  Model& operator=(Model&&) noexcept;
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  // Throw Error("use of moved-from Model") on a moved-from Model (as does Session(Model&)).
  const std::string& architecture() const;
  int64_t n_ctx() const;

 private:
  Model();
  struct Impl;
  std::unique_ptr<Impl> impl_;
  friend class Session;
};

// A Session must not outlive its Model. v1 allows ONE active Session per Model
// at a time (the backend's attention scratch is shared).
class Session {
 public:
  explicit Session(Model&);
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  // Encodes the parts (in order) and samples with GenParams. append = false starts
  // at position 0 (n_past() becomes 0); append = true prefills the new tokens at
  // n_past(). Every sampled token except EOS is forwarded into the KV cache, so
  // after any return (not a throw)
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
  // Empty input: zero new tokens (an empty parts list, or parts that encode to
  // nothing) with append = false throws Error("empty prompt"). With append = true it
  // continues generating from the cached context, unless n_past() == 0, which also
  // throws Error("empty prompt").
  // Invalid sampling params throw Error before any state changes.
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
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace nibble
