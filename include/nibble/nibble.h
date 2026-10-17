#pragma once
// Public API for nibble. Include only this header (plus nibble/error.h, pulled in here).
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>

#include "nibble/error.h"

namespace nibble {

struct ModelParams {
  int64_t n_ctx = 0;   // 0 = auto: min(4096, model context length)
  int n_threads = 0;   // 0 = auto: hardware_concurrency
};

struct GenParams {
  int max_tokens = 256;  // sampling fields arrive in M4
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

  // Greedy decoding. Resets position to 0 each call. on_token receives valid UTF-8
  // pieces; return false to cancel. Throws Error on an empty prompt or a prompt
  // with n_ctx or more tokens. ContextFull is returned once
  // n_prompt_tokens() + n_generated() == n_ctx().
  StopReason generate(std::string_view prompt, const GenParams&,
                      const std::function<bool(std::string_view piece)>& on_token);
  int64_t n_prompt_tokens() const;  // from the last generate call
  int64_t n_generated() const;      // from the last generate call

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace nibble
