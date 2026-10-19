#pragma once
// Flags shared by `nibble run` and `nibble chat`. Public API only.
#include <chrono>
#include <string>

#include "nibble/nibble.h"

namespace nibble::cli {

struct CommonArgs {
  std::string model_path;
  GenParams gp;
  ModelParams mp;
};

enum class FlagResult { NotCommon, Ok, BadValue };

// True for -m -n -t -c --temp --top-k --top-p --seed (flags that take a value).
bool is_common_flag(const std::string& flag);

// Applies one common flag. NotCommon if `flag` is not one; BadValue if `value` does
// not parse or is out of range (the caller prints usage and returns 2).
FlagResult parse_common(const std::string& flag, const char* value, CommonArgs& out);

const char* stop_name(StopReason r);

// Prints the per-reply "[stop] prompt: ... | gen: ..." line to stderr.
void print_stats(StopReason r, int64_t n_prompt, double prefill_ms, int64_t n_gen, double gen_ms);

// Times one reply. The API has no prefill/decode split, so prefill is approximated
// as construction to the first on_piece call (or to print, if no piece arrived);
// decode is the remainder. Construct just before generate/send, call on_piece from
// the token callback, call print after it returns.
class ReplyTimer {
 public:
  void on_piece() {
    if (!seen_) { seen_ = true; first_ = clock::now(); }
  }
  void print(StopReason r, const Session& s) const {
    const auto end = clock::now();
    const auto first = seen_ ? first_ : end;
    const auto ms = [](auto d) { return std::chrono::duration<double, std::milli>(d).count(); };
    print_stats(r, s.n_prompt_tokens(), ms(first - start_), s.n_generated(), ms(end - first));
  }

 private:
  using clock = std::chrono::steady_clock;
  clock::time_point start_ = clock::now(), first_;
  bool seen_ = false;
};

}  // namespace nibble::cli
