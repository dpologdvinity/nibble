#pragma once
// Flags shared by `nibble run` and `nibble chat`. Public API only.
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

}  // namespace nibble::cli
