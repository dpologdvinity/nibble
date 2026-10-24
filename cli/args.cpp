// Shared CLI flag parsing and statistics output (see args.h).
#include "args.h"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace nibble::cli {
namespace {

// Parses the whole string s as a number of type T (integer or floating point). Rejects empty input,
// trailing characters and out-of-range values; returns whether it succeeded.
template <class T>
bool parse_num(const char* s, T& out) {
  const char* end = s + std::strlen(s);
  auto [p, ec] = std::from_chars(s, end, out);
  return ec == std::errc() && p == end && p != s;
}

// Tokens per second for n tokens in ms milliseconds (0 when no time elapsed).
double rate(int64_t n, double ms) { return ms > 0 ? static_cast<double>(n) * 1000.0 / ms : 0.0; }

}  // namespace

// The flags that parse_common understands; all take a value.
bool is_common_flag(const std::string& a) {
  return a == "-m" || a == "-n" || a == "-t" || a == "-c" || a == "--temp" || a == "--top-k" || a == "--top-p" ||
         a == "--seed";
}

// Parses and range-checks one flag's value into `out`: -n, -t and -c must be positive, --temp finite and
// non-negative, --top-k non-negative, --top-p in (0, 1], --seed any unsigned 64-bit number.
FlagResult parse_common(const std::string& a, const char* v, CommonArgs& out) {
  if (!is_common_flag(a)) return FlagResult::NotCommon;
  bool ok = true;
  if (a == "-m")
    out.model_path = v;
  else if (a == "-n")
    ok = parse_num(v, out.gp.max_tokens) && out.gp.max_tokens > 0;
  else if (a == "-t")
    ok = parse_num(v, out.mp.n_threads) && out.mp.n_threads > 0;
  else if (a == "-c")
    ok = parse_num(v, out.mp.n_ctx) && out.mp.n_ctx > 0;
  else if (a == "--temp")
    ok = parse_num(v, out.gp.temperature) && std::isfinite(out.gp.temperature) && out.gp.temperature >= 0;
  else if (a == "--top-k")
    ok = parse_num(v, out.gp.top_k) && out.gp.top_k >= 0;
  else if (a == "--top-p")
    ok = parse_num(v, out.gp.top_p) && out.gp.top_p > 0 && out.gp.top_p <= 1;
  else {  // --seed
    uint64_t s = 0;
    ok = parse_num(v, s);
    if (ok) out.gp.seed = s;
  }
  return ok ? FlagResult::Ok : FlagResult::BadValue;
}

// Used in the "[stop]" statistics line.
const char* stop_name(StopReason r) {
  switch (r) {
    case StopReason::Eos: return "eos";
    case StopReason::MaxTokens: return "max_tokens";
    case StopReason::ContextFull: return "context_full";
    case StopReason::Cancelled: return "cancelled";
  }
  return "unknown";
}

// Writes to stderr so the generated text on stdout stays clean for piping.
void print_stats(StopReason r, int64_t np, double pre_ms, int64_t ng, double gen_ms) {
  std::fprintf(stderr, "\n[%s] prompt: %lld tok, %.1f ms (%.1f tok/s) | gen: %lld tok, %.1f ms (%.1f tok/s)\n",
               stop_name(r), static_cast<long long>(np), pre_ms, rate(np, pre_ms), static_cast<long long>(ng), gen_ms,
               rate(ng, gen_ms));
}

}  // namespace nibble::cli
