// `nibble run`: stream generated text from a GGUF model. Public API only.
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>

#include "commands.h"
#include "nibble/nibble.h"

namespace nibble::cli {
namespace {

void usage() {
  std::fputs(
      "usage: nibble run -m <model.gguf> (-p <prompt> | -f <prompt-file>) "
      "[-n <max_tokens=256>] [-t <threads>] [-c <n_ctx>]\n",
      stderr);
}

template <class T>
bool parse_num(const char* s, T& out) {
  const char* end = s + std::strlen(s);
  auto [p, ec] = std::from_chars(s, end, out);
  return ec == std::errc() && p == end && p != s;
}

const char* stop_name(StopReason r) {
  switch (r) {
    case StopReason::Eos: return "eos";
    case StopReason::MaxTokens: return "max_tokens";
    case StopReason::ContextFull: return "context_full";
    case StopReason::Cancelled: return "cancelled";
  }
  return "unknown";
}

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw Error("cannot read prompt file '" + path + "'");
  std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (f.bad()) throw Error("error reading prompt file '" + path + "'");
  return s;
}

double rate(int64_t n, double ms) { return ms > 0 ? static_cast<double>(n) * 1000.0 / ms : 0.0; }

}  // namespace

int cmd_run(int argc, char** argv) {
  std::string model_path, prompt, prompt_file;
  bool have_p = false, have_f = false;
  GenParams gp;
  gp.temperature = 0;  // greedy, deterministic output. Task 7: flags
  ModelParams mp;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const bool known = a == "-m" || a == "-p" || a == "-f" || a == "-n" || a == "-t" || a == "-c";
    if (!known || i + 1 >= argc) {
      std::fprintf(stderr, known ? "error: flag '%s' needs a value\n" : "error: unknown argument '%s'\n", a.c_str());
      usage();
      return 2;
    }
    const char* v = argv[++i];
    bool ok = true;
    if (a == "-m") model_path = v;
    else if (a == "-p") { prompt = v; have_p = true; }
    else if (a == "-f") { prompt_file = v; have_f = true; }
    else if (a == "-n") ok = parse_num(v, gp.max_tokens) && gp.max_tokens > 0;
    else if (a == "-t") ok = parse_num(v, mp.n_threads) && mp.n_threads > 0;
    else ok = parse_num(v, mp.n_ctx) && mp.n_ctx > 0;
    if (!ok) {
      std::fprintf(stderr, "error: bad value '%s' for %s\n", v, a.c_str());
      usage();
      return 2;
    }
  }
  if (model_path.empty() || have_p == have_f) {
    std::fputs("error: need -m and exactly one of -p / -f\n", stderr);
    usage();
    return 2;
  }
  if (have_f) prompt = read_file(prompt_file);  // bytes passed through unchanged

  Model model = Model::load(model_path, mp);
  Session session(model);

  using clock = std::chrono::steady_clock;
  const auto start = clock::now();
  // The API has no prefill/decode split. Approximation: prefill time = start to the
  // first on_token call (or to return, if EOS arrives before any piece); decode time
  // = the remainder. Tokenization and the first sampled token count as prefill.
  clock::time_point first = start;
  bool seen = false;
  const StopReason r = session.generate(prompt, gp, [&](std::string_view piece) {
    if (!seen) { seen = true; first = clock::now(); }
    std::fwrite(piece.data(), 1, piece.size(), stdout);
    std::fflush(stdout);
    return true;
  });
  const auto end = clock::now();
  if (!seen) first = end;
  const double pre_ms = std::chrono::duration<double, std::milli>(first - start).count();
  const double gen_ms = std::chrono::duration<double, std::milli>(end - first).count();
  const int64_t np = session.n_prompt_tokens(), ng = session.n_generated();
  std::fprintf(stderr, "\n[%s] prompt: %lld tok, %.1f ms (%.1f tok/s) | gen: %lld tok, %.1f ms (%.1f tok/s)\n",
               stop_name(r), static_cast<long long>(np), pre_ms, rate(np, pre_ms), static_cast<long long>(ng),
               gen_ms, rate(ng, gen_ms));
  return 0;
}

}  // namespace nibble::cli
