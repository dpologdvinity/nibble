// `nibble run`: stream generated text from a GGUF model. Public API only.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>

#include "args.h"
#include "commands.h"
#include "nibble/nibble.h"

namespace nibble::cli {
namespace {

void usage() {
  std::fputs(
      "usage: nibble run -m <model.gguf> (-p <prompt> | -f <prompt-file>) "
      "[-n <max_tokens=256>] [-t <threads>] [-c <n_ctx>] "
      "[--temp <f=0.8>] [--top-k <n=40>] [--top-p <f=0.95>] [--seed <u64=random>]\n",
      stderr);
}

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw Error("cannot read prompt file '" + path + "'");
  std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (f.bad()) throw Error("error reading prompt file '" + path + "'");
  return s;
}

}  // namespace

int cmd_run(int argc, char** argv) {
  std::string prompt, prompt_file;
  bool have_p = false, have_f = false;
  CommonArgs ca;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const bool known = is_common_flag(a) || a == "-p" || a == "-f";
    if (!known || i + 1 >= argc) {
      std::fprintf(stderr, known ? "error: flag '%s' needs a value\n" : "error: unknown argument '%s'\n", a.c_str());
      usage();
      return 2;
    }
    const char* v = argv[++i];
    if (a == "-p") { prompt = v; have_p = true; continue; }
    if (a == "-f") { prompt_file = v; have_f = true; continue; }
    if (parse_common(a, v, ca) == FlagResult::BadValue) {
      std::fprintf(stderr, "error: bad value '%s' for %s\n", v, a.c_str());
      usage();
      return 2;
    }
  }
  const std::string& model_path = ca.model_path;
  if (model_path.empty() || have_p == have_f) {
    std::fputs("error: need -m and exactly one of -p / -f\n", stderr);
    usage();
    return 2;
  }
  if (have_f) prompt = read_file(prompt_file);  // bytes passed through unchanged

  Model model = Model::load(model_path, ca.mp);
  Session session(model);

  using clock = std::chrono::steady_clock;
  const auto start = clock::now();
  // The API has no prefill/decode split. Approximation: prefill time = start to the
  // first on_token call (or to return, if EOS arrives before any piece); decode time
  // = the remainder. Tokenization and the first sampled token count as prefill.
  clock::time_point first = start;
  bool seen = false;
  const StopReason r = session.generate(prompt, ca.gp, [&](std::string_view piece) {
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
  print_stats(r, np, pre_ms, ng, gen_ms);
  return 0;
}

}  // namespace nibble::cli
