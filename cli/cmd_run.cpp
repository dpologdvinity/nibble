// `nibble run`: stream generated text from a GGUF model. Public API only.
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

// Prints the `run` usage line to stderr.
void usage() {
  std::fputs(
      "usage: nibble run -m <model.gguf> (-p <prompt> | -f <prompt-file>) "
      "[-n <max_tokens=256>] [-t <threads>] [-c <n_ctx>] "
      "[--temp <f=0.8>] [--top-k <n=40>] [--top-p <f=0.95>] [--seed <u64=random>]\n",
      stderr);
}

// Whole-file read for -f; a prompt file is passed to the model byte for byte.
std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw Error("cannot read prompt file '" + path + "'");
  std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (f.bad()) throw Error("error reading prompt file '" + path + "'");
  return s;
}

}  // namespace

// Flow: parse flags (exit 2 on any problem), load the model, open a Session, stream the reply to stdout
// as it is generated, then print timing statistics to stderr. The prompt is sent as raw text with special
// token parsing enabled, i.e. no chat template is applied (use `chat` for that).
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
    if (a == "-p") {
      prompt = v;
      have_p = true;
      continue;
    }
    if (a == "-f") {
      prompt_file = v;
      have_f = true;
      continue;
    }
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

  ReplyTimer timer;
  const StopReason r = session.generate(prompt, ca.gp, [&](std::string_view piece) {
    timer.on_piece();
    std::fwrite(piece.data(), 1, piece.size(), stdout);
    std::fflush(stdout);
    return true;
  });
  timer.print(r, session);
  return 0;
}

}  // namespace nibble::cli
