// `nibble chat`: multi-turn chat, one user message per stdin line. Public API only.
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>

#include "args.h"
#include "commands.h"
#include "nibble/nibble.h"

namespace nibble::cli {
namespace {

// Prints the `chat` usage line to stderr.
void usage() {
  std::fputs(
      "usage: nibble chat -m <model.gguf> [--system <text>] [-n <max_tokens per reply=512>] "
      "[-t <threads>] [-c <n_ctx>] [--temp <f=0.8>] [--top-k <n=40>] [--top-p <f=0.95>] "
      "[--seed <u64=random>]\n",
      stderr);
}

}  // namespace

// Reads one user message per stdin line (CRLF tolerated, blank lines skipped), sends it through Chat and
// streams the reply. The "> " prompt is printed to stderr only when stdin is a terminal, so piped input
// produces clean output. Ends at EOF or when the context fills up.
int cmd_chat(int argc, char** argv) {
  std::string system;
  CommonArgs ca;
  ca.gp.max_tokens = 512;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const bool known = is_common_flag(a) || a == "--system";
    if (!known || i + 1 >= argc) {
      std::fprintf(stderr, known ? "error: flag '%s' needs a value\n" : "error: unknown argument '%s'\n", a.c_str());
      usage();
      return 2;
    }
    const char* v = argv[++i];
    if (a == "--system") {
      system = v;
      continue;
    }
    if (parse_common(a, v, ca) == FlagResult::BadValue) {
      std::fprintf(stderr, "error: bad value '%s' for %s\n", v, a.c_str());
      usage();
      return 2;
    }
  }
  if (ca.model_path.empty()) {
    std::fputs("error: need -m\n", stderr);
    usage();
    return 2;
  }

  Model model = Model::load(ca.model_path, ca.mp);
  Session session(model);
  Chat chat(session, system);

  const bool tty = isatty(STDIN_FILENO) != 0;
  std::string line;
  for (;;) {
    if (tty) {
      std::fputs("> ", stderr);
      std::fflush(stderr);
    }
    if (!std::getline(std::cin, line)) break;                   // EOF; a last line without '\n' is still returned
    if (!line.empty() && line.back() == '\r') line.pop_back();  // CRLF input
    if (line.empty()) continue;                                 // a blank line is not a user message: skip it

    ReplyTimer timer;
    const StopReason r = chat.send(line, ca.gp, [&](std::string_view piece) {
      timer.on_piece();
      std::fwrite(piece.data(), 1, piece.size(), stdout);
      std::fflush(stdout);
      return true;
    });
    std::fputc('\n', stdout);
    std::fflush(stdout);
    timer.print(r, session);
    if (r == StopReason::ContextFull) {
      std::fputs("[context full]\n", stderr);
      return 0;
    }
  }
  return 0;
}

}  // namespace nibble::cli
