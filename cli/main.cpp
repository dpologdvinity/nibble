// Entry point of the `nibble` command-line tool: dispatches to the subcommands (info, run, chat, bench) and
// turns exceptions into "error: <message>" on stderr with exit status 1 (2 for usage errors).
#include <cstdio>
#include <cstring>

#include "commands.h"
#include "nibble/error.h"
#include "nibble_git.h"  // generated at build time

namespace {
// Prints the top-level help text to `out`: stderr for usage errors, stdout when help was asked for.
void usage(std::FILE* out = stderr) {
  std::fputs(
      "usage: nibble <subcommand> [args]\n\nsubcommands:\n  info <model.gguf>   print GGUF metadata and tensors\n  run "
      "-m <model.gguf> (-p <prompt> | -f <file>) [-n N] [-t N] [-c N] [--temp F] [--top-k N] [--top-p F] [--seed N]   "
      "generate text\n  chat -m <model.gguf> [--system <text>] [-n N] [-t N] [-c N] [sampling flags]   chat, one "
      "message per stdin line\n  bench -m <model.gguf> [-t list] [-p N] [-n N] [-r N] [--mem] [--out file.jsonl]   "
      "pp/tg throughput\n",
      out);
}
}  // namespace

// Exit codes: 0 success, 1 runtime error (bad model file, I/O), 2 usage error (unknown subcommand or flag).
// `--help`, `-h` and `help` print the usage to stdout and succeed; `--version` prints the build's git SHA.
int main(int argc, char** argv) {
  if (argc < 2) {
    usage();
    return 2;
  }
  if (std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0 || std::strcmp(argv[1], "help") == 0) {
    usage(stdout);
    return 0;
  }
  if (std::strcmp(argv[1], "--version") == 0) {
    std::printf("nibble %s\n", NIBBLE_GIT_SHA);
    return 0;
  }
  try {
    if (std::strcmp(argv[1], "info") == 0) return nibble::cli::cmd_info(argc - 1, argv + 1);
    if (std::strcmp(argv[1], "run") == 0) return nibble::cli::cmd_run(argc - 1, argv + 1);
    if (std::strcmp(argv[1], "chat") == 0) return nibble::cli::cmd_chat(argc - 1, argv + 1);
    if (std::strcmp(argv[1], "bench") == 0) return nibble::cli::cmd_bench(argc - 1, argv + 1);
  } catch (const nibble::Error& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  std::fprintf(stderr, "error: unknown subcommand '%s'\n", argv[1]);
  usage();
  return 2;
}
