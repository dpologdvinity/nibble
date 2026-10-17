#include <cstdio>
#include <cstring>

#include "commands.h"
#include "nibble/error.h"

namespace {
void usage() {
  std::fputs("usage: nibble <subcommand> [args]\n\nsubcommands:\n  info <model.gguf>   print GGUF metadata and tensors\n  run -m <model.gguf> (-p <prompt> | -f <file>) [-n N] [-t N] [-c N]   generate text\n", stderr);
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    usage();
    return 2;
  }
  try {
    if (std::strcmp(argv[1], "info") == 0) return nibble::cli::cmd_info(argc - 1, argv + 1);
    if (std::strcmp(argv[1], "run") == 0) return nibble::cli::cmd_run(argc - 1, argv + 1);
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
