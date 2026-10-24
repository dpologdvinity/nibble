#pragma once
// Entry points of the CLI subcommands. Each receives the argument vector starting at the subcommand name
// (argv[0] is e.g. "run"), and returns the process exit status.

namespace nibble::cli {

// Each command throws nibble::Error on user-facing failure; main() reports it.
int cmd_info(int argc, char** argv);   // print GGUF metadata and the tensor table
int cmd_run(int argc, char** argv);  // one-shot text generation; returns 2 (after printing usage) on bad flags
int cmd_chat(int argc, char** argv);  // interactive multi-turn chat; same
int cmd_bench(int argc, char** argv);  // pp/tg throughput benchmark; same

}  // namespace nibble::cli
