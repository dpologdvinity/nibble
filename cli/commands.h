#pragma once

namespace nibble::cli {

// Each command throws nibble::Error on user-facing failure; main() reports it.
int cmd_info(int argc, char** argv);
int cmd_run(int argc, char** argv);  // returns 2 (after printing usage) on bad flags
int cmd_chat(int argc, char** argv);  // same

}  // namespace nibble::cli
