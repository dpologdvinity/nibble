#pragma once

namespace nibble::cli {

// Each command throws nibble::Error on user-facing failure; main() reports it.
int cmd_info(int argc, char** argv);

}  // namespace nibble::cli
