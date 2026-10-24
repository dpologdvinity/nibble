#pragma once
// The single exception type nibble throws for anything that can go wrong at run time because of input the
// caller or a file supplied: a corrupt or unsupported model file, bad parameters, an empty prompt, an
// out-of-memory condition while loading. Programmer errors (violated internal invariants) are not Errors:
// they abort through NIBBLE_DCHECK in debug builds (see src/debug_check.h).
#include <stdexcept>

namespace nibble {
// what() carries a self-contained, human-readable message. Messages that quote text from a model file pass
// it through escape_control first, so control bytes can never reach a terminal.
class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};
}  // namespace nibble
