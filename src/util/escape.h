#pragma once
#include <string>
#include <string_view>

namespace nibble {

// Makes untrusted bytes safe to print to a terminal: control bytes (< 0x20) and
// DEL (0x7F) become escapes (\n, \t, else \xHH); all other bytes pass through, so
// UTF-8 text is unchanged. Used for file-derived text in Error messages and CLI output.
std::string escape_control(std::string_view s);

}  // namespace nibble
