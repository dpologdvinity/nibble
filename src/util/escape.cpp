// Implementation of escape_control (see util/escape.h).
#include "util/escape.h"

namespace nibble {

std::string escape_control(std::string_view s) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string out;
  out.reserve(s.size());
  for (char ch : s) {
    const auto c = static_cast<unsigned char>(ch);
    if (c == '\n') {
      out += "\\n";
    } else if (c == '\t') {
      out += "\\t";
    } else if (c < 0x20 || c == 0x7F) {
      out += "\\x";
      out += kHex[c >> 4];
      out += kHex[c & 0xF];
    } else {
      out += ch;
    }
  }
  return out;
}

}  // namespace nibble
