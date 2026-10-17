#pragma once
#include <array>
#include <string>
#include <string_view>
#include <vector>

namespace nibble::tok {

bool is_letter(char32_t c);      // general category L*
bool is_number(char32_t c);      // general category N*
bool is_whitespace(char32_t c);  // Unicode White_Space property

// Throws nibble::Error("invalid UTF-8 at byte N") on malformed input.
std::vector<char32_t> utf8_decode(std::string_view s);
void utf8_append(std::string& out, char32_t c);

// GPT-2 byte <-> printable code point map.
const std::array<std::string, 256>& byte_to_unicode();
int unicode_to_byte(char32_t c);  // -1 if not a mapped code point

// Holds back an incomplete trailing UTF-8 sequence between pushes.
class Utf8Streamer {
 public:
  std::string push(std::string_view bytes);
  std::string flush();  // held bytes -> U+FFFD, or "" if none held

 private:
  std::string held_;
};

}  // namespace nibble::tok
