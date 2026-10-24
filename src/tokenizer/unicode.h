#pragma once
// Unicode helpers the tokenizer needs: character classes for the pre-tokenizer, strict UTF-8 decode/encode,
// the GPT-2 byte<->character alphabet used by byte-level BPE vocabularies, and an incremental UTF-8 stream
// splitter for printing generated pieces. Character-class tables are generated (tools/gen_unicode_tables.py).
#include <array>
#include <string>
#include <string_view>
#include <vector>

namespace nibble::tok {

// Character classes used by the pre-tokenizer (the \p{L}, \p{N} and \s of the GPT-2/Qwen regexes).
bool is_letter(char32_t c);      // general category L*
bool is_number(char32_t c);      // general category N*
bool is_whitespace(char32_t c);  // Unicode White_Space property

// Decodes strictly: overlong forms, surrogates, code points above U+10FFFF and truncated sequences are all
// errors. Throws nibble::Error("invalid UTF-8 at byte N") on malformed input.
std::vector<char32_t> utf8_decode(std::string_view s);
// Appends the UTF-8 encoding of code point c to out (c must be a valid scalar value).
void utf8_append(std::string& out, char32_t c);

// GPT-2 byte <-> printable code point map: byte-level BPE vocabularies spell every byte as a visible
// character (printable bytes map to themselves, the rest to U+0100 upward). byte_to_unicode gives each
// byte's UTF-8 spelling; unicode_to_byte inverts it.
const std::array<std::string, 256>& byte_to_unicode();
int unicode_to_byte(char32_t c);  // -1 if not a mapped code point

// Turns a stream of byte chunks (token pieces, which can split a multi-byte character) into chunks that are
// always valid UTF-8: an incomplete trailing sequence is held back until the next push, and invalid bytes
// become U+FFFD. Not thread-safe.
class Utf8Streamer {
 public:
  // Appends bytes and returns everything that is now complete.
  std::string push(std::string_view bytes);
  std::string flush();  // held bytes -> U+FFFD, or "" if none held

 private:
  std::string held_;
};

}  // namespace nibble::tok
