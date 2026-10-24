// Tests for the Unicode helpers in src/tokenizer/unicode.h: character classes, strict UTF-8 decoding, the
// GPT-2 byte alphabet and the streaming UTF-8 splitter.
#include <catch2/catch_test_macros.hpp>
#include <set>
#include <string>

#include "nibble/error.h"
#include "tokenizer/unicode.h"

using namespace nibble::tok;

// Letters from several scripts are letters; digits, space, punctuation and emoji are not.
TEST_CASE("unicode letters", "[unicode]") {
  for (char32_t c : {U'a', U'Z', U'é', U'中', U'क', U'א'}) REQUIRE(is_letter(c));
  for (char32_t c : {U'5', U' ', U'!', U'\U0001F600'}) REQUIRE_FALSE(is_letter(c));
}

// Decimal digits, Arabic-Indic digits, fractions and Roman numerals count as numbers (general category N*).
TEST_CASE("unicode numbers", "[unicode]") {
  for (char32_t c : {U'0', U'9', U'٣', U'½', U'Ⅵ'}) REQUIRE(is_number(c));
  REQUIRE_FALSE(is_number(U'a'));
}

// The Unicode White_Space set, including NEL, NBSP and the space separators; control character 0x1C and ZWSP
// are deliberately not whitespace.
TEST_CASE("unicode whitespace", "[unicode]") {
  for (char32_t c : {0x9, 0xA, 0xB, 0xC, 0xD, 0x20, 0x85, 0xA0, 0x1680, 0x2000, 0x2005, 0x200A,
                     0x2028, 0x2029, 0x202F, 0x205F, 0x3000})
    REQUIRE(is_whitespace(c));
  for (char32_t c : {0x1C, 0x200B, 0x61}) REQUIRE_FALSE(is_whitespace(c));
}

// Decodes a mix of 1-, 2- and 4-byte characters, rejects truncated, overlong, surrogate, out-of-range and
// stray-continuation input, and reports the byte offset of the first error.
TEST_CASE("utf8 decode", "[unicode]") {
  auto v = utf8_decode("h\xC3\xA9\xF0\x9F\x98\x80");
  REQUIRE(v == std::vector<char32_t>{U'h', 0xE9, 0x1F600});
  for (const char* bad : {"\xC3", "\xC0\x80", "\xED\xA0\x80", "\xF4\x90\x80\x80", "\xFF", "\xE0\x80\x80",
                          "\xF0\x80\x80\x80", "\x80"})
    REQUIRE_THROWS_AS(utf8_decode(bad), nibble::Error);
  try {
    utf8_decode("ab\xFF");
    FAIL();
  } catch (const nibble::Error& e) {
    REQUIRE(std::string(e.what()).find("byte 2") != std::string::npos);
  }
}

// Encoding then decoding returns the same code point for each length class.
TEST_CASE("utf8 append roundtrip", "[unicode]") {
  for (char32_t c : {U'a', char32_t(0xE9), char32_t(0x4E2D), char32_t(0x1F600)}) {
    std::string s;
    utf8_append(s, c);
    REQUIRE(utf8_decode(s) == std::vector<char32_t>{c});
  }
}

// The GPT-2 byte alphabet: 256 distinct single-character spellings, space is U+0120 and newline U+010A,
// printable ASCII maps to itself, and the reverse map inverts it (unmapped code points give -1).
TEST_CASE("byte map", "[unicode]") {
  const auto& m = byte_to_unicode();
  std::set<std::string> uniq(m.begin(), m.end());
  REQUIRE(uniq.size() == 256);
  REQUIRE(m[0x20] == "\xC4\xA0");
  REQUIRE(m[0x0A] == "\xC4\x8A");
  REQUIRE(m['a'] == "a");
  for (int b = 0; b < 256; ++b) {
    auto cp = utf8_decode(m[b]);
    REQUIRE(cp.size() == 1);
    REQUIRE(unicode_to_byte(cp[0]) == b);
  }
  REQUIRE(unicode_to_byte(0x4E2D) == -1);
  REQUIRE(unicode_to_byte(0x7F) == -1);
}

// A 4-byte character fed one byte at a time is emitted only once complete; flush() turns a dangling prefix
// into one replacement character and is idempotent.
TEST_CASE("utf8 streamer", "[unicode]") {
  Utf8Streamer s;
  const std::string e = "\xF0\x9F\x98\x80";
  REQUIRE(s.push(e.substr(0, 1)) == "");
  REQUIRE(s.push(e.substr(1, 1)) == "");
  REQUIRE(s.push(e.substr(2, 1)) == "");
  REQUIRE(s.push(e.substr(3, 1)) == e);
  REQUIRE(s.flush() == "");
  REQUIRE(s.push("a\xF0\x9F") == "a");
  REQUIRE(s.flush() == "\xEF\xBF\xBD");
  REQUIRE(s.flush() == "");
}

// Invalid leads, bad continuations, overlong/surrogate/out-of-range prefixes become U+FFFD one lead byte at a
// time, without swallowing the valid bytes that follow.
TEST_CASE("utf8 streamer invalid input", "[unicode]") {
  const std::string R = "\xEF\xBF\xBD";
  auto run = [&](std::initializer_list<const char*> parts) {
    Utf8Streamer s;
    std::string o;
    for (auto p : parts) o += s.push(p);
    return o + s.flush();
  };
  REQUIRE(run({"\xFF", "a"}) == R + "a");
  REQUIRE(run({"\xF0", "a"}) == R + "a");
  REQUIRE(run({"\xC3\x28"}) == R + "(");
  REQUIRE(run({"\x80"}) == R);
  REQUIRE(run({"\xE0", "\x80"}) == R + R);
  REQUIRE(run({"\xED\xA0"}) == R + R);
  REQUIRE(run({"\xF4\x90"}) == R + R);
  REQUIRE(run({"\xC0\xC1\xF5"}) == R + R + R);
}

// Randomized: for arbitrary byte soup split into random chunks the output is always valid UTF-8, and valid
// text round-trips unchanged regardless of how it is chunked.
TEST_CASE("utf8 streamer property", "[unicode]") {
  uint32_t st = 12345;
  auto rnd = [&] { st = st * 1664525u + 1013904223u; return st >> 8; };
  const char alpha[] = {'a', '\x80', '\xBF', '\xC2', '\xC3', '\xE0', '\xED', '\xF0', '\xF4', '\xFF', '\x9F', '\xA0'};
  for (int t = 0; t < 10000; ++t) {
    std::string in;
    size_t len = rnd() % 12;
    for (size_t i = 0; i < len; ++i) in += (rnd() % 3 == 0) ? char(rnd() & 0xFF) : alpha[rnd() % sizeof alpha];
    Utf8Streamer s;
    std::string out;
    for (size_t p = 0; p < in.size();) {
      size_t n = 1 + rnd() % 4;
      out += s.push(std::string_view(in).substr(p, n));
      p += n;
    }
    out += s.flush();
    REQUIRE_NOTHROW(utf8_decode(out));
    // valid input round-trips
    std::string v;
    for (size_t i = 0, n = rnd() % 6; i < n; ++i) {
      char32_t c = rnd() % 0x110000;
      if (c >= 0xD800 && c <= 0xDFFF) c = 0x41;
      utf8_append(v, c);
    }
    Utf8Streamer s2;
    std::string o2;
    for (size_t p = 0; p < v.size();) {
      size_t n = 1 + rnd() % 3;
      o2 += s2.push(std::string_view(v).substr(p, n));
      p += n;
    }
    o2 += s2.flush();
    REQUIRE(o2 == v);
  }
}
