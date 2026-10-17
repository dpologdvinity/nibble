#include <catch2/catch_test_macros.hpp>
#include <set>
#include <string>

#include "nibble/error.h"
#include "tokenizer/unicode.h"

using namespace nibble::tok;

TEST_CASE("unicode letters", "[unicode]") {
  for (char32_t c : {U'a', U'Z', U'é', U'中', U'क', U'א'}) REQUIRE(is_letter(c));
  for (char32_t c : {U'5', U' ', U'!', U'\U0001F600'}) REQUIRE_FALSE(is_letter(c));
}

TEST_CASE("unicode numbers", "[unicode]") {
  for (char32_t c : {U'0', U'9', U'٣', U'½', U'Ⅵ'}) REQUIRE(is_number(c));
  REQUIRE_FALSE(is_number(U'a'));
}

TEST_CASE("unicode whitespace", "[unicode]") {
  for (char32_t c : {0x9, 0xA, 0xB, 0xC, 0xD, 0x20, 0x85, 0xA0, 0x1680, 0x2000, 0x2005, 0x200A,
                     0x2028, 0x2029, 0x202F, 0x205F, 0x3000})
    REQUIRE(is_whitespace(c));
  for (char32_t c : {0x1C, 0x200B, 0x61}) REQUIRE_FALSE(is_whitespace(c));
}

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

TEST_CASE("utf8 append roundtrip", "[unicode]") {
  for (char32_t c : {U'a', char32_t(0xE9), char32_t(0x4E2D), char32_t(0x1F600)}) {
    std::string s;
    utf8_append(s, c);
    REQUIRE(utf8_decode(s) == std::vector<char32_t>{c});
  }
}

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
