#include <string>

#include <catch2/catch_test_macros.hpp>

#include "escape.h"

using nibble::cli::escape_control;

TEST_CASE("cli escape_control neutralizes terminal control bytes", "[cli]") {
  const std::string out = escape_control("\x1b[31mred\x1b[0m");
  CHECK(out == "\\x1b[31mred\\x1b[0m");
  CHECK(out.find('\x1b') == std::string::npos);
  CHECK(escape_control("a\nb\tc") == "a\\nb\\tc");
  CHECK(escape_control(std::string("\x00\x01\x1f\x7f", 4)) == "\\x00\\x01\\x1f\\x7f");
  CHECK(escape_control("plain text, \xC4\xA0 utf-8") == "plain text, \xC4\xA0 utf-8");
  CHECK(escape_control("") == "");
}
