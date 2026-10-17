#include "tokenizer/unicode.h"

#include <algorithm>
#include <unordered_map>
#include <utility>

#include "nibble/error.h"

namespace nibble::tok {
namespace {
#include "tokenizer/unicode_tables.inc"

template <size_t N>
bool in_ranges(const std::pair<char32_t, char32_t> (&r)[N], char32_t c) {
  auto it = std::upper_bound(std::begin(r), std::end(r), c,
                             [](char32_t v, const auto& p) { return v < p.first; });
  return it != std::begin(r) && c <= (it - 1)->second;
}
}  // namespace

bool is_letter(char32_t c) {
  if (c < 0x80) return (c | 0x20) >= 'a' && (c | 0x20) <= 'z';
  return in_ranges(kLetterRanges, c);
}

bool is_number(char32_t c) {
  if (c < 0x80) return c >= '0' && c <= '9';
  return in_ranges(kNumberRanges, c);
}

bool is_whitespace(char32_t c) {
  return (c >= 0x9 && c <= 0xD) || c == 0x20 || c == 0x85 || c == 0xA0 || c == 0x1680 ||
         (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F ||
         c == 0x205F || c == 0x3000;
}

std::vector<char32_t> utf8_decode(std::string_view s) {
  std::vector<char32_t> out;
  out.reserve(s.size());
  size_t i = 0;
  auto bad = [&](size_t at) {
    return Error("invalid UTF-8 at byte " + std::to_string(at));
  };
  while (i < s.size()) {
    unsigned char b = static_cast<unsigned char>(s[i]);
    if (b < 0x80) {
      out.push_back(b);
      ++i;
      continue;
    }
    int n;
    char32_t cp, min;
    if (b >= 0xC2 && b <= 0xDF) { n = 1; cp = b & 0x1F; min = 0x80; }
    else if (b >= 0xE0 && b <= 0xEF) { n = 2; cp = b & 0x0F; min = 0x800; }
    else if (b >= 0xF0 && b <= 0xF4) { n = 3; cp = b & 0x07; min = 0x10000; }
    else throw bad(i);
    if (i + n >= s.size()) throw bad(i);
    for (int k = 1; k <= n; ++k) {
      unsigned char cb = static_cast<unsigned char>(s[i + k]);
      if ((cb & 0xC0) != 0x80) throw bad(i);
      cp = (cp << 6) | (cb & 0x3F);
    }
    if (cp < min || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) throw bad(i);
    out.push_back(cp);
    i += n + 1;
  }
  return out;
}

void utf8_append(std::string& out, char32_t c) {
  if (c < 0x80) {
    out += static_cast<char>(c);
  } else if (c < 0x800) {
    out += static_cast<char>(0xC0 | (c >> 6));
    out += static_cast<char>(0x80 | (c & 0x3F));
  } else if (c < 0x10000) {
    out += static_cast<char>(0xE0 | (c >> 12));
    out += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (c & 0x3F));
  } else {
    out += static_cast<char>(0xF0 | (c >> 18));
    out += static_cast<char>(0x80 | ((c >> 12) & 0x3F));
    out += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (c & 0x3F));
  }
}

namespace {
struct ByteMap {
  std::array<std::string, 256> fwd;
  std::unordered_map<char32_t, int> rev;
  ByteMap() {
    int n = 0;
    for (int b = 0; b < 256; ++b) {
      bool printable = (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || b >= 0xAE;
      char32_t cp = printable ? static_cast<char32_t>(b) : static_cast<char32_t>(0x100 + n++);
      utf8_append(fwd[b], cp);
      rev[cp] = b;
    }
  }
};
const ByteMap& byte_map() {
  static const ByteMap m;
  return m;
}
}  // namespace

const std::array<std::string, 256>& byte_to_unicode() { return byte_map().fwd; }

int unicode_to_byte(char32_t c) {
  const auto& r = byte_map().rev;
  auto it = r.find(c);
  return it == r.end() ? -1 : it->second;
}

std::string Utf8Streamer::push(std::string_view bytes) {
  held_.append(bytes);
  // Find start of the trailing (possibly incomplete) sequence.
  size_t n = held_.size();
  size_t keep = 0;
  for (size_t back = 1; back <= std::min<size_t>(3, n); ++back) {
    unsigned char b = static_cast<unsigned char>(held_[n - back]);
    if ((b & 0xC0) == 0x80) continue;  // continuation, keep looking
    size_t need = b >= 0xF0 ? 4 : b >= 0xE0 ? 3 : b >= 0xC0 ? 2 : 1;
    if (need > back) keep = back;
    break;
  }
  std::string out = held_.substr(0, n - keep);
  held_.erase(0, n - keep);
  return out;
}

std::string Utf8Streamer::flush() {
  if (held_.empty()) return {};
  held_.clear();
  return "\xEF\xBF\xBD";
}

}  // namespace nibble::tok
