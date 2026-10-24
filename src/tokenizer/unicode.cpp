// Implementation of the Unicode helpers declared in unicode.h.
#include "tokenizer/unicode.h"

#include <algorithm>
#include <unordered_map>
#include <utility>

#include "nibble/error.h"

namespace nibble::tok {
namespace {
#include "tokenizer/unicode_tables.inc"

// True if c lies in one of the sorted, non-overlapping [first, last] ranges (binary search).
template <size_t N>
bool in_ranges(const std::pair<char32_t, char32_t> (&r)[N], char32_t c) {
  auto it = std::upper_bound(std::begin(r), std::end(r), c,
                             [](char32_t v, const auto& p) { return v < p.first; });
  return it != std::begin(r) && c <= (it - 1)->second;
}
}  // namespace

// ASCII fast path, then the generated Unicode 15 range table.
bool is_letter(char32_t c) {
  if (c < 0x80) return (c | 0x20) >= 'a' && (c | 0x20) <= 'z';
  return in_ranges(kLetterRanges, c);
}

bool is_number(char32_t c) {
  if (c < 0x80) return c >= '0' && c <= '9';
  return in_ranges(kNumberRanges, c);
}

// The Unicode White_Space property, spelled out (it is a short, stable list).
bool is_whitespace(char32_t c) {
  return (c >= 0x9 && c <= 0xD) || c == 0x20 || c == 0x85 || c == 0xA0 || c == 0x1680 ||
         (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F ||
         c == 0x205F || c == 0x3000;
}

// A hand-rolled validator/decoder: lead byte selects the length and minimum value (to reject overlong
// encodings); continuation bytes must be 10xxxxxx; surrogates and values above U+10FFFF are rejected.
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
// The GPT-2 table: bytes that are printable (and not space-like) keep their value as a code point; the others
// are assigned U+0100, U+0101, ... in byte order. Built once on first use.
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

// Reverse lookup in the table above.
int unicode_to_byte(char32_t c) {
  const auto& r = byte_map().rev;
  auto it = r.find(c);
  return it == r.end() ? -1 : it->second;
}

// Scans the held bytes: complete valid sequences are emitted, an invalid lead or a bad continuation byte is
// replaced by U+FFFD (consuming only the lead byte, so later bytes are re-examined), and a valid but
// unfinished prefix at the end stays in held_. The second-byte ranges follow the Unicode well-formedness
// table (e.g. E0 A0..BF, ED 80..9F).
std::string Utf8Streamer::push(std::string_view bytes) {
  held_.append(bytes);
  const std::string_view in = held_;
  std::string out;
  size_t i = 0;
  while (i < in.size()) {
    const unsigned char b = static_cast<unsigned char>(in[i]);
    if (b < 0x80) {
      out += static_cast<char>(b);
      ++i;
      continue;
    }
    size_t total;
    unsigned char lo = 0x80, hi = 0xBF;  // allowed range of the second byte
    if (b >= 0xC2 && b <= 0xDF) total = 2;
    else if (b >= 0xE0 && b <= 0xEF) {
      total = 3;
      if (b == 0xE0) lo = 0xA0;
      if (b == 0xED) hi = 0x9F;
    } else if (b >= 0xF0 && b <= 0xF4) {
      total = 4;
      if (b == 0xF0) lo = 0x90;
      if (b == 0xF4) hi = 0x8F;
    } else {
      out += "\xEF\xBF\xBD";  // invalid lead
      ++i;
      continue;
    }
    bool ok = true;
    size_t k = 1;
    for (; k < total && i + k < in.size(); ++k) {
      const unsigned char c = static_cast<unsigned char>(in[i + k]);
      if (c < (k == 1 ? lo : 0x80) || c > (k == 1 ? hi : 0xBF)) {
        ok = false;
        break;
      }
    }
    if (!ok) {
      out += "\xEF\xBF\xBD";  // bad continuation: replace lead only
      ++i;
    } else if (k < total) {
      break;  // valid but incomplete prefix: hold
    } else {
      out.append(in.substr(i, total));
      i += total;
    }
  }
  held_.erase(0, i);
  return out;
}

// Called at end of stream: a dangling incomplete sequence is reported as one U+FFFD.
std::string Utf8Streamer::flush() {
  if (held_.empty()) return {};
  held_.clear();
  return "\xEF\xBF\xBD";
}

}  // namespace nibble::tok
