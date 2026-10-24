// Hand-written implementation of the Qwen2 / Llama 3 pre-tokenizer regex (see pretokenize.h).
//
// The pattern, as alternatives tried in order at each position:
//   1. contractions         's 't 'm 'd 're 've 'll (case-insensitive)
//   2. [^\r\n\p{L}\p{N}]?\p{L}+   an optional non-letter prefix, then letters
//   3. \p{N}{1,k}           digit runs of at most k (k = 1 Qwen2, 3 Llama 3)
//   4. " ?[^\s\p{L}\p{N}]+[\r\n]*"   punctuation/symbols, optional leading space and trailing newlines
//   5-7. whitespace: runs ending at the last newline; else all but the last blank before a word; else the run
//   8. any single remaining character
#include "tokenizer/pretokenize.h"

#include "nibble/error.h"
#include "tokenizer/unicode.h"

namespace nibble::tok {

namespace {

// Small character predicates matching the regex classes above ("other" = neither whitespace, letter nor digit).
bool is_nl(char32_t c) { return c == U'\r' || c == U'\n'; }
bool is_other(char32_t c) { return !is_whitespace(c) && !is_letter(c) && !is_number(c); }
char32_t lower_ascii(char32_t c) { return (c >= U'A' && c <= U'Z') ? c + 32 : c; }

}  // namespace

// Works on code points but returns byte slices of the input, so no text is copied. The slices tile the input
// exactly (their concatenation is `text`).
std::vector<std::string_view> pretokenize(std::string_view text, int max_digit_run) {
  if (max_digit_run < 1) throw Error("pretokenize: max_digit_run must be >= 1");
  const std::vector<char32_t> cp = utf8_decode(text);
  const size_t n = cp.size();

  // Byte offset of each code point, plus end sentinel.
  std::vector<size_t> off(n + 1, text.size());
  {
    size_t b = 0;
    for (size_t i = 0; i < n; i++) {
      off[i] = b;
      char32_t c = cp[i];
      b += c < 0x80 ? 1 : c < 0x800 ? 2 : c < 0x10000 ? 3 : 4;
    }
  }

  auto at = [&](size_t p) -> char32_t { return p < n ? cp[p] : char32_t{0xFFFFFFFF}; };
  auto letter = [&](size_t p) { return p < n && is_letter(cp[p]); };
  auto number = [&](size_t p) { return p < n && is_number(cp[p]); };
  auto other = [&](size_t p) { return p < n && is_other(cp[p]); };
  auto space = [&](size_t p) { return p < n && is_whitespace(cp[p]); };

  std::vector<std::string_view> out;
  auto emit = [&](size_t a, size_t b) { out.push_back(text.substr(off[a], off[b] - off[a])); };

  size_t i = 0;
  while (i < n) {
    // 1. contractions
    if (cp[i] == U'\'' && i + 1 < n) {
      char32_t a = lower_ascii(at(i + 1));
      if (a == U's' || a == U't' || a == U'm' || a == U'd') {
        emit(i, i + 2);
        i += 2;
        continue;
      }
      char32_t b = lower_ascii(at(i + 2));
      if ((a == U'r' && b == U'e') || (a == U'v' && b == U'e') || (a == U'l' && b == U'l')) {
        emit(i, i + 3);
        i += 3;
        continue;
      }
    }
    // 2. optional prefix + letters
    if (!is_nl(cp[i]) && !number(i) && (letter(i) || letter(i + 1))) {
      size_t j = i + 1;
      while (letter(j)) j++;
      emit(i, j);
      i = j;
      continue;
    }
    // 3. digits
    if (number(i)) {
      size_t j = i;
      while (number(j) && j - i < static_cast<size_t>(max_digit_run)) j++;
      emit(i, j);
      i = j;
      continue;
    }
    // 4. optional space + others + newlines
    {
      size_t j = i;
      if (cp[j] == U' ' && other(j + 1)) j++;
      if (other(j)) {
        while (other(j)) j++;
        while (j < n && is_nl(cp[j])) j++;
        emit(i, j);
        i = j;
        continue;
      }
    }
    // 5-7. whitespace
    size_t k = i;
    size_t last_nl_end = 0;
    while (space(k)) {
      if (is_nl(cp[k])) last_nl_end = k + 1;
      k++;
    }
    if (last_nl_end > 0) {
      emit(i, last_nl_end);
      i = last_nl_end;
      continue;
    }
    if (k - i >= 2 && k < n) {
      emit(i, k - 1);
      i = k - 1;
      continue;
    }
    if (k > i) {
      emit(i, k);
      i = k;
      continue;
    }
    // 8. fallback
    emit(i, i + 1);
    i++;
  }
  return out;
}

}  // namespace nibble::tok
