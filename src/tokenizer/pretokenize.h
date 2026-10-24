#pragma once
// The regex-style pre-tokenizer of byte-level BPE models: splits text into "words" (letters with an optional
// leading character, digit runs, punctuation runs, whitespace) before BPE merges are applied. Implemented by
// hand (no regex engine) following the Qwen2 / Llama 3 patterns used by llama.cpp.
#include <string_view>
#include <vector>

namespace nibble::tok {

// qwen2-style pre-tokenizer split. Slices of `text`, concatenating to `text`.
// max_digit_run: 1 for qwen2, 3 for llama3. Throws nibble::Error on invalid UTF-8.
std::vector<std::string_view> pretokenize(std::string_view text, int max_digit_run);

}  // namespace nibble::tok
