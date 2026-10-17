#pragma once
#include <string_view>
#include <vector>

namespace nibble::tok {

// qwen2-style pre-tokenizer split. Slices of `text`, concatenating to `text`.
// max_digit_run: 1 for qwen2, 3 for llama3. Throws nibble::Error on invalid UTF-8.
std::vector<std::string_view> pretokenize(std::string_view text, int max_digit_run);

}  // namespace nibble::tok
