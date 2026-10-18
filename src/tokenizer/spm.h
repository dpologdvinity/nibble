#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace nibble::tok {

// Inputs of llama.cpp's llm_tokenizer_spm_session. All references outlive the call.
struct SpmVocab {
  const std::unordered_map<std::string, int32_t>& ids;
  const std::vector<float>& scores;
  const std::array<int32_t, 256>& byte_ids;  // <0xXX> (or the 1-byte string) per byte; -1 if absent
  int32_t unk;                               // -1 if absent
};

// Tokenizes one raw-text fragment that already has its space prefix and U+2581 escaping applied.
// Symbols with no vocab entry become byte tokens (UNK, else Error, when a byte token is missing).
void spm_encode(const SpmVocab& v, std::string_view escaped, std::vector<int32_t>& out);

}  // namespace nibble::tok
