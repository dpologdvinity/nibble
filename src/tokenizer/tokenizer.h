#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "gguf/gguf.h"

namespace nibble::tok {

enum class TokenType : int32_t { Normal = 1, Unknown = 2, Control = 3, UserDefined = 4, Unused = 5, Byte = 6 };

// Byte-level BPE (gpt2 model; qwen2 or llama3 pre-tokenizer) or SentencePiece-style BPE (llama model).
// Matches llama.cpp token IDs.
class Tokenizer {
 public:
  static Tokenizer from_gguf(const gguf::File&);
  std::vector<int32_t> encode(std::string_view text, bool parse_special) const;
  std::string decode_token(int32_t id) const;  // raw bytes, may end mid-code-point; Control -> ""
  int32_t n_vocab() const { return static_cast<int32_t>(tokens_.size()); }
  int32_t eos() const { return eos_; }
  int32_t bos() const { return bos_; }  // -1 if none
  bool add_bos() const { return add_bos_; }
  bool add_space_prefix() const { return add_space_prefix_; }  // SPM only
  bool is_eog(int32_t id) const;
  int32_t find(std::string_view token_text) const;  // -1 if absent

 private:
  void init_spm(const gguf::File&);
  void encode_segment(std::string_view text, std::vector<int32_t>& out) const;

  std::vector<std::string> tokens_;
  std::vector<TokenType> types_;
  std::unordered_map<std::string, int32_t> ids_;
  std::unordered_map<std::string, int32_t> ranks_;  // "left right" -> rank
  std::vector<int32_t> specials_;                   // Control/UserDefined/Unknown ids, longest text first
  std::vector<bool> eog_;  // llama.cpp special_eog_ids as a per-id flag
  int32_t eos_ = -1;
  int32_t bos_ = -1;
  bool add_bos_ = false;
  int digit_run_ = 1;          // 1 qwen2, 3 llama3
  bool spm_ = false;           // SentencePiece ("llama") model; no pre-tokenizer or merges
  bool add_space_prefix_ = false;
  int32_t unk_ = -1;
  std::vector<float> scores_;  // spm
  std::array<int32_t, 256> byte_ids_{};  // spm: id of <0xXX> per byte, -1 if absent
  bool ignore_merges_ = false; // llama3: a pre-token that is a whole vocab entry skips BPE
};

}  // namespace nibble::tok
