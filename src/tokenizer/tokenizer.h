#pragma once
// Text <-> token-id conversion, reproducing llama.cpp's token ids exactly (the tokenizer test suite compares
// against llama.cpp output for a corpus of tricky strings).
//
// Two families are supported, selected by the GGUF key tokenizer.ggml.model:
//   "gpt2"  byte-level BPE with a regex-style pre-tokenizer ("qwen2", or "llama3"/"llama-v3"/"llama-bpe"
//           which differ in digit grouping and in skipping BPE for whole-word vocab entries)
//   "llama" SentencePiece-style BPE (TinyLlama): score-ordered merges over UTF-8 symbols, space prefix,
//           U+2581 for spaces, <0xXX> byte tokens
// Anything else is rejected by Tokenizer::from_gguf with an error naming the unsupported value.
#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "gguf/gguf.h"

namespace nibble::tok {

// GGUF token_type values (llama.cpp's LLAMA_TOKEN_TYPE_*): how a vocabulary entry is treated.
//   Normal text piece; Unknown <unk>; Control special marker such as <|im_start|> (never printed);
//   UserDefined added token that always matches in input and is printed; Unused padding; Byte <0xXX>.
enum class TokenType : int32_t { Normal = 1, Unknown = 2, Control = 3, UserDefined = 4, Unused = 5, Byte = 6 };

// One piece of a prompt; only `special` parts match Control/Unknown token text (UserDefined always match).
struct Part {
  std::string_view text;
  bool special;
};

// Byte-level BPE (gpt2 model; qwen2 or llama3 pre-tokenizer) or SentencePiece-style BPE (llama model).
// Matches llama.cpp token IDs.
class Tokenizer {
 public:
  // Builds a tokenizer from the GGUF metadata (tokens, token types, merges or scores, special ids). Throws
  // Error for unsupported tokenizer kinds or inconsistent metadata (length mismatches, out-of-range ids,
  // merges that name tokens not in the vocabulary).
  static Tokenizer from_gguf(const gguf::File&);
  // Tokenizes text without adding BOS. parse_special: whether Control/Unknown token text (e.g.
  // "<|im_end|>") in the text is turned into that token (true) or tokenized as ordinary characters (false).
  // Throws Error on invalid UTF-8.
  std::vector<int32_t> encode(std::string_view text, bool parse_special) const;
  // Same tokens as encode() of the concatenated parts (llama.cpp tokenize_impl), except that a special token's
  // text never spans two parts. Adjacent raw text is merged across parts before encoding. add_bos prepends
  // bos() when the model has one and add_bos() is set, unless the output already starts with it.
  std::vector<int32_t> encode_parts(std::span<const Part> parts, bool add_bos) const;
  // The text of one token as raw bytes (may end mid-UTF-8-sequence, so stream through Utf8Streamer);
  // Control/Unknown/Unused tokens give "". Throws Error for an id outside the vocabulary.
  std::string decode_token(int32_t id) const;
  int32_t n_vocab() const { return static_cast<int32_t>(tokens_.size()); }  // vocabulary size
  int32_t eos() const { return eos_; }  // end-of-sequence id from the metadata
  int32_t bos() const { return bos_; }  // -1 if none
  bool add_bos() const { return add_bos_; }  // whether the model wants BOS prepended to prompts
  bool add_space_prefix() const { return add_space_prefix_; }  // SPM only
  // True for every end-of-generation token (EOS plus end-of-turn markers such as <|im_end|>, <|eot_id|>,
  // </s>), mirroring llama.cpp's special_eog_ids. Sampling one of these ends generation.
  bool is_eog(int32_t id) const;
  int32_t find(std::string_view token_text) const;  // exact-text vocabulary lookup; -1 if absent

 private:
  void init_spm(const gguf::File&);  // SentencePiece-only metadata: scores, byte tokens, unk id, space prefix
  // Byte-level BPE of one run of plain text (no special tokens): pre-tokenize, map bytes to the GPT-2
  // printable alphabet, then merge by rank.
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
