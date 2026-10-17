#pragma once
// Tiny synthetic models for tests (tokenizer half).
#include <string>
#include <vector>

#include "gguf_writer.h"
#include "tokenizer/unicode.h"

struct TinyTokOpts {
  std::string model = "gpt2", pre = "qwen2";
  std::vector<std::string> merges = {"h e", "he l", "\xC4\xA0 w"};
  std::vector<std::string> extra;  // extra Normal tokens after the 5 fixed ones (ids 261..)
  bool drop_type = false;  // token_type one shorter than tokens
};

inline std::vector<uint8_t> tiny_tokenizer_gguf(const TinyTokOpts& o = {}) {
  std::vector<std::string> toks;
  std::vector<int32_t> types;
  for (const auto& s : nibble::tok::byte_to_unicode()) {
    toks.push_back(s);
    types.push_back(1);
  }
  toks.insert(toks.end(), {"he", "hel", "\xC4\xA0w", "<|ctl|>", "<tool>"});
  types.insert(types.end(), {1, 1, 1, 3, 4});
  for (const auto& e : o.extra) {
    toks.push_back(e);
    types.push_back(1);
  }
  if (o.drop_type) types.pop_back();
  GgufWriter w;
  w.add_string("tokenizer.ggml.model", o.model);
  w.add_string("tokenizer.ggml.pre", o.pre);
  w.add_string_array("tokenizer.ggml.tokens", toks);
  w.add_i32_array("tokenizer.ggml.token_type", types);
  w.add_string_array("tokenizer.ggml.merges", o.merges);
  w.add_int("tokenizer.ggml.eos_token_id", GgufWriter::ValueType::U32, 259);
  return w.build();
}
