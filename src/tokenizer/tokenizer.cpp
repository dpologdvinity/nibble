#include "tokenizer/tokenizer.h"

#include <algorithm>
#include <queue>

#include "nibble/error.h"
#include "util/escape.h"
#include "tokenizer/pretokenize.h"
#include "tokenizer/unicode.h"

namespace nibble::tok {

namespace {

struct Sym {
  size_t off, len;
  int prev, next;
};
struct Bigram {
  int left, right;
  int rank;
  size_t size;  // combined byte length; detects stale entries
};
struct BigramCmp {
  bool operator()(const Bigram& a, const Bigram& b) const {
    return a.rank > b.rank || (a.rank == b.rank && a.left > b.left);
  }
};

}  // namespace

Tokenizer Tokenizer::from_gguf(const gguf::File& f) {
  const std::string model = f.get_string("tokenizer.ggml.model");
  if (model != "gpt2") throw Error("unsupported tokenizer model: '" + escape_control(model) + "' (only 'gpt2')");
  const std::string pre = f.get_string("tokenizer.ggml.pre");
  // llama-vocab.cpp:2299-2311 maps these to LLAMA3 (ignore_merges, add_bos default true);
  // qwen2 at :2389 keeps defaults (add_bos false, no ignore_merges).
  const bool llama3 = pre == "llama3" || pre == "llama-v3" || pre == "llama-bpe";
  if (!llama3 && pre != "qwen2")
    throw Error("unsupported tokenizer pre-tokenizer: '" + escape_control(pre) + "' (only 'qwen2', 'llama3', 'llama-v3', 'llama-bpe')");

  const auto& toks = f.get_array("tokenizer.ggml.tokens");
  const auto& types = f.get_array("tokenizer.ggml.token_type");
  const auto& merges = f.get_array("tokenizer.ggml.merges");
  if (!toks.is_string()) throw Error("tokenizer.ggml.tokens is not a string array");
  if (!merges.is_string()) throw Error("tokenizer.ggml.merges is not a string array");
  if (!types.is_int()) throw Error("tokenizer.ggml.token_type is not an integer array");
  if (types.size() != toks.size())
    throw Error("tokenizer.ggml.token_type length " + std::to_string(types.size()) +
                " != tokens length " + std::to_string(toks.size()));
  if (toks.empty() || toks.size() > INT32_MAX) throw Error("bad tokenizer vocab size " + std::to_string(toks.size()));

  Tokenizer t;
  t.digit_run_ = llama3 ? 3 : 1;
  t.ignore_merges_ = llama3;
  t.add_bos_ = llama3;
  t.tokens_.reserve(toks.size());
  for (size_t k = 0; k < toks.size(); ++k) t.tokens_.emplace_back(toks.str_at(k));
  t.types_.reserve(types.size());
  for (size_t k = 0; k < types.size(); ++k) {
    int64_t v = types.int_at(k);
    if (v < 1 || v > 6) throw Error("bad token_type " + std::to_string(v) + " at token " + std::to_string(k));
    t.types_.push_back(static_cast<TokenType>(v));
  }
  t.ids_.reserve(t.tokens_.size());
  for (size_t k = 0; k < t.tokens_.size(); ++k) t.ids_.emplace(t.tokens_[k], static_cast<int32_t>(k));  // first wins

  t.ranks_.reserve(merges.size());
  for (size_t k = 0; k < merges.size(); ++k) {
    const std::string m(merges.str_at(k));
    size_t sp = m.find(' ', 1);  // same as llama.cpp: first space after position 0
    if (sp == std::string::npos) throw Error("malformed merge (no space): '" + escape_control(m) + "'");
    for (const std::string& part : {m.substr(0, sp), m.substr(sp + 1)})
      if (!t.ids_.count(part)) throw Error("merge '" + escape_control(m) + "' references token not in vocab: '" + escape_control(part) + "'");
    t.ranks_.emplace(m, static_cast<int32_t>(k));
  }

  int64_t eos = f.get_int("tokenizer.ggml.eos_token_id");
  if (eos < 0 || eos >= static_cast<int64_t>(t.tokens_.size()))
    throw Error("eos_token_id out of range: " + std::to_string(eos));
  t.eos_ = static_cast<int32_t>(eos);

  if (const gguf::Value* v = f.find("tokenizer.ggml.bos_token_id")) {
    const int64_t* b = std::get_if<int64_t>(v);
    if (!b) throw Error("tokenizer.ggml.bos_token_id is not an integer");
    if (*b < 0 || *b >= static_cast<int64_t>(t.tokens_.size()))
      throw Error("bos_token_id out of range: " + std::to_string(*b));
    t.bos_ = static_cast<int32_t>(*b);
  }
  if (const gguf::Value* v = f.find("tokenizer.ggml.add_bos_token")) {
    const bool* b = std::get_if<bool>(v);
    if (!b) throw Error("tokenizer.ggml.add_bos_token is not a bool");
    t.add_bos_ = *b;
  }

  for (size_t k = 0; k < t.tokens_.size(); ++k) {
    auto ty = t.types_[k];
    if (ty == TokenType::Control || ty == TokenType::UserDefined || ty == TokenType::Unknown)
      if (!t.tokens_[k].empty()) t.specials_.push_back(static_cast<int32_t>(k));
  }
  // llama.cpp partitions per special token, longest text first
  std::stable_sort(t.specials_.begin(), t.specials_.end(), [&](int32_t a, int32_t b) {
    return t.tokens_[a].size() > t.tokens_[b].size();
  });
  return t;
}

int32_t Tokenizer::find(std::string_view s) const {
  auto it = ids_.find(std::string(s));
  return it == ids_.end() ? -1 : it->second;
}

bool Tokenizer::is_eog(int32_t id) const {
  if (id < 0 || id >= n_vocab()) return false;
  if (id == eos_) return true;
  // llama-vocab.cpp:3029-3055 (names that exist in supported vocabularies)
  static const char* const names[] = {"<|eot_id|>", "<|eot|>", "<|im_end|>", "<|end|>", "<|return|>", "<|call|>",
                                      "<|flush|>", "<|calls|>", "<end_of_turn>", "<|endoftext|>", "</s>",
                                      "<|eom_id|>", "<EOT>", "_<EOT>", "[EOT]", "[EOS]", "<|end_of_text|>",
                                      "<end_of_utterance>", "<eos>", "<turn|>", "<|tool_response>",
                                      "<\xEF\xBD\x9C" "end\xE2\x96\x81" "of\xE2\x96\x81" "sentence\xEF\xBD\x9C>",
                                      "[e~[", "<|ifm|im_end|>"};
  for (const char* n : names)
    if (tokens_[id] == n) return true;
  return false;
}

std::string Tokenizer::decode_token(int32_t id) const {
  if (id < 0 || id >= n_vocab()) throw Error("token id out of range: " + std::to_string(id));
  switch (types_[id]) {
    case TokenType::Control:
    case TokenType::Unknown:
    case TokenType::Unused:
      return "";
    case TokenType::UserDefined:
      return tokens_[id];
    default:
      break;
  }
  std::string out;
  for (char32_t c : utf8_decode(tokens_[id])) {
    int b = unicode_to_byte(c);
    if (b < 0) utf8_append(out, c);
    else out.push_back(static_cast<char>(b));
  }
  return out;
}

void Tokenizer::encode_segment(std::string_view text, std::vector<int32_t>& out) const {
  utf8_decode(text);  // validate; throws with byte offset
  const auto& map = byte_to_unicode();
  for (std::string_view word : pretokenize(text, digit_run_)) {
    std::string w;
    for (unsigned char c : word) w += map[c];
    if (ignore_merges_) {  // llama-vocab.cpp:669
      auto whole = ids_.find(w);
      if (whole != ids_.end()) {
        out.push_back(whole->second);
        continue;
      }
    }
    std::vector<Sym> syms;
    for (size_t off = 0; off < w.size();) {
      size_t n = 1;
      unsigned char c = static_cast<unsigned char>(w[off]);
      if (c >= 0xF0) n = 4;
      else if (c >= 0xE0) n = 3;
      else if (c >= 0xC0) n = 2;
      syms.push_back({off, n, static_cast<int>(syms.size()) - 1, -1});
      off += n;
    }
    for (size_t k = 0; k + 1 < syms.size(); ++k) syms[k].next = static_cast<int>(k + 1);

    std::priority_queue<Bigram, std::vector<Bigram>, BigramCmp> q;
    std::string key;
    auto add = [&](int l, int r) {
      if (l < 0 || r < 0) return;
      key.assign(w, syms[l].off, syms[l].len);
      key += ' ';
      key.append(w, syms[r].off, syms[r].len);
      auto it = ranks_.find(key);
      if (it != ranks_.end()) q.push({l, r, it->second, syms[l].len + syms[r].len});
    };
    for (int k = 1; k < static_cast<int>(syms.size()); ++k) add(k - 1, k);
    while (!q.empty()) {
      Bigram b = q.top();
      q.pop();
      Sym& L = syms[b.left];
      Sym& R = syms[b.right];
      if (L.len == 0 || R.len == 0 || L.len + R.len != b.size) continue;  // stale
      L.len += R.len;
      R.len = 0;
      L.next = R.next;
      if (R.next >= 0) syms[R.next].prev = b.left;
      add(L.prev, b.left);
      add(b.left, L.next);
    }
    for (int k = 0; k >= 0 && !syms.empty(); k = syms[k].next) {
      auto it = ids_.find(w.substr(syms[k].off, syms[k].len));
      if (it == ids_.end()) throw Error("symbol not in vocab: '" + escape_control(w.substr(syms[k].off, syms[k].len)) + "'");
      out.push_back(it->second);
    }
  }
}

std::vector<int32_t> Tokenizer::encode(std::string_view text, bool parse_special) const {
  struct Frag {
    size_t off, len;
    int32_t special;  // -1 = raw text
  };
  std::vector<Frag> frags{{0, text.size(), -1}};
  for (int32_t sid : specials_) {
    if (!parse_special && types_[sid] != TokenType::UserDefined) continue;
    const std::string& st = tokens_[sid];
    std::vector<Frag> next;
    for (const Frag& fr : frags) {
      if (fr.special >= 0) {
        next.push_back(fr);
        continue;
      }
      std::string_view sub = text.substr(fr.off, fr.len);
      size_t pos = 0;
      for (;;) {
        size_t m = sub.find(st, pos);
        if (m == std::string_view::npos) break;
        if (m > pos) next.push_back({fr.off + pos, m - pos, -1});
        next.push_back({0, 0, sid});
        pos = m + st.size();
      }
      if (pos < sub.size()) next.push_back({fr.off + pos, sub.size() - pos, -1});
    }
    frags = std::move(next);
  }
  std::vector<int32_t> out;
  for (const Frag& fr : frags) {
    if (fr.special >= 0) out.push_back(fr.special);
    else encode_segment(text.substr(fr.off, fr.len), out);
  }
  return out;
}

}  // namespace nibble::tok
