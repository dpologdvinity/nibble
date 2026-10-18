#include "tokenizer/tokenizer.h"

#include <algorithm>
#include <queue>
#include <set>
#include <variant>

#include "nibble/error.h"
#include "util/escape.h"
#include "tokenizer/pretokenize.h"
#include "tokenizer/spm.h"
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


// Builds llama.cpp's special_eog_ids (llama-vocab.cpp:2993-3173). Names are matched against the
// LAST id carrying that text (token_to_id[word] = i, :2651). Not mirrored: attribute rewrites
// (CONTROL/NORMAL overrides) and the FIM pre/suf/mid ids, which never enter the EOG set.
std::vector<bool> build_eog(const gguf::File& f, const std::vector<std::string>& tokens, int32_t eos) {
  const int32_t n = static_cast<int32_t>(tokens.size());
  std::unordered_map<std::string_view, int32_t> by_text;
  for (int32_t k = 0; k < n; ++k) by_text[tokens[static_cast<size_t>(k)]] = k;
  // Several names present: llama.cpp takes the first hit while iterating token_to_id, a std::unordered_map
  // (llama-vocab.cpp:1969, loop :2825), so the order is unspecified. We pick the lowest id for determinism.
  auto named = [&](std::initializer_list<const char*> names) {
    int32_t best = -1;
    for (const char* nm : names) {
      auto it = by_text.find(nm);
      if (it != by_text.end() && (best < 0 || it->second < best)) best = it->second;
    }
    return best;
  };
  // Optional id key: out of range or non-integer is ignored (:2758-2765, falls back to default NULL).
  auto key_id = [&](const char* key) -> int32_t {
    const gguf::Value* v = f.find(key);
    const int64_t* i = v ? std::get_if<int64_t>(v) : nullptr;
    return i && *i >= 0 && *i < n ? static_cast<int32_t>(*i) : -1;
  };
  // FIM pad/rep/sep: key, else first name match (:2932-2975; names from :2909-2973).
  auto fim = [&](const char* key, std::initializer_list<const char*> names) {
    int32_t id = key_id(key);
    return id >= 0 ? id : named(names);
  };
  std::set<int32_t> eog;
  auto add = [&](int32_t id) { if (id >= 0) eog.insert(id); };
  add(fim("tokenizer.ggml.fim_pad_token_id", {"<|fim_pad|>", "<fim-pad>", "<fim_pad>", "<PAD>", "[PAD]"}));  // :3015-3017
  add(fim("tokenizer.ggml.fim_rep_token_id", {"<|fim_repo|>", "<|repo_name|>", "<fim-repo>", "<REPO>", "<reponame>"}));  // :3019-3021
  add(fim("tokenizer.ggml.fim_sep_token_id", {"<|file_sep|>", "<|plamo:file_separator|>"}));  // :3023-3025
  // EOG names, :3029-3055 (every supported vocab's eot/eom name is in this list, so the
  // eot/eom name searches at :2828-2860 add nothing beyond it).
  static const char* const names[] = {"<|eot_id|>", "<|eot|>", "<|im_end|>", "<|end|>", "<|return|>", "<|call|>",
                                      "<|flush|>", "<|calls|>", "<end_of_turn>", "<|endoftext|>", "</s>",
                                      "<|eom_id|>", "<EOT>", "_<EOT>", "[EOT]", "[EOS]", "<|end_of_text|>",
                                      "<end_of_utterance>", "<eos>", "<turn|>", "<|tool_response>",
                                      "<\xEF\xBD\x9C" "end\xE2\x96\x81" "of\xE2\x96\x81" "sentence\xEF\xBD\x9C>",
                                      "[e~[", "<|ifm|im_end|>"};
  for (const char* nm : names) {
    auto it = by_text.find(nm);
    if (it != by_text.end()) eog.insert(it->second);
  }
  // eos / eot / eom ids always join the set (:3079-3093).
  add(eos);
  add(key_id("tokenizer.ggml.eot_token_id"));
  add(key_id("tokenizer.ggml.eom_token_id"));
  // Removal rules. "<|end|>" is dropped when "<|return|>"+"<|call|>" or "<|calls|>"+"<|flush|>" are
  // EOG too (:3095-3142); "</s>" is dropped next to a normal-token marker (:3144-3173). Both apply to
  // vocabularies (o200k_harmony, solar-open, gemma4, plamo, beam) nibble does not load; implemented
  // because they are cheap and keep the set identical if such a vocab ever appears.
  auto in_eog = [&](const char* nm) {
    auto it = by_text.find(nm);
    return it != by_text.end() && eog.count(it->second) > 0;
  };
  if (in_eog("<|end|>") && ((in_eog("<|return|>") && (in_eog("<|call|>") || in_eog("<|calls|>"))) ||
                            ((in_eog("<|call|>") || in_eog("<|calls|>")) && in_eog("<|flush|>"))))
    eog.erase(by_text.at("<|end|>"));
  if (in_eog("</s>") && (by_text.count("<rfl:parameter name=\"") || in_eog("<|tool_response>") || in_eog("<|plamo:eos|>")))
    eog.erase(by_text.at("</s>"));
  std::vector<bool> out(tokens.size(), false);
  for (int32_t id : eog) out[static_cast<size_t>(id)] = true;
  return out;
}

}  // namespace

Tokenizer Tokenizer::from_gguf(const gguf::File& f) {
  const std::string model = f.get_string("tokenizer.ggml.model");
  const bool spm = model == "llama";
  if (model != "gpt2" && !spm)
    throw Error("unsupported tokenizer model: '" + escape_control(model) + "' (only 'gpt2', 'llama')");
  const std::string pre = spm ? "" : f.get_string("tokenizer.ggml.pre");
  // Deliberately narrowed: llama.cpp also maps falcon3, pixtral, lfm2, ... to LLAMA3 (:2299-2309) and
  // many other pre names; only the families nibble supports (qwen2, llama3/llama-v3/llama-bpe) are accepted.
  // llama-vocab.cpp:2299-2311 maps these to LLAMA3 (ignore_merges, add_bos default true);
  // qwen2 at :2389 keeps defaults (add_bos false, no ignore_merges).
  const bool llama3 = pre == "llama3" || pre == "llama-v3" || pre == "llama-bpe";
  if (!spm && !llama3 && pre != "qwen2")
    throw Error("unsupported tokenizer pre-tokenizer: '" + escape_control(pre) + "' (only 'qwen2', 'llama3', 'llama-v3', 'llama-bpe')");

  const auto& toks = f.get_array("tokenizer.ggml.tokens");
  const auto& types = f.get_array("tokenizer.ggml.token_type");
  static const gguf::Array no_merges;
  const auto& merges = spm ? no_merges : f.get_array("tokenizer.ggml.merges");
  if (!toks.is_string()) throw Error("tokenizer.ggml.tokens is not a string array");
  if (!spm && !merges.is_string()) throw Error("tokenizer.ggml.merges is not a string array");
  if (!types.is_int()) throw Error("tokenizer.ggml.token_type is not an integer array");
  if (types.size() != toks.size())
    throw Error("tokenizer.ggml.token_type length " + std::to_string(types.size()) +
                " != tokens length " + std::to_string(toks.size()));
  if (toks.empty() || toks.size() > INT32_MAX) throw Error("bad tokenizer vocab size " + std::to_string(toks.size()));

  Tokenizer t;
  t.digit_run_ = llama3 ? 3 : 1;
  t.ignore_merges_ = llama3;
  t.add_bos_ = llama3 || spm;  // SPM default: llama-vocab.cpp:2564-2569
  t.spm_ = spm;
  t.tokens_.reserve(toks.size());
  for (size_t k = 0; k < toks.size(); ++k) t.tokens_.emplace_back(toks.str_at(k));
  if (spm)  // llama-vocab.cpp:2663-2667
    for (size_t k = 0; k < t.tokens_.size(); ++k)
      if (t.tokens_[k].empty()) t.tokens_[k] = "[EMPTY_" + std::to_string(k) + "]";
  t.types_.reserve(types.size());
  for (size_t k = 0; k < types.size(); ++k) {
    int64_t v = types.int_at(k);
    if (v < 1 || v > 6) throw Error("bad token_type " + std::to_string(v) + " at token " + std::to_string(k));
    t.types_.push_back(static_cast<TokenType>(v));
  }
  t.ids_.reserve(t.tokens_.size());
  for (size_t k = 0; k < t.tokens_.size(); ++k) {
    const bool fresh = t.ids_.emplace(t.tokens_[k], static_cast<int32_t>(k)).second;  // first wins
    // llama.cpp asserts token_to_id.size() == id_to_token.size() (llama-vocab.cpp:2679).
    if (spm && !fresh) throw Error("duplicate token text at id " + std::to_string(k) + ": '" + escape_control(t.tokens_[k]) + "'");
  }
  if (spm) t.init_spm(f);

  t.ranks_.reserve(merges.size());  // empty for spm
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

  if (spm && t.tokens_.size() > 1) t.bos_ = 1;  // llama-vocab.cpp:1936 default
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

  t.eog_ = build_eog(f, t.tokens_, t.eos_);

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

// Scores, byte tokens, unk id and add_space_prefix for the SPM model.
void Tokenizer::init_spm(const gguf::File& f) {
  const size_t n = tokens_.size();
  scores_.assign(n, 0.0f);  // absent: all zero (llama-vocab.cpp:2635-2637)
  if (const gguf::Value* v = f.find("tokenizer.ggml.scores")) {
    const gguf::Array* a = std::get_if<gguf::Array>(v);
    if (!a || a->elem_type() != gguf::ValueType::F32) throw Error("tokenizer.ggml.scores is not a float array");
    if (a->size() != n)
      throw Error("tokenizer.ggml.scores length " + std::to_string(a->size()) + " != tokens length " + std::to_string(n));
    for (size_t k = 0; k < n; ++k) scores_[k] = static_cast<float>(a->float_at(k));
  }
  for (size_t k = 0; k < n; ++k) {
    if (types_[k] != TokenType::Byte) continue;
    const std::string& s = tokens_[k];
    auto hex = [](char c) { return c >= '0' && c <= '9' ? c - '0' : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1; };
    if (s.size() != 6 || s.compare(0, 3, "<0x") != 0 || s[5] != '>' || hex(s[3]) < 0 || hex(s[4]) < 0)
      throw Error("bad byte token text at id " + std::to_string(k) + ": '" + escape_control(s) + "'");
  }
  // byte_to_token (llama-vocab.cpp:4165-4177): "<0xXX>", else the byte as a 1-char string.
  for (int b = 0; b < 256; ++b) {
    char buf[7];
    snprintf(buf, sizeof buf, "<0x%02X>", b);
    int32_t id = find(buf);
    if (id < 0) id = find(std::string(1, static_cast<char>(b)));
    byte_ids_[static_cast<size_t>(b)] = id;
  }
  auto opt_id = [&](const char* key, int32_t dflt) {  // llama-vocab.cpp:1940 default unk 0
    const gguf::Value* v = f.find(key);
    if (!v) return dflt;
    const int64_t* i = std::get_if<int64_t>(v);
    if (!i) throw Error(std::string(key) + " is not an integer");
    if (*i < 0 || *i >= static_cast<int64_t>(n)) throw Error(std::string(key) + " out of range: " + std::to_string(*i));
    return static_cast<int32_t>(*i);
  };
  unk_ = opt_id("tokenizer.ggml.unknown_token_id", n > 0 ? 0 : -1);
  add_space_prefix_ = true;  // llama-vocab.cpp:2566, overridden by the key at :2591
  if (const gguf::Value* v = f.find("tokenizer.ggml.add_space_prefix")) {
    const bool* b = std::get_if<bool>(v);
    if (!b) throw Error("tokenizer.ggml.add_space_prefix is not a bool");
    add_space_prefix_ = *b;
  }
}

int32_t Tokenizer::find(std::string_view s) const {
  auto it = ids_.find(std::string(s));
  return it == ids_.end() ? -1 : it->second;
}

bool Tokenizer::is_eog(int32_t id) const { return id >= 0 && id < n_vocab() && eog_[static_cast<size_t>(id)]; }

std::string Tokenizer::decode_token(int32_t id) const {
  if (id < 0 || id >= n_vocab()) throw Error("token id out of range: " + std::to_string(id));
  if (spm_) {  // token_to_piece, llama-vocab.cpp:3823-3897 (special=false)
    switch (types_[id]) {
      case TokenType::UserDefined:
        return tokens_[id];
      case TokenType::Normal: {  // llama_unescape_whitespace, :3540
        std::string out;
        const std::string& s = tokens_[id];
        for (size_t k = 0; k < s.size();) {
          if (s.compare(k, 3, "\xE2\x96\x81") == 0) out.push_back(' '), k += 3;
          else out.push_back(s[k++]);
        }
        return out;
      }
      case TokenType::Byte:  // validated as <0xXX> at load
        return std::string(1, static_cast<char>(std::stoi(tokens_[id].substr(3, 2), nullptr, 16)));
      default:  // Control, Unknown, Unused
        return "";
    }
  }
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
  bool prev_special = true;  // llama-vocab.cpp:3586: the first fragment counts as following a special
  const SpmVocab sv{ids_, scores_, byte_ids_, unk_};
  for (const Frag& fr : frags) {
    if (fr.special >= 0) {
      out.push_back(fr.special);
      prev_special = true;
    } else if (spm_) {
      if (fr.len == 0) continue;  // llama.cpp never makes empty raw fragments
      std::string s = add_space_prefix_ && prev_special ? " " : "";  // :3599-3601
      s.append(text.substr(fr.off, fr.len));
      std::string esc;  // llama_escape_whitespace, :3536
      for (char c : s) esc += c == ' ' ? "\xE2\x96\x81" : std::string(1, c);
      spm_encode(sv, esc, out);
      prev_special = false;
    } else {
      encode_segment(text.substr(fr.off, fr.len), out);
    }
  }
  return out;
}

}  // namespace nibble::tok
