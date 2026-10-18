#pragma once
// Tiny synthetic models for tests: tokenizer half, and tokenizer + qwen2 weights.
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "fp16.h"
#include "gguf_writer.h"
#include "quant_test_util.h"
#include "tokenizer/unicode.h"

struct TinyTokOpts {
  std::string model = "gpt2", pre = "qwen2";
  std::vector<std::string> merges = {"h e", "he l", "\xC4\xA0 w"};
  std::vector<std::string> extra;  // extra Normal tokens after the 5 fixed ones (ids 261..)
  bool drop_type = false;  // token_type one shorter than tokens
  // Appends "<|im_start|>", "<|im_end|>" as Control tokens after `extra`. "<|im_end|>"
  // is EOG by name (Tokenizer::is_eog), as for Qwen; eos_token_id is unchanged.
  bool chatml = false;
  std::vector<std::string> control;  // extra Control tokens after chatml's (non-spm only)
  int64_t bos = -1;      // >= 0: write tokenizer.ggml.bos_token_id (U32)
  int add_bos = -1;      // 0/1: write tokenizer.ggml.add_bos_token (Bool); -1: absent
  bool bos_as_string = false;  // write bos_token_id as a string (bad type)
  // SentencePiece ("llama") vocab; see tiny_spm_vocab. Ignores merges/extra/chatml/pre; bos 1, eos 2, unk 0.
  bool spm = false;
  int add_space_prefix = -1;   // spm: 0/1 writes tokenizer.ggml.add_space_prefix; -1 absent (default true)
  int bad_scores = 0;          // spm: 1 = one score too few, 2 = I32 instead of F32, 3 = NaN at id 5, 4 = +inf at id 6
  int byte_0xC3 = 0;           // spm: 1 = <0xC3> becomes the 1-char string "\xC3", 2 = removed (falls back to unk)
};

struct TinySpmTok { const char* text; int type; float score; };

// ids: 0 <unk>, 1 <s>, 2 </s>, 3..258 <0x00>..<0xFF>, then the table below from 259.
inline std::vector<TinySpmTok> tiny_spm_vocab() {
  std::vector<TinySpmTok> v = {{"<unk>", 2, 0}, {"<s>", 3, 0}, {"</s>", 3, 0}};
  static std::vector<std::string> bytes;  // stable storage for c_str()
  if (bytes.empty())
    for (int b = 0; b < 256; ++b) {
      char buf[8];
      snprintf(buf, sizeof buf, "<0x%02X>", b);
      bytes.emplace_back(buf);
    }
  for (const auto& s : bytes) v.push_back({s.c_str(), 6, 0});
#define SP "\xE2\x96\x81"  // U+2581
  for (const TinySpmTok& t : std::vector<TinySpmTok>{
           {SP, 1, -10}, {"h", 1, -10}, {"i", 1, -10}, {"t", 1, -10}, {"e", 1, -10}, {"r", 1, -10},
           {"a", 1, -10}, {"b", 1, -10}, {"c", 1, -10},
           {SP "h", 1, -6}, {"hi", 1, -7}, {SP "hi", 1, -3},
           {SP "t", 1, -6}, {"he", 1, -4}, {"the", 1, -3}, {SP "the", 1, -2},
           {"re", 1, -5}, {SP "there", 1, -1},
           {SP "a", 1, -20}, {SP "b", 1, -20}, {"ab", 1, -1}, {"bc", 1, -2}, {"ca", 1, -1}})
    v.push_back(t);
#undef SP
  return v;
}

inline int64_t tiny_tok_vocab(const TinyTokOpts& o) {
  if (o.spm) return static_cast<int64_t>(tiny_spm_vocab().size());
  return 261 + static_cast<int64_t>(o.extra.size()) + (o.chatml ? 2 : 0) + static_cast<int64_t>(o.control.size());
}

inline void add_tiny_spm_kvs(GgufWriter& w, const TinyTokOpts& o) {
  std::vector<std::string> toks;
  std::vector<int32_t> types;
  std::vector<float> scores;
  for (const auto& t : tiny_spm_vocab()) {
    toks.push_back(t.text);
    types.push_back(t.type);
    scores.push_back(t.score);
  }
  if (o.byte_0xC3) {  // id 3 + 0xC3
    toks[3 + 0xC3] = o.byte_0xC3 == 1 ? "\xC3" : "[no-c3]";
    types[3 + 0xC3] = 1;
  }
  w.add_string("tokenizer.ggml.model", "llama");
  w.add_string_array("tokenizer.ggml.tokens", toks);
  w.add_i32_array("tokenizer.ggml.token_type", types);
  if (o.bad_scores == 1) scores.pop_back();
  if (o.bad_scores == 3) scores[5] = std::numeric_limits<float>::quiet_NaN();
  if (o.bad_scores == 4) scores[6] = std::numeric_limits<float>::infinity();
  if (o.bad_scores == 2) w.add_i32_array("tokenizer.ggml.scores", std::vector<int32_t>(toks.size(), 0));
  else w.add_f32_array("tokenizer.ggml.scores", scores);
  w.add_int("tokenizer.ggml.bos_token_id", GgufWriter::ValueType::U32, 1);
  w.add_int("tokenizer.ggml.eos_token_id", GgufWriter::ValueType::U32, 2);
  w.add_int("tokenizer.ggml.unknown_token_id", GgufWriter::ValueType::U32, 0);
  if (o.add_space_prefix >= 0) w.add_int("tokenizer.ggml.add_space_prefix", GgufWriter::ValueType::Bool, o.add_space_prefix);
  if (o.add_bos >= 0) w.add_int("tokenizer.ggml.add_bos_token", GgufWriter::ValueType::Bool, o.add_bos);
}

inline void add_tiny_tokenizer_kvs(GgufWriter& w, const TinyTokOpts& o, int32_t eos_id = 259) {
  if (o.spm) return add_tiny_spm_kvs(w, o);
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
  if (o.chatml) {
    toks.insert(toks.end(), {"<|im_start|>", "<|im_end|>"});
    types.insert(types.end(), {3, 3});
  }
  for (const auto& c : o.control) {
    toks.push_back(c);
    types.push_back(3);
  }
  if (o.drop_type) types.pop_back();
  w.add_string("tokenizer.ggml.model", o.model);
  w.add_string("tokenizer.ggml.pre", o.pre);
  w.add_string_array("tokenizer.ggml.tokens", toks);
  w.add_i32_array("tokenizer.ggml.token_type", types);
  w.add_string_array("tokenizer.ggml.merges", o.merges);
  w.add_int("tokenizer.ggml.eos_token_id", GgufWriter::ValueType::U32, eos_id);
  if (o.bos_as_string) w.add_string("tokenizer.ggml.bos_token_id", "x");
  else if (o.bos >= 0) w.add_int("tokenizer.ggml.bos_token_id", GgufWriter::ValueType::U32, o.bos);
  if (o.add_bos >= 0) w.add_int("tokenizer.ggml.add_bos_token", GgufWriter::ValueType::Bool, o.add_bos);
}

inline std::vector<uint8_t> tiny_tokenizer_gguf(const TinyTokOpts& o = {}) {
  GgufWriter w;
  add_tiny_tokenizer_kvs(w, o);
  return w.build();
}

// Variants for model-loading tests; defaults give the plain valid model.
struct TinyModelOpts {
  std::string arch = "qwen2";
  std::string drop_tensor;      // omit this tensor
  std::string bad_shape;        // give this 2-D tensor a wrong ne[1]
  std::string q4_tensor;        // store this tensor as Q4_0 (needs ne[0] % 32 == 0, e.g. blk.0.ffn_down.weight)
  int64_t vocab_delta = 0;      // token_embd rows = tokenizer vocab + delta (tokenizer unchanged)
  bool explicit_output = false; // add output.weight (otherwise tied to token_embd)
  bool quant_dims = false;      // n_embd 32, n_ff 64 (head_dim 8) so every ne0 % 32 == 0; needed for Q wtype
  std::string raw_type_tensor;  // store this tensor with unsupported ggml type 12 (Q4_K) and dummy bytes
  int64_t heads_kv = 0, kv_heads_kv = 0;  // if > 0, written as head_count / head_count_kv (tensors unchanged)
  TinyTokOpts tok;              // tokenizer variant
  std::string chat_template;    // non-empty: written as tokenizer.chat_template
  uint32_t context_length = 64; // written as <arch>.context_length
  bool llama_bias = false;      // llama only: also write attn_{q,k,v}.bias in every layer
  bool rope_freqs = false;      // add rope_freqs.weight: F32 {HD/2}, values in [1, 8]
  int bad_rope_freqs = 0;       // with rope_freqs: 1 = wrong ne ({HD/2 + 1}), 2 = F16 instead of F32
  int64_t rope_dim = 0, key_length = 0;  // if > 0, written as <arch>.rope.dimension_count / attention.key_length
};

// qwen2 (or llama, no biases): 2 layers, n_embd 16, 4 heads, 2 kv heads, n_ff 32, vocab tiny_tok_vocab(mo.tok) (261 by default).
inline std::vector<uint8_t> tiny_model_gguf(uint32_t seed, nibble::DType wtype = nibble::DType::F32,
                                            bool zero_output_norm = false, int32_t eos_id = 259,
                                            const TinyModelOpts& mo = {}) {
  using nibble::DType;
  constexpr int64_t H = 4, KV = 2, L = 2;
  const int64_t V = tiny_tok_vocab(mo.tok);
  const int64_t E = mo.quant_dims ? 32 : 16, HD = E / H, FF = mo.quant_dims ? 64 : 32;
  GgufWriter w;
  w.add_string("general.architecture", mo.arch);
  const std::string a = mo.arch + ".";
  w.add_int(a + "block_count", GgufWriter::ValueType::U32, L);
  w.add_int(a + "embedding_length", GgufWriter::ValueType::U32, E);
  w.add_int(a + "feed_forward_length", GgufWriter::ValueType::U32, FF);
  w.add_int(a + "attention.head_count", GgufWriter::ValueType::U32, mo.heads_kv > 0 ? mo.heads_kv : H);
  w.add_int(a + "attention.head_count_kv", GgufWriter::ValueType::U32, mo.kv_heads_kv > 0 ? mo.kv_heads_kv : KV);
  w.add_int(a + "context_length", GgufWriter::ValueType::U32, mo.context_length);
  w.add_f32(a + "attention.layer_norm_rms_epsilon", 1e-6f);
  w.add_f32(a + "rope.freq_base", 10000.0f);
  if (mo.rope_dim > 0) w.add_int(a + "rope.dimension_count", GgufWriter::ValueType::U32, mo.rope_dim);
  if (mo.key_length > 0) w.add_int(a + "attention.key_length", GgufWriter::ValueType::U32, mo.key_length);
  add_tiny_tokenizer_kvs(w, mo.tok, eos_id);
  if (!mo.chat_template.empty()) w.add_string("tokenizer.chat_template", mo.chat_template);

  std::mt19937 g(seed);
  auto rnd = [&](size_t n, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v(n);
    for (auto& x : v) x = d(g);
    return v;
  };
  auto bytes = [](DType t, const std::vector<float>& v, int64_t ne0) {
    std::vector<uint8_t> b;
    if (t == DType::F32) {
      b.resize(v.size() * 4);
      std::memcpy(b.data(), v.data(), b.size());
    } else if (t == DType::F16) {
      for (float f : v) {
        uint16_t h = nibble::fp32_to_fp16(f);
        b.push_back(static_cast<uint8_t>(h));
        b.push_back(static_cast<uint8_t>(h >> 8));
      }
    } else if (t == DType::Q4_0 || t == DType::Q8_0) {
      b = test_quantize_rows(t, v, ne0);
    } else {  // unsupported type: contents are irrelevant (only used to test rejection)
      b.assign(144, 0);
    }
    return b;
  };
  auto mat = [&](const std::string& name, int64_t in, int64_t out) {
    if (name == mo.drop_tensor) return;
    DType t = name == mo.q4_tensor ? DType::Q4_0 : name == mo.raw_type_tensor ? static_cast<DType>(12) : wtype;
    int64_t o = name == mo.bad_shape ? out + 1 : out;
    w.add_tensor(name, t, {in, o}, bytes(t, rnd(static_cast<size_t>(in * o), -0.5f, 0.5f), in));
  };
  auto vec = [&](const std::string& name, int64_t n, float lo, float hi) {
    if (name == mo.drop_tensor) return;
    w.add_tensor(name, DType::F32, {n}, bytes(DType::F32, rnd(static_cast<size_t>(n), lo, hi), n));
  };
  mat("token_embd.weight", E, V + mo.vocab_delta);
  if (zero_output_norm) {
    w.add_tensor("output_norm.weight", DType::F32, {E}, std::vector<uint8_t>(E * 4, 0));
  } else {
    vec("output_norm.weight", E, 0.9f, 1.1f);
  }
  if (mo.explicit_output) {  // own RNG stream, so every other tensor equals the tied model's
    const std::mt19937 saved = g;
    g = std::mt19937(seed ^ 0xA5A5A5A5u);
    mat("output.weight", E, V);
    g = saved;
  }
  if (mo.rope_freqs) {  // own RNG stream, so the other tensors are unchanged by the option
    const std::mt19937 saved = g;
    g = std::mt19937(seed ^ 0x5A5A5A5Au);
    const bool f16 = mo.bad_rope_freqs == 2;
    const int64_t n = mo.bad_rope_freqs == 1 ? HD / 2 + 1 : HD / 2;
    w.add_tensor("rope_freqs.weight", f16 ? DType::F16 : DType::F32, {n},
                 bytes(f16 ? DType::F16 : DType::F32, rnd(static_cast<size_t>(n), 1.f, 8.f), n));
    g = saved;
  }
  for (int64_t i = 0; i < L; ++i) {
    const std::string p = "blk." + std::to_string(i) + ".";
    vec(p + "attn_norm.weight", E, 0.9f, 1.1f);
    mat(p + "attn_q.weight", E, H * HD);
    mat(p + "attn_k.weight", E, KV * HD);
    mat(p + "attn_v.weight", E, KV * HD);
    mat(p + "attn_output.weight", H * HD, E);
    if (mo.arch == "qwen2" || mo.llama_bias) vec(p + "attn_q.bias", H * HD, -0.5f, 0.5f);
    if (mo.arch == "qwen2" || mo.llama_bias) vec(p + "attn_k.bias", KV * HD, -0.5f, 0.5f);
    if (mo.arch == "qwen2" || mo.llama_bias) vec(p + "attn_v.bias", KV * HD, -0.5f, 0.5f);
    vec(p + "ffn_norm.weight", E, 0.9f, 1.1f);
    mat(p + "ffn_gate.weight", E, FF);
    mat(p + "ffn_up.weight", E, FF);
    mat(p + "ffn_down.weight", FF, E);
  }
  return w.build();
}
