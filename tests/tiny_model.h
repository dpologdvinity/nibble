#pragma once
// Tiny synthetic models for tests: tokenizer half, and tokenizer + qwen2 weights.
#include <cstring>
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
};

inline void add_tiny_tokenizer_kvs(GgufWriter& w, const TinyTokOpts& o, int32_t eos_id = 259) {
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
  w.add_string("tokenizer.ggml.model", o.model);
  w.add_string("tokenizer.ggml.pre", o.pre);
  w.add_string_array("tokenizer.ggml.tokens", toks);
  w.add_i32_array("tokenizer.ggml.token_type", types);
  w.add_string_array("tokenizer.ggml.merges", o.merges);
  w.add_int("tokenizer.ggml.eos_token_id", GgufWriter::ValueType::U32, eos_id);
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
  int64_t vocab_delta = 0;      // token_embd rows = 261 + delta (tokenizer unchanged)
  bool explicit_output = false; // add output.weight (otherwise tied to token_embd)
  bool quant_dims = false;      // n_embd 32, n_ff 64 (head_dim 8) so every ne0 % 32 == 0; needed for Q wtype
  std::string raw_type_tensor;  // store this tensor with unsupported ggml type 12 (Q4_K) and dummy bytes
};

// qwen2: 2 layers, n_embd 16, 4 heads, 2 kv heads, n_ff 32, vocab 261.
inline std::vector<uint8_t> tiny_model_gguf(uint32_t seed, nibble::DType wtype = nibble::DType::F32,
                                            bool zero_output_norm = false, int32_t eos_id = 259,
                                            const TinyModelOpts& mo = {}) {
  using nibble::DType;
  constexpr int64_t H = 4, KV = 2, V = 261, L = 2;
  const int64_t E = mo.quant_dims ? 32 : 16, HD = E / H, FF = mo.quant_dims ? 64 : 32;
  GgufWriter w;
  w.add_string("general.architecture", mo.arch);
  const std::string a = mo.arch + ".";
  w.add_int(a + "block_count", GgufWriter::ValueType::U32, L);
  w.add_int(a + "embedding_length", GgufWriter::ValueType::U32, E);
  w.add_int(a + "feed_forward_length", GgufWriter::ValueType::U32, FF);
  w.add_int(a + "attention.head_count", GgufWriter::ValueType::U32, H);
  w.add_int(a + "attention.head_count_kv", GgufWriter::ValueType::U32, KV);
  w.add_int(a + "context_length", GgufWriter::ValueType::U32, 64);
  w.add_f32(a + "attention.layer_norm_rms_epsilon", 1e-6f);
  w.add_f32(a + "rope.freq_base", 10000.0f);
  add_tiny_tokenizer_kvs(w, {}, eos_id);

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
  if (mo.explicit_output) mat("output.weight", E, V);
  for (int64_t i = 0; i < L; ++i) {
    const std::string p = "blk." + std::to_string(i) + ".";
    vec(p + "attn_norm.weight", E, 0.9f, 1.1f);
    mat(p + "attn_q.weight", E, H * HD);
    mat(p + "attn_k.weight", E, KV * HD);
    mat(p + "attn_v.weight", E, KV * HD);
    mat(p + "attn_output.weight", H * HD, E);
    vec(p + "attn_q.bias", H * HD, -0.5f, 0.5f);
    vec(p + "attn_k.bias", KV * HD, -0.5f, 0.5f);
    vec(p + "attn_v.bias", KV * HD, -0.5f, 0.5f);
    vec(p + "ffn_norm.weight", E, 0.9f, 1.1f);
    mat(p + "ffn_gate.weight", E, FF);
    mat(p + "ffn_up.weight", E, FF);
    mat(p + "ffn_down.weight", FF, E);
  }
  return w.build();
}
