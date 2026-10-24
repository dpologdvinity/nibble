// Resolves the transformer's tensors by their GGUF names (token_embd.weight, blk.N.attn_q.weight, ...).
#include "model/weights.h"

#include <initializer_list>
#include <string>

#include "nibble/error.h"
#include "util/escape.h"

namespace nibble::model {
namespace {

// "[a, b, c]" for error messages.
std::string shape_str(const int64_t* ne, int n) {
  std::string s = "[";
  for (int i = 0; i < n; ++i) s += (i ? ", " : "") + std::to_string(ne[i]);
  return s + "]";
}

// The named tensor, or nullptr.
const Tensor* find(const gguf::File& f, const std::string& name) {
  const auto* ti = f.find_tensor(name);
  return ti ? &ti->tensor : nullptr;
}

// The named tensor, checked: it must exist, have exactly the dimensions `ne`, and either be F32 (f32_only) or
// a type the backend supports. Error messages name the tensor and, for types, the supported set.
Tensor get(const gguf::File& f, const Backend& be, const std::string& name, std::initializer_list<int64_t> ne,
           bool f32_only) {
  const Tensor* t = find(f, name);
  if (!t) throw Error("missing tensor " + escape_control(name));
  bool ok = t->n_dims == static_cast<int>(ne.size());
  int i = 0;
  for (int64_t d : ne) ok = ok && t->ne[static_cast<size_t>(i++)] == d;
  if (!ok)
    throw Error("tensor " + escape_control(name) + ": expected " +
                shape_str(std::data(ne), static_cast<int>(ne.size())) + ", got " + shape_str(t->ne.data(), t->n_dims));
  const std::string tn = dtype_info(t->type).name;
  if (f32_only ? t->type != DType::F32 : !be.supports(t->type))
    throw Error("unsupported tensor type " + tn + " for " + escape_control(name) +
                (f32_only ? " — this tensor must be F32"
                          : std::string(" — supported: ") + kSupportedTypes + "; convert with llama-quantize"));
  return *t;
}

}  // namespace

// Binds in file order of use: embedding, output norm, (optional) separate output matrix and rope factors,
// then each layer. A missing output.weight means tied embeddings (the embedding doubles as output matrix).
Weights Weights::bind(const gguf::File& f, const LlamaConfig& c, const Backend& be) {
  const int64_t E = c.n_embd, Q = c.n_heads * c.head_dim, KV = c.n_kv_heads * c.head_dim;
  Weights w;
  w.token_embd = get(f, be, "token_embd.weight", {E, c.n_vocab}, false);
  w.output_norm = get(f, be, "output_norm.weight", {E}, true);
  w.output = find(f, "output.weight") ? get(f, be, "output.weight", {E, c.n_vocab}, false) : w.token_embd;
  if (find(f, "rope_freqs.weight")) w.rope_freqs = get(f, be, "rope_freqs.weight", {c.head_dim / 2}, true);
  for (int64_t i = 0; i < c.n_layers; ++i) {
    const std::string p = "blk." + std::to_string(i) + ".";
    LayerWeights l;
    l.attn_norm = get(f, be, p + "attn_norm.weight", {E}, true);
    l.wq = get(f, be, p + "attn_q.weight", {E, Q}, false);
    l.wk = get(f, be, p + "attn_k.weight", {E, KV}, false);
    l.wv = get(f, be, p + "attn_v.weight", {E, KV}, false);
    l.wo = get(f, be, p + "attn_output.weight", {Q, E}, false);
    if (c.qkv_bias) {
      l.bq = get(f, be, p + "attn_q.bias", {Q}, true);
      l.bk = get(f, be, p + "attn_k.bias", {KV}, true);
      l.bv = get(f, be, p + "attn_v.bias", {KV}, true);
    } else {
      for (const char* b : {"attn_q.bias", "attn_k.bias", "attn_v.bias"})
        if (find(f, p + b))
          throw Error("unsupported tensor " + escape_control(p + b) + " for architecture " + escape_control(c.arch));
    }
    l.ffn_norm = get(f, be, p + "ffn_norm.weight", {E}, true);
    l.w_gate = get(f, be, p + "ffn_gate.weight", {E, c.n_ff}, false);
    l.w_up = get(f, be, p + "ffn_up.weight", {E, c.n_ff}, false);
    l.w_down = get(f, be, p + "ffn_down.weight", {c.n_ff, E}, false);
    w.layers.push_back(std::move(l));
  }
  return w;
}

}  // namespace nibble::model
