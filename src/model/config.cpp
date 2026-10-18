#include "model/config.h"

#include "nibble/error.h"
#include "util/escape.h"

namespace nibble::model {

LlamaConfig LlamaConfig::from_gguf(const gguf::File& f) {
  LlamaConfig c;
  c.arch = f.get_string("general.architecture");
  if (c.arch != "qwen2" && c.arch != "llama")
    throw Error("unsupported architecture '" + escape_control(c.arch) + "' (supported: qwen2, llama)");
  const std::string a = c.arch + ".";
  c.n_layers = f.get_int(a + "block_count");
  c.n_embd = f.get_int(a + "embedding_length");
  c.n_ff = f.get_int(a + "feed_forward_length");
  c.n_heads = f.get_int(a + "attention.head_count");
  c.n_kv_heads = f.find(a + "attention.head_count_kv") ? f.get_int(a + "attention.head_count_kv") : c.n_heads;
  c.rope_base = f.find(a + "rope.freq_base") ? static_cast<float>(f.get_float(a + "rope.freq_base")) : 10000.f;
  c.rms_eps = static_cast<float>(f.get_float(a + "attention.layer_norm_rms_epsilon"));
  c.n_ctx_train = f.get_int(a + "context_length");
  c.rope_neox = c.qkv_bias = c.arch == "qwen2";
  for (auto [k, v] : {std::pair<const char*, int64_t>{"block_count", c.n_layers},
                      {"embedding_length", c.n_embd},
                      {"feed_forward_length", c.n_ff},
                      {"attention.head_count", c.n_heads},
                      {"attention.head_count_kv", c.n_kv_heads}})
    if (v <= 0) throw Error("invalid " + a + k + " = " + std::to_string(v) + " (must be positive)");
  if (c.n_embd % c.n_heads != 0)
    throw Error("invalid model: " + a + "embedding_length " + std::to_string(c.n_embd) + " is not divisible by " + a +
                "attention.head_count " + std::to_string(c.n_heads));
  if (c.n_heads % c.n_kv_heads != 0)
    throw Error("invalid model: " + a + "attention.head_count " + std::to_string(c.n_heads) +
                " is not divisible by " + a + "attention.head_count_kv " + std::to_string(c.n_kv_heads));
  c.head_dim = c.n_embd / c.n_heads;
  if (c.head_dim % 2 != 0) throw Error("invalid model: head_dim " + std::to_string(c.head_dim) + " must be even for rope");
  for (const char* k : {"rope.dimension_count", "attention.key_length"})
    if (f.find(a + k) && f.get_int(a + k) != c.head_dim)
      throw Error("invalid model: " + a + k + " = " + std::to_string(f.get_int(a + k)) +
                  " but embedding_length / head_count = " + std::to_string(c.head_dim));
  const auto* te = f.find_tensor("token_embd.weight");
  if (!te) throw Error("missing tensor token_embd.weight");
  c.n_vocab = te->tensor.ne[1];
  return c;
}

}  // namespace nibble::model
