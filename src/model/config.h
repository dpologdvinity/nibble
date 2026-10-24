#pragma once
// Hyperparameters of the decoder-only transformer, read from GGUF metadata. One struct covers both supported
// architectures because they differ only in a few flags: "qwen2" (Qwen2.5: biases on Q/K/V, NeoX-style rope)
// and "llama" (Llama 3.x and TinyLlama: no biases, interleaved rope).
#include <cstdint>
#include <string>

#include "gguf/gguf.h"

namespace nibble::model {

// Shape and numeric constants of the network. All dimension fields are validated positive and mutually
// consistent by from_gguf, so the rest of the code can use them without re-checking.
struct LlamaConfig {
  std::string arch;           // GGUF general.architecture: "qwen2" or "llama"
  int64_t n_layers = 0;       // number of transformer blocks
  int64_t n_embd = 0;         // model (residual stream) width
  int64_t n_ff = 0;           // feed-forward hidden width
  int64_t n_heads = 0;        // query heads
  int64_t n_kv_heads = 0;     // key/value heads (grouped-query attention when < n_heads; divides n_heads)
  int64_t head_dim = 0;       // n_embd / n_heads, always even (rope rotates pairs)
  int64_t n_vocab = 0;        // rows of token_embd.weight
  int64_t n_ctx_train = 0;    // context length the model was trained for (GGUF context_length)
  float rope_base = 10000.f;  // rotary embedding frequency base (rope.freq_base)
  float rms_eps = 1e-6f;      // epsilon inside RMSNorm
  bool rope_neox = true;      // rotate (i, i + head_dim/2) pairs (qwen2) instead of adjacent pairs (llama)
  bool qkv_bias = true;       // attention Q/K/V projections have bias vectors (qwen2 only)

  // Reads and validates the metadata. Throws Error for an unsupported architecture, missing keys,
  // non-positive or inconsistent dimensions (e.g. embedding_length not divisible by head_count), or a missing
  // token_embd.weight tensor (n_vocab comes from its shape).
  static LlamaConfig from_gguf(const gguf::File&);
};

}  // namespace nibble::model
