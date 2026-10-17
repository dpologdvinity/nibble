#pragma once
#include <cstdint>
#include <string>

#include "gguf/gguf.h"

namespace nibble::model {

struct LlamaConfig {
  std::string arch;
  int64_t n_layers = 0, n_embd = 0, n_ff = 0, n_heads = 0, n_kv_heads = 0, head_dim = 0, n_vocab = 0,
          n_ctx_train = 0;
  float rope_base = 10000.f, rms_eps = 1e-6f;
  bool rope_neox = true, qkv_bias = true;

  static LlamaConfig from_gguf(const gguf::File&);
};

}  // namespace nibble::model
