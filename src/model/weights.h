#pragma once
#include <optional>
#include <vector>

#include "backend/backend.h"
#include "model/config.h"

namespace nibble::model {

struct LayerWeights {
  Tensor attn_norm, wq, wk, wv, wo, ffn_norm, w_gate, w_up, w_down;
  std::optional<Tensor> bq, bk, bv;
};

// Tensors point into the mapped gguf::File, which must outlive the Weights.
struct Weights {
  Tensor token_embd, output_norm, output;
  std::vector<LayerWeights> layers;
  static Weights bind(const gguf::File&, const LlamaConfig&, const Backend&);
};

}  // namespace nibble::model
