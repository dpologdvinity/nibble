#pragma once
// The model's parameters as Tensor views, resolved once by name from the GGUF file. Binding checks every
// tensor's presence, shape and type up front, so the forward pass can use them without further validation.
#include <optional>
#include <vector>

#include "backend/backend.h"
#include "model/config.h"

namespace nibble::model {

// The tensors of one transformer block. Matrices are [in, out] (ne[0] = input width). Norm weights are F32
// vectors; projections may be any supported quantized or float type.
struct LayerWeights {
  Tensor attn_norm, wq, wk, wv, wo, ffn_norm, w_gate, w_up, w_down;
  std::optional<Tensor> bq, bk, bv;  // F32 attention biases; present only when LlamaConfig::qkv_bias
};

// Every parameter of the network. Tensors point into the mapped gguf::File, which must outlive the Weights.
struct Weights {
  Tensor token_embd, output_norm, output;  // `output` is a copy of token_embd when the file ties them
  std::optional<Tensor> rope_freqs;  // F32 [head_dim / 2] frequency factors (llama 3.x)
  std::vector<LayerWeights> layers;
  // Looks up and validates every tensor the architecture needs. Throws Error naming the tensor when one is
  // missing, has the wrong shape, or has a type the backend cannot run (or a non-F32 type where F32 is
  // required), and for stray biases on an architecture that has none.
  static Weights bind(const gguf::File&, const LlamaConfig&, const Backend&);
};

}  // namespace nibble::model
