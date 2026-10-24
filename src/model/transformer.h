#pragma once
// The decoder-only transformer forward pass (RMSNorm, grouped-query attention with rotary embeddings, SwiGLU
// feed-forward), written against the abstract Backend so the same code could drive a GPU backend later.
//
// A Transformer owns the KV cache and every activation buffer, all allocated up front from the backend, so
// forward() performs no allocation. It is stateful: the KV cache holds the keys/values of all tokens fed so
// far, and `pos0` tells forward() where the new tokens go. Not thread-safe.
#include <cstdint>
#include <span>
#include <vector>

#include "model/weights.h"

namespace nibble::model {

class Transformer {
 public:
  ~Transformer();
  Transformer(const Transformer&) = delete;
  Transformer& operator=(const Transformer&) = delete;
  // Allocates the KV cache (n_ctx positions per layer) and all scratch for up to max_batch tokens per chunk
  // from the backend, and calls backend.reserve() with the matching sizes. Throws Error, with a hint to
  // reduce n_ctx, if an allocation fails or a size overflows.
  Transformer(const LlamaConfig&, const Weights&, Backend&, int64_t n_ctx, int64_t max_batch = 512);
  // Runs tokens at positions pos0.. (chunked by max_batch); writes the last token's
  // logits (n_vocab floats) to host memory. Throws Error if pos0 + tokens.size() > n_ctx.
  void forward(std::span<const int32_t> tokens, int64_t pos0, float* logits_out);
  int64_t n_ctx() const { return n_ctx_; }  // KV cache capacity in tokens

 private:
  float* grab(int64_t n_floats);  // backend alloc, recorded for ~Transformer
  void release() noexcept;        // free every owned buffer
  // Runs one chunk of at most max_batch tokens through all layers, leaving the final hidden states in x_
  // and appending the chunk's K/V to the cache at positions pos0...
  void run_chunk(std::span<const int32_t> tokens, int64_t pos0);

  LlamaConfig c_;
  const Weights& w_;   // borrowed; must outlive the Transformer
  Backend& be_;        // borrowed
  int64_t n_ctx_, max_batch_;
  std::vector<float*> owned_;  // every buffer allocated from be_
  std::vector<float*> k_, v_;  // per layer [n_ctx, n_kv, hd]
  // Activations (rows = tokens of the current chunk): x_ residual stream, h_ normed input / matmul output,
  // q_ queries, attn_ attention output, gate_/up_ feed-forward hidden, last_ final normed last row,
  // logits_ output logits before the copy to the host.
  float *x_, *h_, *q_, *attn_, *gate_, *up_, *last_, *logits_;
};

}  // namespace nibble::model
