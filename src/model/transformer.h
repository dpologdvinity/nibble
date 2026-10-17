#pragma once
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
  // Allocates the KV cache and all scratch from the backend; calls backend.reserve().
  Transformer(const LlamaConfig&, const Weights&, Backend&, int64_t n_ctx, int64_t max_batch = 512);
  // Runs tokens at positions pos0.. (chunked by max_batch); writes the last token's
  // logits (n_vocab floats) to host memory. Throws Error if pos0 + tokens.size() > n_ctx.
  void forward(std::span<const int32_t> tokens, int64_t pos0, float* logits_out);
  int64_t n_ctx() const { return n_ctx_; }

 private:
  float* grab(int64_t n_floats);  // backend alloc, recorded for ~Transformer
  void release() noexcept;
  void run_chunk(std::span<const int32_t> tokens, int64_t pos0);

  LlamaConfig c_;
  const Weights& w_;
  Backend& be_;
  int64_t n_ctx_, max_batch_;
  std::vector<float*> owned_;  // every buffer allocated from be_
  std::vector<float*> k_, v_;  // per layer [n_ctx, n_kv, hd]
  float *x_, *h_, *q_, *attn_, *gate_, *up_, *last_, *logits_;
};

}  // namespace nibble::model
