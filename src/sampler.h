#pragma once
#include <cstdint>
#include <span>
#include <vector>

namespace nibble {

struct SamplerParams {
  float temperature;
  int top_k;
  float top_p;
  uint64_t seed;
};

class Sampler {
 public:
  Sampler(const SamplerParams&, int64_t n_vocab);
  int32_t sample(std::span<const float> logits);  // never allocates

 private:
  uint64_t next();

  SamplerParams params_;
  int64_t n_vocab_;
  uint64_t state_[4];
  std::vector<int32_t> idx_;
  std::vector<double> prob_;
};

}  // namespace nibble
