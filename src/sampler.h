#pragma once
#include <cstddef>
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

// Next value of a splitmix64 stream; advances x.
uint64_t splitmix64(uint64_t& x);

// Index drawn by u in [0,1) from probabilities summing to ~1. If rounding leaves
// the cumulative sum <= u, returns the last index with nonzero probability (0 if none).
size_t pick_index(const double* prob, size_t n, double u);

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
