#pragma once
// Next-token sampling from logits: greedy, or temperature + top-k + top-p with a seeded generator. The
// Sampler allocates its buffers at construction, so sample() never allocates in the decode loop.
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace nibble {

// Sampling configuration (the numeric core of GenParams; see nibble.h for the user-facing semantics).
struct SamplerParams {
  float temperature;  // 0 = greedy; > 0 scales logits by 1/temperature
  int top_k;          // keep the k best tokens; <= 0 or >= n_vocab keeps all
  float top_p;        // keep the smallest top set with cumulative probability >= top_p; >= 1 keeps all
  uint64_t seed;      // seeds the xoshiro256** generator (via splitmix64)
};

// Next value of a splitmix64 stream; advances x.
uint64_t splitmix64(uint64_t& x);

// Index drawn by u in [0,1) from probabilities summing to ~1. If rounding leaves
// the cumulative sum <= u, returns the last index with nonzero probability (0 if none).
size_t pick_index(const double* prob, size_t n, double u);

// Draws token ids from logits. One instance per generate call; not thread-safe.
class Sampler {
 public:
  // Throws Error for a NaN or negative temperature or top_p, or n_vocab <= 0.
  Sampler(const SamplerParams&, int64_t n_vocab);
  // Returns the chosen token id. Greedy (temperature 0) picks the highest logit, lowest id on ties. NaN
  // logits count as -inf; a +inf logit wins outright. Throws Error if logits.size() != n_vocab or no logit is
  // usable (all NaN / all -inf). Never allocates.
  int32_t sample(std::span<const float> logits);


 private:
  uint64_t next();  // xoshiro256** step: the next 64 random bits

  SamplerParams params_;
  int64_t n_vocab_;
  uint64_t state_[4];
  std::vector<int32_t> idx_;
  std::vector<double> prob_;
};

}  // namespace nibble
