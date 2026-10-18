#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

#include "nibble/error.h"

namespace nibble {
namespace {

uint64_t splitmix64(uint64_t& x) {
  uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

// NaN sorts as -inf so orderings stay total.
float key(float x) { return std::isnan(x) ? -std::numeric_limits<float>::infinity() : x; }

uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

}  // namespace

Sampler::Sampler(const SamplerParams& p, int64_t n_vocab) : params_(p), n_vocab_(n_vocab) {
  if (std::isnan(p.temperature) || p.temperature < 0.0f)
    throw Error("sampler: invalid temperature " + std::to_string(p.temperature) +
                " (must be >= 0)");
  if (std::isnan(p.top_p) || p.top_p < 0.0f)
    throw Error("sampler: invalid top_p " + std::to_string(p.top_p) + " (must be >= 0)");
  if (n_vocab <= 0) throw Error("sampler: invalid n_vocab " + std::to_string(n_vocab));
  uint64_t x = p.seed;
  for (auto& s : state_) s = splitmix64(x);
  idx_.resize(static_cast<size_t>(n_vocab));
  prob_.resize(static_cast<size_t>(n_vocab));
}

uint64_t Sampler::next() {  // xoshiro256**
  const uint64_t result = rotl(state_[1] * 5, 7) * 9;
  const uint64_t t = state_[1] << 17;
  state_[2] ^= state_[0];
  state_[3] ^= state_[1];
  state_[1] ^= state_[2];
  state_[0] ^= state_[3];
  state_[2] ^= t;
  state_[3] = rotl(state_[3], 45);
  return result;
}

int32_t Sampler::sample(std::span<const float> logits) {
  if (static_cast<int64_t>(logits.size()) != n_vocab_)
    throw Error("sampler: got " + std::to_string(logits.size()) + " logits, expected " +
                std::to_string(n_vocab_));
  const auto better = [&](int32_t a, int32_t b) {
    const float la = key(logits[static_cast<size_t>(a)]), lb = key(logits[static_cast<size_t>(b)]);
    return la > lb || (la == lb && a < b);
  };
  int32_t* idx = idx_.data();
  const size_t n = idx_.size();

  // Special values: NaN counts as -inf. If any +inf, the lowest-index +inf wins
  // (deterministic, no RNG draw). If nothing is above -inf, throw.
  int32_t best = 0;
  bool any_non_nan = false;
  for (size_t i = 0; i < n; ++i) {
    any_non_nan |= !std::isnan(logits[i]);
    if (key(logits[i]) > key(logits[static_cast<size_t>(best)])) best = static_cast<int32_t>(i);
  }
  if (!any_non_nan) throw Error("sampler: all logits are NaN");
  const float top = key(logits[static_cast<size_t>(best)]);
  if (top == -std::numeric_limits<float>::infinity()) throw Error("sampler: no finite logits");
  if (top == std::numeric_limits<float>::infinity() || params_.temperature == 0.0f) return best;

  for (size_t i = 0; i < n; ++i) idx[i] = static_cast<int32_t>(i);
  size_t k = n;
  if (params_.top_k > 0 && static_cast<int64_t>(params_.top_k) < n_vocab_) {
    k = static_cast<size_t>(params_.top_k);
    std::nth_element(idx, idx + (k - 1), idx + n, better);
  }
  std::sort(idx, idx + k, better);  // descending probability

  const double inv_t = 1.0 / static_cast<double>(params_.temperature);
  const double mx = static_cast<double>(key(logits[static_cast<size_t>(idx[0])]));
  double sum = 0.0;
  for (size_t i = 0; i < k; ++i) {
    prob_[i] = std::exp((static_cast<double>(key(logits[static_cast<size_t>(idx[i])])) - mx) * inv_t);
    sum += prob_[i];
  }
  for (size_t i = 0; i < k; ++i) prob_[i] /= sum;

  size_t keep = k;
  if (params_.top_p < 1.0f) {
    double cum = 0.0;
    keep = 0;
    for (size_t i = 0; i < k; ++i) {
      cum += prob_[i];
      keep = i + 1;
      if (cum >= static_cast<double>(params_.top_p)) break;
    }
    double s2 = 0.0;
    for (size_t i = 0; i < keep; ++i) s2 += prob_[i];
    for (size_t i = 0; i < keep; ++i) prob_[i] /= s2;
  }

  const double u = static_cast<double>(next() >> 11) * 0x1p-53;
  double cum = 0.0;
  for (size_t i = 0; i < keep; ++i) {
    cum += prob_[i];
    if (cum > u) return idx[i];
  }
  return idx[keep - 1];  // rounding fallback
}

}  // namespace nibble
