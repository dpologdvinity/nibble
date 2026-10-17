#include "model/transformer.h"

#include <algorithm>
#include <cstdint>
#include <string>

#include "nibble/error.h"

namespace nibble::model {

Transformer::Transformer(const LlamaConfig& c, const Weights& w, Backend& be, int64_t n_ctx, int64_t max_batch)
    : c_(c), w_(w), be_(be), n_ctx_(n_ctx), max_batch_(max_batch) {
  if (n_ctx <= 0 || max_batch <= 0) throw Error("n_ctx and max_batch must be positive");
  try {
    be.reserve(n_ctx, c.n_heads);
  } catch (const Error& e) {
    throw Error(std::string("cannot allocate attention scratch for n_ctx=") + std::to_string(n_ctx) +
                "; try a smaller n_ctx (-c): " + e.what());
  }
  const int64_t B = max_batch, E = c.n_embd;
  // KV cache size with overflow checks; spec 6.2 message on failure.
  uint64_t kv_floats = static_cast<uint64_t>(n_ctx), kv_bytes = 0;
  const bool kv_overflow = __builtin_mul_overflow(kv_floats, static_cast<uint64_t>(c.n_kv_heads), &kv_floats) ||
                           __builtin_mul_overflow(kv_floats, static_cast<uint64_t>(c.head_dim), &kv_floats) ||
                           __builtin_mul_overflow(kv_floats, uint64_t{2 * sizeof(float)}, &kv_bytes) ||
                           __builtin_mul_overflow(kv_bytes, static_cast<uint64_t>(c.n_layers), &kv_bytes) ||
                           kv_floats > static_cast<uint64_t>(INT64_MAX);
  const std::string kv_msg_tail = " for n_ctx=" + std::to_string(n_ctx) + "; try a smaller n_ctx (-c)";
  if (kv_overflow) throw Error("cannot allocate KV cache: size overflows" + kv_msg_tail);
  try {
    for (int64_t i = 0; i < c.n_layers; ++i) {
      k_.push_back(grab(static_cast<int64_t>(kv_floats)));
      v_.push_back(grab(static_cast<int64_t>(kv_floats)));
    }
  } catch (const Error&) {
    release();
    throw Error("cannot allocate KV cache: " + std::to_string(kv_bytes) + " bytes" + kv_msg_tail);
  }
  try {
    x_ = grab(B * E);
    h_ = grab(B * E);
    q_ = grab(B * c.n_heads * c.head_dim);
    attn_ = grab(B * c.n_heads * c.head_dim);
    gate_ = grab(B * c.n_ff);
    up_ = grab(B * c.n_ff);
    last_ = grab(E);
    logits_ = grab(c.n_vocab);
  } catch (...) {
    release();
    throw;
  }
}

float* Transformer::grab(int64_t n_floats) {
  float* p = be_.alloc(n_floats);
  try {
    owned_.push_back(p);
  } catch (...) {
    be_.free(p);
    throw;
  }
  return p;
}

void Transformer::release() noexcept {
  for (float* p : owned_) be_.free(p);
  owned_.clear();
}

Transformer::~Transformer() { release(); }

void Transformer::forward(std::span<const int32_t> tokens, int64_t pos0, float* logits_out) {
  const int64_t n = static_cast<int64_t>(tokens.size());
  if (n == 0) throw Error("forward: no tokens");
  if (pos0 < 0 || pos0 + n > n_ctx_)
    throw Error("context overflow: position " + std::to_string(pos0) + " + " + std::to_string(n) +
                " tokens exceeds n_ctx " + std::to_string(n_ctx_));
  for (int64_t off = 0; off < n; off += max_batch_)
    run_chunk(tokens.subspan(static_cast<size_t>(off), static_cast<size_t>(std::min(max_batch_, n - off))), pos0 + off);
  const int64_t rows = (n - 1) % max_batch_;  // last row within the final chunk
  be_.rmsnorm(x_ + rows * c_.n_embd, w_.output_norm, last_, 1, c_.rms_eps);
  be_.matmul(w_.output, last_, logits_, 1);
  be_.read(logits_out, logits_, c_.n_vocab);
}

void Transformer::run_chunk(std::span<const int32_t> tokens, int64_t pos0) {
  const int64_t n = static_cast<int64_t>(tokens.size()), hd = c_.head_dim, E = c_.n_embd;
  const int64_t kvw = c_.n_kv_heads * hd;
  const RopeParams rp{c_.rope_base, c_.rope_neox, nullptr};
  be_.embed(w_.token_embd, tokens, x_);
  for (int64_t l = 0; l < c_.n_layers; ++l) {
    const LayerWeights& L = w_.layers[static_cast<size_t>(l)];
    // K and V are projected straight into their cache rows for positions pos0..pos0+n.
    float* kc = k_[static_cast<size_t>(l)] + pos0 * kvw;
    float* vc = v_[static_cast<size_t>(l)] + pos0 * kvw;
    be_.rmsnorm(x_, L.attn_norm, h_, n, c_.rms_eps);
    be_.matmul(L.wq, h_, q_, n);
    be_.matmul(L.wk, h_, kc, n);
    be_.matmul(L.wv, h_, vc, n);
    if (L.bq) be_.add_bias(q_, *L.bq, n);
    if (L.bk) be_.add_bias(kc, *L.bk, n);
    if (L.bv) be_.add_bias(vc, *L.bv, n);
    be_.rope(q_, n, c_.n_heads, hd, pos0, rp);
    be_.rope(kc, n, c_.n_kv_heads, hd, pos0, rp);
    be_.attention(q_, k_[static_cast<size_t>(l)], v_[static_cast<size_t>(l)], attn_, n, pos0, c_.n_heads,
                  c_.n_kv_heads, hd);
    be_.matmul(L.wo, attn_, h_, n);
    be_.add(x_, h_, n * E);
    be_.rmsnorm(x_, L.ffn_norm, h_, n, c_.rms_eps);
    be_.matmul(L.w_gate, h_, gate_, n);
    be_.matmul(L.w_up, h_, up_, n);
    be_.silu_mul(gate_, up_, n * c_.n_ff);
    be_.matmul(L.w_down, gate_, h_, n);
    be_.add(x_, h_, n * E);
  }
}

}  // namespace nibble::model
