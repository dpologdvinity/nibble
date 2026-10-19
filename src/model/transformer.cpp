#include "model/transformer.h"

#include <algorithm>
#include <cstdint>
#include <string>

#include "debug_check.h"
#include "nibble/error.h"

namespace nibble::model {
namespace {

// Debug-only shape/dtype agreement checks at op call sites (spec 6.4). Activation
// pointers carry no shape, so the expected widths passed here are the widths the
// Transformer sized each activation buffer with. Compiled out under NDEBUG.
void dcheck_matmul(const Backend& be, const Tensor& w, int64_t in, int64_t out, [[maybe_unused]] const char* op) {
  NIBBLE_DCHECK(w.n_dims == 2 && w.ne[0] == in && w.ne[1] == out && be.supports(w.type),
                "matmul %s: weight [%lld, %lld] (n_dims %d, dtype %u) vs activation in=%lld out=%lld", op,
                static_cast<long long>(w.ne[0]), static_cast<long long>(w.ne[1]), w.n_dims,
                static_cast<unsigned>(w.type), static_cast<long long>(in), static_cast<long long>(out));
}

void dcheck_vec(const Tensor& t, int64_t width, [[maybe_unused]] const char* op) {
  NIBBLE_DCHECK(t.n_dims == 1 && t.ne[0] == width && t.type == DType::F32,
                "%s: tensor [%lld] (n_dims %d, dtype %u) vs activation width %lld (need F32)", op,
                static_cast<long long>(t.ne[0]), t.n_dims, static_cast<unsigned>(t.type),
                static_cast<long long>(width));
}

// rope/attention: the head geometry passed to the op must match the config and the
// row width of the activation buffer, and the rows being written must fit the context.
void dcheck_heads(int64_t heads, int64_t want_heads, int64_t head_dim, int64_t want_head_dim, int64_t row_width,
                  int64_t pos0, int64_t n, int64_t n_ctx, [[maybe_unused]] const char* op) {
  NIBBLE_DCHECK(heads == want_heads && head_dim == want_head_dim && heads * head_dim == row_width && pos0 >= 0 &&
                    pos0 + n <= n_ctx,
                "%s: heads=%lld head_dim=%lld vs config heads=%lld head_dim=%lld, row width %lld; "
                "pos0=%lld + n=%lld vs n_ctx=%lld",
                op, static_cast<long long>(heads), static_cast<long long>(head_dim),
                static_cast<long long>(want_heads), static_cast<long long>(want_head_dim),
                static_cast<long long>(row_width), static_cast<long long>(pos0), static_cast<long long>(n),
                static_cast<long long>(n_ctx));
}

// add/silu_mul: the element count must be rows x the activation width.
void dcheck_count(int64_t count, int64_t rows, int64_t width, [[maybe_unused]] const char* op) {
  NIBBLE_DCHECK(count == rows * width, "%s: count=%lld vs rows=%lld x width=%lld", op,
                static_cast<long long>(count), static_cast<long long>(rows), static_cast<long long>(width));
}

}  // namespace

Transformer::Transformer(const LlamaConfig& c, const Weights& w, Backend& be, int64_t n_ctx, int64_t max_batch)
    : c_(c), w_(w), be_(be), n_ctx_(n_ctx), max_batch_(max_batch) {
  if (n_ctx <= 0 || max_batch <= 0) throw Error("n_ctx and max_batch must be positive");
  if (static_cast<int64_t>(w.layers.size()) != c.n_layers)
    throw Error("weights have " + std::to_string(w.layers.size()) + " layers but config has " +
                std::to_string(c.n_layers));
  try {
    be.reserve({n_ctx, c.n_heads, max_batch, std::max({c.n_embd, c.n_heads * c.head_dim, c.n_ff})});
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
  NIBBLE_DCHECK(logits_out != nullptr, "forward: null logits_out");
  for (int64_t off = 0; off < n; off += max_batch_)
    run_chunk(tokens.subspan(static_cast<size_t>(off), static_cast<size_t>(std::min(max_batch_, n - off))), pos0 + off);
  const int64_t rows = (n - 1) % max_batch_;  // last row within the final chunk
  dcheck_vec(w_.output_norm, c_.n_embd, "rmsnorm output_norm");
  dcheck_matmul(be_, w_.output, c_.n_embd, c_.n_vocab, "output");
  be_.rmsnorm(x_ + rows * c_.n_embd, w_.output_norm, last_, 1, c_.rms_eps);
  be_.matmul(w_.output, last_, logits_, 1);
  be_.read(logits_out, logits_, c_.n_vocab);
}

void Transformer::run_chunk(std::span<const int32_t> tokens, int64_t pos0) {
  const int64_t n = static_cast<int64_t>(tokens.size()), hd = c_.head_dim, E = c_.n_embd;
  const int64_t kvw = c_.n_kv_heads * hd;
  const RopeParams rp{c_.rope_base, c_.rope_neox,
                       w_.rope_freqs ? static_cast<const float*>(w_.rope_freqs->data) : nullptr};
  const int64_t Q = c_.n_heads * hd;
  NIBBLE_DCHECK(n > 0 && n <= max_batch_ && pos0 >= 0 && pos0 + n <= n_ctx_,
                "run_chunk: n=%lld pos0=%lld vs max_batch=%lld n_ctx=%lld", static_cast<long long>(n),
                static_cast<long long>(pos0), static_cast<long long>(max_batch_), static_cast<long long>(n_ctx_));
  NIBBLE_DCHECK(c_.n_kv_heads > 0 && c_.n_heads % c_.n_kv_heads == 0 && hd > 0 && hd % 2 == 0,
                "attention config: n_heads=%lld n_kv_heads=%lld head_dim=%lld", static_cast<long long>(c_.n_heads),
                static_cast<long long>(c_.n_kv_heads), static_cast<long long>(hd));
  NIBBLE_DCHECK(w_.layers.size() == static_cast<size_t>(c_.n_layers), "layers: %zu bound vs n_layers=%lld",
                w_.layers.size(), static_cast<long long>(c_.n_layers));
  if (w_.rope_freqs) {
    dcheck_vec(*w_.rope_freqs, hd / 2, "rope rope_freqs");
    NIBBLE_DCHECK(w_.rope_freqs->data && reinterpret_cast<uintptr_t>(w_.rope_freqs->data) % alignof(float) == 0,
                  "rope rope_freqs: data pointer null or misaligned");
  }
  dcheck_matmul(be_, w_.token_embd, E, c_.n_vocab, "embed token_embd");
  be_.embed(w_.token_embd, tokens, x_);
  for (int64_t l = 0; l < c_.n_layers; ++l) {
    const LayerWeights& L = w_.layers[static_cast<size_t>(l)];
    // K and V are projected straight into their cache rows for positions pos0..pos0+n.
    float* kc = k_[static_cast<size_t>(l)] + pos0 * kvw;
    float* vc = v_[static_cast<size_t>(l)] + pos0 * kvw;
    dcheck_vec(L.attn_norm, E, "rmsnorm attn_norm");
    be_.rmsnorm(x_, L.attn_norm, h_, n, c_.rms_eps);
    dcheck_matmul(be_, L.wq, E, Q, "attn_q");
    dcheck_matmul(be_, L.wk, E, kvw, "attn_k");
    dcheck_matmul(be_, L.wv, E, kvw, "attn_v");
    const Tensor* const qkv_w[] = {&L.wq, &L.wk, &L.wv};
    float* const qkv_y[] = {q_, kc, vc};
    be_.matmul_multi(qkv_w, h_, qkv_y, n);
    NIBBLE_DCHECK(L.bq.has_value() == c_.qkv_bias && L.bk.has_value() == c_.qkv_bias &&
                      L.bv.has_value() == c_.qkv_bias,
                  "qkv bias presence vs config qkv_bias=%d", static_cast<int>(c_.qkv_bias));
    if (L.bq) {
      dcheck_vec(*L.bq, Q, "add_bias attn_q.bias");
      be_.add_bias(q_, *L.bq, n);
    }
    if (L.bk) {
      dcheck_vec(*L.bk, kvw, "add_bias attn_k.bias");
      be_.add_bias(kc, *L.bk, n);
    }
    if (L.bv) {
      dcheck_vec(*L.bv, kvw, "add_bias attn_v.bias");
      be_.add_bias(vc, *L.bv, n);
    }
    dcheck_heads(c_.n_heads, c_.n_heads, hd, c_.head_dim, Q, pos0, n, n_ctx_, "rope q");
    be_.rope(q_, n, c_.n_heads, hd, pos0, rp);   // q_ is [n, n_heads * head_dim] (checked via attn_q)
    dcheck_heads(c_.n_kv_heads, c_.n_kv_heads, hd, c_.head_dim, kvw, pos0, n, n_ctx_, "rope k");
    be_.rope(kc, n, c_.n_kv_heads, hd, pos0, rp);  // kc is [n, n_kv_heads * head_dim] (checked via attn_k)
    dcheck_heads(c_.n_heads, c_.n_heads, hd, c_.head_dim, Q, pos0, n, n_ctx_, "attention q heads");
    dcheck_heads(c_.n_kv_heads, c_.n_kv_heads, hd, c_.head_dim, kvw, pos0, n, n_ctx_, "attention kv heads");
    be_.attention(q_, k_[static_cast<size_t>(l)], v_[static_cast<size_t>(l)], attn_, n, pos0, c_.n_heads,
                  c_.n_kv_heads, hd);
    dcheck_matmul(be_, L.wo, Q, E, "attn_output");
    be_.matmul(L.wo, attn_, h_, n);
    dcheck_count(n * E, n, c_.n_embd, "add attn residual");
    be_.add(x_, h_, n * E);
    dcheck_vec(L.ffn_norm, E, "rmsnorm ffn_norm");
    be_.rmsnorm(x_, L.ffn_norm, h_, n, c_.rms_eps);
    dcheck_matmul(be_, L.w_gate, E, c_.n_ff, "ffn_gate");
    dcheck_matmul(be_, L.w_up, E, c_.n_ff, "ffn_up");
    const Tensor* const ffn_w[] = {&L.w_gate, &L.w_up};
    float* const ffn_y[] = {gate_, up_};
    be_.matmul_multi(ffn_w, h_, ffn_y, n);
    dcheck_count(n * c_.n_ff, n, c_.n_ff, "silu_mul");
    be_.silu_mul(gate_, up_, n * c_.n_ff);
    dcheck_matmul(be_, L.w_down, c_.n_ff, E, "ffn_down");
    be_.matmul(L.w_down, gate_, h_, n);
    dcheck_count(n * E, n, c_.n_embd, "add ffn residual");
    be_.add(x_, h_, n * E);
  }
}

}  // namespace nibble::model
