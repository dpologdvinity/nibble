#pragma once
// Independent plain-loop qwen2 forward pass for tests: no Backend, no model/ code.
#include <cmath>
#include <span>
#include <string>
#include <vector>

#include "fp16.h"
#include "gguf/gguf.h"
#include "quant.h"

namespace naive {

inline std::vector<double> load(const nibble::gguf::File& f, const std::string& name) {
  const auto* ti = f.find_tensor(name);
  if (!ti) throw std::runtime_error("naive: missing " + name);
  int64_t n = 1;
  for (int i = 0; i < ti->tensor.n_dims; ++i) n *= ti->tensor.ne[static_cast<size_t>(i)];
  std::vector<double> v(static_cast<size_t>(n));
  if (ti->tensor.type == nibble::DType::Q4_0 || ti->tensor.type == nibble::DType::Q8_0) {
    std::vector<float> tmp(static_cast<size_t>(n));
    nibble::dequantize_row(ti->tensor.type, ti->tensor.data, tmp.data(), n);
    for (size_t i = 0; i < tmp.size(); ++i) v[i] = tmp[i];
    return v;
  }
  for (int64_t i = 0; i < n; ++i) {
    if (ti->tensor.type == nibble::DType::F32) {
      v[static_cast<size_t>(i)] = static_cast<const float*>(ti->tensor.data)[i];
    } else {
      v[static_cast<size_t>(i)] = nibble::fp16_to_fp32(static_cast<const uint16_t*>(ti->tensor.data)[i]);
    }
  }
  return v;
}

// y[o] = sum_i w[o*in + i] * x[i]
inline std::vector<double> mv(const std::vector<double>& w, const std::vector<double>& x, size_t out) {
  size_t in = x.size();
  std::vector<double> y(out, 0.0);
  for (size_t o = 0; o < out; ++o)
    for (size_t i = 0; i < in; ++i) y[o] += w[o * in + i] * x[i];
  return y;
}

inline std::vector<double> rms(const std::vector<double>& x, const std::vector<double>& w, double eps) {
  double ss = 0;
  for (double v : x) ss += v * v;
  double s = 1.0 / std::sqrt(ss / static_cast<double>(x.size()) + eps);
  std::vector<double> y(x.size());
  for (size_t i = 0; i < x.size(); ++i) y[i] = x[i] * s * w[i];
  return y;
}

inline std::vector<float> naive_logits(const nibble::gguf::File& f, std::span<const int32_t> tokens) {
  const size_t L = static_cast<size_t>(f.get_int("qwen2.block_count")), E = static_cast<size_t>(f.get_int("qwen2.embedding_length")),
               H = static_cast<size_t>(f.get_int("qwen2.attention.head_count")),
               KV = static_cast<size_t>(f.get_int("qwen2.attention.head_count_kv")), HD = E / H,
               FF = static_cast<size_t>(f.get_int("qwen2.feed_forward_length"));
  const double eps = f.get_float("qwen2.attention.layer_norm_rms_epsilon"), base = f.get_float("qwen2.rope.freq_base");
  auto emb = load(f, "token_embd.weight");
  const size_t V = emb.size() / E;
  const size_t T = tokens.size();
  std::vector<std::vector<double>> x(T);
  for (size_t t = 0; t < T; ++t)
    x[t].assign(emb.begin() + static_cast<long>(static_cast<size_t>(tokens[t]) * E),
                emb.begin() + static_cast<long>((static_cast<size_t>(tokens[t]) + 1) * E));

  auto rope = [&](std::vector<double>& v, size_t heads, size_t pos) {
    for (size_t h = 0; h < heads; ++h)
      for (size_t i = 0; i < HD / 2; ++i) {
        double th = static_cast<double>(pos) * std::pow(base, -2.0 * static_cast<double>(i) / static_cast<double>(HD));
        double c = std::cos(th), s = std::sin(th);
        double a = v[h * HD + i], b = v[h * HD + i + HD / 2];
        v[h * HD + i] = a * c - b * s;
        v[h * HD + i + HD / 2] = a * s + b * c;
      }
  };

  for (size_t l = 0; l < L; ++l) {
    const std::string p = "blk." + std::to_string(l) + ".";
    auto an = load(f, p + "attn_norm.weight"), wq = load(f, p + "attn_q.weight"), wk = load(f, p + "attn_k.weight"),
         wv = load(f, p + "attn_v.weight"), wo = load(f, p + "attn_output.weight"), bq = load(f, p + "attn_q.bias"),
         bk = load(f, p + "attn_k.bias"), bv = load(f, p + "attn_v.bias"), fn = load(f, p + "ffn_norm.weight"),
         wg = load(f, p + "ffn_gate.weight"), wu = load(f, p + "ffn_up.weight"), wd = load(f, p + "ffn_down.weight");
    std::vector<std::vector<double>> q(T), k(T), v(T);
    for (size_t t = 0; t < T; ++t) {
      auto h = rms(x[t], an, eps);
      q[t] = mv(wq, h, H * HD);
      k[t] = mv(wk, h, KV * HD);
      v[t] = mv(wv, h, KV * HD);
      for (size_t i = 0; i < q[t].size(); ++i) q[t][i] += bq[i];
      for (size_t i = 0; i < k[t].size(); ++i) k[t][i] += bk[i];
      for (size_t i = 0; i < v[t].size(); ++i) v[t][i] += bv[i];
      rope(q[t], H, t);
      rope(k[t], KV, t);
    }
    for (size_t t = 0; t < T; ++t) {
      std::vector<double> att(H * HD, 0.0);
      for (size_t h = 0; h < H; ++h) {
        size_t kvh = h / (H / KV);
        std::vector<double> sc(t + 1);
        double mx = -1e300;
        for (size_t s = 0; s <= t; ++s) {
          double d = 0;
          for (size_t i = 0; i < HD; ++i) d += q[t][h * HD + i] * k[s][kvh * HD + i];
          sc[s] = d / std::sqrt(static_cast<double>(HD));
          mx = std::max(mx, sc[s]);
        }
        double sum = 0;
        for (double& s : sc) sum += (s = std::exp(s - mx));
        for (size_t s = 0; s <= t; ++s)
          for (size_t i = 0; i < HD; ++i) att[h * HD + i] += sc[s] / sum * v[s][kvh * HD + i];
      }
      auto o = mv(wo, att, E);
      for (size_t i = 0; i < E; ++i) x[t][i] += o[i];
      auto h = rms(x[t], fn, eps);
      auto g = mv(wg, h, FF), u = mv(wu, h, FF);
      for (size_t i = 0; i < FF; ++i) g[i] = g[i] / (1.0 + std::exp(-g[i])) * u[i];
      auto d = mv(wd, g, E);
      for (size_t i = 0; i < E; ++i) x[t][i] += d[i];
    }
  }
  auto h = rms(x[T - 1], load(f, "output_norm.weight"), eps);
  auto out = f.find_tensor("output.weight") ? load(f, "output.weight") : emb;
  auto lg = mv(out, h, V);
  return std::vector<float>(lg.begin(), lg.end());
}

}  // namespace naive
