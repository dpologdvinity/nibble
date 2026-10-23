#include "backend/cpu/ref_ops.h"

#include <algorithm>
#include <cmath>

#include "fp16.h"
#include <nibble/error.h>

namespace nibble::cpu {

namespace {
[[noreturn]] void bad_type() { throw Error(std::string("cpu backend: unsupported weight dtype (only ") + kSupportedTypes + ")"); }

inline const uint16_t* as_f16(const Tensor& t) { return static_cast<const uint16_t*>(t.data); }
inline const float* as_f32(const Tensor& t) { return static_cast<const float*>(t.data); }

bool is_quant(DType t) {
  switch (t) {
    case DType::Q4_0: case DType::Q5_0: case DType::Q5_1: case DType::Q8_0:
    case DType::Q4_K: case DType::Q5_K: case DType::Q6_K:
      return true;
    default:
      return false;
  }
}

// One weight row against activation row `ar` (nb blocks of the weight's act_type).
float dot_row(DType t, const uint8_t* wr, const QuantActs& a, int64_t ar, int64_t nb) {
  switch (t) {
    case DType::Q4_0: return dot_q4_0_q8_0(reinterpret_cast<const BlockQ4_0*>(wr), a.q8_0 + ar * nb, nb);
    case DType::Q5_0: return dot_q5_0_q8_0(reinterpret_cast<const BlockQ5_0*>(wr), a.q8_0 + ar * nb, nb);
    case DType::Q8_0: return dot_q8_0_q8_0(reinterpret_cast<const BlockQ8_0*>(wr), a.q8_0 + ar * nb, nb);
    case DType::Q5_1: return dot_q5_1_q8_1(reinterpret_cast<const BlockQ5_1*>(wr), a.q8_1 + ar * nb, nb);
    case DType::Q4_K: return dot_q4_K_q8_K(reinterpret_cast<const BlockQ4_K*>(wr), a.q8_K + ar * nb, nb);
    case DType::Q5_K: return dot_q5_K_q8_K(reinterpret_cast<const BlockQ5_K*>(wr), a.q8_K + ar * nb, nb);
    case DType::Q6_K: return dot_q6_K_q8_K(reinterpret_cast<const BlockQ6_K*>(wr), a.q8_K + ar * nb, nb);
    default: bad_type();
  }
}

bool has_acts(DType t, const QuantActs& a) {
  switch (act_type(t)) {
    case ActType::Q8_0: return a.q8_0;
    case ActType::Q8_1: return a.q8_1;
    case ActType::Q8_K: return a.q8_K;
    case ActType::None: break;
  }
  return false;
}
}  // namespace

void softmax_inplace(float* x, int64_t n) {
  if (n <= 0) return;
  float m = x[0];
  for (int64_t i = 1; i < n; ++i) m = std::max(m, x[i]);
  double sum = 0;
  for (int64_t i = 0; i < n; ++i) {
    x[i] = std::exp(x[i] - m);
    sum += x[i];
  }
  const float inv = static_cast<float>(1.0 / sum);
  for (int64_t i = 0; i < n; ++i) x[i] *= inv;
}

void embed(const Tensor& table, std::span<const int32_t> ids, float* out) {
  const int64_t d = table.ne[0];
  for (size_t t = 0; t < ids.size(); ++t) {
    const int64_t id = ids[t];
    if (id < 0 || id >= table.rows()) throw Error("embed: token id " + std::to_string(id) + " out of range (n_vocab " + std::to_string(table.rows()) + ")");
    float* o = out + static_cast<int64_t>(t) * d;
    if (table.type == DType::F32) {
      std::copy_n(as_f32(table) + id * d, d, o);
    } else if (table.type == DType::F16) {
      const uint16_t* r = as_f16(table) + id * d;
      for (int64_t k = 0; k < d; ++k) o[k] = fp16_to_fp32(r[k]);
    } else if (is_quant(table.type)) {
      dequantize_row(table.type, static_cast<const uint8_t*>(table.data) + id * table.row_bytes(), o, d);
    } else {
      bad_type();
    }
  }
}

void quantize_rows_q8_0(const float* x, BlockQ8_0* y, int64_t n, int64_t width) {
  const int64_t nb = width / kQK;
  for (int64_t t = 0; t < n; ++t) quantize_row_q8_0(x + t * width, y + t * nb, width);
}

void matmul_rows(const Tensor& w, const float* x, float* y, int64_t n, int64_t row_begin, int64_t row_end,
                 const QuantActs* xq) {
  const int64_t in = w.ne[0], out = w.rows();
  const bool quant = is_quant(w.type);
  if (w.type != DType::F32 && w.type != DType::F16 && !quant) bad_type();
  if (quant) {
    if (n > 0 && !(xq && has_acts(w.type, *xq))) throw Error("matmul: quantized weight needs quantized activations");
    const int64_t nb = in / dtype_info(w.type).block_size, rb = w.row_bytes();
    for (int64_t t = 0; t < n; ++t)
      for (int64_t r = row_begin; r < row_end; ++r)
        y[t * out + r] = dot_row(w.type, static_cast<const uint8_t*>(w.data) + r * rb, *xq, t, nb);
    return;
  }
  for (int64_t t = 0; t < n; ++t) {
    const float* xt = x + t * in;
    for (int64_t r = row_begin; r < row_end; ++r) {
      float acc = 0;
      if (w.type == DType::F32) {
        const float* wr = as_f32(w) + r * in;
        for (int64_t k = 0; k < in; ++k) acc += xt[k] * wr[k];
      } else {
        const uint16_t* wr = as_f16(w) + r * in;
        for (int64_t k = 0; k < in; ++k) acc += xt[k] * fp16_to_fp32(wr[k]);
      }
      y[t * out + r] = acc;
    }
  }
}

void rmsnorm(const float* x, const Tensor& weight, float* y, int64_t n, float eps) {
  const int64_t d = weight.ne[0];
  const float* w = as_f32(weight);
  for (int64_t t = 0; t < n; ++t) {
    const float* xt = x + t * d;
    double ss = 0;
    for (int64_t i = 0; i < d; ++i) ss += static_cast<double>(xt[i]) * xt[i];
    const double scale = 1.0 / std::sqrt(ss / static_cast<double>(d) + static_cast<double>(eps));
    for (int64_t i = 0; i < d; ++i) y[t * d + i] = static_cast<float>(xt[i] * scale * w[i]);
  }
}

void add(float* x, const float* y, int64_t count) {
  for (int64_t i = 0; i < count; ++i) x[i] += y[i];
}

void add_bias(float* x, const Tensor& bias, int64_t n) {
  const int64_t d = bias.ne[0];
  const float* b = as_f32(bias);
  for (int64_t t = 0; t < n; ++t)
    for (int64_t i = 0; i < d; ++i) x[t * d + i] += b[i];
}

void rope(float* x, int64_t n, int64_t n_heads, int64_t head_dim, int64_t pos0, const RopeParams& p) {
  const int64_t half = head_dim / 2;
  for (int64_t t = 0; t < n; ++t) {
    const double pos = static_cast<double>(pos0 + t);
    for (int64_t h = 0; h < n_heads; ++h) {
      float* v = x + (t * n_heads + h) * head_dim;
      for (int64_t i = 0; i < half; ++i) {
        double theta = pos * std::pow(static_cast<double>(p.freq_base), -2.0 * static_cast<double>(i) / static_cast<double>(head_dim));
        if (p.freq_factors) theta /= p.freq_factors[i];
        const float c = static_cast<float>(std::cos(theta)), s = static_cast<float>(std::sin(theta));
        float& a = p.neox ? v[i] : v[2 * i];
        float& b = p.neox ? v[i + half] : v[2 * i + 1];
        const float a0 = a, b0 = b;
        a = a0 * c - b0 * s;
        b = a0 * s + b0 * c;
      }
    }
  }
}

void attention_heads(const float* q, const float* k, const float* v, float* out, int64_t n, int64_t pos0,
                     int64_t n_heads, int64_t n_kv_heads, int64_t head_dim, int64_t head_begin,
                     int64_t head_end, float* scratch) {
  const int64_t group = n_heads / n_kv_heads;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  for (int64_t h = head_begin; h < head_end; ++h) {
    const int64_t kvh = h / group;
    for (int64_t t = 0; t < n; ++t) {
      const float* qh = q + (t * n_heads + h) * head_dim;
      const int64_t len = pos0 + t + 1;
      for (int64_t j = 0; j < len; ++j) {
        const float* kj = k + (j * n_kv_heads + kvh) * head_dim;
        float dot = 0;
        for (int64_t d = 0; d < head_dim; ++d) dot += qh[d] * kj[d];
        scratch[j] = dot * scale;
      }
      softmax_inplace(scratch, len);
      float* o = out + (t * n_heads + h) * head_dim;
      std::fill_n(o, head_dim, 0.0f);
      for (int64_t j = 0; j < len; ++j) {
        const float* vj = v + (j * n_kv_heads + kvh) * head_dim;
        const float pj = scratch[j];
        for (int64_t d = 0; d < head_dim; ++d) o[d] += pj * vj[d];
      }
    }
  }
}

void silu_mul(float* gate, const float* up, int64_t count) {
  for (int64_t i = 0; i < count; ++i) {
    const float g = gate[i];
    gate[i] = g / (1.0f + std::exp(-g)) * up[i];
  }
}

}  // namespace nibble::cpu
