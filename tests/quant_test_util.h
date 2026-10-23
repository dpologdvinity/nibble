#pragma once
// Test-only quantizers producing raw Q4_0/Q8_0 row bytes from F32.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "fp16.h"
#include "quant.h"
#include "quant_golden_gen.h"

// Mirrors llama.cpp quantize_row_q4_0_ref (ggml-quants.c).
inline void test_quantize_row_q4_0(const float* x, nibble::BlockQ4_0* y, int64_t k) {
  const int64_t nb = k / 32;
  for (int64_t i = 0; i < nb; ++i) {
    float amax = 0.0f, max = 0.0f;
    for (int j = 0; j < 32; ++j) {
      const float v = x[i * 32 + j];
      if (amax < std::fabs(v)) {
        amax = std::fabs(v);
        max = v;
      }
    }
    const float d = max / -8;
    const float id = d ? 1.0f / d : 0.0f;
    y[i].d = nibble::fp32_to_fp16(d);
    for (int j = 0; j < 16; ++j) {
      const float x0 = x[i * 32 + j] * id, x1 = x[i * 32 + 16 + j] * id;
      const uint8_t xi0 = static_cast<uint8_t>(std::min(15, static_cast<int>(static_cast<int8_t>(x0 + 8.5f))));
      const uint8_t xi1 = static_cast<uint8_t>(std::min(15, static_cast<int>(static_cast<int8_t>(x1 + 8.5f))));
      y[i].qs[j] = static_cast<uint8_t>(xi0 | (xi1 << 4));
    }
  }
}

// Quantizes `rows` rows of width `ne0` (row-major) into raw bytes of type t (Q4_0 or Q8_0).
inline std::vector<uint8_t> test_quantize_rows(nibble::DType t, const std::vector<float>& v, int64_t ne0) {
  const size_t rows = v.size() / static_cast<size_t>(ne0), nb = static_cast<size_t>(ne0 / 32);
  const size_t bs = t == nibble::DType::Q4_0 ? sizeof(nibble::BlockQ4_0) : sizeof(nibble::BlockQ8_0);
  std::vector<uint8_t> out(rows * nb * bs);
  for (size_t r = 0; r < rows; ++r) {
    uint8_t* dst = out.data() + r * nb * bs;
    if (t == nibble::DType::Q4_0) {
      std::vector<nibble::BlockQ4_0> b(nb);
      test_quantize_row_q4_0(v.data() + r * static_cast<size_t>(ne0), b.data(), ne0);
      std::memcpy(dst, b.data(), nb * bs);
    } else {
      std::vector<nibble::BlockQ8_0> b(nb);
      nibble::quantize_row_q8_0(v.data() + r * static_cast<size_t>(ne0), b.data(), ne0);
      std::memcpy(dst, b.data(), nb * bs);
    }
  }
  return out;
}

// Raw weight bytes for `rows` rows of width ne0 of any quantized type, from the golden generators (random
// quants and finite scales, deterministic in seed).
inline std::vector<uint8_t> test_weight_rows(nibble::DType t, uint64_t seed, int64_t rows, int64_t ne0) {
  using nibble::DType;
  const auto& info = nibble::dtype_info(t);
  const int64_t nb = rows * ne0 / info.block_size;
  std::vector<uint8_t> out(static_cast<size_t>(nb * info.block_bytes));
  golden::Rng r{seed};
  switch (t) {
    case DType::Q4_0: golden::fill_q4_0(r, out.data(), nb); break;
    case DType::Q5_0: golden::fill_q5_0(r, out.data(), nb); break;
    case DType::Q5_1: golden::fill_q5_1(r, out.data(), nb); break;
    case DType::Q8_0: golden::fill_q8_0(r, out.data(), nb); break;
    case DType::Q4_K: golden::fill_q4_K(r, out.data(), nb); break;
    case DType::Q5_K: golden::fill_q5_K(r, out.data(), nb); break;
    case DType::Q6_K: golden::fill_q6_K(r, out.data(), nb); break;
    default: break;
  }
  return out;
}

// n rows of x quantized to the activation type that weights of type t are multiplied against.
struct TestActs {
  std::vector<nibble::BlockQ8_0> q8_0;
  std::vector<nibble::BlockQ8_1> q8_1;
  std::vector<nibble::BlockQ8_K> q8_K;
  nibble::QuantActs view() const { return {q8_0.data(), q8_1.data(), q8_K.data()}; }
};
inline TestActs test_quantize_acts(nibble::DType t, const float* x, int64_t n, int64_t ne0) {
  TestActs a;
  switch (nibble::act_type(t)) {
    case nibble::ActType::Q8_0:
      a.q8_0.resize(static_cast<size_t>(n * ne0 / 32));
      for (int64_t i = 0; i < n; ++i) nibble::quantize_row_q8_0(x + i * ne0, a.q8_0.data() + i * ne0 / 32, ne0);
      break;
    case nibble::ActType::Q8_1:
      a.q8_1.resize(static_cast<size_t>(n * ne0 / 32));
      for (int64_t i = 0; i < n; ++i) nibble::quantize_row_q8_1(x + i * ne0, a.q8_1.data() + i * ne0 / 32, ne0);
      break;
    case nibble::ActType::Q8_K:
      a.q8_K.resize(static_cast<size_t>(n * ne0 / 256));
      for (int64_t i = 0; i < n; ++i) nibble::quantize_row_q8_K(x + i * ne0, a.q8_K.data() + i * ne0 / 256, ne0);
      break;
    case nibble::ActType::None: break;
  }
  return a;
}
