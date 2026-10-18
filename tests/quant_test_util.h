#pragma once
// Test-only quantizers producing raw Q4_0/Q8_0 row bytes from F32.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "fp16.h"
#include "quant.h"

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
