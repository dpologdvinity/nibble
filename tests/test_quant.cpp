#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "data/quant_golden.inc"
#include "fp16.h"
#include "quant.h"
#include "quant_golden_gen.h"

using namespace nibble;

static BlockQ4_0 rand_q4(std::mt19937& g) {
  BlockQ4_0 b;
  b.d = fp32_to_fp16(std::uniform_real_distribution<float>(0.01f, 1.f)(g));
  for (auto& q : b.qs) q = static_cast<uint8_t>(g());
  return b;
}
static BlockQ8_0 rand_q8(std::mt19937& g) {
  BlockQ8_0 b;
  b.d = fp32_to_fp16(std::uniform_real_distribution<float>(0.01f, 1.f)(g));
  for (auto& q : b.qs) q = static_cast<int8_t>(g());
  return b;
}

TEST_CASE("q4_0 dequant", "[quant]") {
  BlockQ4_0 b;
  b.d = fp32_to_fp16(0.5f);
  std::memset(b.qs, 0x88, 16);
  b.qs[0] = 0x9F;
  float out[32];
  dequantize_row(DType::Q4_0, &b, out, 32);
  REQUIRE(out[0] == 3.5f);
  REQUIRE(out[16] == 0.5f);
  REQUIRE(out[1] == 0.f);
  REQUIRE(out[31] == 0.f);
}

TEST_CASE("q8_0 dequant", "[quant]") {
  BlockQ8_0 b{};
  b.d = fp32_to_fp16(0.25f);
  b.qs[0] = -128;
  b.qs[1] = 127;
  float out[32];
  dequantize_row(DType::Q8_0, &b, out, 32);
  REQUIRE(out[0] == -32.f);
  REQUIRE(out[1] == 31.75f);
  REQUIRE(out[2] == 0.f);
}

TEST_CASE("dequant f32/f16", "[quant]") {
  float f[4] = {1, -2, 3.5f, 0};
  float o[4];
  dequantize_row(DType::F32, f, o, 4);
  REQUIRE(o[2] == 3.5f);
  uint16_t h[4];
  for (int i = 0; i < 4; ++i) h[i] = fp32_to_fp16(f[i]);
  dequantize_row(DType::F16, h, o, 4);
  REQUIRE(o[1] == -2.f);
}

TEST_CASE("quantize q8_0 exact and zero", "[quant]") {
  float x[32];
  for (int i = 0; i < 32; ++i) x[i] = (i % 2 ? -1.f : 1.f) * static_cast<float>(i + 1);
  x[5] = 127.f;
  x[6] = 0.f;
  BlockQ8_0 y;
  quantize_row_q8_0(x, &y, 32);
  REQUIRE(y.d == fp32_to_fp16(1.0f));
  for (int i = 0; i < 32; ++i) REQUIRE(y.qs[i] == static_cast<int8_t>(std::roundf(x[i])));

  float z[32] = {};
  BlockQ8_0 zq, w;
  std::mt19937 g(1);
  quantize_row_q8_0(z, &zq, 32);
  REQUIRE(zq.d == 0);
  for (auto q : zq.qs) REQUIRE(q == 0);
  w = rand_q8(g);
  REQUIRE(dot_q8_0_q8_0(&w, &zq, 1) == 0.f);
  BlockQ4_0 w4 = rand_q4(g);
  REQUIRE(dot_q4_0_q8_0(&w4, &zq, 1) == 0.f);
}

TEST_CASE("q8_0 roundtrip error", "[quant]") {
  std::mt19937 g(2);
  std::uniform_real_distribution<float> u(-3.f, 3.f);
  std::vector<float> x(32 * 16), r(x.size());
  for (auto& v : x) v = u(g);
  std::vector<BlockQ8_0> q(16);
  quantize_row_q8_0(x.data(), q.data(), static_cast<int64_t>(x.size()));
  dequantize_row(DType::Q8_0, q.data(), r.data(), static_cast<int64_t>(x.size()));
  for (int b = 0; b < 16; ++b) {
    float amax = 0;
    for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(x[b * 32 + i]));
    for (int i = 0; i < 32; ++i)
      REQUIRE(std::fabs(r[b * 32 + i] - x[b * 32 + i]) <= amax / 254.f + amax * 0x1p-10f);  // d/2 + fp16 rounding of d (up to 127*d*2^-11)
  }
}

TEST_CASE("dots match dequant reference", "[quant]") {
  std::mt19937 g(3);
  const int nb = 24;
  std::vector<BlockQ4_0> w4(nb);
  std::vector<BlockQ8_0> w8(nb), a(nb);
  for (int i = 0; i < nb; ++i) {
    w4[i] = rand_q4(g);
    w8[i] = rand_q8(g);
    a[i] = rand_q8(g);
  }
  std::vector<float> dw(32 * nb), da(32 * nb);
  dequantize_row(DType::Q8_0, a.data(), da.data(), 32 * nb);
  double ref = 0;
  dequantize_row(DType::Q4_0, w4.data(), dw.data(), 32 * nb);
  for (int i = 0; i < 32 * nb; ++i) ref += double(dw[i]) * da[i];
  REQUIRE(std::fabs(dot_q4_0_q8_0(w4.data(), a.data(), nb) - ref) <= 1e-5 * std::fabs(ref));
  ref = 0;
  dequantize_row(DType::Q8_0, w8.data(), dw.data(), 32 * nb);
  for (int i = 0; i < 32 * nb; ++i) ref += double(dw[i]) * da[i];
  REQUIRE(std::fabs(dot_q8_0_q8_0(w8.data(), a.data(), nb) - ref) <= 1e-5 * std::fabs(ref));
}

TEST_CASE("q4_0 hand-built dot", "[quant]") {
  BlockQ4_0 w;
  w.d = fp32_to_fp16(0.5f);
  std::memset(w.qs, 0x88, 16);
  w.qs[0] = 0x9F;  // elem0 = +7, elem16 = +1
  BlockQ8_0 a{};
  a.d = fp32_to_fp16(0.25f);
  a.qs[0] = 2;
  a.qs[16] = -3;
  // isum = 7*2 + 1*(-3) = 11 ; 11 * 0.5 * 0.25
  REQUIRE(dot_q4_0_q8_0(&w, &a, 1) == 11 * 0.5f * 0.25f);
}

TEST_CASE("unaligned blocks", "[quant]") {
  std::mt19937 g(4);
  BlockQ4_0 w = rand_q4(g);
  BlockQ8_0 a = rand_q8(g);
  std::vector<uint8_t> wb(sizeof w + 1), ab(sizeof a + 1);
  std::memcpy(wb.data() + 1, &w, sizeof w);
  std::memcpy(ab.data() + 1, &a, sizeof a);
  float e[32], o[32];
  dequantize_row(DType::Q4_0, &w, e, 32);
  dequantize_row(DType::Q4_0, wb.data() + 1, o, 32);
  REQUIRE(std::memcmp(e, o, sizeof e) == 0);
  dequantize_row(DType::Q8_0, &a, e, 32);
  dequantize_row(DType::Q8_0, ab.data() + 1, o, 32);
  REQUIRE(std::memcmp(e, o, sizeof e) == 0);
  auto* pw = reinterpret_cast<const BlockQ4_0*>(wb.data() + 1);
  auto* pa = reinterpret_cast<const BlockQ8_0*>(ab.data() + 1);
  REQUIRE(dot_q4_0_q8_0(pw, pa, 1) == dot_q4_0_q8_0(&w, &a, 1));
}

// ---- Golden vectors from llama.cpp b11715 reference code (tools/quant_vectors) ----
static uint32_t f32_bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}

TEST_CASE("golden: dot_q8_0_q8_0 bitwise equals ggml_vec_dot_q8_0_q8_0_generic", "[quant][golden]") {
  for (const auto& c : golden::kDotCases) {
    CAPTURE(c.seed, c.n);
    const int nb = c.n / 32;
    std::vector<BlockQ8_0> w8(static_cast<size_t>(nb)), a8(w8.size());
    golden::Rng r{c.seed};
    golden::fill_q8_0(r, reinterpret_cast<uint8_t*>(w8.data()), nb);
    golden::fill_q8_0(r, reinterpret_cast<uint8_t*>(a8.data()), nb);
    std::vector<BlockQ4_0> w4(static_cast<size_t>(nb));
    golden::fill_q4_0(r, reinterpret_cast<uint8_t*>(w4.data()), nb);
    uint64_t h = golden::fnv1a(reinterpret_cast<const uint8_t*>(w8.data()), w8.size() * sizeof(BlockQ8_0));
    h = golden::fnv1a(reinterpret_cast<const uint8_t*>(a8.data()), a8.size() * sizeof(BlockQ8_0), h);
    h = golden::fnv1a(reinterpret_cast<const uint8_t*>(w4.data()), w4.size() * sizeof(BlockQ4_0), h);
    REQUIRE(h == c.input_hash);
    REQUIRE(f32_bits(dot_q8_0_q8_0(w8.data(), a8.data(), nb)) == c.q8q8_bits);
  }
}
