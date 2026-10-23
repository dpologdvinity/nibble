#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <random>
#include <vector>

#include "data/quant_golden.inc"
#include "fp16.h"
#include "quant.h"
#include "quant_golden_gen.h"
#include "quant_test_util.h"

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

TEST_CASE("golden: quantize_row_q8_0 matches quantize_row_q8_0_ref bytes", "[quant][golden]") {
  for (const auto& c : golden::kQuantCases) {
    CAPTURE(c.name);
    BlockQ8_0 y;
    quantize_row_q8_0(c.x, &y, 32);
    REQUIRE(y.d == c.q8_d);
    for (int j = 0; j < 32; ++j) {
      CAPTURE(j);
      REQUIRE(int{y.qs[j]} == int{c.q8_qs[j]});
    }
  }
}

TEST_CASE("golden: test Q4_0 quantizer matches quantize_row_q4_0_ref bytes", "[quant][golden]") {
  for (const auto& c : golden::kQuantCases) {
    CAPTURE(c.name);
    BlockQ4_0 y;
    test_quantize_row_q4_0(c.x, &y, 32);
    REQUIRE(y.d == c.q4_d);
    for (int j = 0; j < 16; ++j) {
      CAPTURE(j);
      REQUIRE(int{y.qs[j]} == int{c.q4_qs[j]});
    }
  }
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

TEST_CASE("golden: dot_q4_0_q8_0 bitwise equals ggml_vec_dot_q4_0_q8_0_generic", "[quant][golden]") {
  for (const auto& c : golden::kDotCases) {
    CAPTURE(c.seed, c.n);
    const int nb = c.n / 32;
    std::vector<BlockQ8_0> w8(static_cast<size_t>(nb)), a8(w8.size());
    std::vector<BlockQ4_0> w4(static_cast<size_t>(nb));
    golden::Rng r{c.seed};
    golden::fill_q8_0(r, reinterpret_cast<uint8_t*>(w8.data()), nb);
    golden::fill_q8_0(r, reinterpret_cast<uint8_t*>(a8.data()), nb);
    golden::fill_q4_0(r, reinterpret_cast<uint8_t*>(w4.data()), nb);
    REQUIRE(f32_bits(dot_q4_0_q8_0(w4.data(), a8.data(), nb)) == c.q4q8_bits);
  }
}

TEST_CASE("golden: dequantize_row Q4_0/Q8_0 bitwise equals dequantize_row_q*_0", "[quant][golden]") {
  constexpr int nb = golden::kDequantBlocks;
  std::vector<uint8_t> q8(nb * sizeof(BlockQ8_0)), q4(nb * sizeof(BlockQ4_0));
  golden::Rng r{golden::kDequantSeed};
  golden::fill_q8_0(r, q8.data(), nb);
  golden::fill_q4_0(r, q4.data(), nb);
  std::vector<float> o(nb * 32);
  dequantize_row(DType::Q8_0, q8.data(), o.data(), nb * 32);
  for (int i = 0; i < nb * 32; ++i) {
    CAPTURE(i);
    REQUIRE(f32_bits(o[static_cast<size_t>(i)]) == golden::kDequantQ8_0[i]);
  }
  dequantize_row(DType::Q4_0, q4.data(), o.data(), nb * 32);
  for (int i = 0; i < nb * 32; ++i) {
    CAPTURE(i);
    REQUIRE(f32_bits(o[static_cast<size_t>(i)]) == golden::kDequantQ4_0[i]);
  }
}

TEST_CASE("q5_0 dequant", "[quant]") {
  BlockQ5_0 b{};
  b.d = fp32_to_fp16(1.0f);
  b.qh[0] = 0x01;  // bit 0 set, bit 16 clear
  b.qs[0] = 0x3F;
  float o[32];
  dequantize_row(DType::Q5_0, &b, o, 32);
  REQUIRE(o[0] == 15.f);
  REQUIRE(o[16] == -13.f);
  REQUIRE(o[1] == -16.f);
}

TEST_CASE("q5_1 dequant", "[quant]") {
  BlockQ5_1 b{};
  b.d = fp32_to_fp16(0.5f);
  b.m = fp32_to_fp16(-1.0f);
  b.qh[2] = 0x01;  // bit 16 set
  b.qs[0] = 0x10;
  float o[32];
  dequantize_row(DType::Q5_1, &b, o, 32);
  REQUIRE(o[0] == -1.f);
  REQUIRE(o[16] == 7.5f);
  REQUIRE(o[17] == -1.f);
}

TEST_CASE("quantize q8_1 exact and zero", "[quant]") {
  float x[32];
  for (auto& v : x) v = 127.f;
  BlockQ8_1 y;
  quantize_row_q8_1(x, &y, 32);
  REQUIRE(y.d == fp32_to_fp16(1.0f));
  REQUIRE(y.s == fp32_to_fp16(4064.f));
  for (auto q : y.qs) REQUIRE(q == 127);

  float z[32] = {};
  BlockQ8_1 zq;
  quantize_row_q8_1(z, &zq, 32);
  REQUIRE(zq.d == 0);
  REQUIRE(zq.s == 0);
  for (auto q : zq.qs) REQUIRE(q == 0);
  BlockQ8_0 z8;
  quantize_row_q8_0(z, &z8, 32);
  golden::Rng r{9};
  BlockQ5_0 w50;
  BlockQ5_1 w51;
  golden::fill_q5_0(r, reinterpret_cast<uint8_t*>(&w50), 1);
  golden::fill_q5_1(r, reinterpret_cast<uint8_t*>(&w51), 1);
  REQUIRE(dot_q5_0_q8_0(&w50, &z8, 1) == 0.f);
  REQUIRE(dot_q5_1_q8_1(&w51, &zq, 1) == 0.f);
}

TEST_CASE("golden fill_q8_1 s is within 1 ulp of fp16(d * sum)", "[quant]") {
  golden::Rng r{11};
  std::vector<BlockQ8_1> a(64);
  golden::fill_q8_1(r, reinterpret_cast<uint8_t*>(a.data()), 64);
  for (const auto& b : a) {
    int sum = 0;
    for (auto q : b.qs) {
      REQUIRE(q >= -127);
      sum += q;
    }
    const uint16_t ref = fp32_to_fp16(static_cast<float>(sum) * fp16_to_fp32(b.d));  // exact product
    REQUIRE((b.s & 0x8000) == (ref & 0x8000));
    REQUIRE(std::abs((b.s & 0x7FFF) - (ref & 0x7FFF)) <= 1);
  }
}

TEST_CASE("golden: quantize_row_q8_1 matches quantize_row_q8_1_ref bytes", "[quant][golden]") {
  STATIC_REQUIRE(std::size(golden::kQ8_1Cases) == std::size(golden::kQuantCases));
  for (size_t i = 0; i < std::size(golden::kQuantCases); ++i) {
    const auto& c = golden::kQuantCases[i];
    CAPTURE(c.name);
    BlockQ8_1 y;
    BlockQ8_0 y8;
    quantize_row_q8_1(c.x, &y, 32);
    quantize_row_q8_0(c.x, &y8, 32);
    REQUIRE(y.d == golden::kQ8_1Cases[i].d);
    REQUIRE(y.d == y8.d);
    REQUIRE(y.s == golden::kQ8_1Cases[i].s);
    for (int j = 0; j < 32; ++j) {
      CAPTURE(j);
      REQUIRE(int{y.qs[j]} == int{y8.qs[j]});
    }
  }
}

TEST_CASE("golden: dot_q5_0_q8_0 and dot_q5_1_q8_1 bitwise equal ggml generic", "[quant][golden]") {
  for (const auto& c : golden::kDot5Cases) {
    CAPTURE(c.seed, c.n);
    const int nb = c.n / 32;
    const auto nbs = static_cast<size_t>(nb);
    std::vector<BlockQ5_0> w50(nbs);
    std::vector<BlockQ8_0> a80(nbs);
    std::vector<BlockQ5_1> w51(nbs);
    std::vector<BlockQ8_1> a81(nbs);
    golden::Rng r{c.seed};
    golden::fill_q5_0(r, reinterpret_cast<uint8_t*>(w50.data()), nb);
    golden::fill_q8_0(r, reinterpret_cast<uint8_t*>(a80.data()), nb);
    golden::fill_q5_1(r, reinterpret_cast<uint8_t*>(w51.data()), nb);
    golden::fill_q8_1(r, reinterpret_cast<uint8_t*>(a81.data()), nb);
    uint64_t h = golden::fnv1a(reinterpret_cast<const uint8_t*>(w50.data()), nbs * sizeof(BlockQ5_0));
    h = golden::fnv1a(reinterpret_cast<const uint8_t*>(a80.data()), nbs * sizeof(BlockQ8_0), h);
    h = golden::fnv1a(reinterpret_cast<const uint8_t*>(w51.data()), nbs * sizeof(BlockQ5_1), h);
    h = golden::fnv1a(reinterpret_cast<const uint8_t*>(a81.data()), nbs * sizeof(BlockQ8_1), h);
    REQUIRE(h == c.input_hash);
    REQUIRE(f32_bits(dot_q5_0_q8_0(w50.data(), a80.data(), nb)) == c.q5_0_bits);
    REQUIRE(f32_bits(dot_q5_1_q8_1(w51.data(), a81.data(), nb)) == c.q5_1_bits);
  }
}

TEST_CASE("golden: dequantize_row Q5_0/Q5_1 bitwise equals dequantize_row_q5_*", "[quant][golden]") {
  constexpr int nb = golden::kDequantBlocks;
  std::vector<uint8_t> q50(nb * sizeof(BlockQ5_0)), q51(nb * sizeof(BlockQ5_1));
  golden::Rng r{golden::kDequantSeed};
  golden::fill_q5_0(r, q50.data(), nb);
  golden::fill_q5_1(r, q51.data(), nb);
  std::vector<float> o(nb * 32);
  dequantize_row(DType::Q5_0, q50.data(), o.data(), nb * 32);
  for (int i = 0; i < nb * 32; ++i) {
    CAPTURE(i);
    REQUIRE(f32_bits(o[static_cast<size_t>(i)]) == golden::kDequantQ5_0[i]);
  }
  dequantize_row(DType::Q5_1, q51.data(), o.data(), nb * 32);
  for (int i = 0; i < nb * 32; ++i) {
    CAPTURE(i);
    REQUIRE(f32_bits(o[static_cast<size_t>(i)]) == golden::kDequantQ5_1[i]);
  }
}

#include "backend/cpu/simd/simd.h"
#ifdef NIBBLE_HAVE_AVX2
// Decision 3: the AVX2 quantizer must reproduce quantize_row_q8_0_ref bytes, including the tie cases
// that _mm256_round_ps(nearest) would round to even.
TEST_CASE("golden: simd quantize_row_q8_0 matches quantize_row_q8_0_ref bytes", "[quant][golden][simd]") {
  for (const auto& c : golden::kQuantCases) {
    CAPTURE(c.name);
    BlockQ8_0 y;
    simd::quantize_row_q8_0(c.x, &y, 32);
    REQUIRE(y.d == c.q8_d);
    for (int j = 0; j < 32; ++j) {
      CAPTURE(j);
      REQUIRE(int{y.qs[j]} == int{c.q8_qs[j]});
    }
  }
}

// Q8_0 x Q8_0 is not checked here: the golden activations are random bytes that include -128, which the
// AVX2 sign trick does not support (quantize_row_q8_0 never emits it); test_simd.cpp covers it against
// the golden-pinned scalar dot instead.
TEST_CASE("golden: simd dot_q4_0_q8_0 bitwise equals ggml generic dot", "[quant][golden][simd]") {
  for (const auto& c : golden::kDotCases) {
    CAPTURE(c.seed, c.n);
    const int nb = c.n / 32;
    std::vector<BlockQ8_0> w8(static_cast<size_t>(nb)), a8(w8.size());
    std::vector<BlockQ4_0> w4(static_cast<size_t>(nb));
    golden::Rng r{c.seed};
    golden::fill_q8_0(r, reinterpret_cast<uint8_t*>(w8.data()), nb);
    golden::fill_q8_0(r, reinterpret_cast<uint8_t*>(a8.data()), nb);
    golden::fill_q4_0(r, reinterpret_cast<uint8_t*>(w4.data()), nb);
    REQUIRE(f32_bits(simd::dot_q4_0_q8_0(w4.data(), a8.data(), nb)) == c.q4q8_bits);
  }
}
#endif
