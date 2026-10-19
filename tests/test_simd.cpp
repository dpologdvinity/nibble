#include <algorithm>
#include <cmath>
#include <cstring>
#include <type_traits>

#include "backend/cpu/simd/simd.h"
#include "fp16.h"
#include "simd_test_util.h"

using namespace nibble;

TEST_CASE("kernel_set names the active macros", "[simd]") {
#if defined(NIBBLE_HAVE_AVXVNNI)
  CHECK(std::strcmp(simd::kernel_set(), "avx2+fma+f16c+avxvnni") == 0);
#elif defined(NIBBLE_HAVE_AVX2)
  CHECK(std::strcmp(simd::kernel_set(), "avx2+fma+f16c") == 0);
#else
  CHECK(std::strcmp(simd::kernel_set(), "scalar") == 0);
#endif
}

TEST_CASE("compare_kernels detects mismatches and accepts equal kernels", "[simd]") {
  auto ident = [](const float* in, float* out, size_t n) { for (size_t i = 0; i < n; ++i) out[i] = in[i]; };
  test::compare_kernels(ident, ident, {0, 1, 7, 100}, {1, 2}, 0.f, 0.f);
  auto off = [](const float* in, float* out, size_t n) { for (size_t i = 0; i < n; ++i) out[i] = in[i] + 1e-3f; };
  test::compare_kernels(ident, off, {16}, {1}, 0.f, 1e-2f);  // within atol passes
}

#ifdef NIBBLE_HAVE_AVX2
TEST_CASE("hsum_ps and hsum_epi32 are exact on integers", "[simd]") {
  alignas(32) float f[8];
  alignas(32) int32_t v[8];
  for (int i = 0; i < 8; ++i) { f[i] = float(i * 1000 + 1); v[i] = -i * 100000 + 7; }
  float ef = 0; int32_t ev = 0;
  for (int i = 0; i < 8; ++i) { ef += f[i]; ev += v[i]; }
  CHECK(simd::hsum_ps(_mm256_load_ps(f)) == ef);
  CHECK(simd::hsum_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(v))) == ev);
}

TEST_CASE("load_f16x8 equals fp16_to_fp32 for all 65536 inputs", "[simd]") {
  for (uint32_t base = 0; base < 65536; base += 8) {
    uint16_t h[8];
    for (int i = 0; i < 8; ++i) h[i] = uint16_t(base + i);
    alignas(32) float out[8];
    _mm256_store_ps(out, simd::load_f16x8(h));
    for (int i = 0; i < 8; ++i) {
      float ref = fp16_to_fp32(h[i]);
      if (std::isnan(ref)) CHECK(std::isnan(out[i]));
      else CHECK(std::memcmp(&ref, &out[i], 4) == 0);
    }
  }
}
#endif

#ifdef NIBBLE_HAVE_AVX2
#include <vector>

#include "backend/cpu/ref_ops.h"
#include "tensor.h"

namespace {
// Compares simd::matmul_rows_f with the scalar oracle over [row_begin,row_end) of a rows x in weight.
// compare_kernels' buffers hold n = nt*max(in,rows) floats: x is the first nt*in, y the first nt*rows.
// x and W are deliberately placed at 4-byte (F32) / 2-byte (F16) offsets from allocation start.
void check_matmul(DType type, int64_t in, int64_t rows, int64_t nt, int64_t rb, int64_t re, float scale) {
  std::mt19937 rng(uint32_t(in * 131 + rows));
  std::uniform_real_distribution<float> d(-1.f, 1.f);
  std::vector<float> wf(size_t(rows * in) + 1);
  std::vector<uint16_t> wh(size_t(rows * in) + 1);
  for (size_t i = 0; i < wf.size(); ++i) { wf[i] = d(rng) * scale; wh[i] = fp32_to_fp16(wf[i]); }
  Tensor w;
  w.type = type;
  w.n_dims = 2;
  w.ne = {in, rows, 1, 1};
  w.data = type == DType::F32 ? static_cast<const void*>(wf.data() + 1) : static_cast<const void*>(wh.data() + 1);
  const size_t n = size_t(nt * std::max(in, rows));
  std::vector<float> xs(n + 1), y(size_t(nt * rows));
  auto run = [&](bool simd_path) {
    return [&, simd_path](const float* x, float* out, size_t) {
      std::copy_n(x, size_t(nt * in), xs.begin() + 1);
      std::fill(out, out + n, -7.f);
      std::fill(y.begin(), y.end(), -7.f);
      if (simd_path) simd::matmul_rows_f(w, xs.data() + 1, y.data(), nt, rb, re);
      else cpu::matmul_rows(w, xs.data() + 1, y.data(), nt, rb, re);
      std::copy(y.begin(), y.end(), out);
    };
  };
  test::compare_kernels(run(false), run(true), {n}, {1, 2}, 1e-5f,
                       1e-6f * std::sqrt(float(in)) * scale);  // atol: the oracle's own sequential-sum error under cancellation
}
}  // namespace

TEST_CASE("simd matmul_rows_f matches scalar for F32 and F16", "[simd]") {
  for (DType t : {DType::F32, DType::F16})
    for (int64_t in : {8, 13, 896, 4864})
      for (int64_t rows : {1, 3, 4, 37})
        for (int64_t nt : {1, 5}) {
          INFO("type=" << int(t) << " in=" << in << " rows=" << rows << " n=" << nt);
          check_matmul(t, in, rows, nt, 0, rows, 1.f);
        }
}

TEST_CASE("simd matmul_rows_f handles sub-ranges and extreme magnitudes", "[simd]") {
  for (DType t : {DType::F32, DType::F16}) {
    check_matmul(t, 70, 37, 2, 5, 22, 1.f);   // row range not 4-aligned
    check_matmul(t, 70, 37, 2, 36, 37, 1.f);  // single trailing row
    check_matmul(t, 70, 37, 1, 9, 9, 1.f);    // empty range
    check_matmul(t, 301, 9, 3, 0, 9, 300.f);  // large (F16 stays finite)
    check_matmul(t, 301, 9, 3, 0, 9, 1e-3f);  // small
  }
}

TEST_CASE("embed_row_f16 equals scalar conversion", "[simd]") {
  std::vector<uint16_t> h(41);
  for (size_t i = 0; i < h.size(); ++i) h[i] = uint16_t(i * 1597 + 3);
  for (int64_t d : {0, 1, 7, 8, 9, 40}) {
    std::vector<float> o(41, -1.f);
    simd::embed_row_f16(h.data() + 1, o.data(), d);
    for (int64_t k = 0; k < 40; ++k) {
      float ref = k < d ? fp16_to_fp32(h[size_t(k + 1)]) : -1.f;
      if (std::isnan(ref)) CHECK(std::isnan(o[size_t(k)]));
      else CHECK(o[size_t(k)] == ref);
    }
  }
}
#endif

#ifdef NIBBLE_HAVE_AVX2
#include "quant.h"

namespace {
uint32_t bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

// Quantizes x with both quantizers into byte buffers at a 1-byte offset (unaligned blocks) and requires equal bytes.
void check_quantize_same(const std::vector<float>& x) {
  const size_t nb = x.size() / 32, bytes = nb * sizeof(BlockQ8_0);
  std::vector<uint8_t> a(bytes + 1, 0xAA), b(bytes + 1, 0x55);
  quantize_row_q8_0(x.data(), reinterpret_cast<BlockQ8_0*>(a.data() + 1), int64_t(x.size()));
  simd::quantize_row_q8_0(x.data(), reinterpret_cast<BlockQ8_0*>(b.data() + 1), int64_t(x.size()));
  for (size_t i = 0; i < bytes; ++i)
    if (a[i + 1] != b[i + 1]) {
      const size_t blk = i / sizeof(BlockQ8_0), j = i % sizeof(BlockQ8_0);
      FAIL("byte " << i << " (block " << blk << ", offset " << j << ") scalar=" << int(a[i + 1]) << " simd=" << int(b[i + 1])
                   << (j >= 2 ? " x=" : " d") << (j >= 2 ? x[blk * 32 + j - 2] : 0.f));
    }
}

template <class B>
B rand_block(std::mt19937& g, bool unit_scale) {
  B b;
  b.d = fp32_to_fp16(unit_scale ? 1.f : std::uniform_real_distribution<float>(0.01f, 1.f)(g));
  for (auto& q : b.qs) q = static_cast<std::remove_reference_t<decltype(q)>>(g());
  return b;
}
BlockQ8_0 rand_act(std::mt19937& g, bool unit_scale) {  // activations: never -128, like quantize_row_q8_0 for normal id
  BlockQ8_0 b = rand_block<BlockQ8_0>(g, unit_scale);
  for (auto& q : b.qs) if (q == -128) q = -127;
  return b;
}
// Copies blocks into a byte buffer at a 1-byte offset so every block is misaligned.
template <class B>
std::vector<uint8_t> misalign(const std::vector<B>& v) {
  std::vector<uint8_t> o(v.size() * sizeof(B) + 1);
  if (!v.empty()) std::memcpy(o.data() + 1, v.data(), v.size() * sizeof(B));
  return o;
}
}  // namespace

TEST_CASE("simd quantize_row_q8_0 is byte-identical to scalar on random rows", "[simd][quant]") {
  std::mt19937 g(11);
  for (int64_t n : {32, 64, 96, 896, 4864})
    for (float scale : {1.f, 1e-3f, 300.f, 1e-30f, 3e4f})
      for (int rep = 0; rep < 8; ++rep) {
        CAPTURE(n, scale, rep);
        std::uniform_real_distribution<float> d(-1.f, 1.f);
        std::vector<float> x(static_cast<size_t>(n));
        for (float& v : x) v = d(g) * scale;
        if (n >= 64) std::fill(x.begin() + 32, x.begin() + 64, rep % 2 ? -0.f : 0.f);  // zero block (d = 0, id = 0)
        if (rep == 3) x[5] = -x[5] * 1e6f;  // one dominant element: the rest quantize to small values and 0
        check_quantize_same(x);
      }
  check_quantize_same({});  // n = 0
  std::vector<float> bad(31);
  CHECK_THROWS(simd::quantize_row_q8_0(bad.data(), nullptr, 31));
}

TEST_CASE("simd quantize_row_q8_0 rounds half away from zero like roundf", "[simd][quant]") {
  // amax = 127 gives d = 1 and id = 1, so x*id == x and the qs are roundf(x) of the inputs themselves.
  std::vector<float> vals = {0.49999997f, 0.5f, 0.50000006f, 1.5f, 2.5f, 126.5f, 0x1p-149f, 0x1p-126f, 0.f, 1.f};
  for (int k = 0; k < 127; ++k)
    for (float c : {float(k), float(k) + 0.5f}) {
      uint32_t u = bits(c);
      for (int off = -64; off <= 64; ++off) {
        if (c == 0.f && off < 0) continue;
        uint32_t v = u + uint32_t(off);
        float f;
        std::memcpy(&f, &v, 4);
        if (f <= 127.f) vals.push_back(f);
      }
    }
  std::vector<float> x;
  for (float v : vals)
    for (float s : {1.f, -1.f}) {
      if (x.size() % 32 == 31) x.push_back(x.size() % 64 == 31 ? 127.f : -127.f);
      x.push_back(s * v);
    }
  while (x.size() % 32) x.push_back(x.size() % 32 == 31 ? 127.f : 0.f);
  for (size_t b = 0; b < x.size(); b += 32) REQUIRE(std::fabs(x[b + 31]) == 127.f);  // every block has d = 1
  check_quantize_same(x);
}

TEST_CASE("mul_sum_i8 maddubs and dispatch variants agree with integer dot", "[simd][quant]") {
  std::mt19937 g(5);
  for (int rep = 0; rep < 2000; ++rep) {
    alignas(32) int8_t x[32], y[32];
    for (int i = 0; i < 32; ++i) {
      x[i] = int8_t(g());
      y[i] = int8_t(g());
      if (rep % 4 == 0) x[i] = i % 2 ? -128 : 127;  // weight extremes with y in [-127,127]
      if (rep % 4 == 1) y[i] = i % 3 ? 127 : -127;
      if (y[i] == -128 && x[i] < 0) y[i] = -127;    // outside the documented exact range
    }
    const __m256i vx = _mm256_load_si256(reinterpret_cast<const __m256i*>(x));
    const __m256i vy = _mm256_load_si256(reinterpret_cast<const __m256i*>(y));
    alignas(32) int32_t a[8], b[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(a), simd::mul_sum_i8(vx, vy));
    _mm256_store_si256(reinterpret_cast<__m256i*>(b), simd::mul_sum_i8_maddubs(vx, vy));
    for (int k = 0; k < 8; ++k) {
      int32_t want = 0;
      for (int i = 4 * k; i < 4 * k + 4; ++i) want += int32_t(x[i]) * y[i];
      REQUIRE(a[k] == want);
      REQUIRE(b[k] == want);
    }
  }
}

TEST_CASE("simd q4_0/q8_0 dots equal scalar bitwise", "[simd][quant]") {
  std::mt19937 g(7);
  for (int64_t nb : {0, 1, 2, 3, 7, 8, 9, 28, 152})
    for (int rep = 0; rep < 20; ++rep) {
      CAPTURE(nb, rep);
      std::vector<BlockQ4_0> w4;
      std::vector<BlockQ8_0> w8, a;
      for (int64_t b = 0; b < nb; ++b) {
        w4.push_back(rand_block<BlockQ4_0>(g, false));
        w8.push_back(rand_block<BlockQ8_0>(g, false));
        a.push_back(rand_act(g, false));
        if (rep == 1) std::memset(a.back().qs, 0, 32);  // zero activation blocks
        if (rep == 2) a.back().d = 0;                     // zero scale
      }
      auto w4b = misalign(w4), w8b = misalign(w8), ab = misalign(a);
      auto* pw4 = reinterpret_cast<const BlockQ4_0*>(w4b.data() + 1);
      auto* pw8 = reinterpret_cast<const BlockQ8_0*>(w8b.data() + 1);
      auto* pa = reinterpret_cast<const BlockQ8_0*>(ab.data() + 1);
      REQUIRE(bits(simd::dot_q4_0_q8_0(pw4, pa, nb)) == bits(dot_q4_0_q8_0(pw4, pa, nb)));
      REQUIRE(bits(simd::dot_q8_0_q8_0(pw8, pa, nb)) == bits(dot_q8_0_q8_0(pw8, pa, nb)));
    }
}

TEST_CASE("tiny-amax activation blocks contribute exactly 0 to scalar and simd dots", "[simd][quant]") {
  // amax 1e-40: id = 1/d overflows, qs are unspecified (scalar UB), but fp16(d) == 0 so the dot is 0.
  std::mt19937 g(5);
  const int64_t nb = 3;
  std::vector<float> x(nb * 32);
  for (auto& v : x) v = std::uniform_real_distribution<float>(-1e-40f, 1e-40f)(g);
  x[0] = 1e-40f;
  std::vector<BlockQ8_0> as(nb), av(nb);
  simd::quantize_row_q8_0(x.data(), av.data(), nb * 32);
  for (auto& b : av) REQUIRE(fp16_to_fp32(b.d) == 0.f);
  as = av;  // scalar quantizer is UB here; reuse the simd blocks (same d) for the scalar dot
  std::vector<BlockQ4_0> w4;
  std::vector<BlockQ8_0> w8;
  for (int64_t b = 0; b < nb; ++b) {
    w4.push_back(rand_block<BlockQ4_0>(g, true));
    w8.push_back(rand_block<BlockQ8_0>(g, true));
  }
  REQUIRE(simd::dot_q8_0_q8_0(w8.data(), av.data(), nb) == 0.f);
  REQUIRE(dot_q8_0_q8_0(w8.data(), as.data(), nb) == 0.f);
  REQUIRE(simd::dot_q4_0_q8_0(w4.data(), av.data(), nb) == 0.f);
  REQUIRE(dot_q4_0_q8_0(w4.data(), as.data(), nb) == 0.f);
}

TEST_CASE("simd dots give exact integer sums on unit-scale extreme blocks", "[simd][quant]") {
  // d = 1 for every block: the dot is the exact integer sum (|sum| < 2^24), so any lost product shows.
  std::mt19937 g(9);
  for (int pattern = 0; pattern < 6; ++pattern)
    for (int64_t nb : {1, 5, 8, 9}) {
      CAPTURE(pattern, nb);
      std::vector<BlockQ4_0> w4(static_cast<size_t>(nb));
      std::vector<BlockQ8_0> w8(static_cast<size_t>(nb)), a(w8.size());
      int64_t want4 = 0, want8 = 0;
      for (int64_t b = 0; b < nb; ++b) {
        w4[size_t(b)] = rand_block<BlockQ4_0>(g, true);
        w8[size_t(b)] = rand_block<BlockQ8_0>(g, true);
        a[size_t(b)] = rand_act(g, true);
        for (int j = 0; j < 32; ++j) {
          int8_t& q = a[size_t(b)].qs[j];
          int8_t& wq = w8[size_t(b)].qs[j];
          if (pattern == 0) { q = 127; wq = 127; }
          if (pattern == 1) { q = -127; wq = -128; }
          if (pattern == 2) { q = j % 2 ? 127 : -127; wq = j % 2 ? -128 : 127; }
          if (pattern == 3 && j < 16) w4[size_t(b)].qs[j] = 0x00;   // nibbles 0 -> -8
          if (pattern == 4 && j < 16) w4[size_t(b)].qs[j] = 0xFF;   // nibbles 15 -> +7
          if (pattern == 5 && j < 16) w4[size_t(b)].qs[j] = 0xF0;   // low 0, high 15
          if (pattern >= 3) q = j % 2 ? 127 : -127;
        }
        for (int j = 0; j < 16; ++j) {
          const uint8_t v = w4[size_t(b)].qs[j];
          want4 += int64_t((v & 0x0F) - 8) * a[size_t(b)].qs[j] + int64_t((v >> 4) - 8) * a[size_t(b)].qs[j + 16];
        }
        for (int j = 0; j < 32; ++j) want8 += int64_t(w8[size_t(b)].qs[j]) * a[size_t(b)].qs[j];
      }
      REQUIRE(simd::dot_q4_0_q8_0(w4.data(), a.data(), nb) == float(want4));
      REQUIRE(simd::dot_q8_0_q8_0(w8.data(), a.data(), nb) == float(want8));
    }
}

namespace {
// simd::matmul_rows_q vs cpu::matmul_rows over [rb,re) of a rows x in quantized weight, n activation rows.
void check_matmul_q(DType type, int64_t in, int64_t rows, int64_t n, int64_t rb, int64_t re) {
  std::mt19937 g(uint32_t(in * 7 + rows * 3 + n));
  const int64_t nb = in / 32;
  std::vector<uint8_t> wbuf(size_t(rows * nb) * (type == DType::Q4_0 ? sizeof(BlockQ4_0) : sizeof(BlockQ8_0)) + 1);
  for (size_t i = 1; i < wbuf.size(); ++i) wbuf[i] = uint8_t(g());
  const size_t bs = type == DType::Q4_0 ? sizeof(BlockQ4_0) : sizeof(BlockQ8_0);
  for (int64_t b = 0; b < rows * nb; ++b) {  // sane scales (random bytes could be inf/NaN halves)
    const uint16_t d = fp32_to_fp16(std::uniform_real_distribution<float>(0.001f, 0.1f)(g));
    std::memcpy(wbuf.data() + 1 + size_t(b) * bs, &d, 2);
  }
  Tensor w;
  w.type = type;
  w.n_dims = 2;
  w.ne = {in, rows, 1, 1};
  w.data = wbuf.data() + 1;  // misaligned weight rows
  std::uniform_real_distribution<float> d(-1.f, 1.f);
  std::vector<float> x(size_t(n * in));
  for (float& v : x) v = d(g);
  std::vector<BlockQ8_0> xq(size_t(n * nb));
  cpu::quantize_rows_q8_0(x.data(), xq.data(), n, in);
  std::vector<float> want(size_t(n * rows), -7.f), got(want);
  cpu::matmul_rows(w, x.data(), want.data(), n, rb, re, xq.data());
  simd::matmul_rows_q(w, xq.data(), got.data(), n, rb, re);
  for (size_t i = 0; i < want.size(); ++i) {
    CAPTURE(i);
    REQUIRE(bits(got[i]) == bits(want[i]));
  }
}
}  // namespace

TEST_CASE("simd matmul_rows_q equals scalar bitwise for Q4_0 and Q8_0", "[simd][quant]") {
  for (DType t : {DType::Q4_0, DType::Q8_0}) {
    for (int64_t in : {32, 96, 896, 4864})
      for (int64_t rows : {1, 3, 4, 37})
        for (int64_t n : {1, 5}) {
          INFO("type=" << int(t) << " in=" << in << " rows=" << rows << " n=" << n);
          check_matmul_q(t, in, rows, n, 0, rows);
        }
    check_matmul_q(t, 64, 37, 2, 5, 22);   // row range not 4-aligned
    check_matmul_q(t, 64, 37, 2, 36, 37);  // single trailing row
    check_matmul_q(t, 64, 37, 1, 9, 9);    // empty range
  }
}
#endif
