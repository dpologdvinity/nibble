// Tests for the AVX2 kernels in src/backend/cpu/simd: the SIMD helpers, the float matmul, the Q8_0 quantizer
// and the quantized dots. Each kernel is compared with its scalar twin, bit for bit where the design promises
// it, with unaligned data and awkward sizes to exercise vector tails. Everything except the first two cases
// is compiled out when the build has no AVX2.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <type_traits>

#include "backend/cpu/simd/simd.h"
#include "fp16.h"
#include "simd_test_util.h"

using namespace nibble;

// The reported kernel-set string matches the macros the build was compiled with.
TEST_CASE("kernel_set names the active macros", "[simd]") {
#if defined(NIBBLE_HAVE_AVXVNNI)
  CHECK(std::strcmp(simd::kernel_set(), "avx2+fma+f16c+avxvnni") == 0);
#elif defined(NIBBLE_HAVE_AVX2)
  CHECK(std::strcmp(simd::kernel_set(), "avx2+fma+f16c") == 0);
#else
  CHECK(std::strcmp(simd::kernel_set(), "scalar") == 0);
#endif
}

// Self-test of the comparison harness used by other tests (tests/simd_test_util.h).
TEST_CASE("compare_kernels detects mismatches and accepts equal kernels", "[simd]") {
  auto ident = [](const float* in, float* out, size_t n) { for (size_t i = 0; i < n; ++i) out[i] = in[i]; };
  test::compare_kernels(ident, ident, {0, 1, 7, 100}, {1, 2}, 0.f, 0.f);
  auto off = [](const float* in, float* out, size_t n) { for (size_t i = 0; i < n; ++i) out[i] = in[i] + 1e-3f; };
  test::compare_kernels(ident, off, {16}, {1}, 0.f, 1e-2f);  // within atol passes
}

#ifdef NIBBLE_HAVE_AVX2
// The horizontal sums are exact for integer-valued inputs.
TEST_CASE("hsum_ps and hsum_epi32 are exact on integers", "[simd]") {
  alignas(32) float f[8];
  alignas(32) int32_t v[8];
  for (int i = 0; i < 8; ++i) { f[i] = float(i * 1000 + 1); v[i] = -i * 100000 + 7; }
  float ef = 0; int32_t ev = 0;
  for (int i = 0; i < 8; ++i) { ef += f[i]; ev += v[i]; }
  CHECK(simd::hsum_ps(_mm256_load_ps(f)) == ef);
  CHECK(simd::hsum_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(v))) == ev);
}

// The F16C conversion agrees with the software conversion for every half bit pattern.
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

// The float matmul agrees with the scalar oracle across shapes (sums are reordered, so within a relative
// tolerance).
TEST_CASE("simd matmul_rows_f matches scalar for F32 and F16", "[simd]") {
  for (DType t : {DType::F32, DType::F16})
    for (int64_t in : {8, 13, 896, 4864})
      for (int64_t rows : {1, 3, 4, 37})
        for (int64_t nt : {1, 5}) {
          INFO("type=" << int(t) << " in=" << in << " rows=" << rows << " n=" << nt);
          check_matmul(t, in, rows, nt, 0, rows, 1.f);
        }
}

// Row sub-ranges (as used by threads) and very large/small values.
TEST_CASE("simd matmul_rows_f handles sub-ranges and extreme magnitudes", "[simd]") {
  for (DType t : {DType::F32, DType::F16}) {
    check_matmul(t, 70, 37, 2, 5, 22, 1.f);   // row range not 4-aligned
    check_matmul(t, 70, 37, 2, 36, 37, 1.f);  // single trailing row
    check_matmul(t, 70, 37, 1, 9, 9, 1.f);    // empty range
    check_matmul(t, 301, 9, 3, 0, 9, 300.f);  // large (F16 stays finite)
    check_matmul(t, 301, 9, 3, 0, 9, 1e-3f);  // small
  }
}

// The vectorized F16 embedding row converter, including the scalar tail for widths not divisible by 8.
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
BlockQ8_0 rand_act(std::mt19937& g, bool unit_scale) {  // activations: never -128, like quantize_row_q8_0 for normal d
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

// Random rows at several magnitudes (including tiny and huge), zero blocks and dominant elements: scales and
// quants are byte-identical, and a length not divisible by 32 throws.
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

// Values at and around every half-integer up to 127, with both signs, quantize exactly as roundf does (a
// plain vector round-to-nearest would round ties to even).
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

// The int8 multiply-accumulate primitive (and its AVX-VNNI variant) equals a plain integer dot, in the
// documented exact range (activation never -128 where the weight is negative).
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

// Random blocks, block counts around the loop unrolling, misaligned pointers, zero activation blocks and zero
// scales.
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

// Rows with tiny magnitudes have an fp16 scale of 0; their dots must be exactly 0 in both implementations.
TEST_CASE("activation blocks with fp16 scale 0 contribute exactly 0 to scalar and simd dots", "[simd][quant]") {
  // Blocks from rows with tiny amax have fp16(d) == 0; in-contract qs (no -128) must still dot to exactly 0.
  std::mt19937 g(5);
  const int64_t nb = 3;
  std::vector<BlockQ4_0> w4;
  std::vector<BlockQ8_0> w8, a;
  for (int64_t b = 0; b < nb; ++b) {
    w4.push_back(rand_block<BlockQ4_0>(g, true));
    w8.push_back(rand_block<BlockQ8_0>(g, true));
    a.push_back(rand_act(g, true));
    a.back().d = fp32_to_fp16(0.0f);
  }
  REQUIRE(simd::dot_q8_0_q8_0(w8.data(), a.data(), nb) == 0.f);
  REQUIRE(dot_q8_0_q8_0(w8.data(), a.data(), nb) == 0.f);
  REQUIRE(simd::dot_q4_0_q8_0(w4.data(), a.data(), nb) == 0.f);
  REQUIRE(dot_q4_0_q8_0(w4.data(), a.data(), nb) == 0.f);

  // The documented d-bit equality holds for an amax = 1e-40 row (qs are not compared: scalar is UB there).
  std::vector<float> x(32, 1e-40f);
  BlockQ8_0 qs{}, qv{};
  nibble::quantize_row_q8_0(x.data(), &qs, 32);
  simd::quantize_row_q8_0(x.data(), &qv, 32);
  REQUIRE(qs.d == qv.d);
}

// With every scale 1 the dot is an exact integer: extreme quants (127, -127, -128 weights, 0/15 nibbles)
// would reveal any int16 saturation in the maddubs path.
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
  const QuantActs qa{xq.data()};
  cpu::matmul_rows(w, x.data(), want.data(), n, rb, re, &qa);
  simd::matmul_rows_q(w, qa, got.data(), n, rb, re);
  for (size_t i = 0; i < want.size(); ++i) {
    CAPTURE(i);
    REQUIRE(bits(got[i]) == bits(want[i]));
  }
}
}  // namespace

// The row-range matmul (4 rows at a time plus a remainder) equals the scalar matmul for various widths, row
// counts, token counts and ranges (including unaligned starts and empty ranges).
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

// ---- Q5_0, Q5_1 and k-quant AVX2 kernels: bit-identical to the scalar dots ----
#include "data/quant_golden.inc"
#include "quant_golden_gen.h"
#include "quant_test_util.h"

namespace {
using DotFn = float (*)(const void*, const void*, int64_t);
template <class W, class A, float (*F)(const W*, const A*, int64_t)>
float call(const void* w, const void* a, int64_t nb) {
  return F(static_cast<const W*>(w), static_cast<const A*>(a), nb);
}
struct NewType {
  DType t;
  size_t act_bytes;
  DotFn scalar, vec;
  std::vector<int64_t> nbs;
};
const NewType kNew[] = {
    {DType::Q5_0, sizeof(BlockQ8_0), call<BlockQ5_0, BlockQ8_0, dot_q5_0_q8_0>,
     call<BlockQ5_0, BlockQ8_0, simd::dot_q5_0_q8_0>, {1, 2, 28, 152}},
    {DType::Q5_1, sizeof(BlockQ8_1), call<BlockQ5_1, BlockQ8_1, dot_q5_1_q8_1>,
     call<BlockQ5_1, BlockQ8_1, simd::dot_q5_1_q8_1>, {1, 2, 28, 152}},
    {DType::Q4_K, sizeof(BlockQ8_K), call<BlockQ4_K, BlockQ8_K, dot_q4_K_q8_K>,
     call<BlockQ4_K, BlockQ8_K, simd::dot_q4_K_q8_K>, {1, 2, 8, 19, 32}},
    {DType::Q5_K, sizeof(BlockQ8_K), call<BlockQ5_K, BlockQ8_K, dot_q5_K_q8_K>,
     call<BlockQ5_K, BlockQ8_K, simd::dot_q5_K_q8_K>, {1, 2, 8, 19, 32}},
    {DType::Q6_K, sizeof(BlockQ8_K), call<BlockQ6_K, BlockQ8_K, dot_q6_K_q8_K>,
     call<BlockQ6_K, BlockQ8_K, simd::dot_q6_K_q8_K>, {1, 2, 8, 19, 32}},
};

// Same bytes at a 1-byte offset (unaligned blocks).
struct Shifted {
  std::vector<uint8_t> buf;
  explicit Shifted(const uint8_t* p, size_t n) : buf(n + 1) { std::memcpy(buf.data() + 1, p, n); }
  const uint8_t* data() const { return buf.data() + 1; }
};

// Both NaN, or the same bits.
bool same_or_nan(float a, float b) { return (std::isnan(a) && std::isnan(b)) || bits(a) == bits(b); }
}  // namespace

// For Q5_0, Q5_1, Q4_K, Q5_K and Q6_K: single-row dots on unaligned blocks and the row-range matmul equal the
// scalar results bit for bit, for the block counts of real model widths, row counts around the 4-row kernel,
// and row ranges starting at odd offsets.
TEST_CASE("simd new-type dots and matmul_rows_q equal scalar bitwise", "[simd][quant]") {
  for (const NewType& nt : kNew) {
    const auto& info = dtype_info(nt.t);
    for (int64_t nb : nt.nbs) {
      const int64_t in = nb * info.block_size;
      for (int64_t rows : {1, 3, 4, 5, 37})
        for (int64_t n : {1, 5}) {
          INFO(info.name << " in=" << in << " rows=" << rows << " n=" << n);
          std::mt19937 g(uint32_t(in * 7 + rows * 3 + n));
          std::uniform_real_distribution<float> u(-1.f, 1.f);
          std::vector<float> x(size_t(n * in));
          for (float& v : x) v = u(g);
          const TestActs acts = test_quantize_acts(nt.t, x.data(), n, in);
          const QuantActs qa = acts.view();
          const auto wb = test_weight_rows(nt.t, uint64_t(in + rows), rows, in);
          const Shifted ws(wb.data(), wb.size());
          Tensor w;
          w.type = nt.t;
          w.n_dims = 2;
          w.ne = {in, rows, 1, 1};
          w.data = ws.data();
          const int64_t ranges[][2] = {{0, rows}, {1, rows}, {rows / 2, rows}, {0, rows / 2}, {rows, rows}};
          for (const auto& r : ranges) {
            std::vector<float> want(size_t(n * rows), -7.f), got(want);
            cpu::matmul_rows(w, x.data(), want.data(), n, r[0], r[1], &qa);
            simd::matmul_rows_q(w, qa, got.data(), n, r[0], r[1]);
            for (size_t i = 0; i < want.size(); ++i) REQUIRE(bits(got[i]) == bits(want[i]));
          }
          // Single-row dot, weight and activation blocks both unaligned.
          const size_t arow = nt.act_bytes * size_t(in / (nt.t == DType::Q4_K || nt.t == DType::Q5_K || nt.t == DType::Q6_K ? 256 : 32));
          const uint8_t* a0 = nullptr;
          switch (act_type(nt.t)) {
            case ActType::Q8_0: a0 = reinterpret_cast<const uint8_t*>(acts.q8_0.data()); break;
            case ActType::Q8_1: a0 = reinterpret_cast<const uint8_t*>(acts.q8_1.data()); break;
            default: a0 = reinterpret_cast<const uint8_t*>(acts.q8_K.data()); break;
          }
          const Shifted as(a0, arow);
          REQUIRE(bits(nt.vec(ws.data(), as.data(), nb)) == bits(nt.scalar(wb.data(), a0, nb)));
        }
    }
  }
}

// Hostile weights: all-0xFF bytes, every Q6_K scale -128, maximal 6-bit scales and NaN/Inf scales. Finite
// cases are bit-identical; non-finite cases must be NaN in both or equal.
TEST_CASE("simd k-quant dots on hostile blocks", "[simd][quant]") {
  for (const NewType& nt : kNew) {
    if (nt.act_bytes != sizeof(BlockQ8_K)) continue;
    const auto& info = dtype_info(nt.t);
    const size_t bb = size_t(info.block_bytes);
    const bool q6 = nt.t == DType::Q6_K;
    const size_t d_off = q6 ? 208 : 0, dmin_off = 2;
    for (int variant = 0; variant < 6; ++variant) {
      INFO(info.name << " variant=" << variant);
      const int64_t nb = 3;
      auto wb = test_weight_rows(nt.t, 40 + uint64_t(variant), 1, nb * 256);
      std::vector<BlockQ8_K> a(static_cast<size_t>(nb));
      golden::Rng r{uint64_t(variant) + 77};
      golden::fill_q8_K(r, reinterpret_cast<uint8_t*>(a.data()), nb);
      const uint16_t one = fp32_to_fp16(1.f);
      for (int64_t b = 0; b < nb; ++b) {
        uint8_t* p = wb.data() + size_t(b) * bb;
        if (variant == 1) std::memset(p, 0xFF, bb);  // all ones everywhere: scales 63/-1, qh all set
        if (variant == 2 && q6) std::memset(p + 192, 0x80, 16);  // every Q6_K scale = -128
        if (variant == 2 && !q6) std::memset(p + 4, 0xFF, 12);
        if (variant == 1 || variant == 2) {
          std::memcpy(p + d_off, &one, 2);
          if (!q6) std::memcpy(p + dmin_off, &one, 2);
        }
        if (variant == 3 || variant == 4 || variant == 5) {
          const uint16_t bad = variant == 3 ? 0x7E00 : variant == 4 ? 0x7C00 : 0xFC00;  // NaN, +Inf, -Inf
          std::memcpy(p + d_off, &bad, 2);
          if (!q6) std::memcpy(p + dmin_off, &bad, 2);
        }
      }
      const Shifted ws(wb.data(), wb.size());
      const float want = nt.scalar(wb.data(), a.data(), nb), got = nt.vec(ws.data(), a.data(), nb);
      if (variant < 3) REQUIRE(bits(got) == bits(want));
      else REQUIRE(same_or_nan(got, want));
    }
  }
}

// d = dmin = 1 and activation d = 1 make every result an exact integer (small scales keep all partial sums
// below 2^24), so the float equals an int64 dot of the dequantized integers.
// Unit scales and small values make every result an exact integer, checked against an int64 dot of the
// dequantized values.
TEST_CASE("simd k-quant and q5 dots equal an exact integer dot", "[simd][quant]") {
  for (const NewType& nt : kNew) {
    const auto& info = dtype_info(nt.t);
    const bool k = nt.act_bytes == sizeof(BlockQ8_K);
    const int64_t nb = k ? 2 : 4, in = nb * info.block_size;
    for (uint64_t seed = 1; seed <= 20; ++seed) {
      INFO(info.name << " seed=" << seed);
      auto wb = test_weight_rows(nt.t, seed, 1, in);
      const uint16_t one = fp32_to_fp16(1.f);
      for (int64_t b = 0; b < nb; ++b) {
        uint8_t* p = wb.data() + size_t(b) * size_t(info.block_bytes);
        if (nt.t == DType::Q6_K) {
          std::memcpy(p + 208, &one, 2);
          for (int i = 0; i < 16; ++i) p[192 + i] = uint8_t(int8_t(int(p[192 + i] % 7) - 3));  // scales in [-3, 3]
        } else if (k) {
          std::memcpy(p, &one, 2);
          std::memcpy(p + 2, &one, 2);
          for (int i = 0; i < 12; ++i) p[4 + i] &= 0x0F;  // unpacked scales and mins are then all <= 15
        } else {
          std::memcpy(p, &one, 2);
          if (nt.t == DType::Q5_1) {
            const uint16_t m = fp32_to_fp16(1.f);
            std::memcpy(p + 2, &m, 2);
          }
        }
      }
      // Activations: small ints so |sum| < 2048 (Q8_1's fp16 s is then exact).
      std::mt19937 g(static_cast<uint32_t>(seed));
      std::vector<uint8_t> abuf(size_t(nb) * nt.act_bytes);
      int64_t want = 0;
      std::vector<float> deq(static_cast<size_t>(in));
      dequantize_row(nt.t, wb.data(), deq.data(), in);
      std::vector<int> aq(static_cast<size_t>(in));
      for (int& v : aq) v = int(g() % 41) - 20;
      for (int64_t b = 0; b < nb; ++b) {
        uint8_t* p = abuf.data() + size_t(b) * nt.act_bytes;
        const int64_t per = 32;
        if (k) {
          const float d1 = 1.f;
          std::memcpy(p, &d1, 4);
          for (int j = 0; j < 256; ++j) p[4 + j] = uint8_t(int8_t(aq[size_t(b * 256 + j)]));
          for (int gq = 0; gq < 16; ++gq) {
            int sum = 0;
            for (int i = 0; i < 16; ++i) sum += aq[size_t(b * 256 + gq * 16 + i)];
            const int16_t sv = int16_t(sum);
            std::memcpy(p + 260 + gq * 2, &sv, 2);
          }
        } else {
          std::memcpy(p, &one, 2);
          int sum = 0;
          const size_t off = nt.act_bytes == sizeof(BlockQ8_1) ? 4 : 2;
          for (int j = 0; j < 32; ++j) {
            p[off + size_t(j)] = uint8_t(int8_t(aq[size_t(b * per + j)]));
            sum += aq[size_t(b * per + j)];
          }
          if (nt.act_bytes == sizeof(BlockQ8_1)) {
            const uint16_t sh = fp32_to_fp16(float(sum));
            std::memcpy(p + 2, &sh, 2);
          }
        }
      }
      for (int64_t i = 0; i < in; ++i) want += int64_t(deq[size_t(i)]) * aq[size_t(i)];
      REQUIRE(nt.vec(wb.data(), abuf.data(), nb) == float(want));
      REQUIRE(nt.scalar(wb.data(), abuf.data(), nb) == float(want));
    }
  }
}

// The AVX2 Q5 and K-quant dots reproduce ggml's generic results for the golden inputs.
TEST_CASE("golden: simd q5 and k-quant dots bitwise equal ggml generic", "[quant][golden][simd]") {
  for (const auto& c : golden::kDot5Cases) {
    CAPTURE(c.seed, c.n);
    const int nb = c.n / 32;
    const auto z = size_t(nb);
    std::vector<BlockQ5_0> w50(z);
    std::vector<BlockQ8_0> a80(z);
    std::vector<BlockQ5_1> w51(z);
    std::vector<BlockQ8_1> a81(z);
    golden::Rng r{c.seed};
    golden::fill_q5_0(r, reinterpret_cast<uint8_t*>(w50.data()), nb);
    golden::fill_q8_0(r, reinterpret_cast<uint8_t*>(a80.data()), nb);
    golden::fill_q5_1(r, reinterpret_cast<uint8_t*>(w51.data()), nb);
    golden::fill_q8_1(r, reinterpret_cast<uint8_t*>(a81.data()), nb);
    // The golden Q8_0 activations are random bytes and include -128, which the AVX2 sign trick cannot take as
    // the activation of a Q8_0 *weight*; Q5_0 weights are small ([-16, 15]) and pass the activation as the
    // sign source, so -128 is exact here.
    REQUIRE(bits(simd::dot_q5_0_q8_0(w50.data(), a80.data(), nb)) == c.q5_0_bits);
    REQUIRE(bits(simd::dot_q5_1_q8_1(w51.data(), a81.data(), nb)) == c.q5_1_bits);
  }
  for (const auto& c : golden::kKDotCases) {
    CAPTURE(c.seed, c.n);
    const int nb = c.n / 256;
    const auto z = size_t(nb);
    std::vector<BlockQ4_K> w4(z);
    std::vector<BlockQ5_K> w5(z);
    std::vector<BlockQ6_K> w6(z);
    std::vector<BlockQ8_K> a8(z);
    golden::Rng r{c.seed};
    golden::fill_q4_K(r, reinterpret_cast<uint8_t*>(w4.data()), nb);
    golden::fill_q5_K(r, reinterpret_cast<uint8_t*>(w5.data()), nb);
    golden::fill_q6_K(r, reinterpret_cast<uint8_t*>(w6.data()), nb);
    golden::fill_q8_K(r, reinterpret_cast<uint8_t*>(a8.data()), nb);
    REQUIRE(bits(simd::dot_q4_K_q8_K(w4.data(), a8.data(), nb)) == c.q4k_bits);
    REQUIRE(bits(simd::dot_q5_K_q8_K(w5.data(), a8.data(), nb)) == c.q5k_bits);
    REQUIRE(bits(simd::dot_q6_K_q8_K(w6.data(), a8.data(), nb)) == c.q6k_bits);
  }
}
#endif
