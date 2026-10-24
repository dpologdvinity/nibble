// Tests for the prefill path: the tiled GEMM kernels (src/backend/cpu/simd/gemm.cpp) must give the same
// results as the row-at-a-time kernels, so prompt processing and token generation agree.
#include <catch2/catch_test_macros.hpp>
#include <bit>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "backend/cpu/cpu_backend.h"
#include "backend/cpu/ref_ops.h"
#include "backend/cpu/simd/simd.h"
#include "fp16.h"
#include "quant_test_util.h"
#include "tensor.h"

using namespace nibble;

namespace {
std::vector<float> rnd(size_t n, std::mt19937& g) {
  std::uniform_real_distribution<float> d(-1.f, 1.f);
  std::vector<float> v(n);
  for (auto& x : v) x = d(g);
  return v;
}

// A rows x in weight of type t built from uniform [-1,1) values; owns its bytes.
struct Weight {
  std::vector<float> f32;
  std::vector<uint16_t> f16;
  std::vector<uint8_t> q;
  Tensor t;
  Weight(DType type, int64_t in, int64_t rows, std::mt19937& g) {
    f32 = rnd(size_t(in * rows), g);
    t.type = type;
    t.n_dims = 2;
    t.ne = {in, rows, 1, 1};
    if (type == DType::F16) {
      for (float v : f32) f16.push_back(fp32_to_fp16(v));
      t.data = f16.data();
    } else if (type == DType::Q4_0 || type == DType::Q8_0) {
      q = test_quantize_rows(type, f32, in);
      t.data = q.data();
    } else {
      t.data = f32.data();
    }
  }
};

bool quant(DType t) { return t == DType::Q4_0 || t == DType::Q8_0; }
}  // namespace

// Prefill-sized matmuls through the backend (tiled GEMM for n >= 4 when SIMD is on) against the scalar
// oracle: quantized results are bit-identical, F32/F16 within 1e-5 of the sum of |x*w| (sum order differs).
TEST_CASE("cpu backend prefill matmul matches scalar for all dtypes", "[gemm][cpu_backend]") {
  ThreadPool pool(4);
  cpu::CpuBackend be(pool);
  be.reserve({64, 4, 513, 4864});
  std::mt19937 g(11);
  for (DType type : {DType::F32, DType::F16, DType::Q8_0, DType::Q4_0})
    for (int64_t in : {32, 896, 4864})
      for (int64_t rows : {1, 37, 896, 4864})
        for (int64_t n : {4, 5, 17, 64, 513}) {
          if (n * rows * in > 40'000'000) continue;  // scalar-oracle budget; every size still appears
          INFO("type=" << int(type) << " in=" << in << " rows=" << rows << " n=" << n);
          Weight w(type, in, rows, g);
          const auto x = rnd(size_t(n * in), g);
          std::vector<float> want(size_t(n * rows)), got(want.size(), -7.f);
          std::vector<BlockQ8_0> xq;
          if (quant(type)) {
            xq.resize(size_t(n * in / 32));
            cpu::quantize_rows_q8_0(x.data(), xq.data(), n, in);
          }
          const QuantActs qa{xq.data()};
          cpu::matmul_rows(w.t, x.data(), want.data(), n, 0, rows, xq.empty() ? nullptr : &qa);
          be.matmul(w.t, x.data(), got.data(), n);
          int64_t mismatches = 0;
          for (int64_t t = 0; t < n; ++t)
            for (int64_t r = 0; r < rows; ++r) {
              const size_t i = size_t(t * rows + r);
              if (quant(type)) {
                if (std::bit_cast<uint32_t>(got[i]) != std::bit_cast<uint32_t>(want[i]) && mismatches++ == 0)
                  UNSCOPED_INFO("first mismatch t=" << t << " r=" << r << " want=" << want[i] << " got=" << got[i]);
                continue;
              }
              double mag = 0;
              for (int64_t k = 0; k < in; ++k)
                mag += std::fabs(double(x[size_t(t * in + k)]) * (type == DType::F16
                                                                      ? fp16_to_fp32(w.f16[size_t(r * in + k)])
                                                                      : w.f32[size_t(r * in + k)]));
              if (!(std::fabs(double(got[i]) - want[i]) <= 1e-5 * mag) && mismatches++ == 0)
                UNSCOPED_INFO("first mismatch t=" << t << " r=" << r << " want=" << want[i] << " got=" << got[i]
                                                  << " mag=" << mag);
            }
          CHECK(mismatches == 0);
        }
}

#ifdef NIBBLE_HAVE_AVX2
namespace {
// gemm_f/gemm_q over [rb,re) vs the row-at-a-time SIMD path: bit-identical for every dtype, any n.
// Weight rows and activations sit at misaligned offsets.
void check_gemm(DType type, int64_t in, int64_t rows, int64_t n, int64_t rb, int64_t re) {
  std::mt19937 g(uint32_t(in * 7 + rows * 3 + n));
  Weight w(type, in, rows, g);
  if (quant(type)) {  // arbitrary quants (Q8_0 -128, every nibble), quantizer scales
    const size_t bs = type == DType::Q4_0 ? sizeof(BlockQ4_0) : sizeof(BlockQ8_0);
    for (size_t i = 0; i < w.q.size(); ++i)
      if (i % bs >= 2) w.q[i] = uint8_t(g());
  }
  // Shift the weight one element / byte off its allocation's alignment.
  const size_t bytes = size_t(w.t.row_bytes() * rows);
  const size_t off = type == DType::F32 ? 4 : type == DType::F16 ? 2 : 1;  // keeps elements aligned
  std::vector<uint8_t> shifted(bytes + 64 + off);
  uint8_t* base = shifted.data() + (64 - reinterpret_cast<uintptr_t>(shifted.data()) % 64) % 64;
  std::memcpy(base + off, w.t.data, bytes);
  w.t.data = base + off;
  const auto xv = rnd(size_t(n * in) + 1, g);
  const float* x = xv.data() + 1;
  std::vector<BlockQ8_0> xq(size_t(n * in / 32));
  if (quant(type)) simd::quantize_row_q8_0(x, xq.data(), n * in);
  std::vector<float> want(size_t(n * rows), -7.f), got(want);
  if (quant(type)) {
    simd::matmul_rows_q(w.t, QuantActs{xq.data()}, want.data(), n, rb, re);
    std::vector<simd::BlockQ8_0x8> xp(size_t((n + 7) / 8 * (in / 32)));
    simd::pack_q8_0_x8(xq.data(), n, in / 32, xp.data());
    simd::gemm_q(w.t, xp.data(), got.data(), n, rb, re);
  } else {
    simd::matmul_rows_f(w.t, x, want.data(), n, rb, re);
    simd::gemm_f(w.t, x, got.data(), n, rb, re);
  }
  int64_t mismatches = 0;
  for (size_t i = 0; i < want.size(); ++i)
    if (std::bit_cast<uint32_t>(got[i]) != std::bit_cast<uint32_t>(want[i]) && mismatches++ == 0)
      UNSCOPED_INFO("first mismatch i=" << i << " want=" << want[i] << " got=" << got[i]);
  CHECK(mismatches == 0);
}
}  // namespace

// Calls the tiled kernels and the row kernels directly on the same misaligned data and row ranges and
// requires identical bits for every dtype and token count.
TEST_CASE("gemm equals the row-at-a-time simd path bitwise", "[gemm][simd]") {
  for (DType t : {DType::F32, DType::F16, DType::Q8_0, DType::Q4_0}) {
    const std::vector<int64_t> ins =
        quant(t) ? std::vector<int64_t>{32, 96, 896, 4864} : std::vector<int64_t>{8, 13, 300, 896, 4864};
    for (int64_t in : ins)
      for (int64_t rows : {1, 3, 4, 37, 64})
        for (int64_t n : {1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 17, 64}) {
          INFO("type=" << int(t) << " in=" << in << " rows=" << rows << " n=" << n);
          check_gemm(t, in, rows, n, 0, rows);
        }
    const int64_t in = quant(t) ? 64 : 70;
    check_gemm(t, in, 37, 9, 5, 22);   // row range not 4-aligned
    check_gemm(t, in, 37, 9, 36, 37);  // single trailing row
    check_gemm(t, in, 37, 9, 9, 9);    // empty range
    check_gemm(t, 896, 896, 513, 0, 896);
  }
}
#endif
