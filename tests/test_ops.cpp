// Tests for the backend operations beyond matmul: fused multi-weight matmul equivalence, and the AVX2
// elementwise kernels (add, RMSNorm, SiLU gating, attention) against their scalar oracles in ref_ops.cpp,
// once per SIMD tier this CPU supports (test::for_each_tier; skipped when there is none). The vector exp is
// tested per tier in simd_helpers_test.cpp.
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "backend/cpu/cpu_backend.h"
#include "backend/cpu/ref_ops.h"
#include "backend/cpu/simd/dispatch.h"
#include "fp16.h"
#include "quant_test_util.h"
#include "simd_test_util.h"
#include "tensor.h"

using namespace nibble;

namespace {
// Uniform random floats in [-scale, scale).
std::vector<float> rnd(size_t n, std::mt19937& g, float scale = 1.f) {
  std::uniform_real_distribution<float> d(-scale, scale);
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
}  // namespace

// Q/K/V-like (unequal row counts, not multiples of 16) and gate/up-like weights sharing one input: one
// matmul_multi call must equal separate matmul calls bit for bit, for decode and GEMM-sized n.
TEST_CASE("matmul_multi equals separate matmul calls bitwise", "[cpu_backend][ops]") {
  const int64_t in = 96;
  std::mt19937 g(21);
  for (unsigned threads : {1u, 4u}) {
    ThreadPool pool(threads);
    cpu::CpuBackend be(pool);
    be.reserve({64, 4, 9, in});
    for (DType a : {DType::F32, DType::F16, DType::Q8_0, DType::Q4_0})
      for (DType b : {a, DType::F16})  // b != a: mixed dtypes fall back to one matmul per weight
        for (int64_t n : {1, 3, 4, 9}) {
          Weight w0(a, in, 37, g), w1(b, in, 13, g), w2(a, in, 20, g);
          const Tensor* ws[] = {&w0.t, &w1.t, &w2.t};
          const int64_t rows[] = {37, 13, 20};
          float* x = be.alloc(n * in);
          const auto xh = rnd(size_t(n * in), g);
          std::copy(xh.begin(), xh.end(), x);
          float* sep[3];
          float* fused[3];
          float* base[3];
          for (int i = 0; i < 3; ++i) {
            sep[i] = be.alloc(n * rows[i]);
            fused[i] = be.alloc(n * rows[i]);
            base[i] = be.alloc(n * rows[i]);
            be.matmul(*ws[i], x, sep[i], n);
          }
          be.matmul_multi(ws, x, fused, n);
          be.Backend::matmul_multi(ws, x, base, n);  // the default other backends inherit
          for (int i = 0; i < 3; ++i) {
            INFO(dtype_info(a).name << "/" << dtype_info(b).name << " n=" << n << " threads=" << threads << " w" << i);
            CHECK(std::memcmp(sep[i], fused[i], size_t(n * rows[i]) * sizeof(float)) == 0);
            CHECK(std::memcmp(sep[i], base[i], size_t(n * rows[i]) * sizeof(float)) == 0);
            be.free(sep[i]);
            be.free(fused[i]);
            be.free(base[i]);
          }
          be.free(x);
        }
  }
}

// Weights with different input widths cannot share a quantized copy of x, so matmul_multi must fall back to
// one call per weight and still give the separate-call results.
TEST_CASE("matmul_multi with different input widths equals separate matmul calls", "[cpu_backend][ops]") {
  ThreadPool pool(4);
  cpu::CpuBackend be(pool);
  std::mt19937 g(22);
  for (DType a : {DType::F32, DType::Q8_0})
    for (int64_t n : {1, 5}) {
      Weight w0(a, 64, 24, g), w1(a, 96, 17, g);  // each reads its own ne0 floats per row of x
      const Tensor* ws[] = {&w0.t, &w1.t};
      float* x = be.alloc(n * 96);
      const auto xh = rnd(size_t(n * 96), g);
      std::copy(xh.begin(), xh.end(), x);
      float* sep[] = {be.alloc(n * 24), be.alloc(n * 17)};
      float* fused[] = {be.alloc(n * 24), be.alloc(n * 17)};
      be.matmul_multi(ws, x, fused, n);  // first, so no earlier call leaves w1-width activations in scratch
      for (int i = 0; i < 2; ++i) be.matmul(*ws[i], x, sep[i], n);
      INFO(dtype_info(a).name << " n=" << n);
      CHECK(std::memcmp(sep[0], fused[0], size_t(n * 24) * sizeof(float)) == 0);
      CHECK(std::memcmp(sep[1], fused[1], size_t(n * 17) * sizeof(float)) == 0);
    }
}

// Elementwise add is exact, so the vector and scalar versions agree bit for bit for sizes around the vector
// width.
TEST_CASE("simd add equals scalar bitwise", "[simd][ops]") {
  test::for_each_tier([&](const simd::Kernels& k) {
    std::mt19937 g(1);
    for (int64_t count : {1, 7, 8, 13, 896, 4867}) {
      auto x = rnd(size_t(count), g), y = rnd(size_t(count), g), ref = x;
      cpu::add(ref.data(), y.data(), count);
      k.add(x.data(), y.data(), count);
      CHECK(std::memcmp(ref.data(), x.data(), size_t(count) * sizeof(float)) == 0);
    }
  });
}

// The vector RMSNorm sums squares in a different order than the scalar, so results agree to 1e-6 relative
// across widths and input magnitudes.
TEST_CASE("simd rmsnorm matches scalar within 1e-6 relative", "[simd][ops]") {
  test::for_each_tier([&](const simd::Kernels& k) {
    std::mt19937 g(2);
    for (int64_t d : {8, 13, 64, 896, 2048})
      for (float scale : {1e-3f, 1.f, 300.f}) {
        const int64_t n = 3;
        auto x = rnd(size_t(n * d), g, scale), wv = rnd(size_t(d), g);
        Tensor w;
        w.type = DType::F32;
        w.n_dims = 1;
        w.ne = {d, 1, 1, 1};
        w.data = wv.data();
        std::vector<float> ref(x.size()), got(x.size());
        cpu::rmsnorm(x.data(), w, ref.data(), n, 1e-6f);
        k.rmsnorm(x.data(), w, got.data(), n, 1e-6f);
        for (size_t i = 0; i < ref.size(); ++i) {
          INFO("d=" << d << " scale=" << scale << " i=" << i);
          CHECK(std::fabs(ref[i] - got[i]) <= 1e-6f * std::fabs(ref[i]));
        }
      }
  });
}

// SwiGLU gating with the polynomial exp agrees with the scalar std::exp version, including very large
// positive and negative gate values.
TEST_CASE("simd silu_mul matches scalar within 1e-6 relative, including saturated gates", "[simd][ops]") {
  test::for_each_tier([&](const simd::Kernels& k) {
    std::mt19937 g(3);
    for (int64_t count : {1, 7, 8, 13, 4864, 4871}) {
      auto gate = rnd(size_t(count), g, 12.f), up = rnd(size_t(count), g);
      if (count > 8) gate[0] = 100.f, gate[1] = -100.f, gate[2] = -88.9f, gate[3] = 88.9f, gate[4] = 0.f;
      auto ref = gate;
      cpu::silu_mul(ref.data(), up.data(), count);
      k.silu_mul(gate.data(), up.data(), count);
      for (int64_t i = 0; i < count; ++i) {
        INFO("count=" << count << " i=" << i << " ref=" << ref[size_t(i)] << " simd=" << gate[size_t(i)]);
        CHECK(std::fabs(ref[size_t(i)] - gate[size_t(i)]) <= 1e-6f * std::fabs(ref[size_t(i)]) + 1e-30f);
      }
    }
  });
}

// silu_mul element i must not depend on where it falls in the call (chunked prefill splits the count
// differently from batch): every element goes through the vector path, tails included.
TEST_CASE("simd silu_mul result is independent of the call's count", "[simd][ops]") {
  test::for_each_tier([&](const simd::Kernels& k) {
    std::mt19937 g(4);
    const int64_t count = 45;
    auto gate = rnd(size_t(count), g, 8.f), up = rnd(size_t(count), g);
    auto whole = gate;
    k.silu_mul(whole.data(), up.data(), count);
    for (int64_t split : {1, 5, 9, 17}) {
      auto parts = gate;
      k.silu_mul(parts.data(), up.data(), split);
      k.silu_mul(parts.data() + split, up.data() + split, count - split);
      CHECK(std::memcmp(whole.data(), parts.data(), size_t(count) * sizeof(float)) == 0);
    }
  });
}

// Score rounding grows with |q|·|k| (sum order differs), so the bound scales with qscale.
TEST_CASE("simd attention matches scalar within 2e-6 per unit of q scale", "[simd][ops]") {
  test::for_each_tier([&](const simd::Kernels& ks) {
    std::mt19937 g(5);
    struct Case {
      int64_t n, pos0, H, KV, D;
      float qscale;
    };
    for (const Case c : {Case{1, 0, 2, 1, 8, 1.f}, Case{7, 3, 6, 2, 16, 1.f}, Case{4, 300, 4, 2, 64, 1.f},
                         Case{3, 40, 2, 2, 13, 1.f}, Case{2, 20, 4, 1, 128, 1.f}, Case{5, 60, 4, 2, 64, 30.f}}) {
      const int64_t ctx = c.pos0 + c.n;
      auto q = rnd(size_t(c.n * c.H * c.D), g, c.qscale), k = rnd(size_t(ctx * c.KV * c.D), g),
           v = rnd(size_t(ctx * c.KV * c.D), g);
      std::vector<float> ref(q.size()), got(q.size()), sc(static_cast<size_t>(ctx));
      cpu::attention_heads(q.data(), k.data(), v.data(), ref.data(), c.n, c.pos0, c.H, c.KV, c.D, 0, c.H, sc.data());
      ks.attention_heads(q.data(), k.data(), v.data(), got.data(), c.n, c.pos0, c.H, c.KV, c.D, 0, c.H, sc.data());
      for (size_t i = 0; i < ref.size(); ++i) {
        INFO("D=" << c.D << " pos0=" << c.pos0 << " qscale=" << c.qscale << " i=" << i);
        CHECK(std::fabs(ref[i] - got[i]) <= 2e-6f * c.qscale);
      }
    }
  });
}
