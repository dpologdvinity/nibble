#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <vector>

#include "backend/cpu/ref_ops.h"
#include <random>

#include "fp16.h"
#include "quant.h"
#include "quant_test_util.h"

using namespace nibble;
using Catch::Approx;

namespace {
Tensor f32_tensor(const std::vector<float>& d, int64_t ne0, int64_t ne1 = 1) {
  Tensor t;
  t.type = DType::F32;
  t.n_dims = ne1 > 1 ? 2 : 1;
  t.ne = {ne0, ne1, 1, 1};
  t.data = d.data();
  return t;
}
Approx rel(double v) { return Approx(v).epsilon(1e-6).margin(1e-7); }
}  // namespace

TEST_CASE("ref rmsnorm", "[ref]") {
  std::vector<float> x{1, 2, 3, 4}, w(4, 1.f), y(4);
  cpu::rmsnorm(x.data(), f32_tensor(w, 4), y.data(), 1, 0.f);
  for (int i = 0; i < 4; ++i) CHECK(y[i] == rel(x[i] / std::sqrt(7.5)));
}

TEST_CASE("ref rope", "[ref]") {
  const double c1 = std::cos(1.0), s1 = std::sin(1.0), c2 = std::cos(0.01), s2 = std::sin(0.01);
  RopeParams p{10000.f, false, nullptr};
  std::vector<float> x{1, 0, 1, 0};
  cpu::rope(x.data(), 1, 1, 4, 1, p);
  CHECK(x[0] == rel(c1));
  CHECK(x[1] == rel(s1));
  CHECK(x[2] == rel(c2));
  CHECK(x[3] == rel(s2));

  p.neox = true;
  x = {1, 0, 1, 0};
  cpu::rope(x.data(), 1, 1, 4, 1, p);
  CHECK(x[0] == rel(c1 - s1));
  CHECK(x[1] == rel(0));
  CHECK(x[2] == rel(s1 + c1));
  CHECK(x[3] == rel(0));

  p.neox = false;
  x = {1, 0, 1, 0};
  cpu::rope(x.data(), 1, 1, 4, 0, p);
  CHECK(x == std::vector<float>{1, 0, 1, 0});

  float ff[2] = {1.f, 2.f};
  p.freq_factors = ff;
  x = {1, 0, 1, 0};
  cpu::rope(x.data(), 1, 1, 4, 1, p);
  CHECK(x[2] == rel(std::cos(0.005)));
  CHECK(x[3] == rel(std::sin(0.005)));
}

TEST_CASE("ref attention", "[ref]") {
  std::vector<float> qkv{1, 0, 0, 1}, out(4), sc(8);
  cpu::attention_heads(qkv.data(), qkv.data(), qkv.data(), out.data(), 2, 0, 1, 1, 2, 0, 1, sc.data());
  CHECK(out[0] == rel(1));
  CHECK(out[1] == rel(0));
  const double e = std::exp(1.0 / std::sqrt(2.0)), p0 = 1.0 / (1.0 + e), p1 = e / (1.0 + e);
  CHECK(out[2] == rel(p0));
  CHECK(out[3] == rel(p1));
}

TEST_CASE("ref attention GQA", "[ref]") {
  const int64_t H = 4, KV = 2, D = 2, N = 3;
  std::vector<float> q(H * D, 1.f), k(N * KV * D, 0.f), v(N * KV * D), out(H * D), sc(8);
  for (int64_t j = 0; j < N; ++j)
    for (int64_t g = 0; g < KV; ++g)
      for (int64_t d = 0; d < D; ++d) v[(j * KV + g) * D + d] = float(10 * g + j + d);
  cpu::attention_heads(q.data(), k.data(), v.data(), out.data(), 1, N - 1, H, KV, D, 0, H, sc.data());
  for (int64_t h = 0; h < H; ++h) {
    const int64_t g = h / 2;
    for (int64_t d = 0; d < D; ++d) CHECK(out[h * D + d] == rel(10 * g + 1 + d));
  }
}

TEST_CASE("ref silu_mul", "[ref]") {
  std::vector<float> g{0, 1}, u{2, 3};
  cpu::silu_mul(g.data(), u.data(), 2);
  CHECK(g[0] == rel(0));
  CHECK(g[1] == rel(3.0 / (1.0 + std::exp(-1.0))));
}

TEST_CASE("ref matmul and embed", "[ref]") {
  std::vector<float> w{1, 2, 3, 4, 5, 6}, x{1, 1, 1}, y(2);
  cpu::matmul_rows(f32_tensor(w, 3, 2), x.data(), y.data(), 1, 0, 2);
  CHECK(y[0] == rel(6));
  CHECK(y[1] == rel(15));

  std::vector<uint16_t> h;
  for (float f : w) h.push_back(fp32_to_fp16(f));
  Tensor th;
  th.type = DType::F16;
  th.n_dims = 2;
  th.ne = {3, 2, 1, 1};
  th.data = h.data();
  std::vector<float> y2(2);
  cpu::matmul_rows(th, x.data(), y2.data(), 1, 0, 2);
  CHECK(y2 == y);

  std::vector<float> e(3);
  int32_t id = 1;
  cpu::embed(th, std::span<const int32_t>(&id, 1), e.data());
  CHECK(e == std::vector<float>{4, 5, 6});
}

TEST_CASE("ref softmax overflow-safe", "[ref]") {
  std::vector<float> x{1000.f, 1000.f};
  cpu::softmax_inplace(x.data(), 2);
  CHECK(x[0] == rel(0.5));
  CHECK(x[1] == rel(0.5));
}

// Pins the llama.cpp method: weights . dequant(quantize_q8_0(x)), not plain F32 x.
TEST_CASE("ref quantized matmul uses q8_0 activations", "[ref][quant]") {
  std::mt19937 g(9);
  std::uniform_real_distribution<float> d(-1.f, 1.f);
  const int64_t in = 96, out = 7, n = 2;
  std::vector<float> wf(static_cast<size_t>(in * out)), x(static_cast<size_t>(in * n));
  for (auto& v : wf) v = d(g);
  for (auto& v : x) v = d(g);
  for (DType t : {DType::Q8_0, DType::Q4_0}) {
    auto wb = test_quantize_rows(t, wf, in);
    Tensor w;
    w.type = t;
    w.n_dims = 2;
    w.ne = {in, out, 1, 1};
    w.data = wb.data();
    std::vector<BlockQ8_0> xq(static_cast<size_t>(n * in / 32));
    cpu::quantize_rows_q8_0(x.data(), xq.data(), n, in);
    std::vector<float> y(static_cast<size_t>(n * out));
    const QuantActs qa{xq.data()};
    cpu::matmul_rows(w, x.data(), y.data(), n, 0, out, &qa);
    std::vector<float> wd(static_cast<size_t>(in * out)), xd(static_cast<size_t>(in * n));
    dequantize_row(t, wb.data(), wd.data(), in * out);
    dequantize_row(DType::Q8_0, xq.data(), xd.data(), in * n);
    for (int64_t tt = 0; tt < n; ++tt)
      for (int64_t r = 0; r < out; ++r) {
        double acc = 0;
        for (int64_t k = 0; k < in; ++k) acc += double(wd[r * in + k]) * xd[tt * in + k];
        CHECK(y[tt * out + r] == Approx(acc).epsilon(1e-5).margin(1e-6));
      }
  }
}
