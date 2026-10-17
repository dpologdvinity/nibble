#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <random>
#include <vector>

#include "backend/cpu/cpu_backend.h"
#include "backend/cpu/ref_ops.h"
#include "fp16.h"

using namespace nibble;

namespace {
std::vector<float> rnd(size_t n, std::mt19937& g) {
  std::uniform_real_distribution<float> d(-1.f, 1.f);
  std::vector<float> v(n);
  for (auto& x : v) x = d(g);
  return v;
}
double maxdiff(const float* a, const float* b, size_t n) {
  double m = 0;
  for (size_t i = 0; i < n; ++i) m = std::max(m, double(std::fabs(a[i] - b[i])));
  return m;
}
}  // namespace

TEST_CASE("cpu backend matmul", "[cpu_backend]") {
  ThreadPool pool(4);
  cpu::CpuBackend be(pool);
  std::mt19937 g(1);
  const int64_t in = 300, out = 37, n = 5;
  auto w = rnd(in * out, g);
  auto xh = rnd(n * in, g);
  std::vector<uint16_t> w16;
  for (float f : w) w16.push_back(fp32_to_fp16(f));
  float* x = be.alloc(n * in);
  float* y = be.alloc(n * out);
  std::copy(xh.begin(), xh.end(), x);
  for (DType ty : {DType::F32, DType::F16}) {
    Tensor t;
    t.type = ty;
    t.n_dims = 2;
    t.ne = {in, out, 1, 1};
    t.data = ty == DType::F32 ? static_cast<const void*>(w.data()) : w16.data();
    std::vector<float> ref(n * out), got(n * out);
    cpu::matmul_rows(t, x, ref.data(), n, 0, out);
    be.matmul(t, x, y, n);
    be.read(got.data(), y, n * out);
    CHECK(maxdiff(ref.data(), got.data(), got.size()) <= 1e-5);
  }
}

TEST_CASE("cpu backend attention", "[cpu_backend]") {
  ThreadPool pool(4);
  cpu::CpuBackend be(pool);
  std::mt19937 g(2);
  const int64_t n = 7, pos0 = 3, H = 6, KV = 2, D = 16, ctx = pos0 + n;
  auto q = rnd(n * H * D, g), k = rnd(ctx * KV * D, g), v = rnd(ctx * KV * D, g);
  std::vector<float> ref(n * H * D), sc(ctx);
  cpu::attention_heads(q.data(), k.data(), v.data(), ref.data(), n, pos0, H, KV, D, 0, H, sc.data());
  float* out = be.alloc(n * H * D);
  for (int rep = 0; rep < 2; ++rep) {
    be.attention(q.data(), k.data(), v.data(), out, n, pos0, H, KV, D);
    CHECK(maxdiff(ref.data(), out, ref.size()) <= 1e-6);
  }
}

TEST_CASE("cpu backend supports", "[cpu_backend]") {
  ThreadPool pool(1);
  cpu::CpuBackend be(pool);
  CHECK(be.supports(DType::F32));
  CHECK(be.supports(DType::F16));
  CHECK_FALSE(be.supports(DType::Q4_0));
}
