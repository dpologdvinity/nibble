#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "backend/cpu/cpu_backend.h"
#include "backend/cpu/ref_ops.h"
#include "fp16.h"
#include "quant_test_util.h"
#include "nibble/error.h"

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
  for (DType t : {DType::Q4_0, DType::Q5_0, DType::Q5_1, DType::Q8_0, DType::Q4_K, DType::Q5_K, DType::Q6_K})
    CHECK(be.supports(t));
}

TEST_CASE("cpu backend reserve prevents scratch growth", "[cpu_backend]") {
  ThreadPool pool(4);
  cpu::CpuBackend be(pool);
  be.reserve({64, 6, 1, 32});
  const size_t cap = be.scratch_capacity();
  std::mt19937 g(3);
  const int64_t H = 6, KV = 2, D = 16;
  auto k = rnd(64 * KV * D, g), v = rnd(64 * KV * D, g), q = rnd(H * D, g);
  float* out = be.alloc(H * D);
  for (int64_t pos = 0; pos < 64; ++pos) {
    be.attention(q.data(), k.data(), v.data(), out, 1, pos, H, KV, D);
    REQUIRE(be.scratch_capacity() == cap);
  }
}

TEST_CASE("cpu backend free releases allocation", "[cpu_backend]") {
  ThreadPool pool(1);
  cpu::CpuBackend be(pool);
  const size_t base = be.live_allocations();
  float* a = be.alloc(100);
  float* b = be.alloc(200);
  REQUIRE(be.live_allocations() == base + 2);
  be.free(a);
  REQUIRE(be.live_allocations() == base + 1);
  be.free(nullptr);
  REQUIRE(be.live_allocations() == base + 1);
  be.free(b);
  REQUIRE(be.live_allocations() == base);
}

TEST_CASE("cpu backend alloc rejects bad sizes", "[cpu_backend]") {
  ThreadPool pool(1);
  cpu::CpuBackend be(pool);
  CHECK_THROWS_AS(be.alloc(-1), Error);
  CHECK_THROWS_AS(be.alloc(INT64_MAX / 2), Error);
  CHECK_THROWS_AS(be.alloc(INT64_MAX), Error);
  CHECK(be.live_allocations() == 0);
}

namespace {
Tensor q_tensor(DType t, const std::vector<uint8_t>& bytes, int64_t in, int64_t out) {
  Tensor w;
  w.type = t;
  w.n_dims = 2;
  w.ne = {in, out, 1, 1};
  w.data = bytes.data();
  return w;
}
}  // namespace

constexpr DType kQuantTypes[] = {DType::Q4_0, DType::Q5_0, DType::Q5_1, DType::Q8_0,
                                 DType::Q4_K, DType::Q5_K, DType::Q6_K};

TEST_CASE("cpu backend quantized matmul equals ref_ops bitwise", "[cpu_backend][quant]") {
  ThreadPool pool(4);
  cpu::CpuBackend be(pool);
  std::mt19937 g(5);
  const int64_t in = 512, out = 37;  // 512 is a block multiple of every type
  be.reserve({64, 4, 5, in});
  for (DType t : kQuantTypes)
    for (int64_t n : {1, 5}) {
      INFO("type=" << dtype_info(t).name << " n=" << n);
      auto wb = test_weight_rows(t, 100 + static_cast<uint64_t>(t), out, in);
      Tensor w = q_tensor(t, wb, in, out);
      auto x = rnd(static_cast<size_t>(n * in), g);
      const TestActs acts = test_quantize_acts(t, x.data(), n, in);
      const QuantActs qa = acts.view();
      std::vector<float> ref(static_cast<size_t>(n * out)), got(ref.size());
      cpu::matmul_rows(w, x.data(), ref.data(), n, 0, out, &qa);
      be.matmul(w, x.data(), got.data(), n);
      CHECK(std::memcmp(ref.data(), got.data(), ref.size() * sizeof(float)) == 0);
    }
}

TEST_CASE("cpu backend quantized embed dequantizes the row", "[cpu_backend][quant]") {
  ThreadPool pool(1);
  cpu::CpuBackend be(pool);
  const int64_t d = 512;
  for (DType t : kQuantTypes) {
    INFO("type=" << dtype_info(t).name);
    auto wb = test_weight_rows(t, 7 + static_cast<uint64_t>(t), 5, d);
    Tensor tab = q_tensor(t, wb, d, 5);
    std::vector<int32_t> ids = {3, 0};
    std::vector<float> got(2 * d), want(d);
    be.embed(tab, ids, got.data());
    const size_t rb = static_cast<size_t>(tab.row_bytes());
    dequantize_row(t, wb.data() + 3 * rb, want.data(), d);
    CHECK(std::memcmp(got.data(), want.data(), d * sizeof(float)) == 0);
    dequantize_row(t, wb.data(), want.data(), d);
    CHECK(std::memcmp(got.data() + d, want.data(), d * sizeof(float)) == 0);
  }
}

namespace {
struct MixCase {
  std::vector<DType> types;
  uint64_t quantizations;  // expected activation-quantization passes for one matmul_multi call
};
}  // namespace

TEST_CASE("matmul_multi with mixed weight types equals separate matmuls", "[cpu_backend][quant]") {
  ThreadPool pool(4);
  cpu::CpuBackend be(pool);
  std::mt19937 g(21);
  const int64_t in = 512, out = 24;
  be.reserve({64, 4, 5, in});
  const MixCase cases[] = {{{DType::Q4_K, DType::Q4_K, DType::Q6_K}, 1},
                           {{DType::Q5_1, DType::Q5_1, DType::Q8_0}, 2},
                           {{DType::Q5_0, DType::Q8_0, DType::Q5_0}, 1},
                           {{DType::F32, DType::Q4_K}, 1}};
  for (const MixCase& c : cases)
    for (int64_t n : {1, 5}) {
      INFO("first type=" << dtype_info(c.types[0]).name << " n=" << n);
      std::vector<std::vector<uint8_t>> bytes;
      std::vector<Tensor> ts;
      for (size_t i = 0; i < c.types.size(); ++i) {
        if (c.types[i] == DType::F32) {
          const auto f = rnd(static_cast<size_t>(in * out), g);
          bytes.emplace_back(reinterpret_cast<const uint8_t*>(f.data()), reinterpret_cast<const uint8_t*>(f.data() + f.size()));
        } else {
          bytes.push_back(test_weight_rows(c.types[i], 300 + i, out, in));
        }
      }
      for (size_t i = 0; i < c.types.size(); ++i) ts.push_back(q_tensor(c.types[i], bytes[i], in, out));
      std::vector<const Tensor*> ws;
      for (const Tensor& t : ts) ws.push_back(&t);
      const auto x = rnd(static_cast<size_t>(n * in), g);
      std::vector<std::vector<float>> want(ts.size(), std::vector<float>(static_cast<size_t>(n * out))), got = want;
      std::vector<float*> ys;
      for (auto& v : got) ys.push_back(v.data());
      for (size_t i = 0; i < ts.size(); ++i) be.matmul(ts[i], x.data(), want[i].data(), n);
      const uint64_t before = be.quantize_calls();
      be.matmul_multi(ws, x.data(), ys, n);
      CHECK(be.quantize_calls() - before == c.quantizations);
      for (size_t i = 0; i < ts.size(); ++i)
        CHECK(std::memcmp(want[i].data(), got[i].data(), want[i].size() * sizeof(float)) == 0);
    }
}

TEST_CASE("cpu backend reserve prevents quantized matmul allocation", "[cpu_backend][quant]") {
  ThreadPool pool(4);
  cpu::CpuBackend be(pool);
  std::mt19937 g(8);
  const int64_t in = 512, out = 24;
  be.reserve({64, 4, 4, in});
  float* y = be.alloc(4 * out);
  const auto x = rnd(static_cast<size_t>(4 * in), g);
  const size_t live = be.live_allocations(), cap = be.quant_scratch_capacity();
  for (DType t : {DType::Q4_0, DType::Q4_K, DType::Q5_1}) {
    auto wb = test_weight_rows(t, 9, out, in);
    Tensor w = q_tensor(t, wb, in, out);
    for (int i = 0; i < 64; ++i) {
      be.matmul(w, x.data(), y, 1);
      REQUIRE(be.live_allocations() == live);
      REQUIRE(be.quant_scratch_capacity() == cap);
    }
  }
  auto w4 = test_weight_rows(DType::Q4_K, 1, out, in), w6 = test_weight_rows(DType::Q6_K, 2, out, in),
       w8 = test_weight_rows(DType::Q8_0, 3, out, in), w51 = test_weight_rows(DType::Q5_1, 4, out, in);
  const Tensor t4 = q_tensor(DType::Q4_K, w4, in, out), t6 = q_tensor(DType::Q6_K, w6, in, out),
               t8 = q_tensor(DType::Q8_0, w8, in, out), t51 = q_tensor(DType::Q5_1, w51, in, out);
  const Tensor* ws[] = {&t4, &t6, &t8, &t51};
  float* y2 = be.alloc(4 * out);
  float* y3 = be.alloc(4 * out);
  float* y4 = be.alloc(4 * out);
  float* const ys[] = {y, y2, y3, y4};
  const size_t live2 = be.live_allocations(), cap2 = be.quant_scratch_capacity();
  for (int64_t n : {1, 4}) {
    be.matmul_multi(ws, x.data(), ys, n);
    REQUIRE(be.live_allocations() == live2);
    REQUIRE(be.quant_scratch_capacity() == cap2);
  }
}

TEST_CASE("cpu backend F16 and F32 embed match the scalar oracle for several ids", "[cpu_backend]") {
  ThreadPool pool(1);
  cpu::CpuBackend be(pool);
  std::mt19937 g(11);
  const int64_t d = 21, vocab = 9;  // d not a multiple of 8: exercises the tail
  auto f = rnd(size_t(d * vocab), g);
  std::vector<uint16_t> h(f.size());
  for (size_t i = 0; i < f.size(); ++i) h[i] = fp32_to_fp16(f[i]);
  std::vector<int32_t> ids = {8, 0, 5, 5, 2};
  for (DType t : {DType::F16, DType::F32}) {
    Tensor tab;
    tab.type = t;
    tab.n_dims = 2;
    tab.ne = {d, vocab, 1, 1};
    tab.data = t == DType::F16 ? static_cast<const void*>(h.data()) : static_cast<const void*>(f.data());
    std::vector<float> want(ids.size() * size_t(d)), got(want.size());
    cpu::embed(tab, ids, want.data());
    be.embed(tab, ids, got.data());
    CHECK(std::memcmp(want.data(), got.data(), want.size() * sizeof(float)) == 0);
  }
}

// Debug/ASan builds abort in the DCHECK at the top of CpuBackend::embed before the throw is reached,
// so the release-mode range check is only testable under NDEBUG.
#ifdef NDEBUG
TEST_CASE("cpu backend embed throws on out-of-range token ids", "[cpu_backend]") {
  ThreadPool pool(1);
  cpu::CpuBackend be(pool);
  std::mt19937 g(12);
  const int64_t d = 16, vocab = 4;
  auto f = rnd(size_t(d * vocab), g);
  std::vector<uint16_t> h(f.size());
  for (size_t i = 0; i < f.size(); ++i) h[i] = fp32_to_fp16(f[i]);
  std::vector<float> out(size_t(2 * d));
  for (DType t : {DType::F16, DType::F32}) {
    Tensor tab;
    tab.type = t;
    tab.n_dims = 2;
    tab.ne = {d, vocab, 1, 1};
    tab.data = t == DType::F16 ? static_cast<const void*>(h.data()) : static_cast<const void*>(f.data());
    for (int32_t bad : {-1, 4}) {
      std::vector<int32_t> ids = {1, bad};
      CHECK_THROWS_WITH(be.embed(tab, ids, out.data()), Catch::Matchers::ContainsSubstring("out of range"));
    }
  }
}
#endif
