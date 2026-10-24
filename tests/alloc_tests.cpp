// Allocation-accounting tests. Built as a separate executable because it replaces
// the global operator new/delete with a counting hook.
//
// Proxy for "the parser's own allocation": every byte requested from operator new
// (all std containers/strings the parser uses go through it) while the hook is armed.
// The cumulative total bounds the peak from above, is deterministic, and works under
// ASan/UBSan (unlike RLIMIT_AS, which ASan's shadow reservation defeats).
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <new>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "backend/cpu/cpu_backend.h"
#include "gguf/gguf.h"
#include "model/transformer.h"
#include "thread_pool.h"
#include "gguf_writer.h"
#include "nibble/nibble.h"
#include "sampler.h"
#include "tiny_model.h"

namespace {
// Hook state: while armed, every operator-new call is counted and its bytes summed; g_fail_at makes the N-th
// armed allocation throw std::bad_alloc, for out-of-memory injection.
std::atomic<bool> g_armed{false};
std::atomic<int64_t> g_count{0};    // allocations while armed
std::atomic<int64_t> g_bytes{0};    // bytes requested while armed
std::atomic<int64_t> g_fail_at{0};  // 1-based index of the armed allocation to fail; 0 = never

// The common allocation path behind all replaced operator new overloads.
void* counted_alloc(std::size_t n, std::size_t align = 0) {
  if (g_armed.load(std::memory_order_relaxed)) {
    const int64_t c = g_count.fetch_add(1) + 1;
    if (c == g_fail_at.load()) throw std::bad_alloc();
    g_bytes.fetch_add(static_cast<int64_t>(n));
  }
  if (align) {
    if (void* p = std::aligned_alloc(align, (n + align - 1) / align * align)) return p;
  } else if (void* p = std::malloc(n ? n : 1)) {
    return p;
  }
  throw std::bad_alloc();
}

// RAII scope that zeroes the counters, optionally sets the failure point, and arms the hook; disarms on
// destruction.
struct Armed {
  explicit Armed(int64_t fail_at = 0) {
    g_count = 0;
    g_bytes = 0;
    g_fail_at = fail_at;
    g_armed = true;
  }
  ~Armed() { g_armed = false; }
};

// A GGUF with one huge u8 array of n elements (a metadata value that would be enormous if decoded eagerly).
std::vector<uint8_t> big_u8_array_gguf(uint64_t n) {
  std::vector<uint8_t> kv;
  GgufWriter::put_str(kv, "junk");
  GgufWriter::put_u32(kv, static_cast<uint32_t>(GgufWriter::ValueType::Array));
  GgufWriter::put_u32(kv, static_cast<uint32_t>(GgufWriter::ValueType::U8));
  GgufWriter::put_u64(kv, n);
  kv.resize(kv.size() + n, 1);
  GgufWriter w;
  w.add_string("general.architecture", "qwen2");
  w.add_raw_kv(std::move(kv));
  return w.build();
}
}  // namespace

void* operator new(std::size_t n) { return counted_alloc(n); }
void* operator new[](std::size_t n) { return counted_alloc(n); }
void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
  try {
    return counted_alloc(n);
  } catch (...) {
    return nullptr;
  }
}
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept {
  try {
    return counted_alloc(n);
  } catch (...) {
    return nullptr;
  }
}
void* operator new(std::size_t n, std::align_val_t a) { return counted_alloc(n, static_cast<std::size_t>(a)); }
void* operator new[](std::size_t n, std::align_val_t a) { return counted_alloc(n, static_cast<std::size_t>(a)); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { std::free(p); }

using nibble::gguf::File;
using Catch::Matchers::ContainsSubstring;

// Parsing a file with a 20 MB numeric array allocates only bookkeeping (< 64 KiB) because arrays are views
// into the file bytes.
TEST_CASE("gguf parse of a 20 MB u8 array allocates O(1), not O(n)", "[gguf][alloc]") {
  constexpr uint64_t kN = 20'000'000;
  const auto bytes = big_u8_array_gguf(kN);
  int64_t used = 0;
  {
    Armed a;
    File f = File::parse(bytes);
    used = g_bytes.load();
    REQUIRE(f.get_array("junk").size() == kN);
    REQUIRE(f.get_array("junk").int_at(kN - 1) == 1);
  }
  INFO("bytes allocated by parse: " << used);
  // Old design: >= 8 * kN (160 MB) of int64 elements. Now only bookkeeping.
  REQUIRE(used < 64 * 1024);
}

// A million-element string array costs at most 8 bytes of offset per element (the worst case is empty
// strings).
TEST_CASE("gguf parse of a large string array allocates <= 8 bytes per element", "[gguf][alloc]") {
  constexpr uint64_t kN = 1'000'000;  // empty strings: 8 file bytes each, the worst case
  std::vector<uint8_t> kv;
  GgufWriter::put_str(kv, "s");
  GgufWriter::put_u32(kv, static_cast<uint32_t>(GgufWriter::ValueType::Array));
  GgufWriter::put_u32(kv, static_cast<uint32_t>(GgufWriter::ValueType::String));
  GgufWriter::put_u64(kv, kN);
  kv.resize(kv.size() + 8 * kN, 0);
  GgufWriter w;
  w.add_raw_kv(std::move(kv));
  const auto bytes = w.build();
  int64_t used = 0;
  {
    Armed a;
    File f = File::parse(bytes);
    used = g_bytes.load();
    REQUIRE(f.get_array("s").str_at(kN - 1).empty());
  }
  INFO("bytes allocated by parse: " << used);
  REQUIRE(used <= static_cast<int64_t>(8 * kN) + 64 * 1024);
}

// Out-of-memory injection: failing each allocation in turn during parsing always yields an Error mentioning
// out of memory, never a crash or std::bad_alloc escape.
TEST_CASE("bad_alloc during File::parse surfaces as nibble::Error", "[gguf][alloc]") {
  const auto bytes = GgufWriter::sample();
  int failures = 0;
  for (int64_t k = 1;; ++k) {
    Armed a(k);
    try {
      File f = File::parse(bytes);
      if (g_count.load() < k) break;  // completed without reaching the injected failure
    } catch (const nibble::Error& e) {
      REQUIRE_THAT(e.what(), ContainsSubstring("out of memory"));
      ++failures;
    }
  }
  REQUIRE(failures > 0);
}

// The same injection over the whole Model::load, ending in the documented "out of memory loading <path>"
// error.
TEST_CASE("bad_alloc during Model::load surfaces as nibble::Error", "[api][alloc]") {
  const std::string path = "alloc_tests_tiny.tmp.gguf";
  {
    const auto b = tiny_model_gguf(1);
    std::ofstream o(path, std::ios::binary);
    o.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
  }
  int oom = 0, failures = 0;
  for (int64_t k = 1;; ++k) {
    bool done = false;
    try {
      Armed a(k);
      nibble::Model m = nibble::Model::load(path, {.n_ctx = 0, .n_threads = 1});
      done = g_count.load() < k;
    } catch (const nibble::Error& e) {
      ++failures;
      if (std::string(e.what()).find("out of memory loading " + path) != std::string::npos) ++oom;
    }
    if (done) break;
  }
  std::remove(path.c_str());
  INFO("injected failures: " << failures << ", reported as out of memory: " << oom);
  REQUIRE(failures > 0);
  REQUIRE(oom > 0);
}

// The sampler's buffers are allocated once at construction; 3000 draws allocate nothing.
TEST_CASE("Sampler::sample does not allocate", "[sampler][alloc]") {
  std::vector<float> l(1000);
  for (size_t i = 0; i < l.size(); ++i) l[i] = static_cast<float>((i * 37) % 101) * 0.1f;
  nibble::Sampler s({0.8f, 50, 0.9f, 5}, 1000);
  nibble::Sampler greedy({0.0f, 0, 1.0f, 5}, 1000);
  nibble::Sampler full({1.0f, 0, 1.0f, 5}, 1000);
  int64_t sink = 0, n = -1;
  {
    Armed armed;
    for (int i = 0; i < 1000; ++i) sink += s.sample(l) + greedy.sample(l) + full.sample(l);
    n = g_count.load();
  }
  REQUIRE(n == 0);
  REQUIRE(sink >= 0);
}

namespace {
// A parsed tiny model with its bound weights, as in test_model.cpp.
struct LoadedTiny {
  std::vector<uint8_t> bytes;
  nibble::gguf::File file;
  nibble::model::LlamaConfig cfg;
  nibble::model::Weights w;
  LoadedTiny(std::vector<uint8_t> b, const nibble::Backend& be)
      : bytes(std::move(b)), file(nibble::gguf::File::parse(bytes)),
        cfg(nibble::model::LlamaConfig::from_gguf(file)), w(nibble::model::Weights::bind(file, cfg, be)) {}
};
}  // namespace

// Mixed-type models: matmul_multi quantizes x once per activation type and dispatches each weight on its type.
// Decode on models with mixed weight types (Q4_K/Q6_K/Q5_0, and Q5_1/Q8_0 with its two activation types):
// after a warm-up prefill, 16 single-token forwards allocate nothing and do not grow any backend scratch.
TEST_CASE("alloc: mixed k-quant model", "[model][alloc]") {
  using nibble::DType;
  struct Mix {
    const char* name;
    DType base;
    DType override_type;
    bool kdims;
  };
  for (const Mix& mix : {Mix{"Q4_K+Q6_K+Q5_0", DType::Q4_K, DType::Q6_K, true}, Mix{"Q5_1+Q8_0", DType::Q5_1, DType::Q8_0, false}}) {
    DYNAMIC_SECTION(mix.name) {
      nibble::ThreadPool pool(4);
      nibble::cpu::CpuBackend be(pool);
      TinyModelOpts mo;
      (mix.kdims ? mo.kquant_dims : mo.quant_dims) = true;
      for (const char* p : {"blk.0.", "blk.1."}) {
        mo.tensor_types.emplace_back(std::string(p) + "attn_v.weight", mix.override_type);
        mo.tensor_types.emplace_back(std::string(p) + "ffn_down.weight", mix.override_type);
      }
      if (mix.kdims) mo.tensor_types.emplace_back("token_embd.weight", DType::Q5_0);
      LoadedTiny m(tiny_model_gguf(3, mix.base, false, 259, mo), be);
      nibble::model::Transformer tr(m.cfg, m.w, be, 64, 8);
      std::vector<float> logits(static_cast<size_t>(m.cfg.n_vocab));
      const std::vector<int32_t> prompt = {1, 50, 257, 3};
      tr.forward(prompt, 0, logits.data());
      const int32_t one[1] = {7};
      const size_t live = be.live_allocations(), sc = be.scratch_capacity(), xq = be.quant_scratch_capacity();
      int64_t n_fwd = -1;
      {
        Armed a;
        for (int i = 0; i < 16; ++i) tr.forward(one, 4 + i, logits.data());
        n_fwd = g_count.load();
      }
      CHECK(n_fwd == 0);
      CHECK(be.live_allocations() == live);
      CHECK(be.scratch_capacity() == sc);
      CHECK(be.quant_scratch_capacity() == xq);
    }
  }
}

// The central no-allocation guarantee for every float/quantized weight type: after warm-up, decoding, short
// and long prefill and a direct matmul perform zero heap allocations and leave the backend's scratch sizes
// unchanged.
TEST_CASE("Transformer::forward decode and CpuBackend::matmul do not allocate", "[model][alloc]") {
  using nibble::DType;
  for (DType t : {DType::F32, DType::F16, DType::Q8_0, DType::Q4_0}) {
    DYNAMIC_SECTION(nibble::dtype_info(t).name) {
      nibble::ThreadPool pool(4);
      nibble::cpu::CpuBackend be(pool);
      TinyModelOpts mo;
      mo.quant_dims = true;
      LoadedTiny m(tiny_model_gguf(3, t, false, 259, mo), be);
      nibble::model::Transformer tr(m.cfg, m.w, be, 64, 8);
      std::vector<float> logits(static_cast<size_t>(m.cfg.n_vocab));
      const std::vector<int32_t> prompt = {1, 50, 257, 3};
      tr.forward(prompt, 0, logits.data());
      float* x = be.alloc(8 * m.w.layers[0].w_up.ne[0]);
      float* y = be.alloc(m.w.layers[0].w_up.rows() * 8);
      const int32_t one[1] = {7};
      std::vector<int32_t> p8 = {1, 50, 257, 3, 9, 10, 11, 12};
      tr.forward(p8, 8, logits.data());  // warm-up prefill n=8
      struct Inv {
        size_t live, sc, xq;
      };
      auto inv = [&] { return Inv{be.live_allocations(), be.scratch_capacity(), be.quant_scratch_capacity()}; };
      const Inv before = inv();
      auto same = [&](const Inv& i) { return i.live == before.live && i.sc == before.sc && i.xq == before.xq; };
      int64_t n_fwd = -1, n_mm = -1, n_pre4 = -1, n_pre8 = -1;
      {
        Armed a;
        for (int i = 0; i < 16; ++i) tr.forward(one, 16 + i, logits.data());
        n_fwd = g_count.load();
      }
      CHECK(same(inv()));
      {
        Armed a;
        tr.forward(std::span(p8).first(4), 32, logits.data());
        n_pre4 = g_count.load();
      }
      CHECK(same(inv()));
      {
        Armed a;
        tr.forward(p8, 40, logits.data());
        n_pre8 = g_count.load();
      }
      CHECK(same(inv()));
      {
        Armed a;
        be.matmul(m.w.layers[0].w_up, x, y, 8);
        n_mm = g_count.load();
      }
      CHECK(same(inv()));
      be.free(x);
      be.free(y);
      CHECK(n_fwd == 0);
      CHECK(n_mm == 0);
      CHECK(n_pre4 == 0);
      CHECK(n_pre8 == 0);
    }
  }
}
