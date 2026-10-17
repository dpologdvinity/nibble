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

#include "gguf/gguf.h"
#include "gguf_writer.h"
#include "nibble/nibble.h"
#include "tiny_model.h"

namespace {
std::atomic<bool> g_armed{false};
std::atomic<int64_t> g_count{0};    // allocations while armed
std::atomic<int64_t> g_bytes{0};    // bytes requested while armed
std::atomic<int64_t> g_fail_at{0};  // 1-based index of the armed allocation to fail; 0 = never

void* counted_alloc(std::size_t n) {
  if (g_armed.load(std::memory_order_relaxed)) {
    const int64_t c = g_count.fetch_add(1) + 1;
    if (c == g_fail_at.load()) throw std::bad_alloc();
    g_bytes.fetch_add(static_cast<int64_t>(n));
  }
  if (void* p = std::malloc(n ? n : 1)) return p;
  throw std::bad_alloc();
}

struct Armed {
  explicit Armed(int64_t fail_at = 0) {
    g_count = 0;
    g_bytes = 0;
    g_fail_at = fail_at;
    g_armed = true;
  }
  ~Armed() { g_armed = false; }
};

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
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { std::free(p); }

using nibble::gguf::File;
using Catch::Matchers::ContainsSubstring;

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
