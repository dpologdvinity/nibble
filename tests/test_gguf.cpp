#include <cstdio>
#include <fstream>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "gguf_writer.h"
#include "nibble/error.h"

using namespace nibble;
using Catch::Matchers::ContainsSubstring;
using gguf::File;
using gguf::ValueType;

namespace {
void expect_error(const std::vector<uint8_t>& b) {
  REQUIRE_THROWS_AS(File::parse(b), nibble::Error);
}
std::vector<uint8_t> one_tensor(DType t, std::vector<int64_t> ne, size_t n = 384) {
  GgufWriter w;
  w.add_tensor("w", t, std::move(ne), std::vector<uint8_t>(n, 1));
  return w.build();
}
}  // namespace

TEST_CASE("gguf valid file", "[gguf]") {
  auto bytes = GgufWriter::sample();
  File f = File::parse(bytes);
  REQUIRE(f.version() == 3);
  REQUIRE(f.get_string("general.architecture") == "qwen2");
  REQUIRE(f.get_int("qwen2.block_count") == 24);
  REQUIRE(f.get_float("qwen2.rope.freq_base") == 10000.0);
  REQUIRE(f.get_array("tokenizer.ggml.tokens").str == std::vector<std::string>{"a", "bc"});
  REQUIRE(f.get_array("tokenizer.ggml.token_type").i == std::vector<int64_t>{1, 3, -2});
  REQUIRE(f.find("nope") == nullptr);
  REQUIRE(f.metadata().size() == 5);
  REQUIRE(f.metadata()[0].first == "general.architecture");
  REQUIRE(f.tensors().size() == 1);
  const auto* t = f.find_tensor("w");
  REQUIRE(t != nullptr);
  REQUIRE(t->tensor.ne == std::array<int64_t, 4>{64, 3, 1, 1});
  REQUIRE(t->tensor.n_dims == 2);
  REQUIRE(t->tensor.type == DType::F16);
  REQUIRE(t->tensor.rows() == 3);
  REQUIRE(t->tensor.row_bytes() == 128);
  REQUIRE(t->nbytes == 384);
  auto* p = static_cast<const uint8_t*>(t->tensor.data);
  REQUIRE(p + t->nbytes == bytes.data() + bytes.size());
  size_t data_start = static_cast<size_t>(p - bytes.data()) - t->offset;
  REQUIRE(data_start % 32 == 0);
  REQUIRE(f.find_tensor("zz") == nullptr);
}

TEST_CASE("gguf version 2 accepted", "[gguf]") {
  GgufWriter w;
  w.add_string("k", "v");
  REQUIRE(File::parse(w.build(2)).version() == 2);
}

TEST_CASE("gguf header rejects", "[gguf]") {
  auto b = GgufWriter::sample();
  SECTION("bad magic") { b[0] = 'X'; expect_error(b); }
  SECTION("version 1") { b[4] = 1; expect_error(b); }
  SECTION("version 4") { b[4] = 4; expect_error(b); }
}

TEST_CASE("gguf every truncation throws", "[gguf]") {
  auto b = GgufWriter::sample();
  for (size_t n = 0; n < b.size(); ++n) {
    std::vector<uint8_t> p(b.begin(), b.begin() + n);
    INFO("prefix " << n);
    REQUIRE_THROWS_AS(File::parse(p), nibble::Error);
  }
}

TEST_CASE("gguf hostile sizes", "[gguf]") {
  SECTION("huge string length") {
    std::vector<uint8_t> kv;
    GgufWriter::put_u64(kv, 1);
    kv.push_back('k');
    GgufWriter::put_u32(kv, 8);
    GgufWriter::put_u64(kv, 0xFFFFFFFFFFFFFFFFull);
    GgufWriter w;
    w.add_raw_kv(kv);
    expect_error(w.build());
  }
  SECTION("huge array count") {
    std::vector<uint8_t> kv;
    GgufWriter::put_u64(kv, 1);
    kv.push_back('k');
    GgufWriter::put_u32(kv, 9);
    GgufWriter::put_u32(kv, 4);
    GgufWriter::put_u64(kv, 1ull << 60);
    GgufWriter w;
    w.add_raw_kv(kv);
    expect_error(w.build());
  }
  SECTION("huge string array count") {
    std::vector<uint8_t> kv;
    GgufWriter::put_u64(kv, 1);
    kv.push_back('k');
    GgufWriter::put_u32(kv, 9);
    GgufWriter::put_u32(kv, 8);
    GgufWriter::put_u64(kv, 1ull << 40);
    GgufWriter w;
    w.add_raw_kv(kv);
    expect_error(w.build());
  }
  SECTION("nested array") {
    std::vector<uint8_t> kv;
    GgufWriter::put_u64(kv, 1);
    kv.push_back('k');
    GgufWriter::put_u32(kv, 9);
    GgufWriter::put_u32(kv, 9);
    GgufWriter::put_u64(kv, 0);
    GgufWriter w;
    w.add_raw_kv(kv);
    expect_error(w.build());
  }
  SECTION("u64 scalar above int64 max") {
    std::vector<uint8_t> kv;
    GgufWriter::put_u64(kv, 1);
    kv.push_back('k');
    GgufWriter::put_u32(kv, 10);
    GgufWriter::put_u64(kv, 1ull << 63);
    GgufWriter w;
    w.add_raw_kv(kv);
    expect_error(w.build());
  }
  SECTION("unknown value type") {
    std::vector<uint8_t> kv;
    GgufWriter::put_u64(kv, 1);
    kv.push_back('k');
    GgufWriter::put_u32(kv, 99);
    GgufWriter w;
    w.add_raw_kv(kv);
    expect_error(w.build());
  }
  SECTION("huge tensor count") {
    auto b = GgufWriter::sample();
    for (int i = 0; i < 8; ++i) b[8 + i] = 0xFF;
    expect_error(b);
  }
  SECTION("huge kv count") {
    auto b = GgufWriter::sample();
    for (int i = 0; i < 8; ++i) b[16 + i] = 0xFF;
    expect_error(b);
  }
}

TEST_CASE("gguf duplicates", "[gguf]") {
  SECTION("key") {
    GgufWriter w;
    w.add_string("k", "a");
    w.add_string("k", "b");
    REQUIRE_THROWS_WITH(File::parse(w.build()), ContainsSubstring("k"));
  }
  SECTION("tensor") {
    GgufWriter w;
    w.add_tensor("w", DType::F32, {4}, std::vector<uint8_t>(16));
    w.add_tensor("w", DType::F32, {4}, std::vector<uint8_t>(16));
    REQUIRE_THROWS_WITH(File::parse(w.build()), ContainsSubstring("w"));
  }
}

TEST_CASE("gguf alignment", "[gguf]") {
  SECTION("zero") {
    GgufWriter w;
    w.add_int("general.alignment", ValueType::U32, 0);
    expect_error(w.build());
  }
  SECTION("not power of two") {
    GgufWriter w;
    w.add_tensor("w", DType::F32, {4}, std::vector<uint8_t>(16));
    expect_error(w.build(3, 48));
  }
  SECTION("wrong type") {
    GgufWriter w;
    w.add_f32("general.alignment", 32.0f);
    expect_error(w.build());
  }
  SECTION("custom valid") {
    GgufWriter w;
    w.add_tensor("w", DType::F32, {4}, std::vector<uint8_t>(16));
    File f = File::parse(w.build(3, 64));
    REQUIRE(f.get_int("general.alignment") == 64);
  }
}

TEST_CASE("gguf tensor shape rejects", "[gguf]") {
  SECTION("n_dims 0") { expect_error(one_tensor(DType::F32, {}, 4)); }
  SECTION("n_dims 5") { expect_error(one_tensor(DType::F32, {1, 1, 1, 1, 1}, 4)); }
  SECTION("zero dim") { expect_error(one_tensor(DType::F32, {4, 0}, 16)); }
  SECTION("Q4_0 ne0 not block multiple") { expect_error(one_tensor(DType::Q4_0, {33}, 100)); }
  SECTION("Q4_0 ok") {
    auto b = one_tensor(DType::Q4_0, {64, 2}, 72);
    REQUIRE(File::parse(b).find_tensor("w")->nbytes == 72);
  }
  SECTION("element overflow") { expect_error(one_tensor(DType::F32, {1ll << 62, 4}, 16)); }
  SECTION("dim above int64") { expect_error(one_tensor(DType::F32, {-1}, 16)); }
  SECTION("row bytes wrap 2^62") { expect_error(one_tensor(DType::F32, {1ll << 62}, 16)); }
  SECTION("row bytes wrap 2^62+1") { expect_error(one_tensor(DType::F32, {(1ll << 62) + 1}, 16)); }
  SECTION("byte overflow") { expect_error(one_tensor(DType::F32, {1ll << 61}, 16)); }
}

TEST_CASE("gguf tensor placement rejects", "[gguf]") {
  SECTION("offset misaligned") {
    GgufWriter w;
    w.add_tensor("w", DType::F32, {4}, std::vector<uint8_t>(16));
    w.set_last_tensor_offset(8);
    expect_error(w.build());
  }
  SECTION("1 byte past end") { expect_error(one_tensor(DType::F16, {64, 3}, 383)); }
  SECTION("offset past end") {
    GgufWriter w;
    w.add_tensor("w", DType::F32, {4}, std::vector<uint8_t>(16));
    w.set_last_tensor_offset(0xFFFFFFFFFFFFFFE0ull);
    expect_error(w.build());
  }
}

TEST_CASE("gguf unsupported tensor type names the type", "[gguf]") {
  auto b = one_tensor(static_cast<DType>(12), {256}, 144);
  REQUIRE_THROWS_WITH(File::parse(b), ContainsSubstring("Q4_K") && ContainsSubstring("w") &&
                                         ContainsSubstring("llama-quantize"));
}

TEST_CASE("gguf typed getters", "[gguf]") {
  auto b = GgufWriter::sample();
  File f = File::parse(b);
  REQUIRE_THROWS_WITH(f.get_string("missing"), ContainsSubstring("missing"));
  REQUIRE_THROWS_AS(f.get_int("general.architecture"), nibble::Error);
  REQUIRE_THROWS_AS(f.get_float("qwen2.block_count"), nibble::Error);
  REQUIRE_THROWS_AS(f.get_array("general.architecture"), nibble::Error);
}

TEST_CASE("gguf open from disk", "[gguf]") {
  auto b = GgufWriter::sample();
  std::string path = "test_gguf_open.tmp.gguf";
  {
    std::ofstream o(path, std::ios::binary);
    o.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
  }
  File f = File::open(path);
  File g = std::move(f);  // tensor pointers stay valid across moves
  REQUIRE(g.find_tensor("w") != nullptr);
  REQUIRE(static_cast<const uint8_t*>(g.find_tensor("w")->tensor.data)[383] == 0x5A);
  std::remove(path.c_str());
  REQUIRE_THROWS_WITH(File::open("/nonexistent/x.gguf"), ContainsSubstring("/nonexistent/x.gguf"));
  std::ofstream("test_gguf_empty.tmp.gguf").close();
  REQUIRE_THROWS_AS(File::open("test_gguf_empty.tmp.gguf"), nibble::Error);
  std::remove("test_gguf_empty.tmp.gguf");
}

TEST_CASE("gguf scalar bool", "[gguf]") {
  GgufWriter w;
  w.add_int("flag", ValueType::Bool, 1);
  w.add_int("flag0", ValueType::Bool, 0);
  auto b = w.build();
  File f = File::parse(b);
  REQUIRE(std::get<bool>(*f.find("flag")) == true);
  REQUIRE(std::get<bool>(*f.find("flag0")) == false);
  REQUIRE_THROWS_AS(f.get_int("flag"), nibble::Error);
}

TEST_CASE("gguf bool alignment rejected", "[gguf]") {
  GgufWriter w;
  w.add_int("general.alignment", ValueType::Bool, 1);
  expect_error(w.build());
}

TEST_CASE("gguf huge kv count within remaining/12 bound is capped", "[gguf]") {
  constexpr uint64_t kN = 2'000'000;  // claims 2M kvs; file is just zero bytes
  std::vector<uint8_t> b = {'G', 'G', 'U', 'F', 3, 0, 0, 0};
  auto put64 = [&](uint64_t v) { for (int i = 0; i < 8; ++i) b.push_back(static_cast<uint8_t>(v >> (8 * i))); };
  put64(0);
  put64(kN);
  b.resize(b.size() + kN * 12, 0);
  CHECK_THROWS_AS(File::parse(b), Error);
}
