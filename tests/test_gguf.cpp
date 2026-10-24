// Tests for the GGUF parser (src/gguf/gguf.cpp). The valid-file cases check that metadata, arrays and tensors
// decode correctly; most of the rest is adversarial: every malformed or hostile file must fail with
// nibble::Error (never crash, hang or over-allocate), because model files are untrusted input. Files are
// built in memory with tests/gguf_writer.h.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "gguf_writer.h"
#include "nibble/error.h"

using namespace nibble;
using Catch::Matchers::ContainsSubstring;
using gguf::File;
using gguf::ValueType;

namespace {
// Requires that parsing the bytes throws nibble::Error.
void expect_error(const std::vector<uint8_t>& b) {
  REQUIRE_THROWS_AS(File::parse(b), nibble::Error);
}
// A GGUF containing one tensor "w" of the given type and shape whose data is n bytes of 0x01 (n defaults to
// 384).
std::vector<uint8_t> one_tensor(DType t, std::vector<int64_t> ne, size_t n = 384) {
  GgufWriter w;
  w.add_tensor("w", t, std::move(ne), std::vector<uint8_t>(n, 1));
  return w.build();
}
}  // namespace

// Round trip of a sample file: header version, typed getters, string and integer arrays, tensor
// shape/type/size, and that tensor data pointers point into the file bytes at an address aligned to 32.
TEST_CASE("gguf valid file", "[gguf]") {
  auto bytes = GgufWriter::sample();
  File f = File::parse(bytes);
  REQUIRE(f.version() == 3);
  REQUIRE(f.get_string("general.architecture") == "qwen2");
  REQUIRE(f.get_int("qwen2.block_count") == 24);
  REQUIRE(f.get_float("qwen2.rope.freq_base") == 10000.0);
  const auto& toks = f.get_array("tokenizer.ggml.tokens");
  REQUIRE(toks.is_string());
  REQUIRE(toks.size() == 2);
  REQUIRE(toks.str_at(0) == "a");
  REQUIRE(toks.str_at(1) == "bc");
  REQUIRE_THROWS_AS(toks.str_at(2), nibble::Error);
  REQUIRE_THROWS_AS(toks.int_at(0), nibble::Error);
  const auto& types = f.get_array("tokenizer.ggml.token_type");
  REQUIRE(types.elem_type() == nibble::gguf::ValueType::I32);
  REQUIRE(types.size() == 3);
  REQUIRE(types.int_at(0) == 1);
  REQUIRE(types.int_at(1) == 3);
  REQUIRE(types.int_at(2) == -2);
  REQUIRE_THROWS_AS(types.float_at(0), nibble::Error);
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

// GGUF version 2 files are accepted as well as version 3.
TEST_CASE("gguf version 2 accepted", "[gguf]") {
  GgufWriter w;
  w.add_string("k", "v");
  REQUIRE(File::parse(w.build(2)).version() == 2);
}

// A wrong magic number or unsupported version (1, 4) is rejected.
TEST_CASE("gguf header rejects", "[gguf]") {
  auto b = GgufWriter::sample();
  SECTION("bad magic") { b[0] = 'X'; expect_error(b); }
  SECTION("version 1") { b[4] = 1; expect_error(b); }
  SECTION("version 4") { b[4] = 4; expect_error(b); }
}

// Every proper prefix of a valid file is rejected: truncation anywhere is detected.
TEST_CASE("gguf every truncation throws", "[gguf]") {
  auto b = GgufWriter::sample();
  for (size_t n = 0; n < b.size(); ++n) {
    std::vector<uint8_t> p(b.begin(), b.begin() + n);
    INFO("prefix " << n);
    REQUIRE_THROWS_AS(File::parse(p), nibble::Error);
  }
}

// Header and value fields claiming absurd sizes (string lengths, array counts, nested arrays, u64 values
// above int64, unknown value types, tensor/kv counts larger than the file) fail fast without allocating.
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

// Duplicate metadata keys or tensor names are rejected.
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

// `general.alignment` must be an integer power of two of at least 8; zero, other values and non-integers are
// rejected, and a valid custom alignment moves the data section accordingly.
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
  SECTION("below 8 rejected, naming the value") {
    for (uint32_t a : {1u, 2u, 4u}) {
      GgufWriter w;
      w.add_tensor("w", DType::F32, {4}, std::vector<uint8_t>(16));
      REQUIRE_THROWS_WITH(File::parse(w.build(3, a)), ContainsSubstring("general.alignment " + std::to_string(a)));
    }
  }
  SECTION("8 accepted") {
    GgufWriter w;
    w.add_tensor("w", DType::F32, {4}, std::vector<uint8_t>(16));
    w.add_tensor("v", DType::F32, {2}, std::vector<uint8_t>(8));
    auto b = w.build(3, 8);
    File f = File::parse(b);
    REQUIRE(f.get_int("general.alignment") == 8);
    REQUIRE(f.find_tensor("v")->offset == 16);
  }
  SECTION("custom valid") {
    GgufWriter w;
    w.add_tensor("w", DType::F32, {4}, std::vector<uint8_t>(16));
    File f = File::parse(w.build(3, 64));
    REQUIRE(f.get_int("general.alignment") == 64);
  }
}

// Invalid dimensions: 0 or 5 dims, zero-sized dimension, ne[0] not a multiple of the block size, and element
// or byte counts that overflow 64 bits.
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

// A tensor offset that is misaligned, or data that extends even one byte past the end of the file, is
// rejected.
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

// Unsupported tensor types (Q3_K, Q4_1, Q8_K, BF16, unknown ids) are rejected with a message naming the type,
// its id, the tensor and the supported set.
TEST_CASE("gguf unsupported tensor type names the type", "[gguf]") {
  auto b = one_tensor(DType(11), {256}, 110);
  REQUIRE_THROWS_WITH(File::parse(b), ContainsSubstring("Q3_K") && ContainsSubstring("(11)") && ContainsSubstring("w") &&
                                         ContainsSubstring(kSupportedTypes));
  struct Case { uint32_t id; const char* name; };
  for (Case c : {Case{3, "Q4_1"}, Case{15, "Q8_K"}, Case{30, "BF16"}, Case{20, "unknown"}}) {
    CAPTURE(c.id);
    REQUIRE_THROWS_WITH(File::parse(one_tensor(DType(c.id), {256}, 1024)),
                        ContainsSubstring(c.name) && ContainsSubstring("(" + std::to_string(c.id) + ")"));
  }
}

// Q5_0, Q5_1, Q4_K, Q5_K and Q6_K tensors load with the right row and tensor byte sizes.
TEST_CASE("gguf accepts the new types", "[gguf]") {
  struct Case { DType t; int64_t bs, bb; };
  for (Case c : {Case{DType::Q5_0, 32, 22}, Case{DType::Q5_1, 32, 24}, Case{DType::Q4_K, 256, 144},
                 Case{DType::Q5_K, 256, 176}, Case{DType::Q6_K, 256, 210}}) {
    CAPTURE(dtype_info(c.t).name);
    const int64_t blocks = 2 * 256 / c.bs;  // 512 elements per row
    auto b = one_tensor(c.t, {512, 3}, static_cast<size_t>(blocks * c.bb * 3));
    File f = File::parse(b);
    const auto* ti = f.find_tensor("w");
    REQUIRE(ti != nullptr);
    REQUIRE(ti->tensor.type == c.t);
    REQUIRE(ti->tensor.row_bytes() == blocks * c.bb);
    REQUIRE(ti->nbytes == static_cast<uint64_t>(blocks * c.bb * 3));
  }
}

// A k-quant (or Q5) tensor whose row length is not a whole number of blocks is rejected with the tensor name,
// the width and the block size.
TEST_CASE("k-quant ne0 not a multiple of 256 is rejected", "[gguf]") {
  REQUIRE_THROWS_WITH(File::parse(one_tensor(DType::Q4_K, {896, 2}, 1000)),
                      ContainsSubstring("'w'") && ContainsSubstring("ne[0]=896") && ContainsSubstring("Q4_K") &&
                          ContainsSubstring("256"));
  REQUIRE_THROWS_WITH(File::parse(one_tensor(DType::Q5_0, {48}, 100)),
                      ContainsSubstring("'w'") && ContainsSubstring("ne[0]=48") && ContainsSubstring("Q5_0") &&
                          ContainsSubstring("32"));
}

// Missing keys and wrong-type lookups throw Error.
TEST_CASE("gguf typed getters", "[gguf]") {
  auto b = GgufWriter::sample();
  File f = File::parse(b);
  REQUIRE_THROWS_WITH(f.get_string("missing"), ContainsSubstring("missing"));
  REQUIRE_THROWS_AS(f.get_int("general.architecture"), nibble::Error);
  REQUIRE_THROWS_AS(f.get_float("qwen2.block_count"), nibble::Error);
  REQUIRE_THROWS_AS(f.get_array("general.architecture"), nibble::Error);
}

// File::open maps a real file; moving the File keeps array views and tensor pointers valid; missing and empty
// files are rejected with a usable message.
TEST_CASE("gguf open from disk", "[gguf]") {
  auto b = GgufWriter::sample();
  std::string path = "test_gguf_open.tmp.gguf";
  {
    std::ofstream o(path, std::ios::binary);
    o.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
  }
  File f = File::open(path);
  const char* tok_before = f.get_array("tokenizer.ggml.tokens").str_at(1).data();
  File g = std::move(f);  // tensor pointers and array views stay valid across moves
  REQUIRE(g.get_array("tokenizer.ggml.tokens").str_at(1).data() == tok_before);
  REQUIRE(g.get_array("tokenizer.ggml.tokens").str_at(1) == "bc");
  REQUIRE(g.find_tensor("w") != nullptr);
  REQUIRE(static_cast<const uint8_t*>(g.find_tensor("w")->tensor.data)[383] == 0x5A);
  std::remove(path.c_str());
  REQUIRE_THROWS_WITH(File::open("/nonexistent/x.gguf"), ContainsSubstring("/nonexistent/x.gguf"));
  std::ofstream("test_gguf_empty.tmp.gguf").close();
  REQUIRE_THROWS_AS(File::open("test_gguf_empty.tmp.gguf"), nibble::Error);
  std::remove("test_gguf_empty.tmp.gguf");
}

// Bool metadata decodes to bool and is not an integer for get_int.
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

// general.alignment given as a Bool is rejected.
TEST_CASE("gguf bool alignment rejected", "[gguf]") {
  GgufWriter w;
  w.add_int("general.alignment", ValueType::Bool, 1);
  expect_error(w.build());
}

// A header claiming 2 million entries in a file of zeros fails cleanly; the parser reserves at most a few
// thousand entries up front.
TEST_CASE("gguf huge kv count within remaining/12 bound is capped", "[gguf]") {
  constexpr uint64_t kN = 2'000'000;  // claims 2M kvs; file is just zero bytes
  std::vector<uint8_t> b = {'G', 'G', 'U', 'F', 3, 0, 0, 0};
  auto put64 = [&](uint64_t v) { for (int i = 0; i < 8; ++i) b.push_back(static_cast<uint8_t>(v >> (8 * i))); };
  put64(0);
  put64(kN);
  b.resize(b.size() + kN * 12, 0);
  CHECK_THROWS_AS(File::parse(b), Error);
}

namespace {
template <class T>
// A raw array key/value entry of the given element type, for building arrays by hand.
std::vector<uint8_t> raw_array_kv(const std::string& key, ValueType t, std::vector<T> v) {
  std::vector<uint8_t> b;
  GgufWriter::put_str(b, key);
  GgufWriter::put_u32(b, static_cast<uint32_t>(ValueType::Array));
  GgufWriter::put_u32(b, static_cast<uint32_t>(t));
  GgufWriter::put_u64(b, v.size());
  for (T x : v) {
    const size_t n = b.size();
    b.resize(n + sizeof x);
    std::memcpy(b.data() + n, &x, sizeof x);
  }
  return b;
}
}  // namespace

// int_at/float_at/str_at return the right value for each numeric element type, including negative and
// boundary values.
TEST_CASE("gguf array elements decode on demand for every type", "[gguf]") {
  GgufWriter w;
  w.add_raw_kv(raw_array_kv<uint8_t>("u8", ValueType::U8, {0, 255}));
  w.add_raw_kv(raw_array_kv<int8_t>("i8", ValueType::I8, {-128, 127}));
  w.add_raw_kv(raw_array_kv<uint16_t>("u16", ValueType::U16, {1, 65535}));
  w.add_raw_kv(raw_array_kv<int16_t>("i16", ValueType::I16, {-32768, 5}));
  w.add_raw_kv(raw_array_kv<uint32_t>("u32", ValueType::U32, {7, 4294967295u}));
  w.add_raw_kv(raw_array_kv<int32_t>("i32", ValueType::I32, {-1, 2}));
  w.add_raw_kv(raw_array_kv<uint64_t>("u64", ValueType::U64, {3, uint64_t{INT64_MAX}}));
  w.add_raw_kv(raw_array_kv<int64_t>("i64", ValueType::I64, {INT64_MIN, 9}));
  w.add_raw_kv(raw_array_kv<uint8_t>("bool", ValueType::Bool, {0, 2}));
  w.add_raw_kv(raw_array_kv<float>("f32", ValueType::F32, {1.5f, -2.0f}));
  w.add_raw_kv(raw_array_kv<double>("f64", ValueType::F64, {0.25, 1e300}));
  w.add_raw_kv(raw_array_kv<uint8_t>("empty", ValueType::U8, {}));
  auto b = w.build();
  File f = File::parse(b);
  auto ints = [&](const char* k) {
    const auto& a = f.get_array(k);
    REQUIRE(a.is_int());
    REQUIRE(a.size() == 2);
    return std::pair{a.int_at(0), a.int_at(1)};
  };
  CHECK(ints("u8") == std::pair<int64_t, int64_t>{0, 255});
  CHECK(ints("i8") == std::pair<int64_t, int64_t>{-128, 127});
  CHECK(ints("u16") == std::pair<int64_t, int64_t>{1, 65535});
  CHECK(ints("i16") == std::pair<int64_t, int64_t>{-32768, 5});
  CHECK(ints("u32") == std::pair<int64_t, int64_t>{7, 4294967295});
  CHECK(ints("i32") == std::pair<int64_t, int64_t>{-1, 2});
  CHECK(ints("u64") == std::pair<int64_t, int64_t>{3, INT64_MAX});
  CHECK(ints("i64") == std::pair<int64_t, int64_t>{INT64_MIN, 9});
  CHECK(ints("bool") == std::pair<int64_t, int64_t>{0, 1});
  const auto& f32 = f.get_array("f32");
  REQUIRE(f32.is_float());
  CHECK(f32.float_at(0) == 1.5);
  CHECK(f32.float_at(1) == -2.0);
  CHECK(f.get_array("f64").float_at(1) == 1e300);
  CHECK_THROWS_AS(f32.int_at(0), Error);
  CHECK_THROWS_AS(f32.float_at(2), Error);
  CHECK(f.get_array("empty").empty());
  CHECK_THROWS_AS(f.get_array("empty").int_at(0), Error);
}

// An unsigned 64-bit array element above INT64_MAX cannot be returned as int64 and is rejected at parse time.
TEST_CASE("gguf u64 array element above INT64_MAX rejected", "[gguf]") {
  GgufWriter w;
  w.add_raw_kv(raw_array_kv<uint64_t>("big", ValueType::U64, {1, uint64_t{INT64_MAX} + 1}));
  REQUIRE_THROWS_WITH(File::parse(w.build()), ContainsSubstring("9223372036854775808"));
}

// Terminal escape bytes inside tensor names or keys never appear raw in error messages.
TEST_CASE("gguf error messages escape control bytes", "[gguf]") {
  auto check = [](const std::vector<uint8_t>& b) {
    try {
      (void)File::parse(b);
      FAIL("expected Error");
    } catch (const nibble::Error& e) {
      const std::string msg = e.what();
      CHECK(msg.find("\\x1b") != std::string::npos);
      CHECK(msg.find('\x1b') == std::string::npos);
    }
  };
  SECTION("tensor name") {
    GgufWriter w;
    w.add_tensor("w\x1b[31m", DType::F32, {}, std::vector<uint8_t>(4, 1));  // n_dims 0
    check(w.build());
  }
  SECTION("key") {
    GgufWriter w;
    w.add_string("k\x1b[0m", "a");
    w.add_string("k\x1b[0m", "b");  // duplicate
    check(w.build());
  }
}
