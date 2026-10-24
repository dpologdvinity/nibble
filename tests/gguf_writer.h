#pragma once
// Test-only builder for GGUF files (valid and deliberately malformed).
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "gguf/gguf.h"
#include "tensor.h"

// Builds GGUF files in memory: well-formed ones for the positive tests, and, through the raw helpers,
// deliberately malformed ones for the hostile-input tests. Typical use: add metadata and tensors, then
// build().
class GgufWriter {
 public:
  using ValueType = nibble::gguf::ValueType;

  // Little-endian primitives, public so tests can hand-encode malformed key/value entries.
  static void put_u32(std::vector<uint8_t>& b, uint32_t v) { put(b, v); }
  static void put_u64(std::vector<uint8_t>& b, uint64_t v) { put(b, v); }
  static void put_str(std::vector<uint8_t>& b, const std::string& s) {
    put_u64(b, s.size());
    b.insert(b.end(), s.begin(), s.end());
  }

  // Adds an integer-valued metadata entry of the given GGUF type (the value is truncated to the type's
  // width).
  void add_int(const std::string& key, ValueType t, int64_t v) {
    auto b = head(key, t);
    switch (t) {
      case ValueType::U8: case ValueType::I8: case ValueType::Bool: put(b, static_cast<uint8_t>(v)); break;
      case ValueType::U16: case ValueType::I16: put(b, static_cast<uint16_t>(v)); break;
      case ValueType::U32: case ValueType::I32: put(b, static_cast<uint32_t>(v)); break;
      default: put(b, static_cast<uint64_t>(v)); break;
    }
    kvs_.push_back(std::move(b));
  }
  // Adds a 32-bit float entry.
  void add_f32(const std::string& key, float v) {
    auto b = head(key, ValueType::F32);
    put(b, v);
    kvs_.push_back(std::move(b));
  }
  // Adds a string entry.
  void add_string(const std::string& key, std::string v) {
    auto b = head(key, ValueType::String);
    put_str(b, v);
    kvs_.push_back(std::move(b));
  }
  // Adds an array of strings.
  void add_string_array(const std::string& key, std::vector<std::string> v) {
    auto b = head(key, ValueType::Array);
    put_u32(b, static_cast<uint32_t>(ValueType::String));
    put_u64(b, v.size());
    for (auto& s : v) put_str(b, s);
    kvs_.push_back(std::move(b));
  }
  // Adds an array of int32.
  void add_i32_array(const std::string& key, std::vector<int32_t> v) {
    auto b = head(key, ValueType::Array);
    put_u32(b, static_cast<uint32_t>(ValueType::I32));
    put_u64(b, v.size());
    for (auto x : v) put(b, x);
    kvs_.push_back(std::move(b));
  }
  // Adds an array of float32.
  void add_f32_array(const std::string& key, std::vector<float> v) {
    auto b = head(key, ValueType::Array);
    put_u32(b, static_cast<uint32_t>(ValueType::F32));
    put_u64(b, v.size());
    for (auto x : v) put(b, x);
    kvs_.push_back(std::move(b));
  }
  // dtype may be any value (cast) to build files with unsupported types.
  void add_tensor(std::string name, nibble::DType t, std::vector<int64_t> ne, std::vector<uint8_t> data) {
    tensors_.push_back({std::move(name), static_cast<uint32_t>(t), std::move(ne), std::move(data), 0, false});
  }
  // Pre-encoded key/value bytes appended verbatim (for malformed-input tests).
  void add_raw_kv(std::vector<uint8_t> encoded) { kvs_.push_back(std::move(encoded)); }
  // Force the stored offset of the most recently added tensor.
  void set_last_tensor_offset(uint64_t off) {
    tensors_.back().offset = off;
    tensors_.back().forced = true;
  }

  // Serializes the file: header, metadata, tensor descriptors (offsets assigned in order unless forced),
  // padding to the alignment, then the tensor data. A non-default alignment is recorded as general.alignment.
  std::vector<uint8_t> build(uint32_t version = 3, uint32_t alignment = 32) const {
    std::vector<std::vector<uint8_t>> kvs = kvs_;
    if (alignment != 32) {
      GgufWriter w;
      w.add_int("general.alignment", ValueType::U32, alignment);
      kvs.push_back(w.kvs_[0]);
    }
    std::vector<uint8_t> b = {'G', 'G', 'U', 'F'};
    put_u32(b, version);
    put_u64(b, tensors_.size());
    put_u64(b, kvs.size());
    for (auto& kv : kvs) b.insert(b.end(), kv.begin(), kv.end());
    uint64_t next = 0;
    std::vector<uint64_t> offs;
    for (auto& t : tensors_) {
      uint64_t off = t.forced ? t.offset : next;
      offs.push_back(off);
      put_str(b, t.name);
      put_u32(b, static_cast<uint32_t>(t.ne.size()));
      for (auto d : t.ne) put_u64(b, static_cast<uint64_t>(d));
      put_u32(b, t.type);
      put_u64(b, off);
      next = align_up(next + t.data.size(), alignment ? alignment : 1);
    }
    b.resize(align_up(b.size(), alignment ? alignment : 1), 0);
    size_t base = b.size();
    for (size_t i = 0; i < tensors_.size(); ++i) {
      if (tensors_[i].forced) continue;
      size_t at = base + offs[i];
      if (b.size() < at) b.resize(at, 0);
      b.insert(b.end(), tensors_[i].data.begin(), tensors_[i].data.end());
    }
    return b;
  }

  // The valid file used by test_gguf and the fuzz seed.
  // The small valid file shared by test_gguf and the fuzz seed.
  static std::vector<uint8_t> sample() {
    GgufWriter w;
    w.add_string("general.architecture", "qwen2");
    w.add_int("qwen2.block_count", ValueType::U32, 24);
    w.add_f32("qwen2.rope.freq_base", 10000.0f);
    w.add_string_array("tokenizer.ggml.tokens", {"a", "bc"});
    w.add_i32_array("tokenizer.ggml.token_type", {1, 3, -2});
    w.add_tensor("w", nibble::DType::F16, {64, 3}, std::vector<uint8_t>(384, 0x5A));
    return w.build();
  }

 private:
  // One queued tensor: name, raw ggml type id, dimensions, data, and an optional forced offset.
  struct T {
    std::string name;
    uint32_t type;
    std::vector<int64_t> ne;
    std::vector<uint8_t> data;
    uint64_t offset;
    bool forced;
  };
  template <class V>
  // Appends the raw bytes of a scalar.
  static void put(std::vector<uint8_t>& b, V v) {
    // resize+memcpy (not insert) sidesteps a GCC 13 -Wstringop-overflow false positive at -O2.
    const size_t n = b.size();
    b.resize(n + sizeof v);
    std::memcpy(b.data() + n, &v, sizeof v);
  }
  // Round up to a multiple of a.
  static uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }
  // Key string plus value-type id: the start of every metadata entry.
  static std::vector<uint8_t> head(const std::string& key, ValueType t) {
    std::vector<uint8_t> b;
    put_str(b, key);
    put_u32(b, static_cast<uint32_t>(t));
    return b;
  }

  std::vector<std::vector<uint8_t>> kvs_;
  std::vector<T> tensors_;
};
