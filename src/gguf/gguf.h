#pragma once
// Reader for GGUF model files (the llama.cpp container format, versions 2 and 3).
//
// A GGUF file is: a header (magic "GGUF", version, tensor count, metadata count), a list of key/value
// metadata entries, a list of tensor descriptors (name, shape, ggml type, offset), then the tensor bytes
// in a data section aligned to `general.alignment` (default 32). This parser validates all of that
// *before* handing anything out, because model files are untrusted input: counts and sizes are checked
// against the file length with overflow-safe arithmetic, so a hostile file can neither crash the process
// nor make it allocate more than a small multiple of the file size. Everything wrong throws
// nibble::Error with a message naming the key or tensor involved.
//
// The result is zero-copy. Tensor::data pointers and metadata array views point straight into the
// (memory-mapped) file bytes, and numeric arrays are never decoded into containers: a tokenizer
// vocabulary of 150k strings costs 8 bytes of bookkeeping per string. See File for lifetime rules.
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "gguf/mapped_file.h"
#include "tensor.h"

namespace nibble::gguf {

// GGUF metadata value type ids, as written in the file (also the element type of arrays).
enum class ValueType : uint32_t {
  U8 = 0,
  I8 = 1,
  U16 = 2,
  I16 = 3,
  U32 = 4,
  I32 = 5,
  F32 = 6,
  Bool = 7,
  String = 8,
  Array = 9,
  U64 = 10,
  I64 = 11,
  F64 = 12
};

namespace detail {
struct ArrayAccess;  // parser-side constructor access: only the parser may fill in an Array
}

// A metadata array as a view into the file bytes; elements decode on demand,
// so parsing allocates O(1) per numeric array and 8 bytes per string element.
// Valid while the owning File lives (File::open owns the mapping; for
// File::parse the caller's bytes must outlive the File). Nested arrays rejected.
class Array {
 public:
  ValueType elem_type() const { return elem_type_; }  // element type as stored in the file
  size_t size() const { return count_; }              // number of elements
  bool empty() const { return count_ == 0; }
  // Classification of the element type: exactly one of is_string / is_float / is_int holds.
  bool is_string() const { return elem_type_ == ValueType::String; }
  bool is_float() const { return elem_type_ == ValueType::F32 || elem_type_ == ValueType::F64; }
  bool is_int() const { return !is_string() && !is_float(); }  // Bool and all integer types

  // Element accessors. Precondition: i < size() and the matching is_*() is true; violations throw Error.
  int64_t int_at(size_t i) const;           // any integer type or Bool (as 0/1), widened to int64
  double float_at(size_t i) const;          // F32 or F64
  std::string_view str_at(size_t i) const;  // a view into the file bytes; not NUL-terminated

 private:
  friend struct detail::ArrayAccess;
  ValueType elem_type_ = ValueType::U8;
  size_t count_ = 0;
  std::span<const uint8_t> data_;  // numeric: count * element size bytes
  std::vector<uint64_t> str_off_;  // string: offset of each element's length prefix in data_
};

// A metadata value. Integer scalars of every width become int64_t (a U64 above INT64_MAX is rejected
// while parsing), F32/F64 become double, Bool stays bool.
using Value = std::variant<int64_t, double, bool, std::string, Array>;

// One tensor descriptor from the file, already validated: supported type, ne[0] a multiple of the type's
// block size, byte size and offset overflow-checked and inside the file.
struct TensorInfo {
  std::string name;
  Tensor tensor;    // shape and type; tensor.data points at the bytes inside the file
  uint64_t offset;  // relative to data section
  uint64_t nbytes;  // size of the tensor's data in bytes
};

// A parsed GGUF file: metadata plus tensor descriptors. Move-only in practice (it may own a mapping).
class File {
 public:
  // Both throw Error on malformed input, including "out of memory ..." if an
  // allocation fails. Tensor data pointers and Array views point into the bytes;
  // moving a File keeps them valid (the mapping address does not change).
  static File open(const std::string& path);          // mmap; owns mapping
  static File parse(std::span<const uint8_t> bytes);  // non-owning; bytes must outlive File

  uint32_t version() const { return version_; }   // GGUF format version, 2 or 3
  const Value* find(std::string_view key) const;  // nullptr if absent
  // Typed lookups: throw Error naming the key if it is absent or has a different value type.
  std::string get_string(std::string_view key) const;
  int64_t get_int(std::string_view key) const;
  double get_float(std::string_view key) const;
  const Array& get_array(std::string_view key) const;
  // All metadata in file order (used by `nibble info`).
  const std::vector<std::pair<std::string, Value>>& metadata() const { return metadata_; }
  // All tensor descriptors in file order.
  const std::vector<TensorInfo>& tensors() const { return tensors_; }
  const TensorInfo* find_tensor(std::string_view name) const;  // nullptr if absent

 private:
  // Hash for string_view lookups into a map keyed by std::string without building a temporary string.
  struct SvHash {
    using is_transparent = void;
    size_t operator()(std::string_view s) const { return std::hash<std::string_view>{}(s); }
  };
  using Index = std::unordered_map<std::string, size_t, SvHash, std::equal_to<>>;

  static File parse_impl(std::span<const uint8_t> bytes);  // lets std::bad_alloc through

  MappedFile map_;  // empty for File::parse; keeps the mapping alive for File::open
  uint32_t version_ = 0;
  std::vector<std::pair<std::string, Value>> metadata_;
  std::vector<TensorInfo> tensors_;
  Index kv_index_, tensor_index_;
};

}  // namespace nibble::gguf
