#pragma once
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

enum class ValueType : uint32_t {
  U8 = 0, I8 = 1, U16 = 2, I16 = 3, U32 = 4, I32 = 5, F32 = 6, Bool = 7,
  String = 8, Array = 9, U64 = 10, I64 = 11, F64 = 12
};

namespace detail {
struct ArrayAccess;  // parser-side constructor access
}

// A metadata array as a view into the file bytes; elements decode on demand,
// so parsing allocates O(1) per numeric array and 8 bytes per string element.
// Valid while the owning File lives (File::open owns the mapping; for
// File::parse the caller's bytes must outlive the File). Nested arrays rejected.
class Array {
 public:
  ValueType elem_type() const { return elem_type_; }
  size_t size() const { return count_; }
  bool empty() const { return count_ == 0; }
  bool is_string() const { return elem_type_ == ValueType::String; }
  bool is_float() const { return elem_type_ == ValueType::F32 || elem_type_ == ValueType::F64; }
  bool is_int() const { return !is_string() && !is_float(); }  // Bool and all integer types

  // Precondition: i < size() and the matching is_*() is true; violations throw Error.
  int64_t int_at(size_t i) const;           // Bool -> 0/1
  double float_at(size_t i) const;
  std::string_view str_at(size_t i) const;

 private:
  friend struct detail::ArrayAccess;
  ValueType elem_type_ = ValueType::U8;
  size_t count_ = 0;
  std::span<const uint8_t> data_;   // numeric: count * element size bytes
  std::vector<uint64_t> str_off_;   // string: offset of each element's length prefix in data_
};

// integer scalars -> int64_t (U64 > INT64_MAX rejected); F32/F64 -> double
using Value = std::variant<int64_t, double, bool, std::string, Array>;

struct TensorInfo {
  std::string name;
  Tensor tensor;
  uint64_t offset;  // relative to data section
  uint64_t nbytes;
};

class File {
 public:
  // Both throw Error on malformed input, including "out of memory ..." if an
  // allocation fails. Tensor data pointers and Array views point into the bytes;
  // moving a File keeps them valid (the mapping address does not change).
  static File open(const std::string& path);           // mmap; owns mapping
  static File parse(std::span<const uint8_t> bytes);   // non-owning; bytes must outlive File

  uint32_t version() const { return version_; }
  const Value* find(std::string_view key) const;       // nullptr if absent
  std::string get_string(std::string_view key) const;  // throws Error if absent/wrong type
  int64_t get_int(std::string_view key) const;
  double get_float(std::string_view key) const;
  const Array& get_array(std::string_view key) const;
  const std::vector<std::pair<std::string, Value>>& metadata() const { return metadata_; }
  const std::vector<TensorInfo>& tensors() const { return tensors_; }
  const TensorInfo* find_tensor(std::string_view name) const;

 private:
  struct SvHash {
    using is_transparent = void;
    size_t operator()(std::string_view s) const { return std::hash<std::string_view>{}(s); }
  };
  using Index = std::unordered_map<std::string, size_t, SvHash, std::equal_to<>>;

  static File parse_impl(std::span<const uint8_t> bytes);  // lets std::bad_alloc through

  MappedFile map_;
  uint32_t version_ = 0;
  std::vector<std::pair<std::string, Value>> metadata_;
  std::vector<TensorInfo> tensors_;
  Index kv_index_, tensor_index_;
};

}  // namespace nibble::gguf
