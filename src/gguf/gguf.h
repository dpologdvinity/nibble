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

// String -> str; Bool and all integer types -> i; F32/F64 -> f. Nested arrays rejected.
struct Array {
  ValueType elem_type = ValueType::U8;
  std::vector<std::string> str;
  std::vector<int64_t> i;
  std::vector<double> f;
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

  MappedFile map_;
  uint32_t version_ = 0;
  std::vector<std::pair<std::string, Value>> metadata_;
  std::vector<TensorInfo> tensors_;
  Index kv_index_, tensor_index_;
};

}  // namespace nibble::gguf
