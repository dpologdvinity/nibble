#pragma once
#include <array>
#include <cstdint>

namespace nibble {

enum class DType : uint32_t { F32 = 0, F16 = 1, Q4_0 = 2, Q8_0 = 8 };

struct DTypeInfo {
  int64_t block_size;
  int64_t block_bytes;
  const char* name;
};

const DTypeInfo& dtype_info(DType t);          // throws Error for unsupported values
const char* ggml_type_name(uint32_t ggml_type);  // "unknown" if not a known ggml type

struct Tensor {
  DType type = DType::F32;
  int n_dims = 0;
  std::array<int64_t, 4> ne{1, 1, 1, 1};  // ne[0] is the contiguous dimension
  const void* data = nullptr;

  int64_t rows() const { return ne[1] * ne[2] * ne[3]; }
  int64_t row_bytes() const;
};

}  // namespace nibble
