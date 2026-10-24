#pragma once
// Tensor metadata shared by the GGUF loader, the model and every backend: the element/block types nibble
// can run (DType, DTypeInfo) and a non-owning view of one weight tensor (Tensor).
//
// A Tensor never owns its bytes. For weights, `data` points straight into the memory-mapped GGUF file
// (see gguf/mapped_file.h), so the file must outlive every Tensor made from it. Quantized tensors are
// stored row by row, each row a whole number of blocks (see quant.h for the block layouts).
#include <array>
#include <cstdint>

namespace nibble {

// Storage type of a tensor. The numeric values are ggml's `ggml_type` ids, so a tensor type read from a
// GGUF file converts with a plain cast. Only the types nibble can run are listed; any other id in a file
// is rejected while parsing (File::parse) with an error naming the type and kSupportedTypes.
//   F32 / F16      plain floats (16-bit half precision for F16)
//   Q4_0, Q5_0     32-weight blocks, one fp16 scale, symmetric 4- / 5-bit values
//   Q5_1           32-weight blocks, fp16 scale and minimum, asymmetric 5-bit values
//   Q8_0           32-weight blocks, one fp16 scale, 8-bit values
//   Q4_K/Q5_K/Q6_K 256-weight "super-blocks" with per-sub-block scales (the llama.cpp k-quants)
enum class DType : uint32_t {
  F32 = 0,
  F16 = 1,
  Q4_0 = 2,
  Q5_0 = 6,
  Q5_1 = 7,
  Q8_0 = 8,
  Q4_K = 12,
  Q5_K = 13,
  Q6_K = 14
};

// Human-readable list of the DTypes above, quoted in error messages about unsupported tensor types.
inline constexpr const char* kSupportedTypes = "F32/F16/Q4_0/Q5_0/Q5_1/Q8_0/Q4_K/Q5_K/Q6_K";

// Static description of a DType: a row of ne[0] elements is ne[0] / block_size blocks of block_bytes bytes
// each (block_size is 1 for the float types). `name` is the ggml spelling, e.g. "Q4_K".
struct DTypeInfo {
  int64_t block_size;   // elements per block (1, 32 or 256)
  int64_t block_bytes;  // bytes per block
  const char* name;     // ggml type name, e.g. "Q4_K"
};

// Returns the static DTypeInfo for t. Throws nibble::Error if t holds a value that is not one of the
// enumerators (possible after a cast from an untrusted file value).
const DTypeInfo& dtype_info(DType t);
// Name of a raw ggml type id (any value, supported or not), for error messages about files that use a type
// nibble does not run. Returns "unknown" for ids outside ggml's table.
const char* ggml_type_name(uint32_t ggml_type);

// Non-owning view of a contiguous, row-major tensor. Weight matrices are stored as ne[1] rows of ne[0]
// elements (the "input" dimension is ne[0], so row r holds the weights of output unit r).
struct Tensor {
  DType type = DType::F32;
  int n_dims = 0;                         // number of used entries of ne (1 or 2 for the tensors nibble loads)
  std::array<int64_t, 4> ne{1, 1, 1, 1};  // ne[0] is the contiguous dimension
  const void* data = nullptr;             // first byte; for weights, inside the mapped file

  // Number of rows: the product of all dimensions above ne[0] (1 for a vector).
  int64_t rows() const { return ne[1] * ne[2] * ne[3]; }
  // Bytes in one row: ne[0] / block_size blocks of block_bytes. Throws Error if the product overflows
  // (a corrupt file can claim an enormous ne[0]). ne[0] must be a multiple of block_size; the GGUF loader
  // enforces that before any Tensor is handed out.
  int64_t row_bytes() const;
};

}  // namespace nibble
