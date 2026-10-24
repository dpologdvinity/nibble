// Implementation of the DType tables declared in tensor.h.
#include "tensor.h"

#include <string>

#include "nibble/error.h"

namespace nibble {

// One function-local static per type keeps the returned references valid for the life of the program.
const DTypeInfo& dtype_info(DType t) {
  static const DTypeInfo f32{1, 4, "F32"}, f16{1, 2, "F16"}, q4_0{32, 18, "Q4_0"},
      q5_0{32, 22, "Q5_0"}, q5_1{32, 24, "Q5_1"}, q8_0{32, 34, "Q8_0"}, q4_k{256, 144, "Q4_K"},
      q5_k{256, 176, "Q5_K"}, q6_k{256, 210, "Q6_K"};
  switch (t) {
    case DType::F32: return f32;
    case DType::F16: return f16;
    case DType::Q4_0: return q4_0;
    case DType::Q5_0: return q5_0;
    case DType::Q5_1: return q5_1;
    case DType::Q8_0: return q8_0;
    case DType::Q4_K: return q4_k;
    case DType::Q5_K: return q5_k;
    case DType::Q6_K: return q6_k;
  }
  throw Error("unsupported dtype " + std::to_string(static_cast<uint32_t>(t)));
}

// ggml numbers its types 0..15 in the order below (ids 4 and 5 were the retired Q4_2/Q4_3); 30 is BF16.
const char* ggml_type_name(uint32_t t) {
  static const char* const names[] = {"F32",  "F16",  "Q4_0", "Q4_1", "Q4_2", "Q4_3",
                                      "Q5_0", "Q5_1", "Q8_0", "Q8_1", "Q2_K", "Q3_K",
                                      "Q4_K", "Q5_K", "Q6_K", "Q8_K"};
  if (t < 16) return names[t];
  if (t == 30) return "BF16";
  return "unknown";
}

// Checked multiply: ne[0] comes from the file, so an absurd value must not wrap to a small size that later
// bounds checks would accept.
int64_t Tensor::row_bytes() const {
  const DTypeInfo& i = dtype_info(type);
  int64_t out;
  if (__builtin_mul_overflow(ne[0] / i.block_size, i.block_bytes, &out)) throw Error("row byte size overflows");
  return out;
}

}  // namespace nibble
