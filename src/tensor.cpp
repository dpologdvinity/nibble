#include "tensor.h"

#include <string>

#include "nibble/error.h"

namespace nibble {

const DTypeInfo& dtype_info(DType t) {
  static const DTypeInfo f32{1, 4, "F32"}, f16{1, 2, "F16"}, q4_0{32, 18, "Q4_0"},
      q8_0{32, 34, "Q8_0"};
  switch (t) {
    case DType::F32: return f32;
    case DType::F16: return f16;
    case DType::Q4_0: return q4_0;
    case DType::Q8_0: return q8_0;
  }
  throw Error("unsupported dtype " + std::to_string(static_cast<uint32_t>(t)));
}

const char* ggml_type_name(uint32_t t) {
  static const char* const names[] = {"F32",  "F16",  "Q4_0", "Q4_1", "Q4_2", "Q4_3",
                                      "Q5_0", "Q5_1", "Q8_0", "Q8_1", "Q2_K", "Q3_K",
                                      "Q4_K", "Q5_K", "Q6_K", "Q8_K"};
  if (t < 16) return names[t];
  if (t == 30) return "BF16";
  return "unknown";
}

int64_t Tensor::row_bytes() const {
  const DTypeInfo& i = dtype_info(type);
  int64_t out;
  if (__builtin_mul_overflow(ne[0] / i.block_size, i.block_bytes, &out)) throw Error("row byte size overflows");
  return out;
}

}  // namespace nibble
