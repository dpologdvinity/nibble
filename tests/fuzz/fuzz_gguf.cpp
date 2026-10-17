#include <cstddef>
#include <cstdint>
#include <span>
#include <variant>

#include "gguf/gguf.h"
#include "nibble/error.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  try {
    auto f = nibble::gguf::File::parse(std::span<const uint8_t>(data, size));
    volatile uint8_t sink = 0;
    for (const auto& t : f.tensors()) {
      if (t.nbytes == 0) __builtin_trap();  // parser must never accept empty tensors
      sink = sink + static_cast<const uint8_t*>(t.tensor.data)[t.nbytes - 1];
    }
    // Every accepted array must decode in bounds via its typed accessor.
    for (const auto& kv : f.metadata()) {
      const auto* a = std::get_if<nibble::gguf::Array>(&kv.second);
      if (!a) continue;
      for (size_t i = 0; i < a->size(); ++i) {
        if (a->is_string()) sink = sink + static_cast<uint8_t>(a->str_at(i).size());
        else if (a->is_float()) sink = sink + static_cast<uint8_t>(a->float_at(i) != 0);
        else sink = sink + static_cast<uint8_t>(a->int_at(i));
      }
    }
    (void)sink;
  } catch (const nibble::Error&) {
  }
  return 0;
}
