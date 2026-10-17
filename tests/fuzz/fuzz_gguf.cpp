#include <cstddef>
#include <cstdint>
#include <span>

#include "gguf/gguf.h"
#include "nibble/error.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  try {
    auto f = nibble::gguf::File::parse(std::span<const uint8_t>(data, size));
    volatile uint8_t sink = 0;
    for (const auto& t : f.tensors()) {
      if (t.nbytes > 0) sink = sink + static_cast<const uint8_t*>(t.tensor.data)[t.nbytes - 1];
    }
    (void)sink;
  } catch (const nibble::Error&) {
  }
  return 0;
}
