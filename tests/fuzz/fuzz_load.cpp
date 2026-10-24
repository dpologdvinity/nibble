// libFuzzer target for the full load path short of a forward pass: GGUF parse,
// tokenizer construction, config extraction and weight binding. Any exception other
// than nibble::Error, and any sanitizer report, is a bug.
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

#include "backend/cpu/cpu_backend.h"
#include "gguf/gguf.h"
#include "model/config.h"
#include "model/weights.h"
#include "nibble/error.h"
#include "thread_pool.h"
#include "tokenizer/tokenizer.h"

using namespace nibble;

// libFuzzer entry point. The backend is static so that the corpus run does not spend its time constructing
// thread pools.
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  static ThreadPool pool(1);
  static cpu::CpuBackend backend(pool);
  std::optional<gguf::File> f;
  try {
    f.emplace(gguf::File::parse(std::span<const uint8_t>(data, size)));
  } catch (const Error&) {
    return 0;
  }
  // Stages are independent so a tokenizer rejection does not hide config/bind bugs.
  try {
    volatile int32_t n = tok::Tokenizer::from_gguf(*f).n_vocab();
    (void)n;
  } catch (const Error&) {
  }
  try {
    const model::LlamaConfig cfg = model::LlamaConfig::from_gguf(*f);
    const model::Weights w = model::Weights::bind(*f, cfg, backend);
    volatile size_t n_layers = w.layers.size();
    (void)n_layers;
  } catch (const Error&) {
  }
  return 0;
}
