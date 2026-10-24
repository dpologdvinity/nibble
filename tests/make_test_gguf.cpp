// Fixture generator used by the CLI tests (CMake invokes it at test time): writes small GGUF files into a
// directory - a minimal model, a tiny runnable model, a ChatML one for `nibble chat`, an escape-sequence
// probe for `nibble info`, a Q6_K tensor file - and, with --big-u8, a file whose metadata holds a huge u8
// array to probe memory amplification.
// Test helper: writes GGUF fixtures into the directory given as argv[1].
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "gguf_writer.h"
#include "tiny_model.h"

// Writes bytes to a file, exiting the process on failure (it is a build-time helper).
static void write_file(const std::string& path, const std::vector<uint8_t>& bytes) {
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  if (!f) {
    std::cerr << "cannot write " << path << "\n";
    std::exit(1);
  }
}

// Usage: make_test_gguf <outdir>, or make_test_gguf --big-u8 <path> <n>.
int main(int argc, char** argv) {
  if (argc == 4 && std::string(argv[1]) == "--big-u8") {
    // A valid GGUF whose metadata holds one u8 array of N elements (memory-amplification probe).
    const uint64_t n = std::stoull(argv[3]);
    std::vector<uint8_t> kv;
    GgufWriter::put_str(kv, "junk");
    GgufWriter::put_u32(kv, static_cast<uint32_t>(GgufWriter::ValueType::Array));
    GgufWriter::put_u32(kv, static_cast<uint32_t>(GgufWriter::ValueType::U8));
    GgufWriter::put_u64(kv, n);
    kv.resize(kv.size() + n, 1);
    GgufWriter w;
    w.add_string("general.architecture", "qwen2");
    w.add_raw_kv(std::move(kv));
    write_file(argv[2], w.build());
    return 0;
  }
  if (argc != 2) {
    std::cerr << "usage: make_test_gguf <outdir> | make_test_gguf --big-u8 <path> <n>\n";
    return 2;
  }
  const std::string dir = argv[1];
  {
    GgufWriter w;
    w.add_string("general.architecture", "qwen2");
    w.add_tensor("token_embd.weight", nibble::DType::F16, {64, 10}, std::vector<uint8_t>(64 * 10 * 2, 0));
    write_file(dir + "/small.gguf", w.build());
  }
  write_file(dir + "/tiny.gguf", tiny_model_gguf(1));
  {
    TinyModelOpts mo;  // ChatML tokens + room for two short turns, for `nibble chat`
    mo.tok.chatml = true;
    mo.context_length = 256;
    write_file(dir + "/tiny_chatml.gguf", tiny_model_gguf(1, nibble::DType::F32, false, 259, mo));
  }
  {
    GgufWriter w;  // terminal-escape injection probe for `nibble info`
    w.add_string("evil", "\x1b[31mred\n");
    w.add_string_array("evil.arr", {"\x1b[0m"});
    write_file(dir + "/escape.gguf", w.build());
  }
  {
    GgufWriter w;  // one Q6_K tensor, for `nibble info`
    w.add_tensor("w", nibble::DType::Q6_K, {256}, std::vector<uint8_t>(210, 0));
    write_file(dir + "/q6k.gguf", w.build());
  }
  return 0;
}
