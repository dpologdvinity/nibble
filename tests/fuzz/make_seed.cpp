// Writes fuzz seeds into <dir>: valid.gguf (the GGUF sample, for fuzz_gguf) and
// tiny.gguf (the tiny qwen2 test model, for fuzz_load).
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

#include "gguf_writer.h"
#include "tiny_model.h"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: make_seed <dir>\n");
    return 2;
  }
  std::string dir = argv[1];
  std::string cmd = "mkdir -p '" + dir + "'";
  if (std::system(cmd.c_str()) != 0) return 1;
  auto write = [&](const std::string& name, const std::vector<uint8_t>& b) {
    std::ofstream o(dir + "/" + name, std::ios::binary);
    o.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
    return static_cast<bool>(o);
  };
  return write("valid.gguf", GgufWriter::sample()) && write("tiny.gguf", tiny_model_gguf(1)) ? 0 : 1;
}
