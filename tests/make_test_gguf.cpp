// Test helper: writes GGUF fixtures into the directory given as argv[1].
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "gguf_writer.h"

static void write_file(const std::string& path, const std::vector<uint8_t>& bytes) {
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  if (!f) {
    std::cerr << "cannot write " << path << "\n";
    std::exit(1);
  }
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: make_test_gguf <outdir>\n";
    return 2;
  }
  const std::string dir = argv[1];
  {
    GgufWriter w;
    w.add_string("general.architecture", "qwen2");
    w.add_tensor("token_embd.weight", nibble::DType::F16, {64, 10}, std::vector<uint8_t>(64 * 10 * 2, 0));
    write_file(dir + "/small.gguf", w.build());
  }
  return 0;
}
