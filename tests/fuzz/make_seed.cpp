// Writes the valid sample GGUF file to <dir>/valid.gguf as a fuzz seed.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

#include "gguf_writer.h"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: make_seed <dir>\n");
    return 2;
  }
  std::string dir = argv[1];
  std::string cmd = "mkdir -p '" + dir + "'";
  if (std::system(cmd.c_str()) != 0) return 1;
  auto b = GgufWriter::sample();
  std::ofstream o(dir + "/valid.gguf", std::ios::binary);
  o.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
  return o ? 0 : 1;
}
