// Writes fuzz seeds into <dir>: valid.gguf (the GGUF sample, for fuzz_gguf), tiny.gguf (the tiny
// qwen2 test model, for fuzz_load) and tiny-kq.gguf (the same with a mix of k-quant weights).
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

#include "gguf_writer.h"
#include "tiny_model.h"

// Writes the seed files into the directory given as argv[1].
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
  TinyModelOpts kq;
  kq.kquant_dims = true;
  kq.tensor_types = {{"token_embd.weight", nibble::DType::Q5_0},
                     {"blk.0.attn_v.weight", nibble::DType::Q6_K},
                     {"blk.1.ffn_down.weight", nibble::DType::Q6_K}};
  return write("valid.gguf", GgufWriter::sample()) && write("tiny.gguf", tiny_model_gguf(1)) &&
                 write("tiny-kq.gguf", tiny_model_gguf(1, nibble::DType::Q4_K, false, 259, kq))
             ? 0
             : 1;
}
