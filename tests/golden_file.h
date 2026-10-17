#pragma once
// Reader for llama_ref golden files (see tools/llama_ref/llama_ref.cpp).
// Layout (little-endian): "NBGL", u32 version=1, u32 n_vocab, u32 n_prompt,
// u32 n_gen, i32 tokens[n_prompt+n_gen], f32 logits[(n_prompt+n_gen)*n_vocab].
#include <cstdint>
#include <cstring>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "corpus.h"

struct Golden {
  uint32_t n_vocab = 0, n_prompt = 0, n_gen = 0;
  std::vector<int32_t> tokens;
  std::vector<float> logits;

  std::span<const float> row(size_t i) const {
    return {logits.data() + i * n_vocab, n_vocab};
  }

  static Golden load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    char magic[4];
    uint32_t hdr[4];
    f.read(magic, 4);
    f.read(reinterpret_cast<char*>(hdr), sizeof hdr);
    if (!f || std::memcmp(magic, "NBGL", 4) != 0) throw std::runtime_error("bad golden magic: " + path);
    if (hdr[0] != 1) throw std::runtime_error("unsupported golden version: " + path);
    Golden g;
    g.n_vocab = hdr[1];
    g.n_prompt = hdr[2];
    g.n_gen = hdr[3];
    size_t n = size_t(g.n_prompt) + g.n_gen;
    g.tokens.resize(n);
    g.logits.resize(n * g.n_vocab);
    f.read(reinterpret_cast<char*>(g.tokens.data()), std::streamsize(n * sizeof(int32_t)));
    f.read(reinterpret_cast<char*>(g.logits.data()), std::streamsize(g.logits.size() * sizeof(float)));
    if (!f) throw std::runtime_error("truncated golden file: " + path);
    return g;
  }
};
