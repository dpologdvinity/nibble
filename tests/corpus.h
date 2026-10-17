#pragma once
// Tokenizer corpus splitting rule, shared by tests and tools/llama_ref.
// Entries are separated by a line that is exactly "====". The newline before
// the separator line belongs to the separator, not the entry; the separator's
// own trailing newline is consumed too. The final entry runs verbatim to EOF
// (a trailing newline at EOF, if present, is part of that entry).
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

inline std::vector<std::string> split_corpus(const std::string& data) {
  static const std::string kSep = "\n====\n";
  std::vector<std::string> out;
  size_t start = 0;
  for (;;) {
    size_t p = data.find(kSep, start);
    if (p == std::string::npos) {
      out.push_back(data.substr(start));
      return out;
    }
    out.push_back(data.substr(start, p - start));
    start = p + kSep.size();
  }
}

inline std::vector<std::string> read_corpus(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::string data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  return split_corpus(data);
}
