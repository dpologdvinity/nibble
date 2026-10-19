// Reference tool built against pinned llama.cpp (see tools/llama_cpp_version.txt).
//   llama_ref tokenize [--special] <model.gguf> <corpus.txt> <out.txt>   (--special: parse_special, no bos)
//   llama_ref eog      <model.gguf> <out.txt>   (ascending ids with llama_vocab_is_eog, vocab only)
//   llama_ref logits   <model.gguf> <prompt.txt> <n_gen> <out.bin>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "corpus.h"
#include "llama.h"

[[noreturn]] static void die(const std::string& m) {
  std::fprintf(stderr, "llama_ref: %s\n", m.c_str());
  std::exit(1);
}

static std::string read_file(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) die("cannot open " + p);
  return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static std::vector<llama_token> tok(const llama_vocab* v, const std::string& s, bool add_special,
                                    bool parse_special) {
  std::vector<llama_token> t(s.size() + 16);
  int n = llama_tokenize(v, s.data(), int32_t(s.size()), t.data(), int32_t(t.size()), add_special, parse_special);
  if (n < 0) {
    t.resize(size_t(-n));
    n = llama_tokenize(v, s.data(), int32_t(s.size()), t.data(), int32_t(t.size()), add_special, parse_special);
  }
  if (n < 0) die("tokenize failed");
  t.resize(size_t(n));
  return t;
}

static llama_model* load(const char* path) {
  llama_backend_init();
  llama_model_params mp = llama_model_default_params();
  llama_model* m = llama_model_load_from_file(path, mp);
  if (!m) die(std::string("cannot load model ") + path);
  return m;
}

static int cmd_tokenize(char** a, bool parse_special) {
  llama_model* m = load(a[0]);
  const llama_vocab* v = llama_model_get_vocab(m);
  std::vector<std::string> entries;
  try { entries = read_corpus(a[1]); } catch (std::exception& e) { die(e.what()); }
  std::ofstream out(a[2], std::ios::binary);
  if (!out) die(std::string("cannot write ") + a[2]);
  for (auto& e : entries) {
    auto t = tok(v, e, false, parse_special);
    for (size_t i = 0; i < t.size(); ++i) out << (i ? " " : "") << t[i];
    out << "\n";
  }
  out.close();
  if (!out) die("write failed");
  llama_model_free(m);
  return 0;
}

static int cmd_eog(char** a) {
  llama_backend_init();
  llama_model_params mp = llama_model_default_params();
  mp.vocab_only = true;
  llama_model* m = llama_model_load_from_file(a[0], mp);
  if (!m) die(std::string("cannot load model ") + a[0]);
  const llama_vocab* v = llama_model_get_vocab(m);
  std::ofstream out(a[1], std::ios::binary);
  if (!out) die(std::string("cannot write ") + a[1]);
  for (int32_t id = 0; id < llama_vocab_n_tokens(v); ++id)
    if (llama_vocab_is_eog(v, id)) out << id << "\n";
  out.close();
  if (!out) die("write failed");
  llama_model_free(m);
  return 0;
}

static bool g_kv_f32 = false;

static int cmd_logits(char** a) {
  int n_gen = std::atoi(a[2]);
  if (n_gen < 0) die("bad n_gen");
  llama_model* m = load(a[0]);
  const llama_vocab* v = llama_model_get_vocab(m);
  std::vector<llama_token> toks = tok(v, read_file(a[1]), true, true);
  const int n_prompt = int(toks.size());
  if (n_prompt == 0) die("empty prompt");
  const int n_vocab = llama_vocab_n_tokens(v);

  llama_context_params cp = llama_context_default_params();
  cp.n_ctx = uint32_t(n_prompt + n_gen + 8);
  cp.n_batch = cp.n_ctx;
  if (g_kv_f32) cp.type_k = cp.type_v = GGML_TYPE_F32;
  cp.n_threads = cp.n_threads_batch = int(std::max(1u, std::thread::hardware_concurrency()));
  llama_context* ctx = llama_init_from_model(m, cp);
  if (!ctx) die("cannot create context");

  std::vector<float> logits;
  logits.reserve(size_t(n_prompt + n_gen) * n_vocab);

  llama_batch b = llama_batch_init(n_prompt, 0, 1);
  b.n_tokens = n_prompt;
  for (int i = 0; i < n_prompt; ++i) {
    b.token[i] = toks[i];
    b.pos[i] = i;
    b.n_seq_id[i] = 1;
    b.seq_id[i][0] = 0;
    b.logits[i] = 1;
  }
  if (llama_decode(ctx, b) != 0) die("decode failed (prompt)");
  for (int i = 0; i < n_prompt; ++i) {
    const float* r = llama_get_logits_ith(ctx, i);
    if (!r) die("no logits for prompt position");
    logits.insert(logits.end(), r, r + n_vocab);
  }
  llama_batch_free(b);

  auto argmax = [&](const float* r) {
    return int32_t(std::max_element(r, r + n_vocab) - r);  // first max = lowest index
  };
  int32_t next = argmax(logits.data() + size_t(n_prompt - 1) * n_vocab);
  llama_batch g = llama_batch_init(1, 0, 1);
  for (int k = 0; k < n_gen; ++k) {
    toks.push_back(next);
    g.n_tokens = 1;
    g.token[0] = next;
    g.pos[0] = n_prompt + k;
    g.n_seq_id[0] = 1;
    g.seq_id[0][0] = 0;
    g.logits[0] = 1;
    if (llama_decode(ctx, g) != 0) die("decode failed (gen)");
    const float* r = llama_get_logits_ith(ctx, 0);
    if (!r) die("no logits for gen position");
    logits.insert(logits.end(), r, r + n_vocab);
    next = argmax(r);
  }
  llama_batch_free(g);

  if (n_gen > 0) {
    std::string piece(64, '\0');
    int n = llama_token_to_piece(v, toks[n_prompt], piece.data(), 64, 0, true);
    std::fprintf(stderr, "first generated token %d piece '%s'\n", toks[n_prompt],
                 piece.substr(0, n > 0 ? size_t(n) : 0).c_str());
  }

  std::ofstream out(a[3], std::ios::binary);
  if (!out) die(std::string("cannot write ") + a[3]);
  const uint32_t hdr[4] = {1, uint32_t(n_vocab), uint32_t(n_prompt), uint32_t(n_gen)};
  out.write("NBGL", 4);
  out.write(reinterpret_cast<const char*>(hdr), sizeof hdr);
  out.write(reinterpret_cast<const char*>(toks.data()), std::streamsize(toks.size() * 4));
  out.write(reinterpret_cast<const char*>(logits.data()), std::streamsize(logits.size() * 4));
  out.close();
  if (!out) die("write failed");
  llama_free(ctx);
  llama_model_free(m);
  return 0;
}

int main(int argc, char** argv) {
  if (argc == 5 && std::string(argv[1]) == "tokenize") return cmd_tokenize(argv + 2, false);
  if (argc == 6 && std::string(argv[1]) == "tokenize" && std::string(argv[2]) == "--special")
    return cmd_tokenize(argv + 3, true);
  if (argc == 7 && std::string(argv[1]) == "logits" && std::string(argv[6]) == "--kv-f32") g_kv_f32 = true, argc = 6;
  if (argc == 4 && std::string(argv[1]) == "eog") return cmd_eog(argv + 2);
  if (argc == 2 && std::string(argv[1]) == "sysinfo") return std::puts(llama_print_system_info()), 0;
  if (argc == 6 && std::string(argv[1]) == "logits") return cmd_logits(argv + 2);
  die("usage: llama_ref eog <model> <out> | tokenize [--special] <model> <corpus> <out> | "
      "logits <model> <prompt> <n_gen> <out> [--kv-f32] | sysinfo");
}
