// `nibble bench`: pp/tg throughput and memory-bandwidth probe.
// Exception to the "CLI uses only nibble.h" rule: this file includes src/ headers
// (Transformer, Weights, ThreadPool) so tokenization and sampling are not measured.
// run/chat must keep using the public API only.
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "backend/cpu/cpu_backend.h"
#include "backend/cpu/simd/simd.h"
#include "bench/membw.h"
#include "commands.h"
#include "gguf/gguf.h"
#include "model/transformer.h"
#include "nibble/error.h"
#include "thread_pool.h"
#include "tokenizer/tokenizer.h"

#include "nibble_git.h"  // generated at build time

namespace nibble::cli {
namespace {

void usage() {
  std::fputs(
      "usage: nibble bench -m <model.gguf> [-t <list, e.g. 1,2,4,8>] [-p <prompt tokens=512>] "
      "[-n <gen tokens=128>] [-r <reps=5>] [--mem] [--out <file.jsonl>]\n",
      stderr);
}

bool parse_int(const char* s, long lo, long& out) {
  char* end = nullptr;
  errno = 0;
  const long v = std::strtol(s, &end, 10);
  if (end == s || *end || errno || v < lo || v > 1000000) return false;
  out = v;
  return true;
}

bool parse_threads(const std::string& s, std::vector<int>& out) {
  size_t i = 0;
  while (true) {
    const size_t j = s.find(',', i);
    long v;
    if (!parse_int(s.substr(i, j == std::string::npos ? j : j - i).c_str(), 1, v)) return false;
    out.push_back(static_cast<int>(v));
    if (j == std::string::npos) return true;
    i = j + 1;
  }
}

double load1() {
  const char* p = std::getenv("NIBBLE_LOADAVG_PATH");
  std::ifstream f(p ? p : "/proc/loadavg");
  double v = 0;
  f >> v;
  return v;
}

std::string iso_now() {
  const std::time_t t = std::time(nullptr);
  char b[32];
  std::tm tm{};
  gmtime_r(&t, &tm);
  std::strftime(b, sizeof b, "%Y-%m-%dT%H:%M:%SZ", &tm);
  return b;
}

struct Stats { double med, mn, mx; };
Stats stats(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  return {n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2, v.front(), v.back()};
}

std::string json_escape(const std::string& s) {
  std::string o;
  for (const unsigned char ch : s) {
    if (ch == '"' || ch == '\\') { o += '\\'; o += static_cast<char>(ch); }
    else if (ch < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", ch); o += b; }
    else o += static_cast<char>(ch);
  }
  return o;
}

struct Sink {
  std::string model, date;
  double load;
  std::ofstream out;
  void emit(const char* test, int threads, int reps, Stats s, const char* unit) {
    std::printf("%-8s %7d %10.2f %10.2f %10.2f  %s%s\n", test, threads, s.med, s.mn, s.mx, unit, std::string(unit) == "GB/s" ? " (1e9 bytes/s)" : "");
    std::fflush(stdout);
    if (!out.is_open()) return;
    char nums[256];
    const int len = std::snprintf(nums, sizeof nums,
                  "\",\"threads\":%d,\"reps\":%d,\"median\":%.4f,\"min\":%.4f,\"max\":%.4f,\"unit\":\"%s\","
                  "\"load1\":%.2f,\"date\":\"%s\"}\n",
                  threads, reps, s.med, s.mn, s.mx, unit, load, date.c_str());
    if (len < 0 || len >= static_cast<int>(sizeof nums)) throw Error("bench: JSON line too long");
    out << "{\"tool\":\"nibble\",\"commit\":\"" << NIBBLE_GIT_SHA << "\",\"kernels\":\"" << simd::kernel_set() << "\",\"model\":\"" << json_escape(model)
        << "\",\"test\":\"" << test << nums;
    out.flush();
  }
};

using clock_ = std::chrono::steady_clock;
double secs(clock_::time_point a) { return std::chrono::duration<double>(clock_::now() - a).count(); }

}  // namespace

int cmd_bench(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--version") {
    std::puts(NIBBLE_GIT_SHA);
    return 0;
  }
  std::string model_path, out_path;
  std::vector<int> threads{1};
  long p = 512, n = 128, r = 5;
  bool mem = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--mem") { mem = true; continue; }
    const bool known = a == "-m" || a == "-t" || a == "-p" || a == "-n" || a == "-r" || a == "--out";
    if (!known || i + 1 >= argc) {
      std::fprintf(stderr, known ? "error: flag '%s' needs a value\n" : "error: unknown argument '%s'\n", a.c_str());
      usage();
      return 2;
    }
    const char* v = argv[++i];
    bool ok = true;
    if (a == "-m") model_path = v;
    else if (a == "--out") out_path = v;
    else if (a == "-t") { threads.clear(); ok = parse_threads(v, threads); }
    else if (a == "-p") ok = parse_int(v, 1, p);
    else if (a == "-n") ok = parse_int(v, 1, n);
    else ok = parse_int(v, 1, r);
    if (!ok) {
      std::fprintf(stderr, "error: bad value '%s' for %s\n", v, a.c_str());
      usage();
      return 2;
    }
  }
  if (model_path.empty()) {
    std::fputs("error: need -m\n", stderr);
    usage();
    return 2;
  }
  const double load = load1();
  if (load > 2.0) std::fprintf(stderr, "warning: load average %.2f \xe2\x80\x94 results will be noisy\n", load);

  Sink sink;
  sink.model = std::filesystem::path(model_path).filename().string();
  sink.date = iso_now();
  sink.load = load;
  if (!out_path.empty()) {
    sink.out.open(out_path, std::ios::app);
    if (!sink.out) throw Error("cannot write '" + out_path + "'");
  }

  const gguf::File file = gguf::File::open(model_path);
  const model::LlamaConfig cfg = model::LlamaConfig::from_gguf(file);
  const tok::Tokenizer tokenizer = tok::Tokenizer::from_gguf(file);
  const int64_t n_ctx = std::max(p, n + 1) + 1;
  if (n_ctx > cfg.n_ctx_train)
    throw Error("bench needs n_ctx " + std::to_string(n_ctx) + " > model context_length " +
                std::to_string(cfg.n_ctx_train));

  std::vector<int32_t> prompt(static_cast<size_t>(p));
  for (int64_t i = 0; i < p; ++i) prompt[static_cast<size_t>(i)] = static_cast<int32_t>(i % cfg.n_vocab);
  if (tokenizer.add_bos() && tokenizer.bos() >= 0) prompt[0] = tokenizer.bos();
  std::vector<float> logits(static_cast<size_t>(cfg.n_vocab));

  std::printf("%-8s %7s %10s %10s %10s\n", "test", "threads", "median", "min", "max");
  const std::string pp_name = "pp" + std::to_string(p), tg_name = "tg" + std::to_string(n);
  for (const int t : threads) {
    ThreadPool pool(t);
    cpu::CpuBackend backend(pool);
    const model::Weights weights = model::Weights::bind(file, cfg, backend);
    model::Transformer tf(cfg, weights, backend, n_ctx);
    std::vector<double> pp, tg;
    for (long rep = -1; rep < r; ++rep) {  // rep -1 is the untimed warm-up
      auto t0 = clock_::now();
      tf.forward(prompt, 0, logits.data());
      const double pp_s = secs(t0);
      // tg: 1-token prompt, then n greedy decode steps.
      const int32_t first = tokenizer.bos() >= 0 ? tokenizer.bos() : 0;
      tf.forward(std::span<const int32_t>(&first, 1), 0, logits.data());
      t0 = clock_::now();
      for (long i = 0; i < n; ++i) {
        const int32_t next = static_cast<int32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
        tf.forward(std::span<const int32_t>(&next, 1), 1 + i, logits.data());
      }
      const double tg_s = secs(t0);
      if (rep >= 0) {
        pp.push_back(static_cast<double>(p) / pp_s);
        tg.push_back(static_cast<double>(n) / tg_s);
      }
    }
    sink.emit(pp_name.c_str(), t, static_cast<int>(r), stats(pp), "tok/s");
    sink.emit(tg_name.c_str(), t, static_cast<int>(r), stats(tg), "tok/s");
  }
  if (mem) {
    const int t = *std::max_element(threads.begin(), threads.end());
    ThreadPool pool(t);
    sink.emit("membw", t, static_cast<int>(r), stats(bench::measure_membw(pool, bench::membw_default_bytes(), static_cast<int>(r))), "GB/s");
  }
  return 0;
}

}  // namespace nibble::cli
