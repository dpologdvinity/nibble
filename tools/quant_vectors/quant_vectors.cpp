// Emits tests/data/quant_golden.inc: expected outputs of llama.cpp's reference quant code (b11715, generic
// non-SIMD build in third_party/build-generic) for fixed inputs. See tools/quant_vectors/CMakeLists.txt
// for how to regenerate.
//
// Functions used (all exported by the generic build):
//   quantize_row_q8_0_ref, quantize_row_q4_0_ref, dequantize_row_q8_0, dequantize_row_q4_0
//     third_party/llama.cpp/ggml/src/ggml-quants.c:276, :113, :553, :459 (libggml-base)
//   ggml_vec_dot_q8_0_q8_0_generic, ggml_vec_dot_q4_0_q8_0_generic
//     third_party/llama.cpp/ggml/src/ggml-cpu/quants.c:451, :225 (libggml-cpu)
//   quantize_row_q8_1_ref, dequantize_row_q5_0, dequantize_row_q5_1
//     third_party/llama.cpp/ggml/src/ggml-quants.c:302, :500, :526 (libggml-base)
//   ggml_vec_dot_q5_0_q8_0_generic, ggml_vec_dot_q5_1_q8_1_generic
//     third_party/llama.cpp/ggml/src/ggml-cpu/quants.c:365, :408 (libggml-cpu)
//   quantize_row_q8_K_ref, dequantize_row_q4_K, dequantize_row_q5_K, dequantize_row_q6_K
//     third_party/llama.cpp/ggml/src/ggml-quants.c:2769, :1530, :1732, :1940 (libggml-base)
//   ggml_vec_dot_q4_K_q8_K_generic, ggml_vec_dot_q5_K_q8_K_generic, ggml_vec_dot_q6_K_q8_K_generic
//     third_party/llama.cpp/ggml/src/ggml-cpu/quants.c:696, :771, :851 (libggml-cpu)
// They are declared here with void* block pointers (ABI-identical to the internal block_q*_0 types in
// ggml-common.h) because ggml-quants.h and ggml-cpu/quants.h are internal headers.
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "ggml-cpu.h"
#include "quant_golden_gen.h"

extern "C" {
void quantize_row_q8_0_ref(const float* x, void* y, int64_t k);
void quantize_row_q4_0_ref(const float* x, void* y, int64_t k);
void dequantize_row_q8_0(const void* x, float* y, int64_t k);
void dequantize_row_q4_0(const void* x, float* y, int64_t k);
void ggml_vec_dot_q8_0_q8_0_generic(int n, float* s, size_t bs, const void* vx, size_t bx, const void* vy, size_t by,
                                    int nrc);
void ggml_vec_dot_q4_0_q8_0_generic(int n, float* s, size_t bs, const void* vx, size_t bx, const void* vy, size_t by,
                                    int nrc);
void quantize_row_q8_1_ref(const float* x, void* y, int64_t k);
void dequantize_row_q5_0(const void* x, float* y, int64_t k);
void dequantize_row_q5_1(const void* x, float* y, int64_t k);
void ggml_vec_dot_q5_0_q8_0_generic(int n, float* s, size_t bs, const void* vx, size_t bx, const void* vy, size_t by,
                                    int nrc);
void ggml_vec_dot_q5_1_q8_1_generic(int n, float* s, size_t bs, const void* vx, size_t bx, const void* vy, size_t by,
                                    int nrc);
void quantize_row_q8_K_ref(const float* x, void* y, int64_t k);
void dequantize_row_q4_K(const void* x, float* y, int64_t k);
void dequantize_row_q5_K(const void* x, float* y, int64_t k);
void dequantize_row_q6_K(const void* x, float* y, int64_t k);
void ggml_vec_dot_q4_K_q8_K_generic(int n, float* s, size_t bs, const void* vx, size_t bx, const void* vy, size_t by,
                                    int nrc);
void ggml_vec_dot_q5_K_q8_K_generic(int n, float* s, size_t bs, const void* vx, size_t bx, const void* vy, size_t by,
                                    int nrc);
void ggml_vec_dot_q6_K_q8_K_generic(int n, float* s, size_t bs, const void* vx, size_t bx, const void* vy, size_t by,
                                    int nrc);
}

namespace {

constexpr uint64_t kDequantSeed = 7;

// Bit pattern of a float.
uint32_t bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}

// Uniform in [lo, hi) from 24 random bits; exact in float.
float uni(golden::Rng& r, float lo, float hi) { return lo + (hi - lo) * static_cast<float>(r.next() >> 40) * 0x1p-24f; }

// One quantizer test input: a name and 32 floats.
struct QCase {
  const char* name;
  float x[32];
};

// The quantizer inputs: ties at both signs, scales that are not fp16-representable, outliers, tiny
// (subnormal-scale) blocks, a Q4_0 maximum that is negative, and some random rows. The same list drives the
// Q8_0, Q4_0 and Q8_1 golden tables.
std::vector<QCase> quant_cases() {
  std::vector<QCase> cs;
  golden::Rng r{42};
  QCase c{};
  // amax = 127 -> d = 1, id = 1: x*id = k + 0.5 exactly, roundf rounds half away from zero.
  c = {"ties_d1", {}};
  for (int j = 0; j < 31; ++j) c.x[j] = static_cast<float>((j - 16) * 7) + 0.5f;
  c.x[31] = -127.0f;
  cs.push_back(c);
  // amax = 63.5 -> d = 0.5, id = 2: x*id = k + 0.5 exactly, small magnitudes incl. +-0.5.
  c = {"ties_d0.5", {}};
  for (int j = 0; j < 31; ++j) c.x[j] = (static_cast<float>(j - 15) + 0.5f) * 0.5f;
  c.x[31] = 63.5f;
  cs.push_back(c);
  // amax = 1 -> d = 1/127, not representable in fp16.
  c = {"d_not_fp16", {}};
  for (int j = 0; j < 32; ++j) c.x[j] = uni(r, -1.f, 1.f);
  c.x[9] = 1.0f;
  cs.push_back(c);
  c = {"mixed_signs", {}};
  for (int j = 0; j < 32; ++j) c.x[j] = uni(r, -3.f, 3.f);
  cs.push_back(c);
  c = {"all_zero", {}};
  cs.push_back(c);
  c = {"outlier_pos", {}};
  for (int j = 0; j < 32; ++j) c.x[j] = uni(r, -0.01f, 0.01f);
  c.x[13] = 1000.25f;
  cs.push_back(c);
  c = {"outlier_neg", {}};
  for (int j = 0; j < 32; ++j) c.x[j] = uni(r, -0.5f, 0.5f);
  c.x[7] = -517.375f;
  cs.push_back(c);
  // amax ~3e-6 -> d below the smallest fp16 subnormal granularity: fp16(d) is subnormal/zero, id is not.
  c = {"tiny_subnormal_d", {}};
  for (int j = 0; j < 32; ++j) c.x[j] = uni(r, -3e-6f, 3e-6f);
  cs.push_back(c);
  // Q4_0 path: max is negative (-8 -> d = 1), others k + 0.5.
  c = {"q4_max_neg", {}};
  for (int j = 0; j < 31; ++j) c.x[j] = static_cast<float>((j % 15) - 7) + 0.5f;
  c.x[20] = -8.0f;
  cs.push_back(c);
  for (int i = 0; i < 3; ++i) {
    c = {i == 0 ? "random0" : i == 1 ? "random1" : "random2", {}};
    for (int j = 0; j < 32; ++j) c.x[j] = uni(r, -40.f, 40.f);
    cs.push_back(c);
  }
  return cs;
}

}  // namespace

// Writes the generated header named by argv[1]: quantizer cases, dot-product cases (Q4_0/Q8_0, Q5, K-quants)
// with their input hashes, and dequantization hashes - every number is computed by ggml's reference code from
// inputs regenerated by tests/quant_golden_gen.h.
int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: quant_vectors OUT.inc\n");
    return 2;
  }
  ggml_cpu_init();  // fills the fp16->fp32 table used by the generic CPU path
  FILE* f = std::fopen(argv[1], "w");
  if (!f) {
    std::perror(argv[1]);
    return 1;
  }
  std::fprintf(f,
               "// GENERATED by tools/quant_vectors from llama.cpp b11715 reference code. DO NOT EDIT.\n"
               "// Regenerate: see tools/quant_vectors/CMakeLists.txt.\n"
               "#pragma once\n#include <cstdint>\n\nnamespace golden {\n\n");

  // Quantizer cases.
  std::fprintf(f,
               "struct QuantCase {\n  const char* name;\n  float x[32];\n  uint16_t q8_d;\n  int8_t q8_qs[32];\n"
               "  uint16_t q4_d;\n  uint8_t q4_qs[16];\n};\n\ninline constexpr QuantCase kQuantCases[] = {\n");
  for (const QCase& c : quant_cases()) {
    uint8_t q8[golden::kQ8Bytes], q4[golden::kQ4Bytes];
    quantize_row_q8_0_ref(c.x, q8, 32);
    quantize_row_q4_0_ref(c.x, q4, 32);
    std::fprintf(f, "    {\"%s\",\n     {", c.name);
    for (int j = 0; j < 32; ++j) std::fprintf(f, "%s%af", j ? ", " : "", static_cast<double>(c.x[j]));
    std::fprintf(f, "},\n     0x%04x,\n     {", q8[0] | (q8[1] << 8));
    for (int j = 0; j < 32; ++j) std::fprintf(f, "%s%d", j ? ", " : "", static_cast<int8_t>(q8[2 + j]));
    std::fprintf(f, "},\n     0x%04x,\n     {", q4[0] | (q4[1] << 8));
    for (int j = 0; j < 16; ++j) std::fprintf(f, "%s0x%02x", j ? ", " : "", q4[2 + j]);
    std::fprintf(f, "}},\n");
  }
  std::fprintf(f, "};\n\n");

  // Dot cases: per (seed, n), blocks w8, a8, w4 from golden::fill_* in that order.
  std::fprintf(f,
               "// Inputs: Rng{seed}; fill_q8_0(w8, nb); fill_q8_0(a8, nb); fill_q4_0(w4, nb). input_hash = FNV-1a\n"
               "// over w8|a8|w4 bytes. q8q8 = vec_dot_q8_0_q8_0_generic(w8, a8), q4q8 = "
               "vec_dot_q4_0_q8_0_generic(w4, a8).\n"
               "struct DotCase {\n  uint64_t seed;\n  int n;\n  uint64_t input_hash;\n  uint32_t q8q8_bits;\n"
               "  uint32_t q4q8_bits;\n};\n\ninline constexpr DotCase kDotCases[] = {\n");
  for (int n : {896, 4864}) {
    for (uint64_t seed : {1ull, 2ull, 3ull, 4ull}) {
      const int nb = n / 32;
      std::vector<uint8_t> w8(static_cast<size_t>(nb) * golden::kQ8Bytes), a8(w8.size()),
          w4(static_cast<size_t>(nb) * golden::kQ4Bytes);
      golden::Rng r{seed};
      golden::fill_q8_0(r, w8.data(), nb);
      golden::fill_q8_0(r, a8.data(), nb);
      golden::fill_q4_0(r, w4.data(), nb);
      uint64_t h = golden::fnv1a(w8.data(), w8.size());
      h = golden::fnv1a(a8.data(), a8.size(), h);
      h = golden::fnv1a(w4.data(), w4.size(), h);
      float s88 = 0, s48 = 0;
      ggml_vec_dot_q8_0_q8_0_generic(n, &s88, 0, w8.data(), 0, a8.data(), 0, 1);
      ggml_vec_dot_q4_0_q8_0_generic(n, &s48, 0, w4.data(), 0, a8.data(), 0, 1);
      std::fprintf(f, "    {%" PRIu64 ", %d, 0x%016" PRIx64 "ull, 0x%08" PRIx32 "u, 0x%08" PRIx32 "u},  // %a, %a\n",
                   seed, n, h, bits(s88), bits(s48), static_cast<double>(s88), static_cast<double>(s48));
    }
  }
  std::fprintf(f, "};\n\n");

  // Dequant: Rng{kDequantSeed}; fill_q8_0(q8, nb); fill_q4_0(q4, nb).
  const int nb = 8;
  std::vector<uint8_t> q8(nb * golden::kQ8Bytes), q4(nb * golden::kQ4Bytes);
  golden::Rng r{kDequantSeed};
  golden::fill_q8_0(r, q8.data(), nb);
  golden::fill_q4_0(r, q4.data(), nb);
  std::vector<float> d8(nb * 32), d4(nb * 32);
  dequantize_row_q8_0(q8.data(), d8.data(), nb * 32);
  dequantize_row_q4_0(q4.data(), d4.data(), nb * 32);
  std::fprintf(f,
               "// Inputs: Rng{kDequantSeed}; fill_q8_0(q8, kDequantBlocks); fill_q4_0(q4, kDequantBlocks).\n"
               "inline constexpr uint64_t kDequantSeed = %" PRIu64 ";\ninline constexpr int kDequantBlocks = %d;\n",
               kDequantSeed, nb);
  for (int t = 0; t < 2; ++t) {
    const std::vector<float>& v = t == 0 ? d8 : d4;
    std::fprintf(f, "inline constexpr uint32_t kDequant%s[%d] = {", t == 0 ? "Q8_0" : "Q4_0", nb * 32);
    for (size_t i = 0; i < v.size(); ++i)
      std::fprintf(f, "%s0x%08" PRIx32 "u", i % 8 ? ", " : (i ? ",\n    " : "\n    "), bits(v[i]));
    std::fprintf(f, "};\n");
  }

  // Q8_1 quantizer: per kQuantCases input (same order), d and s; qs equal the Q8_0 qs.
  std::fprintf(f,
               "\n// quantize_row_q8_1_ref of kQuantCases[i].x; its qs equal kQuantCases[i].q8_qs.\n"
               "struct Q8_1Case {\n  const char* name;\n  uint16_t d;\n  uint16_t s;\n};\n\n"
               "inline constexpr Q8_1Case kQ8_1Cases[] = {\n");
  for (const QCase& c : quant_cases()) {
    uint8_t q[golden::kQ8_1Bytes];
    quantize_row_q8_1_ref(c.x, q, 32);
    std::fprintf(f, "    {\"%s\", 0x%04x, 0x%04x},\n", c.name, q[0] | (q[1] << 8), q[2] | (q[3] << 8));
  }
  std::fprintf(f, "};\n\n");

  // Q5 dot cases: per (seed, n), blocks w50, a80, w51, a81 from golden::fill_* in that order.
  std::fprintf(f,
               "// Inputs: Rng{seed}; fill_q5_0(w50, nb); fill_q8_0(a80, nb); fill_q5_1(w51, nb); fill_q8_1(a81, nb).\n"
               "// input_hash = FNV-1a over w50|a80|w51|a81 bytes. q5_0 = vec_dot_q5_0_q8_0_generic(w50, a80),\n"
               "// q5_1 = vec_dot_q5_1_q8_1_generic(w51, a81).\n"
               "struct Dot5Case {\n  uint64_t seed;\n  int n;\n  uint64_t input_hash;\n  uint32_t q5_0_bits;\n"
               "  uint32_t q5_1_bits;\n};\n\ninline constexpr Dot5Case kDot5Cases[] = {\n");
  for (int n : {32, 896, 4864}) {
    for (uint64_t seed : {1ull, 2ull, 3ull}) {
      const size_t nbs = static_cast<size_t>(n / 32);
      std::vector<uint8_t> w50(nbs * golden::kQ5_0Bytes), a80(nbs * golden::kQ8Bytes), w51(nbs * golden::kQ5_1Bytes),
          a81(nbs * golden::kQ8_1Bytes);
      golden::Rng rr{seed};
      golden::fill_q5_0(rr, w50.data(), n / 32);
      golden::fill_q8_0(rr, a80.data(), n / 32);
      golden::fill_q5_1(rr, w51.data(), n / 32);
      golden::fill_q8_1(rr, a81.data(), n / 32);
      uint64_t h = golden::fnv1a(w50.data(), w50.size());
      h = golden::fnv1a(a80.data(), a80.size(), h);
      h = golden::fnv1a(w51.data(), w51.size(), h);
      h = golden::fnv1a(a81.data(), a81.size(), h);
      float s50 = 0, s51 = 0;
      ggml_vec_dot_q5_0_q8_0_generic(n, &s50, 0, w50.data(), 0, a80.data(), 0, 1);
      ggml_vec_dot_q5_1_q8_1_generic(n, &s51, 0, w51.data(), 0, a81.data(), 0, 1);
      std::fprintf(f, "    {%" PRIu64 ", %d, 0x%016" PRIx64 "ull, 0x%08" PRIx32 "u, 0x%08" PRIx32 "u},  // %a, %a\n",
                   seed, n, h, bits(s50), bits(s51), static_cast<double>(s50), static_cast<double>(s51));
    }
  }
  std::fprintf(f, "};\n\n");

  // Q5 dequant: a fresh Rng{kDequantSeed}; fill_q5_0(q50, nb); fill_q5_1(q51, nb).
  std::vector<uint8_t> q50(nb * golden::kQ5_0Bytes), q51(nb * golden::kQ5_1Bytes);
  golden::Rng r5{kDequantSeed};
  golden::fill_q5_0(r5, q50.data(), nb);
  golden::fill_q5_1(r5, q51.data(), nb);
  std::vector<float> d50(nb * 32), d51(nb * 32);
  dequantize_row_q5_0(q50.data(), d50.data(), nb * 32);
  dequantize_row_q5_1(q51.data(), d51.data(), nb * 32);
  std::fprintf(f, "// Inputs: Rng{kDequantSeed}; fill_q5_0(q50, kDequantBlocks); fill_q5_1(q51, kDequantBlocks).\n");
  for (int t = 0; t < 2; ++t) {
    const std::vector<float>& v = t == 0 ? d50 : d51;
    std::fprintf(f, "inline constexpr uint32_t kDequant%s[%d] = {", t == 0 ? "Q5_0" : "Q5_1", nb * 32);
    for (size_t i = 0; i < v.size(); ++i)
      std::fprintf(f, "%s0x%08" PRIx32 "u", i % 8 ? ", " : (i ? ",\n    " : "\n    "), bits(v[i]));
    std::fprintf(f, "};\n");
  }
  // Q8_K quantizer: crafted rows (id 0..kCraftedQ8K-1) then random rows. The output buffer is zeroed first, so a
  // zero block's bsums hash as 0 (ggml leaves them untouched; they cannot affect a dot because d = 0).
  std::fprintf(
      f,
      "\n// quantize_row_q8_K_ref. crafted >= 0: golden::crafted_q8k_row(crafted, x), n = 256; else\n"
      "// golden::q8k_rows(Rng{seed}, x, n). d_hash = FNV-1a over the d floats' bytes, out_hash = FNV-1a over\n"
      "// each block's qs then bsums bytes, in block order.\n"
      "struct Q8KCase {\n  int crafted;\n  uint64_t seed;\n  int n;\n  uint64_t d_hash;\n  uint64_t out_hash;\n};\n\n"
      "inline constexpr Q8KCase kQ8KCases[] = {\n");
  struct Q8KIn {
    int crafted, n;
    uint64_t seed;
  };
  std::vector<Q8KIn> q8k_ins;
  for (int i = 0; i < golden::kCraftedQ8K; ++i) q8k_ins.push_back({i, 256, 0});
  for (int n : {256, 2048})
    for (uint64_t seed : {1ull, 2ull, 3ull}) q8k_ins.push_back({-1, n, seed});
  for (const Q8KIn& in : q8k_ins) {
    std::vector<float> x(static_cast<size_t>(in.n));
    if (in.crafted >= 0) {
      golden::crafted_q8k_row(in.crafted, x.data());
    } else {
      golden::Rng rk{in.seed};
      golden::q8k_rows(rk, x.data(), in.n);
    }
    const size_t nbk = static_cast<size_t>(in.n / 256);
    std::vector<uint8_t> y(nbk * golden::kQ8_KBytes, 0);
    quantize_row_q8_K_ref(x.data(), y.data(), in.n);
    uint64_t dh = 0xCBF29CE484222325ull, oh = dh;
    for (size_t b = 0; b < nbk; ++b) {
      dh = golden::fnv1a(&y[b * golden::kQ8_KBytes], 4, dh);
      oh = golden::fnv1a(&y[b * golden::kQ8_KBytes + 4], 256 + 32, oh);
    }
    std::fprintf(f, "    {%d, %" PRIu64 ", %d, 0x%016" PRIx64 "ull, 0x%016" PRIx64 "ull},\n", in.crafted, in.seed, in.n,
                 dh, oh);
  }
  std::fprintf(f, "};\n\n");

  // K dot cases: per (seed, n), w4, w5, w6 weights then a8 activations from golden::fill_* in that order.
  std::fprintf(
      f,
      "// Inputs: Rng{seed}; fill_q4_K(w4, nb); fill_q5_K(w5, nb); fill_q6_K(w6, nb); fill_q8_K(a8, nb), nb = n/256.\n"
      "// input_hash = FNV-1a over w4|w5|w6|a8 bytes. qNk = vec_dot_qN_K_q8_K_generic(wN, a8).\n"
      "struct KDotCase {\n  uint64_t seed;\n  int n;\n  uint64_t input_hash;\n  uint32_t q4k_bits;\n"
      "  uint32_t q5k_bits;\n  uint32_t q6k_bits;\n};\n\ninline constexpr KDotCase kKDotCases[] = {\n");
  for (int n : {256, 2048, 4864, 8192}) {
    for (uint64_t seed : {1ull, 2ull, 3ull}) {
      const int nbk = n / 256;
      const size_t z = static_cast<size_t>(nbk);
      std::vector<uint8_t> w4(z * golden::kQ4_KBytes), w5(z * golden::kQ5_KBytes), w6(z * golden::kQ6_KBytes),
          a8(z * golden::kQ8_KBytes);
      golden::Rng rr{seed};
      golden::fill_q4_K(rr, w4.data(), nbk);
      golden::fill_q5_K(rr, w5.data(), nbk);
      golden::fill_q6_K(rr, w6.data(), nbk);
      golden::fill_q8_K(rr, a8.data(), nbk);
      uint64_t h = golden::fnv1a(w4.data(), w4.size());
      h = golden::fnv1a(w5.data(), w5.size(), h);
      h = golden::fnv1a(w6.data(), w6.size(), h);
      h = golden::fnv1a(a8.data(), a8.size(), h);
      float s4 = 0, s5 = 0, s6 = 0;
      ggml_vec_dot_q4_K_q8_K_generic(n, &s4, 0, w4.data(), 0, a8.data(), 0, 1);
      ggml_vec_dot_q5_K_q8_K_generic(n, &s5, 0, w5.data(), 0, a8.data(), 0, 1);
      ggml_vec_dot_q6_K_q8_K_generic(n, &s6, 0, w6.data(), 0, a8.data(), 0, 1);
      std::fprintf(
          f, "    {%" PRIu64 ", %d, 0x%016" PRIx64 "ull, 0x%08" PRIx32 "u, 0x%08" PRIx32 "u, 0x%08" PRIx32 "u},\n",
          seed, n, h, bits(s4), bits(s5), bits(s6));
    }
  }
  std::fprintf(f, "};\n\n");

  // K dequant: a fresh Rng{kDequantSeed} per type, kDequantKBlocks super-blocks; hash = FNV-1a over the output floats.
  constexpr int kKb = 2;
  std::fprintf(
      f,
      "// Inputs: Rng{kDequantSeed}; fill_qN_K(q, kDequantKBlocks), one fresh Rng per type. hash = FNV-1a over\n"
      "// the output floats' bytes.\ninline constexpr int kDequantKBlocks = %d;\n"
      "struct DequantKCase {\n  const char* name;\n  uint64_t hash;\n};\n\n"
      "inline constexpr DequantKCase kDequantK[] = {\n",
      kKb);
  for (int t = 0; t < 3; ++t) {
    const int bytes = t == 0 ? golden::kQ4_KBytes : t == 1 ? golden::kQ5_KBytes : golden::kQ6_KBytes;
    std::vector<uint8_t> q(static_cast<size_t>(kKb) * bytes);
    golden::Rng rd{kDequantSeed};
    if (t == 0) golden::fill_q4_K(rd, q.data(), kKb);
    if (t == 1) golden::fill_q5_K(rd, q.data(), kKb);
    if (t == 2) golden::fill_q6_K(rd, q.data(), kKb);
    std::vector<float> out(static_cast<size_t>(kKb) * 256);
    if (t == 0) dequantize_row_q4_K(q.data(), out.data(), kKb * 256);
    if (t == 1) dequantize_row_q5_K(q.data(), out.data(), kKb * 256);
    if (t == 2) dequantize_row_q6_K(q.data(), out.data(), kKb * 256);
    std::fprintf(f, "    {\"%s\", 0x%016" PRIx64 "ull},\n",
                 t == 0   ? "Q4_K"
                 : t == 1 ? "Q5_K"
                          : "Q6_K",
                 golden::fnv1a(reinterpret_cast<const uint8_t*>(out.data()), out.size() * sizeof(float)));
  }
  std::fprintf(f, "};\n");
  std::fprintf(f, "\n}  // namespace golden\n");
  return std::fclose(f) == 0 ? 0 : 1;
}
