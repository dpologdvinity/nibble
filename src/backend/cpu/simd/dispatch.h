#pragma once
// Runtime selection of the SIMD kernel tier.
//
// The kernels in this directory are compiled once per tier (CMakeLists.txt: nibble_simd_tier), each with its own
// instruction-set flags and inside its own namespace nibble::simd::<tier>, and each tier exports one Kernels table
// (table.cpp). Everything else is compiled for the build's target architecture and calls the kernels only through
// the table select() picks, so one binary runs on any x86-64 CPU: AVX2+FMA+F16C with AVX-VNNI, without it, or the
// scalar code in ref_ops.cpp (select() returns nullptr). This header uses plain types only, so baseline code can
// include it; simd.h (intrinsics) is for the tier sources.
#include <cstdint>
#include <vector>

namespace nibble {
struct Tensor;
struct BlockQ4_0;
struct BlockQ8_0;
struct BlockQ8_1;
struct BlockQ5_0;
struct BlockQ5_1;
struct BlockQ4_K;
struct BlockQ5_K;
struct BlockQ6_K;
struct BlockQ8_K;
struct QuantActs;
}  // namespace nibble

namespace nibble::simd {

// One 32-wide Q8_0 block of 8 tokens, interleaved for the tiled GEMM: qs[g][j] holds token j's quants
// 4g..4g+3, d[j] its scale as float, sum[j] the sum of its 32 quants. Tokens past n are zero.
struct alignas(32) BlockQ8_0x8 {
  int8_t qs[8][8][4];
  float d[8];
  int32_t sum[8];
};

// The instruction-set extensions selection looks at.
struct CpuFeatures {
  bool avx2 = false, fma = false, f16c = false, avxvnni = false;
};

// One tier's kernels. Every member has the contract documented for the function of the same name in simd.h.
struct Kernels {
  const char* name;  // "avx2+fma+f16c" or "avx2+fma+f16c+avxvnni": reported by kernel_set()
  void (*quantize_row_q8_0)(const float* x, BlockQ8_0* y, int64_t n);
  float (*dot_q4_0_q8_0)(const BlockQ4_0* w, const BlockQ8_0* a, int64_t n_blocks);
  float (*dot_q8_0_q8_0)(const BlockQ8_0* w, const BlockQ8_0* a, int64_t n_blocks);
  float (*dot_q5_0_q8_0)(const BlockQ5_0* w, const BlockQ8_0* a, int64_t n_blocks);
  float (*dot_q5_1_q8_1)(const BlockQ5_1* w, const BlockQ8_1* a, int64_t n_blocks);
  float (*dot_q4_K_q8_K)(const BlockQ4_K* w, const BlockQ8_K* a, int64_t n_blocks);
  float (*dot_q5_K_q8_K)(const BlockQ5_K* w, const BlockQ8_K* a, int64_t n_blocks);
  float (*dot_q6_K_q8_K)(const BlockQ6_K* w, const BlockQ8_K* a, int64_t n_blocks);
  void (*matmul_rows_k)(const Tensor& w, const BlockQ8_K* xq, float* y, int64_t n, int64_t row_begin, int64_t row_end);
  void (*gemm_k)(const Tensor& w, const BlockQ8_K* xq, float* y, int64_t n, int64_t row_begin, int64_t row_end);
  void (*matmul_rows_q)(const Tensor& w, const QuantActs& xq, float* y, int64_t n, int64_t row_begin, int64_t row_end);
  void (*matmul_rows_f)(const Tensor& w, const float* x, float* y, int64_t n, int64_t row_begin, int64_t row_end);
  void (*gemm_f)(const Tensor& w, const float* x, float* y, int64_t n, int64_t row_begin, int64_t row_end);
  void (*pack_q8_0_x8)(const BlockQ8_0* xq, int64_t n, int64_t nb, BlockQ8_0x8* xp);
  void (*gemm_q)(const Tensor& w, const BlockQ8_0x8* xp, float* y, int64_t n, int64_t row_begin, int64_t row_end);
  void (*embed_row_f16)(const uint16_t* row, float* out, int64_t d);
  void (*add)(float* x, const float* y, int64_t count);
  void (*rmsnorm)(const float* x, const Tensor& weight, float* y, int64_t n, float eps);
  void (*silu_mul)(float* gate, const float* up, int64_t count);
  void (*attention_heads)(const float* q, const float* k, const float* v, float* out, int64_t n, int64_t pos0,
                          int64_t n_heads, int64_t n_kv_heads, int64_t head_dim, int64_t head_begin, int64_t head_end,
                          float* scratch);
};

// The features of the CPU this runs on (all false when the build has no SIMD tiers).
CpuFeatures cpu_features();
// Whether CPUID reports AVX-VNNI: CPUID leaf 7, sub-leaf 1, EAX bit 4. max_leaf is the highest basic leaf
// (CPUID leaf 0 EAX); below 7, leaf 7 does not exist and leaf7_1_eax is meaningless, so the answer is false.
bool avxvnni_from_cpuid(unsigned max_leaf, unsigned leaf7_1_eax);
// The table to use for CPU features f and an NIBBLE_KERNELS value env (nullptr or "" = the strongest tier f
// allows; "scalar", "avx2", "avx2+vnni" or a full tier name = that tier). nullptr means the scalar kernels.
// Throws nibble::Error for an unknown value, a tier f does not allow, or any tier when none was built. Uncached.
const Kernels* select_for(const char* env, const CpuFeatures& f);
// select_for(getenv("NIBBLE_KERNELS"), cpu_features()), computed on first call and cached (an error is rethrown
// on every call). Thread-safe.
const Kernels* select();
// Every tier this CPU can run, weakest first (the scalar kernels are always available and not listed). For tests.
std::vector<const Kernels*> available();
// Name of the selected tier ("scalar" when select() is nullptr), shown by `nibble info` and recorded in bench rows.
const char* kernel_set();
// Throws nibble::Error(message). The tier sources call this instead of constructing Error themselves, so they
// define no inline copies of Error's members (see tools/check_simd_symbols.sh).
[[noreturn]] void throw_error(const char* message);

}  // namespace nibble::simd
