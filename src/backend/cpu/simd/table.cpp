// The Kernels table of one SIMD tier (see dispatch.h). Compiled once per tier with NIBBLE_SIMD_TIER set, so it
// becomes nibble::simd::avx2::kTable and nibble::simd::vnni::kTable, each pointing at its own tier's functions.
#include "backend/cpu/simd/simd.h"

namespace nibble::simd::NIBBLE_SIMD_TIER {

#ifdef NIBBLE_HAVE_AVX2
extern const Kernels kTable;
const Kernels kTable = {
#ifdef NIBBLE_HAVE_AVXVNNI
    "avx2+fma+f16c+avxvnni",
#else
    "avx2+fma+f16c",
#endif
    quantize_row_q8_0,       dot_q4_0_q8_0, dot_q8_0_q8_0, dot_q5_0_q8_0, dot_q5_1_q8_1,
    dot_q4_K_q8_K,           dot_q5_K_q8_K, dot_q6_K_q8_K, matmul_rows_k, gemm_k,
    matmul_rows_q,           matmul_rows_f, gemm_f,        pack_q8_0_x8,  gemm_q,
    embed_row_f16,           add,           rmsnorm,       silu_mul,      attention_heads,
};
#endif

}  // namespace nibble::simd::NIBBLE_SIMD_TIER
