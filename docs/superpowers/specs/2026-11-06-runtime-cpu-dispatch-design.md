# Runtime CPU dispatch: design

Date: 2026-11-06. Status: approved in conversation; awaiting review of this written spec.

## Goal

One release artifact (tarball and container image) that runs on any x86-64 CPU without `SIGILL` and uses the
fastest kernels the CPU supports: AVX2+FMA+F16C with AVX-VNNI, AVX2+FMA+F16C without it, or scalar. Today the
kernel set is fixed at compile time, so the release ships an `x86-64-v3` tarball (AVX2, never VNNI, crashes on older
CPUs) and a scalar-only `x86-64` tarball.

Non-goals: new kernels (AVX-512, ARM), changing kernel numerics, changing the default `-march=native` for
development builds, a GPU backend.

## Success criteria

1. A build with `NIBBLE_NATIVE=OFF` (plain x86-64) runs on a CPU without AVX2 using the scalar kernels, and on an
   AVX2 or AVX2+VNNI CPU using the matching tier; `nibble info` reports the tier in use.
2. Every tier produces bit-identical results to today's build of the same kernels: unit tests compare every tier the
   machine supports against the scalar oracle, and the 50 end-to-end tests are unchanged.
3. `NIBBLE_KERNELS` forces a tier; an invalid or unsupported value is an error, never a silent fallback.
4. The release workflow publishes one `linux-x86_64` tarball and an image built for plain x86-64.
5. Decode and prefill speed match the `-march=native` build within noise (measured later, on a quiet machine; not
   a merge gate).

## Design

### Build (CMakeLists.txt)

- The kernel translation units (`src/backend/cpu/simd/{quant_avx2,kquant_avx2,gemm,matmul_f,ops}.cpp`) are compiled
  once per tier, as an object library per tier:
  - `avx2`: `-mavx2 -mfma -mf16c`, `NIBBLE_SIMD_TIER=avx2`
  - `vnni`: `-mavx2 -mfma -mf16c -mavxvnni`, `NIBBLE_SIMD_TIER=vnni`
  Both keep `-ffp-contract=off` (the existing bit-exactness rule) and the project's warning flags.
- Tiers are built only when `NIBBLE_SIMD=ON` and the target is x86-64; otherwise the build is scalar-only, as today.
- All other code uses the build's target flags: `-march=native` when `NIBBLE_NATIVE=ON` (development default),
  plain x86-64 when `OFF` (releases). The tier object libraries do not inherit `-march=native`, so a native build and
  a release build run identical kernel code.
- `NIBBLE_HAVE_AVX2` / `NIBBLE_HAVE_AVXVNNI` keep their meaning but are defined only inside tier TUs (from the
  tier's `-m` flags), so the existing `#ifdef`s inside the kernels need no change.

### Namespacing (ODR safety)

- Everything in `simd/simd.h`, including inline helpers (`hsum_ps`, `exp_ps`, loaders) and every kernel, is placed
  in `namespace nibble::simd::NIBBLE_SIMD_TIER` via a macro. Without this, an inline helper compiled with
  `-mavxvnni` and one compiled without it would share a symbol, and the linker could keep the VNNI copy for the
  AVX2 tier, which would crash an AVX2-only CPU.
- `simd.h` becomes internal to the tier TUs. `kernel_set()` moves out of it.
- A test (or a build check script run in CI) confirms no non-tier object file references a tier-only symbol and no
  weak symbol is shared between tiers: `nm` over the tier objects lists only `nibble::simd::<tier>::` symbols.

### Dispatch (new `simd/dispatch.h`, `simd/dispatch.cpp`)

- `struct Kernels { const char* name; <function pointers>; }` with one pointer per kernel entry point declared in
  `simd.h` today: the 12 the backend calls (`matmul_rows_f`, `matmul_rows_q`, `gemm_f`, `gemm_q`, `gemm_k`,
  `pack_q8_0_x8`, `quantize_row_q8_0`, `embed_row_f16`, `rmsnorm`, `add`, `silu_mul`, `attention_heads`) and the ones
  only tests call (`matmul_rows_k` and the seven block dot products). Tests reach every tier through this table, so
  no baseline-compiled file needs the intrinsics headers.
  `BlockQ8_0x8` (the packed activation layout, plain data) moves to `dispatch.h` so the backend can own the buffer.
- Each tier TU set defines `extern const Kernels kernels_<tier>`.
- `const Kernels* simd::select()`: decides once (function-local static) from `__builtin_cpu_init()` +
  `__builtin_cpu_supports("avx2"/"fma"/"f16c"/"avxvnni")`, honouring `NIBBLE_KERNELS`; returns `nullptr` for
  scalar. `simd::available()` returns the tiers this CPU can run (scalar implicit), for tests.
- `const char* simd::kernel_set()` returns the selected tier's name (`"scalar"`, `"avx2+fma+f16c"`,
  `"avx2+fma+f16c+avxvnni"`, the strings used today), used by `nibble info` and bench rows.

### Backend (cpu_backend.cpp / .h)

- `CpuBackend` stores `const simd::Kernels* k_ = simd::select()` at construction.
- The 8 `#ifdef NIBBLE_HAVE_AVX2` blocks become `if (k_) … else …` with the scalar `ref_ops` path unchanged.
  `xp_` (packed activations) is reserved only when `k_` is set.
- Selection happens on the calling thread at construction, so pool tasks never throw for it.

### Override and errors

- `NIBBLE_KERNELS` accepts `scalar`, `avx2`, `avx2+vnni` (also the full names `kernel_set()` prints).
- An unknown value, or a tier the CPU does not support, throws `nibble::Error` from `select()` (first backend
  construction) naming the value and the valid choices for this CPU. Read with `getenv` once (existing NOLINT
  pattern for `concurrency-mt-unsafe`).

### Releases

- `package.yml`: one binary job, `linux-x86_64`, `NIBBLE_NATIVE=OFF`, no `-march`; the smoke test runs
  `nibble info` on the test GGUF and checks the reported tier is a known one. Asset names change from
  `-linux-x86_64-avx2` / `-portable` to `-linux-x86_64`.
- `Dockerfile`: `MARCH` defaults to `x86-64`; the comment about Haswell-or-newer and `SIGILL` is replaced.
- README install/supported-CPU text, ARCHITECTURE.md (kernel selection section), CONTRIBUTING.md (build variants
  table: `NIBBLE_SIMD=OFF` still exists; `NIBBLE_KERNELS` documented) are updated.

### Tests

- Kernel tests (`test_simd`, `test_gemm`, `test_ops`, `test_quant` SIMD sections) run against every tier in
  `simd::available()` through the `Kernels` table instead of `#ifdef`-selected direct calls, and still compare
  bitwise (or within the stated tolerance for float sums) against the scalar oracle.
- New `test_dispatch.cpp`: `select()` returns the best available tier; each valid `NIBBLE_KERNELS` value selects
  it (when available); an unknown value and an unavailable tier throw with the valid choices in the message.
  Selection is cached, so these tests call an uncached helper (`select_for(const char* env, cpu_features)`) that
  takes the environment value and a feature mask, which also lets them cover CPUs the test machine is not.
- CI: the "expected kernel set" check in `ci.yml` matches the runtime tier (`avx2*` on the native and portable
  jobs, since runners have AVX2; `scalar` for the `NIBBLE_SIMD=OFF` job). The portable gcc job becomes the
  release-shaped build. A CI job runs the unit tests once more with `NIBBLE_KERNELS=avx2` so the non-VNNI tier is
  exercised even on VNNI runners.
- The alloc tests and end-to-end tests are unchanged and must pass.

## Risks

- **Speed.** Non-kernel code in release builds loses `-march=native` tuning; kernels lose nothing. Verified by a
  quiet-machine benchmark after merge.
- **Toolchain.** `-mavxvnni` needs GCC 11+/clang 12+; CI uses GCC 13 and clang 18. Building a tier is skipped with a
  CMake message if the compiler rejects its flags.
- **Pointer-call overhead.** One indirect call per matmul/op call, each doing thousands of operations: negligible.
