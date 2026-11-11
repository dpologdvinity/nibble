# Runtime CPU Dispatch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One x86-64 binary that picks the AVX2+VNNI, AVX2 or scalar kernels at startup, so the release ships a single tarball and image that never dies with `SIGILL`.

**Architecture:** The five kernel translation units are compiled once per tier (`avx2`, `vnni`) into object libraries, each inside its own namespace `nibble::simd::<tier>`. Each tier exports a `Kernels` table of function pointers; `simd::select()` picks one table once (CPU features + `NIBBLE_KERNELS`), and `CpuBackend` calls through it, using the existing scalar `ref_ops` path when there is no table.

**Tech Stack:** C++20, CMake ≥ 3.25 (object libraries), GCC 13 / clang 18 (`-mavxvnni`, `__builtin_cpu_supports`), Catch2, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-11-06-runtime-cpu-dispatch-design.md` (read it first).

## Global Constraints

- Tiers: `avx2` = `-mavx2 -mfma -mf16c`; `vnni` = `-mavx2 -mfma -mf16c -mavxvnni`. Both keep `-ffp-contract=off` and the warning flags `-Wall -Wextra -Wpedantic -Werror` and the sanitizer flags; neither gets `-march=native`.
- Tier names reported by `kernel_set()` (unchanged strings): `"scalar"`, `"avx2+fma+f16c"`, `"avx2+fma+f16c+avxvnni"`.
- `NIBBLE_KERNELS` accepts `scalar`, `avx2`, `avx2+vnni`, plus the two full names above; empty or unset = best available.
- Tiers are built only when `NIBBLE_SIMD=ON` and `CMAKE_SYSTEM_PROCESSOR` matches `x86_64|AMD64`. Otherwise everything is scalar and `select()` returns `nullptr`.
- No tier translation unit may define a non-local symbol outside `nibble::simd::<tier>::` (no inline/template instantiations of shared code, no `nibble::Error` construction): the linker could otherwise keep an AVX copy for baseline code.
- Kernel numerics do not change: every tier must stay bit-identical to today's kernels; the 50 end-to-end tests must pass unchanged.
- Conventional Commits, one logical change per commit; `docs/superpowers/`, `CLAUDE.md`, `.claude/` are never committed.
- Work on branch `feat/runtime-dispatch` from `master`; merge by PR only (branch protection requires the 10 CI checks).

## Review Focus

1. **A CPU with AVX2 but no FMA or F16C** (some VMs/emulators mask flags): must select scalar, not `avx2`. Test: `select_for` with `{avx2=true, fma=false, f16c=true, avxvnni=false}` returns `nullptr` (Task 1).
2. **AVX-VNNI reported without AVX2** (impossible on real hardware, possible under a misconfigured hypervisor): must not select `vnni`. Test: `{avx2=false, fma=true, f16c=true, avxvnni=true}` → `nullptr` (Task 1).
3. **`NIBBLE_KERNELS` with surrounding whitespace or different case** (`" AVX2"`): treated as invalid and the error lists the valid values, never silently auto-selects. Test in Task 1.
4. **A weak symbol leaking from a tier object** (e.g. an `Error` destructor or an `fp16_to_fp32` copy compiled with `-mavx2`): the symbol check fails the build. Test: Task 1's `tools/check_simd_symbols.sh` run in CI (Task 4) and once by hand against a deliberately leaking stub.
5. **Release build on a scalar-only CPU**: `nibble info` must work and report `scalar`. Test: `NIBBLE_KERNELS=scalar build/nibble info tests/data/small.gguf` prints `kernels: scalar` in the release-shaped CI job (Task 4).

## File Structure

| file | responsibility |
|---|---|
| `src/backend/cpu/simd/dispatch.h` (new) | `BlockQ8_0x8`, `CpuFeatures`, `Kernels`, `select_for`, `select`, `available`, `cpu_features`, `kernel_set`, `throw_error`. Plain types only: includable from baseline code. |
| `src/backend/cpu/simd/dispatch.cpp` (new, baseline flags) | CPU detection, env parsing, selection cache, `throw_error`. |
| `src/backend/cpu/simd/table.cpp` (new, per tier) | `const Kernels kTable` for the tier. |
| `src/backend/cpu/simd/simd.h` | Internal to tier TUs: intrinsic helpers and kernel declarations inside `namespace nibble::simd::NIBBLE_SIMD_TIER`. `kernel_set()` and `BlockQ8_0x8` move out. |
| `src/backend/cpu/simd/{quant_avx2,kquant_avx2,gemm,matmul_f,ops}.cpp` | Unchanged kernels; namespace becomes the tier namespace; `throw Error(...)` becomes `throw_error(...)`. |
| `src/backend/cpu/cpu_backend.{h,cpp}` | Holds `const simd::Kernels* k_`; runtime branch instead of `#ifdef`. |
| `CMakeLists.txt` | `nibble_warnings` / `nibble_arch` split, `nibble_simd_tier()` function, tier object libraries linked into `nibble`, fuzz targets and helper-test executables. |
| `tools/check_simd_symbols.sh` (new) | Fails if a tier object defines a non-local symbol outside its namespace. |
| `tests/test_dispatch.cpp` (new) | Selection and override behaviour. |
| `tests/simd_helpers_test.cpp` (new, per tier) | Tests of intrinsic helpers moved out of `test_simd.cpp`/`test_ops.cpp`/`test_quant.cpp`. |

---

### Task 1: Tier build, namespaces, dispatch table, backend switch

**Files:**
- Create: `src/backend/cpu/simd/dispatch.h`, `src/backend/cpu/simd/dispatch.cpp`, `src/backend/cpu/simd/table.cpp`, `tools/check_simd_symbols.sh`, `tests/test_dispatch.cpp`
- Modify: `CMakeLists.txt:22-72` (flags split, tiers, fuzz targets at `:264-279`), `src/backend/cpu/simd/simd.h`, the five kernel `.cpp` files, `src/backend/cpu/cpu_backend.h:86`, `src/backend/cpu/cpu_backend.cpp` (8 `#ifdef NIBBLE_HAVE_AVX2` blocks at lines 46, 119, 175, 213, 234, 243, 270, 293), `cli/cmd_info.cpp:91`, `cli/cmd_bench.cpp:129`, `ARCHITECTURE.md` (kernel-selection paragraph)

**Interfaces:**
- Produces (in `namespace nibble::simd`, header `backend/cpu/simd/dispatch.h`):
  - `struct alignas(32) BlockQ8_0x8` — moved unchanged from `simd.h`.
  - `struct CpuFeatures { bool avx2 = false, fma = false, f16c = false, avxvnni = false; };`
  - `struct Kernels { const char* name; ... };` with these members, signatures exactly as in today's `simd.h`:
    `quantize_row_q8_0`, `dot_q4_0_q8_0`, `dot_q8_0_q8_0`, `dot_q5_0_q8_0`, `dot_q5_1_q8_1`, `dot_q4_K_q8_K`, `dot_q5_K_q8_K`, `dot_q6_K_q8_K`, `matmul_rows_k`, `gemm_k`, `matmul_rows_q`, `matmul_rows_f`, `gemm_f`, `pack_q8_0_x8`, `gemm_q`, `embed_row_f16`, `add`, `rmsnorm`, `silu_mul`, `attention_heads` (each a function-pointer member of the same name).
  - `CpuFeatures cpu_features();` — `__builtin_cpu_init()` then `__builtin_cpu_supports("avx2"|"fma"|"f16c"|"avxvnni")`; all false when tiers are not built.
  - `const Kernels* select_for(const char* env, const CpuFeatures& f);` — uncached; `nullptr` means scalar; throws `nibble::Error`.
  - `const Kernels* select();` — `select_for(getenv("NIBBLE_KERNELS"), cpu_features())` computed once (function-local static).
  - `std::vector<const Kernels*> available();` — the tiers `cpu_features()` allows, weakest first (scalar not included).
  - `const char* kernel_set();` — `select() ? select()->name : "scalar"`.
  - `[[noreturn]] void throw_error(const char* message);` — throws `nibble::Error(message)`; defined in `dispatch.cpp` so tier TUs never construct `Error`.
- Produces (per tier): `extern const nibble::simd::Kernels nibble::simd::<tier>::kTable;` with `name` = `"avx2+fma+f16c"` (avx2) or `"avx2+fma+f16c+avxvnni"` (vnni).
- Selection rule: `vnni` needs avx2 && fma && f16c && avxvnni; `avx2` needs avx2 && fma && f16c; auto picks the strongest allowed.
- Error messages (exact prefixes, tests match them):
  - unknown value: `NIBBLE_KERNELS="<value>" is not one of: scalar, avx2, avx2+vnni`
  - unsupported tier: `NIBBLE_KERNELS="<value>" needs a CPU with <AVX2, FMA and F16C | AVX-VNNI>; this CPU supports: <comma-separated list starting with scalar>`
  - tiers not built: `NIBBLE_KERNELS="<value>": this build has no SIMD kernels (NIBBLE_SIMD=OFF or not x86-64)`

- [ ] **Step 1: Write the failing dispatch tests** in `tests/test_dispatch.cpp` (tag `[dispatch]`):

```cpp
using F = nibble::simd::CpuFeatures;
const F none{}, avx2{true, true, true, false}, vnni{true, true, true, true};
TEST_CASE("select_for picks the strongest tier the CPU allows", "[dispatch]") {
  CHECK(select_for(nullptr, none) == nullptr);
  CHECK(std::string(select_for(nullptr, avx2)->name) == "avx2+fma+f16c");
  CHECK(std::string(select_for("", vnni)->name) == "avx2+fma+f16c+avxvnni");
  CHECK(select_for(nullptr, F{true, false, true, false}) == nullptr);   // no FMA
  CHECK(select_for(nullptr, F{false, true, true, true}) == nullptr);    // VNNI without AVX2
}
TEST_CASE("NIBBLE_KERNELS forces a tier by short or full name", "[dispatch]") {
  CHECK(select_for("scalar", vnni) == nullptr);
  CHECK(std::string(select_for("avx2", vnni)->name) == "avx2+fma+f16c");
  CHECK(std::string(select_for("avx2+fma+f16c", vnni)->name) == "avx2+fma+f16c");
  CHECK(std::string(select_for("avx2+vnni", vnni)->name) == "avx2+fma+f16c+avxvnni");
  CHECK(std::string(select_for("avx2+fma+f16c+avxvnni", vnni)->name) == "avx2+fma+f16c+avxvnni");
}
TEST_CASE("bad NIBBLE_KERNELS values throw with the valid choices", "[dispatch]") {
  CHECK_THROWS_WITH(select_for("avx512", vnni), StartsWith("NIBBLE_KERNELS=\"avx512\" is not one of: scalar, avx2, avx2+vnni"));
  CHECK_THROWS_WITH(select_for(" AVX2", vnni), StartsWith("NIBBLE_KERNELS=\" AVX2\" is not one of"));
  CHECK_THROWS_WITH(select_for("avx2+vnni", avx2), ContainsSubstring("needs a CPU with AVX-VNNI; this CPU supports: scalar, avx2"));
  CHECK_THROWS_WITH(select_for("avx2", none), ContainsSubstring("needs a CPU with AVX2, FMA and F16C; this CPU supports: scalar"));
}
TEST_CASE("kernel_set reports the selected tier", "[dispatch]") {
  const auto* k = select();
  CHECK(std::string(kernel_set()) == (k ? k->name : "scalar"));
}
```

Wrap the cases that dereference a tier in `if (!tiers_built()) SKIP("no SIMD tiers in this build");` where `tiers_built()` is `select_for(nullptr, vnni) != nullptr`; in that case also check `select_for("avx2", vnni)` throws the "this build has no SIMD kernels" message.

- [ ] **Step 2: Run to verify it fails**

Run: `cmake --build build -j && build/nibble_tests "[dispatch]"`
Expected: compile error (`dispatch.h` not found).

- [ ] **Step 3: CMake.** Split `nibble_flags` into `nibble_warnings` (warnings, `-Werror`, sanitizer compile/link flags, `NIBBLE_SIMD_ENABLED`) and `nibble_arch` (`-march=native` and the clang `-Wno-invalid-feature-combination` when `NIBBLE_NATIVE`); keep `nibble_flags` as an INTERFACE target linking both so other targets are unchanged. Add `function(nibble_simd_tier name tier)` that creates `OBJECT` library `${name}` from the five kernel files plus `table.cpp`, with `-ffp-contract=off`, `NIBBLE_SIMD_TIER=${tier}`, the tier's `-m` flags from Global Constraints, include dir `src`, linked to `nibble_warnings` only. Exclude `src/backend/cpu/simd/{quant_avx2,kquant_avx2,gemm,matmul_f,ops,table}.cpp` from `NIBBLE_SOURCES`; create tiers `nibble_simd_avx2` and `nibble_simd_vnni` when the Global Constraints condition holds, add their `$<TARGET_OBJECTS:...>` to `nibble`, and define `NIBBLE_SIMD_TIERS=1` on `nibble` (PRIVATE). Fuzz targets add the same objects (create fuzz-instrumented tiers `nibble_simd_avx2_fuzz`/`_vnni_fuzz` with the fuzz flags). Add `tests/test_dispatch.cpp` to the test sources if they are listed explicitly.

- [ ] **Step 4: Namespaces.** In `simd.h`, guard everything with `#ifdef NIBBLE_SIMD_TIER` and open `namespace nibble::simd::NIBBLE_SIMD_TIER`; `NIBBLE_HAVE_AVX2`/`NIBBLE_HAVE_AVXVNNI` stay derived from `__AVX2__ && __FMA__ && __F16C__` / `__AVXVNNI__` inside that guard; remove `kernel_set()` and `BlockQ8_0x8` (include `dispatch.h` instead). In the five kernel files replace `namespace nibble::simd {` with `namespace nibble::simd::NIBBLE_SIMD_TIER {` and every `throw Error(...)` with `throw_error(...)` (same message text). `table.cpp` defines `kTable` for the tier, each member pointing at the tier's function.

- [ ] **Step 5: `dispatch.cpp`.** Implement the Interfaces block. Parse env exactly (no trimming, case-sensitive). `getenv` gets the project's existing `// NOLINT(concurrency-mt-unsafe)` with a reason. When `NIBBLE_SIMD_TIERS` is not defined, `select_for` returns `nullptr` for unset/empty/`scalar` and throws the "no SIMD kernels" message otherwise; `cpu_features()` returns all-false.

- [ ] **Step 6: Backend and CLI.** `CpuBackend` gains `const simd::Kernels* k_` initialised from `simd::select()` in the constructor (so a bad `NIBBLE_KERNELS` throws there, on the calling thread). Each `#ifdef NIBBLE_HAVE_AVX2 … #else … #endif` becomes `if (k_) { …k_->fn(...)… } else { …existing scalar code… }`; the `xp_` packed-scratch reservation and the `need_pack` branch run only when `k_`. `cmd_info.cpp` and `cmd_bench.cpp` include `dispatch.h` and call `simd::kernel_set()`.

- [ ] **Step 7: Symbol check.** `tools/check_simd_symbols.sh <object files...>`: for each file, `nm -C --defined-only` lines whose type is not lowercase (global/weak) and whose name does not start with `nibble::simd::<tier>::` (tier taken from the object library's directory name), ignoring `DW.ref.__gxx_personality_v0`; print offenders and exit 1 if any. Add a CTest test `simd_symbols` that runs it on `$<TARGET_OBJECTS:nibble_simd_avx2>` and `_vnni` (use `COMMAND_EXPAND_LISTS`). Verify it catches a leak by temporarily adding `inline int leak() { return 1; } int use_leak() { return leak(); }` to `table.cpp` outside the namespace: the test must fail; remove it.

- [ ] **Step 8: Run tests**

Run: `cmake -S . -B build && cmake --build build -j && ctest --test-dir build -LE e2e -j6 --output-on-failure`
Expected: all pass, including `[dispatch]`, `simd_symbols`, the alloc tests and `cpu backend prefill matmul matches scalar for all dtypes`. `build/nibble info build/tests/data/small.gguf | grep kernels:` prints `avx2+fma+f16c+avxvnni` on this machine; with `NIBBLE_KERNELS=avx2` prints `avx2+fma+f16c`; with `NIBBLE_KERNELS=scalar` prints `scalar`. (Kernel unit tests that still use `#ifdef NIBBLE_HAVE_AVX2` are compiled out until Task 2; that is expected.)

- [ ] **Step 9: Release-shaped and scalar builds**

Run: `cmake -S . -B build-rel -DCMAKE_BUILD_TYPE=Release -DNIBBLE_NATIVE=OFF && cmake --build build-rel -j && ctest --test-dir build-rel -LE e2e -j6` and the same with `-DNIBBLE_SIMD=OFF` in `build-scalar`.
Expected: all pass; `build-rel/nibble info` reports the AVX2+VNNI tier; `build-scalar` reports `scalar` and `NIBBLE_KERNELS=avx2 build-scalar/nibble info …` exits 1 with the "no SIMD kernels" message.

- [ ] **Step 10: Commit** (three commits)

```bash
git add CMakeLists.txt src/backend/cpu/simd tools/check_simd_symbols.sh && git commit -m "build(simd): compile the kernels once per CPU tier in their own namespaces"
git add src/backend/cpu/cpu_backend.h src/backend/cpu/cpu_backend.cpp cli/cmd_info.cpp cli/cmd_bench.cpp tests/test_dispatch.cpp && git commit -m "feat(cpu): choose the kernel tier at runtime, with a NIBBLE_KERNELS override"
git add ARCHITECTURE.md && git commit -m "docs: describe runtime kernel selection"
```

---

### Task 2: Kernel tests run on every available tier

**Files:**
- Create: `tests/simd_helpers_test.cpp`
- Modify: `tests/test_simd.cpp`, `tests/test_gemm.cpp`, `tests/test_ops.cpp:118-150,184-260`, `tests/test_quant.cpp:803-840`, `CMakeLists.txt` (helper-test executables)

**Interfaces:**
- Consumes: `simd::available()`, `simd::Kernels` members (Task 1).
- Produces: test helper `for_each_tier(fn)` in `tests/quant_test_util.h`: calls `fn(const simd::Kernels&)` inside `DYNAMIC_SECTION(k->name)` for each `available()` entry; if there are none, `SKIP("no SIMD tier on this CPU/build")`.

- [ ] **Step 1: Convert the kernel tests.** Every test that calls `simd::<kernel>(…)` under `#ifdef NIBBLE_HAVE_AVX2` instead calls `k.<kernel>(…)` inside `for_each_tier`, with the same inputs and the same bitwise/tolerance assertions; remove those `#ifdef`s. Replace the test "kernel_set names the active macros" (test_simd.cpp:17) with a check that `simd::kernel_set()` is one of the three names in Global Constraints.

- [ ] **Step 2: Move the intrinsic-helper tests** (`hsum_ps and hsum_epi32 are exact on integers`, `load_f16x8 equals fp16_to_fp32 for all 65536 inputs`, `mul_sum_i8 maddubs and dispatch variants agree with integer dot`, the `exp_ps` cases from test_ops.cpp, and any other case that names an `_mm256_` intrinsic or a `simd.h` helper) into `tests/simd_helpers_test.cpp`, written against `nibble::simd::NIBBLE_SIMD_TIER::`. Build it twice as executables `nibble_simd_helpers_avx2` / `_vnni` with the tier's flags (via `nibble_simd_tier`'s flag list, plus Catch2 and `nibble`), each with its own `main` that returns 77 before running Catch when `cpu_features()` lacks the tier; register both with `add_test` and `SKIP_RETURN_CODE 77`.

- [ ] **Step 3: Run**

Run: `cmake --build build -j && ctest --test-dir build -LE e2e -j6 --output-on-failure && build/nibble_tests "[simd]" --list-tests | wc -l`
Expected: all pass; the `[simd]` count is the same as on `master` minus the moved helper cases; each converted case shows one section per tier (`build/nibble_tests "simd q4_0/q8_0 dots equal scalar bitwise" -s | grep -c "avx2+fma+f16c"` ≥ 1). Under `NIBBLE_KERNELS=avx2` the same suite passes.

- [ ] **Step 4: Sanitizer and clang builds**

Run: `CXX=clang++ cmake -S . -B build-clang && cmake --build build-clang -j && ctest --test-dir build-clang -LE e2e -j6`, then a `-DCMAKE_BUILD_TYPE=Debug -DNIBBLE_SANITIZE=address,undefined` build and a `thread` build, tests under `setarch -R`.
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tests CMakeLists.txt && git commit -m "test(simd): run the kernel tests on every tier the CPU supports"
```

---

### Task 3: End-to-end parity

**Files:** none (verification only; fix in the owning task's files if anything differs).

- [ ] **Step 1:** `ctest --test-dir build -L e2e -j2 --timeout 3000` on the `NIBBLE_NATIVE=ON` build. Expected: 50/50 pass.
- [ ] **Step 2:** `NIBBLE_KERNELS=avx2 ctest --test-dir build-rel -L e2e -j2 --timeout 3000` on the release-shaped build. Expected: 50/50 pass, and the per-file `min cosine` WARN lines equal those of Step 1 (the tiers are bit-identical).
- [ ] **Step 3:** If any line differs, stop: that is a numerics change and must be traced, not tolerated.

---

### Task 4: Release, CI and docs

**Files:**
- Modify: `.github/workflows/package.yml:24-60`, `Dockerfile:1-30`, `.github/workflows/ci.yml` (test matrix `kernels:` values and the "expected kernel set" step at ~:181-188), `README.md` (install/download, supported CPUs, `nibble info` output), `CONTRIBUTING.md` (build-variants table: add `NIBBLE_KERNELS`), `SECURITY.md` only if it names the avx2 tarball

**Interfaces:**
- Consumes: `NIBBLE_KERNELS`, `kernel_set()` names (Task 1).
- Produces: release asset names `nibble-<version>-linux-x86_64.tar.gz` (+ `.sha256`).

- [ ] **Step 1: package.yml.** Replace the two-entry matrix with one job `binary (linux-x86_64)`: `-DNIBBLE_NATIVE=OFF`, no `CMAKE_CXX_FLAGS` `-march`. Smoke test: `build/nibble --version`, `build/nibble --help`, and `build/nibble info` on the test GGUF (build `make_test_gguf` too, or copy the CI's fixture step) with `grep -E '^kernels: (scalar|avx2\+fma\+f16c(\+avxvnni)?)$'`, plus `NIBBLE_KERNELS=scalar` printing `kernels: scalar`.
- [ ] **Step 2: Dockerfile.** `ARG MARCH=x86-64`; replace the comment about x86-64-v3, Haswell and `SIGILL` with one saying the image runs on any x86-64 CPU and picks its kernels at startup. Configure with `-DNIBBLE_NATIVE=OFF` and no `-march` (drop `MARCH` entirely if nothing else uses it). The package.yml image smoke test also runs `docker run --rm -e NIBBLE_KERNELS=scalar nibble:ci --version` (must exit 0).
- [ ] **Step 3: ci.yml.** In the test matrix, `kernels:` for the portable gcc job becomes `avx2` (it now dispatches); the scalar-kernels job (`NIBBLE_SIMD=OFF`) stays `scalar`. Add a step to the portable gcc job: `NIBBLE_KERNELS=avx2 ctest --test-dir build -LE e2e --output-on-failure --parallel "$(nproc)"` so the non-VNNI tier runs even on VNNI runners, and `NIBBLE_KERNELS=scalar build/nibble info build/tests/data/small.gguf | grep -x 'kernels: scalar'`. Update the header comment's job list.
- [ ] **Step 4: Docs.** README: one download (`linux-x86_64`), "runs on any x86-64 CPU; uses AVX-VNNI or AVX2 kernels when present", `NIBBLE_KERNELS` described next to `nibble info`. CONTRIBUTING: build-variants table row for forcing a tier. Run `actionlint`, `shellcheck tools/*.sh`, the clang-format check.
- [ ] **Step 5: Owner ledger.** Add to `STEPS-FOR-OWNER.md` (git-excluded): run the paired benchmark on a quiet machine comparing the release build against the `-march=native` build (spec criterion 5).

- [ ] **Step 5: Commit** (two commits)

```bash
git add .github/workflows Dockerfile && git commit -m "ci: ship one x86-64 release binary that dispatches at runtime"
git add README.md CONTRIBUTING.md && git commit -m "docs: one release binary, NIBBLE_KERNELS"
```

- [ ] **Step 6: PR and CI.** Push `feat/runtime-dispatch`, open a PR, wait for all 10 required checks and the package jobs to pass. Then request the whole-branch review (superpowers:requesting-code-review) before merging; fix Critical/Important findings test-first.
- [ ] **Step 7: Release check after merge.** Tag `v0.8.0-rc.1`, confirm the release workflow publishes one `linux-x86_64` tarball whose checksum verifies and whose `nibble info` reports a tier, and the image prints the commit; delete the rc release, tag and image version; then tag `v0.8.0` and publish.

---

## Not in this plan

- The speed comparison against the `-march=native` build (spec success criterion 5) needs a quiet machine; it is recorded as an owner/quiet-machine step in `STEPS-FOR-OWNER.md` during Task 4.
