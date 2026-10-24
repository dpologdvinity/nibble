# Contributing to nibble

Thanks for helping. This guide covers the local workflow, the checks CI runs, and the conventions for commits and pull
requests. Start with [ARCHITECTURE.md](ARCHITECTURE.md) if you have not read it: the numerical-fidelity rules there
explain several things below that might otherwise look fussy.

## Prerequisites

* g++ 13 or clang 18 (C++20), CMake ≥ 3.25, Python 3 (plots and scripts), git.
* Linux x86-64 for the optimized kernels (other targets build and run the scalar kernels).
* Optional tools used by the checks: `clang-format`, `clang-tidy`, `shellcheck`, `ruff`, `mypy`, `gcovr`.
  A convenient way to get them without touching system Python:

  ```sh
  uv venv .venv && uv pip install --python .venv/bin/python clang-format==23.1.3 clang-tidy==23.1.3 \
      shellcheck-py ruff mypy gcovr
  . .venv/bin/activate
  ```

  (`pip install` into a virtual environment works equally well; the pinned versions match CI.)

## Build and test

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release     # NIBBLE_NATIVE=ON by default: -march=native
cmake --build build -j
ctest --test-dir build -LE e2e --output-on-failure  # everything except the end-to-end tests
```

Useful variants (each in its own build directory):

| goal | configure with |
|---|---|
| portable scalar build | `-DNIBBLE_NATIVE=OFF` |
| scalar kernels on a native build | `-DNIBBLE_SIMD=OFF` |
| AddressSanitizer + UBSan | `-DCMAKE_BUILD_TYPE=Debug -DNIBBLE_SANITIZE=address,undefined` (run tests with `setarch -R`) |
| ThreadSanitizer | `-DCMAKE_BUILD_TYPE=Debug -DNIBBLE_SANITIZE=thread` (with `setarch -R`) |
| clang | `CXX=clang++ cmake ...` |
| fuzzers (clang only) | `CXX=clang++ cmake -DNIBBLE_FUZZ=ON ...`, then `fuzz_gguf` / `fuzz_load` with `make_seed` output as the corpus |

A change that touches kernels or numerics should pass the unit tests in **Release, the SIMD-off build and the
sanitizer build**: bit-exactness bugs often show up in only one of them.

### End-to-end tests

The `e2e` tests compare against llama.cpp on real models and need multi-gigabyte files, so they run locally:

```sh
tools/download_models.sh                 # Qwen2.5-0.5B F16
tools/convert_hf.sh llama32-1b; tools/convert_hf.sh tinyllama-1.1b
tools/build_llama_cpp.sh && tools/build_llama_cpp.sh --generic
tools/make_quants.sh                     # Q8_0, Q4_0 and the Q4_K_M / Q5_K_M / Q6_K mixes
tools/make_golden.sh qwen2.5-0.5b-f16    # repeat for each model/quantization name (see the script)
ctest --test-dir build -L e2e
```

Tests skip themselves when a model or golden file is missing. Do not commit anything under `models/` or
`third_party/`. If you change anything that can move numerics, run the end-to-end suite and compare the
`min cosine` WARN lines before and after; kernels that are meant to be bit-identical must leave them unchanged.

## Code quality checks

CI runs all of these on every push and pull request (`.github/workflows/ci.yml`); run them locally first.

```sh
# C++ formatting (Google style, 120 columns; see .clang-format). Check only, or fix in place:
git ls-files '*.cpp' '*.h' | grep -v '^third_party' | xargs clang-format --dry-run -Werror
git ls-files '*.cpp' '*.h' | grep -v '^third_party' | xargs clang-format -i

# C++ static analysis on the library and CLI (needs a configured build directory):
cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && run-clang-tidy.py -p build 'src/|cli/'

# Shell scripts, Python scripts
shellcheck tools/*.sh
ruff check bench tools && ruff format --check bench tools && mypy bench tools
```

The build itself uses `-Wall -Wextra -Wpedantic -Werror`; warnings are errors.

## Writing code

* **Comments and docs are part of the change.** Every file starts with a comment saying what it is for; every
  type, function and non-obvious member has a comment stating its contract (inputs, outputs, errors, ownership,
  threading). Explain *why*, especially for numerics. Keep `ARCHITECTURE.md` and `README.md` true.
* **Tests come with the change.** New logic gets tests; a bug fix gets a test that fails without the fix. Prefer
  tests that pin behaviour (exact bits, exact messages) over tests that only exercise code.
* **Errors.** Anything caused by input (files, parameters) throws `nibble::Error` with a message that names the culprit
  and passes file-derived text through `escape_control`. Internal invariants use `NIBBLE_DCHECK`. Never read beyond what
  the file length proves.
* **No allocation on the decode path.** Allocate in constructors / `Backend::reserve`; `alloc_tests` enforces it.
* **Numerics.** Kernels that must be bit-identical to the scalar code keep its operation order, and their translation
  units stay in the `-ffp-contract=off` list in `CMakeLists.txt`.
* **Adding a quantized type** touches: `tensor.h/.cpp` (DType, sizes), `gguf.cpp` (accept the id), `quant.h/.cpp`
  (block, dequantize, activation pairing, scalar dot), the AVX2 kernel, `ref_ops.cpp`/`cpu_backend.cpp` dispatch,
  golden generation in `tools/quant_vectors` and `tests/quant_golden_gen.h`, tests at every level, `make_quants.sh`
  and the README support table.

## Commits and pull requests

* [Conventional Commits](https://www.conventionalcommits.org/): `type(scope): summary`, imperative mood, summary ≤ 72
  characters. Types in use: `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `build`, `ci`, `style`, `chore`.
  Describe *why* in the body when it is not obvious.
* Keep commits small and focused: one logical change each (a mechanical reformat is its own commit and is listed in
  `.git-blame-ignore-revs`).
* Never commit secrets, models, golden binaries, or local tooling files (`.claude/`, `CLAUDE.md`, `.env`, ...).
* Open pull requests against `master`. Fill in the template: what and why, how it was tested, whether numerics or
  performance can change (attach before/after numbers if so). CI must be green; reviews look at correctness first, then
  tests, then clarity.
* Performance claims need numbers from `nibble bench` / `tools/bench_final.sh` on a quiet machine, with the protocol
  described in the README.

## Reporting bugs and security issues

Open an issue with the model file name (not the file), the command line, and the output. Because nibble parses
untrusted model files, parser crashes are security-relevant: see [SECURITY.md](SECURITY.md) for how to report those.
