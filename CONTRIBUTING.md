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
  uv venv .venv && uv pip install --python .venv/bin/python -r requirements-ci.txt
  . .venv/bin/activate
  ```

  (`requirements-ci.txt` holds the exact versions CI uses.)

## Build and test

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release     # NIBBLE_NATIVE=ON by default: -march=native
cmake --build build -j
ctest --test-dir build -LE e2e --output-on-failure  # everything except the end-to-end tests
```

Useful variants (each in its own build directory):

| goal | configure with |
|---|---|
| release-shaped build (plain x86-64, kernels chosen at runtime) | `-DNIBBLE_NATIVE=OFF` |
| no SIMD kernels at all | `-DNIBBLE_SIMD=OFF` |
| no AVX-VNNI tier (what a compiler without `-mavxvnni` gets) | `-DNIBBLE_SIMD_VNNI=OFF` |
| force a kernel tier at run time | environment `NIBBLE_KERNELS=scalar`, `avx2` or `avx2+vnni` |
| AddressSanitizer + UBSan | `-DCMAKE_BUILD_TYPE=Debug -DNIBBLE_SANITIZE=address,undefined` (run tests with `setarch -R`) |
| ThreadSanitizer | `-DCMAKE_BUILD_TYPE=Debug -DNIBBLE_SANITIZE=thread` (with `setarch -R`) |
| clang | `CXX=clang++ cmake ...` |
| fuzzers (clang only) | `CXX=clang++ cmake -DNIBBLE_FUZZ=ON ...`, then `fuzz_gguf` / `fuzz_load` with `make_seed` output as the corpus |

A change that touches kernels or numerics should pass the unit tests in **Release, the SIMD-off build and the
sanitizer build**: bit-exactness bugs often show up in only one of them.
The SIMD kernel tests run once per tier the CPU supports (`avx2`, and `vnni` on an AVX-VNNI CPU), and
`NIBBLE_KERNELS=avx2 ctest ...` runs the rest of the suite on the non-VNNI tier.

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
cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DNIBBLE_BUILD_TESTS=OFF && run-clang-tidy.py -p build 'src/|cli/'

# Shell scripts, Python scripts
shellcheck tools/*.sh tests/*.sh
ruff check bench tools && ruff format --check bench tools && mypy
python -m unittest discover -s bench -p 'test_*.py'
python -m unittest discover -s tools/ci -p 'test_*.py'
```

Editors using clangd: link `build/compile_commands.json` (configure with `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`) to
`compile_commands.json` in the repository root; `.clangd` supplies the tier flags `simd.h` needs.

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
* Never commit secrets, models, golden binaries, or machine-local files (`.env*`, `*.local.md`, `.claude/settings.local.json`, ...).
* Open pull requests against `master`. Fill in the template: what and why, how it was tested, whether numerics or
  performance can change (attach before/after numbers if so). CI must be green; reviews look at correctness first, then
  tests, then clarity.
* Performance claims need numbers from `nibble bench` / `tools/bench_final.sh` on a quiet machine, with the protocol
  described in the README.

## CI and automation

Each file in `.github/workflows/` opens with a comment giving its reason, its triggers and how to run it by hand; the
README's "How nibble is tested" section summarizes what each one protects. Only the 10 checks of `ci.yml` are required
on `master`. Anything longer than a few lines of shell lives in typed, stdlib-only Python under `tools/ci/` (see its
README), which ruff and mypy cover and which has its own tests:

```sh
python -m unittest discover -s tools/ci -p 'test_*.py'
```

A workflow can be dispatched once its file is on `master`; `gh workflow run <file> --ref <branch>` then runs the
branch's copy, so a change to an existing workflow can be tried before it merges (nightly and portability open no
issue from another branch). A new workflow can only be dispatched after it is merged.

| workflow | runs | by hand |
|---|---|---|
| `ci.yml` (calls `package.yml`) | every PR and push to `master` | `gh workflow run ci.yml` |
| `pr-report.yml` (`pr-report-comment.yml` posts it) | PRs touching `src/`, `include/`, `cli/`, `CMakeLists.txt`, `tests/` | push to the PR |
| `fix-proof.yml` | PRs; acts on `fix:` PRs that touch `src/`, `include/` or `cli/` | push, edit the title or toggle a label |
| `codeql.yml` | PRs, pushes to `master`, Tuesdays 06:11 UTC | `gh workflow run codeql.yml` |
| `gitleaks.yml` | PRs and pushes (added commits), Wednesdays 02:29 UTC (full history) | `gh workflow run gitleaks.yml`; locally `gitleaks git --redact --no-banner` |
| `actionlint.yml` | PRs and pushes touching `.github/workflows/` or `.github/actions/` | `gh workflow run actionlint.yml`; locally `actionlint` |
| `labeler.yml` | PRs (area labels from `.github/labeler.yml`, generated by `tools/ci/labels.py`) | push to the PR; `python tools/ci/labels.py sync` updates the labels |
| `claude-review.yml`, `agent-approval.yml`, `automerge.yml` | PRs | push to the PR, or re-run the check |
| `nightly.yml` | daily 03:17 UTC | `gh workflow run nightly.yml [-f jobs=accuracy\|sanitizers-real-model\|fuzz\|trend-bench]` |
| `portability.yml` | Sundays 04:23 UTC | `gh workflow run portability.yml` |
| `scorecard.yml` | pushes to `master`, Mondays 05:37 UTC | `gh workflow run scorecard.yml` |
| `link-check.yml` | Mondays 06:41 UTC | `gh workflow run link-check.yml`; locally `lychee --config .lychee.toml README.md ARCHITECTURE.md CONTRIBUTING.md SECURITY.md docs` |
| `dashboard.yml` | after nightly, portability and ci finish on `master`, and daily at 09:47 and 21:47 UTC | `gh workflow run dashboard.yml` |
| `claude-triage.yml` | a new `ci-failure` issue | `gh workflow run claude-triage.yml -f issue=<number>` |
| `claude.yml` | `@claude` from the owner, a member or a collaborator | comment `@claude <request>` |
| `release.yml` | a `v*` tag | push the tag (see below) |

The nightly and portability jobs report through `report-job.yml` and `.github/actions/report-failure`
(`tools/ci/report_failure.py`): one open `ci-failure` issue per failing job, closed when the job is green again. A run
from another branch (`--ref`) opens no issue and saves no cache. The end-to-end jobs get their models from
`.github/actions/setup-models`, a single cache entry that only the nightly accuracy job saves.

### Bug-fix pull requests and fix-proof

`fix-proof.yml` checks the rule above that a bug fix comes with a test that fails without it. For a pull request with
`fix:` or `fix(...)` in its title or in a commit subject that touches `src/`, `include/` or `cli/`, it finds the
Catch2 `TEST_CASE`s the pull request adds or changes in `tests/*.cpp`, builds the merge base with the pull request's
`tests/` in Debug with ASan and UBSan, and runs each of those cases there:

* **proven** (green): at least one case fails on the merge base. A merge base that does not build with the new tests
  also counts, on the assumption that the tests use something the fix adds; the build failure is not checked further,
  so read the base build log in the job before taking that as proof.
* **does not reproduce**, **no test**, **not runnable**, **incomplete** (red): every case passes on the merge base,
  the pull request changes no test case, every selected case skipped or left no result, or no case failed before the
  time budget ran out. A green check always means proven (or waived, below).
* Cases it cannot run prove nothing: names built by a macro are reported as unresolved, and end-to-end cases as
  **not runnable** (it has no models; they skip).
* Only Catch2 `TEST_CASE`s in `tests/*.cpp` are selected. A fix whose reproducer is a CLI test (a `nibble_cli_test` in
  `CMakeLists.txt`) or an end-to-end case always reads **no test** or **not runnable**: say in the pull request which
  test reproduces the bug and how you saw it fail, and ask for the label.

If a fix cannot have a reproducing test that runs here (a build-only fix to code, a CLI or end-to-end reproducer, for
example), say why in the pull request and the owner adds the `no-test-needed` label: it turns **no test** and **not
runnable** green, and only those (**does not reproduce** and **incomplete** stay red). The check is not required yet.
The verdict and the cases are in the job summary.

### Releases

1. In a release-prep pull request, regenerate the changelog with [git-cliff](https://git-cliff.org) (`cliff.toml`):
   `git cliff --tag vX.Y.Z -o CHANGELOG.md`. It turns the "Unreleased" section into the dated `vX.Y.Z` section; read
   the diff, then merge.
2. Tag that merge commit and push the tag: `git tag -a vX.Y.Z -m "..." && git push origin vX.Y.Z`.
3. `release.yml` runs the whole `ci.yml` suite on the tag, fails if CHANGELOG.md has no `## [vX.Y.Z]` section, and
   otherwise creates a draft release with the tarballs, their SPDX SBOMs and attestations, and pushes the image to
   ghcr.io. The owner publishes the draft. The README shows how to verify a download with `gh attestation verify`.

## Working with coding agents

[AGENTS.md](AGENTS.md) holds the instructions coding agents follow in this repository (`CLAUDE.md` imports it), and
its "Claude on GitHub" section describes the Claude workflows. Claude works on `agent/<topic>` branches (a triage fix
on `agent/fix-issue-<n>`), and its pull requests carry the `agent` label: they are never auto-merged, and the owner
reviews and merges them.
Design specs and task-by-task plans for each milestone are in `docs/superpowers/`; they show how larger changes
are planned and executed. The same rules apply to agent-written changes: small Conventional Commits, tests first,
green CI, a pull request.

## Reporting bugs and security issues

Open an issue with the model file name (not the file), the command line, and the output. Because nibble parses
untrusted model files, parser crashes are security-relevant: see [SECURITY.md](SECURITY.md) for how to report those.
