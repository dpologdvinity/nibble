# AGENTS.md

Instructions for coding agents (Claude Code, Codex and others) working in this repository. `CLAUDE.md` is a one-line
stub that imports this file. Humans: [CONTRIBUTING.md](CONTRIBUTING.md) is the full guide; this file is the short
version plus the agent workflow.

## What this is

nibble is a from-scratch LLM inference engine in C++20: it loads GGUF files, runs quantized chat models on a CPU, and
is checked against llama.cpp. No runtime dependencies. Read [ARCHITECTURE.md](ARCHITECTURE.md) before touching
kernels or numerics (its "Repository layout" section maps `src/`); [README.md](README.md) has usage, correctness
and benchmark results.

## Setup, build, test

Needs g++ 13 or clang 18, CMake >= 3.25, Python 3. Dev tools come from `requirements-ci.txt`:

```sh
uv venv .venv && uv pip install --python .venv/bin/python -r requirements-ci.txt && . .venv/bin/activate
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4                              # -j4 at most: the dev machine is small and shared
ctest --test-dir build -LE e2e --output-on-failure   # everything except the end-to-end tests
```

- `e2e` tests need multi-gigabyte models under `models/` (git-ignored) and skip themselves when a model is missing.
  If a change can move numerics, run them (`ctest --test-dir build -L e2e -j2`) before and after and diff the
  `min cosine` lines: kernels meant to be bit-identical must leave them unchanged.
- Other configurations worth testing for kernel changes: `-DNIBBLE_SIMD=OFF`, `-DNIBBLE_SIMD_VNNI=OFF`,
  `-DNIBBLE_NATIVE=OFF`, and the sanitizer builds (see the table in CONTRIBUTING.md).
- `NIBBLE_KERNELS=scalar|avx2|avx2+vnni` forces a kernel tier at run time. No other environment variables are needed.
  CI optionally reads the `CODECOV_TOKEN` repository secret; never put secrets in the repo.

## Lint and format (CI runs all of these)

```sh
git ls-files '*.cpp' '*.h' | grep -v '^third_party' | xargs clang-format --dry-run -Werror
ruff check bench tools && ruff format --check bench tools && mypy
python -m unittest discover -s bench -p 'test_*.py'
shellcheck tools/*.sh tests/*.sh
```

clang-tidy: configure with `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DNIBBLE_BUILD_TESTS=OFF` and run it as in the
`clang-tidy` step of `.github/workflows/ci.yml`. `lefthook install` enables the pre-commit hooks (gitleaks, ruff,
clang-format, typos).

## Conventions

- Warnings are errors (`-Wall -Wextra -Wpedantic -Werror`). Every file starts with a comment saying what it is for;
  public and non-obvious members state their contract. Keep README.md and ARCHITECTURE.md true when behavior changes.
- Tests come with the change; a bug fix gets a test that fails without the fix (write and run it first).
- Input-caused errors throw `nibble::Error` (file-derived text through `escape_control`); internal invariants use
  `NIBBLE_DCHECK`. No allocation on the decode path (`alloc_tests` enforces it).
- Kernel translation units that must match the scalar code bit for bit stay in the `-ffp-contract=off` list in
  `CMakeLists.txt`. SIMD tiers live in `src/backend/cpu/simd/`, one copy per tier, selected at run time by
  `dispatch.cpp`; a tier object must define no symbol outside its namespace (`tools/check_simd_symbols.sh`).
- Fix causes, not symptoms: no skipped tests, no suppressions (`NOLINT` needs a stated reason).

## Commits and pull requests

- `master` is protected: changes go through a pull request with the 10 required CI checks (admins included). Never
  push to `master`.
- Conventional Commits (`feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `build`, `ci`, `style`, `chore`), small
  atomic commits, no attribution trailers. Fill in the PR template.
- Never commit secrets, `models/`, `third_party/`, golden binaries, or machine-local files (`*.local.md`,
  `.claude/settings.local.json`, `.env*`).

## Benchmarks

Numbers in README tables are tool-generated: `tools/bench_final.sh` writes `bench/results/*.jsonl`, and
`bench/plot.py` (`--markdown` for tables, no flag for the SVGs in `docs/bench/`) renders them. Do not hand-type a
table. Run benchmarks only on an idle machine (the driver waits for a 1-minute load average below 2), and never
build or run anything else while one is running: a rebuild replaces `build/nibble` mid-run.

## Agent workflow

Larger changes follow spec -> plan -> execution -> review, and the design record is committed:

- `docs/superpowers/specs/` holds design specs and `docs/superpowers/plans/` the task-by-task plans written from them
  (with the `superpowers` skills: brainstorming, writing-plans, executing-plans). Plans spell out files, test cases,
  commands with expected output, and commit messages; `-ledger.md` files record how earlier milestones went.
- Execution is test-first, one commit per logical step, with a progress ledger in `.superpowers/` (git-ignored
  scratch) so work survives context compaction. Deviations from a plan are logged as rulings and reported at the end.
- Finish with a fresh-context review of the whole branch, then a pull request.
- `.claude/settings.json` enables the shared plugins (clangd LSP, commit commands). Personal permission approvals stay
  in the git-ignored `.claude/settings.local.json`. There are no project MCP servers.
