# CI and Automation Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Grow nibble's CI from one per-PR gate into a visible, around-the-clock verification system. It adds nightly
accuracy against llama.cpp, long fuzzing, a per-PR speed/size/memory/accuracy report, proof that bug fixes carry a
reproducing test, Claude agents that triage and fix, attested releases, and a public dashboard.

**Architecture:** Workflows are thin YAML glue. All logic lives in typed, unit-tested Python modules under
`tools/ci/`, laid out flat like `bench/`, plus two composite actions: `setup-models` and `report-failure`. Producers
upload `dashboard-rows` JSONL artifacts. `dashboard.yml` collects them on a data branch and publishes SVG charts
rendered by `bench/plot.py` to GitHub Pages.

**Tech Stack:** GitHub Actions (SHA-pinned), Python 3.12 stdlib (ruff, mypy strict, unittest), bash (shellcheck),
CMake/ctest, Catch2 v3 JUnit reporter, libFuzzer, the pinned llama.cpp (`tools/llama_cpp_version.txt`),
`anthropics/claude-code-action`, CodeQL, OpenSSF Scorecard, gitleaks, lychee, git-cliff, syft, GitHub attestations.

**Spec:** `docs/superpowers/specs/2026-11-20-ci-design.md` (read it first; this plan argues from it).

## Rulings (deviations and findings from reading the repo)

- **R1. Peak memory.** `nibble bench` has no peak-RSS output: its `--mem` flag measures memory *bandwidth*
  (`cli/cmd_bench.cpp`, `bench::measure_membw`). The PR report measures peak RSS with
  `/usr/bin/time -f %M build/nibble run -m <q4_0> -p "The capital of France is" -n 32 --temp 0 -t 1`, which gives
  KiB. The CLI is not changed.
- **R2. No LTO cell.** `CMakeLists.txt` has no IPO/LTO support. The spec makes the LTO cell conditional on "no build
  changes", so it is dropped.
- **R3. Agent triggers.** Issues and labels created with `GITHUB_TOKEN` do not start other workflows; only
  `workflow_dispatch` and `repository_dispatch` are exempt. So `report-failure` starts triage with
  `gh workflow run claude-triage.yml -f issue=<n>`. `claude-triage.yml` listens to `workflow_dispatch` and to
  `issues: labeled` (for a human adding `ci-failure`). Claude's branches and PRs must be pushed with the Claude
  GitHub App's token, not `GITHUB_TOKEN`, or CI would not run on them. Installing the app is an owner step.
- **R4. Model cache budget.** The Qwen2.5-0.5B family is about 4.0 GB of models plus 0.8 GB of goldens (six
  directories of 136 MB each). GitHub's per-repository cache limit is 10 GB, and it is shared with ccache, pip,
  Catch2 and the Docker `type=gha` cache. Handling:
  - Exactly one cache entry, `models-qwen-v1-<hash>`.
  - Only the nightly accuracy job on `master` saves it.
  - Every other job only restores it (`actions/cache/restore`). PRs can read caches saved on `master`.
  - Daily restores keep it from being evicted. A cold cache never fails a job: the nightly rebuilds it (about 45 to
    75 min), and PR jobs report "skipped: cold cache".
- **R5. Restricting e2e to one family needs no code change.** Every e2e case calls `SKIP("model or golden missing")`,
  so with only Qwen files on disk, `ctest -L e2e` runs exactly the Qwen cases. The Qwen F16 model's case id is
  `f16`; the others are `q8_0`, `q4_0`, `q4_k_m`, `q5_k_m` and `q6_k`.
- **R6. Downloads need no authentication.** `huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/...` returns
  200 without a token (checked 2026-11-20). Llama 3.2 is gated, so it stays local-only.
  - TinyLlama is also left out (builder's call). `tools/convert_hf.sh` needs the Hugging Face converter's Python
    dependencies and another 2.2 GB of F16, which the cache budget in R4 cannot hold.
- **R7. ARM64 is unproven, not known-broken.** Every x86 path is guarded: `src/thread_pool.cpp` uses
  `#if defined(__x86_64__)`; `dispatch.cpp` includes `<cpuid.h>` only under `NIBBLE_SIMD_TIERS`; the tier sources and
  CMake tests are gated on `CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64"`. Expected findings: aarch64 GCC defaults
  to `-ffp-contract=fast`, so translation units outside the `-ffp-contract=off` list may fuse multiply-adds and miss
  exact expected values in tests.
  - The first run is by hand (Task 3.6).
  - Each real finding gets its own `fix:` PR with a failing test.
  - The weekly schedule stays on even if the first run is red, as the owner accepted.
- **R8. Required checks.** No new check joins branch protection in this plan. The owner decides after about two weeks
  green, and the decision is logged in `STEPS-FOR-OWNER.md`.
- **R9. Test outcome for fix-proof.** A fix PR is **proven** when at least one selected case fails on the base. Cases
  that already pass on the base are listed as "already passing", not counted against the PR. The base is built with
  ASan+UBSan (Debug), so fixes for memory errors and undefined behavior also reproduce. A ThreadSanitizer-only fix
  reports "does not reproduce", which is acceptable for a check that is not required.
- **R10. CHANGELOG.** `master` is protected, so `release.yml` cannot prepend to CHANGELOG.md. Instead:
  - The release-prep PR runs `git cliff --unreleased --tag vX.Y.Z --prepend CHANGELOG.md`.
  - `release.yml` fails if CHANGELOG.md has no `## [vX.Y.Z]` section.
  - It uses that section as the release body, replacing `generate_release_notes`.
- **R11. Attestations happen in `release.yml`, not `package.yml`.** `ci.yml` calls `package.yml` with
  `contents: read`, and a called job cannot ask for more permissions than its caller grants. So `package.yml` only
  gains an `image-digest` output, and a new `attest` job in `release.yml` signs the tarballs and the pushed image.
- **R12. Phase 4 is one PR.** pr-report and fix-proof ship in one PR (`ci/pr-checks`), so each rollout phase is one
  PR as asked.

Findings from the PR #49 review (CodeRabbit, each verified against the plan before it was adopted):

- **R13. Scorecard can be run by hand.** `scorecard.yml` also listens to `workflow_dispatch` (Task 2.3), because
  Step 4 of that task dispatches it. The opening comment gives `gh workflow run scorecard.yml`.
- **R14. Every nightly job reports its real outcome.** `dashboard_data.py status` and the artifact upload run under
  `if: always()` and record the job's actual result, so a failed job leaves a `job_status` 0 row instead of none
  (Task 3.5).
- **R15. A skipped job never closes an issue.** When a job skipped its work on a cold cache, `report-failure` is
  skipped, not called with `success`. Only a run that actually executed may close that job's issue (Task 3.5).
- **R16. Dispatch validation happens on `master`.** `workflow_dispatch` needs the workflow file on the default
  branch, so the by-hand runs of `nightly.yml` and `portability.yml` (Tasks 3.5 and 3.6) move to after the Phase 3
  merge. Real failures found then go to issues or `fix/<topic>` PRs, as before.
- **R17. Spread is checked first.** `verdict` returns `inconclusive` when max − min exceeds `SPREAD_MAX`, before it
  looks at faster or slower, so `[1.03, 1.10, 1.20]` is not "faster" (Task 4.1).
- **R18. The report must belong to the triggering run.** `pr_report.post` takes the `head_sha` of the `workflow_run`
  that produced the artifact and refuses to post when the artifact's own `head_sha` differs, so a stale or forged
  artifact cannot comment on another PR (Tasks 4.2 and 4.3).
- **R19. Deleting a line selects its case.** `changed_lines` also maps deleted lines to the enclosing case on the
  head side, so a deletion-only edit to a test still selects that test (Task 4.4).

## Global Constraints

- Every new workflow file starts with `#` comment lines giving its **reason** and **how to run it by hand**
  (`gh workflow run <file>.yml ...`) before `name:`.
- **Pinning:**
  - Every `uses:` is pinned to a full commit SHA with `# vX.Y.Z` after it. Resolve the SHA when you write the file,
    for the latest release tag: `gh api repos/<owner>/<repo>/git/ref/tags/<tag> --jq .object.sha`, and
    dereference annotated tags with `gh api repos/<owner>/<repo>/git/tags/<sha> --jq .object.sha`.
    `pinact run .github/workflows/<file>.yml` (installed) does the same. Never invent a SHA.
  - Reuse the existing pins for actions already in `ci.yml`: checkout `3d3c42e5…` v7.0.1, setup-python
    `6d4fd22b…` v7.1.0, cache `55cc8345…` v6.1.0, upload-artifact `ea165f8d…` v4.6.2, download-artifact `9000827c…`
    v8.0.2.
- **Permissions:**
  - Top-level `permissions:` is `contents: read` or `{}`. Jobs widen only what they use.
  - No job that checks out or builds PR code has a write token or a secret.
- Runners are `ubuntu-24.04` (and `ubuntu-24.04-arm` for ARM64). Python is `3.12` (`PYTHON_VERSION` as in `ci.yml`).
- **Cron times (UTC), off the hour:**

  | Workflow | Cron |
  |---|---|
  | nightly | `17 3 * * *` |
  | portability | `23 4 * * 0` |
  | scorecard | `37 5 * * 1` |
  | link-check | `41 6 * * 1` |
  | codeql | `11 6 * * 2` |
  | gitleaks full history | `29 2 * * 3` |
- **Cache keys:**
  - `models-qwen-v1-${{ hashFiles('tools/llama_cpp_version.txt', 'tools/download_models.sh', 'tools/build_llama_cpp.sh', 'tools/make_quants.sh', 'tools/make_golden.sh', 'tools/llama_ref/**', 'tests/e2e/prompts/**', 'tests/tokenizer/corpus*.txt') }}`
    over paths `models/` and `!models/hf`.
  - `fuzz-corpus-<target>-${{ github.run_id }}`, with restore-key `fuzz-corpus-<target>-`.
- **Labels** (name: colour, description). `tools/ci/labels.py` is the source of truth.
  - Area labels:
    - `kernels`: `1d76db`, "SIMD and scalar compute kernels"
    - `gguf`: `5319e7`, "GGUF parsing and loading"
    - `tokenizer`: `0e8a16`, "Tokenizer and chat templates"
    - `model`: `fbca04`, "Transformer, sampler, engine API"
    - `cli`: `c5def5`, "nibble command line"
    - `build`: `006b75`, "CMake and toolchains"
    - `ci`: `0052cc`, "Workflows and tools/ci"
    - `bench`: `d4c5f9`, "Benchmarks and plots"
    - `documentation`: `0075ca` (exists; it is the docs area)
  - Other labels:
    - `ci-failure`: `b60205`, "A scheduled job failed"
    - `agent`: `7057ff`, "Opened by the Claude agent"
    - `no-test-needed`: `ededed`, "Fix PR without a reproducing test (reason in the PR)"
    - `security`: `ee0701`, "Security-relevant"
    - `performance`: `f9d0c4`, "Speed or memory"
- **Python:** stdlib only (the repo installs no runtime packages for tools). It must pass `ruff check`,
  `ruff format --check` and `mypy` (strict). Module docstrings state purpose and usage, as in `bench/plot.py`.
  Network and `gh` access only goes through `tools/ci/gh.py`'s `GitHub` protocol, which tests replace with
  `tools/ci/fakes.py`.
- **Test command:** `python -m unittest discover -s tools/ci -p 'test_*.py'`. Add it next to the `bench/` line in
  `ci.yml` `lint`, CONTRIBUTING.md and AGENTS.md.
- **Agent workflows:**
  - They run only when the sender's `author_association` is `OWNER`, `MEMBER` or `COLLABORATOR` (or the sender is
    `github-actions[bot]` for dispatches).
  - `CLAUDE_CODE_OAUTH_TOKEN` missing means skip, with a notice in the job summary.
  - Agent PRs carry the `agent` label and are never auto-merged.
- **Process:**
  - Conventional Commits with no attribution trailers.
  - One PR per phase, on the branch named in the phase. Never push to `master`.
  - Log the progress ledger in `.superpowers/ci-automation-ledger.md` (git-ignored scratch).
  - Each phase ends with `actionlint` clean, the full lint command set from AGENTS.md green, and the PR's CI green.
- Never write an absolute home path anywhere: use `~`.

## Review Focus

1. **A fork PR's report artifact is untrusted input to a job with a write token.** The comment poster must:
   - validate the JSON strictly (types, finite numbers, bounded lengths, model names from an allowlist);
   - take the PR number from the API (`commits/{sha}/pulls`), never from the artifact;
   - render Markdown only from validated fields.
   Pinned by `test_pr_report.py::test_rejects_*`.
2. **No secret, cold cache, Dependabot or fork.** Every job skips with a summary notice and exits 0, never red and
   never silent. Pinned by `test_pr_report.py::test_render_skipped_sections` and by the `if:` guards in Tasks 6.x
   (checked in the actionlint step and by a dry run without the secret).
3. **Fix-proof on cases it cannot run.**
   - Macro-built names (`E2E_CASE`) → `unresolved`.
   - Cases that `SKIP` on the base (e2e without models) → `not-runnable`.
   - Names containing `,` `[` `]` `*` `"` `~` must be escaped for Catch2's test spec.
   None of these may report "proven". Pinned in `test_fix_proof.py`.
4. **Runner noise.** A shared runner's ±10% jitter must read as `inconclusive`, never as `slower`. Zero, NaN or
   missing medians must not crash the report. Pinned in `test_perf_ab.py::test_verdict_*`.
5. **Issue spam.**
   - One open issue per job, found by **exact** title match: `gh` search is fuzzy, and `nightly: fuzz_gguf failing`
     must not match `nightly: fuzz_load failing`.
   - Success closes only that job's issue.
   - Two jobs failing in one run open two issues, not four.
   Pinned in `test_report_failure.py`.

---

## Phase 1: Foundations. Branch `ci/foundations`, PR "ci: tools/ci package, labels, issue templates and actionlint"

### Task 1.1: `tools/ci/` package skeleton and the `GitHub` interface

**Files:**
- Create: `tools/ci/README.md` (one paragraph: what lives here, the test command)
- Create: `tools/ci/gh.py`, `tools/ci/fakes.py`, `tools/ci/test_gh.py`
- Modify: `.github/workflows/ci.yml` (lint job: add the test command after "bench/ unit tests")
- Modify: `lefthook.yml` (ruff globs `"{bench,tools}/*.py"` → `"{bench,tools,tools/ci}/*.py"`)
- Modify: `CONTRIBUTING.md`, `AGENTS.md` (lint block: add the test command)

**Interfaces:**
- Produces, in `gh.py`:

```python
Runner = Callable[[Sequence[str], str | None], str]  # (argv, stdin) -> stdout; raises GhError on non-zero exit
class GhError(RuntimeError): ...
class GitHub(Protocol):
    def find_open_issue(self, title: str) -> int | None: ...       # exact title match among open issues
    def create_issue(self, title: str, body: str, labels: Sequence[str]) -> int: ...
    def comment(self, number: int, body: str) -> None: ...
    def close_issue(self, number: int) -> None: ...
    def last_green_sha(self, workflow_file: str, branch: str) -> str | None: ...
    def upsert_comment(self, number: int, marker: str, body: str) -> None: ...  # edits the comment containing marker, else creates
    def pr_for_sha(self, sha: str) -> int | None: ...                # open PR whose head is sha
    def dispatch(self, workflow_file: str, ref: str, inputs: Mapping[str, str]) -> None: ...
class GhCli:  # implements GitHub with `gh api` (repo from $GITHUB_REPOSITORY)
    def __init__(self, repo: str, run: Runner = subprocess_runner) -> None: ...
```

- Produces, in `fakes.py`: `FakeGitHub` (in-memory issues/comments, records dispatches).

- [ ] **Step 1: Write the failing tests.** In `test_gh.py`, use a recording `Runner`:
  - `test_find_open_issue_exact_title`: the runner returns the JSON list `[{"number":3,"title":"nightly: fuzz_load failing"},{"number":4,"title":"nightly: fuzz_gguf failing"}]`. `find_open_issue("nightly: fuzz_gguf failing") == 4`.
  - `test_find_open_issue_paginates`: the call uses `--paginate` against `repos/{repo}/issues?state=open&per_page=100`.
  - `test_upsert_comment_edits_existing`: an existing comment whose body contains `<!-- m -->` is edited with PATCH, not created with POST.
  - `test_runner_error_raises`: a non-zero exit raises `GhError` with the stderr text.
  - `test_fake_matches_protocol`: `FakeGitHub()` is assignable to `GitHub`. mypy checks it through a typed assignment in the test.
- [ ] **Step 2:** Run `python -m unittest discover -s tools/ci -p 'test_*.py'`. Expected: an ImportError for `gh`.
- [ ] **Step 3:** Implement `gh.py` and `fakes.py`. `subprocess_runner` runs `gh` with `check=False`, then raises `GhError(stderr)` on a non-zero exit.
- [ ] **Step 4:** Run the tests, `ruff check bench tools`, `ruff format --check bench tools` and `mypy`. Expected: all pass.
- [ ] **Step 5:** Commit `ci(tools): add the tools/ci package with a testable GitHub client`.

### Task 1.2: Labels as code, and the PR labeler

**Files:**
- Create: `tools/ci/labels.py`, `tools/ci/test_labels.py`, `.github/labeler.yml`, `.github/workflows/labeler.yml`

**Interfaces:**
- Produces: `Label = NamedTuple(name: str, color: str, description: str)`, `LABELS: tuple[Label, ...]` (the Global
  Constraints list), and `AREA_PATHS: Mapping[str, tuple[str, ...]]` (area label → glob list). CLI:
  `python tools/ci/labels.py sync` runs `gh label create <name> --color <c> --description <d> --force` for each label.
- `AREA_PATHS`:
  - `kernels`: `src/backend/**`
  - `gguf`: `src/gguf/**`
  - `tokenizer`: `src/tokenizer/**`, `tests/tokenizer/**`
  - `model`: `src/model/**`, `include/**`, `src/api*`
  - `cli`: `cli/**`
  - `build`: `CMakeLists.txt`, `Dockerfile`
  - `ci`: `.github/**`, `tools/ci/**`
  - `bench`: `bench/**`, `tools/bench_*.sh`
  - `documentation`: `**/*.md`, `docs/**`
  
  Check the `src/` subdirectory names with `ls src` and adjust the globs to what exists.

- [ ] **Step 1: Write the failing tests.**
  - `test_names_unique`
  - `test_colors_are_six_hex`
  - `test_every_area_label_exists`: each `AREA_PATHS` key is in `LABELS`.
  - `test_labeler_yml_matches`: parse `.github/labeler.yml` with a tiny line parser (no PyYAML). Its keys and globs equal `AREA_PATHS`.
- [ ] **Step 2:** Run the tests. Expected: they fail.
- [ ] **Step 3:** Write `labels.py`, and `labeler.yml` in the `actions/labeler` v5+ format (`<label>:` then `- changed-files:` then `- any-glob-to-any-file: [...]`).
- [ ] **Step 4:** Write the workflow `labeler.yml`:
  - It runs on `pull_request_target: [opened, synchronize, reopened]`.
  - Permissions are `contents: read` and `pull-requests: write`.
  - It uses `actions/labeler` (pinned) and never checks out PR code.
  - Its opening comment gives the reason: the changelog, the triage issues and the dashboard group by area.
- [ ] **Step 5:** Run the tests, then `python tools/ci/labels.py sync`. Check with `gh label list`: every label is present.
- [ ] **Step 6:** Commit `ci: define labels as code and label pull requests by area`.

### Task 1.3: Issue templates and repository About

**Files:**
- Create: `.github/ISSUE_TEMPLATE/bug_report.yml`, `.github/ISSUE_TEMPLATE/feature_request.yml`, `.github/ISSUE_TEMPLATE/config.yml`

- [ ] **Step 1:** Write the templates as issue forms.
  - The bug form's fields are: nibble version (`nibble --version`), the `kernels:` line from `nibble info`, the model file, the command, expected vs actual, and an e2e/min-cosine field (optional). Its label is `bug`.
  - The feature form's label is `enhancement`.
  - `config.yml` sets `blank_issues_enabled: false` and has a contact link "Security report → SECURITY.md".
- [ ] **Step 2:** Run `gh repo edit --description "From-scratch LLM inference engine in C++20: GGUF, quantized CPU kernels with runtime SIMD dispatch, checked against llama.cpp" --add-topic llm,inference,cpp20,gguf,simd,quantization,avx2,llama-cpp,cpu-inference`.
- [ ] **Step 3:** Verify with `gh repo view --json description,repositoryTopics`. Expected: both are set.
- [ ] **Step 4:** Commit `docs(github): add bug and feature issue forms`.

### Task 1.4: actionlint

**Files:**
- Create: `.github/workflows/actionlint.yml`

- [ ] **Step 1:** Write the workflow:
  - It runs on `pull_request` and `push` to `master`, with paths `.github/workflows/**` and `.github/actions/**`, plus `workflow_dispatch`.
  - It downloads actionlint at a pinned version with a SHA-256 check. Use the release tarball for the latest tag and record its checksum from `checksums.txt` in the workflow.
  - It runs `actionlint -color` (it shellchecks `run:` blocks with the runner's shellcheck).
  - Opening comment: with many workflows, a typo silently stops one from running.
- [ ] **Step 2:** Run `actionlint` locally (installed). Expected: no output on all current workflows. Fix any real findings in this task.
- [ ] **Step 3:** Commit `ci: lint workflow files with actionlint`.
- [ ] **Step 4:** Push the branch, open the PR (fill in the template), wait for green, and merge with `gh pr merge --merge`.

---

## Phase 2: Security. Branch `ci/security`, PR "ci: CodeQL, gitleaks in CI and OpenSSF Scorecard"

### Task 2.1: CodeQL

**Files:** Create `.github/workflows/codeql.yml`

- [ ] **Step 1:** Write the workflow.
  - It runs on PRs, pushes to `master`, weekly (`11 6 * * 2`) and by hand.
  - It has a matrix of `c-cpp` (build-mode `manual`) and `actions` (build-mode `none`). For `c-cpp`, build with `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNIBBLE_NATIVE=OFF -DNIBBLE_BUILD_TESTS=OFF && cmake --build build -j "$(nproc)"`. That covers both tier object libraries, because tiers build on x86-64 regardless of `NIBBLE_NATIVE`.
  - Queries: `security-and-quality`.
  - Permissions: `security-events: write`, `contents: read`, `actions: read`.
  - Reason: nibble parses untrusted GGUF files.
- [ ] **Step 2:** Run actionlint. Expected: clean.
- [ ] **Step 3:** Commit `ci: scan C++ and workflows with CodeQL`.

### Task 2.2: gitleaks in CI

**Files:** Create `.github/workflows/gitleaks.yml`

- [ ] **Step 1:** Write the workflow.
  - On push and PR, scan the pushed range: `gitleaks git --log-opts="<base>..<head>" --redact --no-banner`. Use the PR base/head SHAs, or `github.event.before..github.sha` on push. A new-branch `before` of all zeros falls back to the full history.
  - Weekly (`29 2 * * 3`) and on dispatch, scan the full history.
  - Install the gitleaks binary from its release at a pinned version with checksum verification. Do not use `gitleaks/gitleaks-action`, which needs a licence key for organizations and adds nothing here.
  - It respects the existing `.gitleaksignore`.
  - Reason: the lefthook hook is optional, and push protection only knows provider patterns.
- [ ] **Step 2:** Run locally `gitleaks git --redact --no-banner`. Expected: `no leaks found`.
- [ ] **Step 3:** Commit `ci: scan for committed secrets with gitleaks`.

### Task 2.3: OpenSSF Scorecard

**Files:** Create `.github/workflows/scorecard.yml`

- [ ] **Step 1:** Write the workflow.
  - It runs on push to `master`, weekly (`37 5 * * 1`), on `branch_protection_rule` and on `workflow_dispatch` (R13). Its opening comment gives `gh workflow run scorecard.yml`.
  - Steps: `ossf/scorecard-action` with `publish_results: true`, then upload SARIF with `github/codeql-action/upload-sarif`.
  - Top-level `permissions: read-all`, as Scorecard requires. The job adds `security-events: write` and `id-token: write`.
- [ ] **Step 2:** Run actionlint. Expected: clean.
- [ ] **Step 3:** Commit `ci: publish an OpenSSF Scorecard`.
- [ ] **Step 4:** Open the PR, wait for green and merge. Then dispatch `codeql.yml` and `scorecard.yml` on `master` and confirm both succeed (`gh run list -w codeql.yml -L1`). Triage any CodeQL alert into an issue labelled `security`.

---

## Phase 3: Nightly. Branch `ci/nightly`, PR "ci: nightly accuracy, real-model sanitizers, long fuzzing and a portability sweep"

### Task 3.1: Parse and compare the `min cosine` lines

**Files:** Create `tools/ci/cosine.py`, `tools/ci/test_cosine.py`

**Interfaces:**
- Produces:

```python
class Cosine(NamedTuple): model: str; prompt: str; min_cos: float; positions: int; agree: int
def parse_min_cosine(log: str) -> dict[str, Cosine]       # key f"{model}/{prompt}"
class CosineChange(NamedTuple): key: str; old: float | None; new: float | None
def diff_cosines(old: Mapping[str, Cosine], new: Mapping[str, Cosine]) -> list[CosineChange]  # sorted by key; exact float compare
```

- The source line is the WARN text from `tests/test_e2e_golden.cpp:194`:
  `<prompt> [<model>]: min cosine over <N> positions = <x>, top-1 agreement <A>/<N>`. Use the regex
  `(\w+) \[([\w.+-]+)\]: min cosine over (\d+) positions = ([-+0-9.eE]+|nan|inf), top-1 agreement (\d+)/(\d+)`.
- Values are printed with the default stream precision (6 significant digits). The docstring says "unchanged" means
  unchanged at that precision.

- [ ] **Step 1: Write the failing tests.**
  - `test_parses_ctest_log_fragment`: a fixture of real ctest `--output-log` text with Catch2's `warning:` framing and two lines (`capital [q4_0]`, `chat [f16]`). Assert the keys `q4_0/capital` and `f16/chat` and `min_cos == 0.999123`.
  - `test_duplicate_key_conflict_raises`: the same key with two values raises `ValueError`.
  - `test_diff_reports_added_removed_changed`.
  - `test_nan_parses_as_nan_and_differs`.
- [ ] **Step 2:** Run the tests. Expected: they fail.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests. Expected: they pass.
- [ ] **Step 5:** Commit `ci(tools): parse and diff e2e min-cosine lines`.

### Task 3.2: `report-failure`: one issue per failing scheduled job

**Files:**
- Create: `tools/ci/report_failure.py`, `tools/ci/test_report_failure.py`, `.github/actions/report-failure/action.yml`

**Interfaces:**
- Consumes: `GitHub` (Task 1.1).
- Produces:

```python
Outcome = Literal["success", "failure"]
Action = Literal["open", "comment", "close", "none"]
@dataclass(frozen=True)
class RunContext: job: str; workflow_file: str; run_url: str; sha: str; repo: str; area: str; attachments_url: str | None
def issue_title(job: str) -> str                          # f"nightly: {job} failing"
def decide(open_issue: int | None, outcome: Outcome) -> Action
def failure_body(ctx: RunContext, last_green: str | None) -> str   # run link, sha, compare link last_green...sha, attachments link
def recovery_body(ctx: RunContext) -> str
def report(gh: GitHub, ctx: RunContext, outcome: Outcome, triage: bool) -> Action   # dispatches claude-triage.yml with {"issue": n} on "open" when triage
def main(argv: Sequence[str]) -> int  # --job --workflow-file --outcome --run-url --sha --area [--attachments-url] [--triage]
```

- Composite action inputs: `job`, `outcome` (both required), `area` (default `ci`), `attachments` (newline-separated
  paths; when non-empty they are uploaded as artifact `failure-<job>` and the artifact URL is passed on), `triage`
  (default `"false"`) and `github-token` (default `${{ github.token }}`). The caller needs `issues: write`, plus
  `actions: write` when `triage` is true. The action runs `python tools/ci/report_failure.py`.
- Labels on an opened issue: `ci-failure` and `<area>`.

- [ ] **Step 1: Write the failing tests** (with `FakeGitHub`).
  - `test_decide_table`: all four combinations.
  - `test_failure_opens_once_then_comments`: report twice for the same job, giving 1 issue with 1 comment.
  - `test_two_jobs_two_issues`: `fuzz_gguf` and `fuzz_load` open separate issues.
  - `test_success_closes_only_own_issue`.
  - `test_success_without_issue_is_noop`.
  - `test_triage_dispatched_only_on_open`.
  - `test_body_has_compare_link`: `https://github.com/<repo>/compare/<green>...<sha>`, and "no green run on record" when it is `None`.
- [ ] **Step 2:** Run the tests. Expected: they fail.
- [ ] **Step 3:** Implement the module and `action.yml`. Steps: an optional `actions/upload-artifact`, then the Python call with `GH_TOKEN` from the input.
- [ ] **Step 4:** Run the tests and actionlint. Expected: they pass.
- [ ] **Step 5:** Commit `ci: open, update and close one issue per failing scheduled job`.

### Task 3.3: `setup-models` composite action

**Files:** Create `.github/actions/setup-models/action.yml`

**Interfaces:**
- Inputs: `save` (`"true"` only in the nightly accuracy job) and `require-warm` (`"false"`). Outputs: `warm` (`true` or
  `false`) and `models-dir` (`models`).
- Behavior:
  1. Restore the cache (key in Global Constraints).
  2. On a miss with `require-warm: true`, set `warm=false` and stop.
  3. On a miss otherwise:
     1. Install `curl`, `git`, `cmake` and `g++`; they are on the runner already, so this is a check only.
     2. Run `tools/download_models.sh qwen2.5-0.5b-f16`.
     3. Run `tools/build_llama_cpp.sh` and `tools/build_llama_cpp.sh --generic`.
     4. Run `tools/make_quants.sh`.
     5. Run `tools/make_golden.sh` for `qwen2.5-0.5b-f16 qwen2.5-0.5b-q8_0 qwen2.5-0.5b-q4_0 qwen2.5-0.5b-q4_k_m qwen2.5-0.5b-q5_k_m qwen2.5-0.5b-q6_k`.
     6. Delete `third_party/build*` to free disk.
     7. Save with `actions/cache/save` when `save` is true.
  4. Print `du -sh models models/golden` to the job summary.

- [ ] **Step 1:** Write `action.yml`. Every step uses `shell: bash`.
- [ ] **Step 2:** Run actionlint. Expected: clean. (It is exercised for real in Task 3.6, Step 5, after the merge.)
- [ ] **Step 3:** Commit `ci: provide Qwen2.5-0.5B models and llama.cpp goldens through one cached action`.

### Task 3.4: Dashboard rows from nightly outputs

**Files:** Create `tools/ci/dashboard_data.py`, `tools/ci/test_dashboard_data.py`

**Interfaces:**
- Row schema (one JSON object per line, all keys required):

```json
{"schema": 1, "date": "2026-11-21T03:17:00Z", "commit": "<40-hex>", "source": "nightly|ci|portability",
 "metric": "bench_tok_s|min_cosine|fuzz_seconds|fuzz_execs|corpus_files|binary_bytes|line_coverage|job_status",
 "key": "<string ≤ 120 chars>", "value": <finite float>, "unit": "tok/s|cosine|s|execs|files|bytes|%|ok",
 "runner": "<cpu model name ≤ 120 chars>", "run_id": <int>}
```

- Keys:

  | Metric | Key | Notes |
  |---|---|---|
  | `bench_tok_s` | `<model file>/<test>/t<threads>` | e.g. `qwen2.5-0.5b-instruct-q4_0.gguf/tg128/t1` |
  | `min_cosine` | `<model>/<prompt>` | |
  | `fuzz_*`, `corpus_files` | `<target>` | |
  | `binary_bytes` | `nibble` | |
  | `line_coverage` | `src+cli` | |
  | `job_status` | `<workflow>/<job>` | value 1 = success, 0 = failure |
- Produces:

```python
Row = dict[str, object]
def validate(row: Mapping[str, object]) -> Row          # raises ValueError naming the bad field
def from_bench(jsonl: str, meta: Meta) -> list[Row]      # nibble bench --out lines; uses "median"
def from_cosine(c: Mapping[str, Cosine], meta: Meta) -> list[Row]
def from_libfuzzer(log: str, target: str, meta: Meta) -> list[Row]  # needs -print_final_stats=1: stat::number_of_executed_units, "Done N runs in S second(s)", and corpus file count passed in
def from_gcovr_summary(text: str, meta: Meta) -> list[Row]   # "lines: 85.3% (1234 out of 1447)"
def status_row(workflow: str, job: str, ok: bool, meta: Meta) -> Row
@dataclass(frozen=True)
class Meta: date: str; commit: str; source: str; runner: str; run_id: int
def merge(existing: str, new_rows: Iterable[Row]) -> str   # JSONL text; drops duplicates on (run_id, metric, key); keeps file order
```

- CLI: `python tools/ci/dashboard_data.py <from-bench|from-cosine|from-libfuzzer|from-gcovr|status> ... --out rows.jsonl`
  (appends). `Meta` comes from the environment: `GITHUB_SHA`, `GITHUB_RUN_ID`, the first `model name` line of
  `/proc/cpuinfo`, and a UTC `date`.

- [ ] **Step 1: Write the failing tests.**
  - One parse test per `from_*`, using real-format fixtures:
    - a `bench/results/abba.jsonl`-style nibble row;
    - a libFuzzer tail with `stat::number_of_executed_units: 1234567` and `Done 1234567 runs in 7200 second(s)`;
    - a gcovr `lines: 85.3% (1234 out of 1447)` line.
  - `test_validate_rejects_nan_and_long_key`.
  - `test_merge_is_idempotent`.
- [ ] **Step 2:** Run the tests. Expected: they fail.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests. Expected: they pass.
- [ ] **Step 5:** Commit `ci(tools): normalize nightly measurements into dashboard rows`.

### Task 3.5: `nightly.yml`

**Files:** Create `.github/workflows/nightly.yml`

**Interfaces:**
- Consumes: `setup-models` (3.3), `report-failure` (3.2), `cosine.py` (3.1) and `dashboard_data.py` (3.4).
- Every job's last steps:
  - `dashboard_data.py status` with the job's real outcome (`${{ job.status }}`), then upload the artifact `dashboard-rows-<job>` (retention 7 days). Both steps run under `if: always()` (R14).
  - `report-failure` with `outcome: ${{ job.status == 'success' && 'success' || 'failure' }}` under `if: always() && github.ref == 'refs/heads/master'`. A job that skipped its work on a cold cache (`steps.<models>.outputs.warm != 'true'`) skips this step too, and records no `ok` status row: only a run that executed may close an issue (R15).
  - Permissions: `contents: read`, `issues: write`, `actions: write`.

Jobs:
- **`accuracy`**, timeout 240:
  1. `setup-models` with `save: true`.
  2. A Release build with `-DNIBBLE_NATIVE=OFF`.
  3. `ctest --test-dir build -L e2e -j2 --output-log e2e.log --timeout 7200`.
  4. `cosine.py` parses `e2e.log`. Assert at least 18 keys: 6 models × 3 prompts. Fewer means models went missing, so fail.
  5. Diff against the previous night's `e2e-cosines` artifact, fetched with `gh run download` from the last successful `nightly.yml` run on `master`. Write the diff table to the job summary.
  6. Upload `e2e-cosines` (the parsed JSON) and add rows with `from-cosine`.
- **`sanitizers-real-model`**, timeout 240:
  1. `setup-models` with `require-warm: true`. When it is cold, write a notice and succeed.
  2. A Debug build with `-DNIBBLE_SANITIZE=address,undefined -DNIBBLE_NATIVE=ON`, after `sudo sysctl -w vm.mmap_rnd_bits=28`, as `ci.yml` does.
  3. `ctest -L e2e -R '\[q4_k_m\]'`.
- **`fuzz`**, matrix target `fuzz_gguf` / `fuzz_load`, timeout 150:
  1. Build as in `ci.yml` `fuzz`.
  2. Restore the corpus cache. Seed it with `make_seed`.
  3. Run `-max_total_time=7200 -print_final_stats=1 -artifact_prefix=crashes/`.
  4. Merge the corpus with `-merge=1` into a fresh directory, then save the cache.
  5. On a crash, pass `attachments: crashes/*`.
  6. Add rows with `from-libfuzzer`.
- **`trend-bench`**, timeout 90:
  1. `setup-models` with `require-warm: true`.
  2. A Release build with `-DNIBBLE_NATIVE=OFF`.
  3. `nibble bench -m models/qwen2.5-0.5b-instruct-{q4_0,q8_0}.gguf -t 1,$(nproc) -r 5 --out bench.jsonl`.
  4. Add rows with `from-bench`.
  5. Strip a copy of `build/nibble`, then add a `binary_bytes` row.

Triggers: `schedule` (`17 3 * * *`), and `workflow_dispatch` with an input `jobs` (default `all`) to run one job by
hand.

- [ ] **Step 1:** Write the workflow. Its opening comment says why each job exists (from the spec, item 7) and gives `gh workflow run nightly.yml -f jobs=fuzz`.
- [ ] **Step 2:** Run actionlint. Expected: clean.
- [ ] **Step 3:** Commit `ci: run accuracy, real-model sanitizers, long fuzzing and trend benchmarks nightly`.
- [ ] **Step 4:** No run yet. `workflow_dispatch` needs the file on the default branch, so the first run happens in Task 3.6, Step 5, after the Phase 3 merge (R16).

### Task 3.6: `portability.yml`

**Files:** Create `.github/workflows/portability.yml`

Jobs, each followed by `report-failure` (area `build`) and a status row:
- **`arm64`** on `ubuntu-24.04-arm`: install `ccache`, then build with `cmake -DCMAKE_BUILD_TYPE=Release -DNIBBLE_NATIVE=OFF`, then `ctest -LE e2e --output-on-failure --parallel "$(nproc)"`. Also confirm `nibble info build/tests/data/small.gguf` prints `kernels: scalar`.
- **`newest-gcc`**: install the newest `g++-NN` from the Ubuntu toolchain PPA (`ppa:ubuntu-toolchain-r/test`). Read the highest NN from `apt-cache search '^g\+\+-[0-9]+$'` and print the version to the summary. Build Release with `-DNIBBLE_NATIVE=OFF`, then run `ctest -LE e2e`.
- **`newest-clang`**: run `apt.llvm.org/llvm.sh` (pin it by SHA-256) for the newest stable release. Build and test as above.
- **`simd-off-portable`**: `-DNIBBLE_SIMD=OFF -DNIBBLE_NATIVE=OFF`, then the full non-e2e suite.

Triggers: weekly (`23 4 * * 0`) and `workflow_dispatch`.

- [ ] **Step 1:** Write the workflow. Its opening comment says it checks portability claims the per-PR matrix does not cover, and that new compilers' warnings are errors here first.
- [ ] **Step 2:** Run actionlint. Expected: clean.
- [ ] **Step 3:** Commit `ci: weekly ARM64, newest-compiler and SIMD-off portability sweep`.
- [ ] **Step 4:** Open the Phase 3 PR, wait for green and merge. Dispatch cannot run before this (R16).
- [ ] **Step 5:** On `master`, validate both workflows by hand.
  1. `gh workflow run nightly.yml -f jobs=accuracy`. Expected: the cache is built (log the cold-run duration in the ledger; confirm with `gh cache list | grep models-qwen-v1`) and 18 cosine keys appear in the summary.
  2. `gh workflow run portability.yml`, and record each job's result in the ledger.
  - For every real failure (R7), the fix goes on its own branch `fix/<topic>`, with a failing test first, and a separate PR.
  - If a fix is larger than one sitting, file an issue (`build` label) and continue. A red weekly job is acceptable.

---

## Phase 4: Per-PR checks. Branch `ci/pr-checks`, PR "ci: per-PR speed/size/memory/accuracy report and bug-fix proof"

### Task 4.1: ABBA performance comparison

**Files:** Create `tools/ci/perf_ab.py`, `tools/ci/test_perf_ab.py`

**Interfaces:**
- Produces:

```python
Side = Literal["base", "head"]
Verdict = Literal["faster", "slower", "same", "inconclusive"]
SAME_BAND = 0.02   # |median ratio - 1| below this, with spread <= SPREAD_MAX, is "same"
SPREAD_MAX = 0.05
def abba_order(rounds: int) -> list[tuple[int, str, Side]]   # (block, position A1|B1|B2|A2, side); A = base, B = head
def block_ratios(rows: Sequence[Mapping[str, object]]) -> dict[tuple[str, str, int], list[float]]
    # rows carry "side", "block", "model", "test", "threads", "median"; ratio per block = (B1+B2)/(A1+A2) of medians (head/base tok/s)
def verdict(ratios: Sequence[float]) -> Verdict
def run(base: str, head: str, models: Sequence[str], threads: Sequence[int], rounds: int, reps: int, out: str) -> None
    # drives `<bin> bench -m M -t T -p 512 -n 128 -r reps --out tmp.jsonl`, tags each row with side/block/position
```

- `verdict` rules:
  - Fewer than 2 ratios, or any ratio that is not finite or is ≤ 0: `inconclusive`.
  - max − min > SPREAD_MAX: `inconclusive`. This is checked before faster and slower (R17).
  - All ratios > 1 + SAME_BAND: `faster`.
  - All ratios < 1 − SAME_BAND: `slower`.
  - |median − 1| < SAME_BAND and max − min ≤ SPREAD_MAX: `same`.
  - Otherwise: `inconclusive`.
- Defaults used by the workflow: models `qwen2.5-0.5b-instruct-q4_0.gguf` and `…-q8_0.gguf`, threads `1` and
  `nproc`, rounds 3, reps 3.

- [ ] **Step 1: Write the failing tests.**
  - `test_abba_order_two_rounds == [(0,'A1','base'),(0,'B1','head'),(0,'B2','head'),(0,'A2','base'),(1,'A1','base'),…]`
  - `test_block_ratio_math`
  - `test_verdict_faster`, `test_verdict_slower`, `test_verdict_same`
  - `test_verdict_noise_is_inconclusive`: ratios `[0.93, 1.08, 1.01]`.
  - `test_verdict_wide_spread_is_inconclusive`: ratios `[1.03, 1.10, 1.20]` (all above 1 + SAME_BAND, spread 0.17).
  - `test_verdict_nan_zero_single`
- [ ] **Step 2:** Run the tests. Expected: they fail.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests. Expected: they pass.
- [ ] **Step 5:** Commit `ci(tools): ABBA speed comparison of two nibble builds`.

### Task 4.2: The PR report: schema, rendering and posting

**Files:** Create `tools/ci/pr_report.py`, `tools/ci/test_pr_report.py`

**Interfaces:**
- Artifact `pr-report/pr-report.json`, `schema` 1:

```json
{"schema": 1, "base_sha": "<40-hex>", "head_sha": "<40-hex>", "runner": "<≤120 chars>",
 "kernels_changed": true,
 "perf": [{"model": "<allowlisted file>", "test": "pp512|tg128", "threads": 1,
           "base_median": 0.0, "head_median": 0.0, "ratios": [0.0], "verdict": "faster|slower|same|inconclusive"}] | null,
 "size": {"base_bytes": 0, "head_bytes": 0} | null,
 "rss": {"base_kib": 0, "head_kib": 0} | null,
 "accuracy": {"status": "ran|skipped-cold-cache", "rows": [{"key": "<model>/<prompt>", "base": 0.0, "head": 0.0}]}}
```

  A `null` section means "skipped". The partial files from the jobs are merged by `assemble`.
- Produces:

```python
MARKER = "<!-- nibble-pr-report -->"
MODEL_ALLOWLIST = frozenset({"qwen2.5-0.5b-instruct-q4_0.gguf", "qwen2.5-0.5b-instruct-q8_0.gguf"})
def load(text: str) -> Report          # strict validation; raises ValueError (Report is a TypedDict tree)
def assemble(parts: Sequence[str]) -> str
def render(r: Report) -> str           # Markdown starting with MARKER
def post(gh: GitHub, r: Report, run_head_sha: str) -> int # raises ValueError when r["head_sha"] != run_head_sha; PR number from gh.pr_for_sha(run_head_sha); upsert_comment; returns PR or 0 when none
def main(argv: Sequence[str]) -> int   # subcommands: assemble <parts...> --out F | post F --head-sha SHA
```

- The comment layout, in order:
  1. A heading `nibble PR report`.
  2. A note: "shared CI runner: trend, not a benchmark".
  3. A speed table: model, test, threads, base tok/s, head tok/s, Δ% (median ratio − 1) and the verdict, with ✅ faster, ⚠️ slower, ➖ same, ❔ inconclusive.
  4. Size and peak-memory lines with Δ.
  5. An accuracy table. When `kernels_changed` is true and any cosine differs, it leads with "❗ bit-identity: N cases changed".
  6. The `<details>` footer: base and head SHAs, the runner, and how to reproduce locally.

- [ ] **Step 1: Write the failing tests.**
  - `test_load_valid_fixture`
  - `test_rejects_unknown_model`
  - `test_rejects_nan`
  - `test_rejects_extra_keys`
  - `test_rejects_long_runner`
  - `test_rejects_bad_sha`
  - `test_render_skipped_sections`: every section `null` renders three "skipped" lines and no tables.
  - `test_render_flags_bit_identity_change`
  - `test_post_updates_existing_comment` (FakeGitHub)
  - `test_post_without_pr_returns_0`
  - `test_post_refuses_mismatched_head_sha`: a report whose `head_sha` names another PR's head raises and posts nothing (R18).
  - `test_markdown_escapes_pipe_in_key`
- [ ] **Step 2:** Run the tests. Expected: they fail.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests. Expected: they pass.
- [ ] **Step 5:** Commit `ci(tools): validate, render and post the PR report`.

### Task 4.3: `pr-report.yml` and `pr-report-comment.yml`

**Files:** Create `.github/workflows/pr-report.yml`, `.github/workflows/pr-report-comment.yml`

- **`pr-report.yml`**:
  - Triggers and permissions: `pull_request` with paths `src/**`, `include/**`, `cli/**`, `CMakeLists.txt` and `tests/**`. `permissions: contents: read`. Concurrency per PR, cancelling runs in progress.
  - **`perf`** job:
    1. Check out with `fetch-depth: 0`. Build head, and base (a worktree at `github.event.pull_request.base.sha`), both Release with `-DNIBBLE_NATIVE=OFF -DNIBBLE_BUILD_TESTS=OFF --target nibble_cli`.
    2. `setup-models` with `require-warm: true`. When cold, perf, rss and accuracy are `null`.
    3. Run `perf_ab.py run`, then size (strip copies) and rss (R1).
    4. Write `part-perf.json`.
  - **`accuracy`** job:
    1. `setup-models` with `require-warm: true`.
    2. Build base and head with tests (Release, `-DNIBBLE_NATIVE=OFF`).
    3. Run `ctest -L e2e --output-log` on each, then `cosine.py`.
    4. `kernels_changed` = the diff touches `src/backend/`.
    5. Write `part-accuracy.json`.
  - **`assemble`** job: `pr_report.py assemble`, then upload the artifact `pr-report`.
- **`pr-report-comment.yml`**:
  - Triggers and permissions: `workflow_run` on `pr-report` completed. `permissions: pull-requests: write, actions: read`.
  - It never checks out PR code. It checks out the default branch (for `tools/ci`), downloads `pr-report` from the triggering run with `run-id`, and runs `pr_report.py post pr-report.json --head-sha "${{ github.event.workflow_run.head_sha }}"` with `GH_TOKEN` (R18).
  - It skips when the artifact is missing.
- Each opening comment states the trust split and gives the manual run: `gh workflow run pr-report.yml` is not
  possible for PRs, so it says "push to the PR".

- [ ] **Step 1:** Write both workflows.
- [ ] **Step 2:** Run actionlint. Expected: clean.
- [ ] **Step 3:** Commit `ci: post a speed, size, memory and accuracy report on pull requests`.

### Task 4.4: Bug-fix proof logic

**Files:** Create `tools/ci/fix_proof.py`, `tools/ci/test_fix_proof.py`

**Interfaces:**
- Produces:

```python
FIX_RE = r"^fix(\([a-z0-9_.,-]+\))?!?: "
CODE_PREFIXES = ("src/", "include/", "cli/")
class Case(NamedTuple): file: str; name: str | None; start: int; end: int; binary: str  # nibble_tests, or nibble_alloc_tests for tests/alloc_tests.cpp
def is_fix(title: str, subjects: Sequence[str]) -> bool
def touches_code(paths: Sequence[str]) -> bool
def test_cases(path: str, source: str) -> list[Case]      # name None when the first TEST_CASE argument is not one plain string literal; end = line before the next TEST_CASE or EOF
def changed_lines(diff: str) -> dict[str, set[int]]        # new-side line numbers of added lines, plus, for each deleted line, the new-side line where it was removed (the hunk position), tests/*.cpp only (R19)
class Selection(NamedTuple): cases: list[Case]; unresolved: list[str]
def select(diff: str, head_sources: Mapping[str, str]) -> Selection
def catch_spec(name: str) -> str                          # backslash-escapes , [ ] * " ~ and a leading ~ / *
CaseResult = Literal["passed", "failed", "skipped", "missing"]
def parse_junit(xml: str, name: str) -> CaseResult         # Catch2 -r junit; xml.etree; "missing" when absent
Outcome = Literal["proven", "proven-base-does-not-compile", "no-test", "does-not-reproduce", "not-runnable"]
def classify(selection: Selection, base_compiled: bool, results: Mapping[str, CaseResult]) -> Outcome
def exit_code(outcome: Outcome, override_label: bool) -> int   # 1 for no-test (unless label) and does-not-reproduce; else 0
def summary(outcome: Outcome, selection: Selection, results: Mapping[str, CaseResult]) -> str
def main(argv: Sequence[str]) -> int  # select --base B --head H (runs git diff, writes cases.json) | classify ...
```

- `classify` rules, in order:
  1. No resolved cases: `no-test`.
  2. `not base_compiled`: `proven-base-does-not-compile`.
  3. Any case `failed`: `proven`.
  4. All cases `skipped` or `missing`: `not-runnable`.
  5. Otherwise: `does-not-reproduce`.

- [ ] **Step 1: Write the failing tests.**
  - `test_is_fix_title_or_subject`: `fix(cpu): x` → True; `feat: x` with a subject `fix: y` → True; `fixup: x` → False.
  - `test_test_cases_names_and_spans`: a fixture with three `TEST_CASE`s, one an `E2E_CASE`-style macro body (`TEST_CASE("e2e golden: " PROMPT ...`). That one's name is None.
  - `test_select_added_and_modified_cases`: a unified-diff fixture adding a case and editing inside another.
  - `test_select_deletion_only_edit`: a diff that only deletes lines inside a case selects that case (R19).
  - `test_select_macro_case_is_unresolved`
  - `test_catch_spec_escapes`: `catch_spec('a, b [x] *') == r'a\, b \[x\] \*'`
  - `test_parse_junit_states`: fixtures for passed, failed, skipped (a Catch2 SKIP gives `<skipped/>`) and missing.
  - `test_classify_table`: one assertion per rule.
  - `test_exit_codes_with_label`
- [ ] **Step 2:** Run the tests. Expected: they fail.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests. Expected: they pass.
- [ ] **Step 5:** Commit `ci(tools): choose and judge the tests that prove a bug fix`.

### Task 4.5: `fix-proof.yml` and the demonstration

**Files:** Create `.github/workflows/fix-proof.yml`

- **Workflow** (`pull_request` on any base branch: opened, synchronize, edited, labeled, unlabeled; `contents: read`):
  1. Check out head with `fetch-depth: 0`.
  2. `fix_proof.py` decides whether the PR applies (`is_fix` on the title and the commit subjects, plus `touches_code`). If not, write "not a code fix: nothing to prove" to the summary and exit 0.
  3. `fix_proof.py select --base <merge-base> --head HEAD` writes `cases.json`.
  4. Create a worktree at the merge-base and copy head's `tests/` over it.
  5. Configure Debug with `-DNIBBLE_SANITIZE=address,undefined -DNIBBLE_NATIVE=ON` (lower ASLR entropy as in `ci.yml`), then build `nibble_tests nibble_alloc_tests`, with `continue-on-error` on the build step and its outcome recorded.
  6. Run each case with `<binary> "<catch_spec(name)>" -r junit -o <n>.xml`.
  7. `fix_proof.py classify` writes the summary and the exit code. The `no-test-needed` label sets `--override`.
- **Opening comment:** this enforces AGENTS.md's "a bug fix gets a test that fails without the fix".

- [ ] **Step 1:** Write the workflow.
- [ ] **Step 2:** Run actionlint. Expected: clean.
- [ ] **Step 3:** Commit `ci: prove that a bug-fix PR's test fails without the fix`.
- [ ] **Step 4: Demonstration** (spec success criterion).
  1. Create `demo/fix-proof-base` from `ci/pr-checks` with one commit that deliberately breaks `sampler: top-k tie goes to the lower index`: flip the tie rule in `src/` (find it with `grep -rn "tie" src/sampler*`).
  2. On `demo/fix-proof-head`, from that base, revert the break and add a new `TEST_CASE("sampler: top-k tie regression demo", "[sampler]")` that fails on the broken code. Open a **draft** PR `fix(sampler): demo` into `demo/fix-proof-base`. Expected summary: `proven`.
  3. Push a second draft PR, `fix(sampler): demo no-repro`, that only adds a case asserting already-true behavior. Expected: `does not reproduce`, and the check fails.
  4. Record both run URLs in the ledger. Close both PRs and delete both `demo/*` branches.
- [ ] **Step 5:** Open the Phase 4 PR, wait for green and merge. The Phase 4 PR touches only `tools/` and `.github/`, so `pr-report.yml` does not run on it. After merging, open a draft PR whose only change is a comment in `src/`, confirm the `nibble PR report` comment appears, record its link in the ledger, and close the PR.

---

## Phase 5: Dashboard. Branch `ci/dashboard`, PR "ci: publish a history dashboard on GitHub Pages"

### Task 5.1: A line chart in `bench/plot.py`

**Files:**
- Modify: `bench/plot.py` (add the type alias and function, reusing `Svg`, `nice_max` and `y_axis`)
- Test: `bench/test_plot.py`

**Interfaces:**
- Produces: `Series = tuple[str, str, Sequence[tuple[float, float]]]` (label, colour, points of x = unix seconds,
  y = value), and `def line_chart(title: str, y_label: str, series: Sequence[Series], y_min: float | None = None) -> str`
  (an SVG text with an x axis of date labels in `YYYY-MM-DD`, and a legend).

- [ ] **Step 1: Write the failing tests.**
  - `test_line_chart_is_svg_with_legend`: the output starts with `<svg`, contains each label (escaped), and has one `<polyline` per series.
  - `test_line_chart_single_point_and_empty_series`: neither raises. Empty data renders the text "no data yet".
  - `test_line_chart_y_min`: cosine charts pass `y_min=0.99` and the axis starts there.
- [ ] **Step 2:** Run `python -m unittest discover -s bench -p 'test_*.py'`. Expected: the new tests fail.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests. Expected: they pass. `python bench/plot.py` output is unchanged: `git status docs/bench` is clean.
- [ ] **Step 5:** Commit `feat(bench): line charts for time series`.

### Task 5.2: Render the site, and `dashboard.yml`

**Files:**
- Create: `tools/ci/dashboard_render.py`, `tools/ci/test_dashboard_render.py`, `.github/workflows/dashboard.yml`
- Modify: `.github/workflows/ci.yml`. In the `coverage` job, on `push` to `master` only, run
  `dashboard_data.py from-gcovr coverage/summary.txt` and upload `dashboard-rows-coverage`.

**Interfaces:**
- Consumes: the row schema (3.4) and `plot.line_chart` (5.1). Import it with
  `sys.path.insert(0, <repo>/bench)` at module top; mypy resolves `plot` because `bench` is in its `files`.
- Produces: `def render_site(rows_jsonl: str, out_dir: str, repo: str) -> list[str]` (the written file names).
  - Pages and charts:
    - `index.html`: a single static page with no JavaScript. It has a status table and inline `<img>` SVGs.
    - `speed.svg`: decode/prefill tok/s per series `<model>/<test>/t<threads>`.
    - `accuracy.svg`: the min cosine per model, the minimum over prompts, with `y_min` 0.99.
    - `fuzz.svg`: cumulative fuzzing hours per target, a running sum of `fuzz_seconds` / 3600.
    - `size.svg`
    - `coverage.svg`
  - The status table shows the latest `job_status` per key, with ✅/❌, the date and a link to the run
    (`https://github.com/<repo>/actions/runs/<run_id>`).
  - Also written: `badge.json`, a shields.io endpoint: `{"schemaVersion":1,"label":"nightly","message":"passing|failing","color":"brightgreen|red"}`. It says passing iff every latest `nightly/*` status is 1.
- **`dashboard.yml`**:
  - Triggers: `workflow_run` on `nightly`, `portability` and `ci` completed, with `branches: [master]`; also `workflow_dispatch`.
  - Concurrency group `dashboard`, without cancelling.
  - Permissions: `contents: write`, `pages: write`, `id-token: write`, `actions: read`.
  - Steps:
    1. Download every `dashboard-rows-*` artifact from the triggering run. On a dispatch, skip this and only re-render.
    2. Check out `dashboard-data` (create it as an orphan branch with `data/rows.jsonl` if it is missing).
    3. Run `dashboard_data.py merge`, then commit `chore(dashboard): rows from run <id>` and push. On a push race, retry up to 3 times with `git pull --rebase`.
    4. `render_site`, then `actions/upload-pages-artifact` and `actions/deploy-pages`.

- [ ] **Step 1: Write the failing tests.**
  - `test_render_writes_all_files`
  - `test_status_table_uses_latest_row_per_key`
  - `test_badge_failing_when_any_latest_nightly_status_zero`
  - `test_cumulative_fuzz_hours`: rows of 7200 s and 7200 s give a last point of 4.0.
  - `test_html_escapes_keys`
- [ ] **Step 2:** Run the tests. Expected: they fail.
- [ ] **Step 3:** Implement the module and the workflow.
- [ ] **Step 4:** Run the tests and actionlint. Expected: they pass.
- [ ] **Step 5:** Enable Pages for workflow deploys: `gh api -X POST repos/dpologdvinity/nibble/pages -f build_type=workflow`. Use `PUT` with the same field if it already exists. If the API refuses, add "Settings → Pages → Source: GitHub Actions" to `STEPS-FOR-OWNER.md`.
- [ ] **Step 6:** Commit `ci: collect nightly rows and publish the dashboard to GitHub Pages`.
- [ ] **Step 7:** Open the PR, wait for green and merge.
  - Run `gh workflow run nightly.yml`. After it finishes, confirm `dashboard.yml` ran and the site URL from `gh api repos/dpologdvinity/nibble/pages --jq .html_url` serves `index.html` with every chart. A chart may say "no data yet" only for coverage until the next push to `master`.
  - Record the URL in the ledger.

---

## Phase 6: Agents. Branch `ci/agents`, PR "ci: Claude on demand, automatic review, failure triage and auto-merge"

Before starting, add the owner steps to `STEPS-FOR-OWNER.md` (the git-excluded file at the repository root):
- the `CLAUDE_CODE_OAUTH_TOKEN` repository secret (`claude setup-token`, then `gh secret set CLAUDE_CODE_OAUTH_TOKEN -R dpologdvinity/nibble`);
- optionally the same token as a Dependabot secret, for reviews of Dependabot PRs;
- installing the Claude GitHub App on the repo (`/install-github-app` in Claude Code), needed for R3.

All three workflows below work, by skipping with a notice, before those steps are done.

### Task 6.1: `claude.yml`, `@claude` on demand

**Files:** Create `.github/workflows/claude.yml`

- [ ] **Step 1:** Write the workflow, modelled on the owner's daydaze `claude.yml`:
  - Triggers: `issue_comment`, `pull_request_review_comment`, `pull_request_review: submitted` and `issues: opened`. Same permissions as daydaze: `contents: write`, `pull-requests: write`, `issues: write`, `id-token: write`, `actions: read`.
  - The gate is the `author_association` in (OWNER, MEMBER, COLLABORATOR) and the text contains `@claude`.
  - A first step checks the secret, and writes a notice and exits when it is empty. The following steps run on `if: steps.secret.outputs.present == 'true'`.
  - Steps: check out, install the build tools (`ccache`; the runner has cmake, g++ and clang), then `anthropics/claude-code-action` (pinned) with `allow_fork_prs: "false"`.
  - `--allowedTools`: `Bash(cmake:*)`, `Bash(ctest:*)`, `Bash(clang-format:*)`, `Bash(git:*)`, `Bash(gh pr:*)`, `Bash(gh issue:*)`, `Bash(python -m unittest:*)`, `Bash(ruff:*)`, `Bash(mypy:*)`, `Bash(shellcheck:*)`, `Bash(build/nibble:*)`.
  - `--append-system-prompt`: "Follow AGENTS.md. Work on a branch named agent/<topic>, never push to master, Conventional Commits, label your PRs `agent`. Before pushing run the lint block from AGENTS.md and `ctest --test-dir build -LE e2e`."
  - Concurrency per issue or PR, without cancelling.
- [ ] **Step 2:** Run actionlint. Expected: clean.
- [ ] **Step 3:** Commit `ci: answer @claude on issues and pull requests`.

### Task 6.2: `claude-review.yml`, automatic review

**Files:** Create `.github/workflows/claude-review.yml`

- [ ] **Step 1:** Write the workflow:
  - Triggers: `pull_request` (opened, synchronize, ready_for_review) when not a draft, and the author is the owner, `dependabot[bot]`, or a PR with the `agent` label.
  - Permissions: `contents: read`, `pull-requests: write`, `id-token: write`.
  - The same secret gate as Task 6.1.
  - The review prompt names the review focus: numerics and bit-identity (the `-ffp-contract=off` list), no allocation on the decode path, `nibble::Error` vs `NIBBLE_DCHECK`, tests coming with the change, AGENTS.md conventions, and workflow least privilege. Allowed tools are read-only plus `mcp__github_inline_comment__create_inline_comment`. Inline comments only: never approve or request changes.
- [ ] **Step 2:** Run actionlint. Expected: clean.
- [ ] **Step 3:** Commit `ci: review pull requests automatically with Claude`.

### Task 6.3: `claude-triage.yml`, and wiring `report-failure` to it

**Files:**
- Create: `.github/workflows/claude-triage.yml`
- Modify: `.github/workflows/nightly.yml` and `portability.yml` (`triage: "true"` on every `report-failure` step)

- [ ] **Step 1:** Write the workflow:
  - Triggers: `workflow_dispatch` (input `issue`, required) and `issues: labeled`, when the label is `ci-failure` and the sender is OWNER, MEMBER or COLLABORATOR.
  - Permissions: as in Task 6.1. The same secret gate. Timeout 120.
  - The prompt includes the issue number and tells Claude to:
    1. Read the issue and the failing run's log (`gh run view --log-failed`).
    2. List the commits between the last green and the first red SHA (the compare link in the issue).
    3. Comment a diagnosis that states its confidence.
    4. Only when there is a reproducer and the confidence is high: create `agent/fix-issue-<n>`. The first commit adds a failing test (`test: reproduce #<n>`), and the second fixes it (`fix(<area>): ... (#<n>)`). Run the lint block and the non-e2e tests, then open a PR with the label `agent` and `Fixes #<n>`.
    5. Never merge and never push to master.
    6. If unsure, stop after the comment.
- [ ] **Step 2:** Run actionlint. Expected: clean.
- [ ] **Step 3:** Commit `ci: let Claude diagnose and propose fixes for failing scheduled jobs`.

### Task 6.4: `automerge.yml`, the repository setting, and AGENTS.md

**Files:**
- Create: `.github/workflows/automerge.yml`
- Modify: `AGENTS.md` (agent workflow section: the `agent/` branches, the `agent` label, never auto-merged, `@claude`)

- [ ] **Step 1:** Write the workflow:
  - Triggers and permissions: `pull_request_target` (opened, reopened, ready_for_review, labeled). Top-level `permissions: {}`.
  - **Job `owner`**: when not a draft, from the same repository, the author is `dpologdvinity`, and there is no `agent` label: `gh pr merge --auto --merge <url>`.
  - **Job `dependabot`**: when the author is `dependabot[bot]`, run `dependabot/fetch-metadata` (pinned). If `update-type` is `version-update:semver-minor` or `version-update:semver-patch`, `gh pr merge --auto --merge`.
  - Both jobs: `contents: write`, `pull-requests: write`. Neither checks out code.
- [ ] **Step 2:** Run `gh api -X PATCH repos/dpologdvinity/nibble -F allow_auto_merge=true`. Verify: `gh api repos/dpologdvinity/nibble --jq .allow_auto_merge` returns `true`.
- [ ] **Step 3:** Run actionlint. Expected: clean.
- [ ] **Step 4:** Commit `ci: auto-merge the owner's and Dependabot's minor/patch PRs once checks pass`.
- [ ] **Step 5:** Open the PR, wait for green and merge. This PR itself shows whether automerge works once the workflow is on master: the next owner PR does.
  - Once the owner has added the secret, comment `@claude what does fix-proof.yml check?` on a scratch issue.
  - Record the run link, then close the issue.

---

## Phase 7: Release. Branch `ci/release-provenance`, PR "ci: attested, SBOM-carrying releases and a generated changelog"

### Task 7.1: git-cliff config and the backfilled CHANGELOG

**Files:** Create `cliff.toml`, `CHANGELOG.md`

- [ ] **Step 1:** Write `cliff.toml`:
  - Keep a Changelog layout with sections from Conventional Commits types: feat → Added, fix → Fixed, perf → Performance, refactor/build/ci/test/style/chore → Changed (`chore(dashboard)` skipped), docs → Documentation, revert → Reverted. Merge commits are skipped.
  - Tag pattern `v[0-9]*`.
  - Link each version header to `https://github.com/dpologdvinity/nibble/compare/<prev>...<tag>`.
- [ ] **Step 2:** Run `git cliff -o CHANGELOG.md` (installed). Expected: sections `[Unreleased]`, `[v0.8.0]`, `[v0.7.1]` and `[v0.7.0]`, with earlier history under v0.7.0 since no earlier tags exist. Read it and fix obviously bad grouping in `cliff.toml`, not by hand.
- [ ] **Step 3:** Commit `docs: add a CHANGELOG generated from Conventional Commits`.

### Task 7.2: Attestations, SBOMs, and release notes from CHANGELOG

**Files:** Modify `.github/workflows/package.yml` and `.github/workflows/release.yml`

- [ ] **Step 1:** In `package.yml`, the `image` job's push step gets an `id`. Expose
  `outputs.image-digest: ${{ steps.push.outputs.digest }}` at the job level and as a `workflow_call` output.
- [ ] **Step 2:** In `release.yml`, add a `changelog` check job before `release`. It extracts the section
  `## [${GITHUB_REF_NAME}]` from CHANGELOG.md with `awk`, writes it to `notes.md` as an artifact, and fails with
  "CHANGELOG.md has no section for <tag>: run git cliff --unreleased --tag <tag> --prepend CHANGELOG.md in the
  release-prep PR".
- [ ] **Step 3:** Add an `attest` job (needs `package`) with permissions `id-token: write`, `attestations: write`,
  `packages: write` and `contents: read`:
  1. Download the tarballs.
  2. `anchore/sbom-action` (pinned) writes an SPDX JSON SBOM per tarball (`nibble-<tag>-linux-x86_64.spdx.json`) and one for the image `ghcr.io/dpologdvinity/nibble@<digest>`.
  3. `actions/attest-build-provenance` (pinned) for `dist/*.tar.gz`, and for the image with `subject-name`, `subject-digest` and `push-to-registry: true`.
  4. `actions/attest-sbom` (pinned) for each SBOM.
  5. Upload the SBOMs as an artifact.
- [ ] **Step 4:** The `release` job needs `[attest, changelog]`. It downloads the SBOMs and `notes.md`, sets
  `body_path: notes.md`, and removes `generate_release_notes`. Its files are `dist/*`, which now includes the SBOMs.
- [ ] **Step 5:** Run actionlint. Expected: clean. Dry-check the awk extraction locally:
  `GITHUB_REF_NAME=v0.8.0 bash -c '<the awk>' < CHANGELOG.md` prints the v0.8.0 section.
- [ ] **Step 6:** Commit `ci: attest release artifacts with provenance and SBOMs`.
- [ ] **Step 7:** Open the PR, wait for green and merge.
  - The release path cannot be exercised without a tag. Do **not** push a release tag. Add to `STEPS-FOR-OWNER.md`: "next release: run the release-prep `git cliff` command, then tag; verify with `gh attestation verify nibble-<tag>-linux-x86_64.tar.gz -R dpologdvinity/nibble`".

---

## Phase 8: Docs. Branch `docs/ci-overview`, PR "docs: how nibble is tested, badges, and the CI in CONTRIBUTING/ARCHITECTURE"

### Task 8.1: README, "How nibble is tested", and badges

**Files:** Modify `README.md`

- [ ] **Step 1:** Add the section "How nibble is tested" after "Correctness" (find the heading with `grep -n '^## ' README.md`):
  1. A two-sentence intro.
  2. A table with columns When, Workflow, What it protects. One row per workflow that exists, with what it protects in plain words, from the spec's reasons. The `ci.yml` row states that the PR matrix builds with the documented compiler floor (g++ 13, clang 18), as spec item 1 requires.
  3. A Mermaid `flowchart LR`: nightly job → issue (`ci-failure`) → Claude diagnosis → `agent/` PR with failing test then fix → fix-proof + CI → owner merges.
  4. A link to the dashboard and a sentence: "CI-runner speed numbers are trends; the benchmark tables above come from an idle dedicated machine."
  5. How to verify a release with `gh attestation verify`.
- [ ] **Step 2:** Badges.
  - Add:
    - nightly (shields endpoint `https://img.shields.io/endpoint?url=<pages>/badge.json`)
    - CodeQL (the workflow badge)
    - OpenSSF Scorecard (`https://api.scorecard.dev/projects/github.com/dpologdvinity/nibble/badge`)
    - latest release (`img.shields.io/github/v/release/dpologdvinity/nibble`)
    - dashboard (static, linking Pages)
  - Keep the existing badges. Change the platform badge to `linux x86-64 | arm64` only if the Phase 3 ARM64 job is green.
  - Check every badge URL with `curl -sfI` (200, image content type), and drop any that do not resolve.
- [ ] **Step 3:** Commit `docs(readme): explain how nibble is tested and add live CI badges`.

### Task 8.2: CONTRIBUTING, ARCHITECTURE, AGENTS, and the owner ledger

**Files:** Modify `CONTRIBUTING.md`, `ARCHITECTURE.md` (Repository layout: `tools/ci/`, `.github/actions/`), `AGENTS.md`

- [ ] **Step 1:** Update CONTRIBUTING.md with:
  - a "CI and automation" subsection: every workflow with its manual-run command;
  - the `tools/ci` test command;
  - what fix-proof expects of a `fix:` PR, and the `no-test-needed` label;
  - the release-prep `git cliff` step.
- [ ] **Step 2:** In ARCHITECTURE.md's layout, add `tools/ci/` and `.github/actions/`. In AGENTS.md, add the `tools/ci` test command (if not already done in 1.1) and one line on fix-proof.
- [ ] **Step 3:** Run `lychee --offline README.md ARCHITECTURE.md CONTRIBUTING.md AGENTS.md docs` (installed). Expected: no broken relative links.
- [ ] **Step 4:** Commit `docs: describe the CI workflows and tools/ci for contributors`.
- [ ] **Step 5:** Update `STEPS-FOR-OWNER.md` (not committed):
  - the secret and Claude App steps, if they are still open;
  - the social preview upload;
  - "after ~2 weeks green, decide which new checks become required (candidates: actionlint, codeql c-cpp, fix-proof)".
- [ ] **Step 6:** Open the PR, wait for green and merge.

### Task 8.3: Link check (spec item 10)

**Files:** Create `.github/workflows/link-check.yml`, `.lychee.toml`

- [ ] **Step 1:** Write `.lychee.toml`: exclude `localhost`, the Pages URL for its first day, and `huggingface.co/.*/resolve/` (large-file redirects). Set `max_retries = 3` and `accept = [200, 206, 429]`.
- [ ] **Step 2:** Write the workflow, modelled on daydaze's `link-check.yml`:
  - Triggers: weekly (`41 6 * * 1`) and dispatch.
  - `lycheeverse/lychee-action` (pinned) over `README.md ARCHITECTURE.md CONTRIBUTING.md SECURITY.md docs`.
  - It keeps one open issue, "docs: broken links found by the link check", labelled `documentation`.
- [ ] **Step 3:** Run `lychee --config .lychee.toml README.md ARCHITECTURE.md CONTRIBUTING.md SECURITY.md docs` locally. Expected: 0 errors, or each failure fixed in this task.
- [ ] **Step 4:** Commit `ci: check documentation links weekly`. This task ships in the Phase 8 PR. It sits here, not in Phase 3, because it checks the docs this phase writes.

---

## Final review

- [ ] Run a fresh-context review of all eight merged PRs together (superpowers:requesting-code-review) against the
  spec's success criteria.
  - Confirm that every workflow has its reason comment (`for f in .github/workflows/*.yml; do head -3 "$f"; done`).
  - Confirm that each scheduled workflow has at least one run on `master` (`gh run list -w <file> -b master -L1`).
- [ ] Report to the owner:
  - the dashboard URL;
  - the demo fix-proof run links;
  - the first nightly results;
  - any `ci-failure` issues opened, and what was found.
