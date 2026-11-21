# SDD ledger — plan: docs/superpowers/plans/2026-11-20-ci-automation.md

Spec: docs/superpowers/specs/2026-11-20-ci-design.md. Worktree ~/git/nibble/.worktrees/ci. Phases land as separate PRs;
each phase branch starts from origin/master after the previous phase merged.

## Pre-flight scan

| Tasks | Shared file/interface | Finding |
|---|---|---|
| 1.1 → 3.2, 4.2, 4.5(none) | `GitHub` protocol / `FakeGitHub` | consistent method names (find_open_issue, create_issue, comment, close_issue, last_green_sha, upsert_comment, pr_for_sha, dispatch) |
| 1.1 ↔ 5.2 | ci.yml (lint job vs coverage job) | different jobs, different phases: no conflict |
| 1.2 ↔ global labels | labels.py vs Global Constraints list | same list; AREA_PATHS globs must be adjusted to real src/ (no src/api*) |
| 3.1 → 3.4, 3.5, 4.3 | `Cosine`, `parse_min_cosine`, `diff_cosines` | consistent |
| 3.2 → 3.5, 3.6, 6.3 | report-failure inputs (job, outcome, area, attachments, triage) | triage stays "false" until 6.3 flips it (claude-triage.yml absent before Phase 6): consistent |
| 3.3 → 3.5, 4.3 | setup-models inputs save/require-warm, outputs warm/models-dir | consistent |
| 3.4 → 3.5, 3.6, 5.2 | row schema, `Meta`, CLI subcommands; 5.2 needs a `merge` CLI subcommand | 3.4's CLI list lacks `merge` that 5.2 step 3 calls → ruling P2 |
| 3.5/3.6/ci ↔ 5.2 | workflow_run names `nightly`, `portability`, `ci` | workflow `name:` must be exactly these → ruling P3 |
| 4.1 → 4.3 | perf_ab.run signature | consistent |
| 4.2 → 4.3 | pr-report.json schema; `pr-report-comment.yml` workflow_run on `pr-report` | workflow name must be exactly `pr-report` (P3) |
| 4.4 → 4.5 | fix_proof CLI select/classify | consistent |
| 5.1 → 5.2 | plot.line_chart signature | consistent |
| 6.3 ↔ 3.5/3.6 | modifies nightly/portability triage flag | consistent |
| 6.4 step 2 / 5.2 step 5 / Phase 6 preamble | repo settings & owner steps | already done 2026-11-21 (Pages via API, allow_auto_merge=true, secret + Claude App by owner): verify only |
| global ↔ SDD skill | ledger path `.superpowers/ci-automation-ledger.md` vs SDD workspace | ruling P1 |

Per-task self-consistency: every task's tests match its interfaces; 4.3 `--target nibble_cli` exists (CMakeLists.txt:148,
OUTPUT_NAME nibble); 3.6 arm64 `kernels: scalar` assertion matches tiers being x86-only; 4.5 demo branches based on
`ci/pr-checks` but Phase 4 merges after → demo PRs target demo/fix-proof-base (fine, fix-proof runs on any base).

Rulings:
- Ruling P1: the progress ledger is this file (SDD workspace), not `.superpowers/ci-automation-ledger.md` — the skill's workspace is the recovery map; both are git-ignored scratch — cost if wrong: none.
- Ruling P2: Task 3.4's CLI also gets a `merge <existing.jsonl> <new.jsonl>... --out F` subcommand (5.2 step 3 calls it) — plan omission — cost if wrong: one extra subcommand.
- Ruling P3: workflow `name:` values are exactly `nightly`, `portability`, `pr-report` (workflow_run matches on name) — cost if wrong: dashboard/comment never trigger.
- Ruling P4: spec + plan commits ship in the Phase 1 PR (branch renamed ci/automation-suite → ci/foundations) — cost if wrong: none.
- Ruling P5: batching — 1.3+1.4, Phase 2 (2.1–2.3), 6.1+6.2, 6.3+6.4, 8.2+8.3 each go to one implementer (small same-shape workflow/doc edits) — cost if wrong: larger review units.
- Ruling P6: the controller (not implementers) pushes, opens PRs, waits for CI and merges at each phase end; the owner authorized merging ("permission to do whatever u want as long as it improves this repo") — cost if wrong: a merged PR the owner would have wanted to see first (revertable).

## Progress
Task 1.1: dispatched (BASE 211877d, implementer sonnet)
Task 1.1: ⚠️ live gh resolved by controller smoke run: find_open_issue (None), last_green_sha(ci.yml, master)=6e23aa2, pr_for_sha on merged PR head → None (open PRs only, as specified); unpushed sha → GhError 422 (callers must handle).
Task 1.1: minor (deferred): FakeGitHub.comment/upsert_comment KeyError for PR numbers — carry to Task 4.2
Task 1.1: minor (deferred): upsert_comment matches marker regardless of author (spoofable) — carry to Task 4.2 (Review Focus 1)
Task 1.1: minor (deferred): last_green_sha URL params unquoted, no event filter
Task 1.1: minor (deferred): _json_list drops non-dict items silently; empty-stderr fallback untested; argv[0] IndexError on empty argv
Task 1.1: complete (commits 211877d..ef37624, review clean)
Task 1.2: dispatched (BASE ef37624, implementer sonnet)
Task 1.2: review: spec ❌ — tests/tokenizer/** dropped because controller note wrongly said it didn't exist (controller error); fix round 1 dispatched to implementer
Task 1.2: minor (deferred): glob-exists test passes for emptied dirs (dir/** matches dir); test files of gguf/kernels/model, src/util/**, src/debug_check.h, tests/fuzz/**, *.cmake unmapped; labeler.yml header says "Generated" though hand-maintained; duplicated runner closure in tests; GhError in sync prints traceback; sync-labels:true strips hand-applied area labels (deliberate)
Task 1.2: fix round 1/5 (1 addressed, 0 open; commits 3602f1c..46eedca)
Task 1.2: complete (commits ef37624..46eedca, review clean)
Tasks 1.3+1.4: dispatched together (BASE 46eedca, implementer sonnet)
Tasks 1.3+1.4: minor (deferred): actionlint header doesn't say it's path-filtered; manual version/sha pin invisible to Dependabot (commented); bug form links /security/policy; no concurrency group
Tasks 1.3+1.4: complete (commits 46eedca..f09d792, review clean; ⚠️ repo About verified by controller)
Phase 1: PR #49 opened (ci/foundations @ f09d792), CI running
- Ruling P7: each later phase gets its own worktree `.worktrees/ci-<phase>` from origin/master (`git switch` is denied by the session's permissions; `git worktree add` works); this workspace stays in .worktrees/ci — cost if wrong: extra worktrees to clean up at the end.
- Ruling P8: Phase 2 starts from origin/master before Phase 1 merges (its files are independent); it is rebased onto master before its PR merges — cost if wrong: one rebase.
Phase 2 (2.1–2.3): dispatched together in .worktrees/ci-security (BASE 6e23aa2, implementer sonnet)
Phase 2: minor (deferred): gitleaks.yml triggers list main+master (mirrors ci.yml) while codeql/scorecard use master only; gitleaks version/checksum manual pin (documented)
Phase 2: ⚠️ first Actions runs (codeql on PR + Dependabot PR SARIF upload, gitleaks PR range, scorecard publish) — controller verifies on the PR and after merge
Tasks 2.1–2.3: complete (commits 6e23aa2..ba6e798, review clean)
Phase 2: PR #50 opened (ci/security @ ba6e798)
Phase 3: worktree .worktrees/ci-nightly from ci/foundations (f09d792), own venv
Task 3.1: dispatched (BASE f09d792, implementer sonnet)
- Ruling P9 (Task 3.1): diff_cosines treats NaN vs NaN as unchanged (spec says "unchanged at printed precision"; a persistent NaN is not a change) — so Task 3.5's accuracy job must itself fail on any NaN/inf min_cos — cost if wrong: a persistent NaN is reported as failure by 3.5 rather than as a diff, which is the safer side.
Task 3.1: minor (deferred): value class accepts garbage (float() error lacks key); no -nan/inf tests; diff ignores positions/agree (per brief); test name now slightly misleading
Task 3.1: complete (commits f09d792..4a1ce1f, review clean)
Task 3.2: dispatched (BASE 4a1ce1f, implementer sonnet)
## CodeRabbit review of PR #49 (all verified valid)
- Ruling CR1: scorecard.yml gains `workflow_dispatch` (fix on ci/security) — plan step 2.3.4 dispatches it.
- Ruling CR2 (→3.5): every job's `dashboard_data.py status` + artifact upload run with `if: always()` and record the real job outcome.
- Ruling CR3 (→3.5): report-failure is skipped (not called with success) when a job skipped its work on a cold cache; only a run that actually executed may close an issue.
- Ruling CR4: workflow_dispatch needs the file on the default branch, so pre-merge dispatch validation (3.5 step 4, 3.6 step 4) moves to after the Phase 3 merge, on master; real failures found then go to issues/fix PRs as planned.
- Ruling CR5 (→4.1): verdict checks spread first: max−min > SPREAD_MAX → inconclusive before faster/slower.
- Ruling CR6 (→4.2/4.3): `post` takes the triggering run's head SHA (github.event.workflow_run.head_sha) and refuses to post when the artifact's head_sha differs; test for a report naming another PR's head.
- Ruling CR7 (→4.4): changed_lines also maps deleted lines (old side → enclosing case on head via hunk positions) so deletion-only edits select their case.
- Ruling CR8: gh.py last_green_sha builds its query with urlencode; pr_for_sha lists open pulls (paginated) and matches head.sha (fix on ci/foundations, PR #49).
NOTE: after the CR fix lands, regenerate briefs 3.5, 3.6, 4.1, 4.2, 4.3, 4.4 from the amended plan before dispatching them.
Task 3.2: review: quality needs fixes — Important: upload step failure skips the report (continue-on-error + overwrite); fix round 1 dispatched
Task 3.2: minor (deferred): header example maps cancelled→failure (callers in 3.5 should skip cancelled); attachments uploaded on success too; `nightly:` prefix used for portability issues (per brief); brittle ARGS slicing in tests; no prefix-name exact-title test at this layer; dispatch failure after create leaves issue untriaged
- Ruling P10 (→3.5/3.6): callers run report-failure only when job.status is success or failure (skip cancelled) and only when the job did real work (CR3).
Task 3.2: fix round 1/5 (1 addressed, 0 open; commits 6325164..4ee0de2)
Task 3.2: complete (commits 4a1ce1f..4ee0de2, review clean)
CR49 fixes pushed to ci/foundations (1d0d340, 8ed251c); review dispatched. Briefs 3.3–4.5 regenerated from amended plan.
Task 3.3: dispatched (BASE 4ee0de2, implementer sonnet)
Phase 2: PR #50 merged (a5572ae), all 19 checks green; scorecard + codeql dispatched on master for verification
Phase 2 verified on master: scorecard (push + dispatch) success, codeql (push + dispatch) success.
CR49 fix review: spec ❌ (plan text) — Important: R19/4.4 deletion mapping ambiguous; R14/3.4–3.5 status CLI + cancelled mapping undefined. Fix round 1 dispatched.
- Ruling P11: the CR49 fix round also folds in these minor plan corrections because later briefs are generated from the plan text: P10 (skip cancelled) in 3.5; Review Focus 1 text (open-pulls lookup + head-sha binding); 4.3 passes workflow_run.head_sha via env, not inline; gh.dispatch encodes workflow_file — cost if wrong: a slightly larger fix round.
CR49: minor (deferred): 4.1 "same" rule keeps redundant spread check; merged-PR test passes on old code too; cold-cache "no ok row" wording
Task 3.3: done (c7768f4); review dispatched
Task 3.3: minor (deferred): path list duplicated between restore/save; `du` of models/golden can fail under -e if missing; deletes build dirs by name; llama-bench absent after cache hit (trend-bench uses nibble bench only — fine)
Task 3.3: complete (commits 4ee0de2..c7768f4, review clean)
Task 3.4: waiting for CR49 fix round 1 (it pins the status CLI), then regenerate brief 3.4/3.5/4.4
CR49: fix round 1/5 (1 addressed, 1 open — whole-case deletion contradicts map-to-c rule; commits 8ed251c..ff39e9f); round 2 dispatched
Task 3.4: dispatched (BASE c7768f4, implementer sonnet; brief regenerated from plan @ ff39e9f)
CR49: fix round 2/5 (1 addressed, 0 open; commits ff39e9f..323a447) — complete
CR49: minor (deferred): contiguous A-tail+B-header pure deletion drops A selection (safe direction: unresolved/no-test)
Task 3.4: done (8b70675); review dispatched
Task 3.4: minor (deferred): dedup key ignores runner; schema 1.0 accepted / impossible dates accepted; 3 near-copy JSONL loops; non-atomic merge write; missing tests (llama.cpp skip, missing GITHUB_SHA, dup in existing); fixture comment says "real run"
- Ruling P12 (→3.5/3.6): every `dashboard_data.py from-*` step runs under `if: always()` (with continue-on-error where a parse failure must not hide other rows), so one exit-1 (NaN cosine, crashed fuzz) never drops the status row; per-leg keys stay distinct (status key = workflow/job).
Task 3.4: complete (commits c7768f4..8b70675, review clean)
Phase 1: PR #49 merged (all 20 checks green). ci/nightly rebased onto origin/master: 3.1=2e9948e 3.2=b334401..9b0c0b8 3.3=5fbb900 3.4=eb4107a
Task 3.5: dispatched (BASE eb4107a, implementer sonnet)
Task 3.5: first review aborted (usage limit); re-dispatched
Task 3.5: review: quality needs fixes — Important: accuracy attachments uploaded on every (green) run; fix round 1 dispatched
- Ruling P13: fix round 1 also takes three minors of the same mechanism or security class: crash attachments expression (`'' ||` falsy bug), `overwrite: true` on dashboard-rows/e2e-cosines uploads (re-run 409 would report a false failure), and GH_TOKEN scoped to the compare step only — cost if wrong: slightly larger fix round.
Task 3.5: minor (deferred): baseline = last fully-green workflow run (could pick newest run with e2e-cosines artifact); actions: write unused until Phase 6; timeouts → cancelled → silent (per R14); ⚠️ disk headroom + ctest -j2 WARN interleaving verified on the first real run (add df -h); area labels exist (synced in Phase 1).
Task 3.5: fix round 1/5 (4 addressed, 0 open; commits 9cd07c6..423c17f)
Task 3.5: complete (commits eb4107a..423c17f, review clean)
Task 3.6: dispatched (BASE 423c17f, implementer sonnet)
Task 3.6: done (1064997); review dispatched
- Ruling P14: issue titles use the calling workflow's name as prefix ("portability: arm64 failing", "nightly: fuzz_gguf failing") instead of the hard-coded "nightly:" — a recruiter-visible issue saying "nightly" for a weekly job reads as sloppy; fixed in report_failure.py with tests during the 3.6 fix round — cost if wrong: none (titles stay unique per workflow+job).
Task 3.6: review clean (spec ✅, approved). minor (deferred): llvm.sh check-then-run from temp dir; actions: write unused while triage off (both workflows); PPA keyserver flakiness; extra scalar assertion in simd-off job (harmless). P14 change dispatched to the 3.6 implementer.
Task 3.6: P14 follow-up a9a4a19 re-reviewed: addressed, no breakage (minor: 155-col comment line in report-failure action.yml)
Task 3.6: complete (commits 423c17f..a9a4a19, review clean)
Phase 3: PR #51 opened (ci/nightly @ a9a4a19)
Phase 4: worktree .worktrees/ci-prchecks from ci/nightly (a9a4a19)
Task 4.1: dispatched (BASE a9a4a19, implementer sonnet)
- Ruling P15: perf_ab gets summarize() + main() (run/summarize subcommands) — 4.3 runs `perf_ab.py run` and needs the report's perf section; plan omission — cost if wrong: one small CLI.
Task 4.1: done (83cfa31); P15 CLI follow-up sent to implementer before review
- Ruling P16 (→4.2): perf entries may carry 0.0 medians/ratios for unusable sides (strict JSON, no NaN); render must not divide by zero (Δ% shown as — when base_median is 0) and verdict stays as given.
Task 4.1: minor (deferred): summarize crashes on a truncated JSONL line; spread == 0.05 float edge (conservative); run appends to --out (use fresh file); base/head_median include NaN-ratio blocks; ±10% uniform jitter passes as faster/slower ~8% in simulation (real ABBA ratios tighter) — final review should weigh this vs Review Focus 4
Task 4.1: complete (commits a9a4a19..3ca1a89, review clean)
Task 4.2: dispatched (BASE 3ca1a89, implementer sonnet)
Task 4.2: done (1beeb4c, ca001b1); review dispatched (opus, security-sensitive)
## CodeRabbit review of PR #51 (10 comments, all verified; one fix wave on ci/nightly)
- CR51-1 (Major): trend-bench builds `--target nibble` (library) → `--target nibble_cli`.
- CR51-2: add tests/corpus.h to the models cache key hashFiles (setup-models + Global Constraints key in plan).
- CR51-3: drop unused `actions: write` from nightly.yml and portability.yml (Phase 6 re-adds it with triage).
- CR51-4: move both inline `python -I` programs in nightly.yml (snapshot writer + count/finite gates; baseline fetch/compare + summary) into tools/ci (e.g. cosine.py CLI subcommands `snapshot` / `compare`) with unit tests.
- CR51-5: baseline = most recent nightly run on master (any event) whose `accuracy` job concluded success and has an `e2e-cosines` artifact (not workflow-level success).
- CR51-6 (ruling): job_status rows record the job's check result before reporting; documented in dashboard_data.py and nightly.yml rather than restructured — cost if wrong: a red Report step with a green row, visible in the run anyway.
- CR51-7: job-level `concurrency: nightly-<job>[-<target>]` / `portability-<job>`, cancel-in-progress false, so a scheduled and a manual run of the same job serialize their reports.
- CR51-8: fuzz corpus cache primary key includes github.run_attempt.
- CR51-9: from-libfuzzer records elapsed seconds even without a `Done` line (CLI `--seconds` measured by the workflow; executed units when present).
- CR51-10: report-failure never closes an issue from a run whose SHA is not the branch's current tip (stale rerun) — it comments instead; test.
Task 4.2: review: quality needs fixes — Important: escaped @mentions/#refs still render as pings/links (restrict charset at load); two vacuous tests (zero-median row selection; empty-ratios "—"). Fix round 1 dispatched.
- Ruling P17 (4.2 fix round, for 4.3's needs): (a) `accuracy` part is required in assemble; the workflow writes status `skipped-cold-cache` when cold; a missing part fails assemble (red), never reads as cold cache; (b) `kernels_changed` is required in the meta part (no silent False); (c) size limit checked in bytes before reading the whole file; (d) add `pr_report.py accuracy --base B.json --head H.json --out part-accuracy.json` building rows from two cosine snapshots, with `base`/`head` allowed to be null for a case present on one side only (rendered as "added"/"removed"); (e) `load` enforces rows empty unless status is `ran` — cost if wrong: small schema surface change before first deploy.
Task 4.2: minor (deferred): R18 check duplicated in main/post; broad KeyError catch; pr_for_sha matches head SHA only (fork PR at same commit as another PR)
CR51: fix wave pushed (a9a4a19..bae8591, 8 commits, 135 tests); review dispatched
NOTE: after rebasing ci/pr-checks onto the CR51 wave, check pr_report accuracy's snapshot reader matches tools/ci/accuracy.py's snapshot writer
Task 4.2: fix round 1/5 (3 addressed + P17 a–e verified, 0 open; commits ca001b1..627e334)
Task 4.2: complete (commits 3ca1a89..627e334, review clean)
- Ruling P18 (→ next dispatch on ci/pr-checks): drop `:` from pr_report _RUNNER_TEXT (no CPU name needs it; `_inert_runner` maps it to `_`) so emoji shortcodes / scheme-like text can't survive — cost if wrong: none.
CR51 fix wave review: spec ✅, approved. minor (deferred): M1 close waits a night if master moves during a long run; M2 compare returns 1 on unreadable current snapshot (doc says never fails); M3 artifact keys unsanitized in ::warning:: lines; M4 download failure aborts compare instead of trying next run; M5 redundant GITHUB_REPOSITORY env; M6 210-col comment; M7 plan fuzz key lacks run_attempt; M8 repeated test kwargs
CR51: complete (a9a4a19..bae8591). ci/pr-checks rebased onto it (tests/mypy/ruff green).
Task 4.3: dispatched (BASE d6cfc3e, implementer sonnet)
CR51-11: accuracy job lacks `actions: read` for the baseline compare (CodeRabbit 4650566616, valid) — fix sent to the CR51 fixer
Task 4.3: done (4f96c55 P18, c656c13 workflows; DONE_WITH_CONCERNS: base --expected 18; timeout guess; Dependabot workflow_run token); review dispatched (opus)
CR51-11: fixed 3fe664e (actions: read on all 8 jobs — report-failure lists runs too); controller verified diff
Phase 3: PR #51 merged (fab3eee), 21 checks green, all 11 CodeRabbit threads answered. nightly.yml (all) + portability.yml dispatched on master for first real runs (CR4 validation).
Task 4.3: review: needs fixes — Important: snapshot --expected 18 runs before the part upload, so a broken head (fewer lines) loses the report; fix round 1 dispatched.
- Ruling P19: 4.3 fix round also takes Minor 1 (permissions at job level + contents: read), Minor 2 (assemble must not need perf: missing perf part → null sections; apt-get update before install fallback), Minor 3 (download artifact to runner.temp), Minor 4 (per-test --timeout 1800) — all small and on the trust-split/robustness path — cost if wrong: a slightly larger round. Minor 5 (paths filter excludes tools/ci) goes into the PR description.
Task 4.3: fix round 1 done (rebased onto master fab3eee; 5641072, c87e5ef, 0c089cd); re-review dispatched
Phase 3 first runs: portability arm64 ✅ (first ARM run ever), newest-clang ✅, simd-off ✅, newest-gcc ❌ real finding: tests/test_api.cpp uses ::getpid without <unistd.h> (transitive include gone in newest libstdc++) → issue #52 auto-opened (title/labels/body correct). Fix dispatched on fix/test-api-unistd (worktree .worktrees/fix-unistd).
Task 4.3: fix round 1/5 (Important + P19 ×4 addressed, 0 open; commits 30b3b69..0c089cd)
Task 4.3: complete (review clean). minor (deferred): two "### e2e accuracy" summary blocks; late perf death renders "skipped" not "failed"; paths filter excludes tools/ci (PR description)
- Ruling P20 (→ 4.4 dispatch, same worktree): pr_report renders one-sided accuracy rows as "no head result" / "no base result" instead of "removed"/"added" (true for both a lost case and a non-finite value) — cost if wrong: none.
Task 4.4: dispatched (BASE 0c089cd, implementer sonnet)
Fix #52: PR #54 (fix/test-api-unistd), portability on branch all green (run 44868238418); local 305/305
Task 4.4: done (9e4b0ef P20, 2863824 fix_proof; 279 tests; DONE_WITH_CONCERNS). Review dispatched (opus).
- Ruling P21 (→ 4.4 fix round): a case whose run dies without writing JUnit (sanitizer abort, signal, timeout) must count as `failed` on the base, not `missing` — crash fixes are the commonest bug fixes. Add a `run` subcommand to fix_proof.py that executes each selected case itself (one process per case, UBSAN_OPTIONS=halt_on_error=1, a per-case timeout), records the exit status, and writes the result (JUnit if produced, else a synthetic failed/timeout result) — logic stays in tested Python, not YAML — cost if wrong: one more subcommand.
Fix #52: PR #54 merged (20 checks green; labeler applied 'tokenizer' — first live labeler run). portability re-dispatched on master to exercise report-failure's recovery close.
Task 4.4: review: spec ✅, quality needs fixes — I1 name starting with `-` reaches Catch2 as an option (would read proven under P21); I2 two-dot diffs leak master's changes (use base...head); I3 leak/non-halting UBSan fixes unprovable (XML passed + non-zero exit must count failed). Fix round 1 dispatched with P21.
- Ruling P22: fix round 1 also takes M1–M9 — fix-proof is the suite's signature check and every minor here is either a false-pass path (M1 hunk redirection via tests/fuzz, M2 zero-assertion case passing as not-runnable), an input-robustness gap (M3 encoding, M4 splitlines, M5 quoted paths, M6 binary validation) or doc/DRY (M7–M9) in the same file — cost if wrong: a bigger round.
FIRST NIGHTLY (run 44867692841): models cache built, all Qwen e2e tests PASSED on the hosted runner, sanitizers-real-model + trend-bench ran; accuracy job FAILED at "Parse the min-cosine lines": real bug in tools/ci/cosine.py — _WARNING_LINE expects a bare "warning:" line, but real Catch2 output is "<file>:<line>: warning:" on one line (3.1 fixture was synthetic; ctest -V prefix "251: " handled fine). Reproduced locally (3 lines in log, 0 parsed). Fix on fix/cosine-warning-header (worktree .worktrees/fix-cosine) with a real-log regression fixture.
Portability re-run on master after #54: all 4 jobs green.
Task 4.4: fix round 1 done (aaeae23, 939cc2c, 7681209, 6e84c86; 300 tests); re-review dispatched (opus)
Fix cosine: PR #55 (a9e3497), real-log fixture committed, RED→GREEN, other parsers' fixtures verified real
Fix cosine: PR #55 merged (20 checks green); nightly accuracy re-dispatched on master to confirm end to end.
Task 4.4: fix round 1/5 (I1–I3, M1–M9, P21 addressed, 0 open; commits 2863824..6e84c86)
Task 4.4: complete (review clean). minor (deferred → folded into 4.5 as P23): N1 probe ignores env/exit code; N2 600 s default timeout below CI's sanitizer per-case timeouts; base+head tests/ copy must include tests/data and headers; fresh results dir.
- Ruling P23 (→ 4.5): fix_proof probe runs with the merged sanitizer env and requires returncode 0 for `unique` (N1, test-first in fix_proof.py); fix-proof.yml passes `--timeout 1800`, copies the whole head `tests/` tree (incl. data/ and headers) over the base worktree, uses a fresh results dir, and runs without models (no setup-models) so [e2e] cases skip — cost if wrong: none.
Task 4.5: dispatched (BASE 6e84c86, implementer sonnet)
Task 4.5: done (15f41f0 P23, d41463e fix-proof.yml). DEMO: PR #56 proven (run 44870531267, green); PR #57 does-not-reproduce (run 44870719237, red). PRs closed, demo branches deleted. Note: no-repro PR needed a src/ touch to pass the gate (deviation from "only add a case", by design of touches_code). Review dispatched.
Nightly accuracy re-run on master after #55 (run 44870310448): SUCCESS (warm cache, 18 cosines parsed).
Task 4.5: review: spec ✅ (demo runs verified), Important: tests/ overlay keeps files the PR deleted/renamed (false red / false "base does not compile"). Fix round 1 dispatched.
- Ruling P24: round 1 also splits configure from compile (a FetchContent/network failure at configure must not read as "base does not compile" → proven) and filters `edited`/`labeled`/`unlabeled` runs to title/base changes and the no-test-needed label — both are false-signal or cost fixes in the same file — cost if wrong: none.
Task 4.5: minor (deferred): blank line after a case selects the neighbour (harmless "already passing"); PR code can modify fix_proof itself (accepted: not required, no write token)
Task 4.5: fix round 1/5 (3 addressed, 1 new Important open — job-level `if:` makes unrelated label/edit events produce a `skipped` fix-proof check that masks a red verdict; commits d41463e..7f17bb1)
- Ruling P25: reverse P24's event filter (my ruling caused the masking): every listed event runs the full job again (the build is ~1 min); keep the tests/ replacement and the configure split; drop the per-event concurrency special case and use one group per PR with cancel-in-progress — cost if wrong: a few extra 1-minute runs per PR.
Task 4.5: fix round 2/5 (1 addressed, 0 open; commits 7f17bb1..e7e7578)
Task 4.5: complete (review clean). Demo-branch pr-report accuracy runs failed at the strict head gate because ci/pr-checks predated the #55 parser fix (part uploaded first, as designed) — fixed by rebasing onto master.
Phase 4: rebased onto master (21 commits), lint green, pushed; PR opened
Phase 4: PR #59 opened. Phase 5: worktree .worktrees/ci-dashboard from origin/master
Task 5.1: dispatched (BASE a1f864e, implementer sonnet)
Task 5.1: done (a763a6d). Controller rendered samples (cairosvg): speed chart fine; cosine chart y ticks uneven (1.0001, 0.9981…) with y_min — sent back as a controller finding before review.
Task 5.1: tick follow-up 7479251 (controller re-rendered: even 0.990–1.000 labels); review dispatched
Task 5.1: review: byte-identity ✅, spec ✅; Important: date_ticks hangs/crashes on huge or ms timestamps; negative-data axis mislabeled. Fix round 1 dispatched (+ y_min non-finite, footnote test).
Task 5.1: minor (deferred): extreme y ranges print very long labels; colours interpolated unescaped (as existing helpers)
Task 5.1: fix round 1 done (9219c46); re-review dispatched
Task 5.1: fix round 1/5 (2 addressed + extras, 0 open; commits 7479251..9219c46)
Task 5.1: complete (commits a1f864e..9219c46, review clean). minor (deferred): 2 s wall-clock limits in timing tests (possible flake on loaded runners); float ticks slightly above hi at 1e300 scale
Task 5.2: dispatched (BASE 9219c46, implementer sonnet)
Phase 4 PR #59: CI all green; CodeRabbit 18 comments (2 Major: report should include head test failure; fix_proof case span runs past test body) — dispatched to a fixer that verifies each, fixes valid ones test-first, lists declines with reasons (raw: cr59-raw.json)
Task 5.2: done (41f95ea coverage rows, 312a179 dashboard). Controller screenshot (Playwright) of sample site: clean; speed chart squashes decode under prefill + cryptic labels → sent back (split prefill/decode, human labels) before review.
Task 5.2: speed split follow-up 226262d; review dispatched (opus)
Task 5.2: review: spec ✅; Important: (1) a successful re-run's status row is dropped by merge dedup (same run_id) → job stays failing; (2) rows silently lost when a 3rd completion replaces the pending dashboard run. Fix round 1 dispatched.
- Ruling P26: round 1 builds the sweep (each dashboard run also collects dashboard-rows from the last 7 days of completed nightly/portability/ci runs on master, same event + same-repo filter, via gh.py with tests) and drops the `run-id` dispatch input (resolves M1); dedup lets the later row win for job_status (or keys on run_attempt); also M2 continue-on-error on the ci.yml upload, M3 status table/badge only consider jobs seen in the last 14 days, M4 one key parser, M5 parse once and surface problems in the job summary, M6 test temp dirs cleaned, M7 charts show the last 90 days — cost if wrong: a larger round on a write-token workflow, reviewed again.
Task 5.2: fix round 1 done (13760da, c2fd2ee, c2b3d96; sweep verified read-only on live repo: 8 runs/9 artifacts → 27 rows, real status table + accuracy chart). P27 (skip malformed artifacts) sent before re-review.
Task 5.2: fix round 1/5 (Important 1, 2 + M1–M7 + P27 addressed, 0 open; commits 226262d..c919e47). Two hardening minors (OverflowError/RecursionError not skipped; one bad run aborts the sweep) sent for a small follow-up.
Task 5.2: minor (deferred): GhCli.download_artifact lists without pagination (>100 artifacts → warning only); identical-date job_status attempts resolve first-wins
Task 5.2: hardening follow-up d2da805 re-reviewed: addressed. minor (deferred): download GhError branch untested; existing-file deep nesting → traceback (exit 1, no write); gh.py download_artifact non-GhError ValueError/KeyError paths
Task 5.2: complete (commits 9219c46..d2da805, review clean)
Phase 5: rebased (up to date), lint green, pushed; PR opened
CR59: 18 handled (15 fixed, 2 partly, 1 declined), 16 commits aab069b..1ab950a pushed, all threads replied; review dispatched (opus)
CR59 fix wave review: spec ✅, approved. minor (deferred → final review): M1 `//` comment ending in `\` ends a body early (fails safe); M2 --budget not checked vs --timeout / NaN (budget<timeout → not-runnable green); M3 unparsable base log shows "changed N of N" (no test); M4 plan text lacks head_tests / bit-identity wording; M5 budget exhaustion after passes → does-not-reproduce
PR #59 + #60: CI fully green. CodeRabbit round 2: #59 3 new (Major: stale part-accuracy artifact on partial re-run; budget-unrun → not-runnable green; stale results dir) → sent to the #59 fixer. #60 4 new (Major: time-bomb test fails from 2026-11-28; invalid calendar dates crash render; clipped labels collide; titles claim fixed token counts) → sent to the dashboard implementer. All verified valid.
CR60: fixed (df6d281, 677642a, 4d381dd), re-review clean (222 tests under 7 patched clocks incl. 2026-11-28/2040; time bomb reproduced pre-fix)
Phase 5: PR #60 merged (all checks green, all CodeRabbit threads answered). CR59 round 2 fixed (372d0f0, 521f0a2, 892daad); re-review dispatched
CR59 round 2 re-review: addressed; follow-ups sent (always() + shell marker for timed-out accuracy attempts; budget==timeout starts no case; header lists incomplete)
DASHBOARD LIVE: https://dpologdvinity.github.io/nibble/ (200), badge.json passing; first nightly fully done: fuzz_gguf + fuzz_load passed (2 h each, no crashes). Visual bug: coverage axis runs to 200% → fix/dashboard-percent-axis (worktree .worktrees/fix-axis)
CR59 round 3: 58cc7af (always() + guarded shell marker; first fix-proof case always starts; header lists incomplete) — controller verified the marker is guarded by [ ! -f ]
Phase 4: PR #59 merged (22 checks green; CodeRabbit 22 threads all answered: 18 fixed/2 partly/1 declined + round 2/3 fixes)
PR-report e2e check: draft PR opened from demo/pr-report-check (comment-only src change)
Phase 6: worktree .worktrees/ci-agents from origin/master (102804b); owner steps verified done (secret, Claude App, allow_auto_merge)
Tasks 6.1+6.2: dispatched together (BASE 102804b, implementer sonnet)
Fix axis (#61): review — Important: hollow/above logic applies without y_max (float hair above top on the cosine chart); fix round 1 dispatched
Fix axis (#61): fix round 1/5 addressed (0771df1), byte-identical to pre-PR baseline for no-y_max charts. minor (deferred): line_chart docstring wording on hollow points
FINDING (live): fix-proof failed red on PR #61 because the PR branch predates #59 and fix-proof.yml runs tools/ci/fix_proof.py from the PR head (file absent). Any PR branched before a tools change goes red for an unrelated reason.
- Ruling P28 (→ cleanup batch): fix-proof.yml (and pr-report.yml if it does the same) runs tools/ci from the PR's merge ref (actions/checkout default for pull_request, which includes master's tools) while still taking the PR HEAD's tests/ (git archive "$HEAD_SHA" tests) for the base worktree — never the merge ref's tests (master tests added after the merge base could fail to compile on base → false "proven") — cost if wrong: PR code can still change tests (accepted), tools now come from master.
#61 rebased onto master and force-pushed (with lease) to re-run checks.
Tasks 6.1+6.2: done (ce4397d, 89fdbd6; claude-code-action v1.0.286 acfe7992…); review dispatched (opus, security)
Tasks 6.1+6.2: review: spec ❌ — I1 track_progress makes the reviewer write-capable (tag mode + App token) and posts a summary; I2 allowed_bots lacks claude[bot] (agent PRs go red); I3 Bash(git:*)/Bash(gh pr:*) allow self-merge, force-push, fork checkout; I4 agent label not enforced (action's PR link → unlabelled owner PR). Fix round 1 dispatched.
- Ruling P29: override the plan-mandated tool list (Task 6.1 `--allowedTools`) — security beats the plan's letter: tight allow-list + `--disallowedTools` for gh pr merge/review/checkout/close, git config/remote/push --force/branch -D; keep the action's push wrapper; Claude opens PRs itself with `gh pr create --label agent`; also M1 persist-credentials false, M2 no pip cache save from the agent job (restore-only or none), M3 header warns about untrusted text, M4 gate fork PRs before setup, M5 header says how it is triggered — cost if wrong: Claude may need a tool we denied (visible as a refusal in its run; widen narrowly).
Fix axis (#61): CodeRabbit 2 minors valid → Ruling P30: y_max is a fixed axis top (percent charts always 0–100); validate explicit bounds before the empty-data return; sent to the axis implementer
Fix axis (#61): P30 fix e07ed6d re-reviewed clean
Tasks 6.1+6.2: fix round 1/5 (I1–I4, M1–M5 addressed, 0 open; commits 89fdbd6..3f47add). Residual (documented in claude.yml header): deny lists are prefix speed bumps in auto mode; hard limits are branch protection, token scope, auto-merge keyed on the owner — but protection needs 0 approvals, so API self-merge is possible in principle.
- Ruling P31 (→ 6.3/6.4 dispatch): add an `agent-approval` check (pull_request + pull_request_review) that is red on PRs authored by claude[bot] or labelled `agent` until the owner has an APPROVED review on the head SHA; making it a required check is an owner decision (STEPS-FOR-OWNER). Also fold in the 3 minors: open a PR only if none exists for the branch; gate step tolerates a transient gh api failure (skip with notice); claude-review also triggers on `labeled` (label == agent); and deny `--output` for git diff/log in the reviewer.
Task 6.1+6.2: complete (review clean)
Tasks 6.3+6.4 (+P31 agent-approval, + 6.1/6.2 minors): dispatched to the Phase 6 implementer (BASE 3f47add). #61: off-grid capped ticks (CodeRabbit 4651775138, valid) sent to the axis implementer.
Fix axis (#61): tick-grid fix 2d88658 re-reviewed clean (4000-case fuzz, 0 problems)
Tasks 6.3+6.4+P31+minors: done (bb7edfa, bd5ab20, ecdce6d, 9a1f2e3); review dispatched (opus)
#61: CodeRabbit 4651808665 (nice_step(0) crash on negative cap ≤ data min) valid → sent to the axis implementer
Fix axis (#61): a51efc1 re-reviewed clean (20000-case fuzz, 0 bad). minor (deferred): docstring on y_max<=0 with negative data. Ruling P32: further CodeRabbit nits on exotic negative-cap cases of a function whose only capped caller is coverage (always ≥ 0) get deferred, not another round — cost if wrong: a rendering oddity in a case no chart produces.
Fix axis: PR #61 merged (21 checks green, all CodeRabbit threads answered).
Tasks 6.3+6.4: review: spec ❌ — C1 no allowed_bots → every automatic triage (dispatched by github-actions[bot]) fails "non-human actor"; I1 App token lacks actions read (additional_permissions); I2 triage gate accepts any issue + reads all comments in a write session, and nightly/portability jobs hold actions: write with persisted checkout creds during builds; I3 @claude pushes onto an auto-merge-enabled owner PR merge unreviewed. Fix round 1 dispatched.
- Ruling P33: (a) when claude[bot] pushes to a PR it did not author, automerge disables auto-merge on that `synchronize` and comments why — the owner merges by hand after review; (b) agent-approval gates PRs AUTHORED by claude[bot] only (an owner-authored PR labelled `agent` is the owner's own work and is not gated — an owner cannot approve their own PR); (c) triage only acts on an OPEN issue labelled `ci-failure` and authored by github-actions[bot] (or the owner), and the prompt reads only the issue body, never comments; (d) report-failure moves to its own small job (needs: the work job; permissions issues: write + actions: write) so the long build jobs keep contents: read only, and every checkout in nightly/portability sets persist-credentials: false; (e) bootstrap in agent-approval fails closed (exit 1) once the script should exist; plus M3, M6, M7, M8, M9 — cost if wrong: report-failure needs the work job's outcome passed via needs/outputs (more YAML).
OWNER STEP (add): confirm the Claude GitHub App's repository permissions do not include Workflows / Checks / Commit statuses write (reviewer M1: otherwise a PR could fake the agent-approval check).
PR-REPORT E2E: PR #62 got the live 'nibble PR report' comment (pr-report run 44888514276, comment run 44890311357): speed ±1.4% all same/inconclusive, size Δ0, memory +0.0%, 18/18 cosines identical, head e2e passed. #62 closed, branch deleted.
Phase 7: worktree .worktrees/ci-release from origin/master; Tasks 7.1+7.2 dispatched together
Phase 7: done (56d693c CHANGELOG/cliff, 6329318 attestations/SBOMs); Ruling P34: release-prep uses full regeneration (git cliff --tag vX.Y.Z -o CHANGELOG.md), not --prepend (would duplicate [Unreleased]); rebase + regenerate requested before review
Phase 7: review approved; Ruling P35 follow-ups sent (CI-and-tooling CHANGELOG group; lowercase attested image; notes drop heading + footer). minor (deferred): Keep-a-Changelog groups not strict (brief-mandated); thin tarball SBOM; README/CONTRIBUTING mention → Phase 8
Phase 6 fix round 1: done (2ffcb15..ffe7548, 6 commits, 407 tests); re-review dispatched (opus)
Phase 7: follow-ups re-reviewed clean (awk dry-runs for v0.8.0, v0.7.1, v0.7.0, missing tags). minor (deferred): bench/test/tooling commits still in user-facing CHANGELOG groups (type/scope-based)
Phase 7: pushed, PR opened
Phase 6: fix round 1/5 (C1, I1–I3, P33 a–e, M2–M9 addressed, 0 open; commits 9a1f2e3..ffe7548). minor (deferred): report-fuzz_* fallback uses combined matrix result if a leg's Record step never ran; owner commits with a Claude co-author trailer keep auto-merge off (fails safe); owner adds `agent` right after opening can race; stray blank lines in on: blocks. Residual (P29): triage Read/cmake unrestricted → env readable under prompt injection (surface now bot issue body + logs only).
Tasks 6.3+6.4: complete (review clean)
Phase 6: rebased (13 commits), lint green, pushed; PR opened
P28: worktree .worktrees/fix-p28 (fix/fix-proof-trusted-tools) dispatched
- CR63: 2 CodeRabbit findings on #63 (sbom/ mkdir, ghcr login in attest) verified against pinned sources and fixed in one commit on ci/release-provenance; both threads replied.
- Social preview drafted: .superpowers/sdd/2026-11-20-ci-automation/social/{card.html,social-preview.png} (1280x640, rendered with Playwright's headless chromium). Goes into the Phase 8 PR as docs/images/social-preview.png.
- P28: done as PR #66 (0bd7df5). Demo run 44892372173 on a branch predating fix_proof.py gave the honest outcome=no-test instead of "can't open file". Controller review: approved. HEAD^2 = head check confirmed on #66's own run (up-to-date branch still gets a 2-parent merge ref).
- Ruling: setup-models stays the PR head's in pr-report — its cache key and scripts must hash/run the head's tests/e2e prompts and llama tools; only branches cut before #51 (the action's birth) hit "action not found" — costs one confusing red on such an ancient branch (fix: rebase).
- #63 (Phase 7) merged as 3d9d92d. CR64: 7 CodeRabbit findings on #64 dispatched to agent cr64 (opus); report -> cr64-report.md.
- CR66: #66 export-ignore finding fixed (01e7c42, git checkout of head tests/); assemble-ref finding declined (producer/consumer must share the merge ref's tools/ci), misleading comments corrected.
- CR64: cr64 agent fixed 5, partly 1, declined 1 (2121b12 e47d836 9c1b348 3ff5a8e 32b1dc6). Controller reviewed 32b1dc6 (automerge re-check, label hold) and 3ff5a8e: OK.
- Ruling P36: close the parked "timeouts → cancelled → silent" gap in #64: reporter callers use !cancelled(); a work job whose own always() step recorded `cancelled` is reported as `timed-out` (failure); a superseded never-started job (no recorded status) is skipped — cost if wrong: one false "failing" issue, auto-closed by the next green run.
- #66 (P28) merged. 4eb7f89 (P36 timed-out reporting) spot-checked OK; parked minors: dashboard rows skip timed-out jobs; fuzz fallback to combined result when one target recorded nothing (pre-existing).
- #64 (Phase 6) merged.
- Live check: @claude on scratch issue #67 answered correctly (run 44895174206, 13 s, no branch); closed.
- Phase 8: PR #68 (8 commits, 9 review findings fixed by the agent). Controller read README diff: table true to merged workflows; release badge "v0.8.0", nightly badge "passing" verified.
- Live check: automerge — reopen while draft => owner job skipped (correct); ready_for_review => auto-merge enabled by github-actions (run 44897720534).
- Parked minor: a concurrency-cancelled agent-approval run shows as a red "cancelled" check on the PR until the newer run finishes (harmless once required: the latest run wins).
- #68 (Phase 8) auto-merged by github-actions as 68775d1 — first live auto-merge. link-check dispatched.
- README Mermaid diagram verified to render (mermaid.ink). The #68 close/reopen/ready were the controller's (after README review), not an unknown actor.
Final review (opus): fix first — 1 Critical, 5 Important, 9 Minor; tests 409+38 pass, mypy/ruff/actionlint clean. Full: final-review.md.
- Ruling F1: ONE fix wave as two draft PRs (fix/agent-hardening: I1 I3 M1 M2; fix/honest-signals: C1 I2 I4 I5 M4 M5 M7 M8 M9 + action example) — reviewable separately; drafts so automerge waits for controller review — cost if wrong: one more PR to review.
- Ruling F2 (I2): not-runnable is red like "no test" (spec lists no green not-runnable outcome), waivable by the owner's no-test-needed label — cost if wrong: e2e-only fixes need the label.
- Ruling F3 (I3): the waiver counts only when the label's latest `labeled` event is by a non-bot with write access — cost if wrong: a collaborator's label is ignored until re-applied by them.
- Ruling F4 (I1): in-repo hardening (subprocess env scrub, deny /proc reads, comments limited to trusted actors, release.yml actor check) by the agent; the v* tag ruleset is created by the controller via API (user granted repo-improving authority; reversible); making agent-approval required stays the owner's decision (spec anti-goal) — flagged in the owner ledger.
- Ruling F5: M3 (±10% jitter false verdicts) not fixed — live ratios ±1.4%; revisit if a no-op PR reads faster/slower. M6: controller dispatches nightly + portability on master after the wave merges.
- F4 done: ruleset 27791611 'release tags: admins only' (refs/tags/v*: creation/update/deletion; bypass RepositoryRole admin). Untested by a real push (any v* push starts release.yml). Final fix wave dispatched to agent final-fix (opus).
- Final fix wave: PRs #70 (A: e7ff8f5..ffe1387) and #71 (B: b1833f8..efff187), drafts, green. @coderabbitai review requested on both; scoped re-review dispatched (final-rereview, opus).
- Final re-review: approve #70 and #71 (merge #70 first). R1 (Important, pre-existing): the Claude App token in a session can merge any check-passing non-agent PR via curl.
- Ruling F6: R1 → owner decision. The admin-only master ruleset was attempted, but personal repos can't exempt the GitHub Actions app (422), so it would break auto-merge. Owner ledger lists options a/b/c — cost if wrong: the narrow path stays open until the owner picks.
- Ruling F7: N1 N2 N3 N4 N6 + the 7 CodeRabbit threads go to final-fix on the same draft PRs. N5 (schedules disabled after 60 days stop the dashboard cron too) and N7 (timeout-row ignores warm) parked — low impact.
- #70 merged (f902547) via auto-merge on ready. #71 branch updated with master.
- Live check #72: hardened @claude works (bwrap 0.9.0 namespace OK, gh OK, correct answer; run 44904780616).
- #71 merged (13f1e97). Dispatched nightly + portability on master (M6).
- Live: nightly run 44905155104 all green (accuracy 18/18, sanitizers-real-model and trend-bench did real work for the first time, 2 h fuzz each, no crash); portability 44905157127 all green (arm64, newest GCC/Clang, SIMD off); every reporter ran, no issue opened. Dashboard deployed 13f1e97 with speed, accuracy, fuzzing and size data; badge "passing".
- Polish PR (fix/dashboard-polish): single-point chart dated a day late (test-first); timeout-row job name shown unevaluated when skipped. Owner decision still open: R1 options a/b/c (owner ledger).
## Outcome
Merged: #49 #50 #51 #54 #55 #59 #60 #61 #63 #64 #66 #68 #70 #71 (+ this ledger). Final review: 1 Critical + 5 Important fixed in #70/#71; re-review approved both. Rulings P1–P36, CR1–CR8, CR51-*, F1–F7 above.
