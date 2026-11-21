"""The nightly accuracy job's gates and its comparison with the previous night, as tested code.

Two subcommands replace what used to be inline Python in .github/workflows/nightly.yml:

  snapshot LOG --out e2e-cosines.json --summary SUMMARY.md --expected 18
      Parses the min-cosine lines of the `ctest -V` log (cosine.parse_min_cosine), writes them as a JSON snapshot
      (`--out`, always, so the artifact shows what happened) and fails (exit 1, `::error::` lines, `- FAILED:` bullets
      in the summary) when fewer than `--expected` results parsed (models or goldens went missing) or any min cosine
      is NaN or infinite. diff_cosines counts NaN == NaN as unchanged, so without the second gate a broken kernel
      would never show up as a change.
      `--lenient` (the PR report) turns those failures into warnings and exits 0, so the snapshot is always written
      and the report is built; the workflow then re-runs the strict form on the head to fail the job.

  compare CURRENT.json --summary SUMMARY.md --dest DIR [--workflow-file nightly.yml --branch master
                                                         --job accuracy --artifact e2e-cosines]
      Finds the baseline (find_baseline), diffs it against CURRENT and writes a table to the summary with one
      `::warning::` per change. It always exits 0: no baseline is a first night, and a GitHub or artifact problem only
      costs the comparison.

The baseline is the newest completed run of the workflow on the branch, whatever started it, whose `accuracy` JOB
concluded success and which still has the snapshot artifact. Judging by the job and not by the whole run matters: a
night where only `fuzz_load` failed still has a valid accuracy artifact, and a hand-run of a different job leaves none.

The snapshot is JSON of {"<model>/<prompt>": {model, prompt, min_cos, positions, agree}}; a non-finite `min_cos` is
stored as the string "nan", "inf" or "-inf" because JSON has no such numbers.

usage (workflow step):
    python tools/ci/accuracy.py snapshot e2e.log --out e2e-cosines.json --summary "$GITHUB_STEP_SUMMARY" --expected 18
Needs GH_TOKEN and GITHUB_REPOSITORY for `compare`. Stdlib only; GitHub access goes through the `GitHub` protocol
(gh.py).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import gh as gh_module
from cosine import Cosine, CosineChange, diff_cosines, parse_min_cosine
from gh import GhError, GitHub

# How many of the newest completed runs find_baseline inspects before giving up.
BASELINE_LOOKBACK = 10


@dataclass(frozen=True)
class Snapshot:
    """The parsed results (keyed "<model>/<prompt>") and the gate failures found in them (empty when all is well)."""

    runs: dict[str, Cosine]
    problems: list[str]


class Baseline(NamedTuple):
    """The previous night's snapshot file, downloaded from run `run_id`."""

    run_id: int
    path: Path


@dataclass(frozen=True)
class Comparison:
    """The summary text for the comparison and the `::warning::` workflow commands to print."""

    markdown: str
    warnings: list[str]


def build_snapshot(log: str, expected: int) -> Snapshot:
    """Parse `log` and apply the two gates: at least `expected` results, and every min cosine finite.

    ValueError (from cosine.parse_min_cosine) means the log contradicts itself; that is not a gate failure but a
    broken input, and the caller reports it as such.
    """
    runs = parse_min_cosine(log)
    problems: list[str] = []
    if len(runs) < expected:
        problems.append(f"only {len(runs)} of the expected {expected} min-cosine results parsed (models missing?)")
    bad = sorted(k for k, c in runs.items() if not math.isfinite(c.min_cos))
    if bad:
        problems.append("non-finite min cosine: " + ", ".join(bad))
    return Snapshot(runs, problems)


def snapshot_json(runs: Mapping[str, Cosine]) -> str:
    """Strict JSON text of `runs`, sorted by key; a non-finite min cosine becomes the string "nan"/"inf"/"-inf"."""
    data = {
        key: {
            "model": c.model,
            "prompt": c.prompt,
            "min_cos": c.min_cos if math.isfinite(c.min_cos) else repr(c.min_cos),
            "positions": c.positions,
            "agree": c.agree,
        }
        for key, c in sorted(runs.items())
    }
    return json.dumps(data, indent=1, allow_nan=False) + "\n"


def load_snapshot(text: str) -> dict[str, Cosine]:
    """Inverse of snapshot_json. Any malformation (not JSON, wrong shape, missing field, bad number) is a ValueError."""
    try:
        raw = json.loads(text)
        if not isinstance(raw, dict):
            raise ValueError("the snapshot is not a JSON object")
        return {
            str(k): Cosine(str(v["model"]), str(v["prompt"]), float(v["min_cos"]), int(v["positions"]), int(v["agree"]))
            for k, v in raw.items()
        }
    except (KeyError, TypeError) as e:
        raise ValueError(f"malformed snapshot entry: {e!r}") from e


def find_baseline(
    gh: GitHub,
    *,
    workflow_file: str,
    branch: str,
    job: str,
    artifact: str,
    filename: str,
    dest: Path,
    limit: int = BASELINE_LOOKBACK,
) -> Baseline | None:
    """The newest of the last `limit` completed runs whose `job` succeeded and that has `artifact` holding `filename`.

    The artifact is downloaded into `dest/previous-<run id>`. Returns None when no such run exists. A GhError from
    listing or downloading propagates: the caller cannot tell "first night" from "could not look".
    """
    for run_id in gh.list_completed_runs(workflow_file, branch, limit):
        if gh.job_conclusion(run_id, job) != "success":
            continue
        target = dest / f"previous-{run_id}"
        if gh.download_artifact(run_id, artifact, target) and (target / filename).is_file():
            return Baseline(run_id, target / filename)
    return None


def _fmt(value: float | None) -> str:
    """A min cosine for the table: six significant digits, or "absent" for a key missing from that night."""
    return "absent" if value is None else f"{value:.6g}"


def _oneline(text: str) -> str:
    """`text` with newlines folded to spaces, so it fits in one workflow command."""
    return " ".join(text.split())


def _render_changes(baseline_run: int, changes: Sequence[CosineChange]) -> Comparison:
    """The summary text and warnings for a successful comparison with run `baseline_run`."""
    head = f"### e2e accuracy versus the previous night\n\nPrevious run: {baseline_run}.\n\n"
    if not changes:
        return Comparison(head + "No min-cosine value changed.\n", [])
    lines = ["| model/prompt | previous | now |", "|---|---|---|"]
    warnings = []
    for c in changes:
        lines.append(f"| {c.key} | {_fmt(c.old)} | {_fmt(c.new)} |")
        warnings.append(f"::warning title=min cosine changed::{c.key}: {_fmt(c.old)} -> {_fmt(c.new)}")
    return Comparison(head + "\n".join(lines) + "\n", warnings)


def _unavailable(summary_line: str, warning: str) -> Comparison:
    """A comparison that could not be made: one summary line and one warning (neither fails the job)."""
    return Comparison(
        f"### e2e accuracy versus the previous night\n\n{summary_line}\n",
        [f"::warning title=nightly::{_oneline(warning)}"],
    )


def compare_with_baseline(
    gh: GitHub,
    current: Path,
    *,
    workflow_file: str,
    branch: str,
    job: str,
    artifact: str,
    dest: Path,
) -> Comparison:
    """Compare the snapshot at `current` with the previous night's; only an unusable `current` raises."""
    now = load_snapshot(current.read_text(encoding="utf-8"))
    try:
        baseline = find_baseline(
            gh,
            workflow_file=workflow_file,
            branch=branch,
            job=job,
            artifact=artifact,
            filename=current.name,
            dest=dest,
        )
    except GhError as e:
        return _unavailable(
            "Comparison unavailable: could not list or download the previous nightly runs.",
            f"could not look up the previous nightly run: {e}",
        )
    if baseline is None:
        return Comparison(
            "### e2e accuracy versus the previous night\n\n"
            f"First night: none of the last {BASELINE_LOOKBACK} runs on {branch} has a successful {job} job "
            "with an e2e-cosines artifact.\n",
            [],
        )
    try:
        before = load_snapshot(baseline.path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return _unavailable(
            "Comparison unavailable: the previous artifact could not be read.",
            f"cannot compare with run {baseline.run_id}: {e}",
        )
    return _render_changes(baseline.run_id, diff_cosines(before, now))


def _snapshot_command(args: argparse.Namespace) -> int:
    """`snapshot`: write the JSON file and the summary, print `::error::` lines, return 1 on a gate failure.

    With `--lenient` the gate failures are `::warning::` lines and `- WARNING:` bullets and the exit status is 0; only
    a log that cannot be read (nothing to write) still returns 1.
    """
    try:
        snap = build_snapshot(args.log.read_text(encoding="utf-8", errors="replace"), args.expected)
    except (OSError, ValueError) as e:
        print("::error title=e2e log::" + _oneline(str(e)))
        return 1
    args.out.write_text(snapshot_json(snap.runs), encoding="utf-8")
    label, command = ("WARNING", "warning") if args.lenient else ("FAILED", "error")
    with args.summary.open("a", encoding="utf-8") as out:
        out.write(f"### e2e accuracy\n\n{len(snap.runs)} min-cosine results parsed.\n\n")
        for problem in snap.problems:
            out.write(f"- {label}: {problem}\n")
    for problem in snap.problems:
        print(f"::{command} title=e2e accuracy::" + problem)
    return 1 if snap.problems and not args.lenient else 0


def _compare_command(args: argparse.Namespace, gh: GitHub) -> int:
    """`compare`: append the comparison to the summary and print its warnings. Returns 0 unless CURRENT is unusable."""
    try:
        result = compare_with_baseline(
            gh,
            args.current,
            workflow_file=args.workflow_file,
            branch=args.branch,
            job=args.job,
            artifact=args.artifact,
            dest=args.dest,
        )
    except (OSError, ValueError) as e:
        print("::error title=e2e accuracy::cannot read the current snapshot: " + _oneline(str(e)))
        return 1
    with args.summary.open("a", encoding="utf-8") as out:
        out.write(result.markdown)
    for warning in result.warnings:
        print(warning)
    return 0


def main(argv: Sequence[str], *, gh: GitHub | None = None, repo: str | None = None) -> int:
    """CLI entry point. `gh` and `repo` default to GhCli and $GITHUB_REPOSITORY; tests inject a fake."""
    parser = argparse.ArgumentParser(
        description="Gate the nightly e2e accuracy results and compare them with last night."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    snap = sub.add_parser("snapshot", help="parse the ctest log into the JSON snapshot and apply the gates")
    snap.add_argument("log", type=Path)
    snap.add_argument("--out", type=Path, required=True)
    snap.add_argument("--summary", type=Path, required=True)
    snap.add_argument("--expected", type=int, required=True)
    snap.add_argument(
        "--lenient",
        action="store_true",
        help="report gate failures as warnings and exit 0 (an unreadable log still fails), for the PR report",
    )
    cmp_ = sub.add_parser("compare", help="diff the snapshot against the previous night's")
    cmp_.add_argument("current", type=Path)
    cmp_.add_argument("--summary", type=Path, required=True)
    cmp_.add_argument("--dest", type=Path, required=True)
    cmp_.add_argument("--workflow-file", default="nightly.yml")
    cmp_.add_argument("--branch", default="master")
    cmp_.add_argument("--job", default="accuracy")
    cmp_.add_argument("--artifact", default="e2e-cosines")
    args = parser.parse_args(list(argv))

    if args.command == "snapshot":
        return _snapshot_command(args)
    repo = repo or os.environ.get("GITHUB_REPOSITORY", "")
    if gh is None and not repo:
        parser.error("GITHUB_REPOSITORY is not set")
    return _compare_command(args, gh if gh is not None else gh_module.GhCli(repo))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
