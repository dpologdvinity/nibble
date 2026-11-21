"""Collect the `dashboard-rows-*` artifacts of the recent nightly, portability and ci runs for dashboard.yml.

dashboard.yml runs once per finished workflow, but GitHub keeps only one *waiting* run per concurrency group, so a
completion that arrives while one dashboard run is active and another is waiting would be dropped, rows and all. To make
that harmless every dashboard run also sweeps the artifacts of the last few days (they expire after 7): rows that were
merged already are dropped by `dashboard_data.py merge`, so repeats cost nothing and nothing is ever lost.

Only trusted runs are swept: completed runs of nightly.yml, portability.yml and ci.yml on the branch whose event is
schedule, push or workflow_dispatch and whose head repository is this one (the same filter as dashboard.yml's `if:`).
That excludes pull requests, including forks' (their artifacts are attacker-controlled input). An artifact is taken only
if its name is `dashboard-rows-` plus letters, digits, `.`, `_` and `-`. Each lands in
DEST/<run id>-<artifact name>/ (the artifact holds `rows.jsonl`); the caller merges everything under DEST.

usage: dashboard_sweep.py --repo OWNER/NAME --dest DIR [--branch master] [--days 7]   (GH_TOKEN set for `gh`)
All GitHub access goes through the `GitHub` protocol of gh.py. Exit status is 0 on success (problems with single
artifacts are printed as `::warning` lines) and 1 when GitHub could not be asked at all.
"""

from __future__ import annotations

import argparse
import datetime
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from gh import GhCli, GhError, GitHub

WORKFLOW_FILES = ("nightly.yml", "portability.yml", "ci.yml")
TRUSTED_EVENTS = ("schedule", "push", "workflow_dispatch")
_ARTIFACT = re.compile(r"dashboard-rows-[A-Za-z0-9._-]+")


@dataclass
class SweepReport:
    """What a sweep did: `runs` trusted runs looked at, `skipped` untrusted runs ignored, `artifacts` downloaded, and a
    message per artifact that was not."""

    runs: int = 0
    skipped: int = 0
    artifacts: int = 0
    problems: list[str] = field(default_factory=list)


def sweep(
    gh: GitHub,
    repo: str,
    branch: str,
    since: str,
    dest: Path,
    workflows: Sequence[str] = WORKFLOW_FILES,
) -> SweepReport:
    """Download the rows artifacts of the trusted completed runs of `workflows` on `branch` created on or after the
    date `since` (`YYYY-MM-DD`) into `dest`. A listing failure raises GhError."""
    report = SweepReport()
    for workflow in workflows:
        for info in gh.list_recent_runs(workflow, branch, since):
            if info.event not in TRUSTED_EVENTS or info.head_repository != repo:
                report.skipped += 1
                continue
            report.runs += 1
            for name in gh.list_artifact_names(info.id):
                if not name.startswith("dashboard-rows-"):
                    continue
                if _ARTIFACT.fullmatch(name) is None:
                    report.problems.append(f"run {info.id}: artifact name {name!r} is not an allowed rows artifact")
                    continue
                if gh.download_artifact(info.id, name, dest / f"{info.id}-{name}"):
                    report.artifacts += 1
                else:
                    report.problems.append(f"run {info.id}: artifact {name} could not be downloaded")
    return report


def main(argv: Sequence[str], *, gh: GitHub | None = None, today: datetime.date | None = None) -> int:
    """Command line entry point (see the module docstring). `gh` and `today` default to `gh api` and the UTC date."""
    parser = argparse.ArgumentParser(description="Download the recent dashboard-rows-* artifacts.")
    parser.add_argument("--repo", required=True, help="OWNER/NAME")
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--branch", default="master")
    parser.add_argument("--days", type=int, default=7, help="how far back to look (artifacts expire after 7 days)")
    args = parser.parse_args(argv)
    now = today or datetime.datetime.now(datetime.UTC).date()
    since = (now - datetime.timedelta(days=args.days)).isoformat()
    try:
        report = sweep(gh or GhCli(args.repo), args.repo, args.branch, since, args.dest)
    except GhError as e:
        print(f"dashboard_sweep: {e}", file=sys.stderr)
        return 1
    for problem in report.problems:
        print(f"::warning::dashboard_sweep: {problem}", file=sys.stderr)
    print(
        f"swept {report.runs} run(s) since {since}: {report.artifacts} artifact(s) downloaded, "
        f"{report.skipped} untrusted run(s) ignored"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
