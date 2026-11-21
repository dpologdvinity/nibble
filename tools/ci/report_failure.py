"""Open, update and close one GitHub issue per failing scheduled job.

Scheduled workflows (nightly, portability) call the `report-failure` composite action at the end of each job. It runs
this script with the job's outcome:

  - failure and no open issue titled "nightly: <job> failing": open one (labels `ci-failure` and the area label);
    when `--triage` is given, also start `claude-triage.yml` with `issue=<number>` (an issue created with
    GITHUB_TOKEN does not trigger other workflows, but `workflow_dispatch` is exempt);
  - failure and an open issue: add a comment, never a second issue;
  - success and an open issue: comment that the job recovered and close it;
  - success and no open issue: do nothing.

The issue is found by exact title (gh.GitHub.find_open_issue), so `fuzz_gguf` and `fuzz_load` never share an issue and
a recovery closes only its own job's issue. The failure text links the run, the commit, a compare view from the last
green run of the same workflow on master (or says none is on record) and, when given, the uploaded artifact.

usage: GH_TOKEN=... GITHUB_REPOSITORY=owner/name python tools/ci/report_failure.py --job fuzz_gguf \
           --workflow-file nightly.yml --outcome failure --run-url URL --sha SHA --area ci \
           [--attachments-url URL] [--triage]
Stdlib only; all GitHub access goes through the `GitHub` protocol in gh.py.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import gh as gh_module
from gh import GitHub

Outcome = Literal["success", "failure"]
Action = Literal["open", "comment", "close", "none"]

# Branch whose last green run anchors the compare link, and the ref the triage workflow is dispatched on.
DEFAULT_BRANCH = "master"
TRIAGE_WORKFLOW = "claude-triage.yml"


@dataclass(frozen=True)
class RunContext:
    """Everything about one job run that the issue text needs.

    `job` names the job (it is part of the issue title), `workflow_file` is e.g. "nightly.yml", `run_url` links the
    run, `sha` is the tested commit, `repo` is "owner/name", `area` is the area label and `attachments_url` links the
    uploaded artifact or is None.
    """

    job: str
    workflow_file: str
    run_url: str
    sha: str
    repo: str
    area: str
    attachments_url: str | None


def issue_title(job: str) -> str:
    """The title of the one issue for `job`; it is matched exactly, so it must stay stable."""
    return f"nightly: {job} failing"


def decide(open_issue: int | None, outcome: Outcome) -> Action:
    """What to do given the number of the job's open issue (None if there is none) and the job's outcome."""
    if outcome == "failure":
        return "open" if open_issue is None else "comment"
    return "none" if open_issue is None else "close"


def failure_body(ctx: RunContext, last_green: str | None) -> str:
    """Issue text or repeat-failure comment: run link, commit, compare link from `last_green`, attachments link."""
    lines = [
        f"The `{ctx.job}` job of `{ctx.workflow_file}` failed.",
        "",
        f"- Run: {ctx.run_url}",
        f"- Commit: `{ctx.sha}`",
    ]
    if last_green is None:
        lines.append("- Changes since the last green run: no green run on record")
    else:
        lines.append(
            f"- Changes since the last green run: https://github.com/{ctx.repo}/compare/{last_green}...{ctx.sha}"
        )
    if ctx.attachments_url:
        lines.append(f"- Attachments: {ctx.attachments_url}")
    return "\n".join(lines)


def recovery_body(ctx: RunContext) -> str:
    """The comment left when the job passes again and its issue is closed."""
    return f"The `{ctx.job}` job of `{ctx.workflow_file}` passed again at `{ctx.sha}` ({ctx.run_url}). Closing."


def report(gh: GitHub, ctx: RunContext, outcome: Outcome, triage: bool) -> Action:
    """Apply `decide` to the job's open issue and return the action taken.

    Dispatches `claude-triage.yml` with the new issue number only when an issue is opened (a comment or close needs no
    new triage) and `triage` is true.
    """
    title = issue_title(ctx.job)
    existing = gh.find_open_issue(title)
    action = decide(existing, outcome)
    if action == "open":
        body = failure_body(ctx, gh.last_green_sha(ctx.workflow_file, DEFAULT_BRANCH))
        number = gh.create_issue(title, body, ["ci-failure", ctx.area])
        if triage:
            gh.dispatch(TRIAGE_WORKFLOW, DEFAULT_BRANCH, {"issue": str(number)})
    elif action == "comment" and existing is not None:
        gh.comment(existing, failure_body(ctx, gh.last_green_sha(ctx.workflow_file, DEFAULT_BRANCH)))
    elif action == "close" and existing is not None:
        gh.comment(existing, recovery_body(ctx))
        gh.close_issue(existing)
    return action


def main(argv: Sequence[str], *, gh: GitHub | None = None, repo: str | None = None) -> int:
    """CLI entry point. `gh` and `repo` default to GhCli and $GITHUB_REPOSITORY; tests inject a fake."""
    parser = argparse.ArgumentParser(description="Open, update or close the issue for one failing scheduled job.")
    parser.add_argument("--job", required=True)
    parser.add_argument("--workflow-file", required=True)
    parser.add_argument("--outcome", required=True, choices=["success", "failure"])
    parser.add_argument("--run-url", required=True)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--area", required=True)
    parser.add_argument("--attachments-url", default="")
    parser.add_argument("--triage", action="store_true")
    args = parser.parse_args(list(argv))

    repo = repo or os.environ.get("GITHUB_REPOSITORY", "")
    if not repo:
        parser.error("GITHUB_REPOSITORY is not set")
    client: GitHub = gh if gh is not None else gh_module.GhCli(repo)
    ctx = RunContext(
        job=args.job,
        workflow_file=args.workflow_file,
        run_url=args.run_url,
        sha=args.sha,
        repo=repo,
        area=args.area,
        attachments_url=args.attachments_url or None,
    )
    outcome: Outcome = "success" if args.outcome == "success" else "failure"
    action = report(client, ctx, outcome, args.triage)
    print(f"report-failure: {ctx.job}: {action}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
