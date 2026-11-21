"""Does the `no-test-needed` label waive fix-proof for this pull request? Only when a trusted person applied it.

fix-proof (`.github/workflows/fix-proof.yml`, verdict in fix_proof.py) is red for a fix pull request that changes no
test case, and the `no-test-needed` label turns that red into green. A label is a weak signal: the Claude sessions can
open a pull request with any label (`gh pr create --label ...`), and anyone with triage access can add one. So the
label counts only when

  1. the pull request carries it now, and
  2. its latest `labeled` event (the issue events API, read-only token) was made by a person, not a bot (account type
     `Bot` or a login ending in `[bot]`), who has write, maintain or admin access to the repository
     (`collaborators/{login}/permission`).

Anything else is no waiver, and so is a failed API call (the check stays red; re-run it). The events list can lag a
moment behind the label: a label applied just before the run may find no event yet and needs a re-run.

usage: GH_TOKEN=... GITHUB_REPOSITORY=owner/name python tools/ci/fix_proof_waiver.py --pr NUMBER \\
           [--label no-test-needed] [--summary "$GITHUB_STEP_SUMMARY"]
Prints `waived=true|false` (for `>> "$GITHUB_OUTPUT"`), the reason on stderr and, with --summary, appends the reason
there. Exits 0 whatever the answer; 2 only for bad arguments.
Stdlib only; all GitHub access goes through the `GitHub` protocol in gh.py.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass

import gh as gh_module
from gh import GhError, GitHub

# The label that waives a "no test" verdict (and its spelling in tools/ci/labels.py).
WAIVER_LABEL = "no-test-needed"
# Repository permissions whose holder may waive. The API reports maintain as write; listed in case that changes.
WAIVING_PERMISSIONS = frozenset({"admin", "maintain", "write"})


@dataclass(frozen=True)
class Waiver:
    """The answer: whether the label waives fix-proof, and a one-line reason for the log and the job summary."""

    waived: bool
    reason: str


def _is_bot(login: str, actor_type: str) -> bool:
    """Whether an event actor is an app or bot account rather than a person."""
    return actor_type.lower() == "bot" or login.lower().endswith("[bot]")


def evaluate(gh: GitHub, number: int, label: str = WAIVER_LABEL) -> Waiver:
    """Whether `label` on pull request `number` waives fix-proof (see the module docstring for the rules).

    Never raises for a GitHub API error: the error is the reason of a `waived=False` answer.
    """
    try:
        if label not in gh.pull_request(number).labels:
            return Waiver(False, f"the pull request does not carry `{label}`")
        applications = [e for e in gh.label_events(number) if e.event == "labeled" and e.label == label]
        if not applications:
            return Waiver(False, f"no `labeled` event for `{label}` is on record yet (re-run the check)")
        latest = applications[-1]
        who = latest.actor or "a deleted account"
        if not latest.actor or _is_bot(latest.actor, latest.actor_type):
            return Waiver(False, f"`{label}` was last applied by {who}, a bot; only a person with write access waives")
        permission = gh.permission(latest.actor)
    except GhError as error:
        return Waiver(False, f"could not check who applied `{label}`: {error}")
    if permission not in WAIVING_PERMISSIONS:
        return Waiver(False, f"`{label}` was last applied by {who}, who has `{permission}` access, not write access")
    return Waiver(True, f"`{label}` was applied by {who} ({permission} access)")


def main(argv: Sequence[str], *, gh: GitHub | None = None, repo: str | None = None) -> int:
    """CLI entry point. `gh` and `repo` default to GhCli and $GITHUB_REPOSITORY; tests inject a fake."""
    parser = argparse.ArgumentParser(description="Does the no-test-needed label waive fix-proof?")
    parser.add_argument("--pr", required=True, type=int, help="pull request number")
    parser.add_argument("--label", default=WAIVER_LABEL, help=f"the waiving label (default {WAIVER_LABEL})")
    parser.add_argument("--summary", help="Markdown file to append the reason to (the job summary)")
    args = parser.parse_args(list(argv))
    repo = repo or os.environ.get("GITHUB_REPOSITORY", "")
    if not repo:
        parser.error("GITHUB_REPOSITORY is not set")
    client: GitHub = gh if gh is not None else gh_module.GhCli(repo)
    answer = evaluate(client, args.pr, args.label)
    # One line: the reason may hold API error text, which must not add lines to the job summary's Markdown.
    reason = " ".join(answer.reason.split())
    print(f"waived={'true' if answer.waived else 'false'}")
    print(f"fix-proof waiver: {'yes' if answer.waived else 'no'}: {reason}", file=sys.stderr)
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as out:
            out.write(f"Waiver label: {'counts' if answer.waived else 'does not count'}: {reason}\n\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
