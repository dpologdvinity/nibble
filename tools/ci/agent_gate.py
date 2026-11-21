"""The `agent-approval` check: hold agent pull requests until a trusted person has approved the current commit.

Branch protection on `master` asks for no approving review, so nothing else stops a pull request opened by the Claude
GitHub App (`claude[bot]`) from being merged without the owner. Making the `agent-approval` check required fixes that.
For a pull request authored by `claude[bot]` this script fails until an APPROVED review stands from an OWNER, MEMBER or
COLLABORATOR on the pull request's current head commit; for every other pull request it passes at once. That includes
an owner's pull request that carries the `agent` label or that Claude pushed to: the owner cannot approve their own
pull request, so gating it would make it unmergeable, and automerge.yml turns auto-merge off for such a pull request
instead (the owner merges it by hand after reading it).

Which reviews stand: for each reviewer only the latest APPROVED, CHANGES_REQUESTED or DISMISSED review counts (a plain
comment changes nothing), it must be an approval of the head commit, and the reviewer must not be the pull request's
author or a bot. A new push therefore withdraws every approval, as it must.

usage: GH_TOKEN=... GITHUB_REPOSITORY=owner/name python tools/ci/agent_gate.py --pr NUMBER
Exit status 0 = pass, 1 = fail; a GitHub API error raises, so the check fails rather than passing blind.
Stdlib only; all GitHub access goes through the `GitHub` protocol in gh.py.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass

import gh as gh_module
from gh import GitHub, Review

# Login of the Claude GitHub App as the REST API reports pull request authors.
AGENT_BOT = "claude[bot]"
# `author_association` values whose approval counts.
TRUSTED_ASSOCIATIONS = frozenset({"OWNER", "MEMBER", "COLLABORATOR"})
# Review states that change a reviewer's standing; COMMENTED and PENDING do not.
_DECIDING_STATES = frozenset({"APPROVED", "CHANGES_REQUESTED", "DISMISSED"})


@dataclass(frozen=True)
class Verdict:
    """The check result: `ok` is whether it passes, `reason` the one-line explanation shown in the log and summary."""

    ok: bool
    reason: str


def standing_approvals(reviews: Sequence[Review], head_sha: str, author: str) -> list[str]:
    """Logins whose latest deciding review is an approval of `head_sha` by a trusted, human, non-author reviewer."""
    latest: dict[str, Review] = {}
    for review in reviews:  # oldest first, so a later review replaces an earlier one
        if review.user and review.state in _DECIDING_STATES:
            latest[review.user.lower()] = review
    return [
        review.user
        for login, review in latest.items()
        if review.state == "APPROVED"
        and review.commit_id == head_sha
        and review.association in TRUSTED_ASSOCIATIONS
        and not review.user.endswith("[bot]")
        and login != author.lower()
    ]


def evaluate(gh: GitHub, number: int) -> Verdict:
    """Decide the `agent-approval` check for pull request `number` (reads the pull request and its reviews)."""
    pr = gh.pull_request(number)
    if pr.author.lower() != AGENT_BOT:
        return Verdict(True, "not a pull request by the Claude app")
    approvers = standing_approvals(gh.list_reviews(number), pr.head_sha, pr.author)
    if approvers:
        return Verdict(True, f"approved on the current head by {', '.join(sorted(approvers))}")
    return Verdict(
        False,
        "pull request by the Claude app: waiting for an approving review by the owner, a member or a collaborator on "
        "the current head commit (a new push withdraws earlier approvals)",
    )


def main(argv: Sequence[str], *, gh: GitHub | None = None, repo: str | None = None) -> int:
    """CLI entry point. `gh` and `repo` default to GhCli and $GITHUB_REPOSITORY; tests inject a fake."""
    parser = argparse.ArgumentParser(description="Hold agent pull requests until the owner approves.")
    parser.add_argument("--pr", required=True, type=int, help="pull request number")
    args = parser.parse_args(list(argv))
    repo = repo or os.environ.get("GITHUB_REPOSITORY", "")
    if not repo:
        parser.error("GITHUB_REPOSITORY is not set")
    client: GitHub = gh if gh is not None else gh_module.GhCli(repo)
    verdict = evaluate(client, args.pr)
    print(f"agent-approval: {'pass' if verdict.ok else 'fail'}: {verdict.reason}")
    return 0 if verdict.ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
