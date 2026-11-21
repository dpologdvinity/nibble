"""In-memory stand-in for gh.GitHub, used by the unit tests of every tools/ci script.

FakeGitHub keeps issues and their comments in a dict, records every workflow dispatch, and answers
`last_green_sha` / `pr_for_sha` / `list_completed_runs` / `job_conclusion` / `download_artifact` from dicts a test
fills in beforehand. It touches the file system only to write a downloaded artifact into the directory it is given.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from gh import GhError


@dataclass
class FakeIssue:
    """One issue: its text, labels, comment bodies in order, and whether it is still open."""

    title: str
    body: str
    labels: list[str]
    comments: list[str] = field(default_factory=list)
    open: bool = True
    # Login of each comment's author, parallel to `comments`; a comment without an entry was written by the bot.
    comment_authors: list[str] = field(default_factory=list)
    # A pull request shares the issue number space and comment API but is never returned by find_open_issue.
    is_pull_request: bool = False


class FakeGitHub:
    """A gh.GitHub that lives in memory.

    Tests inspect `issues` (number -> FakeIssue) and `dispatches` ((workflow, ref, inputs) in call order), and
    seed `green_shas` ((workflow, branch) -> sha), `prs_by_sha` (head sha -> PR number), `completed_runs`
    ((workflow, branch) -> run ids, newest first), `job_conclusions` ((run id, job name) -> conclusion) and
    `artifacts` ((run id, artifact name) -> {file name: text}) and `branch_tips` (branch -> head sha).
    `fail_listing` and `fail_tip` make list_completed_runs and branch_tip raise GhError.
    """

    BOT = "github-actions[bot]"

    def __init__(self) -> None:
        self.issues: dict[int, FakeIssue] = {}
        self.dispatches: list[tuple[str, str, dict[str, str]]] = []
        self.green_shas: dict[tuple[str, str], str] = {}
        self.prs_by_sha: dict[str, int] = {}
        self.completed_runs: dict[tuple[str, str], list[int]] = {}
        self.job_conclusions: dict[tuple[int, str], str] = {}
        self.artifacts: dict[tuple[int, str], dict[str, str]] = {}
        self.branch_tips: dict[str, str] = {}
        self.fail_listing = False
        self.fail_tip = False
        self._next_number = 1

    def find_open_issue(self, title: str) -> int | None:
        for number, issue in self.issues.items():
            if issue.open and not issue.is_pull_request and issue.title == title:
                return number
        return None

    def create_issue(self, title: str, body: str, labels: Sequence[str]) -> int:
        number = self._next_number
        self._next_number += 1
        self.issues[number] = FakeIssue(title, body, list(labels))
        return number

    def add_pull_request(self, head_sha: str) -> int:
        """Create an open pull request whose head commit is `head_sha` and return its number."""
        number = self._next_number
        self._next_number += 1
        self.issues[number] = FakeIssue(f"pull request {number}", "", [], is_pull_request=True)
        self.prs_by_sha[head_sha] = number
        return number

    def comment_as(self, number: int, author: str, body: str) -> None:
        """Add a comment written by `author` (a login); `comment` writes as the bot."""
        issue = self.issues[number]
        # Pad first so a comment appended directly to `comments` by a test keeps its (bot) author slot.
        issue.comment_authors.extend([self.BOT] * (len(issue.comments) - len(issue.comment_authors)))
        issue.comments.append(body)
        issue.comment_authors.append(author)

    def comment(self, number: int, body: str) -> None:
        self.comment_as(number, self.BOT, body)

    def close_issue(self, number: int) -> None:
        self.issues[number].open = False

    def last_green_sha(self, workflow_file: str, branch: str) -> str | None:
        return self.green_shas.get((workflow_file, branch))

    def upsert_comment(self, number: int, marker: str, body: str, author: str | None = None) -> None:
        issue = self.issues[number]
        issue.comment_authors.extend([self.BOT] * (len(issue.comments) - len(issue.comment_authors)))
        for i, existing in enumerate(issue.comments):
            if marker in existing and (author is None or issue.comment_authors[i] == author):
                issue.comments[i] = body
                return
        self.comment_as(number, author or self.BOT, body)

    def pr_for_sha(self, sha: str) -> int | None:
        return self.prs_by_sha.get(sha)

    def dispatch(self, workflow_file: str, ref: str, inputs: Mapping[str, str]) -> None:
        self.dispatches.append((workflow_file, ref, dict(inputs)))

    def list_completed_runs(self, workflow_file: str, branch: str, limit: int) -> list[int]:
        if self.fail_listing:
            raise GhError("simulated listing failure")
        return self.completed_runs.get((workflow_file, branch), [])[:limit]

    def job_conclusion(self, run_id: int, job_name: str) -> str | None:
        return self.job_conclusions.get((run_id, job_name))

    def download_artifact(self, run_id: int, name: str, dest: Path) -> bool:
        files = self.artifacts.get((run_id, name))
        if files is None:
            return False
        for relative, text in files.items():
            target = dest / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        return True

    def branch_tip(self, branch: str) -> str | None:
        if self.fail_tip:
            raise GhError("simulated branch lookup failure")
        return self.branch_tips.get(branch)
