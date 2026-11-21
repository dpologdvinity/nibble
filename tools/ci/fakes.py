"""In-memory stand-in for gh.GitHub, used by the unit tests of every tools/ci script.

FakeGitHub keeps issues and their comments in a dict, records every workflow dispatch, and answers
`last_green_sha` / `pr_for_sha` from dicts a test fills in beforehand. It does no I/O.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


@dataclass
class FakeIssue:
    """One issue: its text, labels, comment bodies in order, and whether it is still open."""

    title: str
    body: str
    labels: list[str]
    comments: list[str] = field(default_factory=list)
    open: bool = True


class FakeGitHub:
    """A gh.GitHub that lives in memory.

    Tests inspect `issues` (number -> FakeIssue) and `dispatches` ((workflow, ref, inputs) in call order), and
    seed `green_shas` ((workflow, branch) -> sha) and `prs_by_sha` (head sha -> PR number).
    """

    def __init__(self) -> None:
        self.issues: dict[int, FakeIssue] = {}
        self.dispatches: list[tuple[str, str, dict[str, str]]] = []
        self.green_shas: dict[tuple[str, str], str] = {}
        self.prs_by_sha: dict[str, int] = {}
        self._next_number = 1

    def find_open_issue(self, title: str) -> int | None:
        for number, issue in self.issues.items():
            if issue.open and issue.title == title:
                return number
        return None

    def create_issue(self, title: str, body: str, labels: Sequence[str]) -> int:
        number = self._next_number
        self._next_number += 1
        self.issues[number] = FakeIssue(title, body, list(labels))
        return number

    def comment(self, number: int, body: str) -> None:
        self.issues[number].comments.append(body)

    def close_issue(self, number: int) -> None:
        self.issues[number].open = False

    def last_green_sha(self, workflow_file: str, branch: str) -> str | None:
        return self.green_shas.get((workflow_file, branch))

    def upsert_comment(self, number: int, marker: str, body: str) -> None:
        comments = self.issues[number].comments
        for i, existing in enumerate(comments):
            if marker in existing:
                comments[i] = body
                return
        comments.append(body)

    def pr_for_sha(self, sha: str) -> int | None:
        return self.prs_by_sha.get(sha)

    def dispatch(self, workflow_file: str, ref: str, inputs: Mapping[str, str]) -> None:
        self.dispatches.append((workflow_file, ref, dict(inputs)))
