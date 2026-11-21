"""In-memory stand-in for gh.GitHub, used by the unit tests of every tools/ci script.

FakeGitHub keeps issues and their comments in a dict, records every workflow dispatch, and answers `last_green_sha`,
`pr_for_sha`, `list_completed_runs`, `list_recent_runs`, `list_artifact_names`, `job_conclusion` and `download_artifact`
from dicts a test fills in beforehand. It touches the file system only to write a downloaded artifact into the
directory it is given.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from gh import GhError, PullRequest, Review, RunInfo


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
    # Login of whoever opened it; find_open_issue only returns the bot's own issues.
    author: str = "github-actions[bot]"


class FakeGitHub:
    """A gh.GitHub that lives in memory.

    Tests inspect `issues` (number -> FakeIssue) and `dispatches` ((workflow, ref, inputs) in call order), and
    seed `green_shas` ((workflow, branch) -> sha), `prs_by_sha` (head sha -> PR number), `completed_runs`
    ((workflow, branch) -> run ids, newest first), `job_conclusions` ((run id, job name) -> conclusion) and
    `recent_runs` ((workflow, branch) -> RunInfo, newest first; list_recent_runs filters by `created_at`),
    `artifacts` ((run id, artifact name) -> {file name: text}) and `branch_tips` (branch -> head sha).
    `pull_requests` (number -> PullRequest, filled by add_pull_request) and `reviews` (number -> Review list).
    `vanished` ((run id, name) pairs that are listed but cannot be downloaded) and `failing_artifact_runs` (run ids
    whose artifact listing raises GhError). `fail_listing`, `fail_tip` and `fail_reviews` make the run listings,
    branch_tip and list_reviews raise GhError.
    """

    BOT = "github-actions[bot]"

    def __init__(self) -> None:
        self.issues: dict[int, FakeIssue] = {}
        self.dispatches: list[tuple[str, str, dict[str, str]]] = []
        self.green_shas: dict[tuple[str, str], str] = {}
        self.prs_by_sha: dict[str, int] = {}
        self.completed_runs: dict[tuple[str, str], list[int]] = {}
        self.recent_runs: dict[tuple[str, str], list[RunInfo]] = {}
        self.job_conclusions: dict[tuple[int, str], str] = {}
        self.artifacts: dict[tuple[int, str], dict[str, str]] = {}
        self.branch_tips: dict[str, str] = {}
        self.vanished: set[tuple[int, str]] = set()
        self.failing_artifact_runs: set[int] = set()
        self.fail_listing = False
        self.fail_tip = False
        self.fail_reviews = False
        self.pull_requests: dict[int, PullRequest] = {}
        self.reviews: dict[int, list[Review]] = {}
        self._next_number = 1

    def find_open_issue(self, title: str, label: str) -> int | None:
        for number, issue in self.issues.items():
            if (
                issue.open
                and not issue.is_pull_request
                and issue.title == title
                and issue.author == self.BOT
                and label in issue.labels
            ):
                return number
        return None

    def create_issue(self, title: str, body: str, labels: Sequence[str]) -> int:
        return self.open_issue_as(self.BOT, title, labels, body)

    def open_issue_as(self, author: str, title: str, labels: Sequence[str], body: str = "") -> int:
        """Open an issue written by `author` (a login) and return its number; `create_issue` opens as the bot."""
        number = self._next_number
        self._next_number += 1
        self.issues[number] = FakeIssue(title, body, list(labels), author=author)
        return number

    def add_pull_request(self, head_sha: str, *, author: str = "someone", labels: Sequence[str] = ()) -> int:
        """Create an open pull request at `head_sha` by `author` with `labels` and return its number."""
        number = self._next_number
        self._next_number += 1
        self.issues[number] = FakeIssue(f"pull request {number}", "", list(labels), is_pull_request=True)
        self.prs_by_sha[head_sha] = number
        self.pull_requests[number] = PullRequest(author, list(labels), head_sha)
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

    def list_recent_runs(self, workflow_file: str, branch: str, since: str) -> list[RunInfo]:
        if self.fail_listing:
            raise GhError("simulated listing failure")
        return [r for r in self.recent_runs.get((workflow_file, branch), []) if r.created_at[:10] >= since]

    def list_artifact_names(self, run_id: int) -> list[str]:
        if run_id in self.failing_artifact_runs:
            raise GhError(f"simulated artifact listing failure for run {run_id}")
        return [name for (run, name) in self.artifacts if run == run_id]

    def job_conclusion(self, run_id: int, job_name: str) -> str | None:
        return self.job_conclusions.get((run_id, job_name))

    def download_artifact(self, run_id: int, name: str, dest: Path) -> bool:
        files = self.artifacts.get((run_id, name))
        if files is None or (run_id, name) in self.vanished:
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

    def pull_request(self, number: int) -> PullRequest:
        return self.pull_requests[number]

    def list_reviews(self, number: int) -> list[Review]:
        if self.fail_reviews:
            raise GhError("simulated review listing failure")
        return list(self.reviews.get(number, []))
