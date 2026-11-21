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

from gh import GhError, RunInfo


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
    seed `green_shas` ((workflow, branch) -> sha), `prs_by_sha` (head sha -> PR number), `completed_runs`
    ((workflow, branch) -> run ids, newest first), `job_conclusions` ((run id, job name) -> conclusion) and
    `recent_runs` ((workflow, branch) -> RunInfo, newest first; list_recent_runs filters by `created_at`),
    `artifacts` ((run id, artifact name) -> {file name: text}) and `branch_tips` (branch -> head sha).
    `vanished` ((run id, name) pairs that are listed but cannot be downloaded) and `failing_artifact_runs` (run ids
    whose artifact listing raises GhError). `fail_listing` and `fail_tip` make the run listings and branch_tip raise
    GhError.
    """

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
