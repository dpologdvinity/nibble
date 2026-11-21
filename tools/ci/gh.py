"""The one place CI helper code talks to GitHub: a `GitHub` protocol and `GhCli`, its `gh api` implementation.

Every workflow script in tools/ci takes a `GitHub` and never shells out or opens a socket itself, so unit tests
pass `fakes.FakeGitHub` instead (see fakes.py). `GhCli` shells out to `gh api` through a `Runner`, which tests
replace with a recording function; `subprocess_runner` is the real one. Stdlib only.

usage (from a workflow step): GhCli(os.environ["GITHUB_REPOSITORY"]) with GH_TOKEN set in the environment.
"""

from __future__ import annotations

import json
import subprocess
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

# (argv, stdin) -> stdout. argv is the full command including the program name. Raises GhError on a non-zero exit.
Runner = Callable[[Sequence[str], str | None], str]


class GhError(RuntimeError):
    """A `gh` invocation failed; the message is its stderr."""


def subprocess_runner(argv: Sequence[str], stdin: str | None) -> str:
    """Run `argv` with `stdin` as input and return stdout; raise GhError(stderr) on a non-zero exit, and GhError too
    when the program cannot start (`gh` missing or not executable): to a caller that is a GitHub problem, not an
    OSError about its own files."""
    try:
        proc = subprocess.run(list(argv), input=stdin, capture_output=True, text=True, check=False)
    except OSError as e:
        raise GhError(f"cannot run {argv[0]}: {e}") from e
    if proc.returncode != 0:
        raise GhError(proc.stderr.strip() or f"{argv[0]} exited with status {proc.returncode}")
    return proc.stdout


@dataclass(frozen=True)
class RunInfo:
    """What a sweep needs to decide whether a workflow run is trusted: its id, the event that started it, the
    `owner/name` of the repository its head commit lives in ("" when GitHub reports none, as for a deleted fork) and its
    creation time (`YYYY-MM-DDTHH:MM:SSZ`)."""

    id: int
    event: str
    head_repository: str
    created_at: str


@dataclass(frozen=True)
class PullRequest:
    """What the agent gate needs to know about a pull request: its author's login, its label names and its head SHA
    (`pull_request` raises GhError when the response has no head commit)."""

    author: str
    labels: list[str]
    head_sha: str


@dataclass(frozen=True)
class Review:
    """One pull request review: the reviewer's login ("" for a deleted account), its state (`APPROVED`,
    `CHANGES_REQUESTED`, `COMMENTED`, `DISMISSED` or `PENDING`), the commit it was made on and the reviewer's
    `author_association` (`OWNER`, `MEMBER`, `COLLABORATOR`, `CONTRIBUTOR`, `NONE`, ...)."""

    user: str
    state: str
    commit_id: str
    association: str


@dataclass(frozen=True)
class LabelEvent:
    """One `labeled` or `unlabeled` event of an issue or pull request: which label, who did it (their login, "" for a
    deleted account) and the actor's account type (`User`, `Bot`, ...; "" when unknown)."""

    event: str
    label: str
    actor: str
    actor_type: str


# Login of the GITHUB_TOKEN identity as the REST API reports issue and comment authors.
BOT_LOGIN = "github-actions[bot]"


class GitHub(Protocol):
    """What the CI scripts need from GitHub. Implemented by GhCli and by fakes.FakeGitHub."""

    def find_open_issue(self, title: str, label: str) -> int | None:
        """Number of the open issue that github-actions[bot] opened with the label `label` and whose title equals
        `title` exactly (search is fuzzy, so never use it), or None. Anyone can open an issue with that title; only the
        workflows' own issue counts, so a stranger's copy never receives the reports nor is closed by a recovery."""
        ...

    def create_issue(self, title: str, body: str, labels: Sequence[str]) -> int:
        """Open an issue and return its number."""
        ...

    def comment(self, number: int, body: str) -> None:
        """Add a comment to issue or pull request `number`."""
        ...

    def close_issue(self, number: int) -> None:
        """Close issue `number` as completed."""
        ...

    def last_green_sha(self, workflow_file: str, branch: str) -> str | None:
        """Head commit of the latest successful run of `workflow_file` on `branch`, or None if there is none."""
        ...

    def upsert_comment(self, number: int, marker: str, body: str, author: str | None = None) -> None:
        """Edit the comment on `number` whose body contains `marker`; create one if there is none.

        With `author` (a login such as "github-actions[bot]") only a comment written by that user counts as a match,
        so an outside commenter who pastes the marker cannot make the caller edit or hide behind their comment.
        """
        ...

    def find_comment(self, number: int, marker: str, author: str) -> str | None:
        """Body of the first comment on `number` that contains `marker` and was written by `author`, or None."""
        ...

    def pr_for_sha(self, sha: str) -> int | None:
        """Number of the open pull request whose head commit is `sha`, or None."""
        ...

    def dispatch(self, workflow_file: str, ref: str, inputs: Mapping[str, str]) -> None:
        """Start `workflow_file` on `ref` through workflow_dispatch (exempt from the GITHUB_TOKEN no-trigger rule)."""
        ...

    def branch_tip(self, branch: str) -> str | None:
        """Head commit of `branch` right now, or None if the API reports no commit for it."""
        ...

    def list_completed_runs(self, workflow_file: str, branch: str, limit: int) -> list[int]:
        """Ids of the `limit` newest runs of `workflow_file` on `branch` (any event or result), newest first."""
        ...

    def list_recent_runs(self, workflow_file: str, branch: str, since: str) -> list[RunInfo]:
        """Completed runs of `workflow_file` on `branch` created on or after the date `since` (`YYYY-MM-DD`), newest
        first, whatever their event or result."""
        ...

    def list_artifact_names(self, run_id: int) -> list[str]:
        """Names of the artifacts of run `run_id` that have not expired."""
        ...

    def job_conclusion(self, run_id: int, job_name: str) -> str | None:
        """Conclusion ("success", "failure", "skipped", ...) of the job named `job_name` in run `run_id`'s latest
        attempt, or None when the run has no such job or it has not concluded."""
        ...

    def download_artifact(self, run_id: int, name: str, dest: Path) -> bool:
        """Download artifact `name` of run `run_id` into the directory `dest`; False if it does not exist or expired."""
        ...

    def pull_request(self, number: int) -> PullRequest:
        """Author, labels and current head SHA of pull request `number`."""
        ...

    def list_reviews(self, number: int) -> list[Review]:
        """Every review of pull request `number`, oldest first."""
        ...

    def label_events(self, number: int) -> list[LabelEvent]:
        """Every `labeled` and `unlabeled` event of issue or pull request `number`, oldest first."""
        ...

    def permission(self, login: str) -> str:
        """`login`'s permission on the repository: `admin`, `write` (maintainers too), `read` (triage too) or `none`.
        Raises GhError when the account does not exist."""
        ...


def _json_values(text: str) -> list[Any]:
    """Every JSON value in `text`, which may hold several back to back (`gh api --paginate` prints one per page)."""
    decoder = json.JSONDecoder()
    values: list[Any] = []
    pos = 0
    while True:
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            return values
        value, pos = decoder.raw_decode(text, pos)
        values.append(value)


def _json_list(text: str) -> list[dict[str, Any]]:
    """Flatten the JSON arrays in `text` (one or several pages) into one list of objects."""
    items: list[dict[str, Any]] = []
    for value in _json_values(text):
        if not isinstance(value, list):
            raise GhError(f"expected a JSON array from gh, got {type(value).__name__}")
        items.extend(item for item in value if isinstance(item, dict))
    return items


class GhCli:
    """GitHub implemented with `gh api` against one repository ("owner/name", normally $GITHUB_REPOSITORY)."""

    def __init__(self, repo: str, run: Runner = subprocess_runner) -> None:
        self.repo = repo
        self._run = run

    def _get(self, path: str, *, paginate: bool = False) -> str:
        """GET `path` (relative to the API root) and return the raw stdout."""
        argv = ["gh", "api"]
        if paginate:
            argv.append("--paginate")
        argv.append(path)
        return self._run(argv, None)

    def _send(self, method: str, path: str, payload: Mapping[str, Any]) -> str:
        """Send `payload` as a JSON body with `method` to `path` and return the raw stdout."""
        argv = ["gh", "api", "-X", method, path, "--input", "-"]
        return self._run(argv, json.dumps(payload))

    def find_open_issue(self, title: str, label: str) -> int | None:
        query = urllib.parse.urlencode({"state": "open", "creator": BOT_LOGIN, "labels": label, "per_page": 100})
        listing = _json_list(self._get(f"repos/{self.repo}/issues?{query}", paginate=True))
        for issue in listing:
            # The query already filters on creator and label; both are checked again so that a filter the API ignored
            # cannot match someone else's issue. The endpoint also returns pull requests ("pull_request" key).
            user = issue.get("user")
            labels = issue.get("labels")
            if (
                "pull_request" not in issue
                and issue.get("title") == title
                and isinstance(user, dict)
                and user.get("login") == BOT_LOGIN
                and isinstance(labels, list)
                and any(isinstance(item, dict) and item.get("name") == label for item in labels)
            ):
                return int(issue["number"])
        return None

    def create_issue(self, title: str, body: str, labels: Sequence[str]) -> int:
        out = self._send("POST", f"repos/{self.repo}/issues", {"title": title, "body": body, "labels": list(labels)})
        return int(json.loads(out)["number"])

    def comment(self, number: int, body: str) -> None:
        self._send("POST", f"repos/{self.repo}/issues/{number}/comments", {"body": body})

    def close_issue(self, number: int) -> None:
        self._send("PATCH", f"repos/{self.repo}/issues/{number}", {"state": "closed", "state_reason": "completed"})

    def last_green_sha(self, workflow_file: str, branch: str) -> str | None:
        # Percent-encode every dynamic piece: a branch like "release/v1#hotfix" would otherwise end the query at '#'.
        query = urllib.parse.urlencode({"branch": branch, "status": "success", "per_page": 1})
        path = f"repos/{self.repo}/actions/workflows/{urllib.parse.quote(workflow_file, safe='')}/runs?{query}"
        runs = json.loads(self._get(path)).get("workflow_runs", [])
        return str(runs[0]["head_sha"]) if runs else None

    def upsert_comment(self, number: int, marker: str, body: str, author: str | None = None) -> None:
        listing = _json_list(self._get(f"repos/{self.repo}/issues/{number}/comments?per_page=100", paginate=True))
        for existing in listing:
            # `user` is null for a deleted account; such a comment never matches an author filter.
            user = existing.get("user")
            login = user.get("login") if isinstance(user, dict) else None
            if marker in str(existing.get("body", "")) and (author is None or login == author):
                self._send("PATCH", f"repos/{self.repo}/issues/comments/{existing['id']}", {"body": body})
                return
        self.comment(number, body)

    def find_comment(self, number: int, marker: str, author: str) -> str | None:
        listing = _json_list(self._get(f"repos/{self.repo}/issues/{number}/comments?per_page=100", paginate=True))
        for existing in listing:
            user = existing.get("user")
            login = user.get("login") if isinstance(user, dict) else None
            body = str(existing.get("body", ""))
            if marker in body and login == author:
                return body
        return None

    def pr_for_sha(self, sha: str) -> int | None:
        # `commits/{sha}/pulls` also lists merged and closed PRs that once contained the commit, so list the open pulls
        # and compare head.sha instead.
        for pr in _json_list(self._get(f"repos/{self.repo}/pulls?state=open&per_page=100", paginate=True)):
            if pr.get("head", {}).get("sha") == sha:
                return int(pr["number"])
        return None

    def dispatch(self, workflow_file: str, ref: str, inputs: Mapping[str, str]) -> None:
        self._send(
            "POST",
            f"repos/{self.repo}/actions/workflows/{urllib.parse.quote(workflow_file, safe='')}/dispatches",
            {"ref": ref, "inputs": dict(inputs)},
        )

    def branch_tip(self, branch: str) -> str | None:
        # Slashes stay (a branch "release/v1" is one path in this endpoint); everything else is percent-encoded.
        commit = json.loads(self._get(f"repos/{self.repo}/branches/{urllib.parse.quote(branch, safe='/')}")).get(
            "commit"
        )
        sha = commit.get("sha") if isinstance(commit, dict) else None
        return str(sha) if sha else None

    def list_completed_runs(self, workflow_file: str, branch: str, limit: int) -> list[int]:
        query = urllib.parse.urlencode({"branch": branch, "status": "completed", "per_page": limit})
        path = f"repos/{self.repo}/actions/workflows/{urllib.parse.quote(workflow_file, safe='')}/runs?{query}"
        return [int(run["id"]) for run in json.loads(self._get(path)).get("workflow_runs", [])]

    def list_recent_runs(self, workflow_file: str, branch: str, since: str) -> list[RunInfo]:
        query = urllib.parse.urlencode(
            {"branch": branch, "status": "completed", "created": f">={since}", "per_page": 100}
        )
        path = f"repos/{self.repo}/actions/workflows/{urllib.parse.quote(workflow_file, safe='')}/runs?{query}"
        runs: list[RunInfo] = []
        for page in _json_values(self._get(path, paginate=True)):
            for run in page.get("workflow_runs", []) if isinstance(page, dict) else []:
                head = run.get("head_repository")
                repo = head.get("full_name") if isinstance(head, dict) else None
                runs.append(
                    RunInfo(int(run["id"]), str(run.get("event", "")), str(repo or ""), str(run.get("created_at", "")))
                )
        return runs

    def list_artifact_names(self, run_id: int) -> list[str]:
        names: list[str] = []
        for page in _json_values(
            self._get(f"repos/{self.repo}/actions/runs/{run_id}/artifacts?per_page=100", paginate=True)
        ):
            for artifact in page.get("artifacts", []) if isinstance(page, dict) else []:
                if not artifact.get("expired"):
                    names.append(str(artifact["name"]))
        return names

    def job_conclusion(self, run_id: int, job_name: str) -> str | None:
        # `filter=latest` keeps a re-run attempt's jobs and drops the superseded ones.
        text = self._get(f"repos/{self.repo}/actions/runs/{run_id}/jobs?filter=latest&per_page=100", paginate=True)
        for page in _json_values(text):
            for job in page.get("jobs", []) if isinstance(page, dict) else []:
                if job.get("name") == job_name:
                    conclusion = job.get("conclusion")
                    return None if conclusion is None else str(conclusion)
        return None

    def download_artifact(self, run_id: int, name: str, dest: Path) -> bool:
        listing = json.loads(self._get(f"repos/{self.repo}/actions/runs/{run_id}/artifacts?per_page=100"))
        if not any(a.get("name") == name and not a.get("expired") for a in listing.get("artifacts", [])):
            return False
        self._run(["gh", "run", "download", str(run_id), "-R", self.repo, "-n", name, "-D", str(dest)], None)
        return True

    def pull_request(self, number: int) -> PullRequest:
        data = json.loads(self._get(f"repos/{self.repo}/pulls/{number}"))
        user = data.get("user")
        labels = [str(label["name"]) for label in data.get("labels", []) if isinstance(label, dict) and "name" in label]
        head = data.get("head")
        sha = head.get("sha") if isinstance(head, dict) else None
        if not sha:
            raise GhError(f"pull request {number} has no head commit in the API response")
        return PullRequest(
            author=str(user.get("login", "")) if isinstance(user, dict) else "",
            labels=labels,
            head_sha=str(sha),
        )

    def list_reviews(self, number: int) -> list[Review]:
        reviews: list[Review] = []
        for item in _json_list(self._get(f"repos/{self.repo}/pulls/{number}/reviews?per_page=100", paginate=True)):
            user = item.get("user")
            reviews.append(
                Review(
                    user=str(user.get("login", "")) if isinstance(user, dict) else "",
                    state=str(item.get("state", "")),
                    commit_id=str(item.get("commit_id", "")),
                    association=str(item.get("author_association", "")),
                )
            )
        return reviews

    def label_events(self, number: int) -> list[LabelEvent]:
        events: list[LabelEvent] = []
        for item in _json_list(self._get(f"repos/{self.repo}/issues/{number}/events?per_page=100", paginate=True)):
            if item.get("event") not in ("labeled", "unlabeled"):
                continue
            actor = item.get("actor")
            label = item.get("label")
            events.append(
                LabelEvent(
                    event=str(item["event"]),
                    label=str(label.get("name", "")) if isinstance(label, dict) else "",
                    actor=str(actor.get("login", "")) if isinstance(actor, dict) else "",
                    actor_type=str(actor.get("type", "")) if isinstance(actor, dict) else "",
                )
            )
        return events

    def permission(self, login: str) -> str:
        data = json.loads(self._get(f"repos/{self.repo}/collaborators/{urllib.parse.quote(login, safe='')}/permission"))
        return str(data.get("permission") or "none") if isinstance(data, dict) else "none"
