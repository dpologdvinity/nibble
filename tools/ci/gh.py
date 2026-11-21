"""The one place CI helper code talks to GitHub: a `GitHub` protocol and `GhCli`, its `gh api` implementation.

Every workflow script in tools/ci takes a `GitHub` and never shells out or opens a socket itself, so unit tests
pass `fakes.FakeGitHub` instead (see fakes.py). `GhCli` shells out to `gh api` through a `Runner`, which tests
replace with a recording function; `subprocess_runner` is the real one. Stdlib only.

usage (from a workflow step): GhCli(os.environ["GITHUB_REPOSITORY"]) with GH_TOKEN set in the environment.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

# (argv, stdin) -> stdout. argv is the full command including the program name. Raises GhError on a non-zero exit.
Runner = Callable[[Sequence[str], str | None], str]


class GhError(RuntimeError):
    """A `gh` invocation failed; the message is its stderr."""


def subprocess_runner(argv: Sequence[str], stdin: str | None) -> str:
    """Run `argv` with `stdin` as input and return stdout; raise GhError(stderr) on a non-zero exit."""
    proc = subprocess.run(list(argv), input=stdin, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise GhError(proc.stderr.strip() or f"{argv[0]} exited with status {proc.returncode}")
    return proc.stdout


class GitHub(Protocol):
    """What the CI scripts need from GitHub. Implemented by GhCli and by fakes.FakeGitHub."""

    def find_open_issue(self, title: str) -> int | None:
        """Number of the open issue whose title equals `title` exactly (search is fuzzy, so never use it), or None."""
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

    def upsert_comment(self, number: int, marker: str, body: str) -> None:
        """Edit the comment on `number` whose body contains `marker`; create one if there is none."""
        ...

    def pr_for_sha(self, sha: str) -> int | None:
        """Number of the open pull request whose head commit is `sha`, or None."""
        ...

    def dispatch(self, workflow_file: str, ref: str, inputs: Mapping[str, str]) -> None:
        """Start `workflow_file` on `ref` through workflow_dispatch (exempt from the GITHUB_TOKEN no-trigger rule)."""
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

    def find_open_issue(self, title: str) -> int | None:
        listing = _json_list(self._get(f"repos/{self.repo}/issues?state=open&per_page=100", paginate=True))
        for issue in listing:
            # The issues endpoint also returns pull requests; they carry a "pull_request" key.
            if "pull_request" not in issue and issue.get("title") == title:
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
        path = f"repos/{self.repo}/actions/workflows/{workflow_file}/runs?branch={branch}&status=success&per_page=1"
        runs = json.loads(self._get(path)).get("workflow_runs", [])
        return str(runs[0]["head_sha"]) if runs else None

    def upsert_comment(self, number: int, marker: str, body: str) -> None:
        listing = _json_list(self._get(f"repos/{self.repo}/issues/{number}/comments?per_page=100", paginate=True))
        for existing in listing:
            if marker in str(existing.get("body", "")):
                self._send("PATCH", f"repos/{self.repo}/issues/comments/{existing['id']}", {"body": body})
                return
        self.comment(number, body)

    def pr_for_sha(self, sha: str) -> int | None:
        for pr in _json_list(self._get(f"repos/{self.repo}/commits/{sha}/pulls")):
            if pr.get("state") == "open" and pr.get("head", {}).get("sha") == sha:
                return int(pr["number"])
        return None

    def dispatch(self, workflow_file: str, ref: str, inputs: Mapping[str, str]) -> None:
        self._send(
            "POST",
            f"repos/{self.repo}/actions/workflows/{workflow_file}/dispatches",
            {"ref": ref, "inputs": dict(inputs)},
        )
