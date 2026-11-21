"""Unit tests for tools/ci/gh.py (GhCli) and tools/ci/fakes.py (FakeGitHub).

Run with: python -m unittest discover -s tools/ci -p 'test_*.py'
GhCli is exercised through a scripted Runner that records every call, so no test touches the network or `gh`.
"""

from __future__ import annotations

import json
import unittest
from collections.abc import Sequence

import gh
from fakes import FakeGitHub

REPO = "owner/nibble"


class ScriptedRunner:
    """A gh.Runner that records (argv, stdin) calls and answers from `responses`.

    `responses` maps a substring of the joined argv to the stdout to return; the first match wins, and an
    unmatched call returns "[]".
    """

    def __init__(self, responses: dict[str, str] | None = None) -> None:
        self.responses = responses or {}
        self.calls: list[tuple[list[str], str | None]] = []

    def __call__(self, argv: Sequence[str], stdin: str | None) -> str:
        self.calls.append((list(argv), stdin))
        joined = " ".join(argv)
        for needle, out in self.responses.items():
            if needle in joined:
                return out
        return "[]"


class FindOpenIssueTests(unittest.TestCase):
    ISSUES = json.dumps(
        [
            {"number": 3, "title": "nightly: fuzz_load failing"},
            {"number": 4, "title": "nightly: fuzz_gguf failing"},
        ]
    )

    def test_find_open_issue_exact_title(self) -> None:
        client = gh.GhCli(REPO, ScriptedRunner({"issues": self.ISSUES}))
        self.assertEqual(client.find_open_issue("nightly: fuzz_gguf failing"), 4)
        self.assertIsNone(client.find_open_issue("nightly: fuzz failing"))

    def test_find_open_issue_paginates(self) -> None:
        runner = ScriptedRunner({"issues": self.ISSUES})
        gh.GhCli(REPO, runner).find_open_issue("x")
        argv, _ = runner.calls[0]
        self.assertIn("--paginate", argv)
        self.assertIn(f"repos/{REPO}/issues?state=open&per_page=100", argv)

    def test_find_open_issue_reads_concatenated_pages(self) -> None:
        # `gh api --paginate` prints one JSON array per page back to back, not one merged array.
        pages = json.dumps([{"number": 1, "title": "a"}]) + json.dumps([{"number": 2, "title": "b"}])
        client = gh.GhCli(REPO, ScriptedRunner({"issues": pages}))
        self.assertEqual(client.find_open_issue("b"), 2)

    def test_find_open_issue_skips_pull_requests(self) -> None:
        # The issues endpoint also lists pull requests; they carry a "pull_request" key.
        listing = json.dumps([{"number": 9, "title": "t", "pull_request": {}}, {"number": 10, "title": "t"}])
        client = gh.GhCli(REPO, ScriptedRunner({"issues": listing}))
        self.assertEqual(client.find_open_issue("t"), 10)


class UpsertCommentTests(unittest.TestCase):
    def test_upsert_comment_edits_existing(self) -> None:
        comments = json.dumps([{"id": 11, "body": "unrelated"}, {"id": 12, "body": "old <!-- m --> text"}])
        runner = ScriptedRunner({"issues/7/comments": comments})
        gh.GhCli(REPO, runner).upsert_comment(7, "<!-- m -->", "new <!-- m --> text")
        writes = [(argv, stdin) for argv, stdin in runner.calls if "-X" in argv]
        self.assertEqual(len(writes), 1)
        argv, stdin = writes[0]
        self.assertEqual(argv[argv.index("-X") + 1], "PATCH")
        self.assertIn(f"repos/{REPO}/issues/comments/12", argv)
        self.assertEqual(json.loads(stdin or ""), {"body": "new <!-- m --> text"})

    def test_upsert_comment_creates_when_absent(self) -> None:
        runner = ScriptedRunner({"issues/7/comments": json.dumps([{"id": 11, "body": "unrelated"}])})
        gh.GhCli(REPO, runner).upsert_comment(7, "<!-- m -->", "body <!-- m -->")
        writes = [argv for argv, _ in runner.calls if "-X" in argv]
        self.assertEqual(len(writes), 1)
        self.assertEqual(writes[0][writes[0].index("-X") + 1], "POST")
        self.assertIn(f"repos/{REPO}/issues/7/comments", writes[0])


class OtherCallTests(unittest.TestCase):
    def test_create_issue_posts_json_and_returns_number(self) -> None:
        runner = ScriptedRunner({"-X POST": '{"number": 21}'})
        number = gh.GhCli(REPO, runner).create_issue("t", "b", ["ci-failure"])
        self.assertEqual(number, 21)
        argv, stdin = runner.calls[0]
        self.assertIn(f"repos/{REPO}/issues", argv)
        self.assertEqual(json.loads(stdin or ""), {"title": "t", "body": "b", "labels": ["ci-failure"]})

    def test_close_issue_patches_state(self) -> None:
        runner = ScriptedRunner()
        gh.GhCli(REPO, runner).close_issue(5)
        argv, stdin = runner.calls[0]
        self.assertEqual(argv[argv.index("-X") + 1], "PATCH")
        self.assertIn(f"repos/{REPO}/issues/5", argv)
        self.assertEqual(json.loads(stdin or "")["state"], "closed")

    def test_last_green_sha(self) -> None:
        runs = json.dumps({"workflow_runs": [{"head_sha": "abc123"}]})
        runner = ScriptedRunner({"runs": runs})
        self.assertEqual(gh.GhCli(REPO, runner).last_green_sha("nightly.yml", "master"), "abc123")
        joined = " ".join(runner.calls[0][0])
        self.assertIn("actions/workflows/nightly.yml/runs", joined)
        self.assertIn("branch=master", joined)
        self.assertIn("status=success", joined)

    def test_last_green_sha_none_when_no_runs(self) -> None:
        runner = ScriptedRunner({"runs": json.dumps({"workflow_runs": []})})
        self.assertIsNone(gh.GhCli(REPO, runner).last_green_sha("nightly.yml", "master"))

    def test_last_green_sha_encodes_branch_and_workflow_path(self) -> None:
        runner = ScriptedRunner({"runs": json.dumps({"workflow_runs": [{"head_sha": "abc123"}]})})
        gh.GhCli(REPO, runner).last_green_sha("my flow.yml", "release/v1#hotfix")
        path = runner.calls[0][0][-1]
        # The '#' must not start a URL fragment that cuts off the status filter.
        self.assertEqual(
            path,
            f"repos/{REPO}/actions/workflows/my%20flow.yml/runs?branch=release%2Fv1%23hotfix&status=success&per_page=1",
        )

    def test_pr_for_sha_lists_open_pulls_and_matches_head(self) -> None:
        page1 = json.dumps([{"number": 2, "head": {"sha": "other"}}])
        page2 = json.dumps([{"number": 3, "head": {"sha": "abc"}}])
        runner = ScriptedRunner({"pulls?state=open&per_page=100": page1 + "\n" + page2})
        client = gh.GhCli(REPO, runner)
        self.assertEqual(client.pr_for_sha("abc"), 3)
        self.assertIsNone(client.pr_for_sha("zzz"))
        argv = runner.calls[0][0]
        self.assertIn("--paginate", argv)
        self.assertEqual(argv[-1], f"repos/{REPO}/pulls?state=open&per_page=100")

    def test_pr_for_sha_ignores_sha_that_only_heads_a_merged_pr(self) -> None:
        # The open-pulls listing no longer contains the merged PR, so its old head SHA matches nothing.
        runner = ScriptedRunner({"pulls?state=open": json.dumps([{"number": 7, "head": {"sha": "live"}}])})
        self.assertIsNone(gh.GhCli(REPO, runner).pr_for_sha("merged-head"))

    def test_dispatch_posts_ref_and_inputs(self) -> None:
        runner = ScriptedRunner()
        gh.GhCli(REPO, runner).dispatch("claude-triage.yml", "master", {"issue": "12"})
        argv, stdin = runner.calls[0]
        self.assertIn(f"repos/{REPO}/actions/workflows/claude-triage.yml/dispatches", argv)
        self.assertEqual(json.loads(stdin or ""), {"ref": "master", "inputs": {"issue": "12"}})

    def test_comment_posts_body(self) -> None:
        runner = ScriptedRunner()
        gh.GhCli(REPO, runner).comment(8, "hello")
        argv, stdin = runner.calls[0]
        self.assertIn(f"repos/{REPO}/issues/8/comments", argv)
        self.assertEqual(json.loads(stdin or ""), {"body": "hello"})


class SubprocessRunnerTests(unittest.TestCase):
    def test_runner_error_raises(self) -> None:
        # `sh -c 'echo boom >&2; exit 3'` stands in for a failing gh invocation.
        with self.assertRaises(gh.GhError) as ctx:
            gh.subprocess_runner(["sh", "-c", "echo boom >&2; exit 3"], None)
        self.assertIn("boom", str(ctx.exception))

    def test_runner_returns_stdout_and_feeds_stdin(self) -> None:
        self.assertEqual(gh.subprocess_runner(["cat"], "hi"), "hi")


class FakeGitHubTests(unittest.TestCase):
    def test_fake_matches_protocol(self) -> None:
        client: gh.GitHub = FakeGitHub()  # mypy verifies the structural match
        self.assertIsNone(client.find_open_issue("nothing"))

    def test_fake_issue_lifecycle(self) -> None:
        fake = FakeGitHub()
        number = fake.create_issue("t", "b", ["ci-failure"])
        self.assertEqual(fake.find_open_issue("t"), number)
        fake.comment(number, "again")
        fake.close_issue(number)
        self.assertIsNone(fake.find_open_issue("t"))
        self.assertEqual(fake.issues[number].comments, ["again"])
        self.assertEqual(fake.issues[number].labels, ["ci-failure"])

    def test_fake_upsert_comment(self) -> None:
        fake = FakeGitHub()
        number = fake.create_issue("t", "b", [])
        fake.upsert_comment(number, "<!-- m -->", "v1 <!-- m -->")
        fake.upsert_comment(number, "<!-- m -->", "v2 <!-- m -->")
        self.assertEqual(fake.issues[number].comments, ["v2 <!-- m -->"])

    def test_fake_records_dispatches_and_answers_lookups(self) -> None:
        fake = FakeGitHub()
        fake.green_shas[("nightly.yml", "master")] = "abc"
        fake.prs_by_sha["def"] = 4
        fake.dispatch("w.yml", "master", {"k": "v"})
        self.assertEqual(fake.last_green_sha("nightly.yml", "master"), "abc")
        self.assertIsNone(fake.last_green_sha("nightly.yml", "other"))
        self.assertEqual(fake.pr_for_sha("def"), 4)
        self.assertIsNone(fake.pr_for_sha("nope"))
        self.assertEqual(fake.dispatches, [("w.yml", "master", {"k": "v"})])


if __name__ == "__main__":
    unittest.main()
