"""Unit tests for tools/ci/gh.py (GhCli) and tools/ci/fakes.py (FakeGitHub).

Run with: python -m unittest discover -s tools/ci -p 'test_*.py'
GhCli is exercised through a scripted Runner that records every call, so no test touches the network or `gh`.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path

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

    def test_dispatch_encodes_workflow_file(self) -> None:
        runner = ScriptedRunner()
        gh.GhCli(REPO, runner).dispatch("my flow#1.yml", "master", {})
        self.assertIn(f"repos/{REPO}/actions/workflows/my%20flow%231.yml/dispatches", runner.calls[0][0])

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


class RunLookupTests(unittest.TestCase):
    """completed_runs, job_conclusion and download_artifact: how the nightly finds the previous night's numbers."""

    def test_list_completed_runs_lists_ids_newest_first(self) -> None:
        runs = json.dumps({"workflow_runs": [{"id": 30}, {"id": 20}, {"id": 10}]})
        runner = ScriptedRunner({"runs": runs})
        self.assertEqual(gh.GhCli(REPO, runner).list_completed_runs("nightly.yml", "master", 10), [30, 20, 10])
        self.assertEqual(
            runner.calls[0][0][-1],
            f"repos/{REPO}/actions/workflows/nightly.yml/runs?branch=master&status=completed&per_page=10",
        )

    def test_job_conclusion_matches_the_job_by_name(self) -> None:
        jobs = json.dumps(
            {"jobs": [{"name": "fuzz_load", "conclusion": "failure"}, {"name": "accuracy", "conclusion": "success"}]}
        )
        runner = ScriptedRunner({"runs/5/jobs": jobs})
        client = gh.GhCli(REPO, runner)
        self.assertEqual(client.job_conclusion(5, "accuracy"), "success")
        self.assertEqual(client.job_conclusion(5, "fuzz_load"), "failure")
        self.assertIsNone(client.job_conclusion(5, "trend-bench"))
        self.assertIn("filter=latest", runner.calls[0][0][-1])

    def test_job_conclusion_reads_every_page_and_none_while_running(self) -> None:
        page1 = json.dumps({"jobs": [{"name": "a", "conclusion": "success"}]})
        page2 = json.dumps({"jobs": [{"name": "accuracy", "conclusion": None}]})
        runner = ScriptedRunner({"runs/5/jobs": page1 + "\n" + page2})
        self.assertIsNone(gh.GhCli(REPO, runner).job_conclusion(5, "accuracy"))

    def test_download_artifact_downloads_when_it_exists(self) -> None:
        listing = json.dumps({"artifacts": [{"name": "e2e-cosines", "expired": False}]})
        runner = ScriptedRunner({"runs/9/artifacts": listing})
        self.assertTrue(gh.GhCli(REPO, runner).download_artifact(9, "e2e-cosines", Path("out dir")))
        argv = runner.calls[-1][0]
        self.assertEqual(argv[:3], ["gh", "run", "download"])
        self.assertEqual(argv[3], "9")
        self.assertEqual(argv[argv.index("-n") + 1], "e2e-cosines")
        self.assertEqual(argv[argv.index("-D") + 1], "out dir")
        self.assertEqual(argv[argv.index("-R") + 1], REPO)

    def test_download_artifact_false_when_absent_or_expired(self) -> None:
        listings: list[dict[str, list[dict[str, object]]]] = [
            {"artifacts": []},
            {"artifacts": [{"name": "e2e-cosines", "expired": True}]},
            {"artifacts": [{"name": "other", "expired": False}]},
        ]
        for listing in listings:
            with self.subTest(listing=listing):
                runner = ScriptedRunner({"runs/9/artifacts": json.dumps(listing)})
                self.assertFalse(gh.GhCli(REPO, runner).download_artifact(9, "e2e-cosines", Path("d")))
                self.assertEqual(len(runner.calls), 1)  # no download attempted


class RecentRunsTests(unittest.TestCase):
    """list_recent_runs and list_artifact_names: what the dashboard sweep asks about."""

    def test_list_recent_runs_reads_every_page_with_the_fields_the_sweep_filters_on(self) -> None:
        def run(run_id: int, event: str, repo: str, created: str) -> dict[str, object]:
            return {"id": run_id, "event": event, "head_repository": {"full_name": repo}, "created_at": created}

        page1 = json.dumps({"workflow_runs": [run(3, "schedule", REPO, "2026-11-21T03:17:00Z")]})
        page2 = json.dumps({"workflow_runs": [run(2, "pull_request", "fork/nibble", "2026-11-20T01:00:00Z")]})
        runner = ScriptedRunner({"workflows/nightly.yml/runs": page1 + page2})
        runs = gh.GhCli(REPO, runner).list_recent_runs("nightly.yml", "master", "2026-11-14")
        self.assertEqual(
            runs,
            [
                gh.RunInfo(3, "schedule", REPO, "2026-11-21T03:17:00Z"),
                gh.RunInfo(2, "pull_request", "fork/nibble", "2026-11-20T01:00:00Z"),
            ],
        )
        argv = runner.calls[0][0]
        self.assertIn("--paginate", argv)
        self.assertEqual(
            argv[-1],
            f"repos/{REPO}/actions/workflows/nightly.yml/runs?branch=master&status=completed"
            "&created=%3E%3D2026-11-14&per_page=100",
        )

    def test_list_recent_runs_tolerates_a_missing_head_repository(self) -> None:
        # GitHub leaves head_repository null when the fork behind a pull request was deleted.
        body = json.dumps(
            {"workflow_runs": [{"id": 1, "event": "pull_request", "head_repository": None, "created_at": "x"}]}
        )
        runs = gh.GhCli(REPO, ScriptedRunner({"/runs": body})).list_recent_runs("ci.yml", "master", "2026-11-14")
        self.assertEqual(runs, [gh.RunInfo(1, "pull_request", "", "x")])

    def test_list_artifact_names_skips_expired_ones_and_reads_every_page(self) -> None:
        page1 = json.dumps(
            {"artifacts": [{"name": "dashboard-rows-a", "expired": False}, {"name": "old", "expired": True}]}
        )
        page2 = json.dumps({"artifacts": [{"name": "dashboard-rows-b", "expired": False}]})
        runner = ScriptedRunner({"runs/9/artifacts": page1 + page2})
        self.assertEqual(gh.GhCli(REPO, runner).list_artifact_names(9), ["dashboard-rows-a", "dashboard-rows-b"])
        self.assertIn("--paginate", runner.calls[0][0])
        self.assertEqual(runner.calls[0][0][-1], f"repos/{REPO}/actions/runs/9/artifacts?per_page=100")


class BranchTipTests(unittest.TestCase):
    def test_branch_tip_reads_the_commit_sha(self) -> None:
        runner = ScriptedRunner({"branches/master": json.dumps({"name": "master", "commit": {"sha": "abc123"}})})
        self.assertEqual(gh.GhCli(REPO, runner).branch_tip("master"), "abc123")
        self.assertEqual(runner.calls[0][0][-1], f"repos/{REPO}/branches/master")

    def test_branch_tip_keeps_slashes_and_encodes_the_rest(self) -> None:
        runner = ScriptedRunner({"branches/": json.dumps({"commit": {"sha": "abc123"}})})
        gh.GhCli(REPO, runner).branch_tip("release/v1#hotfix")
        self.assertEqual(runner.calls[0][0][-1], f"repos/{REPO}/branches/release/v1%23hotfix")

    def test_branch_tip_none_without_a_commit(self) -> None:
        self.assertIsNone(gh.GhCli(REPO, ScriptedRunner({"branches/": "{}"})).branch_tip("master"))


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

    def test_fake_branch_tip(self) -> None:
        fake = FakeGitHub()
        fake.branch_tips["master"] = "abc"
        self.assertEqual(fake.branch_tip("master"), "abc")
        self.assertIsNone(fake.branch_tip("other"))
        fake.fail_tip = True
        with self.assertRaises(gh.GhError):
            fake.branch_tip("master")

    def test_fake_run_lookups(self) -> None:
        fake = FakeGitHub()
        fake.completed_runs[("nightly.yml", "master")] = [3, 2, 1]
        fake.job_conclusions[(3, "accuracy")] = "success"
        fake.artifacts[(3, "snap")] = {"a.json": "{}", "sub/b.txt": "x"}
        self.assertEqual(fake.list_completed_runs("nightly.yml", "master", 2), [3, 2])
        self.assertEqual(fake.job_conclusion(3, "accuracy"), "success")
        self.assertIsNone(fake.job_conclusion(2, "accuracy"))
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(fake.download_artifact(3, "snap", Path(tmp) / "d"))
            self.assertEqual((Path(tmp) / "d" / "a.json").read_text(), "{}")
            self.assertFalse(fake.download_artifact(3, "nope", Path(tmp) / "e"))

    def test_fake_recent_runs_filter_by_creation_date_and_list_artifact_names(self) -> None:
        fake = FakeGitHub()
        old = gh.RunInfo(1, "schedule", REPO, "2026-11-01T03:00:00Z")
        new = gh.RunInfo(2, "schedule", REPO, "2026-11-20T03:00:00Z")
        fake.recent_runs[("nightly.yml", "master")] = [new, old]
        fake.artifacts[(2, "dashboard-rows-a")] = {"rows.jsonl": ""}
        fake.artifacts[(2, "other")] = {}
        self.assertEqual(fake.list_recent_runs("nightly.yml", "master", "2026-11-14"), [new])
        self.assertEqual(fake.list_recent_runs("ci.yml", "master", "2026-11-14"), [])
        self.assertEqual(sorted(fake.list_artifact_names(2)), ["dashboard-rows-a", "other"])
        self.assertEqual(fake.list_artifact_names(1), [])
        fake.fail_listing = True
        with self.assertRaises(gh.GhError):
            fake.list_recent_runs("nightly.yml", "master", "2026-11-14")


if __name__ == "__main__":
    unittest.main()
