"""Unit tests for tools/ci/report_failure.py, run against fakes.FakeGitHub (no `gh`, no network).

Run with: python -m unittest discover -s tools/ci -p 'test_*.py'
They pin the issue-spam rules: one open issue per job found by exact title, repeats comment instead of reopening,
success closes only that job's issue, and triage is dispatched only when an issue is opened.
"""

from __future__ import annotations

import contextlib
import io
import unittest

import report_failure
from fakes import FakeGitHub
from report_failure import RunContext


def make_ctx(
    job: str = "fuzz_gguf",
    *,
    sha: str = "abc1234",
    attachments_url: str | None = None,
    workflow_file: str = "nightly.yml",
) -> RunContext:
    """A RunContext for `job` (of `workflow_file`) with fixed values for everything else."""
    return RunContext(
        job=job,
        workflow_file=workflow_file,
        run_url="https://github.com/o/r/actions/runs/42",
        sha=sha,
        repo="o/r",
        area="ci",
        attachments_url=attachments_url,
    )


def at_tip() -> FakeGitHub:
    """A FakeGitHub whose master tip is make_ctx()'s default sha, so a run of that commit is not stale."""
    gh = FakeGitHub()
    gh.branch_tips["master"] = "abc1234"
    return gh


class DecideTests(unittest.TestCase):
    def test_decide_table(self) -> None:
        self.assertEqual(report_failure.decide(None, "failure"), "open")
        self.assertEqual(report_failure.decide(7, "failure"), "comment")
        self.assertEqual(report_failure.decide(7, "success"), "close")
        self.assertEqual(report_failure.decide(None, "success"), "none")

    def test_title_is_prefixed_with_the_workflow_name(self) -> None:
        self.assertEqual(report_failure.issue_title("nightly.yml", "fuzz_gguf"), "nightly: fuzz_gguf failing")
        self.assertEqual(report_failure.issue_title("portability.yml", "arm64"), "portability: arm64 failing")
        self.assertEqual(report_failure.issue_title("portability.yaml", "arm64"), "portability: arm64 failing")


class ReportTests(unittest.TestCase):
    def test_failure_opens_once_then_comments(self) -> None:
        gh = FakeGitHub()
        self.assertEqual(report_failure.report(gh, make_ctx(), "failure", triage=False), "open")
        self.assertEqual(report_failure.report(gh, make_ctx(), "failure", triage=False), "comment")
        self.assertEqual(len(gh.issues), 1)
        issue = gh.issues[1]
        self.assertEqual(issue.title, "nightly: fuzz_gguf failing")
        self.assertEqual(issue.labels, ["ci-failure", "ci"])
        self.assertEqual(len(issue.comments), 1)
        self.assertTrue(issue.open)

    def test_two_jobs_two_issues(self) -> None:
        gh = FakeGitHub()
        report_failure.report(gh, make_ctx("fuzz_gguf"), "failure", triage=False)
        report_failure.report(gh, make_ctx("fuzz_load"), "failure", triage=False)
        titles = sorted(i.title for i in gh.issues.values())
        self.assertEqual(titles, ["nightly: fuzz_gguf failing", "nightly: fuzz_load failing"])
        self.assertTrue(all(not i.comments for i in gh.issues.values()))

    def test_same_job_name_in_two_workflows_gets_two_issues(self) -> None:
        gh = at_tip()
        nightly = make_ctx("arm64", workflow_file="nightly.yml")
        portability = make_ctx("arm64", workflow_file="portability.yml")
        self.assertEqual(report_failure.report(gh, nightly, "failure", triage=False), "open")
        self.assertEqual(report_failure.report(gh, portability, "failure", triage=False), "open")
        titles = sorted(i.title for i in gh.issues.values())
        self.assertEqual(titles, ["nightly: arm64 failing", "portability: arm64 failing"])
        # Recovery of one workflow's job closes only that workflow's issue.
        self.assertEqual(report_failure.report(gh, portability, "success", triage=False), "close")
        self.assertTrue(gh.issues[1].open)
        self.assertFalse(gh.issues[2].open)

    def test_success_closes_only_own_issue(self) -> None:
        gh = at_tip()
        report_failure.report(gh, make_ctx("fuzz_gguf"), "failure", triage=False)
        report_failure.report(gh, make_ctx("fuzz_load"), "failure", triage=False)
        self.assertEqual(report_failure.report(gh, make_ctx("fuzz_gguf"), "success", triage=False), "close")
        self.assertFalse(gh.issues[1].open)
        self.assertTrue(gh.issues[2].open)
        self.assertEqual(len(gh.issues[1].comments), 1, "closing leaves a recovery comment")
        self.assertEqual(gh.issues[2].comments, [])

    def test_stale_rerun_success_comments_but_never_closes(self) -> None:
        # A newer commit opened the issue; a rerun of an older commit (same SHA and ref as its original run) passes.
        gh = FakeGitHub()
        gh.branch_tips["master"] = "newer99"
        report_failure.report(gh, make_ctx(sha="newer99"), "failure", triage=False)
        self.assertEqual(report_failure.report(gh, make_ctx(sha="older11"), "success", triage=False), "stale")
        issue = gh.issues[1]
        self.assertTrue(issue.open)
        self.assertEqual(len(issue.comments), 1)
        self.assertIn("older11", issue.comments[0])
        self.assertIn("newer99", issue.comments[0])
        self.assertIn("Not closing", issue.comments[0])

    def test_success_with_unknown_branch_tip_is_treated_as_stale(self) -> None:
        gh = FakeGitHub()  # no tip on record: the lookup found nothing, so closing would be a guess
        report_failure.report(gh, make_ctx(), "failure", triage=False)
        self.assertEqual(report_failure.report(gh, make_ctx(), "success", triage=False), "stale")
        self.assertTrue(gh.issues[1].open)

    def test_stale_success_without_issue_is_noop_and_needs_no_lookup(self) -> None:
        gh = FakeGitHub()
        gh.fail_tip = True  # would raise if the tip were looked up
        self.assertEqual(report_failure.report(gh, make_ctx(sha="older11"), "success", triage=False), "none")

    def test_stale_failure_still_comments(self) -> None:
        gh = FakeGitHub()
        gh.branch_tips["master"] = "newer99"
        report_failure.report(gh, make_ctx(sha="newer99"), "failure", triage=False)
        self.assertEqual(report_failure.report(gh, make_ctx(sha="older11"), "failure", triage=False), "comment")
        self.assertTrue(gh.issues[1].open)

    def test_success_without_issue_is_noop(self) -> None:
        gh = FakeGitHub()
        self.assertEqual(report_failure.report(gh, make_ctx(), "success", triage=True), "none")
        self.assertEqual(gh.issues, {})
        self.assertEqual(gh.dispatches, [])

    def test_failure_after_recovery_opens_new_issue(self) -> None:
        gh = at_tip()
        report_failure.report(gh, make_ctx(), "failure", triage=False)
        report_failure.report(gh, make_ctx(), "success", triage=False)
        self.assertEqual(report_failure.report(gh, make_ctx(), "failure", triage=False), "open")
        self.assertEqual(len(gh.issues), 2)

    def test_triage_dispatched_only_on_open(self) -> None:
        gh = at_tip()
        report_failure.report(gh, make_ctx(), "failure", triage=True)
        self.assertEqual(gh.dispatches, [("claude-triage.yml", "master", {"issue": "1"})])
        report_failure.report(gh, make_ctx(), "failure", triage=True)  # comment: no second dispatch
        report_failure.report(gh, make_ctx(), "success", triage=True)  # close: no dispatch
        self.assertEqual(len(gh.dispatches), 1)

    def test_no_dispatch_without_triage(self) -> None:
        gh = FakeGitHub()
        report_failure.report(gh, make_ctx(), "failure", triage=False)
        self.assertEqual(gh.dispatches, [])

    def test_compare_link_uses_last_green_run_of_the_workflow(self) -> None:
        gh = FakeGitHub()
        gh.green_shas[("nightly.yml", "master")] = "green99"
        report_failure.report(gh, make_ctx(), "failure", triage=False)
        self.assertIn("https://github.com/o/r/compare/green99...abc1234", gh.issues[1].body)


class BodyTests(unittest.TestCase):
    def test_body_has_compare_link(self) -> None:
        body = report_failure.failure_body(make_ctx(), "green99")
        self.assertIn("https://github.com/o/r/compare/green99...abc1234", body)
        self.assertIn("https://github.com/o/r/actions/runs/42", body)
        self.assertIn("abc1234", body)

    def test_body_without_green_run(self) -> None:
        body = report_failure.failure_body(make_ctx(), None)
        self.assertIn("no green run on record", body)
        self.assertNotIn("/compare/", body)

    def test_body_attachments_link(self) -> None:
        url = "https://github.com/o/r/actions/runs/42/artifacts/9"
        self.assertIn(url, report_failure.failure_body(make_ctx(attachments_url=url), None))
        self.assertNotIn("ttachments", report_failure.failure_body(make_ctx(), None))

    def test_recovery_body(self) -> None:
        body = report_failure.recovery_body(make_ctx())
        self.assertIn("https://github.com/o/r/actions/runs/42", body)
        self.assertIn("abc1234", body)


class MainTests(unittest.TestCase):
    ARGS = [
        "--job", "fuzz_gguf", "--workflow-file", "nightly.yml", "--outcome", "failure",
        "--run-url", "https://github.com/o/r/actions/runs/42", "--sha", "abc1234", "--area", "build",
    ]  # fmt: skip

    def test_main_opens_issue_with_area_label_and_attachments(self) -> None:
        gh = FakeGitHub()
        with contextlib.redirect_stdout(io.StringIO()):
            code = report_failure.main([*self.ARGS, "--attachments-url", "https://x/y", "--triage"], gh=gh, repo="o/r")
        self.assertEqual(code, 0)
        self.assertEqual(gh.issues[1].labels, ["ci-failure", "build"])
        self.assertIn("https://x/y", gh.issues[1].body)
        self.assertEqual(gh.dispatches, [("claude-triage.yml", "master", {"issue": "1"})])

    def test_main_rejects_unknown_outcome(self) -> None:
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            report_failure.main([*self.ARGS[:5], "cancelled", *self.ARGS[6:]], gh=FakeGitHub(), repo="o/r")

    def test_main_treats_empty_attachments_url_as_none(self) -> None:
        gh = FakeGitHub()
        with contextlib.redirect_stdout(io.StringIO()):
            report_failure.main([*self.ARGS, "--attachments-url", ""], gh=gh, repo="o/r")
        self.assertNotIn("ttachments", gh.issues[1].body)


if __name__ == "__main__":
    unittest.main()
