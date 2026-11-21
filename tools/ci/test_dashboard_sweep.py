"""Unit tests for tools/ci/dashboard_sweep.py: collecting a week of `dashboard-rows-*` artifacts for the dashboard.

Run with: python -m unittest discover -s tools/ci -p 'test_*.py'
GitHub is replaced by fakes.FakeGitHub, so nothing here touches the network; downloads go to a temporary directory.
"""

from __future__ import annotations

import contextlib
import datetime
import io
import tempfile
import unittest
from pathlib import Path

import dashboard_sweep
from fakes import FakeGitHub
from gh import GhError, RunInfo

REPO = "owner/nibble"


def run(run_id: int, event: str = "schedule", repo: str = REPO, created: str = "2026-11-20T03:00:00Z") -> RunInfo:
    return RunInfo(run_id, event, repo, created)


class SweepTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fake = FakeGitHub()
        self.dest = Path(self.enterContext(tempfile.TemporaryDirectory())) / "swept"

    def sweep(self) -> dashboard_sweep.SweepReport:
        return dashboard_sweep.sweep(self.fake, REPO, "master", "2026-11-14", self.dest)

    def downloaded(self) -> list[str]:
        return (
            sorted(str(p.relative_to(self.dest)) for p in self.dest.rglob("rows.jsonl")) if self.dest.exists() else []
        )

    def test_downloads_the_rows_artifacts_of_trusted_runs_of_all_three_workflows(self) -> None:
        self.fake.recent_runs[("nightly.yml", "master")] = [run(3), run(2, "workflow_dispatch")]
        self.fake.recent_runs[("portability.yml", "master")] = [run(4)]
        self.fake.recent_runs[("ci.yml", "master")] = [run(5, "push")]
        for run_id, name in [(3, "accuracy"), (3, "fuzz_gguf"), (2, "accuracy"), (4, "arm64"), (5, "coverage")]:
            self.fake.artifacts[(run_id, f"dashboard-rows-{name}")] = {"rows.jsonl": f"row {run_id} {name}\n"}
        report = self.sweep()
        self.assertEqual(
            self.downloaded(),
            [
                "2-dashboard-rows-accuracy/rows.jsonl",
                "3-dashboard-rows-accuracy/rows.jsonl",
                "3-dashboard-rows-fuzz_gguf/rows.jsonl",
                "4-dashboard-rows-arm64/rows.jsonl",
                "5-dashboard-rows-coverage/rows.jsonl",
            ],
        )
        self.assertEqual((report.runs, report.artifacts, report.problems), (4, 5, []))

    def test_ignores_untrusted_runs(self) -> None:
        # A fork's pull request, a same-repo pull request, a run whose fork is gone and an odd event: all excluded.
        self.fake.recent_runs[("ci.yml", "master")] = [
            run(1, "pull_request", "fork/nibble"),
            run(2, "pull_request"),
            run(3, "push", ""),
            run(4, "push", "fork/nibble"),
            run(5, "issue_comment"),
            run(6, "push"),
        ]
        for run_id in range(1, 7):
            self.fake.artifacts[(run_id, "dashboard-rows-coverage")] = {"rows.jsonl": "x\n"}
        report = self.sweep()
        self.assertEqual(self.downloaded(), ["6-dashboard-rows-coverage/rows.jsonl"])
        self.assertEqual((report.runs, report.skipped), (1, 5))

    def test_only_dashboard_rows_artifacts_with_safe_names_are_downloaded(self) -> None:
        self.fake.recent_runs[("nightly.yml", "master")] = [run(1)]
        self.fake.artifacts[(1, "dashboard-rows-ok")] = {"rows.jsonl": "x\n"}
        self.fake.artifacts[(1, "e2e-cosines")] = {"rows.jsonl": "x\n"}
        self.fake.artifacts[(1, "dashboard-rows-../escape")] = {"rows.jsonl": "x\n"}
        report = self.sweep()
        self.assertEqual(self.downloaded(), ["1-dashboard-rows-ok/rows.jsonl"])
        self.assertEqual(report.artifacts, 1)
        self.assertEqual(len(report.problems), 1)
        self.assertIn("dashboard-rows-../escape", report.problems[0])

    def test_runs_before_the_window_are_not_swept(self) -> None:
        self.fake.recent_runs[("nightly.yml", "master")] = [run(2), run(1, created="2026-11-10T03:00:00Z")]
        for run_id in (1, 2):
            self.fake.artifacts[(run_id, "dashboard-rows-a")] = {"rows.jsonl": "x\n"}
        self.sweep()
        self.assertEqual(self.downloaded(), ["2-dashboard-rows-a/rows.jsonl"])

    def test_an_artifact_that_vanished_is_reported_and_the_rest_still_downloads(self) -> None:
        self.fake.recent_runs[("nightly.yml", "master")] = [run(1)]
        self.fake.artifacts[(1, "dashboard-rows-a")] = {"rows.jsonl": "x\n"}
        self.fake.vanished.add((1, "dashboard-rows-a"))  # listed, but gone by the time it is downloaded
        self.fake.artifacts[(1, "dashboard-rows-b")] = {"rows.jsonl": "y\n"}
        report = self.sweep()
        self.assertEqual(self.downloaded(), ["1-dashboard-rows-b/rows.jsonl"])
        self.assertEqual(len(report.problems), 1)
        self.assertIn("dashboard-rows-a", report.problems[0])

    def test_a_listing_failure_propagates(self) -> None:
        self.fake.fail_listing = True
        with self.assertRaises(GhError):
            self.sweep()

    def test_nothing_to_sweep_is_fine(self) -> None:
        report = self.sweep()
        self.assertEqual((report.runs, report.artifacts, report.problems), (0, 0, []))


class MainTests(unittest.TestCase):
    def test_main_sweeps_the_last_n_days_and_reports(self) -> None:
        fake = FakeGitHub()
        fake.recent_runs[("nightly.yml", "master")] = [
            run(2, created="2026-11-19T03:00:00Z"),
            run(1, created="2026-11-12T03:00:00Z"),
        ]
        for run_id in (1, 2):
            fake.artifacts[(run_id, "dashboard-rows-a")] = {"rows.jsonl": "x\n"}
        with tempfile.TemporaryDirectory() as tmp:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = dashboard_sweep.main(
                    ["--repo", REPO, "--dest", tmp, "--days", "7"], gh=fake, today=datetime.date(2026, 11, 21)
                )
            self.assertEqual(code, 0)
            self.assertEqual(sorted(p.parent.name for p in Path(tmp).rglob("rows.jsonl")), ["2-dashboard-rows-a"])
        self.assertIn("1 artifact", out.getvalue())

    def test_main_warns_about_problems_and_fails_on_a_listing_error(self) -> None:
        fake = FakeGitHub()
        fake.recent_runs[("nightly.yml", "master")] = [run(1)]
        fake.artifacts[(1, "dashboard-rows-../x")] = {}
        with tempfile.TemporaryDirectory() as tmp:
            err = io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(dashboard_sweep.main(["--repo", REPO, "--dest", tmp], gh=fake), 0)
            self.assertIn("::warning", err.getvalue())
            fake.fail_listing = True
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertEqual(dashboard_sweep.main(["--repo", REPO, "--dest", tmp], gh=fake), 1)
            self.assertIn("simulated listing failure", err.getvalue())


if __name__ == "__main__":
    unittest.main()
