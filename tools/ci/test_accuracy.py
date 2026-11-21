"""Unit tests for tools/ci/accuracy.py: the nightly accuracy gates, the snapshot file and the baseline comparison.

Run with: python -m unittest discover -s tools/ci -p 'test_*.py'
GitHub is replaced by fakes.FakeGitHub, so nothing here touches `gh` or the network.
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import accuracy
from cosine import Cosine
from fakes import FakeGitHub


def warn(prompt: str, model: str, value: str, positions: int = 12) -> str:
    """One Catch2 WARN block as ctest prints it, for the given prompt, model and min cosine text."""
    return (
        "tests/test_e2e_golden.cpp:194:\nwarning:\n"
        f"  {prompt} [{model}]: min cosine over {positions} positions = {value}, "
        f"top-1 agreement {positions}/{positions}\n\n"
    )


def log_of(*entries: tuple[str, str, str]) -> str:
    """A ctest log holding one WARN per (prompt, model, value) entry."""
    return "".join(warn(p, m, v) for p, m, v in entries)


def snapshot_text(values: dict[str, float | str]) -> str:
    """The text of an e2e-cosines.json holding `values` ("<model>/<prompt>" -> min_cos)."""
    data = {}
    for key, v in values.items():
        model, prompt = key.split("/")
        data[key] = {"model": model, "prompt": prompt, "min_cos": v, "positions": 12, "agree": 12}
    return json.dumps(data)


class BuildSnapshotTests(unittest.TestCase):
    def test_complete_log_has_no_problems(self) -> None:
        log = log_of(("capital", "q4_0", "0.999123"), ("chat", "f16", "0.5"))
        snap = accuracy.build_snapshot(log, expected=2)
        self.assertEqual(snap.problems, [])
        self.assertEqual(sorted(snap.runs), ["f16/chat", "q4_0/capital"])

    def test_fewer_than_expected_keys_is_a_problem(self) -> None:
        snap = accuracy.build_snapshot(log_of(("capital", "q4_0", "0.9")), expected=18)
        self.assertEqual(len(snap.problems), 1)
        self.assertIn("only 1 of the expected 18", snap.problems[0])

    def test_empty_log_is_a_problem_not_an_exception(self) -> None:
        snap = accuracy.build_snapshot("", expected=18)
        self.assertEqual(snap.runs, {})
        self.assertIn("only 0 of the expected 18", snap.problems[0])

    def test_non_finite_values_are_a_problem_and_sorted(self) -> None:
        log = log_of(("b", "q8_0", "nan"), ("a", "q4_0", "-inf"), ("c", "f16", "0.9"))
        snap = accuracy.build_snapshot(log, expected=3)
        self.assertEqual(snap.problems, ["non-finite min cosine: q4_0/a, q8_0/b"])

    def test_conflicting_duplicate_raises(self) -> None:
        log = log_of(("capital", "q4_0", "0.9"), ("capital", "q4_0", "0.8"))
        with self.assertRaises(ValueError):
            accuracy.build_snapshot(log, expected=1)

    def test_snapshot_json_writes_non_finite_as_strings_and_round_trips(self) -> None:
        snap = accuracy.build_snapshot(log_of(("a", "q4_0", "nan"), ("b", "q4_0", "0.25")), expected=2)
        text = accuracy.snapshot_json(snap.runs)
        raw = json.loads(text)  # strict JSON: no bare NaN
        self.assertEqual(raw["q4_0/a"]["min_cos"], "nan")
        self.assertEqual(raw["q4_0/b"]["min_cos"], 0.25)
        loaded = accuracy.load_snapshot(text)
        self.assertEqual(sorted(loaded), ["q4_0/a", "q4_0/b"])
        self.assertEqual(loaded["q4_0/b"], Cosine("q4_0", "b", 0.25, 12, 12))
        self.assertNotEqual(loaded["q4_0/a"].min_cos, loaded["q4_0/a"].min_cos)  # NaN again


class LoadSnapshotTests(unittest.TestCase):
    def test_rejects_garbage(self) -> None:
        for text in [
            "",
            "not json",
            "[]",
            '{"a/b": 3}',
            '{"a/b": {"model": "a"}}',
            '{"a/b": {"model": "a", "prompt": "b", "min_cos": "x", "positions": 1, "agree": 1}}',
        ]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                accuracy.load_snapshot(text)


class FindBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gh = FakeGitHub()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dest = Path(self.tmp.name)

    def find(self) -> accuracy.Baseline | None:
        return accuracy.find_baseline(
            self.gh,
            workflow_file="nightly.yml",
            branch="master",
            job="accuracy",
            artifact="e2e-cosines",
            filename="e2e-cosines.json",
            dest=self.dest,
        )

    def seed(self, run: int, conclusion: str | None, has_artifact: bool = True, text: str = "{}") -> None:
        self.gh.completed_runs.setdefault(("nightly.yml", "master"), []).append(run)
        if conclusion is not None:
            self.gh.job_conclusions[(run, "accuracy")] = conclusion
        if has_artifact:
            self.gh.artifacts[(run, "e2e-cosines")] = {"e2e-cosines.json": text}

    def test_no_runs_is_none(self) -> None:
        self.assertIsNone(self.find())

    def test_newest_run_with_a_successful_accuracy_job_wins(self) -> None:
        # Run 30: the workflow failed (fuzz_load) but accuracy passed. Run 20: accuracy failed. Run 10: older green.
        self.seed(30, "success", text=snapshot_text({"q4_0/a": 0.9}))
        self.seed(20, "failure", text=snapshot_text({"q4_0/a": 0.1}))
        self.seed(10, "success", text=snapshot_text({"q4_0/a": 0.5}))
        found = self.find()
        assert found is not None
        self.assertEqual(found.run_id, 30)
        self.assertEqual(json.loads(found.path.read_text())["q4_0/a"]["min_cos"], 0.9)

    def test_skips_failed_skipped_and_missing_accuracy_jobs(self) -> None:
        self.seed(40, "failure")
        self.seed(30, "skipped")  # a hand-run of one other job: accuracy never ran
        self.seed(20, None)  # no such job at all
        self.seed(10, "success")
        found = self.find()
        assert found is not None
        self.assertEqual(found.run_id, 10)

    def test_skips_a_successful_run_without_the_artifact(self) -> None:
        self.seed(20, "success", has_artifact=False)
        self.seed(10, "success")
        found = self.find()
        assert found is not None
        self.assertEqual(found.run_id, 10)

    def test_artifact_without_the_expected_file_is_skipped(self) -> None:
        self.seed(20, "success")
        self.gh.artifacts[(20, "e2e-cosines")] = {"other.txt": "x"}
        self.assertIsNone(self.find())

    def test_only_the_first_limit_runs_are_looked_at(self) -> None:
        for run in range(100, 90, -1):  # ten runs without a usable baseline
            self.seed(run, "failure")
        self.seed(5, "success")
        self.assertIsNone(
            accuracy.find_baseline(
                self.gh,
                workflow_file="nightly.yml",
                branch="master",
                job="accuracy",
                artifact="e2e-cosines",
                filename="e2e-cosines.json",
                dest=self.dest,
                limit=10,
            )
        )


class CompareTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gh = FakeGitHub()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.current = self.dir / "e2e-cosines.json"
        self.current.write_text(snapshot_text({"q4_0/a": 0.9, "q4_0/b": 0.8}))

    def compare(self) -> accuracy.Comparison:
        return accuracy.compare_with_baseline(
            self.gh,
            self.current,
            workflow_file="nightly.yml",
            branch="master",
            job="accuracy",
            artifact="e2e-cosines",
            dest=self.dir / "prev",
        )

    def seed(self, text: str, run: int = 7) -> None:
        self.gh.completed_runs[("nightly.yml", "master")] = [run]
        self.gh.job_conclusions[(run, "accuracy")] = "success"
        self.gh.artifacts[(run, "e2e-cosines")] = {"e2e-cosines.json": text}

    def test_first_night_when_nothing_is_on_record(self) -> None:
        got = self.compare()
        self.assertIn("First night", got.markdown)
        self.assertEqual(got.warnings, [])

    def test_unchanged_values(self) -> None:
        self.seed(snapshot_text({"q4_0/a": 0.9, "q4_0/b": 0.8}))
        got = self.compare()
        self.assertIn("Previous run: 7.", got.markdown)
        self.assertIn("No min-cosine value changed.", got.markdown)
        self.assertEqual(got.warnings, [])

    def test_changed_added_and_removed_values_are_tabulated_and_warned(self) -> None:
        self.seed(snapshot_text({"q4_0/a": 0.95, "q4_0/gone": 0.7}))
        got = self.compare()
        self.assertIn("| q4_0/a | 0.95 | 0.9 |", got.markdown)
        self.assertIn("| q4_0/b | absent | 0.8 |", got.markdown)
        self.assertIn("| q4_0/gone | 0.7 | absent |", got.markdown)
        self.assertEqual(len(got.warnings), 3)
        self.assertIn("q4_0/a: 0.95 -> 0.9", got.warnings[0])

    def test_unreadable_baseline_costs_only_the_comparison(self) -> None:
        self.seed("{ this is not json")
        got = self.compare()
        self.assertIn("Comparison unavailable", got.markdown)
        self.assertEqual(len(got.warnings), 1)

    def test_listing_failure_costs_only_the_comparison(self) -> None:
        self.gh.fail_listing = True
        got = self.compare()
        self.assertIn("Comparison unavailable", got.markdown)
        self.assertEqual(len(got.warnings), 1)

    def test_unreadable_current_snapshot_raises(self) -> None:
        self.current.write_text("nope")
        with self.assertRaises(ValueError):
            self.compare()


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def run_cli(self, *argv: str, gh: FakeGitHub | None = None) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = accuracy.main(list(argv), gh=gh, repo="o/r")
        return code, out.getvalue()

    def test_snapshot_ok_writes_both_files_and_exits_0(self) -> None:
        log = self.dir / "e2e.log"
        log.write_text(log_of(("capital", "q4_0", "0.9"), ("chat", "f16", "0.8")))
        code, out = self.run_cli(
            "snapshot",
            str(log),
            "--out",
            str(self.dir / "s.json"),
            "--summary",
            str(self.dir / "sum.md"),
            "--expected",
            "2",
        )
        self.assertEqual(code, 0)
        self.assertEqual(out, "")
        self.assertEqual(sorted(json.loads((self.dir / "s.json").read_text())), ["f16/chat", "q4_0/capital"])
        self.assertIn("2 min-cosine results parsed.", (self.dir / "sum.md").read_text())

    def test_snapshot_too_few_keys_fails_but_still_writes_what_it_found(self) -> None:
        log = self.dir / "e2e.log"
        log.write_text(log_of(("capital", "q4_0", "0.9")))
        code, out = self.run_cli(
            "snapshot",
            str(log),
            "--out",
            str(self.dir / "s.json"),
            "--summary",
            str(self.dir / "sum.md"),
            "--expected",
            "18",
        )
        self.assertEqual(code, 1)
        self.assertIn("::error title=e2e accuracy::only 1 of the expected 18", out)
        self.assertIn("- FAILED: only 1 of the expected 18", (self.dir / "sum.md").read_text())
        self.assertTrue((self.dir / "s.json").exists())

    def test_snapshot_lenient_reports_problems_as_warnings_and_exits_0(self) -> None:
        log = self.dir / "e2e.log"
        log.write_text(log_of(("capital", "q4_0", "nan")))
        code, out = self.run_cli(
            "snapshot",
            str(log),
            "--out",
            str(self.dir / "s.json"),
            "--summary",
            str(self.dir / "sum.md"),
            "--expected",
            "18",
            "--lenient",
        )
        self.assertEqual(code, 0)
        self.assertIn("::warning title=e2e accuracy::only 1 of the expected 18", out)
        self.assertIn("::warning title=e2e accuracy::non-finite min cosine: q4_0/capital", out)
        self.assertNotIn("::error", out)
        self.assertIn("- WARNING: only 1 of the expected 18", (self.dir / "sum.md").read_text())
        self.assertEqual(list(json.loads((self.dir / "s.json").read_text())), ["q4_0/capital"])

    def test_snapshot_lenient_still_fails_on_an_unreadable_log(self) -> None:
        code, out = self.run_cli(
            "snapshot",
            str(self.dir / "missing.log"),
            "--out",
            str(self.dir / "s.json"),
            "--summary",
            str(self.dir / "sum.md"),
            "--expected",
            "0",
            "--lenient",
        )
        self.assertEqual(code, 1)
        self.assertIn("::error title=e2e log::", out)
        self.assertFalse((self.dir / "s.json").exists())

    def test_snapshot_non_finite_fails(self) -> None:
        log = self.dir / "e2e.log"
        log.write_text(log_of(("capital", "q4_0", "nan")))
        code, out = self.run_cli(
            "snapshot",
            str(log),
            "--out",
            str(self.dir / "s.json"),
            "--summary",
            str(self.dir / "sum.md"),
            "--expected",
            "1",
        )
        self.assertEqual(code, 1)
        self.assertIn("non-finite min cosine: q4_0/capital", out)
        json.loads((self.dir / "s.json").read_text())  # still strict JSON

    def test_snapshot_missing_log_fails(self) -> None:
        code, out = self.run_cli(
            "snapshot",
            str(self.dir / "nope.log"),
            "--out",
            str(self.dir / "s.json"),
            "--summary",
            str(self.dir / "sum.md"),
            "--expected",
            "1",
        )
        self.assertEqual(code, 1)
        self.assertIn("::error title=e2e log::", out)

    def test_compare_never_fails_the_job_on_a_fetch_problem(self) -> None:
        cur = self.dir / "e2e-cosines.json"
        cur.write_text(snapshot_text({"q4_0/a": 0.9}))
        gh = FakeGitHub()
        gh.fail_listing = True
        code, out = self.run_cli(
            "compare", str(cur), "--summary", str(self.dir / "sum.md"), "--dest", str(self.dir / "prev"), gh=gh
        )
        self.assertEqual(code, 0)
        self.assertIn("::warning title=nightly::", out)
        self.assertIn("### e2e accuracy versus the previous night", (self.dir / "sum.md").read_text())

    def test_compare_reports_changes(self) -> None:
        cur = self.dir / "e2e-cosines.json"
        cur.write_text(snapshot_text({"q4_0/a": 0.9}))
        gh = FakeGitHub()
        gh.completed_runs[("nightly.yml", "master")] = [3]
        gh.job_conclusions[(3, "accuracy")] = "success"
        gh.artifacts[(3, "e2e-cosines")] = {"e2e-cosines.json": snapshot_text({"q4_0/a": 0.8})}
        code, out = self.run_cli(
            "compare", str(cur), "--summary", str(self.dir / "sum.md"), "--dest", str(self.dir / "prev"), gh=gh
        )
        self.assertEqual(code, 0)
        self.assertIn("::warning title=min cosine changed::q4_0/a: 0.8 -> 0.9", out)
        self.assertIn("| q4_0/a | 0.8 | 0.9 |", (self.dir / "sum.md").read_text())


if __name__ == "__main__":
    unittest.main()
