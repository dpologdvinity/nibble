"""Unit tests for tools/ci/pr_report.py, run against fakes.FakeGitHub (no `gh`, no network).

Run with: python -m unittest discover -s tools/ci -p 'test_*.py'
The `test_rejects_*` tests pin the trust boundary: the report is an artifact written by a (possibly fork) pull
request and read by a job that holds a write token, so `load` must refuse anything but the exact schema, and `post`
must take the pull request number from the API and refuse a report that names another commit.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import re
import tempfile
import unittest
from pathlib import Path
from typing import Any

import pr_report
from fakes import FakeGitHub
from pr_report import MARKER, MODEL_ALLOWLIST, PerfEntry, Report

BASE = "a" * 40
HEAD = "b" * 40
MODEL = "qwen2.5-0.5b-instruct-q4_0.gguf"


def make_dict() -> dict[str, Any]:
    """A valid report as plain JSON data: every section present, one perf entry, two accuracy rows."""
    return {
        "schema": 1,
        "base_sha": BASE,
        "head_sha": HEAD,
        "runner": "ubuntu-24.04 (AMD EPYC 7763 64-Core Processor)",
        "kernels_changed": True,
        "perf": [
            {
                "model": MODEL,
                "test": "tg128",
                "threads": 1,
                "base_median": 40.0,
                "head_median": 44.0,
                "ratios": [1.08, 1.1, 1.12],
                "verdict": "faster",
            }
        ],
        "size": {"base_bytes": 1_000_000, "head_bytes": 1_010_000},
        "rss": {"base_kib": 800_000, "head_kib": 790_000},
        "accuracy": {
            "status": "ran",
            "rows": [
                {"key": "q4_0/capital", "base": 0.999991, "head": 0.999991},
                {"key": "q8_0/capital", "base": 0.999999, "head": 0.999998},
            ],
        },
    }


def make_report() -> Report:
    """The valid dict above, pushed through `load` so it is a validated Report."""
    return pr_report.load(json.dumps(make_dict()))


def perf_of(r: Report) -> list[PerfEntry]:
    """The perf section of `r`, which must not be skipped."""
    perf = r["perf"]
    assert perf is not None
    return perf


def skipped_dict() -> dict[str, Any]:
    """A valid report whose every section was skipped."""
    d = make_dict()
    d.update({"perf": None, "size": None, "rss": None, "accuracy": {"status": "skipped-cold-cache", "rows": []}})
    return d


class LoadTests(unittest.TestCase):
    def reject(self, d: dict[str, Any]) -> None:
        with self.assertRaises(ValueError):
            pr_report.load(json.dumps(d))

    def test_load_valid_fixture(self) -> None:
        r = make_report()
        self.assertEqual(r["head_sha"], HEAD)
        self.assertEqual(perf_of(r)[0]["ratios"], [1.08, 1.1, 1.12])
        self.assertEqual(r["accuracy"]["rows"][1]["key"], "q8_0/capital")

    def test_load_accepts_null_sections_and_zero_medians(self) -> None:
        # Ruling P16: perf_ab writes 0.0 for a side with no usable run.
        pr_report.load(json.dumps(skipped_dict()))
        d = make_dict()
        d["perf"][0].update({"base_median": 0.0, "head_median": 0, "ratios": [0.0], "verdict": "inconclusive"})
        self.assertEqual(perf_of(pr_report.load(json.dumps(d)))[0]["head_median"], 0.0)

    def test_rejects_unknown_model(self) -> None:
        d = make_dict()
        d["perf"][0]["model"] = "../../etc/passwd"
        self.reject(d)
        d["perf"][0]["model"] = "evil [click](http://x)"
        self.reject(d)
        self.assertIn(MODEL, MODEL_ALLOWLIST)

    def test_rejects_nan(self) -> None:
        for literal in ("NaN", "Infinity", "-Infinity", "1e999"):
            with self.subTest(literal=literal):
                text = json.dumps(make_dict()).replace("44.0", literal)
                with self.assertRaises(ValueError):
                    pr_report.load(text)

    def test_rejects_negative_and_non_numbers(self) -> None:
        for field, bad in (("base_median", -1.0), ("head_median", "44"), ("head_median", True), ("ratios", [-0.5])):
            with self.subTest(field=field, bad=bad):
                d = make_dict()
                d["perf"][0][field] = bad
                self.reject(d)
        d = make_dict()
        d["size"]["base_bytes"] = 1.5
        self.reject(d)
        d = make_dict()
        d["size"]["head_bytes"] = True
        self.reject(d)
        d = make_dict()
        d["accuracy"]["rows"][0]["head"] = float(2**60)
        self.reject(d)

    def test_rejects_extra_keys(self) -> None:
        d = make_dict()
        d["pr_number"] = 1
        self.reject(d)
        for section in ("size", "rss"):
            d = make_dict()
            d[section]["extra"] = 1
            self.reject(d)
        d = make_dict()
        d["perf"][0]["extra"] = 1
        self.reject(d)
        d = make_dict()
        d["accuracy"]["rows"][0]["extra"] = 1
        self.reject(d)

    def test_rejects_missing_keys(self) -> None:
        for key in make_dict():
            with self.subTest(key=key):
                d = make_dict()
                del d[key]
                self.reject(d)

    def test_rejects_duplicate_keys(self) -> None:
        # json.loads keeps the last duplicate; a validator that sees one value while another is rendered is a hole.
        text = json.dumps(make_dict()).replace('"schema": 1', '"schema": 1, "schema": 1')
        with self.assertRaises(ValueError):
            pr_report.load(text)

    def test_rejects_long_runner(self) -> None:
        d = make_dict()
        d["runner"] = "x" * 121
        self.reject(d)
        d["runner"] = "x" * 120
        pr_report.load(json.dumps(d))

    def test_rejects_control_and_non_ascii_text(self) -> None:
        for bad in ("two\nlines", "tab\there", "café", "‮evil", ""):
            with self.subTest(bad=bad):
                d = make_dict()
                d["runner"] = bad
                self.reject(d)
        d = make_dict()
        d["accuracy"]["rows"][0]["key"] = "q4_0/a\nb"
        self.reject(d)

    def test_rejects_bad_sha(self) -> None:
        for bad in ("a" * 39, "a" * 41, "A" * 40, "g" * 40, "a" * 40 + "\n", "", 5):
            for field in ("base_sha", "head_sha"):
                with self.subTest(bad=bad, field=field):
                    d = make_dict()
                    d[field] = bad
                    self.reject(d)

    def test_rejects_bad_enums_and_types(self) -> None:
        cases: list[tuple[str, Any]] = [
            ("schema", 2),
            ("schema", True),
            ("kernels_changed", 1),
            ("perf", {}),
            ("size", []),
            ("accuracy", None),
        ]
        for field, bad in cases:
            with self.subTest(field=field):
                d = make_dict()
                d[field] = bad
                self.reject(d)
        for field, bad in (("test", "pp1"), ("verdict", "great"), ("threads", 0), ("threads", True), ("threads", 9999)):
            with self.subTest(field=field, bad=bad):
                d = make_dict()
                d["perf"][0][field] = bad
                self.reject(d)
        d = make_dict()
        d["accuracy"]["status"] = "ok"
        self.reject(d)

    def test_rejects_unbounded_lists_and_input(self) -> None:
        d = make_dict()
        d["perf"] = [copy.deepcopy(d["perf"][0]) for _ in range(pr_report.MAX_PERF + 1)]
        self.reject(d)
        d = make_dict()
        d["accuracy"]["rows"] = [{"key": f"m/p{i}", "base": 1.0, "head": 1.0} for i in range(pr_report.MAX_ROWS + 1)]
        self.reject(d)
        d = make_dict()
        d["perf"][0]["ratios"] = [1.0] * (pr_report.MAX_RATIOS + 1)
        self.reject(d)
        with self.assertRaises(ValueError):
            pr_report.load(json.dumps(make_dict()) + " " * pr_report.MAX_BYTES)

    def test_rejects_malformed_json(self) -> None:
        for text in ("", "[]", "null", "{", "[" * 100_000):
            with self.subTest(text=text[:10]), self.assertRaises(ValueError):
                pr_report.load(text)

    def test_error_message_does_not_start_a_workflow_command(self) -> None:
        d = make_dict()
        d["::set-output name=x::y\n::error::boom"] = 1
        with self.assertRaises(ValueError) as ctx:
            pr_report.load(json.dumps(d))
        self.assertFalse(any(line.startswith("::") for line in str(ctx.exception).splitlines()))
        self.assertNotIn("\n", str(ctx.exception))


class RenderTests(unittest.TestCase):
    def test_render_full_layout(self) -> None:
        text = pr_report.render(make_report())
        self.assertTrue(text.startswith(MARKER))
        order = [
            text.index("nibble PR report"),
            text.index("shared CI runner: trend, not a benchmark"),
            text.index("| model "),
            text.index("Size"),
            text.index("Peak memory"),
            text.index("Accuracy"),
            text.index("<details>"),
        ]
        self.assertEqual(order, sorted(order))
        self.assertIn("✅ faster", text)
        self.assertIn("+10.0%", text)  # median of [1.08, 1.1, 1.12] minus 1
        self.assertIn("40.0", text)
        self.assertIn("44.0", text)
        self.assertIn(BASE, text)
        self.assertIn(HEAD, text)
        self.assertIn("EPYC", text)
        self.assertIn("</details>", text)

    def test_render_verdict_symbols(self) -> None:
        for verdict, symbol in (("faster", "✅"), ("slower", "⚠️"), ("same", "➖"), ("inconclusive", "❔")):
            with self.subTest(verdict=verdict):
                d = make_dict()
                d["perf"][0]["verdict"] = verdict
                self.assertIn(f"{symbol} {verdict}", pr_report.render(pr_report.load(json.dumps(d))))

    def test_render_skipped_sections(self) -> None:
        text = pr_report.render(pr_report.load(json.dumps(skipped_dict())))
        self.assertEqual(sum("skipped" in line for line in text.splitlines()), 3)
        self.assertNotIn("|", text)
        self.assertIn("cold", text)  # the accuracy line says why nothing ran

    def test_render_zero_medians_do_not_divide_by_zero(self) -> None:
        d = make_dict()
        d["perf"][0].update({"base_median": 0.0, "head_median": 0.0, "ratios": [0.0], "verdict": "inconclusive"})
        text = pr_report.render(pr_report.load(json.dumps(d)))
        self.assertIn("❔ inconclusive", text)
        self.assertIn("—", text)
        d["perf"][0].update({"base_median": 0.0, "head_median": 12.0, "ratios": [1.2], "verdict": "faster"})
        row = next(line for line in pr_report.render(pr_report.load(json.dumps(d))).splitlines() if MODEL in line)
        self.assertNotIn("+20.0%", row)  # no base to compare with, so no delta whatever the ratios say

    def test_render_empty_ratios_show_a_dash(self) -> None:
        d = make_dict()
        d["perf"][0]["ratios"] = []
        d["perf"][0]["verdict"] = "inconclusive"
        self.assertIn("—", pr_report.render(pr_report.load(json.dumps(d))))

    def test_render_size_and_memory_deltas(self) -> None:
        text = pr_report.render(make_report())
        size = next(line for line in text.splitlines() if "Size" in line)
        rss = next(line for line in text.splitlines() if "Peak memory" in line)
        self.assertIn("+10,000 B", size)
        self.assertIn("+1.0%", size)
        self.assertIn("-10,000 KiB", rss)
        self.assertIn("-1.2%", rss)
        d = make_dict()
        d["size"] = {"base_bytes": 0, "head_bytes": 5}
        self.assertNotIn(
            "%", next(line for line in pr_report.render(pr_report.load(json.dumps(d))).splitlines() if "Size" in line)
        )

    def test_render_flags_bit_identity_change(self) -> None:
        text = pr_report.render(make_report())  # kernels_changed, and one of two cosines moved
        self.assertIn("❗ bit-identity: 1 cases changed", text)
        self.assertLess(text.index("bit-identity"), text.index("q4_0/capital".replace("_", r"\_").replace("/", r"\/")))

    def test_render_does_not_flag_when_kernels_unchanged_or_identical(self) -> None:
        d = make_dict()
        d["kernels_changed"] = False
        self.assertNotIn("bit-identity: 1 cases", pr_report.render(pr_report.load(json.dumps(d))))
        d = make_dict()
        d["accuracy"]["rows"][1]["head"] = d["accuracy"]["rows"][1]["base"]
        text = pr_report.render(pr_report.load(json.dumps(d)))
        self.assertNotIn("❗", text)
        self.assertIn("bit-identity", text)

    def test_markdown_escapes_pipe_in_key(self) -> None:
        d = make_dict()
        d["accuracy"]["rows"][0]["key"] = "q4_0/a|b `c` <b>@octocat</b> [x](http://e) #1"
        text = pr_report.render(pr_report.load(json.dumps(d)))
        row = next(line for line in text.splitlines() if "octocat" in line)
        # An unescaped pipe would add a column: 4 columns means 5 separators.
        self.assertEqual(len(re.findall(r"(?<!\\)\|", row)), 5)
        self.assertNotIn("<b>", text)
        self.assertNotIn("[x](", text)
        self.assertIn(r"\@octocat", text)
        self.assertNotIn("@octocat</b>", text)

    def test_render_of_maximal_report_fits_a_comment(self) -> None:
        d = make_dict()
        d["perf"] = [copy.deepcopy(d["perf"][0]) for _ in range(pr_report.MAX_PERF)]
        d["perf"][0]["ratios"] = [1.0] * pr_report.MAX_RATIOS
        d["runner"] = "|" * 120
        d["accuracy"]["rows"] = [
            {"key": "|" * pr_report.MAX_KEY, "base": 0.5, "head": 0.25} for _ in range(pr_report.MAX_ROWS)
        ]
        self.assertLess(len(pr_report.render(pr_report.load(json.dumps(d)))), 65_536)


class PostTests(unittest.TestCase):
    def test_post_creates_then_updates_one_comment(self) -> None:
        fake = FakeGitHub()
        number = fake.add_pull_request(HEAD)
        self.assertEqual(pr_report.post(fake, make_report(), HEAD), number)
        d = make_dict()
        d["rss"] = None
        self.assertEqual(pr_report.post(fake, pr_report.load(json.dumps(d)), HEAD), number)
        comments = fake.issues[number].comments
        self.assertEqual(len(comments), 1)
        self.assertIn("skipped", comments[0])

    def test_post_updates_existing_comment(self) -> None:
        fake = FakeGitHub()
        number = fake.add_pull_request(HEAD)
        fake.comment(number, "human remark")
        fake.comment(number, f"{MARKER}\nold report")
        pr_report.post(fake, make_report(), HEAD)
        comments = fake.issues[number].comments
        self.assertEqual(len(comments), 2)
        self.assertEqual(comments[0], "human remark")
        self.assertIn("nibble PR report", comments[1])

    def test_post_ignores_a_marker_planted_by_someone_else(self) -> None:
        fake = FakeGitHub()
        number = fake.add_pull_request(HEAD)
        fake.comment_as(number, "mallory", f"{MARKER}\nplanted")
        pr_report.post(fake, make_report(), HEAD)
        comments = fake.issues[number].comments
        self.assertEqual(comments[0], f"{MARKER}\nplanted")
        self.assertEqual(len(comments), 2)
        self.assertIn("nibble PR report", comments[1])

    def test_post_without_pr_returns_0(self) -> None:
        fake = FakeGitHub()
        self.assertEqual(pr_report.post(fake, make_report(), HEAD), 0)
        self.assertEqual(fake.issues, {})

    def test_post_refuses_mismatched_head_sha(self) -> None:
        # The artifact claims one commit, the triggering run was another: nothing is posted anywhere (R18).
        fake = FakeGitHub()
        mine = fake.add_pull_request(HEAD)
        other = fake.add_pull_request("c" * 40)
        self.assertEqual(pr_report.post(fake, make_report(), "c" * 40), 0)
        self.assertEqual(fake.issues[mine].comments, [])
        self.assertEqual(fake.issues[other].comments, [])

    def test_post_takes_the_pr_from_the_api_not_the_artifact(self) -> None:
        # A forged artifact cannot name a PR: the schema has no such field and the lookup keys on the run's SHA.
        fake = FakeGitHub()
        victim = fake.add_pull_request("c" * 40)
        mine = fake.add_pull_request(HEAD)
        self.assertEqual(pr_report.post(fake, make_report(), HEAD), mine)
        self.assertEqual(fake.issues[victim].comments, [])


class AssembleTests(unittest.TestCase):
    META = {"base_sha": BASE, "head_sha": HEAD, "runner": "runner-1", "kernels_changed": False}

    def test_assemble_merges_parts(self) -> None:
        full = make_dict()
        parts = [
            json.dumps(self.META),
            json.dumps(full["perf"]),  # perf_ab.py summarize writes a bare list
            json.dumps({"size": full["size"], "rss": full["rss"]}),
            json.dumps({"accuracy": full["accuracy"]}),
        ]
        merged = pr_report.load(pr_report.assemble(parts))
        self.assertEqual(merged["perf"], make_report()["perf"])
        self.assertEqual(merged["size"], full["size"])
        self.assertFalse(merged["kernels_changed"])

    def test_assemble_defaults_missing_sections_to_skipped(self) -> None:
        merged = pr_report.load(pr_report.assemble([json.dumps(self.META)]))
        self.assertIsNone(merged["perf"])
        self.assertIsNone(merged["size"])
        self.assertIsNone(merged["rss"])
        self.assertEqual(merged["accuracy"], {"status": "skipped-cold-cache", "rows": []})

    def test_assemble_rejects_conflicts_and_unknown_parts(self) -> None:
        with self.assertRaises(ValueError):
            pr_report.assemble([json.dumps(self.META), json.dumps({"runner": "other"})])
        with self.assertRaises(ValueError):
            pr_report.assemble([json.dumps(self.META), json.dumps({"surprise": 1})])
        with self.assertRaises(ValueError):
            pr_report.assemble([json.dumps({"runner": "x"})])  # no shas
        with self.assertRaises(ValueError):
            pr_report.assemble([json.dumps(self.META), "42"])


class MainTests(unittest.TestCase):
    def run_main(self, argv: list[str], gh: FakeGitHub | None = None) -> tuple[int, str]:
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = pr_report.main(argv, gh=gh)
        return code, err.getvalue()

    def test_assemble_then_post_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            meta, out = Path(tmp, "meta.json"), Path(tmp, "pr-report.json")
            meta.write_text(json.dumps(AssembleTests.META), encoding="utf-8")
            self.assertEqual(self.run_main(["assemble", str(meta), "--out", str(out)])[0], 0)
            fake = FakeGitHub()
            number = fake.add_pull_request(HEAD)
            self.assertEqual(self.run_main(["post", str(out), "--head-sha", HEAD], fake)[0], 0)
            self.assertIn(MARKER, fake.issues[number].comments[0])

    def test_post_with_invalid_report_fails_and_posts_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp, "pr-report.json")
            d = make_dict()
            d["perf"][0]["model"] = "evil"
            bad.write_text(json.dumps(d), encoding="utf-8")
            fake = FakeGitHub()
            number = fake.add_pull_request(HEAD)
            code, err = self.run_main(["post", str(bad), "--head-sha", HEAD], fake)
            self.assertEqual(code, 1)
            self.assertEqual(fake.issues[number].comments, [])
            self.assertNotIn("\n::", "\n" + err)

    def test_post_with_mismatched_sha_exits_0_with_a_notice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "pr-report.json")
            path.write_text(json.dumps(make_dict()), encoding="utf-8")
            fake = FakeGitHub()
            number = fake.add_pull_request("c" * 40)
            code, err = self.run_main(["post", str(path), "--head-sha", "c" * 40], fake)
            self.assertEqual(code, 0)
            self.assertIn("does not match", err)
            self.assertEqual(fake.issues[number].comments, [])

    def test_post_rejects_a_malformed_run_sha(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "pr-report.json")
            path.write_text(json.dumps(make_dict()), encoding="utf-8")
            code, _ = self.run_main(["post", str(path), "--head-sha", "not-a-sha"], FakeGitHub())
            self.assertEqual(code, 1)

    def test_post_missing_file_fails_cleanly(self) -> None:
        code, err = self.run_main(["post", "/nonexistent/pr-report.json", "--head-sha", HEAD], FakeGitHub())
        self.assertEqual(code, 1)
        self.assertIn("No such file", err)


if __name__ == "__main__":
    unittest.main()
