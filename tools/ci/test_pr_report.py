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
from collections.abc import Mapping
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
        "perf_status": "measured",
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
            "head_tests": "passed",
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
    d.update(
        {
            "perf_status": "skipped-cold-cache",
            "perf": None,
            "size": None,
            "rss": None,
            "accuracy": {"status": "skipped-cold-cache", "head_tests": None, "rows": []},
        }
    )
    return d


def speed_cells(text: str) -> list[str]:
    """The cells of the one data row of the speed table: model, test, threads, base, head, delta, verdict."""
    rows = [line for line in text.splitlines() if line.startswith("| ") and "| tg128 |" in line]
    assert len(rows) == 1, rows
    return [cell.strip() for cell in rows[0].strip().strip("|").split("|")]


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

    def test_rejects_mention_and_reference_text(self) -> None:
        # GitHub turns `\\@user` and `\\#1` into a mention and an issue link even though the backslash is consumed by
        # Markdown, so text that reaches the comment may never contain `@` or `#` at all.
        for bad in ("@octocat", "a@b", "see #1", "owner/repo#12", "cpu @ 2.80GHz"):
            with self.subTest(runner=bad):
                d = make_dict()
                d["runner"] = bad
                self.reject(d)
            with self.subTest(key=bad):
                d = make_dict()
                d["accuracy"]["rows"][0]["key"] = bad
                self.reject(d)

    def test_rejects_markup_in_keys_and_runner(self) -> None:
        for bad in ("a|b", "`c`", "<b>x</b>", "[x](http://e)", "a b", "a:b", "*x*", "a\\b", "q4_0/\u00e9"):
            with self.subTest(key=bad):
                d = make_dict()
                d["accuracy"]["rows"][0]["key"] = bad
                self.reject(d)
        for bad in ("a|b", "`c`", "<b>x</b>", "[x](http://e)", "*x*", "a\\b", "a;b", "a=b", "a:b", ":smile:"):
            with self.subTest(runner=bad):
                d = make_dict()
                d["runner"] = bad
                self.reject(d)

    def test_accuracy_rows_must_be_empty_unless_ran(self) -> None:
        d = make_dict()
        d["accuracy"]["status"] = "skipped-cold-cache"
        self.reject(d)
        d = skipped_dict()
        d["accuracy"]["rows"] = [{"key": "q4_0/capital", "base": 1.0, "head": 1.0}]
        self.reject(d)

    def test_head_tests_is_given_exactly_when_the_tests_ran(self) -> None:
        d = make_dict()
        d["accuracy"]["head_tests"] = "failed"
        self.assertEqual(pr_report.load(json.dumps(d))["accuracy"]["head_tests"], "failed")
        for bad in (None, "ok", "", True, 0):
            with self.subTest(head_tests=bad):
                d["accuracy"]["head_tests"] = bad
                self.reject(d)
        d = make_dict()
        del d["accuracy"]["head_tests"]
        self.reject(d)
        for bad in ("passed", "failed", "x"):
            with self.subTest(cold=bad):
                d = skipped_dict()
                d["accuracy"]["head_tests"] = bad
                self.reject(d)

    def test_accuracy_row_may_be_added_or_removed_but_not_both(self) -> None:
        d = make_dict()
        d["accuracy"]["rows"] = [
            {"key": "q4_0/new", "base": None, "head": 0.9},
            {"key": "q4_0/gone", "base": 0.9, "head": None},
        ]
        rows = pr_report.load(json.dumps(d))["accuracy"]["rows"]
        self.assertEqual([(row["base"], row["head"]) for row in rows], [(None, 0.9), (0.9, None)])
        d["accuracy"]["rows"] = [{"key": "q4_0/x", "base": None, "head": None}]
        self.reject(d)
        d["accuracy"]["rows"] = [{"key": "q4_0/x", "base": "nan", "head": None}]
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

    def test_render_reproduction_uses_the_ci_settings(self) -> None:
        # The steps must reproduce what CI compared: portable builds (a plain Release build turns NIBBLE_NATIVE on),
        # both models at 1 and all threads with the same rounds, and accuracy on the base as well as the head.
        text = pr_report.render(make_report())
        block = text[text.index("```sh") : text.index("```", text.index("```sh") + 5)]
        self.assertEqual(block.count("-DNIBBLE_NATIVE=OFF"), 2)
        for model in sorted(pr_report.MODEL_ALLOWLIST):
            self.assertIn(f"models/{model}", block)
        self.assertIn('--threads "1,$(nproc)" --rounds 3 --reps 3', block)
        self.assertIn(f"git worktree add ../nibble-base {BASE}", block)
        self.assertIn("ctest --test-dir ../nibble-base/build -L e2e", block)
        self.assertIn("ctest --test-dir build -L e2e", block)

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
        # ... and that the nightly job only refills master's cache: a PR that changes the hashed model inputs gets no
        # accuracy run before it merges, so the comment must not promise one.
        self.assertIn("changes the model inputs", text)
        self.assertIn("ctest -L e2e", text)

    def test_render_names_the_head_it_describes_in_the_visible_header(self) -> None:
        text = pr_report.render(make_report())
        self.assertIn(f"## nibble PR report for `{HEAD[:7]}`", text)
        self.assertIn(f"<!-- nibble-pr-report-head: {HEAD} -->", text)

    def test_render_a_failed_perf_job_is_failed_not_skipped(self) -> None:
        # A crash in `nibble bench -t 4` or in the RSS run must not read like a cold cache.
        d = skipped_dict()
        d["perf_status"] = "failed"
        d["accuracy"] = make_dict()["accuracy"]
        url = "https://github.com/o/r/actions/runs/42"
        text = pr_report.render(pr_report.load(json.dumps(d)), run_url=url)
        self.assertIn("perf job failed", text)
        self.assertIn(f"]({url})", text)
        self.assertNotIn("skipped", text)
        self.assertIn("perf job failed", pr_report.render(pr_report.load(json.dumps(d))))  # no link without a URL

    def test_render_skipped_names_the_cold_cache(self) -> None:
        text = pr_report.render(pr_report.load(json.dumps(skipped_dict())))
        self.assertIn("**Speed:** skipped (cold model cache)", text)
        self.assertNotIn("failed", text)

    def test_status_and_sections_must_agree(self) -> None:
        # A failed or cold perf job measured nothing, so it carries no numbers.
        for status in ("failed", "skipped-cold-cache"):
            with self.subTest(status=status):
                d = make_dict()
                d["perf_status"] = status
                with self.assertRaises(ValueError):
                    pr_report.load(json.dumps(d))
        d = make_dict()
        d["perf_status"] = "unknown"
        with self.assertRaises(ValueError):
            pr_report.load(json.dumps(d))

    def test_render_zero_medians_do_not_divide_by_zero(self) -> None:
        d = make_dict()
        d["perf"][0].update({"base_median": 0.0, "head_median": 0.0, "ratios": [0.0], "verdict": "inconclusive"})
        cells = speed_cells(pr_report.render(pr_report.load(json.dumps(d))))
        # tok/s columns and the delta are all dashes; the verdict is kept as given.
        self.assertEqual(cells[3:], ["—", "—", "—", "❔ inconclusive"])
        d["perf"][0].update({"base_median": 0.0, "head_median": 12.0, "ratios": [1.2], "verdict": "faster"})
        cells = speed_cells(pr_report.render(pr_report.load(json.dumps(d))))
        # No base median, so no delta whatever the ratios say.
        self.assertEqual(cells[3:], ["—", "12.0", "—", "✅ faster"])

    def test_render_empty_ratios_show_a_dash(self) -> None:
        d = make_dict()
        d["perf"][0]["ratios"] = []
        d["perf"][0]["verdict"] = "inconclusive"
        cells = speed_cells(pr_report.render(pr_report.load(json.dumps(d))))
        self.assertEqual(cells[3:], ["40.0", "44.0", "—", "❔ inconclusive"])

    def test_render_delta_ignores_zero_ratios(self) -> None:
        d = make_dict()
        d["perf"][0]["ratios"] = [0.0, 1.2, 1.4]  # 0.0 marks an unusable block (P16)
        cells = speed_cells(pr_report.render(pr_report.load(json.dumps(d))))
        self.assertEqual(cells[5], "+30.0%")

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

    def test_render_flags_a_changed_printed_min_cosine(self) -> None:
        text = pr_report.render(make_report())  # kernels_changed, and one of two cosines moved
        self.assertIn("❗ printed min cosine changed in 1 of 2 cases", text)
        # Equal values printed to six digits do not show identical logits, so the comment must not claim that.
        self.assertNotIn("bit-identi", text)
        self.assertLess(
            text.index("printed min cosine"), text.index("q4_0/capital".replace("_", r"\_").replace("/", r"\/"))
        )

    def test_render_does_not_flag_when_kernels_unchanged_or_identical(self) -> None:
        d = make_dict()
        d["kernels_changed"] = False
        self.assertNotIn("printed min cosine", pr_report.render(pr_report.load(json.dumps(d))))
        d = make_dict()
        d["accuracy"]["rows"][1]["head"] = d["accuracy"]["rows"][1]["base"]
        text = pr_report.render(pr_report.load(json.dumps(d)))
        self.assertNotIn("❗", text)
        self.assertIn("✅ printed min cosine unchanged in all 2 cases", text)
        self.assertNotIn("bit-identi", text)

    def test_render_says_when_the_head_failed_its_end_to_end_tests(self) -> None:
        # The cosine rows can be identical to the base's while a later check of the same test (batch prefill,
        # greedy continuation) failed: the comment must say so on its own, not only through the red job.
        d = make_dict()
        d["accuracy"]["rows"] = [{"key": "q4_0/capital", "base": 0.99, "head": 0.99}]
        d["accuracy"]["head_tests"] = "failed"
        text = pr_report.render(pr_report.load(json.dumps(d)))
        self.assertIn("❌ end-to-end tests failed on the head", text)
        self.assertLess(text.index("end-to-end tests failed"), text.index("| case |"))
        d["accuracy"]["rows"] = []
        self.assertIn("❌ end-to-end tests failed on the head", pr_report.render(pr_report.load(json.dumps(d))))
        passed = pr_report.render(make_report())
        self.assertNotIn("❌", passed)
        self.assertIn("end-to-end tests passed on the head", passed)
        self.assertNotIn("end-to-end tests", pr_report.render(pr_report.load(json.dumps(skipped_dict()))))

    def test_markdown_escapes_pipe_in_key(self) -> None:
        # `load` refuses such keys; `render` still escapes them (defense in depth for a Report built by hand).
        r = make_report()
        r["accuracy"]["rows"][0]["key"] = "q4_0/a|b `c` <b>@octocat</b> [x](http://e) #1"
        text = pr_report.render(r)
        row = next(line for line in text.splitlines() if "octocat" in line)
        # An unescaped pipe would add a column: 4 columns means 5 separators.
        self.assertEqual(len(re.findall(r"(?<!\\)\|", row)), 5)
        self.assertNotIn("<b>", text)
        self.assertNotIn("[x](", text)

    def test_render_of_a_loaded_report_has_no_mention_or_reference(self) -> None:
        text = pr_report.render(
            pr_report.load(
                pr_report.assemble(
                    [json.dumps(AssembleTests.META | AssembleTests.REAL), json.dumps(AssembleTests.COLD)]
                )
            )
        )
        self.assertIsNone(re.search(r"[@#]\w", text), text)

    def test_render_one_sided_cases_say_which_result_is_missing(self) -> None:
        d = make_dict()
        d["accuracy"]["rows"] = [
            {"key": "q4_0/new", "base": None, "head": 0.9},
            {"key": "q4_0/gone", "base": 0.8, "head": None},
        ]
        text = pr_report.render(pr_report.load(json.dumps(d)))
        rows = {
            line.split("|")[1].strip(): [c.strip() for c in line.split("|")[2:-1]]
            for line in text.splitlines()
            if line.startswith("| q4")
        }
        self.assertEqual(rows[r"q4\_0\/new"], ["—", "0.9", "no base result"])
        self.assertEqual(rows[r"q4\_0\/gone"], ["0.8", "—", "no head result"])
        self.assertIn("❗ printed min cosine changed in 2 of 2 cases", text)  # kernels_changed is true in the fixture

    def test_render_of_maximal_report_fits_a_comment(self) -> None:
        # Built without `load` so that every character is one the escaper doubles: the worst case for the size.
        r = make_report()
        r["perf"] = [copy.deepcopy(perf_of(r)[0]) for _ in range(pr_report.MAX_PERF)]
        perf_of(r)[0]["ratios"] = [1.0] * pr_report.MAX_RATIOS
        r["runner"] = "|" * pr_report.MAX_RUNNER
        r["accuracy"]["rows"] = [
            {"key": "|" * pr_report.MAX_KEY, "base": 0.5, "head": 0.25} for _ in range(pr_report.MAX_ROWS)
        ]
        self.assertLess(len(pr_report.render(r)), 65_536)


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
        self.assertIn("**Peak memory:** not measured", comments[0])  # measured job, section missing: not "skipped"

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

    def test_post_links_the_run_when_given(self) -> None:
        fake = FakeGitHub()
        number = fake.add_pull_request(HEAD)
        d = skipped_dict()
        d["perf_status"] = "failed"
        url = "https://github.com/o/r/actions/runs/7"
        pr_report.post(fake, pr_report.load(json.dumps(d)), HEAD, run_url=url)
        self.assertIn(url, fake.issues[number].comments[0])

    def test_post_takes_the_pr_from_the_api_not_the_artifact(self) -> None:
        # A forged artifact cannot name a PR: the schema has no such field and the lookup keys on the run's SHA.
        fake = FakeGitHub()
        victim = fake.add_pull_request("c" * 40)
        mine = fake.add_pull_request(HEAD)
        self.assertEqual(pr_report.post(fake, make_report(), HEAD), mine)
        self.assertEqual(fake.issues[victim].comments, [])


NEW = "c" * 40
RUN_URL = "https://github.com/o/r/actions/runs/99"


class OutdatedTests(unittest.TestCase):
    """A pr-report run for a newer head that left no report: the older comment must say it is outdated."""

    def setup_pr(self, head: str = NEW) -> tuple[FakeGitHub, int]:
        """An open pull request now at `head` whose bot comment is the report for HEAD."""
        fake = FakeGitHub()
        number = fake.add_pull_request(head)
        fake.comment(number, pr_report.render(make_report()))
        return fake, number

    def test_marks_the_older_report_outdated_with_the_new_head_and_the_run(self) -> None:
        fake, number = self.setup_pr()
        self.assertEqual(pr_report.mark_outdated(fake, NEW, RUN_URL), number)
        (body,) = fake.issues[number].comments
        self.assertTrue(body.startswith(MARKER))
        banner = body[: body.index("## nibble PR report")]
        self.assertIn("Outdated", banner)
        self.assertIn(f"`{NEW[:7]}`", banner)
        self.assertIn(f"`{HEAD[:7]}`", banner)
        self.assertIn(f"]({RUN_URL})", banner)
        self.assertIn("✅ end-to-end tests passed on the head", body)  # the old numbers stay, under the banner

    def test_marking_twice_keeps_one_banner(self) -> None:
        fake, number = self.setup_pr()
        pr_report.mark_outdated(fake, NEW, RUN_URL)
        pr_report.mark_outdated(fake, NEW, "https://github.com/o/r/actions/runs/100")
        (body,) = fake.issues[number].comments
        self.assertEqual(body.count("Outdated"), 1)
        self.assertIn("runs/100", body)
        self.assertNotIn("runs/99", body)

    def test_a_new_report_replaces_the_banner(self) -> None:
        fake, number = self.setup_pr(HEAD)
        fake.prs_by_sha[NEW] = number
        pr_report.mark_outdated(fake, NEW, RUN_URL)
        pr_report.post(fake, make_report(), HEAD)
        self.assertNotIn("Outdated", fake.issues[number].comments[0])

    def test_a_report_for_the_same_head_is_not_outdated(self) -> None:
        fake, number = self.setup_pr(HEAD)
        before = list(fake.issues[number].comments)
        self.assertEqual(pr_report.mark_outdated(fake, HEAD, RUN_URL), 0)
        self.assertEqual(fake.issues[number].comments, before)

    def test_nothing_to_mark_without_a_pull_request_or_a_bot_comment(self) -> None:
        fake = FakeGitHub()
        self.assertEqual(pr_report.mark_outdated(fake, NEW, RUN_URL), 0)  # a superseded head: no open PR at it
        number = fake.add_pull_request(NEW)
        fake.comment_as(number, "mallory", f"{MARKER}\nplanted")
        self.assertEqual(pr_report.mark_outdated(fake, NEW, RUN_URL), 0)
        self.assertEqual(fake.issues[number].comments, [f"{MARKER}\nplanted"])

    def test_a_report_without_a_head_marker_reads_as_an_earlier_commit(self) -> None:
        fake = FakeGitHub()
        number = fake.add_pull_request(NEW)
        fake.comment(number, f"{MARKER}\n## nibble PR report\nold numbers")
        self.assertEqual(pr_report.mark_outdated(fake, NEW, RUN_URL), number)
        body = fake.issues[number].comments[0]
        self.assertIn("an earlier commit", body)
        self.assertIn("old numbers", body)

    def test_cli_outdated_validates_its_inputs(self) -> None:
        fake, number = self.setup_pr()
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(pr_report.main(["outdated", "--head-sha", NEW, "--run-url", RUN_URL], gh=fake), 0)
            self.assertEqual(pr_report.main(["outdated", "--head-sha", "xyz", "--run-url", RUN_URL], gh=fake), 1)
            bad_url = "https://evil.example/o/r/actions/runs/1) [x](y"
            self.assertEqual(pr_report.main(["outdated", "--head-sha", NEW, "--run-url", bad_url], gh=fake), 1)
        self.assertIn("Outdated", fake.issues[number].comments[0])


class AssembleTests(unittest.TestCase):
    META = {"base_sha": BASE, "head_sha": HEAD, "runner": "runner-1", "kernels_changed": False}
    REAL = {"runner": "Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz"}  # what /proc/cpuinfo says on a runner
    ACCURACY = {
        "accuracy": {
            "status": "ran",
            "head_tests": "passed",
            "rows": [{"key": "q4_0/capital", "base": 1.0, "head": 1.0}],
        }
    }
    COLD: dict[str, Any] = {"accuracy": {"status": "skipped-cold-cache", "head_tests": None, "rows": []}}

    def test_assemble_merges_parts(self) -> None:
        full = make_dict()
        parts = [
            json.dumps(self.META),
            json.dumps(full["perf"]),  # perf_ab.py summarize writes a bare list
            json.dumps({"perf_status": "measured", "size": full["size"], "rss": full["rss"]}),
            json.dumps({"accuracy": full["accuracy"]}),
        ]
        merged = pr_report.load(pr_report.assemble(parts))
        self.assertEqual(merged["perf"], make_report()["perf"])
        self.assertEqual(merged["size"], full["size"])
        self.assertFalse(merged["kernels_changed"])

    def test_assemble_without_a_perf_part_reads_as_a_failed_perf_job(self) -> None:
        # The cold cache is an explicit part ({"perf_status": "skipped-cold-cache"}); no part at all means the perf
        # job died before writing one.
        merged = pr_report.load(pr_report.assemble([json.dumps(self.META), json.dumps(self.COLD)]))
        self.assertEqual(merged["perf_status"], "failed")
        self.assertIsNone(merged["perf"])
        self.assertIsNone(merged["size"])
        self.assertIsNone(merged["rss"])
        self.assertEqual(merged["accuracy"], {"status": "skipped-cold-cache", "head_tests": None, "rows": []})

    def test_assemble_requires_the_accuracy_part(self) -> None:
        # A crashed or cancelled accuracy job must not read as "cold cache": the cold case is an explicit status.
        with self.assertRaisesRegex(ValueError, "accuracy"):
            pr_report.assemble([json.dumps(self.META)])

    def test_assemble_refuses_a_failed_accuracy_attempt(self) -> None:
        # The accuracy job uploads this marker when it could not write a real part, so that it replaces an earlier
        # attempt's part-accuracy artifact: a re-run must never assemble a report from that stale head result.
        failed = pr_report.accuracy_failed_part()
        # pr-report.yml writes this exact text with plain shell (it must work before Python is set up).
        self.assertEqual(failed, '{"accuracy_failed": true}')
        with self.assertRaisesRegex(ValueError, "accuracy job failed"):
            pr_report.assemble([json.dumps(self.META), failed])
        with self.assertRaisesRegex(ValueError, "accuracy job failed"):
            pr_report.assemble([json.dumps(self.META), json.dumps(self.ACCURACY), failed])
        # It is not a report section either: load refuses it.
        d = make_dict()
        d["accuracy_failed"] = True
        with self.assertRaises(ValueError):
            pr_report.load(json.dumps(d))

    def test_assemble_requires_kernels_changed(self) -> None:
        meta = {k: v for k, v in self.META.items() if k != "kernels_changed"}
        with self.assertRaisesRegex(ValueError, "kernels_changed"):
            pr_report.assemble([json.dumps(meta), json.dumps(self.COLD)])

    def test_assemble_makes_a_real_runner_name_inert(self) -> None:
        merged = pr_report.load(pr_report.assemble([json.dumps(self.META | self.REAL), json.dumps(self.COLD)]))
        self.assertEqual(merged["runner"], "Intel(R) Xeon(R) Platinum 8370C CPU at 2.80GHz")

    def test_assemble_replaces_colons_in_the_runner_name(self) -> None:
        # ":smile:" would render as an emoji, and no CPU model name needs a colon.
        merged = pr_report.load(
            pr_report.assemble([json.dumps(self.META | {"runner": "cpu :smile: x"}), json.dumps(self.COLD)])
        )
        self.assertEqual(merged["runner"], "cpu _smile_ x")

    def test_assemble_replaces_other_unsafe_runner_characters_and_truncates(self) -> None:
        runner = "a#b <i>|\u00e9" + "x" * 200
        merged = pr_report.load(pr_report.assemble([json.dumps(self.META | {"runner": runner}), json.dumps(self.COLD)]))
        self.assertTrue(merged["runner"].startswith("a_b _i__"))
        self.assertEqual(len(merged["runner"]), pr_report.MAX_RUNNER)

    def test_assemble_rejects_conflicts_and_unknown_parts(self) -> None:
        with self.assertRaises(ValueError):
            pr_report.assemble([json.dumps(self.META), json.dumps(self.COLD), json.dumps({"runner": "other"})])
        with self.assertRaises(ValueError):
            pr_report.assemble([json.dumps(self.META), json.dumps(self.COLD), json.dumps({"surprise": 1})])
        with self.assertRaises(ValueError):
            pr_report.assemble([json.dumps({"runner": "x"}), json.dumps(self.COLD)])  # no shas
        with self.assertRaises(ValueError):
            pr_report.assemble([json.dumps(self.META), json.dumps(self.COLD), "42"])


class MainTests(unittest.TestCase):
    def run_main(self, argv: list[str], gh: FakeGitHub | None = None) -> tuple[int, str]:
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = pr_report.main(argv, gh=gh)
        return code, err.getvalue()

    def test_assemble_then_post_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            meta, out = Path(tmp, "meta.json"), Path(tmp, "pr-report.json")
            acc = Path(tmp, "part-accuracy.json")
            meta.write_text(json.dumps(AssembleTests.META), encoding="utf-8")
            self.assertEqual(self.run_main(["accuracy", "--skipped-cold-cache", "--out", str(acc)])[0], 0)
            self.assertEqual(self.run_main(["assemble", str(meta), str(acc), "--out", str(out)])[0], 0)
            fake = FakeGitHub()
            number = fake.add_pull_request(HEAD)
            self.assertEqual(self.run_main(["post", str(out), "--head-sha", HEAD], fake)[0], 0)
            self.assertIn(MARKER, fake.issues[number].comments[0])

    def snapshot(self, tmp: str, name: str, cosines: Mapping[str, float | str | None]) -> str:
        """Write a cosine snapshot in the format the nightly job uploads as `e2e-cosines.json`; return its path."""
        data = {
            key: {"model": key.split("/")[0], "prompt": key.split("/")[1], "min_cos": v, "positions": 8, "agree": 8}
            for key, v in cosines.items()
        }
        path = Path(tmp, name)
        path.write_text(json.dumps(data), encoding="utf-8")
        return str(path)

    def test_accuracy_subcommand_builds_rows_from_two_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = self.snapshot(tmp, "base.json", {"q4_0/a": 0.99, "q4_0/gone": 0.5, "q8_0/a": 1.0})
            head = self.snapshot(tmp, "head.json", {"q4_0/a": 0.98, "q4_0/new": 0.7, "q8_0/a": 1.0})
            out = Path(tmp, "part-accuracy.json")
            argv = ["accuracy", "--base", base, "--head", head, "--head-outcome", "success", "--out", str(out)]
            self.assertEqual(self.run_main(argv)[0], 0)
            self.assertEqual(
                json.loads(out.read_text(encoding="utf-8")),
                {
                    "accuracy": {
                        "status": "ran",
                        "head_tests": "passed",
                        "rows": [
                            {"key": "q4_0/a", "base": 0.99, "head": 0.98},
                            {"key": "q4_0/gone", "base": 0.5, "head": None},
                            {"key": "q4_0/new", "base": None, "head": 0.7},
                            {"key": "q8_0/a", "base": 1.0, "head": 1.0},
                        ],
                    }
                },
            )

    def test_accuracy_subcommand_records_the_head_test_outcome(self) -> None:
        # Only the step outcome `success` reads as passed; failure, and anything unexpected, reads as failed.
        with tempfile.TemporaryDirectory() as tmp:
            snap = self.snapshot(tmp, "s.json", {"q4_0/a": 0.99})
            out = Path(tmp, "part.json")
            for outcome, want in (("success", "passed"), ("failure", "failed"), ("cancelled", "failed")):
                with self.subTest(outcome=outcome):
                    argv = ["accuracy", "--base", snap, "--head", snap, "--head-outcome", outcome, "--out", str(out)]
                    self.assertEqual(self.run_main(argv)[0], 0)
                    self.assertEqual(json.loads(out.read_text(encoding="utf-8"))["accuracy"]["head_tests"], want)
            for argv in (
                ["accuracy", "--base", snap, "--head", snap, "--out", str(out)],  # no outcome
                ["accuracy", "--skipped-cold-cache", "--head-outcome", "success", "--out", str(out)],
                ["accuracy", "--base", snap, "--head", snap, "--head-outcome", "maybe", "--out", str(out)],
            ):
                with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    pr_report.main(argv)

    def test_accuracy_subcommand_failed_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp, "part-accuracy.json")
            self.assertEqual(self.run_main(["accuracy", "--failed", "--out", str(out)])[0], 0)
            self.assertEqual(json.loads(out.read_text(encoding="utf-8")), {"accuracy_failed": True})
            snap = self.snapshot(tmp, "s.json", {"q4_0/a": 0.99})
            for args in (
                ["--failed", "--skipped-cold-cache"],
                ["--failed", "--base", snap, "--head", snap, "--head-outcome", "success"],
            ):
                with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    pr_report.main(["accuracy", *args, "--out", str(out)])

    def test_accuracy_subcommand_skipped_cold_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp, "part-accuracy.json")
            self.assertEqual(self.run_main(["accuracy", "--skipped-cold-cache", "--out", str(out)])[0], 0)
            self.assertEqual(
                json.loads(out.read_text(encoding="utf-8")),
                {"accuracy": {"status": "skipped-cold-cache", "head_tests": None, "rows": []}},
            )

    def test_accuracy_subcommand_drops_non_finite_cosines_instead_of_failing(self) -> None:
        # A head whose kernel produces NaN must still get a report (the case shows as "no head result", or "no
        # base result" for a NaN on the base side); the workflow's own strict `accuracy.py snapshot` gate is what
        # turns the job red.
        with tempfile.TemporaryDirectory() as tmp:
            good = self.snapshot(tmp, "good.json", {"q4_0/a": 0.99, "q4_0/b": 0.9})
            out = Path(tmp, "part.json")
            for name, value in (("nan.json", "nan"), ("inf.json", "-inf")):
                with self.subTest(name=name):
                    bad = self.snapshot(tmp, name, {"q4_0/a": value, "q4_0/b": 0.9})
                    self.assertEqual(
                        self.run_main(
                            ["accuracy", "--base", good, "--head", bad, "--head-outcome", "success", "--out", str(out)]
                        )[0],
                        0,
                    )
                    rows = json.loads(out.read_text(encoding="utf-8"))["accuracy"]["rows"]
                    self.assertEqual(rows[0], {"key": "q4_0/a", "base": 0.99, "head": None})
                    self.assertEqual(rows[1], {"key": "q4_0/b", "base": 0.9, "head": 0.9})
                    self.assertEqual(
                        self.run_main(
                            ["accuracy", "--base", bad, "--head", good, "--head-outcome", "failure", "--out", str(out)]
                        )[0],
                        0,
                    )
                    rows = json.loads(out.read_text(encoding="utf-8"))["accuracy"]["rows"]
                    self.assertEqual(rows[0], {"key": "q4_0/a", "base": None, "head": 0.99})

    def test_accuracy_subcommand_refuses_bad_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            good = self.snapshot(tmp, "good.json", {"q4_0/a": 0.99})
            out = str(Path(tmp, "part.json"))
            for name, cosines in (("junk.json", {"q4_0/a": "junk"}), ("null.json", {"q4_0/a": None})):
                with self.subTest(name=name):
                    bad = self.snapshot(tmp, name, cosines)
                    code, err = self.run_main(
                        ["accuracy", "--base", good, "--head", bad, "--head-outcome", "success", "--out", out]
                    )
                    self.assertEqual(code, 1)
                    self.assertIn("q4_0/a", err)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as ctx:
                pr_report.main(["accuracy", "--head", good, "--out", out])  # --base missing
            self.assertEqual(ctx.exception.code, 2)
            conflicting: tuple[list[str], ...] = (["--skipped-cold-cache", "--base", good], [])  # and no option at all
            for args in (*conflicting, ["--skipped-cold-cache", "--base", good, "--head", good]):
                with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    pr_report.main(["accuracy", *args, "--out", out])
            self.assertFalse(Path(out).exists())

    def test_files_over_the_size_limit_are_refused_before_decoding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            big = Path(tmp, "pr-report.json")
            big.write_bytes(b" " * (pr_report.MAX_BYTES + 1))
            fake = FakeGitHub()
            code, err = self.run_main(["post", str(big), "--head-sha", HEAD], fake)
            self.assertEqual(code, 1)
            self.assertIn("larger than", err)
            with self.assertRaises(ValueError):
                pr_report.read_limited(str(big))

    def test_read_limited_reads_at_most_the_limit_plus_one_byte(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "f.json")
            path.write_text("x" * 10, encoding="utf-8")
            self.assertEqual(pr_report.read_limited(str(path)), "x" * 10)
        # An endless file must give up after the limit instead of filling memory.
        with self.assertRaises(ValueError):
            pr_report.read_limited("/dev/zero")

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
