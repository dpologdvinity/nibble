"""Unit tests for tools/ci/dashboard_render.py: the static dashboard site built from the data branch's rows.

Run with: python -m unittest discover -s tools/ci -p 'test_*.py'
Rows are built through dashboard_data.validate, so they have the real schema. Nothing here touches the network; the
site goes to a temporary directory.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import dashboard_render
from dashboard_data import METRICS, Row, validate

REPO = "dpologdvinity/nibble"
SHA_A = "a" * 40
SHA_B = "b" * 40
FILES = [
    "index.html",
    "speed-prefill.svg",
    "speed-decode.svg",
    "accuracy.svg",
    "fuzz.svg",
    "size.svg",
    "coverage.svg",
    "badge.json",
]


def row(
    metric: str,
    key: str,
    value: float,
    date: str = "2026-11-21T03:17:00Z",
    run_id: int = 1,
    source: str = "nightly",
    commit: str = SHA_A,
) -> Row:
    """One validated row; the unit comes from the metric."""
    return validate(
        {
            "schema": 1,
            "date": date,
            "commit": commit,
            "source": source,
            "metric": metric,
            "key": key,
            "value": value,
            "unit": METRICS[metric],
            "runner": "Test CPU",
            "run_id": run_id,
        }
    )


def jsonl(rows: list[Row]) -> str:
    return "".join(json.dumps(r) + "\n" for r in rows)


def status(job: str, ok: bool, date: str = "2026-11-21T03:17:00Z", run_id: int = 1, workflow: str = "nightly") -> Row:
    return row("job_status", f"{workflow}/{job}", 1.0 if ok else 0.0, date, run_id, source=workflow)


def render(case: unittest.TestCase, rows: list[Row]) -> tuple[Path, list[str]]:
    """Render `rows` into a temporary directory that `case` removes when the test ends; returns it and the names."""
    out = Path(case.enterContext(tempfile.TemporaryDirectory(prefix="dash-test-"))) / "site"
    return out, dashboard_render.render_site(jsonl(rows), str(out), REPO)


def sample_rows() -> list[Row]:
    return [
        row("bench_tok_s", "qwen2.5-0.5b-instruct-q4_0.gguf/tg128/t1", 20.0, run_id=1),
        row("bench_tok_s", "qwen2.5-0.5b-instruct-q4_0.gguf/pp512/t4", 150.0, run_id=1),
        row("min_cosine", "q4_0/capital", 0.9995, run_id=1),
        row("fuzz_seconds", "fuzz_gguf", 7200.0, run_id=1),
        row("binary_bytes", "nibble", 1500000.0, run_id=1),
        row("line_coverage", "src+cli", 85.3, run_id=2, source="ci", commit=SHA_B, date="2026-11-21T05:00:00Z"),
        status("accuracy", True, run_id=1),
        status("fuzz_gguf", True, run_id=1),
        status("arm64", False, run_id=3, workflow="portability", date="2026-11-21T04:30:00Z"),
    ]


class RenderTest(unittest.TestCase):
    def test_render_writes_all_files(self) -> None:
        out, names = render(self, sample_rows())
        self.assertEqual(sorted(names), sorted(FILES))
        for name in FILES:
            self.assertTrue((out / name).is_file(), name)
        html = (out / "index.html").read_text(encoding="utf-8")
        self.assertNotIn("<script", html.lower())
        imgs = re.findall(r"<img\b[^>]*>", html)
        self.assertEqual(len(imgs), 6)
        for img in imgs:
            self.assertRegex(img, r'alt="[^"]{10,}"')
        for name in FILES[1:7]:
            self.assertIn(f'src="{name}"', html)
        for name in FILES[1:7]:
            self.assertTrue((out / name).read_text(encoding="utf-8").startswith("<svg"))

    def test_render_with_no_rows_still_writes_a_page(self) -> None:
        out, names = render(self, [])
        self.assertEqual(sorted(names), sorted(FILES))
        html = (out / "index.html").read_text(encoding="utf-8")
        self.assertIn("No results yet", html)
        self.assertIn("no data yet", (out / "speed-decode.svg").read_text(encoding="utf-8"))
        self.assertEqual(
            json.loads((out / "badge.json").read_text(encoding="utf-8")),
            {"schemaVersion": 1, "label": "nightly", "message": "no data", "color": "lightgrey"},
        )

    def test_status_table_uses_latest_row_per_key(self) -> None:
        rows = [
            status("accuracy", False, "2026-11-19T03:17:00Z", run_id=10),
            status("accuracy", True, "2026-11-21T03:17:00Z", run_id=12),
            status("accuracy", False, "2026-11-20T03:17:00Z", run_id=11),
        ]
        out, _ = render(self, rows)
        html = (out / "index.html").read_text(encoding="utf-8")
        self.assertEqual(html.count('<th scope="row">accuracy</th>'), 1)
        self.assertIn(f"https://github.com/{REPO}/actions/runs/12", html)
        self.assertNotIn("/actions/runs/10", html)
        self.assertNotIn("/actions/runs/11", html)
        self.assertIn("✅", html)
        self.assertNotIn("❌", html)
        self.assertIn("2026-11-21", html)

    def test_status_table_groups_by_workflow_and_marks_failures(self) -> None:
        out, _ = render(self, sample_rows())
        html = (out / "index.html").read_text(encoding="utf-8")
        self.assertLess(html.index("nightly"), html.index("portability"))
        self.assertIn("❌", html)  # the failing arm64 job
        self.assertRegex(html, r"<th scope=\"row\">arm64</th>\s*<td>❌ failing</td>")
        self.assertIn("<caption>", html)

    def test_header_names_the_newest_row_and_its_commit(self) -> None:
        out, _ = render(self, sample_rows())
        html = (out / "index.html").read_text(encoding="utf-8")
        self.assertIn("2026-11-21 05:00 UTC", html)  # the coverage row is the newest
        self.assertIn(f"https://github.com/{REPO}/commit/{SHA_B}", html)
        self.assertIn(SHA_B[:7], html)
        self.assertIn(f"https://github.com/{REPO}", html)
        self.assertIn("README", html)

    def test_page_is_honest_about_runner_speed(self) -> None:
        out, _ = render(self, sample_rows())
        html = (out / "index.html").read_text(encoding="utf-8")
        self.assertIn("trend", html)
        self.assertIn("not a benchmark", html)

    def test_badge_failing_when_any_latest_nightly_status_zero(self) -> None:
        out, _ = render(self, [status("accuracy", True, run_id=1), status("fuzz_gguf", False, run_id=1)])
        badge = json.loads((out / "badge.json").read_text(encoding="utf-8"))
        self.assertEqual(badge, {"schemaVersion": 1, "label": "nightly", "message": "failing", "color": "red"})

    def test_badge_passing_when_every_latest_nightly_status_is_one(self) -> None:
        rows = [
            status("accuracy", False, "2026-11-20T03:17:00Z", run_id=1),  # an old failure that a later run fixed
            status("accuracy", True, "2026-11-21T03:17:00Z", run_id=2),
            status("fuzz_gguf", True, run_id=2),
        ]
        out, _ = render(self, rows)
        badge = json.loads((out / "badge.json").read_text(encoding="utf-8"))
        self.assertEqual(badge, {"schemaVersion": 1, "label": "nightly", "message": "passing", "color": "brightgreen"})

    def test_badge_ignores_other_workflows(self) -> None:
        rows = [status("accuracy", True), status("arm64", False, workflow="portability")]
        out, _ = render(self, rows)
        badge = json.loads((out / "badge.json").read_text(encoding="utf-8"))
        self.assertEqual(badge["message"], "passing")

    def test_badge_has_no_data_without_nightly_status(self) -> None:
        out, _ = render(self, [status("arm64", True, workflow="portability")])
        badge = json.loads((out / "badge.json").read_text(encoding="utf-8"))
        self.assertEqual(badge["message"], "no data")
        self.assertEqual(badge["color"], "lightgrey")

    def test_cumulative_fuzz_hours(self) -> None:
        rows = [
            row("fuzz_seconds", "fuzz_gguf", 7200.0, "2026-11-21T03:17:00Z", run_id=2),
            row("fuzz_seconds", "fuzz_gguf", 7200.0, "2026-11-20T03:17:00Z", run_id=1),
            row("fuzz_seconds", "fuzz_load", 3600.0, "2026-11-20T03:17:00Z", run_id=1),
        ]
        series = dashboard_render.cumulative_fuzz_hours(rows)
        self.assertEqual([v for _, v in series["fuzz_gguf"]], [2.0, 4.0])
        self.assertLess(series["fuzz_gguf"][0][0], series["fuzz_gguf"][1][0])
        self.assertEqual([v for _, v in series["fuzz_load"]], [1.0])

    def test_accuracy_is_the_minimum_over_prompts_per_run(self) -> None:
        rows = [
            row("min_cosine", "q4_0/capital", 0.9995, run_id=1),
            row("min_cosine", "q4_0/chat", 0.9981, run_id=1),
            row("min_cosine", "q8_0/capital", 0.9999, run_id=1),
        ]
        got = {m: [v for _, v in pts] for m, pts in dashboard_render.min_cosine_by_model(rows).items()}
        self.assertEqual(got, {"q4_0": [0.9981], "q8_0": [0.9999]})

    @staticmethod
    def y_labels(svg: str) -> list[str]:
        """The y axis tick labels of a chart, bottom to top (the text elements right-aligned at the axis)."""
        return re.findall(r'text-anchor="end"[^>]*>([^<]*)</text>', svg)

    def test_coverage_axis_never_exceeds_100_percent(self) -> None:
        # Regression: the default axis (nice_max of 1.1 x the data) ran 0-200 for a 96.3 % value on the live page.
        rows = [row("line_coverage", "src+cli", 96.3, run_id=2, source="ci", commit=SHA_B)]
        out, _ = render(self, rows)
        labels = [float(v) for v in self.y_labels((out / "coverage.svg").read_text(encoding="utf-8"))]
        self.assertEqual(labels[-1], 100.0, labels)
        self.assertLessEqual(max(labels), 100.0, labels)
        self.assertEqual(labels[0], 0.0, labels)

    def test_accuracy_axis_still_tops_out_at_1(self) -> None:
        out, _ = render(self, [row("min_cosine", "q4_0/capital", 0.9995)])
        labels = self.y_labels((out / "accuracy.svg").read_text(encoding="utf-8"))
        self.assertEqual(labels[0], "0.990")
        self.assertEqual(labels[-1], "1.000")

    def test_html_escapes_keys(self) -> None:
        evil = "<script>alert(1)</script>"
        rows = [
            status(evil, True),
            row("bench_tok_s", f"{evil}/tg128/t1", 5.0),
            row("fuzz_seconds", evil, 60.0),
        ]
        out, _ = render(self, rows)
        for name in ("index.html", "speed-prefill.svg", "speed-decode.svg", "fuzz.svg"):
            text = (out / name).read_text(encoding="utf-8")
            self.assertNotIn("<script>", text, name)
            self.assertNotIn("<SCRIPT>", text, name)  # the legend upper-cases the quantization part of the label
        self.assertIn("&lt;SCRIPT&gt;", (out / "speed-decode.svg").read_text(encoding="utf-8"))
        self.assertIn("&lt;script&gt;", (out / "index.html").read_text(encoding="utf-8"))

    def test_prefill_and_decode_are_separate_charts(self) -> None:
        out, _ = render(self, sample_rows())  # holds a tg128/t1 and a pp512/t4 series of the same model
        prefill = (out / "speed-prefill.svg").read_text(encoding="utf-8")
        decode = (out / "speed-decode.svg").read_text(encoding="utf-8")
        self.assertIn("Prefill speed (512-token prompt)", prefill)
        self.assertIn("Decode speed (128 generated tokens)", decode)  # the sizes come from the tests plotted
        self.assertIn(">Q4_0, 4 threads<", prefill)
        self.assertNotIn(">Q4_0, 1 thread<", prefill)
        self.assertIn(">Q4_0, 1 thread<", decode)
        self.assertNotIn(">Q4_0, 4 threads<", decode)
        html = (out / "index.html").read_text(encoding="utf-8")
        self.assertEqual(html.count("<h2>Speed (trend only)</h2>"), 1)
        self.assertLess(html.index("speed-prefill.svg"), html.index("speed-decode.svg"))
        self.assertLess(html.index("speed-decode.svg"), html.index("<h2>Accuracy"))

    def test_speed_legend_labels_are_readable_and_sorted(self) -> None:
        rows = [
            row("bench_tok_s", "qwen2.5-0.5b-instruct-q8_0.gguf/tg128/t4", 15.0),
            row("bench_tok_s", "qwen2.5-0.5b-instruct-q4_0.gguf/tg128/t10", 20.0),
            row("bench_tok_s", "qwen2.5-0.5b-instruct-q4_0.gguf/tg128/t2", 20.0),
            row("bench_tok_s", "qwen2.5-0.5b-instruct-q4_k_m.gguf/tg128/t1", 20.0),
        ]
        out, _ = render(self, rows)
        svg = (out / "speed-decode.svg").read_text(encoding="utf-8")
        labels = re.findall(r">(Q[^<]*threads?)<", svg)
        self.assertEqual(labels, ["Q4_0, 2 threads", "Q4_0, 10 threads", "Q4_K_M, 1 thread", "Q8_0, 4 threads"])
        self.assertNotIn("instruct", svg)

    def test_speed_labels_include_the_model_when_quants_collide(self) -> None:
        keys = ["a-model-q4.gguf/tg128/t1", "b-model-q4.gguf/tg128/t1"]
        self.assertEqual(dashboard_render.speed_labels(keys), ["a-model-q4, 1 thread", "b-model-q4, 1 thread"])

    def test_speed_labels_include_the_test_when_a_chart_mixes_tests(self) -> None:
        keys = ["m-q4_0.gguf/pp512/t1", "m-q4_0.gguf/pp1024/t1"]
        self.assertEqual(dashboard_render.speed_labels(keys), ["Q4_0, 1 thread, pp512", "Q4_0, 1 thread, pp1024"])

    def test_speed_labels_fall_back_to_the_raw_key(self) -> None:
        self.assertEqual(
            dashboard_render.speed_labels(["weird key", "m-q4_0.gguf/tg128/tx"]), ["weird key", "m-q4_0.gguf/tg128/tx"]
        )

    def test_speed_chart_titles_follow_the_plotted_test_sizes(self) -> None:
        self.assertEqual(dashboard_render.speed_title("pp", ["pp512"]), "Prefill speed (512-token prompt)")
        self.assertEqual(dashboard_render.speed_title("tg", ["tg256"]), "Decode speed (256 generated tokens)")
        self.assertEqual(
            dashboard_render.speed_title("pp", ["pp512", "pp1024"]), "Prefill speed (several prompt sizes)"
        )
        self.assertEqual(dashboard_render.speed_title("tg", []), "Decode speed")
        rows = [row("bench_tok_s", "m-q4_0.gguf/pp1024/t1", 5.0), row("bench_tok_s", "m-q4_0.gguf/tg256/t1", 5.0)]
        out, _ = render(self, rows)
        self.assertIn("Prefill speed (1024-token prompt)", (out / "speed-prefill.svg").read_text(encoding="utf-8"))
        self.assertIn("Decode speed (256 generated tokens)", (out / "speed-decode.svg").read_text(encoding="utf-8"))
        self.assertNotIn("512-token", (out / "index.html").read_text(encoding="utf-8"))

    def test_speed_labels_name_the_test_whenever_a_chart_mixes_tests(self) -> None:
        # No two labels collide here, but without the test a reader could not tell which series is which size.
        keys = ["m-q4_0.gguf/pp512/t1", "m-q8_0.gguf/pp1024/t4"]
        self.assertEqual(dashboard_render.speed_labels(keys), ["Q4_0, 1 thread, pp512", "Q8_0, 4 threads, pp1024"])

    def test_clipping_keeps_labels_distinct_and_their_thread_counts(self) -> None:
        keys = [
            "tinyllama-1.1b-chat-v1.0-q4_0.gguf/tg128/t1",
            "tinyllama-1.1b-chat-v1.0-q4_0.gguf/tg128/t4",
            "qwen2.5-0.5b-instruct-q4_0.gguf/tg128/t1",
        ]
        labels = dashboard_render.speed_labels(keys)
        self.assertEqual(len(set(labels)), 3, labels)
        self.assertTrue(all(len(label) <= dashboard_render.MAX_LABEL for label in labels), labels)
        self.assertTrue(labels[0].endswith(", 1 thread") and labels[1].endswith(", 4 threads"), labels)
        self.assertTrue(labels[0].startswith("tinyllama") and labels[2].startswith("qwen"), labels)

    def test_labels_that_only_collide_after_clipping_are_still_made_distinct(self) -> None:
        labels = dashboard_render.speed_labels(["x" * 100 + "a", "x" * 100 + "b"])  # raw keys, clipped to one text
        self.assertEqual(len(set(labels)), 2, labels)
        self.assertTrue(all(len(label) <= dashboard_render.MAX_LABEL for label in labels), labels)

    def test_parse_rows_skips_calendar_invalid_dates_and_the_rest_still_renders(self) -> None:
        good = row("line_coverage", "src+cli", 80.0)
        bad = json.dumps(good).replace("2026-11-21T03:17:00Z", "2026-02-30T03:17:00Z")
        rows, problems = dashboard_render.parse_rows(bad + "\n" + json.dumps(good) + "\n")
        self.assertEqual((rows, len(problems)), ([good], 1))
        self.assertIn("date", problems[0])
        out = Path(self.enterContext(tempfile.TemporaryDirectory())) / "site"
        self.assertEqual(
            dashboard_render.render_site(bad + "\n" + json.dumps(good) + "\n", str(out), REPO), list(FILES)
        )

    def test_labels_are_clipped(self) -> None:
        (label,) = dashboard_render.speed_labels(["x" * 100])
        self.assertLessEqual(len(label), dashboard_render.MAX_LABEL)

    def test_accuracy_legend_uses_upper_case_quant_names(self) -> None:
        out, _ = render(self, [row("min_cosine", "q4_k_m/capital", 0.999), row("min_cosine", "f16/capital", 0.9999)])
        svg = (out / "accuracy.svg").read_text(encoding="utf-8")
        self.assertIn(">Q4_K_M<", svg)
        self.assertIn(">F16<", svg)

    def test_status_table_and_badge_ignore_jobs_silent_for_over_14_days(self) -> None:
        # A removed or renamed job must not keep a stale row (or a stale failure) forever. Age counts back from the
        # newest row, so a quiet repository still shows its last results.
        rows = [
            status("accuracy", True, "2026-11-21T03:17:00Z", run_id=9),
            status("fuzz_old", False, "2026-11-06T03:16:00Z", run_id=3),  # 15 days before the newest row: gone
            status("fuzz_edge", True, "2026-11-07T03:17:00Z", run_id=4),  # exactly 14 days: still shown
        ]
        out, _ = render(self, rows)
        html = (out / "index.html").read_text(encoding="utf-8")
        self.assertIn('<th scope="row">fuzz_edge</th>', html)
        self.assertNotIn("fuzz_old", html)
        badge = json.loads((out / "badge.json").read_text(encoding="utf-8"))
        self.assertEqual(badge["message"], "passing")

    def test_charts_show_only_the_last_90_days(self) -> None:
        rows = [
            row("bench_tok_s", "m-q4_0.gguf/tg128/t1", 20.0, "2026-11-21T03:17:00Z", run_id=9),
            row("bench_tok_s", "m-q8_0.gguf/tg128/t1", 15.0, "2026-08-20T03:17:00Z", run_id=1),  # 93 days earlier
            row("bench_tok_s", "m-q6_k.gguf/tg128/t1", 16.0, "2026-08-23T03:17:00Z", run_id=2),  # 90 days: kept
        ]
        out, _ = render(self, rows)
        svg = (out / "speed-decode.svg").read_text(encoding="utf-8")
        self.assertIn(">Q4_0, 1 thread<", svg)
        self.assertIn(">Q6_K, 1 thread<", svg)
        self.assertNotIn("Q8_0", svg)
        self.assertEqual(len(dashboard_render.recent([], 90)), 0)

    def test_cumulative_fuzz_hours_keep_their_total_inside_the_window(self) -> None:
        rows = [
            row("fuzz_seconds", "fuzz_gguf", 7200.0, "2026-06-01T03:17:00Z", run_id=1),  # far outside the window
            row("fuzz_seconds", "fuzz_gguf", 7200.0, "2026-11-20T03:17:00Z", run_id=2),
            row("fuzz_seconds", "fuzz_old", 3600.0, "2026-06-01T03:17:00Z", run_id=1),  # no point in the window
        ]
        out, _ = render(self, rows)
        svg = (out / "fuzz.svg").read_text(encoding="utf-8")
        self.assertIn(">fuzz_gguf<", svg)
        self.assertNotIn("fuzz_old", svg)
        series = dashboard_render.fuzz_series(rows)
        self.assertEqual([[v for _, v in pts] for _, _, pts in series], [[4.0]])

    def test_keys_that_are_not_bench_keys_belong_to_neither_speed_chart(self) -> None:
        rows = [row("bench_tok_s", "m-q4_0.gguf/pp512/tx", 5.0), row("bench_tok_s", "m-q4_0.gguf/pp512/t4", 6.0)]
        out, _ = render(self, rows)
        svg = (out / "speed-prefill.svg").read_text(encoding="utf-8")
        self.assertIn(">Q4_0, 4 threads<", svg)
        self.assertNotIn("tx", svg)
        self.assertEqual(len(dashboard_render.speed_series(rows, "pp")), 1)

    def test_parse_rows_skips_json_nested_too_deeply_and_huge_numbers(self) -> None:
        huge = json.dumps(row("line_coverage", "src+cli", 1.0)).replace('"value": 1.0', '"value": ' + "9" * 400)
        rows, problems = dashboard_render.parse_rows("[" * 100000 + "\n" + huge + "\n")
        self.assertEqual((rows, len(problems)), ([], 2))

    def test_main_parses_once_and_reports_skipped_rows_in_the_job_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "rows.jsonl"
            src.write_text(jsonl(sample_rows()) + "garbage\n", encoding="utf-8")
            summary = Path(tmp) / "summary.md"
            env = {"GITHUB_STEP_SUMMARY": str(summary)}
            with (
                mock.patch.object(dashboard_render, "parse_rows", wraps=dashboard_render.parse_rows) as parse,
                contextlib.redirect_stderr(io.StringIO()),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                code = dashboard_render.main([str(src), "--out", str(Path(tmp) / "site"), "--repo", REPO], env=env)
            self.assertEqual((code, parse.call_count), (0, 1))
            text = summary.read_text(encoding="utf-8")
            self.assertIn("1 malformed row", text)
            self.assertIn("line 10", text)

    def test_main_leaves_the_summary_alone_when_every_row_is_good(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "rows.jsonl"
            src.write_text(jsonl(sample_rows()), encoding="utf-8")
            summary = Path(tmp) / "summary.md"
            with contextlib.redirect_stdout(io.StringIO()):
                dashboard_render.main(
                    [str(src), "--out", str(Path(tmp) / "site"), "--repo", REPO],
                    env={"GITHUB_STEP_SUMMARY": str(summary)},
                )
            self.assertFalse(summary.exists())

    def test_invalid_repo_is_rejected(self) -> None:
        for bad in ("", "nibble", 'a/b"onmouseover="x', "a/b/c", "a b/c"):
            with self.assertRaisesRegex(ValueError, "repo"), tempfile.TemporaryDirectory() as tmp:
                dashboard_render.render_site("", tmp, bad)

    def test_malformed_rows_are_skipped_and_reported(self) -> None:
        good = row("line_coverage", "src+cli", 80.0)
        text = json.dumps(good) + "\nnot json\n" + json.dumps({"schema": 1}) + "\n\n"
        rows, problems = dashboard_render.parse_rows(text)
        self.assertEqual(rows, [good])
        self.assertEqual(len(problems), 2)
        self.assertIn("line 2", problems[0])
        self.assertIn("line 3", problems[1])

    def test_main_renders_and_warns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "rows.jsonl"
            src.write_text(jsonl(sample_rows()) + "garbage\n", encoding="utf-8")
            out = Path(tmp) / "site"
            err = io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                code = dashboard_render.main([str(src), "--out", str(out), "--repo", REPO], env={})
            self.assertEqual(code, 0)
            self.assertTrue((out / "index.html").is_file())
            self.assertIn("::warning", err.getvalue())

    def test_main_missing_input_renders_an_empty_site(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "site"
            with contextlib.redirect_stdout(io.StringIO()):
                code = dashboard_render.main([str(Path(tmp) / "nope.jsonl"), "--out", str(out), "--repo", REPO], env={})
            self.assertEqual(code, 0)
            self.assertIn("No results yet", (out / "index.html").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
