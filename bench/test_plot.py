"""Unit tests for bench/plot.py: ABBA block selection (abba()) and the time-series line_chart().

Run with: python -m unittest discover -s bench -p 'test_*.py'
Rows mirror bench/results/abba.jsonl: one row per run, note = "abba block <id> <A1|B1|B2|A2>", where A runs are
llama.cpp and B runs are nibble.
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import time
import unittest

import plot

MODEL = "qwen2.5-0.5b-instruct-q4_k_m.gguf"


def row(tool: str, commit: str, block: str, pos: str, median: float, date: str, load1: float = 0.5) -> plot.Row:
    """One abba.jsonl row for MODEL, pp512 at 8 threads."""
    return {
        "tool": tool,
        "commit": commit,
        "model": MODEL,
        "test": "pp512",
        "threads": 8,
        "median": median,
        "unit": "tok/s",
        "load1": load1,
        "date": date,
        "note": f"abba block {block} {pos}",
    }


def block(block_id: str, commit: str, nibble: float, llama: float, day: str, load1: float = 0.5) -> list[plot.Row]:
    """The four runs of one complete block (A1 B1 B2 A2), a minute apart on `day`."""
    return [
        row("llama.cpp", "b11715", block_id, "A1", llama, f"{day}T10:00:00Z", load1),
        row("nibble", commit, block_id, "B1", nibble, f"{day}T10:01:00Z"),
        row("nibble", commit, block_id, "B2", nibble, f"{day}T10:02:00Z"),
        row("llama.cpp", "b11715", block_id, "A2", llama, f"{day}T10:03:00Z"),
    ]


KEY = (MODEL, "pp512", 8)


class AbbaTest(unittest.TestCase):
    def test_abba_keeps_newest_commit(self) -> None:
        rows = block("old-1", "aaaaaaa", 10, 40, "2026-10-24") + block("new-1", "bbbbbbb", 20, 40, "2026-11-12")
        blocks = plot.abba({"abba": rows})[KEY]
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0][0], 2.0)

    def test_abba_keeps_all_blocks_of_one_commit(self) -> None:
        rows = block("2", "bbbbbbb", 20, 40, "2026-11-12") + block("1", "bbbbbbb", 10, 40, "2026-11-11")
        blocks = plot.abba({"abba": rows})[KEY]
        self.assertEqual([b[1] for b in blocks], [10.0, 20.0])  # date order

    def test_abba_marks_noisy_blocks(self) -> None:
        blocks = plot.abba({"abba": block("1", "bbbbbbb", 20, 40, "2026-11-12", load1=2.5)})[KEY]
        self.assertTrue(blocks[0][3])

    def test_abba_skips_incomplete_blocks(self) -> None:
        rows = [r for r in block("1", "bbbbbbb", 20, 40, "2026-11-12") if not r["note"].endswith("A2")]
        self.assertNotIn(KEY, plot.abba({"abba": rows}))


class MixedCommitBlockTest(unittest.TestCase):
    def mixed_block(self, block_id: str, day: str) -> list[plot.Row]:
        """A complete block whose B1 ran on one nibble build and B2 on another (rebuilt mid-block)."""
        rows = block(block_id, "ccccccc", 30, 40, day)
        for r in rows:
            if r["note"].endswith("B2"):
                r["commit"] = "ddddddd"
        return rows

    def test_abba_skips_blocks_with_mixed_commits_and_warns(self) -> None:
        rows = block("ok", "bbbbbbb", 20, 40, "2026-11-12") + self.mixed_block("mixed", "2026-11-13")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            blocks = plot.abba({"abba": rows})[KEY]
        self.assertEqual(
            [b[1] for b in blocks], [20.0]
        )  # the consistent block survives; the mixed one never decides "newest"
        self.assertIn("mixed", err.getvalue())

    def test_abba_with_only_mixed_blocks_reports_nothing_and_warns(self) -> None:
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            result = plot.abba({"abba": self.mixed_block("mixed", "2026-11-13")})
        self.assertNotIn(KEY, result)
        self.assertIn("mixed", err.getvalue())


DAY = 86400
T0 = 1_795_000_000.0  # 2026-11-18 11:06 UTC, an arbitrary unix time for the series


class LineChartTest(unittest.TestCase):
    def two_series(self) -> list[plot.Series]:
        pts = [(T0 + i * DAY, 100.0 + i) for i in range(5)]
        return [("Qwen <Q4_0> & co", "#1f77b4", pts), ("TinyLlama", "#ff7f0e", [(x, y / 2) for x, y in pts])]

    def test_line_chart_is_svg_with_legend(self) -> None:
        svg = plot.line_chart("decode tok/s", "tok/s", self.two_series())
        self.assertTrue(svg.startswith("<svg"))
        self.assertIn("Qwen &lt;Q4_0&gt; &amp; co", svg)  # labels are escaped
        self.assertNotIn("<Q4_0>", svg)
        self.assertIn("TinyLlama", svg)
        self.assertEqual(svg.count("<polyline"), 2)
        self.assertEqual(svg.count("<circle"), 10)  # every point is marked
        self.assertRegex(svg, r">\d{4}-\d{2}-\d{2}</text>")  # date ticks on the x axis
        self.assertIn(">2026-11-19</text>", svg)

    def test_line_chart_single_point_and_empty_series(self) -> None:
        one = plot.line_chart("t", "y", [("a", "#000", [(T0, 3.0)])])
        self.assertEqual(one.count("<circle"), 1)
        self.assertIn("2026-11-18", one)
        self.assertNotIn("no data yet", one)
        no_points: list[list[plot.Series]] = [[], [("a", "#000", []), ("b", "#111", [])]]
        for series in no_points:
            empty = plot.line_chart("t", "y", series)
            self.assertIn("no data yet", empty)
            self.assertNotIn("<polyline", empty)
            self.assertNotIn("nan", empty)
        mixed = plot.line_chart("t", "y", [("a", "#000", []), ("b", "#111", [(T0, 1.0), (T0 + DAY, 2.0)])])
        self.assertEqual(mixed.count("<polyline"), 1)
        self.assertIn(">a</text>", mixed)  # the empty series keeps its legend entry

    def test_line_chart_y_min(self) -> None:
        pts = [(T0 + i * DAY, 0.9995 + i * 1e-5) for i in range(4)]
        svg = plot.line_chart("min cosine", "cosine", [("q4_0", "#000", pts)], y_min=0.99)
        self.assertIn(">0.990</text>", svg)  # the lowest tick is y_min
        self.assertNotIn(">0</text>", svg)

    @staticmethod
    def y_labels(svg: str) -> list[str]:
        """The y axis tick labels, bottom to top (the text elements right-aligned at the axis)."""
        return re.findall(r'text-anchor="end"[^>]*>([^<]*)</text>', svg)

    def test_line_chart_y_min_ticks_are_nice_and_evenly_formatted(self) -> None:
        pts = [(T0, 0.9995), (T0 + DAY, 0.99995)]
        svg = plot.line_chart("min cosine", "cosine", [("q4_0", "#000", pts)], y_min=0.99)
        self.assertEqual(self.y_labels(svg), ["0.990", "0.992", "0.994", "0.996", "0.998", "1.000"])

    def test_line_chart_y_min_ticks_other_scales(self) -> None:
        svg = plot.line_chart("t", "y", [("a", "#000", [(T0, 12.0), (T0 + DAY, 87.0)])], y_min=10.0)
        labels = [float(v) for v in self.y_labels(svg)]
        steps = {round(a - b, 9) for a, b in zip(labels, labels[1:], strict=False)}
        self.assertEqual(len(steps), 1)  # evenly spaced
        self.assertEqual(labels[0], 10.0)  # starts at y_min
        self.assertGreaterEqual(labels[-1], 87.0)  # covers the data

    def test_line_chart_default_ticks_unchanged(self) -> None:
        svg = plot.line_chart("t", "y", self.two_series())
        self.assertEqual(self.y_labels(svg), ["0", "40", "80", "120", "160", "200"])

    def test_date_ticks_cost_does_not_depend_on_the_span(self) -> None:
        start = time.monotonic()
        for hi in (1e9, 1e15, 1e300):
            ticks = plot.date_ticks(0.0, hi)
            self.assertTrue(1 <= len(ticks) <= 9, (hi, len(ticks)))
            self.assertTrue(all(0.0 <= t <= hi for t in ticks))
            if hi <= 1e15:  # beyond 2^53 seconds a float cannot hold a whole midnight
                self.assertTrue(all(t % plot.DAY == 0 for t in ticks))
        self.assertLess(time.monotonic() - start, 2.0)

    def test_date_label_out_of_range_falls_back_to_a_number(self) -> None:
        for t in (1e13, 1e18, -1e18, 1e300):
            self.assertTrue(plot.date_label(t))  # no ValueError or OSError
        self.assertEqual(plot.date_label(0.0), "1970-01-01")

    def test_line_chart_drops_timestamps_outside_the_calendar(self) -> None:
        """Millisecond timestamps (year 58851) and 1e15 s are not dates: dropped, never a crash or a hang."""
        start = time.monotonic()
        ms = plot.line_chart("t", "y", [("a", "#000", [(T0 * 1000, 1.0), (T0 * 1000 + DAY, 2.0)])])
        self.assertIn("no data yet", ms)
        wide = plot.line_chart("t", "y", [("a", "#000", [(0.0, 1.0), (1e15, 2.0), (T0, 3.0)])])
        self.assertEqual(wide.count("<circle"), 2)
        self.assertLess(time.monotonic() - start, 2.0)

    def test_line_chart_negative_data_has_nice_labelled_ticks(self) -> None:
        pts = [(T0, -3.0), (T0 + DAY, 5.0)]
        svg = plot.line_chart("t", "y", [("a", "#000", pts)])
        labels = [float(v) for v in self.y_labels(svg)]
        steps = {round(b - a, 9) for a, b in zip(labels, labels[1:], strict=False)}
        self.assertEqual(len(steps), 1)
        self.assertIn(steps.pop(), (1.0, 2.0, 5.0))  # a nice step
        self.assertLessEqual(labels[0], -3.0)
        self.assertGreaterEqual(labels[-1], 5.0)
        self.assertNotIn("-0", self.y_labels(svg))
        # the gridlines sit exactly where the labels say: evenly spaced and the 0 label on a gridline
        ys = [float(v) for v in re.findall(r'<line x1="70.0" y1="([\d.]+)"', svg)]
        self.assertEqual(len(ys), len(labels))
        zero_y = ys[labels.index(0.0)]
        self.assertAlmostEqual((400.0 - zero_y) / 360.0, (0.0 - labels[0]) / (labels[-1] - labels[0]), places=2)

    def test_line_chart_all_negative_data(self) -> None:
        svg = plot.line_chart("t", "y", [("a", "#000", [(T0, -10.0), (T0 + DAY, -3.0)])])
        labels = [float(v) for v in self.y_labels(svg)]
        self.assertLessEqual(labels[0], -10.0)
        self.assertGreaterEqual(labels[-1], -3.0)

    def test_line_chart_rejects_non_finite_y_min(self) -> None:
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaisesRegex(ValueError, "y_min"):
                plot.line_chart("t", "y", [("a", "#000", [(T0, 1.0)])], y_min=bad)

    def test_line_chart_footnote_names_the_axis_minimum_when_points_are_clamped(self) -> None:
        low = plot.line_chart("t", "y", [("a", "#000", [(T0, 0.5), (T0 + DAY, 1.0)])], y_min=0.99)
        self.assertIn("Hollow = below the axis minimum (0.99)", low)
        fine = plot.line_chart("t", "y", [("a", "#000", [(T0, 0.995), (T0 + DAY, 1.0)])], y_min=0.99)
        self.assertNotIn("Hollow", fine)

    def test_line_chart_default_axis_starts_at_zero(self) -> None:
        svg = plot.line_chart("t", "y", self.two_series())
        self.assertIn(">0</text>", svg)

    def test_line_chart_ticks_are_thinned(self) -> None:
        pts = [(T0 + i * DAY, float(i)) for i in range(400)]
        svg = plot.line_chart("t", "y", [("a", "#000", pts)])
        ticks = re.findall(r">(\d{4}-\d{2}-\d{2})</text>", svg)
        self.assertTrue(2 <= len(ticks) <= 9, ticks)
        self.assertEqual(len(set(ticks)), len(ticks))

    def test_line_chart_ignores_non_finite_points(self) -> None:
        pts = [(T0, 1.0), (T0 + DAY, float("nan")), (T0 + 2 * DAY, float("inf")), (T0 + 3 * DAY, 2.0)]
        svg = plot.line_chart("t", "y", [("a", "#000", pts)])
        self.assertEqual(svg.count("<circle"), 2)
        self.assertNotIn("nan", svg)
        self.assertNotIn("inf", svg)

    def test_line_chart_points_below_y_min_stay_on_the_chart(self) -> None:
        svg = plot.line_chart("t", "y", [("a", "#000", [(T0, 0.5), (T0 + DAY, 1.0)])], y_min=0.99)
        for cy in re.findall(r'<circle cx="[\d.]+" cy="([\d.]+)"', svg):
            self.assertTrue(0 <= float(cy) <= 460, cy)


class ReadmeProtocolTest(unittest.TestCase):
    """The README's protocol paragraph must describe the runs plot.py actually reports."""

    def setUp(self) -> None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "README.md"), encoding="utf-8") as f:
            text = f.read()
        start = text.index("**Protocol**")
        self.protocol = " ".join(text[start : text.index("\n\n", start)].split())
        blocks = plot.abba(plot.load(os.path.join(root, "bench", "results")))
        pp = [b for (_, test, _), bs in blocks.items() if test == plot.PP for b in bs]
        self.reported_runs = 4 * len(pp)  # each ABBA block is four runs, and each run yields one pp512 row
        self.reported_noisy = sum(1 for b in pp if b[3])

    def test_readme_protocol_states_reported_run_count(self) -> None:
        m = re.search(r"All (\d+)(?: reported)? runs", self.protocol)
        self.assertIsNotNone(m, "protocol paragraph has no 'All N runs' sentence")
        assert m is not None
        self.assertEqual(int(m[1]), self.reported_runs)

    def test_readme_protocol_marks_noisy_blocks_only_if_reported(self) -> None:
        claims_marked = "marked †" in self.protocol
        self.assertEqual(claims_marked, self.reported_noisy > 0)


if __name__ == "__main__":
    unittest.main()
