"""Unit tests for tools/ci/dashboard_data.py: turning nightly measurements into validated dashboard rows.

Run with: python -m unittest discover -s tools/ci -p 'test_*.py'
Fixtures are real tool output: a `nibble bench --out` line copied from bench/results/abba.jsonl, the tail of a
libFuzzer run with -print_final_stats=1 (clang 18), a `gcovr --print-summary` tail (gcovr 8.7) and /proc/cpuinfo
excerpts from x86 and aarch64 machines. The environment is always passed in, so no test reads os.environ.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

import dashboard_data
from cosine import Cosine
from dashboard_data import Meta, Row

SHA = "0123456789abcdef0123456789abcdef01234567"
META = Meta(date="2026-11-21T03:17:00Z", commit=SHA, source="nightly", runner="Test CPU @ 3.0GHz", run_id=4242)

# Two lines written by `nibble bench --out` (cli/cmd_bench.cpp Sink::emit) plus its --mem line, as found in
# bench/results/*.jsonl. The "median" field is the one the dashboard plots; membw is GB/s and is not a dashboard metric.
BENCH = (
    '{"tool":"nibble","commit":"39ec442","kernels":"avx2+fma+f16c+avxvnni","model":"qwen2.5-0.5b-instruct-q4_0.gguf",'
    '"test":"tg128","threads":1,"reps":5,"median":24.1292,"min":23.3240,"max":25.9033,"unit":"tok/s","load1":0.40,'
    '"date":"2026-10-19T13:46:54Z","note":"abba block 20261019T134449-1 B1"}\n'
    '{"tool":"nibble","commit":"39ec442","kernels":"avx2+fma+f16c+avxvnni","model":"qwen2.5-0.5b-instruct-q4_0.gguf",'
    '"test":"pp512","threads":4,"reps":5,"median":207.6029,"min":200.7843,"max":230.7439,"unit":"tok/s","load1":0.40,'
    '"date":"2026-10-19T13:46:54Z"}\n'
    '{"tool":"nibble","commit":"36584f9","kernels":"avx2","model":"qwen2.5-0.5b-instruct-q4_0.gguf","test":"membw",'
    '"threads":12,"reps":3,"median":34.0008,"min":32.9766,"max":35.5738,"unit":"GB/s","load1":13.72,'
    '"date":"2026-10-19T04:14:19Z"}\n'
)

# The end of a real libFuzzer run (-max_total_time=3 -print_final_stats=1), with the counts the brief names.
LIBFUZZER = """\
#2097152\tpulse  cov: 5 ft: 5 corp: 4/13b lim: 4096 exec/s: 1048576 rss: 27Mb
#1234567\tDONE   cov: 5 ft: 5 corp: 4/13b lim: 4096 exec/s: 718301 rss: 27Mb
Done 1234567 runs in 7200 second(s)
stat::number_of_executed_units: 1234567
stat::average_exec_per_sec:     171
stat::new_units_added:          4
stat::slowest_unit_time_sec:    0
stat::peak_rss_mb:              27
"""

# The end of `gcovr --print-summary`.
GCOVR = """\
------------------------------------------------------------------------------
TOTAL                                            1447     1234    85%
------------------------------------------------------------------------------
lines: 85.3% (1234 out of 1447)
functions: 100.0% (2 out of 2)
branches: 50.0% (1 out of 2)
"""

CPUINFO_X86 = (
    "processor\t: 0\nvendor_id\t: GenuineIntel\nmodel name\t: 12th Gen Intel(R) Core(TM) i7-1265U\nflags\t: fpu\n"
)
# aarch64 /proc/cpuinfo has no "model name": only implementer/part codes (and "Hardware" on some SoCs).
CPUINFO_ARM = "processor\t: 0\nBogoMIPS\t: 2000.00\nFeatures\t: fp asimd\nCPU implementer\t: 0x41\nCPU part\t: 0xd0c\n"


def good_row(**changes: object) -> Row:
    """A valid row, with `changes` applied on top."""
    row: Row = {
        "schema": 1,
        "date": META.date,
        "commit": SHA,
        "source": "nightly",
        "metric": "binary_bytes",
        "key": "nibble",
        "value": 1234.0,
        "unit": "bytes",
        "runner": META.runner,
        "run_id": 4242,
    }
    row.update(changes)
    return row


class ValidateTests(unittest.TestCase):
    def test_accepts_good_row(self) -> None:
        self.assertEqual(dashboard_data.validate(good_row()), good_row())

    def test_validate_rejects_nan_and_long_key(self) -> None:
        with self.assertRaisesRegex(ValueError, "value"):
            dashboard_data.validate(good_row(value=math.nan))
        with self.assertRaisesRegex(ValueError, "value"):
            dashboard_data.validate(good_row(value=math.inf))
        with self.assertRaisesRegex(ValueError, "key"):
            dashboard_data.validate(good_row(key="k" * 121))
        dashboard_data.validate(good_row(key="k" * 120))  # the limit itself is allowed

    def test_rejects_each_bad_field_by_name(self) -> None:
        bad: dict[str, object] = {
            "schema": 2,
            "date": "2026-11-21",
            "commit": "abc1234",
            "source": "manual",
            "metric": "tok_s",
            "key": "",
            "value": "1.0",
            "unit": "tok/s",  # valid unit, wrong for the metric
            "runner": "r" * 121,
            "run_id": "7",
        }
        for field, value in bad.items():
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                dashboard_data.validate(good_row(**{field: value}))

    def test_rejects_bool_numbers_missing_and_extra_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, "run_id"):
            dashboard_data.validate(good_row(run_id=True))
        with self.assertRaisesRegex(ValueError, "value"):
            dashboard_data.validate(good_row(value=True))
        missing = good_row()
        del missing["runner"]
        with self.assertRaisesRegex(ValueError, "runner"):
            dashboard_data.validate(missing)
        with self.assertRaisesRegex(ValueError, "extra"):
            dashboard_data.validate(good_row(extra=1))

    def test_integer_value_becomes_float(self) -> None:
        row = dashboard_data.validate(good_row(value=7))
        self.assertIsInstance(row["value"], float)


class FromTests(unittest.TestCase):
    def test_from_bench(self) -> None:
        rows = dashboard_data.from_bench(BENCH, META)
        self.assertEqual([r["key"] for r in rows], [
            "qwen2.5-0.5b-instruct-q4_0.gguf/tg128/t1",
            "qwen2.5-0.5b-instruct-q4_0.gguf/pp512/t4",
        ])  # fmt: skip
        self.assertEqual(rows[0]["metric"], "bench_tok_s")
        self.assertEqual(rows[0]["value"], 24.1292)  # the median, not min or max
        self.assertEqual(rows[0]["unit"], "tok/s")
        self.assertEqual(rows[0]["commit"], SHA)  # the tested commit, not the short sha inside the bench line
        self.assertEqual(rows[1]["value"], 207.6029)

    def test_from_bench_rejects_bad_json_and_missing_field(self) -> None:
        with self.assertRaisesRegex(ValueError, "line 1"):
            dashboard_data.from_bench("not json\n", META)
        with self.assertRaisesRegex(ValueError, "median"):
            dashboard_data.from_bench(
                '{"tool":"nibble","model":"m.gguf","test":"tg128","threads":1,"unit":"tok/s"}', META
            )

    def test_from_cosine(self) -> None:
        found = {
            "q4_0/capital": Cosine("q4_0", "capital", 0.999123, 12, 12),
            "f16/chat": Cosine("f16", "chat", 0.999877, 31, 31),
        }
        rows = dashboard_data.from_cosine(found, META)
        self.assertEqual([(r["key"], r["value"]) for r in rows], [("q4_0/capital", 0.999123), ("f16/chat", 0.999877)])
        self.assertEqual({(r["metric"], r["unit"]) for r in rows}, {("min_cosine", "cosine")})

    def test_from_cosine_nan_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "value"):
            dashboard_data.from_cosine({"q4_0/capital": Cosine("q4_0", "capital", math.nan, 12, 12)}, META)

    def test_from_libfuzzer(self) -> None:
        rows = dashboard_data.from_libfuzzer(LIBFUZZER, "fuzz_gguf", META, corpus_files=77)
        got = {r["metric"]: (r["key"], r["value"], r["unit"]) for r in rows}
        self.assertEqual(
            got,
            {
                "fuzz_execs": ("fuzz_gguf", 1234567.0, "execs"),
                "fuzz_seconds": ("fuzz_gguf", 7200.0, "s"),
                "corpus_files": ("fuzz_gguf", 77.0, "files"),
            },
        )

    def test_from_libfuzzer_without_corpus_count_omits_that_row(self) -> None:
        rows = dashboard_data.from_libfuzzer(LIBFUZZER, "fuzz_gguf", META)
        self.assertEqual({r["metric"] for r in rows}, {"fuzz_execs", "fuzz_seconds"})

    def test_from_libfuzzer_crash_keeps_the_measured_time_and_the_counters(self) -> None:
        # A target that crashed after 90 minutes: final stats are printed but there is no `Done` line.
        crashed = "==1==ERROR: libFuzzer: deadly signal\nstat::number_of_executed_units: 99\n"
        rows = dashboard_data.from_libfuzzer(crashed, "fuzz_gguf", META, seconds=5400)
        got = {r["metric"]: r["value"] for r in rows}
        self.assertEqual(got, {"fuzz_execs": 99.0, "fuzz_seconds": 5400.0})

    def test_from_libfuzzer_without_any_stats_still_records_the_time(self) -> None:
        rows = dashboard_data.from_libfuzzer("Segmentation fault\n", "fuzz_gguf", META, seconds=5400)
        self.assertEqual([(r["metric"], r["value"]) for r in rows], [("fuzz_seconds", 5400.0)])

    def test_from_libfuzzer_prefers_the_done_line_to_the_measured_time(self) -> None:
        rows = dashboard_data.from_libfuzzer(LIBFUZZER, "fuzz_gguf", META, seconds=9999)
        self.assertEqual({r["metric"]: r["value"] for r in rows}["fuzz_seconds"], 7200.0)

    def test_from_libfuzzer_done_line_alone_gives_the_time_only(self) -> None:
        rows = dashboard_data.from_libfuzzer("Done 5 runs in 1 second(s)\n", "fuzz_gguf", META)
        self.assertEqual([(r["metric"], r["value"]) for r in rows], [("fuzz_seconds", 1.0)])

    def test_from_libfuzzer_without_any_elapsed_time_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Done"):
            dashboard_data.from_libfuzzer("stat::number_of_executed_units: 5\n", "fuzz_gguf", META)
        with self.assertRaisesRegex(ValueError, "--seconds"):
            dashboard_data.from_libfuzzer("", "fuzz_gguf", META)

    def test_from_libfuzzer_rejects_negative_seconds(self) -> None:
        with self.assertRaisesRegex(ValueError, "seconds"):
            dashboard_data.from_libfuzzer("", "fuzz_gguf", META, seconds=-1)

    def test_from_gcovr_summary(self) -> None:
        (row,) = dashboard_data.from_gcovr_summary(GCOVR, META)
        self.assertEqual(
            (row["metric"], row["key"], row["value"], row["unit"]), ("line_coverage", "src+cli", 85.3, "%")
        )

    def test_from_gcovr_summary_without_lines_line(self) -> None:
        with self.assertRaisesRegex(ValueError, "lines:"):
            dashboard_data.from_gcovr_summary("functions: 100.0% (2 out of 2)\n", META)

    def test_status_row(self) -> None:
        ok = dashboard_data.status_row("nightly", "fuzz_gguf", True, META)
        bad = dashboard_data.status_row("nightly", "fuzz_gguf", False, META)
        self.assertEqual(
            (ok["metric"], ok["key"], ok["value"], ok["unit"]), ("job_status", "nightly/fuzz_gguf", 1.0, "ok")
        )
        self.assertEqual(bad["value"], 0.0)

    def test_binary_size_row(self) -> None:
        (row,) = dashboard_data.from_binary_size(2097152, META)
        self.assertEqual(
            (row["metric"], row["key"], row["value"], row["unit"]), ("binary_bytes", "nibble", 2097152.0, "bytes")
        )


class MergeTests(unittest.TestCase):
    def rows(self) -> list[Row]:
        return [
            dashboard_data.status_row("nightly", "a", True, META),
            dashboard_data.status_row("nightly", "b", False, META),
        ]

    def test_merge_is_idempotent(self) -> None:
        once = dashboard_data.merge("", self.rows())
        twice = dashboard_data.merge(once, self.rows())
        self.assertEqual(once, twice)
        self.assertEqual(len(once.splitlines()), 2)
        self.assertTrue(once.endswith("\n"))

    def test_merge_keeps_file_order_and_existing_text(self) -> None:
        old = '{"legacy": true, "run_id": 1}\n' + json.dumps(good_row(run_id=1)) + "\n"
        merged = dashboard_data.merge(old, self.rows())
        lines = merged.splitlines()
        self.assertEqual(lines[:2], old.splitlines())  # existing rows untouched, even if they are not valid rows
        self.assertEqual(len(lines), 4)

    def test_merge_duplicate_is_on_run_id_metric_key(self) -> None:
        first = dashboard_data.status_row("nightly", "a", True, META)
        again = dashboard_data.status_row("nightly", "a", False, META)  # same (run_id, metric, key), new value
        merged = dashboard_data.merge(dashboard_data.merge("", [first]), [again])
        self.assertEqual([json.loads(line)["value"] for line in merged.splitlines()], [1.0])
        other_run = Meta(META.date, SHA, "nightly", META.runner, 4243)
        merged = dashboard_data.merge(merged, [dashboard_data.status_row("nightly", "a", True, other_run)])
        self.assertEqual(len(merged.splitlines()), 2)

    def test_merge_rejects_an_invalid_new_row_and_bad_existing_json(self) -> None:
        with self.assertRaisesRegex(ValueError, "value"):
            dashboard_data.merge("", [good_row(value=math.nan)])
        with self.assertRaisesRegex(ValueError, "line 1"):
            dashboard_data.merge("garbage\n", [])


class MetaTests(unittest.TestCase):
    NOW = "2026-11-21T03:17:00Z"

    def test_meta_from_env(self) -> None:
        env = {"GITHUB_SHA": SHA, "GITHUB_RUN_ID": "99"}
        meta = dashboard_data.meta_from_env("portability", env, CPUINFO_X86, self.NOW)
        self.assertEqual(
            meta, Meta(self.NOW, SHA, "portability", "12th Gen Intel(R) Core(TM) i7-1265U", 99)
        )  # fmt: skip

    def test_cpu_name_falls_back_on_arm_and_empty(self) -> None:
        self.assertEqual(dashboard_data.cpu_name(CPUINFO_ARM), "unknown (CPU implementer 0x41 part 0xd0c)")
        self.assertEqual(dashboard_data.cpu_name("Hardware\t: BCM2835\n"), "BCM2835")
        self.assertEqual(dashboard_data.cpu_name(""), "unknown")

    def test_long_cpu_name_is_truncated_to_the_row_limit(self) -> None:
        self.assertEqual(len(dashboard_data.cpu_name("model name\t: " + "x" * 300)), 120)

    def test_missing_or_bad_env_is_named(self) -> None:
        with self.assertRaisesRegex(ValueError, "GITHUB_SHA"):
            dashboard_data.meta_from_env("nightly", {"GITHUB_RUN_ID": "1"}, "", self.NOW)
        with self.assertRaisesRegex(ValueError, "GITHUB_RUN_ID"):
            dashboard_data.meta_from_env("nightly", {"GITHUB_SHA": SHA, "GITHUB_RUN_ID": "x"}, "", self.NOW)


class CliTests(unittest.TestCase):
    ENV = {"GITHUB_SHA": SHA, "GITHUB_RUN_ID": "4242"}
    NOW = "2026-11-21T03:17:00Z"

    def run_cli(self, *argv: str) -> tuple[int, str]:
        """Run main() with a fixed environment; return (exit code, stderr text)."""
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = dashboard_data.main(list(argv), env=self.ENV, cpuinfo=CPUINFO_X86, now=self.NOW)
        return code, err.getvalue()

    def test_status_cli_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "rows.jsonl"
            base = ["status", "--workflow", "nightly", "--job", "fuzz_gguf", "--out", str(out)]
            self.assertEqual(self.run_cli(*base, "--outcome", "success")[0], 0)
            (ok,) = [json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual((ok["metric"], ok["key"], ok["value"]), ("job_status", "nightly/fuzz_gguf", 1.0))
            self.assertEqual(ok["runner"], "12th Gen Intel(R) Core(TM) i7-1265U")
            self.assertEqual(ok["run_id"], 4242)
            self.assertEqual(ok["date"], self.NOW)

            self.assertEqual(self.run_cli(*base, "--outcome", "failure")[0], 0)  # appends
            rows = [json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual([r["value"] for r in rows], [1.0, 0.0])

            before = out.read_text()
            self.assertEqual(self.run_cli(*base, "--outcome", "cancelled")[0], 0)
            self.assertEqual(out.read_text(), before)

    def test_status_cancelled_does_not_create_the_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "rows.jsonl"
            args = ["status", "--workflow", "w", "--job", "j", "--outcome", "cancelled", "--out", str(out)]
            self.assertEqual(self.run_cli(*args)[0], 0)
            self.assertFalse(out.exists())

    def test_from_commands_append_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "bench.jsonl").write_text(BENCH)
            (d / "fuzz.log").write_text(LIBFUZZER)
            (d / "summary.txt").write_text(GCOVR)
            (d / "ctest.log").write_text(
                "test_e2e_golden.cpp:194:\nwarning:\n  capital [q4_0]: min cosine over 12 positions = 0.999123, "
                "top-1 agreement 12/\n  12\n"
            )
            (d / "nibble").write_bytes(b"x" * 1000)
            out = str(d / "rows.jsonl")
            cmds = [
                ["from-bench", str(d / "bench.jsonl")],
                ["from-cosine", str(d / "ctest.log")],
                ["from-libfuzzer", str(d / "fuzz.log"), "--target", "fuzz_gguf", "--corpus-files", "5"],
                ["from-gcovr", str(d / "summary.txt")],
                ["from-binary-size", str(d / "nibble")],
            ]
            for cmd in cmds:
                self.assertEqual(self.run_cli(*cmd, "--out", out, "--source", "ci")[0], 0, cmd)
            rows = [json.loads(line) for line in Path(out).read_text().splitlines()]
            self.assertEqual(
                [r["metric"] for r in rows],
                ["bench_tok_s", "bench_tok_s", "min_cosine", "fuzz_execs", "fuzz_seconds", "corpus_files",
                 "line_coverage", "binary_bytes"],
            )  # fmt: skip
            self.assertEqual({r["source"] for r in rows}, {"ci"})
            self.assertEqual(rows[-1]["value"], 1000.0)

    def test_merge_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "existing.jsonl").write_text(dashboard_data.merge("", [good_row()]))
            (d / "new1.jsonl").write_text(dashboard_data.merge("", [good_row(), good_row(key="other")]))
            (d / "new2.jsonl").write_text(dashboard_data.merge("", [good_row(key="third")]))
            out = d / "merged.jsonl"
            args = ["merge", str(d / "existing.jsonl"), str(d / "new1.jsonl"), str(d / "new2.jsonl"), "--out", str(out)]
            self.assertEqual(self.run_cli(*args)[0], 0)
            self.assertEqual([json.loads(x)["key"] for x in out.read_text().splitlines()], ["nibble", "other", "third"])

    def test_merge_cli_missing_existing_file_is_a_first_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "new.jsonl").write_text(dashboard_data.merge("", [good_row()]))
            out = d / "merged.jsonl"
            self.assertEqual(
                self.run_cli("merge", str(d / "none.jsonl"), str(d / "new.jsonl"), "--out", str(out))[0], 0
            )
            self.assertEqual(len(out.read_text().splitlines()), 1)

    def test_merge_cli_rejects_an_invalid_new_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "existing.jsonl").write_text("")
            (d / "new.jsonl").write_text('{"schema":1}\n')
            code, err = self.run_cli(
                "merge", str(d / "existing.jsonl"), str(d / "new.jsonl"), "--out", str(d / "o.jsonl")
            )
            self.assertEqual(code, 1)
            self.assertIn("dashboard_data:", err)

    def test_bad_input_exits_1_with_a_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "fuzz.log"
            log.write_text("nothing useful\n")
            code, err = self.run_cli("from-libfuzzer", str(log), "--target", "t", "--out", str(Path(tmp) / "o.jsonl"))
            self.assertEqual(code, 1)
            self.assertIn("--seconds", err)

    def test_from_libfuzzer_cli_takes_the_measured_seconds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "fuzz.log"
            log.write_text("stat::number_of_executed_units: 12\n")
            out = Path(tmp) / "o.jsonl"
            code, _ = self.run_cli("from-libfuzzer", str(log), "--target", "t", "--seconds", "5400", "--out", str(out))
            self.assertEqual(code, 0)
            rows = [json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual({r["metric"]: r["value"] for r in rows}, {"fuzz_execs": 12.0, "fuzz_seconds": 5400.0})


if __name__ == "__main__":
    unittest.main()
