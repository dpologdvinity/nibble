"""Unit tests for tools/ci/perf_ab.py: ABBA ordering, per-block ratios, the verdict rules and the run driver.

Run with: python -m unittest discover -s tools/ci -p 'test_*.py'
`run` is tested with a fake `nibble bench` (an injected runner) that writes the JSONL a real one would, so no
binary or model is needed.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import tempfile
import unittest
import unittest.mock
from collections.abc import Mapping, Sequence
from pathlib import Path

import perf_ab


def _row(
    side: str, block: int, median: object, model: str = "m.gguf", test: str = "tg128", threads: int = 1
) -> dict[str, object]:
    return {"side": side, "block": block, "model": model, "test": test, "threads": threads, "median": median}


class AbbaOrderTest(unittest.TestCase):
    def test_abba_order_two_rounds(self) -> None:
        self.assertEqual(
            perf_ab.abba_order(2),
            [
                (0, "A1", "base"),
                (0, "B1", "head"),
                (0, "B2", "head"),
                (0, "A2", "base"),
                (1, "A1", "base"),
                (1, "B1", "head"),
                (1, "B2", "head"),
                (1, "A2", "base"),
            ],
        )

    def test_abba_order_zero_rounds_is_empty(self) -> None:
        self.assertEqual(perf_ab.abba_order(0), [])


class BlockRatiosTest(unittest.TestCase):
    def test_block_ratio_math(self) -> None:
        # Block 0: head (11 + 13) / base (10 + 10) = 1.2. Block 1: head (9 + 9) / base (10 + 8) = 1.0.
        rows = [
            _row("base", 0, 10.0),
            _row("head", 0, 11.0),
            _row("head", 0, 13.0),
            _row("base", 0, 10.0),
            _row("base", 1, 10.0),
            _row("head", 1, 9.0),
            _row("head", 1, 9.0),
            _row("base", 1, 8.0),
        ]
        ratios = perf_ab.block_ratios(rows)
        self.assertEqual(list(ratios), [("m.gguf", "tg128", 1)])
        self.assertEqual(ratios[("m.gguf", "tg128", 1)], [1.2, 1.0])

    def test_keys_separate_model_test_and_threads(self) -> None:
        rows = [
            _row(side, 0, med, model=model, test=test, threads=t)
            for model, test, t in [("a.gguf", "pp512", 1), ("a.gguf", "tg128", 4), ("b.gguf", "tg128", 4)]
            for side, med in [("base", 2.0), ("head", 1.0), ("head", 1.0), ("base", 2.0)]
        ]
        ratios = perf_ab.block_ratios(rows)
        self.assertEqual(sorted(ratios), [("a.gguf", "pp512", 1), ("a.gguf", "tg128", 4), ("b.gguf", "tg128", 4)])
        self.assertTrue(all(v == [0.5] for v in ratios.values()))

    def test_blocks_are_ordered_by_block_id(self) -> None:
        # Base is 1.0 in every run; head is 2.0 in block 1 and 1.0 in block 0, and block 1's rows come first.
        rows = [
            _row(s, b, 1.0 if s == "base" else head)
            for b, head in [(1, 2.0), (0, 1.0)]
            for s in ("base", "head", "head", "base")
        ]
        self.assertEqual(perf_ab.block_ratios(rows)[("m.gguf", "tg128", 1)], [1.0, 2.0])

    def test_bad_medians_make_the_block_nan_not_a_crash(self) -> None:
        for bad in (0.0, -1.0, math.nan, math.inf, None, "fast", True):
            with self.subTest(bad=bad):
                rows = [_row("base", 0, 10.0), _row("head", 0, bad), _row("head", 0, 10.0), _row("base", 0, 10.0)]
                (ratio,) = perf_ab.block_ratios(rows)[("m.gguf", "tg128", 1)]
                self.assertTrue(math.isnan(ratio))

    def test_missing_median_field_is_nan(self) -> None:
        rows = [_row("base", 0, 10.0), _row("head", 0, 10.0), _row("head", 0, 10.0), _row("base", 0, 10.0)]
        del rows[1]["median"]
        (ratio,) = perf_ab.block_ratios(rows)[("m.gguf", "tg128", 1)]
        self.assertTrue(math.isnan(ratio))

    def test_incomplete_block_is_nan(self) -> None:
        rows = [_row("base", 0, 10.0), _row("head", 0, 10.0), _row("head", 0, 10.0)]  # A2 never ran
        (ratio,) = perf_ab.block_ratios(rows)[("m.gguf", "tg128", 1)]
        self.assertTrue(math.isnan(ratio))

    def test_a_block_missing_for_one_key_is_nan(self) -> None:
        # Block 1 has rows for pp512 but none for tg128: tg128 must not end up with one fewer, better-looking ratio.
        rows = [
            _row(s, b, 10.0, test=t)
            for t in ("pp512", "tg128")
            for b in (0, 2)
            for s in ("base", "head", "head", "base")
        ]
        rows += [_row(s, 1, 10.0, test="pp512") for s in ("base", "head", "head", "base")]
        ratios = perf_ab.block_ratios(rows)
        self.assertEqual(ratios[("m.gguf", "pp512", 1)], [1.0, 1.0, 1.0])
        tg = ratios[("m.gguf", "tg128", 1)]
        self.assertEqual(len(tg), 3)
        self.assertTrue(math.isnan(tg[1]))

    def test_unattributable_rows_are_ignored(self) -> None:
        junk: list[Mapping[str, object]] = [{}, {"side": "other", "block": 0}, {"side": "base", "block": "x"}]
        self.assertEqual(perf_ab.block_ratios(junk), {})


class VerdictTest(unittest.TestCase):
    def test_verdict_faster(self) -> None:
        self.assertEqual(perf_ab.verdict([1.04, 1.06, 1.05]), "faster")

    def test_verdict_slower(self) -> None:
        self.assertEqual(perf_ab.verdict([0.95, 0.93, 0.94]), "slower")

    def test_verdict_same(self) -> None:
        self.assertEqual(perf_ab.verdict([0.99, 1.01, 1.00]), "same")

    def test_verdict_noise_is_inconclusive(self) -> None:
        self.assertEqual(perf_ab.verdict([0.93, 1.08, 1.01]), "inconclusive")

    def test_verdict_wide_spread_is_inconclusive(self) -> None:
        # All three are above 1 + SAME_BAND, but they disagree by 0.17: not a speed-up to report.
        self.assertEqual(perf_ab.verdict([1.03, 1.10, 1.20]), "inconclusive")

    def test_verdict_same_needs_every_block_in_the_band(self) -> None:
        # The median (1.019) is inside the same band, but the third block (1.06) is a 6% speed-up: not "same".
        self.assertEqual(perf_ab.verdict([1.019, 1.019, 1.06]), "inconclusive")
        self.assertEqual(perf_ab.verdict([0.981, 0.981, 0.95]), "inconclusive")

    def test_verdict_tight_but_between_bands_is_inconclusive(self) -> None:
        # Median 1.03 is outside the same band, and one ratio (1.015) is not above 1 + SAME_BAND.
        self.assertEqual(perf_ab.verdict([1.015, 1.03, 1.04]), "inconclusive")

    def test_verdict_nan_zero_single(self) -> None:
        for ratios in (
            [],
            [1.1],
            [1.1, math.nan],
            [1.1, math.inf],
            [1.1, 0.0],
            [1.1, -1.2],
            [math.nan, math.nan],
        ):
            with self.subTest(ratios=ratios):
                self.assertEqual(perf_ab.verdict(ratios), "inconclusive")

    def test_verdict_two_ratios_suffice(self) -> None:
        self.assertEqual(perf_ab.verdict([1.05, 1.06]), "faster")


class RunTest(unittest.TestCase):
    """`run` drives the two binaries through an injected runner that plays the part of `nibble bench`."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = str(Path(self.tmp.name) / "ab.jsonl")
        self.calls: list[list[str]] = []

    def fake_bench(self, argv: Sequence[str]) -> None:
        """Write what `nibble bench -m M -t T ... --out F` would: a pp row and a tg row; speed depends on the binary."""
        self.calls.append(list(argv))
        speed = {"base-bin": 10.0, "head-bin": 12.0}[argv[0]]
        model, threads, out = argv[argv.index("-m") + 1], int(argv[argv.index("-t") + 1]), argv[argv.index("--out") + 1]
        with open(out, "a", encoding="utf-8") as f:
            for test in ("pp512", "tg128"):
                row = {"tool": "nibble", "model": Path(model).name, "test": test, "threads": threads, "median": speed}
                f.write(json.dumps(row) + "\n")

    def test_run_tags_rows_and_orders_the_binaries_abba(self) -> None:
        perf_ab.run(
            "base-bin", "head-bin", ["dir/a.gguf", "dir/b.gguf"], [1, 4], 2, 3, self.out, runner=self.fake_bench
        )
        # 2 rounds x 2 models x 2 thread counts x 4 runs.
        self.assertEqual(len(self.calls), 32)
        self.assertEqual([c[0] for c in self.calls[:4]], ["base-bin", "head-bin", "head-bin", "base-bin"])
        # Rounds are interleaved: round 0 covers every (model, threads) before round 1 starts.
        self.assertEqual([(c[c.index("-m") + 1], c[c.index("-t") + 1]) for c in self.calls[::4]][:4], [
            ("dir/a.gguf", "1"), ("dir/a.gguf", "4"), ("dir/b.gguf", "1"), ("dir/b.gguf", "4"),
        ])  # fmt: skip
        for call in self.calls:
            self.assertEqual(call[1], "bench")
            self.assertEqual(call[call.index("-p") + 1], "512")
            self.assertEqual(call[call.index("-n") + 1], "128")
            self.assertEqual(call[call.index("-r") + 1], "3")

        rows = [json.loads(line) for line in Path(self.out).read_text().splitlines()]
        self.assertEqual(len(rows), 64)
        first = rows[:2]
        self.assertEqual({(r["side"], r["block"], r["position"]) for r in first}, {("base", 0, "A1")})
        self.assertEqual(
            [(r["side"], r["position"]) for r in rows[2:8:2]], [("head", "B1"), ("head", "B2"), ("base", "A2")]
        )
        ratios = perf_ab.block_ratios(rows)
        self.assertEqual(len(ratios), 8)  # 2 models x 2 thread counts x pp512/tg128
        self.assertTrue(all(v == [1.2, 1.2] for v in ratios.values()))
        self.assertEqual(perf_ab.verdict(ratios[("a.gguf", "tg128", 4)]), "faster")

    def test_run_starts_a_fresh_file(self) -> None:
        # A second session into the same file (a local re-run, a retry after a partial run) must not add its blocks
        # to the first one's: block ids restart at 0, and doubled blocks would read as inconclusive.
        Path(self.out).write_text('{"old": 1}\n')
        perf_ab.run("base-bin", "head-bin", ["a.gguf"], [1], 2, 1, self.out, runner=self.fake_bench)
        perf_ab.run("base-bin", "head-bin", ["a.gguf"], [1], 2, 1, self.out, runner=self.fake_bench)
        rows = [json.loads(line) for line in Path(self.out).read_text().splitlines()]
        self.assertEqual(len(rows), 16)  # 2 rounds x 4 runs x pp512/tg128, from the second session only
        self.assertNotIn({"old": 1}, rows)
        self.assertTrue(all(e["verdict"] == "faster" for e in perf_ab.summarize(rows)))

    def test_run_refuses_an_invocation_that_wrote_no_row_for_a_test(self) -> None:
        # A binary that exits 0 but writes no tg128 row would let a whole round vanish from the ratios.
        def pp_only(argv: Sequence[str]) -> None:
            out = argv[argv.index("--out") + 1]
            row = {"model": "a.gguf", "test": "pp512", "threads": 1, "median": 10.0}
            Path(out).write_text(json.dumps(row) + "\n")

        with self.assertRaisesRegex(RuntimeError, "tg128"):
            perf_ab.run("base-bin", "head-bin", ["a.gguf"], [1], 1, 1, self.out, runner=pp_only)

    def test_run_propagates_a_failing_binary(self) -> None:
        def boom(argv: Sequence[str]) -> None:
            raise RuntimeError("bench failed")

        with self.assertRaises(RuntimeError):
            perf_ab.run("base-bin", "head-bin", ["a.gguf"], [1], 1, 1, self.out, runner=boom)


def _block(block: int, base: tuple[float, float], head: tuple[float, float], **kw: object) -> list[dict[str, object]]:
    """The four rows of one ABBA block (A1 B1 B2 A2) with the given base and head medians."""
    return [
        {**_row(side, block, med), **kw}
        for side, med in (("base", base[0]), ("head", head[0]), ("head", head[1]), ("base", base[1]))
    ]


class SummarizeTest(unittest.TestCase):
    def test_summarize_two_rounds(self) -> None:
        rows = _block(0, (10.0, 12.0), (11.0, 11.0)) + _block(1, (10.0, 10.0), (10.0, 10.0))
        rows += [{**r, "model": "a.gguf", "test": "pp512"} for r in _block(0, (4.0, 4.0), (6.0, 6.0))]
        self.assertEqual(
            perf_ab.summarize(rows),
            [
                {
                    "model": "a.gguf",
                    "test": "pp512",
                    "threads": 1,
                    "base_median": 4.0,
                    "head_median": 6.0,
                    "ratios": [1.5, 0.0],  # block 1 has tg128 rows but no pp512 row: a missing block, not dropped
                    "verdict": "inconclusive",
                },
                {
                    "model": "m.gguf",
                    "test": "tg128",
                    "threads": 1,
                    "base_median": 10.0,
                    "head_median": 10.5,
                    "ratios": [1.0, 1.0],
                    "verdict": "same",
                },
            ],
        )

    def test_summarize_bad_medians_stay_json_safe_and_inconclusive(self) -> None:
        rows = _block(0, (10.0, 10.0), (0.0, math.nan)) + _block(1, (10.0, 10.0), (10.0, 10.0))
        (entry,) = perf_ab.summarize(rows)
        self.assertEqual(entry["ratios"], [0.0, 1.0])  # the broken block reads 0.0, as the report schema has it
        self.assertEqual(entry["head_median"], 10.0)  # only usable runs count
        self.assertEqual(entry["verdict"], "inconclusive")
        json.dumps(entry, allow_nan=False)  # must not raise

    def test_summarize_empty(self) -> None:
        self.assertEqual(perf_ab.summarize([]), [])


class MainTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def test_summarize_round_trip(self) -> None:
        rows = _block(0, (10.0, 10.0), (12.0, 12.0)) + _block(1, (10.0, 10.0), (12.0, 12.0))
        (self.dir / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows) + "\n")
        out = self.dir / "perf.json"
        self.assertEqual(perf_ab.main(["summarize", str(self.dir / "rows.jsonl"), "--out", str(out)]), 0)
        (entry,) = json.loads(out.read_text())
        self.assertEqual(entry["verdict"], "faster")
        self.assertEqual(entry["ratios"], [1.2, 1.2])

    def test_run_parses_arguments(self) -> None:
        seen: list[tuple[object, ...]] = []

        def fake_run(*args: object, **kw: object) -> None:
            seen.append(args)

        with unittest.mock.patch.object(perf_ab, "run", fake_run):
            code = perf_ab.main(
                ["run", "--base", "b/nibble", "--head", "h/nibble", "--model", "x.gguf", "--model", "y.gguf"]
                + ["--threads", "1,4", "--rounds", "2", "--reps", "5", "--out", "rows.jsonl"]
            )
        self.assertEqual(code, 0)
        self.assertEqual(seen, [("b/nibble", "h/nibble", ["x.gguf", "y.gguf"], [1, 4], 2, 5, "rows.jsonl")])

    def test_run_with_a_fake_bench_script(self) -> None:
        bench = self.dir / "fakebin"
        bench.write_text(
            "#!/usr/bin/env python3\nimport sys, json\na = sys.argv\n"
            "for t in ('pp512', 'tg128'):\n"
            "    open(a[a.index('--out') + 1], 'a').write(json.dumps({'model': 'm.gguf', 'test': t, "
            "'threads': int(a[a.index('-t') + 1]), 'median': 5.0}) + '\\n')\n"
        )
        bench.chmod(0o755)
        out = self.dir / "rows.jsonl"
        args = ["run", "--base", str(bench), "--head", str(bench), "--model", "m.gguf", "--threads", "2"]
        self.assertEqual(perf_ab.main(args + ["--rounds", "1", "--reps", "1", "--out", str(out)]), 0)
        rows = [json.loads(line) for line in out.read_text().splitlines()]
        self.assertEqual([(r["side"], r["position"]) for r in rows[::2]], [(s, p) for p, s in perf_ab._POSITIONS])

    def test_bad_threads_is_an_argument_error(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            perf_ab.main(["run", "--base", "a", "--head", "b", "--model", "m", "--threads", "0,x", "--out", "o"])


if __name__ == "__main__":
    unittest.main()
