"""Unit tests for tools/ci/perf_ab.py: ABBA ordering, per-block ratios, the verdict rules and the run driver.

Run with: python -m unittest discover -s tools/ci -p 'test_*.py'
`run` is tested with a fake `nibble bench` (an injected runner) that writes the JSONL a real one would, so no
binary or model is needed.
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
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

    def test_run_appends_to_an_existing_file(self) -> None:
        Path(self.out).write_text('{"old": 1}\n')
        perf_ab.run("base-bin", "head-bin", ["a.gguf"], [1], 1, 1, self.out, runner=self.fake_bench)
        lines = Path(self.out).read_text().splitlines()
        self.assertEqual(lines[0], '{"old": 1}')
        self.assertEqual(len(lines), 1 + 8)

    def test_run_propagates_a_failing_binary(self) -> None:
        def boom(argv: Sequence[str]) -> None:
            raise RuntimeError("bench failed")

        with self.assertRaises(RuntimeError):
            perf_ab.run("base-bin", "head-bin", ["a.gguf"], [1], 1, 1, self.out, runner=boom)


if __name__ == "__main__":
    unittest.main()
