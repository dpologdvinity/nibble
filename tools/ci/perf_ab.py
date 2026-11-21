"""ABBA speed comparison of two nibble builds (a PR's base and head) on one machine.

A shared CI runner drifts: clock speed and neighbours' load change over minutes. So, like
`tools/bench_final.sh --abba`, each (model, threads) gets blocks of four runs in the order base, head, head, base
(positions A1 B1 B2 A2; A is the base build, B the head build), and rounds are interleaved (every (model, threads)
gets its first block before any gets its second). Drift that is linear in time cancels in a block's ratio

    ratio = (B1 + B2) / (A1 + A2)        of the medians in tok/s, i.e. head speed / base speed

`run` drives `<bin> bench -m M -t T -p 512 -n 128 -r <reps> --out <tmp>` for each position and writes the rows to a
fresh JSONL file (one session per file: block ids start at 0 every time, so a second session appended to the first
would double its blocks), adding "side" (base|head), "block" (the round, from 0) and "position" (A1|B1|B2|A2).
`nibble bench` itself writes "model" (the file's base name), "test" (pp512, tg128), "threads" and "median".
`block_ratios` turns such rows into one ratio per block for every (model, test, threads), and `verdict` classifies
the list of ratios.

The verdict is deliberately conservative, because a runner's +-10% jitter must never read as a regression:
fewer than two usable blocks, a block with a missing, zero, negative or non-finite median, or ratios that disagree
by more than SPREAD_MAX are all "inconclusive". Only when every block agrees is the answer faster, slower or same.

`summarize` folds the rows into the `perf` section of the PR-report JSON (one entry per (model, test, threads):
base_median, head_median, ratios, verdict). The report schema has plain numbers only, so a median with no usable run
and a broken block's ratio are written as 0.0 (the verdict for such a row is "inconclusive").

usage (command line):
    python tools/ci/perf_ab.py run --base BIN --head BIN --model M [--model M ...] --threads 1,4 --rounds 3 \\
        --reps 3 --out rows.jsonl
    python tools/ci/perf_ab.py summarize rows.jsonl --out perf.json
usage (library): import perf_ab; perf_ab.run("base/nibble", "head/nibble", models, [1, 4], 3, 3, "ab.jsonl")
    rows = [json.loads(line) for line in open("ab.jsonl")]
    for key, ratios in perf_ab.block_ratios(rows).items(): print(key, perf_ab.verdict(ratios))
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal

Side = Literal["base", "head"]
Verdict = Literal["faster", "slower", "same", "inconclusive"]

SAME_BAND = 0.02  # every |ratio - 1| below this, with spread <= SPREAD_MAX, is "same"
SPREAD_MAX = 0.05  # ratios of one row that differ by more than this (max - min) are noise, not a result

# One ABBA block: (position, side). A is the base build, B the head build.
_POSITIONS: tuple[tuple[str, Side], ...] = (("A1", "base"), ("B1", "head"), ("B2", "head"), ("A2", "base"))

# `nibble bench` arguments fixed by the protocol (the same sizes as tools/bench_final.sh).
_PROMPT_TOKENS = "512"
_GEN_TOKENS = "128"


def abba_order(rounds: int) -> list[tuple[int, str, Side]]:
    """Return the run order `(block, position, side)` for `rounds` blocks: A1 B1 B2 A2 each, block ids from 0."""
    return [(block, pos, side) for block in range(rounds) for pos, side in _POSITIONS]


def _positive_finite(x: object) -> float | None:
    """Return `x` as a float if it is a real, finite, positive number (not a bool), else None."""
    if isinstance(x, bool) or not isinstance(x, int | float):
        return None
    v = float(x)
    return v if math.isfinite(v) and v > 0 else None


def block_ratios(rows: Sequence[Mapping[str, object]]) -> dict[tuple[str, str, int], list[float]]:
    """Group `rows` by (model, test, threads) and return head/base speed ratios per block, in block order.

    A block's ratio is (B1 + B2) / (A1 + A2) over the "median" fields. A block that lacks a side, or has a median
    that is missing, not a number, non-finite or <= 0, contributes NaN (which `verdict` reads as inconclusive)
    instead of being dropped or raising: a broken run must not vanish into a better-looking result. Rows whose
    side, block, model, test or threads cannot be read are ignored, since they cannot be attributed to a block.
    """
    # (model, test, threads) -> block -> side -> medians (None marks an unusable median)
    groups: dict[tuple[str, str, int], dict[int, dict[Side, list[float | None]]]] = {}
    for row in rows:
        side, block = row.get("side"), row.get("block")
        model, test, threads = row.get("model"), row.get("test"), row.get("threads")
        if side not in ("base", "head") or not isinstance(model, str) or not isinstance(test, str):
            continue
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in (block, threads)):
            continue
        assert isinstance(block, int) and isinstance(threads, int)  # narrowed by the check above, for mypy
        sides = groups.setdefault((model, test, threads), {}).setdefault(block, {"base": [], "head": []})
        sides[side].append(_positive_finite(row.get("median")))

    result: dict[tuple[str, str, int], list[float]] = {}
    for key, blocks in groups.items():
        ratios: list[float] = []
        for block in sorted(blocks):
            base, head = blocks[block]["base"], blocks[block]["head"]
            usable = len(base) == 2 and len(head) == 2 and None not in base and None not in head
            if usable:
                ratios.append(sum(v for v in head if v is not None) / sum(v for v in base if v is not None))
            else:
                ratios.append(math.nan)
        result[key] = ratios
    return result


def verdict(ratios: Sequence[float]) -> Verdict:
    """Classify per-block head/base ratios (see the module docstring); noise is "inconclusive", never "slower"."""
    if len(ratios) < 2 or any(not math.isfinite(r) or r <= 0 for r in ratios):
        return "inconclusive"
    # Spread first: blocks that disagree (1.03, 1.10, 1.20 are all "above 1.02") are noise, not a speed-up.
    if max(ratios) - min(ratios) > SPREAD_MAX:
        return "inconclusive"
    if all(r > 1 + SAME_BAND for r in ratios):
        return "faster"
    if all(r < 1 - SAME_BAND for r in ratios):
        return "slower"
    # Every block, not just the median: [1.019, 1.019, 1.06] has a block that is 6% faster, which is not "same".
    if all(abs(r - 1) < SAME_BAND for r in ratios):
        return "same"
    return "inconclusive"


def _run_subprocess(argv: Sequence[str]) -> None:
    """Run one benchmark command, raising CalledProcessError if it fails. Its console table is passed through."""
    subprocess.run(list(argv), check=True)


def run(
    base: str,
    head: str,
    models: Sequence[str],
    threads: Sequence[int],
    rounds: int,
    reps: int,
    out: str,
    *,
    runner: Callable[[Sequence[str]], None] = _run_subprocess,
) -> None:
    """Benchmark `base` and `head` binaries in interleaved ABBA blocks and write the tagged rows to `out`.

    For each round (block id), each model and each thread count, the four runs of `abba_order` call
    `<bin> bench -m <model> -t <threads> -p 512 -n 128 -r <reps> --out <tmp>`; each row the run wrote gets
    "side", "block" and "position" added and goes to the JSONL file `out`, which is truncated first (rows of an
    earlier session would share this session's block ids). `runner`
    executes a command (tests inject a fake `nibble bench`); a failing run raises, so no half-measured block is
    reported as a result. No load gate: unlike a dedicated box, a CI runner is never idle, which is why `verdict`
    tolerates noise.
    """
    binaries: dict[Side, str] = {"base": base, "head": head}
    order = abba_order(rounds)
    with tempfile.TemporaryDirectory() as tmpdir, open(out, "w", encoding="utf-8") as sink:
        tmp = Path(tmpdir) / "bench.jsonl"
        for block in range(rounds):
            for model in models:
                for t in threads:
                    for blk, position, side in order[block * len(_POSITIONS) : (block + 1) * len(_POSITIONS)]:
                        tmp.unlink(missing_ok=True)
                        runner(
                            [binaries[side], "bench", "-m", model, "-t", str(t), "-p", _PROMPT_TOKENS]
                            + ["-n", _GEN_TOKENS, "-r", str(reps), "--out", str(tmp)]
                        )
                        lines = tmp.read_text(encoding="utf-8").splitlines() if tmp.exists() else []
                        for row in (json.loads(line) for line in lines if line.strip()):
                            row.update({"side": side, "block": blk, "position": position})
                            sink.write(json.dumps(row) + "\n")
                        sink.flush()


def _json_number(x: float) -> float:
    """Return `x`, or 0.0 if it is NaN or infinite, so the value survives strict JSON."""
    return x if math.isfinite(x) else 0.0


def summarize(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Return one report entry per (model, test, threads), sorted by that key.

    Each entry is {"model", "test", "threads", "base_median", "head_median", "ratios", "verdict"}. A side's median
    is the median of its usable per-run medians (finite, > 0); `ratios` and `verdict` come from `block_ratios` and
    `verdict`. Values that are NaN (no usable run, a broken block) are written as 0.0.
    """
    per_side: dict[tuple[str, str, int], dict[str, list[float]]] = {}
    for row in rows:
        model, test, threads, side = row.get("model"), row.get("test"), row.get("threads"), row.get("side")
        value = _positive_finite(row.get("median"))
        if (
            isinstance(model, str)
            and isinstance(test, str)
            and isinstance(threads, int)
            and not isinstance(threads, bool)
            and side in ("base", "head")
            and value is not None
        ):
            per_side.setdefault((model, test, threads), {"base": [], "head": []})[side].append(value)

    ratios = block_ratios(rows)
    entries: list[dict[str, object]] = []
    for key in sorted(ratios):
        sides = per_side.get(key, {"base": [], "head": []})
        entries.append(
            {
                "model": key[0],
                "test": key[1],
                "threads": key[2],
                "base_median": statistics.median(sides["base"]) if sides["base"] else 0.0,
                "head_median": statistics.median(sides["head"]) if sides["head"] else 0.0,
                "ratios": [_json_number(r) for r in ratios[key]],
                "verdict": verdict(ratios[key]),
            }
        )
    return entries


def _thread_list(text: str) -> list[int]:
    """Parse "1,4" into [1, 4]; every item must be a positive integer (argparse type)."""
    try:
        values = [int(part) for part in text.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a comma-separated list of integers: {text!r}") from None
    if any(v < 1 for v in values):
        raise argparse.ArgumentTypeError(f"thread counts must be >= 1: {text!r}")
    return values


def main(argv: Sequence[str]) -> int:
    """Command line: `run` benchmarks two builds, `summarize` turns the rows into the report's perf JSON."""
    parser = argparse.ArgumentParser(description="ABBA speed comparison of two nibble builds.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="benchmark base and head in interleaved ABBA blocks")
    p_run.add_argument("--base", required=True, help="the base build's nibble binary")
    p_run.add_argument("--head", required=True, help="the head build's nibble binary")
    p_run.add_argument("--model", action="append", required=True, dest="models", help="a GGUF file (repeatable)")
    p_run.add_argument("--threads", type=_thread_list, default=[1], help="comma-separated thread counts")
    p_run.add_argument("--rounds", type=int, default=3, help="ABBA blocks per (model, threads)")
    p_run.add_argument("--reps", type=int, default=3, help="repetitions per `nibble bench` run")
    p_run.add_argument("--out", required=True, help="JSONL file the tagged rows are written to (replaced)")
    p_sum = sub.add_parser("summarize", help="fold the rows into the report's perf section")
    p_sum.add_argument("rows", help="JSONL file written by `run`")
    p_sum.add_argument("--out", required=True, help="JSON file to write")
    args = parser.parse_args(list(argv))

    if args.command == "run":
        if args.rounds < 1 or args.reps < 1:
            parser.error("--rounds and --reps must be >= 1")
        run(args.base, args.head, args.models, args.threads, args.rounds, args.reps, args.out)
        return 0
    lines = Path(args.rows).read_text(encoding="utf-8").splitlines()
    summary = summarize([json.loads(line) for line in lines if line.strip()])
    Path(args.out).write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
