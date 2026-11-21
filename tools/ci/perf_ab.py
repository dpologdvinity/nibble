"""ABBA speed comparison of two nibble builds (a PR's base and head) on one machine.

A shared CI runner drifts: clock speed and neighbours' load change over minutes. So, like
`tools/bench_final.sh --abba`, each (model, threads) gets blocks of four runs in the order base, head, head, base
(positions A1 B1 B2 A2; A is the base build, B the head build), and rounds are interleaved (every (model, threads)
gets its first block before any gets its second). Drift that is linear in time cancels in a block's ratio

    ratio = (B1 + B2) / (A1 + A2)        of the medians in tok/s, i.e. head speed / base speed

`run` drives `<bin> bench -m M -t T -p 512 -n 128 -r <reps> --out <tmp>` for each position and appends the rows to a
JSONL file, adding "side" (base|head), "block" (the round, from 0) and "position" (A1|B1|B2|A2). `nibble bench`
itself writes "model" (the file's base name), "test" (pp512, tg128), "threads" and "median". `block_ratios` turns
such rows into one ratio per block for every (model, test, threads), and `verdict` classifies the list of ratios.

The verdict is deliberately conservative, because a runner's +-10% jitter must never read as a regression:
fewer than two usable blocks, a block with a missing, zero, negative or non-finite median, or ratios that disagree
by more than SPREAD_MAX are all "inconclusive". Only when every block agrees is the answer faster, slower or same.

usage (library): import perf_ab; perf_ab.run("base/nibble", "head/nibble", models, [1, 4], 3, 3, "ab.jsonl")
    rows = [json.loads(line) for line in open("ab.jsonl")]
    for key, ratios in perf_ab.block_ratios(rows).items(): print(key, perf_ab.verdict(ratios))
"""

from __future__ import annotations

import json
import math
import statistics
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal

Side = Literal["base", "head"]
Verdict = Literal["faster", "slower", "same", "inconclusive"]

SAME_BAND = 0.02  # |median ratio - 1| below this, with spread <= SPREAD_MAX, is "same"
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
    if abs(statistics.median(ratios) - 1) < SAME_BAND:
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
    """Benchmark `base` and `head` binaries in interleaved ABBA blocks and append the tagged rows to `out`.

    For each round (block id), each model and each thread count, the four runs of `abba_order` call
    `<bin> bench -m <model> -t <threads> -p 512 -n 128 -r <reps> --out <tmp>`; each row the run wrote gets
    "side", "block" and "position" added and is appended to the JSONL file `out` (created if absent). `runner`
    executes a command (tests inject a fake `nibble bench`); a failing run raises, so no half-measured block is
    reported as a result. No load gate: unlike a dedicated box, a CI runner is never idle, which is why `verdict`
    tolerates noise.
    """
    binaries: dict[Side, str] = {"base": base, "head": head}
    order = abba_order(rounds)
    with tempfile.TemporaryDirectory() as tmpdir, open(out, "a", encoding="utf-8") as sink:
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
