"""The PR report: validate the `pr-report.json` artifact, render it as one Markdown comment and post it.

A pull request workflow (read-only token) runs the speed, size, memory and accuracy jobs; each writes a partial JSON
file and `assemble` merges them into `pr-report.json`. A separate `workflow_run` job that holds a write token
downloads that artifact and calls `post`. The artifact was written by the pull request's own code, so for a fork it
is untrusted input to a privileged job. Three rules keep that safe:

* `load` accepts exactly the schema below: exact key sets, exact types (booleans are not numbers), finite
  non-negative numbers, bounded lengths and counts, printable ASCII text, 40-hex SHAs, model names from
  MODEL_ALLOWLIST, no duplicate keys, bounded input size. Anything else raises ValueError.
* The pull request is never named by the artifact (the schema has no such field). `post` finds it among the
  OPEN pulls by the triggering run's head SHA (`GitHub.pr_for_sha`) and refuses a report whose own `head_sha` differs
  from that SHA, so an artifact cannot be replayed onto another pull request.
* `render` builds Markdown only from validated fields, and backslash-escapes every ASCII punctuation character in
  free text (keys, runner), so nothing a report says can start a link, a mention, HTML or a new table column. The
  comment is edited in place only when the bot wrote it (a planted marker from another user is ignored).

Schema 1 (`null` for a section means "skipped"; numbers are JSON numbers, 0.0 means "no usable value"):

    {"schema": 1, "base_sha": "<40 hex>", "head_sha": "<40 hex>", "runner": "<1-120 chars of [A-Za-z0-9 ._()/,+-]>",
     "kernels_changed": bool,
     "perf": [{"model": <allowlisted>, "test": "pp512"|"tg128", "threads": int, "base_median": x,
               "head_median": x, "ratios": [x...], "verdict": "faster"|"slower"|"same"|"inconclusive"}] | null,
     "size": {"base_bytes": int, "head_bytes": int} | null,
     "rss": {"base_kib": int, "head_kib": int} | null,
     "accuracy": {"status": "ran"|"skipped-cold-cache", "head_tests": "passed"|"failed"|null,
                  "rows": [{"key": "<model>/<prompt>", "base": x|null, "head": x|null}]}}   # rows only when "ran"

`head_tests` is how the head's end-to-end tests (`ctest -L e2e`) ended: given exactly when the status is "ran", null
otherwise. It is separate from the rows because a test prints its min cosine before its later checks (batch prefill,
greedy continuation), so a head can fail those with every row unchanged; the comment then says the tests failed.

A row's `base` or `head` is null for a case that has a value on one side only (rendered "no base result" / "no head
result", because a case that failed to produce a value is not necessarily gone; never both null). `key` is
`[A-Za-z0-9._/+-]`, so no `@`, `#`, markup or whitespace reaches the comment: GitHub turns even a backslash-escaped
at-sign or hash mark into a mention or an issue link, so those characters are refused, not escaped.

`assemble` takes partial files: a JSON object holding some of the keys above (each key in at most one part), or a
bare JSON list, which is the `perf` section exactly as `perf_ab.py summarize` writes it. `base_sha`, `head_sha`,
`runner`, `kernels_changed` and `accuracy` are required (a missing accuracy part is an error, never "cold cache":
the cold case is the explicit status `skipped-cold-cache`); `perf`, `size` and `rss` default to null (skipped). The
runner is made inert first (`@` becomes "at", other unsafe characters "_", cut to 120), because a CPU model name
such as "Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz" is what a runner reports. The `accuracy` subcommand builds
the accuracy part from two nightly-format cosine snapshots (`e2e-cosines.json`).

usage:
    python tools/ci/pr_report.py accuracy --base base-cosines.json --head head-cosines.json --head-outcome success \\
        --out part-accuracy.json
    python tools/ci/pr_report.py accuracy --skipped-cold-cache --out part-accuracy.json
    python tools/ci/pr_report.py assemble meta.json perf.json part-accuracy.json --out pr-report.json
    python tools/ci/pr_report.py post pr-report.json --head-sha "$HEAD_SHA"   # GITHUB_REPOSITORY and GH_TOKEN set
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypedDict

from gh import GhCli, GhError, GitHub

# Hidden first line of the comment; `post` edits the bot's comment that holds it instead of adding a new one.
MARKER = "<!-- nibble-pr-report -->"
# Login `post` expects the existing report comment to have been written by (the workflow's GITHUB_TOKEN user).
BOT_LOGIN = "github-actions[bot]"
# The only models the report may name; they match the files the CI speed job downloads.
MODEL_ALLOWLIST = frozenset({"qwen2.5-0.5b-instruct-q4_0.gguf", "qwen2.5-0.5b-instruct-q8_0.gguf"})

SCHEMA = 1
TESTS = frozenset({"pp512", "tg128"})
VERDICTS = ("faster", "slower", "same", "inconclusive")
STATUSES = ("ran", "skipped-cold-cache")
HEAD_TESTS = ("passed", "failed")
# `steps.<id>.outcome` of the head's e2e step -> head_tests. Only success is a pass; anything else (failure, and a
# cancelled or skipped step that should never get here) reads as failed, so the report can never claim a pass.
_OUTCOMES = {"success": "passed", "failure": "failed", "cancelled": "failed", "skipped": "failed"}
VERDICT_SYMBOLS = {"faster": "✅", "slower": "⚠️", "same": "➖", "inconclusive": "❔"}

# Bounds. They also cap the rendered comment: the worst case stays well below GitHub's 65,536 character limit
# (pinned by test_render_of_maximal_report_fits_a_comment).
MAX_BYTES = 1_000_000  # size of the artifact text
MAX_RUNNER = 120
MAX_KEY = 120
MAX_PERF = 64
MAX_ROWS = 128
MAX_RATIOS = 64
MAX_THREADS = 4096
MAX_INT = 2**50  # sizes in bytes / KiB: far above any real value, far below float precision limits
MAX_FLOAT = 1e12  # tokens per second, ratios and cosines

_SHA = re.compile(r"[0-9a-f]{40}")
# Allowed text. No `@` and no `#` (mentions and issue links), no markup characters, no control characters.
_KEY_TEXT = re.compile(r"[A-Za-z0-9._/+-]+")  # `<model>/<prompt>`: ids and prompt names, never whitespace
_RUNNER_TEXT = re.compile(r"[A-Za-z0-9 ._()/,+-]+")  # a CPU model name, once `assemble` has made it inert
_RUNNER_UNSAFE = re.compile(r"[^A-Za-z0-9 ._()/,+-]")
_PUNCTUATION = re.compile(r"([!-/:-@\[-`{-~])")

_TOP_KEYS = frozenset(
    {"schema", "base_sha", "head_sha", "runner", "kernels_changed", "perf", "size", "rss", "accuracy"}
)
_SECTIONS = ("perf", "size", "rss", "accuracy")
_META_KEYS = ("base_sha", "head_sha", "runner", "kernels_changed")


class PerfEntry(TypedDict):
    """One (model, test, threads) speed comparison, as written by perf_ab.summarize."""

    model: str
    test: str
    threads: int
    base_median: float
    head_median: float
    ratios: list[float]
    verdict: str


class Size(TypedDict):
    """Size of the nibble binary in bytes."""

    base_bytes: int
    head_bytes: int


class Rss(TypedDict):
    """Peak resident set size in KiB of one `nibble run`."""

    base_kib: int
    head_kib: int


class AccuracyRow(TypedDict):
    """Minimum cosine similarity against the golden logits for `key` ("<model>/<prompt>").

    `base` or `head` is None for a case that exists on one side only (never both).
    """

    key: str
    base: float | None
    head: float | None


class Accuracy(TypedDict):
    """The accuracy section: `rows` is empty and `head_tests` None unless `status` is "ran"."""

    status: str
    head_tests: str | None
    rows: list[AccuracyRow]


class Report(TypedDict):
    """A validated report (see the module docstring). Only `load` should construct one from untrusted text."""

    schema: int
    base_sha: str
    head_sha: str
    runner: str
    kernels_changed: bool
    perf: list[PerfEntry] | None
    size: Size | None
    rss: Rss | None
    accuracy: Accuracy


# ---------------------------------------------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------------------------------------------


def _show(value: object) -> str:
    """A short, single-line, ASCII-only rendering of untrusted `value` for an error message.

    `ascii()` escapes newlines and quotes the text, so an error can never begin a line with `::` (a workflow
    command) however hostile the input.
    """
    return ascii(value)[:60]


def _no_constant(name: str) -> Any:
    """json `parse_constant` hook: NaN, Infinity and -Infinity are not JSON and are refused."""
    raise ValueError(f"non-finite number {name} is not allowed")


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """json `object_pairs_hook`: build the dict but refuse a repeated key (json would silently keep the last)."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {_show(key)}")
        result[key] = value
    return result


def read_limited(path: str, limit: int = MAX_BYTES) -> str:
    """Read the UTF-8 file at `path`, reading at most `limit` + 1 bytes; ValueError if it is larger than `limit`.

    The artifact is untrusted: its size is checked before the whole file is read, so a huge or endless file cannot
    exhaust memory. A decoding error is a ValueError too.
    """
    with open(path, "rb") as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"{os.path.basename(path)}: larger than {limit} bytes")
    return data.decode("utf-8")


def _parse(text: str, what: str) -> Any:
    """Parse untrusted JSON text strictly; every failure is a ValueError that names `what`."""
    if len(text) > MAX_BYTES:  # characters; read_limited has already bounded the bytes of a file
        raise ValueError(f"{what}: larger than {MAX_BYTES} bytes")
    try:
        return json.loads(text, parse_constant=_no_constant, object_pairs_hook=_no_duplicates)
    except RecursionError:
        raise ValueError(f"{what}: nested too deeply") from None
    except ValueError as exc:  # includes json.JSONDecodeError and the hooks' errors
        raise ValueError(f"{what}: {_show(str(exc))}") from None


def _object(value: object, keys: Sequence[str] | frozenset[str], where: str) -> dict[str, Any]:
    """`value` must be a JSON object whose key set is exactly `keys`."""
    if not isinstance(value, dict):
        raise ValueError(f"{where}: expected an object")
    missing = set(keys) - value.keys()
    extra = value.keys() - set(keys)
    if missing or extra:
        raise ValueError(
            f"{where}: missing keys {sorted(map(_show, missing))}, unexpected keys {sorted(map(_show, extra))}"
        )
    return value


def _int(value: object, where: str, low: int, high: int) -> int:
    """A JSON integer in [low, high]. Booleans are not integers here, and 1.0 is not an integer."""
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{where}: expected an integer in [{low}, {high}], got {_show(value)}")
    return value


def _number(value: object, where: str, low: float = 0.0, high: float = MAX_FLOAT) -> float:
    """A finite JSON number in [low, high] (an int is accepted and converted). 0.0 is allowed (ruling P16)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}: expected a number, got {_show(value)}")
    if isinstance(value, int):
        if abs(value) > MAX_INT:  # reject before float(): a huge int would raise OverflowError
            raise ValueError(f"{where}: out of range")
        value = float(value)
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{where}: expected a finite number in [{low}, {high}], got {_show(value)}")
    return value


def _text(value: object, where: str, max_len: int, allowed: re.Pattern[str]) -> str:
    """1..max_len characters, all matching `allowed` (a pattern for one or more allowed characters)."""
    if not isinstance(value, str) or not 1 <= len(value) <= max_len or not allowed.fullmatch(value):
        raise ValueError(f"{where}: expected 1-{max_len} characters from {allowed.pattern}, got {_show(value)}")
    return value


def _choice(value: object, where: str, allowed: Sequence[str] | frozenset[str]) -> str:
    """One of the strings in `allowed`."""
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{where}: not an allowed value: {_show(value)}")
    return value


def _sha(value: object, where: str) -> str:
    """A lower-case 40-hex commit id (fullmatch: `$` would let a trailing newline through)."""
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ValueError(f"{where}: expected 40 lower-case hex digits, got {_show(value)}")
    return value


def _list(value: object, where: str, max_len: int) -> list[Any]:
    """A JSON array of at most `max_len` items."""
    if not isinstance(value, list) or len(value) > max_len:
        raise ValueError(f"{where}: expected an array of at most {max_len} items")
    return value


def _perf_entry(value: object, where: str) -> PerfEntry:
    o = _object(value, ("model", "test", "threads", "base_median", "head_median", "ratios", "verdict"), where)
    return {
        "model": _choice(o["model"], f"{where}.model", MODEL_ALLOWLIST),
        "test": _choice(o["test"], f"{where}.test", TESTS),
        "threads": _int(o["threads"], f"{where}.threads", 1, MAX_THREADS),
        "base_median": _number(o["base_median"], f"{where}.base_median"),
        "head_median": _number(o["head_median"], f"{where}.head_median"),
        "ratios": [_number(r, f"{where}.ratios[]") for r in _list(o["ratios"], f"{where}.ratios", MAX_RATIOS)],
        "verdict": _choice(o["verdict"], f"{where}.verdict", VERDICTS),
    }


def _cosine(value: object, where: str) -> float | None:
    """A cosine in [-1, 1] (the slack covers float rounding of the printed value), or None for a missing side."""
    return None if value is None else _number(value, where, -2.0, 2.0)


def _accuracy(value: object) -> Accuracy:
    o = _object(value, ("status", "head_tests", "rows"), "accuracy")
    rows: list[AccuracyRow] = []
    for i, raw in enumerate(_list(o["rows"], "accuracy.rows", MAX_ROWS)):
        row = _object(raw, ("key", "base", "head"), f"accuracy.rows[{i}]")
        rows.append(
            {
                "key": _text(row["key"], f"accuracy.rows[{i}].key", MAX_KEY, _KEY_TEXT),
                # A cosine lies in [-1, 1]; the slack covers float rounding of the printed value.
                "base": _cosine(row["base"], f"accuracy.rows[{i}].base"),
                "head": _cosine(row["head"], f"accuracy.rows[{i}].head"),
            }
        )
        if rows[-1]["base"] is None and rows[-1]["head"] is None:
            raise ValueError(f"accuracy.rows[{i}]: base and head are both null")
    status = _choice(o["status"], "accuracy.status", STATUSES)
    if rows and status != "ran":
        raise ValueError(f"accuracy: rows given although the status is {status}")
    head_tests: str | None = None
    if status == "ran":
        head_tests = _choice(o["head_tests"], "accuracy.head_tests", HEAD_TESTS)
    elif o["head_tests"] is not None:
        raise ValueError(f"accuracy: head_tests given although the status is {status}")
    return {"status": status, "head_tests": head_tests, "rows": rows}


def _validate(value: object) -> Report:
    """Check a parsed JSON value against schema 1 and return it as a Report built from validated fields only."""
    o = _object(value, _TOP_KEYS, "report")
    _int(o["schema"], "schema", SCHEMA, SCHEMA)  # the only version this code understands
    if not isinstance(o["kernels_changed"], bool):
        raise ValueError("kernels_changed: expected a boolean")
    perf: list[PerfEntry] | None = None
    if o["perf"] is not None:
        perf = [_perf_entry(p, f"perf[{i}]") for i, p in enumerate(_list(o["perf"], "perf", MAX_PERF))]
    size: Size | None = None
    if o["size"] is not None:
        s = _object(o["size"], ("base_bytes", "head_bytes"), "size")
        size = {
            "base_bytes": _int(s["base_bytes"], "size.base_bytes", 0, MAX_INT),
            "head_bytes": _int(s["head_bytes"], "size.head_bytes", 0, MAX_INT),
        }
    rss: Rss | None = None
    if o["rss"] is not None:
        m = _object(o["rss"], ("base_kib", "head_kib"), "rss")
        rss = {
            "base_kib": _int(m["base_kib"], "rss.base_kib", 0, MAX_INT),
            "head_kib": _int(m["head_kib"], "rss.head_kib", 0, MAX_INT),
        }
    return {
        "schema": SCHEMA,
        "base_sha": _sha(o["base_sha"], "base_sha"),
        "head_sha": _sha(o["head_sha"], "head_sha"),
        "runner": _text(o["runner"], "runner", MAX_RUNNER, _RUNNER_TEXT),
        "kernels_changed": o["kernels_changed"],
        "perf": perf,
        "size": size,
        "rss": rss,
        "accuracy": _accuracy(o["accuracy"]),
    }


def load(text: str) -> Report:
    """Parse and strictly validate the text of a pr-report.json; raise ValueError (one-line message) unless schema 1."""
    return _validate(_parse(text, "report"))


def _inert_runner(name: str) -> str:
    """Make a runner description loadable: `@` becomes "at" (a CPU is "... CPU @ 2.80GHz"), any other character
    outside the allowed set becomes "_", and the result is cut to MAX_RUNNER characters."""
    return _RUNNER_UNSAFE.sub("_", name.replace("@", "at"))[:MAX_RUNNER]


def assemble(parts: Sequence[str]) -> str:
    """Merge the JSON texts of the partial files into the text of a validated pr-report.json.

    Each part is an object holding some top-level keys of the schema (no key in two parts) or a bare list, taken as
    the `perf` section. `base_sha`, `head_sha`, `runner`, `kernels_changed` and `accuracy` must be provided (a missing
    accuracy part is an error, not a cold cache); `perf`, `size` and `rss` default to skipped. The runner is made
    inert (`_inert_runner`). The merged result goes through `load`, so `assemble` never writes a report `post` would
    refuse.
    """
    merged: dict[str, Any] = {"schema": SCHEMA, "perf": None, "size": None, "rss": None}
    seen: set[str] = set()
    for index, text in enumerate(parts):
        part = _parse(text, f"part {index}")
        if isinstance(part, list):
            part = {"perf": part}
        if not isinstance(part, dict):
            raise ValueError(f"part {index}: expected an object or a list")
        unknown = part.keys() - (set(_META_KEYS) | set(_SECTIONS))
        if unknown:
            raise ValueError(f"part {index}: unexpected keys {sorted(map(_show, unknown))}")
        repeated = part.keys() & seen
        if repeated:
            raise ValueError(f"part {index}: keys already given by an earlier part: {sorted(repeated)}")
        seen |= part.keys()
        merged.update(part)
    for required in ("base_sha", "head_sha", "runner", "kernels_changed", "accuracy"):
        if required not in seen:
            raise ValueError(f"no part provides {required}")
    if isinstance(merged["runner"], str):
        merged["runner"] = _inert_runner(merged["runner"])
    text = json.dumps(merged, indent=2, allow_nan=False)
    load(text)
    return text


# ---------------------------------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------------------------------


def _esc(text: str) -> str:
    """Backslash-escape every ASCII punctuation character so `text` renders literally (no link, mention, HTML or
    table column), the way CommonMark/GFM allow."""
    return _PUNCTUATION.sub(r"\\\1", text)


def _pct(base: float, head: float) -> str:
    """Signed percent change from `base` to `head`, or an em dash when there is no usable base."""
    if base <= 0:
        return "—"
    return f"{(head / base - 1) * 100:+.1f}%"


def _tok(value: float) -> str:
    """Tokens per second to one decimal, or an em dash for 0.0 (no usable run)."""
    return f"{value:.1f}" if value > 0 else "—"


def _speed_rows(perf: Sequence[PerfEntry]) -> list[str]:
    rows = [
        "| model | test | threads | base tok/s | head tok/s | Δ% | verdict |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for e in perf:
        # Δ% is the median per-block ratio minus one. Ratios of 0.0 stand for unusable blocks (ruling P16) and are
        # left out; with no base median or no usable ratio there is no honest number, so the cell is a dash.
        usable = [r for r in e["ratios"] if r > 0]
        delta = "—"
        if e["base_median"] > 0 and usable:
            delta = f"{(statistics.median(usable) - 1) * 100:+.1f}%"
        symbol = VERDICT_SYMBOLS[e["verdict"]]
        rows.append(
            f"| {_esc(e['model'])} | {e['test']} | {e['threads']} | {_tok(e['base_median'])} "
            f"| {_tok(e['head_median'])} | {delta} | {symbol} {e['verdict']} |"
        )
    return rows


def _accuracy_lines(r: Report) -> list[str]:
    acc = r["accuracy"]
    if acc["status"] != "ran":
        # The nightly job on master is the only one that saves the model cache (ruling R4), and its key hashes the
        # model inputs (setup-models), so a pull request that changes them stays cold until it merges.
        return [
            "**Accuracy:** not run: the model cache was cold. The nightly job on master refills it, but a pull request "
            "that changes the model inputs (the scripts and prompts the cache key hashes) gets no accuracy run before "
            "it merges; run `ctest -L e2e` locally for one."
        ]
    # First, because the rows alone cannot show it: a test can fail after printing an unchanged min cosine.
    tests = (
        "❌ end-to-end tests failed on the head (a failed check need not move a min cosine; see the accuracy job)"
        if acc["head_tests"] == "failed"
        else "✅ end-to-end tests passed on the head"
    )
    rows = acc["rows"]
    if not rows:
        return ["**Accuracy:** ran, no rows", "", tests]
    # A case with a value on one side only counts as changed.
    changed = sum(1 for row in rows if row["base"] != row["head"])
    lines = ["**Accuracy:** min cosine against llama.cpp, per case", "", tests, ""]
    if r["kernels_changed"]:
        # The cosines are compared as printed (six significant digits, the CONTRIBUTING.md check): equal values are
        # strong evidence, not proof, that the logits are bit-identical, so the line says only what was compared.
        lines.append(
            f"❗ printed min cosine changed in {changed} of {len(rows)} cases"
            if changed
            else f"✅ printed min cosine unchanged in all {len(rows)} cases"
        )
        lines.append("")
    lines += ["| case | base | head | Δ |", "|---|---:|---:|---:|"]
    for row in rows:
        base, head = row["base"], row["head"]
        if base is None:
            delta = "no base result"
        elif head is None:
            delta = "no head result"
        else:
            delta = "—" if base == head else f"{head - base:+.6f}"
        cells = ["—" if v is None else f"{v:.6g}" for v in (base, head)]
        lines.append(f"| {_esc(row['key'])} | {cells[0]} | {cells[1]} | {delta} |")
    return lines


def render(r: Report) -> str:
    """The Markdown comment for `r`, starting with MARKER. Every dynamic piece comes from a validated field."""
    lines = [MARKER, "## nibble PR report", "", "_shared CI runner: trend, not a benchmark_", ""]
    if r["perf"] is None:
        lines.append("**Speed:** skipped")
    else:
        lines += ["**Speed**", ""] + (_speed_rows(r["perf"]) if r["perf"] else ["no results"])
    lines.append("")
    if r["size"] is None:
        lines.append("**Size:** skipped")
    else:
        b, h = r["size"]["base_bytes"], r["size"]["head_bytes"]
        lines.append(f"**Size:** {b:,} B → {h:,} B (Δ {h - b:+,} B, {_pct(b, h)})")
    lines.append("")
    if r["rss"] is None:
        lines.append("**Peak memory:** skipped")
    else:
        b, h = r["rss"]["base_kib"], r["rss"]["head_kib"]
        lines.append(f"**Peak memory:** {b:,} KiB → {h:,} KiB (Δ {h - b:+,} KiB, {_pct(b, h)})")
    lines.append("")
    lines += _accuracy_lines(r)
    lines += [
        "",
        "<details>",
        "<summary>Details</summary>",
        "",
        f"- base: `{r['base_sha']}`",
        f"- head: `{r['head_sha']}`",
        f"- runner: {_esc(r['runner'])}",
        "",
        "Reproduce locally (an idle machine gives steadier numbers than a shared runner):",
        "",
        "```sh",
        f"git worktree add ../nibble-base {r['base_sha']}",
        "# build both trees in Release, then compare them with interleaved ABBA blocks",
        "python tools/ci/perf_ab.py run --base ../nibble-base/build/nibble --head build/nibble \\",
        "    --model models/qwen2.5-0.5b-instruct-q4_0.gguf --out rows.jsonl",
        "python tools/ci/perf_ab.py summarize rows.jsonl --out perf.json",
        "ctest --test-dir build -L e2e -j2   # accuracy (needs the models)",
        "```",
        "",
        "</details>",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------------------
# Posting and command line
# ---------------------------------------------------------------------------------------------------------------


def post(gh: GitHub, r: Report, run_head_sha: str) -> int:
    """Create or update the report comment on the open pull request whose head is `run_head_sha`.

    `run_head_sha` is the head SHA of the workflow run that triggered this job (trusted context). Returns the pull
    request number, or 0 without posting anything when the report's `head_sha` is not `run_head_sha` (the artifact
    belongs to another commit: ruling R18) or when no open pull request has that head (it was closed or updated).
    The number comes from `gh.pr_for_sha`, never from the artifact, and only the bot's own comment is edited.
    """
    if r["head_sha"] != run_head_sha:
        return 0
    number = gh.pr_for_sha(run_head_sha)
    if number is None:
        return 0
    gh.upsert_comment(number, MARKER, render(r), author=BOT_LOGIN)
    return number


_NON_FINITE = frozenset({"nan", "inf", "-inf"})  # how accuracy.py's snapshot_json writes a non-finite min cosine


def _snapshot(path: str) -> dict[str, float]:
    """Read a cosine snapshot, the `e2e-cosines.json` the nightly job writes, as {key: min_cos}.

    The file is `{"<model>/<prompt>": {"model", "prompt", "min_cos", "positions", "agree"}, ...}`; a `min_cos` that is
    not finite is written there as a string ("nan", "inf"). Only `min_cos` is used. A non-finite value (the schema
    cannot carry one) leaves the case out, so it shows as "no head result" (or "no base result" when the base has the
    NaN) instead of the report not being written at all; the workflow's strict `accuracy.py snapshot` gate on the
    head is what fails the job for it. Any other
    non-numeric value is a ValueError naming the case.
    """
    raw = _parse(read_limited(path), os.path.basename(path))
    if not isinstance(raw, dict):
        raise ValueError(f"{os.path.basename(path)}: expected an object")
    cosines: dict[str, float] = {}
    for key, entry in raw.items():
        value = entry.get("min_cos") if isinstance(entry, dict) else None
        if isinstance(value, str) and value in _NON_FINITE:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{os.path.basename(path)}: {_show(key)} has no numeric min_cos")
        if math.isfinite(value):
            cosines[key] = float(value)
    return cosines


def accuracy_part(base: Mapping[str, float], head: Mapping[str, float], head_tests: str) -> str:
    """The text of an `accuracy` part: one row per case in either snapshot, sorted by key (None for a missing side),
    and `head_tests` ("passed" or "failed").

    The result is validated like a loaded report's accuracy section, so a key outside the allowed characters, too many
    rows or an out-of-range cosine is a ValueError here, in the job that wrote the data.
    """
    rows = [{"key": k, "base": base.get(k), "head": head.get(k)} for k in sorted(base.keys() | head.keys())]
    section = {"status": "ran", "head_tests": head_tests, "rows": rows}
    _accuracy(section)
    return json.dumps({"accuracy": section}, indent=2, allow_nan=False)


def main(argv: Sequence[str], gh: GitHub | None = None) -> int:
    """Command line: `accuracy` and `assemble` build the report, `post` comments it. `gh` is for tests."""
    parser = argparse.ArgumentParser(description="Assemble and post the nibble PR report.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_asm = sub.add_parser("assemble", help="merge partial JSON files into pr-report.json")
    p_asm.add_argument("parts", nargs="+", help="partial JSON files")
    p_asm.add_argument("--out", required=True, help="file to write")
    p_acc = sub.add_parser("accuracy", help="write the accuracy part from two cosine snapshots")
    p_acc.add_argument("--base", help="e2e-cosines.json of the base build")
    p_acc.add_argument("--head", help="e2e-cosines.json of the head build")
    p_acc.add_argument(
        "--head-outcome",
        choices=sorted(_OUTCOMES),
        help="outcome of the head's e2e test step (steps.<id>.outcome); required with --base and --head",
    )
    p_acc.add_argument("--skipped-cold-cache", action="store_true", help="write the explicit cold-cache status instead")
    p_acc.add_argument("--out", required=True, help="file to write")
    p_post = sub.add_parser("post", help="comment the report on the pull request of the triggering run")
    p_post.add_argument("report", help="downloaded pr-report.json (untrusted)")
    p_post.add_argument("--head-sha", required=True, help="head SHA of the workflow run that triggered this job")
    args = parser.parse_args(list(argv))
    if args.command == "accuracy":
        has_snapshots = bool(args.base and args.head)
        if has_snapshots == args.skipped_cold_cache or (args.skipped_cold_cache and (args.base or args.head)):
            # Neither given, or both, or only one of --base and --head.
            parser.error("give either --skipped-cold-cache, or both --base and --head")
        if has_snapshots != (args.head_outcome is not None):
            parser.error("--head-outcome goes with --base and --head, and only with them")

    try:
        if args.command == "assemble":
            text = assemble([read_limited(p) for p in args.parts])
            Path(args.out).write_text(text + "\n", encoding="utf-8")
            return 0
        if args.command == "accuracy":
            if args.skipped_cold_cache:
                cold: Accuracy = {"status": "skipped-cold-cache", "head_tests": None, "rows": []}
                text = json.dumps({"accuracy": cold}, indent=2)
            else:
                text = accuracy_part(_snapshot(args.base), _snapshot(args.head), _OUTCOMES[args.head_outcome])
            Path(args.out).write_text(text + "\n", encoding="utf-8")
            return 0
        run_sha = _sha(args.head_sha, "--head-sha")
        report = load(read_limited(args.report))
        if report["head_sha"] != run_sha:
            print("pr_report: the report's head_sha does not match the triggering run; nothing posted", file=sys.stderr)
            return 0
        if gh is None and not os.environ.get("GITHUB_REPOSITORY"):
            raise ValueError("GITHUB_REPOSITORY is not set")
        client = gh if gh is not None else GhCli(os.environ["GITHUB_REPOSITORY"])
        number = post(client, report, run_sha)
        print(f"pr_report: {'commented on #' + str(number) if number else 'no open pull request for this commit'}")
        return 0
    except (ValueError, OSError, GhError) as exc:
        # One line, prefixed: a message can never begin a line with `::` (a workflow command).
        print(f"pr_report: {args.command}: {' '.join(str(exc).split())}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
