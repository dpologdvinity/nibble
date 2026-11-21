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

    {"schema": 1, "base_sha": "<40 hex>", "head_sha": "<40 hex>", "runner": "<1-120 printable ASCII>",
     "kernels_changed": bool,
     "perf": [{"model": <allowlisted>, "test": "pp512"|"tg128", "threads": int, "base_median": x,
               "head_median": x, "ratios": [x...], "verdict": "faster"|"slower"|"same"|"inconclusive"}] | null,
     "size": {"base_bytes": int, "head_bytes": int} | null,
     "rss": {"base_kib": int, "head_kib": int} | null,
     "accuracy": {"status": "ran"|"skipped-cold-cache", "rows": [{"key": "<model>/<prompt>", "base": x, "head": x}]}}

`assemble` takes partial files: a JSON object holding some of the keys above (each key in at most one part), or a
bare JSON list, which is the `perf` section exactly as `perf_ab.py summarize` writes it. Sections no part provides
become skipped (null, or accuracy "skipped-cold-cache"); `base_sha`, `head_sha` and `runner` are required.

usage:
    python tools/ci/pr_report.py assemble meta.json perf.json size.json --out pr-report.json
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
from collections.abc import Sequence
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
_PRINTABLE = re.compile(r"[ -~]+")  # printable ASCII: no control characters, newlines or unicode tricks
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
    """Minimum cosine similarity against the golden logits for `key` ("<model>/<prompt>")."""

    key: str
    base: float
    head: float


class Accuracy(TypedDict):
    """The accuracy section: `rows` is empty unless `status` is "ran"."""

    status: str
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


def _parse(text: str, what: str) -> Any:
    """Parse untrusted JSON text strictly; every failure is a ValueError that names `what`."""
    if len(text) > MAX_BYTES:
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


def _text(value: object, where: str, max_len: int) -> str:
    """1..max_len characters of printable ASCII."""
    if not isinstance(value, str) or not 1 <= len(value) <= max_len or not _PRINTABLE.fullmatch(value):
        raise ValueError(f"{where}: expected 1-{max_len} printable ASCII characters, got {_show(value)}")
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


def _accuracy(value: object) -> Accuracy:
    o = _object(value, ("status", "rows"), "accuracy")
    rows: list[AccuracyRow] = []
    for i, raw in enumerate(_list(o["rows"], "accuracy.rows", MAX_ROWS)):
        row = _object(raw, ("key", "base", "head"), f"accuracy.rows[{i}]")
        rows.append(
            {
                "key": _text(row["key"], f"accuracy.rows[{i}].key", MAX_KEY),
                # A cosine lies in [-1, 1]; the slack covers float rounding of the printed value.
                "base": _number(row["base"], f"accuracy.rows[{i}].base", -2.0, 2.0),
                "head": _number(row["head"], f"accuracy.rows[{i}].head", -2.0, 2.0),
            }
        )
    return {"status": _choice(o["status"], "accuracy.status", STATUSES), "rows": rows}


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
        "runner": _text(o["runner"], "runner", MAX_RUNNER),
        "kernels_changed": o["kernels_changed"],
        "perf": perf,
        "size": size,
        "rss": rss,
        "accuracy": _accuracy(o["accuracy"]),
    }


def load(text: str) -> Report:
    """Parse and strictly validate the text of a pr-report.json; raise ValueError (one-line message) unless schema 1."""
    return _validate(_parse(text, "report"))


def assemble(parts: Sequence[str]) -> str:
    """Merge the JSON texts of the partial files into the text of a validated pr-report.json.

    Each part is an object holding some top-level keys of the schema (no key in two parts) or a bare list, taken as
    the `perf` section. Missing sections are skipped; `base_sha`, `head_sha` and `runner` must be provided. The
    merged result goes through `load`, so `assemble` never writes a report `post` would refuse.
    """
    merged: dict[str, Any] = {"schema": SCHEMA, "kernels_changed": False, "perf": None, "size": None, "rss": None}
    merged["accuracy"] = {"status": "skipped-cold-cache", "rows": []}
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
    for required in ("base_sha", "head_sha", "runner"):
        if required not in seen:
            raise ValueError(f"no part provides {required}")
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
        return ["**Accuracy:** not run (the model cache was cold; the nightly run refills it)"]
    rows = acc["rows"]
    if not rows:
        return ["**Accuracy:** ran, no rows"]
    changed = sum(1 for row in rows if row["base"] != row["head"])
    lines = ["**Accuracy:** min cosine against llama.cpp, per case", ""]
    if r["kernels_changed"]:
        lines.append(
            f"❗ bit-identity: {changed} cases changed"
            if changed
            else f"✅ bit-identity: all {len(rows)} cases unchanged"
        )
        lines.append("")
    lines += ["| case | base | head | Δ |", "|---|---:|---:|---:|"]
    for row in rows:
        delta = "—" if row["base"] == row["head"] else f"{row['head'] - row['base']:+.6f}"
        lines.append(f"| {_esc(row['key'])} | {row['base']:.6g} | {row['head']:.6g} | {delta} |")
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


def main(argv: Sequence[str], gh: GitHub | None = None) -> int:
    """Command line: `assemble` merges partial files, `post` comments on the pull request. `gh` is for tests."""
    parser = argparse.ArgumentParser(description="Assemble and post the nibble PR report.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_asm = sub.add_parser("assemble", help="merge partial JSON files into pr-report.json")
    p_asm.add_argument("parts", nargs="+", help="partial JSON files")
    p_asm.add_argument("--out", required=True, help="file to write")
    p_post = sub.add_parser("post", help="comment the report on the pull request of the triggering run")
    p_post.add_argument("report", help="downloaded pr-report.json (untrusted)")
    p_post.add_argument("--head-sha", required=True, help="head SHA of the workflow run that triggered this job")
    args = parser.parse_args(list(argv))

    try:
        if args.command == "assemble":
            text = assemble([Path(p).read_text(encoding="utf-8") for p in args.parts])
            Path(args.out).write_text(text + "\n", encoding="utf-8")
            return 0
        run_sha = _sha(args.head_sha, "--head-sha")
        report = load(Path(args.report).read_text(encoding="utf-8"))
        if report["head_sha"] != run_sha:
            print("pr_report: the report's head_sha does not match the triggering run; nothing posted", file=sys.stderr)
            return 0
        client = gh if gh is not None else GhCli(os.environ["GITHUB_REPOSITORY"])
        number = post(client, report, run_sha)
        print(f"pr_report: {'commented on #' + str(number) if number else 'no open pull request for this commit'}")
        return 0
    except (ValueError, OSError, GhError, KeyError) as exc:
        # One line, prefixed: a message can never begin a line with `::` (a workflow command).
        print(f"pr_report: {args.command}: {' '.join(str(exc).split())}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
