"""Normalize nightly, portability and CI measurements into validated dashboard rows (JSON Lines).

Each job turns what it measured into rows of one fixed shape; a later job merges them onto the data branch and renders
the charts. A row is one JSON object per line with every key present:

    {"schema": 1, "date": "2026-11-21T03:17:00Z", "commit": "<40 hex>", "source": "nightly|ci|portability",
     "metric": "<METRICS>", "key": "<= 120 chars", "value": <finite float>, "unit": "<its metric's unit>",
     "runner": "<cpu model, <= 120 chars>", "run_id": <int>}

Metrics, their keys and units (METRICS below is the single source for the metric -> unit pairing):

    bench_tok_s    "<model file>/<test>/t<threads>"  tok/s   median of `nibble bench --out` (membw lines are skipped)
    min_cosine     "<model>/<prompt>"                cosine  from cosine.parse_min_cosine
    fuzz_execs     "<target>"                        execs   libFuzzer `stat::number_of_executed_units` (if printed)
    fuzz_seconds   "<target>"                        s       libFuzzer `Done N runs in S second(s)`, else --seconds
    corpus_files   "<target>"                        files   passed in by the caller (count of the corpus directory)
    binary_bytes   "nibble"                          bytes   size of the release binary
    line_coverage  "src+cli"                         %       gcovr `lines: 85.3% (1234 out of 1447)`
    job_status     "<workflow>/<job>"                ok      1 = success, 0 = failure (see below)

`job_status` is written by the job's "Dashboard rows (job status)" step from `job.status`, which covers the job's
checks up to that step. A job that hits its timeout-minutes cannot write it (its remaining steps see `cancelled`, the
same as a job superseded in its concurrency group), so report-job.yml writes a `timed-out` row, value 0, for it in a
run that was not cancelled; a hang turns the badge red instead of leaving the last green row standing. The issue
open/comment/close happens afterwards in a separate `report-<job>` job, so a report job that fails by itself (a GitHub
API error) leaves a green row next to a red report job. That is deliberate: the row
says whether the code and the checks passed, and the red report job is visible in the run.

`validate` is the gate: every row built here, and every new row `merge` accepts, passes through it.
`merge` appends rows to existing JSONL text and drops repeats of the same (run_id, metric, key), so re-running a job
or a merge is harmless; existing lines are kept byte for byte. The one exception is `job_status`: a re-run of a job
keeps the run id, so its later row replaces the failed attempt's (see `merge`).

Run metadata (`Meta`) comes from the environment in `meta_from_env`: GITHUB_SHA, GITHUB_RUN_ID, the CPU model from
/proc/cpuinfo and the current UTC time. The pure functions never read the environment.

usage (each appends to --out; --source is nightly|ci|portability, default nightly):
    dashboard_data.py from-bench BENCH.jsonl --out rows.jsonl
    dashboard_data.py from-cosine CTEST.log --out rows.jsonl
    dashboard_data.py from-libfuzzer FUZZ.log --target fuzz_gguf [--corpus-files N] [--seconds S] --out rows.jsonl
        (the fuzzer must run with -print_final_stats=1; S is the wall time the workflow measured around it, used when
        a crash left no `Done` line, so fuzzing time before a crash still counts)
    dashboard_data.py from-gcovr SUMMARY.txt --out rows.jsonl          (gcovr --print-summary output)
    dashboard_data.py from-binary-size build/nibble --out rows.jsonl
    dashboard_data.py status --workflow nightly --job fuzz_gguf --outcome success|failure|timed-out|cancelled \
        --out rows.jsonl
        (timed-out is a failure, value 0; cancelled writes nothing)
    dashboard_data.py merge EXISTING.jsonl NEW.jsonl... --out merged.jsonl [--skip-bad]   (EXISTING may be missing)
        (a NEW file with an invalid row fails the command, or with --skip-bad is skipped whole with a `::warning`
        and a job-summary note, and the other files still merge)
Stdlib only. Exit status is 0 on success and 1 (message on stderr) for input that is not what it should be.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from cosine import Cosine, parse_min_cosine

Row = dict[str, object]

SCHEMA = 1
SOURCES = ("nightly", "ci", "portability")
# metric -> unit. A row whose unit disagrees with its metric is rejected: a chart mixes values of one metric only.
METRICS: dict[str, str] = {
    "bench_tok_s": "tok/s",
    "min_cosine": "cosine",
    "fuzz_seconds": "s",
    "fuzz_execs": "execs",
    "corpus_files": "files",
    "binary_bytes": "bytes",
    "line_coverage": "%",
    "job_status": "ok",
}
# The keys of a row, in the order they are written.
FIELDS = ("schema", "date", "commit", "source", "metric", "key", "value", "unit", "runner", "run_id")
MAX_TEXT = 120  # longest `key` and `runner`
DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"  # the layout of a row's `date`, in UTC
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
_COMMIT = re.compile(r"[0-9a-f]{40}")
COVERAGE_KEY = "src+cli"
BINARY_KEY = "nibble"


@dataclass(frozen=True)
class Meta:
    """What one job run knows about itself: `date` (UTC, `YYYY-MM-DDTHH:MM:SSZ`), the tested 40-hex `commit`, the
    `source` workflow, the `runner` CPU model and the GitHub `run_id`."""

    date: str
    commit: str
    source: str
    runner: str
    run_id: int


def validate(row: Mapping[str, object]) -> Row:
    """Return `row` as a new dict in schema order with `value` as a float, or raise ValueError naming the bad field.

    Checks: exactly the schema's keys; schema 1; date format, and that it is a real date and time; 40-hex commit; known
    source and metric; a unit that matches the metric; a non-empty key and runner of at most 120 characters; a finite
    number for `value`; a non-negative int (not a bool) for `run_id`.
    """
    for field in FIELDS:
        if field not in row:
            raise ValueError(f"row is missing field '{field}'")
    extra = sorted(set(row) - set(FIELDS))
    if extra:
        raise ValueError(f"row has extra field(s) {extra}")
    if row["schema"] != SCHEMA or isinstance(row["schema"], bool):
        raise ValueError(f"field 'schema' must be {SCHEMA}, got {row['schema']!r}")
    _check_str("date", row["date"], _DATE)
    try:
        datetime.strptime(str(row["date"]), DATE_FORMAT)  # the pattern alone lets 2026-02-30 and 24:00:00 through
    except ValueError as e:
        raise ValueError(f"field 'date' must be a real date and time, got {row['date']!r}") from e
    _check_str("commit", row["commit"], _COMMIT)
    if row["source"] not in SOURCES:
        raise ValueError(f"field 'source' must be one of {SOURCES}, got {row['source']!r}")
    metric = row["metric"]
    if not isinstance(metric, str) or metric not in METRICS:
        raise ValueError(f"field 'metric' must be one of {sorted(METRICS)}, got {metric!r}")
    if row["unit"] != METRICS[metric]:
        raise ValueError(f"field 'unit' must be {METRICS[metric]!r} for metric {metric!r}, got {row['unit']!r}")
    for field in ("key", "runner"):
        text = row[field]
        if not isinstance(text, str) or not 1 <= len(text) <= MAX_TEXT:
            raise ValueError(f"field '{field}' must be a string of 1 to {MAX_TEXT} characters, got {text!r:.60}")
    value = row["value"]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"field 'value' must be a finite number, got {value!r:.60}")
    try:
        finite = math.isfinite(value)
    except OverflowError:  # an integer too big for a float
        finite = False
    if not finite:
        raise ValueError(f"field 'value' must be a finite number, got {value!r:.60}")
    run_id = row["run_id"]
    if isinstance(run_id, bool) or not isinstance(run_id, int) or run_id < 0:
        raise ValueError(f"field 'run_id' must be a non-negative integer, got {run_id!r}")
    clean = {field: row[field] for field in FIELDS}
    clean["value"] = float(value)
    return clean


def _check_str(field: str, value: object, pattern: re.Pattern[str]) -> None:
    """Raise ValueError naming `field` unless `value` is a string that matches `pattern` completely."""
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"field '{field}' must match {pattern.pattern}, got {value!r:.60}")


def _row(meta: Meta, metric: str, key: str, value: float) -> Row:
    """Build and validate one row for `metric` (its unit comes from METRICS)."""
    return validate(
        {
            "schema": SCHEMA,
            "date": meta.date,
            "commit": meta.commit,
            "source": meta.source,
            "metric": metric,
            "key": key,
            "value": value,
            "unit": METRICS[metric],
            "runner": meta.runner,
            "run_id": meta.run_id,
        }
    )


def from_bench(jsonl: str, meta: Meta) -> list[Row]:
    """Rows for the `tok/s` lines of `nibble bench --out` (cli/cmd_bench.cpp), keyed `<model>/<test>/t<threads>`.

    The value is the line's "median". Lines of another unit (the `--mem` bandwidth line, GB/s) or another tool
    (llama.cpp reference lines) are skipped; blank lines too. Bad JSON or a missing field raises ValueError with
    the line number. The commit is `meta.commit`, not the short sha inside the line.
    """
    rows: list[Row] = []
    for number, line in enumerate(jsonl.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except ValueError as e:
            raise ValueError(f"bench line {number}: not JSON ({e})") from e
        if not isinstance(obj, dict):
            raise ValueError(f"bench line {number}: not a JSON object")
        if obj.get("tool", "nibble") != "nibble" or obj.get("unit") != "tok/s":
            continue
        for field in ("model", "test", "threads", "median"):
            if field not in obj:
                raise ValueError(f"bench line {number}: missing field '{field}'")
        median = obj["median"]
        if isinstance(median, bool) or not isinstance(median, int | float):
            raise ValueError(f"bench line {number}: field 'median' is not a number: {median!r}")
        rows.append(_row(meta, "bench_tok_s", f"{obj['model']}/{obj['test']}/t{obj['threads']}", float(median)))
    return rows


def from_cosine(c: Mapping[str, Cosine], meta: Meta) -> list[Row]:
    """One `min_cosine` row per entry of `cosine.parse_min_cosine` output, in mapping order.

    A NaN cosine raises ValueError (it is not a finite value); the accuracy test itself fails on such a run.
    """
    return [_row(meta, "min_cosine", f"{e.model}/{e.prompt}", e.min_cos) for e in c.values()]


_EXECS = re.compile(r"^stat::number_of_executed_units:\s*(\d+)\s*$", re.MULTILINE)
_DONE = re.compile(r"^Done (\d+) runs in (\d+) second\(s\)\s*$", re.MULTILINE)


def from_libfuzzer(
    log: str, target: str, meta: Meta, corpus_files: int | None = None, seconds: int | None = None
) -> list[Row]:
    """`fuzz_seconds` and, when present, `fuzz_execs` rows (plus `corpus_files` when given) from a libFuzzer log.

    Time fuzzed counts even when the target crashed, so the elapsed seconds never depend on a clean exit: they come
    from libFuzzer's own `Done N runs in S second(s)` line (the last one) and, when that line is missing because the
    run ended in a crash, from `seconds`, which the caller measured around the fuzzer. With neither, ValueError.
    `fuzz_execs` needs `stat::number_of_executed_units: N` (run with -print_final_stats=1; the last one counts) and is
    left out when the log has none. `corpus_files` is counted by the caller from the corpus directory.
    """
    if seconds is not None and seconds < 0:
        raise ValueError(f"seconds must not be negative, got {seconds}")
    done = _DONE.findall(log)
    if done:
        elapsed = int(done[-1][1])
    elif seconds is not None:
        elapsed = seconds
    else:
        raise ValueError(
            "libFuzzer log has no 'Done N runs in S second(s)' line (did the run end early?) and no --seconds was given"
        )
    rows: list[Row] = []
    execs = _EXECS.findall(log)
    if execs:
        rows.append(_row(meta, "fuzz_execs", target, float(execs[-1])))
    rows.append(_row(meta, "fuzz_seconds", target, float(elapsed)))
    if corpus_files is not None:
        rows.append(_row(meta, "corpus_files", target, float(corpus_files)))
    return rows


_LINES = re.compile(r"^lines:\s*([0-9.]+)%\s*\((\d+) out of (\d+)\)\s*$", re.MULTILINE)


def from_gcovr_summary(text: str, meta: Meta) -> list[Row]:
    """The `line_coverage` row (key `src+cli`) from the `lines: 85.3% (1234 out of 1447)` line of gcovr's summary.

    The value is the percentage as gcovr prints it. ValueError if there is no such line (gcovr prints `nan%` when
    nothing was measured).
    """
    m = _LINES.search(text)
    if m is None:
        raise ValueError("gcovr summary has no 'lines: <pct>% (<n> out of <m>)' line")
    return [_row(meta, "line_coverage", COVERAGE_KEY, float(m.group(1)))]


def from_binary_size(size: int, meta: Meta) -> list[Row]:
    """The `binary_bytes` row (key `nibble`) for a release binary of `size` bytes."""
    return [_row(meta, "binary_bytes", BINARY_KEY, float(size))]


def status_row(workflow: str, job: str, ok: bool, meta: Meta) -> Row:
    """The `job_status` row for `<workflow>/<job>`: value 1 for success, 0 for failure."""
    return _row(meta, "job_status", f"{workflow}/{job}", 1.0 if ok else 0.0)


def _identity(row: Mapping[str, object]) -> tuple[object, object, object] | None:
    """The (run_id, metric, key) triple that makes two rows duplicates, or None if the row lacks one of them."""
    if all(k in row for k in ("run_id", "metric", "key")):
        return (row["run_id"], row["metric"], row["key"])
    return None


def _dump(row: Mapping[str, object]) -> str:
    """Compact one-line JSON for `row`; NaN and infinity are refused."""
    return json.dumps(row, separators=(",", ":"), allow_nan=False)


def merge(existing: str, new_rows: Iterable[Row]) -> str:
    """JSONL text of the `existing` lines followed by `new_rows`, without repeats of (run_id, metric, key).

    Existing lines are kept as they are (not validated, not rewritten) except that a later repeat of an earlier
    line's (run_id, metric, key) is dropped; of two rows with the same identity the first wins, so merging the same
    rows twice changes nothing. One exception: a `job_status` row replaces an earlier row of the same identity when its
    `date` is later (a re-run keeps the run id, so its status must supersede the failed attempt's). The replacement
    takes the earlier row's place in the file, and a row that is not later never replaces one, so the outcome does not
    depend on the order rows arrive in. Every new row is validated first (ValueError names the field). A line of
    `existing` that is not a JSON object raises ValueError with its line number. The result ends with a newline unless
    empty.
    """
    seen: dict[tuple[object, object, object], int] = {}  # identity -> index into `lines`
    lines: list[str] = []
    objects: list[Mapping[str, object]] = []
    for number, line in enumerate(existing.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except ValueError as e:
            raise ValueError(f"existing line {number}: not JSON ({e})") from e
        if not isinstance(obj, dict):
            raise ValueError(f"existing line {number}: not a JSON object")
        ident = _identity(obj)
        if ident is not None:
            if ident in seen:
                continue
            seen[ident] = len(lines)
        lines.append(line)
        objects.append(obj)
    for new in new_rows:
        row = validate(new)
        ident = _identity(row)
        if ident is not None and ident in seen:
            at = seen[ident]
            if row["metric"] == "job_status" and str(row["date"]) > str(objects[at].get("date", "")):
                lines[at] = _dump(row)
                objects[at] = row
            continue
        if ident is not None:
            seen[ident] = len(lines)
        lines.append(_dump(row))
        objects.append(row)
    return "".join(f"{line}\n" for line in lines)


def cpu_name(cpuinfo: str) -> str:
    """The CPU model from the text of /proc/cpuinfo, at most 120 characters.

    x86 has `model name`. aarch64 usually has none, only `CPU implementer` and `CPU part` codes (and `Hardware` on
    some SoCs), so fall back to `Hardware`, then to "unknown (CPU implementer 0x.. part 0x..)", then to "unknown".
    """
    fields: dict[str, str] = {}
    for line in cpuinfo.splitlines():
        name, sep, value = line.partition(":")
        if sep and value.strip():
            fields.setdefault(name.strip().lower(), value.strip())
    if "model name" in fields:
        name = fields["model name"]
    elif "hardware" in fields:
        name = fields["hardware"]
    elif "cpu implementer" in fields and "cpu part" in fields:
        name = f"unknown (CPU implementer {fields['cpu implementer']} part {fields['cpu part']})"
    else:
        name = "unknown"
    return name[:MAX_TEXT]


def meta_from_env(source: str, env: Mapping[str, str], cpuinfo: str, now: str) -> Meta:
    """Build a Meta from GITHUB_SHA and GITHUB_RUN_ID in `env`, the /proc/cpuinfo text and a UTC timestamp `now`.

    ValueError names the variable that is missing or malformed.
    """
    sha = env.get("GITHUB_SHA", "")
    if _COMMIT.fullmatch(sha) is None:
        raise ValueError(f"GITHUB_SHA must be a 40-hex commit, got {sha!r:.60}")
    run = env.get("GITHUB_RUN_ID", "")
    if not run.isascii() or not run.isdigit():
        raise ValueError(f"GITHUB_RUN_ID must be an integer, got {run!r:.60}")
    return Meta(date=now, commit=sha, source=source, runner=cpu_name(cpuinfo), run_id=int(run))


def _read_cpuinfo() -> str:
    """The text of /proc/cpuinfo, or "" where it cannot be read (non-Linux)."""
    try:
        return Path("/proc/cpuinfo").read_text(errors="replace")
    except OSError:
        return ""


def _utc_now() -> str:
    """The current UTC time as `YYYY-MM-DDTHH:MM:SSZ`."""
    return datetime.now(UTC).strftime(DATE_FORMAT)


def _append(out: Path, rows: Iterable[Row]) -> None:
    """Append `rows`, one JSON line each, to `out` (created if absent). Writes nothing for an empty list."""
    text = "".join(f"{_dump(r)}\n" for r in rows)
    if text:
        with out.open("a", encoding="utf-8") as f:
            f.write(text)


def _parser() -> argparse.ArgumentParser:
    """The argument parser for every subcommand (see the module docstring)."""
    parser = argparse.ArgumentParser(description="Turn job measurements into dashboard rows.")
    sub = parser.add_subparsers(dest="command", required=True)

    def command(name: str, *positionals: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name)
        for pos in positionals:
            p.add_argument(pos, type=Path)
        p.add_argument("--out", type=Path, required=True)
        if name != "merge":
            p.add_argument("--source", choices=SOURCES, default="nightly")
        return p

    command("from-bench", "input")
    command("from-cosine", "input")
    fuzz = command("from-libfuzzer", "input")
    fuzz.add_argument("--target", required=True)
    fuzz.add_argument("--corpus-files", type=int)
    fuzz.add_argument("--seconds", type=int)
    command("from-gcovr", "input")
    command("from-binary-size", "input")
    status = command("status")
    status.add_argument("--workflow", required=True)
    status.add_argument("--job", required=True)
    status.add_argument("--outcome", required=True, choices=["success", "failure", "timed-out", "cancelled"])
    merge_cmd = command("merge", "existing")
    merge_cmd.add_argument("new", type=Path, nargs="+")
    merge_cmd.add_argument(
        "--skip-bad",
        action="store_true",
        help="skip a NEW file that is invalid as a whole (warning) instead of failing",
    )
    return parser


def _read_rows(path: Path) -> list[Row]:
    """The validated rows of the JSONL file `path`; ValueError (naming the file, the line and the field) if any line is
    not a valid row, OSError if it cannot be read. All or nothing: a bad line fails the whole file."""
    rows: list[Row] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError("not a JSON object")
            rows.append(validate(obj))
        except RecursionError as e:  # JSON nested deeper than the interpreter allows
            raise ValueError(f"{path.name} line {number}: JSON is nested too deeply") from e
        except ValueError as e:
            raise ValueError(f"{path.name} line {number}: {e}") from e
    return rows


def _merge_files(
    existing: Path, new: Sequence[Path], out: Path, skip_bad: bool = False, env: Mapping[str, str] | None = None
) -> None:
    """Write the merge of `existing` (a missing file counts as empty) and every row of the `new` files to `out`.

    A new file with an invalid row (or an unreadable one) raises ValueError/OSError, unless `skip_bad`: then that file
    is skipped as a whole (none of its rows are merged) and reported as a `::warning` on stderr naming the path and the
    reason and, when GITHUB_STEP_SUMMARY is set in `env`, in the job summary; the other files are merged. `existing` is
    never skipped: a broken data file is a real error.
    """
    old_text = existing.read_text(encoding="utf-8") if existing.exists() else ""
    rows: list[Row] = []
    skipped: list[str] = []
    for path in new:
        try:
            rows.extend(_read_rows(path))
        except (ValueError, OSError) as e:
            if not skip_bad:
                raise
            skipped.append(f"{path}: {e}")
    out.write_text(merge(old_text, rows), encoding="utf-8")
    for message in skipped:
        print(f"::warning::dashboard_data: skipped rows file {message}", file=sys.stderr)
    summary = (env or {}).get("GITHUB_STEP_SUMMARY")
    if skipped and summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(f"### Dashboard: {len(skipped)} rows file(s) skipped\n\n" + "".join(f"- `{m}`\n" for m in skipped))


def main(
    argv: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    cpuinfo: str | None = None,
    now: str | None = None,
) -> int:
    """Run one subcommand. `env`, `cpuinfo` and `now` default to the real environment, /proc/cpuinfo and the clock;
    tests pass them in. Returns 0, or 1 after printing the reason to stderr when the input is unusable."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "merge":
            _merge_files(args.existing, args.new, args.out, args.skip_bad, os.environ if env is None else env)
            return 0
        if args.command == "status" and args.outcome == "cancelled":
            return 0  # a cancelled job says nothing about the code; leave the file alone
        meta = meta_from_env(
            args.source,
            os.environ if env is None else env,
            _read_cpuinfo() if cpuinfo is None else cpuinfo,
            _utc_now() if now is None else now,
        )
        rows: list[Row]
        if args.command == "status":
            rows = [status_row(args.workflow, args.job, args.outcome == "success", meta)]
        elif args.command == "from-binary-size":
            rows = from_binary_size(args.input.stat().st_size, meta)
        else:
            text = args.input.read_text(encoding="utf-8", errors="replace")
            if args.command == "from-bench":
                rows = from_bench(text, meta)
            elif args.command == "from-cosine":
                rows = from_cosine(parse_min_cosine(text), meta)
            elif args.command == "from-libfuzzer":
                rows = from_libfuzzer(text, args.target, meta, args.corpus_files, args.seconds)
            else:
                rows = from_gcovr_summary(text, meta)
        _append(args.out, rows)
    except (ValueError, OSError) as e:
        print(f"dashboard_data: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
