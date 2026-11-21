"""Render the CI dashboard: one static page, six SVG charts and a shields.io badge, from the rows of the data branch.

usage: dashboard_render.py ROWS.jsonl --out DIR --repo OWNER/NAME [--now YYYY-MM-DDTHH:MM:SSZ]

ROWS.jsonl holds validated rows (see dashboard_data.py; a missing file counts as empty). DIR receives:

    index.html    the page: status table of every job's latest result plus the charts as inline <img>. No JavaScript,
                  no external fonts or scripts; it reads fine on a phone.
    speed-prefill.svg, speed-decode.svg
                  tok/s per model and thread count, from the `<model>/<test>/t<threads>` bench keys (shared CI
                  runners: a trend, not a benchmark); the legend reads `Q4_0, 1 thread`
    accuracy.svg  min cosine against llama.cpp per model, the minimum over prompts (y axis from 0.99)
    fuzz.svg      cumulative fuzzing hours per target
    size.svg      stripped release binary size
    coverage.svg  line coverage of src/ and cli/
    badge.json    shields.io endpoint: nightly "passing" iff every current job's latest `nightly/*` status is 1 and
                  the newest of them is at most STALE_HOURS (36) old by the clock; "stale" (orange) when they all
                  passed but the newest is older, so rows that stopped arriving (schedules disabled, a broken
                  dashboard run, a skipped artifact) never leave a green badge standing; "failing" wins over "stale"

The charts are drawn by bench/plot.py's `line_chart` and show the last CHART_DAYS (90) days; the status table and
badge only count jobs that reported within STATUS_DAYS (14) days, so a removed or renamed job ages out. Both spans are
counted back from the newest row, not from the clock. Only staleness is measured against the clock (`--now`, default
the current time): the badge as above, a notice above the status table, and "stale" on every nightly job's row older
than STALE_HOURS (weekly portability rows are not marked). Every value that reaches HTML is escaped (the SVG builder
escapes its own text), and `--repo` is validated before it goes into a URL. A malformed row is skipped and reported
on stderr as a `::warning` (and in the job summary): merge already validated what it stored, so this only guards
against a hand-edited data branch.
Stdlib only. Exit status is 0 on success and 1 (message on stderr) for an unusable `--repo` or an unwritable `--out`.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench"))

import plot
from dashboard_data import Row, validate

Point = tuple[float, float]  # (x = unix seconds, y = value)

FILES = (
    "index.html",
    "speed-prefill.svg",
    "speed-decode.svg",
    "accuracy.svg",
    "fuzz.svg",
    "size.svg",
    "coverage.svg",
    "badge.json",
)
_REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
MAX_LABEL = 30  # characters that fit the chart legend (190 px at 11 px sans-serif)
DAY = 86400  # seconds
CHART_DAYS = 90  # charts show this much history, counted back from the newest row (see `recent`)
STATUS_DAYS = 14  # a job with no result for this long (counted the same way) is gone from the status table and badge
STALE_HOURS = 36  # a nightly result older than this by the clock is stale (a night missed, with a margin)
NIGHTLY = "nightly/"  # key prefix of the nightly jobs' statuses, the only ones the badge and the staleness rule read
WORKFLOW_ORDER = ("nightly", "portability", "ci")  # status table order; other workflows follow alphabetically
# One colour per series, cycling: the first ten of plot.MODELS' palette (Tableau 10).
PALETTE = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf")


def parse_rows(text: str) -> tuple[list[Row], list[str]]:
    """The valid rows of JSONL `text`, and one message per line that was skipped ("line N: why"). Blank lines are
    ignored silently."""
    rows: list[Row] = []
    problems: list[str] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError("not a JSON object")
            rows.append(validate(obj))
        except RecursionError:  # JSON nested deeper than the interpreter allows
            problems.append(f"line {number}: JSON is nested too deeply")
        except ValueError as e:
            problems.append(f"line {number}: {e}"[:200])
    return rows, problems


def _unix(date: object) -> float:
    """Unix seconds of a row's `YYYY-MM-DDTHH:MM:SSZ` date (validated by parse_rows)."""
    return datetime.strptime(str(date), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC).timestamp()


def _order(r: Row) -> tuple[float, int]:
    """Sort key that puts a row after every older one: date first, run id to break ties."""
    run_id = r["run_id"]
    return (_unix(r["date"]), run_id if isinstance(run_id, int) else 0)


def _num(r: Row) -> float:
    """The row's `value` (validate made it a float)."""
    value = r["value"]
    return float(value) if isinstance(value, int | float) else float("nan")


def recent(rows: Sequence[Row], days: int) -> list[Row]:
    """The rows dated within `days` days of the newest row (counting from the data, not the clock, so a quiet
    repository still shows its last results and the output depends on the rows alone). [] for no rows."""
    if not rows:
        return []
    cutoff = max(_unix(r["date"]) for r in rows) - days * DAY
    return [r for r in rows if _unix(r["date"]) >= cutoff]


def current_statuses(rows: Sequence[Row]) -> dict[str, Row]:
    """The latest `job_status` row of every `<workflow>/<job>` that reported within STATUS_DAYS, so a job that was
    renamed or removed ages out instead of keeping its last (maybe failing) row forever."""
    return latest_by_key(recent(rows, STATUS_DAYS), "job_status")


def latest_by_key(rows: Sequence[Row], metric: str) -> dict[str, Row]:
    """The newest row of `metric` for every key (newest = latest date, then highest run id)."""
    newest: dict[str, Row] = {}
    for r in rows:
        if r["metric"] != metric:
            continue
        key = str(r["key"])
        if key not in newest or _order(r) > _order(newest[key]):
            newest[key] = r
    return newest


def _clip(label: str) -> str:
    """`label` cut to MAX_LABEL characters, ending in an ellipsis when it was cut."""
    return label if len(label) <= MAX_LABEL else label[: MAX_LABEL - 1] + "…"


def _series(points: Mapping[str, Sequence[Point]], labels: Sequence[str]) -> list[plot.Series]:
    """Chart series for `points` (name -> points), one colour each in PALETTE order, labelled with `labels`."""
    return [
        (label, PALETTE[i % len(PALETTE)], pts)
        for i, (label, pts) in enumerate(zip(labels, points.values(), strict=True))
    ]


_BENCH_KEY = re.compile(r"(?P<model>.+)/(?P<test>[^/]+)/t(?P<threads>\d+)")


def _bench_parts(key: str) -> tuple[str, str, int] | None:
    """(model file stem without `.gguf`, test, threads) of a bench key `<model file>/<test>/t<threads>`, or None."""
    m = _BENCH_KEY.fullmatch(key)
    if m is None:
        return None
    return m["model"].removesuffix(".gguf"), m["test"], int(m["threads"])


def _clip_middle(text: str, width: int) -> str:
    """`text` cut to `width` characters by replacing its middle with an ellipsis (both ends stay recognisable)."""
    if len(text) <= width:
        return text
    head = width // 2
    return text[:head] + "…" + text[len(text) - (width - 1 - head) :]


def speed_labels(keys: Sequence[str]) -> list[str]:
    """Legend labels for the bench keys of one chart, readable without knowing the key format: `Q4_0, 1 thread` (the
    quantization is the last dash-separated part of the model file name, upper-cased; the count comes from `t<N>`).
    The test is named too (`, pp512`) when the chart mixes tests. If two keys would still get the same label, more is
    shown, in order until they differ: the test, the whole model stem instead of the quantization, then both. A key
    that does not parse keeps its raw text. Labels are at most MAX_LABEL characters: a long model name is shortened
    in the middle so the thread count and test always stay, and labels that are still equal after that get ` (2)`,
    ` (3)` and so on, so no two series ever share a label."""
    parsed = [_bench_parts(k) for k in keys]
    mixed = len({p[1] for p in parsed if p is not None}) > 1

    def build(key: str, parts: tuple[str, str, int] | None, stage: int) -> str:
        if parts is None:
            return _clip(key)
        stem, test, threads = parts
        model = stem.rsplit("-", 1)[-1].upper() if stage < 2 else stem
        suffix = f", {threads} thread{'' if threads == 1 else 's'}" + (f", {test}" if stage % 2 == 1 else "")
        room = MAX_LABEL - len(suffix)
        return _clip(model + suffix) if room < 4 else _clip_middle(model, room) + suffix

    # Stage 0: quantization only; 1: plus the test; 2: full stem; 3: full stem plus the test.
    labels: list[str] = []
    for stage in range(1 if mixed else 0, 4):
        labels = [build(k, p, stage) for k, p in zip(keys, parsed, strict=True)]
        if len(set(labels)) == len(labels):
            return labels
    seen: dict[str, int] = {}
    unique: list[str] = []
    for label in labels:
        seen[label] = seen.get(label, 0) + 1
        tag = f" ({seen[label]})" if seen[label] > 1 else ""
        unique.append(label[: MAX_LABEL - len(tag)] + tag)
    return unique


def _speed_order(key: str) -> tuple[int, str, int]:
    """Sort key for bench keys: parsed ones by model then thread count (numerically), the rest after them."""
    parts = _bench_parts(key)
    return (0, parts[0], parts[2]) if parts else (1, key, 0)


def _speed_points(rows: Sequence[Row], test_prefix: str) -> dict[str, list[Point]]:
    """Points per bench key whose test starts with `test_prefix` (`pp` prefill, `tg` decode), keys ordered by model and
    thread count. A key without a `<model>/<test>/t<threads>` shape belongs to neither chart."""
    by_key: dict[str, list[Point]] = {}
    for r in rows:
        if r["metric"] != "bench_tok_s":
            continue
        key = str(r["key"])
        parts = _bench_parts(key)
        if parts is not None and parts[1].startswith(test_prefix):
            by_key.setdefault(key, []).append((_unix(r["date"]), _num(r)))
    return dict(sorted(by_key.items(), key=lambda kv: _speed_order(kv[0])))


def speed_series(rows: Sequence[Row], test_prefix: str) -> list[plot.Series]:
    """One series per bench key of the `test_prefix` tests, ordered by model and thread count."""
    by_key = _speed_points(rows, test_prefix)
    return _series(by_key, speed_labels(list(by_key)))


def speed_tests(rows: Sequence[Row], test_prefix: str) -> list[str]:
    """The distinct test names (`pp512`, `tg128`, ...) among the bench keys a `test_prefix` chart plots, sorted."""
    return sorted({parts[1] for key in _speed_points(rows, test_prefix) if (parts := _bench_parts(key)) is not None})


def speed_title(test_prefix: str, tests: Sequence[str]) -> str:
    """Title of a speed chart that does not claim more than its data: `Prefill speed (512-token prompt)` when every
    plotted test is the same `pp<N>` (likewise `Decode speed (128 generated tokens)` for `tg<N>`), the bare name when
    nothing or something other than `<prefix><N>` is plotted, and `(several ...)` when the sizes differ."""
    prefill = test_prefix == "pp"
    name = "Prefill speed" if prefill else "Decode speed"
    if len(tests) > 1:
        return f"{name} (several {'prompt sizes' if prefill else 'generation lengths'})"
    if len(tests) == 1 and (m := re.fullmatch(rf"{test_prefix}(\d+)", tests[0])) is not None:
        return f"{name} ({m[1]}-token prompt)" if prefill else f"{name} ({m[1]} generated tokens)"
    return name


def min_cosine_by_model(rows: Sequence[Row]) -> dict[str, list[Point]]:
    """Per model (the key's part before `/`), one point per run: the minimum cosine over that run's prompts, at the
    run's earliest row date. Points are in time order."""
    best: dict[tuple[str, object], tuple[float, float]] = {}  # (model, run_id) -> (x, min cosine)
    for r in rows:
        if r["metric"] != "min_cosine":
            continue
        ident = (str(r["key"]).partition("/")[0], r["run_id"])
        x, v = _unix(r["date"]), _num(r)
        if ident in best:
            x0, v0 = best[ident]
            best[ident] = (min(x, x0), min(v, v0))
        else:
            best[ident] = (x, v)
    by_model: dict[str, list[Point]] = {}
    for (model, _run), point in best.items():
        by_model.setdefault(model, []).append(point)
    return {m: sorted(p) for m, p in sorted(by_model.items())}


def cumulative_fuzz_hours(rows: Sequence[Row]) -> dict[str, list[Point]]:
    """Per fuzz target, the running sum of `fuzz_seconds` / 3600 at each run, in time order."""
    runs: dict[str, list[Row]] = {}
    for r in rows:
        if r["metric"] == "fuzz_seconds":
            runs.setdefault(str(r["key"]), []).append(r)
    out: dict[str, list[Point]] = {}
    for target, rs in sorted(runs.items()):
        total = 0.0
        pts: list[Point] = []
        for r in sorted(rs, key=_order):
            total += _num(r) / 3600
            pts.append((_unix(r["date"]), total))
        out[target] = pts
    return out


def fuzz_series(rows: Sequence[Row]) -> list[plot.Series]:
    """Chart series of the cumulative fuzz hours: the running totals are summed over all `rows`, then only the points
    within CHART_DAYS of the newest row are drawn (a target with none is left out), so the line starts at the total
    reached by then rather than at zero."""
    if not rows:
        return []
    cutoff = max(_unix(r["date"]) for r in rows) - CHART_DAYS * DAY
    shown = {t: [p for p in pts if p[0] >= cutoff] for t, pts in cumulative_fuzz_hours(rows).items()}
    shown = {t: pts for t, pts in shown.items() if pts}
    return _series(shown, [_clip(t) for t in shown])


def _simple_series(rows: Sequence[Row], metric: str, scale: float = 1.0) -> list[plot.Series]:
    """One series per key of `metric`, values divided by `scale`, in key order."""
    by_key: dict[str, list[Point]] = {}
    for r in rows:
        if r["metric"] == metric:
            by_key.setdefault(str(r["key"]), []).append((_unix(r["date"]), _num(r) / scale))
    by_key = dict(sorted(by_key.items()))
    return _series(by_key, [_clip(k) for k in by_key])


def charts(rows: Sequence[Row]) -> dict[str, str]:
    """File name -> SVG text for the six charts, each over the last CHART_DAYS days (the fuzz total counts all runs)."""
    shown = recent(rows, CHART_DAYS)
    cosine = min_cosine_by_model(shown)
    return {
        "speed-prefill.svg": plot.line_chart(
            speed_title("pp", speed_tests(shown, "pp")), "tokens / second", speed_series(shown, "pp")
        ),
        "speed-decode.svg": plot.line_chart(
            speed_title("tg", speed_tests(shown, "tg")), "tokens / second", speed_series(shown, "tg")
        ),
        "accuracy.svg": plot.line_chart(
            "Accuracy against llama.cpp: min cosine per model",
            "min cosine",
            _series(cosine, [_clip(m.upper()) for m in cosine]),
            y_min=0.99,
        ),
        "fuzz.svg": plot.line_chart("Cumulative fuzzing time", "hours", fuzz_series(rows)),
        "size.svg": plot.line_chart(
            "Stripped release binary size", "MiB", _simple_series(shown, "binary_bytes", 1024 * 1024)
        ),
        "coverage.svg": plot.line_chart(
            "Line coverage of src/ and cli/", "% of lines", _simple_series(shown, "line_coverage"), y_max=100
        ),
    }


def parse_now(text: str) -> float:
    """Unix seconds of a `YYYY-MM-DDTHH:MM:SSZ` time (the `--now` option); ValueError for anything else."""
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC).timestamp()


def _is_stale(r: Row, now: float | None) -> bool:
    """Whether a row is older than STALE_HOURS at `now` (None: never, for rendering that ignores the clock)."""
    return now is not None and now - _unix(r["date"]) > STALE_HOURS * 3600


def newest_nightly(rows: Sequence[Row]) -> Row | None:
    """The newest current `nightly/*` status row, or None when there is none."""
    nightly = [r for k, r in current_statuses(rows).items() if k.startswith(NIGHTLY)]
    return max(nightly, key=_order) if nightly else None


def badge(rows: Sequence[Row], now: float | None = None) -> dict[str, object]:
    """The shields.io endpoint object: "failing" (red) if any current job's latest `nightly/*` status is not 1;
    otherwise "stale" (orange) when the newest of them is older than STALE_HOURS at `now` (unix seconds; None never
    counts as stale), else "passing" (green); "no data" (grey) while there is none. Current means reported within
    STATUS_DAYS."""
    nightly = [r for k, r in current_statuses(rows).items() if k.startswith(NIGHTLY)]
    newest = newest_nightly(rows)
    if not nightly or newest is None:
        message, colour = "no data", "lightgrey"
    elif any(_num(r) != 1.0 for r in nightly):
        message, colour = "failing", "red"
    elif _is_stale(newest, now):
        message, colour = "stale", "orange"
    else:
        message, colour = "passing", "brightgreen"
    return {"schemaVersion": 1, "label": "nightly", "message": message, "color": colour}


def _when(r: Row) -> str:
    """`2026-11-21 03:17 UTC` for a row's date."""
    return str(r["date"])[:16].replace("T", " ") + " UTC"


def _stale_notice(rows: Sequence[Row], now: float | None) -> str:
    """A paragraph saying the nightly results are stale (the newest is older than STALE_HOURS at `now`), or ""."""
    newest = newest_nightly(rows)
    if newest is None or not _is_stale(newest, now):
        return ""
    return (
        f"<p><strong>⚠️ The nightly results are stale:</strong> the newest is from {_when(newest)}, more than "
        f"{STALE_HOURS} hours ago. The nightly workflow has not reported since; check its runs.</p>"
    )


def _status_table(rows: Sequence[Row], repo: str, now: float | None = None) -> str:
    """The HTML status table: the latest `job_status` per `<workflow>/<job>`, grouped by workflow. A nightly job's row
    older than STALE_HOURS at `now` says "stale" after its result."""
    latest = current_statuses(rows)
    if not latest:
        return "<p>No job status recorded yet.</p>"
    groups: dict[str, list[tuple[str, Row]]] = {}
    for key, r in latest.items():
        workflow, _, job = key.partition("/")
        groups.setdefault(workflow, []).append((job or key, r))
    known = [w for w in WORKFLOW_ORDER if w in groups]
    ordered = known + sorted(set(groups) - set(known))
    body: list[str] = []
    for workflow in ordered:
        body.append(f'<tbody><tr><th scope="rowgroup" colspan="4">{html.escape(workflow)}</th></tr>')
        for job, r in sorted(groups[workflow], key=lambda g: g[0]):
            ok = _num(r) == 1.0
            run = f"https://github.com/{repo}/actions/runs/{r['run_id']}"
            stale = " (⚠️ stale)" if key_is_nightly(workflow) and _is_stale(r, now) else ""
            body.append(
                f'<tr><th scope="row">{html.escape(job)}</th>'
                f"<td>{'✅ passing' if ok else '❌ failing'}{stale}</td>"
                f"<td>{_when(r)}</td>"
                f'<td><a href="{run}">run {r["run_id"]}</a></td></tr>'
            )
        body.append("</tbody>")
    head = (
        "<thead><tr>" + "".join(f'<th scope="col">{c}</th>' for c in ("Job", "Result", "When", "Run")) + "</tr></thead>"
    )
    return (
        '<div class="scroll"><table><caption>Latest result of each job</caption>'
        + head
        + "".join(body)
        + "</table></div>"
    )


CSS = """\
:root { color-scheme: light; }
body {
  margin: 0; font: 16px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; color: #1f2328; background: #fff;
}
main { max-width: 960px; margin: 0 auto; padding: 1rem; }
h1 { margin: 0.5rem 0; font-size: 1.6rem; }
h2 { margin-top: 2rem; font-size: 1.25rem; border-bottom: 1px solid #d0d7de; padding-bottom: 0.25rem; }
a { color: #0969da; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; }
caption { text-align: left; font-weight: 600; padding-bottom: 0.25rem; }
th, td { text-align: left; padding: 0.3rem 0.6rem; border-bottom: 1px solid #eaeef2; white-space: nowrap; }
thead th { background: #f6f8fa; }
tbody th[scope="rowgroup"] {
  background: #f6f8fa; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.04em;
}
figure { margin: 1.25rem 0; }
figure img + img { margin-top: 0.75rem; }
figure img { display: block; max-width: 100%; height: auto; border: 1px solid #d0d7de; border-radius: 6px; }
figcaption { font-size: 0.9rem; color: #57606a; }
.muted { color: #57606a; font-size: 0.9rem; }
footer { margin-top: 2rem; }
"""

# (heading, caption, [(file, alt text)]). The alt text says what the chart shows; the SVGs have no scripts or text
# fallback. The speed section has two images under one caption.
SECTIONS = (
    (
        "Speed (trend only)",
        "Measured on shared GitHub-hosted runners, whose speed varies from run to run: read this as a trend, "
        "not a benchmark. The README tables come from an idle dedicated machine. Each chart title gives the prompt or "
        "generation size measured.",
        (
            (
                "speed-prefill.svg",
                "Line chart of prefill speed in tokens per second for each model and thread count, "
                "one point per nightly run",
            ),
            (
                "speed-decode.svg",
                "Line chart of decode speed in tokens per second for each model and thread "
                "count, one point per nightly run",
            ),
        ),
    ),
    (
        "Accuracy against llama.cpp",
        "Minimum cosine over the test prompts, per model and nightly run. Higher is closer to llama.cpp; the axis "
        "starts at 0.99.",
        (
            (
                "accuracy.svg",
                "Line chart of the minimum cosine similarity to llama.cpp per model over time, on an axis starting "
                "at 0.99",
            ),
        ),
    ),
    (
        "Fuzzing",
        "Total time each fuzz target has run in the nightly job.",
        (("fuzz.svg", "Line chart of cumulative fuzzing hours per fuzz target over time"),),
    ),
    (
        "Binary size",
        "Size of the stripped nibble binary built each night.",
        (("size.svg", "Line chart of the stripped release binary size in MiB over time"),),
    ),
    (
        "Line coverage",
        "Line coverage of src/ and cli/ from the unit and integration tests, measured on each push to master.",
        (
            (
                "coverage.svg",
                "Line chart of the percentage of src and cli lines covered by the tests, per push to master",
            ),
        ),
    ),
)


def key_is_nightly(workflow: str) -> bool:
    """Whether a status key's workflow part names the nightly workflow."""
    return f"{workflow}/" == NIGHTLY


def index_html(rows: Sequence[Row], repo: str, now: float | None = None) -> str:
    """The dashboard page for `rows`. `repo` must already be validated; `now` (unix seconds, None to ignore the clock)
    decides what is stale."""
    repo_url = f"https://github.com/{repo}"
    if rows:
        newest = max(rows, key=_order)
        sha = str(newest["commit"])
        updated = (
            f'<p class="muted">Last updated {_when(newest)}, from commit '
            f'<a href="{repo_url}/commit/{sha}"><code>{sha[:7]}</code></a>.</p>'
        )
    else:
        updated = '<p class="muted">No results yet: the first nightly run fills this page.</p>'
    figures = "".join(
        f"<section><h2>{html.escape(heading)}</h2><figure>"
        + "".join(
            f'<img src="{name}" alt="{html.escape(alt, quote=True)}" width="900" height="460">' for name, alt in images
        )
        + f"<figcaption>{html.escape(caption)}</figcaption></figure></section>"
        for heading, caption, images in SECTIONS
    )
    return (
        '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>nibble CI dashboard</title>"
        '<meta name="description" content="Nightly and weekly CI results for nibble, a CPU LLM inference engine.">'
        f"<style>\n{CSS}</style></head><body><main>\n"
        "<h1>nibble CI dashboard</h1>\n"
        f'<p>Results of the scheduled checks for <a href="{repo_url}">{html.escape(repo)}</a>: nightly accuracy, '
        "fuzzing and trend runs, the weekly portability builds, and line coverage on every push to master. "
        f'For what nibble is and its benchmark numbers, see the <a href="{repo_url}#readme">README</a>.</p>\n'
        f"{updated}\n"
        f"<h2>Status</h2>\n{_stale_notice(rows, now)}{_status_table(rows, repo, now)}\n"
        f"{figures}\n"
        f'<footer class="muted">The data is <a href="{repo_url}/blob/dashboard-data/data/rows.jsonl">one JSON Lines '
        "file</a> on the <code>dashboard-data</code> branch. This page is static and needs no JavaScript.</footer>\n"
        "</main></body></html>\n"
    )


def render_rows(rows: Sequence[Row], out_dir: str, repo: str, now: float | None = None) -> list[str]:
    """Write the dashboard for already parsed `rows` into `out_dir` (created if needed) and return the names of the
    files written (FILES). `now` is the clock for the staleness rule (unix seconds; None ignores it, for tests of
    everything else). ValueError if `repo` is not `owner/name`."""
    if _REPO.fullmatch(repo) is None:
        raise ValueError(f"repo must look like owner/name, got {repo!r:.60}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pages = {
        "index.html": index_html(rows, repo, now),
        **charts(rows),
        "badge.json": json.dumps(badge(rows, now)) + "\n",
    }
    for name in FILES:
        (out / name).write_text(pages[name], encoding="utf-8")
    return list(FILES)


def render_site(rows_jsonl: str, out_dir: str, repo: str, now: float | None = None) -> list[str]:
    """Write the dashboard for the rows in `rows_jsonl` (JSONL text) into `out_dir` and return the names of the files
    written; `now` as for render_rows. Malformed rows are skipped silently here; `main` is the caller that reports
    them (see parse_rows)."""
    rows, _problems = parse_rows(rows_jsonl)
    return render_rows(rows, out_dir, repo, now)


def _report_problems(problems: Sequence[str], env: Mapping[str, str]) -> None:
    """Print a `::warning` per skipped row to stderr and, if GITHUB_STEP_SUMMARY is set, list them in the summary."""
    for problem in problems:
        print(f"::warning::dashboard_render: skipped row, {problem}", file=sys.stderr)
    summary = env.get("GITHUB_STEP_SUMMARY")
    if problems and summary:
        lines = [f"### Dashboard: {len(problems)} malformed row(s) skipped", ""]
        lines += [f"- {problem}" for problem in problems[:20]]
        if len(problems) > 20:
            lines.append(f"- ... and {len(problems) - 20} more")
        with open(summary, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


def main(argv: Sequence[str], *, env: Mapping[str, str] | None = None) -> int:
    """Command line entry point (see the module docstring). `env` defaults to the real environment. Returns the exit
    status."""
    parser = argparse.ArgumentParser(description="Render the CI dashboard site from dashboard rows.")
    parser.add_argument("rows", type=Path, help="rows JSONL (a missing file counts as empty)")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--repo", required=True, help="OWNER/NAME, used for links")
    parser.add_argument("--now", help="the time staleness counts to, YYYY-MM-DDTHH:MM:SSZ (default: the clock)")
    args = parser.parse_args(argv)
    try:
        now = parse_now(args.now) if args.now else datetime.now(UTC).timestamp()
        rows, problems = parse_rows(args.rows.read_text(encoding="utf-8") if args.rows.exists() else "")
        _report_problems(problems, os.environ if env is None else env)
        names = render_rows(rows, str(args.out), args.repo, now)
    except (ValueError, OSError) as e:
        print(f"dashboard_render: {e}", file=sys.stderr)
        return 1
    print(f"wrote {', '.join(names)} to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
