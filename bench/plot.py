#!/usr/bin/env python3
"""Charts and the README table from bench/results/*.jsonl. Stdlib only; writes SVG text, never parses XML.

usage: bench/plot.py [--results DIR] [--out DIR] [--markdown]

Writes OUT/ladder.svg, OUT/vs-llamacpp.svg and OUT/roofline.svg (default docs/bench), or with --markdown
prints the README tables instead. Rows with load1 > 2 are noisy: drawn hollow, marked † in tables.
nibble vs llama.cpp comes from the paired ABBA blocks in abba.jsonl (tools/bench_final.sh --abba); the
thread sweep and roofline come from the single session in m6-final.jsonl / llamacpp-b11715-final.jsonl.
Other files (session2-discarded.jsonl, llamacpp-b11715.jsonl) are read but not plotted.
Rows without a "model" (the hand-recorded dispatch-empty timings) are left out.
"""
import argparse
import glob
import html
import json
import math
import os
import re
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOISY = 2.0
MODELS = [  # (file, label, colour)
    ("qwen2.5-0.5b-instruct-q4_0.gguf", "Qwen2.5-0.5B Q4_0", "#1f77b4"),
    ("qwen2.5-0.5b-instruct-q8_0.gguf", "Qwen2.5-0.5B Q8_0", "#ff7f0e"),
    ("qwen2.5-0.5b-instruct-fp16.gguf", "Qwen2.5-0.5B F16", "#2ca02c"),
    ("llama-3.2-1b-instruct-f16.gguf", "Llama-3.2-1B F16", "#d62728"),
    ("tinyllama-1.1b-chat-f16.gguf", "TinyLlama-1.1B F16", "#9467bd"),
]
# GGUF file sizes stand in for bytes read per token (weights dominate). The sizes of models/<file> are
# used when present; this table is the fallback for checkouts without the models.
MODEL_BYTES = {
    "qwen2.5-0.5b-instruct-q4_0.gguf": 360662880,
    "qwen2.5-0.5b-instruct-q8_0.gguf": 675710816,
    "qwen2.5-0.5b-instruct-fp16.gguf": 1266425696,
    "llama-3.2-1b-instruct-f16.gguf": 2479595264,
    "tinyllama-1.1b-chat-f16.gguf": 2201017664,
}
TOOLS = [("nibble", "#1f77b4"), ("llama.cpp", "#ff7f0e")]
TG, PP = "tg128", "pp512"
FINAL = ("m6-final", "llamacpp-b11715-final")
ABBA_NOTE = re.compile(r"^abba block (\S+) (A1|B1|B2|A2)$")


def load(results):
    """{file stem: [row]}, rows without a model dropped."""
    out = {}
    for path in sorted(glob.glob(os.path.join(results, "*.jsonl"))):
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        out[os.path.basename(path)[: -len(".jsonl")]] = [r for r in rows if "model" in r]
    return out


def latest(rows):
    """One row per (tool, model, test, threads): the newest by date."""
    best = {}
    for r in rows:
        k = (r["tool"], r["model"], r["test"], r["threads"])
        if k not in best or r["date"] >= best[k]["date"]:
            best[k] = r
    return best


def final_rows(data):
    rows = [r for stem, rs in data.items() if stem in FINAL for r in rs]
    return latest(rows)


def session(final):
    """'YYYY-MM-DD HH:MM–HH:MM UTC, nibble <commit>' span of the single-session rows."""
    dates = sorted(r["date"] for r in final.values())
    commits = ", ".join(sorted({r["commit"] for r in final.values() if r["tool"] == "nibble"}))
    return f"{dates[0][:10]} {dates[0][11:16]}–{dates[-1][11:16]} UTC, nibble {commits}" if dates else "no data"


def abba(data):
    """{(model, test, threads): [block]} from complete ABBA blocks; block = (ratio, nibble, llama.cpp, noisy).
    ratio = mean(llama.cpp A1, A2) / mean(nibble B1, B2), each run's median tok/s."""
    runs = {}
    for r in data.get("abba", []):
        m = ABBA_NOTE.match(r.get("note", ""))
        if m and r["unit"] == "tok/s":
            runs.setdefault((r["model"], r["test"], r["threads"], m[1]), {})[m[2]] = r
    out = {}
    for (model, test, t, _), pos in sorted(runs.items(), key=lambda kv: min(r["date"] for r in kv[1].values())):
        if len(pos) < 4:
            continue  # incomplete block
        a = (pos["A1"]["median"] + pos["A2"]["median"]) / 2
        b = (pos["B1"]["median"] + pos["B2"]["median"]) / 2
        out.setdefault((model, test, t), []).append((a / b, b, a, any(noisy(r) for r in pos.values())))
    return out


def pick(final, tool, model, test, threads=None):
    """Best row for tool/model/test (see rank), or the row at `threads`."""
    rs = [r for (t, m, te, th), r in final.items() if (t, m, te) == (tool, model, test)]
    if threads is not None:
        rs = [r for r in rs if r["threads"] == threads]
    return max(rs, key=rank) if rs else None


def noisy(r):
    return r["load1"] > NOISY


def rank(r):
    """Clean rows beat noisy ones; then the highest median."""
    return (not noisy(r), r["median"])


# --- SVG ---------------------------------------------------------------------------------------------

def esc(s):
    return html.escape(str(s), quote=True)


class Svg:
    def __init__(self, w, h, title):
        self.w, self.h = w, h
        self.parts = [f'<rect width="{w}" height="{h}" fill="white"/>',
                      self.text(w / 2, 22, title, size=15, anchor="middle", weight="bold")]

    @staticmethod
    def text(x, y, s, size=11, anchor="start", weight="normal", fill="#222", rotate=None):
        rot = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate is not None else ""
        return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="sans-serif" font-size="{size}" '
                f'text-anchor="{anchor}" font-weight="{weight}" fill="{fill}"{rot}>{esc(s)}</text>')

    def add(self, s):
        self.parts.append(s)

    def line(self, x1, y1, x2, y2, stroke="#ccc", width=1, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" '
                 f'stroke-width="{width}"{d}/>')

    def rect(self, x, y, w, h, colour, hollow):
        fill = "white" if hollow else colour
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{max(h, 0):.1f}" fill="{fill}" '
                 f'stroke="{colour}" stroke-width="2"/>')

    def dot(self, x, y, colour, hollow):
        fill = "white" if hollow else colour
        self.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{fill}" stroke="{colour}" stroke-width="2"/>')

    def legend(self, x, y, items):
        for i, (label, colour) in enumerate(items):
            self.add(f'<rect x="{x:.1f}" y="{y + i * 16 - 9:.1f}" width="10" height="10" fill="{colour}"/>')
            self.add(self.text(x + 15, y + i * 16, label))

    def note(self, s):
        self.add(self.text(10, self.h - 8, s, size=10, fill="#555"))

    def __str__(self):
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
                f'viewBox="0 0 {self.w} {self.h}">\n' + "\n".join(self.parts) + "\n</svg>\n")


def nice_max(v):
    """Round v up to 1, 2 or 5 times a power of ten."""
    if v <= 0:
        return 1.0
    p = 10 ** math.floor(math.log10(v))
    return next(m * p for m in (1, 2, 5, 10) if m * p >= v)


def y_axis(svg, x0, x1, y_of, ticks, fmt):
    for v in ticks:
        y = y_of(v)
        svg.line(x0, y, x1, y)
        svg.add(svg.text(x0 - 5, y + 4, fmt(v), size=10, anchor="end"))


NOISE_NOTE = "Hollow = noisy (1-min load average > 2 when the run started)."


def ladder(data):
    steps = [s for s in data if s.startswith("m6-")]
    pts = {}  # (step, model) -> best nibble tg row at that step
    sizes = {}
    for s in steps:
        tg = [r for r in data[s] if r["tool"] == "nibble" and r["test"].startswith("tg")]
        sizes[s] = sorted({r["test"] for r in tg})
        for r in tg:
            k = (s, r["model"])
            if k not in pts or rank(r) > rank(pts[k]):
                pts[k] = r
    w, h, left, right, top, bottom = 900, 460, 60, 170, 40, 90
    svg = Svg(w, h, "nibble decode tok/s per M6 step (each step one session, best t, log scale)")
    vals = [r["median"] for r in pts.values()] or [1]
    lo = 10 ** math.floor(math.log10(min(vals)))
    hi = nice_max(max(vals))
    y_of = lambda v: top + (h - top - bottom) * (1 - (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)))
    ticks = [m * 10 ** e for e in range(int(math.log10(lo)), int(math.log10(hi)) + 1) for m in (1, 2, 5)
             if lo <= m * 10 ** e <= hi]
    y_axis(svg, left, w - right, y_of, ticks, lambda v: f"{v:g}")
    x_of = lambda i: left + 20 + i * (w - left - right - 40) / max(len(steps) - 1, 1)
    for i, s in enumerate(steps):
        svg.add(svg.text(x_of(i), h - bottom + 16, s[3:], size=10, anchor="middle"))
        reduced = [t for t in sizes[s] if t != TG]
        svg.add(svg.text(x_of(i), h - bottom + 30, ",".join(sizes[s]), size=9, anchor="middle",
                         fill="#b00" if reduced else "#555"))
    for model, label, colour in MODELS:
        xy = [(x_of(i), y_of(pts[(s, model)]["median"]), pts[(s, model)]) for i, s in enumerate(steps)
              if (s, model) in pts]
        if len(xy) > 1:
            d = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in xy)
            svg.add(f'<polyline points="{d}" fill="none" stroke="{colour}" stroke-width="1.5"/>')
        for x, y, r in xy:
            svg.dot(x, y, colour, noisy(r))
    svg.add(svg.text(left - 45, top + (h - top - bottom) / 2, "tok/s", anchor="middle", rotate=-90))
    svg.legend(w - right + 15, top + 20, [(lab, c) for _, lab, c in MODELS])
    svg.add(svg.text(left, h - 40, "Sizes under each step; red = reduced size (tg16, not tg128): earlier "
                     "steps used shorter runs on a busier machine, so", size=10, fill="#b00"))
    svg.add(svg.text(left, h - 26, "they are not directly comparable with the final tg128 points. Each step "
                     "keeps its own thread counts and load.", size=10, fill="#b00"))
    svg.note(NOISE_NOTE)
    return str(svg)


def bars(final, title, panels, value, unit, ref=None, caption=None):
    """Grouped bars: one panel per test, per model one bar per tool at that tool's best t."""
    w, h, top, bottom = 900, 440, 50, 90
    pw = (w - 80) / len(panels)
    svg = Svg(w, h, title)
    for pi, test in enumerate(panels):
        x0 = 60 + pi * pw
        rows = {(m, t): pick(final, t, m, test) for m, _, _ in MODELS for t, _ in TOOLS}
        rows = {k: r for k, r in rows.items() if r and value(r) is not None}
        vals = [value(r) for r in rows.values()]
        hi = nice_max(max(vals + ([ref] if ref else []) or [1]) * 1.1)
        y_of = lambda v: top + (h - top - bottom) * (1 - v / hi)
        y_axis(svg, x0, x0 + pw - 20, y_of, [hi * i / 5 for i in range(6)], lambda v: f"{v:g}")
        svg.add(svg.text(x0 + (pw - 20) / 2, top - 8, f"{test} ({unit})", size=12, anchor="middle", weight="bold"))
        if ref:
            svg.line(x0, y_of(ref), x0 + pw - 20, y_of(ref), stroke="#b00", width=1.5, dash="4 3")
        gw = (pw - 20) / len(MODELS)
        for mi, (model, label, _) in enumerate(MODELS):
            gx = x0 + mi * gw
            for ti, (tool, colour) in enumerate(TOOLS):
                r = rows.get((model, tool))
                if not r:
                    continue
                bw = gw * 0.35
                bx = gx + gw * 0.12 + ti * bw
                v = value(r)
                svg.rect(bx, y_of(v), bw - 2, y_of(0) - y_of(v), colour, noisy(r))
                svg.add(svg.text(bx + bw / 2, y_of(v) - 14, f"{v:.3g}", size=9, anchor="middle"))
                svg.add(svg.text(bx + bw / 2, y_of(v) - 4, f"t{r['threads']}", size=8, anchor="middle", fill="#555"))
            svg.add(svg.text(gx + gw / 2, h - bottom + 14, label.split(" ")[0], size=9, anchor="middle"))
            svg.add(svg.text(gx + gw / 2, h - bottom + 26, label.split(" ")[1], size=9, anchor="middle"))
    svg.legend(70, h - 42, TOOLS)
    svg.add(svg.text(170, h - 42, "Each tool at its own best thread count (t under the value).", size=10, fill="#555"))
    if caption:
        svg.add(svg.text(170, h - 26, caption, size=10, fill="#555"))
    svg.note(NOISE_NOTE)
    return str(svg)


def ratio_chart(blocks):
    """Per model and test: median llama.cpp/nibble block ratio per t, with the min-max range."""
    ts = sorted({t for (_, _, t) in blocks})
    tcol = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    w, h, top, bottom = 900, 440, 50, 90
    panels = [TG, PP]
    pw = (w - 80) / len(panels)
    svg = Svg(w, h, "llama.cpp ÷ nibble tok/s, paired ABBA blocks (median, bar = min–max over blocks)")
    hi = math.ceil(max([r for bs in blocks.values() for r, *_ in bs] + [1.5]) * 1.1 * 2) / 2  # 0.5 steps
    y_of = lambda v: top + (h - top - bottom) * (1 - v / hi)
    for pi, test in enumerate(panels):
        x0 = 60 + pi * pw
        y_axis(svg, x0, x0 + pw - 20, y_of, [hi * i / 5 for i in range(6)], lambda v: f"{v:.2g}")
        svg.add(svg.text(x0 + (pw - 20) / 2, top - 8, test, size=12, anchor="middle", weight="bold"))
        svg.line(x0, y_of(1), x0 + pw - 20, y_of(1), stroke="#444", width=1.5)
        svg.line(x0, y_of(1.5), x0 + pw - 20, y_of(1.5), stroke="#b00", width=1.5, dash="4 3")
        gw = (pw - 20) / len(MODELS)
        bw = gw * 0.7 / max(len(ts), 1)
        for mi, (model, label, _) in enumerate(MODELS):
            gx = x0 + mi * gw
            for ti, t in enumerate(ts):
                bs = blocks.get((model, test, t))
                if not bs:
                    continue
                rs = [r for r, *_ in bs]
                med, colour = statistics.median(rs), tcol[ti % len(tcol)]
                bx = gx + gw * 0.15 + ti * bw
                svg.rect(bx, y_of(med), bw - 3, y_of(0) - y_of(med), colour, any(n for *_, n in bs))
                cx = bx + (bw - 3) / 2
                svg.line(cx, y_of(min(rs)), cx, y_of(max(rs)), stroke="#222", width=1.5)
                svg.line(cx - 4, y_of(min(rs)), cx + 4, y_of(min(rs)), stroke="#222", width=1.5)
                svg.line(cx - 4, y_of(max(rs)), cx + 4, y_of(max(rs)), stroke="#222", width=1.5)
                svg.add(svg.text(cx, y_of(max(rs)) - 4, f"{med:.2f}", size=9, anchor="middle"))
            svg.add(svg.text(gx + gw / 2, h - bottom + 14, label.split(" ")[0], size=9, anchor="middle"))
            svg.add(svg.text(gx + gw / 2, h - bottom + 26, label.split(" ")[1], size=9, anchor="middle"))
    svg.legend(70, h - 42, [(f"t = {t}", tcol[i % len(tcol)]) for i, t in enumerate(ts)])
    svg.add(svg.text(170, h - 42, "Above 1: llama.cpp faster. Solid line = parity; dashed red = the 1.5× target.",
                     size=10, fill="#555"))
    svg.add(svg.text(170, h - 26, "Block = llama.cpp, nibble, nibble, llama.cpp back to back; ratio = mean A ÷ mean B.",
                     size=10, fill="#555"))
    svg.note("Hollow = some run in the blocks was noisy (1-min load average > 2 when it started).")
    return str(svg)


def membw(final, threads):
    """The highest nibble membw probe at `threads` across all models' runs, else the highest at any t."""
    rs = [r for (t, m, te, th), r in final.items() if te == "membw"]
    at = [r["median"] for r in rs if r["threads"] == threads]
    return max(at or [r["median"] for r in rs] or [float("nan")])


def model_bytes(model):
    path = os.path.join(ROOT, "models", model)
    return os.path.getsize(path) if os.path.exists(path) else MODEL_BYTES.get(model)


def weight_bw_pct(final, r):
    """None if the model's size is unknown."""
    b = model_bytes(r["model"])
    return 100 * b * r["median"] / 1e9 / membw(final, r["threads"]) if b else None


def markdown(data):
    final, blocks = final_rows(data), abba(data)
    mark = lambda r: "†" if noisy(r) else ""
    out = ["Paired ABBA blocks (llama.cpp ÷ nibble; tok/s are medians of the per-block means):", "",
           "| model | test | t | blocks | llama.cpp ÷ nibble, median | range | nibble tok/s | llama.cpp tok/s |",
           "|---|---|---|---|---|---|---|---|"]
    for model, label, _ in MODELS:
        for test in (TG, PP):
            for t in sorted({t for (m, te, t) in blocks if (m, te) == (model, test)}):
                bs = blocks[(model, test, t)]
                rs = [r for r, *_ in bs]
                med = lambda i: statistics.median(b[i] for b in bs)
                nz = "†" if any(n for *_, n in bs) else ""
                out.append(f"| {label} | {test} | {t} | {len(bs)}{nz} | {statistics.median(rs):.2f}× | "
                           f"{min(rs):.2f}–{max(rs):.2f}× | {med(1):.1f} | {med(2):.1f} |")
    for model, label, _ in MODELS:
        ts = sorted({th for (t, m, te, th) in final if m == model and te == TG})
        if len(ts) < 3:
            continue
        out += ["", f"Thread sweep, {label}, tok/s: single session ({session(final)}), used only to choose "
                "the default thread count:", "",
                "| t | nibble tg128 | llama.cpp tg128 | nibble pp512 | llama.cpp pp512 |", "|---|---|---|---|---|"]
        for th in ts:
            c = [pick(final, tool, model, test, th) for test in (TG, PP) for tool in ("nibble", "llama.cpp")]
            f = lambda r: f"{r['median']:.1f}{mark(r)}" if r else "–"
            out.append(f"| {th} | {f(c[0])} | {f(c[1])} | {f(c[2])} | {f(c[3])} |")
    out += ["", "† noisy: 1-minute load average > 2 when a run started (for blocks: in any of its runs)."]
    return "\n".join(out) + "\n"


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=os.path.join(ROOT, "bench", "results"))
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "bench"))
    ap.add_argument("--markdown", action="store_true", help="print the README tables instead of writing SVGs")
    a = ap.parse_args(argv)
    data = load(a.results)
    if a.markdown:
        sys.stdout.write(markdown(data))
        return 0
    final = final_rows(data)
    charts = {
        "ladder.svg": ladder(data),
        "vs-llamacpp.svg": ratio_chart(abba(data)),
        "roofline.svg": bars(final, f"Decode weight-read BW, % of membw probe (single session, {session(final)})",
                             [TG], lambda r: weight_bw_pct(final, r), "% of membw", ref=100,
                             caption="membw = highest probe across models at that t. Bytes = GGUF file size; "
                                     "TinyLlama's untied token_embd (~131 MB) is counted but not read per token."),
    }
    os.makedirs(a.out, exist_ok=True)
    for name, svg in charts.items():
        with open(os.path.join(a.out, name), "w", encoding="utf-8") as f:
            f.write(svg)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
