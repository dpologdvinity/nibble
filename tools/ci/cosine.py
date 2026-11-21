"""Parse the e2e accuracy lines out of a ctest log and compare two runs.

tests/test_e2e_golden.cpp prints one Catch2 WARN per (model, prompt):

    <prompt> [<model>]: min cosine over <N> positions = <x>, top-1 agreement <A>/<N>

`parse_min_cosine` reads those from the text of `ctest --output-log` (or of the test binary's console output) and
returns them keyed `"<model>/<prompt>"`. `diff_cosines` lists what changed between two such maps; the nightly
accuracy job, the dashboard rows and the PR report all build on it.

Precision: `x` is printed with the default stream precision (6 significant digits), so two values are "unchanged"
when they are equal at that precision. `diff_cosines` compares the parsed floats exactly, which is the same thing
for text that came from the same printer.

Catch2's console reporter frames a WARN as a `<file>:<line>:` line, a `warning:` line, then the message indented two
spaces and wrapped at 79 columns. The wrap can fall inside a token (`12/` newline `12`), so the parser joins the
indented lines of each `warning:` block and ignores whitespace before matching; model ids and prompt names never
contain whitespace, so nothing is lost.

usage (library): import cosine; runs = cosine.parse_min_cosine(Path("ctest.log").read_text())
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator, Mapping
from typing import NamedTuple

# The WARN text with every whitespace character removed (see the module docstring). Groups: prompt, model,
# positions, min cosine, agreeing positions, total. The character set of `model` covers ids like `llama32-q4_k_m`.
_LINE = re.compile(
    r"(\w+)\[([\w.+-]+)\]:mincosineover(\d+)positions=([-+0-9.eE]+|nan|inf|-nan|-inf),top-1agreement(\d+)/(\d+)"
)
# ctest -V prefixes every line with the test number ("31: "); `--output-log` does not.
_CTEST_PREFIX = re.compile(r"^\d+: ?")
_WARNING_LINE = re.compile(r"^warning:\s*$")


class Cosine(NamedTuple):
    """One e2e result: the lowest logit cosine over `positions` positions and the top-1 agreement count."""

    model: str
    prompt: str
    min_cos: float
    positions: int
    agree: int


class CosineChange(NamedTuple):
    """A key whose result differs between two runs; `old` or `new` is None when the key is absent from that run."""

    key: str
    old: float | None
    new: float | None


def _warning_messages(log: str) -> Iterator[str]:
    """Yield each Catch2 `warning:` message with its wrapped lines joined and all whitespace removed."""
    block: list[str] | None = None
    for raw in [*log.splitlines(), ""]:
        line = _CTEST_PREFIX.sub("", raw.rstrip())
        if block is not None and line[:1].isspace():
            block.append("".join(line.split()))
            continue
        if block:
            yield "".join(block)
        block = [] if _WARNING_LINE.match(line) else None


def parse_min_cosine(log: str) -> dict[str, Cosine]:
    """Return every `min cosine` WARN in `log` keyed `"<model>/<prompt>"`.

    Other warnings are ignored. The same key printed twice with different values raises ValueError (two runs were
    concatenated, or a test case ran twice with different results); an exact repeat is accepted. A line whose
    position count disagrees with its top-1 denominator also raises ValueError.
    """
    found: dict[str, Cosine] = {}
    for message in _warning_messages(log):
        m = _LINE.fullmatch(message)
        if m is None:
            continue
        prompt, model, positions, value, agree, total = m.groups()
        key = f"{model}/{prompt}"
        if positions != total:
            raise ValueError(f"{key}: {positions} positions but top-1 agreement out of {total}")
        entry = Cosine(model, prompt, float(value), int(positions), int(agree))
        previous = found.get(key)
        if previous is not None and not _same(previous.min_cos, entry.min_cos):
            raise ValueError(f"{key}: conflicting values {previous.min_cos!r} and {entry.min_cos!r}")
        found[key] = entry
    return found


def _same(a: float, b: float) -> bool:
    """Exact equality, except that two NaNs count as equal (an unchanged NaN is not a change)."""
    return a == b or (math.isnan(a) and math.isnan(b))


def diff_cosines(old: Mapping[str, Cosine], new: Mapping[str, Cosine]) -> list[CosineChange]:
    """List keys that were added, removed or whose `min_cos` differs, sorted by key. Unchanged keys are omitted."""
    changes: list[CosineChange] = []
    for key in sorted(old.keys() | new.keys()):
        before = old[key].min_cos if key in old else None
        after = new[key].min_cos if key in new else None
        if before is None or after is None or not _same(before, after):
            changes.append(CosineChange(key, before, after))
    return changes
