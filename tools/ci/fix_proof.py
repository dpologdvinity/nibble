"""Bug-fix proof: pick the Catch2 test cases a pull request added or changed, and judge how they ran on the base.

nibble requires that a bug fix comes with a test that fails without the fix. CI checks that in four steps:

1. `gate` says whether the pull request is a fix (`fix:` title or commit subject) that touches code
   (`src/`, `include/`, `cli/`). Other pull requests are not checked.
2. `select` reads `git diff -U0 base...head` (changes since the merge base), maps every added or modified line of
   `tests/*.cpp` to the TEST_CASE whose header or body contains it (a span ends at the body's closing brace), and
   writes `cases.json`.
3. The workflow builds the merge base with the PR's tests applied (ASan and UBSan), and `run` executes each selected
   case there in a process of its own, recording the JUnit XML and how the process ended.
4. `classify` judges (R9 of the plan): the fix is **proven** when at least one selected case fails on the base. Cases
   that already pass are only listed. A base that does not compile with the new tests also counts as proven (the
   tests use something the fix adds).

A case counts as failed when its JUnit says so, or when its process crashed (signal), timed out or exited non-zero
(a LeakSanitizer report comes after the JUnit and leaves the case passed in it; so does a non-halting sanitizer
finding), provided `--list-tests` (run with the same environment, and exiting 0) showed that the spec matches
exactly that one case. Without that proof a process outcome is not attributed to the case, because Catch2 also exits 1
without running anything when it refuses its command line, and a sanitizer can fail every invocation.

What can never be proven, by design (the "unresolved" and "not-runnable" paths):

* A TEST_CASE whose name is not one plain string literal (the `E2E_CASE` macro in tests/test_e2e_golden.cpp builds
  its names with the preprocessor) cannot be named to Catch2, so a change to it is listed as unresolved.
* A change that deletes a whole TEST_CASE selects nothing and is listed as `deleted: <name>`.
* A change outside every TEST_CASE body (a helper, the `E2E_MODEL` invocations, code between or after cases) selects
  nothing and is listed by line, even when a case uses the helper: the check cannot tell which cases a helper
  affects, and running the neighbouring case would prove nothing about it.
* A case that SKIPs on the base (the end-to-end cases without models), or whose run left no result, is not-runnable.
* Edits to tests/*.cpp files that neither nibble_tests nor nibble_alloc_tests builds (simd_helpers_test.cpp and so on)
  are listed as unresolved.

Catch2 facts the code relies on (checked against Catch2 v3.16.1, the version CMakeLists.txt pins, by running real
binaries with `-r junit`):

* A test spec treats `,` `[` `]` `"` `~` `*` and `\\` specially; each needs a backslash (`catch_spec`), and so does a
  leading `-`, which Catch2 would read as an option. A leading or trailing `*` stays a wildcard even when escaped, so
  such a name also matches other cases; results are therefore looked up by exact name, and for such a name counted
  only when the listing showed the spec to match that one case (a case named `parse*/legacy` would otherwise pass
  for a section of `parse*`). A name starting with
  `exclude:` turns the spec into an exclusion that runs everything else, so `test_cases` reports such a name as
  unresolved.
* Spec matching ignores case; the JUnit `<testcase name>` is the exact name. A section is its own `<testcase>` named
  `<case>/<section>`. A `<testcase>` is written only for a case or section that made an assertion, so a passing case
  without assertions has none. Only the first failing assertion of a section is written, as `<failure>` (or `<error>`
  for an exception or a caught signal such as SIGABRT), and a SKIP is `<skipped type="SKIP">`. A `[!mayfail]` case
  that fails gets both a `<skipped>` and a `<failure>`; it is reported as skipped, never as failed.
* Exit codes: 0 passed, 42 an assertion failed, 4 every case skipped, 2/3/5/6 nothing ran (no match, unmatched or bad
  spec, bad reporter), 1 an unspecified error (and what the sanitizers exit with).

usage:
    python tools/ci/fix_proof.py gate --title "$PR_TITLE" --base "$BASE_SHA" --head "$HEAD_SHA"
        prints `fix=true|false` and `code=true|false` (`>> "$GITHUB_OUTPUT"`).
    python tools/ci/fix_proof.py select --base "$BASE_SHA" --head "$HEAD_SHA" --out cases.json
        `--base` may be the base branch tip or the merge base: both diffs are `base...head`, which starts at the merge
        base. Run in a git checkout with both commits (or pass `--repo DIR`). Prints `selected=<n>`.
    python tools/ci/fix_proof.py run --cases cases.json --build-dir build --results-dir results [--timeout 600]
        for each selected case, in order, runs `<build-dir>/<binary> <spec> -r junit -o <results-dir>/<n>.xml` (n is
        the case's index in cases.json) with `UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1` and
        `ASAN_OPTIONS=detect_leaks=1` (options already set in the environment are kept), and a timeout per case
        (seconds, default 600). Next to the XML it writes `<n>.status.json`
        (`{"started", "exit_code", "signal", "timed_out", "unique"}`; `unique` is the `--list-tests` probe) and
        `<n>.log` (the tail of the output, evidence for a failure). Always exits 0 unless the arguments are bad.
    python tools/ci/fix_proof.py classify --cases cases.json --results-dir results --base-compiled true|false \\
        [--override] --summary "$GITHUB_STEP_SUMMARY"
        reads `<n>.xml` and `<n>.status.json` of every case from --results-dir, appends a Markdown summary to
        --summary, prints `outcome=<outcome>` and exits 1 for no-test (unless --override, which the workflow sets for
        the `no-test-needed` label) and does-not-reproduce, else 0. Other errors exit 2. Per case: failed JUnit is
        failed; passed JUnit with a failed process is failed; no JUnit element with exit 0 is passed (a case with no
        assertions); skipped is skipped; anything not attributable is missing (see `case_result`).
    python tools/ci/fix_proof.py specs --cases cases.json
        one line per selected case, `<index>\\t<binary>\\t<catch2 test spec>`, for running cases by hand.

cases.json is `{"cases": [{"file", "name", "start", "end", "binary", "spec"}...], "unresolved": ["..."]}`; only
`file`, `name`, `start`, `end`, `binary` (nibble_tests or nibble_alloc_tests) and `unresolved` are read back (the
spec is recomputed).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, NamedTuple

# Conventional Commit `fix` type (with optional scope and `!`), as the commit-lint and the PR title check accept it.
FIX_RE = r"^fix(\([a-z0-9_.,-]+\))?!?: "
# A fix that changes none of these directories changes no behaviour of the engine, so it needs no reproducing test.
CODE_PREFIXES = ("src/", "include/", "cli/")

TESTS_BINARY = "nibble_tests"
ALLOC_BINARY = "nibble_alloc_tests"
# `tests/*.cpp` directly under tests/ (the fuzz drivers in tests/fuzz are not Catch2 binaries).
_TEST_FILE = re.compile(r"^tests/[^/]+\.cpp$")
MACRO_BUILT = "<macro-built>"
MAX_XML_BYTES = 20_000_000
MAX_LOG_BYTES = 64_000  # the tail of a case's output kept as evidence
DEFAULT_TIMEOUT = 600.0  # seconds per case in `run`
LIST_TIMEOUT = 60.0  # seconds for one `--list-tests` probe


class Case(NamedTuple):
    """One Catch2 TEST_CASE: `name` is None when its first argument is not one plain string literal.

    `start` is the line of the TEST_CASE token and `end` the line of the closing brace of its body, both 1-based and
    inclusive, so helpers, macro invocations and comments between two cases belong to neither. When the body cannot
    be delimited (see `_body_end`), `end` is the line before the next TEST_CASE (or the last line of the file).
    `binary` is the test executable that registers the case: nibble_tests, or nibble_alloc_tests for
    tests/alloc_tests.cpp.
    """

    file: str
    name: str | None
    start: int
    end: int
    binary: str


class Selection(NamedTuple):
    """The cases a pull request touched (all with a resolved name) and what could not be turned into a case."""

    cases: list[Case]
    unresolved: list[str]


CaseResult = Literal["passed", "failed", "skipped", "missing"]
Outcome = Literal["proven", "proven-base-does-not-compile", "no-test", "does-not-reproduce", "not-runnable"]


# ---------------------------------------------------------------------------------------------------------------
# Pull request gate


def is_fix(title: str, subjects: Sequence[str]) -> bool:
    """True when the pull request title or any commit subject is a Conventional Commit `fix`."""
    return any(re.match(FIX_RE, text) for text in (title, *subjects))


def touches_code(paths: Sequence[str]) -> bool:
    """True when any changed path is under src/, include/ or cli/."""
    return any(p.startswith(CODE_PREFIXES) for p in paths)


# ---------------------------------------------------------------------------------------------------------------
# Finding TEST_CASE headers in C++ source


def _scan(source: str) -> tuple[str, str]:
    """Return (no_comments, skeleton): `source` with comments blanked, and with string/char contents blanked too.

    Both keep the length and every newline, so offsets and line numbers carry over. The skeleton keeps the quote
    characters, which is what lets the header search ignore a `TEST_CASE(` that sits in a comment or in a string.
    Understands raw strings, char literals and digit separators (`1'000`); it does not run the preprocessor.
    """
    n = len(source)
    plain = list(source)  # comments blanked
    skel = list(source)  # comments and literal contents blanked

    def blank(buf: list[str], a: int, b: int) -> None:
        for k in range(a, b):
            if buf[k] != "\n":
                buf[k] = " "

    i = 0
    while i < n:
        c = source[i]
        nxt = source[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":
            j = source.find("\n", i)
            j = n if j < 0 else j
            blank(plain, i, j)
            blank(skel, i, j)
            i = j
        elif c == "/" and nxt == "*":
            j = source.find("*/", i + 2)
            j = n if j < 0 else j + 2
            blank(plain, i, j)
            blank(skel, i, j)
            i = j
        elif c == '"':
            j = _string_end(source, i)
            blank(skel, i + 1, max(i + 1, j - 1))
            i = j
        elif c == "'":
            if _is_digit_separator(source, i):
                i += 1
                continue
            j = i + 1
            while j < n and source[j] != "'" and source[j] != "\n":
                j += 2 if source[j] == "\\" else 1
            j = min(n, j + 1)
            blank(skel, i + 1, max(i + 1, j - 1))
            i = j
        else:
            i += 1
    return "".join(plain), "".join(skel)


def _is_digit_separator(source: str, i: int) -> bool:
    """True when the `'` at `i` sits inside a numeric literal such as `20'000'000`."""
    j = i
    while j > 0 and (source[j - 1].isalnum() or source[j - 1] in "_.'"):
        j -= 1
    return j < i and source[j].isdigit()


def _string_end(source: str, i: int) -> int:
    """Offset just past the string literal whose opening quote is at `i` (the end of the source if unterminated)."""
    n = len(source)
    k = i
    while k > 0 and (source[k - 1].isalnum() or source[k - 1] == "_"):
        k -= 1
    if source[k:i] in ("R", "u8R", "uR", "UR", "LR"):  # raw string: R"delim( ... )delim"
        paren = source.find("(", i)
        if paren >= 0:
            end = source.find(")" + source[i + 1 : paren] + '"', paren)
            return n if end < 0 else end + (paren - i - 1) + 2
        return n
    j = i + 1
    while j < n and source[j] != '"' and source[j] != "\n":
        j += 2 if source[j] == "\\" else 1
    return min(n, j + 1)


# Every Catch2 (or project) macro whose name contains TEST_CASE or SCENARIO. Only the exact name TEST_CASE has its
# name read; the others (TEST_CASE_METHOD, TEMPLATE_TEST_CASE, SCENARIO, ...) are cases with an unresolved name.
_HEADER = re.compile(r"(?<![A-Za-z0-9_])([A-Za-z0-9_]*(?:TEST_CASE|SCENARIO)[A-Za-z0-9_]*)\s*\(")
_ESCAPES = {'"': '"', "\\": "\\", "'": "'"}


def _plain_literal(text: str, pos: int) -> str | None:
    """The value of the one plain string literal that is the first macro argument starting at `pos`, else None.

    `text` has its comments blanked. None for anything else: a prefixed or raw literal, two adjacent literals
    (`"a" "b"`), a literal followed by a macro name, any escape other than `\\"`, `\\\\` and `\\'`.
    """
    n = len(text)
    while pos < n and text[pos].isspace():
        pos += 1
    if pos >= n or text[pos] != '"':
        return None
    pos += 1
    out: list[str] = []
    while pos < n and text[pos] != '"':
        ch = text[pos]
        if ch == "\\":
            if pos + 1 >= n or text[pos + 1] not in _ESCAPES:
                return None
            out.append(_ESCAPES[text[pos + 1]])
            pos += 2
        elif ch == "\n":
            return None
        else:
            out.append(ch)
            pos += 1
    if pos >= n:
        return None
    pos += 1
    while pos < n and text[pos].isspace():
        pos += 1
    if pos >= n or text[pos] not in ",)":
        return None
    return "".join(out)


def _usable_name(name: str) -> bool:
    """True when `name` can be given to Catch2 as a test spec and mean exactly that case.

    Not usable: an empty name (Catch2 invents "Anonymous test case N"), leading or trailing whitespace (a spec
    skips leading spaces), control characters, and the `exclude:` prefix (it turns the spec into an exclusion).
    """
    return (
        bool(name) and name == name.strip() and all(ch.isprintable() for ch in name) and not name.startswith("exclude:")
    )


def _matching(skeleton: str, pos: int, open_ch: str, close_ch: str) -> int | None:
    """Offset of the `close_ch` that closes the `open_ch` at `pos` in `skeleton`, or None if it is never closed."""
    depth = 0
    for k in range(pos, len(skeleton)):
        ch = skeleton[k]
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return k
    return None


def _body_end(skeleton: str, open_paren: int) -> int | None:
    """Offset of the `}` that closes the body of the header whose argument list opens at `open_paren`, else None.

    The skeleton has comments and the contents of string and char literals blanked, so the braces counted are the
    code's own. None (the caller falls back to a longer span) when the argument list or the body is never closed, or
    when what follows the argument list is not a `{` (a macro that expands to a header, say). It does not run the
    preprocessor: braces in both arms of an `#if` can still unbalance a body, which the caller catches by checking
    that the body ends before the next header.
    """
    close_paren = _matching(skeleton, open_paren, "(", ")")
    if close_paren is None:
        return None
    k = close_paren + 1
    while k < len(skeleton) and skeleton[k].isspace():
        k += 1
    if k >= len(skeleton) or skeleton[k] != "{":
        return None
    return _matching(skeleton, k, "{", "}")


class _Header(NamedTuple):
    """A TEST_CASE-like header: its line, its name (None if unresolved) and the line its body ends on (None if the
    body could not be delimited)."""

    line: int
    name: str | None
    body_end: int | None


def _find_headers(source: str) -> list[_Header]:
    """Every TEST_CASE-like header in `source`, in order."""
    plain, skeleton = _scan(source)
    found: list[_Header] = []
    for m in _HEADER.finditer(skeleton):
        line = skeleton.count("\n", 0, m.start(1)) + 1
        name = _plain_literal(plain, m.end()) if m.group(1) == "TEST_CASE" else None
        end = _body_end(skeleton, m.end() - 1)
        found.append(
            _Header(
                line,
                name if name is not None and _usable_name(name) else None,
                None if end is None else skeleton.count("\n", 0, end) + 1,
            )
        )
    return found


def _headers(source: str) -> list[tuple[int, str | None]]:
    """(line, name) of every TEST_CASE-like header in `source`, in order; name None when it cannot be resolved."""
    return [(h.line, h.name) for h in _find_headers(source)]


def test_cases(path: str, source: str) -> list[Case]:
    """Every TEST_CASE of `source` (the text of `path`) with its line span; see `Case` for the fields."""
    # `select` never passes a file that no binary builds; any other path (a fixture in a test) gets nibble_tests.
    binary = _binary_for(path) or TESTS_BINARY
    last = source.count("\n") + (0 if source.endswith("\n") or not source else 1)
    headers = _find_headers(source)
    cases: list[Case] = []
    for idx, header in enumerate(headers):
        limit = headers[idx + 1].line if idx + 1 < len(headers) else last
        # A body that ends past the next header (or not at all) was not delimited correctly: keep every line up to
        # the next header rather than guess, so an edit of this case is never lost.
        if header.body_end is not None and header.body_end <= limit:
            end = header.body_end
        else:
            end = limit - 1 if idx + 1 < len(headers) else last
        cases.append(Case(path, header.name, header.line, max(end, header.line), binary))
    return cases


# ---------------------------------------------------------------------------------------------------------------
# Reading a `git diff -U0` listing


class _Hunk(NamedTuple):
    """One hunk of a -U0 diff: head-side range (`new_start`, `new_count`) and the removed lines' text."""

    path: str
    new_start: int
    new_count: int
    old_count: int
    removed: list[str]


_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _diff_path(line: str) -> str | None:
    """The path in a `--- a/x` or `+++ b/x` line; None for /dev/null and for a path git quoted (not a test file)."""
    text = line[4:].split("\t", 1)[0]
    if text == "/dev/null" or text.startswith('"'):
        return None
    return text[2:] if text[:2] in ("a/", "b/") else text


def _hunks(diff: str) -> list[_Hunk]:
    """Parse `git diff -U0 --no-ext-diff --no-color` output into hunks of tests/*.cpp files only.

    The hunk header counts say how many content lines follow, so a content line that looks like a header (a removed
    `-- ...`, an added `++ b/tests/x.cpp`) is never read as one. The accounting runs for every file in the diff, also
    those whose hunks are not returned (the `tests/*.cpp` pathspec matches tests/fuzz/*.cpp too), so text in one
    file cannot redirect a later hunk. Lines are split at `\n` only, as git writes them.
    """
    hunks: list[_Hunk] = []
    old_path: str | None = None
    path: str | None = None
    old_left = new_left = 0
    current: _Hunk | None = None
    for line in diff.split("\n"):
        if old_left > 0 or new_left > 0:
            if line.startswith("-"):
                old_left -= 1
                if current is not None:
                    current.removed.append(line[1:])
            elif line.startswith("+"):
                new_left -= 1
            # `\ No newline at end of file` and anything else carry no count
            continue
        if line.startswith("diff "):
            old_path = path = None
        elif line.startswith("--- "):
            old_path = _diff_path(line)
        elif line.startswith("+++ "):
            new_path = _diff_path(line)
            path = new_path if new_path is not None else old_path
        elif m := _HUNK.match(line):
            old_count = 1 if m.group(2) is None else int(m.group(2))
            new_count = 1 if m.group(4) is None else int(m.group(4))
            old_left, new_left = old_count, new_count
            current = None
            if path is not None and _TEST_FILE.match(path):
                current = _Hunk(path, int(m.group(3)), new_count, old_count, [])
                hunks.append(current)
        else:
            current = None
    return hunks


def _deleted_headers(hunk: _Hunk) -> list[str | None]:
    """Names (None when not one plain literal) of the TEST_CASE headers among the lines a deletion hunk removes."""
    # The sentinel closes a header whose argument list was cut off by the end of the removed text.
    return [name for _, name in _headers("\n".join(hunk.removed) + "\n)")]


class _Touched(NamedTuple):
    """What a diff did to one file, head-side: lines it added or changed, and the gaps pure deletions left.

    `gaps` maps `c` of a pure-deletion hunk `@@ -a,b +c,0 @@` (the removed text sat after head line `c`; 0 is the top
    of the file) to the removed lines.
    """

    lines: set[int]
    gaps: dict[int, list[str]]


def _touched(diff: str) -> dict[str, _Touched]:
    """Per tests/*.cpp file, the lines and deletion gaps of `diff` (see `changed_lines` for which hunks count)."""
    out: dict[str, _Touched] = {}
    for h in _hunks(diff):
        if h.new_count > 0:
            out.setdefault(h.path, _Touched(set(), {})).lines.update(range(h.new_start, h.new_start + h.new_count))
        elif h.old_count > 0 and not _deleted_headers(h):
            out.setdefault(h.path, _Touched(set(), {})).gaps.setdefault(h.new_start, []).extend(h.removed)
    return out


def changed_lines(diff: str) -> dict[str, set[int]]:
    """Per tests/*.cpp file, the head-side line numbers a diff added or changed.

    A pure-deletion hunk `@@ -a,b +c,0 @@` counts as line `c` (the line before the gap; 0 at the top of the file).
    Exception (R19): a deletion that removes a TEST_CASE header contributes nothing, because `c` would be the last
    line of the previous case, which did not change; see `deleted_case_names`. Files with no counted line are left
    out. (`select` tells the gap of a deletion from a changed line: a gap selects a case only from inside its body.)
    """
    return {path: t.lines | set(t.gaps) for path, t in _touched(diff).items()}


def deleted_case_names(diff: str) -> dict[str, list[str]]:
    """Per tests/*.cpp file, the names on TEST_CASE headers that pure-deletion hunks removed.

    A header whose name is not one plain string literal is listed as "<macro-built>".
    """
    out: dict[str, list[str]] = {}
    for h in _hunks(diff):
        if h.new_count == 0 and h.old_count > 0:
            for name in _deleted_headers(h):
                out.setdefault(h.path, []).append(MACRO_BUILT if name is None else name)
    return out


def _binary_for(path: str) -> str | None:
    """The executable that runs the TEST_CASEs of `path`; None for tests/*.cpp files that are not built into one."""
    if path == "tests/alloc_tests.cpp":
        return ALLOC_BINARY
    if re.match(r"^tests/test_[^/]*\.cpp$", path):  # CMakeLists.txt globs tests/test_*.cpp into nibble_tests
        return TESTS_BINARY
    return None


def _line_list(numbers: set[int]) -> str:
    """ "line 4" or "lines 1, 4-6" for a non-empty set of line numbers."""
    ordered = sorted(numbers)
    runs: list[list[int]] = []
    for n in ordered:
        if runs and n == runs[-1][1] + 1:
            runs[-1][1] = n
        else:
            runs.append([n, n])
    text = ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)
    return f"line {text}" if len(ordered) == 1 else f"lines {text}"


def select(diff: str, head_sources: Mapping[str, str]) -> Selection:
    """The cases `diff` adds or changes, found in `head_sources` (path to the file's text at the head).

    A changed line selects the case whose span (header to closing brace) contains it, and a deletion selects the case
    whose body it removed lines from (its gap lies after a line of the span other than the closing brace). Changes
    outside every span (helpers, macro invocations such as E2E_MODEL, code between or after cases) select nothing and
    are listed in `unresolved`, except blank lines. A selected case with an unresolved name goes to `unresolved`, as
    does a changed file no test binary builds, and each deleted TEST_CASE header adds `deleted: <name>`.
    """
    cases: list[Case] = []
    unresolved: list[str] = []
    for path, touched in _touched(diff).items():
        if _binary_for(path) is None:
            unresolved.append(f"{path}: not built into nibble_tests or nibble_alloc_tests, so not run")
            continue
        source = head_sources.get(path)
        if source is None:
            continue
        spans = test_cases(path, source)
        for case in spans:
            hit = any(case.start <= n <= case.end for n in touched.lines) or any(
                case.start <= c < case.end for c in touched.gaps
            )
            if not hit:
                continue
            if case.name is None:
                unresolved.append(f"{path}:{case.start}: test name is not one plain string literal")
            else:
                cases.append(case)
        text = source.split("\n")
        outside = {
            n
            for n in touched.lines
            if not any(c.start <= n <= c.end for c in spans) and 0 < n <= len(text) and text[n - 1].strip()
        }
        outside |= {
            max(c, 1)
            for c, removed in touched.gaps.items()
            if not any(case.start <= c < case.end for case in spans) and any(r.strip() for r in removed)
        }
        if outside:
            unresolved.append(
                f"{path}: {_line_list(outside)} changed outside every TEST_CASE body (helpers, macro invocations): "
                "no case selected for that"
            )
    for names in deleted_case_names(diff).values():
        unresolved.extend(f"deleted: {name}" for name in names)
    return Selection(cases, unresolved)


# ---------------------------------------------------------------------------------------------------------------
# Catch2 side: test spec and JUnit results


_SPEC_SPECIAL = frozenset(',[]*"~\\')


def catch_spec(name: str) -> str:
    """Escape `name` for a Catch2 test spec: a backslash before each of `, [ ] * " ~` and backslash itself.

    Escaping `~` and `[` also covers a leading `~` (exclusion) and a leading `[` (tag). A leading `-` is escaped too:
    unescaped, Catch2 reads the name as a command-line option ("Unrecognised token") and runs nothing. Catch2 still
    treats a leading or trailing `*` as a wildcard after unescaping, so such a spec may match more cases than `name`.
    """
    spec = "".join("\\" + ch if ch in _SPEC_SPECIAL else ch for ch in name)
    return "\\" + spec if spec.startswith("-") else spec


def _element_state(case: ET.Element) -> CaseResult:
    """One `<testcase>`: skipped when it has a `<skipped>` child, else failed on `<failure>`/`<error>`, else passed."""
    if case.find("skipped") is not None:
        return "skipped"  # a SKIP, or `[!mayfail]` (which also carries the failure)
    if case.find("failure") is not None or case.find("error") is not None:
        return "failed"
    return "passed"


_RANK: dict[CaseResult, int] = {"missing": 0, "skipped": 1, "passed": 2, "failed": 3}


def parse_junit(xml: str, name: str) -> CaseResult:
    """How the case `name` ended in a Catch2 `-r junit` document.

    The case's own `<testcase name=NAME>` and its sections (`NAME/section`) are combined: failed if any failed, else
    passed if any passed, else skipped. "missing" when there is no such element, and also when `xml` is not
    well-formed (a run that died before writing it), so a broken file can never prove anything.
    """
    if "<!DOCTYPE" in xml or "<!ENTITY" in xml:  # Catch2 writes neither; refuse entity tricks in stdlib ElementTree
        return "missing"
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return "missing"
    best: CaseResult = "missing"
    for tc in root.iter("testcase"):
        tc_name = tc.get("name", "")
        if tc_name == name or tc_name.startswith(name + "/"):
            state = _element_state(tc)
            if _RANK[state] > _RANK[best]:
                best = state
    return best


class RunStatus(NamedTuple):
    """What `run` recorded about the process that ran one case (`<index>.status.json`).

    `started` is False when the binary could not be started. `exit_code` is None when a signal or the timeout ended
    the process, `signal` the number of that signal. `unique` says that `<binary> <spec> --list-tests` showed the
    spec to match exactly this one case, so that whatever the process did can be attributed to the case.
    """

    started: bool
    exit_code: int | None
    signal: int | None
    timed_out: bool
    unique: bool


# Catch2's own exit codes (catch_session.hpp): 0 passed, 4 every case skipped (also true when one section of a case
# skipped), 42 an assertion failed; 2, 3, 5 and 6 mean it ran nothing (no match, unmatched spec, bad spec, bad
# reporter). 1 is both Catch2's "unspecified error" (a command line it refused) and what ASan, UBSan and LeakSanitizer
# exit with, hence the `unique` requirement before a non-zero exit counts against a case.
_BENIGN_EXITS = frozenset({0, 4})
_REFUSED_EXITS = frozenset({2, 3, 5, 6})


def case_result(name: str, xml: str | None, run: RunStatus | None) -> tuple[CaseResult, str]:
    """Combine the JUnit text (None when no file) and the process outcome of one case into (result, note).

    * JUnit says failed, or skipped: that, except for a name with a leading or trailing `*`. Catch2 keeps such a `*`
      a wildcard, so the spec may also run another case whose name extends this one (`parse*` runs `parse*/legacy`,
      whose <testcase> reads like a section of `parse*`); its JUnit (whatever it says) then counts only when the
      listing proved the spec unique, and the case is missing otherwise.
    * JUnit says passed, but the process failed (a signal, the timeout, any exit but 0 and 4): failed. This is how
      LeakSanitizer (reports at exit, after the XML) and non-halting sanitizer findings show up. Only when the
      listing proved the spec unique.
    * No JUnit element (Catch2 writes none for a case without assertions): exit 0 means the case passed, exit 4
      skipped, and a process that died (signal, timeout, other exit) failed, again only for a unique spec; if Catch2
      itself refused the run (exit 2, 3, 5, 6) or the binary did not start, the case is missing.
    * No status file: the JUnit alone decides (a missing or unreadable file is missing).
    The note names the process outcome when it decided the result.
    """
    state = parse_junit(xml, name) if xml is not None else "missing"
    wildcard = name.startswith("*") or name.endswith("*")
    if wildcard and state != "missing" and (run is None or not run.unique):
        return "missing", "the wildcard test spec did not match exactly one case, so its JUnit is not attributed"
    if state in ("failed", "skipped") or run is None:
        return state, ""
    if not run.started:
        return "missing", "the test binary did not start"
    died = run.timed_out or run.signal is not None or run.exit_code not in _BENIGN_EXITS | _REFUSED_EXITS | {None}
    how = (
        "timed out"
        if run.timed_out
        else f"killed by signal {run.signal}"
        if run.signal is not None
        else f"exited with status {run.exit_code}"
    )
    if state == "passed":
        if died and run.unique:
            return "failed", f"{how} although the JUnit reported a pass (sanitizer or leak report)"
        return "passed", ""
    # no JUnit element for the case
    if run.exit_code in _REFUSED_EXITS:
        return "missing", f"Catch2 ran nothing ({how})"
    if not run.unique:
        return "missing", "the test spec did not match exactly one case, so the process outcome is not attributed"
    if died:
        return "failed", f"{how}, no JUnit result"
    if run.exit_code == 4:
        return "skipped", ""
    return "passed", "no JUnit element (the case made no assertion) and exit status 0"


def classify(selection: Selection, base_compiled: bool, results: Mapping[str, CaseResult]) -> Outcome:
    """Judge the run of the selected cases on the base (rules in order, R9 of the plan)."""
    if not selection.cases:
        return "no-test"
    if not base_compiled:
        return "proven-base-does-not-compile"
    states = [results.get(_name(c), "missing") for c in selection.cases]
    if "failed" in states:
        return "proven"
    if all(s in ("skipped", "missing") for s in states):
        return "not-runnable"
    return "does-not-reproduce"


def exit_code(outcome: Outcome, override_label: bool) -> int:
    """1 for no-test (unless the `no-test-needed` label is set) and for does-not-reproduce, else 0."""
    if outcome == "no-test":
        return 0 if override_label else 1
    return 1 if outcome == "does-not-reproduce" else 0


def _name(case: Case) -> str:
    """The name of a selected case (select only keeps resolved ones)."""
    if case.name is None:
        raise ValueError(f"{case.file}:{case.start}: case without a name")
    return case.name


# Characters that start emphasis, code, links, HTML, tables, mentions or entities in GitHub Markdown.
_MARKDOWN_ACTIVE = re.compile(r"([\\`*_{}\[\]()<>#+!|~&@])")


def _md(text: str) -> str:
    """`text` safe inside a Markdown list item: non-printable characters become `?`, Markdown syntax is backslashed.

    Test names come from the pull request's own source, so they are untrusted text in a job summary.
    """
    return _MARKDOWN_ACTIVE.sub(r"\\\1", "".join(ch if ch.isprintable() else "?" for ch in text))


_HEADLINES: dict[Outcome, str] = {
    "proven": "Proven: at least one test of this fix fails without the fix.",
    "proven-base-does-not-compile": (
        "Proven: the base does not compile with this pull request's tests, so they use something the fix adds."
    ),
    "no-test": (
        "No reproducing test: no TEST_CASE added or changed by this pull request could be selected. Add a test that "
        "fails without the fix, or put the `no-test-needed` label on the pull request and give the reason in its "
        "description."
    ),
    "does-not-reproduce": (
        "Does not reproduce: every selected test passes on the base, so none of them shows the bug. (A fix for a "
        "data race that only ThreadSanitizer sees ends up here; the base is built with ASan and UBSan.)"
    ),
    "not-runnable": (
        "Not runnable: every selected test skipped or left no result on the base (end-to-end cases skip without "
        "models), so nothing could be proven."
    ),
}


def summary(
    outcome: Outcome,
    selection: Selection,
    results: Mapping[str, CaseResult],
    notes: Mapping[str, str] | None = None,
) -> str:
    """The Markdown job summary: the verdict, one line per selected case (with `notes[name]` when given) and every
    unresolved item."""
    lines = [f"### Bug-fix proof: {outcome}", "", _HEADLINES[outcome], ""]
    if selection.cases:
        lines.append("Selected test cases and how they ran on the base:")
        lines.append("")
        for case in selection.cases:
            state = results.get(_name(case), "missing")
            note = " (already passing)" if state == "passed" else ""
            why = f" [{_md(notes[_name(case)])}]" if notes and notes.get(_name(case)) else ""
            lines.append(f"- {state}{note}: {_md(_name(case))} ({_md(case.file)}:{case.start}){why}")
        lines.append("")
    if selection.unresolved:
        lines.append("Not checked:")
        lines.append("")
        lines += [f"- {_md(item)}" for item in selection.unresolved]
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------------
# Command line


_REF = re.compile(r"[0-9A-Za-z][0-9A-Za-z._/~^-]*")


def _git(repo: str, *args: str) -> str:
    """Run `git <args>` in `repo` (argv list, no shell) and return stdout; a failure raises ValueError with stderr."""
    # quotePath=false: git prints non-ASCII paths as they are instead of as quoted octal escapes. Bytes that are not
    # UTF-8 (a stray Latin-1 comment in a test file) become U+FFFD instead of failing the whole check.
    proc = subprocess.run(
        ["git", "-C", repo, "-c", "core.quotePath=false", *args],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        raise ValueError(f"git {args[0]} failed: {proc.stderr.strip()}")
    return proc.stdout


def _check_ref(ref: str) -> str:
    """Refuse a revision that could be read as a git option or that contains anything but ref characters."""
    if not _REF.fullmatch(ref):
        raise ValueError(f"not a plain git revision: {ref!r}")
    return ref


def _case_to_json(case: Case) -> dict[str, object]:
    return {
        "file": case.file,
        "name": case.name,
        "start": case.start,
        "end": case.end,
        "binary": case.binary,
        "spec": catch_spec(_name(case)),
    }


def _read_selection(path: str) -> Selection:
    """Load cases.json, strictly: wrong types raise ValueError."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"cases", "unresolved"}:
        raise ValueError(f"{path}: expected an object with exactly the keys cases and unresolved")
    if not isinstance(raw["cases"], list):  # an object or a string would iterate as no case at all
        raise ValueError(f"{path}: cases must be a list")
    cases: list[Case] = []
    for item in raw["cases"]:
        if not isinstance(item, dict):
            raise ValueError(f"{path}: a case must be an object")
        file, name, start, end, binary = (item.get(k) for k in ("file", "name", "start", "end", "binary"))
        ints_ok = all(isinstance(v, int) and not isinstance(v, bool) for v in (start, end))
        if not (isinstance(file, str) and isinstance(name, str) and ints_ok and binary in (TESTS_BINARY, ALLOC_BINARY)):
            raise ValueError(f"{path}: bad case {item!r}")
        assert isinstance(start, int) and isinstance(end, int)
        cases.append(Case(file, name, start, end, binary))
    unresolved = raw["unresolved"]
    if not isinstance(unresolved, list) or not all(isinstance(u, str) for u in unresolved):
        raise ValueError(f"{path}: unresolved must be a list of strings")
    return Selection(cases, unresolved)


def _read_status(path: Path) -> RunStatus | None:
    """`<index>.status.json` as written by `run`; None when absent or not exactly that shape."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict) or set(raw) != {"started", "exit_code", "signal", "timed_out", "unique"}:
        return None
    started, code, sig, timed_out, unique = (raw[k] for k in ("started", "exit_code", "signal", "timed_out", "unique"))

    def small_int_or_none(v: object) -> bool:
        return v is None or (isinstance(v, int) and not isinstance(v, bool) and -1000 < v < 1000)

    if not (
        isinstance(started, bool)
        and isinstance(timed_out, bool)
        and isinstance(unique, bool)
        and small_int_or_none(code)
        and small_int_or_none(sig)
    ):
        return None
    return RunStatus(started, code, sig, timed_out, unique)


def read_results(results_dir: str, selection: Selection) -> tuple[dict[str, CaseResult], dict[str, str]]:
    """(result, note) per selected case name from the files `run` wrote: `<index>.xml` and `<index>.status.json`.

    `<index>` is the position of the case in `selection.cases`. An absent, oversized or unreadable XML counts as no
    JUnit; see `case_result` for how it combines with the process status.
    """
    results: dict[str, CaseResult] = {}
    notes: dict[str, str] = {}
    for index, case in enumerate(selection.cases):
        name = _name(case)
        xml_path = Path(results_dir, f"{index}.xml")
        xml: str | None = None
        if xml_path.is_file() and xml_path.stat().st_size <= MAX_XML_BYTES:
            xml = xml_path.read_text(encoding="utf-8", errors="replace")
        state, note = case_result(name, xml, _read_status(Path(results_dir, f"{index}.status.json")))
        if _RANK[state] >= _RANK[results.get(name, "missing")]:
            results[name] = state
            notes[name] = note
    return results, notes


def _merge_options(existing: str | None, wanted: str) -> str:
    """`existing` (a sanitizer options string `a=1:b=2`) plus those `wanted` options it does not already set."""
    parts = [p for p in (existing or "").split(":") if p]
    present = {p.split("=", 1)[0] for p in parts}
    return ":".join(parts + [w for w in wanted.split(":") if w.split("=", 1)[0] not in present])


def _list_probe(binary: Path, spec: str, name: str, env: Mapping[str, str]) -> bool:
    """True when `<binary> <spec> --list-tests` exits 0 and lists exactly one case, `name`.

    It runs with `env`, the environment of the run itself: a failure that happens without running any case
    (LeakSanitizer unable to run, a leak in static initialisation) then shows up here too, and the exit status keeps it
    from being attributed to the case.
    """
    try:
        proc = subprocess.run(
            [str(binary), spec, "--list-tests", "-r", "xml"],
            env=env,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=LIST_TIMEOUT,
            check=False,
        )
        listed = [el.findtext("Name", "").strip() for el in ET.fromstring(proc.stdout).iter("TestCase")]
    except (OSError, subprocess.TimeoutExpired, ET.ParseError):
        return False
    return proc.returncode == 0 and listed == [name]


def run_case(build_dir: Path, results_dir: Path, index: int, case: Case, timeout: float) -> RunStatus:
    """Run one case in its own process and leave `<index>.xml`, `<index>.log` and `<index>.status.json` behind."""
    binary = build_dir / case.binary
    spec = catch_spec(_name(case))
    xml_path = results_dir / f"{index}.xml"
    xml_path.unlink(missing_ok=True)
    env = dict(os.environ)
    env["UBSAN_OPTIONS"] = _merge_options(env.get("UBSAN_OPTIONS"), "halt_on_error=1:print_stacktrace=1")
    env["ASAN_OPTIONS"] = _merge_options(env.get("ASAN_OPTIONS"), "detect_leaks=1")
    log = b""
    if not binary.is_file():
        status = RunStatus(False, None, None, False, False)
    else:
        unique = _list_probe(binary, spec, _name(case), env)
        try:
            proc = subprocess.run(
                [str(binary), spec, "-r", "junit", "-o", str(xml_path)],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
            log = proc.stdout or b""
            code = proc.returncode
            status = RunStatus(True, None if code < 0 else code, -code if code < 0 else None, False, unique)
        except subprocess.TimeoutExpired as exc:
            log = exc.stdout if isinstance(exc.stdout, bytes) else b""
            status = RunStatus(True, None, None, True, unique)
        except OSError:
            status = RunStatus(False, None, None, False, False)
    (results_dir / f"{index}.log").write_bytes(log[-MAX_LOG_BYTES:])
    payload = {
        "started": status.started,
        "exit_code": status.exit_code,
        "signal": status.signal,
        "timed_out": status.timed_out,
        "unique": status.unique,
    }
    (results_dir / f"{index}.status.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return status


def _cmd_gate(args: argparse.Namespace) -> int:
    base, head = _check_ref(args.base), _check_ref(args.head)
    subjects = _git(args.repo, "log", "--format=%s", f"{base}..{head}").split("\n")
    names = [p for p in _git(args.repo, "diff", "--name-only", "-z", f"{base}...{head}").split("\0") if p]
    print(f"fix={str(is_fix(args.title, subjects)).lower()}")
    print(f"code={str(touches_code(names)).lower()}")
    return 0


def _cmd_select(args: argparse.Namespace) -> int:
    base, head = _check_ref(args.base), _check_ref(args.head)
    diff = _git(args.repo, "diff", "-U0", "--no-ext-diff", "--no-color", "-M", f"{base}...{head}", "--", "tests/*.cpp")
    sources: dict[str, str] = {}
    for path in changed_lines(diff):
        try:
            sources[path] = _git(args.repo, "show", f"{head}:{path}")
        except ValueError:
            continue  # the file does not exist at the head (deleted): nothing to select in it
    selection = select(diff, sources)
    payload = {"cases": [_case_to_json(c) for c in selection.cases], "unresolved": selection.unresolved}
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"selected={len(selection.cases)}")
    return 0


def _cmd_specs(args: argparse.Namespace) -> int:
    for i, case in enumerate(_read_selection(args.cases).cases):
        print(f"{i}\t{case.binary}\t{catch_spec(_name(case))}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    selection = _read_selection(args.cases)
    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    build_dir = Path(args.build_dir).resolve()
    for index, case in enumerate(selection.cases):
        status = run_case(build_dir, results_dir, index, case, args.timeout)
        print(
            f"{index}: {_name(case)}: started={status.started} exit={status.exit_code} signal={status.signal}",
            file=sys.stderr,
        )
    return 0


def _cmd_classify(args: argparse.Namespace) -> int:
    selection = _read_selection(args.cases)
    results, notes = read_results(args.results_dir, selection)
    outcome = classify(selection, args.base_compiled == "true", results)
    with open(args.summary, "a", encoding="utf-8") as handle:
        handle.write(summary(outcome, selection, results, notes) + "\n")
    print(f"outcome={outcome}")
    return exit_code(outcome, args.override)


def main(argv: Sequence[str]) -> int:
    """Command line entry point; see the module docstring. Returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0] if __doc__ else None)
    sub = parser.add_subparsers(dest="command", required=True)

    gate = sub.add_parser("gate", help="is this a fix pull request that touches code")
    gate.add_argument("--title", required=True)
    gate.add_argument("--base", required=True)
    gate.add_argument("--head", required=True)
    gate.add_argument("--repo", default=".")
    gate.set_defaults(func=_cmd_gate)

    sel = sub.add_parser("select", help="write cases.json from the diff between two revisions")
    sel.add_argument("--base", required=True)
    sel.add_argument("--head", required=True)
    sel.add_argument("--out", required=True)
    sel.add_argument("--repo", default=".")
    sel.set_defaults(func=_cmd_select)

    specs = sub.add_parser("specs", help="print `index<TAB>binary<TAB>spec` per selected case")
    specs.add_argument("--cases", required=True)
    specs.set_defaults(func=_cmd_specs)

    run = sub.add_parser("run", help="run every selected case in its own process and record how it ended")
    run.add_argument("--cases", required=True)
    run.add_argument("--build-dir", required=True, help="directory holding nibble_tests and nibble_alloc_tests")
    run.add_argument("--results-dir", required=True)
    run.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="seconds per case (default 600)")
    run.set_defaults(func=_cmd_run)

    cls = sub.add_parser("classify", help="judge the JUnit results and exit non-zero when the fix is not proven")
    cls.add_argument("--cases", required=True)
    cls.add_argument("--results-dir", required=True)
    cls.add_argument("--base-compiled", required=True, choices=("true", "false"))
    cls.add_argument("--override", action="store_true", help="the no-test-needed label is set")
    cls.add_argument("--summary", required=True, help="file the Markdown summary is appended to")
    cls.set_defaults(func=_cmd_classify)

    args = parser.parse_args(list(argv))
    try:
        code: int = args.func(args)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"fix_proof: {exc}", file=sys.stderr)
        return 2
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
