"""Unit tests for tools/ci/fix_proof.py (no network; the command-line tests use a throwaway git repository).

Run with: python -m unittest discover -s tools/ci -p 'test_*.py'
They pin what may and may not be called a proof (Review Focus 3 of the CI plan): a macro-built name is unresolved,
a case that SKIPs on the base is not-runnable, and names with `,` `[` `]` `*` `"` `~` reach Catch2 escaped. The JUnit
fixtures are real output of `-r junit` from Catch2 v3.16.1 (the version CMakeLists.txt pins), built from a small test
program with a passing, a failing, a SKIPping, a throwing and an aborting case, sections, and `[!mayfail]`.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from unittest import mock

import fix_proof
from fix_proof import Case, CaseResult, Selection

REPO_ROOT = Path(__file__).resolve().parents[2]

# A test file shaped like the real ones: helpers, a name with brackets, commas and a star, a header that spans
# lines, an E2E_CASE-style macro (tests/test_e2e_golden.cpp) with its invocations, and a plain last case.
SOURCE = """\
#include <catch2/catch_test_macros.hpp>

namespace {
int helper() { return 20'000; }  // a digit separator is not a char literal
}  // namespace

// TEST_CASE("commented out", "[x]") { }
TEST_CASE("alpha, with [tags] and * stars", "[a][b]") {
  SECTION("one") {
    REQUIRE(helper() == 20'000);
  }
  const char* s = "TEST_CASE(\\"in a string\\")";
}

TEST_CASE(
    "beta multi-line header",
    "[c]") {
  CHECK(true);
}

#define E2E_CASE(MODEL, PROMPT) \\
  TEST_CASE("e2e golden: " PROMPT " [" MODEL "]", "[e2e]") { run_prompt(MODEL, PROMPT); }
E2E_CASE("f16", "capital")
E2E_CASE("q4_0", "chat")

TEST_CASE("gamma") {
  CHECK(true);
}
"""


# Shaped like the real files around a case body: braces inside strings, a char literal, comments and a raw string
# (none of which may end the body), a lambda and nested SECTION/if blocks, a helper and a struct between and after the
# cases, and a one-line case.
BODIES = """\
// Tests for the thing.
#include <catch2/catch_test_macros.hpp>
#include <string>

namespace {
struct Fixture {
  int value = 1;
};
}  // namespace

TEST_CASE("braces in literals and comments", "[x]") {
  const std::string open = "{";  // a } in a comment
  const char close = '}';
  /* { */
  const char* raw = R"json({"a": {"b": 1}})json";
  auto f = [](int v) { return v + 1; };
  SECTION("nested") {
    if (f(1) == 2) {
      CHECK(open.size() == 1);
    }
  }
  CHECK(close == '}');
}

// A helper the next case uses, added between two cases.
static int helper_between(int v) {
  return v * 2;
}

TEST_CASE("uses the helper") { CHECK(helper_between(2) == 4); }

struct After {
  int x = 0;
};
"""


def line_of(text: str, needle: str) -> int:
    """1-based number of the first line of `text` containing `needle` (keeps fixtures free of hand-counted lines)."""
    for number, line in enumerate(text.splitlines(), start=1):
        if needle in line:
            return number
    raise AssertionError(f"{needle!r} not in fixture")


def hunk(old_start: int, removed: Sequence[str], new_start: int, added: Sequence[str]) -> str:
    """One -U0 hunk; the header counts follow from the lines, as `git diff -U0` writes them."""
    old = f"{old_start},{len(removed)}" if len(removed) != 1 else str(old_start)
    new = f"{new_start},{len(added)}" if len(added) != 1 else str(new_start)
    body = [f"-{r}" for r in removed] + [f"+{a}" for a in added]
    return "\n".join([f"@@ -{old} +{new} @@", *body]) + "\n"


def file_diff(path: str, *hunks: str) -> str:
    """The `git diff` text for one modified file."""
    head = f"diff --git a/{path} b/{path}\nindex 1111111..2222222 100644\n--- a/{path}\n+++ b/{path}\n"
    return head + "".join(hunks)


def case(name: str, start: int, end: int, file: str = "tests/test_x.cpp") -> Case:
    return Case(file, name, start, end, "nibble_tests")


class GateTests(unittest.TestCase):
    def test_is_fix_title_or_subject(self) -> None:
        self.assertTrue(fix_proof.is_fix("fix(cpu): x", []))
        self.assertTrue(fix_proof.is_fix("feat: x", ["wip", "fix: y"]))
        self.assertTrue(fix_proof.is_fix("fix(cpu,gguf)!: breaking", []))
        self.assertFalse(fix_proof.is_fix("fixup: x", []))
        self.assertFalse(fix_proof.is_fix("fix x", ["fix:no space", "Fix: caps"]))
        self.assertFalse(fix_proof.is_fix("feat: prefix fix: later", []))

    def test_touches_code(self) -> None:
        self.assertTrue(fix_proof.touches_code(["README.md", "src/gguf.cpp"]))
        self.assertTrue(fix_proof.touches_code(["include/nibble/api.h"]))
        self.assertTrue(fix_proof.touches_code(["cli/main.cpp"]))
        self.assertFalse(fix_proof.touches_code(["tests/test_gguf.cpp", "docs/src/x.md", "tools/ci/a.py"]))
        self.assertFalse(fix_proof.touches_code([]))


class TestCasesTests(unittest.TestCase):
    def test_test_cases_names_and_spans(self) -> None:
        cases = fix_proof.test_cases("tests/test_x.cpp", SOURCE)
        self.assertEqual(
            [c.name for c in cases],
            ["alpha, with [tags] and * stars", "beta multi-line header", None, "gamma"],
        )
        alpha, beta, e2e, gamma = cases
        # A span runs from the TEST_CASE line to the line of the body's closing brace.
        self.assertEqual(alpha.start, line_of(SOURCE, 'TEST_CASE("alpha'))
        self.assertEqual(alpha.end, line_of(SOURCE, 'const char* s = "TEST_CASE') + 1)
        self.assertEqual(beta.start, line_of(SOURCE, "beta multi-line") - 1)
        self.assertEqual(beta.end, line_of(SOURCE, '"[c]") {') + 2)
        # The macro's case is its one line; the invocations below it are not inside any body.
        self.assertEqual(e2e.start, line_of(SOURCE, 'TEST_CASE("e2e golden'))
        self.assertEqual(e2e.end, e2e.start)
        self.assertEqual(gamma.end, len(SOURCE.splitlines()))
        self.assertTrue(all(c.binary == "nibble_tests" for c in cases))

    def test_commented_and_quoted_headers_are_not_cases(self) -> None:
        names = [c.name for c in fix_proof.test_cases("tests/test_x.cpp", SOURCE)]
        self.assertNotIn("commented out", names)
        self.assertNotIn("in a string", names)

    def test_alloc_tests_use_their_own_binary(self) -> None:
        (c,) = fix_proof.test_cases("tests/alloc_tests.cpp", 'TEST_CASE("a", "[alloc]") {}\n')
        self.assertEqual(c.binary, "nibble_alloc_tests")

    def test_names_that_are_not_one_plain_literal_are_none(self) -> None:
        forms = [
            'TEST_CASE("a" "b") {}',  # adjacent literals
            'TEST_CASE(NAME, "[t]") {}',
            'TEST_CASE(R"(raw)", "[t]") {}',
            'TEST_CASE(u8"wide") {}',
            'TEST_CASE("new\\nline") {}',  # an escape other than \\" \\\\ \\'
            'TEST_CASE("") {}',  # Catch2 invents a name
            'TEST_CASE(" padded ") {}',
            'TEST_CASE("exclude:foo") {}',  # as a spec it would exclude instead of select
            'TEMPLATE_TEST_CASE("t", "[t]", int, long) {}',
            'TEST_CASE_METHOD(Fixture, "m", "[t]") {}',
            'SCENARIO("s") {}',
        ]
        for text in forms:
            with self.subTest(text):
                (c,) = fix_proof.test_cases("tests/test_x.cpp", text + "\n")
                self.assertIsNone(c.name)

    def test_escaped_quote_and_backslash_in_a_name_are_decoded(self) -> None:
        (c,) = fix_proof.test_cases("tests/test_x.cpp", r'TEST_CASE("say \"hi\" a\\b") {}' + "\n")
        self.assertEqual(c.name, r'say "hi" a\b')

    def test_comment_between_arguments_is_ignored(self) -> None:
        (c,) = fix_proof.test_cases("tests/test_x.cpp", 'TEST_CASE("a" /* why */, "[t]") {}\n')
        self.assertEqual(c.name, "a")

    def test_span_ends_at_the_closing_brace_of_the_body(self) -> None:
        first, second = fix_proof.test_cases("tests/test_x.cpp", BODIES)
        self.assertEqual(
            (first.start, first.end), (line_of(BODIES, "braces in literals"), line_of(BODIES, "CHECK(close") + 1)
        )
        self.assertEqual((second.start, second.end), (line_of(BODIES, "uses the helper"),) * 2)

    def test_span_falls_back_to_the_next_header_when_the_body_cannot_be_delimited(self) -> None:
        # Unbalanced braces across a preprocessor branch: no closing brace before the next case, so the old span
        # (up to the line before the next TEST_CASE) is kept rather than a guess.
        unbalanced = (
            'TEST_CASE("a") {\n#if X\n  if (x) {\n#else\n  if (y) {\n#endif\n  }\n}\n\nint h;\nTEST_CASE("b") {}\n'
        )
        a, b = fix_proof.test_cases("tests/test_x.cpp", unbalanced)
        self.assertEqual((a.start, a.end), (1, b.start - 1))
        # A header whose body is not a brace right after it (a macro that expands to the header).
        no_body = '#define CASE TEST_CASE("m")\nCASE {\n  CHECK(true);\n}\n'
        (m,) = fix_proof.test_cases("tests/test_x.cpp", no_body)
        self.assertEqual((m.start, m.end), (1, 4))
        # An unterminated body at the end of the file runs to the last line.
        (u,) = fix_proof.test_cases("tests/test_x.cpp", 'TEST_CASE("u") {\n  CHECK(true);\n')
        self.assertEqual((u.start, u.end), (1, 2))

    def test_real_repository_files(self) -> None:
        # Every real test file parses; only the E2E_CASE macro (tests/test_e2e_golden.cpp) has an unnamed case.
        files = sorted((REPO_ROOT / "tests").glob("*.cpp"))
        self.assertTrue(files)
        unnamed: list[str] = []
        total = 0
        for path in files:
            rel = f"tests/{path.name}"
            for c in fix_proof.test_cases(rel, path.read_text(encoding="utf-8")):
                total += 1
                if c.name is None:
                    unnamed.append(rel)
        self.assertGreater(total, 100)
        self.assertEqual(unnamed, ["tests/test_e2e_golden.cpp"])
        # Every span of the real files ends on its body's closing brace, before the next case (no fallback).
        for path in files:
            text = path.read_text(encoding="utf-8")
            lines = text.splitlines()
            cases = fix_proof.test_cases(f"tests/{path.name}", text)
            for i, c in enumerate(cases):
                with self.subTest(file=path.name, line=c.start):
                    self.assertIn("}", lines[c.end - 1])
                    if i + 1 < len(cases):
                        self.assertLess(c.end, cases[i + 1].start)
        real = {c.name for c in fix_proof.test_cases("t.cpp", (REPO_ROOT / "tests/test_quant.cpp").read_text("utf-8"))}
        self.assertIn("golden: dot_q{4,5,6}_K_q8_K bitwise equals ggml generic", real)


class ChangedLinesTests(unittest.TestCase):
    def test_added_lines_and_other_files(self) -> None:
        diff = (
            file_diff("tests/test_a.cpp", hunk(10, [], 11, ["x", "y"]), hunk(30, ["old"], 32, ["new"]))
            + file_diff("tests/alloc_tests.cpp", hunk(1, [], 2, ["z"]))
            + file_diff("src/a.cpp", hunk(5, [], 6, ["no"]))
            + file_diff("tests/fuzz/fuzz_gguf.cpp", hunk(5, [], 6, ["no"]))
            + file_diff("tests/data/quant_golden.inc", hunk(5, [], 6, ["no"]))
        )
        self.assertEqual(
            fix_proof.changed_lines(diff),
            {"tests/test_a.cpp": {11, 12, 32}, "tests/alloc_tests.cpp": {2}},
        )

    def test_pure_deletion_counts_the_line_before_the_gap(self) -> None:
        diff = file_diff("tests/test_a.cpp", hunk(10, ["gone", "gone2"], 9, []), hunk(1, ["top"], 0, []))
        self.assertEqual(fix_proof.changed_lines(diff), {"tests/test_a.cpp": {9, 0}})

    def test_removed_text_that_looks_like_a_file_header_stays_content(self) -> None:
        diff = file_diff("tests/test_a.cpp", hunk(10, ["-- a comment", "++ b"], 9, []))
        self.assertEqual(fix_proof.changed_lines(diff), {"tests/test_a.cpp": {9}})

    def test_a_non_test_file_cannot_redirect_a_later_hunk(self) -> None:
        # tests/fuzz/*.cpp matches the `tests/*.cpp` pathspec. Its added line `++ b/tests/test_x.cpp` is diff text
        # `+++ b/tests/test_x.cpp`; its second hunk must not be read as a hunk of test_x.cpp.
        diff = file_diff(
            "tests/fuzz/f.cpp", hunk(3, [], 4, ["++ b/tests/test_x.cpp"]), hunk(40, ["a"], 41, ["b"])
        ) + file_diff("tests/test_a.cpp", hunk(7, [], 8, ["ok"]))
        self.assertEqual(fix_proof.changed_lines(diff), {"tests/test_a.cpp": {8}})

    def test_form_feed_inside_a_line_does_not_split_it(self) -> None:
        # str.splitlines would cut this added line at the form feed and read `+++ b/...` as a file header.
        diff = file_diff(
            "tests/test_a.cpp", hunk(3, [], 4, ["x\x0c+++ b/tests/test_y.cpp"]), hunk(40, ["a"], 41, ["b"])
        )
        self.assertEqual(fix_proof.changed_lines(diff), {"tests/test_a.cpp": {4, 41}})

    def test_deleted_header_contributes_no_line_and_is_named(self) -> None:
        diff = file_diff(
            "tests/test_a.cpp",
            hunk(20, ['TEST_CASE("B, [x]", "[t]") {', "  CHECK(true);", "}", ""], 19, []),
        )
        self.assertEqual(fix_proof.changed_lines(diff), {})
        self.assertEqual(fix_proof.deleted_case_names(diff), {"tests/test_a.cpp": ["B, [x]"]})

    def test_deleted_macro_built_header_is_named_macro_built(self) -> None:
        diff = file_diff("tests/test_a.cpp", hunk(5, ['  TEST_CASE("e2e: " PROMPT) { run(); }'], 4, []))
        self.assertEqual(fix_proof.deleted_case_names(diff), {"tests/test_a.cpp": ["<macro-built>"]})

    def test_deleted_multiline_header_is_named(self) -> None:
        diff = file_diff("tests/test_a.cpp", hunk(5, ["TEST_CASE(", '    "two lines",', '    "[t]") {', "}"], 4, []))
        self.assertEqual(fix_proof.deleted_case_names(diff), {"tests/test_a.cpp": ["two lines"]})

    def test_whole_file_deletion(self) -> None:
        diff = (
            "diff --git a/tests/test_old.cpp b/tests/test_old.cpp\ndeleted file mode 100644\nindex 111..000\n"
            "--- a/tests/test_old.cpp\n+++ /dev/null\n" + hunk(1, ['TEST_CASE("only") {', "}"], 0, [])
        )
        self.assertEqual(fix_proof.changed_lines(diff), {})
        self.assertEqual(fix_proof.deleted_case_names(diff), {"tests/test_old.cpp": ["only"]})

    def test_hunk_header_without_counts(self) -> None:
        diff = (
            "--- a/tests/test_a.cpp\n+++ b/tests/test_a.cpp\n@@ -4 +4 @@ void f()\n-a\n+b\n"
            "\\ No newline at end of file\n"
        )
        self.assertEqual(fix_proof.changed_lines(diff), {"tests/test_a.cpp": {4}})

    def test_empty_diff(self) -> None:
        self.assertEqual(fix_proof.changed_lines(""), {})
        self.assertEqual(fix_proof.deleted_case_names(""), {})


class SelectTests(unittest.TestCase):
    PATH = "tests/test_x.cpp"

    def select(self, *hunks: str, source: str = SOURCE) -> Selection:
        return fix_proof.select(file_diff(self.PATH, *hunks), {self.PATH: source})

    def names(self, sel: Selection) -> list[str | None]:
        return [c.name for c in sel.cases]

    def test_select_added_and_modified_cases(self) -> None:
        added_line = line_of(SOURCE, "gamma") + 1  # a line added inside gamma
        modified = line_of(SOURCE, "REQUIRE(helper()")  # a line changed inside alpha
        sel = self.select(hunk(modified - 1, ["old"], modified, ["new"]), hunk(added_line - 1, [], added_line, ["x"]))
        self.assertEqual(self.names(sel), ["alpha, with [tags] and * stars", "gamma"])
        self.assertEqual(sel.unresolved, [])
        self.assertEqual(sel.cases[0].binary, "nibble_tests")

    def test_select_helper_lines_outside_every_case_select_nothing(self) -> None:
        sel = self.select(hunk(3, [], 4, ["int more() { return 2; }"]), hunk(0, [], 1, ["// top"]))
        self.assertEqual(sel.cases, [])
        # Listed, so that the summary says why such a change proves nothing.
        self.assertEqual(len(sel.unresolved), 1)
        self.assertIn(f"{self.PATH}: lines 1, 4 changed outside every TEST_CASE body", sel.unresolved[0])

    def test_select_a_helper_between_two_cases_selects_neither(self) -> None:
        # The helper follows a closed case. With the span running on to the next TEST_CASE it selected that case,
        # and if the unchanged case failed on the base anyway, the check read "proven" for a test the PR never touched.
        helper = line_of(BODIES, "return v * 2;")
        sel = self.select(hunk(helper, ["  return v + v;"], helper, ["  return v * 2;"]), source=BODIES)
        self.assertEqual(sel.cases, [])
        self.assertEqual(
            sel.unresolved,
            [
                f"{self.PATH}: line {helper} changed outside every TEST_CASE body (helpers, macro invocations): "
                "no case selected for that"
            ],
        )
        # Added after the last case: the struct is outside too.
        after = line_of(BODIES, "int x = 0;")
        self.assertEqual(self.select(hunk(after - 1, [], after, ["  int x = 0;"]), source=BODIES).cases, [])

    def test_select_deletions_inside_and_just_after_a_body(self) -> None:
        close = line_of(BODIES, "CHECK(close") + 1  # the first case's closing brace
        # Lines removed just before the closing brace: inside the body.
        sel = self.select(hunk(close, ["  CHECK(true);"], close - 1, []), source=BODIES)
        self.assertEqual(self.names(sel), ["braces in literals and comments"])
        # Lines removed just after it (a helper deleted): outside.
        sel = self.select(hunk(close + 1, ["static int gone() { return 1; }"], close, []), source=BODIES)
        self.assertEqual(sel.cases, [])
        self.assertEqual(len(sel.unresolved), 1)

    def test_select_blank_lines_between_cases_are_not_listed(self) -> None:
        blank = line_of(BODIES, "// A helper the next case uses") - 1
        sel = self.select(hunk(blank - 1, [], blank, [""]), hunk(blank + 5, [""], blank + 4, []), source=BODIES)
        self.assertEqual(sel, Selection([], []))

    def test_select_multi_line_header_change_selects_the_case(self) -> None:
        tag_line = line_of(SOURCE, '"[c]") {')
        sel = self.select(hunk(tag_line, ['"[old]") {'], tag_line, ['"[c]") {']))
        self.assertEqual(self.names(sel), ["beta multi-line header"])

    def test_select_deletion_only_edit(self) -> None:
        # Two lines removed from inside alpha: the hunk is `+c,0` with c the line before the gap.
        inside = line_of(SOURCE, "SECTION")
        sel = self.select(hunk(inside + 1, ["a", "b"], inside, []))
        self.assertEqual(self.names(sel), ["alpha, with [tags] and * stars"])

    def test_select_deletion_at_case_boundary(self) -> None:
        beta = line_of(SOURCE, "beta multi-line") - 1  # the TEST_CASE( line of the two-line header
        # Last lines of alpha removed, right before beta's TEST_CASE line: the gap follows alpha's last kept line.
        sel = self.select(hunk(beta - 2, ["x", "}"], beta - 3, []))
        self.assertEqual(self.names(sel), ["alpha, with [tags] and * stars"])
        # Lines right after beta's TEST_CASE line removed: the line before the gap is the header line itself.
        sel = self.select(hunk(beta + 1, ['    "beta multi-line header",'], beta, []))
        self.assertEqual(self.names(sel), ["beta multi-line header"])

    def test_select_whole_case_deletion(self) -> None:
        source = 'TEST_CASE("A") {\n  CHECK(true);\n}\n\nTEST_CASE("C") {\n  CHECK(true);\n}\n'
        # B (lines 5-8 of the old file) was between A and C; its removal leaves the gap after line 4.
        sel = self.select(
            hunk(5, ['TEST_CASE("B") {', "  CHECK(true);", "}", ""], 4, []),
            source=source,
        )
        self.assertEqual(sel.cases, [])
        self.assertEqual(sel.unresolved, ["deleted: B"])

    def test_select_deleting_a_case_while_editing_another_keeps_the_edit(self) -> None:
        source = 'TEST_CASE("A") {\n  CHECK(true);\n}\n\nTEST_CASE("C") {\n  CHECK(true);\n}\n'
        sel = self.select(
            hunk(2, ["  CHECK(false);"], 2, ["  CHECK(true);"]),
            hunk(5, ['TEST_CASE("B") {', "}", ""], 4, []),
            source=source,
        )
        self.assertEqual(self.names(sel), ["A"])
        self.assertEqual(sel.unresolved, ["deleted: B"])

    def test_select_macro_case_is_unresolved(self) -> None:
        # An edit of the macro's TEST_CASE line lands in the unnamed case; an edited E2E_CASE invocation is outside
        # every body. Neither is ever a selected case, and both are listed.
        invocation = line_of(SOURCE, 'E2E_CASE("q4_0"')
        sel = self.select(hunk(invocation, ['E2E_CASE("q8_0", "chat")'], invocation, ['E2E_CASE("q4_0", "chat")']))
        self.assertEqual(sel.cases, [])
        self.assertEqual(len(sel.unresolved), 1)
        self.assertIn(f"{self.PATH}: line {invocation} changed outside every TEST_CASE body", sel.unresolved[0])
        macro = line_of(SOURCE, 'TEST_CASE("e2e golden')
        sel = self.select(hunk(macro, ["a"], macro, ["b"]), hunk(macro + 1, ["c"], macro + 1, ["d"]))
        self.assertEqual(sel.cases, [])
        self.assertEqual(len(sel.unresolved), 2)
        self.assertIn(f"{self.PATH}:{macro}: test name is not one plain string literal", sel.unresolved)

    def test_select_file_no_test_binary_builds_is_unresolved(self) -> None:
        diff = file_diff("tests/simd_helpers_test.cpp", hunk(5, [], 6, ["x"]))
        sel = fix_proof.select(diff, {"tests/simd_helpers_test.cpp": 'TEST_CASE("h") {}\n'})
        self.assertEqual(sel.cases, [])
        self.assertEqual(len(sel.unresolved), 1)
        self.assertIn("tests/simd_helpers_test.cpp", sel.unresolved[0])

    def test_select_missing_source_selects_nothing(self) -> None:
        diff = file_diff(self.PATH, hunk(5, [], 6, ["x"]))
        self.assertEqual(fix_proof.select(diff, {}), Selection([], []))

    def test_select_alloc_tests_binary(self) -> None:
        diff = file_diff("tests/alloc_tests.cpp", hunk(1, [], 2, ["x"]))
        sel = fix_proof.select(diff, {"tests/alloc_tests.cpp": 'TEST_CASE("a", "[alloc]") {\n  x;\n}\n'})
        self.assertEqual([c.binary for c in sel.cases], ["nibble_alloc_tests"])


class CatchSpecTests(unittest.TestCase):
    def test_catch_spec_escapes(self) -> None:
        self.assertEqual(fix_proof.catch_spec("a, b [x] *"), r"a\, b \[x\] \*")

    def test_catch_spec_escapes_each_special_character(self) -> None:
        for ch in ',[]*"~\\':
            with self.subTest(ch):
                self.assertEqual(fix_proof.catch_spec(f"a{ch}b"), f"a\\{ch}b")

    def test_catch_spec_leading_tilde_and_star_and_tag_bracket(self) -> None:
        self.assertEqual(fix_proof.catch_spec("~lead"), r"\~lead")
        self.assertEqual(fix_proof.catch_spec("*lead"), r"\*lead")
        self.assertEqual(fix_proof.catch_spec("[tag] x"), r"\[tag\] x")

    def test_catch_spec_leading_dash_never_reaches_catch2_as_an_option(self) -> None:
        # `build/nibble_tests "-1 handling"` dies with "Unrecognised token: -1"; Catch2 accepts the escaped form
        # (checked with a real binary: `--list-tests '\\-1 handling'` lists the case).
        self.assertEqual(fix_proof.catch_spec("-1 handling"), r"\-1 handling")
        self.assertEqual(fix_proof.catch_spec("--list-tests"), r"\--list-tests")
        self.assertEqual(fix_proof.catch_spec("a-b"), "a-b")
        (c,) = fix_proof.test_cases("tests/test_x.cpp", 'TEST_CASE("-1 handling") {}\n')
        self.assertEqual(c.name, "-1 handling")

    def test_catch_spec_leaves_other_characters_alone(self) -> None:
        # `{`, `}`, `/`, `:` and `<` need no escape: the real name below runs under Catch2 as written.
        name = "golden: dot_q{4,5,6}_K_q8_K a/b <= 2e-7"
        self.assertEqual(fix_proof.catch_spec(name), r"golden: dot_q{4\,5\,6}_K_q8_K a/b <= 2e-7")


# Real `-r junit` output (Catch2 v3.16.1), trimmed to the testcase elements.
def _junit(*elements: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n<testsuites>\n'
        '  <testsuite name="t" errors="0" failures="1" skipped="1" tests="3" hostname="tbd" time="0.000">\n'
        '    <properties>\n      <property name="random-seed" value="1"/>\n    </properties>\n'
        + "".join(f"    {e}\n" for e in elements)
        + "    <system-out/>\n    <system-err/>\n  </testsuite>\n</testsuites>\n"
    )


PASSED = '<testcase classname="t.global" name="passes" time="0.000" status="run"/>'
FAILED = (
    '<testcase classname="t.global" name="fails, with [x] &amp; &lt;b> &quot;q&quot;" time="0.000" status="run">\n'
    '      <failure message="1 == 2" type="REQUIRE">\nFAILED:\n  REQUIRE( 1 == 2 )\nat t.cpp:3\n      </failure>\n'
    "    </testcase>"
)
SKIPPED = (
    '<testcase classname="t.global" name="skips" time="0.000" status="run">\n'
    '      <skipped type="SKIP">\nSKIPPED\nmodel or golden missing\nat t.cpp:4\n      </skipped>\n    </testcase>'
)
THREW = (
    '<testcase classname="s.global" name="thrower" time="0.000" status="run">\n'
    '      <error type="TEST_CASE">\nFAILED:\nboom\nat s.cpp:4\n      </error>\n    </testcase>'
)
ABORTED = (
    '<testcase classname="s.global" name="exitcrash" time="0.000" status="run">\n      <error type="TEST_CASE">\n'
    "FAILED:\nSIGABRT - Abort (abnormal termination) signal\nat s.cpp:9\n      </error>\n    </testcase>"
)
MAYFAIL = (
    '<testcase classname="s.global" name="mayfail" time="0.000" status="run">\n'
    '      <skipped message="TEST_CASE tagged with !mayfail"/>\n'
    '      <failure message="1==2" type="REQUIRE">\nFAILED:\n  REQUIRE( 1==2 )\nat s.cpp:7\n      </failure>\n'
    "    </testcase>"
)
SECTIONS = (
    '<testcase classname="s.global" name="sec/one" time="0.000" status="run"/>\n'
    '    <testcase classname="s.global" name="sec/two" time="0.000" status="run">\n'
    '      <failure message="1 == 2" type="REQUIRE">\nFAILED:\nat s.cpp:3\n      </failure>\n    </testcase>'
)
SECTION_SKIP = (
    '<testcase classname="s.global" name="part" time="0.000" status="run"/>\n'
    '    <testcase classname="s.global" name="part/a" time="0.000" status="run">\n'
    '      <skipped type="SKIP">\nSKIPPED\nx\nat s.cpp:5\n      </skipped>\n    </testcase>\n'
    '    <testcase classname="s.global" name="part/b" time="0.000" status="run"/>'
)


class ParseJunitTests(unittest.TestCase):
    def test_parse_junit_states(self) -> None:
        xml = _junit(PASSED, FAILED, SKIPPED)
        self.assertEqual(fix_proof.parse_junit(xml, "passes"), "passed")
        self.assertEqual(fix_proof.parse_junit(xml, 'fails, with [x] & <b> "q"'), "failed")
        self.assertEqual(fix_proof.parse_junit(xml, "skips"), "skipped")
        self.assertEqual(fix_proof.parse_junit(xml, "absent"), "missing")

    def test_exception_and_caught_signal_are_failures(self) -> None:
        self.assertEqual(fix_proof.parse_junit(_junit(THREW), "thrower"), "failed")
        self.assertEqual(fix_proof.parse_junit(_junit(ABORTED), "exitcrash"), "failed")

    def test_mayfail_failure_is_not_a_failure(self) -> None:
        self.assertEqual(fix_proof.parse_junit(_junit(MAYFAIL), "mayfail"), "skipped")

    def test_sections_count_for_their_case(self) -> None:
        self.assertEqual(fix_proof.parse_junit(_junit(SECTIONS), "sec"), "failed")
        self.assertEqual(fix_proof.parse_junit(_junit(SECTION_SKIP), "part"), "passed")
        self.assertEqual(fix_proof.parse_junit(_junit(SECTIONS), "se"), "missing")  # a prefix is not a section

    def test_empty_run_and_broken_xml_are_missing(self) -> None:
        self.assertEqual(fix_proof.parse_junit(_junit(), "passes"), "missing")
        self.assertEqual(fix_proof.parse_junit("", "passes"), "missing")
        self.assertEqual(fix_proof.parse_junit(_junit(FAILED)[:200], 'fails, with [x] & <b> "q"'), "missing")

    def test_entity_declarations_are_refused(self) -> None:
        bomb = '<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa">]><testsuites>' + PASSED + "</testsuites>"
        self.assertEqual(fix_proof.parse_junit(bomb, "passes"), "missing")


def selection_of(*names: str, unresolved: Sequence[str] = ()) -> Selection:
    return Selection([case(n, 1 + i * 10, 9 + i * 10) for i, n in enumerate(names)], list(unresolved))


class ClassifyTests(unittest.TestCase):
    def test_classify_table(self) -> None:
        one = selection_of("a")
        two = selection_of("a", "b")
        res: dict[str, CaseResult]
        # 1. no resolved case, whatever else is true
        self.assertEqual(fix_proof.classify(Selection([], []), True, {}), "no-test")
        self.assertEqual(fix_proof.classify(Selection([], ["x: macro"]), False, {"a": "failed"}), "no-test")
        # 2. the base does not compile
        self.assertEqual(fix_proof.classify(one, False, {}), "proven-base-does-not-compile")
        # 3. any failed case proves it, the passing ones do not count against it (R9)
        res = {"a": "passed", "b": "failed"}
        self.assertEqual(fix_proof.classify(two, True, res), "proven")
        # 4. all skipped or missing
        self.assertEqual(fix_proof.classify(two, True, {"a": "skipped"}), "not-runnable")
        self.assertEqual(fix_proof.classify(two, True, {}), "not-runnable")
        self.assertEqual(fix_proof.classify(two, True, {"a": "skipped", "b": "missing"}), "not-runnable")
        # 5. otherwise
        self.assertEqual(fix_proof.classify(two, True, {"a": "passed", "b": "skipped"}), "does-not-reproduce")
        self.assertEqual(fix_proof.classify(two, True, {"a": "passed", "b": "passed"}), "does-not-reproduce")

    def test_unresolved_and_unrunnable_cases_are_never_proven(self) -> None:
        # Review Focus 3: a macro-built name, a case that skips, and a name needing escapes.
        macro_only = Selection([], ["tests/test_e2e_golden.cpp:233: test name is not one plain string literal"])
        self.assertEqual(fix_proof.classify(macro_only, True, {}), "no-test")
        skipping = selection_of("e2e prompt tokens match llama.cpp")
        self.assertEqual(
            fix_proof.classify(skipping, True, {"e2e prompt tokens match llama.cpp": "skipped"}), "not-runnable"
        )
        tricky = selection_of('x, [y] * "z" ~')
        self.assertEqual(fix_proof.classify(tricky, True, {}), "not-runnable")

    def test_exit_codes_with_label(self) -> None:
        self.assertEqual(fix_proof.exit_code("no-test", False), 1)
        self.assertEqual(fix_proof.exit_code("no-test", True), 0)
        self.assertEqual(fix_proof.exit_code("does-not-reproduce", False), 1)
        self.assertEqual(fix_proof.exit_code("does-not-reproduce", True), 1)  # the label excuses a missing test only
        for outcome in ("proven", "proven-base-does-not-compile", "not-runnable"):
            with self.subTest(outcome):
                self.assertEqual(fix_proof.exit_code(outcome, False), 0)

    def test_summary_lists_cases_and_unresolved(self) -> None:
        sel = selection_of(
            "fails", "passes", unresolved=["deleted: gone", "tests/t.cpp:9: test name is not one plain string literal"]
        )
        res: dict[str, CaseResult] = {"fails": "failed", "passes": "passed"}
        text = fix_proof.summary("proven", sel, res)
        self.assertIn("Bug-fix proof: proven", text)
        self.assertIn("- failed: fails", text)
        self.assertIn("- passed (already passing): passes", text)
        self.assertIn("deleted: gone", text)
        for outcome in ("no-test", "does-not-reproduce", "not-runnable", "proven-base-does-not-compile"):
            with self.subTest(outcome):
                body = fix_proof.summary(outcome, sel, res)
                self.assertIn(f"Bug-fix proof: {outcome}", body)
        self.assertNotIn("Proven:", fix_proof.summary("not-runnable", sel, {}))
        self.assertIn("no-test-needed", fix_proof.summary("no-test", Selection([], []), {}))

    def test_summary_neutralises_markdown_in_names(self) -> None:
        sel = selection_of("[x](http://evil) @user `tick`\nnewline")
        text = fix_proof.summary("does-not-reproduce", sel, {})
        self.assertNotIn("](", text)
        self.assertNotIn("\nnewline", text)
        self.assertNotIn("`tick`", text)


def status(**changes: Any) -> fix_proof.RunStatus:
    """What `run` records about one process: a started one that exited 0 on a unique spec, with `changes` applied."""
    fields: dict[str, Any] = {"started": True, "exit_code": 0, "signal": None, "timed_out": False, "unique": True}
    return fix_proof.RunStatus(**(fields | changes))


class CaseResultTests(unittest.TestCase):
    """How one case's JUnit and process outcome combine (P21: crashes, sanitizers and zero-assertion cases)."""

    def result(self, xml: str | None, st: fix_proof.RunStatus | None) -> CaseResult:
        return fix_proof.case_result("passes", xml, st)[0]

    def test_junit_alone_when_there_is_no_status(self) -> None:
        self.assertEqual(self.result(_junit(PASSED), None), "passed")
        self.assertEqual(self.result(_junit(SKIPPED.replace("skips", "passes")), None), "skipped")
        self.assertEqual(self.result(None, None), "missing")
        self.assertEqual(self.result("not xml", None), "missing")

    def test_failed_and_skipped_junit_win_over_the_exit_status(self) -> None:
        failed = _junit(FAILED.replace("fails, with [x] &amp; &lt;b> &quot;q&quot;", "passes"))
        skipped = _junit(SKIPPED.replace("skips", "passes"))
        self.assertEqual(self.result(failed, status(exit_code=0)), "failed")
        self.assertEqual(self.result(skipped, status(exit_code=4)), "skipped")  # exit 4 is Catch2's "all skipped"
        self.assertEqual(self.result(skipped, status(exit_code=1)), "skipped")

    def test_passed_junit_with_a_failing_process_is_failed(self) -> None:
        # LeakSanitizer reports at exit, after Catch2 wrote the XML with the case passed (I3).
        passed = _junit(PASSED)
        for kwargs in (
            {"exit_code": 1},
            {"exit_code": 23},
            {"exit_code": None, "signal": 6},
            {"exit_code": None, "timed_out": True},
        ):
            with self.subTest(kwargs):
                self.assertEqual(self.result(passed, status(**kwargs)), "failed")
        for kwargs in ({"exit_code": 0}, {"exit_code": 4}):  # 4: a skipped section next to a passing one
            with self.subTest(kwargs):
                self.assertEqual(self.result(passed, status(**kwargs)), "passed")
        # Without proof that the spec named exactly this case, a non-zero exit is not attributed to it.
        self.assertEqual(self.result(passed, status(exit_code=1, unique=False)), "passed")

    def test_no_junit_and_a_died_process_is_failed_only_for_a_unique_spec(self) -> None:
        for kwargs in (
            {"exit_code": None, "signal": 6},
            {"exit_code": None, "signal": 11},
            {"exit_code": None, "timed_out": True},
            {"exit_code": 1},
        ):
            with self.subTest(kwargs):
                self.assertEqual(self.result(None, status(**kwargs)), "failed")
                self.assertEqual(self.result(None, status(unique=False, **kwargs)), "missing")

    def test_no_junit_and_catch2_refusing_the_run_is_missing(self) -> None:
        # 1 with a bad command line ("Unrecognised token") is indistinguishable from a sanitizer exit, which is
        # why a unique `--list-tests` match is required above; 2, 3, 5 and 6 say that Catch2 ran nothing.
        for code in (2, 3, 5, 6):
            with self.subTest(code):
                self.assertEqual(self.result(None, status(exit_code=code)), "missing")
        self.assertEqual(self.result(None, fix_proof.RunStatus(False, None, None, False, True)), "missing")

    def test_zero_assertion_case_that_exits_zero_passed(self) -> None:
        # Catch2 writes no <testcase> for a case without assertions (M2).
        self.assertEqual(self.result(_junit(), status(exit_code=0)), "passed")
        self.assertEqual(self.result(None, status(exit_code=0)), "passed")
        self.assertEqual(self.result(_junit(), status(exit_code=0, unique=False)), "missing")
        self.assertEqual(self.result(_junit(), status(exit_code=4)), "skipped")

    def test_notes_say_why(self) -> None:
        note = fix_proof.case_result("passes", None, status(exit_code=None, signal=6))[1]
        self.assertIn("signal 6", note)
        note = fix_proof.case_result("passes", None, status(exit_code=None, timed_out=True))[1]
        self.assertIn("timed out", note)
        note = fix_proof.case_result("passes", _junit(PASSED), status(exit_code=1))[1]
        self.assertIn("status 1", note)
        self.assertIn("reported a pass", note)


# A stand-in for a Catch2 test binary: it understands `<spec> --list-tests -r xml` and `<spec> -r junit -o FILE`, and
# behaves according to the test name, so `run` can be exercised with real processes (signals, timeouts, exit codes)
# without building the project. Like the real one it rejects an argument starting with `-` it does not know.
FAKE_BINARY = """\
#!{python}
import json, os, re, signal, sys, time
args = sys.argv[1:]
spec = args[0]
if spec.startswith("-"):
    print("Error(s) in input:\\n  Unrecognised token: " + spec)
    sys.exit(1)
name = re.sub(r"\\\\(.)", r"\\1", spec)
if "--list-tests" in args:
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "list-env.json"), "w") as f:
        json.dump({{k: os.environ.get(k) for k in ("ASAN_OPTIONS", "UBSAN_OPTIONS")}}, f)
    if name.startswith("listcrash"):
        # What a LeakSanitizer that cannot run, or a leak in static initialisation, does to every invocation.
        print("<MatchingTests><TestCase><Name>" + name + "</Name></TestCase></MatchingTests>")
        sys.exit(1)
    count = 2 if name.startswith("ambiguous") else 0 if name.startswith("nomatch") else 1
    cases = "".join(f"<TestCase><Name>{{name}}</Name></TestCase>" for _ in range(count))
    print("<MatchingTests>" + cases + "</MatchingTests>")
    sys.exit(0)
out = args[args.index("-o") + 1]
with open(os.path.join(os.path.dirname(out), "env-" + os.path.basename(out) + ".json"), "w") as f:
    json.dump({{k: os.environ.get(k) for k in ("ASAN_OPTIONS", "UBSAN_OPTIONS")}}, f)
def write(element, code):
    body = ""
    if element is not None:
        body = f'<testcase classname="t.global" name="{{name}}" status="run">{{element}}</testcase>'
    open(out, "w").write(f'<?xml version="1.0"?><testsuites><testsuite name="t">{{body}}</testsuite></testsuites>')
    print("leak report on stderr" if name == "leaks" else "done", file=sys.stderr)
    sys.exit(code)
if name == "aborts":
    os.kill(os.getpid(), signal.SIGABRT)
elif name == "hangs":
    time.sleep(60)
elif name == "fails":
    write("<failure message='x'>FAILED</failure>", 42)
elif name == "leaks":
    write("", 1)
elif name == "noassert":
    write(None, 0)
elif name == "skips":
    write("<skipped type='SKIP'>SKIPPED</skipped>", 4)
elif name.startswith("nomatch"):
    write(None, 2)
elif name.startswith("ambiguous") or name.startswith("listcrash"):
    os.kill(os.getpid(), signal.SIGABRT)
else:
    write("", 0)
"""


class RunTests(unittest.TestCase):
    """`run` and `classify` with fake Catch2 binaries as real subprocesses."""

    NAMES = [
        "passes",
        "fails",
        "leaks",
        "aborts",
        "hangs",
        "noassert",
        "skips",
        "ambiguous abort",
        "-1 handling",
        "x, [y] *",
    ]

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.build = Path(self.tmp.name, "build")
        self.build.mkdir()
        for binary in ("nibble_tests", "nibble_alloc_tests"):
            path = self.build / binary
            path.write_text(FAKE_BINARY.format(python=sys.executable), encoding="utf-8")
            path.chmod(path.stat().st_mode | stat.S_IXUSR)
        self.results = Path(self.tmp.name, "results")
        self.cases = Path(self.tmp.name, "cases.json")

    def write_cases(self, names: list[str], binary: str = "nibble_tests") -> Selection:
        selection = Selection(
            [Case("tests/test_x.cpp", n, 1 + 10 * i, 9 + 10 * i, binary) for i, n in enumerate(names)], []
        )
        payload = {"cases": [fix_proof._case_to_json(c) for c in selection.cases], "unresolved": []}
        self.cases.write_text(json.dumps(payload), encoding="utf-8")
        return selection

    def main(self, *argv: str) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = fix_proof.main(argv)
        return code, out.getvalue()

    def run_all(self, *extra: str) -> None:
        code, _ = self.main(
            "run", "--cases", str(self.cases), "--build-dir", str(self.build), "--results-dir", str(self.results),
            "--timeout", "2", *extra,
        )  # fmt: skip
        self.assertEqual(code, 0)

    def test_run_then_classify_each_kind_of_process_outcome(self) -> None:
        selection = self.write_cases(self.NAMES)
        self.run_all()
        results, notes = fix_proof.read_results(str(self.results), selection)
        self.assertEqual(
            results,
            {
                "passes": "passed",
                "fails": "failed",
                "leaks": "failed",  # the JUnit says passed, the process exited 1
                "aborts": "failed",  # no JUnit, SIGABRT, spec unique
                "hangs": "failed",  # timed out
                "noassert": "passed",  # no <testcase>, exit 0
                "skips": "skipped",
                "ambiguous abort": "missing",  # the spec matched two cases: not attributable
                "-1 handling": "passed",  # the leading dash was escaped; an unescaped one is refused with exit 1
                "x, [y] *": "passed",
            },
        )
        self.assertIn("signal 6", notes["aborts"])
        self.assertIn("timed out", notes["hangs"])

    def test_run_files(self) -> None:
        self.write_cases(["passes", "aborts"])
        self.run_all()
        self.assertTrue((self.results / "0.xml").is_file())
        self.assertFalse((self.results / "1.xml").exists())
        recorded = json.loads((self.results / "1.status.json").read_text(encoding="utf-8"))
        self.assertEqual(
            recorded, {"started": True, "exit_code": None, "signal": 6, "timed_out": False, "unique": True}
        )
        self.assertTrue((self.results / "0.log").is_file())  # the process output, as evidence for a failure
        self.assertTrue((self.results / "1.log").is_file())

    def test_run_sets_sanitizer_options_and_keeps_existing_values(self) -> None:
        self.write_cases(["passes"])
        env = {"ASAN_OPTIONS": "abort_on_error=1", "UBSAN_OPTIONS": "halt_on_error=0"}
        with mock.patch.dict(os.environ, env):
            self.run_all()
        seen = json.loads((self.results / "env-0.xml.json").read_text(encoding="utf-8"))
        self.assertEqual(seen["ASAN_OPTIONS"], "abort_on_error=1:detect_leaks=1")
        self.assertEqual(seen["UBSAN_OPTIONS"], "halt_on_error=0:print_stacktrace=1")
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ASAN_OPTIONS", None)
            os.environ.pop("UBSAN_OPTIONS", None)
            self.run_all()
        seen = json.loads((self.results / "env-0.xml.json").read_text(encoding="utf-8"))
        self.assertEqual(
            seen, {"ASAN_OPTIONS": "detect_leaks=1", "UBSAN_OPTIONS": "halt_on_error=1:print_stacktrace=1"}
        )

    def test_the_listing_probe_runs_with_the_environment_of_the_run(self) -> None:
        self.write_cases(["passes"])
        with mock.patch.dict(os.environ, {"ASAN_OPTIONS": "abort_on_error=1"}):
            self.run_all()
        seen = json.loads((self.build / "list-env.json").read_text(encoding="utf-8"))
        self.assertEqual(seen["ASAN_OPTIONS"], "abort_on_error=1:detect_leaks=1")
        self.assertIn("halt_on_error=1", seen["UBSAN_OPTIONS"])

    def test_a_failing_listing_is_not_a_unique_match(self) -> None:
        # The probe lists the case but exits 1 (a sanitizer failing before any test ran): the later crash of the
        # run is then not attributable to the case, so it must not count as proof.
        selection = self.write_cases(["listcrash case"])
        self.run_all()
        recorded = json.loads((self.results / "0.status.json").read_text(encoding="utf-8"))
        self.assertFalse(recorded["unique"])
        results, _ = fix_proof.read_results(str(self.results), selection)
        self.assertEqual(results, {"listcrash case": "missing"})

    def test_merge_options(self) -> None:
        self.assertEqual(fix_proof._merge_options(None, "a=1:b=2"), "a=1:b=2")
        self.assertEqual(fix_proof._merge_options("", "a=1"), "a=1")
        self.assertEqual(fix_proof._merge_options("b=0:c=3", "a=1:b=2"), "b=0:c=3:a=1")

    def test_missing_binary_is_recorded_not_fatal(self) -> None:
        self.write_cases(["passes"], binary="nibble_alloc_tests")
        (self.build / "nibble_alloc_tests").unlink()
        self.run_all()
        recorded = json.loads((self.results / "0.status.json").read_text(encoding="utf-8"))
        self.assertFalse(recorded["started"])
        results, _ = fix_proof.read_results(str(self.results), self.write_cases(["passes"], "nibble_alloc_tests"))
        self.assertEqual(results, {"passes": "missing"})

    def test_classify_cli_proves_a_crash_and_explains_it(self) -> None:
        self.write_cases(["passes", "aborts"])
        self.run_all()
        summary_file = Path(self.tmp.name, "summary.md")
        code, out = self.main(
            "classify", "--cases", str(self.cases), "--results-dir", str(self.results),
            "--base-compiled", "true", "--summary", str(summary_file),
        )  # fmt: skip
        self.assertEqual((code, out.strip()), (0, "outcome=proven"))
        text = summary_file.read_text(encoding="utf-8")
        self.assertIn("- failed: aborts", text)
        self.assertIn("signal 6", text)

    def test_a_refused_command_line_never_proves_anything(self) -> None:
        # Had the leading dash not been escaped, the binary would exit 1 without a JUnit file: with a unique
        # `--list-tests` match that would read as a crash. Here the listing finds nothing, so it stays missing.
        selection = self.write_cases(["nomatch case"])
        self.run_all()
        results, _ = fix_proof.read_results(str(self.results), selection)
        self.assertEqual(results, {"nomatch case": "missing"})


def git(repo: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True)
    return done.stdout.strip()


class CommandLineTests(unittest.TestCase):
    """`main` end to end, against a real temporary git repository and real-format JUnit files."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name, "repo")
        (self.repo / "tests").mkdir(parents=True)
        (self.repo / "src").mkdir()
        git(self.repo, "init", "-q", "-b", "master")
        git(self.repo, "config", "user.email", "t@example.invalid")
        git(self.repo, "config", "user.name", "t")
        git(self.repo, "config", "commit.gpgsign", "false")

    def commit(self, files: dict[str, str], message: str) -> str:
        for rel, text in files.items():
            Path(self.repo, rel).write_text(text, encoding="utf-8")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", message)
        return git(self.repo, "rev-parse", "HEAD")

    def run_main(self, *argv: str) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = fix_proof.main(argv)
        return code, out.getvalue()

    def make_prs(self) -> tuple[str, str]:
        base_src = (
            'TEST_CASE("keeps", "[a]") {\n  CHECK(true);\n}\n\nTEST_CASE("old, [one]", "[a]") {\n  CHECK(true);\n}\n'
        )
        base = self.commit({"tests/test_x.cpp": base_src, "src/a.cpp": "int a;\n"}, "feat: base")
        head_src = base_src + '\nTEST_CASE("new fails", "[a]") {\n  CHECK(false);\n}\n'
        head_src = head_src.replace(
            'CHECK(true);\n}\n\nTEST_CASE("new', 'CHECK(true);\n  CHECK(1 == 1);\n}\n\nTEST_CASE("new'
        )
        head = self.commit({"tests/test_x.cpp": head_src, "src/a.cpp": "int b;\n"}, "fix(cpu): the bug")
        return base, head

    def test_select_runs_git_and_writes_cases_json(self) -> None:
        base, head = self.make_prs()
        out = Path(self.tmp.name, "cases.json")
        code, stdout = self.run_main(
            "select", "--repo", str(self.repo), "--base", base, "--head", head, "--out", str(out)
        )
        self.assertEqual((code, stdout.strip()), (0, "selected=2"))
        data = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual([c["name"] for c in data["cases"]], ["old, [one]", "new fails"])
        self.assertEqual(data["cases"][0]["spec"], r"old\, \[one\]")
        self.assertEqual(data["cases"][0]["binary"], "nibble_tests")
        self.assertEqual(data["unresolved"], [])
        code, stdout = self.run_main("specs", "--cases", str(out))
        self.assertEqual(stdout.splitlines(), ["0\tnibble_tests\told\\, \\[one\\]", "1\tnibble_tests\tnew fails"])

    def test_gate(self) -> None:
        base, head = self.make_prs()
        code, stdout = self.run_main(
            "gate", "--repo", str(self.repo), "--title", "chore: x", "--base", base, "--head", head
        )
        self.assertEqual(stdout.splitlines(), ["fix=true", "code=true"])  # the commit subject is a fix
        code, stdout = self.run_main(
            "gate", "--repo", str(self.repo), "--title", "docs: x", "--base", head, "--head", head
        )
        self.assertEqual(stdout.splitlines(), ["fix=false", "code=false"])

    def moved_base(self) -> tuple[str, str, str]:
        """(merge base, base branch tip, head) where the base branch gained a test case, a deletion and a src change."""
        src = 'TEST_CASE("keeps") {\n  CHECK(true);\n}\n\nTEST_CASE("master drops") {\n  CHECK(true);\n}\n'
        merge_base = self.commit({"tests/test_x.cpp": src, "src/a.cpp": "int a;\n"}, "feat: base")
        git(self.repo, "checkout", "-q", "-b", "pr")
        head_src = src.replace("CHECK(true);\n}\n\nTEST_CASE", "CHECK(true);\n  CHECK(1);\n}\n\nTEST_CASE", 1)
        head = self.commit({"tests/test_x.cpp": head_src}, "fix(docs): only a test")
        git(self.repo, "checkout", "-q", "master")
        master_src = src.replace('TEST_CASE("master drops") {\n  CHECK(true);\n}\n', 'TEST_CASE("master adds") {\n}\n')
        tip = self.commit({"tests/test_x.cpp": master_src, "src/a.cpp": "int moved;\n"}, "feat: master moved on")
        return merge_base, tip, head

    def test_select_diffs_against_the_merge_base_not_the_base_tip(self) -> None:
        merge_base, tip, head = self.moved_base()
        outs = []
        for base in (merge_base, tip):
            out = Path(self.tmp.name, f"{base[:7]}.json")
            self.assertEqual(
                self.run_main("select", "--repo", str(self.repo), "--base", base, "--head", head, "--out", str(out))[0],
                0,
            )
            outs.append(json.loads(out.read_text(encoding="utf-8")))
        self.assertEqual([c["name"] for c in outs[0]["cases"]], ["keeps"])
        self.assertEqual(outs[0]["unresolved"], [])
        self.assertEqual(outs[1], outs[0])  # master's added case and deleted case do not leak into the PR

    def test_gate_diffs_against_the_merge_base_not_the_base_tip(self) -> None:
        merge_base, tip, head = self.moved_base()
        for base in (merge_base, tip):
            _, stdout = self.run_main("gate", "--repo", str(self.repo), "--title", "x", "--base", base, "--head", head)
            self.assertEqual(
                stdout.splitlines(), ["fix=true", "code=false"], base
            )  # master's src change is not the PR's

    def test_select_reads_non_utf8_and_non_ascii_paths(self) -> None:
        base = self.commit({"tests/test_a.cpp": 'TEST_CASE("a") {\n}\n'}, "feat: base")
        Path(self.repo, "tests/test_\u00e9.cpp").write_text('TEST_CASE("accent") {\n}\n', encoding="utf-8")
        Path(self.repo, "tests/test_a.cpp").write_bytes(b'TEST_CASE("a") {\n  // latin-1 \xe9 and \xff\n}\n')
        head = self.commit({}, "fix: encodings")
        out = Path(self.tmp.name, "cases.json")
        code, _ = self.run_main("select", "--repo", str(self.repo), "--base", base, "--head", head, "--out", str(out))
        self.assertEqual(code, 0)
        names = sorted(c["name"] for c in json.loads(out.read_text(encoding="utf-8"))["cases"])
        self.assertEqual(names, ["a", "accent"])  # the git-quoted path "tests/test_\303\251.cpp" is not dropped

    def test_cases_json_with_an_unknown_binary_is_refused(self) -> None:
        bad = Path(self.tmp.name, "bad.json")
        case = {"file": "tests/test_a.cpp", "name": "a", "start": 1, "end": 2, "binary": "../../bin/sh"}
        bad.write_text(json.dumps({"cases": [case], "unresolved": []}), encoding="utf-8")
        self.assertEqual(self.run_main("specs", "--cases", str(bad))[0], 2)

    def test_select_refuses_an_option_looking_revision(self) -> None:
        code, _ = self.run_main("select", "--repo", str(self.repo), "--base=--output=x", "--head", "HEAD", "--out", "o")
        self.assertEqual(code, 2)

    def write_results(self, results: dict[str, str]) -> Path:
        directory = Path(self.tmp.name, "results")
        directory.mkdir(exist_ok=True)
        for name, text in results.items():
            (directory / name).write_text(text, encoding="utf-8")
        return directory

    def classify(self, results: dict[str, str], *extra: str, compiled: str = "true") -> tuple[int, str, str]:
        base, head = self.make_prs()
        cases = Path(self.tmp.name, "cases.json")
        self.run_main("select", "--repo", str(self.repo), "--base", base, "--head", head, "--out", str(cases))
        directory = self.write_results(results)
        summary_file = Path(self.tmp.name, "summary.md")
        code, stdout = self.run_main(
            "classify", "--cases", str(cases), "--results-dir", str(directory), "--base-compiled", compiled,
            "--summary", str(summary_file), *extra,
        )  # fmt: skip
        return code, stdout.strip(), summary_file.read_text(encoding="utf-8")

    def test_classify_proven(self) -> None:
        failing = _junit(FAILED.replace("fails, with [x] &amp; &lt;b> &quot;q&quot;", "new fails"))
        passing = _junit(PASSED.replace("passes", r"old, [one]"))
        code, stdout, text = self.classify({"0.xml": passing, "1.xml": failing})
        self.assertEqual((code, stdout), (0, "outcome=proven"))
        self.assertIn("- failed: new fails", text)
        self.assertIn("already passing", text)

    def test_classify_all_pass_does_not_reproduce(self) -> None:
        results = {
            "0.xml": _junit(PASSED.replace("passes", "old, [one]")),
            "1.xml": _junit(PASSED.replace("passes", "new fails")),
        }
        self.assertEqual(self.classify(results)[:2], (1, "outcome=does-not-reproduce"))

    def test_classify_no_results_is_not_runnable_and_exit_zero(self) -> None:
        self.assertEqual(self.classify({"0.xml": "not xml"})[:2], (0, "outcome=not-runnable"))

    def test_classify_base_does_not_compile(self) -> None:
        self.assertEqual(self.classify({}, compiled="false")[:2], (0, "outcome=proven-base-does-not-compile"))

    def test_classify_no_test_needs_the_label(self) -> None:
        base = self.commit({"tests/test_x.cpp": 'TEST_CASE("a") {}\n'}, "feat: base")
        head = self.commit({"src/a.cpp": "int c;\n"}, "fix: no test")
        cases = Path(self.tmp.name, "cases.json")
        self.run_main("select", "--repo", str(self.repo), "--base", base, "--head", head, "--out", str(cases))
        self.assertEqual(json.loads(cases.read_text(encoding="utf-8")), {"cases": [], "unresolved": []})
        results = self.write_results({})
        summary_file = Path(self.tmp.name, "s.md")
        args = ["classify", "--cases", str(cases), "--results-dir", str(results), "--base-compiled", "true"]
        args += ["--summary", str(summary_file)]
        self.assertEqual(self.run_main(*args), (1, "outcome=no-test\n"))
        self.assertEqual(self.run_main(*args, "--override"), (0, "outcome=no-test\n"))
        self.assertEqual(summary_file.read_text(encoding="utf-8").count("Bug-fix proof: no-test"), 2)  # appended

    def test_malformed_cases_json_is_an_error(self) -> None:
        bad = Path(self.tmp.name, "bad.json")
        bad.write_text('{"cases": [{"file": 1}], "unresolved": []}', encoding="utf-8")
        results = self.write_results({})
        code, _ = self.run_main(
            "classify", "--cases", str(bad), "--results-dir", str(results), "--base-compiled", "true",
            "--summary", str(Path(self.tmp.name, "s.md")),
        )  # fmt: skip
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
