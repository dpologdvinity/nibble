"""Unit tests for tools/ci/cosine.py: parsing the e2e `min cosine` WARN lines and comparing two runs.

Run with: python -m unittest discover -s tools/ci -p 'test_*.py'
The log fixture is real Catch2 v3 console output (captured from a WARN built like the one in
tests/test_e2e_golden.cpp), including its 79-column wrapping, which splits a line mid-token.
"""

from __future__ import annotations

import math
import unittest

import cosine
from cosine import Cosine, CosineChange

# Real ctest `--output-log` text: Catch2 prints `<file>:<line>: ` then `warning:` then the message, indented two
# spaces and wrapped at 79 columns (the first message wraps inside "12/12").
LOG = """\
Test project /work/build
    Start 31: e2e_golden.q4_0.capital
...............................................................................

test_e2e_golden.cpp:194:
warning:
  capital [q4_0]: min cosine over 12 positions = 0.999123, top-1 agreement 12/
  12

test_e2e_golden.cpp:205:
warning:
  capital: prefill last-row cosine = 0.999987

test_e2e_golden.cpp:194:
warning:
  chat [f16]: min cosine over 31 positions = 0.999877, top-1 agreement 31/31

test_e2e_golden.cpp:194:
warning:
  code [llama32-q4_k_m]: min cosine over 1234 positions = 0.97, top-1 agreement
  1234/1234

===============================================================================
All tests passed (1 assertion in 1 test case)
"""


def cos(key: str, value: float) -> Cosine:
    model, prompt = key.split("/")
    return Cosine(model, prompt, value, 10, 10)


class ParseTest(unittest.TestCase):
    def test_parses_ctest_log_fragment(self) -> None:
        got = cosine.parse_min_cosine(LOG)
        self.assertEqual(sorted(got), ["f16/chat", "llama32-q4_k_m/code", "q4_0/capital"])
        self.assertEqual(got["q4_0/capital"], Cosine("q4_0", "capital", 0.999123, 12, 12))
        self.assertEqual(got["f16/chat"].min_cos, 0.999877)
        self.assertEqual(got["llama32-q4_k_m/code"], Cosine("llama32-q4_k_m", "code", 0.97, 1234, 1234))

    def test_unwrapped_and_ctest_verbose_prefixed_lines(self) -> None:
        text = "31: warning:\n31:   chat [q8_0]: min cosine over 3 positions = 0.5, top-1 agreement 2/3\n"
        self.assertEqual(cosine.parse_min_cosine(text), {"q8_0/chat": Cosine("q8_0", "chat", 0.5, 3, 2)})

    def test_other_warnings_and_empty_input_yield_nothing(self) -> None:
        self.assertEqual(cosine.parse_min_cosine(""), {})
        self.assertEqual(cosine.parse_min_cosine("warning:\n  capital: prefill last-row cosine = 0.9\n"), {})

    def test_duplicate_key_conflict_raises(self) -> None:
        a = "warning:\n  chat [f16]: min cosine over 3 positions = 0.9, top-1 agreement 3/3\n"
        b = "warning:\n  chat [f16]: min cosine over 3 positions = 0.8, top-1 agreement 3/3\n"
        with self.assertRaisesRegex(ValueError, "f16/chat"):
            cosine.parse_min_cosine(a + "\n" + b)

    def test_duplicate_key_with_same_value_is_accepted(self) -> None:
        a = "warning:\n  chat [f16]: min cosine over 3 positions = 0.9, top-1 agreement 3/3\n"
        self.assertEqual(len(cosine.parse_min_cosine(a + "\n" + a)), 1)

    def test_nan_parses_as_nan_and_differs(self) -> None:
        text = "warning:\n  chat [f16]: min cosine over 3 positions = nan, top-1 agreement 0/3\n"
        got = cosine.parse_min_cosine(text)
        self.assertTrue(math.isnan(got["f16/chat"].min_cos))
        # nan != nan under IEEE, but an unchanged nan must not be reported as a change.
        self.assertEqual(cosine.diff_cosines(got, got), [])
        change = cosine.diff_cosines({"f16/chat": cos("f16/chat", 0.9)}, got)
        self.assertEqual(len(change), 1)
        self.assertEqual(change[0].old, 0.9)
        new = change[0].new
        assert new is not None
        self.assertTrue(math.isnan(new))

    def test_inconsistent_totals_raise(self) -> None:
        text = "warning:\n  chat [f16]: min cosine over 3 positions = 0.9, top-1 agreement 3/4\n"
        with self.assertRaises(ValueError):
            cosine.parse_min_cosine(text)


class DiffTest(unittest.TestCase):
    def test_diff_reports_added_removed_changed(self) -> None:
        old = {"a/x": cos("a/x", 0.9), "b/y": cos("b/y", 0.8), "c/z": cos("c/z", 0.7)}
        new = {"a/x": cos("a/x", 0.9), "b/y": cos("b/y", 0.81), "d/w": cos("d/w", 0.6)}
        self.assertEqual(
            cosine.diff_cosines(old, new),
            [
                CosineChange("b/y", 0.8, 0.81),
                CosineChange("c/z", 0.7, None),
                CosineChange("d/w", None, 0.6),
            ],
        )

    def test_identical_runs_have_no_changes(self) -> None:
        run = {"a/x": cos("a/x", 0.123456)}
        self.assertEqual(cosine.diff_cosines(run, dict(run)), [])


if __name__ == "__main__":
    unittest.main()
