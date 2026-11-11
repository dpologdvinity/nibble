"""Unit tests for bench/plot.py's ABBA block selection (abba()).

Run with: python -m unittest discover -s bench -p 'test_*.py'
Rows mirror bench/results/abba.jsonl: one row per run, note = "abba block <id> <A1|B1|B2|A2>", where A runs are
llama.cpp and B runs are nibble.
"""

from __future__ import annotations

import unittest

import plot

MODEL = "qwen2.5-0.5b-instruct-q4_k_m.gguf"


def row(tool: str, commit: str, block: str, pos: str, median: float, date: str, load1: float = 0.5) -> plot.Row:
    """One abba.jsonl row for MODEL, pp512 at 8 threads."""
    return {
        "tool": tool,
        "commit": commit,
        "model": MODEL,
        "test": "pp512",
        "threads": 8,
        "median": median,
        "unit": "tok/s",
        "load1": load1,
        "date": date,
        "note": f"abba block {block} {pos}",
    }


def block(block_id: str, commit: str, nibble: float, llama: float, day: str, load1: float = 0.5) -> list[plot.Row]:
    """The four runs of one complete block (A1 B1 B2 A2), a minute apart on `day`."""
    return [
        row("llama.cpp", "b11715", block_id, "A1", llama, f"{day}T10:00:00Z", load1),
        row("nibble", commit, block_id, "B1", nibble, f"{day}T10:01:00Z"),
        row("nibble", commit, block_id, "B2", nibble, f"{day}T10:02:00Z"),
        row("llama.cpp", "b11715", block_id, "A2", llama, f"{day}T10:03:00Z"),
    ]


KEY = (MODEL, "pp512", 8)


class AbbaTest(unittest.TestCase):
    def test_abba_keeps_newest_commit(self) -> None:
        rows = block("old-1", "aaaaaaa", 10, 40, "2026-10-24") + block("new-1", "bbbbbbb", 20, 40, "2026-11-12")
        blocks = plot.abba({"abba": rows})[KEY]
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0][0], 2.0)

    def test_abba_keeps_all_blocks_of_one_commit(self) -> None:
        rows = block("2", "bbbbbbb", 20, 40, "2026-11-12") + block("1", "bbbbbbb", 10, 40, "2026-11-11")
        blocks = plot.abba({"abba": rows})[KEY]
        self.assertEqual([b[1] for b in blocks], [10.0, 20.0])  # date order

    def test_abba_marks_noisy_blocks(self) -> None:
        blocks = plot.abba({"abba": block("1", "bbbbbbb", 20, 40, "2026-11-12", load1=2.5)})[KEY]
        self.assertTrue(blocks[0][3])

    def test_abba_skips_incomplete_blocks(self) -> None:
        rows = [r for r in block("1", "bbbbbbb", 20, 40, "2026-11-12") if not r["note"].endswith("A2")]
        self.assertNotIn(KEY, plot.abba({"abba": rows}))


if __name__ == "__main__":
    unittest.main()
