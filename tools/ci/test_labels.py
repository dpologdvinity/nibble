"""Unit tests for tools/ci/labels.py: the label table is well formed and .github/labeler.yml mirrors AREA_PATHS.

Run with: python -m unittest discover -s tools/ci -p 'test_*.py'
labeler.yml is read with a tiny line parser (the repo installs no PyYAML); `sync` is checked against a recording
Runner, so no test touches `gh` or the network.
"""

from __future__ import annotations

import re
import unittest
from collections.abc import Sequence
from pathlib import Path

import labels

REPO_ROOT = Path(__file__).resolve().parents[2]
LABELER_YML = REPO_ROOT / ".github" / "labeler.yml"


def parse_labeler_yml(text: str) -> dict[str, tuple[str, ...]]:
    """Read the actions/labeler v5 file we generate: `<label>:`, `- changed-files:`, `- any-glob-to-any-file:`, globs.

    Only that exact shape is understood; any other non-comment line raises ValueError so format drift is caught.
    """
    result: dict[str, list[str]] = {}
    current: str | None = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" ") and line.endswith(":"):
            current = line[:-1]
            result[current] = []
        elif line.strip() in ("- changed-files:", "- any-glob-to-any-file:"):
            continue
        elif current is not None and (m := re.fullmatch(r"\s+- '([^']+)'", line)):
            result[current].append(m.group(1))
        else:
            raise ValueError(f"unexpected line in labeler.yml: {raw!r}")
    return {name: tuple(globs) for name, globs in result.items()}


class LabelTableTests(unittest.TestCase):
    def test_names_unique(self) -> None:
        names = [label.name for label in labels.LABELS]
        self.assertEqual(len(names), len(set(names)))

    def test_colors_are_six_hex(self) -> None:
        for label in labels.LABELS:
            self.assertRegex(label.color, r"^[0-9a-f]{6}$", label.name)

    def test_descriptions_fit_github_limit(self) -> None:
        for label in labels.LABELS:
            self.assertTrue(0 < len(label.description) <= 100, label.name)

    def test_every_area_label_exists(self) -> None:
        names = {label.name for label in labels.LABELS}
        for area in labels.AREA_PATHS:
            self.assertIn(area, names)

    def test_every_area_has_globs(self) -> None:
        for area, globs in labels.AREA_PATHS.items():
            self.assertTrue(globs, area)


class LabelerYmlTests(unittest.TestCase):
    def test_labeler_yml_matches(self) -> None:
        parsed = parse_labeler_yml(LABELER_YML.read_text(encoding="utf-8"))
        self.assertEqual(parsed, dict(labels.AREA_PATHS))

    def test_parser_rejects_unknown_shape(self) -> None:
        with self.assertRaises(ValueError):
            parse_labeler_yml("kernels:\n  - changed-files: [a]\n")

    def test_globs_name_something_that_exists(self) -> None:
        # A glob that matches nothing is a typo or a renamed directory; `**/*.md` and friends match by pattern.
        for area, globs in labels.AREA_PATHS.items():
            for glob in globs:
                self.assertTrue(list(REPO_ROOT.glob(glob)), f"{area}: {glob} matches no file")


class SyncTests(unittest.TestCase):
    def test_sync_creates_every_label_with_force(self) -> None:
        calls: list[list[str]] = []

        def runner(argv: Sequence[str], stdin: str | None) -> str:
            calls.append(list(argv))
            return ""

        labels.sync(runner)
        self.assertEqual(len(calls), len(labels.LABELS))
        for label, argv in zip(labels.LABELS, calls, strict=True):
            self.assertEqual(
                argv,
                [
                    "gh",
                    "label",
                    "create",
                    label.name,
                    "--color",
                    label.color,
                    "--description",
                    label.description,
                    "--force",
                ],
            )

    def test_sync_never_deletes(self) -> None:
        calls: list[list[str]] = []

        def runner(argv: Sequence[str], stdin: str | None) -> str:
            calls.append(list(argv))
            return ""

        labels.sync(runner)
        self.assertFalse(any("delete" in argv for argv in calls))


if __name__ == "__main__":
    unittest.main()
