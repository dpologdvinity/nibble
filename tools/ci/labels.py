"""The repository's labels as code, and the path globs that map changed files to area labels.

`LABELS` is the source of truth for every label the CI automation uses (the changelog, triage issues and the
dashboard group by area). `AREA_PATHS` feeds .github/labeler.yml (actions/labeler); a test keeps the two in step.
`sync` creates or updates each label in the current repository with `gh label create --force`. It never deletes a
label, so labels added by hand survive.

usage: python tools/ci/labels.py sync        (needs `gh` authenticated for the repository; run from a clone)
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from typing import NamedTuple

from gh import Runner, subprocess_runner


class Label(NamedTuple):
    """One GitHub label: `color` is six lowercase hex digits without the `#`."""

    name: str
    color: str
    description: str


LABELS: tuple[Label, ...] = (
    # Area labels (applied by actions/labeler from AREA_PATHS).
    Label("kernels", "1d76db", "SIMD and scalar compute kernels"),
    Label("gguf", "5319e7", "GGUF parsing and loading"),
    Label("tokenizer", "0e8a16", "Tokenizer and chat templates"),
    Label("model", "fbca04", "Transformer, sampler, engine API"),
    Label("cli", "c5def5", "nibble command line"),
    Label("build", "006b75", "CMake and toolchains"),
    Label("ci", "0052cc", "Workflows and tools/ci"),
    Label("bench", "d4c5f9", "Benchmarks and plots"),
    Label("documentation", "0075ca", "Improvements or additions to documentation"),
    # Other labels (applied by workflows or by hand).
    Label("ci-failure", "b60205", "A scheduled job failed"),
    Label("agent", "7057ff", "Opened by the Claude agent"),
    Label("no-test-needed", "ededed", "Fix PR without a reproducing test (reason in the PR)"),
    Label("security", "ee0701", "Security-relevant"),
    Label("performance", "f9d0c4", "Speed or memory"),
)

# Area label -> globs (actions/labeler `any-glob-to-any-file`). Checked against the tree by test_labels.py.
AREA_PATHS: Mapping[str, tuple[str, ...]] = {
    "kernels": (
        "src/backend/**",
        "src/quant.*",
        "src/fp16.h",
        "src/tensor.*",
        "src/thread_pool.*",
    ),
    "gguf": ("src/gguf/**",),
    "tokenizer": (
        "src/tokenizer/**",
        "src/chat.*",
        "tests/tokenizer/**",
        "tests/test_tokenizer*.cpp",
        "tests/test_pretokenize.cpp",
        "tests/test_chat.cpp",
    ),
    "model": (
        "src/model/**",
        "include/**",
        "src/nibble.cpp",
        "src/sampler.*",
    ),
    "cli": ("cli/**",),
    "build": ("CMakeLists.txt", "Dockerfile"),
    "ci": (".github/**", "tools/ci/**"),
    "bench": ("bench/**", "src/bench/**", "tools/bench_*.sh"),
    "documentation": ("**/*.md", "docs/**"),
}


def sync(runner: Runner = subprocess_runner) -> None:
    """Create or update every label in LABELS in the current repository (`--force` updates an existing one)."""
    for label in LABELS:
        runner(
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
            None,
        )


def main(argv: Sequence[str]) -> int:
    """CLI entry point: `sync` is the only subcommand."""
    if list(argv) != ["sync"]:
        print("usage: labels.py sync", file=sys.stderr)
        return 2
    sync()
    print(f"synced {len(LABELS)} labels")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
