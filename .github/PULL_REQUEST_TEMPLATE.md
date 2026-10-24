## What and why

<!-- What changes, and why. Link issues. -->

## How it was tested

<!-- Commands run: unit tests, sanitizer / SIMD-off builds, end-to-end tests if numerics are involved. -->

## Checklist

- [ ] Conventional Commit messages, small focused commits
- [ ] Tests added or updated (a failing test first for bug fixes)
- [ ] `clang-format`, `clang-tidy`, `shellcheck`, `ruff`, `mypy` are clean (see CONTRIBUTING.md)
- [ ] Comments / docs updated (`README.md`, `ARCHITECTURE.md`)
- [ ] Numerics: end-to-end results unchanged, or the change is explained with before/after numbers
- [ ] Performance: before/after `nibble bench` numbers attached if relevant
