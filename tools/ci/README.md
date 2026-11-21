# tools/ci

Typed, stdlib-only Python helpers that the GitHub Actions workflows call (failure reporting, labels, PR reports and
so on). All GitHub access goes through the `GitHub` protocol in `gh.py`; `GhCli` implements it with `gh api`, and
`fakes.FakeGitHub` replaces it in tests. Modules import each other by bare name (`import gh`), the same way `bench/`
does: `unittest discover -s tools/ci` and mypy both put this directory on the module path (it has no
`__init__.py`), so keep module names unique across `bench/` and `tools/`. Test with
`python -m unittest discover -s tools/ci -p 'test_*.py'`; ruff and mypy cover it through `tools`.
