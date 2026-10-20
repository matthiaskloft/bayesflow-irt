# Contributing to bayesflow-irt

Pull requests are accepted from collaborators only. If you are not a
collaborator, please open an [issue](https://github.com/matthiaskloft/bayesflow-irt/issues)
for bug reports, questions or feature ideas; pull requests from
non-collaborators will be closed. The rest of this guide is for collaborators.

## Setup

```bash
git clone https://github.com/matthiaskloft/bayesflow-irt.git
cd bayesflow-irt
pip install -e ".[dev,torch]"
```

Set the Keras backend before importing `keras`, `bayesflow`, or
`bayesflow_irt`.

PowerShell:

```powershell
$env:KERAS_BACKEND = "torch"
```

Bash:

```bash
export KERAS_BACKEND=torch
```

## Checks

```bash
pytest tests/ -v
ruff check src/
```

Default pytest runs exclude tests marked `slow`. Run them explicitly when
you touch training-heavy paths.

## Code expectations

- Follow the existing public API patterns and update `__all__` exports when
  adding public symbols.
- Add tests for behavior changes.
- Keep type hints and docstrings on public classes and functions.
- Update `README.md` when public behavior or workflow changes.

## Releases

The `Release` workflow publishes pushed `vX.Y.Z` tags to PyPI, then creates
a GitHub release with the changelog entry and the checked wheel and sdist.
It requires the tagged commit to be on `main`, reruns the test and lint
workflow, checks distribution metadata and contents, and smoke-tests the
installed wheel in a clean environment with the CPU torch backend.

One-time setup:

1. Create GitHub environments `pypi` and `testpypi`. Require a reviewer for
   `pypi` and restrict it to `v*` tags; allow only `main` for `testpypi`.
2. Register Trusted Publishers on [PyPI](https://pypi.org/manage/account/publishing/)
   and [TestPyPI](https://test.pypi.org/manage/account/publishing/) for project
   `bayesflow-irt`, owner `matthiaskloft`, repository `bayesflow-irt`, workflow
   `release.yml`, and the respective environment. No API token is needed.
3. After merging, rehearse with
   `gh workflow run release.yml --ref main -f destination=testpypi`.

For each stable release, prepare a PR updating `pyproject.toml`, the
`__version__` fallback in `src/bayesflow_irt/__init__.py`, and the version
and release date in `CITATION.cff`. Add a non-empty
`## [X.Y.Z] - YYYY-MM-DD` entry to `CHANGELOG.md` with the same date.
After merging and green CI, tag the merged commit and push `vX.Y.Z`.
Approve the `pypi` deployment when prompted, then verify a fresh install
and `pip check`.

If validation or CI fails, fix the cause before publishing. Never move a
published tag or replace an index upload; use a new version. If PyPI succeeds
but the GitHub release fails, rerun only the failed jobs. An existing draft
release must be completed manually with the run's notes and artifacts.

## References maintenance

- Keep root `references.md` in sync with public theory-facing behavior.
- For new or changed algorithmic claims, add an APA 7 entry with an
  OpenAlex verification link when available.

## Pull requests

1. Branch from `main`.
2. Keep the diff focused.
3. Run the relevant tests and lint checks.
4. Describe behavior changes, migration impact, and any remaining risks.

Merged branches are deleted. Before deleting an abandoned branch, tag its
last commit as `archive/failed/<name>` so the attempt stays reachable.

## Releases

Releases are semantic version tags (`v0.1.1`, `v0.2.0`), each with an entry
in `CHANGELOG.md`.

## Reporting issues

Use the [GitHub issue tracker](https://github.com/matthiaskloft/bayesflow-irt/issues).
