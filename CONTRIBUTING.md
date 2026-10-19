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
