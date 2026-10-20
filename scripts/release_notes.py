"""Validate a release ref against package metadata and extract its changelog entry.

Run by ``.github/workflows/release.yml`` with ``RELEASE_REF`` and
``RELEASE_EVENT`` set from the GitHub context; writes ``release-notes.md``.
Pure stdlib, so it needs no install and no Keras backend.
"""

from __future__ import annotations

import ast
import os
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BRANCH = "main"


def _fallback_version(root: Path) -> str | None:
    """Return the ``__version__`` literal in ``__init__.py``'s except handler."""
    source = (root / "src" / "bayesflow_irt" / "__init__.py").read_text(
        encoding="utf-8"
    )
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Try):
            continue
        for handler in node.handlers:
            for statement in ast.walk(handler):
                if (
                    isinstance(statement, ast.Assign)
                    and any(
                        isinstance(t, ast.Name) and t.id == "__version__"
                        for t in statement.targets
                    )
                    and isinstance(statement.value, ast.Constant)
                ):
                    return str(statement.value.value)
    return None


def release_notes(root: Path, ref: str, event: str) -> str:
    """Check that *ref* may release the checked-out version; return its notes.

    Parameters
    ----------
    root : Path
        Repository root.
    ref : str
        Full Git ref of the run, e.g. ``refs/tags/v0.5.2``.
    event : str
        ``"push"`` (tag release) or ``"workflow_dispatch"`` (TestPyPI rehearsal).

    Returns
    -------
    str
        The changelog body for the version, without its heading.
    """
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    version = pyproject["project"]["version"]
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("Only stable X.Y.Z releases are supported")
    if event == "push":
        if ref != f"refs/tags/v{version}":
            raise ValueError(f"Release tag {ref} must match package version v{version}")
    elif event == "workflow_dispatch":
        if ref != f"refs/heads/{DEFAULT_BRANCH}":
            raise ValueError(f"TestPyPI rehearsals must run from {DEFAULT_BRANCH}")
    else:
        raise ValueError(f"Unsupported release event: {event}")
    if _fallback_version(root) != version:
        raise ValueError(
            "__version__ fallback in __init__.py must match package version"
        )
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    match = re.search(
        rf"(?ms)^## \[{re.escape(version)}\] - (\d{{4}}-\d{{2}}-\d{{2}})\n"
        r"(.*?)(?=^## |\Z)",
        changelog,
    )
    if not match or not match[2].strip():
        raise ValueError(
            f"CHANGELOG.md needs a non-empty '## [{version}] - YYYY-MM-DD' entry"
        )
    citation = (root / "CITATION.cff").read_text(encoding="utf-8")
    if not re.search(rf'(?m)^version: "?{re.escape(version)}"?$', citation):
        raise ValueError("CITATION.cff version must match package version")
    if not re.search(rf'(?m)^date-released: "?{re.escape(match[1])}"?$', citation):
        raise ValueError("CITATION.cff date-released must match the changelog date")
    return match[2].strip() + "\n"


if __name__ == "__main__":
    notes = release_notes(ROOT, os.environ["RELEASE_REF"], os.environ["RELEASE_EVENT"])
    (ROOT / "release-notes.md").write_text(notes, encoding="utf-8")
    print(notes)
