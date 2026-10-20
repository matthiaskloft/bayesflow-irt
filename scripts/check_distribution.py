"""Inspect release archives and smoke-test the wheel outside the source checkout.

Run after ``python -m build``. Installs the wheel into a fresh virtual
environment (with the CPU torch backend, since ``[dev]`` and the core
dependencies pull none) and checks that the installed package, not the
checkout, is imported and reports the release version.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tarfile
import tempfile
import tomllib
import venv
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PARTS = {".git", "__pycache__", ".venv", ".pytest_cache", ".mypy_cache"}
REQUIRED = (
    "__init__.py",
    "py.typed",
    "_card_assets/dashboard.js",
    "_card_assets/dashboard.css",
)


def check_members(names: list[str], *, sdist: bool) -> None:
    """Reject local runtime content and require the core package files."""
    paths = [Path(name) for name in names]
    if sdist:
        paths = [Path(*path.parts[1:]) for path in paths if len(path.parts) > 1]
    for path in paths:
        if any(part in FORBIDDEN_PARTS for part in path.parts):
            raise ValueError(f"Local runtime content in distribution: {path}")
        if path.suffix == ".db":
            raise ValueError(f"Study database in distribution: {path}")
    prefix = "src/" if sdist else ""
    for member in REQUIRED:
        if Path(f"{prefix}bayesflow_irt/{member}") not in paths:
            raise ValueError(f"Missing packaged file: {member}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    wheels = list(args.dist.glob("*.whl"))
    sdists = list(args.dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("Expected exactly one wheel and one sdist")
    wheel = wheels[0].resolve()
    with zipfile.ZipFile(wheel) as archive:
        check_members(archive.namelist(), sdist=False)
    with tarfile.open(sdists[0]) as archive:
        check_members(archive.getnames(), sdist=True)
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    version = pyproject["project"]["version"]
    with tempfile.TemporaryDirectory(prefix="bayesflow-irt-wheel-check-") as tmp:
        work = Path(tmp)
        environment = work / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        scripts = environment / ("Scripts" if os.name == "nt" else "bin")
        python = scripts / ("python.exe" if os.name == "nt" else "python")
        env = {
            key: value
            for key, value in os.environ.items()
            if key not in {"PYTHONPATH", "PYTHONHOME"}
        }
        env["KERAS_BACKEND"] = "torch"

        def run(*arguments: str) -> None:
            subprocess.run([str(python), *arguments], cwd=work, env=env, check=True)

        run(
            "-m",
            "pip",
            "install",
            "torch",
            "--index-url",
            "https://download.pytorch.org/whl/cpu",
        )
        run("-m", "pip", "install", str(wheel))
        run("-m", "pip", "check")
        run(
            "-c",
            "import importlib.metadata as m, bayesflow_irt as b; "
            f"assert m.version('bayesflow-irt') == {version!r}; "
            f"assert b.__version__ == {version!r}; "
            f"assert not str(b.__file__).startswith({str(ROOT)!r}); "
            "print('Installed wheel:', b.__file__)",
        )
    print(f"Distribution checks passed for {version}")


if __name__ == "__main__":
    main()
