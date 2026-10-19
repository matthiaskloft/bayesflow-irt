"""Tests for :func:`bayesflow_irt.session_info`."""

import json
import subprocess
from pathlib import Path

import keras

import bayesflow_irt
from bayesflow_irt import metadata, session_info


def test_fields_and_json_round_trip():
    info = session_info()
    assert set(info) == {
        "timestamp", "git_commit", "git_dirty", "python", "platform",
        "keras_backend", "packages", "gpu",
    }
    assert info["keras_backend"] == keras.backend.backend()
    assert info["packages"]["keras"] == keras.__version__
    assert "bayesflow" in info["packages"]
    assert json.loads(json.dumps(info)) == info


def test_git_is_read_at_the_package_source():
    info = session_info()
    src = Path(bayesflow_irt.__file__).resolve().parent
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=src, capture_output=True, text=True
    )
    if head.returncode == 0:
        assert info["git_commit"] == head.stdout.strip()
        assert isinstance(info["git_dirty"], bool)


def test_untracked_package_reports_no_git(monkeypatch):
    """An installed copy inside another repository is not that repository."""
    real = metadata._git

    def fake(args, cwd):
        return None if args[0] == "ls-files" else real(args, cwd)

    monkeypatch.setattr(metadata, "_git", fake)
    info = session_info()
    assert info["git_commit"] is None and info["git_dirty"] is None


def test_non_torch_backend_has_no_gpu_block(monkeypatch):
    monkeypatch.setattr(keras.backend, "backend", lambda: "jax")
    info = session_info()
    assert info["keras_backend"] == "jax" and info["gpu"] is None


def test_no_git_gives_none(monkeypatch):
    monkeypatch.setattr(metadata, "_git", lambda args, cwd: None)
    info = session_info()
    assert info["git_commit"] is None
    assert info["git_dirty"] is None


def test_missing_packages_are_omitted(monkeypatch):
    real = metadata.version

    def fake(name):
        if name == "numba":
            raise metadata.PackageNotFoundError(name)
        return real(name)

    monkeypatch.setattr(metadata, "version", fake)
    assert "numba" not in session_info()["packages"]
