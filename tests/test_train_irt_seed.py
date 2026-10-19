"""``scripts/train_irt.py`` applies the config's seed to network initialisation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from bayesflow_irt.compat import weights_digest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

SMOKE = SCRIPTS / "configs" / "smoke_1pl.json"


def _initial_digest(monkeypatch, tmp_path, cfg_path) -> str:
    train_irt = pytest.importorskip("train_irt")
    seen: list[str] = []
    real_build = train_irt.build_approximator

    def build(cfg, adapter):
        approximator = real_build(cfg, adapter)
        simulator = train_irt.build_simulator_factory(cfg)[0](
            n_items=tuple(cfg["n_items"]), n_persons=tuple(cfg["n_persons"])
        )
        approximator.build_from_data(approximator.adapter(simulator.sample((2,))))
        seen.append(weights_digest(approximator))
        return approximator

    monkeypatch.setattr(train_irt, "build_approximator", build)
    train_irt.main(
        [
            "--config",
            str(cfg_path),
            "--out-dir",
            str(tmp_path / "out"),
            "--no-probe",
            "--dry-run",
        ]
    )
    return seen[0]


def test_same_seed_gives_identical_initial_weights(monkeypatch, tmp_path):
    first = _initial_digest(monkeypatch, tmp_path / "a", SMOKE)
    second = _initial_digest(monkeypatch, tmp_path / "b", SMOKE)
    assert first == second


def test_different_seed_gives_different_initial_weights(monkeypatch, tmp_path):
    cfg = json.loads(SMOKE.read_text(encoding="utf-8"))
    cfg["seed"] = int(cfg["seed"]) + 1
    other = tmp_path / "other.json"
    other.write_text(json.dumps(cfg), encoding="utf-8")
    assert _initial_digest(monkeypatch, tmp_path / "a", SMOKE) != _initial_digest(
        monkeypatch, tmp_path / "b", other
    )
