"""Tests for the packaged ``WarmupCosine`` schedule."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np
import pytest

from bayesflow_irt import WarmupCosine


def _make(warmup=10, base=1e-2, total=110, min_lr=1e-4):
    cos = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=base,
        decay_steps=total - warmup,
        alpha=min_lr / base,
    )
    return WarmupCosine(warmup, base, cos)


def _lr(s, step):
    return float(keras.ops.convert_to_numpy(s(step)))


def test_values_at_warmup_and_cosine_points():
    s = _make()
    assert _lr(s, 0) == pytest.approx(0.0)
    assert _lr(s, 5) == pytest.approx(5e-3, rel=1e-5)
    assert _lr(s, 10) == pytest.approx(1e-2, rel=1e-5)
    mid = 1e-4 + (1e-2 - 1e-4) * 0.5 * (1 + np.cos(np.pi * 0.5))
    assert _lr(s, 60) == pytest.approx(mid, rel=1e-4)
    assert _lr(s, 110) == pytest.approx(1e-4, rel=1e-4)


def test_config_round_trip():
    s = _make()
    s2 = WarmupCosine.from_config(s.get_config())
    assert s2.warmup_steps == 10 and s2.base_lr == 1e-2
    for step in (0, 5, 10, 60, 110):
        assert _lr(s2, step) == pytest.approx(_lr(s, step))


def test_registered_identity_is_legacy_name():
    assert (
        keras.saving.get_registered_name(WarmupCosine)
        == "bayesflow_irt_scripts>WarmupCosine"
    )


def test_load_compiled_in_fresh_process(tmp_path):
    path = tmp_path / "m.keras"
    s = _make()
    model = keras.Sequential([keras.layers.Input((3,)), keras.layers.Dense(2)])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=s), loss="mse")
    model.save(path)
    expected = [_lr(s, k) for k in (0, 5, 10, 60, 110)]

    code = textwrap.dedent(
        f"""
        import os, sys, json
        os.environ["KERAS_BACKEND"] = "torch"
        import keras
        import bayesflow_irt
        assert not any("train_irt" in m for m in sys.modules)
        m = keras.saving.load_model(r"{path}", compile=True)
        sch = m.optimizer._learning_rate
        assert type(sch).__name__ == "WarmupCosine", type(sch)
        out = [float(keras.ops.convert_to_numpy(sch(k)))
               for k in (0, 5, 10, 60, 110)]
        print(json.dumps([sch.warmup_steps, sch.base_lr, out]))
        """
    )
    src = str(Path(__file__).resolve().parents[1] / "src")
    env = {
        **os.environ,
        "PYTHONPATH": src + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    res = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(tmp_path),
    )
    assert res.returncode == 0, res.stderr
    import json

    warm, base, vals = json.loads(res.stdout.strip().splitlines()[-1])
    assert warm == 10 and base == pytest.approx(1e-2)
    assert vals == pytest.approx(expected, rel=1e-5)
