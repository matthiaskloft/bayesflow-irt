"""Tests for the model-card joint metrics (``card.joint_metrics``)."""

from types import SimpleNamespace

import numpy as np
import pytest

from bayesflow_irt.card import Cell, CellSample, joint_metrics
from bayesflow_irt.hpo import make_tcc_bias_metric
from bayesflow_irt.sbc import classical_reference_points, compute_tarp_coverage

PRIOR_SD = {"a": 0.5, "b": 1.0}  # log a ~ N(0, 0.5), b ~ N(0, 1)


def _sample(
    n=300, n_items=6, n_persons=200, draws=80, noise=0.3, seed=0, ignore_data=False
):
    """2PL cell whose draws are an exact posterior, or the prior.

    Each parameter is observed once with Gaussian noise, ``c = truth +
    N(0, noise)``, and the draws come from the conjugate posterior given
    ``c``. The responses are simulated from ``c``, not from the truth, so
    given ``c`` they carry no further information and the draws stay the
    exact posterior given everything the classical reference sees.
    """
    rng = np.random.default_rng(seed)
    truth, post = {}, {}
    for key, tau in PRIOR_SD.items():
        t = rng.normal(0, tau, (n, n_items))
        c = t + noise * rng.normal(size=t.shape)
        var = 1 / (1 / tau**2 + 1 / noise**2)
        if ignore_data:
            post[key] = rng.normal(0, tau, (n, draws, n_items))
        else:
            mean = var * c / noise**2
            post[key] = mean[:, None] + np.sqrt(var) * rng.normal(
                size=(n, draws, n_items)
            )
        truth[key] = (t, c)
    theta = rng.normal(size=(n, n_persons, 1))
    a_c, b_c = np.exp(truth["a"][1]), truth["b"][1]
    p = 1 / (1 + np.exp(-a_c[:, None] * (theta - b_c[:, None])))
    resp = (rng.uniform(size=p.shape) < p).astype(np.float32)
    return CellSample(
        cell=Cell(n_items, n_persons, True),
        seed=11,
        data={"a": np.exp(truth["a"][0]), "b": truth["b"][0], "responses": resp},
        draws={
            "a": np.exp(post["a"]).astype(np.float32),
            "b": post["b"].astype(np.float32),
        },
        item=np.zeros(n, int),
        person=np.zeros(n, int),
        threshold=None,
        retries=[],
        seconds=0.0,
        chunk_size=50,
    )


def test_matches_the_library_functions():
    s = _sample()
    out = joint_metrics(s, chunk_size=7)
    log_draws = np.concatenate([np.log(s.draws["a"]), s.draws["b"]], axis=-1).astype(
        float
    )
    truth = np.concatenate([np.log(s.data["a"]), s.data["b"]], axis=-1)
    inputs = SimpleNamespace(param_keys=("a", "b"), draws=log_draws, true_values=truth)
    for scale, key in (("z", "tcc_z_bias"), ("tcc", "tcc_bias")):
        ref = make_tcc_bias_metric(scale=scale)(inputs)[key]
        assert out["tcc"][key][0] == pytest.approx(ref, rel=1e-6)
    refs = classical_reference_points(s.data["responses"], ("a", "b"))
    ref = compute_tarp_coverage(log_draws, truth, reference_points=refs, resolution=39)
    assert out["tarp"]["classical"]["tarp_error"][0] == ref["tarp_error"]
    assert len(out["tarp"]["random"]["levels"]) == 39


def test_reproducible_and_independent_of_chunk_size():
    one = joint_metrics(_sample(), chunk_size=50)
    two = joint_metrics(_sample(), chunk_size=13)
    assert one["tarp"] == two["tarp"]
    for key in ("tcc_bias", "tcc_z_bias"):
        assert one["tcc"][key][0] == pytest.approx(two["tcc"][key][0], rel=1e-12)
    assert one["tarp"]["random"]["reference_mode"] == "random"
    assert one["tarp"]["classical"]["reference_mode"] == "provided"
    assert one["tarp"]["random"]["n_draws"] == 80


def test_calibrated_stays_near_the_floor_under_both_references():
    out = joint_metrics(_sample())
    for block in out["tarp"].values():
        err, se = block["tarp_error"]
        assert err < block["null_floor"] + 4 * se


def test_only_the_classical_reference_sees_a_posterior_that_ignores_data():
    """Lemos et al. (2023, Sec. 4.3)."""
    out = joint_metrics(_sample(ignore_data=True))
    random, classical = out["tarp"]["random"], out["tarp"]["classical"]
    assert random["tarp_error"][0] < random["null_floor"] + 4 * random["tarp_error"][1]
    err, se = classical["tarp_error"]
    assert err > classical["null_floor"] + 10 * se


def _too_narrow(sample, factor=0.7):
    """Shrink each dataset's draws about their mean: centred, too narrow."""
    for k, x in sample.draws.items():
        x = np.log(x) if k == "a" else x
        m = x.mean(axis=1, keepdims=True)
        x = m + factor * (x - m)
        sample.draws[k] = (np.exp(x) if k == "a" else x).astype(np.float32)
    return sample


def test_bootstrap_mcse_matches_spread_away_from_the_null():
    """Near the floor the bootstrap overstates (documented); here it must not."""
    reps = [
        joint_metrics(_too_narrow(_sample(n=150, draws=40, seed=s))) for s in range(30)
    ]
    for ref in ("random", "classical"):
        values = np.array([r["tarp"][ref]["tarp_error"] for r in reps])
        assert values[:, 0].mean() > 2 * reps[0]["tarp"][ref]["null_floor"]
        ratio = values[:, 1].mean() / values[:, 0].std(ddof=1)
        assert 0.6 < ratio < 1.5, (ref, ratio)


def test_null_floor_is_the_calibrated_mean():
    reps = [joint_metrics(_sample(n=150, draws=40, seed=s)) for s in range(30)]
    values = [r["tarp"]["random"]["tarp_error"][0] for r in reps]
    floor = reps[0]["tarp"]["random"]["null_floor"]
    assert abs(np.mean(values) - floor) < 3 * np.std(values, ddof=1) / np.sqrt(30)


def test_1pl_tarp_runs():
    s = _sample()
    s.draws.pop("a")
    s.data.pop("a")
    out = joint_metrics(s)
    assert out["tarp"]["classical"]["tarp_error"][1] > 0


def test_1pl_tcc_matches_library():
    s = _sample()
    s.draws.pop("a")
    s.data.pop("a")
    out = joint_metrics(s)
    inputs = SimpleNamespace(
        param_keys=("b",), draws=s.draws["b"].astype(float), true_values=s.data["b"]
    )
    ref = make_tcc_bias_metric(model="1pl", scale="tcc")(inputs)["tcc_bias"]
    assert out["tcc"]["tcc_bias"][0] == pytest.approx(ref, rel=1e-6)


def test_polytomous_is_omitted():
    s = _sample()
    s.draws = {"delta": np.zeros((3, 4, 6, 2), np.float32)}
    assert "PCM/GPCM" in joint_metrics(s)["omitted"]


def test_unexpected_keys_raise():
    s = _sample()
    s.draws["c"] = s.draws["b"]
    with pytest.raises(ValueError, match="Unexpected item parameters"):
        joint_metrics(s)


def test_zero_tcc_spread_raises():
    s = _sample()
    s.draws["b"][0] = s.draws["b"][0, :1]
    s.draws["a"][0] = s.draws["a"][0, :1]
    with pytest.raises(ValueError, match="zero spread"):
        joint_metrics(s)
