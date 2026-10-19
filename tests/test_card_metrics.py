"""Tests for the per-parameter model-card metrics (``card_metrics.py``)."""

import numpy as np
import pytest

from bayesflow_irt.card_metrics import CARD_LEVELS, jackknife_se, parameter_metrics
from bayesflow_irt.sbc import _compute_recovery_metrics, compute_calibration_error


def _calibrated(rng, n=400, s=200):
    """Conjugate normal: truth ~ N(0, 1), y ~ N(truth, 1), exact posterior."""
    truth = rng.normal(size=n)
    y = truth + rng.normal(size=n)
    draws = y[:, None] / 2 + np.sqrt(0.5) * rng.normal(size=(n, s))
    return draws, truth


def test_values_match_the_library_definitions():
    draws, truth = _calibrated(np.random.default_rng(0))
    m = parameter_metrics(draws, truth)
    rec = _compute_recovery_metrics(truth, draws.mean(axis=1))
    for k in ("corr", "nrmse", "nrmse_excess", "rmse"):
        assert np.isclose(m[k][0], rec[k])
    med = compute_calibration_error(draws, truth)["calibration_error"]
    mean = compute_calibration_error(draws, truth, aggregation="mean")
    assert np.isclose(m["calibration_error"][0], med)
    assert np.isclose(m["mean_calibration_error"][0], mean["mean_calibration_error"])
    assert {f"coverage_{lv}" for lv in CARD_LEVELS} <= set(m)
    curve = m.pop("sbc_ecdf")
    pairs = m.pop("pairs")
    cc = m.pop("calibration_curve")
    assert len(cc["levels"]) == len(cc["coverage"]) == len(cc["mcse"]) == 20
    # The mean calibration error is the mean |coverage - level| of the curve.
    dev = np.abs(np.array(cc["coverage"]) - cc["levels"])
    assert np.isclose(dev.mean(), m["mean_calibration_error"][0], atol=1e-6)
    assert set(pairs) == {"truth", "mean", "sd", "q025", "q975"}
    assert all(len(v) == len(truth) for v in pairs.values())
    assert all(len(v) == 2 for v in m.values())
    assert len(curve) == 39 and curve == sorted(curve)


def test_contraction_matches_the_variance_ratio():
    # Prior N(0, 1), posterior variance 0.5: contraction 0.5.
    draws, truth = _calibrated(np.random.default_rng(3), n=2000)
    value, se = parameter_metrics(draws, truth)["contraction"]
    assert abs(value - 0.5) < 4 * se + 0.01
    assert 0 < se < 0.05


def test_z_score_of_a_calibrated_posterior_is_standard():
    draws, truth = _calibrated(np.random.default_rng(4), n=2000)
    m = parameter_metrics(draws, truth)
    assert abs(m["z_mean"][0]) < 4 * m["z_mean"][1]
    assert abs(m["z_sd"][0] - 1) < 4 * m["z_sd"][1] + 0.02
    post, prior = m["posterior_variance"][0], m["truth_variance"][0]
    assert np.isclose(m["contraction"][0], 1 - post / prior)


def test_calibrated_posterior_scores_well():
    draws, truth = _calibrated(np.random.default_rng(1), n=2000)
    m = parameter_metrics(draws, truth)
    for lv in CARD_LEVELS:
        value, se = m[f"coverage_{lv}"]
        assert abs(value - lv) < 4 * se + 0.01
    assert abs(m["bias"][0]) < 4 * m["bias"][1]
    # Exact posterior mean: NRMSE = sqrt(1 - r^2), so the excess is ~0.
    assert abs(m["nrmse_excess"][0]) < 0.02
    # 95% width of N(., 0.5) is 2 * 1.96 * sqrt(0.5); at 200 draws the sample
    # quantiles fall about 2.5% short, so check it at 4000 draws.
    wide = parameter_metrics(*_calibrated(np.random.default_rng(6), n=200, s=4000))
    assert np.isclose(wide["width_0.95"][0], 2 * 1.959964 * np.sqrt(0.5), rtol=0.01)


def test_overconfident_posterior_is_caught():
    rng = np.random.default_rng(2)
    draws, truth = _calibrated(rng, n=1000)
    narrow = draws.mean(axis=1, keepdims=True) + 0.3 * (
        draws - draws.mean(axis=1, keepdims=True)
    )
    m = parameter_metrics(narrow, truth)
    assert m["coverage_0.95"][0] < 0.6
    assert m["mean_calibration_error"][0] > 0.2
    assert m["sbc_ecdf_max_dev"][0] > 0.1


def test_mcse_matches_the_spread_over_replications():
    """The reported MCSEs describe the actual sampling spread."""
    rng = np.random.default_rng(3)
    keys = (
        "bias",
        "rmse",
        "corr",
        "coverage_0.9",
        "width_0.95",
        "mean_calibration_error",
        "sbc_ecdf_max_dev",
    )
    est = {k: [] for k in keys}
    se = {k: [] for k in keys}
    for _ in range(150):
        m = parameter_metrics(*_calibrated(rng, n=300, s=100))
        for k in keys:
            est[k].append(m[k][0])
            se[k].append(m[k][1])
    for k in keys:
        ratio = np.mean(se[k]) / np.std(est[k], ddof=1)
        # The jackknife on a max/median of indicator means is rough, so
        # allow a wide band; the analytic MCSEs are tighter.
        assert 0.6 < ratio < 1.6, (k, ratio)


def test_jackknife_se_of_mean_equals_the_analytic_se():
    x = np.random.default_rng(4).normal(size=50)
    loo = (x.sum() - x) / (len(x) - 1)
    assert np.isclose(jackknife_se(loo), x.std(ddof=1) / np.sqrt(len(x)))


def test_degenerate_truth_gives_nan_not_an_error():
    rng = np.random.default_rng(5)
    m = parameter_metrics(rng.normal(size=(10, 50)), np.zeros(10))
    assert np.isnan(m["corr"][0]) and np.isnan(m["corr"][1])
    assert np.isfinite(m["bias"][0])


def test_input_validation():
    with pytest.raises(ValueError, match="n_datasets"):
        parameter_metrics(np.zeros(5), np.zeros(5))
    with pytest.raises(ValueError, match="at least 3"):
        parameter_metrics(np.zeros((2, 5)), np.zeros(2))
