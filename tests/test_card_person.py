"""Tests for the model-card person metrics (``card.person_metrics``)."""

import json
from pathlib import Path

import numpy as np
import pytest
from test_card_sampling import SPEC_2PL, FakeApproximator

from bayesflow_irt.card import (
    Cell,
    EvaluationProtocol,
    _cluster_mean,
    _item_curve,
    person_metrics,
    sample_cell,
    theta_bin_count,
)
from bayesflow_irt.metadata import MetadataTracker
from bayesflow_irt.scoring import _log_likelihood_gpcm

SIDECARS = Path(__file__).resolve().parent / "fixtures" / "sidecars"


def _protocol(**kw):
    kw.setdefault("datasets_per_cell", 200)
    kw.setdefault("posterior_draws", 40)
    kw.setdefault("persons_scored", 30)
    return EvaluationProtocol.preset("development", **kw)


def _exact(spec=SPEC_2PL, cell=Cell(10, 50, True), **kw):
    """Item draws at the truth: theta posteriors are then exact."""
    proto = _protocol(**kw)
    return sample_cell(FakeApproximator(sd=0.0), spec, cell, proto), proto


def test_exact_items_give_calibrated_theta_and_zero_rimse():
    s, proto = _exact()
    out = person_metrics(s, proto)
    cov, se = out["theta"]["coverage_0.95"]
    assert abs(cov - 0.95) < 4 * se + 0.01
    bias, se = out["theta"]["bias"]
    assert abs(bias) < 4 * se
    assert out["rimse"]["mean"][0] == pytest.approx(0.0, abs=1e-9)
    diff, se = out["reliability"]["difference"]["mean"]
    assert abs(diff) < 4 * se + 0.01
    assert 0.5 < out["reliability"]["posterior_variance"]["mean"][0] < 1
    assert out["reliability"]["persons_scored"] == 30
    assert json.loads(json.dumps(out, allow_nan=False))


def test_conditional_bins_hold_every_scored_person():
    s, proto = _exact()
    bins = person_metrics(s, proto)["conditional"]
    assert len(bins) == theta_bin_count(proto) == 10
    assert sum(b["n"] for b in bins) == 200 * 30
    assert bins[0]["edges"][0] is None and bins[-1]["edges"][1] is None
    assert bins[4]["edges"][1] == pytest.approx(0.0, abs=1e-12)  # median
    assert all(b["clusters"] >= 30 for b in bins)


def test_exact_posteriors_cover_on_average_and_shrink_in_the_tails():
    # Conditional on the true theta, a correct posterior under-covers in
    # the tails (Feuerstahler, 2018); pooled over bins it is nominal. 400
    # draws: quantile intervals from 40 draws cover only about 0.89.
    s, proto = _exact(datasets_per_cell=60, posterior_draws=400)
    bins = person_metrics(s, proto)["conditional"]
    cov = np.array([b["coverage_0.95"][0] for b in bins])
    n = np.array([b["n"] for b in bins])
    assert abs((cov * n).sum() / n.sum() - 0.95) < 0.02
    mid = len(bins) // 2
    assert cov[0] < cov[mid] and cov[-1] < cov[mid]


def test_bin_count_follows_the_smallest_cell():
    assert theta_bin_count(_protocol(datasets_per_cell=20)) == 2  # 20 x 10 / 100
    assert theta_bin_count(_protocol(datasets_per_cell=3, persons_scored=1)) == 1
    assert theta_bin_count(EvaluationProtocol.preset("validation")) == 10


def test_cluster_mean_matches_the_sandwich_formula():
    """Cameron & Miller (2015), Eqs. 10-12 with X a column of ones, K = 1."""
    rng = np.random.default_rng(0)
    cluster = rng.integers(0, 12, size=200)
    y = rng.normal(size=200) + cluster * 0.3  # correlated within cluster
    mean, se, g = _cluster_mean(y, cluster)
    u = y - y.mean()
    b = sum(u[cluster == c].sum() ** 2 for c in np.unique(cluster))
    n = y.size
    c = g / (g - 1) * (n - 1) / (n - 1)
    assert mean == pytest.approx(y.mean())
    assert se == pytest.approx(np.sqrt(c * b / n**2))
    assert g == np.unique(cluster).size


def test_cluster_mean_reduces_to_the_independent_se():
    y = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    _, se, g = _cluster_mean(y, np.arange(6))
    p = y.mean()
    assert se == pytest.approx(np.sqrt(p * (1 - p) / 6 * g / (g - 1)))
    assert _cluster_mean(y, np.zeros(6))[1] is None


def test_noisy_items_raise_rimse():
    proto = _protocol(datasets_per_cell=50)
    s = sample_cell(FakeApproximator(sd=0.3), SPEC_2PL, Cell(10, 50, True), proto)
    assert person_metrics(s, proto)["rimse"]["mean"][0] > 0.3


def test_reproducible_and_fewer_persons_than_requested():
    s, proto = _exact(cell=Cell(5, 10, True), datasets_per_cell=20)
    one, two = person_metrics(s, proto), person_metrics(s, proto)
    assert one == two
    assert one["reliability"]["persons_scored"] == 10


def test_polytomous_curve_matches_the_scoring_likelihood():
    theta = np.linspace(-3, 3, 7)
    a, delta = np.array([1.4]), np.array([[-0.5, 0.8]])
    probs = np.exp(np.stack([
        _log_likelihood_gpcm(theta, np.array([k], float), a, delta)
        for k in range(3)
    ], axis=-1))
    expected = (probs * np.arange(3)).sum(axis=-1) / 2
    got = _item_curve(theta, a, delta, True)[0]
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_pcm_runs():
    path = SIDECARS / "pcm_equivariant_metadata.json"
    md = MetadataTracker.load(path).to_dict()
    proto = _protocol(datasets_per_cell=30, persons_scored=20)
    s = sample_cell(FakeApproximator(sd=0.0), md["model_spec"], Cell(6, 30, True),
                    proto, trained_at=md["provenance"]["timestamp"])
    out = person_metrics(s, proto)
    assert out["rimse"]["item_curve"] == "expected score / (K-1)"
    assert out["rimse"]["mean"][0] == pytest.approx(0.0, abs=1e-9)


def test_information_reliability_matches_direct_quadrature():
    """Lee (2026, Eqs. A.3-A.5), recomputed from the generating items."""
    s, proto = _exact(datasets_per_cell=20)
    out = person_metrics(s, proto)["reliability"]["information"]
    q = np.linspace(-5, 5, 101)
    w = np.exp(-0.5 * q**2)
    values = []
    for k in range(20):
        a, b = s.data["a"][k], s.data["b"][k]
        p = 1 / (1 + np.exp(-a[:, None] * (q - b[:, None])))
        info = (a[:, None] ** 2 * p * (1 - p)).sum(axis=0)
        msem = (w / info).sum() / w.sum()
        values.append(1 / (1 + msem))
    assert out["mean"][0] == pytest.approx(np.mean(values), rel=1e-10)


def test_information_reliability_scales_with_the_theta_prior():
    """Nodes and weights follow a non-standard prior (0.5, 2)."""
    s, proto = _exact(datasets_per_cell=20)
    out = person_metrics(s, proto, theta_prior=(0.5, 2.0))["reliability"]
    z = np.linspace(-5, 5, 101)
    q, w = 0.5 + 2.0 * z, np.exp(-0.5 * z**2)
    values = []
    for k in range(20):
        a, b = s.data["a"][k], s.data["b"][k]
        p = 1 / (1 + np.exp(-a[:, None] * (q - b[:, None])))
        info = (a[:, None] ** 2 * p * (1 - p)).sum(axis=0)
        values.append(4.0 / (4.0 + (w / info).sum() / w.sum()))
    assert out["information"]["mean"][0] == pytest.approx(np.mean(values), rel=1e-10)


def test_two_scored_persons_give_null_correlation():
    s, proto = _exact(datasets_per_cell=10, persons_scored=2)
    rel = person_metrics(s, proto)["reliability"]
    assert rel["squared_correlation"]["mean"] == (None, None)
    assert rel["posterior_variance"]["mean"][0] is not None


def test_conditional_se_is_in_the_bins_for_dichotomous_only():
    s, proto = _exact(datasets_per_cell=50)
    bins = person_metrics(s, proto)["conditional"]
    middle = bins[2]
    assert middle["conditional_se"][0] > 0
    path = SIDECARS / "pcm_equivariant_metadata.json"
    md = MetadataTracker.load(path).to_dict()
    proto = _protocol(datasets_per_cell=20, persons_scored=10)
    s = sample_cell(FakeApproximator(sd=0.0), md["model_spec"], Cell(6, 30, True),
                    proto, trained_at=md["provenance"]["timestamp"])
    out = person_metrics(s, proto)
    assert "omitted" in out["reliability"]["information"]
    assert all("conditional_se" not in b for b in out["conditional"])


def test_theta_prior_defaults_to_the_normal_sampler_defaults():
    """A normal theta prior record without loc/scale is valid for
    build_simulator and resolves to the sampler defaults (0, 1)."""
    from bayesflow_irt.card import _theta_prior

    assert _theta_prior({"priors": {"theta": {"name": "normal"}}}) == (0.0, 1.0)
    assert _theta_prior(
        {"priors": {"theta": {"name": "normal", "loc": 0.5}}}
    ) == (0.5, 1.0)
