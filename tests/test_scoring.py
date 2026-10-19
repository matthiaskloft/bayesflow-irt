"""Tests for the person scoring module."""

import os
import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import bayesflow as bf
import numpy as np
import pytest
from numpy.testing import assert_allclose

from bayesflow_irt import (
    EquivariantIRTApproximator,
    OneHotEncode,
    PersonScore,
    PersonScorer,
    Prior,
    make_2pl_adapter,
    make_2pl_simulator,
    make_gpcm_adapter,
    make_gpcm_simulator,
    make_pcm_adapter,
    make_pcm_simulator,
)
from bayesflow_irt.networks import MatrixEquivariantIRTSummary
from bayesflow_irt.scoring import (
    _aggregate_grid_posterior,
    _detect_model_type,
    _gauss_hermite_grid,
    _log_likelihood_1pl,
    _log_likelihood_2pl,
    _log_likelihood_all_persons,
    _log_likelihood_gpcm,
    _log_likelihood_pcm,
    _log_likelihood_polytomous_chunk,
    _uniform_grid,
    _validate_posteriors,
    _weighted_hdi_grid,
    _weighted_quantile_grid,
)

# Standard test priors
_PRIOR_A = Prior("lognormal", loc=0, scale=0.5)
_PRIOR_B = Prior("normal", loc=0, scale=2.0)
_PRIOR_THETA = Prior("normal", loc=0, scale=1.0)
_PRIOR_DELTA = Prior("normal", loc=0, scale=2.0)


# ---------------------------------------------------------------------------
# Unit tests (no approximator needed)
# ---------------------------------------------------------------------------


class TestLogLikelihood:
    """Test log-likelihood functions against manually computed values."""

    def test_log_likelihood_1pl_symmetric(self):
        """1PL: symmetric responses (equal 0s and 1s) with b=0 -> EAP ≈ 0."""
        # Symmetric item difficulties centered at 0
        b = np.array([-1.0, 0.0, 1.0])
        # Symmetric responses (two 1s, one 0)
        responses = np.array([1.0, 1.0, 0.0])

        # Create quadrature grid
        theta_grid, log_prior = _gauss_hermite_grid(31)
        log_likelihood = _log_likelihood_1pl(theta_grid, responses, b)

        # Compute posterior
        log_post = log_prior + log_likelihood
        log_post -= np.logaddexp.reduce(log_post)
        posterior = np.exp(log_post)

        # EAP should be positive but not too large due to symmetry
        eap = np.sum(posterior * theta_grid)
        assert abs(eap) < 0.5, f"EAP = {eap}, expected near 0"

    def test_log_likelihood_2pl_symmetric(self):
        """2PL: symmetric responses with a=1, b=0 -> EAP ≈ 0."""
        a = np.array([1.0, 1.0, 1.0])
        b = np.array([-0.5, 0.0, 0.5])
        responses = np.array([1.0, 1.0, 0.0])

        theta_grid, log_prior = _gauss_hermite_grid(31)
        log_likelihood = _log_likelihood_2pl(theta_grid, responses, a, b)

        log_post = log_prior + log_likelihood
        log_post -= np.logaddexp.reduce(log_post)
        posterior = np.exp(log_post)

        eap = np.sum(posterior * theta_grid)
        assert abs(eap) < 0.5, f"EAP = {eap}, expected near 0"

    def test_log_likelihood_pcm_structure(self):
        """PCM: verify cumulative logit structure."""
        # 3-category PCM (K=3, so K-1=2 step parameters)
        delta = np.array(
            [
                [-1.0, 0.5],  # item 0
                [0.0, 1.0],  # item 1
            ]
        )
        # Person with theta=0 should prefer middle category
        theta = np.array([0.0])
        responses = np.array([1, 1])  # Middle category (index 1)

        log_likelihood = _log_likelihood_pcm(theta, responses, delta)
        assert log_likelihood.shape == (1,), (
            f"Expected shape (1,), got {log_likelihood.shape}"
        )
        # Likelihood should be positive (well-defined)
        assert not np.any(np.isnan(log_likelihood)), "Log-likelihood contains NaN"
        assert not np.any(np.isinf(log_likelihood)), "Log-likelihood contains Inf"

    def test_log_likelihood_gpcm_structure(self):
        """GPCM: verify structure with discrimination."""
        a = np.array([1.0, 1.5])
        delta = np.array(
            [
                [-1.0, 0.5],
                [0.0, 1.0],
            ]
        )
        theta = np.array([0.0])
        responses = np.array([1, 1])

        log_likelihood = _log_likelihood_gpcm(theta, responses, a, delta)
        assert log_likelihood.shape == (1,), (
            f"Expected shape (1,), got {log_likelihood.shape}"
        )
        assert not np.any(np.isnan(log_likelihood)), "Log-likelihood contains NaN"
        assert not np.any(np.isinf(log_likelihood)), "Log-likelihood contains Inf"


class TestQuadrature:
    """Test quadrature grid setup."""

    def test_gauss_hermite_weights_sum_to_one(self):
        """Gauss-Hermite weights should sum to 1 after normalization."""
        theta_grid, log_weights = _gauss_hermite_grid(31)
        weights = np.exp(log_weights)
        assert_allclose(weights.sum(), 1.0, rtol=1e-6)

    def test_gauss_hermite_grid_coverage(self):
        """Grid should cover approximately [-6, 6] for N(0,1)."""
        theta_grid, _ = _gauss_hermite_grid(31)
        assert theta_grid[0] < -5.0, "Grid should extend below -5"
        assert theta_grid[-1] > 5.0, "Grid should extend above +5"

    def test_uniform_grid_weights_sum_to_one(self):
        """Uniform grid weights should sum to 1."""
        theta_grid, log_weights = _uniform_grid(41, bounds=(-6.0, 6.0))
        weights = np.exp(log_weights)
        assert_allclose(weights.sum(), 1.0, rtol=1e-6)

    def test_uniform_grid_bounds(self):
        """Uniform grid should respect specified bounds."""
        theta_grid, _ = _uniform_grid(41, bounds=(-4.0, 4.0))
        assert theta_grid[0] == -4.0
        assert theta_grid[-1] == 4.0


class TestModelDetection:
    """Test model type detection from posterior keys."""

    def test_detect_1pl(self):
        posteriors = {"b": np.zeros((10, 5))}
        assert _detect_model_type(posteriors) == "1PL"

    def test_detect_2pl(self):
        posteriors = {"a": np.zeros((10, 5)), "b": np.zeros((10, 5))}
        assert _detect_model_type(posteriors) == "2PL"

    def test_detect_pcm(self):
        posteriors = {"delta": np.zeros((10, 5, 3))}
        assert _detect_model_type(posteriors) == "PCM"

    def test_detect_gpcm(self):
        posteriors = {
            "a": np.zeros((10, 5)),
            "delta": np.zeros((10, 5, 3)),
        }
        assert _detect_model_type(posteriors) == "GPCM"

    def test_detect_filters_metadata(self):
        """Should filter out metadata keys like d, d_delta, sigma_a."""
        posteriors = {
            "a": np.zeros((10, 5)),
            "b": np.zeros((10, 5)),
            "d": np.zeros((10, 5)),  # Derived key, should be filtered
            "sigma_a": np.zeros((10, 5)),  # Meta key, should be filtered
            "item_ids": np.arange(5),
        }
        assert _detect_model_type(posteriors) == "2PL"

    def test_detect_unrecognized(self):
        posteriors = {"unknown": np.zeros((10, 5))}
        with pytest.raises(ValueError, match="Unrecognized posterior keys"):
            _detect_model_type(posteriors)


class TestPosteriorsValidation:
    """Test posterior sample validation."""

    def test_validate_1pl_correct(self):
        posteriors = {"b": np.random.randn(100, 10)}
        n_samples, n_items = _validate_posteriors(posteriors, "1PL")
        assert n_samples == 100
        assert n_items == 10

    def test_validate_2pl_correct(self):
        posteriors = {
            "a": np.random.rand(100, 10) + 0.5,
            "b": np.random.randn(100, 10),
        }
        n_samples, n_items = _validate_posteriors(posteriors, "2PL")
        assert n_samples == 100
        assert n_items == 10

    def test_validate_pcm_correct(self):
        # PCM posteriors are (S, I, K-1) - 3D without batch dimension
        posteriors = {"delta": np.random.randn(100, 10, 4)}
        n_samples, n_items = _validate_posteriors(posteriors, "PCM")
        assert n_samples == 100
        assert n_items == 10

    def test_validate_missing_key(self):
        posteriors = {"a": np.random.randn(100, 10)}
        with pytest.raises(ValueError, match="Missing required keys"):
            _validate_posteriors(posteriors, "2PL")

    def test_validate_shape_mismatch(self):
        posteriors = {
            "a": np.random.randn(100, 10),
            "b": np.random.randn(100, 15),  # Different item count
        }
        with pytest.raises(ValueError, match="Inconsistent item count"):
            _validate_posteriors(posteriors, "2PL")

    def test_validate_nan_inf(self):
        posteriors = {
            "a": np.random.randn(100, 10),
            "b": np.random.randn(100, 10),
        }
        posteriors["a"][0, 0] = np.nan
        with pytest.raises(ValueError, match="Non-finite values"):
            _validate_posteriors(posteriors, "2PL")

    def test_validate_2pl_rejects_nonpositive_discrimination(self):
        posteriors = {
            "a": np.ones((10, 5)),
            "b": np.zeros((10, 5)),
        }
        posteriors["a"][0, 0] = 0.0
        with pytest.raises(ValueError, match="strictly positive"):
            _validate_posteriors(posteriors, "2PL")

    def test_validate_batch_dimension_squeezed(self):
        """Should handle batch dimension correctly (squeeze to 2D/3D)."""
        posteriors = {
            "b": np.random.randn(1, 100, 10),  # (batch, S, I)
        }
        n_samples, n_items = _validate_posteriors(posteriors, "1PL")
        assert n_samples == 100, f"Expected 100 samples, got {n_samples}"
        assert n_items == 10, f"Expected 10 items, got {n_items}"


class TestGridPosteriorAggregation:
    """Test grid-posterior summary aggregation."""

    def test_map_uses_discrete_posterior_mass(self):
        theta_grid = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
        mixture_weights = np.array([[0.2, 0.3, 0.5]], dtype=np.float64)
        misleading_log_density = np.array([[0.0, 10.0, 0.0]], dtype=np.float64)

        result = _aggregate_grid_posterior(
            theta_grid,
            mixture_weights,
            misleading_log_density,
        )

        assert result["map"][0] == pytest.approx(1.0)


class TestCellUniformQuantiles:
    """Quantiles treat each grid mass as uniform on its cell."""

    def test_two_cell_grid_known_answer(self):
        grid = np.array([0.0, 1.0])
        weights = np.array([[0.25, 0.75]])
        probs = np.array([0.0, 0.25, 0.5, 0.625, 1.0])
        out = _weighted_quantile_grid(grid, weights, probs)
        # Edges 0, 0.5, 1 with CDF 0, 0.25, 1.
        assert_allclose(out[0], [0.0, 0.5, 0.5 + 0.5 / 3, 0.75, 1.0])

    def test_single_occupied_cell(self):
        grid = np.array([0.0, 1.0, 2.0])
        weights = np.array([[0.0, 1.0, 0.0]])
        out = _weighted_quantile_grid(grid, weights, np.array([0.25, 0.5, 0.75]))
        # All mass uniform on [0.5, 1.5].
        assert_allclose(out[0], [0.75, 1.0, 1.25])

    def test_per_person_grids(self):
        grids = np.array([[0.0, 1.0, 2.0], [10.0, 12.0, 14.0]])
        weights = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        out = _weighted_quantile_grid(grids, weights, np.array([0.5]))
        assert_allclose(out[:, 0], [1.0, 12.0])

    def test_hdi_of_uniform_cell_is_centered(self):
        grid = np.array([0.0, 1.0, 2.0])
        weights = np.array([[0.0, 1.0, 0.0]])
        lower, upper = _weighted_hdi_grid(grid, weights, 0.5)
        assert_allclose(upper - lower, 0.5, atol=1e-9)
        assert 0.5 - 1e-9 <= lower[0] and upper[0] <= 1.5 + 1e-9


class TestAdaptiveGrid:
    """Two-pass per-person grid."""

    @staticmethod
    def _simulate(n_items=50, n_persons=40, seed=3):
        rng = np.random.default_rng(seed)
        a = np.exp(rng.normal(0.0, 0.5, n_items))
        b = rng.normal(0.0, 2.0, n_items)
        theta = rng.normal(0.0, 1.0, n_persons)
        p = 1.0 / (1.0 + np.exp(-a * (theta[:, None] - b)))
        x = (rng.random(p.shape) < p).astype(float)
        return a, b, theta, x

    def test_matches_very_fine_fixed_grid_at_long_test(self):
        a, b, _, x = self._simulate()
        scorer = PersonScorer(
            None, item_posteriors={"a": a[None], "b": b[None]}, model_type="2PL"
        )
        adaptive = scorer.score(x, n_plausible_values=20)
        fine = scorer.score(
            x, theta_grid_size=4001, adaptive_grid=False, n_plausible_values=20
        )
        assert_allclose(adaptive.mean, fine.mean, atol=2e-3)
        assert_allclose(adaptive.sd, fine.sd, atol=2e-3)
        assert_allclose(adaptive.median, fine.median, atol=1e-3)
        assert_allclose(adaptive.eti_lower, fine.eti_lower, atol=1.5e-2)
        assert_allclose(adaptive.plausible_values, fine.plausible_values, atol=1.5e-2)

    def test_matches_fixed_grid_with_item_draws_and_missing(self):
        a, b, _, x = self._simulate(n_items=30, n_persons=15)
        rng = np.random.default_rng(0)
        x[rng.random(x.shape) < 0.2] = np.nan
        draws = {
            "a": a * np.exp(rng.normal(0, 0.05, (20, a.size))),
            "b": b + rng.normal(0, 0.1, (20, b.size)),
        }
        scorer = PersonScorer(None, item_posteriors=draws, model_type="2PL")
        adaptive = scorer.score(x)
        fine = scorer.score(x, theta_grid_size=4001, adaptive_grid=False)
        assert_allclose(adaptive.mean, fine.mean, atol=2e-3)
        assert_allclose(adaptive.sd, fine.sd, atol=2e-3)

    def test_pcm_matches_fixed_grid(self):
        rng = np.random.default_rng(5)
        delta = np.sort(rng.normal(0, 1.0, (20, 3)), axis=-1)
        responses = rng.integers(0, 4, size=(10, 20)).astype(float)
        scorer = PersonScorer(
            None, item_posteriors={"delta": delta[None]}, model_type="PCM"
        )
        adaptive = scorer.score(responses)
        fine = scorer.score(responses, theta_grid_size=4001, adaptive_grid=False)
        assert_allclose(adaptive.mean, fine.mean, atol=2e-3)
        assert_allclose(adaptive.sd, fine.sd, atol=2e-3)

    def test_all_correct_pattern_does_not_crash(self):
        a, b, _, x = self._simulate(n_items=20, n_persons=5)
        x[0] = 1.0
        x[1] = 0.0
        scorer = PersonScorer(
            None, item_posteriors={"a": a[None], "b": b[None]}, model_type="2PL"
        )
        with pytest.warns(UserWarning, match="all-correct or all-incorrect"):
            result = scorer.score(x, n_plausible_values=10)
        assert np.all(np.isfinite(result.mean))
        assert np.all(np.isfinite(result.sd))
        assert np.all(np.isfinite(result.plausible_values))
        assert result.mean[0] > 0 > result.mean[1]
        assert np.all(np.abs(result.plausible_values) <= 6.0 + 1e-9)

    def test_plausible_values_calibrated_with_true_item_parameters(self):
        """True item parameters => PIT of true theta is uniform.

        The adaptive grid with cell-uniform quantiles keeps this coverage
        error at the Monte Carlo floor (~0.01).
        """
        a, b, theta, x = self._simulate(n_items=100, n_persons=1000, seed=0)
        scorer = PersonScorer(
            None, item_posteriors={"a": a[None], "b": b[None]}, model_type="2PL"
        )
        result = scorer.score(x, n_plausible_values=100)
        k = result.plausible_values.shape[1]
        pit = ((result.plausible_values < theta[:, None]).sum(axis=1) + 0.5) / (
            k + 1
        )
        alphas = np.linspace(0.005, 0.995, 20)
        coverage = np.array([np.mean(np.abs(pit - 0.5) <= al / 2) for al in alphas])
        assert np.median(np.abs(coverage - alphas)) < 0.025

    def test_long_test_window_is_refined(self):
        """Posterior SD (~0.08) far below the coarse SD floor (0.2).

        Without re-centring the pass-2 window on the pass-2 SD the ETI is
        ~2-8% too wide and bounds are off by ~0.18 SD at 1000 items.
        """
        a, b, _, x = self._simulate(n_items=1000, n_persons=12, seed=11)
        scorer = PersonScorer(
            None, item_posteriors={"a": a[None], "b": b[None]}, model_type="2PL"
        )
        r = scorer.score(x, n_plausible_values=50)
        ref = scorer.score(
            x, theta_grid_size=8001, adaptive_grid=False, n_plausible_values=50
        )
        assert np.median(ref.sd) < 0.15
        assert_allclose(r.sd, ref.sd, rtol=5e-3)
        assert np.max(np.abs(r.eti_lower - ref.eti_lower) / ref.sd) < 0.05
        assert np.max(np.abs(r.eti_upper - ref.eti_upper) / ref.sd) < 0.05
        width_ratio = (r.eti_upper - r.eti_lower) / (ref.eti_upper - ref.eti_lower)
        assert np.all(np.abs(width_ratio - 1.0) < 0.03)
        hdi_gap = (r.hdi_upper - r.hdi_lower) - (ref.hdi_upper - ref.hdi_lower)
        assert np.max(np.abs(hdi_gap) / ref.sd) < 0.08

    @staticmethod
    def _separated_draw_scorer():
        """200 item draws, 3 of which shift every difficulty by +6."""
        rng = np.random.default_rng(4)
        n_items, n_draws = 40, 200
        b = rng.normal(0.0, 1.5, n_items)
        theta = rng.normal(0.0, 0.5, 15)
        p = 1.0 / (1.0 + np.exp(-(theta[:, None] - b)))
        x = (rng.random(p.shape) < p).astype(float)
        shift = np.zeros(n_draws)
        shift[:3] = 6.0
        draws = {"b": b[None, :] + shift[:, None]}
        scorer = PersonScorer(None, item_posteriors=draws, model_type="1PL")
        return scorer, x

    def test_separated_item_draws_are_contained(self):
        """Low-weight, far-away draws must not be pushed onto the window edge."""
        scorer, x = self._separated_draw_scorer()
        kw = {"eti_level": 0.99}
        ref = scorer.score(x, theta_grid_size=4001, adaptive_grid=False, **kw)
        out = scorer.score(x, **kw)
        # Without the guard the 1.5% component is moved from ~5.5 to the
        # window edge (~3.7): mean off by ~0.03, upper ETI bound by ~1.5.
        assert_allclose(out.mean, ref.mean, atol=5e-3)
        assert_allclose(out.eti_upper, ref.eti_upper, atol=0.1)
        assert_allclose(out.sd, ref.sd, atol=2e-2)

    def test_containment_guard_is_what_fixes_separated_draws(self, monkeypatch):
        import bayesflow_irt.scoring as scoring

        scorer, x = self._separated_draw_scorer()
        ref = scorer.score(x, theta_grid_size=4001, adaptive_grid=False)
        monkeypatch.setattr(scoring, "_ADAPTIVE_GRID_MAX_OUTSIDE_MASS", np.inf)
        unguarded = scorer.score(x)
        assert np.max(np.abs(unguarded.mean - ref.mean)) > 2e-2

    @pytest.mark.parametrize("device", ["cpu"])
    def test_separated_draws_torch_matches_numpy(self, device):
        pytest.importorskip("torch")
        scorer, x = self._separated_draw_scorer()
        ref = scorer.score(x, n_plausible_values=10)
        out = scorer.score(x, n_plausible_values=10, device=device)
        assert_allclose(out.mean, ref.mean, atol=1e-8)
        assert_allclose(out.plausible_values, ref.plausible_values, atol=1e-8)

    def test_binary_person_blocking_is_equivalent(self, monkeypatch):
        import bayesflow_irt.scoring as scoring

        a, b, _, x = self._simulate(n_items=30, n_persons=17)
        scorer = PersonScorer(
            None, item_posteriors={"a": a[None], "b": b[None]}, model_type="2PL"
        )
        ref = scorer.score(x, n_plausible_values=10)
        # budget // (S * Q * I) = 1 -> one person per block (17 blocks)
        monkeypatch.setattr(scoring, "_PERSON_GRID_BLOCK_ELEMENTS", 1)
        blocked = scorer.score(x, n_plausible_values=10)
        for name in ("mean", "sd", "median", "eti_lower", "hdi_upper",
                     "plausible_values"):
            assert_allclose(getattr(blocked, name), getattr(ref, name), atol=1e-12)

    @pytest.mark.parametrize("model", ["PCM", "GPCM"])
    def test_polytomous_per_person_grid_blocking(self, model, monkeypatch):
        import bayesflow_irt.scoring as scoring

        rng = np.random.default_rng(8)
        n_items, n_persons = 12, 9
        delta = rng.normal(0, 1.0, (n_items, 3))  # unsorted steps are allowed
        item_params = {"delta": delta}
        if model == "GPCM":
            item_params["a"] = np.exp(rng.normal(0, 0.4, n_items))
        responses = rng.integers(0, 4, (n_persons, n_items)).astype(float)
        responses[rng.random(responses.shape) < 0.15] = np.nan
        grids = np.sort(rng.normal(0, 1.5, (n_persons, 7)), axis=-1)

        def fn(*_args, **_kwargs):  # never called for built-in models
            raise AssertionError

        # Oracle: the shared-grid implementation, one person at a time.
        expected = np.stack(
            [
                _log_likelihood_all_persons(
                    grids[p], responses[p : p + 1], item_params, model, fn
                )[0]
                for p in range(n_persons)
            ]
        )
        whole = _log_likelihood_all_persons(grids, responses, item_params, model, fn)
        assert_allclose(whole, expected, atol=1e-10)
        # budget // (Q * I * K) = 0 -> one person per block
        monkeypatch.setattr(scoring, "_POLYTOMOUS_PERSON_GRID_BLOCK_ELEMENTS", 1)
        blocked = _log_likelihood_all_persons(grids, responses, item_params, model, fn)
        assert_allclose(blocked, expected, atol=1e-10)

    def test_polytomous_score_blocked_matches_unblocked(self, monkeypatch):
        import bayesflow_irt.scoring as scoring

        rng = np.random.default_rng(2)
        delta = np.sort(rng.normal(0, 1.0, (15, 3)), axis=-1)
        draws = {"delta": delta[None] + rng.normal(0, 0.05, (4, 15, 3))}
        x = rng.integers(0, 4, (11, 15)).astype(float)
        scorer = PersonScorer(None, item_posteriors=draws, model_type="PCM")
        ref = scorer.score(x, n_plausible_values=8)
        monkeypatch.setattr(scoring, "_POLYTOMOUS_PERSON_GRID_BLOCK_ELEMENTS", 1)
        blocked = scorer.score(x, n_plausible_values=8)
        assert_allclose(blocked.mean, ref.mean, atol=1e-12)
        assert_allclose(blocked.plausible_values, ref.plausible_values, atol=1e-10)

    def test_adaptive_grid_can_be_disabled(self):
        a, b, _, x = self._simulate(n_items=10, n_persons=4)
        scorer = PersonScorer(
            None, item_posteriors={"a": a[None], "b": b[None]}, model_type="2PL"
        )
        fixed = scorer.score(x, adaptive_grid=False)
        grid, _ = _uniform_grid(31, bounds=(-6.0, 6.0))
        assert np.all(np.isin(np.round(fixed.map, 9), np.round(grid, 9)))


class TestReturnPosteriorGrid:
    """``score(return_posterior="grid")`` exposes the summarized masses."""

    @staticmethod
    def _scorer_2pl(n_items=30, n_persons=12):
        a, b, _, x = TestAdaptiveGrid._simulate(n_items=n_items, n_persons=n_persons)
        scorer = PersonScorer(
            None, item_posteriors={"a": a[None], "b": b[None]}, model_type="2PL"
        )
        return scorer, x

    @staticmethod
    def _assert_consistent(res, n_persons):
        assert res.theta_grid.shape == res.theta_posterior_mass.shape
        assert res.theta_grid.shape[0] == n_persons
        assert res.theta_grid.dtype == np.float64
        assert res.theta_posterior_mass.dtype == np.float64
        assert_allclose(res.theta_posterior_mass.sum(axis=1), 1.0, atol=1e-10)
        mean = np.sum(res.theta_posterior_mass * res.theta_grid, axis=1)
        var = np.sum(
            res.theta_posterior_mass * (res.theta_grid - mean[:, None]) ** 2, axis=1
        )
        assert_allclose(mean, res.mean, atol=1e-12)
        assert_allclose(np.sqrt(var), res.sd, atol=1e-12)

    def test_default_fields_are_none(self):
        scorer, x = self._scorer_2pl()
        res = scorer.score(x)
        assert res.theta_grid is None
        assert res.theta_posterior_mass is None

    def test_grid_adaptive(self):
        scorer, x = self._scorer_2pl()
        res = scorer.score(x, return_posterior="grid")
        assert res.theta_grid.shape == (x.shape[0], 21)
        self._assert_consistent(res, x.shape[0])
        # Adaptive grids differ between persons.
        assert not np.allclose(res.theta_grid[0], res.theta_grid[1])

    def test_grid_fixed_is_shared_and_writable(self):
        scorer, x = self._scorer_2pl()
        res = scorer.score(x, adaptive_grid=False, return_posterior="grid")
        assert res.theta_grid.shape == (x.shape[0], 31)
        self._assert_consistent(res, x.shape[0])
        assert np.all(res.theta_grid == res.theta_grid[0])
        assert res.theta_grid.flags.writeable
        res.theta_grid[0, 0] = 99.0
        assert res.theta_grid[1, 0] != 99.0

    def test_grid_with_missing_and_all_missing_person(self):
        scorer, x = self._scorer_2pl(n_persons=6)
        x[0, ::2] = np.nan
        x[1] = np.nan
        res = scorer.score(x, return_posterior="grid")
        self._assert_consistent(res, x.shape[0])
        assert np.all(np.isfinite(res.theta_posterior_mass))
        assert np.all(res.theta_posterior_mass >= 0)
        assert np.all(np.isfinite(res.theta_grid))

    @pytest.mark.parametrize("model", ["PCM", "GPCM"])
    def test_grid_polytomous(self, model):
        rng = np.random.default_rng(4)
        delta = np.sort(rng.normal(0, 1.0, (15, 3)), axis=-1)
        draws = {"delta": delta[None] + rng.normal(0, 0.05, (4, 15, 3))}
        if model == "GPCM":
            draws["a"] = np.exp(rng.normal(0, 0.3, (4, 15)))
        x = rng.integers(0, 4, (7, 15)).astype(float)
        scorer = PersonScorer(None, item_posteriors=draws, model_type=model)
        res = scorer.score(x, return_posterior="grid")
        self._assert_consistent(res, 7)

    def test_grid_torch_path_matches_numpy(self):
        pytest.importorskip("torch")
        scorer, x = self._scorer_2pl(n_persons=5)
        ref = scorer.score(x, return_posterior="grid")
        res = scorer.score(x, device="cpu", return_posterior="grid")
        self._assert_consistent(res, 5)
        assert_allclose(res.theta_grid, ref.theta_grid, atol=1e-10)
        assert_allclose(res.theta_posterior_mass, ref.theta_posterior_mass, atol=1e-10)

    def test_accepts_collections(self):
        scorer, x = self._scorer_2pl(n_persons=3)
        for value in ("grid", ("grid",), ["grid"], {"grid"}):
            res = scorer.score(x, return_posterior=value)
            assert res.theta_grid is not None
        res = scorer.score(x, return_posterior=())
        assert res.theta_grid is None

    @pytest.mark.parametrize("value", ["foo", ("grid", "foo"), 3])
    def test_invalid_value_raises(self, value):
        scorer, x = self._scorer_2pl(n_persons=3)
        with pytest.raises(ValueError, match="return_posterior"):
            scorer.score(x, return_posterior=value)


class TestReturnPosteriorDraws:
    """``score(return_posterior="draws")``: joint (items_s, theta_s) draws."""

    _BOTH = ("grid", "draws")

    @staticmethod
    def _scorer(n_draws=8, n_items=30, n_persons=10, model="2PL", identical=False):
        rng = np.random.default_rng(7)
        a = np.exp(rng.normal(0.0, 0.5, n_items))
        b = rng.normal(0.0, 1.5, n_items)
        theta = rng.normal(0.0, 1.0, n_persons)
        p = 1.0 / (1.0 + np.exp(-a * (theta[:, None] - b)))
        x = (rng.random(p.shape) < p).astype(float)
        if identical:
            draws = {
                "a": np.repeat(a[None], n_draws, 0),
                "b": np.repeat(b[None], n_draws, 0),
            }
        else:
            draws = {
                "a": a * np.exp(rng.normal(0, 0.05, (n_draws, n_items))),
                "b": b + rng.normal(0, 0.1, (n_draws, n_items)),
            }
        if model == "1PL":
            draws.pop("a")
        return PersonScorer(None, item_posteriors=draws, model_type=model), x

    @staticmethod
    def _polytomous(model, n_draws=5, n_persons=7, n_items=15):
        rng = np.random.default_rng(4)
        delta = np.sort(rng.normal(0, 1.0, (n_items, 3)), axis=-1)
        draws = {"delta": delta[None] + rng.normal(0, 0.05, (n_draws, n_items, 3))}
        if model == "GPCM":
            draws["a"] = np.exp(rng.normal(0, 0.3, (n_draws, n_items)))
        x = rng.integers(0, 4, (n_persons, n_items)).astype(float)
        return PersonScorer(None, item_posteriors=draws, model_type=model), x

    @staticmethod
    def _assert_in_cells(res):
        from bayesflow_irt.scoring import _cell_edges

        edges = _cell_edges(res.theta_grid)
        assert np.all(res.theta_draws >= edges[:, [0]].T - 1e-12)
        assert np.all(res.theta_draws <= edges[:, [-1]].T + 1e-12)

    @staticmethod
    def _ks_pvalues(res, persons):
        from scipy.stats import kstest

        from bayesflow_irt.scoring import _edge_cdf

        pvals = []
        for p in persons:
            edges, cdf = _edge_cdf(
                res.theta_grid[p : p + 1], res.theta_posterior_mass[p : p + 1]
            )
            pvals.append(
                kstest(
                    res.theta_draws[:, p], lambda t: np.interp(t, edges[0], cdf[0])
                ).pvalue
            )
        return np.array(pvals)

    def test_shapes_bounds_and_item_draws(self):
        scorer, x = self._scorer(n_draws=8, n_persons=10)
        res = scorer.score(x, return_posterior="draws", seed=1)
        assert res.theta_draws.shape == (8, 10)
        assert res.theta_draws.dtype == np.float64
        assert np.all(np.isfinite(res.theta_draws))
        assert res.item_draws["b"].shape[0] == 8
        assert res.item_draws["a"].shape[0] == 8
        assert res.theta_grid is None  # "grid" not requested
        assert res.theta_window_uncontained.shape == (10,)
        assert res.theta_window_uncontained.dtype == bool
        assert res.theta_draw_mean is None and res.theta_draw_sd is None
        res = scorer.score(x, return_posterior=self._BOTH, seed=1)
        self._assert_in_cells(res)

    def test_iid_draws_match_per_draw_masses(self):
        scorer, x = self._scorer(n_draws=4000, n_persons=4, identical=True)
        res = scorer.score(
            x, return_posterior=self._BOTH, draw_moments=True, seed=11
        )
        # Identical item draws: the marginal masses are every draw's masses.
        assert_allclose(res.theta_draw_mean, np.tile(res.mean, (4000, 1)), atol=1e-10)
        assert_allclose(res.theta_draw_sd, np.tile(res.sd, (4000, 1)), atol=1e-10)
        h = res.theta_grid[:, 1] - res.theta_grid[:, 0]
        n = res.theta_draws.shape[0]
        mean_se = res.sd / np.sqrt(n)
        assert np.all(np.abs(res.theta_draws.mean(0) - res.mean) < 4 * mean_se + 1e-3)
        # Cell-uniform density adds h^2 / 12 to the grid-point variance.
        sd_cell = np.sqrt(res.sd**2 + h**2 / 12)
        assert np.all(
            np.abs(res.theta_draws.std(0) - sd_cell) < 4 * sd_cell / np.sqrt(2 * n)
        )
        assert np.all(self._ks_pvalues(res, range(4)) > 1e-3)

    def test_mean_of_draws_matches_marginal_mean(self):
        scorer, x = self._scorer(n_draws=2000, n_persons=8)
        res = scorer.score(x, return_posterior=self._BOTH, seed=3)
        n = res.theta_draws.shape[0]
        tol = 5 * res.sd / np.sqrt(n) + 1e-3
        assert np.all(np.abs(res.theta_draws.mean(0) - res.mean) < tol)

    def test_seed_reproducibility(self):
        scorer, x = self._scorer()
        a = scorer.score(x, return_posterior="draws", seed=5)
        b = scorer.score(x, return_posterior="draws", seed=5)
        c = scorer.score(x, return_posterior="draws", seed=6)
        assert np.array_equal(a.theta_draws, b.theta_draws)
        assert not np.allclose(a.theta_draws, c.theta_draws)

    def test_n_cores_and_chunking_do_not_change_draws(self, monkeypatch):
        import bayesflow_irt.scoring as scoring

        scorer, x = self._scorer(n_draws=10)
        ref = scorer.score(
            x, return_posterior="draws", draw_moments=True, seed=2
        )
        monkeypatch.setattr(scoring, "_BINARY_POSTERIOR_CHUNK_SIZE", 3)
        for n_cores in (1, 2):
            out = scorer.score(
                x,
                return_posterior="draws",
                draw_moments=True,
                seed=2,
                n_cores=n_cores,
            )
            assert_allclose(out.theta_draws, ref.theta_draws, atol=1e-12)
            assert_allclose(out.theta_draw_mean, ref.theta_draw_mean, atol=1e-12)
            assert_allclose(out.theta_draw_sd, ref.theta_draw_sd, atol=1e-12)

    @pytest.mark.parametrize("adaptive", [True, False])
    def test_torch_matches_numpy(self, adaptive, monkeypatch):
        pytest.importorskip("torch")
        import bayesflow_irt.scoring as scoring

        scorer, x = self._scorer(n_draws=10, n_persons=9)
        kw = {
            "return_posterior": "draws",
            "draw_moments": True,
            "seed": 4,
            "adaptive_grid": adaptive,
        }
        ref = scorer.score(x, **kw)
        # Tiny budget and chunk size: several draw chunks and person blocks.
        monkeypatch.setattr(scoring, "_BINARY_POSTERIOR_CHUNK_SIZE", 3)
        monkeypatch.setitem(scoring._TORCH_BLOCK_ELEMENTS, "cpu", 400)
        out = scorer.score(x, device="cpu", **kw)
        assert_allclose(out.theta_draws, ref.theta_draws, atol=1e-6)
        assert_allclose(out.theta_draw_mean, ref.theta_draw_mean, atol=1e-6)
        assert_allclose(out.theta_draw_sd, ref.theta_draw_sd, atol=1e-6)

    def test_refined_persons_use_matching_prior_weights(self):
        """Long test: refinement replaces the pass-2 grid and prior weights."""
        scorer, x = self._scorer(n_draws=3000, n_items=200, n_persons=5, identical=True)
        res = scorer.score(
            x, return_posterior=self._BOTH, draw_moments=True, seed=8
        )
        h = res.theta_grid[:, 1] - res.theta_grid[:, 0]
        assert np.all(h <= 0.5 * res.sd + 1e-12)  # refinement resolved the SD
        # Stale (pass-2) prior weights would make these differ from res.mean.
        assert_allclose(res.theta_draw_mean[0], res.mean, atol=1e-10)
        assert_allclose(res.theta_draw_sd[0], res.sd, atol=1e-10)
        self._assert_in_cells(res)
        assert np.all(self._ks_pvalues(res, range(5)) > 1e-3)

    def test_draw_sd_is_precise_at_shifted_scale(self):
        # raw moments E[t^2] - E[t]^2 cancel catastrophically at |theta| ~ 1e8
        scorer = PersonScorer(
            None,
            item_posteriors={"b": np.full((2, 4), 1e8)},
            model_type="1PL",
            theta_prior=(1e8, 1.0),
        )
        res = scorer.score(
            np.array([[0.0, 1.0, 0.0, 1.0]]),
            return_posterior="draws",
            draw_moments=True,
            seed=0,
        )
        assert_allclose(res.theta_draw_sd[:, 0], res.sd[0], rtol=1e-6)

    def test_single_draw(self):
        scorer, x = self._scorer(n_draws=1)
        res = scorer.score(x, return_posterior=self._BOTH, draw_moments=True, seed=0)
        assert res.theta_draws.shape == (1, x.shape[0])
        self._assert_in_cells(res)

    def test_all_missing_person_draws_from_truncated_prior(self):
        scorer, x = self._scorer(n_draws=2000, n_persons=4)
        x[1] = np.nan
        res = scorer.score(x, return_posterior=self._BOTH, seed=9)
        self._assert_in_cells(res)
        assert np.all(np.isfinite(res.theta_draws))
        assert abs(res.theta_draws[:, 1].mean()) < 0.1
        assert abs(res.theta_draws[:, 1].std() - 1.0) < 0.1

    @pytest.mark.parametrize("model", ["PCM", "GPCM"])
    def test_polytomous(self, model):
        scorer, x = self._polytomous(model)
        res = scorer.score(x, return_posterior=self._BOTH, draw_moments=True, seed=1)
        assert res.theta_draws.shape == (5, 7)
        assert res.theta_draw_mean.shape == (5, 7)
        assert "delta" in res.item_draws
        self._assert_in_cells(res)

    def test_fixed_grid(self):
        scorer, x = self._scorer()
        res = scorer.score(
            x, return_posterior=self._BOTH, adaptive_grid=False, seed=1
        )
        assert res.theta_draws.shape == (8, 10)
        assert not res.theta_window_uncontained.any()
        self._assert_in_cells(res)

    def test_uncontained_flag(self):
        scorer, x = TestAdaptiveGrid._separated_draw_scorer()
        res = scorer.score(x, return_posterior=self._BOTH, seed=1)
        assert res.theta_window_uncontained.any()
        self._assert_in_cells(res)
        only_grid = scorer.score(x, return_posterior="grid")
        assert_allclose(
            only_grid.theta_window_uncontained, res.theta_window_uncontained
        )

    def test_defaults_none(self):
        scorer, x = self._scorer()
        res = scorer.score(x)
        for name in (
            "theta_window_uncontained",
            "theta_draws",
            "item_draws",
            "theta_draw_mean",
            "theta_draw_sd",
        ):
            assert getattr(res, name) is None
        res = scorer.score(x, return_posterior="grid")
        assert res.theta_window_uncontained is not None
        assert res.theta_draws is None

    def test_seed_or_moments_without_draws_raise(self):
        scorer, x = self._scorer()
        with pytest.raises(ValueError, match="seed"):
            scorer.score(x, seed=1)
        with pytest.raises(ValueError, match="seed"):
            scorer.score(x, return_posterior="grid", seed=1)
        with pytest.raises(ValueError, match="draw_moments"):
            scorer.score(x, draw_moments=True)
        with pytest.raises(ValueError, match="seed"):
            scorer.score(x, return_posterior="draws", seed=-1)

    @pytest.mark.parametrize("device", [None, "cpu"])
    def test_summaries_unchanged_by_sampling(self, device):
        if device is not None:
            pytest.importorskip("torch")
        scorer, x = self._scorer()
        kw = {"n_plausible_values": 7, "eti_level": [0.5, 0.9], "device": device}
        ref = scorer.score(x, **kw)
        out = scorer.score(
            x, return_posterior=self._BOTH, draw_moments=True, seed=1, **kw
        )
        for name in (
            "mean",
            "map",
            "median",
            "sd",
            "eti_lower",
            "eti_upper",
            "hdi_lower",
            "hdi_upper",
            "plausible_values",
        ):
            assert np.array_equal(getattr(out, name), getattr(ref, name)), name


class TestTorchDevicePath:
    """score(device=...) matches the NumPy path (float64)."""

    @staticmethod
    def _scorer(model="2PL", n_items=30, n_persons=25, n_draws=12, seed=2):
        rng = np.random.default_rng(seed)
        a = np.exp(rng.normal(0.0, 0.5, n_items))
        b = rng.normal(0.0, 2.0, n_items)
        theta = rng.normal(0.0, 1.0, n_persons)
        if model in ("PCM", "GPCM"):
            delta = np.sort(rng.normal(0.0, 1.0, (n_items, 3)), axis=-1)
            draws = {"delta": delta + rng.normal(0, 0.1, (n_draws, n_items, 3))}
            if model == "GPCM":
                draws["a"] = a * np.exp(rng.normal(0, 0.05, (n_draws, n_items)))
            x = rng.integers(0, 4, (n_persons, n_items)).astype(float)
            return PersonScorer(None, item_posteriors=draws, model_type=model), x
        p = 1.0 / (1.0 + np.exp(-a * (theta[:, None] - b)))
        x = (rng.random(p.shape) < p).astype(float)
        draws = {"b": b + rng.normal(0, 0.1, (n_draws, n_items))}
        if model == "2PL":
            draws["a"] = a * np.exp(rng.normal(0, 0.05, (n_draws, n_items)))
        return PersonScorer(None, item_posteriors=draws, model_type=model), x

    @staticmethod
    def _devices():
        torch = pytest.importorskip("torch")
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        return devices

    @staticmethod
    def _assert_same(r, ref):
        for name in ("mean", "map", "median", "sd", "eti_lower", "eti_upper",
                     "hdi_lower", "hdi_upper", "plausible_values"):
            if getattr(ref, name) is None:
                continue
            assert_allclose(getattr(r, name), getattr(ref, name), atol=1e-8, rtol=0)

    @pytest.mark.parametrize("model", ["1PL", "2PL", "PCM", "GPCM"])
    @pytest.mark.parametrize("adaptive", [True, False])
    def test_matches_numpy(self, model, adaptive):
        scorer, x = self._scorer(model)
        ref = scorer.score(
            x, n_plausible_values=20, eti_level=[0.5, 0.95], adaptive_grid=adaptive
        )
        for device in self._devices():
            r = scorer.score(
                x,
                n_plausible_values=20,
                eti_level=[0.5, 0.95],
                adaptive_grid=adaptive,
                device=device,
            )
            self._assert_same(r, ref)

    @pytest.mark.parametrize("model", ["2PL", "GPCM"])
    def test_missing_data_and_edge_patterns(self, model):
        scorer, x = self._scorer(model)
        rng = np.random.default_rng(0)
        x[rng.random(x.shape) < 0.2] = np.nan
        x[0] = 3.0 if model == "GPCM" else 1.0
        x[1] = 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = scorer.score(x, n_plausible_values=10)
            for device in self._devices():
                r = scorer.score(x, n_plausible_values=10, device=device)
                self._assert_same(r, ref)

    @pytest.mark.parametrize("model", ["2PL", "PCM", "GPCM"])
    @pytest.mark.parametrize("budget", [200, 50_000])
    def test_draw_chunking(self, monkeypatch, model, budget):
        """Small budgets block the persons, the offset gather and (shared
        grid) the polytomous normalizer's items; 50,000 gives partial item
        blocks."""
        import bayesflow_irt.scoring as scoring

        scorer, x = self._scorer(model, n_draws=20)
        x[np.random.default_rng(1).random(x.shape) < 0.1] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = scorer.score(x)
        monkeypatch.setattr(scoring, "_BINARY_POSTERIOR_CHUNK_SIZE", 3)
        monkeypatch.setitem(scoring._TORCH_BLOCK_ELEMENTS, "cpu", budget)
        pytest.importorskip("torch")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._assert_same(scorer.score(x, device="cpu"), ref)

    def test_custom_likelihood_raises(self):
        pytest.importorskip("torch")
        scorer, x = self._scorer()
        custom = PersonScorer(
            None,
            item_posteriors={"a": np.ones((1, 30)), "b": np.zeros((1, 30))},
            log_likelihood=lambda theta, resp, a, b: np.zeros_like(theta),
        )
        with pytest.raises(NotImplementedError):
            custom.score(x, device="cpu")

    def test_device_check_uses_redetected_model_type(self):
        """The torch/model check runs after posteriors fix the model type."""
        pytest.importorskip("torch")
        rng = np.random.default_rng(0)
        x_bin = rng.integers(0, 2, (4, 6)).astype(float)
        pcm = {"delta": np.sort(rng.normal(0, 1, (3, 6, 2)), axis=-1)}
        binary = {"a": np.ones((3, 6)), "b": rng.normal(0, 1, (3, 6))}

        # Declared 2PL, but the drawn posteriors are PCM -> scored as PCM.
        scorer = PersonScorer(None, model_type="2PL")
        scorer._model_type_explicit = False  # as after adapter-based detection
        scorer._draw_item_posteriors = lambda *a, **k: dict(pcm)
        ref = scorer.score(x_bin)
        out = scorer.score(x_bin, device="cpu")
        assert scorer.model_type == "PCM"
        assert_allclose(out.mean, ref.mean, atol=1e-8)

        # Declared PCM, but the drawn posteriors are 2PL -> scored as 2PL.
        scorer = PersonScorer(None, model_type="PCM")
        scorer._model_type_explicit = False
        scorer._draw_item_posteriors = lambda *a, **k: dict(binary)
        ref = scorer.score(x_bin)
        scorer = PersonScorer(None, model_type="PCM")
        scorer._model_type_explicit = False
        scorer._draw_item_posteriors = lambda *a, **k: dict(binary)
        out = scorer.score(x_bin, device="cpu")
        assert scorer.model_type == "2PL"
        assert_allclose(out.mean, ref.mean, atol=1e-8)

    def test_auto_falls_back_without_torch(self, monkeypatch):
        """'auto' uses NumPy when torch is missing; explicit devices raise."""
        scorer, x = self._scorer()
        ref = scorer.score(x)
        monkeypatch.setitem(sys.modules, "torch", None)  # import torch -> ImportError
        assert_allclose(scorer.score(x, device="auto").mean, ref.mean)
        with pytest.raises(ImportError, match="PyTorch"):
            scorer.score(x, device="cpu")
        with pytest.raises(ImportError, match="PyTorch"):
            scorer.score(x, device="cuda")

    @pytest.mark.parametrize("model", ["2PL", "GPCM"])
    def test_float32_close_to_numpy(self, model):
        scorer, x = self._scorer(model)
        ref = scorer.score(x, return_posterior="draws", seed=3)
        for device in self._devices():
            r = scorer.score(x, device=device, torch_dtype="float32",
                             return_posterior="draws", seed=3)
            assert r.mean.dtype == np.float64
            np.testing.assert_allclose(r.mean, ref.mean, atol=1e-4)
            np.testing.assert_allclose(r.sd, ref.sd, rtol=1e-4)
            np.testing.assert_allclose(r.theta_draws, ref.theta_draws, atol=1e-3)

    def test_invalid_torch_dtype(self):
        scorer, x = self._scorer()
        with pytest.raises(ValueError, match="torch_dtype"):
            scorer.score(x, device="cpu", torch_dtype="float16")

    def test_invalid_device(self):
        scorer, x = self._scorer()
        with pytest.raises(ValueError, match="device"):
            scorer.score(x, device="tpu")


class TestEdgeCases:
    """Test edge cases like all-correct/all-incorrect responses."""

    def test_all_correct_warning(self):
        """All-correct responses should produce bounded estimates with warning."""
        b = np.array([-1.0, 0.0, 1.0])
        responses = np.array([1.0, 1.0, 1.0])  # All correct

        theta_grid, log_prior = _gauss_hermite_grid(31)
        log_likelihood = _log_likelihood_1pl(theta_grid, responses, b)

        # Posterior should be shifted to the right (high theta)
        log_post = log_prior + log_likelihood
        log_post -= np.logaddexp.reduce(log_post)
        posterior = np.exp(log_post)

        # EAP should be positive (person has high ability)
        eap = np.sum(posterior * theta_grid)
        assert eap > 0.5, f"EAP = {eap}, expected positive for all-correct"

        # EAP should be bounded by grid limits
        assert eap < theta_grid[-1], "EAP should be clipped to grid bounds"

    def test_all_incorrect_warning(self):
        """All-incorrect responses should produce bounded estimates with warning."""
        b = np.array([-1.0, 0.0, 1.0])
        responses = np.array([0.0, 0.0, 0.0])  # All incorrect

        theta_grid, log_prior = _gauss_hermite_grid(31)
        log_likelihood = _log_likelihood_1pl(theta_grid, responses, b)

        # Posterior should be shifted to the left (low theta)
        log_post = log_prior + log_likelihood
        log_post -= np.logaddexp.reduce(log_post)
        posterior = np.exp(log_post)

        # EAP should be negative (person has low ability)
        eap = np.sum(posterior * theta_grid)
        assert eap < -0.5, f"EAP = {eap}, expected negative for all-incorrect"

        # EAP should be bounded by grid limits
        assert eap > theta_grid[0], "EAP should be clipped to grid bounds"


class TestPersonScorerRegressions:
    """Regression tests for scorer semantics and validation."""

    def test_fixed_item_posterior_keeps_theta_uncertainty(self):
        """Intervals should not collapse when item parameters are known."""
        posteriors = {"b": np.zeros((100, 5), dtype=np.float64)}
        responses = np.array([[1, 1, 1, 0, 0]], dtype=np.float64)

        scorer = PersonScorer(object(), item_posteriors=posteriors, model_type="1PL")
        result = scorer.score(
            responses,
            theta_grid_size=61,
            n_plausible_values=100,
        )

        assert result.sd[0] > 0.5
        assert result.eti_upper[0] - result.eti_lower[0] > 1.0
        assert result.plausible_values is not None
        assert np.unique(result.plausible_values[0]).size > 1

    def test_eti_levels_can_be_vector(self):
        posteriors = {"b": np.zeros((5, 4), dtype=np.float64)}
        responses = np.array([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=np.float64)

        scorer = PersonScorer(object(), item_posteriors=posteriors, model_type="1PL")
        result = scorer.score(responses, eti_level=[0.5, 0.95], theta_grid_size=41)

        assert result.eti_lower.shape == (2, 2)
        assert result.eti_upper.shape == (2, 2)
        assert result.hdi_lower.shape == (2, 2)
        assert result.hdi_upper.shape == (2, 2)
        assert_allclose(result.eti_levels, np.array([0.5, 0.95]))
        assert_allclose(result.hdi_levels, np.array([0.5, 0.95]))
        assert np.all(result.eti_lower[0] >= result.eti_lower[1])
        assert np.all(result.eti_upper[0] <= result.eti_upper[1])

    def test_plausible_values_are_optional_and_count_is_explicit(self):
        posteriors = {"b": np.zeros((3, 4), dtype=np.float64)}
        responses = np.array([[1, 1, 0, 0]], dtype=np.float64)

        scorer = PersonScorer(object(), item_posteriors=posteriors, model_type="1PL")
        default_result = scorer.score(responses, theta_grid_size=41)
        sampled_result = scorer.score(
            responses,
            theta_grid_size=41,
            n_plausible_values=17,
        )

        assert default_result.plausible_values is None
        assert sampled_result.plausible_values is not None
        assert sampled_result.plausible_values.shape == (1, 17)
        assert np.unique(sampled_result.plausible_values[0]).size > 1

    def test_parallel_chunks_match_serial_results(self):
        rng = np.random.default_rng(123)
        posteriors = {"delta": rng.normal(size=(32, 5, 3)).astype(np.float64)}
        responses = rng.integers(0, 4, size=(6, 5)).astype(np.float64)

        scorer = PersonScorer(object(), item_posteriors=posteriors, model_type="PCM")
        serial = scorer.score(responses, theta_grid_size=31, n_cores=1)
        parallel = scorer.score(responses, theta_grid_size=31, n_cores=4)

        assert_allclose(serial.mean, parallel.mean)
        assert_allclose(serial.map, parallel.map)
        assert_allclose(serial.median, parallel.median)
        assert_allclose(serial.sd, parallel.sd)
        assert_allclose(serial.eti_lower, parallel.eti_lower)
        assert_allclose(serial.eti_upper, parallel.eti_upper)
        assert_allclose(serial.hdi_lower, parallel.hdi_lower)
        assert_allclose(serial.hdi_upper, parallel.hdi_upper)

    def test_score_rejects_non_integer_n_cores(self):
        posteriors = {"b": np.zeros((2, 3), dtype=np.float64)}
        responses = np.array([[1, 0, 1]], dtype=np.float64)
        scorer = PersonScorer(object(), item_posteriors=posteriors, model_type="1PL")

        with pytest.raises(ValueError, match="positive integer"):
            scorer.score(responses, theta_grid_size=11, n_cores=2.5)

    def test_theta_grid_size_controls_exact_polytomous_grid_resolution(self):
        posteriors = {"delta": np.zeros((2, 3, 2), dtype=np.float64)}
        scorer = PersonScorer(object(), item_posteriors=posteriors, model_type="PCM")

        theta_grid, _ = scorer._get_quadrature_grid(21)

        assert theta_grid.shape == (21,)

    def test_precomputed_polytomous_batch_dimension_is_squeezed(self):
        """PCM posteriors from approximator.sample() may be (1, S, I, K-1)."""
        posteriors = {"delta": np.zeros((1, 2, 3, 2), dtype=np.float64)}
        responses = np.zeros((4, 3), dtype=np.float64)

        scorer = PersonScorer(object(), item_posteriors=posteriors, model_type="PCM")
        with pytest.warns(UserWarning, match="all-lowest-category"):
            result = scorer.score(responses, theta_grid_size=21)

        assert result.n_items == 3
        assert np.all(np.isfinite(result.mean))

    def test_polytomous_category_upper_bound_is_validated(self):
        posteriors = {"delta": np.zeros((2, 3, 2), dtype=np.float64)}
        responses = np.array([[0, 3, 1]], dtype=np.float64)

        scorer = PersonScorer(object(), item_posteriors=posteriors, model_type="PCM")
        with pytest.raises(ValueError, match=r"\{0, \.\.\., 2\}"):
            scorer.score(responses, theta_grid_size=21)

    def test_polytomous_edge_detection_uses_max_category(self):
        posteriors = {"delta": np.zeros((2, 3, 2), dtype=np.float64)}
        scorer = PersonScorer(object(), item_posteriors=posteriors, model_type="PCM")

        middle_category = np.ones((2, 3), dtype=np.float64)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            scorer.score(middle_category, theta_grid_size=21)
        assert not any("all-highest-category" in str(w.message) for w in caught)

        highest_category = np.full((2, 3), 2.0, dtype=np.float64)
        with pytest.warns(UserWarning, match="all-highest-category"):
            scorer.score(highest_category, theta_grid_size=21)

    def test_binary_missing_responses_are_omitted_from_likelihood(self):
        posteriors = {"b": np.zeros((5, 4), dtype=np.float64)}
        responses = np.array(
            [
                [1, np.nan, 0, np.nan],
                [np.nan, np.nan, np.nan, np.nan],
            ],
            dtype=np.float64,
        )

        scorer = PersonScorer(object(), item_posteriors=posteriors, model_type="1PL")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = scorer.score(responses, theta_grid_size=41)

        assert not caught
        assert abs(result.mean[0]) < 1e-12
        assert abs(result.mean[1]) < 1e-12
        assert result.sd[0] > 0
        assert result.sd[1] > 0

    def test_binary_prior_only_interval_matches_normal_prior(self):
        posteriors = {"b": np.zeros((5, 4), dtype=np.float64)}
        responses = np.full((1, 4), np.nan, dtype=np.float64)

        scorer = PersonScorer(object(), item_posteriors=posteriors, model_type="1PL")
        result = scorer.score(responses, theta_grid_size=121)

        assert result.eti_lower[0] == pytest.approx(-1.96, abs=0.12)
        assert result.eti_upper[0] == pytest.approx(1.96, abs=0.12)
        assert result.hdi_lower[0] == pytest.approx(-result.hdi_upper[0], abs=0.15)

    def test_polytomous_missing_responses_are_omitted_from_likelihood(self):
        posteriors = {"delta": np.zeros((5, 3, 2), dtype=np.float64)}
        responses = np.array([[0, np.nan, 2]], dtype=np.float64)

        scorer = PersonScorer(object(), item_posteriors=posteriors, model_type="PCM")
        result = scorer.score(responses, theta_grid_size=31)

        assert np.all(np.isfinite(result.mean))
        assert result.sd[0] > 0

    def test_single_sample_polytomous_posteriors_are_accepted(self):
        posteriors = {"delta": np.zeros((1, 3, 2), dtype=np.float64)}
        responses = np.array([[0, 1, 2]], dtype=np.float64)

        scorer = PersonScorer(object(), item_posteriors=posteriors, model_type="PCM")
        result = scorer.score(responses, theta_grid_size=31)

        assert np.all(np.isfinite(result.mean))

    def test_autodraw_rejects_missing_responses(self):
        class DummyApprox:
            adapter = SimpleNamespace(transforms=[])

            def sample(self, *, num_samples, conditions):
                raise AssertionError("sample should not be called with NaN responses")

        scorer = PersonScorer(DummyApprox(), model_type="1PL")
        responses = np.array([[1, np.nan, 0]], dtype=np.float64)

        with pytest.raises(ValueError, match="NaN missing responses"):
            scorer.score(responses, n_item_samples=3, theta_grid_size=11)

    def test_polytomous_autodraw_accepts_single_item_sample(self):
        class DummyApprox:
            adapter = SimpleNamespace(
                transforms=[
                    SimpleNamespace(
                        include="responses",
                        transform_constructor=lambda: OneHotEncode(3),
                    )
                ]
            )

            def sample(self, *, num_samples, conditions):
                assert num_samples == 1
                n_items = conditions["responses"].shape[-1]
                return {
                    "delta": np.zeros((1, 1, n_items, 2), dtype=np.float64),
                }

        scorer = PersonScorer(DummyApprox(), model_type="PCM")
        responses = np.array([[0, 1, 2], [1, 1, 0]], dtype=np.float64)

        result = scorer.score(responses, n_item_samples=1, theta_grid_size=21)

        assert np.all(np.isfinite(result.mean))

    def test_inf_responses_are_rejected(self):
        posteriors = {"b": np.zeros((2, 3), dtype=np.float64)}
        scorer = PersonScorer(object(), item_posteriors=posteriors, model_type="1PL")

        with pytest.raises(ValueError, match="Inf"):
            scorer.score(np.array([[1, np.inf, 0]], dtype=np.float64))

    def test_explicit_model_type_rejects_incompatible_posterior_keys(self):
        posteriors = {
            "a": np.ones((2, 3), dtype=np.float64),
            "b": np.zeros((2, 3), dtype=np.float64),
        }

        with pytest.raises(ValueError, match="Unexpected posterior keys"):
            PersonScorer(object(), item_posteriors=posteriors, model_type="1PL")

    def test_polytomous_grid_tracks_theta_prior_location_and_scale(self):
        posteriors = {"delta": np.zeros((2, 3, 2), dtype=np.float64)}
        scorer = PersonScorer(
            object(),
            item_posteriors=posteriors,
            model_type="PCM",
            theta_prior=(5.0, 2.0),
        )

        theta_grid, _ = scorer._get_quadrature_grid(21)

        assert theta_grid[0] == pytest.approx(-7.0)
        assert theta_grid[-1] == pytest.approx(17.0)

    def test_response_transform_detection_requires_one_hot(self):
        class NotOneHot:
            pass

        approx = SimpleNamespace(
            adapter=SimpleNamespace(
                transforms=[
                    SimpleNamespace(
                        include="responses",
                        transform_constructor=NotOneHot,
                    )
                ]
            )
        )

        scorer = PersonScorer(approx)

        assert scorer.model_type == "1PL"

    def test_autodraw_passes_size_conditions(self):
        class DummyApprox:
            def __init__(self):
                self.adapter = SimpleNamespace(
                    transforms=[
                        SimpleNamespace(
                            keys=["n_items", "n_persons"],
                            into="inference_conditions",
                        )
                    ]
                )
                self.seen_conditions = None

            def sample(self, *, num_samples, conditions):
                self.seen_conditions = conditions
                return {
                    "b": np.zeros(
                        (1, num_samples, conditions["responses"].shape[-1]),
                        dtype=np.float64,
                    )
                }

        approx = DummyApprox()
        responses = np.array(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 1, 0, 0],
            ],
            dtype=np.float64,
        )
        scorer = PersonScorer(approx, model_type="1PL")
        scorer.score(responses, n_item_samples=3, theta_grid_size=11)

        assert sorted(approx.seen_conditions) == ["n_items", "n_persons", "responses"]
        assert approx.seen_conditions["n_items"].shape == (1, 1)
        assert approx.seen_conditions["n_persons"].shape == (1, 1)
        assert approx.seen_conditions["n_items"][0, 0] == responses.shape[1]
        assert approx.seen_conditions["n_persons"][0, 0] == responses.shape[0]

    def test_autodraw_requires_extra_adapter_conditions(self):
        class DummyApprox:
            adapter = SimpleNamespace(
                transforms=[
                    SimpleNamespace(
                        keys=["n_items", "n_persons", "sigma_b"],
                        into="inference_conditions",
                    )
                ]
            )

            def sample(self, *, num_samples, conditions):
                raise AssertionError(
                    "sample should not be called with missing conditions"
                )

        scorer = PersonScorer(DummyApprox(), model_type="1PL")
        responses = np.array([[1, 0, 1]], dtype=np.float64)

        with pytest.raises(ValueError, match="sigma_b"):
            scorer.score(responses, n_item_samples=3, theta_grid_size=11)

    def test_autodraw_rejects_reserved_size_condition_overrides(self):
        class DummyApprox:
            adapter = SimpleNamespace(transforms=[])

            def sample(self, *, num_samples, conditions):
                raise AssertionError("sample should not be called")

        scorer = PersonScorer(DummyApprox(), model_type="1PL")
        responses = np.array([[1, 0, 1]], dtype=np.float64)

        with pytest.raises(ValueError, match="must not contain"):
            scorer.score(
                responses,
                n_item_samples=3,
                theta_grid_size=11,
                item_conditions={"n_items": np.array([2], dtype=np.float32)},
            )

    def test_constructor_uses_sample_fallback_for_ambiguous_2pl_adapter(self):
        class DummyApprox:
            adapter = SimpleNamespace(transforms=[])

            def sample(self, *, num_samples, conditions):
                n_items = conditions["responses"].shape[-1]
                return {
                    "a": np.ones((1, num_samples, n_items), dtype=np.float64),
                    "b": np.zeros((1, num_samples, n_items), dtype=np.float64),
                }

        scorer = PersonScorer(DummyApprox())

        assert scorer.model_type == "2PL"

    def test_autodraw_redetects_2pl_when_constructor_fallback_lacks_conditions(self):
        class DummyApprox:
            adapter = SimpleNamespace(
                transforms=[
                    SimpleNamespace(
                        keys=["n_items", "n_persons", "sigma_b"],
                        into="inference_conditions",
                    )
                ]
            )

            def sample(self, *, num_samples, conditions):
                assert "sigma_b" in conditions
                n_items = conditions["responses"].shape[-1]
                return {
                    "a": np.ones((1, num_samples, n_items), dtype=np.float64),
                    "b": np.zeros((1, num_samples, n_items), dtype=np.float64),
                }

        scorer = PersonScorer(DummyApprox())
        assert scorer.model_type == "1PL"

        responses = np.array([[1, 0, 1]], dtype=np.float64)
        result = scorer.score(
            responses,
            n_item_samples=3,
            theta_grid_size=11,
            item_conditions={"sigma_b": np.array([1.0], dtype=np.float32)},
        )

        assert scorer.model_type == "2PL"
        assert np.all(np.isfinite(result.mean))

    def test_autodraw_rejects_invalid_binary_responses_before_sampling(self):
        class DummyApprox:
            adapter = SimpleNamespace(transforms=[])

            def sample(self, *, num_samples, conditions):
                raise AssertionError("sample should not be called")

        scorer = PersonScorer(DummyApprox(), model_type="1PL")

        with pytest.raises(ValueError, match=r"\{0, 1\}"):
            scorer.draw_item_posteriors(
                np.array([[0, 2, 1]], dtype=np.float64),
                n_item_samples=3,
            )

    def test_autodraw_rejects_invalid_polytomous_category_before_sampling(self):
        class DummyApprox:
            adapter = SimpleNamespace(
                transforms=[
                    SimpleNamespace(
                        include="responses",
                        transform_constructor=lambda: OneHotEncode(3),
                    )
                ]
            )

            def sample(self, *, num_samples, conditions):
                raise AssertionError("sample should not be called")

        scorer = PersonScorer(DummyApprox(), model_type="PCM")

        with pytest.raises(ValueError, match=r"\{0, \.\.\., 2\}"):
            scorer.draw_item_posteriors(
                np.array([[0, 3, 1]], dtype=np.float64),
                n_item_samples=3,
            )

    def test_cached_item_ids_are_enforced(self):
        posteriors = {
            "b": np.zeros((2, 3), dtype=np.float64),
            "item_ids": np.array(["item-a", "item-b", "item-c"]),
        }
        responses = np.array([[1, 0, 1]], dtype=np.float64)

        scorer = PersonScorer(object(), item_posteriors=posteriors, model_type="1PL")

        with pytest.raises(ValueError, match="Pass matching item_ids"):
            scorer.score(responses, theta_grid_size=11)

        with pytest.raises(ValueError, match="do not match"):
            scorer.score(
                responses,
                theta_grid_size=11,
                item_ids=np.array(["item-a", "item-c", "item-b"]),
            )

        result = scorer.score(
            responses,
            theta_grid_size=11,
            item_ids=np.array(["item-a", "item-b", "item-c"]),
        )
        assert np.all(np.isfinite(result.mean))

    def test_constructor_item_ids_must_match_posterior_metadata(self):
        posteriors = {
            "b": np.zeros((2, 3), dtype=np.float64),
            "item_ids": np.array(["item-a", "item-b", "item-c"]),
        }

        with pytest.raises(ValueError, match="do not match"):
            PersonScorer(
                object(),
                item_posteriors=posteriors,
                model_type="1PL",
                item_ids=np.array(["item-a", "item-c", "item-b"]),
            )

    def test_score_time_item_ids_cannot_bind_idless_cached_posteriors(self):
        posteriors = {"b": np.zeros((2, 3), dtype=np.float64)}
        responses = np.array([[1, 0, 1]], dtype=np.float64)

        scorer = PersonScorer(object(), item_posteriors=posteriors, model_type="1PL")

        with pytest.raises(ValueError, match="created without item_ids"):
            scorer.score(
                responses,
                theta_grid_size=11,
                item_ids=np.array(["item-a", "item-b", "item-c"]),
            )

    def test_custom_model_type_uses_custom_posterior_keys(self):
        posteriors = {"gamma": np.ones((2, 3), dtype=np.float64)}
        responses = np.array([[0.0, 1.0, 0.0]], dtype=np.float64)

        def custom_ll(theta, responses, **item_params):
            assert sorted(item_params) == ["gamma"]
            assert item_params["gamma"].shape == (3,)
            return np.zeros_like(theta, dtype=np.float64)

        scorer = PersonScorer(
            object(),
            item_posteriors=posteriors,
            model_type="custom",
            log_likelihood=custom_ll,
        )
        result = scorer.score(responses, theta_grid_size=11)

        assert result.n_items == 3
        assert np.all(np.isfinite(result.mean))

    def test_autodraw_rejects_non_integer_n_item_samples(self):
        class DummyApprox:
            adapter = SimpleNamespace(transforms=[])

            def sample(self, *, num_samples, conditions):
                raise AssertionError("sample should not be called")

        scorer = PersonScorer(DummyApprox(), model_type="1PL")

        with pytest.raises(ValueError, match="positive integer"):
            scorer.draw_item_posteriors(
                np.array([[1, 0, 1]], dtype=np.float64),
                n_item_samples=2.5,
            )

    def test_score_rejects_non_integer_n_plausible_values(self):
        posteriors = {"b": np.zeros((2, 3), dtype=np.float64)}
        responses = np.array([[1, 0, 1]], dtype=np.float64)
        scorer = PersonScorer(object(), item_posteriors=posteriors, model_type="1PL")

        with pytest.raises(ValueError, match="positive integer"):
            scorer.score(
                responses,
                theta_grid_size=11,
                n_plausible_values=2.9,
            )


# ---------------------------------------------------------------------------
# Integration tests (with trained approximator)
# ---------------------------------------------------------------------------


class TestPersonScorerApproximatorInterface:
    """Approximator-interface smoke tests without real fitting."""

    def test_autodraw_1pl_smoke_without_training(self):
        class DummyApprox:
            def __init__(self):
                self.adapter = SimpleNamespace(
                    transforms=[
                        SimpleNamespace(
                            keys=["n_items", "n_persons"],
                            into="inference_conditions",
                        )
                    ]
                )

            def sample(self, *, num_samples, conditions):
                n_items = conditions["responses"].shape[-1]
                return {
                    "b": np.zeros((1, num_samples, n_items), dtype=np.float64),
                }

        responses = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1]],
            dtype=np.float64,
        )
        scorer = PersonScorer(DummyApprox())

        result = scorer.score(responses, n_item_samples=5, theta_grid_size=31)

        assert isinstance(result, PersonScore)
        assert np.all(np.isfinite(result.mean))
        assert result.n_items == responses.shape[1]

    def test_autodraw_pcm_smoke_without_training(self):
        class DummyApprox:
            def __init__(self):
                self.adapter = SimpleNamespace(
                    transforms=[
                        SimpleNamespace(
                            include="responses",
                            transform_constructor=lambda: OneHotEncode(3),
                        ),
                        SimpleNamespace(
                            keys=["n_items", "n_persons"],
                            into="inference_conditions",
                        ),
                    ]
                )

            def sample(self, *, num_samples, conditions):
                n_items = conditions["responses"].shape[-1]
                return {
                    "delta": np.zeros((1, num_samples, n_items, 2), dtype=np.float64),
                }

        responses = np.array([[0, 1, 2], [1, 1, 0]], dtype=np.float64)
        scorer = PersonScorer(DummyApprox())

        result = scorer.score(responses, n_item_samples=4, theta_grid_size=31)

        assert isinstance(result, PersonScore)
        assert np.all(np.isfinite(result.mean))
        assert result.n_items == responses.shape[1]

    def test_loading_parameterization_smoke_without_training(self):
        class DummyApprox:
            def __init__(self):
                self.adapter = SimpleNamespace(
                    transforms=[],
                    _bayesflow_irt_internal_parameterization="loading",
                )

            def sample(self, *, num_samples, conditions):
                n_items = conditions["responses"].shape[-1]
                return {
                    "a": np.ones((1, num_samples, n_items), dtype=np.float64),
                    "b": np.zeros((1, num_samples, n_items), dtype=np.float64),
                    "d": np.zeros((1, num_samples, n_items), dtype=np.float64),
                }

        responses = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float64)
        scorer = PersonScorer(DummyApprox())

        result = scorer.score(responses, n_item_samples=4, theta_grid_size=31)

        assert isinstance(result, PersonScore)
        assert np.all(np.isfinite(result.mean))
        assert result.n_items == responses.shape[1]


@pytest.fixture(scope="session")
def trained_2pl_approximator():
    """Build and train a small 2PL approximator for testing."""
    simulator = make_2pl_simulator(
        n_items=(10, 15),
        n_persons=(20, 30),
        prior_a=_PRIOR_A,
        prior_b=_PRIOR_B,
        prior_theta=_PRIOR_THETA,
    )
    adapter = make_2pl_adapter(
        simulator,
        log_transform_a=True,
        internal_parameterization="standard",
    )

    summary_net = MatrixEquivariantIRTSummary(
        embed_dim=16,
        matrix_hidden_dim=16,
        num_matrix_blocks=1,
        num_outer_sabs=1,
        outer_embed_dim=16,
        outer_mlp_width=32,
    )
    inference_net = bf.networks.FlowMatching(
        subnet_kwargs={"widths": (32, 32)},
    )

    approximator = EquivariantIRTApproximator(
        adapter=adapter,
        inference_network=inference_net,
        summary_network=summary_net,
    )
    approximator.compile(optimizer="adam")

    approximator.fit(
        simulator=simulator,
        epochs=1,
        num_batches=4,
        batch_size=10,
        workers=1,
        verbose=False,
    )

    return approximator, simulator


@pytest.fixture(scope="session")
def trained_pcm_approximator():
    """Build and train a small PCM approximator for testing."""
    simulator = make_pcm_simulator(
        n_items=(5, 8),
        n_persons=(20, 30),
        n_categories=4,
        prior_delta=_PRIOR_DELTA,
        prior_theta=_PRIOR_THETA,
    )
    adapter = make_pcm_adapter(simulator)

    summary_net = MatrixEquivariantIRTSummary(
        embed_dim=16,
        matrix_hidden_dim=16,
        num_matrix_blocks=1,
        num_outer_sabs=1,
        outer_embed_dim=16,
        outer_mlp_width=32,
    )
    inference_net = bf.networks.FlowMatching(
        subnet_kwargs={"widths": (32, 32)},
    )

    approximator = EquivariantIRTApproximator(
        adapter=adapter,
        inference_network=inference_net,
        summary_network=summary_net,
    )
    approximator.compile(optimizer="adam")

    approximator.fit(
        simulator=simulator,
        epochs=1,
        num_batches=4,
        batch_size=10,
        workers=1,
        verbose=False,
    )

    return approximator, simulator


@pytest.fixture(scope="session")
def trained_gpcm_approximator():
    """Build and train a small GPCM approximator for testing."""
    simulator = make_gpcm_simulator(
        n_items=(5, 8),
        n_persons=(20, 30),
        n_categories=4,
        prior_a=_PRIOR_A,
        prior_delta=_PRIOR_DELTA,
        prior_theta=_PRIOR_THETA,
    )
    adapter = make_gpcm_adapter(
        simulator,
        log_transform_a=True,
        internal_parameterization="standard",
    )

    summary_net = MatrixEquivariantIRTSummary(
        embed_dim=16,
        matrix_hidden_dim=16,
        num_matrix_blocks=1,
        num_outer_sabs=1,
        outer_embed_dim=16,
        outer_mlp_width=32,
    )
    inference_net = bf.networks.FlowMatching(
        subnet_kwargs={"widths": (32, 32)},
    )

    approximator = EquivariantIRTApproximator(
        adapter=adapter,
        inference_network=inference_net,
        summary_network=summary_net,
    )
    approximator.compile(optimizer="adam")

    approximator.fit(
        simulator=simulator,
        epochs=1,
        num_batches=4,
        batch_size=10,
        workers=1,
        verbose=False,
    )

    return approximator, simulator


# ---------------------------------------------------------------------------
# Fresh-process save/reload regression helpers
#
# Regression coverage: a one-hot response transform whose
# ``transform_constructor`` is a
# ``lambda: OneHotEncode(n_categories)`` closure fails
# to deserialize when a checkpoint was reloaded in a genuinely fresh process
# with ``keras.saving.load_model(..., safe_mode=False)``. Keras's
# ``safe_mode=False`` lambda deserialization reconstructs the lambda's
# bytecode but does not repopulate its ``__globals__`` with the defining
# module's namespace, so calling it raised
# ``NameError: name 'OneHotEncode' is not defined``. Reloading in the *same*
# process the checkpoint was saved in does not reproduce this bug, because
# the defining module's globals are still live from the original import --
# hence the ``subprocess`` isolation used here.
# ---------------------------------------------------------------------------

_REPO_SRC_DIR = Path(__file__).resolve().parents[1] / "src"

_RELOAD_AND_SCORE_SCRIPT = """
import os
os.environ.setdefault("KERAS_BACKEND", "torch")
import sys

import numpy as np
import keras

import bayesflow_irt as bfi
from bayesflow_irt.scoring import PersonScorer

model_path, responses_path = sys.argv[1], sys.argv[2]

approximator = keras.saving.load_model(
    model_path,
    safe_mode=False,
    custom_objects={
        "EquivariantIRTApproximator": bfi.EquivariantIRTApproximator,
        "MatrixEquivariantIRTSummary": bfi.MatrixEquivariantIRTSummary,
    },
)

responses = np.load(responses_path)
scorer = PersonScorer(approximator)
results = scorer.score(responses, n_item_samples=20, theta_grid_size=11)
assert np.all(np.isfinite(results.mean)), results.mean
print("SCORE_OK")
"""


def _assert_fresh_process_reload_and_score(model_path, responses):
    """Reload ``model_path`` and score ``responses`` in a fresh subprocess.

    Runs the reload + score entirely in a new Python process (not just a
    fresh Keras deserialization call within this test process) so that stale
    module globals from the original ``adapters`` import cannot mask the
    lambda-deserialization bug this guards against.
    """
    responses_path = model_path.with_suffix(".responses.npy")
    np.save(responses_path, responses)

    env = dict(os.environ)
    env["KERAS_BACKEND"] = "torch"
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(_REPO_SRC_DIR), existing_pythonpath) if p
    )

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _RELOAD_AND_SCORE_SCRIPT,
            str(model_path),
            str(responses_path),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert result.returncode == 0 and "SCORE_OK" in result.stdout, (
        "Fresh-process reload+score failed.\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )


@pytest.mark.slow
class TestPersonScorerIntegration:
    """Integration tests with trained approximators."""

    def test_1pl_autodraw_workflow(self, trained_1pl_approximator):
        """Test PersonScorer with auto-draw from approximator."""
        approximator, simulator = trained_1pl_approximator

        # Generate test data
        test_data = simulator.sample(1)
        responses = test_data["responses"][0]  # (P, I)

        # Create scorer without pre-computed posteriors
        scorer = PersonScorer(approximator)

        # Score with auto-draw
        results = scorer.score(
            responses,
            n_item_samples=100,
            theta_grid_size=21,
        )

        # Check output structure
        assert isinstance(results, PersonScore)
        assert results.mean.shape == (responses.shape[0],)
        assert results.median.shape == (responses.shape[0],)
        assert results.sd.shape == (responses.shape[0],)
        assert results.eti_lower.shape == (responses.shape[0],)
        assert results.eti_upper.shape == (responses.shape[0],)
        assert results.hdi_lower.shape == (responses.shape[0],)
        assert results.hdi_upper.shape == (responses.shape[0],)
        assert results.n_items == responses.shape[1]

        # Sanity checks
        assert np.all(np.isfinite(results.mean))
        assert np.all(results.sd >= 0)
        assert np.all(results.eti_lower <= results.eti_upper)
        assert np.all(results.hdi_lower <= results.hdi_upper)

        # EAP should be in reasonable range (quadrature grid bounds)
        assert np.all(results.mean > -6.0) and np.all(results.mean < 6.0)

    def test_1pl_precomputed_workflow(self, trained_1pl_approximator):
        """Test PersonScorer with pre-computed posteriors."""
        approximator, simulator = trained_1pl_approximator

        # Generate test data
        test_data = simulator.sample(1)
        responses = test_data["responses"][0]

        # Draw posteriors once with the scorer helper so size conditions are
        # included exactly as they were during training.
        posterior_drawer = PersonScorer(approximator)
        posteriors = posterior_drawer.draw_item_posteriors(
            responses,
            n_item_samples=100,
        )

        # Create scorer with pre-computed posteriors
        scorer = PersonScorer(approximator, item_posteriors=posteriors)

        # Score should use cached posteriors (no re-draw)
        results = scorer.score(responses, theta_grid_size=21)

        assert results.mean.shape == (responses.shape[0],)
        assert np.all(np.isfinite(results.mean))

    def test_1pl_new_persons_same_items(self, trained_1pl_approximator):
        """Test scoring new persons with the same items."""
        approximator, simulator = trained_1pl_approximator

        # Draw posteriors from one dataset
        data1 = simulator.sample(1)
        responses1 = data1["responses"][0]
        n_items = responses1.shape[1]

        posteriors = PersonScorer(approximator).draw_item_posteriors(
            responses1,
            n_item_samples=100,
        )

        # Create scorer with pre-computed posteriors
        scorer = PersonScorer(approximator, item_posteriors=posteriors)

        results1 = scorer.score(responses1, theta_grid_size=21)

        # Generate new data with the SAME number of items
        # We can't use simulator.sample() directly because it varies
        # So we generate and then select matching item count
        for _ in range(10):  # Try up to 10 times to get matching item count
            data2 = simulator.sample(1)
            responses2 = data2["responses"][0]
            if responses2.shape[1] == n_items:
                results2 = scorer.score(responses2, theta_grid_size=21)
                assert results1.n_items == results2.n_items
                return

        # If we couldn't get matching items, skip this test
        pytest.skip("Could not generate data with matching item count")

    def test_wrong_item_count_raises_error(self, trained_1pl_approximator):
        """Scoring responses with different item count should raise error."""
        approximator, simulator = trained_1pl_approximator

        data1 = simulator.sample(1)
        posteriors = PersonScorer(approximator).draw_item_posteriors(
            data1["responses"][0],
            n_item_samples=100,
        )

        scorer = PersonScorer(approximator, item_posteriors=posteriors)

        # Create responses with different item count
        wrong_responses = np.random.randint(0, 2, size=(10, 20)).astype(np.float64)

        with pytest.raises(ValueError, match="Response data has .* items, but cached"):
            scorer.score(wrong_responses)

    def test_2pl_scoring(self, trained_2pl_approximator):
        """Test 2PL scoring."""
        approximator, simulator = trained_2pl_approximator

        test_data = simulator.sample(1)
        responses = test_data["responses"][0]

        scorer = PersonScorer(approximator)
        results = scorer.score(responses, n_item_samples=100, theta_grid_size=21)

        assert isinstance(results, PersonScore)
        assert np.all(np.isfinite(results.mean))
        assert results.n_items == responses.shape[1]

    def test_pcm_scoring(self, trained_pcm_approximator):
        """Test PCM scoring (polytomous)."""
        approximator, simulator = trained_pcm_approximator

        test_data = simulator.sample(1)
        responses = test_data["responses"][0]

        scorer = PersonScorer(approximator)
        results = scorer.score(responses, n_item_samples=100, theta_grid_size=21)

        assert isinstance(results, PersonScore)
        assert np.all(np.isfinite(results.mean))
        assert results.n_items == responses.shape[1]

    def test_gpcm_scoring(self, trained_gpcm_approximator):
        """Test GPCM scoring (polytomous with discrimination)."""
        approximator, simulator = trained_gpcm_approximator

        test_data = simulator.sample(1)
        responses = test_data["responses"][0]

        scorer = PersonScorer(approximator)
        results = scorer.score(responses, n_item_samples=100, theta_grid_size=21)

        assert isinstance(results, PersonScore)
        assert np.all(np.isfinite(results.mean))
        assert results.n_items == responses.shape[1]

    def test_pcm_scoring_survives_fresh_process_reload(
        self, trained_pcm_approximator, tmp_path
    ):
        """PersonScorer must work on a PCM checkpoint reloaded fresh.

        Regression test: see the module-level comment above
        ``_assert_fresh_process_reload_and_score`` for the bug this guards
        against (one-hot ``transform_constructor`` lambda failing to
        deserialize outside the process that saved it).
        """
        approximator, simulator = trained_pcm_approximator

        test_data = simulator.sample(1)
        responses = test_data["responses"][0]

        model_path = tmp_path / "pcm_roundtrip.keras"
        approximator.save(model_path)

        _assert_fresh_process_reload_and_score(model_path, responses)

    def test_gpcm_scoring_survives_fresh_process_reload(
        self, trained_gpcm_approximator, tmp_path
    ):
        """PersonScorer must work on a GPCM checkpoint reloaded fresh.

        Regression test: see the module-level comment above
        ``_assert_fresh_process_reload_and_score`` for the bug this guards
        against. GPCM uses the identical one-hot ``FilterTransform`` pattern
        as PCM, so the same fix applies.
        """
        approximator, simulator = trained_gpcm_approximator

        test_data = simulator.sample(1)
        responses = test_data["responses"][0]

        model_path = tmp_path / "gpcm_roundtrip.keras"
        approximator.save(model_path)

        _assert_fresh_process_reload_and_score(model_path, responses)

    def test_edge_case_all_correct_warning(self, trained_1pl_approximator):
        """All-correct responses should emit UserWarning."""
        approximator, simulator = trained_1pl_approximator

        # Manually create all-correct responses
        test_data = simulator.sample(1)
        n_items = test_data["responses"].shape[2]
        all_correct = np.ones((5, n_items), dtype=np.float64)

        scorer = PersonScorer(approximator)

        with pytest.warns(UserWarning, match="all-correct or all-incorrect"):
            results = scorer.score(all_correct, n_item_samples=50, theta_grid_size=21)

        # Estimates should be bounded
        assert np.all(results.mean < 6.0)
        assert np.all(results.mean > -6.0)

    def test_plausible_values_integration(self, trained_1pl_approximator):
        """Test that n_plausible_values returns theta plausible values."""
        approximator, simulator = trained_1pl_approximator

        test_data = simulator.sample(1)
        responses = test_data["responses"][0]

        scorer = PersonScorer(approximator)
        results = scorer.score(
            responses,
            n_item_samples=100,
            theta_grid_size=21,
            n_plausible_values=100,
        )

        assert results.plausible_values is not None
        assert results.plausible_values.shape == (
            responses.shape[0],
            100,
        )

    def test_custom_log_likelihood(self, trained_1pl_approximator):
        """Test custom log-likelihood function."""
        approximator, simulator = trained_1pl_approximator

        test_data = simulator.sample(1)
        responses = test_data["responses"][0]

        # Custom likelihood that just delegates to 1PL
        def custom_ll(theta, responses, **item_params):
            return _log_likelihood_1pl(theta, responses, **item_params)

        scorer = PersonScorer(approximator, log_likelihood=custom_ll)
        results = scorer.score(responses, n_item_samples=50, theta_grid_size=21)

        assert np.all(np.isfinite(results.mean))

    def test_loading_parameterization(self, trained_2pl_approximator):
        """Test scorer with loading parameterization."""
        approximator, _ = trained_2pl_approximator
        simulator = make_2pl_simulator(
            n_items=(10, 15),
            n_persons=(20, 30),
            prior_a=_PRIOR_A,
            prior_b=_PRIOR_B,
            prior_theta=_PRIOR_THETA,
        )

        # Retrain with loading parameterization
        adapter = make_2pl_adapter(
            simulator,
            log_transform_a=True,
            internal_parameterization="loading",
        )

        summary_net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=16,
            num_matrix_blocks=1,
            num_outer_sabs=1,
            outer_embed_dim=16,
            outer_mlp_width=32,
        )
        inference_net = bf.networks.FlowMatching(
            subnet_kwargs={"widths": (32, 32)},
        )

        loading_approx = EquivariantIRTApproximator(
            adapter=adapter,
            inference_network=inference_net,
            summary_network=summary_net,
        )
        loading_approx.compile(optimizer="adam")

        loading_approx.fit(
            simulator=simulator,
            epochs=1,
            num_batches=4,
            batch_size=10,
            workers=1,
            verbose=False,
        )

        # Test scorer with loading parameterization
        test_data = simulator.sample(1)
        responses = test_data["responses"][0]

        scorer = PersonScorer(loading_approx)
        results = scorer.score(responses, n_item_samples=50, theta_grid_size=21)

        assert np.all(np.isfinite(results.mean))


@pytest.mark.parametrize("model_type", ["PCM", "GPCM"])
@pytest.mark.parametrize("person_grids", [False, True])
@pytest.mark.parametrize("horner", [True, False])
def test_polytomous_chunk_matches_per_draw_likelihood(
    model_type, person_grids, horner, monkeypatch
):
    """Both normalizer forms (Horner and max-shifted) match the per-draw path."""
    import bayesflow_irt.scoring as scoring

    if not horner:
        monkeypatch.setattr(scoring, "_HORNER_MAX_EXPONENT", 0.0)
    rng = np.random.default_rng(7)
    n_draws, n_persons, n_items, n_cat = 5, 6, 4, 4
    delta = rng.normal(size=(n_draws, n_items, n_cat - 1))
    a = np.exp(rng.normal(0, 0.3, size=(n_draws, n_items)))
    responses = rng.integers(0, n_cat, size=(n_persons, n_items)).astype(float)
    responses[1, 2] = np.nan
    theta = (
        np.sort(rng.normal(size=(n_persons, 9)), axis=1)
        if person_grids
        else np.linspace(-3, 3, 9)
    )
    chunk = _log_likelihood_polytomous_chunk(
        theta,
        responses,
        model_type=model_type,
        delta_chunk=delta,
        a_chunk=a if model_type == "GPCM" else None,
    )
    for s in range(n_draws):
        params = {"delta": delta[s]}
        if model_type == "GPCM":
            params["a"] = a[s]
        expected = _log_likelihood_all_persons(
            theta, responses, params, model_type, None
        )
        assert_allclose(chunk[s], expected, rtol=1e-10, atol=1e-10)


def test_polytomous_chunk_extreme_values_stay_finite():
    """Exponents past the Horner limit fall back to the max-shifted form."""
    delta = np.array([[[-300.0, 0.0, 300.0]]])
    a = np.array([[2.0]])
    theta = np.array([-200.0, 0.0, 200.0])
    responses = np.array([[0.0], [3.0]])
    chunk = _log_likelihood_polytomous_chunk(
        theta, responses, model_type="GPCM", delta_chunk=delta, a_chunk=a
    )
    expected = _log_likelihood_all_persons(
        theta, responses, {"delta": delta[0], "a": a[0]}, "GPCM", None
    )
    assert np.all(np.isfinite(chunk))
    assert_allclose(chunk[0], expected, rtol=1e-10)


@pytest.mark.parametrize(
    "intervals, eti, hdi",
    [
        (("eti", "hdi"), True, True),
        ("eti", True, False),
        (["hdi"], False, True),
        (None, False, False),
        ((), False, False),
    ],
)
def test_intervals_selects_the_computed_intervals(intervals, eti, hdi):
    rng = np.random.default_rng(0)
    draws = {"b": rng.normal(0, 1, (8, 10))}
    x = rng.integers(0, 2, (5, 10)).astype(float)
    scorer = PersonScorer(None, item_posteriors=draws, model_type="1PL")
    ref = scorer.score(x, eti_level=[0.5, 0.9])
    out = scorer.score(x, eti_level=[0.5, 0.9], intervals=intervals)
    for kind, on in (("eti", eti), ("hdi", hdi)):
        for side in ("lower", "upper"):
            value = getattr(out, f"{kind}_{side}")
            if on:
                assert_allclose(value, getattr(ref, f"{kind}_{side}"))
            else:
                assert value is None
        assert len(getattr(out, f"{kind}_levels")) == (2 if on else 0)
    assert_allclose(out.mean, ref.mean)
    assert_allclose(out.sd, ref.sd)


def test_intervals_rejects_unknown_values():
    scorer = PersonScorer(None, item_posteriors={"b": np.zeros((2, 3))})
    with pytest.raises(ValueError, match="intervals"):
        scorer.score(np.zeros((1, 3)), intervals="ci")
