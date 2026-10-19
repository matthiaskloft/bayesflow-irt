"""Tests for bayesflow_irt.priors."""

import numpy as np
import pytest

from bayesflow_irt.priors import (
    Prior,
    VariablePrior,
    make_incremental_prior,
    make_variable_incremental_prior,
    prior_moments,
    resolve_prior,
    sample_prior_predictive,
)

# ---------------------------------------------------------------------------
# Prior dataclass
# ---------------------------------------------------------------------------


class TestPriorDataclass:
    def test_basic_construction(self):
        p = Prior("normal", loc=0, scale=2)
        assert p.name == "normal"
        assert p.kwargs == {"loc": 0, "scale": 2}

    def test_no_kwargs(self):
        p = Prior("lognormal")
        assert p.kwargs == {}

    def test_frozen(self):
        p = Prior("normal")
        with pytest.raises(AttributeError):
            p.name = "other"

    def test_repr_with_kwargs(self):
        p = Prior("normal", loc=0, scale=2)
        r = repr(p)
        assert "normal" in r
        assert "loc=0" in r
        assert "scale=2" in r

    def test_repr_no_kwargs(self):
        assert repr(Prior("lognormal")) == "Prior('lognormal')"


# ---------------------------------------------------------------------------
# resolve_prior — Prior dataclass
# ---------------------------------------------------------------------------


class TestResolvePriorDataclass:
    def test_lognormal_a(self):
        sampler = resolve_prior(Prior("lognormal", loc=0, scale=0.5), "a")
        rng = np.random.default_rng(42)
        samples = sampler((500,), rng)
        assert samples.shape == (500,)
        assert np.all(samples > 0)

    def test_halfnorm_a(self):
        sampler = resolve_prior(Prior("halfnorm"), "a")
        rng = np.random.default_rng(42)
        samples = sampler((500,), rng)
        assert np.all(samples >= 0)

    def test_truncnorm_a(self):
        sampler = resolve_prior(Prior("truncnorm"), "a")
        rng = np.random.default_rng(42)
        samples = sampler((500,), rng)
        assert np.all(samples >= 0.1)

    def test_normal_b(self):
        sampler = resolve_prior(Prior("normal", loc=0, scale=2.0), "b")
        rng = np.random.default_rng(42)
        samples = sampler((5000,), rng)
        assert abs(samples.std() - 2.0) < 0.2

    def test_normal_b_no_kwargs_uses_function_defaults(self):
        """Prior('normal') without kwargs uses the function default scale=1.0."""
        sampler = resolve_prior(Prior("normal"), "b")
        rng = np.random.default_rng(42)
        samples = sampler((5000,), rng)
        # Function default: _sample_normal has scale=1.0
        assert abs(samples.std() - 1.0) < 0.1

    def test_normal_theta(self):
        sampler = resolve_prior(Prior("normal", loc=0, scale=1.0), "theta")
        rng = np.random.default_rng(42)
        samples = sampler((5000,), rng)
        assert abs(samples.std() - 1.0) < 0.1

    def test_custom_hyperparams(self):
        sampler = resolve_prior(Prior("normal", loc=5, scale=0.1), "b")
        rng = np.random.default_rng(42)
        samples = sampler((1000,), rng)
        assert abs(samples.mean() - 5.0) < 0.05

    def test_2d_shape(self):
        sampler = resolve_prior(Prior("lognormal"), "a")
        rng = np.random.default_rng(42)
        samples = sampler((8, 20), rng)
        assert samples.shape == (8, 20)
        assert np.all(samples > 0)


# ---------------------------------------------------------------------------
# resolve_prior — callable
# ---------------------------------------------------------------------------


class TestResolvePriorCallable:
    def test_basic_callable(self):
        def my_prior(shape, rng):
            return rng.exponential(1.0, size=shape)

        sampler = resolve_prior(my_prior, "a")
        rng = np.random.default_rng(42)
        samples = sampler((100,), rng)
        assert samples.shape == (100,)
        assert np.all(samples > 0)

    def test_callable_wrong_shape_raises(self):
        def bad_prior(shape, rng):
            return rng.normal(size=(3,))

        sampler = resolve_prior(bad_prior, "b")
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="shape"):
            sampler((100,), rng)

    def test_callable_negative_a_raises(self):
        def neg_prior(shape, rng):
            return -np.abs(rng.normal(size=shape))

        sampler = resolve_prior(neg_prior, "a")
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="positive"):
            sampler((10,), rng)

    def test_callable_negative_b_allowed(self):
        """Negative values are fine for difficulty."""

        def neg_prior(shape, rng):
            return -np.abs(rng.normal(size=shape))

        sampler = resolve_prior(neg_prior, "b")
        rng = np.random.default_rng(42)
        samples = sampler((50,), rng)
        assert np.all(samples <= 0)

    def test_callable_returns_list(self):
        """Callables returning lists are auto-converted to ndarray."""

        def list_prior(shape, rng):
            return [1.0] * shape[0]

        sampler = resolve_prior(list_prior, "a")
        rng = np.random.default_rng(42)
        samples = sampler((5,), rng)
        assert isinstance(samples, np.ndarray)


# ---------------------------------------------------------------------------
# resolve_prior — validation
# ---------------------------------------------------------------------------


class TestResolvePriorValidation:
    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="Prior or callable"):
            resolve_prior(42, "b")

    def test_string_raises(self):
        """Strings are not accepted as PriorSpec."""
        with pytest.raises(TypeError, match="Prior or callable"):
            resolve_prior("normal", "b")

    def test_unknown_distribution_raises(self):
        with pytest.raises(ValueError, match="Unknown distribution"):
            resolve_prior(Prior("cauchy"), "b")

    def test_role_mismatch_normal_a(self):
        with pytest.raises(ValueError, match="not valid for parameter role"):
            resolve_prior(Prior("normal"), "a")

    def test_role_mismatch_student_t_a(self):
        with pytest.raises(ValueError, match="not valid for parameter role"):
            resolve_prior(Prior("student_t"), "a")

    def test_uniform_a_negative_low_raises(self):
        with pytest.raises(ValueError, match="low > 0"):
            resolve_prior(Prior("uniform", low=-1, high=2), "a")


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------


class TestDistributions:
    @pytest.mark.parametrize(
        "prior,role",
        [
            (Prior("uniform", low=-4, high=4), "b"),
            (Prior("uniform", low=-4, high=4), "theta"),
            (Prior("uniform", low=0.3, high=3.0), "a"),
            (Prior("student_t", df=3, loc=0, scale=1), "b"),
            (Prior("student_t", df=3, loc=0, scale=1), "theta"),
            (Prior("gamma", concentration=2, scale=0.5), "a"),
        ],
    )
    def test_samples_correct_shape(self, prior, role):
        sampler = resolve_prior(prior, role)
        rng = np.random.default_rng(42)
        samples = sampler((500,), rng)
        assert samples.shape == (500,)
        if role == "a":
            assert np.all(samples > 0)

    def test_gamma_custom_params(self):
        sampler = resolve_prior(Prior("gamma", concentration=4, scale=0.25), "a")
        rng = np.random.default_rng(42)
        samples = sampler((5000,), rng)
        assert np.all(samples > 0)
        # Gamma(4, 0.25) has mean=1.0
        assert abs(samples.mean() - 1.0) < 0.1

    def test_student_t_heavier_tails(self):
        """Student-t(df=3) should have heavier tails than normal."""
        rng = np.random.default_rng(42)
        t_sampler = resolve_prior(Prior("student_t", df=3, loc=0, scale=1), "b")
        t_samples = t_sampler((10_000,), rng)
        assert np.sum(np.abs(t_samples) > 5) > 50


# ---------------------------------------------------------------------------
# sample_prior_predictive
# ---------------------------------------------------------------------------


class TestSamplePriorPredictive:
    def test_basic_usage(self):
        samples = sample_prior_predictive(
            Prior("lognormal"), "a", n_samples=500, seed=0
        )
        assert samples.shape == (500,)
        assert np.all(samples > 0)

    def test_with_prior_object(self):
        samples = sample_prior_predictive(
            Prior("normal", loc=2, scale=0.5), "b", n_samples=1000, seed=0
        )
        assert abs(samples.mean() - 2.0) < 0.1

    def test_reproducible_with_seed(self):
        p = Prior("normal", loc=0, scale=1)
        s1 = sample_prior_predictive(p, "b", seed=123)
        s2 = sample_prior_predictive(p, "b", seed=123)
        np.testing.assert_array_equal(s1, s2)

    def test_default_n_samples(self):
        samples = sample_prior_predictive(
            Prior("normal", loc=0, scale=1), "theta", seed=0
        )
        assert samples.shape == (10_000,)


# ---------------------------------------------------------------------------
# make_incremental_prior
# ---------------------------------------------------------------------------


class TestMakeIncrementalPrior:
    """Tests for the incremental (mean + cumsum) compound prior."""

    def test_output_shape_3d(self):
        sampler = make_incremental_prior(
            Prior("normal", loc=0, scale=1),
            Prior("normal", loc=0, scale=1),
            n_categories=5,
        )
        rng = np.random.default_rng(42)
        result = sampler((8, 10, 4), rng)
        assert result.shape == (8, 10, 4)

    def test_centering_matches_mean_prior(self):
        """Mean of output thresholds ≈ sampled mean prior location."""
        sampler = make_incremental_prior(
            Prior("normal", loc=3.0, scale=0.5),
            Prior("normal", loc=0, scale=1),
            n_categories=4,
        )
        rng = np.random.default_rng(42)
        result = sampler((1000, 20, 3), rng)
        # Per-item mean across steps
        item_means = result.mean(axis=-1)  # (1000, 20)
        assert abs(item_means.mean() - 3.0) < 0.1

    def test_sample_prior_predictive_1d(self):
        """1D shape works for sample_prior_predictive / prior_moments."""
        sampler = make_incremental_prior(
            Prior("normal", loc=0, scale=1),
            Prior("normal", loc=0, scale=1),
            n_categories=5,
        )
        rng = np.random.default_rng(42)
        result = sampler((10_000,), rng)
        assert result.shape == (10_000,)
        assert np.isfinite(result).all()

    def test_prior_moments_finite(self):
        """prior_moments returns finite (mean, std) for the compound prior."""
        sampler = make_incremental_prior(
            Prior("normal", loc=0, scale=1.5),
            Prior("normal", loc=0, scale=1),
            n_categories=5,
        )
        mean, std = prior_moments(sampler, "delta")
        assert np.isfinite(mean)
        assert np.isfinite(std) and std > 0

    def test_k2_edge_case(self):
        """K=2: single threshold, no increments, delta ≈ mu."""
        sampler = make_incremental_prior(
            Prior("normal", loc=2.0, scale=0.5),
            Prior("normal", loc=0, scale=1),
            n_categories=2,
        )
        rng = np.random.default_rng(42)
        result = sampler((500, 10, 1), rng)
        assert result.shape == (500, 10, 1)
        # Should be drawn from the mean prior only
        assert abs(result.mean() - 2.0) < 0.15
        assert abs(result.std() - 0.5) < 0.1

    def test_mismatched_k_raises(self):
        """Passing a 3D shape with wrong last dim raises ValueError."""
        sampler = make_incremental_prior(
            Prior("normal"),
            Prior("normal"),
            n_categories=5,
        )
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="K-1=4"):
            sampler((8, 10, 3), rng)

    def test_invalid_shape_raises(self):
        sampler = make_incremental_prior(
            Prior("normal"),
            Prior("normal"),
            n_categories=3,
        )
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="1D or 3D"):
            sampler((8, 10), rng)

    def test_scaling_stabilizes_variance(self):
        """Marginal std of thresholds shouldn't grow much with K."""
        rng = np.random.default_rng(42)
        stds = {}
        for n_categories in [3, 10]:
            sampler = make_incremental_prior(
                Prior("normal", loc=0, scale=1),
                Prior("normal", loc=0, scale=1),
                n_categories=n_categories,
            )
            result = sampler((2000, 20, n_categories - 1), rng)
            stds[n_categories] = float(result.std())

        # Without scaling, K=10 would have ~sqrt(9/2) ≈ 2.1× the std of K=3.
        # With scaling, the ratio should be much closer to 1.
        ratio = stds[10] / stds[3]
        assert ratio < 1.5, f"Scaling failed: std ratio K=10/K=3 = {ratio:.2f}"

    def test_invalid_n_categories(self):
        with pytest.raises(ValueError, match="n_categories"):
            make_incremental_prior(Prior("normal"), Prior("normal"), n_categories=1)

    @pytest.mark.parametrize("n_categories", [3, 4, 5, 7, 10])
    def test_thresholds_are_sorted_for_all_k(self, n_categories):
        """Thresholds must be non-decreasing along the last axis for all K."""
        sampler = make_incremental_prior(
            Prior("normal", loc=0, scale=2.0),
            Prior("normal", loc=2.0, scale=0.5),
            n_categories=n_categories,
        )
        rng = np.random.default_rng(0)
        delta = sampler((500, 20, n_categories - 1), rng)
        diffs = np.diff(delta, axis=-1)
        assert np.all(diffs >= 0), (
            f"K={n_categories}: found {(diffs < 0).sum()} disordered threshold pairs"
        )

    @pytest.mark.parametrize("n_categories", [3, 5, 7])
    def test_constant_expected_span_across_k(self, n_categories):
        """E[max(delta) - min(delta)] ≈ loc for all K >= 3."""
        loc = 2.0
        sampler = make_incremental_prior(
            Prior("normal", loc=0, scale=2.0),
            Prior("normal", loc=loc, scale=0.5),
            n_categories=n_categories,
        )
        rng = np.random.default_rng(0)
        delta = sampler((2000, 20, n_categories - 1), rng)
        span = delta[..., -1] - delta[..., 0]  # (2000, 20)
        mean_span = float(span.mean())
        assert abs(mean_span - loc) < 0.15, (
            f"K={n_categories}: mean span={mean_span:.3f}, expected ≈{loc}"
        )

    def test_end_to_end_with_pcm_simulator(self):
        """Compound prior works with make_pcm_simulator + make_pcm_adapter."""
        from bayesflow_irt import make_pcm_adapter, make_pcm_simulator

        prior_delta = make_incremental_prior(
            Prior("normal", loc=0, scale=1.0),
            Prior("normal", loc=0, scale=1.0),
            n_categories=4,
        )
        sim = make_pcm_simulator(
            n_items=(5, 8),
            n_persons=(10, 15),
            n_categories=4,
            prior_delta=prior_delta,
        )
        sample = sim.sample((4,))
        assert sample["delta"].shape[2] == 3  # K-1 = 3

        adapter = make_pcm_adapter(sim)
        adapted = adapter(sample)
        assert "inference_variables" in adapted
        assert adapted["inference_variables"].shape[2] == 3


# ---------------------------------------------------------------------------
# VariablePrior
# ---------------------------------------------------------------------------


class TestVariablePrior:
    """Tests for the VariablePrior class."""

    def test_construction(self):
        vp = VariablePrior("normal", role="b", scale_range=(0.5, 3.0))
        assert vp.name == "normal"
        assert vp.role == "b"
        assert vp.scale_range == (0.5, 3.0)

    def test_meta_keys(self):
        vp = VariablePrior("normal", role="b", scale_range=(0.5, 3.0))
        assert vp.meta_keys == ["sigma_b"]

    def test_meta_keys_role_a(self):
        vp = VariablePrior("lognormal", role="a", scale_range=(0.1, 1.0))
        assert vp.meta_keys == ["sigma_a"]

    def test_repr(self):
        vp = VariablePrior("normal", role="b", scale_range=(0.5, 3.0), loc=0.0)
        r = repr(vp)
        assert "VariablePrior" in r
        assert "normal" in r

    def test_unknown_distribution_raises(self):
        with pytest.raises(ValueError, match="Unknown distribution"):
            VariablePrior("cauchy", role="b", scale_range=(0.5, 3.0))

    def test_role_mismatch_raises(self):
        with pytest.raises(ValueError, match="not valid for role"):
            VariablePrior("normal", role="a", scale_range=(0.5, 3.0))

    def test_invalid_scale_range_zero_raises(self):
        with pytest.raises(ValueError, match="0 < lo < hi"):
            VariablePrior("normal", role="b", scale_range=(0.0, 3.0))

    def test_invalid_scale_range_inverted_raises(self):
        with pytest.raises(ValueError, match="0 < lo < hi"):
            VariablePrior("normal", role="b", scale_range=(3.0, 0.5))

    def test_marginal_sampling_shape(self):
        vp = VariablePrior("normal", role="b", scale_range=(0.5, 3.0))
        rng = np.random.default_rng(42)
        samples = vp((100,), rng)
        assert samples.shape == (100,)
        assert samples.dtype == np.float32

    def test_marginal_sampling_2d(self):
        vp = VariablePrior("lognormal", role="a", scale_range=(0.1, 1.0))
        rng = np.random.default_rng(42)
        samples = vp((8, 20), rng)
        assert samples.shape == (8, 20)
        assert np.all(samples > 0)

    def test_conditional_sampling_shape(self):
        vp = VariablePrior("normal", role="b", scale_range=(0.5, 3.0))
        rng = np.random.default_rng(42)
        samples, meta = vp.sample_with_meta((8, 20), rng)
        assert samples.shape == (8, 20)
        assert "sigma_b" in meta
        assert meta["sigma_b"].shape == (8,)

    def test_conditional_sigma_range(self):
        vp = VariablePrior("normal", role="b", scale_range=(0.5, 3.0))
        rng = np.random.default_rng(42)
        _, meta = vp.sample_with_meta((100, 10), rng)
        assert np.all(meta["sigma_b"] >= 0.5)
        assert np.all(meta["sigma_b"] <= 3.0)

    def test_conditional_1d_raises(self):
        vp = VariablePrior("normal", role="b", scale_range=(0.5, 3.0))
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="at least 2D"):
            vp.sample_with_meta((100,), rng)

    def test_resolve_prior_works(self):
        vp = VariablePrior("normal", role="b", scale_range=(0.5, 3.0))
        sampler = resolve_prior(vp, "b")
        rng = np.random.default_rng(42)
        samples = sampler((100,), rng)
        assert samples.shape == (100,)

    def test_prior_moments_works(self):
        vp = VariablePrior("normal", role="b", scale_range=(0.5, 3.0))
        mean, std = prior_moments(vp, "b", n_samples=5000, seed=0)
        assert np.isfinite(mean)
        assert np.isfinite(std) and std > 0

    def test_sample_prior_predictive_works(self):
        vp = VariablePrior("normal", role="b", scale_range=(0.5, 3.0))
        samples = sample_prior_predictive(vp, "b", n_samples=500, seed=0)
        assert samples.shape == (500,)


class TestMakeVariableIncrementalPrior:
    """Tests for make_variable_incremental_prior."""

    def test_construction(self):
        vip = make_variable_incremental_prior(
            mu_sigma_range=(0.5, 3.0),
            eps_sigma_range=(0.3, 2.0),
            n_categories=5,
        )
        assert vip.n_categories == 5
        assert vip.meta_keys == ["sigma_mu", "sigma_eps"]

    def test_invalid_n_categories(self):
        with pytest.raises(ValueError, match="n_categories"):
            make_variable_incremental_prior(
                mu_sigma_range=(0.5, 3.0),
                eps_sigma_range=(0.3, 2.0),
                n_categories=1,
            )

    def test_invalid_mu_sigma_range(self):
        with pytest.raises(ValueError, match="mu_sigma_range"):
            make_variable_incremental_prior(
                mu_sigma_range=(0.0, 3.0),
                eps_sigma_range=(0.3, 2.0),
                n_categories=5,
            )

    def test_invalid_eps_sigma_range(self):
        with pytest.raises(ValueError, match="eps_sigma_range"):
            make_variable_incremental_prior(
                mu_sigma_range=(0.5, 3.0),
                eps_sigma_range=(3.0, 0.3),
                n_categories=5,
            )

    def test_conditional_sampling_shape(self):
        vip = make_variable_incremental_prior(
            mu_sigma_range=(0.5, 3.0),
            eps_sigma_range=(0.3, 2.0),
            n_categories=5,
        )
        rng = np.random.default_rng(42)
        delta, meta = vip.sample_with_meta((4, 10, 4), rng)
        assert delta.shape == (4, 10, 4)
        assert "sigma_mu" in meta
        assert "sigma_eps" in meta
        assert meta["sigma_mu"].shape == (4,)
        assert meta["sigma_eps"].shape == (4,)

    def test_conditional_sigma_ranges(self):
        vip = make_variable_incremental_prior(
            mu_sigma_range=(0.5, 3.0),
            eps_sigma_range=(0.3, 2.0),
            n_categories=5,
        )
        rng = np.random.default_rng(42)
        _, meta = vip.sample_with_meta((100, 10, 4), rng)
        assert np.all(meta["sigma_mu"] >= 0.5) and np.all(meta["sigma_mu"] <= 3.0)
        assert np.all(meta["sigma_eps"] >= 0.3) and np.all(meta["sigma_eps"] <= 2.0)

    def test_marginal_sampling_1d(self):
        vip = make_variable_incremental_prior(
            mu_sigma_range=(0.5, 3.0),
            eps_sigma_range=(0.3, 2.0),
            n_categories=5,
        )
        rng = np.random.default_rng(42)
        samples = vip((1000,), rng)
        assert samples.shape == (1000,)
        assert np.isfinite(samples).all()

    def test_marginal_sampling_3d(self):
        vip = make_variable_incremental_prior(
            mu_sigma_range=(0.5, 3.0),
            eps_sigma_range=(0.3, 2.0),
            n_categories=5,
        )
        rng = np.random.default_rng(42)
        samples = vip((4, 10, 4), rng)
        assert samples.shape == (4, 10, 4)

    def test_conditional_wrong_shape_raises(self):
        vip = make_variable_incremental_prior(
            mu_sigma_range=(0.5, 3.0),
            eps_sigma_range=(0.3, 2.0),
            n_categories=5,
        )
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="3D shape"):
            vip.sample_with_meta((100,), rng)

    def test_resolve_prior_works(self):
        vip = make_variable_incremental_prior(
            mu_sigma_range=(0.5, 3.0),
            eps_sigma_range=(0.3, 2.0),
            n_categories=5,
        )
        sampler = resolve_prior(vip, "delta")
        rng = np.random.default_rng(42)
        samples = sampler((1000,), rng)
        assert samples.shape == (1000,)


class TestDiscriminationPositivity:
    @pytest.mark.parametrize("low", [-0.5, 0.0])
    def test_truncnorm_rejects_non_positive_low(self, low):
        with pytest.raises(ValueError, match="low > 0"):
            resolve_prior(Prior("truncnorm", loc=1, scale=0.5, low=low), "a")

    def test_truncnorm_default_low_accepted(self):
        sampler = resolve_prior(Prior("truncnorm", loc=1, scale=0.5), "a")
        assert (sampler((200,), np.random.default_rng(0)) > 0).all()

    def test_variable_truncnorm_rejects_negative_low(self):
        with pytest.raises(ValueError, match="low > 0"):
            VariablePrior("truncnorm", role="a", scale_range=(0.3, 0.6), low=-1.0)

    def test_uniform_omitted_low_reports_default(self):
        with pytest.raises(ValueError, match="low=0.0"):
            resolve_prior(Prior("uniform", high=2.0), "a")

    def test_uniform_negative_low_rejected(self):
        with pytest.raises(ValueError, match="low=-1"):
            resolve_prior(Prior("uniform", low=-1, high=2.0), "a")
