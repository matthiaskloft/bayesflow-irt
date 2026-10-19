"""Tests for bayesflow_irt.adapters."""

import numpy as np
import pytest

from bayesflow_irt.adapters import (
    LogTransform,
    OneHotEncode,
    _GPCMLoadingReparamTransform,
    _prior_moments_d,
    _prior_moments_d_gpcm,
    make_1pl_adapter,
    make_2pl_adapter,
    make_gpcm_adapter,
    make_pcm_adapter,
)
from bayesflow_irt.priors import Prior, VariablePrior, make_variable_incremental_prior
from bayesflow_irt.simulators import (
    make_1pl_simulator,
    make_2pl_simulator,
    make_gpcm_simulator,
    make_pcm_simulator,
)

# Standard test priors — must be specified explicitly at every call site.
_PRIOR_A = Prior("lognormal", loc=0, scale=0.5)
_PRIOR_B = Prior("normal", loc=0, scale=2.0)
_PRIOR_THETA = Prior("normal", loc=0, scale=1.0)
_PRIOR_DELTA = Prior("normal", loc=0, scale=2.0)


def _make_2pl_sim(n_items=10, n_persons=30):
    return make_2pl_simulator(
        n_items=(n_items, n_items + 2),
        n_persons=(n_persons, n_persons + 2),
        prior_a=_PRIOR_A,
        prior_b=_PRIOR_B,
        prior_theta=_PRIOR_THETA,
    )


def _make_1pl_sim(n_items=10, n_persons=30):
    return make_1pl_simulator(
        n_items=(n_items, n_items + 2),
        n_persons=(n_persons, n_persons + 2),
        prior_b=_PRIOR_B,
        prior_theta=_PRIOR_THETA,
    )


# ---------------------------------------------------------------------------
# LogTransform
# ---------------------------------------------------------------------------


class TestLogTransform:
    def test_forward_applies_log(self):
        """Forward: data -> log(data)."""
        transform = LogTransform()
        x = np.array([1.0, 2.0, 0.5], dtype=np.float32).reshape(1, 3, 1)
        out = transform.forward(x)
        np.testing.assert_allclose(out, np.log(x), rtol=1e-6)

    def test_inverse_applies_exp(self):
        """Inverse: data -> exp(data)."""
        transform = LogTransform()
        x = np.array([0.0, 0.5, -0.5], dtype=np.float32).reshape(1, 3, 1)
        out = transform.inverse(x)
        np.testing.assert_allclose(out, np.exp(x), rtol=1e-6)

    def test_round_trip(self):
        """forward -> inverse is identity for positive values."""
        transform = LogTransform()
        x = np.random.uniform(0.1, 5.0, (4, 10, 1)).astype(np.float32)
        np.testing.assert_allclose(
            transform.inverse(transform.forward(x)), x, rtol=1e-5
        )

    def test_inverse_guarantees_positive(self):
        """Inverse (exp) always returns positive values."""
        transform = LogTransform()
        x = np.random.randn(3, 5, 1).astype(np.float32)
        out = transform.inverse(x)
        assert np.all(out > 0)

    def test_get_config(self):
        transform = LogTransform()
        assert transform.get_config() == {}


# ---------------------------------------------------------------------------
# 2PL Adapter
# ---------------------------------------------------------------------------


class TestMake2PLAdapter:
    @staticmethod
    def _sim_and_sample(batch=4, n_items=10, n_persons=30):
        sim = _make_2pl_sim(n_items, n_persons)
        return sim, sim.sample((batch,))

    def test_forward_keys(self):
        """Forward should produce exactly the three BayesFlow keys."""
        sim, raw = self._sim_and_sample()
        adapter = make_2pl_adapter(sim)
        out = adapter(raw)
        assert set(out.keys()) == {
            "inference_variables",
            "summary_variables",
            "inference_conditions",
        }

    def test_forward_inference_variables_shape(self):
        """inference_variables: (batch, n_items, 2) from stacking a and b."""
        sim, raw = self._sim_and_sample(batch=4)
        adapter = make_2pl_adapter(sim)
        out = adapter(raw)
        iv = out["inference_variables"]
        assert iv.ndim == 3
        assert iv.shape[0] == 4
        assert iv.shape[2] == 2  # param_dim = 2 (a, b)

    def test_forward_summary_variables_shape(self):
        """summary_variables: (batch, n_persons, n_items) — response matrix."""
        sim, raw = self._sim_and_sample(batch=4)
        adapter = make_2pl_adapter(sim)
        out = adapter(raw)
        sv = out["summary_variables"]
        assert sv.ndim == 3
        assert sv.shape[0] == 4

    def test_forward_inference_conditions_shape(self):
        """inference_conditions: (batch, 2) from [n_items, n_persons]."""
        sim, raw = self._sim_and_sample(batch=4)
        adapter = make_2pl_adapter(sim)
        out = adapter(raw)
        ic = out["inference_conditions"]
        assert ic.shape == (4, 2)

    def test_theta_dropped(self):
        """Theta (nuisance) should not appear in adapter output."""
        sim, raw = self._sim_and_sample()
        adapter = make_2pl_adapter(sim)
        out = adapter(raw)
        for key in out.keys():
            assert "theta" not in key.lower()

    def test_dtype_float32(self):
        """All outputs should be float32."""
        sim, raw = self._sim_and_sample()
        adapter = make_2pl_adapter(sim)
        out = adapter(raw)
        for v in out.values():
            assert v.dtype == np.float32

    def test_log_transform_a_applied(self):
        """Standard mode: first column is log(a), second column is b."""
        sim, raw = self._sim_and_sample(batch=4)
        adapter = make_2pl_adapter(
            sim, log_transform_a=True, internal_parameterization="standard"
        )
        out = adapter(raw)
        iv = out["inference_variables"]
        # With standardization, values should be roughly unit-normal
        assert iv.ndim == 3
        assert iv.shape[2] == 2

    def test_no_log_transform(self):
        """Standard mode, no log: first column should be raw a (standardized)."""
        sim, raw = self._sim_and_sample(batch=4)
        adapter = make_2pl_adapter(
            sim, log_transform_a=False, internal_parameterization="standard"
        )
        out = adapter(raw)
        iv = out["inference_variables"]
        assert iv.ndim == 3
        assert iv.shape[2] == 2

    def test_rejects_missing_simulator_attrs(self):
        """Adapter should reject objects without prior_a / prior_b."""
        with pytest.raises(TypeError, match="prior_a"):
            make_2pl_adapter(object())


# ---------------------------------------------------------------------------
# 1PL Adapter
# ---------------------------------------------------------------------------


class TestMake1PLAdapter:
    @staticmethod
    def _sim_and_sample(batch=4, n_items=10, n_persons=30):
        sim = _make_1pl_sim(n_items, n_persons)
        return sim, sim.sample((batch,))

    def test_forward_keys(self):
        sim, raw = self._sim_and_sample()
        adapter = make_1pl_adapter(sim)
        out = adapter(raw)
        assert set(out.keys()) == {
            "inference_variables",
            "summary_variables",
            "inference_conditions",
        }

    def test_shape(self):
        """inference_variables has param_dim=1."""
        sim, raw = self._sim_and_sample(batch=3)
        adapter = make_1pl_adapter(sim)
        out = adapter(raw)
        iv = out["inference_variables"]
        assert iv.ndim == 3
        assert iv.shape[0] == 3
        assert iv.shape[2] == 1

    def test_dtype_float32(self):
        sim, raw = self._sim_and_sample()
        adapter = make_1pl_adapter(sim)
        out = adapter(raw)
        for v in out.values():
            assert v.dtype == np.float32

    def test_rejects_missing_simulator_attrs(self):
        """Adapter should reject objects without prior_b."""
        with pytest.raises(TypeError, match="prior_b"):
            make_1pl_adapter(object())


# ---------------------------------------------------------------------------
# 2PL Adapter — loading reparameterization
# ---------------------------------------------------------------------------


class TestMake2PLAdapterLoading:
    """Tests for make_2pl_adapter(simulator, internal_parameterization='loading')."""

    @staticmethod
    def _sim_and_sample(batch=4, n_items=10, n_persons=30):
        sim = _make_2pl_sim(n_items, n_persons)
        return sim, sim.sample((batch,))

    def test_rejects_invalid_parameterization(self):
        sim = _make_2pl_sim()
        with pytest.raises(ValueError, match="internal_parameterization"):
            make_2pl_adapter(sim, internal_parameterization="unknown")

    def test_forward_keys(self):
        """Loading adapter output must have same three BayesFlow keys."""
        sim, raw = self._sim_and_sample()
        adapter = make_2pl_adapter(sim, internal_parameterization="loading")
        out = adapter(raw)
        assert set(out.keys()) == {
            "inference_variables",
            "summary_variables",
            "inference_conditions",
        }

    def test_forward_inference_variables_shape(self):
        """inference_variables shape is (batch, n_items, 2) — same as standard."""
        sim, raw = self._sim_and_sample(batch=5)
        adapter = make_2pl_adapter(sim, internal_parameterization="loading")
        out = adapter(raw)
        iv = out["inference_variables"]
        assert iv.ndim == 3
        assert iv.shape[0] == 5
        assert iv.shape[2] == 2

    def test_dtype_float32(self):
        sim, raw = self._sim_and_sample()
        adapter = make_2pl_adapter(sim, internal_parameterization="loading")
        out = adapter(raw)
        for v in out.values():
            assert v.dtype == np.float32

    def test_internal_representation_differs_from_standard(self):
        """Loading and standard adapters must encode differently in iv."""
        sim, raw = self._sim_and_sample(batch=4)
        iv_std = make_2pl_adapter(sim, internal_parameterization="standard")(raw)[
            "inference_variables"
        ]
        iv_load = make_2pl_adapter(sim, internal_parameterization="loading")(raw)[
            "inference_variables"
        ]
        # Second channel: standard = b, loading = d = -a*b; they differ whenever b != 0
        assert not np.allclose(iv_std[..., 1], iv_load[..., 1])

    def test_round_trip_recovers_a_and_b(self):
        """adapter → adapter(inverse=True) must recover original (a, b) values."""
        sim, raw = self._sim_and_sample(batch=6)
        adapter = make_2pl_adapter(sim, internal_parameterization="loading")
        adapted = adapter(raw)
        recovered = adapter(adapted, inverse=True)

        np.testing.assert_allclose(
            recovered["a"].astype(np.float32),
            raw["a"].astype(np.float32),
            rtol=1e-4,
            atol=1e-5,
            err_msg="Round-trip failed for 'a'",
        )
        np.testing.assert_allclose(
            recovered["b"].astype(np.float32),
            raw["b"].astype(np.float32),
            rtol=1e-4,
            atol=1e-5,
            err_msg="Round-trip failed for 'b'",
        )

    def test_no_log_transform_round_trip(self):
        """Round-trip also works when log_transform_a=False."""
        sim, raw = self._sim_and_sample(batch=4)
        adapter = make_2pl_adapter(
            sim,
            log_transform_a=False,
            internal_parameterization="loading",
        )
        recovered = adapter(adapter(raw), inverse=True)
        np.testing.assert_allclose(
            recovered["b"].astype(np.float32),
            raw["b"].astype(np.float32),
            rtol=1e-4,
            atol=1e-5,
        )

    def test_condition_only_input_does_not_crash_with_strict_false(self):
        """Condition-only dictionaries should not trigger loading reparameterization."""
        sim, raw = self._sim_and_sample(batch=2)
        adapter = make_2pl_adapter(sim, internal_parameterization="loading")

        # BayesFlow concatenate caches indices on first strict=True pass.
        adapter(raw)

        condition_only = {
            "responses": raw["responses"],
            "n_items": raw["n_items"],
            "n_persons": raw["n_persons"],
        }
        out = adapter(condition_only, strict=False)

        assert "summary_variables" in out
        assert "inference_conditions" in out


# ---------------------------------------------------------------------------
# OneHotEncode transform
# ---------------------------------------------------------------------------


class TestOneHotEncode:
    def test_forward_shape(self):
        """(B, n_persons, n_items) → (B, n_persons, n_items, n_categories)."""
        enc = OneHotEncode(n_categories=5)
        x = np.array([[[0, 1, 4], [2, 3, 0]]], dtype=np.float32)  # (1, 2, 3)
        out = enc.forward(x)
        assert out.shape == (1, 2, 3, 5)

    def test_forward_values(self):
        """Each row should be a valid one-hot vector."""
        enc = OneHotEncode(n_categories=3)
        x = np.array([0, 1, 2], dtype=np.float32)
        out = enc.forward(x)
        expected = np.eye(3, dtype=np.float32)
        np.testing.assert_array_equal(out, expected)

    def test_inverse_recovers_categories(self):
        """inverse(forward(x)) == x for integer categories."""
        enc = OneHotEncode(n_categories=5)
        x = np.array([[[0, 1, 4], [2, 3, 0]]], dtype=np.float32)
        out = enc.forward(x)
        recovered = enc.inverse(out)
        np.testing.assert_array_equal(recovered, x)

    def test_round_trip(self):
        """forward → inverse is identity on integer data."""
        enc = OneHotEncode(n_categories=7)
        rng = np.random.default_rng(42)
        x = rng.integers(0, 7, size=(4, 20, 10)).astype(np.float32)
        np.testing.assert_array_equal(enc.inverse(enc.forward(x)), x)

    def test_get_config(self):
        enc = OneHotEncode(n_categories=5)
        assert enc.get_config() == {"n_categories": 5}

    @pytest.mark.parametrize(
        "bad", [0.9, 1.9, -0.1, 3.0, -1.0, np.nan, np.inf], ids=str
    )
    def test_rejects_non_integer_or_out_of_range_values(self, bad):
        """Fractional, negative, out-of-range and non-finite values must raise
        rather than be truncated into a valid-looking category."""
        enc = OneHotEncode(n_categories=3)
        x = np.array([0.0, 1.0, bad], dtype=np.float32)
        with pytest.raises(ValueError, match="OneHotEncode.forward"):
            enc.forward(x)

    def test_output_dtype(self):
        enc = OneHotEncode(n_categories=4)
        x = np.array([0, 1, 2, 3], dtype=np.float32)
        out = enc.forward(x)
        assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# PCM Adapter
# ---------------------------------------------------------------------------


class TestMakePCMAdapter:
    @staticmethod
    def _sim_and_sample(batch=4, n_items=10, n_persons=30, n_categories=5):
        sim = make_pcm_simulator(
            n_items=(n_items, n_items + 2),
            n_persons=(n_persons, n_persons + 2),
            n_categories=n_categories,
            prior_delta=_PRIOR_DELTA,
        )
        return sim, sim.sample((batch,))

    def test_forward_keys(self):
        """Forward should produce exactly the three BayesFlow keys."""
        sim, raw = self._sim_and_sample()
        adapter = make_pcm_adapter(sim)
        out = adapter(raw)
        assert set(out.keys()) == {
            "inference_variables",
            "summary_variables",
            "inference_conditions",
        }

    def test_inference_variables_shape(self):
        """inference_variables: (batch, n_items, n_categories-1)."""
        sim, raw = self._sim_and_sample(batch=4, n_categories=5)
        adapter = make_pcm_adapter(sim)
        out = adapter(raw)
        iv = out["inference_variables"]
        assert iv.ndim == 3
        assert iv.shape[0] == 4
        assert iv.shape[2] == 5 - 1

    def test_summary_variables_shape(self):
        """summary_variables is one-hot encoded over response categories."""
        sim, raw = self._sim_and_sample(batch=4, n_categories=5)
        adapter = make_pcm_adapter(sim)
        out = adapter(raw)
        sv = out["summary_variables"]
        assert sv.ndim == 4
        assert sv.shape[0] == 4
        assert sv.shape[3] == 5

    def test_inference_conditions_shape(self):
        """inference_conditions: (batch, 2)."""
        sim, raw = self._sim_and_sample(batch=4)
        adapter = make_pcm_adapter(sim)
        out = adapter(raw)
        ic = out["inference_conditions"]
        assert ic.shape == (4, 2)

    def test_dtype_float32(self):
        """All outputs should be float32."""
        sim, raw = self._sim_and_sample()
        adapter = make_pcm_adapter(sim)
        out = adapter(raw)
        for v in out.values():
            assert v.dtype == np.float32

    def test_one_hot_sums_to_one(self):
        """Each one-hot vector in summary_variables should sum to 1."""
        sim, raw = self._sim_and_sample(n_categories=5)
        adapter = make_pcm_adapter(sim)
        out = adapter(raw)
        sv = out["summary_variables"]
        sums = sv.sum(axis=-1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-6)

    def test_rejects_missing_simulator_attrs(self):
        """Adapter should reject objects without prior_delta / n_categories."""
        with pytest.raises(TypeError, match="prior_delta"):
            make_pcm_adapter(object())


# ---------------------------------------------------------------------------
# Standardization correctness — both parameterizations
# ---------------------------------------------------------------------------


class TestAdapterStandardization:
    """Statistical check: after forward adapter with priors, each channel
    of inference_variables should be approximately N(0, 1).

    Uses a large number of samples (many batches × batch_size × n_items items)
    so the empirical mean and std are reliable.  Tolerances are 0.05
    on the mean and 0.03 on the std — tight enough to catch wrong prior
    moment calculations but loose enough to pass with 100k-200k observations.
    """

    _PRIOR_A = Prior("lognormal", loc=0, scale=0.5)
    _PRIOR_B = Prior("normal", loc=0, scale=2.0)
    _PRIOR_THETA = Prior("normal", loc=0, scale=1.0)
    _N_BATCHES = 120
    _BATCH_SIZE = 32

    @classmethod
    def _collect_channels(cls, internal_parameterization: str):
        sim = make_2pl_simulator(
            n_items=(18, 22),
            n_persons=(80, 100),
            prior_a=cls._PRIOR_A,
            prior_b=cls._PRIOR_B,
            prior_theta=cls._PRIOR_THETA,
        )
        adapter = make_2pl_adapter(
            sim,
            log_transform_a=True,
            internal_parameterization=internal_parameterization,
        )
        ch0_parts, ch1_parts = [], []
        for _ in range(cls._N_BATCHES):
            iv = adapter(sim.sample((cls._BATCH_SIZE,)))["inference_variables"]
            ch0_parts.append(iv[..., 0].ravel())
            ch1_parts.append(iv[..., 1].ravel())
        return np.concatenate(ch0_parts), np.concatenate(ch1_parts)

    def test_standard_channel0_is_unit_normal(self):
        """Standard mode: log(a) channel is standardized to ~N(0,1)."""
        ch0, _ = self._collect_channels("standard")
        assert abs(ch0.mean()) < 0.05, f"ch0 mean = {ch0.mean():.4f}"
        assert abs(ch0.std() - 1.0) < 0.03, f"ch0 std = {ch0.std():.4f}"

    def test_standard_channel1_is_unit_normal(self):
        """Standard mode: b channel is standardized to ~N(0,1)."""
        _, ch1 = self._collect_channels("standard")
        assert abs(ch1.mean()) < 0.05, f"ch1 (b) mean = {ch1.mean():.4f}"
        assert abs(ch1.std() - 1.0) < 0.03, f"ch1 (b) std = {ch1.std():.4f}"

    def test_loading_channel0_is_unit_normal(self):
        """Loading mode: log(a) channel is standardized to ~N(0,1)."""
        ch0, _ = self._collect_channels("loading")
        assert abs(ch0.mean()) < 0.05, f"ch0 mean = {ch0.mean():.4f}"
        assert abs(ch0.std() - 1.0) < 0.03, f"ch0 std = {ch0.std():.4f}"

    def test_loading_channel1_is_unit_normal(self):
        """Loading mode: d = -a*b channel is standardized to ~N(0,1)."""
        _, ch1 = self._collect_channels("loading")
        assert abs(ch1.mean()) < 0.05, f"ch1 (d) mean = {ch1.mean():.4f}"
        assert abs(ch1.std() - 1.0) < 0.03, f"ch1 (d) std = {ch1.std():.4f}"

    def test_prior_moments_d_matches_analytic(self):
        """Empirical std(d) should match sqrt(E[a^2] * Var[b]) analytically."""
        # For LogNormal(0, sigma): E[a^2] = exp(2*mu + 2*sigma^2) = exp(0.5)
        # For Normal(0, 2): Var[b] = 4
        d_mean, d_std = _prior_moments_d(self._PRIOR_A, self._PRIOR_B)
        analytic_std = np.sqrt(np.exp(2 * 0 + 2 * 0.5**2) * 2.0**2)
        assert abs(d_mean) < 0.05, f"d_mean = {d_mean:.4f}"
        assert abs(d_std - analytic_std) < 0.05, (
            f"d_std = {d_std:.4f}, analytic = {analytic_std:.4f}"
        )


# ---------------------------------------------------------------------------
# GPCM Adapter
# ---------------------------------------------------------------------------


def _make_gpcm_sim(n_items=10, n_persons=30, n_categories=5):
    return make_gpcm_simulator(
        n_items=(n_items, n_items + 2),
        n_persons=(n_persons, n_persons + 2),
        n_categories=n_categories,
        prior_a=_PRIOR_A,
        prior_delta=_PRIOR_DELTA,
    )


class TestMakeGPCMAdapter:
    @staticmethod
    def _sim_and_sample(batch=4, n_items=10, n_persons=30, n_categories=5):
        sim = _make_gpcm_sim(n_items, n_persons, n_categories)
        return sim, sim.sample((batch,))

    def test_forward_keys(self):
        sim, raw = self._sim_and_sample()
        adapter = make_gpcm_adapter(sim)
        out = adapter(raw)
        assert set(out.keys()) == {
            "inference_variables",
            "summary_variables",
            "inference_conditions",
        }

    def test_inference_variables_shape(self):
        """inference_variables: (batch, n_items, n_categories)."""
        n_categories = 5
        sim, raw = self._sim_and_sample(batch=4, n_categories=n_categories)
        adapter = make_gpcm_adapter(sim)
        out = adapter(raw)
        iv = out["inference_variables"]
        assert iv.ndim == 3
        assert iv.shape[0] == 4
        assert iv.shape[2] == n_categories

    def test_summary_variables_shape(self):
        """summary_variables: (batch, n_persons, n_items, n_categories) -- one-hot."""
        n_categories = 5
        sim, raw = self._sim_and_sample(batch=4, n_categories=n_categories)
        adapter = make_gpcm_adapter(sim)
        sv = adapter(raw)["summary_variables"]
        assert sv.ndim == 4
        assert sv.shape[3] == n_categories

    def test_dtype_float32(self):
        sim, raw = self._sim_and_sample()
        adapter = make_gpcm_adapter(sim)
        for v in adapter(raw).values():
            assert v.dtype == np.float32

    def test_rejects_missing_simulator_attrs(self):
        with pytest.raises(TypeError, match="prior_a"):
            make_gpcm_adapter(object())

    def test_rejects_invalid_parameterization(self):
        sim = _make_gpcm_sim()
        with pytest.raises(ValueError, match="internal_parameterization"):
            make_gpcm_adapter(sim, internal_parameterization="x")


class TestMakeGPCMAdapterLoading:
    @staticmethod
    def _sim_and_sample(batch=4, n_items=10, n_persons=30, n_categories=5):
        sim = _make_gpcm_sim(n_items, n_persons, n_categories)
        return sim, sim.sample((batch,))

    def test_round_trip_recovers_a_and_delta(self):
        sim, raw = self._sim_and_sample(batch=6)
        adapter = make_gpcm_adapter(
            sim,
            internal_parameterization="loading",
        )
        recovered = adapter(adapter(raw), inverse=True)
        np.testing.assert_allclose(
            recovered["a"],
            raw["a"],
            rtol=1e-4,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            recovered["delta"],
            raw["delta"],
            rtol=1e-4,
            atol=1e-5,
        )

    def test_standard_round_trip(self):
        sim, raw = self._sim_and_sample(batch=4)
        adapter = make_gpcm_adapter(
            sim,
            internal_parameterization="standard",
        )
        recovered = adapter(adapter(raw), inverse=True)
        np.testing.assert_allclose(
            recovered["delta"],
            raw["delta"],
            rtol=1e-4,
            atol=1e-5,
        )

    def test_internal_representation_differs(self):
        sim, raw = self._sim_and_sample(batch=4)
        iv_std = make_gpcm_adapter(
            sim,
            internal_parameterization="standard",
        )(raw)["inference_variables"]
        iv_load = make_gpcm_adapter(
            sim,
            internal_parameterization="loading",
        )(raw)["inference_variables"]
        assert not np.allclose(iv_std[..., 1:], iv_load[..., 1:])

    def test_condition_only_no_crash(self):
        sim, raw = self._sim_and_sample(batch=2)
        adapter = make_gpcm_adapter(
            sim,
            internal_parameterization="loading",
        )
        adapter(raw)  # prime
        cond = {
            "responses": raw["responses"],
            "n_items": raw["n_items"],
            "n_persons": raw["n_persons"],
        }
        out = adapter(cond, strict=False)
        assert "summary_variables" in out

    def test_loading_inverse_supports_sample_axis(self):
        transform = _GPCMLoadingReparamTransform()
        a = np.array(
            [[[1.5, 2.0], [0.5, 4.0]]],
            dtype=np.float32,
        )  # (B, S, n_items)
        delta = np.array(
            [[[[0.2, -0.1], [1.0, 0.5]], [[-0.4, 0.3], [0.25, -0.75]]]],
            dtype=np.float32,
        )  # (B, S, n_items, n_categories-1)

        forward = transform.forward({"a": a, "delta": delta})
        recovered = transform.inverse(forward)

        np.testing.assert_allclose(recovered["a"], a, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(recovered["delta"], delta, rtol=1e-6, atol=1e-6)


class TestGPCMAdapterStandardization:
    """GPCM adapter channels should be ~N(0, 1)."""

    _N_BATCHES = 120
    _BATCH_SIZE = 32

    @classmethod
    def _collect_channels(cls, internal_parameterization):
        sim = make_gpcm_simulator(
            n_items=(18, 22),
            n_persons=(80, 100),
            n_categories=5,
            prior_a=_PRIOR_A,
            prior_delta=_PRIOR_DELTA,
        )
        adapter = make_gpcm_adapter(
            sim,
            internal_parameterization=internal_parameterization,
        )
        ch0, ch_rest = [], []
        for _ in range(cls._N_BATCHES):
            iv = adapter(sim.sample((cls._BATCH_SIZE,)))["inference_variables"]
            ch0.append(iv[..., 0].ravel())
            ch_rest.append(iv[..., 1:].ravel())
        return np.concatenate(ch0), np.concatenate(ch_rest)

    def test_standard_channel0(self):
        ch0, _ = self._collect_channels("standard")
        assert abs(ch0.mean()) < 0.05
        assert abs(ch0.std() - 1.0) < 0.03

    def test_standard_delta_channels(self):
        _, ch = self._collect_channels("standard")
        assert abs(ch.mean()) < 0.05
        assert abs(ch.std() - 1.0) < 0.03

    def test_loading_channel0(self):
        ch0, _ = self._collect_channels("loading")
        assert abs(ch0.mean()) < 0.05
        assert abs(ch0.std() - 1.0) < 0.03

    def test_loading_d_delta_channels(self):
        _, ch = self._collect_channels("loading")
        assert abs(ch.mean()) < 0.05
        assert abs(ch.std() - 1.0) < 0.03

    def test_prior_moments_d_gpcm_sanity(self):
        d_mean, d_std = _prior_moments_d_gpcm(_PRIOR_A, _PRIOR_DELTA)
        assert np.isfinite(d_mean)
        assert np.isfinite(d_std) and d_std > 0


# ---------------------------------------------------------------------------
# Variable prior adapter integration
# ---------------------------------------------------------------------------


class TestMake2PLAdapterVariable:
    def test_conditions_include_sigma_keys(self):
        vp_a = VariablePrior("lognormal", role="a", scale_range=(0.1, 1.0))
        vp_b = VariablePrior("normal", role="b", scale_range=(0.5, 3.0))
        sim = make_2pl_simulator(
            n_items=(5, 8),
            n_persons=(10, 15),
            prior_a=vp_a,
            prior_b=vp_b,
        )
        adapter = make_2pl_adapter(sim)
        out = adapter(sim.sample((4,)))
        ic = out["inference_conditions"]
        # 2 (n_items, n_persons) + 2 (sigma_a, sigma_b) = 4
        assert ic.shape == (4, 4)

    def test_mixed_fixed_variable_conditions(self):
        vp_a = VariablePrior("lognormal", role="a", scale_range=(0.1, 1.0))
        sim = make_2pl_simulator(
            n_items=(5, 8),
            n_persons=(10, 15),
            prior_a=vp_a,
            prior_b=_PRIOR_B,
        )
        adapter = make_2pl_adapter(sim)
        out = adapter(sim.sample((4,)))
        ic = out["inference_conditions"]
        # 2 + 1 (sigma_a only) = 3
        assert ic.shape == (4, 3)

    def test_round_trip_with_variable_prior(self):
        vp_b = VariablePrior("normal", role="b", scale_range=(0.5, 3.0))
        sim = make_2pl_simulator(
            n_items=(5, 8),
            n_persons=(10, 15),
            prior_a=_PRIOR_A,
            prior_b=vp_b,
        )
        adapter = make_2pl_adapter(sim, internal_parameterization="loading")
        raw = sim.sample((4,))
        recovered = adapter(adapter(raw), inverse=True)
        np.testing.assert_allclose(
            recovered["a"],
            raw["a"],
            rtol=1e-4,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            recovered["b"],
            raw["b"],
            rtol=1e-4,
            atol=1e-5,
        )


class TestSizeConditionStandardization:
    """`n_items` / `n_persons` must reach the network on the same scale as
    everything else.

    Passed raw they arrive at 5-500 while parameters and responses sit near
    unit scale. Under fixed size that is absorbed by the first-layer bias;
    under variable size it is a live, badly scaled input, so all inputs are
    kept on a comparable scale.
    """

    def _sim(self, **kwargs):
        defaults = dict(
            n_items=(5, 50),
            n_persons=(10, 500),
            prior_a=_PRIOR_A,
            prior_b=_PRIOR_B,
            prior_theta=_PRIOR_THETA,
            seed=7,
        )
        return make_2pl_simulator(**{**defaults, **kwargs})

    def test_default_brings_sizes_to_unit_scale(self):
        sim = self._sim()
        adapter = make_2pl_adapter(sim)

        cols = np.concatenate(
            [
                np.asarray(adapter(sim.sample((16,)))["inference_conditions"])
                for _ in range(12)
            ]
        )

        # Raw draws span 5-500; standardized ones must not.
        assert np.abs(cols).max() < 4.0
        assert abs(float(cols[:, 0].mean())) < 1.0
        assert abs(float(cols[:, 1].mean())) < 1.0

    @pytest.mark.parametrize(
        "factory",
        [make_2pl_adapter, make_1pl_adapter, make_pcm_adapter, make_gpcm_adapter],
    )
    def test_raw_size_conditioning_cannot_be_requested(self, factory):
        """There is no opt-out, and re-adding one must be a deliberate act.

        `standardize_size_conditions=False` would request raw conditioning;
        there is no such option, rather than a parameter whose only correct
        value is its default.
        """
        with pytest.raises(TypeError, match="standardize_size_conditions"):
            factory(self._sim(), standardize_size_conditions=False)

    def test_variable_prior_meta_keys_are_not_rescaled(self):
        """The transform is per-key, before concatenation.

        Applying it to the concatenated tensor would silently rescale the
        sigma columns with size moments.
        """
        vp_a = VariablePrior("lognormal", role="a", scale_range=(0.1, 1.0))
        sim = self._sim(prior_a=vp_a)
        batch = sim.sample((8,))

        ic = np.asarray(make_2pl_adapter(sim)(batch)["inference_conditions"])

        assert ic.shape == (8, 3)
        # sigma_a is column 2 and must arrive exactly as the simulator drew it.
        sigma_a = np.asarray(batch["sigma_a"]).reshape(8)
        np.testing.assert_allclose(ic[:, 2], sigma_a, rtol=1e-6)
        # The size columns, by contrast, must not be their raw draws.
        assert np.abs(ic[:, :2]).max() < 4.0

    def test_pinned_size_is_finite_and_centered(self):
        """A fixed-size simulator has zero spread in log size.

        Dividing by that sd would produce inf/nan; the transform must reduce to
        a pure shift instead.
        """
        sim = self._sim(n_items=(20, 20), n_persons=(300, 300))
        adapter = make_2pl_adapter(sim)

        ic = np.asarray(adapter(sim.sample((8,)))["inference_conditions"])

        assert np.all(np.isfinite(ic))
        np.testing.assert_allclose(ic, 0.0, atol=1e-5)

    def test_moments_follow_the_bucketed_sampler_not_the_integer_range(self):
        """Bucketed draws are uniform over bucket representatives, not integers.

        Using the integer-range moments for a bucketed simulator would
        standardize against a distribution the simulator never draws from.
        """
        from bayesflow_irt.adapters import _log_size_moments, _size_support
        from bayesflow_irt.simulators import _bucket_representatives

        assert _size_support((5, 50), None).size == 46
        bucket_support = _size_support((5, 50), 15)
        # ceil(46 / 15) = 4 log-spaced values from lo to hi.
        np.testing.assert_array_equal(bucket_support, [5.0, 11.0, 23.0, 50.0])
        np.testing.assert_array_equal(
            bucket_support, _bucket_representatives(5, 50, 15)
        )

        assert _log_size_moments((5, 50), 15) != _log_size_moments((5, 50), None)

    def test_log_uniform_moments_are_probability_weighted(self):
        """Log-uniform draws share the uniform support but not its weights, so
        the moments must come from the draws' probabilities: log(size) is then
        close to uniform on [log lo, log(hi + 1)]."""
        from bayesflow_irt.adapters import _log_size_moments
        from bayesflow_irt.simulators import _draw_bucketed_size

        mean, std = _log_size_moments((10, 500), None, "log_uniform")
        assert (mean, std) != _log_size_moments((10, 500), None)
        rng = np.random.default_rng(3)
        logs = np.log(
            [
                _draw_bucketed_size(rng, (10, 500), None, "log_uniform")
                for _ in range(20000)
            ]
        )
        assert logs.mean() == pytest.approx(mean, abs=0.02)
        assert logs.std() == pytest.approx(std, abs=0.02)

    @pytest.mark.parametrize(
        "factory,sim_factory,extra",
        [
            (make_1pl_adapter, make_1pl_simulator, {}),
            (make_pcm_adapter, make_pcm_simulator, {"n_categories": 4}),
        ],
    )
    def test_other_families_are_fixed_too(self, factory, sim_factory, extra):
        """All four adapters concatenated raw sizes, so all four needed it."""
        kwargs = dict(n_items=(5, 50), n_persons=(10, 500), seed=7, **extra)
        if factory is make_1pl_adapter:
            kwargs.update(prior_b=_PRIOR_B, prior_theta=_PRIOR_THETA)
        else:
            kwargs.update(prior_delta=_PRIOR_DELTA, prior_theta=_PRIOR_THETA)
        sim = sim_factory(**kwargs)

        batch = sim.sample((16,))
        ic = np.asarray(factory(sim)(batch)["inference_conditions"])

        assert np.abs(ic[:, :2]).max() < 4.0
        # The raw draws these came from are far outside that range, so the
        # bound above is evidence of the transform rather than of small inputs.
        raw_sizes = np.stack(
            [
                np.asarray(batch["n_items"]).reshape(16),
                np.asarray(batch["n_persons"]).reshape(16),
            ],
            axis=1,
        )
        assert raw_sizes.max() > 4.0


class TestMake1PLAdapterVariable:
    def test_conditions_include_sigma_b(self):
        vp_b = VariablePrior("normal", role="b", scale_range=(0.5, 3.0))
        sim = make_1pl_simulator(
            n_items=(5, 8),
            n_persons=(10, 15),
            prior_b=vp_b,
        )
        adapter = make_1pl_adapter(sim)
        out = adapter(sim.sample((4,)))
        ic = out["inference_conditions"]
        # 2 + 1 (sigma_b) = 3
        assert ic.shape == (4, 3)

    def test_fixed_prior_conditions_shape_unchanged(self):
        sim = make_1pl_simulator(
            n_items=(5, 8),
            n_persons=(10, 15),
            prior_b=_PRIOR_B,
        )
        adapter = make_1pl_adapter(sim)
        out = adapter(sim.sample((4,)))
        assert out["inference_conditions"].shape == (4, 2)


class TestMakePCMAdapterVariable:
    def test_conditions_include_sigma_keys(self):
        vip = make_variable_incremental_prior((0.5, 3.0), (0.3, 2.0), 5)
        sim = make_pcm_simulator(
            n_items=(5, 8),
            n_persons=(10, 15),
            n_categories=5,
            prior_delta=vip,
        )
        adapter = make_pcm_adapter(sim)
        out = adapter(sim.sample((4,)))
        ic = out["inference_conditions"]
        # 2 + 2 (sigma_mu, sigma_eps) = 4
        assert ic.shape == (4, 4)


class TestMakeGPCMAdapterVariable:
    def test_conditions_include_all_sigma_keys(self):
        vp_a = VariablePrior("lognormal", role="a", scale_range=(0.1, 1.0))
        vip = make_variable_incremental_prior((0.5, 3.0), (0.3, 2.0), 5)
        sim = make_gpcm_simulator(
            n_items=(5, 8),
            n_persons=(10, 15),
            n_categories=5,
            prior_a=vp_a,
            prior_delta=vip,
        )
        adapter = make_gpcm_adapter(sim)
        out = adapter(sim.sample((4,)))
        ic = out["inference_conditions"]
        # 2 + 1 (sigma_a) + 2 (sigma_mu, sigma_eps) = 5
        assert ic.shape == (4, 5)

    def test_mixed_fixed_variable_conditions(self):
        vip = make_variable_incremental_prior((0.5, 3.0), (0.3, 2.0), 5)
        sim = make_gpcm_simulator(
            n_items=(5, 8),
            n_persons=(10, 15),
            n_categories=5,
            prior_a=_PRIOR_A,
            prior_delta=vip,
        )
        adapter = make_gpcm_adapter(sim)
        out = adapter(sim.sample((4,)))
        ic = out["inference_conditions"]
        # 2 + 2 (sigma_mu, sigma_eps) = 4
        assert ic.shape == (4, 4)
