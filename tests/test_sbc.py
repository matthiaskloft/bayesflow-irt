"""Tests for pooled SBC helper utilities."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest
from scipy.stats import truncnorm

matplotlib.use("Agg")

import bayesflow_irt.sbc as sbc_module
from bayesflow_irt.sbc import (
    _compute_recovery_metrics,
    chunked_posterior_sample,
    classical_item_statistics,
    compute_tarp_coverage,
    plot_edge_parameter_diagnostics,
    pooled_sbc_for_parameter,
    pooled_sbc_random_conditions,
    run_person_theta_sbc,
    run_sbc_diagnostics,
    sample_edge_corner_batches,
    sample_random_size_conditions,
    sample_random_theta_sbc_conditions,
    summarize_edge_parameter_diagnostics,
)


class _DummySimulator:
    def __init__(self, n_items=(3, 3), n_persons=(5, 5)):
        self.n_items = n_items
        self.n_persons = n_persons

    def clone_with_sizes(self, *, n_items=None, n_persons=None):
        return _DummySimulator(
            n_items=n_items if n_items is not None else self.n_items,
            n_persons=n_persons if n_persons is not None else self.n_persons,
        )

    def sample(self, shape):
        if tuple(shape) != (1,):
            raise ValueError("Dummy simulator only supports sample((1,)).")
        n_items = int(self.n_items[0])
        n_persons = int(self.n_persons[0])
        b = np.linspace(-1.0, 1.0, n_items, dtype=np.float32)[np.newaxis, :]
        responses = np.zeros((1, n_persons, n_items), dtype=np.float32)
        return {
            "b": b,
            "responses": responses,
            "n_items": np.array([[n_items]], dtype=np.float32),
            "n_persons": np.array([[n_persons]], dtype=np.float32),
        }


class _DummyApproximator:
    def __init__(self):
        self.sample_calls = []

    def sample(self, *, num_samples, conditions, flow_sample_kwargs=None):
        self.sample_calls.append(
            {
                "num_samples": num_samples,
                "batch_size": len(next(iter(conditions.values()))),
                "flow_sample_kwargs": flow_sample_kwargs,
            }
        )
        b_true = np.asarray(conditions["b"], dtype=np.float32)
        b_draws = np.repeat(b_true[:, np.newaxis, :], repeats=num_samples, axis=1)
        # Construct deterministic ranks at num_samples // 2.
        b_draws[:, : num_samples // 2, :] -= 1.0
        b_draws[:, num_samples // 2 :, :] += 1.0
        return {"b": b_draws}


class _DummyEdgeApproximator:
    def __init__(self):
        self.sample_calls = []

    def sample(self, *, num_samples, conditions, flow_sample_kwargs=None):
        self.sample_calls.append(
            {
                "num_samples": num_samples,
                "flow_sample_kwargs": flow_sample_kwargs,
            }
        )
        out = {}
        for key in ("a", "b"):
            vals = np.asarray(conditions[key], dtype=np.float32)
            draws = np.repeat(vals[:, np.newaxis, :], repeats=num_samples, axis=1)
            # Add deterministic spread around truth to exercise error bars.
            draws[:, : num_samples // 2, :] -= 0.1
            draws[:, num_samples // 2 :, :] += 0.1
            out[key] = draws
        return out


class _DummyEdgeSimulator:
    def __init__(self, n_items=(2, 3), n_persons=(4, 5)):
        self.n_items = n_items
        self.n_persons = n_persons

    def clone_with_sizes(self, *, n_items=None, n_persons=None):
        return _DummyEdgeSimulator(
            n_items=n_items if n_items is not None else self.n_items,
            n_persons=n_persons if n_persons is not None else self.n_persons,
        )

    def sample(self, shape):
        if tuple(shape) != (3,):
            raise ValueError("Dummy edge simulator only supports sample((3,)).")
        n_items = int(self.n_items[0])
        n_persons = int(self.n_persons[0])
        base = np.linspace(0.5, 1.5, n_items, dtype=np.float32)
        a = np.tile(base[np.newaxis, :], (3, 1))
        b = np.tile((-base)[np.newaxis, :], (3, 1))
        responses = np.zeros((3, n_persons, n_items), dtype=np.float32)
        return {"a": a, "b": b, "responses": responses}


def test_chunked_posterior_sample_concatenates_batches():
    approx = _DummyApproximator()
    cond = {
        "b": np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=np.float32),
        "responses": np.zeros((3, 2, 2), dtype=np.float32),
    }
    out = chunked_posterior_sample(
        approx,
        cond,
        num_samples=10,
        chunk_size=2,
    )
    assert out["b"].shape == (3, 10, 2)


def test_chunked_posterior_sample_forwards_flow_sample_kwargs():
    class RecordingApproximator:
        def __init__(self):
            self.calls = []

        def sample(self, *, num_samples, conditions, flow_sample_kwargs=None):
            self.calls.append(
                {
                    "num_samples": num_samples,
                    "flow_sample_kwargs": flow_sample_kwargs,
                    "batch": len(next(iter(conditions.values()))),
                }
            )
            b_true = np.asarray(conditions["b"], dtype=np.float32)
            return {
                "b": np.repeat(b_true[:, np.newaxis, :], repeats=num_samples, axis=1)
            }

    approx = RecordingApproximator()
    cond = {
        "b": np.array([[0.0], [1.0], [2.0]], dtype=np.float32),
        "responses": np.zeros((3, 2, 1), dtype=np.float32),
    }
    flow_kwargs = {"method": "euler", "steps": 24}
    out = chunked_posterior_sample(
        approx,
        cond,
        num_samples=5,
        chunk_size=2,
        flow_sample_kwargs=flow_kwargs,
    )

    assert out["b"].shape == (3, 5, 1)
    assert [call["flow_sample_kwargs"] for call in approx.calls] == [
        flow_kwargs,
        flow_kwargs,
    ]


def test_sample_random_size_conditions_count():
    sim = _DummySimulator(n_items=(2, 4), n_persons=(5, 6))
    sampled = sample_random_size_conditions(
        sim,
        n_conditions=4,
        datasets_per_condition=2,
        seed=123,
    )
    assert len(sampled) == 8
    assert all("conditions" in entry for entry in sampled)


def test_pooled_sbc_for_parameter_shapes_and_ranks():
    approx = _DummyApproximator()
    cond_batches = [
        {
            "conditions": {
                "b": np.array([[0.0, 1.0, 2.0]], dtype=np.float32),
                "responses": np.zeros((1, 3, 3), dtype=np.float32),
            }
        },
        {
            "conditions": {
                "b": np.array([[3.0, 4.0, 5.0]], dtype=np.float32),
                "responses": np.zeros((1, 3, 3), dtype=np.float32),
            }
        },
    ]
    out = pooled_sbc_for_parameter(
        approx,
        condition_batches=cond_batches,
        param_key="b",
        n_posterior_samples=20,
        sample_chunk_size=1,
    )
    assert out["ranks"].shape == (6,)
    assert np.all(out["ranks"] == 10)
    assert out["posterior_flat"].shape == (6, 20)
    assert out["true_flat"].shape == (6,)


def test_pooled_sbc_random_conditions_end_to_end():
    sim = _DummySimulator(n_items=(3, 3), n_persons=(5, 5))
    approx = _DummyApproximator()
    out = pooled_sbc_random_conditions(
        sim,
        approx,
        param_key="b",
        n_conditions=5,
        datasets_per_condition=1,
        n_posterior_samples=10,
        sample_chunk_size=1,
        seed=1234,
    )
    assert len(out["sampled_conditions"]) == 5
    assert out["ranks"].shape == (15,)
    assert np.all(out["ranks"] == 5)
    assert out["posterior_flat"].shape == (15, 10)
    assert out["true_flat"].shape == (15,)


def test_run_sbc_diagnostics_without_plots_returns_expected_keys():
    ranks = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    post = np.array(
        [
            np.linspace(-1, 1, 5),
            np.linspace(-0.5, 1.5, 5),
            np.linspace(0, 2, 5),
            np.linspace(0.5, 2.5, 5),
            np.linspace(1, 3, 5),
        ],
        dtype=np.float64,
    )
    true = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float64)

    out = run_sbc_diagnostics(
        ranks,
        post,
        true,
        title_prefix="unit-test",
        show_plots=False,
    )
    for key in (
        "ks_stat",
        "ks_pval",
        "chi2_stat",
        "chi2_pval",
        "r_global",
        "rmse_global",
    ):
        assert key in out
        assert isinstance(out[key], float)


def test_plot_edge_parameter_diagnostics_keeps_corner_counts_separate():
    sim = _DummyEdgeSimulator(n_items=(2, 3), n_persons=(4, 5))
    approx = _DummyEdgeApproximator()

    out = plot_edge_parameter_diagnostics(
        sim,
        approx,
        param_key="a",
        samples_per_condition=3,
        n_posterior_draws=12,
        title_prefix="unit-test",
    )

    assert out["axes"].shape == (4, 4)
    assert out["corners"] == [(2, 4), (2, 5), (3, 4), (3, 5)]
    assert out["equalize_observations"] is True
    assert out["plot_n_observations"] == 6

    for n_items_case, n_persons_case in out["corners"]:
        corner = (n_items_case, n_persons_case)
        corner_result = out["corner_results"][corner]
        expected = 6
        assert corner_result["true_flat"].shape == (expected,)
        assert corner_result["posterior_flat"].shape == (expected, 12)
        assert corner_result["ranks"].shape == (expected,)
        assert corner_result["n_observations"] == expected
        assert np.isfinite(corner_result["corr"])
        assert np.isclose(corner_result["nrmse"], 0.0, atol=1e-6)


def test_plot_edge_parameter_diagnostics_scales_columns_to_corner_count():
    """One column per corner, not a hardcoded four.

    Pinning training to a size that is not an edge corner means that size
    must be scored alongside the usual four, so the figure has to widen.
    """
    sim = _DummyEdgeSimulator(n_items=(2, 3), n_persons=(4, 5))
    approx = _DummyEdgeApproximator()

    corners = [(2, 4), (2, 5), (3, 4), (3, 5), (3, 9)]
    out = plot_edge_parameter_diagnostics(
        sim,
        approx,
        param_key="a",
        samples_per_condition=3,
        n_posterior_draws=12,
        corners=corners,
        title_prefix="unit-test",
    )

    assert out["axes"].shape == (4, 5)
    assert out["corners"] == corners
    assert (3, 9) in out["corner_results"]


@pytest.mark.parametrize("log_scale", [True, False])
@pytest.mark.parametrize(
    "point_estimate",
    [
        pytest.param([-1.0, 2.0, 3.0], id="non_positive"),
        pytest.param([np.nan, np.nan, np.nan], id="all_nan"),
        pytest.param([np.inf, 2.0, 3.0], id="infinite"),
    ],
)
def test_recovery_axes_survive_degenerate_posteriors(log_scale, point_estimate):
    """A diagnostic figure must survive the model it is diagnosing.

    Scoring a size-pinned model far out of distribution yields non-positive
    or non-finite posterior summaries; those must not produce NaN axis
    limits that crash a run after training has completed.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from bayesflow_irt.sbc import _set_recovery_identity_axes

    fig, ax = plt.subplots()
    try:
        if log_scale:
            ax.set_xscale("log")
            ax.set_yscale("log")
        _set_recovery_identity_axes(
            ax,
            np.array([1.0, 2.0, 3.0]),
            np.asarray(point_estimate, dtype=float),
            np.array([[0.8, 1.8, 2.8], [1.2, 2.2, 3.2]]),
            log_scale=log_scale,
        )
        low, high = ax.get_xlim()
        assert np.isfinite(low) and np.isfinite(high)
        assert low < high
    finally:
        plt.close(fig)


def test_summarize_edge_parameter_diagnostics_matches_corner_results(capsys):
    sim = _DummyEdgeSimulator(n_items=(2, 3), n_persons=(4, 5))
    approx = _DummyEdgeApproximator()

    diag = plot_edge_parameter_diagnostics(
        sim,
        approx,
        param_key="a",
        samples_per_condition=3,
        n_posterior_draws=12,
        title_prefix="unit-test",
    )

    rows = summarize_edge_parameter_diagnostics(diag, print_summary=True)

    assert len(rows) == len(diag["corners"])
    seen_corners = set()
    for row in rows:
        corner = (row["n_items"], row["n_persons"])
        seen_corners.add(corner)
        corner_result = diag["corner_results"][corner]
        assert row["param_key"] == "a"
        assert row["n_observations"] == corner_result["n_observations"]
        assert np.isclose(row["corr"], corner_result["corr"])
        assert np.isclose(row["nrmse"], corner_result["nrmse"])
        assert np.isclose(row["nrmse_excess"], corner_result["nrmse_excess"])
        assert np.isfinite(row["calibration_error"])
        assert 0.0 <= row["calibration_error"] <= 1.0

    assert seen_corners == set(diag["corners"])

    captured = capsys.readouterr()
    for row in rows:
        assert f"I={row['n_items']:>3}, P={row['n_persons']:>4}" in captured.out
        assert "dNRMSE=" in captured.out
        assert "CalErr=" in captured.out


def test_nrmse_excess_is_zero_for_a_correctly_scaled_posterior_mean():
    # An exact posterior mean is unbiased and shrunk by exactly the
    # correlation factor, sd(m) = r sd(t), which collapses the MSE
    # decomposition to NRMSE = sqrt(1 - r**2).
    rng = np.random.default_rng(0)
    truth = rng.normal(size=20_000)
    for shrink in (0.95, 0.70, 0.43):
        estimate = shrink * truth + np.sqrt(shrink * (1.0 - shrink)) * rng.normal(
            size=truth.size
        )
        metrics = _compute_recovery_metrics(truth, estimate)
        assert np.isclose(metrics["nrmse"], metrics["nrmse_ideal"], atol=1e-3)
        assert abs(metrics["nrmse_excess"]) < 1e-3


@pytest.mark.parametrize(
    ("estimate_fn", "label"),
    [
        (lambda t, rng: t + 0.5 * rng.normal(size=t.size), "under-shrunk"),
        (
            lambda t, rng: 0.9 * t + 0.4 * rng.normal(size=t.size) + 0.5,
            "biased",
        ),
    ],
)
def test_nrmse_excess_is_positive_when_the_scale_is_wrong(estimate_fn, label):
    # Ordering is preserved, so correlation stays high; only the spread or
    # location is wrong, which is exactly what nrmse_excess is meant to catch.
    rng = np.random.default_rng(1)
    truth = rng.normal(size=20_000)
    metrics = _compute_recovery_metrics(truth, estimate_fn(truth, rng))
    assert metrics["corr"] > 0.8, label
    assert metrics["nrmse_excess"] > 0.01, label


def test_nrmse_excess_is_nan_when_correlation_is_undefined():
    constant = np.ones(5)
    metrics = _compute_recovery_metrics(constant, constant)
    assert np.isnan(metrics["corr"])
    assert np.isnan(metrics["nrmse_ideal"])
    assert np.isnan(metrics["nrmse_excess"])


def test_plot_edge_parameter_diagnostics_can_use_log_scale_for_positive_parameter():
    sim = _DummyEdgeSimulator(n_items=(2, 3), n_persons=(4, 5))
    approx = _DummyEdgeApproximator()

    out = plot_edge_parameter_diagnostics(
        sim,
        approx,
        param_key="a",
        samples_per_condition=3,
        n_posterior_draws=12,
        log_scale=True,
    )

    assert out["log_scale"] is True
    assert out["axes"][0, 0].get_xscale() == "log"
    assert out["axes"][0, 0].get_yscale() == "log"
    assert out["axes"][1, 0].get_xscale() == "linear"
    assert out["axes"][1, 0].get_yscale() == "linear"


def test_plot_edge_parameter_diagnostics_frees_log_axes_from_shared_groups(monkeypatch):
    import matplotlib.pyplot as plt

    original_subplots = plt.subplots

    def shared_subplots(*args, **kwargs):
        kwargs["sharex"] = "col"
        kwargs["sharey"] = "row"
        return original_subplots(*args, **kwargs)

    monkeypatch.setattr(plt, "subplots", shared_subplots)

    sim = _DummyEdgeSimulator(n_items=(2, 3), n_persons=(4, 5))
    approx = _DummyEdgeApproximator()

    out = plot_edge_parameter_diagnostics(
        sim,
        approx,
        param_key="a",
        samples_per_condition=3,
        n_posterior_draws=12,
        log_scale=True,
    )

    assert out["axes"][0, 0].get_xscale() == "log"
    assert out["axes"][0, 0].get_yscale() == "log"
    assert out["axes"][1, 0].get_xscale() == "linear"
    assert out["axes"][1, 0].get_yscale() == "linear"
    assert (
        out["axes"][0, 0]
        .get_shared_x_axes()
        .joined(
            out["axes"][0, 0],
            out["axes"][1, 0],
        )
        is False
    )
    assert (
        out["axes"][0, 0]
        .get_shared_y_axes()
        .joined(
            out["axes"][0, 0],
            out["axes"][0, 1],
        )
        is False
    )


def test_plot_edge_parameter_diagnostics_uses_positive_log_limits():
    corners = [(2, 4), (2, 5), (3, 4), (3, 5)]
    conditions = []
    posteriors = []
    for col, (n_items, _n_persons) in enumerate(corners):
        base = np.geomspace(0.01 * (col + 1), 0.04 * (col + 1), n_items).astype(
            np.float32
        )
        true = np.tile(base[np.newaxis, :], (3, 1))
        draws = np.repeat(true[:, np.newaxis, :], repeats=12, axis=1)
        draws[:, :6, :] *= 0.8
        draws[:, 6:, :] *= 1.2
        conditions.append({"a": true})
        posteriors.append({"a": draws})

    out = plot_edge_parameter_diagnostics(
        param_key="a",
        edge_data={
            "corners": corners,
            "conditions": conditions,
            "posteriors": posteriors,
            "n_posterior_draws": 12,
        },
        log_scale=True,
    )

    for ax in out["axes"][0]:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        assert xlim[0] > 0.0
        assert ylim[0] > 0.0
        assert xlim == ylim


def test_plot_edge_parameter_diagnostics_log_scale_metrics_use_log_posterior_samples():
    corners = [(2, 4), (2, 5), (3, 4), (3, 5)]
    true = np.array([[1.0, 4.0]], dtype=np.float32)
    draws = np.array(
        [
            [
                [0.5, 2.0],
                [2.0, 8.0],
            ]
        ],
        dtype=np.float32,
    )
    edge_data = {
        "corners": corners,
        "conditions": [{"a": true} for _corner in corners],
        "posteriors": [{"a": draws} for _corner in corners],
        "n_posterior_draws": 2,
    }

    out = plot_edge_parameter_diagnostics(
        param_key="a",
        edge_data=edge_data,
        log_scale=True,
        equalize_observations=False,
    )

    corner_result = out["corner_results"][corners[0]]
    assert np.isclose(corner_result["rmse"], 0.0, atol=1e-7)
    assert np.isclose(corner_result["nrmse"], 0.0, atol=1e-7)
    assert out["axes"][0, 0].get_title().endswith("NRMSE=0.000")


def test_plot_edge_parameter_diagnostics_rejects_non_positive_log_scale_values():
    sim = _DummyEdgeSimulator(n_items=(2, 3), n_persons=(4, 5))
    approx = _DummyEdgeApproximator()

    with np.testing.assert_raises(ValueError):
        plot_edge_parameter_diagnostics(
            sim,
            approx,
            param_key="b",
            samples_per_condition=3,
            n_posterior_draws=12,
            log_scale=True,
        )


def test_sample_edge_corner_batches_can_be_reused_across_parameters():
    sim = _DummyEdgeSimulator(n_items=(2, 3), n_persons=(4, 5))
    approx = _DummyEdgeApproximator()

    edge_data = sample_edge_corner_batches(
        sim,
        approx,
        samples_per_condition=3,
        n_posterior_draws=12,
    )

    out_a = plot_edge_parameter_diagnostics(
        param_key="a",
        edge_data=edge_data,
        title_prefix="unit-test-a",
    )
    out_b = plot_edge_parameter_diagnostics(
        param_key="b",
        edge_data=edge_data,
        title_prefix="unit-test-b",
    )

    assert out_a["corners"] == edge_data["corners"]
    assert out_b["corners"] == edge_data["corners"]
    assert out_a["plot_n_observations"] == 6
    assert out_b["plot_n_observations"] == 6


def test_sample_edge_corner_batches_forwards_flow_sample_kwargs():
    sim = _DummyEdgeSimulator(n_items=(2, 3), n_persons=(4, 5))
    approx = _DummyEdgeApproximator()
    flow_kwargs = {"method": "tsit5", "steps": 32}

    edge_data = sample_edge_corner_batches(
        sim,
        approx,
        samples_per_condition=3,
        n_posterior_draws=12,
        flow_sample_kwargs=flow_kwargs,
    )

    assert edge_data["n_posterior_draws"] == 12
    assert len(approx.sample_calls) == 4
    assert all(
        call["flow_sample_kwargs"] == flow_kwargs for call in approx.sample_calls
    )


def test_compute_calibration_error_calibrated_draws_are_near_zero():
    from bayesflow_irt.sbc import compute_calibration_error

    rng = np.random.default_rng(1)
    n_obs, n_draws = 4000, 100
    # Standard SBC setup: truth and posterior draws share the same joint
    # (true ~ N(mu, 1), draws ~ N(mu, 1)), so calibration error should be
    # near zero.
    mu = rng.normal(size=n_obs)
    true_vals = mu + rng.normal(size=n_obs)
    posterior = mu[:, None] + rng.normal(size=(n_obs, n_draws))
    out = compute_calibration_error(posterior, true_vals)
    assert out["calibration_error"] < 0.03


def test_compute_calibration_error_detects_overconfidence():
    from bayesflow_irt.sbc import compute_calibration_error

    rng = np.random.default_rng(2)
    n_obs, n_draws = 2000, 100
    mu = rng.normal(size=n_obs)
    true_vals = mu + rng.normal(size=n_obs)
    # Overconfident posterior: draws far too narrow around a biased center.
    posterior = mu[:, None] + 0.1 * rng.normal(size=(n_obs, n_draws))
    out = compute_calibration_error(posterior, true_vals)
    assert out["calibration_error"] > 0.3


def test_compute_calibration_error_is_monotone_transform_invariant():
    from bayesflow_irt.sbc import compute_calibration_error

    rng = np.random.default_rng(3)
    mu = rng.normal(size=500)
    true_vals = mu + rng.normal(size=500)
    posterior = mu[:, None] + rng.normal(size=(500, 80))
    raw = compute_calibration_error(posterior, true_vals)
    logged = compute_calibration_error(np.exp(posterior), np.exp(true_vals))
    assert np.isclose(raw["calibration_error"], logged["calibration_error"])


def test_compute_calibration_error_validates_inputs():
    import pytest

    from bayesflow_irt.sbc import compute_calibration_error

    with pytest.raises(ValueError, match="2D"):
        compute_calibration_error(np.zeros(5), np.zeros(5))
    with pytest.raises(ValueError, match="one entry per"):
        compute_calibration_error(np.zeros((5, 10)), np.zeros(4))
    with pytest.raises(ValueError, match="resolution"):
        compute_calibration_error(np.zeros((5, 10)), np.zeros(5), resolution=0)
    with pytest.raises(ValueError, match="min_level"):
        compute_calibration_error(
            np.zeros((5, 10)), np.zeros(5), min_level=0.9, max_level=0.1
        )


def test_sample_random_theta_sbc_conditions_within_range():
    conditions = sample_random_theta_sbc_conditions(
        (5, 20), (10, 100), n_conditions=6, seed=0
    )
    assert len(conditions) == 6
    for n_items, n_persons in conditions:
        assert 5 <= n_items <= 20
        assert 10 <= n_persons <= 100


def test_sample_random_theta_sbc_conditions_rejects_bad_count():
    import pytest

    with pytest.raises(ValueError, match="n_conditions"):
        sample_random_theta_sbc_conditions((5, 20), (10, 100), n_conditions=0)


@pytest.mark.slow
def test_run_person_theta_sbc_fixed_mode_shapes(trained_1pl_approximator):
    from bayesflow_irt import make_1pl_simulator

    approximator, simulator = trained_1pl_approximator

    def _simulator_factory(n_items, n_persons, seed):
        return make_1pl_simulator(
            n_items=(n_items, n_items),
            n_persons=(n_persons, n_persons),
            prior_b=simulator.prior_b,
            prior_theta=simulator.prior_theta,
            seed=seed,
        )

    results = run_person_theta_sbc(
        _simulator_factory,
        approximator,
        conditions=[(10, 20), (12, 25)],
        n_pilot=2,
        adaptive=False,
        n_item_samples=20,
        n_plausible_values=20,
        theta_grid_size=11,
        seed=0,
        flow_sample_kwargs={"method": "euler", "steps": 4},
    )

    assert len(results) == 2
    for entry, (n_items, n_persons) in zip(results, [(10, 20), (12, 25)]):
        assert entry["n_items"] == n_items
        assert entry["n_persons"] == n_persons
        assert entry["n_replications"] == 2
        assert entry["hit_cap"] is False
        assert "achieved_mcse_corr" not in entry
        for key in ("mean_corr", "mean_coverage", "mean_precision_sd", "mean_rmse"):
            assert np.isfinite(entry[key])


@pytest.mark.slow
def test_run_person_theta_sbc_forwards_flow_sample_kwargs_and_restores_sample(
    trained_1pl_approximator,
):
    """PersonScorer.score() has no flow_sample_kwargs passthrough, so
    run_person_theta_sbc must monkeypatch approximator.sample() to inject it
    -- and must restore the original sample() afterward, even though a spy
    sits underneath recording what was actually passed through."""
    from bayesflow_irt import make_1pl_simulator

    approximator, simulator = trained_1pl_approximator
    original_sample = approximator.sample
    recorded_kwargs = []

    def _spy_sample(*args, **kwargs):
        recorded_kwargs.append(kwargs.get("flow_sample_kwargs"))
        return original_sample(*args, **kwargs)

    approximator.sample = _spy_sample

    def _simulator_factory(n_items, n_persons, seed):
        return make_1pl_simulator(
            n_items=(n_items, n_items),
            n_persons=(n_persons, n_persons),
            prior_b=simulator.prior_b,
            prior_theta=simulator.prior_theta,
            seed=seed,
        )

    try:
        run_person_theta_sbc(
            _simulator_factory,
            approximator,
            conditions=[(10, 20)],
            n_pilot=1,
            adaptive=False,
            n_item_samples=20,
            n_plausible_values=20,
            theta_grid_size=11,
            seed=0,
            flow_sample_kwargs={"method": "euler", "steps": 4},
        )
        assert len(recorded_kwargs) >= 1
        assert all(kw == {"method": "euler", "steps": 4} for kw in recorded_kwargs)
        # approximator.sample must be restored to the spy (what it was before
        # the call), not left pointing at run_person_theta_sbc's internal patch.
        assert approximator.sample is _spy_sample
    finally:
        approximator.sample = original_sample


@pytest.mark.slow
def test_run_person_theta_sbc_adaptive_mode_reports_mcse(trained_1pl_approximator):
    from bayesflow_irt import make_1pl_simulator

    approximator, simulator = trained_1pl_approximator

    def _simulator_factory(n_items, n_persons, seed):
        return make_1pl_simulator(
            n_items=(n_items, n_items),
            n_persons=(n_persons, n_persons),
            prior_b=simulator.prior_b,
            prior_theta=simulator.prior_theta,
            seed=seed,
        )

    results = run_person_theta_sbc(
        _simulator_factory,
        approximator,
        conditions=[(10, 20)],
        n_pilot=2,
        adaptive=True,
        max_replications=3,
        target_mcse_corr=1e-6,  # unreachable target forces the max_replications cap
        target_mcse_coverage=1e-6,
        n_item_samples=20,
        n_plausible_values=20,
        theta_grid_size=11,
        seed=0,
        flow_sample_kwargs={"method": "euler", "steps": 4},
    )

    assert len(results) == 1
    entry = results[0]
    assert entry["n_replications"] == 3
    assert entry["hit_cap"] is True
    for key in (
        "achieved_mcse_corr",
        "achieved_mcse_coverage",
        "achieved_mcse_precision",
        "achieved_mcse_rmse",
    ):
        assert np.isfinite(entry[key])


def _conjugate_gaussian_case(kind, n_sims=1500, n_draws=200, n_params=2, seed=0):
    """Build a case with a known-correct posterior and a controlled defect.

    Conjugate model: ``theta ~ N(0, 1)``, ``x | theta ~ N(theta, 1)``, so the
    exact posterior is ``N(x / 2, 0.5)``. Returns
    ``(draws, truth, data_dependent_reference_points)``.
    """
    rng = np.random.default_rng(seed)
    truth = rng.normal(size=(n_sims, n_params))
    x = truth + rng.normal(size=(n_sims, n_params))
    mean, sd = x / 2.0, np.sqrt(0.5)
    if kind == "calibrated":
        draws = mean[:, None, :] + sd * rng.normal(size=(n_sims, n_draws, n_params))
    elif kind == "overconfident":
        draws = mean[:, None, :] + 0.5 * sd * rng.normal(
            size=(n_sims, n_draws, n_params)
        )
    elif kind == "underconfident":
        draws = mean[:, None, :] + 2.0 * sd * rng.normal(
            size=(n_sims, n_draws, n_params)
        )
    elif kind == "prior":
        # The failure mode that matters: the estimator ignores x entirely.
        draws = rng.normal(size=(n_sims, n_draws, n_params))
    else:  # pragma: no cover - guard against typos in the test itself
        raise ValueError(kind)
    references = x + rng.uniform(size=(n_sims, n_params))
    return draws, truth, references


def test_tarp_is_near_nominal_for_an_exact_posterior():
    draws, truth, refs = _conjugate_gaussian_case("calibrated")
    assert compute_tarp_coverage(draws, truth, seed=1)["tarp_error"] < 0.03
    assert (
        compute_tarp_coverage(draws, truth, reference_points=refs)["tarp_error"] < 0.03
    )


def test_tarp_reference_seed_is_independent_of_the_data_seed():
    """Passing the data's own seed must not correlate references with truths.

    The docstring directs callers to pass a seed for anything committed, and
    the obvious seed to reach for is the run's. Without domain separation a
    fresh ``default_rng(s)`` would replay the stream that drew the truths:
    where the consumptions align, the references land on the truths, every
    ``f_i`` collapsed to zero, and a calibrated posterior scored exactly 0.5.

    Truths are drawn with ``uniform`` here precisely because that is the
    aligned case. With ``normal`` the mismatch between the Ziggurat's
    consumption and a plain uniform draw blunts the collision to a 50%
    inflation of a small number -- real, but invisible.

    The posterior is the *exact* one for this model, so "calibrated" is a
    property of the fixture and not a label on it: theta ~ U(-3, 3) with
    x | theta ~ N(theta, 1) gives a posterior proportional to N(x, 1)
    truncated to [-3, 3]. It is a function of ``x`` alone and never sees
    ``truth``. Its TARP error under an independent reference is 0.005-0.015,
    which is what makes the 0.5 meaningful.
    """
    n_sims, n_draws, n_params, seed = 1500, 200, 2, 0
    lo, hi = -3.0, 3.0
    rng = np.random.default_rng(seed)
    truth = rng.uniform(lo, hi, size=(n_sims, n_params))
    x = truth + rng.normal(size=(n_sims, n_params))
    draws = truncnorm.rvs(
        (lo - x)[:, None, :],
        (hi - x)[:, None, :],
        loc=x[:, None, :],
        scale=1.0,
        size=(n_sims, n_draws, n_params),
        random_state=rng,
    )

    colliding = compute_tarp_coverage(draws, truth, seed=seed)["tarp_error"]
    independent = compute_tarp_coverage(draws, truth, seed=seed + 999983)["tarp_error"]

    # The fixture must be calibrated for the assertion below to mean
    # anything: an estimator that is simply bad would also score badly.
    assert independent < 0.05, f"fixture is not calibrated: {independent:.4f}"
    # 0.5 is the signature of the collision: ECP saturates at 1.0 at every
    # level, so the median deviation is the median of |1 - level|.
    assert colliding < 0.1, (
        f"tarp_error={colliding:.4f} with seed == the data seed; 0.5 means "
        "the reference points reproduced the truths."
    )
    # The two seeds must agree on the verdict, not merely both avoid 0.5.
    assert abs(colliding - independent) < 0.02


def test_tarp_reference_draw_is_still_reproducible():
    """Domain separation must not cost determinism."""
    draws, truth, _ = _conjugate_gaussian_case("calibrated", n_sims=200, n_draws=50)
    a = compute_tarp_coverage(draws, truth, seed=3)
    b = compute_tarp_coverage(draws, truth, seed=3)
    assert a["tarp_error"] == b["tarp_error"]
    assert a["reference_mode"] == "random"
    c = compute_tarp_coverage(draws, truth, seed=4)
    assert a["tarp_error"] != c["tarp_error"]


def test_tarp_reference_draw_is_pinned_to_the_salt():
    """Lock the reference stream, so the salt cannot change silently.

    ``_TARP_REFERENCE_SALT`` promises not to change once results are
    committed, and comparing two calls to the same revision cannot enforce
    that -- changing 0x7A297 to 0x7A298 would leave every other test
    passing. This pins the actual draw.

    ``coverage_fractions`` rather than ``tarp_error``: the latter is a median
    over levels, so different reference draws can collapse to the same value.
    Changing the salt, the argument order, or the sequence form all move
    these numbers.
    """
    rng = np.random.default_rng(12345)
    truth = rng.uniform(-3.0, 3.0, size=(8, 2))
    x = truth + rng.normal(size=(8, 2))
    draws = x[:, None, :] + 0.7 * rng.normal(size=(8, 16, 2))

    out = compute_tarp_coverage(draws, truth, seed=5, resolution=4)
    np.testing.assert_allclose(
        out["coverage_fractions"],
        [0.9375, 0.0625, 0.5, 0.0, 0.25, 0.0, 0.4375, 0.125],
        rtol=0,
        atol=0,
    )


def test_tarp_curve_is_not_mirrored():
    """Overconfident sits below the diagonal, underconfident above it.

    This is the orientation of the paper's Figure 2. A sign error or a
    flipped inequality in Algorithm 2 would swap them while leaving
    ``tarp_error`` looking entirely reasonable, so the direction is asserted
    rather than only the magnitude.
    """
    over = compute_tarp_coverage(*_conjugate_gaussian_case("overconfident")[:2], seed=1)
    under = compute_tarp_coverage(
        *_conjugate_gaussian_case("underconfident")[:2], seed=1
    )
    levels = over["credibility_levels"]
    mid = (levels > 0.3) & (levels < 0.7)
    # The direction of the whole band, not each point: the overconfident
    # curve crosses the diagonal near the low end before settling below it,
    # so a pointwise assertion would be brittle without testing anything
    # stronger.
    assert np.mean(over["expected_coverage"][mid] - levels[mid]) < -0.02
    assert np.mean(under["expected_coverage"][mid] - levels[mid]) > 0.02


def test_tarp_with_random_reference_points_is_blind_to_an_ignored_dataset():
    """Section 4.3 of Lemos et al. (2023), reproduced.

    An estimator with ``p_hat(theta|x) = p(theta)`` has learned nothing, yet
    TARP with x-independent reference points returns a result as clean as a
    perfectly calibrated posterior. Only reference points that depend on the
    data expose it. This test exists so nobody "simplifies" the reference
    point away and reports the resulting all-clear as evidence of
    calibration.
    """
    draws, truth, refs = _conjugate_gaussian_case("prior")
    blind = compute_tarp_coverage(draws, truth, seed=1)
    sighted = compute_tarp_coverage(draws, truth, reference_points=refs)
    assert blind["reference_mode"] == "random"
    # "provided", not "data_dependent": the function cannot verify how a
    # supplied reference array was built, and only an x-dependent one closes
    # the Sec. 4.3 blind spot.
    assert sighted["reference_mode"] == "provided"
    assert blind["tarp_error"] < 0.03
    assert sighted["tarp_error"] > 0.15
    assert sighted["tarp_error"] > 5 * blind["tarp_error"]


def test_tarp_raises_on_a_constant_parameter_dimension():
    """A point-mass truth must not be silently dropped.

    Dropping such dimensions would suppress a real failure: a posterior
    placing sd-50 mass on a point-mass parameter would score as calibrated.
    Keeping the dimension costs a *correct* posterior nothing (see the next
    test), so a drop has no upside.
    """
    draws, truth, _ = _conjugate_gaussian_case("calibrated", n_sims=200)
    truth = truth.copy()
    truth[:, 1] = 3.0
    with pytest.raises(ValueError, match="constant across"):
        compute_tarp_coverage(draws, truth, seed=1)
    # standardize=False is unaffected: nothing is divided by the zero scale
    # and every dimension is already on a common footing.
    out = compute_tarp_coverage(draws, truth, seed=1, standardize=False)
    assert 0.0 <= out["tarp_error"] <= 1.0


def test_tarp_keeping_a_point_mass_dimension_is_free_for_a_correct_posterior():
    """Adding a coordinate constant in both truth and draws cannot move f_i.

    The same term enters both sides of ``d(draw, ref) < d(truth, ref)``, so
    the comparison is unchanged. This is the fact that makes raising -- rather
    than dropping -- the right call: nothing is lost by keeping such a
    dimension, and a wrong posterior there is caught.
    """
    draws, truth, _ = _conjugate_gaussian_case("calibrated", n_sims=200)
    refs = np.zeros((truth.shape[0], truth.shape[1] + 1))
    base = compute_tarp_coverage(
        draws, truth, reference_points=refs[:, :-1], standardize=False
    )
    padded_truth = np.concatenate([truth, np.full((truth.shape[0], 1), 3.0)], axis=1)
    padded_draws = np.concatenate(
        [draws, np.full((draws.shape[0], draws.shape[1], 1), 3.0)], axis=2
    )
    for metric in ("euclidean", "manhattan"):
        padded = compute_tarp_coverage(
            padded_draws,
            padded_truth,
            reference_points=refs,
            standardize=False,
            metric=metric,
        )
        base_metric = compute_tarp_coverage(
            draws,
            truth,
            reference_points=refs[:, :-1],
            standardize=False,
            metric=metric,
        )
        assert np.array_equal(
            padded["coverage_fractions"], base_metric["coverage_fractions"]
        )
    assert base["n_simulations"] == truth.shape[0]


def test_tarp_catches_a_wrong_posterior_on_a_point_mass_dimension():
    """The failure a drop would suppress."""
    draws, truth, _ = _conjugate_gaussian_case("calibrated", n_sims=800)
    truth = truth.copy()
    draws = draws.copy()
    truth[:, 1] = 3.0
    rng = np.random.default_rng(5)
    draws[:, :, 1] = 3.0 + 50.0 * rng.normal(size=draws.shape[:2])
    out = compute_tarp_coverage(draws, truth, seed=1, standardize=False)
    assert out["tarp_error"] > 0.2


def test_tarp_rejects_non_finite_inputs_instead_of_biasing_quietly():
    """NaN compares False, so it would be silently counted as "not closer"."""
    draws, truth, _ = _conjugate_gaussian_case("calibrated")
    draws = draws.copy()
    draws[:50] = np.nan
    with pytest.raises(ValueError, match="finite"):
        compute_tarp_coverage(draws, truth, seed=1)


def test_tarp_chunking_is_exactly_equivalent(monkeypatch):
    """Force several chunks and demand bit-identical results.

    At the default element budget every realistic call fits in one chunk, so
    without shrinking it this test would pass even if the chunk loop were
    deleted.
    """
    draws, truth, refs = _conjugate_gaussian_case("overconfident", n_sims=400)
    whole = compute_tarp_coverage(draws, truth, reference_points=refs)
    monkeypatch.setattr(sbc_module, "_TARP_CHUNK_ELEMENTS", 200 * 2 * 7)
    chunked = compute_tarp_coverage(draws, truth, reference_points=refs)
    assert np.array_equal(whole["coverage_fractions"], chunked["coverage_fractions"])
    assert chunked["tarp_error"] == whole["tarp_error"]

    for metric in ("euclidean", "manhattan"):
        a = compute_tarp_coverage(draws, truth, reference_points=refs, metric=metric)
        monkeypatch.setattr(sbc_module, "_TARP_CHUNK_ELEMENTS", 1.0)
        b = compute_tarp_coverage(draws, truth, reference_points=refs, metric=metric)
        assert np.array_equal(a["coverage_fractions"], b["coverage_fractions"])
        monkeypatch.setattr(sbc_module, "_TARP_CHUNK_ELEMENTS", 200 * 2 * 7)


def test_tarp_validates_shapes_and_arguments():
    draws, truth, _ = _conjugate_gaussian_case("calibrated", n_sims=50, n_draws=10)
    with pytest.raises(ValueError, match="3D"):
        compute_tarp_coverage(draws[:, 0, :], truth)
    with pytest.raises(ValueError, match="n_simulations, n_parameters"):
        compute_tarp_coverage(draws, truth[:10])
    with pytest.raises(ValueError, match="metric"):
        compute_tarp_coverage(draws, truth, metric="cosine")
    with pytest.raises(ValueError, match="reference_points"):
        compute_tarp_coverage(draws, truth, reference_points=truth[:10])
    with pytest.raises(ValueError, match="at least 2 draws"):
        compute_tarp_coverage(draws[:, :1, :], truth)
    with pytest.raises(ValueError, match="resolution"):
        compute_tarp_coverage(draws, truth, resolution=0)


def test_tarp_rejects_non_finite_truth_and_reference_points():
    """Not only non-finite draws: the other two inputs matter too."""
    draws, truth, refs = _conjugate_gaussian_case("calibrated", n_sims=60, n_draws=10)
    bad_truth = truth.copy()
    bad_truth[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        compute_tarp_coverage(draws, bad_truth, seed=1)
    bad_refs = refs.copy()
    bad_refs[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        compute_tarp_coverage(draws, truth, reference_points=bad_refs)


@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
def test_tarp_rejects_a_scale_that_overflowed_to_infinity():
    """`np.std` squares first, so huge magnitudes give scale = inf.

    The z-scores would then collapse to zero and the dimension would vanish
    from every distance — a silent drop would hide real failures: a
    prior-equals-posterior defect on such a dimension would
    score as well calibrated.
    """
    draws, truth, _ = _conjugate_gaussian_case("calibrated", n_sims=100, n_draws=10)
    truth = truth.copy()
    draws = draws.copy()
    truth[:, 1] *= 1e200
    draws[:, :, 1] *= 1e200
    with pytest.raises(ValueError, match="non-finite mean or"):
        compute_tarp_coverage(draws, truth, seed=1)


def test_tarp_rejects_an_empty_simulation_set():
    with pytest.raises(ValueError, match="at least one simulation"):
        compute_tarp_coverage(np.zeros((0, 5, 2)), np.zeros((0, 2)))


def test_classical_item_statistics_recovers_difficulty_ordering():
    """`z_difficulty` must be monotone in the true difficulty."""
    rng = np.random.default_rng(3)
    n_persons, n_items = 4000, 12
    b = np.linspace(-2.0, 2.0, n_items)
    theta = rng.normal(size=n_persons)
    p = 1.0 / (1.0 + np.exp(-(theta[:, None] - b[None, :])))
    responses = (rng.uniform(size=p.shape) < p).astype(float)

    stats = classical_item_statistics(responses)
    assert stats["p_value"].shape == (n_items,)
    # Harder items are answered correctly less often.
    assert np.all(np.diff(stats["p_value"]) < 0)
    # z_difficulty rises with b, which is what makes it usable as a
    # monotone-but-attenuated reference coordinate.
    assert np.corrcoef(stats["z_difficulty"], b)[0, 1] > 0.99


def test_classical_item_statistics_is_attenuated_not_an_estimate():
    """`z_difficulty` equals b * r_bis, so it is systematically shrunk.

    This attenuation is the point: a reference coordinate that matched `b`
    closely would collapse the TARP fractions and report a correct posterior
    as maximally miscalibrated.
    """
    rng = np.random.default_rng(4)
    n_persons, n_items = 4000, 12
    b = np.linspace(-2.0, 2.0, n_items)
    theta = rng.normal(size=n_persons)
    p = 1.0 / (1.0 + np.exp(-(theta[:, None] - b[None, :])))
    responses = (rng.uniform(size=p.shape) < p).astype(float)
    z = classical_item_statistics(responses)["z_difficulty"]
    # Shrunk toward zero relative to the true difficulties.
    assert np.std(z) < np.std(b)


def test_classical_item_statistics_handles_degenerate_items_and_batches():
    rng = np.random.default_rng(5)
    responses = (rng.uniform(size=(200, 6)) < 0.5).astype(float)
    responses[:, 0] = 1.0  # everyone correct: no variance, p at the ceiling
    stats = classical_item_statistics(responses)
    assert stats["point_biserial"][0] == 0.0
    assert np.all(np.isfinite(stats["z_difficulty"]))

    batch = np.stack([responses, responses])
    batched = classical_item_statistics(batch)
    assert batched["p_value"].shape == (2, 6)
    assert np.allclose(batched["z_difficulty"][0], stats["z_difficulty"])

    with pytest.raises(ValueError, match="n_persons, n_items"):
        classical_item_statistics(responses[0])


def test_classical_item_statistics_rejects_polytomous_responses():
    """Non-binary input yields plausible nonsense rather than an error.

    `p_value` becomes a mean category score instead of a proportion correct,
    and the half-response clip then drags it into [0, 1], collapsing many
    items onto the same extreme `z_difficulty`. A TARP reference built from
    that looks entirely ordinary, so the failure has to be loud.
    """
    rng = np.random.default_rng(0)
    polytomous = rng.integers(0, 5, size=(40, 6)).astype(float)
    with pytest.raises(ValueError, match="binary"):
        classical_item_statistics(polytomous)


def test_classical_item_statistics_accepts_binary_including_degenerate():
    """All-zero and all-one items are binary and must still be handled."""
    responses = np.zeros((20, 4))
    responses[:, 1] = 1.0
    responses[:10, 2] = 1.0
    stats = classical_item_statistics(responses)
    assert set(stats) == {"p_value", "z_difficulty", "point_biserial"}
    assert np.all(np.isfinite(stats["z_difficulty"]))


def _binary_batch(n_sims=3, n_persons=200, n_items=6, seed=0):
    rng = np.random.default_rng(seed)
    b = rng.normal(size=(n_sims, 1, n_items))
    theta = rng.normal(size=(n_sims, n_persons, 1))
    p = 1.0 / (1.0 + np.exp(-(theta - b)))
    return (rng.uniform(size=p.shape) < p).astype(float)


def test_classical_reference_points_blocks_follow_param_keys():
    from bayesflow_irt.sbc import classical_reference_points

    responses = _binary_batch()
    stats = classical_item_statistics(responses)
    refs = classical_reference_points(responses, ["a", "b"])
    assert refs.shape == (3, 12)
    np.testing.assert_array_equal(refs[:, :6], stats["point_biserial"])
    np.testing.assert_array_equal(refs[:, 6:], stats["z_difficulty"])
    np.testing.assert_array_equal(
        classical_reference_points(responses, ["b"]), stats["z_difficulty"]
    )


def test_classical_reference_points_jitter_and_refusals():
    from bayesflow_irt.sbc import classical_reference_points

    responses = _binary_batch()
    base = classical_reference_points(responses, ["b"])
    jittered = classical_reference_points(
        responses, ["b"], jitter=0.5, rng=np.random.default_rng(1)
    )
    assert np.all((jittered - base >= 0) & (jittered - base < 0.5))
    with pytest.raises(ValueError, match="rng is required"):
        classical_reference_points(responses, ["b"], jitter=0.5)
    with pytest.raises(ValueError, match="category-appropriate"):
        classical_reference_points(responses, ["delta"])


def test_compute_calibration_error_mean_aggregation():
    import pytest

    from bayesflow_irt.sbc import compute_calibration_error

    rng = np.random.default_rng(4)
    mu = rng.normal(size=3000)
    true_vals = mu + rng.normal(size=3000)
    # Right in the middle, too light in the tails: draws clipped at +-1.6 SD
    # cover correctly at low levels and undercover only at the top levels.
    posterior = mu[:, None] + np.clip(rng.normal(size=(3000, 200)), -1.6, 1.6)
    median = compute_calibration_error(posterior, true_vals)
    mean = compute_calibration_error(posterior, true_vals, aggregation="mean")
    assert set(mean) == {"mean_calibration_error"}
    assert mean["mean_calibration_error"] > 2 * median["calibration_error"]

    alphas = np.linspace(0.005, 0.995, 20)
    lo = np.quantile(posterior, (1 - alphas) / 2, axis=1)
    hi = np.quantile(posterior, (1 + alphas) / 2, axis=1)
    cov = ((lo <= true_vals) & (true_vals <= hi)).mean(axis=1)
    assert np.isclose(mean["mean_calibration_error"], np.mean(np.abs(cov - alphas)))

    with pytest.raises(ValueError, match="aggregation"):
        compute_calibration_error(posterior, true_vals, aggregation="max")


def test_compute_calibration_error_matches_bayesflow_mean():
    """Same statistic as bayesflow-hpo's mean_calibration_error."""
    import pytest

    bf = pytest.importorskip("bayesflow")
    from bayesflow_irt.sbc import compute_calibration_error

    rng = np.random.default_rng(5)
    mu = rng.normal(size=400)
    true_vals = mu + rng.normal(size=400)
    posterior = mu[:, None] + 0.8 * rng.normal(size=(400, 150))
    ours = compute_calibration_error(posterior, true_vals, aggregation="mean")
    ref = bf.diagnostics.calibration_error(
        estimates=posterior[:, :, None], targets=true_vals[:, None],
        aggregation=np.mean,
    )
    assert np.isclose(
        ours["mean_calibration_error"], float(np.mean(ref["values"])), atol=1e-12
    )


def test_sample_random_size_conditions_independent_seeded_streams():
    from bayesflow_irt import Prior, make_2pl_simulator

    def run():
        sim = make_2pl_simulator(
            n_items=(4, 4),
            n_persons=(6, 6),
            prior_a=Prior("lognormal"),
            prior_b=Prior("normal"),
            seed=7,
        )
        return sample_random_size_conditions(sim, n_conditions=3, seed=11)

    first, second = run(), run()
    b = [entry["conditions"]["b"] for entry in first]
    assert not np.allclose(b[0], b[1])
    assert not np.allclose(b[1], b[2])
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left["conditions"]["b"], right["conditions"]["b"])
