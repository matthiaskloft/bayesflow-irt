"""Tests for the shared posterior sampling driver (``evaluation.sample_posterior``)."""

import math

import keras
import numpy as np
import pytest
import torch

from bayesflow_irt import Prior, make_2pl_adapter, make_2pl_simulator
from bayesflow_irt.evaluation import SamplingFailedError, sample_posterior


def _data(n=10, n_items=4, n_persons=6, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "a": rng.lognormal(0, 0.5, (n, n_items)),
        "b": rng.normal(0, 2, (n, n_items)),
        "responses": rng.integers(0, 2, (n, n_persons, n_items)).astype(float),
    }


class RowwiseApproximator:
    """Draws that depend only on each dataset's own truth: chunking-invariant."""

    def __init__(self, autoregressive=False, max_chunk=None):
        self.autoregressive = autoregressive
        self.max_chunk = max_chunk
        self.calls = []

    def sample(self, conditions, num_samples, **kwargs):
        n = conditions["a"].shape[0]
        self.calls.append((n, kwargs))
        if self.max_chunk is not None and n > self.max_chunk:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (fake)")
        z = np.linspace(-2, 2, num_samples)[None, :, None]
        return {
            "a": conditions["a"][:, None] * np.exp(0.3 * z),
            "b": conditions["b"][:, None] + 0.5 * z,
        }


def test_chunks_reassemble_in_dataset_order():
    data = _data()
    whole = sample_posterior(RowwiseApproximator(), data, 7, ["a", "b"], chunk_size=10)
    split = sample_posterior(RowwiseApproximator(), data, 7, ["a", "b"], chunk_size=3)
    for key in ("a", "b"):
        assert whole.draws[key].shape == (10, 7, 4)
        assert whole.draws[key].dtype == np.float32
        np.testing.assert_array_equal(whole.draws[key], split.draws[key])


def test_flow_rows_are_batched_only_when_asked():
    data = _data(n_items=4)
    fake = RowwiseApproximator()
    sample_posterior(fake, data, 7, ["a"], chunk_size=5, batch_flow_rows=True)
    assert [k for _, k in fake.calls] == [
        {"condition_chunk_size": 20, "sample_chunk_size": 7}
    ] * 2
    fake = RowwiseApproximator()
    sample_posterior(fake, data, 7, ["a"], chunk_size=5, sample_kwargs={"x": 1})
    assert [k for _, k in fake.calls] == [{"x": 1}] * 2


def test_autoregressive_rows_are_the_whole_chunk():
    fake = RowwiseApproximator(autoregressive=True)
    res = sample_posterior(fake, _data(), 7, ["a"], chunk_size=4, batch_flow_rows=True)
    assert [k["condition_chunk_size"] for _, k in fake.calls] == [28, 28, 14]
    assert res.autoregressive_rows == "whole chunk"


def test_out_of_memory_shrinks_and_keeps_the_draws():
    data = _data()
    ref = sample_posterior(RowwiseApproximator(), data, 5, ["b"], chunk_size=10)
    res = sample_posterior(
        RowwiseApproximator(max_chunk=4), data, 5, ["b"], chunk_size=10
    )
    assert res.retries and res.retries[0]["start"] == 0
    np.testing.assert_array_equal(res.draws["b"], ref.draws["b"])
    with pytest.raises(SamplingFailedError, match="Out of memory"):
        sample_posterior(RowwiseApproximator(max_chunk=0), data, 5, ["b"], chunk_size=2)


@pytest.mark.parametrize("bad", [0, -1, 1.5, "big"])
def test_rejects_bad_chunk_size(bad):
    with pytest.raises(ValueError, match="chunk_size"):
        sample_posterior(RowwiseApproximator(), _data(), 5, ["b"], chunk_size=bad)


# --- A real flow-matching approximator --------------------------------------

hpo = pytest.importorskip("bayesflow_irt.hpo")
pytest.importorskip("bayesflow_hpo", reason="bayesflow_hpo not installed")


@pytest.fixture(scope="module")
def flow_model():
    keras.utils.set_random_seed(3)
    sim = make_2pl_simulator(
        n_items=(3, 5),
        n_persons=(20, 30),
        prior_a=Prior("lognormal", loc=0, scale=0.5),
        prior_b=Prior("normal", loc=0, scale=2.0),
    )
    adapter = make_2pl_adapter(sim)
    hooks = hpo.make_irt_hooks(
        sim, adapter, corners=[(5, 30)], sims_per_corner=2, metrics=["nrmse"]
    )
    params: dict = {}
    for sub in ("summary_space", "inference_space", "training_space"):
        for dim in getattr(hooks.search_space, sub).dimensions:
            choices = getattr(dim, "choices", None)
            low = getattr(dim, "low", None)
            if choices:
                params[dim.name] = list(choices)[0]
            elif isinstance(low, (int, float)):
                params[dim.name] = low
            else:
                params[dim.name] = getattr(dim, "constant", None)
    model = hooks.build_fn(params)
    model.build_from_data(adapter(sim.sample((2,))))
    corner = make_2pl_simulator(
        n_items=(5, 5),
        n_persons=(30, 30),
        prior_a=Prior("lognormal", loc=0, scale=0.5),
        prior_b=Prior("normal", loc=0, scale=2.0),
    ).sample((6,))
    return model, {k: np.asarray(v) for k, v in corner.items()}


def test_a_seeded_chunk_is_reproducible(flow_model):
    model, data = flow_model
    one = sample_posterior(model, data, 16, ["a", "b"], chunk_size=3, seed=11)
    two = sample_posterior(model, data, 16, ["a", "b"], chunk_size=3, seed=11)
    np.testing.assert_array_equal(one.draws["b"], two.draws["b"])


def test_batched_flow_rows_sample_the_same_posterior(flow_model):
    """One ODE solve per chunk draws from the same distribution as 256-row calls.

    The streams differ, so the check is statistical: per (dataset, item), the
    posterior means under both batchings agree within 5 Monte Carlo SEs.
    """
    model, data = flow_model
    n_draws = 400
    default = sample_posterior(model, data, n_draws, ["a", "b"], chunk_size=6, seed=1)
    batched = sample_posterior(
        model, data, n_draws, ["a", "b"], chunk_size=6, seed=2, batch_flow_rows=True
    )
    for key in ("a", "b"):
        x, y = default.draws[key].astype(float), batched.draws[key].astype(float)
        assert x.shape == y.shape
        se = np.sqrt((x.var(axis=1) + y.var(axis=1)) / n_draws)
        assert np.all(np.abs(x.mean(axis=1) - y.mean(axis=1)) < 5 * se)


# --- HPO validation through the driver --------------------------------------


def _hooks(chunk, metrics=("calibration_error", "nrmse")):
    # Seeded: the same validation datasets for every call.
    sim = make_2pl_simulator(
        n_items=(3, 5),
        n_persons=(20, 30),
        prior_a=Prior("lognormal", loc=0, scale=0.5),
        prior_b=Prior("normal", loc=0, scale=2.0),
        seed=5,
    )
    return hpo.make_irt_hooks(
        sim,
        make_2pl_adapter(sim),
        corners=[(3, 20), (5, 30)],
        param_keys=["a", "b"],
        sims_per_corner=8,
        metrics=list(metrics),
        validation_chunk_size=chunk,
    )


def test_validation_metrics_do_not_depend_on_the_chunking():
    whole = _hooks(8).validate_fn(RowwiseApproximator(), None, 20)
    split = _hooks(3).validate_fn(RowwiseApproximator(), None, 20)
    auto = _hooks("auto").validate_fn(RowwiseApproximator(), None, 20)
    assert whole.keys() == split.keys() == auto.keys()
    for key, value in whole.items():
        if isinstance(value, float) and math.isnan(value):
            assert math.isnan(split[key])
        else:
            assert split[key] == pytest.approx(value)
            assert auto[key] == pytest.approx(value)


def test_validation_matches_the_metric_on_one_shot_draws():
    """The value equals the registry metric on one unchunked sample() call."""
    from bayesflow_hpo.validation.registry import resolve_metrics

    hooks = _hooks(3, metrics=("calibration_error",))
    result = hooks.validate_fn(RowwiseApproximator(), None, 20)
    fn = resolve_metrics(["calibration_error"])["calibration_error"]
    values = []
    for corner in hooks.validation_data.simulations:
        post = RowwiseApproximator().sample(corner, 20)
        for key in ("a", "b"):
            d = np.asarray(post[key], dtype=np.float32)
            draws_flat = np.moveaxis(d, 1, -1).reshape(-1, d.shape[1])
            values.append(
                fn(draws_flat, np.asarray(corner[key]).reshape(-1))["calibration_error"]
            )
    assert result["calibration_error/worst"] == pytest.approx(max(values))


@pytest.mark.parametrize("bad", [0, True, "big"])
def test_validation_rejects_bad_chunk_size(bad):
    with pytest.raises(ValueError, match="validation_chunk_size"):
        _hooks(bad)
