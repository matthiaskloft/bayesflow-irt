"""Tests for the shared posterior sampling driver (``evaluation.sample_posterior``)."""

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

    def __init__(self, max_chunk=None, max_rows=None):
        self.max_chunk = max_chunk
        self.max_rows = max_rows
        self.calls = []

    def sample(self, conditions, num_samples, **kwargs):
        n = conditions["a"].shape[0]
        self.calls.append((n, kwargs))
        rows = kwargs.get("condition_chunk_size", 0)
        if (self.max_chunk is not None and n > self.max_chunk) or (
            self.max_rows is not None and rows > self.max_rows
        ):
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


def test_rows_are_the_whole_chunk():
    fake = RowwiseApproximator()
    res = sample_posterior(
        fake, _data(), 7, ["a"], chunk_size=4, sample_kwargs={"x": 1}
    )
    assert [k for _, k in fake.calls] == [
        {"x": 1, "condition_chunk_size": r} for r in (28, 28, 14)
    ]
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


def test_autoregressive_row_batch_shrinks_for_one_dataset():
    data = _data()
    ref = sample_posterior(RowwiseApproximator(), data, 8, ["b"], chunk_size=10)
    fake = RowwiseApproximator(max_rows=5)
    res = sample_posterior(fake, data, 8, ["b"], chunk_size=1)
    # 8 rows -> 6 -> 4 (CHUNK_SHRINK), then every dataset fits.
    assert [r["retry_rows"] for r in res.retries] == [6, 4]
    assert res.autoregressive_rows == "at most 4"
    np.testing.assert_array_equal(res.draws["b"], ref.draws["b"])


def test_auto_probes_device_memory(monkeypatch):
    """The CUDA probe path, with torch.cuda's memory counters stubbed."""
    import bayesflow_irt.evaluation as ev

    mib = 2**20
    state = {"peak": 0}
    fake = RowwiseApproximator()
    original = fake.sample

    def sample(conditions, num_samples, **kwargs):
        state["peak"] = conditions["a"].shape[0] * 10 * mib
        return original(conditions, num_samples, **kwargs)

    fake.sample = sample
    monkeypatch.setattr(ev, "_torch_cuda", lambda: True)
    for name, value in {
        "memory_allocated": lambda: 0,
        "memory_reserved": lambda: 0,
        "reset_peak_memory_stats": lambda: None,
        "max_memory_allocated": lambda: state["peak"],
        "mem_get_info": lambda: (50 * mib, 100 * mib),
        "get_device_name": lambda: "fake",
    }.items():
        monkeypatch.setattr(torch.cuda, name, value)
    data = _data(n=10)
    ref = sample_posterior(RowwiseApproximator(), data, 5, ["b"], chunk_size=10)
    res = sample_posterior(fake, data, 5, ["b"], chunk_size="auto")
    # 10 MiB per dataset, 0.8 of 50 MiB free: 4 datasets per chunk.
    assert res.chunk_size == 4
    assert res.chunk_rule["mib_per_dataset"] == 10
    np.testing.assert_array_equal(res.draws["b"], ref.draws["b"])


@pytest.mark.parametrize("bad", [0, -1, 1.5, "big", True])
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
