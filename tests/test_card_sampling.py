"""Tests for sampling one model-card cell (``card.sample_cell``)."""

import warnings
from pathlib import Path

import keras
import numpy as np
import pytest
import torch

from bayesflow_irt.card import (
    Cell,
    CellFailedError,
    EvaluationProtocol,
    item_parameter_metrics,
    reset_posterior_seed,
    sample_cell,
)

ROOT = Path(__file__).resolve().parents[1]
SIDECARS = Path(__file__).resolve().parent / "fixtures" / "sidecars"
N01 = {"name": "normal", "loc": 0.0, "scale": 1.0}
SPEC_2PL = {
    "model": "2PL",
    "priors": {
        "a": {"name": "lognormal", "loc": 0.0, "scale": 0.5},
        "b": {"name": "normal", "loc": 0.0, "scale": 2.0},
        "theta": N01,
    },
    "size_ranges": {"n_items": [5, 35], "n_persons": [5, 1000]},
}


class FakeApproximator:
    """Draws centred on the truth; optionally runs out of memory."""

    def __init__(self, max_chunk=None, sd=0.3):
        self.max_chunk = max_chunk
        self.sd = sd
        self.calls = []
        self.generator = keras.random.SeedGenerator(0)

    def sample(self, conditions, num_samples, **kwargs):
        n = conditions["responses"].shape[0]
        self.calls.append(n)
        if self.max_chunk is not None and n > self.max_chunk:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (fake)")
        out = {}
        for key in ("a", "b", "delta"):
            if key not in conditions:
                continue
            truth = np.asarray(conditions[key], dtype=float)
            noise = np.random.normal(size=(n, num_samples, *truth.shape[1:]))
            scale = np.log(truth) if key == "a" else truth
            draws = scale[:, None] + self.sd * noise
            out[key] = np.exp(draws) if key == "a" else draws
        return out


def _protocol(**kw):
    kw.setdefault("datasets_per_cell", 12)
    kw.setdefault("posterior_draws", 40)
    return EvaluationProtocol.preset("development", **kw)


def test_shapes_choices_and_metrics():
    cell = Cell(6, 20, True)
    s = sample_cell(FakeApproximator(), SPEC_2PL, cell, _protocol(), chunk_size=5)
    assert s.draws["a"].shape == (12, 40, 6) and s.draws["a"].dtype == np.float32
    assert s.data["responses"].shape == (12, 20, 6)
    assert s.item.shape == (12,) and s.item.max() < 6 and s.person.max() < 20
    assert s.threshold is None and s.retries == [] and s.chunk_size == 5
    m = item_parameter_metrics(s)
    assert set(m) == {"log_a", "b"}
    assert "mean_calibration_error" in m["b"]


def test_metrics_pair_draws_with_the_chosen_items_truth():
    """Draws centred on the truth: a wrong index on either side shows as bias."""
    proto = _protocol(datasets_per_cell=200, posterior_draws=50)
    s = sample_cell(FakeApproximator(sd=0.05), SPEC_2PL, Cell(10, 20, True), proto)
    for name, metrics in item_parameter_metrics(s).items():
        bias, se = metrics["bias"]
        assert abs(bias) < 4 * se + 1e-3, name
        assert metrics["rmse"][0] < 0.1, name


def test_pcm_threshold_path():
    from bayesflow_irt.metadata import MetadataTracker

    path = SIDECARS / "pcm_equivariant_metadata.json"
    md = MetadataTracker.load(path).to_dict()
    s = sample_cell(
        FakeApproximator(sd=0.05),
        md["model_spec"],
        Cell(6, 30, True),
        _protocol(datasets_per_cell=60),
        chunk_size=25,
        trained_at=md["provenance"]["timestamp"],
    )
    n_thresholds = s.data["delta"].shape[-1]
    assert s.draws["delta"].shape == (60, 40, 6, n_thresholds)
    assert s.threshold.shape == (60,) and s.threshold.max() < n_thresholds
    bias, se = item_parameter_metrics(s)["delta"]["bias"]
    assert abs(bias) < 4 * se + 1e-3


def test_split_halves_are_seeded_from_their_own_start():
    """A half reproduces what an unsplit run with that chunk boundary gives."""
    cell = Cell(5, 10, True)
    split = sample_cell(
        FakeApproximator(max_chunk=3), SPEC_2PL, cell, _protocol(), chunk_size=6
    )
    small = sample_cell(FakeApproximator(), SPEC_2PL, cell, _protocol(), chunk_size=3)
    assert np.array_equal(split.draws["b"], small.draws["b"])


def test_same_protocol_same_cell_same_result():
    cell = Cell(5, 10, True)
    one = sample_cell(FakeApproximator(), SPEC_2PL, cell, _protocol())
    two = sample_cell(FakeApproximator(), SPEC_2PL, cell, _protocol())
    assert np.array_equal(one.draws["b"], two.draws["b"])
    assert np.array_equal(one.item, two.item)
    other = sample_cell(FakeApproximator(), SPEC_2PL, cell, _protocol(seed=1))
    assert not np.array_equal(one.data["b"], other.data["b"])


def test_out_of_memory_chunk_shrinks_and_stays_small():
    fake = FakeApproximator(max_chunk=3)
    s = sample_cell(fake, SPEC_2PL, Cell(5, 10, True), _protocol(), chunk_size=6)
    assert s.retries == [
        {"start": 0, "size": 6, "retry_size": 4},
        {"start": 0, "size": 4, "retry_size": 3},
    ]
    assert fake.calls == [6, 4, 3, 3, 3, 3]
    assert np.isfinite(s.draws["b"]).all()


def test_shrink_is_to_three_quarters():
    fake = FakeApproximator(max_chunk=5)
    s = sample_cell(
        fake, SPEC_2PL, Cell(5, 10, True), _protocol(datasets_per_cell=8), chunk_size=8
    )
    assert s.retries == [
        {"start": 0, "size": 8, "retry_size": 6},
        {"start": 0, "size": 6, "retry_size": 4},
    ]
    assert fake.calls == [8, 6, 4, 4]


def test_out_of_memory_at_one_dataset_fails_the_cell():
    fake = FakeApproximator(max_chunk=0)
    with pytest.raises(CellFailedError) as info:
        sample_cell(fake, SPEC_2PL, Cell(5, 10, True), _protocol(), chunk_size=6)
    # Datasets shrink to one, then that dataset's row batch shrinks to one.
    assert fake.calls[:5] == [6, 4, 3, 2, 1]
    assert set(fake.calls[5:]) == {1}
    assert info.value.record["size"] == 1
    cell = {"n_items": 5, "n_persons": 10, "in_range": True}
    assert info.value.record["cell"] == cell


def test_auto_chunk_size_is_recorded():
    s = sample_cell(
        FakeApproximator(), SPEC_2PL, Cell(5, 10, True), _protocol(), chunk_size="auto"
    )
    assert s.chunk_rule["rule"] == "auto"
    assert 1 <= s.chunk_size <= 12 and s.chunk_rule["chunk_size"] == s.chunk_size
    fixed = sample_cell(
        FakeApproximator(),
        SPEC_2PL,
        Cell(5, 10, True),
        _protocol(),
        chunk_size=s.chunk_size,
    )
    assert np.array_equal(s.draws["b"], fixed.draws["b"])  # probe leaves no trace


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_auto_chunk_size_uses_marginal_memory():
    from bayesflow_irt.card import _auto_chunk_size

    def probe(k):  # k**2 MiB: 2 MiB/dataset at k=2, 10 MiB marginal to k=8
        x = torch.empty(k * k * 2**18, device="cuda")  # 1 MiB per 2**18 floats
        del x

    _, rule = _auto_chunk_size(probe, 100)
    assert rule["probe_datasets"] == [2, 8]
    assert rule["mib_per_dataset"] == pytest.approx(10.0, abs=0.5)


class FakeAutoregressive(FakeApproximator):
    """Runs out of memory above ``max_rows`` rows per batch."""

    autoregressive = True

    def __init__(self, max_rows, **kw):
        super().__init__(**kw)
        self.max_rows = max_rows
        self.rows = []

    def sample(self, conditions, num_samples, condition_chunk_size):
        self.rows.append(condition_chunk_size)
        if condition_chunk_size > self.max_rows:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (fake)")
        return super().sample(conditions, num_samples)


def test_autoregressive_row_batch_shrinks_at_one_dataset():
    fake = FakeAutoregressive(max_rows=25)
    s = sample_cell(fake, SPEC_2PL, Cell(5, 10, True), _protocol(), chunk_size=2)
    # 2 datasets x 40 draws, then 1 x 40, then rows 30 and 22.
    assert fake.rows[:4] == [80, 40, 30, 22]
    assert set(fake.rows[4:]) == {22}
    assert s.autoregressive_rows == "at most 22"
    assert [r.get("retry_rows") for r in s.retries] == [None, 30, 22]
    assert s.draws["b"].shape[0] == 12


def test_autoregressive_one_row_out_of_memory_fails_the_cell():
    with pytest.raises(CellFailedError):
        sample_cell(
            FakeAutoregressive(max_rows=0),
            SPEC_2PL,
            Cell(5, 10, True),
            _protocol(),
            chunk_size=1,
        )


@pytest.mark.parametrize("share", [0, -0.5, 1.5])
def test_sample_cell_rejects_memory_share(share):
    with pytest.raises(ValueError, match="memory_share"):
        sample_cell(
            FakeApproximator(),
            SPEC_2PL,
            Cell(5, 10, True),
            _protocol(),
            chunk_size="auto",
            memory_share=share,
        )


def test_one_dataset_chunk_out_of_memory_fails_the_cell():
    fake = FakeApproximator(max_chunk=0)
    with pytest.raises(CellFailedError) as info:
        sample_cell(fake, SPEC_2PL, Cell(5, 10, True), _protocol(), chunk_size=1)
    assert info.value.record["size"] == 1 and "retry_size" not in info.value.record


def test_retry_runs_after_the_failed_attempt_is_released():
    """The failed call's frames must be gone before the halves run."""
    import gc
    import weakref

    class Holder:
        pass

    class Tracking(FakeApproximator):
        def __init__(self):
            super().__init__(max_chunk=3)
            self.alive_at_retry = []
            self.ref = None

        def sample(self, conditions, num_samples, **kwargs):
            n = conditions["responses"].shape[0]
            if n > self.max_chunk:
                held = Holder()  # stands in for activations on the failed frame
                self.ref = weakref.ref(held)
                raise torch.cuda.OutOfMemoryError("CUDA out of memory (fake)")
            gc.collect()
            self.alive_at_retry.append(self.ref is not None and self.ref() is not None)
            return super().sample(conditions, num_samples)

    fake = Tracking()
    sample_cell(
        fake, SPEC_2PL, Cell(5, 10, True), _protocol(datasets_per_cell=6), chunk_size=6
    )
    assert fake.alive_at_retry == [False, False]


def test_other_errors_are_not_swallowed():
    class Broken(FakeApproximator):
        def sample(self, conditions, num_samples, **kwargs):
            raise ValueError("not memory")

    with pytest.raises(ValueError, match="not memory"):
        sample_cell(Broken(), SPEC_2PL, Cell(5, 10, True), _protocol())


def test_reset_posterior_seed_resets_generators():
    fake = FakeApproximator()
    assert reset_posterior_seed(fake, 5) == 1
    first = keras.ops.convert_to_numpy(keras.random.normal((3,), seed=fake.generator))
    reset_posterior_seed(fake, 5)
    again = keras.ops.convert_to_numpy(keras.random.normal((3,), seed=fake.generator))
    assert np.array_equal(first, again)


def test_unsupported_family():
    with pytest.raises(ValueError, match="Unsupported model family"):
        sample_cell(
            FakeApproximator(), {"model": "3PL"}, Cell(5, 10, True), _protocol()
        )


@pytest.mark.slow
def test_real_checkpoint_is_reproducible():
    from bayesflow_irt.metadata import MetadataTracker

    path = ROOT / "models_trained/examples/2pl_equivariant.keras"
    if not path.exists():
        pytest.skip(f"needs a trained 2PL checkpoint at {path}")
    md = MetadataTracker.load(path.with_name("2pl_equivariant_metadata.json"))
    md = md.to_dict()
    appr = keras.saving.load_model(str(path), compile=False)
    proto = _protocol(datasets_per_cell=6, posterior_draws=30)
    kw = {"trained_at": md["provenance"]["timestamp"], "chunk_size": 4}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        one = sample_cell(appr, md["model_spec"], Cell(5, 20, True), proto, **kw)
        two = sample_cell(appr, md["model_spec"], Cell(5, 20, True), proto, **kw)
    assert np.array_equal(one.draws["b"], two.draws["b"])
    assert np.isfinite(one.draws["a"]).all() and (one.draws["a"] > 0).all()
