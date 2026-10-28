"""Tests for the autoregressive item posterior of ``EquivariantIRTApproximator``."""

import bayesflow as bf
import keras
import numpy as np
import pytest

from bayesflow_irt import (
    EquivariantIRTApproximator,
    MatrixEquivariantIRTSummary,
    Prior,
    make_2pl_adapter,
    make_2pl_simulator,
)


def _simulator(n_items=(3, 6), n_persons=(8, 12)):
    return make_2pl_simulator(
        n_items=n_items,
        n_persons=n_persons,
        prior_a=Prior("lognormal", loc=0, scale=0.5),
        prior_b=Prior("normal", loc=0, scale=1.0),
        seed=0,
    )


def _make(simulator, inference_network=None, **kw):
    if inference_network is None:
        inference_network = bf.networks.CouplingFlow(
            depth=2, subnet_kwargs={"widths": (16, 16)}
        )
    return EquivariantIRTApproximator(
        adapter=make_2pl_adapter(simulator),
        summary_network=MatrixEquivariantIRTSummary(
            embed_dim=8,
            matrix_hidden_dim=8,
            num_matrix_blocks=1,
            num_outer_sabs=1,
            outer_embed_dim=8,
            outer_mlp_width=16,
        ),
        inference_network=inference_network,
        draws_summary_hidden_dim=8,
        draws_summary_dim=8,
        **kw,
    )


def _adapted(approximator, data):
    adapted = approximator.adapter(data)
    return {k: keras.ops.convert_to_tensor(v) for k, v in adapted.items()}


_POOLINGS = ("mean", "mean_std", "mean_max", "attention", "pma")


@pytest.fixture(scope="module", params=_POOLINGS)
def built(request):
    simulator = _simulator(n_items=(10, 10), n_persons=(8, 8))
    approximator = _make(simulator, draws_summary_pooling=request.param)
    approximator.build_from_data(_adapted(approximator, simulator.sample((4,))))
    return simulator, approximator


@pytest.mark.parametrize("item_posterior", [None, "flow_matching"])
def test_independent_item_posterior_config_is_rejected(item_posterior):
    """Configs saved with the removed independent item posterior fail clearly."""
    approximator = _make(_simulator())
    config = approximator.get_config()
    assert config["item_posterior"] == "autoregressive_coupling_flow"
    if item_posterior is None:
        del config["item_posterior"]
    else:
        config["item_posterior"] = item_posterior
    with pytest.raises(ValueError, match="independent item posterior"):
        EquivariantIRTApproximator.from_config(config)


def test_flow_matching_trains_samples_and_reloads(tmp_path):
    """A FlowMatching per-item flow works in the autoregressive loop."""
    simulator = _simulator()
    approximator = _make(
        simulator,
        inference_network=bf.networks.FlowMatching(
            subnet_kwargs={"widths": (16, 16)},
            integrate_kwargs={"method": "euler", "steps": 2},
        ),
    )
    approximator.compile(optimizer="adam")
    history = approximator.fit(
        simulator=simulator,
        epochs=1,
        num_batches=2,
        batch_size=4,
        workers=1,
        verbose=False,
    )
    assert np.all(np.isfinite(history.history["loss"]))

    data = simulator.sample((2,))
    out = approximator.sample(num_samples=3, conditions=data)
    assert out["a"].shape == (2, 3, data["a"].shape[-1])
    assert np.all(np.isfinite(out["a"]))

    path = tmp_path / "ar_fm.keras"
    approximator.save(path)
    restored = keras.saving.load_model(path)
    assert isinstance(restored.inference_network, bf.networks.FlowMatching)
    assert np.all(np.isfinite(restored.sample(num_samples=2, conditions=data)["b"]))


@pytest.mark.parametrize("n_items,n_persons", [(13, 7), (4, 5), (1, 3)])
def test_sample_unseen_sizes(built, n_items, n_persons):
    simulator, approximator = built  # built at I=10, P=8
    sim = simulator.clone_with_sizes(
        n_items=(n_items, n_items), n_persons=(n_persons, n_persons)
    )
    out = approximator.sample(num_samples=3, conditions=sim.sample((2,)))
    for k in ("a", "b", "d"):
        assert out[k].shape == (2, 3, n_items)
        assert np.all(np.isfinite(out[k]))
    assert np.all(out["a"] > 0)


@pytest.mark.parametrize("row_chunk", [256, 4])
def test_sampling_conditions_match_teacher_forcing(built, monkeypatch, row_chunk):
    """Item ``i`` is sampled under the condition training uses for the drawn
    items, i.e. the sampler's draws summary is the teacher-forced one. A row
    chunk of 4 does not divide the 6 (dataset, draw) rows."""
    simulator, approximator = built
    data = simulator.sample((2,))
    adapted = _adapted(approximator, data)
    summary = adapted["summary_variables"]
    glob = adapted["inference_conditions"]
    flow = approximator.inference_network
    seen = []
    rng = np.random.default_rng(0)

    def fake_sample(batch_shape, conditions=None, **kwargs):
        seen.append(keras.ops.convert_to_numpy(conditions[:, 0]))
        x = rng.normal(size=(*batch_shape, 2)).astype("float32")
        seen.append(x[:, 0])
        return keras.ops.convert_to_tensor(x)

    monkeypatch.setattr(flow, "sample", fake_sample)
    num_samples = 3
    samples = approximator._sample(
        num_samples,
        summary_variables=summary,
        inference_conditions=glob,
        condition_chunk_size=row_chunk,
    )
    per_call = seen[0::2]
    num_items = summary.shape[-1]
    conds = np.concatenate(
        [
            np.stack(per_call[k : k + num_items], axis=1)
            for k in range(0, len(per_call), num_items)
        ],
        axis=0,
    )  # (B*S, I, C)

    h = approximator.summary_network(summary)
    h_rows = keras.ops.repeat(h, num_samples, axis=0)
    items = keras.ops.reshape(samples, (-1, *samples.shape[2:]))
    per_item = approximator._item_conditions(h_rows, items)
    g = keras.ops.repeat(glob, num_samples, axis=0)
    g = keras.ops.broadcast_to(
        keras.ops.expand_dims(g, 1), (*per_item.shape[:2], g.shape[-1])
    )
    expected = keras.ops.convert_to_numpy(keras.ops.concatenate([per_item, g], axis=-1))
    np.testing.assert_allclose(conds, expected, atol=1e-5)


class _Rows:
    """Spy on the flow's sample calls of ``built`` (10 items), recording the
    rows per call; calls above ``oom_above`` rows raise a fake CUDA OOM."""

    def __init__(self, built, monkeypatch, oom_above=None):
        import torch

        simulator, self.approximator = built
        self.adapted = _adapted(self.approximator, simulator.sample((2,)))
        flow = self.approximator.inference_network
        real_sample = flow.sample
        self.rows = []

        def spy(batch_shape, **kwargs):
            if oom_above is not None and batch_shape[0] > oom_above:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory (fake)")
            self.rows.append(batch_shape[0])
            return real_sample(batch_shape, **kwargs)

        monkeypatch.setattr(flow, "sample", spy)
        # the fixture is shared: keep its stored measurement out of other tests
        monkeypatch.setattr(self.approximator, "_ar_row_bytes", None)

    def draw(self, num_samples=300, **kw):
        self.rows.clear()
        out = self.approximator._sample(
            num_samples,
            summary_variables=self.adapted["summary_variables"],
            inference_conditions=self.adapted["inference_conditions"],
            **kw,
        )
        assert out.shape == (2, num_samples, 10, 2)
        return list(self.rows)


def _fake_cuda(monkeypatch, free_bytes, peak_bytes):
    """Make ``"auto"`` see a CUDA device with these memory readings; return
    the list that counts peak measurements."""
    from bayesflow_irt import approximators as module

    starts = []
    monkeypatch.setattr(module, "_torch_cuda", lambda: True)
    monkeypatch.setattr(module, "_device_peak_start", lambda: starts.append(1) or 0)
    monkeypatch.setattr(module, "_device_peak_since", lambda base: peak_bytes)
    monkeypatch.setattr(module, "_device_free_bytes", lambda: free_bytes)
    return starts


def test_row_chunk_without_cuda_is_fixed(built, monkeypatch):
    """Off CUDA, ``"auto"`` is ``AR_ROW_CHUNK_SIZE`` rows per flow call. A flow
    call costs about the same for 1 or thousands of rows, so the old default
    of 256 rows made sampling 15-70x slower than one chunk."""
    from bayesflow_irt import approximators as module

    monkeypatch.setattr(module, "_torch_cuda", lambda: False)
    spy = _Rows(built, monkeypatch)
    assert spy.draw() == [600] * 10
    monkeypatch.setattr(module, "AR_ROW_CHUNK_SIZE", 256)
    assert spy.draw() == [256] * 10 + [256] * 10 + [88] * 10
    assert spy.approximator._ar_row_bytes is None


def test_auto_rows_are_measured_once_and_reused(built, monkeypatch):
    """The first chunk of at least ``AR_MEASURE_MIN_ROWS`` rows stores the
    peak per row; later calls size chunks from it without measuring again."""
    # 1200 rows peak at 1.2 MB: 1000 B/row; 0.8 * 625 kB fits 500 rows.
    starts = _fake_cuda(monkeypatch, free_bytes=625_000, peak_bytes=1_200_000)
    spy = _Rows(built, monkeypatch)
    assert spy.draw(600) == [1200] * 10  # unmeasured: up to AR_ROW_CHUNK_SIZE
    assert spy.approximator._ar_row_bytes == pytest.approx(1000)
    assert spy.draw(600) == [500] * 10 + [500] * 10 + [200] * 10
    assert spy.draw(600) == [500] * 10 + [500] * 10 + [200] * 10
    assert len(starts) == 1
    # an integer is used as given and leaves the measurement alone
    assert spy.draw(600, condition_chunk_size=700) == [700] * 10 + [500] * 10
    assert len(starts) == 1


def test_auto_rows_skip_measuring_small_chunks(built, monkeypatch):
    """A chunk below ``AR_MEASURE_MIN_ROWS`` is dominated by fixed overhead,
    so it is not stored."""
    starts = _fake_cuda(monkeypatch, free_bytes=625_000, peak_bytes=1_200_000)
    spy = _Rows(built, monkeypatch)
    assert spy.draw(100) == [200] * 10
    assert starts == [] and spy.approximator._ar_row_bytes is None


def test_auto_rows_scale_with_items(built, monkeypatch):
    """More items than measured at cost proportionally more per row; fewer
    keep the stored cost."""
    _fake_cuda(monkeypatch, free_bytes=1_000_000, peak_bytes=1)
    approximator = built[1]
    monkeypatch.setattr(approximator, "_ar_row_bytes", 1000.0)
    monkeypatch.setattr(approximator, "_ar_row_items", 10)
    assert approximator._auto_row_chunk(10, 10**6) == 800
    assert approximator._auto_row_chunk(20, 10**6) == 400
    assert approximator._auto_row_chunk(5, 10**6) == 800
    assert approximator._auto_row_chunk(10, 300) == 300


def test_auto_rows_shrink_on_out_of_memory(built, monkeypatch):
    """A chunk that runs out of memory shrinks by ``CHUNK_SHRINK`` until it
    fits, and the stored per-row cost grows so the next call starts there."""
    # stored 1000 B/row and 1.5 MB free: chunks of 1200, which OOM above 600
    _fake_cuda(monkeypatch, free_bytes=1_500_000, peak_bytes=1)
    spy = _Rows(built, monkeypatch, oom_above=600)
    monkeypatch.setattr(spy.approximator, "_ar_row_bytes", 1000.0)
    monkeypatch.setattr(spy.approximator, "_ar_row_items", 10)
    # 1200 -> 900 -> 675 -> 506 fits: chunks 506, 506, 188
    assert spy.draw(600) == [506] * 10 + [506] * 10 + [188] * 10
    assert spy.approximator._ar_row_bytes == pytest.approx(1000 / 0.75**3)
    rows = spy.draw(600)
    assert max(rows) <= 600 and sum(rows) == 1200 * 10


def test_auto_rows_measure_the_chunk_that_fits_after_out_of_memory(built, monkeypatch):
    """Without a stored cost, the first chunk that fits after an
    out-of-memory error is measured even below ``AR_MEASURE_MIN_ROWS``, so a
    loop of calls does not run out of memory on every call."""
    # 1200 -> 900 -> 675 -> 506 fits; its 506 kB peak is 1000 B/row.
    starts = _fake_cuda(monkeypatch, free_bytes=750_000, peak_bytes=506_000)
    spy = _Rows(built, monkeypatch, oom_above=600)
    # 0.8 * 750 kB / 1000 B = 600 rows for the rest and for later calls
    assert spy.draw(600) == [506] * 10 + [600] * 10 + [94] * 10
    assert spy.approximator._ar_row_bytes == pytest.approx(1000)
    attempts = len(starts)  # one per attempt of the first chunk
    assert spy.draw(600) == [600] * 10 + [600] * 10
    assert len(starts) == attempts


def test_fixed_rows_do_not_retry_out_of_memory(built, monkeypatch):
    """An integer ``condition_chunk_size`` is the caller's choice: an
    out-of-memory error propagates (``evaluation.sample_posterior`` handles
    it)."""
    import torch

    _fake_cuda(monkeypatch, free_bytes=10**9, peak_bytes=1)
    spy = _Rows(built, monkeypatch, oom_above=100)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        spy.draw(600, condition_chunk_size=200)


@pytest.mark.parametrize("pooling", _POOLINGS)
def test_fit_save_load(tmp_path, pooling):
    simulator = _simulator(n_items=(2, 6), n_persons=(5, 12))
    approximator = _make(simulator, draws_summary_pooling=pooling)
    approximator.compile(optimizer="adam")
    history = approximator.fit(
        simulator=simulator,
        epochs=1,
        num_batches=3,
        batch_size=4,
        workers=1,
        verbose=False,
    )
    assert np.all(np.isfinite(history.history["loss"]))

    path = tmp_path / "ar.keras"
    approximator.save(path)
    restored = keras.saving.load_model(path)
    assert restored.draws_summary_pooling == pooling

    data = simulator.sample((3,))
    adapted = _adapted(approximator, data)
    x = adapted["inference_variables"]
    y = adapted["summary_variables"]
    n_items = x.shape[1]
    conds = []
    for model in (approximator, restored):
        h = model.summary_network(y)
        c = model._flatten_conditions(
            model._item_conditions(h, x),
            adapted["inference_conditions"],
            x.shape[0],
            n_items,
        )
        conds.append(c)
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(conds[0]),
        keras.ops.convert_to_numpy(conds[1]),
        atol=1e-5,
    )
    flat_x = keras.ops.reshape(x, (-1, 2))
    lp = [
        keras.ops.convert_to_numpy(m.inference_network.log_prob(flat_x, conditions=c))
        for m, c in zip((approximator, restored), conds)
    ]
    np.testing.assert_allclose(lp[0], lp[1], atol=1e-4)

    out = restored.sample(num_samples=2, conditions=data)
    assert out["a"].shape == (3, 2, data["a"].shape[-1])


def test_draws_summary_is_causal(built):
    """Position i's summary reads only draws j < i (and, with attention, its
    own embedding h_i as the query). Perturbing positions > k therefore leaves
    positions <= k unchanged and changes positions > k + 1."""
    _, approximator = built
    layer = approximator.draws_summary
    rng = np.random.default_rng(2)
    x = rng.normal(size=(2, 8, 2)).astype("float32")
    h = rng.normal(size=(2, 8, 8)).astype("float32")

    def run(x_, h_):
        return keras.ops.convert_to_numpy(
            layer(keras.ops.convert_to_tensor(x_), keras.ops.convert_to_tensor(h_))
        )

    base = run(x, h)
    k = 4
    x2, h2 = x.copy(), h.copy()
    x2[:, k + 1 :] += rng.normal(size=x2[:, k + 1 :].shape)
    h2[:, k + 1 :] += rng.normal(size=h2[:, k + 1 :].shape)
    pert = run(x2, h2)
    np.testing.assert_allclose(base[:, : k + 1], pert[:, : k + 1], atol=1e-5)
    assert np.abs(base[:, k + 2 :] - pert[:, k + 2 :]).max() > 1e-4


def test_invalid_draws_summary_pooling():
    with pytest.raises(ValueError, match="pooling"):
        _make(_simulator(), draws_summary_pooling="sum")


@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"flow_chunk_size": 5},
        {"gradient_accumulation_steps": 2},
        {"draws_summary_pooling": "attention"},
        {"draws_summary_pooling": "pma"},
        {"draws_summary_pooling": "mean_std"},
    ],
)
def test_fit_trains_draws_summary(kw):
    simulator = _simulator()
    approximator = _make(simulator, **kw)
    approximator.build_from_data(_adapted(approximator, simulator.sample((4,))))
    before = [np.array(w) for w in approximator.draws_summary.weights]
    approximator.compile(optimizer="adam")
    history = approximator.fit(
        simulator=simulator,
        epochs=1,
        num_batches=4,
        batch_size=4,
        workers=1,
        verbose=False,
    )
    assert np.all(np.isfinite(history.history["loss"]))
    after = [np.array(w) for w in approximator.draws_summary.weights]
    # every draws-summary weight, including the position-0 constant, trains
    assert all(np.abs(a - b).max() > 0 for a, b in zip(after, before))


def test_one_pl():
    from bayesflow_irt import make_1pl_adapter, make_1pl_simulator

    simulator = make_1pl_simulator(
        n_items=(3, 6), n_persons=(8, 12), prior_b=Prior("normal", loc=0, scale=1)
    )
    approximator = EquivariantIRTApproximator(
        adapter=make_1pl_adapter(simulator),
        summary_network=MatrixEquivariantIRTSummary(
            embed_dim=8, matrix_hidden_dim=8, num_matrix_blocks=1, num_outer_sabs=1
        ),
        inference_network=bf.networks.CouplingFlow(depth=2),
        draws_summary_hidden_dim=8,
        draws_summary_dim=8,
    )
    approximator.compile(optimizer="adam")
    approximator.fit(
        simulator=simulator,
        epochs=1,
        num_batches=2,
        batch_size=4,
        workers=1,
        verbose=False,
    )
    data = simulator.sample((2,))
    out = approximator.sample(num_samples=3, conditions=data)
    assert out["b"].shape == (2, 3, data["b"].shape[-1])
    assert np.all(np.isfinite(out["b"]))


def test_utils_paths(built):
    from bayesflow_irt.utils import _build_probe_approximator

    _, approximator = built
    clone = _build_probe_approximator(approximator)
    assert clone.draws_summary_dim == approximator.draws_summary_dim


def test_integer_seed_does_not_repeat_streams(built):
    """An integer ``seed`` in ``flow_sample_kwargs`` must not restart the same
    latent stream for every item and chunk (bayesflow versions that honour
    the seed would otherwise repeat one stream)."""
    simulator, approximator = built
    data = simulator.sample((2,))
    adapted = _adapted(approximator, data)
    kw = {"flow_sample_kwargs": {"seed": 7}}
    out = approximator._sample(
        4,
        summary_variables=adapted["summary_variables"],
        inference_conditions=adapted["inference_conditions"],
        condition_chunk_size=4,
        **kw,
    )
    out = keras.ops.convert_to_numpy(out)  # (B, S, I, P)
    flat = out.reshape(-1, out.shape[2], out.shape[3])
    # Draws of different rows/items are not identical copies of one stream.
    assert len(np.unique(np.round(flat.reshape(flat.shape[0], -1), 5), axis=0)) == len(
        flat
    )
