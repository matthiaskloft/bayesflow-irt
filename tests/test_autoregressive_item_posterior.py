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

_AR = "autoregressive_coupling_flow"


def _simulator(n_items=(3, 6), n_persons=(8, 12)):
    return make_2pl_simulator(
        n_items=n_items,
        n_persons=n_persons,
        prior_a=Prior("lognormal", loc=0, scale=0.5),
        prior_b=Prior("normal", loc=0, scale=1.0),
        seed=0,
    )


def _make(simulator, **kw):
    kw.setdefault("item_posterior", _AR)
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
        inference_network=bf.networks.CouplingFlow(
            depth=2, subnet_kwargs={"widths": (16, 16)}
        ),
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


def test_default_is_flow_matching():
    simulator = _simulator()
    approximator = EquivariantIRTApproximator(
        adapter=make_2pl_adapter(simulator),
        summary_network=MatrixEquivariantIRTSummary(embed_dim=8),
        inference_network=bf.networks.FlowMatching(),
    )
    assert approximator.item_posterior == "flow_matching"
    assert approximator.draws_summary is None


def test_invalid_item_posterior():
    with pytest.raises(ValueError, match="item_posterior"):
        _make(_simulator(), item_posterior="nope")


def test_rejects_standardize_params():
    with pytest.raises(ValueError, match="standardize_params"):
        _make(_simulator(), standardize_params=True)


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
    assert restored.item_posterior == _AR
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


def test_requires_coupling_flow():
    simulator = _simulator()
    with pytest.raises(TypeError, match="CouplingFlow"):
        EquivariantIRTApproximator(
            adapter=make_2pl_adapter(simulator),
            summary_network=MatrixEquivariantIRTSummary(embed_dim=8),
            inference_network=bf.networks.FlowMatching(),
            item_posterior=_AR,
        )


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
        item_posterior=_AR,
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
    from bayesflow_irt.utils import (
        _build_probe_approximator,
        profile_posterior_inference,
    )

    simulator, approximator = built
    clone = _build_probe_approximator(approximator)
    assert clone.item_posterior == _AR
    assert clone.draws_summary_dim == approximator.draws_summary_dim
    with pytest.raises(NotImplementedError):
        profile_posterior_inference(approximator, simulator.sample((1,)), num_samples=2)


def test_calibrated_rejects_autoregressive():
    pytest.importorskip("bayesflow_calibration_loss")
    from bayesflow_irt import CalibratedIRTApproximator

    simulator = _simulator()
    with pytest.raises(NotImplementedError):
        CalibratedIRTApproximator(
            adapter=make_2pl_adapter(simulator),
            summary_network=MatrixEquivariantIRTSummary(embed_dim=8),
            inference_network=bf.networks.CouplingFlow(),
            item_posterior=_AR,
        )


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
