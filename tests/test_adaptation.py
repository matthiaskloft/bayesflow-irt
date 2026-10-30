"""Target-size adaptation persists a distinct, independently usable model."""

import bayesflow as bf
import keras
import numpy as np
import pytest

from bayesflow_irt import (
    EquivariantIRTApproximator,
    MatrixEquivariantIRTSummary,
    MetadataTracker,
    PersonScorer,
    Prior,
    VariablePrior,
    adapt_and_fit,
    make_1pl_adapter,
    make_1pl_simulator,
)
from bayesflow_irt.compat import weights_digest
from bayesflow_irt.metadata import _sha256_file


@pytest.fixture
def parent_checkpoint(tmp_path, request):
    variable = getattr(request, "param", False)
    prior_b = (
        VariablePrior("normal", role="b", scale_range=(0.5, 2), loc=0)
        if variable
        else Prior("normal", loc=0, scale=2)
    )
    simulator = make_1pl_simulator(
        n_items=(4, 6),
        n_persons=(10, 15),
        prior_b=prior_b,
        prior_theta=Prior("normal", loc=0, scale=1),
        seed=3,
    )
    model = EquivariantIRTApproximator(
        adapter=make_1pl_adapter(simulator),
        inference_network=bf.networks.CouplingFlow(
            subnet_kwargs={"widths": (8, 8)},
        ),
        summary_network=MatrixEquivariantIRTSummary(
            embed_dim=8,
            matrix_hidden_dim=8,
            num_matrix_blocks=1,
            num_outer_sabs=0,
            outer_embed_dim=8,
            outer_mlp_width=8,
        ),
    )
    model.compile(optimizer="adam")
    model.fit(
        simulator=simulator, epochs=1, num_batches=1, batch_size=2, workers=0, verbose=0
    )
    checkpoint = tmp_path / "parent.keras"
    model.save(checkpoint)
    tracker = MetadataTracker(model="1PL", variant="fixed", version="test")
    tracker.add_training_config(
        priors={
            "b": (
                {
                    "kind": "variable",
                    "name": "normal",
                    "role": "b",
                    "scale_range": [0.5, 2],
                    "fixed_kwargs": {"loc": 0},
                }
                if variable
                else {"name": "normal", "loc": 0, "scale": 2}
            ),
            "theta": {"name": "normal", "loc": 0, "scale": 1},
        },
        size_ranges={"n_items": [10, 15], "n_persons": [20, 30]},
    )
    tracker.attach(model)
    sidecar = tmp_path / "parent.json"
    tracker.save(sidecar, checkpoint_path=checkpoint)
    return checkpoint, sidecar


def run(parent, output, **kwargs):
    checkpoint, metadata = parent
    return adapt_and_fit(
        checkpoint,
        np.zeros((12, 4)),
        output,
        metadata=metadata,
        epochs=1,
        num_batches=1,
        batch_size=2,
        learning_rate=1e-4,
        seed=42,
        n_item_samples=5,
        fit_kwargs={"workers": 0, "verbose": 0},
        **kwargs,
    )


@pytest.mark.parametrize(
    "override",
    [
        None,
        Prior("normal", loc=0, scale=0.75),
        Prior("student_t", df=7, loc=1, scale=0.5),
    ],
)
def test_end_to_end(parent_checkpoint, tmp_path, monkeypatch, override):
    import bayesflow_irt.adaptation as module

    checkpoint, _ = parent_checkpoint
    digest = _sha256_file(checkpoint)
    initial = weights_digest(keras.saving.load_model(checkpoint, compile=False))
    parent_adapter = keras.saving.load_model(
        checkpoint, compile=False
    ).adapter.get_config()
    real_fit = module.fit_with_oom_retry

    def fit(model, simulator, **kwargs):
        assert weights_digest(model) == initial
        batch = simulator.sample((2,))
        assert batch["responses"].shape == (2, 12, 4)
        assert np.all(batch["n_items"] == 4)
        assert np.all(batch["n_persons"] == 12)
        expected = Prior("normal", loc=0, scale=2) if override is None else override
        assert simulator.prior_b == expected
        return real_fit(model, simulator, **kwargs)

    monkeypatch.setattr(module, "fit_with_oom_retry", fit)
    result = run(
        parent_checkpoint,
        tmp_path / "adapted",
        score_kwargs={},
        prior_overrides={} if override is None else {"b": override},
    )
    assert _sha256_file(checkpoint) == digest
    loaded = keras.saving.load_model(result.checkpoint, compile=False)
    assert weights_digest(loaded) == weights_digest(result.scorer.approximator)
    assert weights_digest(loaded) != initial
    assert loaded.get_config() == result.scorer.approximator.get_config()
    assert loaded.adapter.get_config() == parent_adapter
    record = MetadataTracker.load(result.metadata_path).to_dict()
    assert record == result.provenance
    assert record["adaptation"]["parent_checkpoint_sha256"] == digest
    assert record["model_spec"]["size_ranges"]["n_items"] == [4, 4]
    assert record["model_spec"]["priors"]["b"]["scale"] == (
        2 if override is None else override.kwargs["scale"]
    )
    parent = MetadataTracker.load(parent_checkpoint[1]).to_dict()
    assert record["adaptation"]["parent_metadata"] == parent
    assert parent["model_spec"]["priors"]["b"]["scale"] == 2
    assert (
        record["model_spec"]["priors"]["theta"]
        == parent["model_spec"]["priors"]["theta"]
    )
    assert record["adaptation"]["prior_overrides"] == (
        {} if override is None else {"b": {"name": override.name, **override.kwargs}}
    )
    assert result.item_posteriors["b"].shape == (5, 4)
    assert result.scores is not None
    assert result.adaptation_seconds > 0
    assert result.item_inference_seconds > 0
    assert result.scoring_seconds > 0
    loaded.inference_network.base_distribution.seed_generator.state.assign([91, 0])
    expected = PersonScorer(loaded).draw_item_posteriors(np.zeros((12, 4)), 5)
    result.scorer.approximator.inference_network.base_distribution.seed_generator.state.assign(
        [91, 0]
    )
    actual = PersonScorer(result.scorer.approximator).draw_item_posteriors(
        np.zeros((12, 4)),
        5,
    )
    np.testing.assert_allclose(expected["b"], actual["b"], atol=1e-6)


@pytest.mark.parametrize("failure", ["train", "save", "metadata", "partial"])
def test_failure_prevents_inference(parent_checkpoint, tmp_path, monkeypatch, failure):
    import bayesflow_irt.adaptation as module

    def fail(*args, **kwargs):
        raise RuntimeError("injected failure")

    if failure == "train":
        monkeypatch.setattr(module, "fit_with_oom_retry", fail)
    elif failure == "save":
        monkeypatch.setattr(keras.Model, "save", fail)
    elif failure == "metadata":
        monkeypatch.setattr(MetadataTracker, "save", fail)
    else:
        from types import SimpleNamespace

        monkeypatch.setattr(
            module,
            "fit_with_oom_retry",
            lambda *a, **k: SimpleNamespace(history=SimpleNamespace(history={})),
        )
    monkeypatch.setattr(
        PersonScorer,
        "draw_item_posteriors",
        lambda *a, **k: pytest.fail("inference must not run"),
    )
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError):
        run(parent_checkpoint, output)
    assert not output.exists()


def test_existing_output_rejected(parent_checkpoint):
    with pytest.raises(FileExistsError):
        run(parent_checkpoint, parent_checkpoint[0].parent)


@pytest.mark.parametrize(
    "overrides, error",
    [
        ({"missing": Prior("normal")}, ValueError),
        ({"b": {"scale": 1}}, TypeError),
        ({"b": Prior("normal", scale=-1)}, ValueError),
        ({"b": Prior("gamma")}, ValueError),
        ({"b": VariablePrior("lognormal", role="a", scale_range=(0.1, 1))}, ValueError),
        (
            {"theta": VariablePrior("normal", role="theta", scale_range=(0.5, 2))},
            ValueError,
        ),
    ],
)
def test_invalid_prior_overrides(parent_checkpoint, tmp_path, overrides, error):
    with pytest.raises(error):
        run(parent_checkpoint, tmp_path / "invalid", prior_overrides=overrides)


def test_variable_prior_cannot_add_condition_keys(parent_checkpoint, tmp_path):
    with pytest.raises(ValueError, match="preserve the adapter's sigma conditions"):
        run(
            parent_checkpoint,
            tmp_path / "variable",
            prior_overrides={
                "b": VariablePrior("normal", role="b", scale_range=(0.5, 2)),
            },
        )


def test_theta_override_controls_scoring(parent_checkpoint, tmp_path):
    result = run(
        parent_checkpoint,
        tmp_path / "theta",
        prior_overrides={
            "theta": Prior("normal", loc=0.5, scale=1.4),
        },
        score_kwargs={},
    )
    assert result.scorer.theta_prior_mean == 0.5
    assert result.scorer.theta_prior_scale == 1.4


def test_non_normal_theta_inference(parent_checkpoint, tmp_path):
    result = run(
        parent_checkpoint,
        tmp_path / "theta_t",
        prior_overrides={
            "theta": Prior("student_t", df=8),
        },
    )
    assert result.scorer is None
    assert result.item_posteriors["b"].shape == (5, 4)
    with pytest.raises(ValueError, match="fixed Normal theta prior"):
        run(
            parent_checkpoint,
            tmp_path / "unsupported_scoring",
            prior_overrides={
                "theta": Prior("student_t", df=8),
            },
            score_kwargs={},
        )


@pytest.mark.parametrize("parent_checkpoint", [True], indirect=True)
def test_variable_prior_round_trip(parent_checkpoint, tmp_path):
    from bayesflow_irt import build_simulator

    prior = VariablePrior("normal", role="b", scale_range=(0.2, 1), loc=0.3)
    result = run(
        parent_checkpoint,
        tmp_path / "variable",
        prior_overrides={"b": prior},
        item_conditions={"sigma_b": 0.6},
    )
    metadata = MetadataTracker.load(result.metadata_path).to_dict()
    sim = build_simulator(metadata["model_spec"], seed=3)
    assert isinstance(sim.prior_b, VariablePrior)
    assert sim.prior_b.scale_range == prior.scale_range
    assert sim.prior_b.fixed_kwargs == prior.fixed_kwargs
    sigma = sim.sample((20,))["sigma_b"]
    assert np.all((sigma >= 0.2) & (sigma <= 1))
    assert result.item_posteriors["b"].shape == (5, 4)
