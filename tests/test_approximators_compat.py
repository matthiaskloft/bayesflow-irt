"""Compatibility regression tests for EquivariantIRTApproximator."""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock, patch

import bayesflow as bf
import keras
import numpy as np
import pytest
from bayesflow.approximators.continuous_approximator import ContinuousApproximator
from conftest import bare_model_init, strip_registered_name_prefixes

from bayesflow_irt.adapters import make_2pl_adapter
from bayesflow_irt.approximators import EquivariantIRTApproximator
from bayesflow_irt.networks import MatrixEquivariantIRTSummary
from bayesflow_irt.priors import Prior
from bayesflow_irt.simulators import make_2pl_simulator


def test_init_accepts_legacy_standardize_kwarg() -> None:
    """Constructor should tolerate serialized `standardize` kwargs."""
    with patch.object(
        ContinuousApproximator, "__init__", autospec=True, side_effect=bare_model_init
    ) as super_init:
        EquivariantIRTApproximator(
            adapter=MagicMock(),
            inference_network=MagicMock(),
            summary_network=MagicMock(),
            standardize=["legacy"],
        )

    assert super_init.called
    _, kwargs = super_init.call_args
    assert kwargs["standardize"] == []


def test_sample_accepts_positional_conditions_and_n_samples_alias() -> None:
    """sample() should support legacy call patterns used by downstream code."""
    with patch.object(ContinuousApproximator, "__init__", bare_model_init):
        approx = EquivariantIRTApproximator(
            adapter=MagicMock(),
            inference_network=MagicMock(),
            summary_network=MagicMock(),
        )

    adapted = {
        "summary_variables": np.zeros((1, 4, 3), dtype=np.float32),
        "inference_conditions": np.zeros((1, 0), dtype=np.float32),
    }
    inv = {
        "a": np.ones((1, 5, 3, 1), dtype=np.float32),
        "b": np.full((1, 5, 3, 1), 2.0, dtype=np.float32),
    }
    approx.adapter = MagicMock(side_effect=[adapted, inv])
    setattr(approx.adapter, "_bayesflow_irt_internal_parameterization", "loading")
    approx._sample = MagicMock(return_value=np.zeros((1, 5, 3, 2), dtype=np.float32))

    out = approx.sample({"responses": adapted["summary_variables"]}, n_samples=5)

    assert approx._sample.call_count == 1
    assert approx._sample.call_args.kwargs["num_samples"] == 5
    assert "d" in out
    assert out["d"].shape == (1, 5, 3, 1)


def test_sample_summary_batch_size_chunks_summary_forward() -> None:
    with patch.object(ContinuousApproximator, "__init__", bare_model_init):
        approx = EquivariantIRTApproximator(
            adapter=MagicMock(),
            inference_network=MagicMock(),
            summary_network=MagicMock(),
        )

    class SummaryNet:
        def __init__(self):
            self.calls = 0

        def __call__(self, x, training=False):
            self.calls += 1
            b = int(x.shape[0])
            return keras.ops.zeros((b, 2, 3), dtype="float32")

    class InferenceNet:
        def __init__(self):
            self.calls = []

        def sample(self, shape, conditions=None, **kwargs):
            self.calls.append((shape, kwargs))
            n_cond, n_sample = shape
            return keras.ops.zeros((n_cond, n_sample, 1), dtype="float32")

    approx.summary_network = SummaryNet()
    approx.inference_network = InferenceNet()
    approx._global_condition_dim = 0
    approx.draws_summary.build((None, None, 1), (None, None, 3))

    summary_variables = keras.ops.zeros((3, 4, 2), dtype="float32")
    out = approx._sample(
        num_samples=4,
        summary_variables=summary_variables,
        summary_batch_size=1,
        condition_chunk_size=2,
        flow_sample_kwargs={"method": "euler", "steps": 16},
    )

    assert approx.summary_network.calls == 3
    assert approx.inference_network.calls
    assert all(
        kwargs == {"method": "euler", "steps": 16}
        for _, kwargs in approx.inference_network.calls
    )
    assert tuple(out.shape) == (3, 4, 2, 1)


def test_init_rejects_per_chunk_backward_mode() -> None:
    with patch.object(ContinuousApproximator, "__init__", bare_model_init):
        with pytest.raises(ValueError, match="per_chunk"):
            EquivariantIRTApproximator(
                adapter=MagicMock(),
                inference_network=MagicMock(),
                summary_network=MagicMock(),
                flow_chunk_backward="per_chunk",
            )


def test_init_normalizes_legacy_none_backward_mode() -> None:
    with patch.object(ContinuousApproximator, "__init__", bare_model_init):
        approx = EquivariantIRTApproximator(
            adapter=MagicMock(),
            inference_network=MagicMock(),
            summary_network=MagicMock(),
            flow_chunk_backward="none",
        )

    assert approx.flow_chunk_backward is None


def test_chunked_weighted_loss_equals_unchunked_weighted_loss() -> None:
    with patch.object(ContinuousApproximator, "__init__", bare_model_init):
        approx = EquivariantIRTApproximator(
            adapter=MagicMock(),
            inference_network=MagicMock(),
            summary_network=MagicMock(),
            flow_chunk_size=2,
            flow_chunk_backward="none",
        )

    class WeightedInferenceNet:
        def compute_metrics(
            self, params, conditions=None, sample_weight=None, stage="training"
        ):
            value = params[:, 0]
            if sample_weight is None:
                loss = keras.ops.mean(value)
            else:
                # BayesFlow's reduction: mean(loss * sample_weight) over rows.
                loss = keras.ops.mean(value * sample_weight)
            return {"loss": loss}

    approx.inference_network = WeightedInferenceNet()

    flat_params = keras.ops.convert_to_tensor(
        np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
    )
    flat_conditions = keras.ops.zeros((4, 1), dtype="float32")
    sample_weight = keras.ops.convert_to_tensor(
        np.array([1.0, 100.0, 1.0, 1.0], dtype=np.float32)
    )

    weighted_chunked = approx._chunked_inference_loss(
        flat_params,
        flat_conditions,
        sample_weight=sample_weight,
        stage="training",
    )

    approx.flow_chunk_size = None
    weighted_unchunked = approx._chunked_inference_loss(
        flat_params,
        flat_conditions,
        sample_weight=sample_weight,
        stage="training",
    )

    weighted_chunked_np = float(keras.ops.convert_to_numpy(weighted_chunked))
    weighted_unchunked_np = float(keras.ops.convert_to_numpy(weighted_unchunked))
    assert weighted_chunked_np == pytest.approx(weighted_unchunked_np, abs=1e-5)


def test_checkpoint_chunking_preserves_summary_gradient_path() -> None:
    if keras.backend.backend() != "torch":
        pytest.skip("checkpoint chunking test is torch-specific")

    import torch

    with patch.object(ContinuousApproximator, "__init__", bare_model_init):
        approx = EquivariantIRTApproximator(
            adapter=MagicMock(),
            inference_network=MagicMock(),
            summary_network=MagicMock(),
            flow_chunk_size=2,
            flow_chunk_backward="checkpoint",
        )

    class InferenceNet:
        def compute_metrics(
            self, params, conditions=None, sample_weight=None, stage="training"
        ):
            return {"loss": (params[:, 0] * conditions[:, 0]).mean()}

    approx.inference_network = InferenceNet()

    projector = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        projector.weight.fill_(2.0)
    with torch.enable_grad():
        flat_conditions = projector(torch.ones(4, 1))
        flat_params = torch.ones(4, 1)
        loss = approx._chunked_inference_loss(
            flat_params,
            flat_conditions,
            sample_weight=None,
            stage="training",
        )
        loss.backward()

    assert projector.weight.grad is not None
    assert torch.any(projector.weight.grad != 0)


def test_chunked_loss_equals_unchunked_loss() -> None:
    """Chunked weighted-average loss must equal the plain per-sample mean."""
    with patch.object(ContinuousApproximator, "__init__", bare_model_init):
        approx = EquivariantIRTApproximator(
            adapter=MagicMock(),
            inference_network=MagicMock(),
            summary_network=MagicMock(),
            flow_chunk_size=3,
        )

    class MeanInferenceNet:
        """Returns mean of first column — easy to verify analytically."""

        def compute_metrics(
            self, params, conditions=None, sample_weight=None, stage="training"
        ):
            return {"loss": keras.ops.mean(params[:, 0])}

    approx.inference_network = MeanInferenceNet()

    # flat_params[:, 0] = [0, 1, 2, 3, 4, 5]; expected mean = 2.5
    flat_params = keras.ops.convert_to_tensor(
        np.arange(6, dtype=np.float32).reshape(6, 1)
    )
    flat_conditions = keras.ops.zeros((6, 2), dtype="float32")

    # Unchunked baseline (chunk_size=None bypasses loop)
    approx.flow_chunk_size = None
    loss_unchunked = float(
        keras.ops.convert_to_numpy(
            approx._chunked_inference_loss(
                flat_params, flat_conditions, stage="validation"
            )
        )
    )

    # Equal chunks: size=3 → two chunks of 3
    approx.flow_chunk_size = 3
    loss_equal = float(
        keras.ops.convert_to_numpy(
            approx._chunked_inference_loss(
                flat_params, flat_conditions, stage="validation"
            )
        )
    )

    # Unequal chunks: size=4 → chunks of [4, 2]; weighted avg must still = 2.5
    # chunk1: mean([0,1,2,3])=1.5 weight=4; chunk2: mean([4,5])=4.5 weight=2
    # → (1.5*4 + 4.5*2) / 6 = 2.5
    approx.flow_chunk_size = 4
    loss_unequal = float(
        keras.ops.convert_to_numpy(
            approx._chunked_inference_loss(
                flat_params, flat_conditions, stage="validation"
            )
        )
    )

    assert loss_unchunked == pytest.approx(2.5, rel=1e-5)
    assert loss_equal == pytest.approx(2.5, rel=1e-5)
    assert loss_unequal == pytest.approx(2.5, rel=1e-5)


def test_gradient_accumulation_applies_at_correct_cadence() -> None:
    """optimizer.apply called once per ga_steps; _accum_step resets after apply."""
    if keras.backend.backend() != "torch":
        pytest.skip("gradient accumulation uses torch-native backward")

    import torch

    with patch.object(ContinuousApproximator, "__init__", bare_model_init):
        approx = EquivariantIRTApproximator(
            adapter=MagicMock(),
            inference_network=MagicMock(),
            summary_network=MagicMock(),
            gradient_accumulation_steps=3,
        )

    # Real torch parameter so loss.backward() actually computes gradients
    param = torch.nn.Parameter(torch.zeros(1))
    fake_weight = MagicMock()
    fake_weight.value = param

    optimizer = MagicMock()
    optimizer.apply = MagicMock()
    approx.optimizer = optimizer
    approx.zero_grad = MagicMock()
    approx._update_metrics = MagicMock()
    approx._batch_size_from_data = MagicMock(return_value=4)

    fake_data = {"inference_variables": MagicMock(), "summary_variables": MagicMock()}

    def fake_compute_metrics(**kwargs):
        return {"loss": param.sum()}  # grad_fn present → backward works

    approx.compute_metrics = fake_compute_metrics

    with patch.object(
        type(approx),
        "trainable_weights",
        new_callable=PropertyMock,
        return_value=[fake_weight],
    ):
        approx.train_step(fake_data)  # micro-step 1
        assert approx._accum_step == 1, "accum_step should be 1 after step 1"
        assert optimizer.apply.call_count == 0, "no apply after step 1"

        approx.train_step(fake_data)  # micro-step 2
        assert approx._accum_step == 2
        assert optimizer.apply.call_count == 0, "no apply after step 2"

        approx.train_step(fake_data)  # micro-step 3 → apply + reset
        assert approx._accum_step == 0, "step counter must reset after apply"
        assert optimizer.apply.call_count == 1, "optimizer applied exactly once"
        assert approx.zero_grad.called, "zero_grad called to clear accum"

        approx.train_step(fake_data)  # micro-step 4 → new window
        assert approx._accum_step == 1, "new accumulation window started"
        assert optimizer.apply.call_count == 1, "no second apply at step 4"


def test_gradient_accumulation_matches_large_batch_gradient() -> None:
    if keras.backend.backend() != "torch":
        pytest.skip("gradient accumulation uses torch-native backward")

    import torch

    with patch.object(ContinuousApproximator, "__init__", bare_model_init):
        approx = EquivariantIRTApproximator(
            adapter=MagicMock(),
            inference_network=MagicMock(),
            summary_network=MagicMock(),
            gradient_accumulation_steps=3,
        )

    param = torch.nn.Parameter(torch.tensor([0.0]))
    fake_weight = MagicMock()
    fake_weight.value = param
    captured = {}

    def fake_apply(grads, _weights):
        captured["grad"] = grads[0].detach().clone()

    approx.optimizer = MagicMock()
    approx.optimizer.apply = fake_apply
    approx.zero_grad = MagicMock()
    approx._update_metrics = MagicMock()
    approx._batch_size_from_data = MagicMock(return_value=2)

    def compute_metrics(**kwargs):
        x = kwargs["inference_variables"]
        loss = ((param * x[:, 0] - 1.0) ** 2).mean()
        return {"loss": loss}

    approx.compute_metrics = compute_metrics

    micro_batches = [
        {
            "inference_variables": torch.tensor([[1.0], [2.0]]),
            "summary_variables": MagicMock(),
        },
        {
            "inference_variables": torch.tensor([[3.0], [4.0]]),
            "summary_variables": MagicMock(),
        },
        {
            "inference_variables": torch.tensor([[5.0], [6.0]]),
            "summary_variables": MagicMock(),
        },
    ]

    with patch.object(
        type(approx),
        "trainable_weights",
        new_callable=PropertyMock,
        return_value=[fake_weight],
    ):
        for batch in micro_batches:
            approx.train_step(batch)

    with torch.enable_grad():
        full_x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
        ref = torch.nn.Parameter(torch.tensor([0.0]))
        ref_loss = ((ref * full_x[:, 0] - 1.0) ** 2).mean()
        ref_loss.backward()

    assert "grad" in captured
    assert float(captured["grad"]) == pytest.approx(float(ref.grad), abs=1e-5)


def test_gradient_accumulation_clears_state_on_exception() -> None:
    """_accum_step resets and zero_grad is called when compute_metrics raises."""
    if keras.backend.backend() != "torch":
        pytest.skip("gradient accumulation uses torch-native backward")

    import torch

    with patch.object(ContinuousApproximator, "__init__", bare_model_init):
        approx = EquivariantIRTApproximator(
            adapter=MagicMock(),
            inference_network=MagicMock(),
            summary_network=MagicMock(),
            gradient_accumulation_steps=3,
        )

    param = torch.nn.Parameter(torch.zeros(1))
    fake_weight = MagicMock()
    fake_weight.value = param

    approx.optimizer = MagicMock()
    approx.zero_grad = MagicMock()
    approx._update_metrics = MagicMock()
    approx._batch_size_from_data = MagicMock(return_value=4)

    fake_data = {"inference_variables": MagicMock(), "summary_variables": MagicMock()}

    with patch.object(
        type(approx),
        "trainable_weights",
        new_callable=PropertyMock,
        return_value=[fake_weight],
    ):
        # Step 1 succeeds — uses a function so param.sum() is recomputed
        # inside the train_step's torch.enable_grad() context (giving it a
        # grad_fn).  A pre-computed tensor would have no grad_fn because
        # BayesFlow disables autograd globally by default.
        def make_loss(**kwargs):
            return {"loss": param.sum()}

        approx.compute_metrics = make_loss
        approx.train_step(fake_data)
        assert approx._accum_step == 1

        # Step 2 raises inside compute_metrics — state must be fully reset
        def raise_oom(**kwargs):
            raise RuntimeError("simulated OOM")

        approx.compute_metrics = raise_oom
        with pytest.raises(RuntimeError, match="simulated OOM"):
            approx.train_step(fake_data)

        assert approx._accum_step == 0, "step counter must reset on exception"
        approx.zero_grad.assert_called()


def test_epoch_end_flushes_partial_accumulation() -> None:
    """The flush discards unfinished micro-steps without applying gradients."""
    with patch.object(ContinuousApproximator, "__init__", bare_model_init):
        approx = EquivariantIRTApproximator(
            adapter=MagicMock(),
            inference_network=MagicMock(),
            summary_network=MagicMock(),
            gradient_accumulation_steps=3,
        )

    approx.optimizer = MagicMock()
    approx.zero_grad = MagicMock()
    # Simulate dirty state: 1 micro-step accumulated out of 3
    approx._accum_step = 1

    approx._flush_partial_accumulation(epoch=0)

    assert approx._accum_step == 0, "step counter must be reset"
    approx.zero_grad.assert_called()
    approx.optimizer.apply_gradients.assert_not_called()


def _build_tiny_2pl_approximator():
    """Return a minimal built-ready 2PL approximator with its simulator/adapter."""
    simulator = make_2pl_simulator(
        n_items=(3, 3),
        n_persons=(5, 5),
        prior_a=Prior("lognormal", loc=0.0, scale=0.5),
        prior_b=Prior("normal", loc=0.0, scale=2.0),
        prior_theta=Prior("normal", loc=0.0, scale=1.0),
    )
    adapter = make_2pl_adapter(simulator)
    summary = MatrixEquivariantIRTSummary(
        embed_dim=8,
        matrix_hidden_dim=8,
        num_matrix_blocks=1,
        num_outer_sabs=1,
        outer_embed_dim=8,
        num_outer_heads=2,
        outer_mlp_depth=1,
        outer_mlp_width=16,
        dropout=0.0,
    )
    flow = bf.networks.FlowMatching(
        subnet_kwargs={"widths": (16,), "time_embedding_dim": 8},
        integrate_kwargs={"method": "euler", "steps": 2},
    )
    approximator = EquivariantIRTApproximator(
        adapter=adapter,
        inference_network=flow,
        summary_network=summary,
    )
    return approximator, simulator, adapter


def test_matrix_summary_model_save_load_round_trip(tmp_path) -> None:
    """Matrix summary network should round-trip through Keras save/load."""
    summary = MatrixEquivariantIRTSummary(
        embed_dim=8,
        matrix_hidden_dim=8,
        num_matrix_blocks=1,
        num_outer_sabs=1,
        outer_embed_dim=8,
        num_outer_heads=2,
        outer_mlp_depth=1,
        outer_mlp_width=16,
        dropout=0.0,
    )
    inputs = keras.Input(shape=(6, 4), dtype="float32")
    outputs = summary(inputs, training=False)
    model = keras.Model(inputs=inputs, outputs=outputs)

    x = np.random.default_rng(7).integers(0, 2, size=(2, 6, 4)).astype(np.float32)
    y_before = keras.ops.convert_to_numpy(model(x, training=False))

    model_path = tmp_path / "matrix_summary_roundtrip.keras"
    model.save(model_path)
    restored = keras.models.load_model(
        model_path,
        custom_objects={"MatrixEquivariantIRTSummary": MatrixEquivariantIRTSummary},
    )
    y_after = keras.ops.convert_to_numpy(restored(x, training=False))

    assert y_before.shape == y_after.shape
    np.testing.assert_allclose(y_before, y_after, rtol=1e-6, atol=1e-6)


def test_equivariant_approximator_save_load_restores_sampling_state(tmp_path) -> None:
    """Subclassed approximator checkpoints should reload weights and flow dim."""
    approximator, simulator, adapter = _build_tiny_2pl_approximator()

    batch = adapter(simulator.sample((2,)))
    approximator.build({key: value.shape for key, value in batch.items()})
    weights_before = [
        np.array(value, copy=True) for value in approximator.get_weights()
    ]
    assert weights_before
    assert approximator.inference_network.base_distribution.dim == 2

    model_path = tmp_path / "equivariant_2pl.keras"
    approximator.save(model_path)
    restored = keras.saving.load_model(
        model_path,
        custom_objects={
            "EquivariantIRTApproximator": EquivariantIRTApproximator,
            "MatrixEquivariantIRTSummary": MatrixEquivariantIRTSummary,
        },
    )

    assert restored.inference_network.base_distribution.dim == 2
    weights_after = restored.get_weights()
    assert len(weights_after) == len(weights_before)
    for before, after in zip(weights_before, weights_after):
        np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-6)

    samples = restored.sample(
        num_samples=3,
        conditions=simulator.sample((1,)),
        flow_sample_kwargs={"method": "euler", "steps": 2},
    )
    assert samples["a"].shape == (1, 3, 3)
    assert samples["b"].shape == (1, 3, 3)


@pytest.mark.parametrize(
    "cls",
    [
        EquivariantIRTApproximator,
        MatrixEquivariantIRTSummary,
    ],
)
def test_public_classes_are_registered_keras_serializables(cls) -> None:
    """Importing the package must be enough for Keras to locate the classes."""
    registry = keras.saving.get_custom_objects()
    assert registry.get(f"bayesflow_irt>{cls.__name__}") is cls
    # Bare-name alias keeps pre-registration checkpoints loadable.
    assert registry.get(cls.__name__) is cls


def test_matrix_summary_loads_without_custom_objects(tmp_path) -> None:
    """The summary network should reload with no `custom_objects` argument."""
    summary = MatrixEquivariantIRTSummary(
        embed_dim=8,
        matrix_hidden_dim=8,
        num_matrix_blocks=1,
        num_outer_sabs=1,
        outer_embed_dim=8,
        num_outer_heads=2,
        outer_mlp_depth=1,
        outer_mlp_width=16,
        dropout=0.0,
    )
    inputs = keras.Input(shape=(6, 4), dtype="float32")
    model = keras.Model(inputs=inputs, outputs=summary(inputs, training=False))

    x = np.random.default_rng(7).integers(0, 2, size=(2, 6, 4)).astype(np.float32)
    y_before = keras.ops.convert_to_numpy(model(x, training=False))

    model_path = tmp_path / "matrix_summary_bare_load.keras"
    model.save(model_path)
    restored = keras.models.load_model(model_path)

    y_after = keras.ops.convert_to_numpy(restored(x, training=False))
    np.testing.assert_allclose(y_before, y_after, rtol=1e-6, atol=1e-6)


def test_equivariant_approximator_loads_without_custom_objects(tmp_path) -> None:
    """`keras.saving.load_model(path)` alone must restore the approximator.

    Covers both the checkpoint this version writes and the legacy form that
    predates registration, whose `registered_name` is the bare class name.
    """
    approximator, simulator, adapter = _build_tiny_2pl_approximator()
    batch = adapter(simulator.sample((2,)))
    approximator.build({key: value.shape for key, value in batch.items()})
    weights_before = [
        np.array(value, copy=True) for value in approximator.get_weights()
    ]
    assert weights_before

    model_path = tmp_path / "equivariant_2pl_bare_load.keras"
    approximator.save(model_path)

    restored = keras.saving.load_model(model_path)
    assert isinstance(restored, EquivariantIRTApproximator)
    assert restored.inference_network.base_distribution.dim == 2
    for before, after in zip(weights_before, restored.get_weights()):
        np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-6)

    legacy_path = tmp_path / "equivariant_2pl_legacy.keras"
    strip_registered_name_prefixes(model_path, legacy_path)
    legacy_restored = keras.saving.load_model(legacy_path)
    assert isinstance(legacy_restored, EquivariantIRTApproximator)
    for before, after in zip(weights_before, legacy_restored.get_weights()):
        np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-6)


def test_gradient_accumulation_resets_at_epoch_boundary_in_real_fit() -> None:
    """A partial accumulation window must not leak into the next epoch."""
    if keras.backend.backend() != "torch":
        pytest.skip("gradient accumulation uses torch-native backward")

    simulator = make_2pl_simulator(
        n_items=(4, 5),
        n_persons=(8, 10),
        prior_a=Prior("lognormal", loc=0, scale=0.5),
        prior_b=Prior("normal", loc=0, scale=1.0),
        prior_theta=Prior("normal", loc=0, scale=1.0),
    )
    approx = EquivariantIRTApproximator(
        adapter=make_2pl_adapter(simulator),
        inference_network=bf.networks.FlowMatching(subnet_kwargs={"widths": (16, 16)}),
        summary_network=MatrixEquivariantIRTSummary(
            embed_dim=8,
            matrix_hidden_dim=8,
            num_matrix_blocks=1,
            num_outer_sabs=1,
            outer_embed_dim=8,
            outer_mlp_width=16,
        ),
        gradient_accumulation_steps=3,
    )
    approx.compile(optimizer="adam")

    seen: list[int] = []

    class _Probe(keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            seen.append(approx._accum_step)

    # 5 batches per epoch with a window of 3: 2 micro-steps are left over.
    approx.fit(
        simulator=simulator,
        epochs=2,
        num_batches=5,
        batch_size=2,
        callbacks=[_Probe()],
        verbose=0,
    )

    assert seen == [0, 0]
    assert approx._accum_step == 0
    # One full window per epoch; leftovers are discarded, not carried over.
    assert int(approx.optimizer.iterations.numpy()) == 2


@pytest.mark.parametrize("chunk_size", [None, 4])
def test_compute_metrics_broadcasts_per_dataset_sample_weight(chunk_size) -> None:
    """Per-dataset weights (B,) are repeated over items in flatten order."""
    with patch.object(ContinuousApproximator, "__init__", bare_model_init):
        approx = EquivariantIRTApproximator(
            adapter=MagicMock(),
            inference_network=MagicMock(),
            summary_network=MagicMock(),
            flow_chunk_size=chunk_size,
        )

    seen: list[np.ndarray] = []

    class WeightedNet:
        def compute_metrics(
            self, params, conditions=None, sample_weight=None, stage="training"
        ):
            assert sample_weight is not None
            assert sample_weight.shape[0] == params.shape[0]
            w = keras.ops.convert_to_numpy(sample_weight)
            seen.append(w)
            p = keras.ops.convert_to_numpy(params)[:, 0]
            return {"loss": keras.ops.convert_to_tensor((w * p).mean())}

    approx.inference_network = WeightedNet()
    approx._compute_summary_outputs = lambda sv, stage: keras.ops.zeros(
        (2, 3, 4), dtype="float32"
    )

    # Rows (flatten order): dataset 0 items 0-2, then dataset 1 items 0-2.
    params = keras.ops.convert_to_tensor(
        np.arange(6, dtype=np.float32).reshape(2, 3, 1)
    )
    weights = keras.ops.convert_to_tensor(np.array([1.0, 3.0], dtype=np.float32))

    loss = approx.compute_metrics(
        inference_variables=params,
        summary_variables=keras.ops.zeros((2, 5, 3), dtype="float32"),
        sample_weight=weights,
        stage="validation",
    )["loss"]

    full = np.concatenate(seen)
    np.testing.assert_allclose(full, [1, 1, 1, 3, 3, 3])
    expected = (np.array([1, 1, 1, 3, 3, 3]) * np.arange(6)).mean()
    assert float(keras.ops.convert_to_numpy(loss)) == pytest.approx(expected)
