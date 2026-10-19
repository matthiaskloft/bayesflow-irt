"""Regression tests for chunked-loss, accumulation and parameterization fixes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import bayesflow as bf
import keras
import numpy as np
import pytest
from bayesflow.approximators.continuous_approximator import ContinuousApproximator

from bayesflow_irt.approximators import (
    EquivariantIRTApproximator,
    _collect_seed_generators,
)


def _bare_approx(**kwargs) -> EquivariantIRTApproximator:
    with patch.object(ContinuousApproximator, "__init__", return_value=None):
        return EquivariantIRTApproximator(
            adapter=MagicMock(),
            inference_network=MagicMock(),
            summary_network=MagicMock(),
            **kwargs,
        )


def _seed_states(net) -> list[np.ndarray]:
    generators = _collect_seed_generators(net)
    assert len(generators) >= 2  # flow and base distribution
    return [keras.ops.convert_to_numpy(g.state).copy() for g in generators]


def _set_seed_states(net, values) -> None:
    for generator, value in zip(_collect_seed_generators(net), values):
        generator.state.assign(value)


def _flow_grads(backward, chunk_size, n=7):
    import torch

    net = bf.networks.FlowMatching(subnet_kwargs={"widths": (8, 8)})
    torch.manual_seed(0)
    net.build((n, 2), (n, 3))
    return net


def _run(net, mode, seeds, weights=None):
    import torch

    approx = _bare_approx(flow_chunk_size=3, flow_chunk_backward=mode)
    object.__setattr__(approx, "inference_network", net)
    _set_seed_states(net, seeds)
    torch.manual_seed(123)
    rng = np.random.default_rng(1)
    # Keras' default device, so inputs match the flow's own draws on CUDA.
    params = keras.ops.convert_to_tensor(rng.normal(size=(8, 2)), dtype="float32")
    cond = keras.ops.convert_to_tensor(rng.normal(size=(8, 3)), dtype="float32")
    cond.requires_grad_()
    sw = (
        None
        if weights is None
        else keras.ops.convert_to_tensor(weights, dtype="float32")
    )
    for w in net.trainable_weights:
        w.value.grad = None
    with torch.enable_grad():
        loss = approx._chunked_inference_loss(
            params, cond, sample_weight=sw, stage="training"
        )
        loss.backward()
    grads = [w.value.grad.detach().cpu().clone().numpy() for w in net.trainable_weights]
    return float(loss), grads, _seed_states(net)


@pytest.mark.parametrize("weighted", [False, True])
def test_checkpoint_matches_plain_chunking_with_real_flow_matching(weighted) -> None:
    if keras.backend.backend() != "torch":
        pytest.skip("checkpoint chunking is torch-specific")

    net = _flow_grads(None, None)
    seeds = _seed_states(net)
    weights = np.linspace(0.2, 2.0, 8) if weighted else None

    loss_plain, g_plain, end_plain = _run(net, None, seeds, weights)
    loss_ckpt, g_ckpt, end_ckpt = _run(net, "checkpoint", seeds, weights)

    assert loss_ckpt == pytest.approx(loss_plain, rel=1e-5)
    for a, b in zip(g_plain, g_ckpt):
        np.testing.assert_allclose(b, a, rtol=1e-4, atol=1e-6)
    for a, b in zip(end_plain, end_ckpt):
        np.testing.assert_array_equal(b, a)


def test_checkpoint_weighted_chunks_bind_their_own_weights() -> None:
    """Recompute during backward must use each chunk's own sample weights."""
    if keras.backend.backend() != "torch":
        pytest.skip("checkpoint chunking is torch-specific")

    import torch

    class WeightedNet:
        def compute_metrics(
            self, params, conditions=None, sample_weight=None, stage="training"
        ):
            return {"loss": (params[:, 0] * conditions[:, 0] * sample_weight).sum()}

    def grad(mode):
        approx = _bare_approx(flow_chunk_size=3, flow_chunk_backward=mode)
        approx.inference_network = WeightedNet()
        base = torch.arange(1.0, 9.0).reshape(8, 1)
        w = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        scale = torch.ones(1, requires_grad=True)
        with torch.enable_grad():
            loss = approx._chunked_inference_loss(
                base, base * scale, sample_weight=w, stage="training"
            )
            loss.backward()
        return scale.grad.clone()

    torch.testing.assert_close(grad("checkpoint"), grad(None))


class _StopFitError(Exception):
    pass


def test_fit_resets_stale_accumulation_state() -> None:
    if keras.backend.backend() != "torch":
        pytest.skip("gradient accumulation uses torch-native backward")

    import torch

    approx = _bare_approx(gradient_accumulation_steps=3)
    param = torch.nn.Parameter(torch.zeros(1))
    param.grad = torch.ones(1)
    approx._accum_step = 2
    approx.zero_grad = MagicMock(side_effect=lambda: setattr(param, "grad", None))

    seen = {}

    def fake_super_fit(self, *args, **kwargs):
        seen["accum_at_start"] = self._accum_step
        seen["grad_at_start"] = param.grad
        self._accum_step = 1
        param.grad = torch.ones(1)
        raise _StopFitError

    with patch.object(ContinuousApproximator, "fit", fake_super_fit):
        with pytest.raises(_StopFitError):
            approx.fit()
        assert seen["accum_at_start"] == 0
        assert seen["grad_at_start"] is None
        # Interrupted fit leaves nothing behind either.
        assert approx._accum_step == 0
        assert param.grad is None


def test_loading_parameterization_survives_save_load(tmp_path) -> None:
    from bayesflow_irt.adapters import make_2pl_adapter
    from bayesflow_irt.networks import MatrixEquivariantIRTSummary
    from bayesflow_irt.priors import Prior
    from bayesflow_irt.simulators import make_2pl_simulator

    simulator = make_2pl_simulator(
        n_items=(3, 3),
        n_persons=(5, 5),
        prior_a=Prior("lognormal", loc=0.0, scale=0.5),
        prior_b=Prior("normal", loc=0.0, scale=2.0),
        prior_theta=Prior("normal", loc=0.0, scale=1.0),
    )
    adapter = make_2pl_adapter(simulator, internal_parameterization="loading")
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
    approximator = EquivariantIRTApproximator(
        adapter=adapter,
        inference_network=bf.networks.FlowMatching(
            subnet_kwargs={"widths": (16,), "time_embedding_dim": 8},
        ),
        summary_network=summary,
    )
    batch = adapter(simulator.sample((2,)))
    approximator.build({key: value.shape for key, value in batch.items()})

    path = tmp_path / "loading_2pl.keras"
    approximator.save(path)
    restored = keras.saving.load_model(path)

    assert restored.adapter._bayesflow_irt_internal_parameterization == "loading"
    samples = restored.sample(
        num_samples=3,
        conditions=simulator.sample((1,)),
        flow_sample_kwargs={"method": "euler", "steps": 2},
    )
    assert {"a", "b", "d"} <= set(samples)
    np.testing.assert_allclose(samples["d"], -samples["a"] * samples["b"], rtol=1e-5)


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("mode", [None, "checkpoint"])
def test_chunked_loss_matches_unchunked_with_bayesflow_reduction(
    weighted, mode
) -> None:
    """Chunk aggregation must reproduce BayesFlow's row-mean reduction."""
    if keras.backend.backend() != "torch":
        pytest.skip("checkpoint chunking is torch-specific")

    import torch
    from bayesflow.utils import weighted_mean

    class RowLossNet:
        # Deterministic per-row loss reduced exactly as BayesFlow's inference
        # networks do (mean of loss * sample_weight over rows).
        def compute_metrics(
            self, params, conditions=None, sample_weight=None, stage="training"
        ):
            per_row = (params[:, 0] - conditions[:, 0]) ** 2
            return {"loss": weighted_mean(per_row, sample_weight)}

    rng = np.random.default_rng(1)
    params = torch.tensor(rng.normal(size=(8, 2)), dtype=torch.float32)
    cond = torch.tensor(rng.normal(size=(8, 3)), dtype=torch.float32)
    sw = (
        torch.tensor(np.linspace(0.2, 3.0, 8), dtype=torch.float32)
        if weighted
        else None
    )

    def loss_for(chunk_size, backward):
        approx = _bare_approx(flow_chunk_size=chunk_size, flow_chunk_backward=backward)
        approx.inference_network = RowLossNet()
        with torch.enable_grad():
            return float(
                approx._chunked_inference_loss(
                    params,
                    cond.clone().requires_grad_(),
                    sample_weight=sw,
                    stage="training",
                ).detach()
            )

    # 8 rows in chunks of 3 -> the final chunk has 2 rows.
    assert loss_for(3, mode) == pytest.approx(loss_for(None, None), rel=1e-5)
