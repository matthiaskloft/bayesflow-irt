"""Slow regression gate for 2PL discrimination recovery quality."""

from __future__ import annotations

import os

import bayesflow as bf
import keras
import numpy as np
import pytest

from bayesflow_irt.adapters import make_2pl_adapter
from bayesflow_irt.approximators import EquivariantIRTApproximator
from bayesflow_irt.networks import MatrixEquivariantIRTSummary
from bayesflow_irt.priors import Prior
from bayesflow_irt.simulators import make_2pl_simulator
from bayesflow_irt.utils import PrecomputedDataset


def _seeded_2pl_simulator(seed: int):
    simulator = make_2pl_simulator(
        n_items=(20, 20),
        n_persons=(200, 200),
        prior_a=Prior("lognormal", loc=0.0, scale=0.4),
        prior_b=Prior("normal", loc=0.0, scale=2.0),
        prior_theta=Prior("normal", loc=0.0, scale=1.0),
    )
    simulator._rng = np.random.default_rng(seed)
    return simulator


def _clone_batches(batches: list[dict[str, np.ndarray]]) -> list[dict[str, np.ndarray]]:
    return [{k: np.array(v, copy=True) for k, v in batch.items()} for batch in batches]


def _pearson_flat(pred: np.ndarray, truth: np.ndarray) -> float:
    x = pred.reshape(-1).astype(np.float64)
    y = truth.reshape(-1).astype(np.float64)
    x_std = float(x.std())
    y_std = float(y.std())
    if x_std == 0.0 or y_std == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _std_rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    y = truth.reshape(-1).astype(np.float64)
    y_scale = float(y.std())
    if y_scale == 0.0:
        y_scale = 1.0
    err = (pred.reshape(-1).astype(np.float64) - y) / y_scale
    return float(np.sqrt(np.mean(err**2)))


def _fit_and_eval(
    summary_network,
    *,
    train_batches: list[dict[str, np.ndarray]],
    eval_raw: dict[str, np.ndarray],
) -> tuple[float, float]:
    adapter_sim = _seeded_2pl_simulator(777)
    adapter = make_2pl_adapter(adapter_sim)
    _ = adapter(adapter_sim.sample((1,)))  # initialize strict-mode adapter transforms
    approx = EquivariantIRTApproximator(
        adapter=adapter,
        inference_network=bf.networks.FlowMatching(subnet_kwargs={"widths": (64, 64)}),
        summary_network=summary_network,
    )
    approx.compile(optimizer=keras.optimizers.Adam(1e-3))
    approx.fit(
        dataset=PrecomputedDataset(_clone_batches(train_batches)),
        epochs=6,
        verbose=0,
    )

    samples = approx.sample(
        num_samples=120,
        conditions=eval_raw,
        condition_chunk_size=128,
    )
    pred_a = samples["a"].mean(axis=1)
    true_a = eval_raw["a"]
    return _pearson_flat(pred_a, true_a), _std_rmse(pred_a, true_a)


@pytest.mark.slow
@pytest.mark.skipif(
    os.getenv("RUN_SLOW_TESTS") != "1",
    reason="Set RUN_SLOW_TESTS=1 to run slow discrimination-recovery gates.",
)
def test_matrix_summary_recovers_2pl_discrimination():
    """Matrix summary should reach strong discrimination recovery on 2PL.

    Checks an absolute correlation floor.
    """
    keras.utils.set_random_seed(1337)

    train_sim = _seeded_2pl_simulator(123)
    train_adapter = make_2pl_adapter(train_sim)
    train_batches = [train_adapter(train_sim.sample((16,))) for _ in range(12)]

    eval_sim = _seeded_2pl_simulator(2026)
    eval_raw = eval_sim.sample((8,))

    matrix_corr, matrix_rmse = _fit_and_eval(
        MatrixEquivariantIRTSummary(
            embed_dim=24,
            matrix_hidden_dim=24,
            num_matrix_blocks=2,
            num_outer_sabs=1,
            outer_embed_dim=24,
            outer_mlp_width=64,
        ),
        train_batches=train_batches,
        eval_raw=eval_raw,
    )

    assert matrix_corr >= 0.20, (
        "2PL discrimination recovery gate failed: "
        f"matrix_corr={matrix_corr:.3f}, matrix_std_rmse={matrix_rmse:.3f}."
    )
