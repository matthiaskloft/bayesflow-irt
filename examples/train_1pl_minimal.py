"""Minimal, non-notebook, reproducible 1PL train/infer example.

Trains a small variable-size 1PL `EquivariantIRTApproximator` for a few
epochs, draws posteriors for one held-out batch, and reports a quick
correlation between true and recovered difficulty (b). This is intentionally
small in scope: no HPO, no memory probing,
no diagnostics plots, no frequentist cross-check -- just the shortest path
from "fresh checkout" to "a trained model producing posteriors", runnable
from a plain terminal.

Usage
-----
    export KERAS_BACKEND=torch  # must be set before bayesflow/keras import
    python examples/train_1pl_minimal.py
"""

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

from pathlib import Path

import bayesflow as bf
import keras
import numpy as np

from bayesflow_irt import (
    EquivariantIRTApproximator,
    MatrixEquivariantIRTSummary,
    Prior,
    fit_with_oom_retry,
    make_1pl_adapter,
    make_1pl_simulator,
)

N_ITEMS = (5, 50)
N_PERSONS = (10, 500)
BATCH_SIZE = 32
EPOCHS = 3
BATCHES_PER_EPOCH = 20


def main() -> None:
    prior_b = Prior("normal", loc=0.0, scale=2.0)
    prior_theta = Prior("normal", loc=0.0, scale=1.0)

    simulator = make_1pl_simulator(
        n_items=N_ITEMS,
        n_persons=N_PERSONS,
        prior_b=prior_b,
        prior_theta=prior_theta,
    )
    adapter = make_1pl_adapter(simulator)

    summary_net = MatrixEquivariantIRTSummary(
        embed_dim=16,
        matrix_hidden_dim=32,
        num_matrix_blocks=2,
        matrix_activation="mish",
        person_pooling="mean",
        num_outer_sabs=0,
        outer_embed_dim=32,
        num_outer_heads=4,
        outer_mlp_depth=2,
        outer_mlp_width=48,
        dropout=0.05,
        use_outer_isab=True,
        num_outer_inducing_points=32,
        gradient_checkpointing=True,
    )
    flow = bf.networks.FlowMatching()
    approximator = EquivariantIRTApproximator(
        adapter=adapter,
        inference_network=flow,
        summary_network=summary_net,
    )
    approximator.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))

    result = fit_with_oom_retry(
        approximator,
        simulator,
        start_batch_size=BATCH_SIZE,
        fit_kwargs=dict(
            epochs=EPOCHS,
            num_batches=BATCHES_PER_EPOCH,
            workers=0,
            use_multiprocessing=False,
            max_queue_size=2,
        ),
    )
    print(f"Trained: batch_size={result.batch_size}, attempts={result.attempts}")

    # Draw posteriors for one held-out simulated dataset and check recovery.
    test_batch = simulator.sample((1,))
    n_items = int(np.asarray(test_batch["n_items"]).reshape(-1)[0])
    posterior = approximator.sample(num_samples=500, conditions=test_batch)
    b_true = np.asarray(test_batch["b"]).reshape(n_items)
    b_post_mean = np.asarray(posterior["b"]).mean(axis=1).reshape(n_items)
    corr = np.corrcoef(b_true, b_post_mean)[0, 1]
    print(f"n_items={n_items}, posterior-mean vs. true b correlation: {corr:.3f}")

    cwd = Path.cwd().resolve()
    repo_root = cwd if (cwd / "pyproject.toml").exists() else cwd.parent
    out_path = repo_root / "models_trained" / "examples" / "1pl_minimal_example.keras"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    approximator.save(str(out_path))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
