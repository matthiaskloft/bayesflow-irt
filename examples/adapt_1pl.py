"""Small fine-tune → save → fit example, with synthetic observed responses.

Set KERAS_BACKEND=torch before running. Supply a pretrained 1PL .keras file
and its metadata sidecar. For example:

    python examples/adapt_1pl.py parent.keras parent_metadata.json adapted_run

The output directory must be new. Training uses fresh simulations at 30 persons
and 8 items; the synthetic observed matrix is used only for inference/scoring.
This is an API demonstration, not a calibration or large-scale feasibility study.
"""

import argparse

import keras
import numpy as np

from bayesflow_irt import PersonScorer, Prior, adapt_and_fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("metadata")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    rng = np.random.default_rng(12)
    theta = rng.normal(size=(30, 1))
    b = rng.normal(0, 2, size=(1, 8))
    responses = rng.binomial(1, 1 / (1 + np.exp(b - theta)))
    result = adapt_and_fit(
        args.checkpoint,
        responses,
        args.output_dir,
        metadata=args.metadata,
        epochs=1,
        num_batches=2,
        batch_size=2,
        learning_rate=1e-4,
        seed=13,
        prior_overrides={"b": Prior("normal", loc=0, scale=1.5)},
        n_item_samples=20,
        fit_kwargs={"workers": 0},
        score_kwargs={},
    )
    print(f"Saved: {result.checkpoint}")
    print(f"Adaptation: {result.adaptation_seconds:.3f}s")
    print(f"Item inference: {result.item_inference_seconds:.3f}s")
    print(f"Scoring: {result.scoring_seconds:.3f}s")
    # Independent reuse in another process requires no further adaptation.
    reloaded = keras.saving.load_model(result.checkpoint, compile=False)
    later_scorer = PersonScorer(
        reloaded,
        item_posteriors=result.item_posteriors,
        theta_prior=(result.scorer.theta_prior_mean, result.scorer.theta_prior_scale),
    )
    later_scorer.score(responses[:3])


if __name__ == "__main__":
    main()
