"""Explicit target-size adaptation followed by persisted-model inference."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import keras
import numpy as np

from .compat import weights_digest
from .envelope import check_data
from .metadata import MetadataTracker, _sanitize_for_json, _sha256_file, session_info
from .priors import ParameterRole, Prior, VariablePrior, sample_prior_predictive
from .scoring import PersonScore, PersonScorer
from .simulators import build_simulator, describe_size_sampling
from .utils import FitResult, fit_with_oom_retry


@dataclass
class AdaptationResult:
    """Saved artifact, provenance, draws, and reusable person scorer.

    Times are wall-clock seconds. Adaptation includes loading, simulation,
    training, persistence and reload; item inference and scoring are separate.
    """

    checkpoint: Path
    metadata_path: Path
    training: FitResult
    provenance: dict[str, Any]
    item_posteriors: dict[str, np.ndarray]
    scorer: PersonScorer | None
    scores: PersonScore | None
    adaptation_seconds: float
    item_inference_seconds: float
    scoring_seconds: float | None


def adapt_and_fit(
    checkpoint: str | Path,
    responses: np.ndarray,
    output_dir: str | Path,
    *,
    metadata: str | Path,
    epochs: int,
    num_batches: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    prior_overrides: Mapping[str, Prior | VariablePrior] | None = None,
    n_item_samples: int = 1000,
    fit_kwargs: dict[str, Any] | None = None,
    memory_kwargs: dict[str, Any] | None = None,
    item_conditions: dict[str, Any] | None = None,
    score_kwargs: dict[str, Any] | None = None,
) -> AdaptationResult:
    """Fine-tune pretrained weights on target-size simulations, save, then infer.

    ``metadata`` is the parent's metadata sidecar. Recorded priors (including
    theta) must be reconstructible. ``prior_overrides`` replaces named priors
    with existing ``Prior`` or ``VariablePrior`` objects, for example
    ``{"b": Prior("normal", loc=0, scale=1.5)}``. Unspecified priors are kept.
    Bare callables cannot be persisted and are not accepted as overrides.
    Adapter transforms and conditioning conventions stay fixed: variable-prior
    overrides must retain the adapter's existing sigma condition keys.
    The override governs fresh training simulations, not observed responses.
    A fresh Adam optimizer uses the explicit learning rate. ``fit_kwargs``
    supplies optional Keras fit controls; ``memory_kwargs`` supplies optional
    :func:`fit_with_oom_retry` controls. Ordinary inference never adapts.

    ``output_dir`` must not exist. It contains ``adapted.keras`` and
    ``adapted_metadata.json`` and is published only after both save and reload
    succeed. Failures propagate; no original-model inference fallback occurs.
    Pass ``score_kwargs={}`` to also score the observed persons, or use the
    returned ``scorer.score(new_responses)`` later. Scoring uses the recorded
    Normal theta prior. For other theta distributions, item inference remains
    supported but ``scorer`` is None and requesting scoring raises before
    training, since PersonScorer currently supports Normal theta priors only.
    The saved model can be loaded independently
    with ``keras.saving.load_model(path, compile=False)``.

    This workflow provides no accuracy/calibration guarantee: validate on fresh
    simulations at the target size before making scientific claims.
    """
    started = perf_counter()
    checkpoint = Path(checkpoint).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    for name, value in (
        ("epochs", epochs),
        ("num_batches", num_batches),
        ("batch_size", batch_size),
        ("n_item_samples", n_item_samples),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    parent = MetadataTracker.load(metadata).to_dict()
    digest = _sha256_file(checkpoint)
    binding = parent.get("provenance", {}).get("checkpoint_sha256")
    if binding is not None and binding != digest:
        raise ValueError("Parent metadata checkpoint digest does not match")
    spec = deepcopy(parent["model_spec"])
    if "theta" not in spec.get("priors", {}):
        raise ValueError("Parent metadata must record the theta prior")
    overrides = {}
    for key, prior in (prior_overrides or {}).items():
        if key not in spec["priors"]:
            raise ValueError(f"Cannot override unrecorded prior {key!r}")
        if not isinstance(prior, (Prior, VariablePrior)):
            raise TypeError("prior_overrides values must be Prior or VariablePrior")
        role = cast(ParameterRole, "delta" if key.startswith("delta") else key)
        if isinstance(prior, VariablePrior) and prior.role != role:
            raise ValueError(f"VariablePrior role must match {key!r}")
        # Reuse the prior registry's role, parameter and support validation.
        draws = sample_prior_predictive(prior, role, n_samples=32, seed=0)
        if not np.all(np.isfinite(draws)):
            raise ValueError(f"Prior override {key!r} produced non-finite samples")
        if isinstance(prior, Prior):
            overrides[key] = {"name": prior.name, **deepcopy(prior.kwargs)}
        else:
            overrides[key] = {
                "kind": "variable",
                "name": prior.name,
                "role": prior.role,
                "scale_range": list(prior.scale_range),
                "fixed_kwargs": deepcopy(prior.fixed_kwargs),
            }
    spec["priors"].update(overrides)
    theta = spec["priors"]["theta"]
    normal_theta = theta.get("name") == "normal" and "kind" not in theta
    if score_kwargs is not None and not normal_theta:
        raise ValueError("PersonScorer requires a fixed Normal theta prior")
    theta_prior = (theta.get("loc", 0.0), theta.get("scale", 1.0))
    responses = np.asarray(responses)
    if responses.ndim != 2:
        raise ValueError("responses must be a 2-D (persons, items) matrix")
    n_persons, n_items = responses.shape
    sizes = {"n_items": [n_items, n_items], "n_persons": [n_persons, n_persons]}
    # Validate categories and shape at the new size, without an OOD warning.
    target_spec = dict(spec)
    target_spec["size_ranges"] = sizes
    check_data(responses, metadata={"model_spec": target_spec})
    fit_options = dict(fit_kwargs or {})
    if {
        "epochs",
        "num_batches",
        "batch_size",
        "simulator",
        "x",
        "initial_epoch",
    } & fit_options.keys():
        raise ValueError("fit_kwargs cannot override explicit training inputs")
    memory_options = {"min_batch_size": 1, **(memory_kwargs or {})}
    if {
        "start_batch_size",
        "fit_kwargs",
        "approximator",
        "simulator",
    } & memory_options.keys():
        raise ValueError("memory_kwargs cannot override explicit training inputs")
    keras.utils.set_random_seed(seed)
    simulator = build_simulator(
        spec,
        n_items=(n_items, n_items),
        n_persons=(n_persons, n_persons),
        seed=seed,
        trained_at=parent.get("provenance", {}).get("timestamp"),
    )
    model = keras.saving.load_model(str(checkpoint), compile=False)
    sigma_keys = {
        key
        for key in PersonScorer._adapter_condition_keys(model.adapter)
        if key.startswith("sigma_")
    }
    simulated_sigma_keys = {
        key
        for role in ("a", "b", "delta")
        for key in getattr(getattr(simulator, f"prior_{role}", None), "meta_keys", [])
    }
    if sigma_keys != simulated_sigma_keys:
        raise ValueError("Prior overrides must preserve the adapter's sigma conditions")
    initial_weights = weights_digest(model)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate))
    training_start = perf_counter()
    training = fit_with_oom_retry(
        model,
        simulator,
        start_batch_size=batch_size,
        fit_kwargs={**fit_options, "epochs": epochs, "num_batches": num_batches},
        **memory_options,
    )
    history = training.history.history
    if len(history.get("loss", [])) != epochs or not all(
        np.all(np.isfinite(values)) for values in history.values()
    ):
        raise RuntimeError("Adaptation did not complete all epochs with finite metrics")
    training_seconds = perf_counter() - training_start
    tracker = MetadataTracker(
        model=spec["model"],
        variant=spec["variant"],
        version=spec["version"],
    )
    tracker.add_training_config(
        priors=spec["priors"],
        size_ranges={
            **sizes,
            **(
                {"n_categories": spec["n_categories"]} if "n_categories" in spec else {}
            ),
        },
        size_sampling=describe_size_sampling(simulator),
        seed=seed,
        internal_parameterization=spec.get("internal_parameterization"),
    )
    tracker.add_phase2_metrics(epochs=epochs)
    tracker.attach(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        stage = Path(temporary) / "artifact"
        stage.mkdir()
        saved = stage / "adapted.keras"
        model.save(str(saved))
        record = tracker.to_dict()
        record["model_spec"] = {
            **spec,
            **record["model_spec"],
        }
        record["architecture"] = model.get_config()
        record["adaptation"] = {
            "parent_checkpoint_sha256": digest,
            "parent_metadata": parent,
            "prior_overrides": overrides,
            "initial_weights_digest": initial_weights,
            "target_dimensions": {"n_persons": n_persons, "n_items": n_items},
            "seed": seed,
            "epochs": epochs,
            "num_batches": num_batches,
            "batch_size": batch_size,
            "physical_batch_size": training.batch_size,
            "learning_rate": learning_rate,
            "optimizer": "Adam",
            "fit_kwargs": fit_options,
            "memory_kwargs": memory_options,
            "attempts": training.attempts,
            "memory_config": asdict(training.config),
            "history": history,
            "training_seconds": training_seconds,
            "adaptation_seconds_before_reload": perf_counter() - started,
            "session": session_info(),
        }
        # Round-trip through the tracker to retain its checkpoint binding writer.
        sidecar = stage / "adapted_metadata.json"
        sidecar.write_text(json.dumps(_sanitize_for_json(record)), encoding="utf-8")
        tracker = MetadataTracker.load(sidecar)
        tracker.save(sidecar, checkpoint_path=saved)
        persisted = keras.saving.load_model(str(saved), compile=False)
        if weights_digest(persisted) != weights_digest(model):
            raise RuntimeError("Persisted adapted weights differ from trained weights")
        record = tracker.to_dict()
        record["adaptation"]["adaptation_seconds_before_publication"] = (
            perf_counter() - started
        )
        sidecar.write_text(json.dumps(_sanitize_for_json(record)), encoding="utf-8")
        tracker = MetadataTracker.load(sidecar)
        stage.rename(output)
    adaptation_seconds = perf_counter() - started
    drawer = PersonScorer(persisted)
    inference_start = perf_counter()
    posteriors = drawer.draw_item_posteriors(
        responses,
        n_item_samples=n_item_samples,
        item_conditions=item_conditions,
    )
    inference_seconds = perf_counter() - inference_start
    scorer = (
        PersonScorer(persisted, item_posteriors=posteriors, theta_prior=theta_prior)
        if normal_theta
        else None
    )
    scores = None
    scoring_seconds = None
    if score_kwargs is not None:
        assert scorer is not None
        scoring_start = perf_counter()
        scores = scorer.score(responses, **score_kwargs)
        scoring_seconds = perf_counter() - scoring_start
    return AdaptationResult(
        output / "adapted.keras",
        output / "adapted_metadata.json",
        training,
        tracker.to_dict(),
        posteriors,
        scorer,
        scores,
        adaptation_seconds,
        inference_seconds,
        scoring_seconds,
    )
