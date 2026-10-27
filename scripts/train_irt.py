"""Config-driven, non-interactive IRT training entry point.

It runs the simulator -> adapter -> architecture -> memory-probe ->
``fit_with_oom_retry`` -> save-with-metadata pipeline, but reads every choice
from a JSON config and writes only files (no plots, no interactive state), so
it can run unattended without a live connection.

Diagnostics and prior-predictive checks are left to interactive work; this
script only produces the checkpoint + metadata sidecar.

Usage
-----
    export KERAS_BACKEND=torch
    python scripts/train_irt.py --config scripts/configs/2pl_fixed.json

Notable flags
-------------
``--resume-weights PATH``
    Load weights before fitting -- for restarting an interrupted run.
    Combine with ``--epochs`` to train only the remaining epochs.
``--no-probe``
    Skip :func:`probe_memory_config` and trust ``training.person_chunk_size``
    from the config. Worth using once a GPU's safe chunk size is known, since
    the probe adds startup time on every run.
``--dry-run``
    Build everything, run the probe if enabled, then exit before fitting.
    Use this once on a fresh machine to validate a config cheaply.

Every run writes, under ``--out-dir``:

``<tag>.keras``                     final checkpoint
``<tag>_metadata.json``             :class:`MetadataTracker` sidecar (UUID-linked)
``<tag>_history.csv``               per-epoch loss/val_loss (Keras ``CSVLogger``)
``<tag>_run.json``                  resolved config + environment provenance
``checkpoints/<tag>.epoch*.weights.h5``  per-epoch weights, for interruption recovery
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Keras must see the backend before keras/bayesflow are imported anywhere.
os.environ.setdefault("KERAS_BACKEND", "torch")
# Variable-size batches make the CUDA caching allocator ratchet up reserved
# memory (see utils._PeriodicCudaCacheRelease). expandable_segments is the
# allocator's own fix for this pattern; it is a documented no-op on Windows
# CUDA (use max_split_size_mb there) but does work on Linux.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import bayesflow as bf  # noqa: E402
import keras  # noqa: E402
import torch  # noqa: E402

import bayesflow_irt  # noqa: E402
from bayesflow_irt import (  # noqa: E402
    ChunkSearchOptions,
    EquivariantIRTApproximator,
    MatrixEquivariantIRTSummary,
    MetadataTracker,
    build_simulator,
    fit_with_oom_retry,
    make_1pl_adapter,
    make_2pl_adapter,
    make_gpcm_adapter,
    make_pcm_adapter,
    probe_memory_config,
)
from bayesflow_irt.metadata import _sanitize_for_json  # noqa: E402
from bayesflow_irt.simulators import describe_size_sampling  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# Learning-rate schedule
# --------------------------------------------------------------------------
# Defined at a stable, importable module path. A checkpoint serializes its
# LR schedule by registered name, so a notebook-local definition cannot be
# deserialized from a fresh kernel without ``compile=False``; keeping it here
# lets ``keras.saving.load_model`` restore a compiled, further-trainable model
# via ``custom_objects={"WarmupCosine": WarmupCosine}``.
@keras.saving.register_keras_serializable(package="bayesflow_irt_scripts")
class WarmupCosine(keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup over ``warmup_steps``, then cosine decay."""

    def __init__(self, warmup_steps: int, base_lr: float, cosine: Any) -> None:
        super().__init__()
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.cosine = cosine

    def __call__(self, step):
        step = keras.ops.cast(step, "float32")
        warmup_steps = keras.ops.cast(self.warmup_steps, "float32")
        warmup_lr = self.base_lr * step / keras.ops.maximum(warmup_steps, 1.0)
        return keras.ops.where(
            step < warmup_steps, warmup_lr, self.cosine(step - warmup_steps)
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "warmup_steps": self.warmup_steps,
            "base_lr": self.base_lr,
            "cosine": keras.saving.serialize_keras_object(self.cosine),
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> WarmupCosine:
        config = dict(config)
        config["cosine"] = keras.saving.deserialize_keras_object(config["cosine"])
        return cls(**config)


# --------------------------------------------------------------------------
# Config handling
# --------------------------------------------------------------------------
def _has_numba() -> bool:
    import importlib.util

    return importlib.util.find_spec("numba") is not None


def build_simulator_factory(cfg: dict[str, Any]):
    """Return ``(make_simulator, make_adapter, adapter_kwargs)`` for *cfg*.

    ``make_simulator`` takes only ``n_items``/``n_persons`` -- the signature
    :func:`probe_memory_config` requires -- with priors, categories, numba, and
    size buckets closed over from *cfg*. The simulator itself comes from
    :func:`bayesflow_irt.build_simulator`, the builder model-card evaluation
    uses, so both simulate from the same code. Only the 2PL simulator exposes
    ``item_bucket``/``person_bucket``/``size_distribution``, so those keys
    apply to that family alone.
    """
    model = cfg["model"].lower()
    if model not in _ADAPTERS:
        raise ValueError(f"Unknown model {cfg['model']!r}; expected 1pl/2pl/pcm/gpcm.")
    use_numba = cfg.get("use_numba", _has_numba())
    seed = cfg.get("seed")
    size_sampling: dict[str, Any] = {
        "size_distribution": cfg.get("size_distribution", "uniform"),
        "n_items": {"bucket": cfg.get("item_bucket")},
        "n_persons": {"bucket": cfg.get("person_bucket")},
    }
    if "log_uniform_share" in cfg:
        size_sampling["log_uniform_share"] = cfg["log_uniform_share"]
    model_spec = {
        "model": model,
        "priors": cfg["priors"],
        "n_categories": cfg.get("n_categories"),
        "size_sampling": size_sampling if model == "2pl" else None,
    }

    def factory(*, n_items, n_persons, seed=seed):
        return build_simulator(
            model_spec,
            n_items=n_items,
            n_persons=n_persons,
            seed=seed,
            use_numba=use_numba,
            apply_size_sampling=True,
        )

    adapter_kwargs: dict[str, Any] = {}
    if model in ("2pl", "gpcm"):
        adapter_kwargs["internal_parameterization"] = cfg.get(
            "internal_parameterization", "loading"
        )
    return factory, _ADAPTERS[model], adapter_kwargs


_ADAPTERS = {
    "1pl": make_1pl_adapter,
    "2pl": make_2pl_adapter,
    "pcm": make_pcm_adapter,
    "gpcm": make_gpcm_adapter,
}


def build_approximator(cfg: dict[str, Any], adapter) -> EquivariantIRTApproximator:
    """Build the summary net, per-item FlowMatching flow, and approximator."""
    s = cfg["summary"]
    summary_net = MatrixEquivariantIRTSummary(
        embed_dim=s["embed_dim"],
        matrix_hidden_dim=s["matrix_hidden_dim"],
        num_matrix_blocks=s["num_matrix_blocks"],
        matrix_activation=s.get("matrix_activation", "mish"),
        person_pooling=s.get("person_pooling", "mean"),
        num_outer_sabs=s.get("num_outer_sabs", 0),
        dropout=s.get("dropout", 0.05),
        gradient_checkpointing=s.get("gradient_checkpointing", True),
        # Defaults to () -- the summary net's own default -- so configs without
        # the key build the default network. Without this passthrough the
        # script cannot express the late-fused raw channels at all, and silently
        # builds a different architecture than a config asking for them intends,
        # e.g. one with ("item_rest_correlation",).
        raw_feature_channels=tuple(s.get("raw_feature_channels", ())),
    )

    f = cfg["flow"]
    # euler/16 is this script's deliberate default, and it is NOT BayesFlow's:
    # `FlowMatching.INTEGRATE_DEFAULT_CONFIG` is {"method": "tsit5", "steps":
    # "adaptive"}. A config that omits `flow.integrate_kwargs` therefore builds
    # a model that samples differently from one built without the key, which
    # inherits the adaptive solver.
    #
    # The divergence is intended here -- fixed-step Euler is much faster to
    # sample.
    # But it is load-bearing for provenance: `integrate_kwargs` is serialized
    # into the checkpoint's `get_config()`, so a checkpoint built without it
    # carries tsit5/adaptive and is NOT reproduced bit-for-bit by
    # rebuilding from a config that leaves this at the default. Before
    # comparing posterior draws across checkpoints, check which path produced
    # each one.
    flow = bf.networks.FlowMatching(
        subnet_kwargs={"widths": tuple(f.get("widths", (64, 64)))},
        integrate_kwargs=f.get("integrate_kwargs", {"method": "euler", "steps": 16}),
    )

    a = cfg.get("approximator", {})
    return EquivariantIRTApproximator(
        adapter=adapter,
        inference_network=flow,
        summary_network=summary_net,
        flow_chunk_size=a.get("flow_chunk_size", 1024),
        flow_chunk_backward=a.get("flow_chunk_backward", "checkpoint"),
    )


def compile_approximator(
    approximator: EquivariantIRTApproximator,
    cfg: dict[str, Any],
    total_steps: int,
) -> dict[str, Any]:
    """Compile with Adam + warmup-cosine; return the schedule for metadata."""
    t = cfg["training"]
    base_lr = float(t.get("base_lr", 1e-3))
    min_lr = float(t.get("min_lr", 1e-4))
    warmup_steps = int(float(t.get("warmup_frac", 0.05)) * total_steps)

    cosine = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=base_lr,
        decay_steps=max(1, total_steps - warmup_steps),
        alpha=min_lr / base_lr,
    )
    approximator.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=WarmupCosine(warmup_steps, base_lr, cosine)
        )
    )
    return {
        "optimizer": "Adam",
        "base_lr": base_lr,
        "min_lr": min_lr,
        "warmup_steps": warmup_steps,
        "total_steps": total_steps,
    }


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------
def environment_provenance() -> dict[str, Any]:
    """Record what hardware and software produced a checkpoint.

    The device can change between runs, and the memory-related
    settings this pipeline tunes (batch size, person chunk size) are only
    interpretable against the card they were tuned on -- so the device name
    and VRAM are recorded alongside the usual version pins.
    """
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=REPO_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        commit = None

    device: dict[str, Any] = {"cuda_available": bool(torch.cuda.is_available())}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        device.update(
            name=props.name,
            total_memory_gb=round(props.total_memory / 1e9, 2),
            capability=f"{props.major}.{props.minor}",
            device_count=torch.cuda.device_count(),
        )
    return {
        "git_commit": commit,
        # `MetadataTracker` requires both git_commit and timestamp whenever a
        # provenance block is present (metadata._REQUIRED_PROVENANCE_KEYS), and
        # a sidecar written from `to_dict()` rather than `save()` does not get
        # the tracker's own auto-filled block. Without this, every sidecar this
        # function feeds fails the repo's own validate() -- and a run record
        # with no wall-clock time cannot be placed against a log after the
        # fact.
        "timestamp": datetime.now(UTC).isoformat(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "keras": keras.__version__,
        "keras_backend": keras.backend.backend(),
        "bayesflow": bf.__version__,
        "bayesflow_irt": bayesflow_irt.__version__,
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "device": device,
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an IRT approximator from a JSON config (batch mode)."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: config.out_dir or models_trained).",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Artifact basename (default: config.tag or <model>_<variant>).",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs.")
    parser.add_argument(
        "--resume-weights",
        type=Path,
        default=None,
        help="Load these weights before fitting (resume an interrupted run).",
    )
    parser.add_argument(
        "--no-probe", action="store_true", help="Skip the memory probe."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build (and probe), then exit without training.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    tag = args.tag or cfg.get("tag") or f"{cfg['model'].lower()}_{cfg['variant']}"
    out_dir = Path(args.out_dir or cfg.get("out_dir", REPO_ROOT / "models_trained"))
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    t = cfg["training"]
    if args.epochs is not None:
        t["epochs"] = args.epochs
    batch_size = int(t["batch_size"])
    epochs = int(t["epochs"])
    batches_per_epoch = int(t["batches_per_epoch"])
    val_batches = int(t.get("val_batches", 50))
    n_items = tuple(cfg["n_items"])
    n_persons = tuple(cfg["n_persons"])

    provenance = environment_provenance()
    print(json.dumps({"run": tag, "environment": provenance}, indent=2), flush=True)
    if not provenance["device"]["cuda_available"]:
        print(
            "WARNING: CUDA is not available -- this run will use the CPU. On a "
            "GPU machine that almost always means the wrong interpreter is "
            "active or a CPU-only torch wheel is installed.",
            flush=True,
        )

    # The config's seed also drives weight initialisation and the flow and
    # dropout draws, not only the simulators.
    if cfg.get("seed") is not None:
        keras.utils.set_random_seed(int(cfg["seed"]))

    make_simulator, make_adapter, adapter_kwargs = build_simulator_factory(cfg)
    simulator = make_simulator(n_items=n_items, n_persons=n_persons)
    adapter = make_adapter(simulator, **adapter_kwargs)
    approximator = build_approximator(cfg, adapter)
    schedule_info = compile_approximator(approximator, cfg, epochs * batches_per_epoch)

    # --- Memory probe -----------------------------------------------------
    probe_cfg = cfg.get("probe", {})
    person_chunk_size = t.get("person_chunk_size")
    probe_best: dict[str, Any] | None = None
    if not args.no_probe and probe_cfg.get("enabled", True):
        report = probe_memory_config(
            approximator,
            make_simulator,
            batch_size=batch_size,
            n_items=n_items,
            n_persons=n_persons,
            # MatrixEquivariantIRTSummary chunks the person axis, not the item
            # axis, so items stay dense ([None]); the person axis uses the
            # probe's own boundary search rather than a guessed candidate list,
            # which is what makes one config portable across GPU sizes.
            search=ChunkSearchOptions(
                candidate_item_chunk_sizes=[None],
                checkpoint_options=[
                    cfg["summary"].get("gradient_checkpointing", True),
                    False,
                ],
            ),
            search_tolerance=probe_cfg.get("search_tolerance", 8),
            safety_margin=probe_cfg.get("safety_margin", 0.9),
            verbose=True,
        )
        if report.best is not None:
            probed = getattr(report.best, "person_chunk_size", None)
            probe_best = {
                "person_chunk_size": probed,
                "batch_size": getattr(report.best, "batch_size", None),
            }
            if probed is not None:
                # The probe validates against a 1-2 step trial at the margin,
                # which can still be OOM-prone in full variable-shape
                # training; starting below it skips that fragile top rung.
                factor = float(probe_cfg.get("person_chunk_safety_factor", 0.8))
                person_chunk_size = max(8, int(probed * factor))
                print(
                    f"Probe best person_chunk_size={probed}; seeding fit with "
                    f"{person_chunk_size} ({factor:.0%} of it).",
                    flush=True,
                )

    if args.dry_run:
        print("--dry-run: built and probed successfully; exiting before fit.")
        return 0

    if args.resume_weights is not None:
        # Weights can only be loaded into a built model. The memory probe
        # builds it as a side effect of running real train steps, but with
        # --no-probe nothing has touched the networks yet, so build explicitly
        # from one adapted batch first.
        if not approximator.built:
            approximator.build_from_data(approximator.adapter(simulator.sample((2,))))
        approximator.load_weights(str(args.resume_weights))
        print(f"Resumed weights from {args.resume_weights}", flush=True)

    # --- Validation data --------------------------------------------------
    # A seeded config would otherwise hand the validation simulator the same
    # seed as the training simulator, so val_loss would be measured on a replay
    # of the training stream rather than on independent draws. Offset the seed
    # to keep the two streams independent; an unseeded config is already
    # independent and stays that way.
    cfg_seed = cfg.get("seed")
    val_seed = None if cfg_seed is None else int(cfg_seed) + 1
    val_simulator = make_simulator(n_items=n_items, n_persons=n_persons, seed=val_seed)
    val_dataset = approximator.build_dataset(
        simulator=val_simulator,
        adapter=approximator.adapter,
        batch_size=batch_size,
        num_batches=val_batches,
        workers=0,
        use_multiprocessing=False,
        max_queue_size=2,
    )

    # --- Fit --------------------------------------------------------------
    # Per-epoch weight snapshots plus a CSV history are what make an unattended
    # run recoverable: a machine that dies at epoch 60 of 100 restarts with
    # --resume-weights and --epochs 40 rather than from scratch.
    ckpt_pattern = str(out_dir / "checkpoints" / f"{tag}.epoch{{epoch:03d}}.weights.h5")
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=ckpt_pattern, save_weights_only=True, save_freq="epoch"
        ),
        keras.callbacks.CSVLogger(str(out_dir / f"{tag}_history.csv"), append=True),
        keras.callbacks.TerminateOnNaN(),
    ]

    started = time.time()
    result = fit_with_oom_retry(
        approximator,
        simulator,
        start_batch_size=batch_size,
        start_person_chunk_size=person_chunk_size,
        fit_kwargs=dict(
            epochs=epochs,
            num_batches=batches_per_epoch,
            workers=0,
            use_multiprocessing=False,
            max_queue_size=2,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=2,  # one line per epoch: readable in a log file
        ),
    )
    elapsed = time.time() - started
    history = getattr(result.history, "history", {}) or {}
    print(
        f"Training finished in {elapsed / 3600:.2f} h: "
        f"batch_size={result.batch_size}, "
        f"person_chunk_size={result.person_chunk_size}, "
        f"attempts={result.attempts}",
        flush=True,
    )

    # --- Save checkpoint + metadata --------------------------------------
    model_path = out_dir / f"{tag}.keras"
    metadata_path = out_dir / f"{tag}_metadata.json"
    approximator.save(str(model_path))

    tracker = MetadataTracker(
        model=cfg["model"].upper(),
        variant=cfg["variant"],
        version=bayesflow_irt.__version__,
    )
    tracker.add_training_config(
        priors={k: dict(v) for k, v in cfg["priors"].items()},
        size_ranges={
            "n_items": list(n_items),
            "n_persons": list(n_persons),
            **({"n_categories": cfg["n_categories"]} if "n_categories" in cfg else {}),
        },
        size_sampling=describe_size_sampling(simulator),
        hparams={
            "epochs": epochs,
            "batches_per_epoch": batches_per_epoch,
            "batch_size_requested": batch_size,
            "batch_size_effective": result.batch_size,
            "person_chunk_size_effective": result.person_chunk_size,
            "oom_retry_attempts": result.attempts,
            "val_batches": val_batches,
            "summary": cfg["summary"],
            "flow": cfg["flow"],
            "wall_clock_hours": round(elapsed / 3600, 3),
            "device": provenance["device"].get("name"),
            **schedule_info,
        },
        seed=cfg.get("seed"),
        internal_parameterization=(
            cfg.get("internal_parameterization", "loading")
            if cfg["model"].lower() in ("2pl", "gpcm")
            else None
        ),
    )
    loss = history.get("loss", [])
    val_loss = history.get("val_loss", [])
    tracker.add_phase2_metrics(
        epochs=epochs,
        per_epoch_metrics=[
            {
                "epoch": i + 1,
                "loss": float(loss[i]),
                "val_loss": (
                    float(val_loss[i])
                    if i < len(val_loss) and val_loss[i] is not None
                    else None
                ),
            }
            for i in range(len(loss))
        ],
    )
    tracker.attach(approximator)
    tracker.save(metadata_path, checkpoint_path=model_path)
    if not MetadataTracker.verify_file(approximator, metadata_path):
        raise RuntimeError("Metadata UUID mismatch between model object and sidecar.")

    (out_dir / f"{tag}_run.json").write_text(
        json.dumps(
            _sanitize_for_json(
                {
                    "tag": tag,
                    "config": cfg,
                    "environment": provenance,
                    "probe_best": probe_best,
                    "effective": {
                        "batch_size": result.batch_size,
                        "person_chunk_size": result.person_chunk_size,
                        "attempts": result.attempts,
                        "wall_clock_hours": round(elapsed / 3600, 3),
                    },
                }
            ),
            indent=2,
            default=str,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    print(f"Saved model:    {model_path}")
    print(f"Saved metadata: {metadata_path}")
    print("Metadata validation errors:", tracker.validate())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
