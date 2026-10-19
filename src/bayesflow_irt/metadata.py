"""Metadata tracking for IRT model training pipelines.

``MetadataTracker`` incrementally accumulates phase-specific data (HPO,
main training, calibration refinement, robustness) and persists it as a
sidecar JSON file alongside Keras checkpoints.  A UUID key-lock binds the
JSON to a specific model instance so mismatches can be detected.

Sidecars carry a ``schema_version``.  Version 2 (this module's output) has the
typed sections ``model_spec``, ``architecture``, ``training`` and
``provenance``.  Files without ``schema_version`` are version 1 and are
converted in memory by :func:`upgrade_v1_to_v2` when loaded.
Old files are never rewritten.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 2

_REQUIRED_MODEL_SPEC_KEYS = {"model", "variant", "version"}
_REQUIRED_PROVENANCE_KEYS = {"git_commit", "timestamp"}

# v1 ``training_config.hparams`` entries and where they go in v2.
_ARCHITECTURE_KEYS = ("summary", "flow")
_REFINEMENT_KEYS = (
    "base_checkpoint",
    "refine_epochs",
    "refine_batches_per_epoch",
    "effective_batch_size",
    "hpo_trial_params",
)
# Keys a v1 training sidecar may hold; the shim writes ``None`` for the ones a
# file lacks so a reader can tell "not recorded" from "absent from the schema".
_TRAINING_KEYS = (
    "epochs",
    "batches_per_epoch",
    "batch_size_requested",
    "batch_size_effective",
    "person_chunk_size_effective",
    "oom_retry_attempts",
    "val_batches",
    "optimizer",
    "base_lr",
    "min_lr",
    "warmup_steps",
    "total_steps",
    "wall_clock_hours",
    "device",
    "seed",
)
_MODEL_SPEC_KEYS = (
    "model",
    "variant",
    "version",
    "n_categories",
    "internal_parameterization",
    "priors",
    "size_ranges",
    "size_sampling",
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert numpy types to JSON-serializable Python types.

    Non-finite floats become ``None``. ``json.dump`` would otherwise write
    them as bare ``Infinity`` / ``-Infinity`` / ``NaN`` literals, which
    Python reads back but which are not JSON -- RFC 8259 has no token for
        them, and strict parsers (e.g. R's ``jsonlite``) refuse the file.

    This is not hypothetical: ``add_hpo_results(pareto_front=...)`` stores
    trial records verbatim and those carry ``log_gamma``, which reaches
    ``-inf`` whenever the rank ECDF discrepancy underflows to zero.

    ``None`` rather than a string: it reads back as ``NA`` in R and ``None``
    in Python, which is what the value means for analysis, and the key stays
    present so "underflowed" remains distinguishable from "not computed".

    Both a ``np.floating`` branch and a plain ``float`` branch are needed.
    ``np.float64`` is a ``float`` subclass but ``np.float32`` is not, so a
    check written against either one alone misses half the cases.

    Parameters
    ----------
    obj : Any
        Object to sanitize — typically a dict/list/value tree containing
        numpy scalars or arrays.

    Returns
    -------
    Any
        A JSON-serializable copy of *obj*.
    """
    import numpy as np

    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.ndarray):
        return _sanitize_for_json(obj.tolist())
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _get_provenance() -> dict[str, str]:
    """Gather provenance data: git commit, timestamp, package versions."""
    # Get git commit
    git_commit = "unknown"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_commit = result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    def get_version(name: str) -> str:
        try:
            return version(name)
        except PackageNotFoundError:
            return "unknown"

    return {
        "git_commit": git_commit,
        "timestamp": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        "python_version": platform.python_version(),
        "keras_version": get_version("keras"),
        "bayesflow_version": get_version("bayesflow"),
        "bayesflow_irt_version": get_version("bayesflow-irt"),
    }


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=5
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def session_info() -> dict[str, Any]:
    """Describe the software and hardware a computation runs on.

    In the spirit of R's ``sessionInfo()``: recorded next to results such as
    a model card's ``evaluation.json`` so a reader can tell what produced
    them. Git is queried at this package's own source directory, so the
    commit is that of the code actually imported (a worktree reports its own
    branch). The git fields are ``None`` unless that directory is a checkout
    tracking this module, so an installed copy inside another repository
    (for example a ``.venv`` in a project) does not report that repository.
    ``git_dirty`` counts changes to tracked files only, not untracked ones.

    Package versions come from :mod:`importlib.metadata` and are ``None``
    when a package is not installed. The GPU block is filled only when the
    Keras backend is ``torch`` and CUDA is available.

    Returns
    -------
    dict
        ``timestamp`` (UTC, ISO 8601), ``git_commit``, ``git_dirty``,
        ``python``, ``platform``, ``keras_backend``, ``packages`` (name to
        version) and ``gpu`` (``None`` or name, VRAM in GB, compute
        capability, device count). The hostname is deliberately not
        recorded: the result is meant to be published with a model card.
    """
    src = Path(__file__).resolve().parent
    # Only a checkout that tracks this very file describes the running code:
    # an installed copy inside some other repo (a .venv in a project) would
    # otherwise report that repo's commit.
    tracked = _git(["ls-files", "--error-unmatch", Path(__file__).name], src)
    commit = _git(["rev-parse", "HEAD"], src) if tracked is not None else None
    status = (
        _git(["status", "--porcelain", "--untracked-files=no"], src) if commit else None
    )

    def get_version(name: str) -> str | None:
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    packages = {
        name: get_version(name)
        for name in (
            "bayesflow-irt",
            "bayesflow",
            "keras",
            "numpy",
            "scipy",
            "torch",
            "jax",
            "tensorflow",
            "numba",
        )
    }
    import keras

    backend = keras.backend.backend()
    gpu = None
    if backend == "torch":
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu = {
                "name": props.name,
                "total_memory_gb": round(props.total_memory / 1e9, 2),
                "capability": f"{props.major}.{props.minor}",
                "device_count": torch.cuda.device_count(),
                "cuda": torch.version.cuda,
            }
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "keras_backend": backend,
        "packages": {k: v for k, v in packages.items() if v is not None},
        "gpu": gpu,
    }


def _sha256_file(path: str | Path) -> str:
    """Return the hex SHA-256 of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_hparams(
    hparams: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Any]:
    """Split a flat ``hparams`` dict into the v2 sections.

    Returns ``(architecture, training, refinement, n_categories)``. Keys with
    no designated home stay in ``training`` rather than being dropped.
    """
    architecture = {k: hparams[k] for k in _ARCHITECTURE_KEYS if k in hparams}
    refinement = {k: hparams[k] for k in _REFINEMENT_KEYS if k in hparams}
    n_categories = hparams.get("n_categories")
    consumed = {*_ARCHITECTURE_KEYS, *_REFINEMENT_KEYS, "n_categories"}
    training = {k: v for k, v in hparams.items() if k not in consumed}
    return architecture, training, refinement, n_categories


def upgrade_v1_to_v2(
    data: dict[str, Any], *, run_config: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Convert a version-1 sidecar dict to the version-2 layout, in memory.

    A pure function: *data* is not modified and nothing is written to disk.
    Values v1 never held (for example the seed or ``size_sampling``) become
    explicit ``None``, which a reader should show as "not recorded".

    Parameters
    ----------
    data : dict
        A parsed v1 sidecar (no ``schema_version`` key).
    run_config : dict, optional
        The ``config`` block of the sibling ``<tag>_run.json``. v1 sidecars
        carry neither the seed nor the internal parameterization; this is the
        only place either is recorded.

    Returns
    -------
    dict
        The same information in the v2 layout.

    Raises
    ------
    KeyError
        If *data* has no ``model_info``.
    """
    if "model_info" not in data:
        raise KeyError("Missing required key 'model_info' in metadata file.")
    rest = copy.deepcopy(data)
    run_config = run_config or {}
    info = rest.pop("model_info")
    config = rest.pop("training_config", None) or {}
    phase2 = rest.pop("phase2_metrics", None) or {}
    provenance = rest.pop("provenance", None)

    sizes = copy.deepcopy(config.get("size_ranges"))
    n_categories = None
    if sizes is not None:
        n_categories = sizes.pop("n_categories", None)
    architecture_part, training_part, refinement, hp_categories = _split_hparams(
        config.get("hparams") or {}
    )
    if n_categories is None:
        n_categories = hp_categories

    model_spec: dict[str, Any] = dict.fromkeys(_MODEL_SPEC_KEYS)
    model_spec.update(
        model=info.get("model"),
        variant=info.get("variant"),
        version=info.get("version"),
        n_categories=n_categories,
        internal_parameterization=run_config.get("internal_parameterization"),
        priors=config.get("priors"),
        size_ranges=sizes,
        size_sampling=config.get("size_sampling"),
    )

    training: dict[str, Any] = dict.fromkeys(_TRAINING_KEYS)
    training.update(training_part)
    if training["seed"] is None:
        training["seed"] = run_config.get("seed")
    final = {
        name: phase2[key]
        for name, key in (
            ("sbc_ecdf", "final_sbc_ecdf"),
            ("coverage", "final_coverage"),
            ("nrmse", "final_nrmse"),
        )
        if key in phase2
    }
    training.update(
        refinement=refinement or None,
        epochs_completed=phase2.get("epochs"),
        per_epoch_metrics=phase2.get("per_epoch_metrics"),
        final_metrics=final or None,
    )

    out: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model_spec": model_spec,
        "architecture": {**dict.fromkeys(_ARCHITECTURE_KEYS), **architecture_part},
        "training": training,
    }
    if provenance is not None:
        out["provenance"] = {
            "checkpoint_filename": None,
            "checkpoint_sha256": None,
            **provenance,
        }
    out.update(rest)  # hpo_results, phase3/4_metrics, metadata_uuid
    return out


def _read_run_config(metadata_path: Path) -> dict[str, Any] | None:
    """Return ``config`` from the sibling ``<tag>_run.json``, if there is one."""
    stem = metadata_path.name.removesuffix("_metadata.json")
    run_path = metadata_path.with_name(f"{stem}_run.json")
    if stem == metadata_path.name or not run_path.is_file():
        return None
    try:
        config = json.loads(run_path.read_text(encoding="utf-8")).get("config")
    except (OSError, ValueError):
        return None
    return config if isinstance(config, dict) else None


# ---------------------------------------------------------------------------
# MetadataTracker
# ---------------------------------------------------------------------------


class MetadataTracker:
    """Incremental metadata tracker for IRT model training pipelines.

    Builder-pattern class that accumulates phase-specific data across the
    4-phase deployment pipeline (HPO → main training → calibration
    refinement → robustness).  On ``save()``, provenance is auto-filled
    and the full metadata dict is written as a sidecar JSON file.

    A UUID key-lock binds the JSON to a specific Keras model checkpoint.

    Parameters
    ----------
    model : str
        Model name (e.g. ``"1PL"``, ``"2PL"``, ``"PCM"``, ``"GPCM"``).
    variant : str
        Prior variant (``"fixed"`` or ``"variable"``).
    version : str
        Package version string (e.g. ``"0.1.0"``).

    Examples
    --------
    >>> tracker = MetadataTracker(model="1PL", variant="fixed", version="0.1.0")
    >>> tracker.add_hpo_results(num_trials=100, best_params={"lr": 0.001})
    >>> tracker.attach(model)  # doctest: +SKIP
    >>> tracker.save("checkpoints/1pl_fixed_metadata.json")  # doctest: +SKIP
    """

    def __init__(self, *, model: str, variant: str, version: str) -> None:
        self._data: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "model_spec": {
                "model": model,
                "variant": variant,
                "version": version,
            },
        }
        self._metadata_uuid: str | None = None

    # ------------------------------------------------------------------
    # Builder methods
    # ------------------------------------------------------------------

    def add_training_config(
        self,
        *,
        priors: dict[str, Any] | None = None,
        size_ranges: dict[str, Any] | None = None,
        hparams: dict[str, Any] | None = None,
        size_sampling: dict[str, Any] | None = None,
        seed: int | None = None,
        internal_parameterization: str | None = None,
    ) -> None:
        """Store training configuration.

        Repeated calls merge: only the arguments given are written, so a later
        call can add the seed without restating the rest.

        Parameters
        ----------
        priors : dict, optional
            Prior distribution specifications (``model_spec``).
        size_ranges : dict, optional
            Ranges for P and I (``model_spec``). A ``n_categories`` entry is
            moved to ``model_spec.n_categories``.
        hparams : dict, optional
            Hyperparameters. ``summary`` and ``flow`` become ``architecture``;
            ``base_checkpoint`` and the ``refine_*`` entries become
            ``training.refinement``; ``n_categories`` goes to ``model_spec``;
            everything else lands in ``training``.
        size_sampling : dict, optional
            Which sizes training actually drew, with their probabilities, as
            returned by :func:`bayesflow_irt.simulators.describe_size_sampling`.
            A range alone does not say this: bucketing can leave parts of it
            unvisited.
        seed : int, optional
            Training seed (``training.seed``).
        internal_parameterization : str, optional
            Parameterization the network works in, for example ``"loading"``
            (slope-intercept), as opposed to the reported one.
        """
        spec = self._data["model_spec"]
        if priors is not None:
            spec["priors"] = priors
        if size_ranges is not None:
            sizes = dict(size_ranges)
            n_categories = sizes.pop("n_categories", None)
            spec["size_ranges"] = sizes
            if n_categories is not None:
                spec["n_categories"] = n_categories
        if size_sampling is not None:
            spec["size_sampling"] = size_sampling
        if internal_parameterization is not None:
            spec["internal_parameterization"] = internal_parameterization

        training = self._data.setdefault("training", {})
        if hparams is not None:
            architecture, rest, refinement, n_categories = _split_hparams(hparams)
            if architecture:
                self._data.setdefault("architecture", {}).update(architecture)
            training.update(rest)
            if refinement:
                if not isinstance(training.get("refinement"), dict):
                    training["refinement"] = {}  # v1 upgrade stores None
                training["refinement"].update(refinement)
            if n_categories is not None:
                spec["n_categories"] = n_categories
        if seed is not None:
            training["seed"] = seed

    def add_intended_use(self, *, notes: str) -> None:
        """Store free-text author notes in ``intended_use.notes``.

        The model card renders this separately and labels it as written by the
        author; everything else the card says about use is derived from the
        sidecar (see :func:`bayesflow_irt.envelope.data_contract`).

        Parameters
        ----------
        notes : str
            Context a human wants to add, such as the population or test the
            model was prepared for.
        """
        self._data.setdefault("intended_use", {})["notes"] = notes

    def add_hpo_results(
        self,
        *,
        num_trials: int,
        best_params: dict[str, Any] | None = None,
        pareto_front: list[dict[str, Any]] | None = None,
    ) -> None:
        """Store HPO results (stage 1 of the pipeline).

        Parameters
        ----------
        num_trials : int
            Number of Optuna trials completed.
        best_params : dict, optional
            Best hyperparameters found.
        pareto_front : list of dict, optional
            Pareto-optimal trial records.
        """
        results: dict[str, Any] = {"num_trials": num_trials}
        if best_params is not None:
            results["best_params"] = best_params
        if pareto_front is not None:
            results["pareto_front"] = pareto_front
        self._data["hpo_results"] = results

    def add_phase2_metrics(
        self,
        *,
        epochs: int,
        final_sbc_ecdf: float | None = None,
        final_coverage: float | None = None,
        final_nrmse: float | None = None,
        per_epoch_metrics: list[dict[str, Any]] | None = None,
    ) -> None:
        """Store main training metrics (stage 2) in ``training``.

        Parameters
        ----------
        epochs : int
            Total training epochs completed.
        final_sbc_ecdf : float, optional
            Final SBC ECDF deviation.
        final_coverage : float, optional
            Final 95% CI coverage.
        final_nrmse : float, optional
            Final NRMSE across parameters.
        per_epoch_metrics : list of dict, optional
            Per-epoch metric records.
        """
        training = self._data.setdefault("training", {})
        for stale in ("final_metrics", "per_epoch_metrics"):
            training.pop(stale, None)
        training["epochs_completed"] = epochs
        final = {
            name: value
            for name, value in (
                ("sbc_ecdf", final_sbc_ecdf),
                ("coverage", final_coverage),
                ("nrmse", final_nrmse),
            )
            if value is not None
        }
        if final:
            training["final_metrics"] = final
        if per_epoch_metrics is not None:
            training["per_epoch_metrics"] = per_epoch_metrics

    def add_phase3_metrics(
        self,
        *,
        applied: bool,
        epochs: int | None = None,
        improvement: dict[str, Any] | None = None,
    ) -> None:
        """Store calibration refinement metrics (stage 3).

        Parameters
        ----------
        applied : bool
            Whether calibration refinement was actually performed.
        epochs : int, optional
            Number of refinement epochs.
        improvement : dict, optional
            Before/after improvement metrics.
        """
        metrics: dict[str, Any] = {"applied": applied}
        if epochs is not None:
            metrics["epochs"] = epochs
        if improvement is not None:
            metrics["improvement"] = improvement
        self._data["phase3_metrics"] = metrics

    def add_phase4_metrics(
        self,
        *,
        ood_validation: dict[str, Any] | None = None,
        robustness_analysis: dict[str, Any] | None = None,
    ) -> None:
        """Store robustness training metrics (stage 4).

        Parameters
        ----------
        ood_validation : dict, optional
            Out-of-distribution validation results at corner points.
        robustness_analysis : dict, optional
            Degradation analysis across OOD conditions.
        """
        metrics: dict[str, Any] = {}
        if ood_validation is not None:
            metrics["ood_validation"] = ood_validation
        if robustness_analysis is not None:
            metrics["robustness_analysis"] = robustness_analysis
        self._data["phase4_metrics"] = metrics

    # ------------------------------------------------------------------
    # UUID key-lock
    # ------------------------------------------------------------------

    def attach(self, model: Any) -> str:
        """Bind this tracker to a Keras model via a UUID key-lock.

        Sets ``model.metadata`` (a dict snapshot) and
        ``model.metadata_uuid`` on the model object.  The same UUID is
        stored internally so ``verify()`` can later check consistency.

        Re-attaching overwrites the previous UUID.

        Parameters
        ----------
        model : Keras model
            Any Python object that supports attribute assignment.

        Returns
        -------
        str
            The generated UUID.
        """
        self._metadata_uuid = str(uuid.uuid4())
        self._data["metadata_uuid"] = self._metadata_uuid
        model.metadata = self.to_dict()
        model.metadata_uuid = self._metadata_uuid
        return self._metadata_uuid

    def verify(self, model: Any) -> bool:
        """Check whether a model's UUID matches this tracker's UUID.

        Parameters
        ----------
        model : Keras model
            Model that was previously passed to ``attach()``.

        Returns
        -------
        bool
            ``True`` if the UUIDs match, ``False`` otherwise.
        """
        model_uuid = getattr(model, "metadata_uuid", None)
        return model_uuid is not None and model_uuid == self._metadata_uuid

    @classmethod
    def verify_file(cls, model: Any, path: str | Path) -> bool:
        """Check whether a model's UUID matches the UUID in a JSON file.

        Parameters
        ----------
        model : Keras model
            Model with ``metadata_uuid`` attribute.
        path : str or Path
            Path to the sidecar JSON metadata file.

        Returns
        -------
        bool
            ``True`` if the UUIDs match.
        """
        with open(path, encoding="utf-8") as f:
            file_data = json.load(f)
        model_uuid = getattr(model, "metadata_uuid", None)
        file_uuid = file_data.get("metadata_uuid")
        return model_uuid is not None and model_uuid == file_uuid

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Check metadata against the deployment spec schema.

        Returns a list of error strings.  An empty list means the
        metadata is valid.  Works on partial trackers — only checks
        sections that have been added.

        Returns
        -------
        list of str
            Validation errors (empty if valid).
        """
        errors: list[str] = []

        # model_spec is always present (set in __init__)
        missing_model = _REQUIRED_MODEL_SPEC_KEYS - set(
            self._data.get("model_spec", {})
        )
        if missing_model:
            errors.append(f"model_spec missing keys: {sorted(missing_model)}")

        # provenance is present only after save(); skip if not yet filled
        provenance = self._data.get("provenance")
        if provenance is not None:
            missing_prov = _REQUIRED_PROVENANCE_KEYS - set(provenance)
            if missing_prov:
                errors.append(f"provenance missing keys: {sorted(missing_prov)}")

        # Require at least one phase results section
        training = self._data.get("training") or {}
        has_phase_data = "hpo_results" in self._data or any(
            training.get(k) is not None
            for k in ("epochs_completed", "final_metrics", "per_epoch_metrics")
        )
        if not has_phase_data and provenance is not None:
            errors.append(
                "at least one of hpo_results or training results "
                "(epochs_completed, final_metrics, per_epoch_metrics) is required"
            )

        return errors

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(
        self, path: str | Path, *, checkpoint_path: str | Path | None = None
    ) -> None:
        """Write metadata to a JSON file with provenance auto-fill.

        Raises ``ValueError`` if ``attach()`` has not been called yet.

        Parameters
        ----------
        path : str or Path
            Output file path.
        checkpoint_path : str or Path, optional
            The saved ``.keras`` file. Its filename and SHA-256 are stored in
            ``provenance``. This is the binding that survives a Keras
            save/load, which the UUID does not: ``metadata_uuid`` is a plain
            attribute and is gone from a loaded model. Save the checkpoint
            first, then pass its path. When omitted, a binding already
            present (a loaded sidecar re-saved after adding stage 3/4 metrics)
            is kept, not dropped.
        """
        if self._metadata_uuid is None:
            raise ValueError("Must call attach(model) before save().")

        # Auto-fill provenance at save time
        # Keep fields recorded earlier (checkpoint binding, later additions);
        # refresh the ones _get_provenance() owns.
        provenance: dict[str, Any] = dict(self._data.get("provenance") or {})
        provenance.update(_get_provenance())
        if checkpoint_path is not None:
            provenance["checkpoint_filename"] = Path(checkpoint_path).name
            provenance["checkpoint_sha256"] = _sha256_file(checkpoint_path)
        self._data["provenance"] = provenance
        self._data["metadata_uuid"] = self._metadata_uuid

        sanitized = _sanitize_for_json(self._data)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            # Mostly belt and braces: once `_sanitize_for_json` has nulled
            # the non-finite float *values* this cannot fire on them. It is
            # not dead, though -- the sanitizer rewrites dict values and
            # leaves dict *keys* alone, and `json.dump` coerces a float key
            # to a string, so a non-finite key would otherwise be written as
            # `{"-Infinity": ...}`. With the flag it raises instead. It also
            # catches a future path that reaches the writer without the
            # sanitizer.
            json.dump(sanitized, f, indent=2, allow_nan=False)

    @classmethod
    def load(cls, path: str | Path) -> MetadataTracker:
        """Load a ``MetadataTracker`` from a JSON file.

        The returned tracker is mutable — builder methods remain
        available so you can load after training and add stage 3/4 metrics later.

        A file without ``schema_version`` is read as version 1 and converted
        in memory by :func:`upgrade_v1_to_v2`; the file itself is not touched.
        For such a file the sibling ``<tag>_run.json`` is read, if present,
        for the seed and the internal parameterization.

        Parameters
        ----------
        path : str or Path
            Path to the JSON metadata file.

        Returns
        -------
        MetadataTracker
            Reconstructed tracker with all stored data.
        """
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        file_version = data.get("schema_version")
        if file_version is None:
            data = upgrade_v1_to_v2(data, run_config=_read_run_config(path))
        elif file_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported metadata schema_version {file_version!r}; "
                f"this bayesflow_irt reads versions 1 and {SCHEMA_VERSION}."
            )
        if "model_spec" not in data:
            raise KeyError("Missing required key 'model_spec' in metadata file.")

        spec = data["model_spec"]
        tracker = cls(
            model=spec["model"],
            variant=spec["variant"],
            version=spec["version"],
        )
        # Keep every key, including ones this version does not know about.
        tracker._data = copy.deepcopy(data)

        tracker._metadata_uuid = data.get("metadata_uuid")

        return tracker

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a deep copy of the internal metadata dict."""
        return copy.deepcopy(self._data)
