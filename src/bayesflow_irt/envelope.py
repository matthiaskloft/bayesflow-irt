"""Input envelope of a trained model and the guard that checks data against it.

A checkpoint is only valid for data like what it was trained on.  The sidecar
records that (``model_spec.size_ranges``, ``n_categories``); this module reads
those same fields for both the runtime check, :func:`check_data`, and the model
card, so the two cannot disagree.

The check is on the nominal bounds only, never on exact sizes: the networks are
variable-size and a size inside the range that bucketing never drew is still
inside the envelope.  Which sizes training actually drew is a separate record,
``model_spec.size_sampling``, and affects only the card's trained-cells map.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .metadata import SCHEMA_VERSION, MetadataTracker, upgrade_v1_to_v2

__all__ = ["DataContractError", "EnvelopeCheck", "check_data", "data_contract"]

_BINARY = frozenset({"1PL", "2PL"})
_POLYTOMOUS = frozenset({"PCM", "GPCM"})
_AXES = ("n_persons", "n_items")

Status = Literal["inside", "outside", "unknown"]


class DataContractError(ValueError):
    """Data the model cannot be run on at all, whatever ``strict`` says."""


@dataclass(frozen=True)
class EnvelopeCheck:
    """Result of :func:`check_data`.

    Attributes
    ----------
    status : {"inside", "outside", "unknown"}
        ``"outside"`` if any axis is outside the trained range, else
        ``"unknown"`` if any axis has no recorded range, else ``"inside"``.
    axes : dict
        Per-axis status for ``"n_persons"`` and ``"n_items"``.
    trained_ranges : dict
        The ``[low, high]`` range recorded for each axis, or ``None``.
    messages : list of str
        Human-readable findings; empty when everything is inside.
    """

    status: Status
    axes: dict[str, Status]
    trained_ranges: dict[str, list[int] | None]
    messages: list[str] = field(default_factory=list)


def _model_spec(metadata: Any) -> dict[str, Any]:
    """Return the v2 ``model_spec`` from a tracker, a dict, or a sidecar path."""
    if isinstance(metadata, MetadataTracker):
        data = metadata.to_dict()
    elif isinstance(metadata, (str, Path)):
        data = MetadataTracker.load(metadata).to_dict()
    elif isinstance(metadata, dict):
        data = metadata
        version = data.get("schema_version")
        if version is None and "model_info" in data:
            data = upgrade_v1_to_v2(data)
        elif version is not None and version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported metadata schema_version {version!r}; "
                f"this bayesflow_irt reads versions 1 and {SCHEMA_VERSION}."
            )
    else:
        raise TypeError(
            "metadata must be a MetadataTracker, a sidecar dict, or a path; "
            f"got {type(metadata).__name__}."
        )
    if "model_spec" not in data:
        raise KeyError("metadata has no 'model_spec' section.")
    return data["model_spec"]


def _family(spec: dict[str, Any]) -> str | None:
    name = str(spec.get("model") or "").upper()
    return name if name in _BINARY | _POLYTOMOUS else None


def _orientation(family: str | None) -> str:
    base = (
        "All items must be keyed in the same direction (a higher response "
        "means more of the trait)"
    )
    if family in ("2PL", "GPCM"):
        why = "the discrimination prior is positive, so"
    elif family in ("1PL", "PCM"):
        why = "discrimination is fixed at 1 (not estimated), so"
    else:
        return base + "; reverse-keyed items must be recoded first."
    return f"{base}: {why} reverse-keyed items must be recoded first."


def _score_scale(priors: dict[str, Any] | None) -> str:
    theta = (priors or {}).get("theta")
    if not isinstance(theta, dict) or not theta.get("name"):
        prior = "the theta prior is not recorded in this sidecar"
    else:
        name = str(theta["name"])
        loc, scale = theta.get("loc"), theta.get("scale")
        if name == "normal" and loc == 0 and scale == 1:
            prior = "theta ~ N(0, 1)"
        else:
            extra = ", ".join(f"{k}={v}" for k, v in theta.items() if k != "name")
            prior = f"the recorded theta prior is {name}({extra})"
    return (
        f"theta lives on the scale of its training prior ({prior}), which fixes "
        "the location and scale of the latent trait. Scores from "
        "separate fits are not on a common scale."
    )


_NO_MISSING_NOTE = (
    "Not supported. The summary network has no mask, so a missing "
    "cell or a padding sentinel is pooled in as if it were a real "
    "response (MatrixEquivariantIRTSummary docstring). Remove "
    "incomplete persons or items before inference."
)


def _missing_responses(spec: dict[str, Any]) -> dict[str, Any]:
    """Structured missing-response record; unsupported unless training says so.

    ``model_spec.missing_responses`` is where a model trained with missing
    cells will record its ``coding``, ``mechanisms`` and
    ``max_missing_rate``. No current simulator produces missing cells, so no
    sidecar has it. ``supported`` and ``note`` always describe this code,
    whatever the record claims: :func:`check_data` rejects missing values,
    so the contract must not advertise them.
    """
    record = spec.get("missing_responses")
    out: dict[str, Any] = {
        "supported": False,
        "coding": None,
        "mechanisms": [],
        "max_missing_rate": None,
    }
    if isinstance(record, dict):
        out.update(record)  # unknown keys are carried through
    out["supported"] = False
    out["note"] = _NO_MISSING_NOTE
    return out


def data_contract(metadata: Any) -> dict[str, Any]:
    """State what data the model accepts, derived from its ``model_spec``.

    Derived rather than stored, so a sidecar cannot contradict the code and
    old (v1) sidecars get the same statement.  The model card prints these
    fields verbatim.

    Parameters
    ----------
    metadata : MetadataTracker, dict or path
        The sidecar.

    Returns
    -------
    dict
        ``response_coding``, ``orientation`` and ``score_scale``, each a
        sentence, and ``missing_responses``, a record with ``supported``,
        ``coding`` (how a missing cell is marked), ``mechanisms`` (the
        missingness simulated in training, e.g. ``"mcar"``),
        ``max_missing_rate`` (the highest rate trained on) and ``note``.
        Read from ``model_spec.missing_responses`` when present; otherwise
        unsupported, which is true of every current model.
    """
    spec = _model_spec(metadata)
    family = _family(spec)
    n_cat = spec.get("n_categories")
    if family in _POLYTOMOUS and n_cat:
        coding = (
            f"Integer categories 0..{int(n_cat) - 1} ({int(n_cat)} categories); "
            "the model is specific to this K and cannot be used with another."
        )
    elif family in _POLYTOMOUS:
        coding = "Integer categories 0..K-1; the model is specific to its K."
    elif family in _BINARY:
        coding = "Integer 0/1 (1 = correct or endorsed)."
    else:
        coding = "Not recorded for this model family."
    return {
        "missing_responses": _missing_responses(spec),
        "response_coding": coding,
        "orientation": _orientation(family),
        "score_scale": _score_scale(spec.get("priors")),
    }


def _validate_responses(
    responses: Any, family: str | None, n_categories: int | None
) -> np.ndarray:
    if np.ma.is_masked(responses):
        raise DataContractError(
            "responses is a masked array with masked cells. Missing responses "
            "are not supported (see data_contract()); remove incomplete "
            "persons or items."
        )
    arr = np.asarray(responses)
    if arr.ndim not in (2, 3):
        raise DataContractError(
            "responses must have shape (persons, items) or "
            f"(batch, persons, items); got {arr.shape}."
        )
    if arr.dtype.kind not in "iubf":
        raise DataContractError(f"responses must be numeric; got dtype {arr.dtype}.")
    if 0 in arr.shape:
        raise DataContractError(
            f"responses has a zero-size axis (shape {arr.shape}); the model "
            "needs at least one person and one item."
        )
    if arr.dtype.kind == "f":
        values = arr.astype(float)
        if not np.all(np.isfinite(values)):
            raise DataContractError(
                "responses contain NaN or Inf. Missing responses are not "
                "supported (see data_contract()); remove incomplete persons "
                "or items."
            )
        if not np.all(values == np.round(values)):
            raise DataContractError("responses must be integer-valued.")
    else:
        values = arr  # integers and bools: finite and integral by construction
    top = 1 if family in _BINARY else (n_categories - 1 if n_categories else None)
    if values.min() < 0 or (top is not None and values.max() > top):
        where = f"0..{top}" if top is not None else ">= 0"
        raise DataContractError(
            f"responses must lie in {where}; found "
            f"[{float(values.min()):g}, {float(values.max()):g}]."
        )
    return arr


def check_data(
    responses: Any = None,
    *,
    n_persons: int | None = None,
    n_items: int | None = None,
    n_categories: int | None = None,
    metadata: Any,
    strict: bool = False,
    stacklevel: int = 2,
) -> EnvelopeCheck:
    """Check data against the envelope a model was trained on.

    Pass either ``responses`` (shape ``(persons, items)`` or
    ``(batch, persons, items)``), which are then also validated against the
    data contract, or the sizes directly.

    Parameters
    ----------
    responses : array-like, optional
        Response matrix; sizes are read from its last two axes.
    n_persons, n_items : int, optional
        Sizes, when ``responses`` is not given.
    n_categories : int, optional
        Number of categories of the data, for PCM/GPCM.
    metadata : MetadataTracker, dict or path
        The model's sidecar.  A loaded checkpoint carries none, so it must be
        passed explicitly.
    strict : bool
        Raise instead of warn when a size is outside the trained range.
    stacklevel : int
        Passed to :func:`warnings.warn`, so a wrapper can point the warning at
        its own caller.

    Returns
    -------
    EnvelopeCheck

    Raises
    ------
    DataContractError
        Always, regardless of ``strict``, for data the model cannot run on:
        a category count different from the model's, responses outside
        ``0..K-1``, or missing/non-finite responses.
    ValueError
        With ``strict=True``, for a size outside the trained range.

    Notes
    -----
    A missing ``size_ranges`` (for example an HPO-only sidecar) gives
    ``"unknown"`` for that axis and a warning; it never blocks.
    """
    spec = _model_spec(metadata)
    family = _family(spec)
    model_k = spec.get("n_categories")

    expected_k = 2 if family in _BINARY else model_k
    if n_categories is not None and expected_k is not None:
        if n_categories != expected_k:
            raise DataContractError(
                f"Data has {n_categories} categories but the model expects "
                f"K={expected_k}; each PCM/GPCM model is trained for one "
                "number of categories."
            )
    if responses is not None:
        arr = _validate_responses(
            responses, family, model_k if model_k is not None else n_categories
        )
        shape = {"n_persons": arr.shape[-2], "n_items": arr.shape[-1]}
        for axis, given in (("n_persons", n_persons), ("n_items", n_items)):
            if given is not None and given != shape[axis]:
                raise DataContractError(
                    f"{axis}={given} disagrees with the response matrix "
                    f"({shape[axis]}); pass the matrix alone or matching sizes."
                )
        n_persons, n_items = shape["n_persons"], shape["n_items"]
    sizes = {"n_persons": n_persons, "n_items": n_items}
    if any(v is None for v in sizes.values()):
        raise TypeError("Provide responses, or both n_persons and n_items.")
    for axis, value in sizes.items():
        if value is None or int(value) != value or value < 1:
            raise DataContractError(
                f"{axis} must be a positive integer; got {value!r}."
            )

    ranges = spec.get("size_ranges") or {}
    axes: dict[str, Status] = {}
    trained: dict[str, list[int] | None] = {}
    messages: list[str] = []
    for axis in _AXES:
        rng = ranges.get(axis)
        trained[axis] = list(rng) if rng else None
        if not rng:
            axes[axis] = "unknown"
            messages.append(f"{axis}: no trained range recorded; cannot check.")
        elif rng[0] <= sizes[axis] <= rng[1]:
            axes[axis] = "inside"
        else:
            axes[axis] = "outside"
            messages.append(
                f"{axis}={sizes[axis]} is outside the trained range "
                f"[{rng[0]}, {rng[1]}]; posteriors there are not validated."
            )

    if "outside" in axes.values():
        status: Status = "outside"
    elif "unknown" in axes.values():
        status = "unknown"
    else:
        status = "inside"
    result = EnvelopeCheck(status, axes, trained, messages)

    if status == "outside":
        text = " ".join(messages)
        if strict:
            raise ValueError(text)
        warnings.warn(text, UserWarning, stacklevel=stacklevel)
    elif status == "unknown":
        warnings.warn(" ".join(messages), UserWarning, stacklevel=stacklevel)
    return result
