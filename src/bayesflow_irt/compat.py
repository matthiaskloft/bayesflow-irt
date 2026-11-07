"""Checkpoint-adjacent helpers: weight digests and size-condition moments.

:func:`weights_digest` fingerprints a model's weight values so callers can tell
a real checkpoint restore from a swallowed load failure (``bayesflow_hpo``'s
loader swallows load errors). The size-moment helpers reproduce the
standardization of the size conditions a weights-only checkpoint was trained
with.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any

__all__ = [
    "SIZE_MOMENTS_CHANGED_AT",
    "adapter_size_moments",
    "apply_checkpoint_size_moments",
    "legacy_log_size_moments",
    "legacy_size_moments",
    "resolve_checkpoint_size_moments",
    "set_adapter_size_moments",
    "weights_digest",
]

#: When ``adapters._log_size_moments`` changed.
#: A weights-only checkpoint trained before this instant saw size conditions
#: standardized with :func:`legacy_log_size_moments`.
SIZE_MOMENTS_CHANGED_AT = datetime(
    2026, 10, 5, 13, 8, 58, tzinfo=timezone(timedelta(hours=2))
)

_SIZE_KEYS = ("n_items", "n_persons")


def weights_digest(model: Any) -> str:
    """Return a short digest of ``model``'s current weight values.

    Comparing this before and after a load is the only reliable way to tell a
    real restore from a swallowed failure.
    """
    import numpy as np

    h = hashlib.sha256()
    for variable in model.weights:
        h.update(np.ascontiguousarray(np.asarray(variable)).tobytes())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Size-condition moments of weights-only checkpoints
# ---------------------------------------------------------------------------
#
# A weights-only checkpoint (``weights.weights.h5`` + ``checkpoint.json``)
# carries no adapter; readers rebuild it from the study config with the
# *current* code. ``adapters._log_size_moments`` changed (the mean/sd used to
# standardize ``log(n_items)`` / ``log(n_persons)`` in
# ``_standardize_size_conditions``; see ``SIZE_MOMENTS_CHANGED_AT``).
# A checkpoint trained earlier therefore receives different size conditions
# on reload and may be mis-scored. Full
# ``.keras`` saves serialize the adapter and are unaffected.


def legacy_log_size_moments(
    value_range: tuple[int, int], bucket: int | None
) -> tuple[float, float]:
    """Mean and sd of ``log(size)`` as computed before the size-moments change.

    The support is every integer in ``[lo, hi]`` when ``bucket`` is ``None``,
    otherwise one representative per bucket,
    ``min(lo + i * bucket + bucket - 1, hi)`` for
    ``i = 0 .. ceil((hi - lo + 1) / bucket) - 1`` (the bucket's upper edge).
    The moments are the unweighted mean and the population standard deviation
    (``np.std``) of ``log(support)``; the sd is forced to 1.0 when it is 0.

    Parameters
    ----------
    value_range : tuple of int
        Inclusive ``(lo, hi)`` size range.
    bucket : int or None
        Bucket width, or ``None`` for unbucketed sizes.

    Returns
    -------
    tuple of float
        ``(mean, std)``.
    """
    import math

    import numpy as np

    lo, hi = int(value_range[0]), int(value_range[1])
    if bucket is None:
        support = np.arange(lo, hi + 1, dtype=np.float64)
    else:
        n = math.ceil((hi - lo + 1) / bucket)
        support = np.array(
            [min(lo + i * bucket + bucket - 1, hi) for i in range(n)],
            dtype=np.float64,
        )
    logs = np.log(support)
    std = float(np.std(logs))
    return float(np.mean(logs)), std if std > 0 else 1.0


def legacy_size_moments(simulator: Any) -> dict[str, list[float]]:
    """Pre-change moments for ``n_items`` and ``n_persons`` of ``simulator``.

    Uses ``simulator.n_items`` / ``item_bucket`` and ``simulator.n_persons`` /
    ``person_bucket`` with :func:`legacy_log_size_moments`.

    Returns
    -------
    dict
        ``{"n_items": [mean, std], "n_persons": [mean, std]}``.
    """
    pairs = (("n_items", "item_bucket"), ("n_persons", "person_bucket"))
    return {
        key: list(
            legacy_log_size_moments(
                getattr(simulator, key), getattr(simulator, bucket_attr, None)
            )
        )
        for key, bucket_attr in pairs
    }


def _size_standardize_transform(adapter: Any, key: str) -> Any:
    """The adapter's ``Standardize`` transform for ``key``, or ``None``."""
    from bayesflow.adapters.transforms import Standardize

    for transform in adapter.transforms:
        if (
            getattr(transform, "include", None) == [key]
            and getattr(transform, "transform_constructor", None) is Standardize
        ):
            return transform
    return None


def adapter_size_moments(adapter: Any) -> dict[str, list[float]]:
    """Mean and sd that ``adapter`` uses to standardize the size conditions.

    Record this next to a weights-only checkpoint (``checkpoint.json``) so a
    later reader can restore exactly the conditions the network was trained
    with, whatever the current ``_log_size_moments`` computes.

    Returns
    -------
    dict
        ``{"n_items": [mean, std], "n_persons": [mean, std]}`` as plain floats.

    Raises
    ------
    ValueError
        If the adapter has no ``Standardize`` transform for a size key.
    """
    out: dict[str, list[float]] = {}
    for key in _SIZE_KEYS:
        transform = _size_standardize_transform(adapter, key)
        if transform is None:
            raise ValueError(f"adapter has no Standardize transform for {key!r}")
        out[key] = [
            float(transform.kwargs["mean"]),
            float(transform.kwargs["std"]),
        ]
    return out


def set_adapter_size_moments(adapter: Any, moments: dict[str, Any]) -> None:
    """Overwrite the size-condition moments of ``adapter`` in place.

    Replaces the ``mean`` / ``std`` kwargs of the ``Standardize`` transforms
    for the keys in ``moments`` (as ``np.float32``) and clears their lazily
    filled ``transform_map`` cache, so the new values take effect on the next
    call.

    Parameters
    ----------
    adapter : bayesflow.adapters.Adapter
        An adapter built by one of the ``make_*_adapter`` factories.
    moments : dict
        ``{key: [mean, std]}`` as returned by :func:`adapter_size_moments`.

    Raises
    ------
    ValueError
        If a key's ``Standardize`` transform is missing. Nothing is modified
        in that case.
    """
    import numpy as np

    targets = {}
    for key in moments:
        transform = _size_standardize_transform(adapter, key)
        if transform is None:
            raise ValueError(f"adapter has no Standardize transform for {key!r}")
        targets[key] = transform
    for key, transform in targets.items():
        mean, std = moments[key]
        transform.kwargs = {
            **transform.kwargs,
            "mean": np.float32(mean),
            "std": np.float32(std),
        }
        transform.transform_map = {}


def resolve_checkpoint_size_moments(
    record: dict[str, Any],
    simulator: Any,
    trained_at: datetime | None = None,
) -> dict[str, list[float]] | None:
    """Decide which size moments a weights-only checkpoint was trained with.

    Precedence: moments recorded in ``record["adapter_size_moments"]``; else,
    if ``trained_at`` is earlier than :data:`SIZE_MOMENTS_CHANGED_AT`, the
    :func:`legacy_size_moments` of ``simulator``; else ``None``.

    Parameters
    ----------
    record : dict
        The checkpoint's ``checkpoint.json`` contents (may be empty).
    simulator : object
        The simulator the adapter was built from.
    trained_at : datetime, optional
        When training finished. Must be timezone-aware to be compared.

    Returns
    -------
    dict or None
        Moments to apply with :func:`set_adapter_size_moments`, or ``None``
        meaning the current code's moments apply unchanged.
    """
    if record and "adapter_size_moments" in record:
        return record["adapter_size_moments"]
    if trained_at is not None and trained_at < SIZE_MOMENTS_CHANGED_AT:
        return legacy_size_moments(simulator)
    return None


def apply_checkpoint_size_moments(
    adapter: Any,
    simulator: Any,
    mode: str = "auto",
    record: dict[str, Any] | None = None,
    trained_at: datetime | None = None,
) -> str:
    """Set ``adapter``'s size moments for a weights-only checkpoint.

    For code that rebuilds an adapter from a study config and then loads
    ``weights.weights.h5``.

    Parameters
    ----------
    adapter, simulator
        As for :func:`set_adapter_size_moments` /
        :func:`resolve_checkpoint_size_moments`.
    mode : {"auto", "current", "legacy"}
        ``"auto"`` uses :func:`resolve_checkpoint_size_moments`;
        ``"current"`` leaves the adapter as built; ``"legacy"`` forces
        :func:`legacy_size_moments`.
    record : dict, optional
        The checkpoint's ``checkpoint.json`` contents.
    trained_at : datetime, optional
        When the checkpoint was trained. A naive value (as Optuna's
        ``datetime_complete``) is read in the UTC offset of
        :data:`SIZE_MOMENTS_CHANGED_AT`.

    Returns
    -------
    str
        A one-line description of which moments are in effect.
    """
    if mode not in ("auto", "current", "legacy"):
        raise ValueError(f"mode must be auto, current or legacy; got {mode!r}")
    if trained_at is not None and trained_at.tzinfo is None:
        trained_at = trained_at.replace(tzinfo=SIZE_MOMENTS_CHANGED_AT.tzinfo)
    if mode == "current":
        return "current code"
    moments: dict[str, list[float]] | None
    if mode == "legacy":
        moments = legacy_size_moments(simulator)
        label = "legacy size moments (forced)"
    else:
        moments = resolve_checkpoint_size_moments(record or {}, simulator, trained_at)
        label = (
            "from checkpoint"
            if record and "adapter_size_moments" in record
            else "legacy size moments"
        )
    if moments is None:
        return "current code"
    set_adapter_size_moments(adapter, moments)
    return f"{label}: {moments}"
