"""Shared posterior sampling for evaluation: the model card and HPO validation.

:func:`sample_posterior` draws item posteriors for a batch of datasets in
chunks of datasets. Each chunk is one ``approximator.sample`` call whose
rows are passed as one batch, so the GPU sees few large calls instead of
many small ones. The chunk size can be chosen from free device memory
(:func:`_auto_chunk_size`), and a chunk that runs out of memory shrinks and
is retried. Draws are returned as NumPy arrays; every metric downstream
runs on them unchanged.
"""

from __future__ import annotations

import gc
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _containing(value: Any, depth: int = 0) -> Iterator[Any]:
    """Objects reachable from ``value`` through attributes and containers."""
    if depth > 6:
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            yield from _containing(item, depth + 1)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _containing(item, depth + 1)
    elif hasattr(value, "__dict__"):
        yield value


def reset_posterior_seed(model: Any, seed: int) -> int:
    """Seed the global RNGs and every ``SeedGenerator`` of a loaded model.

    ``keras.utils.set_random_seed`` does not touch a ``SeedGenerator`` that
    already exists, so a loaded model samples differently on every call
    under a fixed global seed. Each generator found by a
    walk over attributes and containers is reset to ``[seed + offset, 0]``,
    offsets in traversal order. The
    order is fixed for one architecture and environment, which is what a
    regenerated card needs.

    Returns
    -------
    int
        The number of generators reset.
    """
    import keras

    seed = int(seed) % 2**31
    keras.utils.set_random_seed(seed)
    sg_cls = keras.random.SeedGenerator
    found, stack, seen = [], [model], set()
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, sg_cls):
            found.append(obj)
            continue
        for attr in getattr(obj, "__dict__", {}).values():
            stack.extend(_containing(attr))
    for offset, generator in enumerate(found):
        generator.state.assign([seed + offset, 0])
    return len(found)


def _chunk_seed(seed: int, start: int) -> int:
    return int(np.random.SeedSequence([seed, start]).generate_state(1)[0])


#: On an out-of-memory error the chunk shrinks to this fraction of its size
#: and stays there for the rest of the cell.
CHUNK_SHRINK = 0.75
#: ``chunk_size="auto"``: datasets in the two memory probes, the share of
#: free device memory a chunk may use, and the size without a CUDA device.
PROBE_DATASETS = (2, 8)
MEMORY_SHARE = 0.8
FALLBACK_CHUNK = 50


def _torch_cuda() -> bool:
    """Whether Keras runs on PyTorch with a CUDA device.

    The memory probe and the GPU person scorer use PyTorch's allocator; under
    another Keras backend that framework owns the device, so both stay off.
    """
    import keras

    if keras.backend.backend() != "torch":
        return False
    import torch

    return torch.cuda.is_available()


def _drop_tracebacks(exc: BaseException) -> BaseException:
    """Strip the tracebacks of ``exc`` and the exceptions chained to it.

    Their frames hold a failed attempt's device tensors for as long as the
    exception lives.
    """
    seen: set[int] = set()
    stack: list[BaseException | None] = [exc]
    while stack:
        link = stack.pop()
        if link is None or id(link) in seen:
            continue
        seen.add(id(link))
        link.__traceback__ = None
        stack += [link.__cause__, link.__context__]
    return exc


def _auto_chunk_size(
    probe: Any, n: int, share: float = MEMORY_SHARE
) -> tuple[int, dict[str, Any]]:
    """Datasets per chunk from a probe's peak device memory.

    ``probe(k)`` samples ``k`` datasets; their draws are overwritten later,
    because every real chunk resets the posterior seed. Two probes
    (:data:`PROBE_DATASETS`) measure peak device memory; memory per dataset
    is the larger of the small probe's peak over its size (which includes
    the fixed overhead) and the growth between the probes per added dataset
    (the marginal cost, which a small probe can understate). The chunk is
    ``share`` (default :data:`MEMORY_SHARE`) of the free device memory (free
    on the device plus held by the caching allocator but unused) over that.
    Without CUDA, or under a Keras backend other than PyTorch, it is
    :data:`FALLBACK_CHUNK`.
    """
    from .utils import is_oom_error, safe_empty_cache

    if not _torch_cuda():
        size = min(FALLBACK_CHUNK, n)
        return size, {"rule": "auto", "device": None, "chunk_size": size}
    import torch

    sizes = sorted({min(k, n) for k in PROBE_DATASETS})
    peaks: list = []
    for k in sizes:
        safe_empty_cache()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        oom = False
        try:
            probe(k)
        except Exception as exc:
            if not is_oom_error(exc):
                raise
            oom = True
        if oom:
            gc.collect()
            safe_empty_cache()
            # The larger probe failed: the smaller one fit.
            size = sizes[0] if peaks else 1
            rule = {"rule": "auto", "probe": "out of memory", "chunk_size": size}
            return size, rule
        peaks.append(max(torch.cuda.max_memory_allocated() - base, 1))
    per_dataset = peaks[0] / sizes[0]
    if len(sizes) > 1:
        per_dataset = max(per_dataset, (peaks[1] - peaks[0]) / (sizes[1] - sizes[0]))
    safe_empty_cache()
    free = torch.cuda.mem_get_info()[0] + (
        torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    )
    size = int(min(n, max(1, share * free // per_dataset)))
    return size, {
        "rule": "auto",
        "device": torch.cuda.get_device_name(),
        "probe_datasets": sizes,
        "probe_peak_mib": [round(p / 2**20, 1) for p in peaks],
        "mib_per_dataset": round(per_dataset / 2**20, 2),
        "free_mib": round(free / 2**20),
        "memory_share": share,
        "chunk_size": size,
    }


class SamplingFailedError(RuntimeError):
    """A chunk ran out of memory at one dataset and one row batch.

    ``record`` holds the chunk start and the sizes tried; ``sizes`` the
    sizes as text, for a caller's own message.
    """

    def __init__(self, message: str, record: dict[str, Any], sizes: str):
        super().__init__(message)
        self.record = record
        self.sizes = sizes


@dataclass
class PosteriorDraws:
    """Item-parameter draws of a batch of datasets, and how they were drawn.

    Attributes
    ----------
    draws : dict
        Draws per key, ``(n_datasets, n_draws, ...)``, float32.
    retries : list of dict
        Out-of-memory retries: chunk start, its size, and the shrunk size
        (``retry_size``) or row batch (``retry_rows``) used from then on.
    seconds : float
        Wall time spent sampling, including the memory probe.
    chunk_size : int
        Datasets per ``sample`` call before any retry.
    final_chunk_size : int
        Datasets per call after the last out-of-memory shrink, a safe size
        to reuse for the same data shape.
    autoregressive_rows : str or None
        For an autoregressive item posterior, ``"whole chunk"`` or
        ``"at most <k>"`` after a row-batch shrink; ``None`` otherwise.
    chunk_rule : dict or None
        For ``chunk_size="auto"``, how the size was chosen.
    """

    draws: dict[str, np.ndarray]
    retries: list[dict[str, Any]] = field(default_factory=list)
    seconds: float = 0.0
    chunk_size: int = 0
    final_chunk_size: int = 0
    autoregressive_rows: str | None = None
    chunk_rule: dict[str, Any] | None = None


def sample_posterior(
    approximator: Any,
    conditions: Mapping[str, Any],
    num_samples: int,
    keys: Sequence[str],
    *,
    chunk_size: int | str = 50,
    memory_share: float = MEMORY_SHARE,
    seed: int | None = None,
    batch_flow_rows: bool = False,
    sample_kwargs: Mapping[str, Any] | None = None,
    responses_key: str = "responses",
) -> PosteriorDraws:
    """Draw item posteriors for every dataset of ``conditions`` in chunks.

    Datasets are sampled ``chunk_size`` at a time. With ``seed``, the
    posterior seed is reset before each chunk from ``seed`` and the chunk's
    first dataset index (:func:`reset_posterior_seed`), so the draws are
    reproducible for a given chunking. ``chunk_size="auto"`` picks the size
    from free device memory (:func:`_auto_chunk_size`). On an out-of-memory
    error the chunk shrinks to :data:`CHUNK_SHRINK` of its size and is
    retried from the same start; the smaller size is kept for the remaining
    datasets and every shrink is recorded.

    An autoregressive item posterior samples a chunk's ``(dataset, draw)``
    rows as one batch; when a one-dataset chunk runs out of memory, that
    row batch shrinks the same way. The approximator's default of 256 rows
    makes one flow call per item for every 256 rows, and that many small
    calls leave the GPU idle. ``batch_flow_rows=True`` does the same for a
    flow-matching item posterior: all ``(dataset, item)`` conditions and
    all draws of a chunk go into one ODE solve. It is off by default
    because it changes the random stream, so the model card keeps its
    existing draws.

    Parameters
    ----------
    approximator
        Has ``sample(conditions=..., num_samples=...)``.
    conditions : mapping
        Simulator output, leading axis datasets; sliced per chunk.
    num_samples : int
        Posterior draws per dataset.
    keys : sequence of str
        Parameter keys to keep from each ``sample`` result.
    chunk_size : int or "auto"
        Datasets per ``sample`` call.
    memory_share : float
        With ``"auto"``, the share of free device memory a chunk may use.
    seed : int, optional
        Base posterior seed, reset per chunk; ``None`` leaves the RNG alone.
    batch_flow_rows : bool
        Pass a flow-matching chunk as one batch of conditions and draws. If
        one dataset does not fit that way, the remaining datasets fall back
        to the approximator's default row chunks (``retry_rows="default"``).
    sample_kwargs : mapping, optional
        Extra keyword arguments for every ``sample`` call (for example
        ``flow_sample_kwargs``).
    responses_key : str
        Key of the response matrix ``(n, persons, items[, ...])``, read for
        the item count when ``batch_flow_rows`` is set.

    Returns
    -------
    PosteriorDraws

    Raises
    ------
    SamplingFailedError
        If a chunk runs out of memory at one dataset and one row.
    """
    from .utils import is_oom_error, safe_empty_cache

    if chunk_size != "auto" and (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size < 1
    ):
        raise ValueError(f"chunk_size must be >= 1 or 'auto'; got {chunk_size!r}.")
    if not 0 < memory_share <= 1:
        raise ValueError(f"memory_share must be in (0, 1]; got {memory_share}.")
    n = int(np.shape(next(iter(conditions.values())))[0])
    # `is True`: a mock answers any attribute with a truthy object.
    autoregressive = getattr(approximator, "autoregressive", False) is True
    batch_flow = batch_flow_rows and not autoregressive
    n_items = int(np.shape(conditions[responses_key])[2]) if batch_flow else 0
    base_kwargs = dict(sample_kwargs or {})
    draws: dict[str, np.ndarray] = {}
    retries: list[dict[str, Any]] = []
    # Cap on an autoregressive row batch; None is the whole chunk.
    row_cap: int | None = None
    # Set when one dataset does not fit as one flow-matching batch: the rest
    # is sampled at the approximator's own row chunks.
    flow_unbatched = False

    def run(start: int, stop: int) -> None:
        if seed is not None:
            reset_posterior_seed(approximator, _chunk_seed(seed, start))
        chunk = {k: v[start:stop] for k, v in conditions.items()}
        extra = dict(base_kwargs)
        if autoregressive:
            rows = (stop - start) * num_samples
            extra["condition_chunk_size"] = (
                rows if row_cap is None else min(rows, row_cap)
            )
        elif batch_flow and not flow_unbatched:
            extra["condition_chunk_size"] = (stop - start) * n_items
            extra["sample_chunk_size"] = num_samples
        post = approximator.sample(conditions=chunk, num_samples=num_samples, **extra)
        for key in keys:
            values = np.asarray(post[key], dtype=np.float32)
            if key not in draws:
                draws[key] = np.empty((n, *values.shape[1:]), dtype=np.float32)
            draws[key][start:stop] = values

    def failed(record: dict[str, Any], exc: BaseException) -> SamplingFailedError:
        sizes = ", ".join(str(record[k]) for k in ("size", "retry_size") if k in record)
        err = SamplingFailedError(
            f"Out of memory at chunk size(s) {sizes}.", record, sizes
        )
        err.__cause__ = exc
        return err

    t0 = time.perf_counter()
    chunk_rule = None
    if chunk_size == "auto":
        first, chunk_rule = _auto_chunk_size(lambda k: run(n - k, n), n, memory_share)
    else:
        first = int(chunk_size)
    size = first
    start = 0
    while start < n:
        stop = min(start + size, n)
        oom = None
        try:
            run(start, stop)
        except Exception as exc:
            if not is_oom_error(exc):
                raise
            oom = _drop_tracebacks(exc)
        if oom is None:
            start = stop
            continue
        # Shrink outside the except block: the live traceback would otherwise
        # keep the failed attempt's device tensors allocated. Frames left in
        # reference cycles are freed only by the cycle collector.
        gc.collect()
        safe_empty_cache()
        record: dict[str, Any] = {"start": start, "size": stop - start}
        if stop - start < 2:
            if batch_flow and not flow_unbatched:
                # One dataset does not fit as one ODE batch: fall back to the
                # approximator's default row chunks for the remaining datasets.
                flow_unbatched = True
                record["retry_rows"] = "default"
                retries.append(record)
                continue
            rows = num_samples if row_cap is None else row_cap
            if not autoregressive or rows < 2:
                raise failed(record, oom)
            # One dataset does not fit as one row batch: shrink the batch.
            row_cap = min(rows - 1, max(1, int(rows * CHUNK_SHRINK)))
            record["retry_rows"] = row_cap
            retries.append(record)
            continue
        size = min(stop - start - 1, max(1, int((stop - start) * CHUNK_SHRINK)))
        record["retry_size"] = size
        retries.append(record)
    return PosteriorDraws(
        draws=draws,
        retries=retries,
        seconds=time.perf_counter() - t0,
        chunk_size=first,
        final_chunk_size=size,
        autoregressive_rows=(
            None
            if not autoregressive
            else "whole chunk"
            if row_cap is None
            else f"at most {row_cap}"
        ),
        chunk_rule=chunk_rule,
    )
