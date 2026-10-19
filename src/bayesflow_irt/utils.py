"""Training helpers and memory-management utilities for variable-size IRT data."""

from __future__ import annotations

import gc
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import keras
import keras.ops as ops
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training-memory helpers
# ---------------------------------------------------------------------------


def is_oom_error(exc: BaseException) -> bool:
    """Return ``True`` if *exc* looks like a backend out-of-memory error.

    Detection is backend-aware: each Keras 3 backend raises a distinct
    exception type for device OOM (``torch.cuda.OutOfMemoryError`` on
    PyTorch, ``tf.errors.ResourceExhaustedError`` on TensorFlow,
    ``jaxlib.xla_extension.XlaRuntimeError`` with a ``RESOURCE_EXHAUSTED``
    status on JAX). Falling back to substring matching on ``str(exc)`` alone
    both over-matches unrelated exceptions and under-matches on backends
    whose message text this package doesn't special-case, so the exception
    type is checked first per active backend, with message-substring
    matching only as a fallback for that backend's known message shapes.
    """
    backend = keras.backend.backend()

    if backend == "torch":
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
        msg = str(exc).lower()
        if isinstance(exc, RuntimeError) and (
            "out of memory" in msg
            or "cudaerrormemoryallocation" in msg
            # CPU allocator: "DefaultCPUAllocator: not enough memory" on
            # Windows, "... can't allocate memory" on Linux.
            or "defaultcpuallocator" in msg
        ):
            return True
        # torch's C++ autograd engine runs backward() on worker threads; a
        # genuine CUDA allocation failure inside a worker thread does not
        # always marshal back into a typed torch.cuda.OutOfMemoryError on the
        # calling thread the way a same-thread CUDA OOM normally does. On
        # Windows this can surface instead as a bare
        # SystemError with CPython's generic "a C API call failed without
        # setting an exception" message -- e.g. hit deep inside a chunked
        # backward recompute (_ChunkedTrainableMatrixPool.backward) under
        # real memory pressure. Narrowly scoped to that exact message (not
        # SystemError in general, which can also indicate an unrelated
        # interpreter-level bug) so this doesn't mask genuine defects as
        # retriable OOMs.
        return isinstance(exc, SystemError) and (
            "error return without exception set" in msg
        )

    if backend == "tensorflow":
        import tensorflow as tf

        return isinstance(exc, tf.errors.ResourceExhaustedError)

    if backend == "jax":
        msg = str(exc).lower()
        return "resource_exhausted" in msg or "out of memory" in msg

    msg = str(exc).lower()
    return "out of memory" in msg or "cudaerrormemoryallocation" in msg


def safe_empty_cache() -> None:
    """Call ``torch.cuda.empty_cache()`` if CUDA is available, swallowing errors."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, RuntimeError) as exc:
        logger.debug("safe_empty_cache skipped: %s", exc)


_cuda_memory_fraction_capped: set[int] = set()

# Physical-safety ceiling for _cap_cuda_memory_fraction (see docstring): just
# under 100% of total device memory, deliberately *not* the same number as
# probe_memory_config's safety_margin (default 0.9). safety_margin is a
# config-selection policy ("only prefer configs whose peak leaves 10%
# headroom, because a single probe isn't representative of a full training
# run") applied on top of real, successful allocations. This constant instead
# governs whether the CUDA allocator itself is even allowed to attempt an
# allocation -- it should only ever intervene on a genuine near-total
# overflow, not silently claw back 10% of usable VRAM from every real
# training step. Left just below 1.0 (rather than exactly 1.0) for CUDA
# context/driver bookkeeping overhead that is not itself "free" memory.
_CUDA_SPILLOVER_GUARD_FRACTION = 0.95


def _cap_cuda_memory_fraction(fraction: float = _CUDA_SPILLOVER_GUARD_FRACTION) -> None:
    """Cap the CUDA caching allocator to *fraction* of physical device memory.

    On Windows, the WDDM driver can silently satisfy an over-budget CUDA
    allocation by spilling into "shared GPU memory" (system RAM over PCIe)
    instead of raising ``torch.cuda.OutOfMemoryError`` -- ``torch.cuda``'s own
    memory bookkeeping (``max_memory_reserved`` etc.) has no visibility into
    this and reports the call as having "succeeded". This is slow (10-100x
    slower than real VRAM), can destabilize the driver, and -- critically for
    :func:`probe_memory_config` -- is not fully released by
    ``torch.cuda.empty_cache()``/``gc.collect()`` afterward within the same
    process, so a single over-budget trial (e.g. an unchunked dense baseline
    at a worst-case tensor shape) can poison every subsequent trial in the
    same probing run, even ones that would otherwise comfortably fit in VRAM.

    ``torch.cuda.set_per_process_memory_fraction`` closes this at the source:
    it makes the *allocator itself* refuse to exceed ``fraction`` of total
    device memory, raising a real (catchable) ``torch.cuda.OutOfMemoryError``
    in milliseconds instead of a slow multi-GB shared-memory allocation. This
    is called once per device -- before :func:`probe_memory_config` runs any
    trials, and again (a no-op by then) before :func:`fit_with_oom_retry`
    starts real training -- using :data:`_CUDA_SPILLOVER_GUARD_FRACTION`
    (~95%), **not** ``safety_margin``: this cap should only ever fire on a
    genuine near-total overflow, so it applies equally during real training
    without permanently sacrificing the 10% headroom that ``safety_margin``
    reserves purely for probe-time config *selection*.

    Idempotent per device index: only the first call actually applies the
    cap on a given device, so whichever of :func:`probe_memory_config` /
    :func:`fit_with_oom_retry` runs first in a process wins. Both call sites
    pass the same fraction, so this is purely a cheap-repeat-call guard, not
    a "stricter caller wins" tie-break.

    Never *loosens* an existing limit: if an embedding application already
    called ``torch.cuda.set_per_process_memory_fraction`` with a stricter
    value than *fraction* (e.g. 0.5, to share one GPU across processes),
    this only ever tightens further, never overrides that down to a looser
    ~95%. ``torch.cuda.get_per_process_memory_fraction`` defaults to 1.0
    (no limit) when nothing has set one yet, so an untouched device is
    always tightened to *fraction* as intended.

    References
    ----------
    PyTorch Team. (2026). *torch.cuda.memory.set_per_process_memory_fraction*
    (documents that exceeding the fraction "will raise an out of memory
    error in allocator").
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return
        device = torch.cuda.current_device()
        if device in _cuda_memory_fraction_capped:
            return
        current_fraction = torch.cuda.get_per_process_memory_fraction(device)
        if fraction < current_fraction:
            torch.cuda.set_per_process_memory_fraction(fraction, device)
        _cuda_memory_fraction_capped.add(device)
    except (ImportError, RuntimeError, AttributeError) as exc:
        logger.debug("_cap_cuda_memory_fraction skipped: %s", exc)


def set_summary_memory_config(
    summary_net: Any,
    item_chunk_size: int | None,
    gradient_checkpointing: bool,
    person_chunk_size: int | None = None,
) -> None:
    """Update the summary-network memory knobs in-place.

    This helper is used by :func:`probe_memory_config` and
    :func:`fit_with_oom_retry` to toggle the summary-side controls exposed by
    the current summary API:

    - ``item_chunk_size`` for summaries that support item-axis chunking;
    - ``gradient_checkpointing`` for summaries that support rematerialization
      on the PyTorch backend;
    - ``person_chunk_size`` for summaries that support person-axis chunking.

    The function also patches the internal ``_config`` dict when present so
    that ``get_config()`` and serialization stay in sync with the live object.
    It does not validate whether a given summary class actually uses these
    settings at runtime.  For example,
    :class:`bayesflow_irt.networks.MatrixEquivariantIRTSummary` stores
    ``item_chunk_size`` for compatibility but ignores it during execution.
    ``person_chunk_size`` is only written when the network already exposes
    that attribute, so calling this on a summary class that predates person
    chunking (no ``person_chunk_size`` attribute at all) is a no-op for that
    argument rather than an error.

    Parameters
    ----------
    summary_net : Any
        Summary network instance to mutate.
    item_chunk_size : int or None
        New item-chunk size to store on the network.
    gradient_checkpointing : bool
        New checkpointing flag to store on the network.
    person_chunk_size : int or None
        New person-chunk size to store on the network. Ignored (left
        untouched) if the network has no ``person_chunk_size`` attribute.
    """
    summary_net.item_chunk_size = item_chunk_size
    summary_net.gradient_checkpointing = gradient_checkpointing
    if hasattr(summary_net, "_config") and isinstance(summary_net._config, dict):
        summary_net._config["item_chunk_size"] = item_chunk_size
        summary_net._config["gradient_checkpointing"] = gradient_checkpointing
    if hasattr(summary_net, "person_chunk_size"):
        summary_net.person_chunk_size = person_chunk_size
        if hasattr(summary_net, "_config") and isinstance(summary_net._config, dict):
            summary_net._config["person_chunk_size"] = person_chunk_size


def _supports_item_chunking(summary_net: Any) -> bool:
    """Return whether ``item_chunk_size`` changes summary-network execution.

    Older/custom summary networks may not expose an explicit capability flag.
    For backward compatibility, the presence of ``item_chunk_size`` still means
    "supports chunking" unless the network opts out with
    ``supports_item_chunking = False``.
    """
    if summary_net is None or not hasattr(summary_net, "item_chunk_size"):
        return False
    return bool(getattr(summary_net, "supports_item_chunking", True))


def _supports_person_chunking(summary_net: Any) -> bool:
    """Return whether ``person_chunk_size`` changes summary-network execution.

    Unlike :func:`_supports_item_chunking`, this defaults to ``False`` when a
    summary network has no explicit ``supports_person_chunking = True`` flag:
    person-axis chunking is new and, unlike
    ``item_chunk_size``, no pre-existing summary class ever relied on a bare
    ``hasattr`` check implying support.

    Beyond the class-level flag, this also re-checks the *instance's*
    eligibility via ``_person_chunking_config_is_eligible()`` when the
    network exposes it. ``MatrixEquivariantIRTSummary.__init__`` only
    validates ``person_pooling == "mean"`` / no cell-level raw features
    against ``person_chunk_size`` at construction time; an instance built
    with ``person_pooling="pma"`` (and ``person_chunk_size=None``, which
    passes that check) would otherwise look "supported" here even though
    ``call()`` will never actually take the chunked path for it. Skipping
    such instances up front avoids wasting probe/retry attempts on a
    person-chunk-size ladder that is silently a no-op for them.
    """
    if summary_net is None or not hasattr(summary_net, "person_chunk_size"):
        return False
    if not bool(getattr(summary_net, "supports_person_chunking", False)):
        return False
    is_eligible = getattr(summary_net, "_person_chunking_config_is_eligible", None)
    if callable(is_eligible):
        return bool(is_eligible())
    return True


# -- Probe data-classes -----------------------------------------------------


@dataclass
class ProbeResult:
    """Result of one ``probe_memory_config`` trial.

    Each trial corresponds to a single ``(item_chunk_size,
    gradient_checkpointing, person_chunk_size)`` combination evaluated at a
    fixed batch size and fixed maximum ``(n_items, n_persons)`` shape.  When
    the trial succeeds, CUDA peak-memory counters are recorded; when it OOMs,
    the trial is marked as unsuccessful and the peak-memory fields remain
    ``None``.
    """

    item_chunk_size: int | None
    gradient_checkpointing: bool
    ok: bool
    peak_allocated_bytes: int | None = None
    """Peak ``torch.cuda.max_memory_allocated`` in bytes (``None`` on OOM)."""
    peak_reserved_bytes: int | None = None
    """Peak ``torch.cuda.max_memory_reserved`` in bytes (``None`` on OOM)."""
    person_chunk_size: int | None = None
    """Person-chunk size tried in this trial (``None`` for summary networks
    that do not support person-axis chunking, or when untried)."""
    reason: str = "ok"
    """Short outcome code: ``"ok"``, ``"oom"`` (CUDA OOM), ``"spill"``
    (WDDM shared-memory spillover -- see :func:`_probe_one`), ``"margin"``
    (exceeded ``safety_margin``), or ``"ckpt"`` (gradient-checkpointing
    incompatible with this layer config). A failure to reset CUDA
    peak-memory counters before a trial is *not* one of these codes -- it
    indicates a CUDA/driver-state problem unrelated to memory feasibility,
    so :func:`_probe_one` raises instead of returning a misleading
    ``ProbeResult`` for it (see :func:`_probe_one`)."""
    elapsed_s: float | None = None
    """Wall-clock seconds spent running this trial's ``train_step``."""
    device_total_bytes: int | None = None
    """Physical VRAM (``torch.cuda.get_device_properties(...).total_memory``)
    read during this trial, when available."""
    device_budget_bytes: int | None = None
    """VRAM actually *available to this process* when the trial started, in
    bytes: ``torch.cuda.mem_get_info()``'s free figure plus whatever this
    process's own caching allocator already held (which it can reuse). This
    is the quantity ``safety_margin`` is enforced against -- see
    :func:`_probe_one`. It is strictly smaller than ``device_total_bytes``
    whenever anything else holds VRAM, which is exactly the case where budgeting
    against physical VRAM overestimates real headroom."""

    @property
    def peak_bytes(self) -> int | None:
        """Backward-compatible alias for peak allocated bytes."""
        return self.peak_allocated_bytes

    @property
    def peak_mib(self) -> float | None:
        return (
            self.peak_allocated_bytes / 2**20
            if self.peak_allocated_bytes is not None
            else None
        )

    @property
    def peak_reserved_mib(self) -> float | None:
        return (
            self.peak_reserved_bytes / 2**20
            if self.peak_reserved_bytes is not None
            else None
        )

    @property
    def peak_reserved_pct(self) -> float | None:
        """Peak reserved memory as a percentage of the usable VRAM budget.

        Reported against ``device_budget_bytes`` (what this process could
        actually get) when available, falling back to ``device_total_bytes``
        (physical VRAM) otherwise. The budget is the meaningful denominator:
        when other processes hold part of the card, a trial can look
        comfortable against physical VRAM while using most of what is
        actually free, and it is the latter that predicts whether real
        training OOMs.
        """
        denominator = self.device_budget_bytes or self.device_total_bytes
        if self.peak_reserved_bytes is None or not denominator:
            return None
        return 100.0 * self.peak_reserved_bytes / denominator


def _format_probe_row(t: ProbeResult, note: str = "") -> str:
    """One console/log line for a single probe trial.

    Shared by the live ``verbose`` prints in :func:`probe_memory_config` /
    :func:`_search_chunk_size_boundary` and by
    :meth:`MemoryProbeReport.summary_lines`, so all three stay in the same
    format instead of drifting apart.
    """
    tag = " OK" if t.ok else "FAIL"
    item = "None" if t.item_chunk_size is None else str(t.item_chunk_size)
    person = "None" if t.person_chunk_size is None else str(t.person_chunk_size)
    cp = "ON" if t.gradient_checkpointing else "OFF"
    alloc = f"{t.peak_mib:>7.0f} MiB" if t.peak_mib is not None else "    n/a"
    reserved = (
        f"{t.peak_reserved_mib:>7.0f} MiB"
        if t.peak_reserved_mib is not None
        else "    n/a"
    )
    pct = f" ({t.peak_reserved_pct:>4.1f}%)" if t.peak_reserved_pct is not None else ""
    elapsed = f"{t.elapsed_s:>5.2f}s" if t.elapsed_s is not None else "   n/a"
    row = (
        f"  [{tag}] item={item:>4}  person={person:>6}  cp={cp:>3}  "
        f"alloc={alloc}  reserved={reserved}{pct}  time={elapsed}"
    )
    if not t.ok:
        row += f"  reason={t.reason}"
    if note:
        row += f"  {note}"
    return row


@dataclass
class MemoryProbeReport:
    """Aggregated result of :func:`probe_memory_config`.

    The report keeps the full set of attempted configurations in ``trials``
    and the first successful one in ``best``.  When ``early_stop=False`` the
    report can be used as a lightweight memory profile across several summary
    settings; when ``early_stop=True`` it acts more like a record of the
    search path that found a viable configuration.
    """

    batch_size: int
    n_items: int
    n_persons: int
    response_dtype: str = "float32"
    estimated_raw_response_bytes: int | None = None
    trials: list[ProbeResult] = field(default_factory=list)
    best: ProbeResult | None = None
    probe_failed: bool = False
    probe_error: str | None = None

    def summary_lines(self) -> list[str]:
        raw_mib = (
            f"{self.estimated_raw_response_bytes / 2**20:.1f} MiB"
            if self.estimated_raw_response_bytes is not None
            else "n/a"
        )
        lines = [
            (
                "Memory probe  "
                f"batch={self.batch_size}  I={self.n_items}  P={self.n_persons}  "
                f"dtype={self.response_dtype}  raw={raw_mib}"
            ),
        ]
        for t in self.trials:
            lines.append(_format_probe_row(t))
        total_s = sum(t.elapsed_s for t in self.trials if t.elapsed_s is not None)
        if self.best is not None:
            chunk = self.best.item_chunk_size
            pchunk = self.best.person_chunk_size
            cp = self.best.gradient_checkpointing
            alloc = (
                f"{self.best.peak_mib:.0f} MiB"
                if self.best.peak_mib is not None
                else "n/a"
            )
            reserv = (
                f"{self.best.peak_reserved_mib:.0f} MiB"
                if self.best.peak_reserved_mib is not None
                else "n/a"
            )
            pct = (
                f" ({self.best.peak_reserved_pct:.1f}%)"
                if self.best.peak_reserved_pct is not None
                else ""
            )
            lines.append(
                f"  -> best: item={chunk}, person={pchunk}, cp={cp}, "
                f"alloc={alloc}, reserved={reserv}{pct}"
            )
        else:
            lines.append("  -> no config succeeded")
        lines.append(
            f"  ({len(self.trials)} trial(s), {total_s:.2f}s total probing time)"
        )
        return lines


@dataclass
class TrainingMemoryConfig:
    """Memory-affecting knobs for large-scale training/sampling.

    This dataclass is a compact record of the runtime configuration applied to
    an approximator after probing or OOM recovery.  It deliberately combines
    summary-side controls, flow-loss controls, and sampling controls so that
    downstream code can persist or report a single coherent memory policy.

    Attributes
    ----------
    item_chunk_size : int or None
        Item-chunk size applied to the summary network.
    gradient_checkpointing : bool
        Whether gradient checkpointing is enabled on the summary network.
    flow_chunk_size : int or None
        Flow-loss chunk size applied to the inference network.
    flow_chunk_backward : str or None
        Backward mode for chunked flow loss (``None`` or ``"checkpoint"``).
    gradient_accumulation_steps : int
        Number of micro-steps accumulated before an optimizer update.
    person_chunk_size : int or None
        Person-chunk size applied to the summary network. ``None`` for summary
        classes that do not support person-axis chunking.
    """

    item_chunk_size: int | None = None
    gradient_checkpointing: bool = False
    flow_chunk_size: int | None = None
    flow_chunk_backward: str | None = None
    gradient_accumulation_steps: int = 1
    person_chunk_size: int | None = None

    def __post_init__(self) -> None:
        if self.item_chunk_size is not None and int(self.item_chunk_size) <= 0:
            raise ValueError("item_chunk_size must be a positive integer or None.")
        if self.flow_chunk_size is not None and int(self.flow_chunk_size) <= 0:
            raise ValueError("flow_chunk_size must be a positive integer or None.")
        if self.flow_chunk_backward == "none":
            self.flow_chunk_backward = None
        if self.flow_chunk_backward not in {None, "checkpoint"}:
            raise ValueError("flow_chunk_backward must be None or 'checkpoint'.")
        if int(self.gradient_accumulation_steps) <= 0:
            raise ValueError("gradient_accumulation_steps must be a positive integer.")
        if self.person_chunk_size is not None and int(self.person_chunk_size) <= 0:
            raise ValueError("person_chunk_size must be a positive integer or None.")


@dataclass
class InferenceProfile:
    """Timing breakdown for one posterior-sampling call.

    This helper is intended for notebook-side profiling of
    ``approximator.sample(...)``. It separates the main stages of the custom
    IRT sampling path:

    - raw-condition adaptation
    - summary-network forward pass
    - flow-based posterior sampling
    - inverse adaptation back to user-facing parameter keys
    """

    num_samples: int
    batch_size: int
    num_items: int
    flat_conditions: int
    sample_chunk_size: int
    condition_chunk_size: int
    summary_batch_size: int | None
    num_sample_chunks: int
    num_condition_chunks: int
    inference_sample_calls: int
    adapter_s: float
    summary_s: float
    flow_sampling_s: float
    inverse_adapter_s: float
    total_profiled_s: float
    output_shapes: dict[str, tuple[int, ...]]

    def summary_lines(self) -> list[str]:
        """Return a compact human-readable breakdown."""
        lines = [
            (
                "Inference profile  "
                f"batch={self.batch_size}  items={self.num_items}  "
                f"draws={self.num_samples}  flat={self.flat_conditions}"
            ),
            (
                "  chunks  "
                f"sample={self.sample_chunk_size} ({self.num_sample_chunks})  "
                f"condition={self.condition_chunk_size} ({self.num_condition_chunks})  "
                f"calls={self.inference_sample_calls}"
            ),
            (
                "  timings  "
                f"adapter={self.adapter_s:.3f}s  "
                f"summary={self.summary_s:.3f}s  "
                f"flow={self.flow_sampling_s:.3f}s  "
                f"inverse={self.inverse_adapter_s:.3f}s  "
                f"total={self.total_profiled_s:.3f}s"
            ),
        ]
        return lines


def _sync_cuda_if_available() -> None:
    """Synchronize CUDA when available so timings reflect actual work."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except (ImportError, RuntimeError):
        return


def _tensor_dim_as_int(x, axis: int) -> int:
    """Resolve one tensor dimension to a concrete Python int."""
    dim = x.shape[axis]
    if dim is not None:
        return int(dim)
    shape = ops.convert_to_numpy(ops.shape(x))
    return int(shape[axis])


def _adapt_conditions_for_inference(approximator, conditions):
    """Run the condition-only adapter path, priming strict mode if needed."""
    try:
        return approximator.adapter(conditions, strict=False)
    except ValueError as exc:
        msg = str(exc)
        if "strict=False" not in msg or "strict=True" not in msg:
            raise
        approximator.adapter(conditions, strict=True)
        return approximator.adapter(conditions, strict=False)


def profile_posterior_inference(
    approximator,
    conditions,
    *,
    num_samples: int,
    sample_chunk_size: int = 64,
    condition_chunk_size: int = 256,
    summary_batch_size: int | None = None,
    flow_sample_kwargs: dict[str, Any] | None = None,
) -> InferenceProfile:
    """Profile posterior sampling for a fitted IRT approximator.

    Parameters
    ----------
    approximator
        A fitted ``EquivariantIRTApproximator`` (or compatible object exposing
        the same adapter / summary / inference interfaces).
    conditions
        Raw simulator output or any condition payload accepted by
        ``approximator.sample(conditions=...)``.
    num_samples : int
        Number of posterior draws per dataset/item.
    sample_chunk_size : int
        Sampling-time chunk size over the draw axis.
    condition_chunk_size : int
        Sampling-time chunk size over the flattened ``batch × items`` axis.
    summary_batch_size : int or None
        Optional batch chunking for the summary-network forward pass.
    flow_sample_kwargs : dict or None
        Extra keyword arguments forwarded to ``inference_network.sample(...)``.
        For ``FlowMatching`` networks, use this to compare solver settings
        such as ``{"method": "euler", "steps": 24}``.

    Returns
    -------
    InferenceProfile
        Timing breakdown plus chunk/call counts and output shapes.
    """
    if getattr(approximator, "autoregressive", False):
        raise NotImplementedError(
            "profile_posterior_inference does not support "
            "item_posterior='autoregressive_coupling_flow'."
        )
    if int(num_samples) <= 0:
        raise ValueError("num_samples must be a positive integer.")
    if int(sample_chunk_size) <= 0:
        raise ValueError("sample_chunk_size must be a positive integer.")
    if int(condition_chunk_size) <= 0:
        raise ValueError("condition_chunk_size must be a positive integer.")
    if summary_batch_size is not None and int(summary_batch_size) <= 0:
        raise ValueError("summary_batch_size must be a positive integer or None.")
    if flow_sample_kwargs is None:
        flow_sample_kwargs = {}
    elif not isinstance(flow_sample_kwargs, dict):
        raise TypeError("flow_sample_kwargs must be a dict or None.")

    sample_chunk_size = int(sample_chunk_size)
    condition_chunk_size = int(condition_chunk_size)
    num_samples = int(num_samples)
    if summary_batch_size is not None:
        summary_batch_size = int(summary_batch_size)

    _sync_cuda_if_available()
    t0 = time.perf_counter()
    adapted = _adapt_conditions_for_inference(approximator, conditions)
    adapted = {k: ops.convert_to_tensor(v) for k, v in adapted.items()}
    _sync_cuda_if_available()
    t1 = time.perf_counter()

    summary_variables = adapted.get("summary_variables")
    inference_conditions = adapted.get("inference_conditions")

    if summary_batch_size is None:
        summary_outputs = approximator.summary_network(summary_variables)
    else:
        total_batch = _tensor_dim_as_int(summary_variables, 0)
        outs = []
        for start in range(0, total_batch, summary_batch_size):
            end = min(start + summary_batch_size, total_batch)
            outs.append(approximator.summary_network(summary_variables[start:end]))
        summary_outputs = ops.concatenate(outs, axis=0)
    _sync_cuda_if_available()
    t2 = time.perf_counter()

    batch_size = _tensor_dim_as_int(summary_outputs, 0)
    num_items = _tensor_dim_as_int(summary_outputs, 1)
    flat_n = batch_size * num_items

    global_condition_dim = int(getattr(approximator, "_global_condition_dim", 0))
    if inference_conditions is None and global_condition_dim > 0:
        inference_conditions = ops.zeros(
            (batch_size, global_condition_dim),
            dtype=keras.ops.dtype(summary_outputs),
        )

    flat_conditions = approximator._flatten_conditions(
        summary_outputs,
        inference_conditions,
        batch_size,
        num_items,
    )

    condition_chunks = []
    inference_sample_calls = 0
    for cond_start in range(0, flat_n, condition_chunk_size):
        n_cond = min(condition_chunk_size, flat_n - cond_start)
        flat_conditions_cond = flat_conditions[cond_start : cond_start + n_cond]

        sample_chunks = []
        for sample_start in range(0, num_samples, sample_chunk_size):
            n_sample = min(sample_chunk_size, num_samples - sample_start)
            flat_conditions_chunk = ops.expand_dims(flat_conditions_cond, axis=1)
            flat_conditions_chunk = ops.broadcast_to(
                flat_conditions_chunk,
                (n_cond, n_sample, ops.shape(flat_conditions_cond)[-1]),
            )
            chunk = approximator.inference_network.sample(
                (n_cond, n_sample),
                conditions=flat_conditions_chunk,
                **flow_sample_kwargs,
            )
            sample_chunks.append(chunk)
            inference_sample_calls += 1

        condition_chunks.append(ops.concatenate(sample_chunks, axis=1))

    samples = ops.concatenate(condition_chunks, axis=0)
    param_dim = _tensor_dim_as_int(samples, -1)

    if getattr(approximator, "standardize_params", False):
        flat_samples = ops.reshape(samples, (-1, param_dim))
        flat_samples = approximator._flat_standardize(
            flat_samples,
            stage="inference",
            forward=False,
        )
        samples = ops.reshape(flat_samples, (flat_n, num_samples, param_dim))

    samples = ops.reshape(samples, (batch_size, num_items, num_samples, param_dim))
    samples = ops.transpose(samples, axes=(0, 2, 1, 3))
    _sync_cuda_if_available()
    t3 = time.perf_counter()

    samples_np = ops.convert_to_numpy(samples)
    out = approximator.adapter(
        {"inference_variables": samples_np},
        inverse=True,
        strict=False,
    )

    is_loading = (
        getattr(approximator.adapter, "_bayesflow_irt_internal_parameterization", None)
        == "loading"
    )
    if (
        is_loading
        and isinstance(out, dict)
        and "d" not in out
        and "a" in out
        and "b" in out
    ):
        out = dict(out)
        out["d"] = -(np.asarray(out["a"]) * np.asarray(out["b"]))

    _sync_cuda_if_available()
    t4 = time.perf_counter()

    output_shapes = {k: tuple(np.asarray(v).shape) for k, v in out.items()}
    num_condition_chunks = int(np.ceil(flat_n / condition_chunk_size))
    num_sample_chunks = int(np.ceil(num_samples / sample_chunk_size))

    return InferenceProfile(
        num_samples=num_samples,
        batch_size=batch_size,
        num_items=num_items,
        flat_conditions=flat_n,
        sample_chunk_size=sample_chunk_size,
        condition_chunk_size=condition_chunk_size,
        summary_batch_size=summary_batch_size,
        num_sample_chunks=num_sample_chunks,
        num_condition_chunks=num_condition_chunks,
        inference_sample_calls=inference_sample_calls,
        adapter_s=t1 - t0,
        summary_s=t2 - t1,
        flow_sampling_s=t3 - t2,
        inverse_adapter_s=t4 - t3,
        total_profiled_s=t4 - t0,
        output_shapes=output_shapes,
    )


# -- Probe implementation ---------------------------------------------------


def _build_probe_approximator(approximator):
    """Create a disposable approximator clone for memory probing.

    Reads architecture config from *approximator* (read-only) and builds a
    **new** ``EquivariantIRTApproximator`` with fresh random weights and a
    throwaway Adam optimiser. The adapter is shared (it is stateless).

    Parameters
    ----------
    approximator
        The live approximator whose architecture to clone.

    Returns
    -------
    EquivariantIRTApproximator
        A freshly constructed, compiled clone.  Caller is responsible for
        deleting it after use.
    """
    from .approximators import EquivariantIRTApproximator

    # Clone whichever summary-network class is attached to the approximator.
    summary_net = approximator.summary_network
    clone_summary = type(summary_net).from_config(summary_net.get_config())

    # Clone inference network via Keras serialisation round-trip.
    inf_net = approximator.inference_network
    clone_inference = type(inf_net).from_config(inf_net.get_config())

    # Build a new approximator sharing the (stateless) adapter.
    clone = EquivariantIRTApproximator(
        adapter=approximator.adapter,
        inference_network=clone_inference,
        summary_network=clone_summary,
        standardize_params=getattr(approximator, "standardize_params", False),
        flow_chunk_size=getattr(approximator, "flow_chunk_size", None),
        flow_chunk_backward=getattr(approximator, "flow_chunk_backward", None),
        gradient_accumulation_steps=getattr(
            approximator, "gradient_accumulation_steps", 1
        ),
        item_posterior=getattr(approximator, "item_posterior", "flow_matching"),
        draws_summary_pooling=getattr(approximator, "draws_summary_pooling", "mean"),
        draws_summary_hidden_dim=getattr(approximator, "draws_summary_hidden_dim", 32),
        draws_summary_dim=getattr(approximator, "draws_summary_dim", 32),
        draws_summary_num_heads=getattr(approximator, "draws_summary_num_heads", 4),
    )
    clone.compile(optimizer="adam")
    return clone


def _probe_one(
    approximator,
    make_simulator,
    batch_size: int,
    item_chunk_size: int | None,
    gradient_checkpointing: bool,
    n_items: int,
    n_persons: int,
    safety_margin: float = 0.9,
    person_chunk_size: int | None = None,
) -> ProbeResult:
    """Run a single train-step probe and return a :class:`ProbeResult`.

    Parameters
    ----------
    safety_margin : float
        A config is only accepted if ``peak_reserved <= safety_margin *
        device_budget``, where ``device_budget`` is the VRAM actually
        available to this process at the start of the trial
        (``torch.cuda.mem_get_info()``'s free figure plus this process's
        own already-reserved pool, which it can reuse) -- *not* the card's
        physical VRAM. Anything else holding VRAM (on Windows, the desktop
        compositor) shrinks the real budget, and it is what is actually
        obtainable that predicts whether real training OOMs. Budgeting
        against physical VRAM therefore silently spends the entire safety
        margin before the first trial runs. A single probed ``train_step``
        is not a reliable stand-in for thousands of variable-shape steps in
        real training: a config probed with only a few percent headroom can
        OOM a few epochs into training, while the same architecture with
        ample headroom trains without issue. Requiring headroom up front
        catches this before a long run wastes hours on partial epochs and
        OOM/retry cycles. Default 0.9 (10% headroom); tune down only if you
        have verified your workload's actual real-training peak stays within
        a smaller margin.
    """
    try:
        import torch
    except ImportError as exc:
        raise NotImplementedError(
            "Memory probing (item_chunk_size / gradient_checkpointing grid "
            "search) only supports the torch backend today; PyTorch is not "
            "installed in this environment."
        ) from exc

    summary_net = approximator.summary_network
    set_summary_memory_config(
        summary_net,
        item_chunk_size,
        gradient_checkpointing,
        person_chunk_size=person_chunk_size,
    )
    start_time = time.perf_counter()

    def _device_total_bytes() -> int | None:
        # Best-effort only: called from paths where CUDA may be in a bad
        # state (post-OOM, failed peak-stats reset), so a query failure here
        # must not mask the real outcome of this trial. Narrowed to the
        # exception types this call is actually known to raise in that
        # state (matching safe_empty_cache/_cap_cuda_memory_fraction
        # elsewhere in this module) so an unrelated bug doesn't silently
        # show up as a permanently missing device_total_bytes.
        try:
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            return props.total_memory
        except (RuntimeError, AttributeError) as exc:
            logger.debug("Could not query CUDA device total memory: %s", exc)
            return None

    def _device_budget_bytes() -> int | None:
        """VRAM obtainable by this process right now, in bytes.

        ``mem_get_info()``'s free figure excludes memory held by *any*
        process -- including this one's caching-allocator pool, which this
        process can freely reuse -- so that pool is added back. Best-effort
        for the same reason as ``_device_total_bytes``: a query failure
        here must degrade to "unknown budget" (callers fall back to
        physical VRAM) rather than mask the trial's real outcome.
        """
        try:
            free_bytes, _total = torch.cuda.mem_get_info()
            return int(free_bytes) + int(torch.cuda.memory_reserved())
        except (RuntimeError, AttributeError) as exc:
            logger.debug("Could not query CUDA free memory: %s", exc)
            return None

    try:
        if hasattr(make_simulator, "clone_with_sizes"):
            sim = make_simulator.clone_with_sizes(n_items=n_items, n_persons=n_persons)
        else:
            sim = make_simulator(n_items=n_items, n_persons=n_persons)
        dummy = approximator.adapter(sim.sample((batch_size,)))

        # Ensure the disposable clone is explicitly built from concrete shapes
        # before the first optimizer step. This avoids backend-specific dynamic
        # shape paths inside FlowMatching that can fail during nested subnet
        # construction (e.g., Dense seeing tuple-valued input dims).
        if not approximator.built:
            data_shapes = {
                k: tuple(int(d) for d in v.shape)
                for k, v in dummy.items()
                if hasattr(v, "shape")
            }
            approximator.build(data_shapes)

        safe_empty_cache()
        try:
            torch.cuda.reset_peak_memory_stats()
        except (RuntimeError, AttributeError) as exc:
            # A failed reset leaves stale peak-memory counters from a prior
            # trial in place. Reading them afterward would attribute another
            # trial's peak usage to this one, potentially selecting a config
            # that then OOMs in real training. Retry once (a transient
            # driver hiccup can clear on its own) before giving up.
            logger.warning(
                "Could not reset CUDA peak memory stats before probe "
                "(chunk=%s, cp=%s); retrying once: %s",
                item_chunk_size,
                gradient_checkpointing,
                exc,
            )
            gc.collect()
            safe_empty_cache()
            try:
                torch.cuda.reset_peak_memory_stats()
            except (RuntimeError, AttributeError) as exc2:
                # This is a CUDA/driver-state problem, not a memory-
                # feasibility result. Reporting it as a failed ProbeResult
                # would make callers (e.g. _search_chunk_size_boundary,
                # probe_memory_config) treat it identically to a genuine
                # OOM/infeasible config -- silently narrowing the search
                # boundary or reporting "no valid configuration (OOM or
                # unsupported)" for what is actually an infrastructure
                # failure unrelated to any config's memory footprint.
                # Raise instead so the probe aborts with a clear cause.
                raise RuntimeError(
                    "Could not reset CUDA peak-memory stats before probing "
                    f"(item_chunk_size={item_chunk_size}, "
                    f"gradient_checkpointing={gradient_checkpointing}, "
                    f"person_chunk_size={person_chunk_size}) even after "
                    "retrying. This indicates a CUDA/driver-state problem, "
                    "not that this configuration is infeasible."
                ) from exc2

        # Sampled here, after the cache flush above and before the measured
        # step, so it reflects the memory this trial could actually draw on
        # (rather than a figure polluted by this trial's own peak, which is
        # what reading it afterward would give).
        device_budget = _device_budget_bytes()

        _ = approximator.train_step(dummy)
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()

        del dummy, sim
        gc.collect()
        safe_empty_cache()

        device_total = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).total_memory
        if peak_reserved > device_total:
            # No exception was raised, but peak reserved memory exceeds this
            # device's actual VRAM. On Windows, the WDDM driver can silently
            # spill CUDA allocations into "shared GPU memory" (system RAM
            # accessed over PCIe) instead of raising
            # torch.cuda.OutOfMemoryError -- so train_step() "succeeds" here,
            # but real training with this config would keep paging over PCIe,
            # which is on the order of 10-100x slower than genuine VRAM and
            # can eventually destabilize the driver. Treat this as a failed
            # trial so a smaller/checkpointed config that actually fits in
            # VRAM is preferred instead.
            logger.warning(
                "Probe config (chunk=%s, cp=%s) reserved %.0f MiB, which exceeds "
                "this device's %.0f MiB of physical VRAM. No CUDA OOM was "
                "raised, which usually means the driver silently fell back to "
                "slower shared/system memory (a known Windows WDDM behavior) "
                "-- treating this configuration as unusable.",
                item_chunk_size,
                gradient_checkpointing,
                peak_reserved / 2**20,
                device_total / 2**20,
            )
            return ProbeResult(
                item_chunk_size,
                gradient_checkpointing,
                ok=False,
                peak_allocated_bytes=peak_allocated,
                peak_reserved_bytes=peak_reserved,
                person_chunk_size=person_chunk_size,
                reason="spill",
                elapsed_s=time.perf_counter() - start_time,
                device_total_bytes=device_total,
                device_budget_bytes=device_budget,
            )

        # Enforce the margin against what this process could actually
        # obtain, not the card's physical VRAM -- see the safety_margin
        # docstring. Falls back to physical VRAM only when the free-memory
        # query failed.
        budget = device_budget or device_total
        margin_limit = safety_margin * budget
        if peak_reserved > margin_limit:
            # Technically under physical VRAM, but with too little headroom.
            # A single probed train_step is not representative of thousands
            # of variable-shape steps in real training -- see the
            # safety_margin docstring above.
            logger.warning(
                "Probe config (chunk=%s, cp=%s) reserved %.0f MiB, which is "
                "within %.0f MiB physical VRAM but exceeds the %.0f%% safety "
                "margin (%.0f MiB) on the %.0f MiB actually available to this "
                "process. A single probe step is not a reliable predictor of a "
                "full variable-shape training run at this margin -- treating "
                "this configuration as unusable. Reduce batch_size, enable "
                "person-axis chunking, free VRAM held by other processes, or "
                "lower safety_margin if you have verified this workload's real "
                "training peak stays within a thinner margin.",
                item_chunk_size,
                gradient_checkpointing,
                peak_reserved / 2**20,
                device_total / 2**20,
                safety_margin * 100,
                margin_limit / 2**20,
                budget / 2**20,
            )
            return ProbeResult(
                item_chunk_size,
                gradient_checkpointing,
                ok=False,
                peak_allocated_bytes=peak_allocated,
                peak_reserved_bytes=peak_reserved,
                person_chunk_size=person_chunk_size,
                reason="margin",
                elapsed_s=time.perf_counter() - start_time,
                device_total_bytes=device_total,
                device_budget_bytes=device_budget,
            )

        return ProbeResult(
            item_chunk_size,
            gradient_checkpointing,
            ok=True,
            peak_allocated_bytes=peak_allocated,
            peak_reserved_bytes=peak_reserved,
            person_chunk_size=person_chunk_size,
            elapsed_s=time.perf_counter() - start_time,
            device_total_bytes=device_total,
            device_budget_bytes=device_budget,
        )

    except Exception as exc:
        gc.collect()
        safe_empty_cache()
        if is_oom_error(exc):
            return ProbeResult(
                item_chunk_size,
                gradient_checkpointing,
                ok=False,
                person_chunk_size=person_chunk_size,
                reason="oom",
                elapsed_s=time.perf_counter() - start_time,
                device_total_bytes=_device_total_bytes(),
                device_budget_bytes=_device_budget_bytes(),
            )
        # CheckpointError: gradient checkpointing is incompatible with this
        # layer configuration — treat as a failed (unusable) probe config.
        try:
            from torch.utils.checkpoint import CheckpointError  # torch ≥ 2.1

            if isinstance(exc, CheckpointError):
                logger.debug(
                    "Probe config (chunk=%s, cp=%s) raised CheckpointError; "
                    "marking as failed.",
                    item_chunk_size,
                    gradient_checkpointing,
                )
                return ProbeResult(
                    item_chunk_size,
                    gradient_checkpointing,
                    ok=False,
                    person_chunk_size=person_chunk_size,
                    reason="ckpt",
                    elapsed_s=time.perf_counter() - start_time,
                    device_total_bytes=_device_total_bytes(),
                    device_budget_bytes=_device_budget_bytes(),
                )
        except ImportError:
            pass
        raise


def _resolve_search_tolerance(tolerance: int | float, ceiling: int) -> int:
    """Resolve an absolute or relative boundary-search tolerance to an int.

    An ``int`` is an absolute chunk-size tolerance (used as-is, minimum 1).
    A ``float`` is interpreted as a fraction of ``ceiling`` (must lie in
    ``(0, 1)``), converted to an absolute count and floored at 1 -- e.g.
    ``tolerance=0.05`` on ``ceiling=5000`` stops once the bracket is within
    250 of the true boundary, instead of narrowing all the way to an exact
    single-integer boundary.
    """
    if isinstance(tolerance, float):
        if not 0.0 < tolerance < 1.0:
            raise ValueError(
                "A float search_tolerance is a fraction of the search "
                f"ceiling and must lie in (0, 1); got {tolerance}."
            )
        return max(1, int(round(tolerance * ceiling)))
    if int(tolerance) < 1:
        raise ValueError(f"An int search_tolerance must be >= 1; got {tolerance}.")
    return int(tolerance)


def _search_chunk_size_boundary(
    clone,
    make_simulator,
    batch_size: int,
    gradient_checkpointing: bool,
    n_items_spec,
    n_persons_spec,
    safety_margin: float,
    axis: str,
    ceiling: int,
    floor: int = 8,
    tolerance: int | float = 1,
    verbose: bool = True,
) -> tuple[list[ProbeResult], ProbeResult | None]:
    """Binary search for the largest feasible chunk size.

    Finds the largest ``item_chunk_size`` (``axis="item"``) or
    ``person_chunk_size`` (``axis="person"``) that keeps a single real
    ``train_step`` within ``safety_margin`` of physical VRAM, in
    ``O(log2(ceiling - floor))`` probe trials rather than the fixed
    geometric ladder's coarser power-of-``backoff_factor`` rungs (see
    ``probe_memory_config``'s ``backoff_factor``). This assumes feasibility
    is monotonic in chunk size -- true by construction, since both item- and
    person-axis chunking bound peak memory to ``O(chunk_size)`` regardless
    of the full batch/person extent.

    Algorithm: first confirm ``floor`` itself fits (if not, nothing will).
    Then binary-search directly between ``floor`` (known-feasible) and
    ``ceiling`` (assumed infeasible -- see the ``ceiling`` parameter below)
    until the bracket width is within ``tolerance`` of the true boundary.
    It does not first double up from ``floor``
    (``floor, 2*floor, 4*floor, ...``) to bracket the boundary before
    bisecting: that phase is redundant once ``ceiling`` can be trusted as
    an infeasible upper bound, and it would waste real ``train_step`` trials
    (usually several, each as expensive as a bisection trial) whenever the
    true boundary sits closer to ``ceiling`` than to ``floor``, as when the
    dense baseline OOMs only at the largest sizes.

    Parameters
    ----------
    ceiling : int
        Upper bound for the chunk size (``batch_size`` for the item axis,
        the worst-case ``n_persons`` for the person axis), and the upper
        end of the search bracket. The caller is required to have already
        probed this axis's "no chunking" (``None``) dense candidate and
        confirmed it OOMs before calling this function (see the call sites
        in :func:`probe_memory_config`) -- chunking at
        ``chunk_size == ceiling`` processes the exact same single
        full-extent chunk as no chunking, so that dense failure is reused
        here as proof ``ceiling`` itself is infeasible, without spending a
        further probe trial to re-derive it.
    floor : int
        Smallest chunk size to consider (default 8, matching the floor used
        elsewhere in this module's chunk ladders).
    tolerance : int or float
        How precisely to locate the boundary -- see
        :func:`_resolve_search_tolerance`. Default 1 (exact integer
        boundary). Raising this trades a smaller (more conservative) final
        chunk size for fewer real ``train_step`` probe trials, which matters
        most at large ``ceiling`` (e.g. ``n_persons`` in the thousands),
        where locating the exact boundary costs several extra trials for
        precision beyond what real variable-shape training needs.

    Returns
    -------
    trials : list[ProbeResult]
        Every trial run during the search, in execution order.
    best : ProbeResult or None
        The largest feasible chunk size found, or ``None`` if even
        ``floor`` OOMs.
    """
    abs_tolerance = _resolve_search_tolerance(tolerance, ceiling)
    trials: list[ProbeResult] = []

    def probe(
        cs: int, phase: str, bracket: tuple[int, int] | None = None
    ) -> ProbeResult:
        item_cs = cs if axis == "item" else None
        person_cs = cs if axis == "person" else None
        result = _probe_one(
            clone,
            make_simulator,
            batch_size,
            item_cs,
            gradient_checkpointing,
            n_items_spec,
            n_persons_spec,
            safety_margin=safety_margin,
            person_chunk_size=person_cs,
        )
        trials.append(result)
        if verbose:
            note = f"[search:{axis} {phase}"
            note += f" lo={bracket[0]} hi={bracket[1]}]" if bracket else "]"
            print(_format_probe_row(result, note=note))
        return result

    floor = min(floor, ceiling)
    floor_result = probe(floor, "floor")
    if not floor_result.ok:
        return trials, None
    if floor == ceiling:
        # Nothing left to bracket: floor is the only candidate in range.
        return trials, floor_result

    lo, lo_result = floor, floor_result
    hi = ceiling

    while hi - lo > abs_tolerance:
        mid = (lo + hi) // 2
        result = probe(mid, "bisect", bracket=(lo, hi))
        if result.ok:
            lo, lo_result = mid, result
        else:
            hi = mid

    return trials, lo_result


@dataclass
class ChunkSearchOptions:
    """Fixed-candidate-grid knobs for :func:`probe_memory_config`.

    These only matter for the fixed-grid scan fallback: an explicit
    ``candidate_item_chunk_sizes`` / ``candidate_person_chunk_sizes`` list,
    or ``early_stop=False`` (full peak-memory profiling across every grid
    point). When neither applies, the supported chunking axis instead runs
    the binary boundary search (see
    :func:`probe_memory_config`'s "Search strategy" note and
    :func:`_search_chunk_size_boundary`), which needs only
    ``search_tolerance`` and the axis being searched -- none of these
    fields.
    """

    candidate_item_chunk_sizes: list[int | None] | None = None
    """Item-chunk sizes to try (``None`` = no chunking). Default builds a
    sequence from *batch_size* down to 8, multiplying by ``backoff_factor``
    at each step (a power-of-2 sequence when ``backoff_factor=0.5``, the
    default)."""
    candidate_person_chunk_sizes: list[int | None] | None = None
    """Person-chunk sizes to try (``None`` = no chunking). Default builds a
    sequence from the worst-case ``n_persons`` down to 8, multiplying by
    ``backoff_factor`` at each step, mirroring
    ``candidate_item_chunk_sizes``'s default. Ignored (collapsed to
    ``[None]``) for summary classes that do not support person-axis
    chunking."""
    checkpoint_options: list[bool] | None = None
    """Gradient-checkpointing flags to try. Default ``[False, True]``."""
    backoff_factor: float = 0.5
    """Multiplicative factor applied at each step when building the default
    ``candidate_item_chunk_sizes`` / ``candidate_person_chunk_sizes`` grids
    (default 0.5, i.e. halving). Must lie in ``(0, 1]`` -- a smaller value
    steps down faster. Matches the parameter of the same name on
    :func:`fit_with_oom_retry`, so a probed grid and a real-training retry
    ladder step down at the same rate when both are left at their
    defaults. Ignored when the corresponding ``candidate_*`` list is
    passed explicitly."""


def probe_memory_config(
    approximator,
    make_simulator,
    *,
    batch_size: int,
    n_items: int | tuple[int, int],
    n_persons: int | tuple[int, int],
    search: ChunkSearchOptions | None = None,
    search_tolerance: int | float = 1,
    early_stop: bool = True,
    force_fallback: bool = False,
    safety_margin: float = 0.9,
    verbose: bool = True,
) -> MemoryProbeReport:
    """Probe summary-side memory settings for a target batch size.

    The function evaluates a grid over ``item_chunk_size``,
    ``gradient_checkpointing``, and ``person_chunk_size`` at a fixed
    worst-case tensor shape and records which combinations fit into GPU
    memory.  Probing runs on a **disposable clone** of *approximator*, so the
    original weights, optimizer state, and training history are left
    untouched.  Only after probing completes is the best configuration
    copied back onto the original summary network.

    For summary classes that do not support item-axis chunking, the chunk-size
    grid is collapsed to ``[None]`` automatically.  This matters for
    :class:`bayesflow_irt.networks.MatrixEquivariantIRTSummary`, where
    ``gradient_checkpointing`` and ``person_chunk_size`` are the operative
    summary-side knobs and ``item_chunk_size`` is compatibility-only
    metadata. Symmetrically, ``person_chunk_size`` is collapsed to ``[None]``
    for summary classes that do not support person-axis chunking (see
    :func:`_supports_person_chunking`), which today is every class except
    :class:`bayesflow_irt.networks.MatrixEquivariantIRTSummary`.

    **Search strategy.** When ``early_stop=True`` (the default) and the
    relevant ``search.candidate_*`` list is left at its default (``None``),
    the supported chunking axis is *not* scanned as a fixed candidate grid.
    Instead, dense (``None``, no chunking) is probed first as usual, and if
    that OOMs, :func:`_search_chunk_size_boundary` reuses that failure as
    proof the chunk-size ceiling itself is infeasible and binary-searches
    directly between a known-feasible floor and that ceiling -- finding the
    largest feasible chunk size in ``O(log2(n))`` real probe trials, fewer
    and more precise than the fixed power-of-``backoff_factor`` ladder
    below.
    The fixed-grid scan (governed by ``search``'s
    ``candidate_item_chunk_sizes`` / ``candidate_person_chunk_sizes`` /
    ``backoff_factor``) is still used whenever the caller passes an explicit
    candidate list for that axis, or when ``early_stop=False`` (which asks
    for full peak-memory data across every grid point, a request the
    boundary search can't satisfy since it deliberately skips most of the
    range).

    Parameters
    ----------
    approximator
        A compiled ``EquivariantIRTApproximator`` (or compatible).
        Used as an architecture template only — its weights and optimiser
        are never touched.
    make_simulator
        Factory callable ``make_simulator(n_items=..., n_persons=...)``
        returning a BayesFlow simulator, **or** a simulator instance that
        exposes a ``clone_with_sizes(n_items, n_persons)`` method (e.g. the
        training ``simulator`` created by :func:`make_2pl_simulator`).
    batch_size : int
        Target training batch size to test.
    n_items, n_persons : int or (int, int)
        Worst-case item/person counts to probe with.  If a ``(lo, hi)`` range
        is passed (as used for training), the maximum value is used for every
        trial so that peak-memory comparisons are meaningful and reproducible
        across configurations.
    search : ChunkSearchOptions, optional
        Fixed-candidate-grid knobs (``candidate_item_chunk_sizes``,
        ``candidate_person_chunk_sizes``, ``checkpoint_options``,
        ``backoff_factor``) -- see :class:`ChunkSearchOptions`. Defaults to
        ``ChunkSearchOptions()`` (all fields at their own defaults) when
        omitted. These only matter for the fixed-grid fallback path (see
        "Search strategy" above); the binary boundary search that normally
        runs instead only needs ``search_tolerance`` below.
    search_tolerance : int or float
        How precisely the binary boundary search (see
        "Search strategy" above and :func:`_search_chunk_size_boundary`)
        needs to locate the largest feasible chunk size. An ``int`` is an
        absolute chunk-size tolerance; a ``float`` in ``(0, 1)`` is a
        fraction of the search ceiling (``batch_size`` for the item axis,
        the worst-case ``n_persons`` for the person axis) -- e.g.
        ``search_tolerance=0.05`` on ``n_persons=5000`` stops once the
        bracket is within 250 of the true boundary. Default 1 (exact
        integer boundary). The result is always a *feasible* (probed ``ok``)
        chunk size regardless of tolerance -- a looser tolerance only makes
        it more conservative (smaller) than the true boundary, trading
        precision for fewer real ``train_step`` probe trials. Ignored when
        the fixed-grid scan runs instead (explicit candidate list or
        ``early_stop=False``).
    early_stop : bool
        If ``True`` (default), stop probing as soon as the first working
        config is found.  Set to ``False`` to run the full grid and collect
        peak-memory data for every combination.
    force_fallback : bool
        If ``True``, apply an emergency fallback config when probing finds no
        viable option (or when probe setup fails). If ``False`` (default),
        raise ``RuntimeError`` in these cases.
    safety_margin : float
        Fraction of physical VRAM a trial's peak reserved memory must stay
        under to be accepted (default 0.9, i.e. 10% headroom). See
        :func:`_probe_one` for the justification: a single probed step with
        little headroom can OOM a few epochs into real variable-shape
        training.  If
        this rejects every configuration, reduce ``batch_size`` and re-probe
        rather than lowering this value, unless you have separately verified
        your workload's real peak stays within a thinner margin.
    verbose : bool
        If ``True``, print each trial result immediately as it completes.

    Returns
    -------
    MemoryProbeReport
        Report containing every attempted configuration and the first
        successful configuration in ``best``.

    Raises
    ------
    RuntimeError
        If the probe clone cannot be built or if all configurations fail and
        ``force_fallback=False``.

    Notes
    -----
    Successful probing mutates ``approximator.summary_network`` by applying
    the chosen ``item_chunk_size`` / ``gradient_checkpointing`` pair.
    Nothing else on the original approximator is modified.

    References
    ----------
    Chen, T., Xu, B., Zhang, C., & Guestrin, C. (2016). *Training Deep Nets
    with Sublinear Memory Cost*.
    PyTorch Team. (2026). *torch.utils.checkpoint*.
    """
    try:
        import torch
    except ImportError as exc:
        raise NotImplementedError(
            "probe_memory_config only supports the torch backend today "
            "(item_chunk_size / gradient_checkpointing probing relies on "
            "torch.cuda memory stats); PyTorch is not installed in this "
            "environment."
        ) from exc

    if search is None:
        search = ChunkSearchOptions()
    candidate_item_chunk_sizes = search.candidate_item_chunk_sizes
    candidate_person_chunk_sizes = search.candidate_person_chunk_sizes
    checkpoint_options = search.checkpoint_options
    backoff_factor = search.backoff_factor

    if not 0.0 < backoff_factor <= 1.0:
        raise ValueError(
            f"backoff_factor must be a fraction in (0, 1] (e.g. 0.5 to halve "
            f"each step; 1.0 is a no-op); got {backoff_factor}. For a "
            "divisor-style value, invert it (2 -> 0.5, 4 -> 0.25)."
        )

    # Resolve worst-case sizes: if a (lo, hi) range is passed use hi so that
    # every trial probes the same (maximum) tensor shape.
    probe_items = (
        int(max(n_items)) if isinstance(n_items, (tuple, list)) else int(n_items)
    )
    probe_persons = (
        int(max(n_persons)) if isinstance(n_persons, (tuple, list)) else int(n_persons)
    )

    response_dtype = getattr(make_simulator, "response_dtype", "float32")
    try:
        dtype_name = str(response_dtype.name)  # np.dtype
    except AttributeError:
        dtype_name = str(response_dtype)
    try:
        itemsize = np.dtype(dtype_name).itemsize
    except TypeError:
        logger.warning(
            "Unknown response dtype %r during memory probe; using itemsize=4.",
            dtype_name,
        )
        itemsize = 4
    estimated_raw_bytes = (
        int(batch_size) * int(probe_items) * int(probe_persons) * itemsize
    )
    report = MemoryProbeReport(
        batch_size,
        probe_items,
        probe_persons,
        response_dtype=dtype_name,
        estimated_raw_response_bytes=estimated_raw_bytes,
    )

    if not torch.cuda.is_available():
        logger.info("CUDA unavailable — skipping memory probe.")
        return report

    _cap_cuda_memory_fraction()

    # Pin the simulator to exactly (max, max) so _draw_sizes always returns max.
    probe_items_spec = (probe_items, probe_items)
    probe_persons_spec = (probe_persons, probe_persons)

    # Captured before the defaulting logic below overwrites these to a
    # concrete list, so the search-vs-fixed-grid branch (below) can tell
    # "caller left this at the default" apart from "caller explicitly asked
    # for this exact list".
    item_chunks_were_default = candidate_item_chunk_sizes is None
    person_chunks_were_default = candidate_person_chunk_sizes is None

    if candidate_item_chunk_sizes is None:
        candidate_item_chunk_sizes = [None]
        item_cs = int(batch_size)
        while item_cs >= 1:
            candidate_item_chunk_sizes.append(item_cs)
            if item_cs <= 8:
                break
            next_item_cs = int(item_cs * backoff_factor)
            item_cs = next_item_cs if next_item_cs < item_cs else item_cs - 1
        # deduplicate while preserving order
        candidate_item_chunk_sizes = list(dict.fromkeys(candidate_item_chunk_sizes))

    if candidate_person_chunk_sizes is None:
        candidate_person_chunk_sizes = [None]
        pcs = int(probe_persons)
        while pcs >= 1:
            candidate_person_chunk_sizes.append(pcs)
            if pcs <= 8:
                break
            next_pcs = int(pcs * backoff_factor)
            pcs = next_pcs if next_pcs < pcs else pcs - 1
        candidate_person_chunk_sizes = list(dict.fromkeys(candidate_person_chunk_sizes))

    summary_template = getattr(approximator, "summary_network", None)
    supports_chunking = _supports_item_chunking(summary_template)
    if not supports_chunking:
        ignored = [
            item_cs for item_cs in candidate_item_chunk_sizes if item_cs is not None
        ]
        if ignored:
            logger.info(
                "Summary network does not support item chunking; "
                "ignoring candidate chunk sizes %s.",
                ignored,
            )
        candidate_item_chunk_sizes = [None]

    supports_person_chunking = _supports_person_chunking(summary_template)
    if not supports_person_chunking:
        ignored_person = [
            pcs for pcs in candidate_person_chunk_sizes if pcs is not None
        ]
        if ignored_person:
            logger.info(
                "Summary network does not support person chunking; "
                "ignoring candidate person chunk sizes %s.",
                ignored_person,
            )
        candidate_person_chunk_sizes = [None]

    # Use the binary boundary search (see
    # _search_chunk_size_boundary) instead of the fixed candidate grid when
    # the caller left the relevant axis at its default and early_stop=True
    # (see the "Search strategy" note above). At most one of these is ever
    # True in practice: item and person chunking are mutually exclusive on
    # every summary class today, so whichever axis is unsupported was just
    # collapsed to [None] above and its "were_default" flag no longer
    # matters. If a future summary class supported both simultaneously, the
    # item axis would take precedence and the person axis would be pinned
    # to None (no chunking) for that call, not run through either search
    # strategy -- this branch only ever probes one chunking axis per call.
    use_item_search = item_chunks_were_default and supports_chunking and early_stop
    use_person_search = (
        person_chunks_were_default
        and supports_person_chunking
        and early_stop
        and not use_item_search
    )

    if checkpoint_options is None:
        checkpoint_options = [False, True]

    probe_start_time = time.perf_counter()
    if verbose:
        print(f"Memory probe  batch={batch_size}  I={probe_items}  P={probe_persons}")
        # Make the margin's denominator visible: reserved-% figures below are
        # against the usable budget, which on a card also driving a display
        # is materially smaller than physical VRAM.
        try:
            import torch as _torch

            _free, _total = _torch.cuda.mem_get_info()
            _held = _total - _free
            print(
                f"  VRAM  total={_total / 2**20:.0f} MiB  "
                f"available={_free / 2**20:.0f} MiB  "
                f"held by other processes={_held / 2**20:.0f} MiB  "
                f"(safety_margin={safety_margin:.0%} applies to available)"
            )
        except Exception:  # pragma: no cover - purely informational
            pass

    # Build a disposable clone — the original is never touched.
    try:
        clone = _build_probe_approximator(approximator)
    except Exception as exc:
        report.probe_failed = True
        report.probe_error = f"clone failed: {exc}"
        if not force_fallback:
            raise RuntimeError(
                "Memory probe failed before running any trial "
                f"({report.probe_error}). Re-run with force_fallback=True "
                "to apply an emergency config."
            ) from exc
        logger.warning(
            "Could not build probe approximator (%s). "
            "Applying fallback because force_fallback=True.",
            report.probe_error,
        )
        fallback_chunk = (
            8 if _supports_item_chunking(approximator.summary_network) else None
        )
        fallback_person_chunk = (
            8 if _supports_person_chunking(approximator.summary_network) else None
        )
        set_summary_memory_config(
            approximator.summary_network,
            item_chunk_size=fallback_chunk,
            gradient_checkpointing=True,
            person_chunk_size=fallback_person_chunk,
        )
        if verbose:
            print(
                f"  -> UNVALIDATED FALLBACK APPLIED (probe never ran: "
                f"{report.probe_error}): item_chunk_size={fallback_chunk}  "
                f"person_chunk_size={fallback_person_chunk}  "
                "gradient_checkpointing=True -- training may still OOM."
            )
        return report

    try:
        if use_item_search or use_person_search:
            axis = "item" if use_item_search else "person"
            ceiling = int(batch_size) if axis == "item" else int(probe_persons)
            for cp in checkpoint_options:
                dense = _probe_one(
                    clone,
                    make_simulator,
                    batch_size,
                    None,
                    cp,
                    probe_items_spec,
                    probe_persons_spec,
                    safety_margin=safety_margin,
                    person_chunk_size=None,
                )
                report.trials.append(dense)
                if verbose:
                    print(_format_probe_row(dense, note="dense baseline"))
                if dense.ok:
                    if report.best is None:
                        report.best = dense
                    break

                search_trials, best = _search_chunk_size_boundary(
                    clone,
                    make_simulator,
                    batch_size,
                    cp,
                    probe_items_spec,
                    probe_persons_spec,
                    safety_margin,
                    axis,
                    ceiling,
                    tolerance=search_tolerance,
                    verbose=verbose,
                )
                report.trials.extend(search_trials)
                if best is not None and report.best is None:
                    report.best = best
                if report.best is not None:
                    break
        else:
            for cp in checkpoint_options:
                for cs in candidate_item_chunk_sizes:
                    for pcs in candidate_person_chunk_sizes:
                        result = _probe_one(
                            clone,
                            make_simulator,
                            batch_size,
                            cs,
                            cp,
                            probe_items_spec,
                            probe_persons_spec,
                            safety_margin=safety_margin,
                            person_chunk_size=pcs,
                        )
                        report.trials.append(result)
                        if result.ok and report.best is None:
                            report.best = result

                        if verbose:
                            print(_format_probe_row(result))

                        if early_stop and report.best is not None:
                            break
                    if early_stop and report.best is not None:
                        break
                if early_stop and report.best is not None:
                    break

    finally:
        # Tear down the clone and free GPU memory.
        del clone
        gc.collect()
        safe_empty_cache()

    # Apply the best config (or a safe fallback) to the ORIGINAL summary net.
    summary_net = approximator.summary_network
    if report.best is not None:
        set_summary_memory_config(
            summary_net,
            report.best.item_chunk_size,
            report.best.gradient_checkpointing,
            person_chunk_size=report.best.person_chunk_size,
        )
    else:
        report.probe_failed = True
        report.probe_error = "all probe configurations failed (OOM or unsupported)"
        if not force_fallback:
            raise RuntimeError(
                "probe_memory_config found no valid configuration. "
                "Re-run with force_fallback=True to apply an emergency fallback."
            )
        logger.warning(
            "probe_memory_config: all probe configurations OOM'd. "
            "Applying emergency fallback (item_chunk_size=8, "
            "person_chunk_size=8, gradient_checkpointing=True). This has NOT "
            "been validated to fit in memory — training may still OOM."
        )
        fallback_chunk = 8 if _supports_item_chunking(summary_net) else None
        fallback_person_chunk = 8 if _supports_person_chunking(summary_net) else None
        set_summary_memory_config(
            summary_net,
            item_chunk_size=fallback_chunk,
            gradient_checkpointing=True,
            person_chunk_size=fallback_person_chunk,
        )

    if verbose:
        if report.best is not None:
            print(_format_probe_row(report.best, note="-> best"))
        elif report.probe_failed:
            fallback_chunk = 8 if _supports_item_chunking(summary_net) else None
            fallback_person_chunk = (
                8 if _supports_person_chunking(summary_net) else None
            )
            print(
                "  -> no config succeeded; UNVALIDATED FALLBACK APPLIED: "
                f"item_chunk_size={fallback_chunk}  "
                f"person_chunk_size={fallback_person_chunk}  "
                "gradient_checkpointing=True -- training may still OOM."
            )
        else:
            print("  -> no config succeeded")
        total_s = time.perf_counter() - probe_start_time
        print(f"  ({len(report.trials)} trial(s), {total_s:.2f}s total probing time)")

    return report


# -- Fit with OOM retry -----------------------------------------------------


class _PeriodicCudaCacheRelease(keras.callbacks.Callback):
    """Calls ``torch.cuda.empty_cache()`` after every train/test batch.

    Variable-size IRT training draws a different ``(n_items, n_persons)``
    shape every batch, so PyTorch's CUDA caching allocator keeps seeing
    tensor sizes it hasn't cached a block for yet. Its allocator normally
    only *grows* the reserved pool to satisfy each new largest-shape-so-far
    request and never shrinks it back down mid-training, so reserved memory
    can ratchet far above ``torch.cuda.memory_allocated()`` -- the actual
    live-tensor footprint. Once reserved memory exceeds the
    physical VRAM this way, the Windows WDDM driver silently spills into
    much slower shared/system memory (see the VRAM check in ``_probe_one``
    above), so training can start fast and get progressively slower.

    Calling ``empty_cache()`` after every batch releases now-unused cached
    blocks back to the driver, so the reserved watermark stays close to the
    current batch's actual peak instead of the running maximum across every
    shape seen so far. This can make training faster overall (fewer WDDM
    shared-memory spills to pay for), despite the added per-step
    ``empty_cache()`` call.
    """

    def __init__(self):
        super().__init__()
        try:
            import torch

            self._torch = torch
            self._enabled = torch.cuda.is_available()
        except ImportError:
            self._torch = None
            self._enabled = False

    def on_train_batch_end(self, batch, logs=None):
        if self._enabled:
            self._torch.cuda.empty_cache()

    def on_test_batch_end(self, batch, logs=None):
        if self._enabled:
            self._torch.cuda.empty_cache()


@dataclass
class FitResult:
    """Return value of :func:`fit_with_oom_retry`.

    This bundles the successful Keras ``History`` together with the effective
    training batch size and the memory configuration that made the run
    possible.  It is intended to be a small provenance record for notebooks
    and training scripts.
    """

    history: Any
    """Keras ``History`` object covering every epoch actually trained,
    across every OOM-triggered restart -- not just the final (or, on
    ``KeyboardInterrupt``, final-before-interrupt) attempt's epochs. Earlier
    attempts' metric values are merged in via
    :func:`_merge_epoch_metrics_into_history`."""
    batch_size: int
    """Effective batch size that succeeded."""
    item_chunk_size: int | None
    """Effective ``item_chunk_size`` on the summary network that succeeded
    (``None`` if the approximator has no summary network with chunking support,
    or if no chunking was needed)."""
    attempts: int
    """Number of fit attempts (1 = no retry)."""
    config: TrainingMemoryConfig
    """Final memory config applied on the approximator."""
    person_chunk_size: int | None = None
    """Effective ``person_chunk_size`` on the summary network that succeeded
    (``None`` if the approximator has no summary network with person-chunking
    support, or if no person chunking was needed)."""


def _merge_epoch_metrics_into_history(
    history_so_far: dict[str, list[Any]], history: Any
) -> None:
    """Prepend ``history_so_far`` onto a Keras ``History`` object, in place.

    Used by :func:`fit_with_oom_retry` to make its returned ``History``
    cover every epoch actually trained across OOM-triggered restarts, not
    just the final (or final-before-interrupt) attempt's epochs. A no-op
    if ``history_so_far`` is empty (the common case: no restart happened).
    """
    if not history_so_far:
        return
    merged: dict[str, list[Any]] = {}
    for key in dict.fromkeys([*history_so_far.keys(), *history.history.keys()]):
        merged[key] = list(history_so_far.get(key, [])) + list(
            history.history.get(key, [])
        )
    history.history = merged
    if hasattr(history, "epoch"):
        history.epoch = list(range(len(next(iter(merged.values()), []))))


def fit_with_oom_retry(
    approximator,
    simulator,
    *,
    start_batch_size: int,
    min_batch_size: int = 8,
    backoff_factor: float = 0.5,
    n_persistent_tries: int = 2,
    start_person_chunk_size: int | None = None,
    release_cuda_cache_every_step: bool = True,
    fit_kwargs: dict[str, Any] | None = None,
) -> FitResult:
    """Train *approximator* with automatic CUDA OOM recovery.

    The retry logic is designed for large IRT workloads where the exact
    feasible batch size depends on the response shape and summary
    architecture.  On a CUDA out-of-memory error the function progressively
    relaxes the summary-side memory settings and, if necessary, the batch
    size, until training succeeds or the configured floor is reached.

    Recovery order on CUDA OOM, from statistically free to
    calibration-affecting:

    1. **Reduce ``item_chunk_size``** on the summary network — processes fewer
       ``batch × items`` rows through the inner SABs per pass.  This has no
       effect on the mathematical result (same gradients, same loss value).
       Tries ``None → batch_size → batch_size * backoff_factor → … → 8``.
    2. **Reduce ``person_chunk_size``** on summary networks that support
       person-axis chunking — bounds peak memory to
       ``O(person_chunk_size)`` regardless of how many persons are in the
       drawn batch.  Exact (same gradients as the dense path) on the torch
       backend, including matrix blocks with nonzero dropout on CPU/CUDA
       (RNG state is preserved across the chunked backward's recompute so it
       reproduces the dropout mask actually used by the forward pass that
       produced the loss; on other torch devices such as MPS, nonzero
       dropout falls back to the dense path).  Tried at every
       ``item_chunk_size`` rung above, independently:
       ``None → 8192 → … → 8`` (halving) when ``start_person_chunk_size``
       is left at its default (``None``, meaning nothing is known about a
       safe value yet); or, when the caller passes an explicit
       ``start_person_chunk_size`` (e.g. already validated by
       :func:`probe_memory_config`), starting directly at that value and
       stepping down more gently (multiplying by ``sqrt(backoff_factor)``, not
       ``backoff_factor``) instead of retrying dense first, since dense
       would only re-derive -- via a real training attempt -- a config
       already known to be infeasible, and a real OOM this close to a
       validated value likely means "just barely not enough," not license
       to jump straight to a much smaller, slower chunk size. Item and
       person chunking are mutually orthogonal knobs on the current
       summary classes (no class supports both today), so nesting them is
       a full enumeration, not a combinatorial
       explosion in practice.
    3. **Halve ``batch_size`` while doubling ``gradient_accumulation_steps``**
       once all chunk sizes at the current batch size are exhausted, holding
       *effective* batch size (``batch_size x gradient_accumulation_steps``)
       constant — no calibration effect, just a smaller physical microbatch.
       Only engaged when the approximator already exposes a mutable
       ``gradient_accumulation_steps`` attribute and does *not* have
       ``standardize_params=True`` (accumulation would double-advance the
       running statistics of standardization layers).
    4. **Plain halve ``batch_size``** (an actual effective-batch-size cut,
       the step that can affect calibration) when step 3's compensation is
       unavailable — the fallback for approximators without a settable
       ``gradient_accumulation_steps`` or with ``standardize_params=True``.

    Each batch-size step restarts the chunk ladders -- from ``None`` for
    ``item_chunk_size``, and from ``None`` (default) or
    ``start_person_chunk_size`` (when explicitly given) for
    ``person_chunk_size``.

    If the approximator has no ``summary_network`` with ``item_chunk_size``
    or ``person_chunk_size`` support, the function falls back to
    batch-size-only back-off (steps 3/4 above). This is the path used by
    matrix-summary models without person chunking, where memory relief typically
    came only from ``gradient_checkpointing`` and from flow/batch controls.

    **LR schedule sync:** after each OOM, ``optimizer.iterations`` is reset to
    ``epochs_done × num_batches``.  Without this, ``CosineDecay`` (and any
    schedule that reads ``optimizer.iterations``) would see artificially high
    step counts caused by partial epochs that OOM'd, decaying the LR faster
    than the actual training progress justifies.

    Parameters
    ----------
    approximator
        A compiled BayesFlow approximator.
    simulator
        A BayesFlow simulator.
    start_batch_size : int
        Initial batch size to try.
    min_batch_size : int
        Stop retrying once the batch size drops below this value.
    backoff_factor : float
        Multiplicative factor (in ``(0, 1]``) applied to ``item_chunk_size``
        and (when all chunks are exhausted) to ``batch_size`` on each retry;
        its reciprocal (``1 / backoff_factor``) is the multiplier applied to
        ``gradient_accumulation_steps`` when compensating a batch-size
        halving (default 0.5, i.e. halving each step). Also applied to
        ``person_chunk_size`` when ``start_person_chunk_size`` is left at
        its default; ``sqrt(backoff_factor)`` is used instead (a gentler,
        less-aggressive-than-halving step) when the caller passes an
        explicit, already-validated ``start_person_chunk_size`` -- see that
        parameter.
    n_persistent_tries : int
        Number of times to retry the **same** ``(batch_size, item_chunk_size,
        person_chunk_size)`` configuration after an OOM before stepping down
        to a smaller one (default 2, i.e. one retry after the first OOM).
    start_person_chunk_size : int, optional
        Starting rung for the ``person_chunk_size`` ladder, for summary
        networks that support person-axis chunking. Two modes:

        - ``None`` (default): nothing is known about a safe value yet, so
          the ladder includes ``None`` (dense, no chunking) as its first
          rung -- mirroring ``item_chunk_size``'s behavior -- then falls
          back to an internal default starting rung (8192, clamped below
          ``simulator``'s worst-case ``n_persons`` when that attribute is
          available -- a chunk size at or above the max persons a batch
          can ever draw chunks nothing, so an un-clamped rung would just
          re-fail the same way as the dense rung already tried) if that
          OOMs.
        - An explicit value: treated as *already validated* (e.g. by
          :func:`probe_memory_config`), so the ladder starts directly
          there with **no** ``None``/dense rung first -- retrying dense
          would only re-derive, via a real training attempt, a config
          already known to be infeasible at the caller's actual
          ``n_persons`` range. Steps down by ``sqrt(backoff_factor)``
          instead of ``backoff_factor`` in this case: the single-step
          probe that validated the seed is not a perfect predictor of a
          full variable-shape training run (see
          :func:`probe_memory_config`'s ``safety_margin``), so a real OOM
          this close to a validated value likely means "just barely not
          enough," not license to jump straight to a much smaller, slower
          chunk size the way a full halving would.

        Unlike the ``item_chunk_size`` ladder (seeded from ``batch_size``,
        since item chunking operates on ``batch x items`` rows), person
        chunking operates on the person axis *within* each simulated batch
        element -- hence the internal default (8192, clamped to
        ``simulator.n_persons`` when available -- see above) instead of a
        ``batch_size``-derived one. Ignored for summary networks that do
        not support person chunking.
    release_cuda_cache_every_step : bool
        If ``True`` (default) and running on the torch backend with CUDA,
        inject a callback that calls ``torch.cuda.empty_cache()`` after
        every train/test batch. Variable-size batches make PyTorch's CUDA
        caching allocator ratchet its *reserved* memory up to the largest
        shape seen so far and never release it mid-training, which can
        silently exceed physical VRAM (see ``_PeriodicCudaCacheRelease``)
        well before an actual ``torch.cuda.OutOfMemoryError`` would fire.
        Set ``False`` to disable if you supply your own memory-management
        callback.
    fit_kwargs : dict, optional
        Extra keyword arguments forwarded to ``approximator.fit()`` (e.g.
        ``epochs``, ``num_batches``, ``workers``).

    Returns
    -------
    FitResult
        History plus the effective batch size, retry count, and memory config
        that succeeded.

    Raises
    ------
    RuntimeError
        If training OOMs through all chunk sizes down to *min_batch_size*.

    Notes
    -----
    This function mutates the live approximator as it searches for a viable
    configuration.  In particular, ``summary_network.item_chunk_size``,
    ``summary_network.person_chunk_size``, and
    ``approximator.gradient_accumulation_steps`` may be changed between
    retries, and the final successful settings are left in place on return.

    References
    ----------
    Chen, T., Xu, B., Zhang, C., & Guestrin, C. (2016). *Training Deep Nets
    with Sublinear Memory Cost*.
    PyTorch Team. (2026). *CUDA semantics*.
    """
    if not 0.0 < backoff_factor <= 1.0:
        raise ValueError(
            f"backoff_factor must be a fraction in (0, 1] (e.g. 0.5 to halve "
            f"each step; 1.0 is a no-op); got {backoff_factor}. For a "
            "divisor-style value, invert it (2 -> 0.5, 4 -> 0.25)."
        )

    if fit_kwargs is None:
        fit_kwargs = {}

    # Cap the CUDA allocator to ~95% of physical VRAM (not the same as the
    # probe's safety_margin -- see _cap_cuda_memory_fraction's docstring) so
    # an actual near-total overflow during real training raises a real,
    # catchable OutOfMemoryError for fit_with_oom_retry's own backoff ladder
    # to react to, instead of Windows' WDDM driver silently spilling into
    # slow shared/system memory. No-op if already capped (e.g.
    # probe_memory_config ran first in this process) or CUDA is unavailable.
    if keras.backend.backend() == "torch":
        _cap_cuda_memory_fraction()

    if release_cuda_cache_every_step and keras.backend.backend() == "torch":
        fit_kwargs = {
            **fit_kwargs,
            "callbacks": [
                *fit_kwargs.get("callbacks", []),
                _PeriodicCudaCacheRelease(),
            ],
        }

    summary_net = getattr(approximator, "summary_network", None)
    has_item_chunking = _supports_item_chunking(summary_net)
    has_person_chunking = _supports_person_chunking(summary_net)

    # person_chunk_size's natural scale is the person axis, not batch_size
    # (unlike item_chunk_size, seeded from batch_size below), so this ladder
    # is built once from a fixed starting rung rather than being rebuilt
    # every time batch_size changes.
    if has_person_chunking:
        # An explicit start_person_chunk_size is treated as already
        # validated (e.g. by probe_memory_config): start the ladder there
        # directly, with no None/dense rung first. Left at the default
        # (None), nothing is known yet, so try dense first and fall back to
        # an internal default starting rung (8192) on OOM -- the prior
        # behavior for every existing caller.
        is_seeded = start_person_chunk_size is not None
        person_chunk_ladder: list[int | None] = [] if is_seeded else [None]
        # A seeded (already-validated) start point steps down gently --
        # sqrt(backoff_factor) rather than backoff_factor -- since a real
        # OOM this close to a validated value likely means "just barely
        # not enough," not license to jump straight to a much smaller,
        # slower chunk size.
        pc_backoff = (
            float(backoff_factor) ** 0.5 if is_seeded else float(backoff_factor)
        )
        # The internal default starting rung (8192) is only a sensible
        # upper bound when the simulator can actually draw that many
        # persons. chunk_size >= n_persons chunks nothing -- it processes
        # the exact same single full-extent chunk as dense (None), per
        # _search_chunk_size_boundary's docstring -- so an un-clamped 8192
        # rung against e.g. n_persons=(5, 500) is a functional no-op
        # duplicate of the dense rung already tried above, wasting a full
        # persistent-retry cycle (a real, possibly multi-minute partial
        # training attempt) reproving a failure already known from the
        # dense trial before ever reaching a rung that changes memory
        # behavior. simulator.n_persons (set by every simulator in this
        # module's __init__) gives this function the visibility to clamp
        # the default rung below that ceiling; fall back to the
        # unclamped 8192 default for simulators that don't expose it.
        default_start_pcs = 8192
        sim_n_persons = getattr(simulator, "n_persons", None)
        if isinstance(sim_n_persons, (tuple, list)) and len(sim_n_persons) == 2:
            person_ceiling = int(max(sim_n_persons))
            if person_ceiling > 0:
                default_start_pcs = max(8, min(default_start_pcs, person_ceiling - 1))
        pcs = int(start_person_chunk_size) if is_seeded else default_start_pcs
        while pcs >= 8:
            person_chunk_ladder.append(pcs)
            next_pcs = int(pcs * pc_backoff)
            if next_pcs >= pcs:
                # Guard against a backoff factor <= 1 (or rounding to a
                # no-op step) stalling the ladder forever.
                next_pcs = pcs - 1
            pcs = next_pcs
        person_chunk_ladder = list(dict.fromkeys(person_chunk_ladder))
        if not person_chunk_ladder:
            # A seeded start_person_chunk_size already below 8: fall back
            # to dense rather than trying nothing.
            person_chunk_ladder = [None]
    else:
        person_chunk_ladder = [None]

    # gradient_accumulation_steps compensation (step 3 of the ladder) is only
    # safe when:
    # - the approximator exposes a mutable gradient_accumulation_steps
    #   attribute;
    # - it does not combine that with running-stats standardization
    #   (accumulation would double-advance the running statistics);
    # - the active Keras backend is torch. EquivariantIRTApproximator's
    #   train_step() raises NotImplementedError (not an OOM) for
    #   gradient_accumulation_steps > 1 on any other backend -- without this
    #   check, enabling compensation after an OOM on JAX/TensorFlow would
    #   turn a recoverable OOM into a hard abort of the whole retry loop.
    # A further, per-step_down check (below, using num_batches) additionally
    # guards against silently discarding every epoch's accumulated
    # gradients -- see that check's comment for why.
    # When any guard fails, batch-size step-downs fall back to a plain cut
    # (step 4), matching this function's behavior without person chunking exactly.
    can_compensate_with_accumulation = (
        hasattr(approximator, "gradient_accumulation_steps")
        and not getattr(approximator, "standardize_params", False)
        and keras.backend.backend() == "torch"
    )

    batch_size = int(start_batch_size)
    last_exc: BaseException | None = None
    attempts = 0
    total_epochs: int | None = fit_kwargs.get("epochs")
    num_batches: int | None = fit_kwargs.get("num_batches")
    epochs_done = 0  # epochs completed across all (possibly OOM'd) attempts
    # Per-epoch metrics (loss, val_loss, …) from every attempt that completed
    # at least one epoch before an OOM forced a restart. epochs_done (above)
    # only tracks a *count* so the next retry's epochs= arg is computed
    # correctly; the metric values are kept here so a later attempt's
    # History does not replace them in the returned FitResult, which would
    # truncate the training curve to the final attempt's epochs.
    # Merged into whichever History object is ultimately returned (success
    # or KeyboardInterrupt) so FitResult.history always covers every epoch
    # actually trained, in order.
    history_so_far: dict[str, list[Any]] = {}
    # optimizer.apply() calls completed across all attempts. NOT simply
    # epochs_done * num_batches once gradient_accumulation_steps
    # compensation (below) is active: EquivariantIRTApproximator.train_step
    # calls optimizer.apply() once per accumulation window, i.e.
    # num_batches // gradient_accumulation_steps times per epoch, not once
    # per batch. Tracked using whichever gradient_accumulation_steps was
    # actually active on the approximator during each attempt's completed
    # epochs (read at the moment those epochs are counted, before any later
    # step-down changes it), since compensation can raise it partway
    # through a retry sequence.
    completed_optimizer_steps = 0
    # optimizer.iterations on entry: a model that was trained or restored
    # before this call already carries optimizer steps, and the schedule resync
    # below must restore baseline + completed steps, not completed steps alone.
    try:
        baseline_iterations = int(approximator.optimizer.iterations)
    except (RuntimeError, AttributeError, TypeError, ValueError):
        baseline_iterations = 0

    while batch_size >= min_batch_size:
        # Build item-chunk ladder for this batch_size: [None, bs, bs//2, …, 8]
        if has_item_chunking:
            item_chunk_ladder: list[int | None] = [None]
            item_cs = batch_size
            while item_cs >= 8:
                item_chunk_ladder.append(item_cs)
                next_item_cs = int(item_cs * backoff_factor)
                if next_item_cs >= item_cs:
                    # Guard against a backoff factor >= 1 (or rounding to a
                    # no-op step) stalling the ladder forever.
                    next_item_cs = item_cs - 1
                item_cs = next_item_cs
            item_chunk_ladder = list(dict.fromkeys(item_chunk_ladder))  # deduplicate
        else:
            item_chunk_ladder = [None]

        oom_in_chunk_loop = False
        for item_chunk_size in item_chunk_ladder:
            for person_chunk_size in person_chunk_ladder:
                for persistent_try in range(n_persistent_tries):
                    attempts += 1
                    if has_item_chunking or has_person_chunking:
                        set_summary_memory_config(
                            summary_net,
                            item_chunk_size,
                            summary_net.gradient_checkpointing,
                            person_chunk_size=person_chunk_size
                            if has_person_chunking
                            else None,
                        )
                        logger.info(
                            "Trying batch_size=%d, item_chunk_size=%s, "
                            "person_chunk_size=%s …",
                            batch_size,
                            item_chunk_size,
                            person_chunk_size if has_person_chunking else None,
                        )
                    else:
                        logger.info("Trying training with batch_size=%d …", batch_size)

                    # Subtract already-completed epochs so restarts don't
                    # re-train them.
                    if total_epochs is not None:
                        remaining = total_epochs - epochs_done
                        current_fit_kwargs = {**fit_kwargs, "epochs": remaining}
                    else:
                        current_fit_kwargs = fit_kwargs

                    try:
                        history = approximator.fit(
                            simulator=simulator,
                            batch_size=batch_size,
                            **current_fit_kwargs,
                        )
                        _merge_epoch_metrics_into_history(history_so_far, history)
                        config = TrainingMemoryConfig(
                            item_chunk_size=(
                                item_chunk_size if has_item_chunking else None
                            ),
                            gradient_checkpointing=bool(
                                getattr(summary_net, "gradient_checkpointing", False)
                            )
                            if summary_net is not None
                            else False,
                            flow_chunk_size=getattr(
                                approximator, "flow_chunk_size", None
                            ),
                            flow_chunk_backward=getattr(
                                approximator, "flow_chunk_backward", None
                            ),
                            gradient_accumulation_steps=int(
                                getattr(approximator, "gradient_accumulation_steps", 1)
                            ),
                            person_chunk_size=person_chunk_size
                            if has_person_chunking
                            else None,
                        )
                        return FitResult(
                            history=history,
                            batch_size=batch_size,
                            item_chunk_size=(
                                item_chunk_size if has_item_chunking else None
                            ),
                            attempts=attempts,
                            config=config,
                            person_chunk_size=person_chunk_size
                            if has_person_chunking
                            else None,
                        )

                    except KeyboardInterrupt:
                        # Return whatever partial history Keras accumulated so
                        # downstream cells (loss plot, etc.) don't crash.
                        partial = getattr(approximator, "history", None)
                        if partial is not None and getattr(partial, "history", None):
                            n_done = len(next(iter(partial.history.values()), []))
                            _merge_epoch_metrics_into_history(history_so_far, partial)
                            logger.info(
                                "Training interrupted after %d epoch(s) — "
                                "returning partial history.",
                                n_done,
                            )
                            return FitResult(
                                history=partial,
                                batch_size=batch_size,
                                item_chunk_size=item_chunk_size
                                if has_item_chunking
                                else None,
                                attempts=attempts,
                                config=TrainingMemoryConfig(
                                    item_chunk_size=item_chunk_size
                                    if has_item_chunking
                                    else None,
                                    gradient_checkpointing=bool(
                                        getattr(
                                            summary_net,
                                            "gradient_checkpointing",
                                            False,
                                        )
                                    )
                                    if summary_net is not None
                                    else False,
                                    flow_chunk_size=getattr(
                                        approximator, "flow_chunk_size", None
                                    ),
                                    flow_chunk_backward=getattr(
                                        approximator, "flow_chunk_backward", None
                                    ),
                                    gradient_accumulation_steps=int(
                                        getattr(
                                            approximator,
                                            "gradient_accumulation_steps",
                                            1,
                                        )
                                    ),
                                    person_chunk_size=person_chunk_size
                                    if has_person_chunking
                                    else None,
                                ),
                                person_chunk_size=person_chunk_size
                                if has_person_chunking
                                else None,
                            )
                        raise

                    except Exception as exc:
                        if not is_oom_error(exc):
                            raise
                        last_exc = exc
                        n_this_attempt = 0

                        # Count epochs completed during this failed attempt
                        # and accumulate so the next retry starts where this
                        # one left off.
                        if total_epochs is not None:
                            partial = getattr(approximator, "history", None)
                            if partial is not None and getattr(
                                partial, "history", None
                            ):
                                n_this_attempt = len(
                                    next(iter(partial.history.values()), [])
                                )
                                if n_this_attempt:
                                    # Preserve this attempt's actual metric
                                    # values (not just the epoch count) so
                                    # the eventually-returned History covers
                                    # every epoch really trained, not just
                                    # the final attempt's — see
                                    # _merge_epoch_metrics_into_history.
                                    for key, values in partial.history.items():
                                        history_so_far.setdefault(key, []).extend(
                                            values
                                        )
                                    epochs_done += n_this_attempt
                                    if num_batches is not None:
                                        # gradient_accumulation_steps active
                                        # on the approximator right now is
                                        # exactly what was active during the
                                        # epochs this attempt just completed
                                        # -- compensation (below) only ever
                                        # changes it after this whole
                                        # persistent-try loop has exited.
                                        active_ga = max(
                                            1,
                                            int(
                                                getattr(
                                                    approximator,
                                                    "gradient_accumulation_steps",
                                                    1,
                                                )
                                            ),
                                        )
                                        completed_optimizer_steps += n_this_attempt * (
                                            num_batches // active_ga
                                        )
                                    logger.info(
                                        "OOM after %d epoch(s); %d / %d total "
                                        "epochs done.",
                                        n_this_attempt,
                                        epochs_done,
                                        total_epochs,
                                    )

                        gc.collect()
                        safe_empty_cache()

                        # --- Sync LR schedule to actual progress ---
                        # optimizer.iterations advances once per
                        # optimizer.apply() call -- once per batch normally,
                        # but once per gradient_accumulation_steps-sized
                        # window when accumulation is active (see
                        # EquivariantIRTApproximator.train_step) -- and is
                        # never reset between fit() calls. If OOM fires
                        # mid-epoch the counter overshoots: the next attempt
                        # starts with a lower LR than the clean schedule would
                        # give for the optimizer steps that actually
                        # completed. Fix: clamp optimizer.iterations back to
                        # baseline_iterations + completed_optimizer_steps, NOT
                        # epochs_done * num_batches (that assumes
                        # gradient_accumulation_steps == 1 throughout, which
                        # no longer holds once step 3's compensation raises
                        # it mid-retry). Only done when this attempt completed
                        # an epoch: an OOM before the first completed epoch
                        # keeps the weights and Adam slots of its partial
                        # updates, so the counter (used for Adam's bias
                        # correction as well as the schedule) must keep
                        # them too.
                        if num_batches is not None and n_this_attempt > 0:
                            target_step = (
                                baseline_iterations + completed_optimizer_steps
                            )
                            try:
                                opt = approximator.optimizer
                                if (
                                    hasattr(opt, "iterations")
                                    and opt.iterations is not None
                                ):
                                    opt.iterations.assign(target_step)
                            except (RuntimeError, AttributeError, TypeError) as exc:
                                logger.warning(
                                    "Could not resync optimizer.iterations to %d: %s",
                                    target_step,
                                    exc,
                                )

                        is_last_persistent_try = (
                            persistent_try == n_persistent_tries - 1
                        )
                        if not is_last_persistent_try:
                            logger.warning(
                                "OOM at batch_size=%d, item_chunk_size=%s, "
                                "person_chunk_size=%s — retrying same config "
                                "(try %d/%d).",
                                batch_size,
                                item_chunk_size,
                                person_chunk_size if has_person_chunking else None,
                                persistent_try + 1,
                                n_persistent_tries,
                            )
                        else:
                            if has_item_chunking or has_person_chunking:
                                logger.warning(
                                    "OOM at batch_size=%d, item_chunk_size=%s, "
                                    "person_chunk_size=%s after %d tries "
                                    "— trying next config.",
                                    batch_size,
                                    item_chunk_size,
                                    person_chunk_size if has_person_chunking else None,
                                    n_persistent_tries,
                                )
                            else:
                                logger.warning(
                                    "OOM at batch_size=%d after %d tries — halving.",
                                    batch_size,
                                    n_persistent_tries,
                                )
                            oom_in_chunk_loop = True
                            break  # exit persistent loop → chunk loops advance

        # The batch-size step must strictly decrease -- a backoff_factor
        # close enough to 1 (including the boundary value 1.0, which is
        # explicitly allowed for the chunk-size ladders above) would
        # otherwise leave batch_size unchanged and retry the same failing
        # configuration forever instead of ever reaching min_batch_size.
        next_batch_size = int(batch_size * backoff_factor)
        if next_batch_size >= batch_size:
            next_batch_size = batch_size - 1

        if oom_in_chunk_loop:
            current_accumulation_steps = int(
                getattr(approximator, "gradient_accumulation_steps", 1)
            )
            proposed_accumulation_steps = int(
                current_accumulation_steps / backoff_factor
            )
            # Compensation must actually hold the effective batch size
            # (batch_size * gradient_accumulation_steps) constant, not just
            # produce *some* accumulation value that happens to divide
            # num_batches evenly -- e.g. backoff_factor=0.75 with
            # accumulation_steps=1 truncates proposed_accumulation_steps
            # back down to 1 (int(1 / 0.75) == 1), which trivially divides
            # any num_batches but leaves accumulation unchanged while the
            # batch size still shrinks by 25%, silently breaking the
            # documented guarantee.
            old_effective_batch = batch_size * current_accumulation_steps
            new_effective_batch = next_batch_size * proposed_accumulation_steps
            # A compensated halving is also only safe if num_batches is known,
            # the new accumulation window still fits within one epoch, and
            # it evenly divides num_batches. EquivariantIRTApproximator's
            # fit-time _AccumulationEpochReset callback silently discards
            # any partial accumulation window left at epoch end (to avoid
            # cross-epoch gradient contamination) -- if
            # gradient_accumulation_steps > num_batches,
            # that window *never* completes within an epoch, so every
            # optimizer.apply() call is skipped and fit() can return a
            # "successful" History despite zero parameter updates. A
            # non-divisible value is safer (some windows still complete) but
            # still discards part of every epoch, undershooting the
            # documented "effective batch size held constant" guarantee, so
            # it is also rejected here rather than silently accepted.
            can_compensate_this_step = (
                can_compensate_with_accumulation
                and num_batches is not None
                and proposed_accumulation_steps <= num_batches
                and num_batches % proposed_accumulation_steps == 0
                and new_effective_batch >= old_effective_batch
            )
            if can_compensate_this_step:
                approximator.gradient_accumulation_steps = proposed_accumulation_steps
                logger.warning(
                    "OOM exhausted all chunk configs at batch_size=%d — halving "
                    "batch size to %d while raising gradient_accumulation_steps "
                    "to %d, holding effective batch size constant.",
                    batch_size,
                    next_batch_size,
                    proposed_accumulation_steps,
                )
            else:
                if not can_compensate_with_accumulation:
                    if getattr(approximator, "standardize_params", False):
                        reason = "standardize_params=True"
                    elif keras.backend.backend() != "torch":
                        reason = (
                            "gradient_accumulation_steps > 1 is only "
                            "implemented on the torch backend"
                        )
                    else:
                        reason = "approximator has no gradient_accumulation_steps"
                elif num_batches is None:
                    reason = "num_batches is unknown"
                elif new_effective_batch < old_effective_batch:
                    reason = (
                        f"gradient_accumulation_steps={proposed_accumulation_steps} "
                        f"would leave the effective batch size at "
                        f"{new_effective_batch} (down from {old_effective_batch}) "
                        "despite compensation, e.g. because backoff_factor's "
                        "reciprocal is not an integer"
                    )
                else:
                    reason = (
                        f"gradient_accumulation_steps={proposed_accumulation_steps} "
                        f"would not evenly divide num_batches={num_batches} (or "
                        "exceeds it), which would silently discard some or all "
                        "of every epoch's accumulated gradients"
                    )
                logger.warning(
                    "OOM exhausted all chunk configs at batch_size=%d — halving "
                    "batch size (no gradient_accumulation_steps compensation "
                    "available: %s).",
                    batch_size,
                    reason,
                )
        batch_size = next_batch_size

    raise RuntimeError(
        "Training OOM'd through all chunk sizes down to "
        f"min_batch_size={min_batch_size}."
    ) from last_exc


# ---------------------------------------------------------------------------
# Pre-computed multi-condition validation dataset
# ---------------------------------------------------------------------------


class PrecomputedDataset(keras.utils.PyDataset):
    """A static validation dataset wrapping a pre-generated list of adapted batches.

    Unlike :class:`bayesflow.datasets.OnlineDataset`, this dataset simulates
    data **once at construction time** and caches it.  The same fixed batches
    are returned every epoch, which stabilises the validation loss curve and
    eliminates simulation overhead during training.

    Each batch may have a different inner shape (e.g. different ``(I, P)``),
    because Keras evaluates variable-size batches one at a time via
    ``__getitem__``.

    Parameters
    ----------
    batches : list[dict[str, np.ndarray]]
        Pre-adapted batches produced by an adapter.  Each element is a dict
        mapping BayesFlow key names (``"inference_variables"``,
        ``"summary_variables"``, …) to numpy arrays.
    """

    def __init__(self, batches: list[dict]) -> None:
        super().__init__()
        self._batches = batches

    def __len__(self) -> int:
        return len(self._batches)

    def __getitem__(self, idx: int) -> dict:
        return self._batches[idx]

    @property
    def num_batches(self) -> int:
        return len(self._batches)


def build_multicondition_val_dataset(
    simulator,
    adapter,
    *,
    num_conditions: int = 50,
    samples_per_condition: int = 4,
) -> PrecomputedDataset:
    """Build a pre-computed validation dataset spanning multiple (I, P) conditions.

    **Why this is faster than** ``build_dataset(batch_size=1, num_batches=N)``:

    ``OnlineDataset`` with ``batch_size=1`` calls ``simulator.sample((1,))``
    once per validation step, yielding 100 single-sample GPU forward passes.
    This function instead groups ``samples_per_condition`` samples per
    condition — calling ``simulator.sample((samples_per_condition,))`` once
    per condition — and pre-caches the result.  The validation loop then
    processes ``num_conditions`` batches of size ``samples_per_condition``
    instead of ``num_conditions * samples_per_condition`` batches of size 1.

    The key property that makes this work: calling ``simulator.sample((k,))``
    draws ``(I, P)`` **once** for the whole call, so all ``k`` samples share
    the same shape and can be stacked into a single batch.  Different calls
    get independent ``(I, P)`` draws, spanning the full condition space.

    Parameters
    ----------
    simulator :
        A BayesFlow simulator whose ``sample((k,))`` call draws ``(I, P)``
        once and assigns it to all ``k`` samples.
    adapter :
        BayesFlow adapter to transform raw simulator output into the
        ``"inference_variables"`` / ``"summary_variables"`` format expected
        by the approximator.
    num_conditions : int
        Number of distinct ``(I, P)`` configurations to include (default 50).
        Corresponds to ``num_batches`` in ``build_dataset``.
    samples_per_condition : int
        Samples drawn per ``(I, P)`` configuration (default 4).
        Corresponds to ``batch_size`` in ``build_dataset``.

    Returns
    -------
    PrecomputedDataset
        A ``keras.utils.PyDataset`` with ``num_conditions`` batches, each
        of shape ``(samples_per_condition, ...)``.  Pass directly as
        ``validation_data=`` in ``approximator.fit()`` or ``fit_with_oom_retry``.

    Examples
    --------
    >>> val_dataset = build_multicondition_val_dataset(
    ...     val_simulator, approximator.adapter,
    ...     num_conditions=50, samples_per_condition=4,
    ... )
    >>> result = fit_with_oom_retry(
    ...     approximator, simulator,
    ...     start_batch_size=256,
    ...     fit_kwargs=dict(epochs=20, num_batches=100,
    ...                     validation_data=val_dataset),
    ... )
    """
    batches = [
        adapter(simulator.sample((samples_per_condition,)))
        for _ in range(num_conditions)
    ]
    return PrecomputedDataset(batches)


def configure_large_scale_training(
    *,
    mixed_precision: bool = True,
    torch_float32_matmul_precision: str | None = "high",
    strict: bool = True,
) -> None:
    """Configure optional process-wide training defaults for large-scale runs.

    References
    ----------
    Keras mixed precision API:
    https://keras.io/api/mixed_precision/
    PyTorch float32 matmul precision:
    https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    PyTorch CUDA allocator config (`PYTORCH_CUDA_ALLOC_CONF`):
    https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf

    Parameters
    ----------
    mixed_precision : bool
        If ``True``, call
        ``keras.mixed_precision.set_global_policy("mixed_float16")``.

        .. warning::
            On the PyTorch backend (recommended for this package) this sets a
            Keras-level dtype policy that may have **no effect** on PyTorch ops,
            which use ``torch.autocast`` for AMP.  Consider setting this to
            ``False`` and managing ``torch.autocast`` directly if you need AMP.
    torch_float32_matmul_precision : str or None
        Passed to ``torch.set_float32_matmul_precision()``.  Typical values:
        ``"high"`` (trades a small amount of precision for throughput on
        Ampere+ GPUs) or ``"highest"`` (strict IEEE-754).  ``None`` skips
        this call entirely.
    strict : bool
        If ``True`` (default), setup failures raise ``RuntimeError``.
        If ``False``, failures are downgraded to warnings.
    """

    def _handle_config_error(context: str, exc: Exception) -> None:
        if strict:
            raise RuntimeError(f"{context}: {exc}") from exc
        logger.warning("%s: %s", context, exc)

    if mixed_precision:
        try:
            keras.mixed_precision.set_global_policy("mixed_float16")
            if keras.backend.backend() == "torch":
                logger.warning(
                    "mixed_precision=True on the PyTorch backend sets a Keras "
                    "dtype policy but does NOT enable torch.autocast. Keras-level "
                    "AMP may have no effect on PyTorch operations. Consider setting "
                    "mixed_precision=False and using torch.autocast manually."
                )
        except (ValueError, RuntimeError) as exc:
            _handle_config_error("Could not enable mixed precision", exc)

    if torch_float32_matmul_precision is not None:
        try:
            import torch

            torch.set_float32_matmul_precision(torch_float32_matmul_precision)
        except (ImportError, ValueError, RuntimeError) as exc:
            # ImportError included: this call is skipped intentionally by
            # setting torch_float32_matmul_precision=None on non-torch
            # backends, but the default is non-None, so a JAX/TensorFlow-only
            # install without the optional torch dependency must still be
            # downgradable to a warning under strict=False rather than
            # always raising ImportError regardless of strict.
            _handle_config_error("Could not set torch matmul precision", exc)

    alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments:True" not in alloc_conf:
        logger.warning(
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is not set. "
            "For lower GPU allocator fragmentation, add this to your process "
            "environment before importing torch (takes effect on next launch)."
        )
