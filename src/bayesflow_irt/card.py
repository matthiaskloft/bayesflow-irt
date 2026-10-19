"""Model card: evaluation protocol and cell planning.

The protocol fixes what a card is evaluated on: the standard grid of
(items, persons) cells, the datasets and posterior draws per cell, and the
seed rule. Its values are recorded in ``evaluation.json`` so two cards are
comparable or visibly not.
The grid values are a design choice, not a source-backed result.
"""

from __future__ import annotations

import dataclasses
import re
import time
import warnings
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

#: Version of the protocol definition below. Bump it whenever a preset value,
#: the grid, the cell plan or the seed rule changes, so a stored evaluation
#: says which definition produced it.
PROTOCOL_VERSION = 1

#: The standard grid. Part of the protocol version.
STANDARD_GRID: dict[str, tuple[int, ...]] = {
    "n_items": (3, 5, 10, 20, 35, 50),
    "n_persons": (10, 25, 50, 100, 250, 500, 1000, 5000),
}

#: Named presets. ``validation`` is the full protocol for a final card;
#: ``development`` is a cheap run for development work; 100 datasets give
#: about three times the MCSE of 1000. The draw count is validated for
#: the families in :data:`DRAWS_VALIDATED_FOR` only.
PRESETS: dict[str, dict[str, Any]] = {
    "validation": {
        "datasets_per_cell": 1000,
        "posterior_draws": 2000,
        "include_out_of_range": True,
        "persons_scored": 100,
    },
    "development": {
        "datasets_per_cell": 100,
        "posterior_draws": 500,
        "include_out_of_range": False,
        "persons_scored": 100,
    },
}


#: Model families for which the ``posterior_draws`` default has been checked
#: against a larger draw count. 1PL uses it as a 2PL with ``a`` fixed;
#: PCM/GPCM are not yet checked.
DRAWS_VALIDATED_FOR = ("2pl",)

#: How seeds are derived, stored with every evaluation. The posterior part is
#: implemented by the cell runner (resetting the global seed alone
#: leaves a loaded model's ``keras.random.SeedGenerator`` state untouched).
SEED_RULE = (
    "cell seed = SeedSequence([seed, n_items, n_persons]); the simulator and "
    "the item/person choice are seeded from it, and the posterior seed is "
    "reset per chunk from it and the chunk's first dataset index, including "
    "every keras.random.SeedGenerator of the loaded model; the item/person "
    "choice is default_rng([cell seed, 1]); TARP's random reference points "
    "use seed = cell seed, and the joint-metric bootstrap and null floor "
    "default_rng([cell seed, 2]); the scored persons come from "
    "default_rng([cell seed, 3]) and each dataset's theta draws are seeded "
    "from SeedSequence([cell seed, 4, dataset index]); the simulator never "
    "uses the numba kernel (its RNG ignores the seed)"
)


@dataclass(frozen=True)
class EvaluationProtocol:
    """The constants a card evaluation runs under.

    Build one with :meth:`preset`, which records every field that differs
    from the named preset in :attr:`overrides`.

    Attributes
    ----------
    preset_name : str
        ``"validation"`` or ``"development"``.
    datasets_per_cell : int
        Simulated datasets per cell.
    posterior_draws : int
        Posterior draws per dataset.
    include_out_of_range : bool
        Also evaluate the out-of-range cells (:func:`plan_cells`).
    persons_scored : int
        Persons scored per dataset for reliability: the dataset's random
        person plus a seeded subsample (all persons if there are fewer).
        Scoring every person is slow and can take hours per validation
        card.
    seed : int
        Base seed; per-cell seeds come from :func:`cell_seed`.
    overrides : tuple of str
        Fields set by the caller to values other than the preset's.
    """

    preset_name: str
    datasets_per_cell: int
    posterior_draws: int
    include_out_of_range: bool
    persons_scored: int
    seed: int = 0
    overrides: tuple[str, ...] = ()

    @classmethod
    def preset(cls, name: str = "validation", **overrides: Any) -> EvaluationProtocol:
        """Return the named preset, with optional field overrides.

        Parameters
        ----------
        name : {"validation", "development"}
            The preset.
        **overrides
            Any of ``datasets_per_cell``, ``posterior_draws``,
            ``include_out_of_range``, ``persons_scored`` and ``seed``. Only
            values that differ from the preset are recorded as overrides.
        """
        if name not in PRESETS:
            raise ValueError(f"Unknown preset {name!r}; choose from {sorted(PRESETS)}.")
        values = {**PRESETS[name], "seed": 0}
        unknown = set(overrides) - set(values)
        if unknown:
            raise TypeError(f"Unknown protocol fields: {sorted(unknown)}.")
        changed = tuple(sorted(k for k, v in overrides.items() if values[k] != v))
        values.update(overrides)
        return cls(preset_name=name, overrides=changed, **values)

    def __post_init__(self) -> None:
        # Checked here, not in preset(), so direct construction is checked too.
        for key in ("datasets_per_cell", "posterior_draws", "persons_scored", "seed"):
            value = getattr(self, key)
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
                raise TypeError(f"{key} must be an int; got {value!r}.")
            object.__setattr__(self, key, int(value))
        flag = self.include_out_of_range
        if not isinstance(flag, bool):
            raise TypeError(f"include_out_of_range must be a bool; got {flag!r}.")
        # Per-parameter MCSEs need 3 datasets (card_metrics); the TCC bias
        # needs 4 draws (half-split z, joint_metrics).
        minimum = {"datasets_per_cell": 3, "posterior_draws": 4, "persons_scored": 1}
        for key, low in minimum.items():
            if getattr(self, key) < low:
                raise ValueError(f"{key} must be >= {low}; got {getattr(self, key)}.")
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0; got {self.seed}.")

    def to_dict(self) -> dict[str, Any]:
        """The protocol as stored in ``evaluation.json``."""
        return {
            "protocol_version": PROTOCOL_VERSION,
            "grid": {k: list(v) for k, v in STANDARD_GRID.items()},
            "seed_rule": SEED_RULE,
            **dataclasses.asdict(self),
            "overrides": list(self.overrides),
        }


@dataclass(frozen=True)
class Cell:
    """One (items, persons) cell of a card.

    ``in_range`` is true when both sizes lie inside the model's trained
    range; an out-of-range cell has at least one size beyond it.
    """

    n_items: int
    n_persons: int
    in_range: bool


def _axis(values: Sequence[int], bounds: Sequence[int]) -> tuple[list[int], list[int]]:
    """In-range values (grid plus exact bounds) and the next grid value out."""
    lo, hi = int(bounds[0]), int(bounds[1])
    if lo > hi:
        raise ValueError(f"Size range lower bound exceeds upper bound: {bounds}.")
    inside = sorted({v for v in values if lo <= v <= hi} | {lo, hi})
    below = [v for v in values if v < lo]
    above = [v for v in values if v > hi]
    outside = ([max(below)] if below else []) + ([min(above)] if above else [])
    return inside, outside


def plan_cells(
    size_ranges: Mapping[str, Sequence[int]], protocol: EvaluationProtocol
) -> list[Cell]:
    """Cells a model is evaluated on.

    Per axis, the in-range values are the standard-grid values inside the
    trained range plus the range's exact bounds, so the corners a model
    claims are always tested. The out-of-range values are the next grid
    value beyond each bound. Out-of-range cells (only with
    ``protocol.include_out_of_range``) are the edges, where one axis is out
    of range, and the corners, where both are.

    Parameters
    ----------
    size_ranges : mapping
        ``{"n_items": (lo, hi), "n_persons": (lo, hi)}``, the trained range
        from a sidecar's ``model_spec["size_ranges"]``.
    protocol : EvaluationProtocol
        The protocol in effect.

    Returns
    -------
    list of Cell
        In-range cells first, then out-of-range cells; each block sorted by
        items, then persons.
    """
    items_in, items_out = _axis(STANDARD_GRID["n_items"], size_ranges["n_items"])
    persons_in, persons_out = _axis(
        STANDARD_GRID["n_persons"], size_ranges["n_persons"]
    )
    cells = [Cell(i, p, True) for i in items_in for p in persons_in]
    if protocol.include_out_of_range:
        out = {(i, p) for i in items_out for p in persons_in + persons_out}
        out |= {(i, p) for i in items_in for p in persons_out}
        cells += [Cell(i, p, False) for i, p in sorted(out)]
    return cells


def cell_seed(seed: int, cell: Cell) -> int:
    """Seed for one cell, derived from the base seed and the cell's sizes.

    Keyed by the sizes rather than the cell's position, so adding or
    dropping cells (for example the out-of-range ones) leaves every other
    cell's data and posterior draws unchanged.
    """
    state = np.random.SeedSequence([int(seed), cell.n_items, cell.n_persons])
    return int(state.generate_state(1)[0])


# --- Posterior sampling for one cell ---------------------------------------

#: Item-parameter keys per model family, as the simulator and the
#: approximator name them (``simulators._REQUIRED_PRIORS``).
ITEM_PARAMETERS: dict[str, tuple[str, ...]] = {
    "1pl": ("b",),
    "2pl": ("a", "b"),
    "pcm": ("delta",),
    "gpcm": ("a", "delta"),
}


class CellFailedError(RuntimeError):
    """A cell could not be sampled, even after the out-of-memory retry."""

    def __init__(self, message: str, record: dict[str, Any]):
        super().__init__(message)
        self.record = record


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


#: The numba simulator kernel draws from its own RNG and ignores the seed,
#: so a card never uses it; recorded as ``numba_kernel`` in the record.
USE_NUMBA = False


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
    peaks = []
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


@dataclass
class CellSample:
    """Simulated data and posterior draws of one cell.

    Attributes
    ----------
    cell : Cell
        The cell.
    seed : int
        The cell seed (:func:`cell_seed`).
    data : dict
        The simulator output: true parameters and ``responses``.
    draws : dict
        Item-parameter draws per key, ``(n_datasets, n_draws, n_items)``,
        with a trailing threshold axis for ``delta``; float32.
    item, person : np.ndarray
        The randomly chosen item and person per dataset, ``(n_datasets,)``.
    threshold : np.ndarray or None
        For polytomous models, the randomly chosen threshold per dataset.
    retries : list of dict
        Out-of-memory retries: chunk start, its size, and the shrunk size
        (``retry_size``) used from that start on.
    seconds : float
        Wall time spent in posterior sampling.
    chunk_size : int
        Datasets per ``sample`` call. Part of the record because each chunk
        is seeded from its start index, so the draws depend on it.
    autoregressive_rows : str or None
        For an autoregressive item posterior, how its rows are batched:
        ``"whole chunk"`` (all ``(dataset, draw)`` rows of a chunk in one
        batch) or ``"at most <k>"`` after an out-of-memory shrink of the
        row batch. The batching changes the random stream, so the draws depend
        on it. ``None`` otherwise.
    chunk_rule : dict or None
        For ``chunk_size="auto"``, how the size was chosen
        (:func:`_auto_chunk_size`); ``None`` for a fixed size.
    """

    cell: Cell
    seed: int
    data: dict[str, np.ndarray]
    draws: dict[str, np.ndarray]
    item: np.ndarray
    person: np.ndarray
    threshold: np.ndarray | None
    retries: list[dict[str, Any]]
    seconds: float
    chunk_size: int
    autoregressive_rows: str | None = None
    chunk_rule: dict[str, Any] | None = None


def sample_cell(
    approximator: Any,
    model_spec: Mapping[str, Any],
    cell: Cell,
    protocol: EvaluationProtocol,
    *,
    trained_at: Any = None,
    chunk_size: int | str = 50,
    memory_share: float = MEMORY_SHARE,
) -> CellSample:
    """Simulate a cell's datasets and draw their item posteriors.

    The simulator is rebuilt from ``model_spec`` with the cell's fixed sizes
    (:func:`bayesflow_irt.build_simulator`). Datasets are sampled in chunks
    of ``chunk_size``; before each chunk the posterior seed is reset from the
    cell seed and the chunk's first dataset index
    (:func:`reset_posterior_seed`). ``chunk_size="auto"`` picks the size
    from free device memory (:func:`_auto_chunk_size`). On an out-of-memory
    error the chunk shrinks to :data:`CHUNK_SHRINK` of its size and is
    retried from the same start; the smaller size is kept for the rest of
    the cell and every shrink is recorded. An autoregressive item posterior
    samples a chunk's ``(dataset, draw)`` rows as one batch; when a
    one-dataset chunk runs out of memory, that row batch shrinks the same
    way. A chunk that runs out of memory at one dataset and one row raises
    :class:`CellFailedError`. The draws depend on the chunk boundaries and
    row batches, which the chunk size and the retries determine.

    Parameters
    ----------
    approximator
        A loaded approximator with ``sample(conditions=..., num_samples=...)``.
    model_spec : mapping
        The sidecar's ``model_spec``.
    cell : Cell
        The cell to sample.
    protocol : EvaluationProtocol
        Datasets, draws and base seed.
    trained_at
        The checkpoint's training time, forwarded to ``build_simulator``.
    chunk_size : int or "auto"
        Datasets per ``sample`` call, or ``"auto"``.
    memory_share : float
        With ``"auto"``, the share of free device memory a chunk may use.

    Returns
    -------
    CellSample
        Data, draws, chosen indices and the retry record.
    """
    from .simulators import build_simulator
    from .utils import is_oom_error, safe_empty_cache

    family = str(model_spec.get("model", "")).lower()
    if family not in ITEM_PARAMETERS:
        raise ValueError(f"Unsupported model family {model_spec.get('model')!r}.")
    if chunk_size != "auto" and (not isinstance(chunk_size, int) or chunk_size < 1):
        raise ValueError(f"chunk_size must be >= 1 or 'auto'; got {chunk_size!r}.")
    if not 0 < memory_share <= 1:
        raise ValueError(f"memory_share must be in (0, 1]; got {memory_share}.")
    seed = cell_seed(protocol.seed, cell)
    n = protocol.datasets_per_cell
    simulator = build_simulator(
        dict(model_spec),
        n_items=(cell.n_items, cell.n_items),
        n_persons=(cell.n_persons, cell.n_persons),
        seed=seed,
        trained_at=trained_at,
        use_numba=USE_NUMBA,
    )
    data = {k: np.asarray(v) for k, v in simulator.sample(n).items()}
    choice = np.random.default_rng([seed, 1])
    item = choice.integers(cell.n_items, size=n)
    person = choice.integers(cell.n_persons, size=n)
    keys = ITEM_PARAMETERS[family]
    threshold = None
    if "delta" in keys:
        threshold = choice.integers(data["delta"].shape[-1], size=n)

    draws: dict[str, np.ndarray] = {}
    retries: list[dict[str, Any]] = []

    autoregressive = bool(getattr(approximator, "autoregressive", False))
    # Cap on an autoregressive row batch; None is the whole chunk.
    row_cap: int | None = None

    def run(start: int, stop: int) -> None:
        reset_posterior_seed(approximator, _chunk_seed(seed, start))
        conditions = {k: v[start:stop] for k, v in data.items()}
        # The autoregressive sampler makes one flow call per item for every
        # 256 (dataset, draw) rows by default; that many small calls leave the
        # GPU idle. Pass the whole chunk as one batch of rows instead.
        rows = (stop - start) * protocol.posterior_draws
        extra = (
            {"condition_chunk_size": rows if row_cap is None else min(rows, row_cap)}
            if autoregressive
            else {}
        )
        post = approximator.sample(
            conditions=conditions, num_samples=protocol.posterior_draws, **extra
        )
        for key in keys:
            values = np.asarray(post[key], dtype=np.float32)
            if key not in draws:
                draws[key] = np.empty((n, *values.shape[1:]), dtype=np.float32)
            draws[key][start:stop] = values

    def failed(record: dict[str, Any], exc: BaseException) -> CellFailedError:
        sizes = ", ".join(str(record[k]) for k in ("size", "retry_size") if k in record)
        err = CellFailedError(
            f"Out of memory in cell {cell.n_items}x{cell.n_persons} at chunk "
            f"size(s) {sizes}.",
            {**record, "cell": dataclasses.asdict(cell)},
        )
        err.__cause__ = exc
        return err

    t0 = time.perf_counter()
    chunk_rule = None
    if chunk_size == "auto":
        chunk_size, chunk_rule = _auto_chunk_size(
            lambda k: run(n - k, n), n, memory_share
        )
    size = chunk_size
    start = 0
    while start < n:
        stop = min(start + size, n)
        oom = None
        try:
            run(start, stop)
        except Exception as exc:
            if not is_oom_error(exc):
                raise
            oom = exc
        if oom is None:
            start = stop
            continue
        # Shrink outside the except block: the live traceback would otherwise
        # keep the failed attempt's device tensors allocated.
        safe_empty_cache()
        record: dict[str, Any] = {"start": start, "size": stop - start}
        if stop - start < 2:
            rows = protocol.posterior_draws if row_cap is None else row_cap
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
    return CellSample(
        cell=cell,
        seed=seed,
        data=data,
        draws=draws,
        item=item,
        person=person,
        threshold=threshold,
        retries=retries,
        seconds=time.perf_counter() - t0,
        chunk_size=chunk_size,
        autoregressive_rows=(
            None
            if not autoregressive
            else "whole chunk"
            if row_cap is None
            else f"at most {row_cap}"
        ),
        chunk_rule=chunk_rule,
    )


def item_parameter_metrics(sample: CellSample) -> dict[str, dict[str, Any]]:
    """Per-parameter metrics for the randomly chosen item of each dataset.

    ``a`` is evaluated on the log scale, as ``log_a``: the scale TARP and the
    classical references use. Coverage and calibration are invariant to it;
    bias and RMSE are not. ``delta`` uses each dataset's chosen threshold.

    Returns
    -------
    dict
        Parameter name to :func:`bayesflow_irt.card_metrics.parameter_metrics`.
    """
    from .card_metrics import parameter_metrics

    rows = np.arange(sample.item.shape[0])
    out = {}
    for key, values in sample.draws.items():
        truth = np.asarray(sample.data[key], dtype=float)
        if key == "delta":
            d = values[rows, :, sample.item, sample.threshold]
            t = truth[rows, sample.item, sample.threshold]
        else:
            d = values[rows, :, sample.item]
            t = truth[rows, sample.item]
        d = np.asarray(d, dtype=float)
        name = key
        if key == "a":
            name, d, t = "log_a", np.log(d), np.log(t)
        out[name] = parameter_metrics(d, t)
    return out


# --- Joint metrics over all items of a dataset ------------------------------

#: Why the joint metrics are missing for polytomous models.
POLYTOMOUS_JOINT_OMITTED = (
    "TARP needs a data-dependent reference and TCC bias a test characteristic "
    "curve; neither exists for PCM/GPCM yet"
)


def _joint_item_draws(sample: CellSample) -> tuple[tuple[str, ...], Any, Any]:
    """Draws ``(n, S, P*I)`` and truth ``(n, P*I)``, ``a`` as ``log a``."""
    keys = tuple(sample.draws)
    draws = np.concatenate(
        [np.log(sample.draws[k]) if k == "a" else sample.draws[k] for k in keys],
        axis=-1,
    )
    truth = np.concatenate(
        [
            np.log(np.asarray(sample.data[k], dtype=float))
            if k == "a"
            else np.asarray(sample.data[k], dtype=float)
            for k in keys
        ],
        axis=-1,
    )
    return keys, draws, truth


#: Bootstrap replicates for the MCSE of the joint summaries.
JOINT_BOOTSTRAP = 200


def _bootstrap_se(stat: Any, n: int, rng: np.random.Generator) -> float:
    """Bootstrap SE over datasets of ``stat(indices)``."""
    reps = [stat(rng.integers(n, size=n)) for _ in range(JOINT_BOOTSTRAP)]
    return float(np.std(reps, ddof=1))


def _tarp_error(fractions: np.ndarray, levels: np.ndarray) -> float:
    ecp = (fractions[:, None] < levels).mean(axis=0)
    return float(np.median(np.abs(ecp - levels)))


def _tarp_metrics(
    draws: np.ndarray,
    truth: np.ndarray,
    refs: np.ndarray | None,
    seed: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    from .card_metrics import SBC_GRID
    from .sbc import compute_tarp_coverage

    res = compute_tarp_coverage(
        draws,
        truth,
        reference_points=refs,
        resolution=len(SBC_GRID),
        seed=seed,
    )
    levels = res["credibility_levels"]
    f = res["coverage_fractions"]
    n = f.shape[0]
    ecp = res["expected_coverage"]
    # Null floor: the same statistic for an exact posterior, whose n_draws * f
    # is discrete uniform on {0, ..., n_draws} (Talts et al., 2018, Theorem 1,
    # applied to the distance as the scalar function).
    n_draws = res["n_draws"]
    floor = np.mean(
        [
            _tarp_error(rng.integers(n_draws + 1, size=n) / n_draws, levels)
            for _ in range(JOINT_BOOTSTRAP)
        ]
    )
    return {
        "tarp_error": (
            res["tarp_error"],
            _bootstrap_se(lambda idx: _tarp_error(f[idx], levels), n, rng),
        ),
        "null_floor": float(floor),
        "max_deviation": res["max_deviation"],
        "reference_mode": res["reference_mode"],
        "n_draws": n_draws,
        "levels": levels.tolist(),
        "expected_coverage": ecp.tolist(),
        "expected_coverage_se": np.sqrt(ecp * (1 - ecp) / n).tolist(),
    }


def _tcc(log_a: np.ndarray | None, b: np.ndarray, thetas: np.ndarray) -> Any:
    """Test characteristic curve, ``(..., I) -> (..., n_theta)``."""
    a = 1.0 if log_a is None else np.exp(log_a)[..., None]
    logits = a * (thetas - b[..., None])
    return (1.0 / (1.0 + np.exp(-logits))).sum(axis=-2)


def joint_metrics(
    sample: CellSample,
    thetas: Sequence[float] = (-1.0, 0.0, 1.0),
    *,
    chunk_size: int = 50,
) -> dict[str, Any]:
    """Metrics over all items of each dataset: TARP and TCC bias.

    **TARP** (Lemos et al., 2023) on the joint item vector, ``a`` as
    ``log a``, with two reference rules: random references (seeded from the
    cell seed) and data-dependent classical references
    (:func:`bayesflow_irt.sbc.classical_reference_points`). Section 4.3 of
    Lemos et al. shows only the second detects a posterior that ignores the
    data. The ECP curve is stored on the :data:`card_metrics.SBC_GRID`
    levels for the diagnostic panel.

    **TCC bias**: ``tcc_z_bias`` and ``tcc_bias`` exactly as
    :func:`bayesflow_irt.hpo.make_tcc_bias_metric` defines them (Birnbaum,
    1968; Schad et al., 2021; Modrák et al., 2025), plus the per-theta means.

    The summaries (a median over levels, a max over thetas) are non-smooth,
    so their MCSE is a bootstrap over datasets (:data:`JOINT_BOOTSTRAP`
    replicates, seeded from the cell seed) rather than the delete-one
    jackknife; Siepe et al. (2024) name bootstrap intervals as an MCSE
    option. Checked by replication (``tests/test_card_joint.py``): it
    matches the spread of the summary for a miscalibrated posterior, and
    overstates it by up to about 2x near the null, where the summary is
    pinned against its floor. Both summaries sit above zero even for an
    exact posterior, so
    TARP also stores ``null_floor``, the expected ``tarp_error`` of an exact
    posterior at this cell's size and draw count. The linear pieces carry
    analytic SEs (Siepe et al., 2024, Table 3): the ECP at each level and
    the per-theta TCC means. PCM/GPCM return ``{"omitted": reason}``
    (:data:`POLYTOMOUS_JOINT_OMITTED`).

    Parameters
    ----------
    sample : CellSample
        The cell's data and draws.
    thetas : sequence of float
        Abilities at which the TCC is evaluated.
    chunk_size : int
        Datasets per TCC evaluation, bounding memory.

    Returns
    -------
    dict
        ``tarp`` (``random`` and ``classical``) and ``tcc``; each scalar is a
        ``(value, mcse)`` pair.
    """
    from .sbc import classical_reference_points

    if "delta" in sample.draws:
        return {"omitted": POLYTOMOUS_JOINT_OMITTED}
    if set(sample.draws) not in ({"b"}, {"a", "b"}):
        raise ValueError(f"Unexpected item parameters {sorted(sample.draws)}.")
    rng = np.random.default_rng([sample.seed, 2])
    keys, draws, truth = _joint_item_draws(sample)
    refs = classical_reference_points(sample.data["responses"], keys)
    tarp = {
        "random": _tarp_metrics(draws, truth, None, sample.seed, rng),
        "classical": _tarp_metrics(draws, truth, refs, sample.seed, rng),
    }

    theta = np.asarray(thetas, dtype=float)
    n_items = sample.cell.n_items
    n, n_draws = draws.shape[:2]
    if n_draws < 4:
        raise ValueError("TCC bias needs at least 4 posterior draws.")
    z = np.empty((n, theta.size))
    d = np.empty((n, theta.size))

    def split(x: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
        if keys == ("b",):
            return None, x
        ia, ib = keys.index("a"), keys.index("b")
        block = slice(ia * n_items, (ia + 1) * n_items)
        return x[..., block], x[..., ib * n_items : (ib + 1) * n_items]

    true_tcc = _tcc(*split(truth), theta)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        tcc = _tcc(*split(draws[start:stop].astype(float)), theta)
        true = true_tcc[start:stop]
        d[start:stop] = (tcc.mean(axis=1) - true) / n_items
        # Disjoint halves: mean from even draws, sd from odd draws.
        sd = tcc[:, 1::2].std(axis=1, ddof=1)
        if np.any(np.ptp(tcc[:, 1::2], axis=1) == 0):
            raise ValueError("A posterior TCC has zero spread.")
        z[start:stop] = (tcc[:, 0::2].mean(axis=1) - true) / sd

    def max_abs(values: np.ndarray) -> tuple[float, float]:
        def stat(idx: np.ndarray) -> float:
            return float(np.max(np.abs(values[idx].mean(axis=0))))

        return stat(np.arange(n)), _bootstrap_se(stat, n, rng)

    def se(values: np.ndarray) -> list[float]:
        return (values.std(axis=0, ddof=1) / np.sqrt(n)).tolist()

    tcc_out = {
        "thetas": theta.tolist(),
        "tcc_z_bias": max_abs(z),
        "tcc_bias": max_abs(d),
        "z_mean": z.mean(axis=0).tolist(),
        "z_mean_se": se(z),
        "d_mean": d.mean(axis=0).tolist(),
        "d_mean_se": se(d),
    }
    return {"tarp": tarp, "tcc": tcc_out}


# --- Person metrics ----------------------------------------------------------

#: PersonScorer model names per item-parameter set.
_SCORER_MODEL = {
    frozenset({"b"}): "1PL",
    frozenset({"a", "b"}): "2PL",
    frozenset({"delta"}): "PCM",
    frozenset({"a", "delta"}): "GPCM",
}

#: Expected scored persons per theta bin in the smallest cell, and the most
#: bins. A package choice; see :func:`theta_bin_count`.
THETA_BIN_TARGET = 100
THETA_BIN_MAX = 10


def theta_bin_count(protocol: EvaluationProtocol) -> int:
    """Number of equal-probability theta bins, fixed once per protocol.

    Chosen so the smallest cell of :data:`STANDARD_GRID` expects at least
    :data:`THETA_BIN_TARGET` scored persons per bin, at most
    :data:`THETA_BIN_MAX` bins. Every cell uses the same bins, so cells
    stay comparable.
    """
    smallest = protocol.datasets_per_cell * min(
        protocol.persons_scored, min(STANDARD_GRID["n_persons"])
    )
    return int(np.clip(smallest // THETA_BIN_TARGET, 1, THETA_BIN_MAX))


def theta_bin_edges(n_bins: int) -> np.ndarray:
    """Equal-probability edges in prior SDs: standard normal quantiles.

    Feuerstahler (2018) reports fixed generating values instead, which a
    random-theta simulation cannot.
    """
    from scipy.stats import norm

    return norm.ppf(np.arange(n_bins + 1) / n_bins)


def _cluster_mean(
    values: np.ndarray, cluster: np.ndarray
) -> tuple[float, float | None, int]:
    """Mean over persons with a cluster-robust SE, clusters = datasets.

    The mean is the OLS estimate of an intercept-only model, and the SE its
    cluster-robust sandwich estimate with the finite-sample factor
    ``G / (G - 1)`` (Cameron & Miller, 2015, Eqs. 10-12, with K = 1):
    ``SE = sqrt(G / (G - 1) * sum_g (s_g - n_g * mean)^2) / N``, ``s_g`` the
    cluster sum, ``n_g`` its persons, ``N`` all persons. Persons of one
    dataset share its item draws, so they are not independent. ``G`` counts
    the clusters with at least one person; the SE is None below two.
    """
    ids, inv = np.unique(cluster, return_inverse=True)
    g = ids.size
    mean = float(values.mean())
    if g < 2:
        return mean, None, g
    s = np.bincount(inv, weights=values, minlength=g)
    n_g = np.bincount(inv, minlength=g)
    var = g / (g - 1) * np.sum((s - n_g * mean) ** 2) / values.size**2
    return mean, float(np.sqrt(var)), g


#: Rectangular quadrature of RIMSE: Q = 101 nodes on [-5, 5] (Falk and
#: Feuerstahler, 2021), in prior SDs around the prior mean.
RIMSE_NODES = np.linspace(-5.0, 5.0, 101)


def _item_curve(
    theta: np.ndarray, a: np.ndarray | None, b: np.ndarray, polytomous: bool
) -> np.ndarray:
    """Item curve on ``theta`` ``(Q,)``, shape ``(..., Q)``.

    Dichotomous: the IRF ``logistic(a (theta - b))``, ``b`` of shape
    ``(...)``. Polytomous: the expected item score divided by ``K - 1``,
    ``b`` the step difficulties ``(..., K-1)``, with the category
    probabilities of ``scoring._log_likelihood_pcm`` / ``_gpcm``.
    """
    if not polytomous:
        scale = 1.0 if a is None else a[..., None]
        return 1.0 / (1.0 + np.exp(-scale * (theta - b[..., None])))
    scale = 1.0 if a is None else a[..., None, None]
    steps = scale * (theta[:, None] - b[..., None, :])  # (..., Q, K-1)
    cum = np.concatenate(
        [np.zeros((*steps.shape[:-1], 1)), np.cumsum(steps, axis=-1)], axis=-1
    )
    cum -= cum.max(axis=-1, keepdims=True)
    p = np.exp(cum)
    p /= p.sum(axis=-1, keepdims=True)
    return (p * np.arange(p.shape[-1])).sum(axis=-1) / (p.shape[-1] - 1)


def _test_information(
    theta: np.ndarray, a: np.ndarray | None, b: np.ndarray
) -> np.ndarray:
    """Dichotomous test information ``J(theta)``, ``(..., I) -> (..., Q)``.

    Lee (2026, Appendix A, Eqs. A.2-A.4): ``J_i = a_i^2 P_i (1 - P_i)``
    summed over items; ``a = 1`` for the 1PL.
    """
    p = _item_curve(theta, a, b, False)  # (..., I, Q)
    scale = 1.0 if a is None else a[..., None] ** 2
    return (scale * p * (1.0 - p)).sum(axis=-2)


def _mean_se_range(values: np.ndarray) -> dict[str, Any]:
    """Mean with analytic MCSE, and the 2.5-97.5% range across datasets.

    Fewer than two values give nulls (no MCSE exists).
    """
    n = values.shape[0]
    if n < 2:
        return {"mean": (None, None), "range_95": [None, None], "n": n}
    return {
        "mean": (float(values.mean()), float(values.std(ddof=1) / np.sqrt(n))),
        "range_95": np.percentile(values, [2.5, 97.5]).tolist(),
    }


def person_metrics(
    sample: CellSample,
    protocol: EvaluationProtocol,
    *,
    theta_prior: tuple[float, float] = (0.0, 1.0),
    chunk_size: int = 50,
) -> dict[str, Any]:
    """Person metrics and item-curve recovery for one cell.

    Each dataset's persons are scored with
    :class:`bayesflow_irt.PersonScorer` from that dataset's item draws,
    returning joint theta draws (``return_posterior="draws"``). Scored are
    the dataset's random person and a seeded subsample of the others,
    ``protocol.persons_scored`` in all.

    - ``theta``: :func:`card_metrics.parameter_metrics` for the random
      person.
    - ``conditional``: per bin of the scored persons' true theta
      (:func:`theta_bin_edges`, :func:`theta_bin_count` bins), the count,
      the datasets contributing (``clusters``), bias of the posterior mean,
      95% central-interval coverage and width, and the mean posterior SD,
      with cluster-robust MCSEs (:func:`_cluster_mean`). Below-nominal
      coverage in the outer bins is the expected shrinkage of a correct
      posterior (Feuerstahler, 2018): a description, not a check.
    - ``reliability``: per dataset over the scored persons,
      ``1 - mean posterior variance / prior variance`` (de la Torre et al.,
      Eq. 19) and ``corr(theta, posterior mean)^2``. For a calibrated
      posterior the two agree in expectation, so their difference is
      reported too. Context, not a calibration check: both are subsample
      estimates and the identity holds only over the prior.
    - ``rimse``: for the random item, ``100 sqrt(sum_q (P_hat - P)^2 w_q /
      sum_q w_q)`` (Falk and Feuerstahler, 2021; Ramsay, 1991, without the
      factor 100) on :data:`RIMSE_NODES`, ``P_hat`` the posterior-mean
      curve and ``w`` the theta prior density. Polytomous items use the
      expected score / (K - 1), a labelled extension.
    - 1PL/2PL only, from each dataset's generating items: the
      information-based marginal reliability ``w = sigma^2 / (sigma^2 +
      MSEM)`` with ``MSEM = E[1 / J(theta)]`` over the theta prior (Lee,
      2026, Eq. A.5; rectangular quadrature on :data:`RIMSE_NODES`), how
      informative the simulated tests were; and in the conditional table
      the conditional SE ``1 / sqrt(J(theta))`` at each scored person's true
      theta, next to the posterior SD. PCM/GPCM omit both (no source read
      for their information function yet).

    Parameters
    ----------
    sample : CellSample
        The cell's data and item draws.
    protocol : EvaluationProtocol
        Supplies ``persons_scored``.
    theta_prior : (float, float)
        Mean and SD of the normal theta prior (the model's ``theta`` prior).
    chunk_size : int
        Datasets per RIMSE evaluation, bounding memory.
    """
    from .card_metrics import parameter_metrics
    from .scoring import PersonScorer

    keys = frozenset(sample.draws)
    if keys not in _SCORER_MODEL:
        raise ValueError(f"Unexpected item parameters {sorted(keys)}.")
    model_type = _SCORER_MODEL[keys]
    polytomous = "delta" in keys
    gpu_scoring = _torch_cuda()
    loc, sd = (float(v) for v in theta_prior)
    n = sample.item.shape[0]
    n_persons = sample.cell.n_persons
    m = min(protocol.persons_scored, n_persons)
    theta_true = np.asarray(sample.data["theta"], dtype=float).reshape(n, n_persons)
    responses = np.asarray(sample.data["responses"], dtype=float)
    pick = np.random.default_rng([sample.seed, 3])

    n_draws = next(iter(sample.draws.values())).shape[1]
    draws_rand = np.empty((n, n_draws))
    scored = np.empty((n, m), dtype=int)
    p_mean, p_lo, p_hi, p_sd = (np.empty((n, m)) for _ in range(4))
    rel_var = np.empty(n)
    rel_corr = np.full(n, np.nan)
    for k in range(n):
        others = np.delete(np.arange(n_persons), sample.person[k])
        persons = np.concatenate(
            [[sample.person[k]], pick.choice(others, size=m - 1, replace=False)]
        )
        scorer = PersonScorer(
            None,
            item_posteriors={
                key: np.asarray(v[k], dtype=float) for key, v in sample.draws.items()
            },
            model_type=model_type,
            theta_prior=(loc, sd),
        )
        seed = int(np.random.SeedSequence([sample.seed, 4, k]).generate_state(1)[0])
        res = scorer.score(
            responses[k, persons],
            # Only the draws, mean and SD are used.
            intervals=None,
            return_posterior="draws",
            seed=seed % 2**31,
            # The torch path agrees with NumPy to about 1e-6 (float32).
            device="cuda" if gpu_scoring else None,
            torch_dtype="float32",
        )
        draws_rand[k] = res.theta_draws[:, 0]
        scored[k] = persons
        p_mean[k] = res.theta_draws.mean(axis=0)
        p_lo[k], p_hi[k] = np.quantile(res.theta_draws, [0.025, 0.975], axis=0)
        p_sd[k] = res.theta_draws.std(axis=0, ddof=1)
        rel_var[k] = 1.0 - np.mean(res.sd**2) / sd**2
        if m > 2 and np.std(res.mean) > 0:
            rel_corr[k] = np.corrcoef(theta_true[k, persons], res.mean)[0, 1] ** 2

    truth_rand = theta_true[np.arange(n), sample.person]
    a_true = None
    if "a" in sample.data and not polytomous:
        a_true = np.asarray(sample.data["a"], dtype=float)
    truth = np.take_along_axis(theta_true, scored, axis=1)  # (n, m)
    cond_se = None
    if not polytomous:
        b_true = np.asarray(sample.data["b"], dtype=float)
        cond_se = np.empty((n, m))
        for k in range(n):
            info = _test_information(
                truth[k], None if a_true is None else a_true[k], b_true[k]
            )
            cond_se[k] = 1.0 / np.sqrt(info)
    theta_out = parameter_metrics(draws_rand, truth_rand)

    covered = ((p_lo <= truth) & (truth <= p_hi)).astype(float)
    per_person = {
        "bias": p_mean - truth,
        "coverage_0.95": covered,
        "width_0.95": p_hi - p_lo,
        "posterior_sd": p_sd,
        **({} if cond_se is None else {"conditional_se": cond_se}),
    }
    z = ((truth - loc) / sd).ravel()
    dataset = np.repeat(np.arange(n), m)
    edges_z = theta_bin_edges(theta_bin_count(protocol))
    bins = []
    for lo_edge, hi_edge in zip(edges_z[:-1], edges_z[1:]):
        rows = (z >= lo_edge) & (z < hi_edge)
        count = int(rows.sum())
        edges = [float(e) if np.isfinite(e) else None for e in (lo_edge, hi_edge)]
        entry: dict[str, Any] = {"edges": edges, "n": count, "clusters": 0}
        if count >= 2:
            for name, values in per_person.items():
                mean, se, g = _cluster_mean(values.ravel()[rows], dataset[rows])
                entry[name] = (mean, se)
                entry["clusters"] = g
        else:
            entry["clusters"] = int(np.unique(dataset[rows]).size)
        bins.append(entry)

    ok = np.isfinite(rel_corr)
    reliability = {
        "persons_scored": m,
        "posterior_variance": _mean_se_range(rel_var),
        "squared_correlation": _mean_se_range(rel_corr[ok]),
        "difference": _mean_se_range(rel_var[ok] - rel_corr[ok]),
    }

    nodes = loc + sd * RIMSE_NODES
    weight = np.exp(-0.5 * RIMSE_NODES**2)
    if polytomous:
        reliability["information"] = {"omitted": "no PCM/GPCM information source"}
    else:
        info = _test_information(nodes, a_true, b_true)  # (n, Q)
        msem = (weight / info).sum(axis=-1) / weight.sum()
        reliability["information"] = {
            **_mean_se_range(sd**2 / (sd**2 + msem)),
            "basis": "all generating items; missing responses not masked",
        }
    a_draws = sample.draws.get("a")
    b_key = "delta" if polytomous else "b"
    rimse = np.empty(n)
    for start in range(0, n, chunk_size):
        rows = np.arange(start, min(start + chunk_size, n))
        item = sample.item[rows]
        a_d = a_t = None
        if a_draws is not None:
            a_d = a_draws[rows, :, item].astype(float)
            a_t = np.asarray(sample.data["a"], dtype=float)[rows, item]
        b_d = sample.draws[b_key][rows, :, item].astype(float)
        b_t = np.asarray(sample.data[b_key], dtype=float)[rows, item]
        p_hat = _item_curve(nodes, a_d, b_d, polytomous).mean(axis=1)
        p_true = _item_curve(nodes, a_t, b_t, polytomous)
        rimse[rows] = 100.0 * np.sqrt(
            ((p_hat - p_true) ** 2 * weight).sum(axis=-1) / weight.sum()
        )

    return {
        # The torch float32 path agrees with NumPy to about 1e-6 in mean/SD.
        "scorer": "cuda float32" if gpu_scoring else "numpy float64",
        "theta": theta_out,
        "conditional": bins,
        "conditional_basis": (
            "all scored persons; equal-probability theta-prior bins; "
            "cluster-robust MCSE by dataset (Cameron & Miller, 2015)"
        ),
        **(
            {}
            if polytomous
            else {
                "conditional_se_basis": (
                    "all generating items; missing responses not masked"
                )
            }
        ),
        "reliability": reliability,
        "rimse": {
            **_mean_se_range(rimse),
            "form": "Falk and Feuerstahler (2021): 100 x Ramsay (1991)",
            "item_curve": "expected score / (K-1)" if polytomous else "IRF",
        },
    }


# --- Orchestrator, evaluation.json and CLI ----------------------------------

#: Format version of ``evaluation.json``, independent of the protocol version.
EVALUATION_FORMAT = 3


def network_config(approximator: Any) -> dict[str, Any] | None:
    """The approximator's full configuration, defaults included (None if it
    is not a Keras object).

    ``keras.saving.serialize_keras_object`` gives the same nested
    ``{module, class_name, config, registered_name}`` tree that
    ``Model.save`` writes as the ``.keras`` archive's ``config.json`` (Keras
    serialization utilities), so the card can list every setting, not only
    the ones a training config set. A JSON round trip keeps it serializable.
    """
    import json

    import keras

    try:
        config = json.loads(
            json.dumps(keras.saving.serialize_keras_object(approximator), default=str)
        )
    except TypeError:  # not a Keras object (a caller-supplied stand-in)
        return None
    try:
        n_parameters = int(approximator.count_params())
    except ValueError:  # an unbuilt model has no weights to count
        n_parameters = None
    return {
        "config": config,
        "n_parameters": n_parameters,
        "source": "serialize_keras_object of the loaded checkpoint",
    }


#: ``PersonScorer.score``'s warning for all-correct/all-incorrect patterns.
#: 1PL/2PL say "all-correct ...", PCM/GPCM "all-highest-category ...".
_EXTREME_PATTERNS = re.compile(r"(\d+) response pattern\(s\) are all-")


def _theta_prior(model_spec: Mapping[str, Any]) -> tuple[float, float]:
    prior = (model_spec.get("priors") or {}).get("theta")
    if not prior or prior.get("name") != "normal":
        raise ValueError(f"Expected a normal theta prior in model_spec; got {prior!r}.")
    # Defaults match the "normal" sampler in priors.py.
    return float(prior.get("loc", 0.0)), float(prior.get("scale", 1.0))


def _write_json(path: Path, record: Mapping[str, Any]) -> None:
    """Strict JSON (non-finite values become null), written atomically."""
    import json
    import os

    from .metadata import _sanitize_for_json

    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(_sanitize_for_json(record), indent=1, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def evaluate_cell(
    approximator: Any,
    model_spec: Mapping[str, Any],
    cell: Cell,
    protocol: EvaluationProtocol,
    *,
    trained_at: Any = None,
    chunk_size: int | str = 50,
    memory_share: float = MEMORY_SHARE,
) -> dict[str, Any]:
    """Sample one cell and compute all its metrics.

    Returns the cell's ``evaluation.json`` entry: the cell, its seed, the
    sampling record (seconds, chunk size, out-of-memory retries) and the
    item, joint and person metrics. A cell whose sampling fails even after
    the out-of-memory retry, or whose metrics raise, gets a ``failed``
    record (``stage`` "sampling" or "metrics"; metrics finished before the
    error are kept) instead of aborting the run. ``PersonScorer``'s warning
    about extreme response patterns (all-correct/all-incorrect, or
    all-highest/all-lowest category) is counted instead of
    shown: ``persons.extreme_patterns`` is the number of scored persons
    with such a pattern in the cell; other warnings pass through.
    """
    entry: dict[str, Any] = {"cell": dataclasses.asdict(cell)}
    failure = None
    try:
        sample = sample_cell(
            approximator,
            model_spec,
            cell,
            protocol,
            trained_at=trained_at,
            chunk_size=chunk_size,
            memory_share=memory_share,
        )
    except CellFailedError as exc:
        failure = {"stage": "sampling", "message": str(exc), **exc.record}
    if failure is not None:
        # The exception chain's frames hold the failed attempt's tensors in
        # reference cycles; collect them before freeing the device cache so
        # the next cell does not inherit the allocation.
        import gc

        from .utils import safe_empty_cache

        gc.collect()
        safe_empty_cache()
        return {**entry, "failed": failure}
    entry.update(
        seed=sample.seed,
        sampling={
            "seconds": sample.seconds,
            "chunk_size": sample.chunk_size,
            "autoregressive_rows": sample.autoregressive_rows,
            "chunk_rule": sample.chunk_rule,
            "retries": sample.retries,
        },
    )
    t0 = time.perf_counter()
    try:
        _cell_metrics(sample, model_spec, protocol, entry)
    except Exception as exc:  # noqa: BLE001 - record the cell, keep the run
        entry["failed"] = {
            "stage": "metrics",
            "message": f"{type(exc).__name__}: {exc}",
        }
    entry["metrics_seconds"] = time.perf_counter() - t0
    return entry


def _cell_metrics(
    sample: CellSample,
    model_spec: Mapping[str, Any],
    protocol: EvaluationProtocol,
    entry: dict[str, Any],
) -> None:
    """Add the item, joint and person metrics to ``entry`` one by one."""
    entry["items"] = item_parameter_metrics(sample)
    entry["joint"] = joint_metrics(sample)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        persons = person_metrics(sample, protocol, theta_prior=_theta_prior(model_spec))
    extreme = 0
    for w in caught:
        match = _EXTREME_PATTERNS.search(str(w.message))
        if match:
            extreme += int(match.group(1))
        else:
            warnings.warn_explicit(
                w.message, w.category, w.filename, w.lineno, source=w.source
            )
    persons["extreme_patterns"] = extreme
    entry["persons"] = persons


def evaluate_for_card(
    checkpoint: str | Path,
    metadata: str | Path | None = None,
    protocol: EvaluationProtocol | None = None,
    *,
    out: str | Path | None = None,
    approximator: Any = None,
    chunk_size: int | str = "auto",
    memory_share: float = MEMORY_SHARE,
    progress: bool = False,
) -> dict[str, Any]:
    """Evaluate a checkpoint on its card cells and write ``evaluation.json``.

    A thin orchestrator: it plans the cells from the sidecar's trained range
    (:func:`plan_cells`) and runs :func:`evaluate_cell` per cell,
    on data from the model's own simulator rebuilt from ``model_spec``. The
    record points back to the
    sidecar by ``metadata_uuid`` and the checkpoint's SHA-256. A sidecar that
    records ``checkpoint_sha256`` must match the file. Without a hash, a
    recorded ``provenance.checkpoint_filename`` must name the checkpoint;
    ``checkpoint.match`` is then "filename only", as it is when the sidecar
    follows the ``<stem>_metadata.json`` convention, and "unverified"
    otherwise. The file is rewritten after every cell, so a run that stops keeps
    its finished cells; ``complete`` is false until the last.

    Parameters
    ----------
    checkpoint : path
        The full ``.keras`` checkpoint.
    metadata : path, optional
        The sidecar; default ``<checkpoint stem>_metadata.json``.
    protocol : EvaluationProtocol, optional
        Default ``EvaluationProtocol.preset("validation")``.
    out : path, optional
        Where to write: a ``.json`` path, or a directory that gets
        ``evaluation.json``. Nothing is written when omitted.
    approximator : optional
        An already-loaded approximator for ``checkpoint``; loaded with
        ``keras.saving.load_model`` when omitted.
    chunk_size : int or "auto"
        Datasets per ``sample`` call (:func:`sample_cell`).
    memory_share : float
        With ``"auto"``, the share of free device memory a chunk may use.
    progress : bool
        Print one line per finished cell.

    Returns
    -------
    dict
        The evaluation record.
    """
    from .metadata import MetadataTracker, _sha256_file, session_info

    checkpoint = Path(checkpoint)
    meta_path = (
        Path(metadata)
        if metadata is not None
        else checkpoint.with_name(checkpoint.stem + "_metadata.json")
    )
    md = MetadataTracker.load(meta_path).to_dict()
    spec = md["model_spec"]
    if not spec.get("size_ranges") or not spec.get("priors"):
        raise ValueError(
            f"{meta_path.name}: model_spec lacks size_ranges or priors, so the "
            "simulator cannot be rebuilt and the model stays unevaluated."
        )
    _theta_prior(spec)
    provenance = md.get("provenance") or {}
    sha = _sha256_file(checkpoint)
    recorded = provenance.get("checkpoint_sha256")
    if recorded is not None and recorded != sha:
        raise ValueError(
            f"{checkpoint.name} does not match the SHA-256 recorded in "
            f"{meta_path.name}."
        )
    if recorded is not None:
        match = "sha256"
    else:
        # No hash: fall back to the recorded checkpoint filename, else the
        # sidecar naming convention; neither proves the pairing.
        named = provenance.get("checkpoint_filename")
        if named is not None and Path(named).name != checkpoint.name:
            raise ValueError(
                f"{meta_path.name} names checkpoint {named!r}, not {checkpoint.name}."
            )
        conventional = meta_path.name == checkpoint.stem + "_metadata.json"
        match = "filename only" if named or conventional else "unverified"
    protocol = protocol or EvaluationProtocol.preset("validation")
    if approximator is None:
        import keras

        approximator = keras.saving.load_model(str(checkpoint), compile=False)

    path = None
    if out is not None:
        path = Path(out)
        if path.suffix != ".json":
            path.mkdir(parents=True, exist_ok=True)
            path = path / "evaluation.json"
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
    if chunk_size != "auto" and (not isinstance(chunk_size, int) or chunk_size < 1):
        raise ValueError(f"chunk_size must be >= 1 or 'auto'; got {chunk_size!r}.")
    if not 0 < memory_share <= 1:
        raise ValueError(f"memory_share must be in (0, 1]; got {memory_share}.")
    cells = plan_cells(spec["size_ranges"], protocol)
    record: dict[str, Any] = {
        "evaluation_format": EVALUATION_FORMAT,
        "complete": False,
        "checkpoint": {
            "filename": checkpoint.name,
            "sha256": sha,
            "match": match,
        },
        "metadata": {
            "filename": meta_path.name,
            "metadata_uuid": md.get("metadata_uuid"),
            "schema_version": md.get("schema_version"),
        },
        "model_spec": spec,
        "protocol": protocol.to_dict(),
        "numba_kernel": USE_NUMBA,
        "draws_validated_for": list(DRAWS_VALIDATED_FOR),
        "chunk_size": chunk_size,
        "session_info": session_info(),
        "network": network_config(approximator),
        "cells": [],
    }
    t0 = time.perf_counter()
    for i, cell in enumerate(cells):
        entry = evaluate_cell(
            approximator,
            spec,
            cell,
            protocol,
            trained_at=provenance.get("timestamp"),
            chunk_size=chunk_size,
            memory_share=memory_share,
        )
        record["cells"].append(entry)
        if path is not None:
            _write_json(path, record)
        if progress:
            state = "FAILED" if "failed" in entry else "ok"
            print(
                f"[{i + 1}/{len(cells)}] {cell.n_items}x{cell.n_persons} "
                f"{state} ({time.perf_counter() - t0:.0f} s)",
                flush=True,
            )
    record["complete"] = True
    record["failed_cells"] = [c["cell"] for c in record["cells"] if "failed" in c]
    record["seconds"] = time.perf_counter() - t0
    if path is not None:
        _write_json(path, record)
    return record


def main(argv: Sequence[str] | None = None) -> None:
    """CLI: ``python -m bayesflow_irt.card CHECKPOINT [options]``."""
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m bayesflow_irt.card",
        description="Evaluate a .keras checkpoint for its model card.",
    )
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--metadata", type=Path, default=None)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory or .json path; default <checkpoint stem>_card/.",
    )
    p.add_argument("--preset", choices=sorted(PRESETS), default="validation")
    p.add_argument("--datasets-per-cell", type=int)
    p.add_argument("--posterior-draws", type=int)
    p.add_argument("--persons-scored", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument(
        "--include-out-of-range", action=argparse.BooleanOptionalAction, default=None
    )
    p.add_argument(
        "--chunk-size",
        type=lambda v: v if v == "auto" else int(v),
        default="auto",
        help="Datasets per sample call, or 'auto' (from free GPU memory).",
    )
    p.add_argument(
        "--memory-share",
        type=float,
        default=MEMORY_SHARE,
        help="With --chunk-size auto, the share of free GPU memory a chunk "
        "may use; lower it on a GPU shared with other jobs.",
    )
    p.add_argument(
        "--render",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render model_card.md and figures after evaluating (default on).",
    )
    args = p.parse_args(argv)
    fields = (
        "datasets_per_cell",
        "posterior_draws",
        "persons_scored",
        "seed",
        "include_out_of_range",
    )
    overrides = {k: getattr(args, k) for k in fields if getattr(args, k) is not None}
    protocol = EvaluationProtocol.preset(args.preset, **overrides)
    out = args.out or args.checkpoint.with_name(args.checkpoint.stem + "_card")
    evaluate_for_card(
        args.checkpoint,
        args.metadata,
        protocol,
        out=out,
        chunk_size=args.chunk_size,
        memory_share=args.memory_share,
        progress=True,
    )
    if args.render:
        from .card_render import render_model_card

        if out.suffix == ".json" and out.name != "evaluation.json":
            print("Not rendering: the record is not named evaluation.json.")
            return
        card_dir = out.parent if out.suffix == ".json" else out
        # The same sidecar evaluate_for_card read, wherever --out points.
        meta = args.metadata or args.checkpoint.with_name(
            args.checkpoint.stem + "_metadata.json"
        )
        try:
            print(render_model_card(card_dir, meta))
        except Exception as exc:  # the evaluation itself succeeded
            warnings.warn(
                f"evaluation.json was written, but rendering the card failed: "
                f"{type(exc).__name__}: {exc}. Rerun with "
                f"python -m bayesflow_irt.card_render {card_dir}.",
                stacklevel=2,
            )


if __name__ == "__main__":
    main()
