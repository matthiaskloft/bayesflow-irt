"""IRT data-generating processes (simulators) for BayesFlow.

Provides factory functions that return ``bayesflow.Simulator`` instances for
1PL and 2PL IRT models.  Each simulator produces a dictionary with keys
suitable for the adapters in :mod:`bayesflow_irt.adapters`.

Design note — variable-size batches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``n_items`` and ``n_persons`` are ranges ``(lo, hi)`` with ``hi >= lo``.
Use ``hi == lo`` for a fixed size (e.g. validation edge conditions).
Sizes are drawn **once per batch** (not per sample). This keeps every
sample in a batch the same shape so NumPy can stack them into a clean
tensor — no padding or masking needed. Across batches the sizes vary.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any

import numpy as np
from bayesflow.simulators import Simulator
from bayesflow.utils.decorators import allow_batch_size
from scipy.special import expit as _sigmoid

from .priors import (
    ParameterRole,
    Prior,
    PriorSpec,
    make_incremental_prior,
    resolve_prior,
)

try:
    from ._numba_kernels import binary_response_kernel, categorical_response_kernel

    _HAS_NUMBA = True
    _NUMBA_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001 - numba installed-but-broken (e.g. a
    # mismatched/corrupt llvmlite binary raising OSError or RuntimeError, not
    # just ImportError) must not break importing bayesflow_irt when
    # use_numba is never requested; the error only surfaces if a caller
    # actually asks for use_numba=True, via _check_numba_available below.
    _HAS_NUMBA = False
    _NUMBA_IMPORT_ERROR = exc

_VALID_PARAMETERIZATIONS = ("standard", "loading")
_VALID_RESPONSE_DTYPES = {"float32", "uint8"}


def _check_numba_available(use_numba: bool) -> None:
    """Raise if ``use_numba=True`` was requested but numba isn't usable."""
    if not use_numba or _HAS_NUMBA:
        return
    if isinstance(_NUMBA_IMPORT_ERROR, ImportError):
        raise ValueError(
            "use_numba=True but numba is not installed. "
            "Install it with `pip install bayesflow-irt[numba]`."
        ) from _NUMBA_IMPORT_ERROR
    raise ValueError(
        "use_numba=True but numba failed to import "
        f"({type(_NUMBA_IMPORT_ERROR).__name__}: {_NUMBA_IMPORT_ERROR}). "
        "Check your numba/llvmlite installation, e.g. by reinstalling "
        "with `pip install --force-reinstall bayesflow-irt[numba]`."
    ) from _NUMBA_IMPORT_ERROR


# ---------------------------------------------------------------------------
# Helpers shared by 1PL, 2PL, and PCM
# ---------------------------------------------------------------------------


def _validate_variable_range(name: str, spec: tuple[int, int]) -> tuple[int, int]:
    """Validate and normalize a variable-size range specification."""
    if not isinstance(spec, (list, tuple)) or len(spec) != 2:
        raise ValueError(f"{name} must be a tuple/list (lo, hi) with hi >= lo.")
    lo, hi = int(spec[0]), int(spec[1])
    if lo <= 0:
        raise ValueError(f"{name} lower bound must be positive, got {lo}.")
    if hi < lo:
        raise ValueError(f"{name} must satisfy hi >= lo (use hi == lo for fixed size).")
    return lo, hi


def _resolve_named_prior(spec: PriorSpec, name: ParameterRole) -> Callable:
    """Resolve a prior spec, prefixing errors with the parameter name."""
    try:
        return resolve_prior(spec, name)
    except (ValueError, TypeError) as e:
        raise type(e)(f"prior_{name}: {e}") from e


def _resolve_theta_prior(spec: PriorSpec) -> Callable:
    """Resolve the person prior, rejecting scale-varying (variable) priors.

    Person parameters are drawn from the marginal sampler and no ``sigma_theta``
    is emitted, so a variable prior would silently lose its per-dataset scale.
    """
    if hasattr(spec, "sample_with_meta"):
        raise ValueError(
            "prior_theta: a variable prior (per-dataset scale) is not "
            "supported for theta; pass a fixed Prior or a callable."
        )
    return _resolve_named_prior(spec, "theta")


def _bucket_representatives(lo: int, hi: int, bucket: int) -> np.ndarray:
    """Return the sorted distinct sizes a bucketed draw can produce.

    The range ``[lo, hi]`` is covered by ``max(2, ceil((hi - lo + 1) /
    bucket))`` values, log-spaced from ``lo`` to ``hi`` inclusive and rounded
    to integers, so both ends are always included (rounding may merge
    neighbours at small sizes, hence "at most" that many). ``lo == hi``
    returns ``[lo]``. ``lo >= 1`` is guaranteed by
    ``_validate_variable_range``.

    The number of values matches the number of fixed-width buckets, which is
    what bounds the count of distinct tensor shapes (see
    ``_draw_bucketed_size`` for why that matters). Log spacing puts more of
    those few values at small sizes, which an evenly spaced set would leave
    nearly untrained. That is a design choice, not a derived one.

    Returning each fixed-width bucket's upper edge instead would, with
    ``n_persons=(10, 500)`` and ``bucket=50``, train only on
    ``{59, 109, ..., 459, 500}`` and never draw the smallest sizes, although
    they are validation corners.

    Parameters
    ----------
    lo, hi : int
        Inclusive size range, ``1 <= lo <= hi``.
    bucket : int
        Nominal bucket width; sets the number of representatives.

    Returns
    -------
    numpy.ndarray
        Sorted unique integer sizes, containing ``lo`` and ``hi``.
    """
    if lo == hi:
        return np.array([lo], dtype=np.int64)
    n_values = max(2, -(-(hi - lo + 1) // bucket))  # ceil division
    values = np.rint(np.geomspace(lo, hi, n_values)).astype(np.int64)
    return np.unique(values)


_VALID_SIZE_DISTRIBUTIONS = ("uniform", "log_uniform", "mixed")

#: Share of log-uniform draws under ``size_distribution="mixed"``; the rest
#: are uniform. 0.7 is a design choice (see :func:`_size_distribution`).
DEFAULT_LOG_UNIFORM_SHARE = 0.7

#: Widest size support :func:`describe_size_sampling` lists value by value.
_MAX_LISTED_SIZES = 2000


def _validate_size_distribution(size_distribution: str) -> str:
    """Validate the unbucketed size distribution's name."""
    if size_distribution not in _VALID_SIZE_DISTRIBUTIONS:
        raise ValueError(
            f"size_distribution={size_distribution!r} is not valid. "
            f"Choose from {_VALID_SIZE_DISTRIBUTIONS}."
        )
    return size_distribution


def _validate_log_uniform_share(share: float) -> float:
    """Validate the log-uniform share of a ``"mixed"`` size distribution."""
    share = float(share)
    if not 0.0 <= share <= 1.0:
        raise ValueError(f"log_uniform_share must be in [0, 1], got {share}.")
    return share


def _size_distribution(
    value_range: tuple[int, int],
    bucket: int | None,
    size_distribution: str = "uniform",
    log_uniform_share: float = DEFAULT_LOG_UNIFORM_SHARE,
) -> tuple[np.ndarray, np.ndarray]:
    """Every size a draw can return, with its probability.

    The single source of truth for what :func:`_draw_bucketed_size` draws:
    the adapter's size-standardization moments and the training metadata
    are both computed from it.

    - ``bucket`` given: the bucket representatives, equally likely.
      ``size_distribution`` is ignored, since the representatives are
      already log-spaced.
    - ``"uniform"``: every integer in ``[lo, hi]``, equally likely.
    - ``"log_uniform"``: every integer ``k`` in ``[lo, hi]`` with
      probability ``log((k + 1) / k) / log((hi + 1) / lo)``, i.e. ``log k``
      is close to uniform. Small sizes are drawn far more often than under
      ``"uniform"`` -- over ``(10, 500)``, 10-24 persons come up about 23% of
      the time instead of 3%. A design choice, not derived from theory (see
      :func:`_bucket_representatives`).
    - ``"mixed"``: a draw is log-uniform with probability
      ``log_uniform_share`` and uniform otherwise, so each integer's
      probability is the same mixture of the two. It aims to keep most of
      the small-size benefit of log-uniform draws without starving the
      large sizes.

    Returns
    -------
    values : numpy.ndarray of int64
    probabilities : numpy.ndarray of float64, summing to 1
    """
    lo, hi = int(value_range[0]), int(value_range[1])
    if bucket is not None:
        values = _bucket_representatives(lo, hi, bucket)
        return values, np.full(values.size, 1.0 / values.size)
    values = np.arange(lo, hi + 1, dtype=np.int64)
    flat = np.full(values.size, 1.0 / values.size)
    if size_distribution in ("log_uniform", "mixed") and hi > lo:
        log_probs = np.log((values + 1) / values) / np.log((hi + 1) / lo)
        if size_distribution == "log_uniform":
            return values, log_probs
        return values, log_uniform_share * log_probs + (1 - log_uniform_share) * flat
    return values, flat


def _draw_bucketed_size(
    rng: np.random.Generator,
    value_range: tuple[int, int],
    bucket: int | None,
    size_distribution: str = "uniform",
    log_uniform_share: float = DEFAULT_LOG_UNIFORM_SHARE,
) -> int:
    """Draw one size from ``value_range``, optionally quantized to buckets.

    With ``bucket=None`` (default), draws every integer in ``[lo, hi]``,
    uniformly (``size_distribution="uniform"``, the original behavior),
    log-uniformly (``"log_uniform"``) or as a mixture of the two
    (``"mixed"``); see :func:`_size_distribution`. With a ``bucket`` size given,
    the draw is uniform over ``_bucket_representatives(lo, hi, bucket)``: a
    small, *repeating*, log-spaced set that always contains both ``lo`` and
    ``hi``. This trades a small amount of shape-range fidelity for few
    possible sizes -- e.g. ``bucket=50`` over ``n_persons=(5, 1000)`` yields
    at most 20 distinct values instead of 996 nearly always-unique ones.

    This exists because PyTorch's CUDA caching allocator on the torch
    backend requests a fresh, rarely-reusable memory block whenever a
    genuinely novel tensor shape appears; with variable-size batches drawing
    a new size on almost every training step, that fragments the allocator's
    reserved pool (confirmed via a `torch.cuda.memory_reserved()` vs.
    `memory_allocated()` reproduction -- reserved climbed to 2.5x physical
    VRAM while allocated stayed bounded). Padding still lets one axis
    (persons, via `MatrixEquivariantIRTSummary`'s person-chunking) reuse
    blocks across steps, but the item axis has no such chunking and drives
    a long tail of one-off free block sizes -- see the person-chunk padding
    in `networks.py`'s `_ChunkedTrainableMatrixPool` for the analogous,
    chunk-level version of this same problem. Bucketing at the simulator
    level fixes both axes at the source, without touching any autograd-sensitive code.
    """
    lo, hi = value_range
    if bucket is None:
        # `hi > lo` first: a pinned range consumes the same random numbers
        # under every distribution, so validation clones stay comparable.
        log_draw = hi > lo and (
            size_distribution == "log_uniform"
            or (size_distribution == "mixed" and rng.random() < log_uniform_share)
        )
        if log_draw:
            # floor(exp(U(log lo, log(hi + 1)))) gives exactly the
            # probabilities _size_distribution states; min() guards the
            # float edge at hi + 1.
            u = rng.uniform(np.log(lo), np.log(hi + 1))
            return min(int(np.floor(np.exp(u))), hi)
        return int(rng.integers(lo, hi + 1))
    representatives = _bucket_representatives(lo, hi, bucket)
    return int(representatives[rng.integers(0, representatives.size)])


def _draw_sizes(
    rng: np.random.Generator,
    items_range: tuple[int, int],
    persons_range: tuple[int, int],
    item_bucket: int | None = None,
    person_bucket: int | None = None,
    size_distribution: str = "uniform",
    log_uniform_share: float = DEFAULT_LOG_UNIFORM_SHARE,
) -> tuple[int, int]:
    """Draw ``(n_items, n_persons)`` from validated ranges, optionally bucketed."""
    n_items = _draw_bucketed_size(
        rng, items_range, item_bucket, size_distribution, log_uniform_share
    )
    n_persons = _draw_bucketed_size(
        rng, persons_range, person_bucket, size_distribution, log_uniform_share
    )
    return n_items, n_persons


def describe_size_sampling(simulator: Any) -> dict[str, Any]:
    """Record exactly which data sizes a simulator trains on, for metadata.

    A size range alone does not say what a model saw: a bucketing rule can
    leave whole parts of the range undrawn. This records the rule and, per
    axis, every size the simulator can draw with its probability, computed by
    :func:`_size_distribution`, the function the sampler itself uses.

    Parameters
    ----------
    simulator
        Any simulator built by this module. Families without buckets or a
        ``size_distribution`` attribute are recorded as unbucketed uniform.

    Returns
    -------
    dict
        JSON-serializable: ``{"rule": ..., "n_items": {...},
        "n_persons": {...}}`` where each axis has ``range``, ``bucket``,
        ``values`` and ``probabilities`` (full precision; ``None`` for a
        support wider than ``_MAX_LISTED_SIZES``, which the rule and range
        then define exactly).
    """
    distribution = getattr(simulator, "size_distribution", "uniform")
    share = getattr(simulator, "log_uniform_share", DEFAULT_LOG_UNIFORM_SHARE)
    record: dict[str, Any] = {"size_distribution": distribution}
    if distribution == "mixed":
        record["log_uniform_share"] = share
    axes = (("n_items", "item_bucket"), ("n_persons", "person_bucket"))
    rules = set()
    for key, bucket_attr in axes:
        low, high = (int(v) for v in getattr(simulator, key))
        value_range = (low, high)
        bucket = getattr(simulator, bucket_attr, None)
        values, probs = _size_distribution(value_range, bucket, distribution, share)
        rules.add("log_spaced_buckets" if bucket is not None else distribution)
        listed = values.size <= _MAX_LISTED_SIZES
        record[key] = {
            "range": list(value_range),
            "bucket": bucket,
            # Full float precision: the record must reproduce the draw. A
            # support too wide to list is fully defined by its rule and range.
            "values": values.tolist() if listed else None,
            "probabilities": probs.tolist() if listed else None,
        }
    record["rule"] = "+".join(sorted(rules))
    return record


def _validate_bucket(name: str, bucket: int | None) -> int | None:
    """Validate an optional size-bucket argument."""
    if bucket is None:
        return None
    bucket = int(bucket)
    if bucket <= 0:
        raise ValueError(f"{name} must be a positive integer or None, got {bucket}.")
    return bucket


def _sample_variable_prior(
    spec: PriorSpec,
    resolved_sampler: Callable,
    shape: tuple,
    rng: np.random.Generator,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Sample from a prior spec, returning metadata for variable priors.

    If *spec* has ``sample_with_meta`` (duck-typing for VariablePrior and
    _VariableIncrementalPrior), uses conditional sampling and returns the
    sigma metadata as ``(batch_size, 1)`` float32 arrays.  Otherwise uses
    the resolved sampler directly.
    """
    if hasattr(spec, "sample_with_meta"):
        samples, raw_meta = spec.sample_with_meta(shape, rng)
        meta = {
            key: np.asarray(val, dtype=np.float32).reshape(batch_size, 1)
            for key, val in raw_meta.items()
        }
        return samples.astype(np.float32), meta
    return resolved_sampler(shape, rng).astype(np.float32), {}


def _normalize_response_dtype(response_dtype: str | np.dtype) -> np.dtype:
    """Validate response dtype used for simulated response matrices."""
    dtype = np.dtype(response_dtype)
    if dtype.name not in _VALID_RESPONSE_DTYPES:
        raise ValueError(
            "response_dtype must be one of "
            f"{_VALID_RESPONSE_DTYPES}, got {dtype.name!r}."
        )
    return dtype


def _validate_categorical_dtype_capacity(
    response_dtype: np.dtype, n_categories: int, model_name: str
) -> None:
    """Raise if an integer ``response_dtype`` can't represent all category indices."""
    if np.issubdtype(response_dtype, np.integer):
        dtype_max = int(np.iinfo(response_dtype).max)
        if n_categories > dtype_max + 1:
            raise ValueError(
                f"response_dtype {response_dtype!r} cannot represent all {model_name} "
                f"category indices: n_categories={n_categories} but dtype "
                f"supports indices 0..{dtype_max} only."
            )


def _sample_binary_responses_from_logits(
    logits: np.ndarray,
    rng: np.random.Generator,
    response_dtype: np.dtype,
    use_numba: bool = False,
) -> np.ndarray:
    """Draw Bernoulli responses from logits using an in-place sigmoid buffer.

    If ``use_numba``, dispatches to a fused numba kernel instead (see
    ``_numba_kernels.py``) — same Bernoulli(sigmoid(logits)) distribution,
    but drawn via numba's own RNG rather than *rng*, so results differ from
    the numpy path for a given seed even though both are statistically
    correct draws from the same model.
    """
    if use_numba:
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        out = np.empty(logits.shape, dtype=response_dtype)
        binary_response_kernel(logits, out, seed)
        return out
    _sigmoid(logits, out=logits)
    return (rng.random(logits.shape, dtype=np.float32) < logits).astype(
        response_dtype,
        copy=False,
    )


def _sample_categorical_responses_from_logits(
    logits: np.ndarray,
    rng: np.random.Generator,
    response_dtype: np.dtype,
    use_numba: bool = False,
) -> np.ndarray:
    """Draw categorical responses from logits with the Gumbel-max trick.

    Uses the exponential-noise form of Gumbel-max described by
    Maddison et al. (2014); see ``references.md``.

    If ``use_numba``, dispatches to a fused numba kernel instead (see
    ``_numba_kernels.py``) — same construction, drawn via numba's own RNG
    rather than *rng*, so results differ from the numpy path for a given
    seed even though both are statistically correct draws from the same
    model.
    """
    if use_numba:
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        out = np.empty(logits.shape[:-1], dtype=response_dtype)
        categorical_response_kernel(logits, out, seed)
        return out
    # If E ~ Exponential(1), then -log(E) is standard Gumbel noise.
    # standard_exponential(dtype=float32) draws directly in float32, avoiding
    # the float64 allocation + cast that rng.exponential(...).astype() incurs.
    noise = rng.standard_exponential(size=logits.shape, dtype=np.float32)
    np.maximum(noise, np.finfo(np.float32).tiny, out=noise)
    np.log(noise, out=noise)
    logits -= noise
    return np.argmax(logits, axis=-1).astype(response_dtype, copy=False)


def _cumulative_step_offsets(delta: np.ndarray, n_categories: int) -> np.ndarray:
    """Return cumulative step difficulties with a zero offset for category 0."""
    offsets = np.empty((*delta.shape[:-1], n_categories), dtype=np.float32)
    offsets[..., 0] = 0.0
    np.cumsum(delta, axis=-1, out=offsets[..., 1:])
    return offsets


# ---------------------------------------------------------------------------
# 2PL Simulator
# ---------------------------------------------------------------------------


class _TwoPLSimulator(Simulator):
    """Batched 2PL IRT simulator.

    Sizes are drawn once per ``sample()`` call so every element in the batch
    has the same ``(n_persons, n_items)`` shape.
    """

    def __init__(
        self,
        n_items: tuple[int, int],
        n_persons: tuple[int, int],
        prior_a: PriorSpec,
        prior_b: PriorSpec,
        prior_theta: PriorSpec,
        parameterization: str,
        response_dtype: str | np.dtype = "float32",
        seed: int | None = None,
        use_numba: bool = False,
        item_bucket: int | None = None,
        person_bucket: int | None = None,
        size_distribution: str = "uniform",
        log_uniform_share: float = DEFAULT_LOG_UNIFORM_SHARE,
    ):
        self.n_items = n_items
        self.n_persons = n_persons
        self._n_items_range = _validate_variable_range("n_items", n_items)
        self._n_persons_range = _validate_variable_range("n_persons", n_persons)
        self.item_bucket = _validate_bucket("item_bucket", item_bucket)
        self.person_bucket = _validate_bucket("person_bucket", person_bucket)
        self.size_distribution = _validate_size_distribution(size_distribution)
        self.log_uniform_share = _validate_log_uniform_share(log_uniform_share)

        if parameterization not in _VALID_PARAMETERIZATIONS:
            raise ValueError(
                f"parameterization={parameterization!r} is not valid. "
                f"Choose from {_VALID_PARAMETERIZATIONS}."
            )

        self._sample_a = _resolve_named_prior(prior_a, "a")
        self._sample_b = _resolve_named_prior(prior_b, "b")
        self._sample_theta = _resolve_theta_prior(prior_theta)

        self.prior_a = prior_a
        self.prior_b = prior_b
        self.prior_theta = prior_theta
        self.parameterization = parameterization
        self.response_dtype = _normalize_response_dtype(response_dtype)
        self.seed = seed
        _check_numba_available(use_numba)
        self.use_numba = use_numba
        self._rng = np.random.default_rng(seed)

    def clone_with_sizes(
        self,
        n_items: tuple[int, int] | None = None,
        n_persons: tuple[int, int] | None = None,
    ) -> _TwoPLSimulator:
        """Return a new simulator with the same priors but different size ranges.

        Parameters
        ----------
        n_items : (int, int) or None
            New item range. Defaults to the current range if not provided.
        n_persons : (int, int) or None
            New person range. Defaults to the current range if not provided.
        """
        return _TwoPLSimulator(
            n_items=n_items if n_items is not None else self.n_items,
            n_persons=n_persons if n_persons is not None else self.n_persons,
            prior_a=self.prior_a,
            prior_b=self.prior_b,
            prior_theta=self.prior_theta,
            parameterization=self.parameterization,
            response_dtype=self.response_dtype.name,
            seed=self.seed,
            use_numba=self.use_numba,
            item_bucket=self.item_bucket,
            person_bucket=self.person_bucket,
            size_distribution=self.size_distribution,
            log_uniform_share=self.log_uniform_share,
        )

    @allow_batch_size
    def sample(self, batch_shape, **kwargs) -> dict[str, np.ndarray]:
        rng = self._rng
        batch_size = int(np.prod(batch_shape))
        n_items, n_persons = _draw_sizes(
            rng,
            self._n_items_range,
            self._n_persons_range,
            item_bucket=self.item_bucket,
            person_bucket=self.person_bucket,
            size_distribution=self.size_distribution,
            log_uniform_share=self.log_uniform_share,
        )

        a, meta_a = _sample_variable_prior(
            self.prior_a,
            self._sample_a,
            (batch_size, n_items),
            rng,
            batch_size,
        )
        b, meta_b = _sample_variable_prior(
            self.prior_b,
            self._sample_b,
            (batch_size, n_items),
            rng,
            batch_size,
        )
        theta = self._sample_theta((batch_size, n_persons), rng).astype(np.float32)

        # a: (batch_size, n_items), theta: (batch_size, n_persons)
        # -> logits: (batch_size, n_persons, n_items)
        if self.parameterization == "standard":
            logits = theta[:, :, np.newaxis] - b[:, np.newaxis, :]
            logits *= a[:, np.newaxis, :]
        else:
            d = -a * b
            logits = theta[:, :, np.newaxis] * a[:, np.newaxis, :]
            logits += d[:, np.newaxis, :]

        responses = _sample_binary_responses_from_logits(
            logits,
            rng,
            self.response_dtype,
            self.use_numba,
        )

        out = dict(
            a=a,
            b=b,
            theta=theta,
            responses=responses,
            n_items=np.full((batch_size, 1), n_items, dtype=np.float32),
            n_persons=np.full((batch_size, 1), n_persons, dtype=np.float32),
        )
        out.update(meta_a)
        out.update(meta_b)
        return out


def make_2pl_simulator(
    n_items: tuple[int, int] = (15, 25),
    n_persons: tuple[int, int] = (150, 250),
    *,
    prior_a: PriorSpec,
    prior_b: PriorSpec,
    prior_theta: PriorSpec = Prior("normal", loc=0, scale=1.0),
    parameterization: str = "standard",
    response_dtype: str | np.dtype = "float32",
    seed: int | None = None,
    use_numba: bool = False,
    item_bucket: int | None = None,
    person_bucket: int | None = None,
    size_distribution: str = "uniform",
    log_uniform_share: float = DEFAULT_LOG_UNIFORM_SHARE,
) -> Simulator:
    """Create a BayesFlow simulator for the 2PL IRT model.

    Parameters
    ----------
    n_items : (int, int)
        Variable item range ``(lo, hi)`` with ``hi >= lo``. Each **batch**
        draws a random number of items from ``Uniform(lo, hi)`` (inclusive).
        All samples in the batch share the same size.
    n_persons : (int, int)
        Variable person range ``(lo, hi)`` with ``hi >= lo``.
    prior_a : Prior | callable
        Prior on item discriminations.  Pass a :class:`~bayesflow_irt.Prior`
        (e.g. ``Prior("lognormal", loc=0, scale=0.3)``) or a callable
        ``(shape, rng) -> ndarray``.  No default — must be specified
        explicitly to prevent silent train/validation mismatches.
    prior_b : Prior | callable
        Prior on item difficulties.  No default — must be specified explicitly.
    prior_theta : Prior | callable
        Prior on person abilities.  Default ``Normal(0, 1)``.
    parameterization : str
        ``"standard"`` uses :math:`a_i (\\theta_p - b_i)`.
        ``"loading"``  uses :math:`a_i \\theta_p + d_i` where
        :math:`d_i = -a_i b_i`.
    response_dtype : str | np.dtype
        Output dtype for ``responses``. Supported: ``"float32"`` (default)
        and ``"uint8"``. Use ``"uint8"`` to reduce memory for binary
        responses.
    seed : int or None
        Seed for the simulator's internal RNG. ``None`` (default) draws an
        unseeded generator, matching prior behavior.
    use_numba : bool
        If ``True``, draw binary responses with a fused numba kernel instead
        of the numpy path (requires the ``numba`` extra:
        ``pip install bayesflow-irt[numba]``). Same distribution, but numba
        uses its own internal RNG independent of *seed* — enabling this
        changes the exact simulated values for a given seed (not bit-for-bit
        reproducible against ``use_numba=False``). Default ``False``.
    item_bucket : int or None
        If given, quantizes each batch's drawn ``n_items`` to a small,
        repeating set of values instead of a nearly-unique integer per
        batch: ``n_items`` is drawn uniformly from ``ceil(range / item_bucket)``
        log-spaced sizes that include both range ends. Reduces the number of
        distinct ``(n_items, n_persons)`` tensor shapes PyTorch's CUDA
        caching allocator has to serve on the torch backend, which otherwise
        fragments its reserved memory pool under long variable-size training runs (see
        ``_draw_bucketed_size``'s docstring for the measured evidence).
        ``None`` (default) preserves the original per-integer draw.
    person_bucket : int or None
        Same quantization as ``item_bucket``, applied to ``n_persons``.
    size_distribution : {"uniform", "log_uniform", "mixed"}
        How an UNBUCKETED axis draws its size. ``"uniform"`` (default) is
        the original per-integer draw. ``"log_uniform"`` draws small sizes
        far more often; ``"mixed"`` draws log-uniformly with probability
        *log_uniform_share* and uniformly otherwise (see
        ``_size_distribution``).
        A bucketed axis ignores it; its representatives are already
        log-spaced.
    log_uniform_share : float
        Share of log-uniform draws under ``size_distribution="mixed"``, in
        ``[0, 1]``. Default :data:`DEFAULT_LOG_UNIFORM_SHARE` (0.7).

    Returns
    -------
    bayesflow.simulators.Simulator
    """
    return _TwoPLSimulator(
        n_items=n_items,
        n_persons=n_persons,
        prior_a=prior_a,
        prior_b=prior_b,
        prior_theta=prior_theta,
        parameterization=parameterization,
        seed=seed,
        response_dtype=response_dtype,
        use_numba=use_numba,
        item_bucket=item_bucket,
        person_bucket=person_bucket,
        size_distribution=size_distribution,
        log_uniform_share=log_uniform_share,
    )


# ---------------------------------------------------------------------------
# 1PL (Rasch) Simulator
# ---------------------------------------------------------------------------


class _OnePLSimulator(Simulator):
    """Batched 1PL (Rasch) IRT simulator."""

    def __init__(
        self,
        n_items: tuple[int, int],
        n_persons: tuple[int, int],
        prior_b: PriorSpec,
        prior_theta: PriorSpec,
        response_dtype: str | np.dtype = "float32",
        seed: int | None = None,
        use_numba: bool = False,
    ):
        self.n_items = n_items
        self.n_persons = n_persons
        self._n_items_range = _validate_variable_range("n_items", n_items)
        self._n_persons_range = _validate_variable_range("n_persons", n_persons)

        self._sample_b = _resolve_named_prior(prior_b, "b")
        self._sample_theta = _resolve_theta_prior(prior_theta)

        self.prior_b = prior_b
        self.prior_theta = prior_theta
        self.response_dtype = _normalize_response_dtype(response_dtype)
        self.seed = seed
        _check_numba_available(use_numba)
        self.use_numba = use_numba
        self._rng = np.random.default_rng(seed)

    def clone_with_sizes(
        self,
        n_items: tuple[int, int] | None = None,
        n_persons: tuple[int, int] | None = None,
    ) -> _OnePLSimulator:
        """Return a new simulator with the same priors but different size ranges.

        Parameters
        ----------
        n_items : (int, int) or None
            New item range. Defaults to the current range if not provided.
        n_persons : (int, int) or None
            New person range. Defaults to the current range if not provided.
        """
        return _OnePLSimulator(
            n_items=n_items if n_items is not None else self.n_items,
            n_persons=n_persons if n_persons is not None else self.n_persons,
            prior_b=self.prior_b,
            prior_theta=self.prior_theta,
            response_dtype=self.response_dtype.name,
            seed=self.seed,
            use_numba=self.use_numba,
        )

    @allow_batch_size
    def sample(self, batch_shape, **kwargs) -> dict[str, np.ndarray]:
        rng = self._rng
        batch_size = int(np.prod(batch_shape))
        n_items, n_persons = _draw_sizes(
            rng,
            self._n_items_range,
            self._n_persons_range,
        )

        b, meta_b = _sample_variable_prior(
            self.prior_b,
            self._sample_b,
            (batch_size, n_items),
            rng,
            batch_size,
        )
        theta = self._sample_theta((batch_size, n_persons), rng).astype(np.float32)

        # (batch_size, n_persons, n_items)
        logits = theta[:, :, np.newaxis] - b[:, np.newaxis, :]
        responses = _sample_binary_responses_from_logits(
            logits,
            rng,
            self.response_dtype,
            self.use_numba,
        )

        out = dict(
            b=b,
            theta=theta,
            responses=responses,
            n_items=np.full((batch_size, 1), n_items, dtype=np.float32),
            n_persons=np.full((batch_size, 1), n_persons, dtype=np.float32),
        )
        out.update(meta_b)
        return out


def make_1pl_simulator(
    n_items: tuple[int, int] = (15, 25),
    n_persons: tuple[int, int] = (150, 250),
    *,
    prior_b: PriorSpec,
    prior_theta: PriorSpec = Prior("normal", loc=0, scale=1.0),
    response_dtype: str | np.dtype = "float32",
    seed: int | None = None,
    use_numba: bool = False,
) -> Simulator:
    """Create a BayesFlow simulator for the 1PL (Rasch) IRT model.

    In the 1PL model discrimination is fixed at 1 for all items:
    :math:`P(X = 1) = \\sigma(\\theta_p - b_i)`.

    Parameters
    ----------
    n_items : (int, int)
        Variable item range ``(lo, hi)`` with ``hi >= lo``.
    n_persons : (int, int)
        Variable person range ``(lo, hi)`` with ``hi >= lo``.
    prior_b : Prior | callable
        Prior on item difficulties.  No default — must be specified explicitly.
    prior_theta : Prior | callable
        Prior on person abilities.  Default ``Normal(0, 1)``.
    response_dtype : str | np.dtype
        Output dtype for ``responses``. Supported: ``"float32"`` (default)
        and ``"uint8"``. Use ``"uint8"`` to reduce memory for binary
        responses.
    seed : int or None
        Seed for the simulator's internal RNG. ``None`` (default) draws an
        unseeded generator, matching prior behavior.
    use_numba : bool
        If ``True``, draw binary responses with a fused numba kernel instead
        of the numpy path (requires the ``numba`` extra:
        ``pip install bayesflow-irt[numba]``). Same distribution, but numba
        uses its own internal RNG independent of *seed* — enabling this
        changes the exact simulated values for a given seed (not bit-for-bit
        reproducible against ``use_numba=False``). Default ``False``.

    Returns
    -------
    bayesflow.simulators.Simulator
    """
    return _OnePLSimulator(
        n_items=n_items,
        n_persons=n_persons,
        prior_b=prior_b,
        prior_theta=prior_theta,
        response_dtype=response_dtype,
        seed=seed,
        use_numba=use_numba,
    )


# ---------------------------------------------------------------------------
# PCM (Partial Credit Model) Simulator
# ---------------------------------------------------------------------------


class _PCMSimulator(Simulator):
    """Batched Partial Credit Model (PCM) simulator.

    The PCM (Masters, 1982) is a polytomous extension of the Rasch model:

    .. math::

        P(X_{pi} = k) =
            \\frac{\\exp\\bigl(\\sum_{j=1}^{k}
            (\\theta_p - \\delta_{ij})\\bigr)}
            {\\sum_{c=0}^{K-1}\\exp\\bigl(\\sum_{j=1}^{c}
            (\\theta_p - \\delta_{ij})\\bigr)}

    Each item has ``K`` response categories (0 … K-1) and ``K-1`` step
    difficulty parameters.
    """

    def __init__(
        self,
        n_items: tuple[int, int],
        n_persons: tuple[int, int],
        n_categories: int,
        prior_delta: PriorSpec,
        prior_theta: PriorSpec,
        response_dtype: str | np.dtype = "float32",
        seed: int | None = None,
        use_numba: bool = False,
    ):
        self.n_items = n_items
        self.n_persons = n_persons
        self._n_items_range = _validate_variable_range("n_items", n_items)
        self._n_persons_range = _validate_variable_range("n_persons", n_persons)

        if not isinstance(n_categories, int) or n_categories < 2:
            raise ValueError(
                f"n_categories must be an integer >= 2, got {n_categories!r}."
            )
        self.n_categories = n_categories

        self._sample_delta = _resolve_named_prior(prior_delta, "delta")
        self._sample_theta = _resolve_theta_prior(prior_theta)

        self.prior_delta = prior_delta
        self.prior_theta = prior_theta
        self.response_dtype = _normalize_response_dtype(response_dtype)
        _validate_categorical_dtype_capacity(
            self.response_dtype, self.n_categories, "PCM"
        )
        self.seed = seed
        _check_numba_available(use_numba)
        self.use_numba = use_numba
        self._rng = np.random.default_rng(seed)

    def clone_with_sizes(
        self,
        n_items: tuple[int, int] | None = None,
        n_persons: tuple[int, int] | None = None,
    ) -> _PCMSimulator:
        """Return a new simulator with the same priors but different size ranges.

        Parameters
        ----------
        n_items : (int, int) or None
            New item range. Defaults to the current range if not provided.
        n_persons : (int, int) or None
            New person range. Defaults to the current range if not provided.
        """
        return _PCMSimulator(
            n_items=n_items if n_items is not None else self.n_items,
            n_persons=n_persons if n_persons is not None else self.n_persons,
            n_categories=self.n_categories,
            prior_delta=self.prior_delta,
            prior_theta=self.prior_theta,
            response_dtype=self.response_dtype.name,
            seed=self.seed,
            use_numba=self.use_numba,
        )

    @allow_batch_size
    def sample(self, batch_shape, **kwargs) -> dict[str, np.ndarray]:
        rng = self._rng
        batch_size = int(np.prod(batch_shape))
        n_categories = self.n_categories
        n_items, n_persons = _draw_sizes(
            rng,
            self._n_items_range,
            self._n_persons_range,
        )

        # Prior draws
        delta, meta_delta = _sample_variable_prior(
            self.prior_delta,
            self._sample_delta,
            (batch_size, n_items, n_categories - 1),
            rng,
            batch_size,
        )
        if n_categories > 2:
            delta = np.sort(delta, axis=-1)
        theta = self._sample_theta((batch_size, n_persons), rng).astype(np.float32)

        if n_categories == 2:
            logits = theta[:, :, np.newaxis] - delta[:, np.newaxis, :, 0]
            responses = _sample_binary_responses_from_logits(
                logits,
                rng,
                self.response_dtype,
                self.use_numba,
            )
        else:
            category_scores = np.arange(n_categories, dtype=np.float32)
            cum_logits = (
                theta[:, :, np.newaxis, np.newaxis] * category_scores
                - _cumulative_step_offsets(delta, n_categories)[:, np.newaxis, :, :]
            )  # (batch_size, n_persons, n_items, n_categories)
            responses = _sample_categorical_responses_from_logits(
                cum_logits,
                rng,
                self.response_dtype,
                self.use_numba,
            )

        out = dict(
            delta=delta,
            theta=theta,
            responses=responses,
            n_items=np.full((batch_size, 1), n_items, dtype=np.float32),
            n_persons=np.full((batch_size, 1), n_persons, dtype=np.float32),
        )
        out.update(meta_delta)
        return out


# ---------------------------------------------------------------------------
# GPCM (Generalized Partial Credit Model) Simulator
# ---------------------------------------------------------------------------


class _GPCMSimulator(Simulator):
    """Batched Generalized Partial Credit Model (GPCM) simulator.

    The GPCM (Muraki, 1992) extends the PCM by adding item-level
    discrimination parameters:

    .. math::

        P(X_{pi} = k) =
            \\frac{\\exp\\bigl(\\sum_{j=1}^{k}
            a_i(\\theta_p - \\delta_{ij})\\bigr)}
            {\\sum_{c=0}^{K-1}\\exp\\bigl(\\sum_{j=1}^{c}
            a_i(\\theta_p - \\delta_{ij})\\bigr)}

    When ``a_i = 1`` for all items, the GPCM reduces to the PCM.

    References
    ----------
    Muraki, E. (1992). A generalized partial credit model: Application of
    an EM algorithm. *Applied Psychological Measurement*, 16(2), 159-176.
    """

    def __init__(
        self,
        n_items: tuple[int, int],
        n_persons: tuple[int, int],
        n_categories: int,
        prior_a: PriorSpec,
        prior_delta: PriorSpec,
        prior_theta: PriorSpec,
        parameterization: str,
        response_dtype: str | np.dtype = "float32",
        seed: int | None = None,
        use_numba: bool = False,
    ):
        self.n_items = n_items
        self.n_persons = n_persons
        self._n_items_range = _validate_variable_range("n_items", n_items)
        self._n_persons_range = _validate_variable_range("n_persons", n_persons)

        if not isinstance(n_categories, int) or n_categories < 2:
            raise ValueError(
                f"n_categories must be an integer >= 2, got {n_categories!r}."
            )
        self.n_categories = n_categories

        if parameterization not in _VALID_PARAMETERIZATIONS:
            raise ValueError(
                f"parameterization={parameterization!r} is not valid. "
                f"Choose from {_VALID_PARAMETERIZATIONS}."
            )

        self._sample_a = _resolve_named_prior(prior_a, "a")
        self._sample_delta = _resolve_named_prior(prior_delta, "delta")
        self._sample_theta = _resolve_theta_prior(prior_theta)

        self.prior_a = prior_a
        self.prior_delta = prior_delta
        self.prior_theta = prior_theta
        self.parameterization = parameterization
        self.response_dtype = _normalize_response_dtype(response_dtype)
        _validate_categorical_dtype_capacity(
            self.response_dtype, self.n_categories, "GPCM"
        )
        self.seed = seed
        _check_numba_available(use_numba)
        self.use_numba = use_numba
        self._rng = np.random.default_rng(seed)

    def clone_with_sizes(
        self,
        n_items: tuple[int, int] | None = None,
        n_persons: tuple[int, int] | None = None,
    ) -> _GPCMSimulator:
        """Return a new simulator with the same priors but different size ranges.

        Parameters
        ----------
        n_items : (int, int) or None
            New item range. Defaults to the current range if not provided.
        n_persons : (int, int) or None
            New person range. Defaults to the current range if not provided.
        """
        return _GPCMSimulator(
            n_items=n_items if n_items is not None else self.n_items,
            n_persons=n_persons if n_persons is not None else self.n_persons,
            n_categories=self.n_categories,
            prior_a=self.prior_a,
            prior_delta=self.prior_delta,
            prior_theta=self.prior_theta,
            parameterization=self.parameterization,
            response_dtype=self.response_dtype.name,
            seed=self.seed,
            use_numba=self.use_numba,
        )

    @allow_batch_size
    def sample(self, batch_shape, **kwargs) -> dict[str, np.ndarray]:
        rng = self._rng
        batch_size = int(np.prod(batch_shape))
        n_categories = self.n_categories
        n_items, n_persons = _draw_sizes(
            rng,
            self._n_items_range,
            self._n_persons_range,
        )

        # Prior draws
        a, meta_a = _sample_variable_prior(
            self.prior_a,
            self._sample_a,
            (batch_size, n_items),
            rng,
            batch_size,
        )
        delta, meta_delta = _sample_variable_prior(
            self.prior_delta,
            self._sample_delta,
            (batch_size, n_items, n_categories - 1),
            rng,
            batch_size,
        )
        if n_categories > 2:
            delta = np.sort(delta, axis=-1)
        theta = self._sample_theta((batch_size, n_persons), rng).astype(np.float32)

        if n_categories == 2:
            logits = theta[:, :, np.newaxis] - delta[:, np.newaxis, :, 0]
            logits *= a[:, np.newaxis, :]
            responses = _sample_binary_responses_from_logits(
                logits,
                rng,
                self.response_dtype,
                self.use_numba,
            )
        else:
            category_scores = np.arange(n_categories, dtype=np.float32)
            cum_logits = (
                theta[:, :, np.newaxis, np.newaxis] * category_scores
                - _cumulative_step_offsets(delta, n_categories)[:, np.newaxis, :, :]
            )
            cum_logits *= a[:, np.newaxis, :, np.newaxis]
            responses = _sample_categorical_responses_from_logits(
                cum_logits,
                rng,
                self.response_dtype,
                self.use_numba,
            )

        out = dict(
            a=a,
            delta=delta,
            theta=theta,
            responses=responses,
            n_items=np.full((batch_size, 1), n_items, dtype=np.float32),
            n_persons=np.full((batch_size, 1), n_persons, dtype=np.float32),
        )
        out.update(meta_a)
        out.update(meta_delta)
        return out


def make_gpcm_simulator(
    n_items: tuple[int, int] = (15, 25),
    n_persons: tuple[int, int] = (150, 250),
    n_categories: int = 5,
    *,
    prior_a: PriorSpec,
    prior_delta: PriorSpec,
    prior_theta: PriorSpec = Prior("normal", loc=0, scale=1.0),
    parameterization: str = "standard",
    response_dtype: str | np.dtype = "float32",
    seed: int | None = None,
    use_numba: bool = False,
) -> Simulator:
    """Create a BayesFlow simulator for the Generalized Partial Credit Model.

    The GPCM (Muraki, 1992) extends the PCM by adding item-level
    discrimination parameters ``a_i``.  When ``a_i = 1`` for all items,
    the GPCM reduces to the PCM.

    Parameters
    ----------
    n_items : (int, int)
        Variable item range ``(lo, hi)`` with ``hi >= lo``.
    n_persons : (int, int)
        Variable person range ``(lo, hi)`` with ``hi >= lo``.
    n_categories : int
        Number of response categories per item (K).  Must be >= 2.
    prior_a : Prior | callable
        Prior on item discriminations.  No default — must be specified
        explicitly to prevent silent train/validation mismatches.
    prior_delta : Prior | callable
        Prior on step difficulties.  No default — must be specified
        explicitly to prevent silent train/validation mismatches.
    prior_theta : Prior | callable
        Prior on person abilities.  Default ``Normal(0, 1)``.
    parameterization : str
        ``"standard"`` uses ``a_i * (theta_p - delta_ij)``.
        ``"loading"``  uses ``a_i * theta_p + d_ij`` where
        ``d_ij = -a_i * delta_ij``.
    response_dtype : str | np.dtype
        Output dtype for ``responses``. Supported: ``"float32"`` (default)
        and ``"uint8"``. Integer dtypes must represent category indices
        ``0 .. n_categories-1``; for ``uint8`` this requires
        ``n_categories <= 256``.
    seed : int or None
        Seed for the simulator's internal RNG. ``None`` (default) draws an
        unseeded generator, matching prior behavior.
    use_numba : bool
        If ``True``, draw responses with a fused numba kernel instead of the
        numpy path (requires the ``numba`` extra:
        ``pip install bayesflow-irt[numba]``). Same distribution, but numba
        uses its own internal RNG independent of *seed* — enabling this
        changes the exact simulated values for a given seed (not bit-for-bit
        reproducible against ``use_numba=False``). Default ``False``.

    Returns
    -------
    bayesflow.simulators.Simulator

    References
    ----------
    Muraki, E. (1992). A generalized partial credit model: Application of
    an EM algorithm. *Applied Psychological Measurement*, 16(2), 159-176.
    """
    return _GPCMSimulator(
        n_items=n_items,
        n_persons=n_persons,
        n_categories=n_categories,
        prior_a=prior_a,
        prior_delta=prior_delta,
        prior_theta=prior_theta,
        parameterization=parameterization,
        response_dtype=response_dtype,
        seed=seed,
        use_numba=use_numba,
    )


def make_pcm_simulator(
    n_items: tuple[int, int] = (15, 25),
    n_persons: tuple[int, int] = (150, 250),
    n_categories: int = 5,
    *,
    prior_delta: PriorSpec,
    prior_theta: PriorSpec = Prior("normal", loc=0, scale=1.0),
    response_dtype: str | np.dtype = "float32",
    seed: int | None = None,
    use_numba: bool = False,
) -> Simulator:
    """Create a BayesFlow simulator for the Partial Credit Model (PCM).

    The PCM (Masters, 1982) is a polytomous extension of the Rasch model.
    Each item has ``n_categories`` response categories (0 … K-1) and
    ``K-1`` step difficulty parameters.

    Parameters
    ----------
    n_items : (int, int)
        Variable item range ``(lo, hi)`` with ``hi >= lo``.
    n_persons : (int, int)
        Variable person range ``(lo, hi)`` with ``hi >= lo``.
    n_categories : int
        Number of response categories per item (K).  Must be >= 2.
        Default ``5`` (5-point Likert scale).
    prior_delta : Prior | callable
        Prior on step difficulties.  No default — must be specified
        explicitly to prevent silent train/validation mismatches.
    prior_theta : Prior | callable
        Prior on person abilities.  Default ``Normal(0, 1)``.
    response_dtype : str | np.dtype
        Output dtype for ``responses``. Supported: ``"float32"`` (default)
        and ``"uint8"``. Integer dtypes must represent category indices
        ``0 .. n_categories-1``; for ``uint8`` this requires
        ``n_categories <= 256``.
    seed : int or None
        Seed for the simulator's internal RNG. ``None`` (default) draws an
        unseeded generator, matching prior behavior.
    use_numba : bool
        If ``True``, draw responses with a fused numba kernel instead of the
        numpy path (requires the ``numba`` extra:
        ``pip install bayesflow-irt[numba]``). Same distribution, but numba
        uses its own internal RNG independent of *seed* — enabling this
        changes the exact simulated values for a given seed (not bit-for-bit
        reproducible against ``use_numba=False``). Default ``False``.

    Returns
    -------
    bayesflow.simulators.Simulator
    """
    return _PCMSimulator(
        n_items=n_items,
        n_persons=n_persons,
        n_categories=n_categories,
        prior_delta=prior_delta,
        prior_theta=prior_theta,
        response_dtype=response_dtype,
        seed=seed,
        use_numba=use_numba,
    )


# ---------------------------------------------------------------------------
# Rebuilding a simulator from a recorded model spec
# ---------------------------------------------------------------------------

# make_incremental_prior uses 1/max(K-2, 1) scaling (previously 1/sqrt(K-1)).
# A delta_mean/delta_cat_sep record written before the change describes a
# different threshold prior. Detection is by date: records from the change
# day itself are treated as legacy.
_INCREMENTAL_PRIOR_SINCE = datetime(2026, 7, 6, tzinfo=UTC)

# Prior keys each family needs; ``theta`` is optional everywhere because every
# factory defaults it to Normal(0, 1).
_REQUIRED_PRIORS = {
    "1pl": ("b",),
    "2pl": ("a", "b"),
    "pcm": ("delta",),
    "gpcm": ("a", "delta"),
}


def _prior_from_record(record: Any, key: str) -> Prior:
    """Build a :class:`Prior` from a recorded ``{"name": ..., **kwargs}``."""
    if not isinstance(record, Mapping) or not isinstance(record.get("name"), str):
        raise ValueError(
            f"Prior {key!r} is recorded as {record!r}, not as "
            "{'name': ..., **kwargs}; a custom prior cannot be rebuilt from "
            "metadata. Pass the training simulator explicitly instead."
        )
    kwargs = {k: v for k, v in record.items() if k != "name"}
    return Prior(record["name"], **kwargs)


def _check_incremental_prior_version(trained_at: Any) -> None:
    """Refuse an incremental-prior record that may predate its current scaling."""
    try:
        when = datetime.fromisoformat(str(trained_at))
    except ValueError:
        when = None
    if when is not None and when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    if when is None or when < _INCREMENTAL_PRIOR_SINCE:
        raise ValueError(
            "delta_mean/delta_cat_sep priors are rebuilt with make_incremental_prior; "
            "records written before its scaling change (i.e. before "
            f"{_INCREMENTAL_PRIOR_SINCE.date().isoformat()}) describe a different "
            f"threshold prior. This model's training time is {trained_at!r}, so "
            "its threshold prior cannot be identified. Pass the training "
            "simulator explicitly instead."
        )


def _priors_from_spec(
    priors: Mapping[str, Any],
    family: str,
    n_categories: int | None,
    trained_at: Any,
) -> dict[str, PriorSpec]:
    """Map recorded priors to the factory's ``prior_*`` keyword arguments.

    Step difficulties are recorded either as one ``delta`` prior
    (``scripts/train_irt.py``) or as ``delta_mean`` plus ``delta_cat_sep``,
    which combine through :func:`make_incremental_prior`.
    """
    priors = dict(priors)
    if "delta" not in priors and {"delta_mean", "delta_cat_sep"} <= priors.keys():
        if n_categories is None:
            raise ValueError("model_spec.n_categories is needed for a PCM/GPCM.")
        _check_incremental_prior_version(trained_at)
        delta = make_incremental_prior(
            delta_mean=_prior_from_record(priors.pop("delta_mean"), "delta_mean"),
            delta_cat_sep=_prior_from_record(
                priors.pop("delta_cat_sep"), "delta_cat_sep"
            ),
            n_categories=int(n_categories),
        )
    else:
        delta = None
    missing = [k for k in _REQUIRED_PRIORS[family] if k not in priors]
    if delta is not None:
        missing = [k for k in missing if k != "delta"]
    if missing:
        raise ValueError(
            f"model_spec.priors has no record of {missing} for a {family} model."
        )
    expected = set(_REQUIRED_PRIORS[family]) | {"theta"}
    unknown = sorted(set(priors) - expected)
    if unknown:
        raise ValueError(
            f"model_spec.priors has entries {unknown} that a {family} simulator "
            "does not take."
        )
    if "theta" not in priors:
        warnings.warn(
            "model_spec.priors records no 'theta' prior; using the simulator "
            "default Normal(0, 1).",
            UserWarning,
            stacklevel=3,
        )
    out: dict[str, PriorSpec] = {
        f"prior_{k}": _prior_from_record(v, k) for k, v in priors.items()
    }
    if delta is not None:
        out["prior_delta"] = delta
    return out


def _bucket_kwargs(size_sampling: Mapping[str, Any] | None) -> dict[str, Any]:
    """Read 2PL size-sampling settings from a ``describe_size_sampling`` record."""
    if not size_sampling:
        return {}
    out: dict[str, Any] = {}
    if "size_distribution" in size_sampling:
        out["size_distribution"] = size_sampling["size_distribution"]
    if "log_uniform_share" in size_sampling:
        out["log_uniform_share"] = size_sampling["log_uniform_share"]
    for axis, kwarg in (("n_items", "item_bucket"), ("n_persons", "person_bucket")):
        record = size_sampling.get(axis)
        if isinstance(record, Mapping) and "bucket" in record:
            out[kwarg] = record["bucket"]
    return out


def build_simulator(
    model_spec: Mapping[str, Any],
    *,
    n_items: tuple[int, int] | None = None,
    n_persons: tuple[int, int] | None = None,
    seed: int | None = None,
    use_numba: bool = False,
    apply_size_sampling: bool | None = None,
    trained_at: str | None = None,
) -> Simulator:
    """Rebuild a model's training simulator from its recorded ``model_spec``.

    A ``.keras`` checkpoint holds the approximator but not the simulator, so
    anything that must simulate from a model's own prior -- training scripts
    and model-card evaluation alike -- builds it here from the metadata
    sidecar's ``model_spec``.

    Parameters
    ----------
    model_spec : mapping
        The ``model_spec`` section of a version-2 metadata sidecar. Uses
        ``model``, ``priors``, ``n_categories`` (PCM/GPCM), ``size_ranges``
        and, for a 2PL, ``size_sampling``. Other keys are ignored.
    n_items, n_persons : (int, int), optional
        Size ranges to simulate. Default: ``model_spec.size_ranges``.
    seed : int or None
        Seed for the simulator's RNG.
    use_numba : bool
        Use the numba response kernel. Default ``False``, the path whose draws
        are fixed by *seed*.
    apply_size_sampling : bool or None
        Apply the recorded 2PL size buckets and size distribution
        (``model_spec.size_sampling``). Default ``None``: only when neither
        *n_items* nor *n_persons* is given, since the record describes the
        training ranges.
    trained_at : str or None
        ISO timestamp of training, e.g. the sidecar's
        ``provenance.timestamp``. Needed only for ``delta_mean`` +
        ``delta_cat_sep`` priors: records written before the
        :func:`make_incremental_prior` scaling change (before 2026-07-06) and
        undated records are refused.

    Returns
    -------
    bayesflow.simulators.Simulator

    Raises
    ------
    ValueError
        If the family is unknown, a needed prior is missing, or a prior is not
        recorded as ``{"name": ..., **kwargs}`` (a custom callable prior), or
        an incremental threshold prior is older than its current scaling.
    """
    family = str(model_spec.get("model") or "").lower()
    if family not in _REQUIRED_PRIORS:
        raise ValueError(
            f"model_spec.model is {model_spec.get('model')!r}; expected one of "
            f"{sorted(_REQUIRED_PRIORS)}."
        )
    n_categories = model_spec.get("n_categories")
    priors = _priors_from_spec(
        model_spec.get("priors") or {}, family, n_categories, trained_at
    )

    sizes = model_spec.get("size_ranges") or {}
    if apply_size_sampling is None:
        apply_size_sampling = n_items is None and n_persons is None
    if n_items is None:
        n_items = sizes.get("n_items")
    if n_persons is None:
        n_persons = sizes.get("n_persons")
    if n_items is None or n_persons is None:
        raise ValueError("model_spec.size_ranges lacks n_items or n_persons.")
    common: dict[str, Any] = dict(
        n_items=tuple(int(v) for v in n_items),
        n_persons=tuple(int(v) for v in n_persons),
        seed=seed,
        use_numba=use_numba,
        **priors,
    )

    if family == "1pl":
        return make_1pl_simulator(**common)
    if family == "2pl":
        buckets = _bucket_kwargs(
            model_spec.get("size_sampling") if apply_size_sampling else None
        )
        return make_2pl_simulator(**common, **buckets)
    if n_categories is None:
        raise ValueError("model_spec.n_categories is needed for a PCM/GPCM.")
    if family == "pcm":
        return make_pcm_simulator(n_categories=int(n_categories), **common)
    return make_gpcm_simulator(n_categories=int(n_categories), **common)
