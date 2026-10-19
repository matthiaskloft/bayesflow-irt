"""Adapter factories for IRT models.

Builds ``bayesflow.adapters.Adapter`` pipelines that map simulator dictionaries
into the ``inference_variables`` / ``summary_variables`` / ``inference_conditions``
schema expected by ``EquivariantIRTApproximator`` with ``FlowMatching``.
"""

from __future__ import annotations

import bayesflow as bf
import keras
import numpy as np
from bayesflow.adapters.transforms import (
    ElementwiseTransform,
    FilterTransform,
    Transform,
)

from .priors import PriorSpec, prior_moments, sample_prior_predictive
from .simulators import DEFAULT_LOG_UNIFORM_SHARE, _size_distribution


def _collect_meta_keys(*specs) -> list[str]:
    """Collect meta_keys from all prior specs that support variable priors."""
    keys: list[str] = []
    for spec in specs:
        if hasattr(spec, "meta_keys"):
            keys.extend(spec.meta_keys)
    return keys


def _size_support(
    value_range: tuple[int, int],
    bucket: int | None,
    size_distribution: str = "uniform",
) -> np.ndarray:
    """Every size value a simulator can draw.

    Taken from ``simulators._size_distribution``, the same function the
    sampler draws from, rather than assuming a continuous uniform. The
    values are not always equally likely (``"log_uniform"``); use
    :func:`_log_size_moments` for anything that depends on the weights.
    """
    values, _ = _size_distribution(value_range, bucket, size_distribution)
    return values.astype(np.float64)


def _log_size_moments(
    value_range: tuple[int, int],
    bucket: int | None,
    size_distribution: str = "uniform",
    log_uniform_share: float = DEFAULT_LOG_UNIFORM_SHARE,
) -> tuple[float, float]:
    """Mean and sd of ``log(size)`` under a simulator's actual size draw.

    Probability-weighted over ``simulators._size_distribution``, so a
    log-uniform or bucketed draw is standardized against the distribution
    the simulator really samples from.

    A pinned range (``lo == hi``) has zero spread. The sd is forced to 1.0
    there so the transform stays finite and reduces to a pure shift, which is
    the correct behavior: a constant carries no information and only needs to
    be moved off a large magnitude.
    """
    values, probs = _size_distribution(
        value_range, bucket, size_distribution, log_uniform_share
    )
    logs = np.log(values.astype(np.float64))
    mean = float(np.sum(probs * logs))
    std = float(np.sqrt(np.sum(probs * (logs - mean) ** 2)))
    return mean, std if std > 0 else 1.0


def _standardize_size_conditions(adapter, simulator) -> None:
    """Log-and-standardize ``n_items`` / ``n_persons`` before they are fused.

    Why this exists
    ---------------
    Both keys are concatenated into ``inference_conditions`` and reach the
    inference network directly, while every other input is standardized to
    approximately unit scale (parameters via prior moments, responses being
    already 0/1). Left raw they arrive at 5-500 -- one to two orders of
    magnitude larger than anything else the network sees.

    Under a *fixed* size that is harmless: the value is constant, so the first
    layer's bias absorbs it. Under *variable* size it is a live, badly scaled
    input, and it risks 2PL discrimination not being recovered at all.

    Such a failure would not be merely a worse fit: the discrimination
    posterior would stay essentially the prior while difficulty is
    recovered. That state is
    *well calibrated*, because returning the prior is calibrated -- it is
    visible in contraction, not in calibration (Modrak et al., 2023).

    There is deliberately no opt-out: a parameter whose only correct value
    is the default is a trap rather than a choice.

    Applied per key *before* ``concatenate``, not to the concatenated tensor,
    so variable-prior meta keys sharing ``inference_conditions`` are left
    alone.
    """
    size_keys = (("n_items", "item_bucket"), ("n_persons", "person_bucket"))
    for key, bucket_attr in size_keys:
        value_range = getattr(simulator, key)
        bucket = getattr(simulator, bucket_attr, None)
        distribution = getattr(simulator, "size_distribution", "uniform")
        share = getattr(simulator, "log_uniform_share", DEFAULT_LOG_UNIFORM_SHARE)
        mean, std = _log_size_moments(value_range, bucket, distribution, share)
        adapter.log(key)
        adapter.standardize(key, mean=np.float32(mean), std=np.float32(std))


# ---------------------------------------------------------------------------
# Elementwise transforms
# ---------------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="bayesflow_irt")
class LogTransform(ElementwiseTransform):
    """Apply ``log`` on `forward`, ``exp`` on `inverse`.

    Useful for strictly positive parameters (e.g., discrimination ``a``):
    maps the lognormal prior to an approximately normal distribution that
    the flow can model efficiently.  The inverse guarantees positivity.
    """

    def forward(self, data: np.ndarray, **kwargs) -> np.ndarray:  # noqa: D401
        if np.any(data <= 0):
            raise ValueError(
                "LogTransform.forward requires strictly positive input; got "
                f"values <= 0 (min={np.min(data)!r})."
            )
        return np.log(data)

    def inverse(self, data: np.ndarray, **kwargs) -> np.ndarray:  # noqa: D401
        return np.exp(data)

    def get_config(self) -> dict:
        return {}


@keras.saving.register_keras_serializable(package="bayesflow_irt")
class OneHotEncodeFactory:
    """Serializable ``transform_constructor`` factory for :class:`OneHotEncode`.

    ``FilterTransform.transform_constructor`` must be a zero-argument
    callable. A ``lambda: OneHotEncode(n_categories)`` closure would not
    survive reloading: Keras's ``safe_mode=False`` lambda deserialization
    reconstructs the lambda's bytecode without repopulating its
    ``__globals__`` with the defining module's namespace: calling the
    deserialized lambda in a fresh process then raises ``NameError: name
    'OneHotEncode' is not defined``.

    This class replaces that lambda. Being a registered, named class with
    ``get_config``/``from_config``, it round-trips through Keras
    serialization (``keras.saving.save_model`` / ``load_model``) via its
    config dict instead of via unsafe bytecode, so it works identically
    whether reloaded in the same process or a fresh one.

    See Keras's serialization guidance on preferring named, serializable
    callables over lambdas: https://keras.io/guides/serialization_and_saving/
    """

    def __init__(self, n_categories: int):
        self.n_categories = n_categories
        # FilterTransform.__repr__ / extra_repr access transform_constructor
        # .__name__, which plain instances lack; provide one.
        self.__name__ = f"OneHotEncodeFactory(n_categories={n_categories})"

    def __call__(self) -> OneHotEncode:
        return OneHotEncode(self.n_categories)

    def get_config(self) -> dict:
        return {"n_categories": self.n_categories}

    @classmethod
    def from_config(cls, config: dict) -> OneHotEncodeFactory:
        return cls(**config)


@keras.saving.register_keras_serializable(package="bayesflow_irt")
class OneHotEncode(ElementwiseTransform):
    """One-hot encode integer responses: ``(…) → (…, K)``.

    Forward maps integer category indices ``{0, 1, …, K-1}`` to one-hot
    vectors of length *K*.  Inverse recovers the integer index via argmax.
    """

    def __init__(self, n_categories: int):
        self.n_categories = n_categories

    def forward(self, data: np.ndarray, **kwargs) -> np.ndarray:  # noqa: D401
        values = np.asarray(data)
        # Validate before casting: astype(int) would silently truncate
        # 0.9 / 1.9 / -0.1 into valid-looking categories.
        if (
            not np.all(np.isfinite(values))
            or np.any(values != np.floor(values))
            or np.any(values < 0)
            or np.any(values >= self.n_categories)
        ):
            finite = values[np.isfinite(values)]
            observed = (
                f"[{finite.min():g}, {finite.max():g}]" if finite.size else "none"
            )
            raise ValueError(
                "OneHotEncode.forward requires finite integer category indices "
                f"in [0, {self.n_categories}); got finite range {observed}"
                + ("" if np.all(np.isfinite(values)) else " plus non-finite values")
                + "."
            )
        idx = values.astype(int)
        return np.eye(self.n_categories, dtype=np.float32)[idx]

    def inverse(self, data: np.ndarray, **kwargs) -> np.ndarray:  # noqa: D401
        return np.argmax(data, axis=-1).astype(np.float32)

    def get_config(self) -> dict:
        return {"n_categories": self.n_categories}


# ---------------------------------------------------------------------------
# Loading reparameterization transform
# ---------------------------------------------------------------------------


class _BaseLoadingReparamTransform(Transform):
    """Shared cross-key reparameterization: ``(a, X)`` <-> ``(a, d = -a·X)``.

    Both the 2PL's scalar ``b`` and the GPCM's per-item-step ``delta`` array
    use the same ``d = -a * X`` / ``X = -d / a`` algebra; they differ only in
    which keys are read/written and whether ``a`` needs a trailing axis
    inserted to broadcast against a rank-3 ``delta``. Subclasses set
    ``_source_key``/``_target_key`` and override ``_broadcast_a`` when needed.

    Placed **first** in the adapter's forward chain, this automatically runs
    **last** during inversion — exactly when ``a`` and the target key are
    both in their original scales.

    Both ``forward`` and ``inverse`` guard against missing keys by checking
    membership before operating. This is necessary because
    ``approximator.sample(conditions=...)`` runs the adapter in condition-only
    mode where inference variables are absent — only the response matrix and
    metadata flow through.
    """

    _source_key: str
    _target_key: str

    @staticmethod
    def _broadcast_a(a: np.ndarray) -> np.ndarray:
        return a

    def forward(self, data: dict, **kwargs) -> dict:
        data = dict(data)
        # In sampling-time condition paths, adapter input may not include
        # inference variables. In that case this transform should be a no-op.
        if "a" in data and self._source_key in data:
            data[self._target_key] = -(
                self._broadcast_a(data["a"]) * data[self._source_key]
            )
            del data[self._source_key]
        return data

    def inverse(self, data: dict, **kwargs) -> dict:
        data = dict(data)
        if "a" in data and self._target_key in data:
            data[self._source_key] = -(
                data[self._target_key] / self._broadcast_a(data["a"])
            )
            del data[self._target_key]
        return data

    def get_config(self) -> dict:
        return {}


@keras.saving.register_keras_serializable(package="bayesflow_irt")
class _LoadingReparamTransform(_BaseLoadingReparamTransform):
    """Cross-key reparameterization: ``(a, b)`` ↔ ``(a, d = -a·b)``.

    **Forward** (training path): receives raw per-item arrays ``a`` and ``b``
    from the simulator, computes the intercept ``d = -a * b``, replaces ``b``
    with ``d`` in the data dict.

    **Inverse** (sampling path): receives ``a`` (already restored to original
    lognormal scale by prior inverse transforms) and ``d`` (already
    un-standardized), computes ``b = -d / a``, replaces ``d`` with ``b``.
    """

    _source_key = "b"
    _target_key = "d"


@keras.saving.register_keras_serializable(package="bayesflow_irt")
class _GPCMLoadingReparamTransform(_BaseLoadingReparamTransform):
    """Cross-key reparameterization: ``(a, delta)`` <-> ``(a, d_delta)``.

    **Forward**: computes ``d_delta = -a[:, :, None] * delta``.
    **Inverse**: computes ``delta = -d_delta / a[:, :, None]``.

    Uses key ``d_delta`` to avoid collision with the 2PL's scalar ``d``.

    Note: ``a`` must be positive (guaranteed by ``log_transform_a=True``
    default); division by zero is not possible under normal usage.
    """

    _source_key = "delta"
    _target_key = "d_delta"

    @staticmethod
    def _broadcast_a(a: np.ndarray) -> np.ndarray:
        return a[..., np.newaxis]


def _prior_moments_d_gpcm(
    prior_a: PriorSpec,
    prior_delta: PriorSpec,
    n_samples: int = 100_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Empirical ``(mean, std)`` of ``d = -a * delta`` under the joint prior.

    Draws 1D marginal samples and returns moments of ``d = -a * delta``.
    Used to standardize all K-1 ``d_delta`` channels in the GPCM loading
    adapter.  The independence assumption matches ``_prior_moments_d``.
    """
    a_samples = sample_prior_predictive(
        prior_a,
        "a",
        n_samples=n_samples,
        seed=seed,
    )
    delta_samples = sample_prior_predictive(
        prior_delta,
        "delta",
        n_samples=n_samples,
        seed=seed + 1,
    )
    d = -(a_samples * delta_samples)
    return float(d.mean()), float(d.std())


def _prior_moments_d(
    prior_a: PriorSpec,
    prior_b: PriorSpec,
    n_samples: int = 100_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Empirical ``(mean, std)`` of ``d = -a * b`` under the joint prior.

    Draws ``n_samples`` independent pairs ``(a, b)`` from their respective
    priors and returns the empirical moments of ``d = -a * b``.  Used to
    standardize the ``d`` channel in the loading adapter.
    """
    a_samples = sample_prior_predictive(prior_a, "a", n_samples=n_samples, seed=seed)
    b_samples = sample_prior_predictive(
        prior_b, "b", n_samples=n_samples, seed=seed + 1
    )
    d = -(a_samples * b_samples)
    return float(d.mean()), float(d.std())


# ---------------------------------------------------------------------------
# 2PL adapters
# ---------------------------------------------------------------------------


def make_2pl_adapter(
    simulator,
    *,
    log_transform_a: bool = True,
    internal_parameterization: str = "loading",
) -> bf.adapters.Adapter:
    """Build a BayesFlow Adapter for variable-size 2PL IRT.

    The output ``inference_variables`` has shape ``(batch, I, 2)`` —
    per-item parameter pairs.  Prior-based standardization is always
    applied so the flow sees approximately unit-variance inputs.

    Parameters
    ----------
    simulator
        A 2PL simulator (returned by :func:`make_2pl_simulator`).
        The adapter reads ``prior_a`` and ``prior_b`` from the
        simulator to compute standardization moments.
    log_transform_a : bool
        If ``True`` (default), apply ``log`` to the discrimination
        parameter *a* before it enters the flow, and ``exp`` on the
        inverse path.  This maps the lognormal prior to an
        approximately normal distribution that the flow can model
        efficiently, and guarantees positivity of posterior samples.
        Disable only if your prior on *a* is already roughly symmetric
        (e.g. a truncated normal).
    internal_parameterization : str
        ``"loading"`` (default): internally reparameterizes to the intercept
        ``d_i = -a_i b_i`` before the flow and converts back via
        ``b_i = -d_i / a_i`` when sampling posteriors.  The
        user-facing API (priors, returned posterior keys ``"a"`` and
        ``"b"``) is unchanged.  This can improve *b* recovery by
        decoupling the intercept from the discrimination scale —
        ``d_i`` is more directly informed by item endorsement rates,
        giving the flow better-conditioned gradients for the difficulty
        component.

        ``"standard"``: the flow learns ``(a_i, b_i)`` directly;
        ``inference_variables[..., 0]`` = (log-)a,
        ``inference_variables[..., 1]`` = b.

    Returns
    -------
    bayesflow.adapters.Adapter
    """
    if internal_parameterization not in ("standard", "loading"):
        raise ValueError(
            f"internal_parameterization={internal_parameterization!r} is not valid. "
            "Choose 'standard' or 'loading'."
        )

    for attr in ("prior_a", "prior_b"):
        if not hasattr(simulator, attr):
            raise TypeError(
                f"Expected a 2PL simulator with a {attr!r} attribute "
                f"(from make_2pl_simulator), got {type(simulator).__name__}."
            )

    prior_a = simulator.prior_a
    prior_b = simulator.prior_b

    adapter = bf.adapters.Adapter()

    # Convert everything to arrays and float32
    adapter.to_array()
    adapter.convert_dtype("float64", "float32")

    if internal_parameterization == "loading":
        # Reparameterize b → d = -a*b BEFORE any per-key transforms so that
        # both a and b are still in their raw (unstandardized, unlogged) scales.
        # During inversion this transform runs LAST, after a has been exp()'d
        # back to original scale, and computes b = -d/a.
        adapter.add_transform(_LoadingReparamTransform())

    # ── a channel ──────────────────────────────────────────────────────────
    adapter.expand_dims("a", axis=-1)

    if log_transform_a:
        # log(a) ≈ Normal → much easier for the velocity network;
        # inverse (exp) guarantees a > 0 when sampling posteriors.
        adapter.add_transform(
            FilterTransform(
                include="a",
                transform_constructor=LogTransform,
            )
        )

    a_mean, a_std = prior_moments(
        prior_a,
        "a",
        transform=np.log if log_transform_a else None,
    )
    adapter.standardize("a", mean=np.float32(a_mean), std=np.float32(a_std))

    # ── b / d channel ───────────────────────────────────────────────────────
    second_key = "d" if internal_parameterization == "loading" else "b"
    adapter.expand_dims(second_key, axis=-1)

    if internal_parameterization == "loading":
        d_mean, d_std = _prior_moments_d(prior_a, prior_b)
        adapter.standardize("d", mean=np.float32(d_mean), std=np.float32(d_std))
    else:
        b_mean, b_std = prior_moments(prior_b, "b")
        adapter.standardize("b", mean=np.float32(b_mean), std=np.float32(b_std))

    adapter.concatenate(["a", second_key], into="inference_variables")
    adapter.rename("responses", "summary_variables")

    # Keep size info as conditions, plus sigma values from variable priors
    _standardize_size_conditions(adapter, simulator)
    condition_keys = ["n_items", "n_persons"] + _collect_meta_keys(prior_a, prior_b)
    adapter.concatenate(condition_keys, into="inference_conditions")

    # Drop theta (nuisance)
    adapter.keep(["inference_variables", "summary_variables", "inference_conditions"])

    # Internal flag used by EquivariantIRTApproximator.sample() to expose
    # auxiliary keys expected by external validation tooling.
    adapter._bayesflow_irt_internal_parameterization = internal_parameterization

    return adapter


# ---------------------------------------------------------------------------
# 1PL adapters
# ---------------------------------------------------------------------------


def make_1pl_adapter(
    simulator,
) -> bf.adapters.Adapter:
    """Build a BayesFlow Adapter for variable-size 1PL (Rasch) IRT.

    The output ``inference_variables`` has shape ``(batch, I, 1)`` —
    per-item difficulty only.  Prior-based standardization is always
    applied so the flow sees approximately unit-variance inputs.

    Parameters
    ----------
    simulator
        A 1PL simulator (returned by :func:`make_1pl_simulator`).
        The adapter reads ``prior_b`` from the simulator to compute
        standardization moments.

    Returns
    -------
    bayesflow.adapters.Adapter
    """
    if not hasattr(simulator, "prior_b"):
        raise TypeError(
            f"Expected a 1PL simulator with a 'prior_b' attribute "
            f"(from make_1pl_simulator), got {type(simulator).__name__}."
        )

    prior_b = simulator.prior_b

    adapter = bf.adapters.Adapter()
    adapter.to_array()
    adapter.convert_dtype("float64", "float32")

    # For 1PL, only b is the target: (batch, I) → (batch, I, 1)
    adapter.expand_dims("b", axis=-1)

    b_mean, b_std = prior_moments(prior_b, "b")
    adapter.standardize("b", mean=np.float32(b_mean), std=np.float32(b_std))

    adapter.rename("b", "inference_variables")
    adapter.rename("responses", "summary_variables")
    _standardize_size_conditions(adapter, simulator)
    condition_keys = ["n_items", "n_persons"] + _collect_meta_keys(prior_b)
    adapter.concatenate(condition_keys, into="inference_conditions")
    adapter.keep(["inference_variables", "summary_variables", "inference_conditions"])

    return adapter


# ---------------------------------------------------------------------------
# PCM adapters
# ---------------------------------------------------------------------------


def make_pcm_adapter(
    simulator,
) -> bf.adapters.Adapter:
    """Build a BayesFlow Adapter for variable-size PCM IRT.

    The output ``inference_variables`` has shape ``(batch, I, K-1)`` —
    per-item step difficulty vectors.  Responses are one-hot encoded to
    ``(batch, P, I, K)`` for ``summary_variables``.  Prior-based
    standardization is always applied so the flow sees approximately
    unit-variance inputs.

    Parameters
    ----------
    simulator
        A PCM simulator (returned by :func:`make_pcm_simulator`).
        The adapter reads ``prior_delta`` and ``n_categories`` from
        the simulator.

    Returns
    -------
    bayesflow.adapters.Adapter
    """
    for attr in ("prior_delta", "n_categories"):
        if not hasattr(simulator, attr):
            raise TypeError(
                f"Expected a PCM simulator with a {attr!r} attribute "
                f"(from make_pcm_simulator), got {type(simulator).__name__}."
            )

    prior_delta = simulator.prior_delta
    n_categories = simulator.n_categories

    adapter = bf.adapters.Adapter()
    adapter.to_array()
    adapter.convert_dtype("float64", "float32")

    # Step difficulties are already (B, I, K-1) — no expand_dims needed
    d_mean, d_std = prior_moments(prior_delta, "delta")
    adapter.standardize("delta", mean=np.float32(d_mean), std=np.float32(d_std))

    adapter.rename("delta", "inference_variables")

    # One-hot encode responses: (B, P, I) → (B, P, I, K)
    adapter.add_transform(
        FilterTransform(
            include="responses",
            transform_constructor=OneHotEncodeFactory(n_categories),
        )
    )
    adapter.rename("responses", "summary_variables")

    _standardize_size_conditions(adapter, simulator)
    condition_keys = ["n_items", "n_persons"] + _collect_meta_keys(prior_delta)
    adapter.concatenate(condition_keys, into="inference_conditions")
    adapter.keep(["inference_variables", "summary_variables", "inference_conditions"])

    return adapter


# ---------------------------------------------------------------------------
# GPCM adapters
# ---------------------------------------------------------------------------


def make_gpcm_adapter(
    simulator,
    *,
    log_transform_a: bool = True,
    internal_parameterization: str = "loading",
) -> bf.adapters.Adapter:
    """Build a BayesFlow Adapter for variable-size GPCM IRT.

    The output ``inference_variables`` has shape ``(batch, I, K)`` where
    ``K = n_categories = 1 + (K-1)`` -- per-item parameter vectors
    containing the discrimination and step parameters.

    Parameters
    ----------
    simulator
        A GPCM simulator (returned by :func:`make_gpcm_simulator`).
    log_transform_a : bool
        If ``True`` (default), apply ``log`` to ``a`` before the flow.
    internal_parameterization : str
        ``"loading"`` (default) or ``"standard"``.

    Returns
    -------
    bayesflow.adapters.Adapter

    References
    ----------
    Muraki, E. (1992). A generalized partial credit model: Application
    of an EM algorithm. *Applied Psychological Measurement*, 16(2),
    159-176.
    """
    if internal_parameterization not in ("standard", "loading"):
        raise ValueError(
            f"internal_parameterization="
            f"{internal_parameterization!r} is not valid. "
            "Choose 'standard' or 'loading'."
        )

    for attr in ("prior_a", "prior_delta", "n_categories"):
        if not hasattr(simulator, attr):
            raise TypeError(
                f"Expected a GPCM simulator with a {attr!r} "
                f"attribute (from make_gpcm_simulator), "
                f"got {type(simulator).__name__}."
            )

    prior_a = simulator.prior_a
    prior_delta = simulator.prior_delta
    n_categories = simulator.n_categories

    adapter = bf.adapters.Adapter()
    adapter.to_array()
    adapter.convert_dtype("float64", "float32")

    if internal_parameterization == "loading":
        adapter.add_transform(_GPCMLoadingReparamTransform())

    # -- a channel: (B, I) -> (B, I, 1) --
    adapter.expand_dims("a", axis=-1)

    if log_transform_a:
        adapter.add_transform(
            FilterTransform(
                include="a",
                transform_constructor=LogTransform,
            )
        )

    a_mean, a_std = prior_moments(
        prior_a,
        "a",
        transform=np.log if log_transform_a else None,
    )
    adapter.standardize(
        "a",
        mean=np.float32(a_mean),
        std=np.float32(a_std),
    )

    # -- delta / d_delta channel: already 3D (B, I, K-1) --
    second_key = "d_delta" if internal_parameterization == "loading" else "delta"

    if internal_parameterization == "loading":
        d_mean, d_std = _prior_moments_d_gpcm(prior_a, prior_delta)
        adapter.standardize(
            "d_delta",
            mean=np.float32(d_mean),
            std=np.float32(d_std),
        )
    else:
        d_mean, d_std = prior_moments(prior_delta, "delta")
        adapter.standardize(
            "delta",
            mean=np.float32(d_mean),
            std=np.float32(d_std),
        )

    # (B, I, 1) + (B, I, K-1) -> (B, I, K)
    adapter.concatenate(
        ["a", second_key],
        into="inference_variables",
    )

    # One-hot encode responses: (B, P, I) -> (B, P, I, K)
    adapter.add_transform(
        FilterTransform(
            include="responses",
            transform_constructor=OneHotEncodeFactory(n_categories),
        )
    )
    adapter.rename("responses", "summary_variables")

    _standardize_size_conditions(adapter, simulator)
    condition_keys = ["n_items", "n_persons"] + _collect_meta_keys(prior_a, prior_delta)
    adapter.concatenate(
        condition_keys,
        into="inference_conditions",
    )
    adapter.keep(
        [
            "inference_variables",
            "summary_variables",
            "inference_conditions",
        ]
    )

    adapter._bayesflow_irt_internal_parameterization = internal_parameterization

    return adapter
