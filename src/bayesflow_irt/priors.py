"""Prior specifications for IRT model parameters.

Provides the :class:`Prior` dataclass for configurable priors, a registry of
built-in distributions, and :func:`sample_prior_predictive` for inspecting
priors before fitting.

Built-in distributions
----------------------

.. list-table::
   :header-rows: 1
   :widths: 15 35 15 15

   * - Name
     - Parameters (defaults)
     - Support
     - Roles
   * - ``"normal"``
     - ``loc=0.0``, ``scale=1.0``
     - :math:`(-\\infty, \\infty)`
     - b, theta
   * - ``"lognormal"``
     - ``loc=0.0``, ``scale=0.5``  (of the underlying normal)
     - :math:`(0, \\infty)`
     - a
   * - ``"halfnorm"``
     - ``scale=1.0``
     - :math:`[0, \\infty)`
     - a
   * - ``"truncnorm"``
     - ``loc=1.0``, ``scale=0.5``, ``low=0.1``, ``high=None``
     - :math:`[\\text{low}, \\text{high}]`
     - a
   * - ``"uniform"``
     - ``low=0.0``, ``high=1.0``
     - :math:`[\\text{low}, \\text{high}]`
     - a, b, theta
   * - ``"gamma"``
     - ``concentration=2.0``, ``scale=0.5``
     - :math:`(0, \\infty)`
     - a
   * - ``"student_t"``
     - ``df=5.0``, ``loc=0.0``, ``scale=1.0``
     - :math:`(-\\infty, \\infty)`
     - b, theta

The **Roles** column shows which parameter roles accept each distribution.
Discriminations (``"a"``) must be positive, so only positive-support
distributions are allowed.  Difficulties (``"b"``) and abilities
(``"theta"``) are unconstrained.

Examples
--------
>>> from bayesflow_irt import Prior, sample_prior_predictive
>>> samples = sample_prior_predictive(
...     Prior("lognormal", loc=0.3, scale=0.4), "a", seed=0
... )
>>> samples.shape
(10000,)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Union

import numpy as np
from scipy.stats import truncnorm

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

#: A prior specification: a :class:`Prior` dataclass (named distribution with
#: hyperparameters), a :class:`VariablePrior`, or a callable
#: ``(shape, rng) -> ndarray``.
PriorSpec = Union[
    "Prior", "VariablePrior", Callable[[tuple, np.random.Generator], np.ndarray]
]

#: Which IRT parameter the prior is for.
ParameterRole = Literal["a", "b", "theta", "delta"]


# ---------------------------------------------------------------------------
# Prior dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Prior:
    """Specification of a prior distribution with configurable hyperparameters.

    Parameters
    ----------
    name : str
        Distribution name.  See the module-level table for available
        distributions, their parameters, and which roles accept them.
    **kwargs
        Hyperparameters passed to the sampling function.  Any parameter
        not provided falls back to the distribution's default (shown in
        the module-level table).

    Notes
    -----
    For ``"lognormal"``, ``loc`` and ``scale`` parameterize the
    **underlying normal**, not the lognormal moments.  That is,
    ``Prior("lognormal", loc=0, scale=1)`` draws from
    :math:`\\text{LogNormal}(\\mu{=}0,\\,\\sigma{=}1)` where
    :math:`\\mu` and :math:`\\sigma` are the mean and standard deviation
    of :math:`\\ln X`.

    Instead of a ``Prior``, simulators also accept a bare callable with
    signature ``(shape, rng) -> np.ndarray`` for full control over
    sampling (see :func:`resolve_prior`).

    Examples
    --------
    >>> Prior("normal", loc=0, scale=2)
    Prior('normal', loc=0, scale=2)
    >>> Prior("lognormal", loc=0, scale=0.5)
    Prior('lognormal', loc=0, scale=0.5)
    >>> Prior("halfnorm", scale=1.0)
    Prior('halfnorm', scale=1.0)
    >>> Prior("truncnorm", loc=1, scale=0.5, low=0.1)
    Prior('truncnorm', loc=1, scale=0.5, low=0.1)
    >>> Prior("uniform", low=0.3, high=3.0)
    Prior('uniform', low=0.3, high=3.0)
    >>> Prior("gamma", concentration=4, scale=0.25)
    Prior('gamma', concentration=4, scale=0.25)
    >>> Prior("student_t", df=3, loc=0, scale=1)
    Prior('student_t', df=3, loc=0, scale=1)
    """

    name: str
    kwargs: dict = field(default_factory=dict)

    def __init__(self, name: str, **kwargs):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kwargs", kwargs)

    def __repr__(self) -> str:
        if self.kwargs:
            kw = ", ".join(f"{k}={v!r}" for k, v in self.kwargs.items())
            return f"Prior({self.name!r}, {kw})"
        return f"Prior({self.name!r})"


# ---------------------------------------------------------------------------
# Variable-scale prior
# ---------------------------------------------------------------------------


class VariablePrior:
    """A prior whose scale parameter varies across batches.

    Each call draws a new scale ``σ`` from ``Uniform(*scale_range)`` and
    then samples from the base distribution with that ``σ``.  The ``σ``
    value is emitted as metadata so that the adapter can feed it as an
    inference condition.

    The ``__call__`` method performs *marginal* sampling (each element
    gets an independent σ) — suitable for ``prior_moments`` and
    ``sample_prior_predictive``.  The ``sample_with_meta`` method
    performs *conditional* sampling (one σ per batch) and returns the
    metadata dict — used by simulators.

    Parameters
    ----------
    name : str
        Distribution name from the built-in registry (e.g. ``"normal"``).
    role : str
        Parameter role (``"a"``, ``"b"``, ``"theta"``, or ``"delta"``).
        Used for validation.
    scale_range : tuple[float, float]
        ``(lo, hi)`` for the uniform meta-prior on the scale parameter.
        Must satisfy ``0 < lo < hi``.
    **fixed_kwargs
        Additional hyperparameters forwarded to the sampling function
        (e.g. ``loc=0.0``).

    Examples
    --------
    >>> vp = VariablePrior("normal", role="b", scale_range=(0.5, 3.0))
    >>> rng = np.random.default_rng(42)
    >>> samples = vp((100,), rng)  # marginal
    >>> samples.shape
    (100,)
    >>> samples, meta = vp.sample_with_meta((8, 20), rng)  # conditional
    >>> samples.shape
    (8, 20)
    >>> "sigma_b" in meta
    True
    """

    def __init__(
        self,
        name: str,
        *,
        role: str,
        scale_range: tuple[float, float],
        **fixed_kwargs,
    ):
        if name not in _DISTRIBUTION_REGISTRY:
            raise ValueError(
                f"Unknown distribution {name!r}. "
                f"Available: {sorted(_DISTRIBUTION_REGISTRY)}."
            )
        # VariablePrior varies the 'scale' parameter, so distributions that
        # don't use scale (e.g., 'uniform' uses low/high) are not supported.
        if name == "uniform":
            raise ValueError(
                "VariablePrior is not compatible with 'uniform' distribution "
                "(which uses low/high, not scale). Use a distribution that "
                "accepts a scale parameter (e.g., 'normal', 'lognormal')."
            )
        allowed = _ROLE_ALLOWED.get(role, set())
        if name not in allowed:
            raise ValueError(
                f"Distribution {name!r} is not valid for role {role!r}. "
                f"Allowed: {sorted(allowed)}."
            )
        _check_positive_truncnorm(name, role, fixed_kwargs)
        lo, hi = scale_range
        if lo <= 0 or hi <= lo:
            raise ValueError(
                f"scale_range must satisfy 0 < lo < hi, got {scale_range!r}."
            )

        self.name = name
        self.role = role
        self.scale_range = scale_range
        self.fixed_kwargs = fixed_kwargs
        self.meta_keys: list[str] = [f"sigma_{role}"]

    def __call__(self, shape, rng) -> np.ndarray:
        """Marginal sampling: each element gets an independent σ."""
        n = int(np.prod(shape))
        sigmas = rng.uniform(
            self.scale_range[0],
            self.scale_range[1],
            size=n,
        )
        sample_fn = _DISTRIBUTION_REGISTRY[self.name]
        kwargs = dict(self.fixed_kwargs)
        kwargs["scale"] = sigmas
        samples = sample_fn((n,), rng, **kwargs)
        return samples.reshape(shape).astype(np.float32)

    def sample_with_meta(
        self,
        shape,
        rng,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Conditional sampling: one σ per batch element.

        Returns
        -------
        samples : np.ndarray
            Shape ``shape``.
        meta : dict[str, float]
            ``{f"sigma_{role}": σ}`` with one σ per batch.
        """
        if len(shape) < 2:
            raise ValueError(
                f"sample_with_meta requires at least 2D shape (B, ...), got {shape}."
            )
        batch_size = shape[0]
        sigmas = rng.uniform(
            self.scale_range[0],
            self.scale_range[1],
            size=batch_size,
        ).astype(np.float32)

        sample_fn = _DISTRIBUTION_REGISTRY[self.name]
        kwargs = dict(self.fixed_kwargs)
        kwargs["scale"] = sigmas.reshape((batch_size,) + (1,) * (len(shape) - 1))
        results = sample_fn(shape, rng, **kwargs).astype(np.float32)

        meta = {self.meta_keys[0]: sigmas}
        return results, meta

    def __repr__(self) -> str:
        kw = ", ".join(
            [f"role={self.role!r}", f"scale_range={self.scale_range!r}"]
            + [f"{k}={v!r}" for k, v in self.fixed_kwargs.items()]
        )
        return f"VariablePrior({self.name!r}, {kw})"


# ---------------------------------------------------------------------------
# Built-in sampling functions
# ---------------------------------------------------------------------------


def _sample_normal(shape, rng, *, loc=0.0, scale=1.0):
    return rng.normal(loc, scale, size=shape)


def _sample_lognormal(shape, rng, *, loc=0.0, scale=0.5):
    return rng.lognormal(loc, scale, size=shape)


def _sample_halfnorm(shape, rng, *, scale=1.0):
    return np.abs(rng.normal(0, scale, size=shape))


def _sample_truncnorm(shape, rng, *, loc=1.0, scale=0.5, low=0.1, high=None):
    a = (low - loc) / scale
    b = np.inf if high is None else (high - loc) / scale
    return truncnorm.rvs(a, b, loc=loc, scale=scale, size=shape, random_state=rng)


def _sample_uniform(shape, rng, *, low=0.0, high=1.0):
    return rng.uniform(low, high, size=shape)


def _sample_gamma(shape, rng, *, concentration=2.0, scale=0.5):
    return rng.gamma(concentration, scale, size=shape)


def _sample_student_t(shape, rng, *, df=5.0, loc=0.0, scale=1.0):
    return rng.standard_t(df, size=shape) * scale + loc


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Maps distribution name -> sampling function.
_DISTRIBUTION_REGISTRY: dict[str, Callable] = {
    "normal": _sample_normal,
    "lognormal": _sample_lognormal,
    "halfnorm": _sample_halfnorm,
    "truncnorm": _sample_truncnorm,
    "uniform": _sample_uniform,
    "gamma": _sample_gamma,
    "student_t": _sample_student_t,
}

#: Maps parameter role -> set of allowed distribution names.
#: Discriminations ("a") must be strictly positive in IRT, so only
#: distributions with positive support are permitted.  Difficulties ("b")
#: and abilities ("theta") are unconstrained in sign.
_ROLE_ALLOWED: dict[str, set[str]] = {
    "a": {"lognormal", "halfnorm", "truncnorm", "gamma", "uniform"},
    "b": {"normal", "uniform", "student_t"},
    "theta": {"normal", "uniform", "student_t"},
    "delta": {"normal", "uniform", "student_t"},
}


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def resolve_prior(
    spec: PriorSpec,
    role: ParameterRole,
) -> Callable[[tuple, np.random.Generator], np.ndarray]:
    """Resolve a prior specification into a sampling callable.

    Accepts either a :class:`Prior` dataclass (validated against the
    distribution registry and role constraints) or a bare callable for
    full flexibility.

    Parameters
    ----------
    spec : Prior | callable
        Prior specification.
    role : {"a", "b", "theta", "delta"}
        Which IRT parameter the prior is for.  Used for validation
        (e.g. discriminations must be positive).

    Returns
    -------
    callable
        ``(shape, rng) -> np.ndarray``

    Raises
    ------
    TypeError
        If *spec* is not :class:`Prior` or callable.
    ValueError
        If the distribution name is unknown or not allowed for *role*.

    Examples
    --------
    Using a :class:`Prior`:

    >>> sampler = resolve_prior(Prior("normal", loc=0, scale=2), "b")
    >>> samples = sampler((1000,), np.random.default_rng(0))
    >>> samples.shape
    (1000,)

    Using a custom callable:

    >>> sampler = resolve_prior(
    ...     lambda shape, rng: np.exp(rng.normal(0, 0.3, size=shape)),
    ...     "a",
    ... )
    >>> samples = sampler((500,), np.random.default_rng(0))
    >>> bool((samples > 0).all())
    True
    """
    # --- callable (advanced) ---
    if callable(spec) and not isinstance(spec, Prior):
        return _wrap_callable(spec, role)

    # --- Prior dataclass ---
    if isinstance(spec, Prior):
        return _resolve_prior_dataclass(spec, role)

    raise TypeError(f"Prior spec must be Prior or callable, got {type(spec).__name__}.")


def _check_positive_truncnorm(name: str, role: str, kwargs: dict) -> None:
    """Reject truncated normals for 'a' whose support includes non-positive values."""
    if role == "a" and name == "truncnorm" and kwargs.get("low", 0.1) <= 0:
        raise ValueError(
            f"Truncnorm prior for 'a' requires low > 0, got low={kwargs['low']}."
        )


def _resolve_prior_dataclass(
    prior: Prior,
    role: ParameterRole,
) -> Callable[[tuple, np.random.Generator], np.ndarray]:
    """Resolve a :class:`Prior` into a sampling callable."""
    name = prior.name
    allowed = _ROLE_ALLOWED.get(role, set())

    if name not in _DISTRIBUTION_REGISTRY:
        raise ValueError(
            f"Unknown distribution {name!r}. "
            f"Available: {sorted(_DISTRIBUTION_REGISTRY)}."
        )
    if name not in allowed:
        raise ValueError(
            f"Distribution {name!r} is not valid for parameter role {role!r}. "
            f"Allowed for {role!r}: {sorted(allowed)}."
        )

    sample_fn = _DISTRIBUTION_REGISTRY[name]

    # Use only the user-provided kwargs (no role-specific defaults)
    merged = dict(prior.kwargs)

    # Validate uniform positivity for "a" at construction time
    if role == "a" and name == "uniform" and merged.get("low", 0.0) <= 0:
        raise ValueError(
            f"Uniform prior for 'a' requires low > 0, got low={merged.get('low', 0.0)}."
        )
    _check_positive_truncnorm(name, role, merged)

    # Return a bare closure — no per-call validation overhead
    def _sampler(shape, rng):
        return sample_fn(shape, rng, **merged)

    return _sampler


def _wrap_callable(
    fn: Callable,
    role: ParameterRole,
) -> Callable[[tuple, np.random.Generator], np.ndarray]:
    """Wrap a user callable with runtime shape and constraint validation."""

    def _validated_sampler(shape, rng):
        result = fn(shape, rng)
        if not isinstance(result, np.ndarray):
            result = np.asarray(result)
        expected = shape if isinstance(shape, tuple) else (shape,)
        if result.shape != expected:
            raise ValueError(
                f"Custom prior for {role!r} returned shape {result.shape}, "
                f"expected {expected}."
            )
        if role == "a" and np.any(result <= 0):
            raise ValueError(
                f"Custom prior for 'a' must return strictly positive values, "
                f"but got min={result.min():.4f}."
            )
        return result

    return _validated_sampler


# ---------------------------------------------------------------------------
# Compound priors
# ---------------------------------------------------------------------------


def _cumsum_sort_center(mu: np.ndarray, eps: np.ndarray, k_minus_1: int) -> np.ndarray:
    """Cumsum -> sort -> mean-center construction shared by incremental priors.

    Builds ordered, mean-centered step difficulties from a per-item mean
    ``mu`` and raw per-step increments ``eps``, per the construction
    documented in :func:`make_incremental_prior`. Assumes ``k_minus_1 > 1``;
    callers handle the ``K=2`` / single-threshold case separately (no
    increments exist then).
    """
    span_factor = np.float32(max(k_minus_1 - 1, 1))
    scaled = eps / span_factor
    cumsum = np.sort(np.cumsum(scaled, axis=-1), axis=-1)
    cumsum_mean = cumsum.mean(axis=-1, keepdims=True)
    return cumsum - cumsum_mean + mu[..., np.newaxis]


def make_incremental_prior(
    delta_mean: PriorSpec,
    delta_cat_sep: PriorSpec,
    n_categories: int,
) -> Callable[[tuple, np.random.Generator], np.ndarray]:
    """Create a structured step-difficulty prior from mean + increments.

    Decomposes PCM step difficulties into a per-item **mean difficulty**
    ``delta_mean`` and **category-separation increments** ``delta_cat_sep``
    between successive thresholds.  This is natural for testlet sum scores,
    where thresholds are approximately ordered and the overall difficulty
    level varies across items.

    Construction (for each item):

    .. math::

        \\mu_i &\\sim \\text{delta\\_mu} \\\\
        \\varepsilon_{ij} &\\sim \\text{delta\\_cat\\_sep},
        \\quad j = 1, \\ldots, K{-}1 \\
        c_j &= \\operatorname{sort}\\!\\left(
        \\sum_{l=1}^{j} \\varepsilon_{il}\\right) / \\max(K{-}2,\\,1) \\
        \\delta_{ij} &= c_j - \\bar{c} + \\mu_i

    For increments that are (almost surely) nonnegative, the ``1/max(K-2, 1)``
    scaling keeps the **expected total threshold span** (``max δ − min δ``)
    equal to ``loc`` for all ``K ≥ 3``: the span is the sum of the ``K-2``
    increments after the first, each scaled by ``1/(K-2)``.  The prior then
    produces comparable threshold coverage regardless of the number of
    categories.  Thresholds are sorted along the last axis to guarantee
    ordering when a step increment is negative.  With signed increments the
    span is the range of the partial sums, which is not ``loc``: it is close
    to ``loc`` only when ``loc`` is large relative to ``scale`` (e.g. about
    2.0 for ``loc=2.0, scale=0.5`` and 2.03-2.07 for ``loc=2.0, scale=1.5``,
    ``K = 4..7``), and shrinks with ``K`` for zero-mean increments (about 0.8
    for ``K=3`` to 0.5 for ``K=7`` with ``loc=0, scale=1``).

    Parameters
    ----------
    delta_mean : Prior | callable
        Prior on the per-item mean step difficulty (unconstrained).
        E.g., ``Prior("normal", loc=0, scale=1.5)``.
    delta_cat_sep : Prior | callable
        Prior on the category-separation increments between successive
        thresholds (unconstrained).  The ``loc`` parameter sets the
        **expected total threshold span** ``max(δ) − min(δ)`` in logits.
        For nonnegative increments this stays equal to ``loc`` for all
        ``K ≥ 3`` due to the ``1/max(K-2, 1)`` normalisation; for signed
        increments it only approximates ``loc`` (see above).  Recommended:
        ``Prior("normal", loc=2.0, scale=0.5)``.
    n_categories : int
        Number of response categories ``K``.  Must be >= 2 and must match
        the simulator's ``n_categories``.

    Returns
    -------
    callable
        ``(shape, rng) -> np.ndarray``, compatible with ``PriorSpec``
        for ``prior_delta``.

    Notes
    -----
    Why not ``1/sqrt(K-1)``: that normalisation keeps the per-threshold
    marginal variance stable but lets the expected span grow as
    ``(K-2)/sqrt(K-1) * loc``.  For ``K=7`` this inflates the expected span
    to roughly ``4.1 * loc / 2`` logits, pushing outer thresholds beyond the
    typical person-ability range and causing category sparsity.  With the
    normalisation used here, ``loc=2.0`` with nonnegative increments gives an
    expected span of 2 logits (about 2 logits as well for
    ``Prior("normal", loc=2.0, scale=0.5)``).

    Examples
    --------
    >>> prior_delta = make_incremental_prior(
    ...     delta_mean=Prior("normal", loc=0, scale=2.0),
    ...     delta_cat_sep=Prior("normal", loc=2.0, scale=0.5),
    ...     n_categories=5,
    ... )
    >>> rng = np.random.default_rng(0)
    >>> delta = prior_delta((8, 10, 4), rng)
    >>> delta.shape
    (8, 10, 4)
    """
    if not isinstance(n_categories, int) or n_categories < 2:
        raise ValueError(f"n_categories must be an integer >= 2, got {n_categories!r}.")
    k_minus_1 = n_categories - 1

    mean_sampler = resolve_prior(delta_mean, "delta")
    incr_sampler = resolve_prior(delta_cat_sep, "delta")

    def _sample_3d(shape: tuple, rng: np.random.Generator) -> np.ndarray:
        batch_size, n_items, k_minus_1 = shape
        mu = mean_sampler((batch_size, n_items), rng).astype(np.float32)

        if k_minus_1 == 1:
            # K=2: single threshold, no increments
            return mu[:, :, np.newaxis]

        eps = incr_sampler((batch_size, n_items, k_minus_1), rng).astype(np.float32)
        return _cumsum_sort_center(mu, eps, k_minus_1)

    def _sampler(shape: tuple, rng: np.random.Generator) -> np.ndarray:
        if len(shape) == 3:
            if shape[-1] != k_minus_1:
                raise ValueError(
                    f"make_incremental_prior was created with n_categories="
                    f"{n_categories} (K-1={k_minus_1}), but received shape "
                    f"{shape} with last dimension {shape[-1]}."
                )
            return _sample_3d(shape, rng)

        if len(shape) == 1:
            # 1D fallback for sample_prior_predictive / prior_moments:
            # simulate virtual items and flatten to get marginal samples
            n = shape[0]
            n_items = max(1, n // k_minus_1 + 1)
            full = _sample_3d((1, n_items, k_minus_1), rng)
            return full.ravel()[:n]

        raise ValueError(f"make_incremental_prior expects 1D or 3D shape, got {shape}.")

    return _sampler


class _VariableIncrementalPrior:
    """Variable-scale incremental prior for PCM/GPCM.

    Both ``mu_sigma`` (mean difficulty scale) and ``eps_sigma`` (category
    separation scale) are drawn from uniform meta-distributions each batch.
    Emits ``["sigma_mu", "sigma_eps"]`` as meta keys.

    The increments are zero-mean normal with SD ``σ_ε`` and are divided by
    ``max(K-2, 1)`` before the cumulative sum, so ``σ_ε`` is the SD of the
    raw increments, not the threshold span.  The expected span
    ``max(δ) − min(δ)`` is proportional to ``σ_ε`` but depends on ``K``
    (for ``σ_ε = 1`` it is about 0.80 for ``K=3`` and about 0.52 for
    ``K=7``).

    Parameters
    ----------
    mu_sigma_range : tuple[float, float]
        ``(lo, hi)`` for the uniform meta-prior on μ scale (item mean
        difficulty in logits).
    eps_sigma_range : tuple[float, float]
        ``(lo, hi)`` for the uniform meta-prior on σ_ε, the SD of the
        zero-mean normal step increments (before the ``1/max(K-2, 1)``
        normalisation).
    n_categories : int
        Number of response categories K (>= 2).
    """

    def __init__(
        self,
        mu_sigma_range: tuple[float, float],
        eps_sigma_range: tuple[float, float],
        n_categories: int,
    ):
        if not isinstance(n_categories, int) or n_categories < 2:
            raise ValueError(
                f"n_categories must be an integer >= 2, got {n_categories!r}."
            )
        for name, rng in [
            ("mu_sigma_range", mu_sigma_range),
            ("eps_sigma_range", eps_sigma_range),
        ]:
            lo, hi = rng
            if lo <= 0 or hi <= lo:
                raise ValueError(f"{name} must satisfy 0 < lo < hi, got {rng!r}.")
        self.mu_sigma_range = mu_sigma_range
        self.eps_sigma_range = eps_sigma_range
        self.n_categories = n_categories
        self.meta_keys: list[str] = ["sigma_mu", "sigma_eps"]

    def _compute_delta(
        self,
        rng,
        batch_size,
        n_items,
        sigma_mu,
        sigma_eps,
    ) -> np.ndarray:
        """Shared construction: incremental prior with given scales."""
        k_minus_1 = self.n_categories - 1
        mu = rng.normal(0, sigma_mu, size=(batch_size, n_items)).astype(np.float32)
        if k_minus_1 == 1:
            return mu[:, :, np.newaxis]
        eps = rng.normal(
            0,
            sigma_eps,
            size=(batch_size, n_items, k_minus_1),
        ).astype(np.float32)
        return _cumsum_sort_center(mu, eps, k_minus_1)

    def __call__(self, shape, rng) -> np.ndarray:
        """Marginal sampling: each element gets independent σ_μ and σ_ε."""
        if len(shape) == 1:
            n = shape[0]
            k_minus_1 = self.n_categories - 1
            n_items = max(1, n // k_minus_1 + 1)
            sigmas_mu = rng.uniform(
                self.mu_sigma_range[0],
                self.mu_sigma_range[1],
                size=n_items,
            )
            sigmas_eps = rng.uniform(
                self.eps_sigma_range[0],
                self.eps_sigma_range[1],
                size=n_items,
            )
            # sigma_mu/sigma_eps vary per synthetic item, so pass them
            # element-shaped (1, n_items) / (1, n_items, 1) -- _compute_delta's
            # rng.normal calls broadcast against these directly, replacing the
            # former per-item Python loop with one vectorized draw.
            all_samples = self._compute_delta(
                rng,
                1,
                n_items,
                sigmas_mu.reshape(1, n_items),
                sigmas_eps.reshape(1, n_items, 1),
            ).ravel()
            return all_samples[:n].astype(np.float32)
        if len(shape) == 3:
            batch_size, n_items, k_minus_1 = shape
            expected_k_minus_1 = self.n_categories - 1
            if k_minus_1 != expected_k_minus_1:
                raise ValueError(
                    f"_VariableIncrementalPrior was created with n_categories="
                    f"{self.n_categories} (K-1={expected_k_minus_1}), but received "
                    f"shape {shape} with last dimension {k_minus_1}."
                )
            sigmas_mu = rng.uniform(
                self.mu_sigma_range[0],
                self.mu_sigma_range[1],
                size=batch_size * n_items,
            )
            sigmas_eps = rng.uniform(
                self.eps_sigma_range[0],
                self.eps_sigma_range[1],
                size=batch_size * n_items,
            )
            # Each (batch, item) gets an independent sigma, so reshape to
            # match (batch_size, n_items) exactly (not just broadcast
            # per-batch) and draw all elements in one call instead of the
            # former nested for-batch/for-item loop.
            return self._compute_delta(
                rng,
                batch_size,
                n_items,
                sigmas_mu.reshape(batch_size, n_items),
                sigmas_eps.reshape(batch_size, n_items, 1),
            )
        raise ValueError(f"Expected 1D or 3D shape, got {shape}.")

    def sample_with_meta(
        self,
        shape,
        rng,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Conditional sampling: one σ_μ and σ_ε per batch element."""
        if len(shape) != 3:
            raise ValueError(
                f"sample_with_meta requires 3D shape (B, I, K-1), got {shape}."
            )
        batch_size, n_items, k_minus_1 = shape
        expected_k_minus_1 = self.n_categories - 1
        if k_minus_1 != expected_k_minus_1:
            raise ValueError(
                f"_VariableIncrementalPrior was created with n_categories="
                f"{self.n_categories} (K-1={expected_k_minus_1}), but received "
                f"shape {shape} with last dimension {k_minus_1}."
            )
        sigmas_mu = rng.uniform(
            self.mu_sigma_range[0],
            self.mu_sigma_range[1],
            size=batch_size,
        ).astype(np.float32)
        sigmas_eps = rng.uniform(
            self.eps_sigma_range[0],
            self.eps_sigma_range[1],
            size=batch_size,
        ).astype(np.float32)

        # sigma_mu/sigma_eps broadcast against (batch_size, n_items) /
        # (batch_size, n_items, k_minus_1) inside _compute_delta's rng.normal
        # calls, so all batch elements are drawn in one vectorized call
        # instead of a Python loop over batch_size.
        results = self._compute_delta(
            rng,
            batch_size,
            n_items,
            sigmas_mu[:, None],
            sigmas_eps[:, None, None],
        )

        meta = {"sigma_mu": sigmas_mu, "sigma_eps": sigmas_eps}
        return results, meta


def make_variable_incremental_prior(
    mu_sigma_range: tuple[float, float],
    eps_sigma_range: tuple[float, float],
    n_categories: int,
) -> _VariableIncrementalPrior:
    """Create a variable-scale incremental prior for PCM/GPCM.

    Like :func:`make_incremental_prior`, but both the mean-difficulty scale
    (σ_μ) and the category-separation scale (σ_ε) are drawn from uniform
    meta-distributions each batch.  This allows the amortized network to
    learn a mapping from the scale to the posterior, enabling inference
    across heterogeneous datasets.

    ``σ_ε`` is the SD of the zero-mean normal increments, applied before the
    ``1/max(K-2, 1)`` normalisation.  The expected threshold span
    ``max(δ) − min(δ)`` scales with ``σ_ε`` but is ``K``-dependent (for
    ``σ_ε = 1`` about 0.80 logits at ``K=3`` and 0.52 at ``K=7``).

    Parameters
    ----------
    mu_sigma_range : tuple[float, float]
        ``(lo, hi)`` for ``σ_μ ~ Uniform(lo, hi)`` (item mean difficulty
        scale in logits).
    eps_sigma_range : tuple[float, float]
        ``(lo, hi)`` for ``σ_ε ~ Uniform(lo, hi)``, where ``σ_ε`` is the SD
        of the zero-mean normal step increments.
    n_categories : int
        Number of response categories K (>= 2).

    Returns
    -------
    _VariableIncrementalPrior
        A prior object with ``sample_with_meta`` protocol.

    Examples
    --------
    >>> vp = make_variable_incremental_prior(
    ...     mu_sigma_range=(0.5, 3.0),
    ...     eps_sigma_range=(0.3, 2.0),
    ...     n_categories=5,
    ... )
    >>> rng = np.random.default_rng(0)
    >>> delta, meta = vp.sample_with_meta((4, 10, 4), rng)
    >>> delta.shape
    (4, 10, 4)
    >>> "sigma_mu" in meta
    True
    >>> "sigma_eps" in meta
    True
    """
    return _VariableIncrementalPrior(
        mu_sigma_range=mu_sigma_range,
        eps_sigma_range=eps_sigma_range,
        n_categories=n_categories,
    )


# ---------------------------------------------------------------------------
# Prior moments & prior predictive sampling
# ---------------------------------------------------------------------------


def prior_moments(
    spec: PriorSpec,
    role: ParameterRole,
    transform: Callable | None = None,
    n_samples: int = 100_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute (mean, std) of a prior distribution.

    Draws a large sample and returns empirical moments.  Optionally applies
    a transform (e.g. ``np.log``) before computing statistics — useful for
    getting the moments of ``log(a)`` when the prior is on ``a``.

    Parameters
    ----------
    spec : Prior | callable
        Prior specification.
    role : {"a", "b", "theta", "delta"}
        Parameter role.
    transform : callable or None
        Optional elementwise transform applied to samples before computing
        moments (e.g. ``np.log`` for log-transformed discriminations).
    n_samples : int
        Number of samples for the empirical estimate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    (float, float)
        ``(mean, std)`` of the (optionally transformed) prior.

    Examples
    --------
    >>> prior_moments(Prior("normal", loc=0, scale=2), "b")
    (0.0..., 2.0...)
    >>> prior_moments(Prior("lognormal", loc=0, scale=0.5), "a", transform=np.log)
    (0.0..., 0.5...)
    """
    samples = sample_prior_predictive(spec, role, n_samples=n_samples, seed=seed)
    if transform is not None:
        samples = transform(samples)
    return float(samples.mean()), float(samples.std())


def sample_prior_predictive(
    spec: PriorSpec,
    role: ParameterRole,
    n_samples: int = 10_000,
    seed: int | None = None,
) -> np.ndarray:
    """Draw samples from a prior for inspection (prior predictive check).

    Parameters
    ----------
    spec : Prior | callable
        Prior specification (same types accepted by simulators).
    role : {"a", "b", "theta", "delta"}
        Parameter role.
    n_samples : int
        Number of samples to draw.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Shape ``(n_samples,)``.

    Examples
    --------
    >>> samples = sample_prior_predictive(
    ...     Prior("lognormal"), "a", n_samples=500, seed=0
    ... )
    >>> samples.shape
    (500,)
    """
    rng = np.random.default_rng(seed)
    sampler = resolve_prior(spec, role)
    return sampler((n_samples,), rng)
