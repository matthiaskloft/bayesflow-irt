"""Person scoring module for IRT models.

Provides classical IRT person scoring (EAP, MAP) with uncertainty propagation
over item parameter posterior samples. Supports 1PL (Rasch), 2PL, PCM, and
GPCM models with configurable theta priors.

The scoring approach computes a conditional theta posterior by numerical
quadrature for each item posterior draw, then averages the normalized
posterior masses. Summaries are taken from that marginal theta posterior, so
intervals preserve both response-level theta uncertainty and item-parameter
uncertainty.

By default the quadrature is two-pass: a coarse grid on the prior bounds
gives a per-person EAP and SD, and the posterior is then re-evaluated on a
per-person grid centred there (``adaptive_grid=True``). Quantile-type
summaries (median, ETI, HDI, plausible values) treat each grid mass as uniform
on its cell, which removes the half-cell low bias of reading quantiles off the
CDF at the grid points. A residual O(h^2) discretization error remains (about
0.025 SD on interval bounds with the default grid, measured for 5-3000
items).

References
----------
Bock, R. D., & Mislevy, R. J. (1982). Adaptive EAP estimation of ability
in a microcomputer environment. *Applied Psychological Measurement*, 6(4),
431-444. https://doi.org/10.1177/014662168200600405

Devroye, L. (1986). *Non-Uniform Random Variate Generation*. Springer.
https://doi.org/10.1007/978-1-4613-8643-8

Naylor, J. C., & Smith, A. F. M. (1982). Applications of a method for the
efficient computation of posterior distributions. *Journal of the Royal
Statistical Society: Series C (Applied Statistics)*, 31(3), 214-225.
https://doi.org/10.2307/2347995

Schilling, S., & Bock, R. D. (2005). High-dimensional maximum marginal
likelihood item factor analysis by adaptive quadrature. *Psychometrika*,
70(3), 533-555. https://doi.org/10.1007/s11336-003-1141-x

Thissen, D., & Wainer, H. (2001). *Test Scoring*. Lawrence Erlbaum.

Lord, F. M. (1980). *Applications of Item Response Theory to Practical
Testing Problems*. Lawrence Erlbaum.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Collection, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import logsumexp

from .adapters import OneHotEncode

if TYPE_CHECKING:
    from .approximators import EquivariantIRTApproximator

# Type alias for log-likelihood functions
LogLikelihoodFn = Callable[[np.ndarray, np.ndarray, ...], np.ndarray]

_POSTERIOR_METADATA_KEYS = {
    "n_items",
    "n_persons",
    "item_ids",
    "d",
    "d_delta",
    "sigma_a",
    "sigma_b",
    "sigma_delta",
    "sigma_mu",
    "sigma_eps",
}

_BINARY_POSTERIOR_CHUNK_SIZE = 128
_GENERAL_POSTERIOR_CHUNK_SIZE = 16
# Element budget of the (S, P_block, Q, I) work array for per-person grids.
_PERSON_GRID_BLOCK_ELEMENTS = 2**22
# Smaller (cache-sized) budget for the polytomous (P_block, Q, I, K) array:
# measured fastest at 2**16 in a block-size benchmark.
_POLYTOMOUS_PERSON_GRID_BLOCK_ELEMENTS = 2**16
# Largest |exponent| for the Horner form of the PCM/GPCM normalizer; float64
# exp overflows near 709, and the sum of K terms adds at most log K.
_HORNER_MAX_EXPONENT = 600.0
# Adaptive second-pass grid: half-width in pass-1 SDs and number of points
# (both chosen by a sweep of span and size), and the floor on the
# pass-1 SD in units of the coarse-grid spacing (a coarse pass can badly
# underestimate a posterior much narrower than its spacing).
_ADAPTIVE_GRID_HALF_WIDTH_SD = 4.0
_ADAPTIVE_GRID_SIZE = 21
_ADAPTIVE_GRID_MIN_SD_SPACINGS = 0.5
# Refinement of the second pass for persons whose window is too wide for the
# posterior (long tests): re-run while grid spacing > 0.5 pass-2 SD (the
# default window gives about 0.4), at most this many times; the new half-width
# is at least this many spacings so a posterior narrower than one spacing
# still shrinks the window by a large factor each round.
_ADAPTIVE_GRID_MAX_SPACING_SD = 0.5
# Containment guard: if any single item-posterior draw puts more than this
# conditional posterior mass (on the coarse grid) outside a person's pass-2
# window, the window is widened to the union of all draws' own windows.
# Pass 2 renormalizes every draw inside the window, so mass outside it would
# otherwise be moved onto the window edge at full mixture weight.
_ADAPTIVE_GRID_MAX_OUTSIDE_MASS = 1e-3
_ADAPTIVE_GRID_MAX_REFINEMENTS = 6
_ADAPTIVE_GRID_MIN_HALF_WIDTH_SPACINGS = 1.5


# Element budget of the (S, P_block, Q, I) work array on the torch path.
_TORCH_BLOCK_ELEMENTS = {"cpu": 2**22, "cuda": 2**26}
_TORCH_DTYPE_NAME = "float64"


def _resolve_device(device: str | None) -> str | None:
    """Map the ``device`` argument of ``score()`` to None, 'cpu' or 'cuda'."""
    if device is None:
        return None
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError(
            f"device must be None, 'auto', 'cpu' or 'cuda', got {device!r}."
        )
    try:
        import torch
    except ImportError as exc:
        if device == "auto":
            return None
        raise ImportError(
            "score(device=...) requires PyTorch, which is an optional "
            "dependency. Install torch or use device=None."
        ) from exc
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else None
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but CUDA is not available.")
    return device


def _to_float64(tensor) -> np.ndarray:
    """A torch tensor as a float64 NumPy array (no copy if already float64)."""
    return tensor.cpu().numpy().astype(np.float64, copy=False)


def _mixture_weights_torch(
    theta_grid: np.ndarray,
    log_prior_weights: np.ndarray,
    responses: np.ndarray,
    params: Mapping[str, np.ndarray],
    model_type: str,
    *,
    device: str,
    draw_chunk_size: int,
    window: tuple | None = None,
    on_post: Callable | None = None,
    dtype_name: str = _TORCH_DTYPE_NAME,
):
    """Draw-averaged theta posterior masses on a torch device.

    Same computation as the NumPy path, for the built-in 1PL, 2PL, PCM and
    GPCM likelihoods (``params``: the item-posterior draws ``b``/``delta``
    and, for 2PL/GPCM, ``a``): for each item-posterior draw the
    conditional posterior on the grid is normalized, then averaged over
    draws. ``theta_grid`` is ``(Q,)`` (shared) or ``(P, Q)`` (per person).
    Draws are chunked and persons are blocked (on both the shared- and the
    per-person-grid path) under an element budget to bound device memory.
    The PCM/GPCM likelihood uses the algebra of
    :func:`_log_likelihood_polytomous_chunk`.
    Returns ``(P, Q)`` masses as float64 NumPy.

    ``on_post(draw_start, draw_stop, post, person_slice)``, if given, is called
    for every (draw chunk, person block) with the normalized per-draw masses
    ``post`` as a float64 NumPy array ``(draw_stop - draw_start, n_block, Q)``;
    the block is not kept afterwards. ``dtype_name`` is the torch precision
    of the computation; outputs are float64 either way.
    """
    import torch

    dtype = getattr(torch, dtype_name)
    observed = ~np.isnan(responses)
    r = torch.as_tensor(np.where(observed, responses, 0.0), dtype=dtype, device=device)
    o = torch.as_tensor(observed.astype(np.float64), dtype=dtype, device=device)
    grid = torch.as_tensor(theta_grid, dtype=dtype, device=device)
    log_w = torch.as_tensor(log_prior_weights, dtype=dtype, device=device)
    n_persons = responses.shape[0]
    polytomous = model_type in ("PCM", "GPCM")
    first = params["delta"] if polytomous else params["b"]
    n_samples, n_items = first.shape[:2]
    a = params.get("a") if model_type in ("2PL", "GPCM") else None
    n_grid = grid.shape[-1]
    if polytomous:
        n_categories = first.shape[2] + 1
        idx = torch.as_tensor(
            np.where(observed, responses, 0).astype(np.int64), device=device
        )
        # x_pi and the item index, for gathering D_s,i,x_pi.
        item_idx = torch.arange(n_items, device=device)[None, :]
    budget = _TORCH_BLOCK_ELEMENTS[device]
    total = torch.zeros((n_persons, n_grid), dtype=dtype, device=device)
    if window is not None:
        w_lo, w_hi, w_span, w_floor = window
        w_lo = torch.as_tensor(w_lo, dtype=dtype, device=device)
        w_hi = torch.as_tensor(w_hi, dtype=dtype, device=device)
        inf = float("inf")
        outside_max = torch.zeros(n_persons, dtype=dtype, device=device)
        union_lo = torch.full((n_persons,), inf, dtype=dtype, device=device)
        union_hi = torch.full((n_persons,), -inf, dtype=dtype, device=device)

    for start in range(0, n_samples, draw_chunk_size):
        stop = min(start + draw_chunk_size, n_samples)
        n_s = stop - start
        a_c = (
            torch.as_tensor(a[start:stop], dtype=dtype, device=device)
            if a is not None
            else torch.ones((n_s, n_items), dtype=dtype, device=device)
        )
        if polytomous:
            d_c = torch.as_tensor(
                params["delta"][start:stop], dtype=dtype, device=device
            )
            cum = torch.cat([torch.zeros_like(d_c[..., :1]), d_c.cumsum(-1)], -1)
            k_vec = torch.arange(n_categories, dtype=dtype, device=device)
            # sum_i a_si x_pi and sum_i a_si D_s,i,x_pi over observed items.
            slope = (r @ a_c.T).T  # (S, P); r holds x_pi (0 if missing)

            def block_offset(p0, p1):
                # Gathered per person block: the full (S, P, I) gather
                # would grow with the cohort, outside the budget.
                gathered = cum[:, item_idx, idx[p0:p1]]
                return torch.einsum("spi,si,pi->sp", gathered, a_c, o[p0:p1])

        else:
            b_c = torch.as_tensor(params["b"][start:stop], dtype=dtype, device=device)
        if grid.ndim == 1:
            if polytomous:
                # Person-free item normalizers (S, Q, I), built in item
                # blocks so the (S, Q, I, K) intermediate stays in budget.
                i_block = max(1, budget // (n_s * n_grid * n_categories))
                norm = torch.cat(
                    [
                        torch.logsumexp(
                            a_c[:, None, i0 : i0 + i_block, None]
                            * (
                                grid[:, None, None] * k_vec
                                - cum[:, None, i0 : i0 + i_block]
                            ),
                            -1,
                        )
                        for i0 in range(0, n_items, i_block)
                    ],
                    -1,
                )
            else:
                logits = a_c[:, None, :] * (grid[None, :, None] - b_c[:, None, :])
                norm = torch.logaddexp(torch.zeros_like(logits), logits)
            # The block also bounds the (S, Pb, I) offset gather.
            block = max(1, budget // (n_s * max(n_grid, n_items)))
            for p0 in range(0, n_persons, block):
                p1 = min(p0 + block, n_persons)
                if polytomous:
                    ll = grid * slope[:, p0:p1, None] - block_offset(p0, p1)[..., None]
                else:
                    ll = torch.einsum("pi,sgi->spg", r[p0:p1], logits)
                ll = ll - torch.einsum("pi,sgi->spg", o[p0:p1], norm)
                log_mass = ll + log_w
                post = torch.exp(log_mass - torch.logsumexp(log_mass, -1, keepdim=True))
                total[p0:p1] += post.sum(0)
                if on_post is not None:
                    on_post(start, stop, _to_float64(post), slice(p0, p1))
                if window is not None:
                    out_mask = (grid[None, :] < w_lo[p0:p1, None]) | (
                        grid[None, :] > w_hi[p0:p1, None]
                    )
                    outside_max[p0:p1] = torch.maximum(
                        outside_max[p0:p1], (post * out_mask).sum(-1).amax(0)
                    )
                    mean_s = post @ grid
                    sd_s = torch.sqrt(
                        torch.clamp((post @ grid**2) - mean_s**2, min=0.0)
                    )
                    half_s = w_span * torch.clamp(sd_s, min=w_floor)
                    union_lo[p0:p1] = torch.minimum(
                        union_lo[p0:p1], (mean_s - half_s).amin(0)
                    )
                    union_hi[p0:p1] = torch.maximum(
                        union_hi[p0:p1], (mean_s + half_s).amax(0)
                    )
            continue
        if not polytomous:
            slope = (r @ a_c.T).T  # (S, P)
            offset = (r @ (a_c * b_c).T).T

            def block_offset(p0, p1):
                return offset[:, p0:p1]

        per_person = n_grid * n_items * (n_categories if polytomous else 1)
        block = max(1, budget // (n_s * per_person))
        for p0 in range(0, n_persons, block):
            p1 = min(p0 + block, n_persons)
            th = grid[p0:p1]
            if polytomous:
                norm = torch.logsumexp(
                    a_c[:, None, None, :, None]
                    * (th[:, :, None, None] * k_vec - cum[:, None, None]),
                    -1,
                )
            else:
                logits = a_c[:, None, None, :] * (
                    th[None, :, :, None] - b_c[:, None, None, :]
                )
                norm = torch.logaddexp(torch.zeros_like(logits), logits)
            ll = (
                th[None] * slope[:, p0:p1, None]
                - block_offset(p0, p1)[..., None]
                - torch.einsum("pi,spgi->spg", o[p0:p1], norm)
            )
            log_mass = ll + log_w[p0:p1]
            post = torch.exp(log_mass - torch.logsumexp(log_mass, -1, keepdim=True))
            total[p0:p1] += post.sum(0)
            if on_post is not None:
                on_post(start, stop, _to_float64(post), slice(p0, p1))
    weights = (total / n_samples).cpu().numpy().astype(np.float64)
    if window is None:
        return weights
    return weights, (
        outside_max.cpu().numpy().astype(np.float64),
        union_lo.cpu().numpy().astype(np.float64),
        union_hi.cpu().numpy().astype(np.float64),
    )


def _draw_window_stats(
    post: np.ndarray,
    grid: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    span: float,
    sd_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-person containment statistics over a chunk of draws.

    ``post`` holds per-draw conditional posterior masses ``(S, P, Q)`` on the
    shared ``grid`` ``(Q,)``. Returns, per person ``(P,)``: the largest mass
    any draw places outside ``[lower, upper]``, and the lowest/highest edge of
    the draws' own windows ``mean_s -/+ span * max(sd_s, sd_floor)``.
    """
    outside_mask = (grid[np.newaxis, :] < lower[:, np.newaxis]) | (
        grid[np.newaxis, :] > upper[:, np.newaxis]
    )
    outside = np.max(np.sum(post * outside_mask, axis=-1), axis=0)
    mean_s = post @ grid
    sd_s = np.sqrt(np.maximum(post @ grid**2 - mean_s**2, 0.0))
    half_s = span * np.maximum(sd_s, sd_floor)
    return outside, np.min(mean_s - half_s, axis=0), np.max(mean_s + half_s, axis=0)


def _combine_window_stats(stats_list):
    """Combine :func:`_draw_window_stats` results of several chunks."""
    outside = np.max([s[0] for s in stats_list], axis=0)
    lo = np.min([s[1] for s in stats_list], axis=0)
    hi = np.max([s[2] for s in stats_list], axis=0)
    return outside, lo, hi


def _grid_mean_sd(
    theta_grid: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and SD of grid masses; ``theta_grid`` is ``(Q,)`` or ``(P, Q)``."""
    w = weights / weights.sum(axis=-1, keepdims=True)
    mean = np.sum(w * theta_grid, axis=-1)
    var = np.sum(w * (theta_grid - mean[:, np.newaxis]) ** 2, axis=-1)
    return mean, np.sqrt(np.maximum(var, 0.0))


def _validate_n_cores(n_cores: int | None) -> int:
    """Validate requested thread count for chunk-parallel scoring."""
    if n_cores is None:
        return 1
    return _validate_positive_int("n_cores", n_cores)


_RETURN_POSTERIOR_VALUES = frozenset({"grid", "draws"})
_INTERVAL_VALUES = frozenset({"eti", "hdi"})


def _validate_seed(seed: int | None, requested: frozenset[str]) -> int | None:
    """Check ``seed`` (a non-negative int) and that ``"draws"`` is requested."""
    if seed is None:
        return None
    if "draws" not in requested:
        raise ValueError(
            "seed is only used when return_posterior includes 'draws'; "
            "got a seed without it."
        )
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError(f"seed must be None or a non-negative int, got {seed!r}.")
    return int(seed)


def _invert_cell_uniform(
    post: np.ndarray,
    edges: np.ndarray,
    u: np.ndarray,
) -> np.ndarray:
    """Inverse-CDF draws from cell-uniform grid posteriors.

    Parameters
    ----------
    post : np.ndarray
        Normalized masses ``(S, P, Q)``, one distribution per (draw, person).
    edges : np.ndarray
        Cell edges ``(P, Q + 1)`` (see :func:`_cell_edges`).
    u : np.ndarray
        Uniform variates in ``[0, 1)``, shape ``(S, P)``.

    Returns
    -------
    np.ndarray
        Draws ``(S, P)``; each mass is uniform on its cell, i.e. the CDF is
        piecewise linear through the edges, as in :func:`_edge_cdf`.
    """
    cdf = np.cumsum(post, axis=-1)
    cdf = cdf / cdf[..., -1:]
    n_grid = post.shape[-1]
    k = np.minimum(np.sum(cdf < u[..., np.newaxis], axis=-1), n_grid - 1)
    cdf_pad = np.concatenate([np.zeros_like(cdf[..., :1]), cdf], axis=-1)
    idx = k[..., np.newaxis]
    lo = np.take_along_axis(cdf_pad, idx, axis=-1)[..., 0]
    hi = np.take_along_axis(cdf_pad, idx + 1, axis=-1)[..., 0]
    e = np.broadcast_to(edges, (post.shape[0],) + edges.shape)
    e_lo = np.take_along_axis(e, idx, axis=-1)[..., 0]
    e_hi = np.take_along_axis(e, idx + 1, axis=-1)[..., 0]
    width = hi - lo
    frac = np.where(width > 0, (u - lo) / np.where(width > 0, width, 1.0), 0.0)
    return e_lo + np.clip(frac, 0.0, 1.0) * (e_hi - e_lo)


def _validate_return_posterior(
    return_posterior: str | Collection[str] | None,
) -> frozenset[str]:
    """Normalize ``return_posterior`` to a set of allowed string values."""
    if return_posterior is None:
        return frozenset()
    if isinstance(return_posterior, str):
        requested = {return_posterior}
    else:
        try:
            requested = set(return_posterior)
        except TypeError:
            raise ValueError(
                "return_posterior must be None, a string, or a collection of "
                f"strings, got {return_posterior!r}."
            ) from None
    invalid = [v for v in requested if v not in _RETURN_POSTERIOR_VALUES]
    if invalid:
        allowed = sorted(_RETURN_POSTERIOR_VALUES)
        raise ValueError(
            f"Invalid return_posterior value(s) {sorted(map(repr, invalid))}; "
            f"allowed values: {allowed}."
        )
    return frozenset(requested)


def _validate_intervals(intervals: str | Collection[str] | None) -> frozenset[str]:
    """Normalize ``intervals`` to a subset of ``{"eti", "hdi"}``."""
    if intervals is None:
        return frozenset()
    if isinstance(intervals, str):
        requested = {intervals}
    else:
        try:
            requested = set(intervals)
        except TypeError:
            raise ValueError(
                "intervals must be None, a string, or a collection of strings, "
                f"got {intervals!r}."
            ) from None
    invalid = [v for v in requested if v not in _INTERVAL_VALUES]
    if invalid:
        raise ValueError(
            f"Invalid intervals value(s) {sorted(map(repr, invalid))}; "
            f"allowed values: {sorted(_INTERVAL_VALUES)}."
        )
    return frozenset(requested)


@dataclass
class PersonScore:
    """Structured output from person scoring.

    All array fields have shape ``(P',)`` where ``P'`` is the number of persons
    scored, except interval arrays with multiple requested levels, which have
    shape ``(L, P')``, and ``plausible_values``, which is ``(P', S)`` or
    ``None``.

    Attributes
    ----------
    mean : np.ndarray
        Posterior mean under the marginal theta posterior.
    map : np.ndarray
        Maximum a posteriori estimate (grid mode of aggregated posterior).
    median : np.ndarray
        Median of the marginal theta posterior.
    sd : np.ndarray
        Posterior standard deviation of theta.
    eti_lower : np.ndarray or None
        Lower bound of equal-tailed interval.
        Shape is ``(P',)`` for one requested level and ``(L, P')`` for many;
        ``None`` when skipped via ``score(intervals=...)``.
    eti_upper : np.ndarray or None
        Upper bound of equal-tailed interval.
        Shape is ``(P',)`` for one requested level and ``(L, P')`` for many;
        ``None`` when skipped via ``score(intervals=...)``.
    hdi_lower : np.ndarray or None
        Lower bound of highest density interval.
        Shape is ``(P',)`` for one requested level and ``(L, P')`` for many;
        ``None`` when skipped via ``score(intervals=...)``.
    hdi_upper : np.ndarray or None
        Upper bound of highest density interval.
        Shape is ``(P',)`` for one requested level and ``(L, P')`` for many;
        ``None`` when skipped via ``score(intervals=...)``.
    plausible_values : np.ndarray or None
        Optional deterministic inverse-CDF theta plausible values from the
        marginal theta posterior.
    n_items : int
        Number of items used for scoring (for validation).
    eti_levels : np.ndarray
        Equal-tailed interval levels used for ``eti_lower``/``eti_upper``.
    hdi_levels : np.ndarray
        Highest-density interval levels used for ``hdi_lower``/``hdi_upper``.
    theta_grid : np.ndarray or None
        Final quadrature grid of each scored person, shape ``(P', Q)``
        (float64). Only set when ``score(return_posterior="grid")`` was
        requested, else ``None``. With ``adaptive_grid=False`` every row is
        the same shared grid. ``P'`` is the number of rows of the scored
        response matrix; no persons are dropped, so rows align with the
        other per-person fields.
    theta_posterior_mass : np.ndarray or None
        Draw-averaged normalized posterior masses on ``theta_grid``, shape
        ``(P', Q)`` (float64), each row summing to 1. Only set together with
        ``theta_grid``. ``mean``, ``sd``, ``map``, ``median``, the intervals
        and the plausible values are computed from exactly these masses;
        quantile-type summaries treat each mass as uniform over its grid
        cell.
    theta_window_uncontained : np.ndarray or None
        Boolean ``(P',)``. ``True`` for persons whose adaptive window failed
        the containment guard (some item draw put more than
        ``_ADAPTIVE_GRID_MAX_OUTSIDE_MASS`` of its mass outside the
        coarse window); their window was widened and not refined. All
        ``False`` with ``adaptive_grid=False``. Only set when ``"grid"`` or
        ``"draws"`` was requested, else ``None``.
    theta_draws : np.ndarray or None
        Joint draws of theta, shape ``(S, P')`` (float64), where ``S`` is the
        number of item posterior draws used. Row ``s`` holds, for every
        person, one draw from ``p(theta | y_p, items_s)`` on that person's
        final grid, so ``(item_draws[s], theta_draws[s])`` is a draw from
        ``p(items | y) p(theta | items, y)``. The layout ``(S, P')`` is the
        transpose of ``plausible_values`` ``(P', S)``. Only set when
        ``score(return_posterior="draws")`` was requested. Each draw's
        conditional is truncated to the person's grid window: the lost mass
        is at most 1e-3 per draw for contained persons, but up to a
        ``span``-SD tail for persons flagged in ``theta_window_uncontained``,
        and the grid may be coarse relative to a single draw's SD. Memory is
        ``S * P'`` float64. To reduce it, draw fewer item samples:
        ``n_item_samples`` applies only to a scorer without cached posteriors;
        cached ``item_posteriors`` must be subset with the same draw indices
        for every parameter before building a new scorer.
    item_draws : dict or None
        Shallow copy of the item posterior draws the scorer used (the same
        arrays as ``PersonScorer._item_posteriors``), aligned with the rows
        of ``theta_draws``. Only set together with ``theta_draws``.
    theta_draw_mean : np.ndarray or None
        Per-draw conditional posterior mean on the grid, ``(S, P')``
        (grid-point moments, same convention as ``mean``). Only set when
        ``draw_moments=True``.
    theta_draw_sd : np.ndarray or None
        Per-draw conditional posterior SD on the grid, ``(S, P')``. Only set
        when ``draw_moments=True``.
    """

    mean: np.ndarray
    map: np.ndarray
    median: np.ndarray
    sd: np.ndarray
    eti_lower: np.ndarray | None
    eti_upper: np.ndarray | None
    hdi_lower: np.ndarray | None
    hdi_upper: np.ndarray | None
    plausible_values: np.ndarray | None
    n_items: int
    eti_levels: np.ndarray
    hdi_levels: np.ndarray
    theta_grid: np.ndarray | None = None
    theta_posterior_mass: np.ndarray | None = None
    theta_window_uncontained: np.ndarray | None = None
    theta_draws: np.ndarray | None = None
    item_draws: dict[str, np.ndarray] | None = None
    theta_draw_mean: np.ndarray | None = None
    theta_draw_sd: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Log-likelihood functions (matching simulator formulas)
# ---------------------------------------------------------------------------


def _log_likelihood_1pl(
    theta: np.ndarray,
    responses: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """1PL log-likelihood vectorized over quadrature points.

    Parameters
    ----------
    theta : np.ndarray
        Quadrature grid points ``(Q,)``.
    responses : np.ndarray
        Person's response vector ``(I,)`` with values in ``{0, 1}``.
    b : np.ndarray
        Item difficulties ``(I,)``.

    Returns
    -------
    np.ndarray
        Log-likelihood at each quadrature point ``(Q,)``.

    Notes
    -----
    The 1PL likelihood uses the logistic function:

    .. math::

        \\log p(x | \\theta, b) = \\sum_i x_i(\\theta - b_i)
        - \\sum_i \\log(1 + \\exp(\\theta - b_i))

    This matches the 1PL simulator formula in ``simulators.py``.
    """
    observed = ~np.isnan(responses)
    responses_filled = np.where(observed, responses, 0.0)

    # theta: (Q,), b: (I,) -> logits: (Q, I)
    logits = theta[:, np.newaxis] - b[np.newaxis, :]
    # Log-sum-exp for numerical stability: log(1 + exp(x)) = logsumexp([0, x])
    log_norm = logsumexp(np.stack([np.zeros_like(logits), logits], axis=-1), axis=-1)
    # Weighted sum: x * logit - log_norm
    log_likelihood = np.sum(
        responses_filled[np.newaxis, :] * logits
        - observed[np.newaxis, :].astype(np.float64) * log_norm,
        axis=-1,
    )
    return log_likelihood


def _log_likelihood_2pl(
    theta: np.ndarray,
    responses: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """2PL log-likelihood vectorized over quadrature points.

    Parameters
    ----------
    theta : np.ndarray
        Quadrature grid points ``(Q,)``.
    responses : np.ndarray
        Person's response vector ``(I,)`` with values in ``{0, 1}``.
    a : np.ndarray
        Item discriminations ``(I,)``.
    b : np.ndarray
        Item difficulties ``(I,)``.

    Returns
    -------
    np.ndarray
        Log-likelihood at each quadrature point ``(Q,)``.

    Notes
    -----
    The 2PL likelihood:

    .. math::

        \\log p(x | \\theta, a, b) = \\sum_i x_i a_i(\\theta - b_i)
        - \\sum_i \\log(1 + \\exp(a_i(\\theta - b_i)))

    This matches the 2PL simulator formula in ``simulators.py``.
    """
    observed = ~np.isnan(responses)
    responses_filled = np.where(observed, responses, 0.0)

    # theta: (Q,), a: (I,), b: (I,) -> logits: (Q, I)
    logits = a[np.newaxis, :] * (theta[:, np.newaxis] - b[np.newaxis, :])
    log_norm = logsumexp(np.stack([np.zeros_like(logits), logits], axis=-1), axis=-1)
    log_likelihood = np.sum(
        responses_filled[np.newaxis, :] * logits
        - observed[np.newaxis, :].astype(np.float64) * log_norm,
        axis=-1,
    )
    return log_likelihood


def _log_likelihood_pcm(
    theta: np.ndarray,
    responses: np.ndarray,
    delta: np.ndarray,
) -> np.ndarray:
    """PCM log-likelihood vectorized over quadrature points.

    Parameters
    ----------
    theta : np.ndarray
        Quadrature grid points ``(Q,)``.
    responses : np.ndarray
        Person's response vector ``(I,)`` with values in ``{0, ..., K-1}``.
    delta : np.ndarray
        Step difficulties ``(I, K-1)`` where ``K`` is the number of categories.

    Returns
    -------
    np.ndarray
        Log-likelihood at each quadrature point ``(Q,)``.

    Notes
    -----
    PCM uses the cumulative logit structure. For item i with response k:

    .. math::

        \\log p(x_i = k | \\theta, \\delta_i) = \\sum_{j=1}^k (\\theta - \\delta_{ij})
        - \\log \\sum_{c=0}^{K-1} \\exp(\\sum_{j=1}^c (\\theta - \\delta_{ij}))

    where the sum ``\\sum_{j=1}^0(...)`` is 0 (category 0 has no step).
    This matches the PCM simulator formula in ``simulators.py``.
    """
    n_items, n_steps_minus_1 = delta.shape  # noqa: N806 (standard IRT notation)
    # Step logits: (Q, I, K-1) = theta - delta
    step_logits = theta[:, np.newaxis, np.newaxis] - delta[np.newaxis, :, :]

    # Cumulative sum with zero-prepend for category 0
    zeros = np.zeros(
        (step_logits.shape[0], step_logits.shape[1], 1),
        dtype=step_logits.dtype,
    )
    cum_logits = np.concatenate(
        [zeros, np.cumsum(step_logits, axis=-1)],
        axis=-1,
    )

    # Log-sum-exp for normalization: (Q, I, K)
    log_norm = logsumexp(cum_logits, axis=-1, keepdims=True)

    # Gather the log-likelihood for each item's response category
    observed = ~np.isnan(responses)
    response_indices = np.where(observed, responses, 0).astype(int)
    log_likelihood_per_item = np.zeros(
        (theta.shape[0], n_items),
        dtype=theta.dtype,
    )

    for i in range(n_items):
        if not observed[i]:
            continue
        k = response_indices[i]
        log_likelihood_per_item[:, i] = cum_logits[:, i, k] - log_norm[:, i, 0]

    return np.sum(log_likelihood_per_item, axis=-1)


def _log_likelihood_gpcm(
    theta: np.ndarray,
    responses: np.ndarray,
    a: np.ndarray,
    delta: np.ndarray,
) -> np.ndarray:
    """GPCM log-likelihood vectorized over quadrature points.

    Parameters
    ----------
    theta : np.ndarray
        Quadrature grid points ``(Q,)``.
    responses : np.ndarray
        Person's response vector ``(I,)`` with values in ``{0, ..., K-1}``.
    a : np.ndarray
        Item discriminations ``(I,)``.
    delta : np.ndarray
        Step difficulties ``(I, K-1)`` where ``K`` is the number of categories.

    Returns
    -------
    np.ndarray
        Log-likelihood at each quadrature point ``(Q,)``.

    Notes
    -----
    GPCM extends PCM with discrimination scaling:

    .. math::

        \\log p(x_i = k | \\theta, a, \\delta) = \\sum_{j=1}^k a_i(
        \\theta - \\delta_{ij}) - \\log \\sum_{c=0}^{K-1} \\exp(
        \\sum_{j=1}^c a_i(\\theta - \\delta_{ij}))

    This matches the GPCM simulator formula in ``simulators.py``.
    """
    n_items, _ = delta.shape  # noqa: N806 (standard IRT notation)

    # Step logits: (Q, I, K-1) = a * (theta - delta)
    step_logits = a[np.newaxis, :, np.newaxis] * (
        theta[:, np.newaxis, np.newaxis] - delta[np.newaxis, :, :]
    )

    # Cumulative sum with zero-prepend for category 0
    zeros = np.zeros(
        (step_logits.shape[0], step_logits.shape[1], 1),
        dtype=step_logits.dtype,
    )
    cum_logits = np.concatenate(
        [zeros, np.cumsum(step_logits, axis=-1)],
        axis=-1,
    )

    # Log-sum-exp for normalization: (Q, I, K)
    log_norm = logsumexp(cum_logits, axis=-1, keepdims=True)

    observed = ~np.isnan(responses)
    response_indices = np.where(observed, responses, 0).astype(int)
    log_likelihood_per_item = np.zeros(
        (theta.shape[0], n_items),
        dtype=theta.dtype,
    )

    for i in range(n_items):
        if not observed[i]:
            continue
        k = response_indices[i]
        log_likelihood_per_item[:, i] = cum_logits[:, i, k] - log_norm[:, i, 0]

    return np.sum(log_likelihood_per_item, axis=-1)


def _log_likelihood_all_persons(
    theta: np.ndarray,
    responses: np.ndarray,
    item_params: dict[str, np.ndarray],
    model_type: str,
    log_likelihood_fn: LogLikelihoodFn,
    custom_log_likelihood: bool = False,
) -> np.ndarray:
    """Compute log-likelihoods for all persons at all grid points.

    Returns an array with shape ``(P, Q)``. Built-in likelihoods use
    vectorized NumPy operations; custom likelihoods fall back to the public
    single-person callback contract.

    ``theta`` is ``(Q,)`` for a grid shared by all persons or ``(P, Q)`` for
    one grid per person (adaptive scoring grids).
    """
    if custom_log_likelihood:
        grids = (
            np.broadcast_to(theta, (responses.shape[0], theta.shape[-1]))
            if theta.ndim == 2
            else [theta] * responses.shape[0]
        )
        return np.stack(
            [
                log_likelihood_fn(person_theta, person_responses, **item_params)
                for person_theta, person_responses in zip(grids, responses)
            ],
            axis=0,
        )

    if theta.ndim == 2:
        if model_type not in ("PCM", "GPCM"):
            raise ValueError(
                "Per-person theta grids are only supported for PCM/GPCM here; "
                "binary models use the chunked path."
            )
        # With a_i (k * theta - D_ik), D_ik = sum_{j<=k} delta_ij, the log
        # probability of category k is that linear term minus the item
        # normalizer. The observed-category terms are linear in theta (matrix
        # products); only the normalizer needs the (P_block, Q, I, K) array,
        # which is evaluated in person blocks to bound memory.
        delta = item_params["delta"]
        slope_a = item_params["a"] if model_type == "GPCM" else np.ones(delta.shape[0])
        cum_delta = np.concatenate(
            [np.zeros((delta.shape[0], 1)), np.cumsum(delta, axis=-1)], axis=-1
        )  # (I, K)
        n_categories = cum_delta.shape[-1]
        observed = ~np.isnan(responses)
        obs_float = observed.astype(np.float64)
        idx = np.where(observed, responses, 0).astype(int)
        slope = (obs_float * idx) @ slope_a  # sum_i a_i x_pi
        offset = np.sum(
            obs_float
            * slope_a[np.newaxis, :]
            * cum_delta[np.arange(delta.shape[0])[np.newaxis, :], idx],
            axis=1,
        )
        out = theta * slope[:, np.newaxis] - offset[:, np.newaxis]
        n_persons, n_grid = theta.shape
        n_items = delta.shape[0]
        block = max(
            1,
            _POLYTOMOUS_PERSON_GRID_BLOCK_ELEMENTS // (n_grid * n_items * n_categories),
        )
        for start in range(0, n_persons, block):
            stop = min(start + block, n_persons)
            # Category axis first: reductions over a short last axis are slow
            # in NumPy, whereas elementwise ops on (P_block, Q, I) are fast.
            th = theta[start:stop, :, np.newaxis]
            c = np.empty((n_categories, stop - start, n_grid, n_items))
            for k in range(n_categories):
                c[k] = slope_a * (k * th - cum_delta[:, k])
            m = c.max(axis=0)
            c -= m
            np.exp(c, out=c)
            log_z = m + np.log(c.sum(axis=0))
            out[start:stop] -= np.einsum("pi,pgi->pg", obs_float[start:stop], log_z)
        return out

    if model_type == "1PL":
        b = item_params["b"]
        logits = theta[:, np.newaxis] - b[np.newaxis, :]
        log_norm = np.logaddexp(0.0, logits)
        observed = ~np.isnan(responses)
        responses_filled = np.where(observed, responses, 0.0)
        return responses_filled @ logits.T - observed.astype(np.float64) @ log_norm.T

    if model_type == "2PL":
        a = item_params["a"]
        b = item_params["b"]
        logits = a[np.newaxis, :] * (theta[:, np.newaxis] - b[np.newaxis, :])
        log_norm = np.logaddexp(0.0, logits)
        observed = ~np.isnan(responses)
        responses_filled = np.where(observed, responses, 0.0)
        return responses_filled @ logits.T - observed.astype(np.float64) @ log_norm.T

    if model_type == "PCM":
        delta = item_params["delta"]
        step_logits = theta[:, np.newaxis, np.newaxis] - delta[np.newaxis, :, :]
    elif model_type == "GPCM":
        a = item_params["a"]
        delta = item_params["delta"]
        step_logits = a[np.newaxis, :, np.newaxis] * (
            theta[:, np.newaxis, np.newaxis] - delta[np.newaxis, :, :]
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    zeros = np.zeros(
        (step_logits.shape[0], step_logits.shape[1], 1),
        dtype=step_logits.dtype,
    )
    cum_logits = np.concatenate([zeros, np.cumsum(step_logits, axis=-1)], axis=-1)
    log_probs = cum_logits - logsumexp(cum_logits, axis=-1, keepdims=True)

    observed = ~np.isnan(responses)
    response_indices = np.where(observed, responses, 0).astype(int)
    n_categories = log_probs.shape[-1]
    item_offsets = (
        np.arange(
            response_indices.shape[1],
            dtype=np.int64,
        )[np.newaxis, :]
        * n_categories
    )
    flat_indices = item_offsets + response_indices
    log_probs_flat = log_probs.reshape(log_probs.shape[0], -1)
    gathered = np.take(log_probs_flat, flat_indices, axis=1).transpose(1, 0, 2)
    return np.sum(gathered * observed[:, np.newaxis, :], axis=-1, dtype=np.float64)


def _log_likelihood_binary_chunk_person_grids(
    theta: np.ndarray,
    responses_filled: np.ndarray,
    observed_float: np.ndarray,
    *,
    model_type: str,
    b_chunk: np.ndarray,
    a_chunk: np.ndarray | None,
) -> np.ndarray:
    """Binary log-likelihoods on one theta grid per person, ``(S, P, Q)``.

    With ``a_si (theta - b_si)`` the response-weighted logit term is linear in
    theta and reduces to matrix products; only the normalizer
    ``log(1 + exp(logit))`` needs the full ``(S, P, Q, I)`` array, which is
    evaluated in person blocks to bound memory.
    """
    if model_type == "1PL":
        a_chunk = np.ones_like(b_chunk)
    elif a_chunk is None:
        raise ValueError("a_chunk is required for 2PL chunked scoring.")
    elif model_type != "2PL":
        raise ValueError(
            f"Unsupported binary model_type for chunk scoring: {model_type!r}"
        )

    n_draws, n_items = b_chunk.shape
    n_persons, n_grid = theta.shape
    # sum_i r_pi a_si (theta_pg - b_si) = theta_pg * slope_sp - offset_sp
    slope = responses_filled @ a_chunk.T  # (P, S)
    offset = responses_filled @ (a_chunk * b_chunk).T  # (P, S)
    out = (
        theta[np.newaxis, :, :] * slope.T[:, :, np.newaxis] - offset.T[:, :, np.newaxis]
    )

    block = max(1, _PERSON_GRID_BLOCK_ELEMENTS // (n_draws * n_grid * n_items))
    for start in range(0, n_persons, block):
        stop = min(start + block, n_persons)
        logits = a_chunk[:, np.newaxis, np.newaxis, :] * (
            theta[np.newaxis, start:stop, :, np.newaxis]
            - b_chunk[:, np.newaxis, np.newaxis, :]
        )
        log_norm = np.logaddexp(0.0, logits)
        out[:, start:stop, :] -= np.einsum(
            "pi,spgi->spg", observed_float[start:stop], log_norm
        )
    return out


def _log_likelihood_binary_chunk(
    theta: np.ndarray,
    responses: np.ndarray,
    *,
    model_type: str,
    b_chunk: np.ndarray,
    a_chunk: np.ndarray | None = None,
) -> np.ndarray:
    """Compute binary-model log-likelihoods for a chunk of posterior draws.

    Returns an array with shape ``(S_chunk, P, Q)``. ``theta`` is ``(Q,)``
    for a shared grid or ``(P, Q)`` for one grid per person.
    """
    observed = ~np.isnan(responses)
    responses_filled = np.where(observed, responses, 0.0)
    observed_float = observed.astype(np.float64)

    if theta.ndim == 2:
        return _log_likelihood_binary_chunk_person_grids(
            theta,
            responses_filled,
            observed_float,
            model_type=model_type,
            b_chunk=b_chunk,
            a_chunk=a_chunk,
        )

    if model_type == "1PL":
        logits = theta[np.newaxis, :, np.newaxis] - b_chunk[:, np.newaxis, :]
    elif model_type == "2PL":
        if a_chunk is None:
            raise ValueError("a_chunk is required for 2PL chunked scoring.")
        logits = a_chunk[:, np.newaxis, :] * (
            theta[np.newaxis, :, np.newaxis] - b_chunk[:, np.newaxis, :]
        )
    else:
        raise ValueError(
            f"Unsupported binary model_type for chunk scoring: {model_type!r}"
        )

    log_norm = np.logaddexp(0.0, logits)
    return np.einsum(
        "pi,sqi->spq", responses_filled, logits, optimize=True
    ) - np.einsum("pi,sqi->spq", observed_float, log_norm, optimize=True)


def _log_likelihood_polytomous_chunk(
    theta: np.ndarray,
    responses: np.ndarray,
    *,
    model_type: str,
    delta_chunk: np.ndarray,
    a_chunk: np.ndarray | None = None,
) -> np.ndarray:
    """PCM/GPCM log-likelihoods for a chunk of posterior draws, ``(S, P, Q)``.

    The per-draw algebra of :func:`_log_likelihood_all_persons` with a draw
    axis: with ``D_sik = sum_{j<=k} delta_sij``, the log probability of
    category ``k`` is ``a_si (k theta - D_sik)`` minus the item normalizer,
    so the observed-category terms reduce to matrix products and only the
    normalizer needs the full array. ``theta`` is ``(Q,)`` for a shared grid
    (the normalizer is then person-free, ``(S, Q, I)``) or ``(P, Q)`` for one
    grid per person (``(S, P_block, Q, I)``, in person blocks).
    """
    if model_type == "PCM":
        slope_a = np.ones(delta_chunk.shape[:2])
    elif model_type == "GPCM":
        if a_chunk is None:
            raise ValueError("a_chunk is required for GPCM chunked scoring.")
        slope_a = a_chunk
    else:
        raise ValueError(
            f"Unsupported polytomous model_type for chunk scoring: {model_type!r}"
        )
    n_draws, n_items, _ = delta_chunk.shape
    cum_delta = np.concatenate(
        [np.zeros((n_draws, n_items, 1)), np.cumsum(delta_chunk, axis=-1)], axis=-1
    )  # (S, I, K)
    n_categories = cum_delta.shape[-1]
    observed = ~np.isnan(responses)
    obs_float = observed.astype(np.float64)
    idx = np.where(observed, responses, 0).astype(int)
    slope = (obs_float * idx) @ slope_a.T  # (P, S): sum_i a_si x_pi
    picked = cum_delta[:, np.arange(n_items)[np.newaxis, :], idx]  # (S, P, I)
    offset = np.einsum("spi,si,pi->sp", picked, slope_a, obs_float)

    # sum_k exp(a (k theta - D_k)) = sum_k v^k w_k with v = exp(a theta) and
    # w_k = exp(-a D_k): a polynomial in v, evaluated by Horner's rule with
    # one exp per (draw, person, grid point, item) instead of one per
    # category. All terms are positive and w_0 = 1, so the sum is >= 1 and
    # has no cancellation; it is used only while no term can overflow
    # (exponents below _HORNER_MAX_EXPONENT), else the max-shifted form.
    max_exponent = (n_categories - 1) * np.abs(slope_a).max() * np.abs(
        theta
    ).max() + np.abs(slope_a[..., np.newaxis] * cum_delta).max()
    horner = bool(max_exponent < _HORNER_MAX_EXPONENT)
    if horner:
        w = np.exp(-slope_a[..., np.newaxis] * cum_delta)  # (S, I, K)

    def log_normalizer(th: np.ndarray) -> np.ndarray:
        # th broadcasts against (S, ..., I); category axis first, as in
        # _log_likelihood_all_persons (short last-axis reductions are slow).
        extra = (np.newaxis,) * (th.ndim - 1)
        a = slope_a[(slice(None), *extra, slice(None))]
        if horner:
            v = np.exp(a * th)
            z = np.broadcast_to(
                w[(slice(None), *extra, slice(None), n_categories - 1)], v.shape
            ).copy()
            for k in range(n_categories - 2, -1, -1):
                z *= v
                z += w[(slice(None), *extra, slice(None), k)]
            return np.log(z, out=z)
        c = np.stack(
            [
                a * (k * th - cum_delta[(slice(None), *extra, slice(None), k)])
                for k in range(n_categories)
            ]
        )
        m = c.max(axis=0)
        c -= m
        np.exp(c, out=c)
        return m + np.log(c.sum(axis=0))

    if theta.ndim == 1:
        out = theta * slope.T[:, :, np.newaxis] - offset[:, :, np.newaxis]
        log_z = log_normalizer(theta[:, np.newaxis])  # (S, Q, I)
        out -= np.einsum("pi,sqi->spq", obs_float, log_z)
        return out

    n_persons, n_grid = theta.shape
    out = theta[np.newaxis] * slope.T[:, :, np.newaxis] - offset[:, :, np.newaxis]
    block = max(
        1,
        _POLYTOMOUS_PERSON_GRID_BLOCK_ELEMENTS
        // (n_draws * n_grid * n_items * n_categories),
    )
    for start in range(0, n_persons, block):
        stop = min(start + block, n_persons)
        log_z = log_normalizer(theta[start:stop, :, np.newaxis])  # (S, Pb, Q, I)
        out[:, start:stop] -= np.einsum("pi,spgi->spg", obs_float[start:stop], log_z)
    return out


def _compute_log_likelihood_chunk(
    chunk_start: int,
    chunk_stop: int,
    *,
    theta_grid: np.ndarray,
    responses: np.ndarray,
    param_samples: dict[str, np.ndarray],
    param_keys: Sequence[str],
    model_type: str,
    log_likelihood_fn: LogLikelihoodFn,
    custom_log_likelihood: bool,
) -> np.ndarray:
    """Compute log-likelihoods for one posterior-draw chunk."""
    use_binary_chunk = (not custom_log_likelihood) and model_type in ("1PL", "2PL")
    if use_binary_chunk:
        b_chunk = param_samples["b"][chunk_start:chunk_stop]
        a_chunk = (
            param_samples["a"][chunk_start:chunk_stop] if model_type == "2PL" else None
        )
        return _log_likelihood_binary_chunk(
            theta_grid,
            responses,
            model_type=model_type,
            b_chunk=b_chunk,
            a_chunk=a_chunk,
        )
    if not custom_log_likelihood and model_type in ("PCM", "GPCM"):
        return _log_likelihood_polytomous_chunk(
            theta_grid,
            responses,
            model_type=model_type,
            delta_chunk=param_samples["delta"][chunk_start:chunk_stop],
            a_chunk=(
                param_samples["a"][chunk_start:chunk_stop]
                if model_type == "GPCM"
                else None
            ),
        )

    return np.stack(
        [
            _log_likelihood_all_persons(
                theta_grid,
                responses,
                {key: param_samples[key][s] for key in param_keys},
                model_type,
                log_likelihood_fn,
                custom_log_likelihood=custom_log_likelihood,
            )
            for s in range(chunk_start, chunk_stop)
        ],
        axis=0,
    )


# ---------------------------------------------------------------------------
# Quadrature grid setup
# ---------------------------------------------------------------------------


def _gauss_hermite_grid(
    n_points: int,
    prior_mean: float = 0.0,
    prior_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate Gauss-Hermite quadrature grid for normal prior.

    Parameters
    ----------
    n_points : int
        Number of quadrature points.
    prior_mean : float
        Mean of the normal prior (default 0.0).
    prior_scale : float
        Standard deviation of the normal prior (default 1.0).

    Returns
    -------
    theta_grid : np.ndarray
        Quadrature points ``(Q,)`` on the original scale.
    log_prior_weights : np.ndarray
        Log prior weights ``(Q,)`` (normalized to sum to 1).

    Notes
    -----
    Uses ``numpy.polynomial.hermite.hermgauss`` which returns nodes and
    weights for the standard normal distribution. Transforms to the
    user-specified prior scale.
    """
    # Get standard normal nodes and weights
    nodes, weights = hermgauss(n_points)

    # Transform to N(prior_mean, prior_scale^2)
    theta_grid = prior_mean + np.sqrt(2) * prior_scale * nodes

    # Weights need to be scaled: w_i -> w_i / sqrt(pi) for the transform,
    # then normalize to sum to 1 for proper Bayesian updating
    log_weights = np.log(weights / np.sqrt(np.pi))
    log_prior_weights = log_weights - logsumexp(log_weights)

    return theta_grid.astype(np.float64), log_prior_weights.astype(np.float64)


def _uniform_grid(
    n_points: int,
    bounds: tuple[float, float] = (-6.0, 6.0),
    prior_mean: float = 0.0,
    prior_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate uniform quadrature grid for normal prior.

    Parameters
    ----------
    n_points : int
        Number of quadrature points.
    bounds : tuple[float, float]
        Lower and upper bounds for the grid (default (-6, 6)).
    prior_mean : float
        Mean of the normal prior (default 0.0).
    prior_scale : float
        Standard deviation of the normal prior (default 1.0).

    Returns
    -------
    theta_grid : np.ndarray
        Quadrature points ``(Q,)``.
    log_prior_weights : np.ndarray
        Log prior weights ``(Q,)`` (normalized to sum to 1).

    Notes
    -----
    Uniform grid with trapezoidal rule for polytomous models where
    Gauss-Hermite may not provide sufficient density in the tails.
    """
    lower, upper = bounds
    theta_grid = np.linspace(lower, upper, n_points, dtype=np.float64)

    # Log prior density under N(prior_mean, prior_scale^2)
    log_prior = -0.5 * ((theta_grid - prior_mean) / prior_scale) ** 2
    log_prior -= 0.5 * np.log(2 * np.pi) + np.log(prior_scale)

    # Trapezoidal rule weights (endpoints get half weight)
    weights = np.ones(n_points, dtype=np.float64)
    weights[0] = weights[-1] = 0.5
    log_weights = np.log(weights) + log_prior

    # Normalize to sum to 1
    log_prior_weights = log_weights - logsumexp(log_weights)

    return theta_grid, log_prior_weights


def _uniform_person_grids(
    lower: np.ndarray,
    upper: np.ndarray,
    n_points: int,
    prior_mean: float = 0.0,
    prior_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-person uniform grids with trapezoid-times-prior log weights.

    Same construction as :func:`_uniform_grid`, but with one grid
    ``linspace(lower[p], upper[p], n_points)`` per person.

    Returns
    -------
    theta_grid : np.ndarray
        Grids ``(P, Q)``.
    log_prior_weights : np.ndarray
        Log prior weights ``(P, Q)``, each row normalized to sum to 1.
    """
    unit = np.linspace(0.0, 1.0, n_points, dtype=np.float64)
    theta_grid = lower[:, np.newaxis] + (upper - lower)[:, np.newaxis] * unit
    log_weights = _normal_log_density(theta_grid, prior_mean, prior_scale)
    log_weights[:, [0, -1]] += np.log(0.5)
    log_prior_weights = log_weights - logsumexp(log_weights, axis=-1, keepdims=True)
    return theta_grid, log_prior_weights


def _normal_log_density(
    theta_grid: np.ndarray,
    prior_mean: float,
    prior_scale: float,
) -> np.ndarray:
    """Evaluate the normal prior log-density on the theta grid."""
    return (
        -0.5 * ((theta_grid - prior_mean) / prior_scale) ** 2
        - 0.5 * np.log(2 * np.pi)
        - np.log(prior_scale)
    )


def _validate_probability_level(name: str, value: float) -> float:
    """Validate a probability level in the open unit interval."""
    value = float(value)
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be in (0, 1), got {value!r}.")
    return value


def _validate_probability_levels(
    name: str,
    value: float | Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Validate one or more probability levels in the open unit interval."""
    arr = np.asarray(value, dtype=np.float64)
    scalar = arr.ndim == 0
    if scalar:
        arr = arr.reshape(1)
    elif arr.ndim != 1:
        raise ValueError(f"{name} must be a scalar or 1-D sequence of levels.")
    if arr.size < 1:
        raise ValueError(f"{name} must contain at least one level.")
    if not np.all(np.isfinite(arr)) or np.any((arr <= 0.0) | (arr >= 1.0)):
        raise ValueError(f"{name} levels must all be in (0, 1), got {arr!r}.")
    return arr, scalar


def _squeeze_level_axis(values: np.ndarray, scalar_level: bool) -> np.ndarray:
    """Return ``(P,)`` for scalar levels and ``(L, P)`` for vector levels."""
    return values[0] if scalar_level else values


def _validate_positive_int(name: str, value: int) -> int:
    """Validate a strictly positive integer argument."""
    scalar = np.asarray(value)
    if scalar.ndim != 0:
        raise ValueError(f"{name} must be a scalar positive integer, got {value!r}.")
    scalar = scalar.item()
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    if isinstance(scalar, (int, np.integer)):
        value = int(scalar)
    elif isinstance(scalar, (float, np.floating)):
        if not np.isfinite(scalar) or not float(scalar).is_integer():
            raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        value = int(scalar)
    else:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}.")
    return value


def _validate_responses_for_model(
    responses: np.ndarray,
    model_type: str,
    *,
    max_category: int | None = None,
    allow_nan: bool = True,
) -> None:
    """Validate response values for a scorer model type."""
    if np.any(np.isinf(responses)):
        n_inf = np.sum(np.isinf(responses))
        raise ValueError(
            f"Responses contain {n_inf} Inf value(s). Use np.nan for missing responses."
        )
    if not allow_nan and np.any(np.isnan(responses)):
        raise ValueError(
            "Drawing item posteriors from an approximator does not support "
            "NaN missing responses. Provide item_posteriors estimated from "
            "calibration data, then call score() with missing responses."
        )

    observed_responses = responses[~np.isnan(responses)]
    if model_type in ("1PL", "2PL"):
        unique_values = np.unique(observed_responses)
        invalid_values = unique_values[(unique_values != 0) & (unique_values != 1)]
        if len(invalid_values) > 0:
            raise ValueError(
                "Binary models (1PL, 2PL) require responses in {0, 1}. "
                f"Found invalid values: {sorted(invalid_values)}."
            )
    elif model_type in ("PCM", "GPCM"):
        if observed_responses.size > 0 and not np.all(observed_responses >= 0):
            raise ValueError(
                "Polytomous models (PCM, GPCM) require non-negative "
                "response values. Found minimum value: "
                f"{np.min(observed_responses)}."
            )
        if not np.allclose(observed_responses, np.round(observed_responses)):
            raise ValueError(
                "Polytomous models (PCM, GPCM) require integer response "
                "values. Found non-integer values."
            )
        if max_category is not None:
            observed_max = (
                np.max(observed_responses) if observed_responses.size > 0 else None
            )
            if observed_max is not None and observed_max > max_category:
                raise ValueError(
                    f"Polytomous responses for {model_type} must be in "
                    f"{{0, ..., {max_category}}}. "
                    f"Found maximum value: {observed_max}."
                )


# ---------------------------------------------------------------------------
# Model type detection
# ---------------------------------------------------------------------------


def _detect_model_type(posteriors: dict[str, np.ndarray]) -> str:
    """Detect IRT model type from posterior dictionary keys.

    Parameters
    ----------
    posteriors : dict
        Dictionary with keys like ``'a'``, ``'b'``, ``'delta'``, etc.

    Returns
    -------
    str
        One of ``'1PL'``, ``'2PL'``, ``'PCM'``, ``'GPCM'``.

    Raises
    ------
    ValueError
        If the key combination is not recognized.

    Notes
    -----
    Filters out metadata and derived keys before inspection:
    ``'n_items'``, ``'n_persons'``, ``'d'``, ``'d_delta'``, ``'sigma_a'``,
    ``'sigma_b'``, ``'sigma_delta'``.

    Detection logic:
    - ``'b'`` only → 1PL
    - ``{'a', 'b'}`` → 2PL
    - ``'delta'`` only → PCM
    - ``{'a', 'delta'}`` → GPCM
    """
    # Filter out metadata and derived keys
    core_keys = set(posteriors.keys()) - _POSTERIOR_METADATA_KEYS

    if core_keys == {"b"}:
        return "1PL"
    elif core_keys == {"a", "b"}:
        return "2PL"
    elif core_keys == {"delta"}:
        return "PCM"
    elif core_keys == {"a", "delta"}:
        return "GPCM"
    else:
        raise ValueError(
            f"Unrecognized posterior keys: {sorted(core_keys)}. "
            "Expected: 'b' (1PL), {'a', 'b'} (2PL), 'delta' (PCM), "
            "or {'a', 'delta'} (GPCM)."
        )


# ---------------------------------------------------------------------------
# Posteriors validation
# ---------------------------------------------------------------------------


def _squeeze_batch_dim(
    arr: np.ndarray,
    *,
    expected_ndim: int | None = None,
) -> np.ndarray:
    """Squeeze a leading singleton batch dimension when unambiguous.

    For built-in models, ``expected_ndim`` disambiguates item posterior shapes:
    binary posteriors expect ``(S, I)`` while polytomous posteriors expect
    ``(S, I, K-1)``. This avoids misreading valid single-sample polytomous
    arrays ``(1, I, K-1)`` as batched arrays.
    """
    if expected_ndim is not None:
        if arr.ndim == expected_ndim + 1 and arr.shape[0] == 1:
            return arr.squeeze(axis=0)
        return arr
    if arr.ndim >= 4 and arr.shape[0] == 1:
        return arr.squeeze(axis=0)
    return arr


def _expected_posterior_ndims(model_type: str) -> dict[str, int]:
    """Return expected posterior ndim by parameter key for a built-in model."""
    if model_type == "1PL":
        return {"b": 2}
    if model_type == "2PL":
        return {"a": 2, "b": 2}
    if model_type == "PCM":
        return {"delta": 3}
    if model_type == "GPCM":
        return {"a": 2, "delta": 3}
    return {}


def _validate_item_ids(
    item_ids: np.ndarray | None,
    n_items: int,
    *,
    source: str = "item_ids",
) -> np.ndarray | None:
    """Validate optional item identifiers used to guard cached posteriors."""
    if item_ids is None:
        return None

    ids = np.asarray(item_ids)
    if ids.ndim != 1:
        raise ValueError(f"{source} must be 1-D with one id per item.")
    if ids.shape[0] != n_items:
        raise ValueError(
            f"{source} has {ids.shape[0]} entries, but expected {n_items}."
        )
    if len(set(ids.tolist())) != n_items:
        raise ValueError(f"{source} must contain unique item identifiers.")
    return ids.copy()


def _posterior_item_ids(
    posteriors: Mapping[str, np.ndarray],
    n_items: int,
    explicit_item_ids: np.ndarray | None = None,
) -> np.ndarray | None:
    """Extract and validate item ids from explicit args or posterior metadata."""
    validated_explicit = _validate_item_ids(explicit_item_ids, n_items)
    validated_posterior = None
    if "item_ids" in posteriors:
        validated_posterior = _validate_item_ids(
            posteriors["item_ids"],
            n_items,
            source="posteriors['item_ids']",
        )

    if (
        validated_explicit is not None
        and validated_posterior is not None
        and not np.array_equal(validated_explicit, validated_posterior)
    ):
        raise ValueError("Explicit item_ids do not match posteriors['item_ids'].")

    if validated_explicit is not None:
        return validated_explicit
    return validated_posterior


def _validate_custom_posteriors(
    posteriors: Mapping[str, np.ndarray],
) -> tuple[int, int]:
    """Validate custom posterior samples for custom log-likelihood scoring."""
    param_keys = [key for key in posteriors if key not in _POSTERIOR_METADATA_KEYS]
    if not param_keys:
        raise ValueError("Custom posteriors must contain at least one parameter key.")

    n_samples = None
    n_items = None
    for key in param_keys:
        arr = _squeeze_batch_dim(np.asarray(posteriors[key]))
        if arr.ndim < 1:
            raise ValueError(
                f"Unexpected shape for custom posterior {key}: {arr.shape}. "
                "Expected at least a sample axis."
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Non-finite values (NaN/Inf) found in {key}.")
        if n_samples is None:
            n_samples = arr.shape[0]
        elif arr.shape[0] != n_samples:
            raise ValueError(
                f"Inconsistent sample count: expected {n_samples}, "
                f"{key} has {arr.shape[0]} samples."
            )
        if arr.ndim >= 2:
            if n_items is None:
                n_items = arr.shape[1]
            elif arr.shape[1] != n_items:
                raise ValueError(
                    f"Inconsistent item count: expected {n_items}, "
                    f"{key} has {arr.shape[1]} items."
                )

    if n_samples is None or n_samples < 1:
        raise ValueError("Custom posteriors must contain at least one sample.")
    if n_items is None:
        if "item_ids" not in posteriors:
            raise ValueError(
                "Custom posteriors must include at least one item-indexed "
                "parameter with shape (S, I, ...) or item_ids metadata."
            )
        n_items = np.asarray(posteriors["item_ids"]).shape[0]
    return int(n_samples), int(n_items)


def _posterior_parameter_keys(
    posteriors: Mapping[str, np.ndarray],
    model_type: str,
) -> list[str]:
    """Return posterior sample keys used as item parameters for scoring."""
    required_keys = {
        "1PL": ["b"],
        "2PL": ["a", "b"],
        "PCM": ["delta"],
        "GPCM": ["a", "delta"],
    }
    if model_type in required_keys:
        return required_keys[model_type]
    if model_type == "custom":
        return [key for key in posteriors if key not in _POSTERIOR_METADATA_KEYS]
    raise ValueError(f"Unknown model_type: {model_type!r}")


def _validate_posteriors(
    posteriors: dict[str, np.ndarray],
    model_type: str,
) -> tuple[int, int]:
    """Validate posterior dictionary structure and contents.

    Parameters
    ----------
    posteriors : dict
        Posterior samples dictionary.
    model_type : str
        One of ``'1PL'``, ``'2PL'``, ``'PCM'``, ``'GPCM'``.

    Returns
    -------
    n_samples : int
        Number of posterior samples (S).
    n_items : int
        Number of items (I).

    Raises
    ------
    ValueError
        If required keys are missing, shapes are inconsistent, or NaN/Inf
        values are present.
    """
    # Define required keys per model type
    required_keys = {
        "1PL": ["b"],
        "2PL": ["a", "b"],
        "PCM": ["delta"],
        "GPCM": ["a", "delta"],
    }

    if model_type == "custom":
        return _validate_custom_posteriors(posteriors)

    if model_type not in required_keys:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    expected_keys = required_keys[model_type]
    core_keys = set(posteriors.keys()) - _POSTERIOR_METADATA_KEYS
    unexpected_keys = core_keys - set(expected_keys)
    if unexpected_keys:
        raise ValueError(
            f"Unexpected posterior keys for {model_type}: "
            f"{sorted(unexpected_keys)}. Expected exactly "
            f"{sorted(expected_keys)} plus metadata keys."
        )
    missing_keys = set(expected_keys) - set(posteriors.keys())
    if missing_keys:
        raise ValueError(
            f"Missing required keys for {model_type}: {sorted(missing_keys)}. "
            f"Got: {sorted(posteriors.keys())}"
        )

    # Extract shape info from the model's item-axis parameter.
    first_key = expected_keys[0]
    first_array = _squeeze_batch_dim(np.asarray(posteriors[first_key]))

    expected_ndims = _expected_posterior_ndims(model_type)

    first_expected_ndim = expected_ndims[first_key]
    first_array = _squeeze_batch_dim(
        first_array,
        expected_ndim=first_expected_ndim,
    )
    if first_array.ndim != first_expected_ndim:
        raise ValueError(
            f"Unexpected shape for {first_key}: {first_array.shape}. "
            f"Expected {first_expected_ndim}-D after squeezing batch dimension."
        )

    n_samples, n_items = first_array.shape[:2]

    # Validate shapes of all required keys before checking values, so callers
    # get structural errors before domain-specific parameter errors.
    validated_arrays = {}
    for key in expected_keys:
        expected_ndim = expected_ndims[key]
        arr_squeezed = _squeeze_batch_dim(
            np.asarray(posteriors[key]),
            expected_ndim=expected_ndim,
        )
        if arr_squeezed.ndim != expected_ndim:
            raise ValueError(
                f"Unexpected shape for {key}: {arr_squeezed.shape}. "
                f"Expected {expected_ndim}-D after squeezing batch dimension."
            )

        if arr_squeezed.shape[0] != n_samples:
            raise ValueError(
                f"Inconsistent sample count: {first_key} has {n_samples} samples, "
                f"{key} has {arr_squeezed.shape[0]} samples."
            )

        if arr_squeezed.shape[1] != n_items:
            raise ValueError(
                f"Inconsistent item count: {first_key} has {n_items} items, "
                f"{key} has {arr_squeezed.shape[1]} items."
            )
        validated_arrays[key] = arr_squeezed

    for key, arr_squeezed in validated_arrays.items():
        # Check for NaN/Inf
        if not np.all(np.isfinite(arr_squeezed)):
            raise ValueError(f"Non-finite values (NaN/Inf) found in {key}.")

        if key == "a" and np.any(arr_squeezed <= 0):
            raise ValueError(
                "Discrimination posteriors 'a' must be strictly positive. "
                f"Found minimum value {np.min(arr_squeezed)}."
            )

        if key == "delta" and arr_squeezed.shape[-1] < 1:
            raise ValueError(
                "Polytomous posteriors require at least one step parameter "
                f"per item, got shape {arr_squeezed.shape}."
            )

    return n_samples, n_items


def _cell_edges(theta_grid: np.ndarray) -> np.ndarray:
    """Cell edges for grid-supported posterior masses.

    Each grid mass is treated as uniform on its cell. Interior cell edges are
    the midpoints between neighbouring grid points; the outer edges are the
    first and last grid point. The outer cells therefore have half width,
    which matches the half-weight end points of the trapezoidal rule used by
    :func:`_uniform_grid` (the mass of an end point is the integral over half
    a cell).

    Parameters
    ----------
    theta_grid : np.ndarray
        Strictly increasing grid ``(G,)`` or one grid per person ``(P, G)``.

    Returns
    -------
    np.ndarray
        Edges with shape ``(G + 1,)`` or ``(P, G + 1)``.
    """
    theta_grid = np.asarray(theta_grid, dtype=np.float64)
    mids = 0.5 * (theta_grid[..., 1:] + theta_grid[..., :-1])
    return np.concatenate(
        [theta_grid[..., :1], mids, theta_grid[..., -1:]],
        axis=-1,
    )


def _edge_cdf(
    theta_grid: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Cell edges and the CDF at those edges, both ``(P, G + 1)``.

    The CDF is piecewise linear between the edges, i.e. the density is
    piecewise constant (each grid mass uniform on its cell).
    """
    n_persons = weights.shape[0]
    edges = _cell_edges(theta_grid)
    if edges.ndim == 1:
        edges = np.broadcast_to(edges, (n_persons, edges.shape[0]))
    cdf = np.concatenate(
        [np.zeros((n_persons, 1)), np.cumsum(weights, axis=-1)],
        axis=-1,
    )
    # Guard against tiny numerical drift below/above 1.
    cdf = cdf / cdf[:, [-1]]
    return edges, cdf


def _weighted_quantile_grid(
    theta_grid: np.ndarray,
    weights: np.ndarray,
    probs: np.ndarray,
) -> np.ndarray:
    """Quantiles of a grid posterior with each mass uniform on its cell.

    The CDF is piecewise linear through the cell edges (see
    :func:`_cell_edges`) and is inverted by linear interpolation. Placing
    the CDF value of a grid point's whole mass *at* that point (the
    discrete-quantile convention) biases quantiles low by up to half a grid
    cell; the cell-edge convention removes that bias.

    Parameters
    ----------
    theta_grid : np.ndarray
        Grid ``(G,)`` shared by all persons or per-person grids ``(P, G)``.
    weights : np.ndarray
        Posterior masses ``(P, G)``.
    probs : np.ndarray
        Probability levels in ``[0, 1]``, shape ``(K,)``.

    Returns
    -------
    np.ndarray
        Quantiles with shape ``(P, K)``.

    Notes
    -----
    Inverse-CDF sampling is the inversion method (Devroye, 1986, Ch. II,
    Theorem 2.1): it applies because a piecewise-constant density has a
    continuous CDF.
    """
    edges, cdf = _edge_cdf(theta_grid, weights)
    out = np.zeros((weights.shape[0], probs.shape[0]), dtype=np.float64)
    for p in range(weights.shape[0]):
        out[p] = np.interp(probs, cdf[p], edges[p])
    return out


def _weighted_hdi_grid(
    theta_grid: np.ndarray,
    weights: np.ndarray,
    hdi_level: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Shortest interval with ``hdi_level`` mass under the cell-uniform density.

    Candidate intervals start at evenly spaced CDF levels and are read off
    the same piecewise-linear CDF as :func:`_weighted_quantile_grid`, so the
    HDI is consistent with the ETI and plausible values.
    """
    n_persons, n_grid = weights.shape
    lower = np.zeros(n_persons, dtype=np.float64)
    upper = np.zeros(n_persons, dtype=np.float64)
    candidate_count = max(257, 8 * n_grid + 1)
    start_probs = np.linspace(
        0.0,
        1.0 - hdi_level,
        candidate_count,
        dtype=np.float64,
    )
    end_probs = start_probs + hdi_level
    edges, cdf = _edge_cdf(theta_grid, weights)
    grid = np.broadcast_to(np.asarray(theta_grid, dtype=np.float64), weights.shape)

    for p in range(n_persons):
        mode_theta = grid[p, int(np.argmax(weights[p]))]
        lower_candidates = np.interp(start_probs, cdf[p], edges[p])
        upper_candidates = np.interp(end_probs, cdf[p], edges[p])
        widths = upper_candidates - lower_candidates
        best_width = np.min(widths)
        best_mask = np.isclose(widths, best_width, atol=1e-12, rtol=0.0)
        center_distance = np.abs(
            0.5 * (lower_candidates + upper_candidates) - mode_theta
        )
        best_idx = int(np.argmin(np.where(best_mask, center_distance, np.inf)))

        lower[p] = lower_candidates[best_idx]
        upper[p] = upper_candidates[best_idx]

    return lower, upper


def _aggregate_grid_posterior(
    theta_grid: np.ndarray,
    mixture_weights: np.ndarray,
    log_mixture_density: np.ndarray | None = None,
    eti_level: float | Sequence[float] | np.ndarray | None = 0.95,
    hdi_level: float | Sequence[float] | np.ndarray | None = 0.95,
    n_plausible_values: int | None = None,
) -> dict[str, np.ndarray | None]:
    """Aggregate marginal theta posterior masses on a (per-person) grid.

    ``theta_grid`` is ``(G,)`` for a grid shared by all persons or ``(P, G)``
    for one grid per person. Mean, SD and MAP use the grid masses directly;
    median, ETI, HDI and plausible values use the cell-uniform CDF. A level
    of ``None`` skips that interval: its bounds are ``None`` and its levels
    empty.
    """
    eti_levels, eti_scalar = (
        _validate_probability_levels("eti_level", eti_level)
        if eti_level is not None
        else (np.empty(0), False)
    )
    hdi_levels, hdi_scalar = (
        _validate_probability_levels("hdi_level", hdi_level)
        if hdi_level is not None
        else (np.empty(0), False)
    )

    mixture_weights = np.asarray(mixture_weights, dtype=np.float64)
    theta_grid = np.asarray(theta_grid, dtype=np.float64)
    row_sums = mixture_weights.sum(axis=-1, keepdims=True)
    if np.any(row_sums <= 0) or not np.all(np.isfinite(row_sums)):
        raise ValueError("Posterior mixture weights are not normalizable.")
    mixture_weights = mixture_weights / row_sums

    mean = np.sum(mixture_weights * theta_grid, axis=-1)
    centered = theta_grid - mean[:, np.newaxis]
    variance = np.sum(mixture_weights * centered**2, axis=-1)
    sd = np.sqrt(np.maximum(variance, 0.0))

    median = _weighted_quantile_grid(
        theta_grid, mixture_weights, np.array([0.5], dtype=np.float64)
    )[:, 0]

    eti_lower = np.zeros(
        (eti_levels.shape[0], mixture_weights.shape[0]),
        dtype=np.float64,
    )
    eti_upper = np.zeros_like(eti_lower)
    for idx, level in enumerate(eti_levels):
        alpha = 1.0 - level
        eti = _weighted_quantile_grid(
            theta_grid,
            mixture_weights,
            np.array([alpha / 2.0, 1.0 - alpha / 2.0], dtype=np.float64),
        )
        eti_lower[idx] = eti[:, 0]
        eti_upper[idx] = eti[:, 1]

    hdi_lower = np.zeros(
        (hdi_levels.shape[0], mixture_weights.shape[0]),
        dtype=np.float64,
    )
    hdi_upper = np.zeros_like(hdi_lower)
    for idx, level in enumerate(hdi_levels):
        lower, upper = _weighted_hdi_grid(theta_grid, mixture_weights, float(level))
        hdi_lower[idx] = lower
        hdi_upper[idx] = upper

    # MAP should use the same discretized posterior mass as every other
    # summary on the evaluation grid.
    map_idx = np.argmax(mixture_weights, axis=-1)
    map_est = np.broadcast_to(theta_grid, mixture_weights.shape)[
        np.arange(mixture_weights.shape[0]), map_idx
    ].astype(np.float64)

    plausible_values = None
    if n_plausible_values is not None:
        if n_plausible_values < 1:
            raise ValueError("n_plausible_values must be at least 1.")
        probs = (
            np.arange(n_plausible_values, dtype=np.float64) + 0.5
        ) / n_plausible_values
        plausible_values = _weighted_quantile_grid(theta_grid, mixture_weights, probs)

    return {
        "mean": mean.astype(np.float64),
        "map": map_est,
        "median": median.astype(np.float64),
        "sd": sd.astype(np.float64),
        **{
            f"{kind}_{side}": (
                _squeeze_level_axis(values, scalar).astype(np.float64)
                if level is not None
                else None
            )
            for kind, level, scalar, bounds in (
                ("eti", eti_level, eti_scalar, (eti_lower, eti_upper)),
                ("hdi", hdi_level, hdi_scalar, (hdi_lower, hdi_upper)),
            )
            for side, values in zip(("lower", "upper"), bounds)
        },
        "plausible_values": plausible_values,
        "eti_levels": eti_levels,
        "hdi_levels": hdi_levels,
    }


# ---------------------------------------------------------------------------
# PersonScorer class
# ---------------------------------------------------------------------------


class PersonScorer:
    """Person scoring with item-posterior uncertainty propagation.

    Estimates person abilities θ from response matrices given item
    parameter posteriors. Supports all IRT models (1PL, 2PL, PCM, GPCM)
    with configurable θ priors and multiple scoring outputs (EAP, MAP,
    median, CI, HDI).

    The scorer caches item posteriors to avoid re-drawing from the
    approximator on subsequent ``score()`` calls with different persons
    but the same items.

    Parameters
    ----------
    approximator : EquivariantIRTApproximator
        Trained approximator for model type detection and posterior
        sampling (if ``item_posteriors`` not provided).
    item_posteriors : dict or None
        Pre-computed item posterior samples. If provided, the scorer
        caches them and skips re-drawing. Expected format:
        ``{a: (S, I), b: (S, I)}`` or similar.
    model_type : str or None
        Explicit model type override. If ``None``, detected from
        ``item_posteriors`` keys or adapter config.
    theta_prior : tuple[float, float]
        Mean and scale of the normal prior on θ (default ``(0.0, 1.0)``).
    log_likelihood : callable or None
        Custom log-likelihood function. If ``None``, built-in likelihoods
        are used based on ``model_type``.
    item_ids : array-like or None
        Optional stable item identifiers for cached posteriors. When present,
        ``score(..., item_ids=...)`` must provide the same identifiers in the
        same order.

    Examples
    --------
    Auto-draw posteriors from approximator:

    >>> scorer = PersonScorer(approximator)
    >>> results = scorer.score(
    ...     empirical_responses,  # (P, I)
    ...     n_item_samples=1000,
    ... )

    Reuse pre-computed posteriors:

    >>> posterior_drawer = PersonScorer(approximator)
    >>> posteriors = posterior_drawer.draw_item_posteriors(
    ...     empirical_responses,
    ...     n_item_samples=1000,
    ... )
    >>> scorer = PersonScorer(approximator, item_posteriors=posteriors)
    >>> results_a = scorer.score(empirical_responses)
    >>> results_b = scorer.score(new_person_responses)  # no re-drawing
    """

    def __init__(
        self,
        approximator: EquivariantIRTApproximator,
        item_posteriors: dict[str, np.ndarray] | None = None,
        model_type: str | None = None,
        theta_prior: tuple[float, float] = (0.0, 1.0),
        log_likelihood: LogLikelihoodFn | None = None,
        item_ids: np.ndarray | None = None,
    ):
        self.approximator = approximator
        self._model_type_explicit = model_type is not None
        self.theta_prior_mean, self.theta_prior_scale = theta_prior
        self.theta_prior_mean = float(self.theta_prior_mean)
        self.theta_prior_scale = float(self.theta_prior_scale)
        if not np.isfinite(self.theta_prior_mean):
            raise ValueError(
                f"theta_prior mean must be finite, got {self.theta_prior_mean!r}."
            )
        if not np.isfinite(self.theta_prior_scale) or self.theta_prior_scale <= 0:
            raise ValueError(
                "theta_prior scale must be finite and strictly positive, "
                f"got {self.theta_prior_scale!r}."
            )
        self.custom_log_likelihood = log_likelihood

        # Resolve model type
        if model_type is not None:
            self.model_type = model_type
        elif item_posteriors is not None:
            try:
                self.model_type = _detect_model_type(item_posteriors)
            except ValueError:
                if log_likelihood is None:
                    raise
                self.model_type = "custom"
        else:
            self.model_type = self._detect_model_type_from_adapter()

        # Cache posteriors if provided
        self._item_posteriors: dict | None = None
        self._n_items: int | None = None
        self._n_samples: int | None = None
        self._item_ids: np.ndarray | None = None

        if item_posteriors is not None:
            # Validate and cache posteriors
            n_samples, n_items = _validate_posteriors(item_posteriors, self.model_type)
            self._item_ids = _posterior_item_ids(
                item_posteriors,
                n_items,
                explicit_item_ids=item_ids,
            )

            # Squeeze batch dimension if present
            self._item_posteriors = {}
            expected_ndims = _expected_posterior_ndims(self.model_type)
            for key, val in item_posteriors.items():
                if key == "item_ids":
                    continue
                self._item_posteriors[key] = _squeeze_batch_dim(
                    np.asarray(val),
                    expected_ndim=expected_ndims.get(key),
                )

            self._n_samples = n_samples
            self._n_items = n_items

    def _detect_model_type_from_adapter(self) -> str:
        """Detect model type from approximator's adapter configuration."""
        adapter = self.approximator.adapter

        has_one_hot_responses = self._adapter_has_response_encoding(adapter)
        fallback_model_type = "PCM" if has_one_hot_responses else "1PL"

        # Check parameterization flag (2PL/GPCM adapters set this)
        internal_param = getattr(
            adapter,
            "_bayesflow_irt_internal_parameterization",
            None,
        )

        if internal_param in ("standard", "loading"):
            return "GPCM" if has_one_hot_responses else "2PL"

        try:
            return self._detect_from_sample()
        except Exception:
            return fallback_model_type

    @staticmethod
    def _adapter_has_response_encoding(adapter) -> bool:
        """Return whether the adapter one-hot encodes response categories."""
        for transform in getattr(adapter, "transforms", ()):
            include = getattr(transform, "include", None)
            constructor = getattr(transform, "transform_constructor", None)
            if constructor is not None and include is not None:
                include_set = (
                    set(include)
                    if isinstance(include, (list, tuple, set))
                    else {include}
                )
                if "responses" in include_set and PersonScorer._is_one_hot_constructor(
                    constructor
                ):
                    return True
        return False

    @staticmethod
    def _is_one_hot_constructor(constructor) -> bool:
        """Return whether an adapter transform constructor builds OneHotEncode."""
        if constructor is OneHotEncode:
            return True
        if not callable(constructor):
            return False
        try:
            transform = constructor()
        except TypeError:
            return False
        return isinstance(transform, OneHotEncode)

    @staticmethod
    def _adapter_response_n_categories(adapter) -> int | None:
        """Infer response category count from an adapter one-hot transform."""
        for transform in getattr(adapter, "transforms", ()):
            include = getattr(transform, "include", None)
            constructor = getattr(transform, "transform_constructor", None)
            if constructor is None or include is None:
                continue
            include_set = (
                set(include) if isinstance(include, (list, tuple, set)) else {include}
            )
            if "responses" not in include_set:
                continue
            if constructor is OneHotEncode:
                return None
            if callable(constructor):
                try:
                    transform_instance = constructor()
                except TypeError:
                    continue
                if isinstance(transform_instance, OneHotEncode):
                    return int(transform_instance.n_categories)
        return None

    @staticmethod
    def _adapter_condition_keys(adapter) -> list[str]:
        """Infer raw condition keys concatenated into ``inference_conditions``."""
        for transform in getattr(adapter, "transforms", ()):
            if getattr(transform, "into", None) == "inference_conditions":
                return list(getattr(transform, "keys", ()))
        return []

    def _detect_from_sample(self) -> str:
        """Draw a test sample from approximator to detect model type."""
        test_conditions = self._make_dummy_conditions()
        test_sample = self.approximator.sample(
            num_samples=1,
            conditions=test_conditions,
        )
        return _detect_model_type(test_sample)

    def _make_dummy_conditions(self) -> dict:
        """Create minimal dummy conditions for model type detection."""
        # Find a small validation dataset or use minimal synthetic data
        # This is a fallback path; ideally users provide model_type explicitly
        return {
            "responses": np.zeros((1, 10, 10), dtype=np.float32),
            "n_items": np.array([[10]], dtype=np.float32),
            "n_persons": np.array([[10]], dtype=np.float32),
        }

    def _get_log_likelihood_fn(self) -> LogLikelihoodFn:
        """Get the appropriate log-likelihood function."""
        if self.custom_log_likelihood is not None:
            return self.custom_log_likelihood

        likelihoods = {
            "1PL": _log_likelihood_1pl,
            "2PL": _log_likelihood_2pl,
            "PCM": _log_likelihood_pcm,
            "GPCM": _log_likelihood_gpcm,
        }
        if self.model_type not in likelihoods:
            raise ValueError(
                f"Invalid model_type: {self.model_type!r}. "
                f"Expected one of: {sorted(likelihoods.keys())}. "
                f"This may indicate a bug in model type detection. "
                f"Try passing model_type explicitly to PersonScorer.__init__()."
            )
        return likelihoods[self.model_type]

    def _validate_or_redetect_posteriors(
        self,
        posteriors: dict[str, np.ndarray],
    ) -> tuple[int, int]:
        """Validate posteriors and recover inferred model types from sample keys."""
        try:
            return _validate_posteriors(posteriors, self.model_type)
        except ValueError:
            if self._model_type_explicit:
                raise
            detected_model_type = _detect_model_type(posteriors)
            if detected_model_type == self.model_type:
                raise
            self.model_type = detected_model_type
            return _validate_posteriors(posteriors, self.model_type)

    def _get_quadrature_grid(
        self,
        theta_grid_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the theta evaluation grid used for posterior summaries.

        Score summaries use a deterministic uniform grid for all model
        types. This keeps CDF-, ETI-, HDI-, and plausible-value summaries
        numerically coherent; Gauss-Hermite nodes remain available as a helper
        for direct integration tests, but are not used for score() output.
        """
        half_width = 6.0 * self.theta_prior_scale
        return _uniform_grid(
            theta_grid_size,
            bounds=(
                self.theta_prior_mean - half_width,
                self.theta_prior_mean + half_width,
            ),
            prior_mean=self.theta_prior_mean,
            prior_scale=self.theta_prior_scale,
        )

    def _draw_item_posteriors(
        self,
        responses: np.ndarray,
        n_item_samples: int,
        item_conditions: Mapping[str, np.ndarray | float | int] | None = None,
        item_ids: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Draw item posterior samples from the approximator."""
        n_item_samples = _validate_positive_int("n_item_samples", n_item_samples)
        max_category = None
        if self.model_type in ("PCM", "GPCM"):
            n_categories = self._adapter_response_n_categories(
                self.approximator.adapter
            )
            if n_categories is not None:
                max_category = n_categories - 1
        _validate_responses_for_model(
            responses,
            self.model_type,
            max_category=max_category,
            allow_nan=False,
        )
        validated_item_ids = _validate_item_ids(item_ids, responses.shape[1])

        # Prepare conditions. Adapters train with n_items/n_persons as global
        # conditions, so do not let the approximator inject zero placeholders.
        conditions = {
            "responses": responses[np.newaxis, ...].astype(np.float32),
            "n_items": np.array([[responses.shape[1]]], dtype=np.float32),
            "n_persons": np.array([[responses.shape[0]]], dtype=np.float32),
        }
        if item_conditions is not None:
            for key, value in item_conditions.items():
                if key in {"responses", "n_items", "n_persons"}:
                    raise ValueError(
                        "item_conditions must not contain 'responses', 'n_items', "
                        "or 'n_persons'; pass responses as the first score() "
                        "argument and let the scorer infer size conditions."
                    )
                arr = np.asarray(value, dtype=np.float32)
                if arr.ndim == 0:
                    arr = arr.reshape(1, 1)
                elif arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                conditions[key] = arr

        expected_condition_keys = self._adapter_condition_keys(
            self.approximator.adapter
        )
        missing_condition_keys = [
            key for key in expected_condition_keys if key not in conditions
        ]
        if missing_condition_keys:
            raise ValueError(
                "Cannot draw item posteriors because the approximator adapter "
                "requires additional condition keys: "
                f"{missing_condition_keys}. Pass them via "
                "score(..., item_conditions=...)."
            )

        # Sample from approximator
        samples = self.approximator.sample(
            num_samples=n_item_samples,
            conditions=conditions,
        )

        # Squeeze batch dimension and return
        posteriors = {}
        for key, val in samples.items():
            arr = np.asarray(val)
            if arr.ndim == 4 and arr.shape[0] == 1:
                posteriors[key] = arr.squeeze(axis=0)  # (1, S, I, ...) -> (S, I, ...)
            elif arr.ndim == 3 and arr.shape[0] == 1:
                posteriors[key] = arr.squeeze(axis=0)  # (1, S, I) -> (S, I)
            else:
                posteriors[key] = arr

        if validated_item_ids is not None:
            posteriors["item_ids"] = validated_item_ids

        return posteriors

    def draw_item_posteriors(
        self,
        responses: np.ndarray,
        n_item_samples: int = 1000,
        item_conditions: Mapping[str, np.ndarray | float | int] | None = None,
        item_ids: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Draw reusable item posterior samples with the scorer's safe conditions.

        This helper includes ``n_items`` and ``n_persons`` automatically, and
        validates extra adapter conditions before sampling. Use it when item
        posteriors should be reused across several person-scoring calls.
        """
        responses = np.asarray(responses, dtype=np.float64)
        if responses.ndim != 2:
            raise ValueError(
                f"responses must be 2-D (P', I), got shape {responses.shape}"
            )
        if np.any(np.isinf(responses)):
            raise ValueError("Responses contain Inf values. Use NaN for missing data.")

        posteriors = self._draw_item_posteriors(
            responses,
            n_item_samples,
            item_conditions=item_conditions,
            item_ids=item_ids,
        )
        _, n_items = self._validate_or_redetect_posteriors(posteriors)
        if n_items != responses.shape[1]:
            raise ValueError(
                "Approximator returned posteriors for "
                f"{n_items} items, but responses have {responses.shape[1]} items."
            )
        return posteriors

    def score(
        self,
        responses: np.ndarray,
        n_item_samples: int = 1000,
        eti_level: float | Sequence[float] | np.ndarray = 0.95,
        hdi_level: float | Sequence[float] | np.ndarray | None = None,
        theta_grid_size: int = 31,
        n_plausible_values: int | None = None,
        n_cores: int | None = None,
        item_conditions: Mapping[str, np.ndarray | float | int] | None = None,
        item_ids: np.ndarray | None = None,
        adaptive_grid: bool = True,
        adaptive_span: float = _ADAPTIVE_GRID_HALF_WIDTH_SD,
        adaptive_grid_size: int = _ADAPTIVE_GRID_SIZE,
        device: str | None = None,
        torch_dtype: str = _TORCH_DTYPE_NAME,
        return_posterior: str | Collection[str] | None = None,
        draw_moments: bool = False,
        seed: int | None = None,
        intervals: str | Collection[str] | None = ("eti", "hdi"),
    ) -> PersonScore:
        """Score persons given their response matrix.

        Parameters
        ----------
        responses : np.ndarray
            Response matrix ``(P', I)`` for binary models or ``(P', I)``
            with integer categories ``{0, ..., K-1}`` for polytomous models.
            Use ``np.nan`` for missing/not-administered item responses when
            scoring against cached ``item_posteriors``.
        n_item_samples : int
            Number of item posterior samples to draw (only used if
            posteriors are not cached). Default 1000.
        eti_level : float or sequence
            Equal-tailed interval level or 1-D sequence of levels
            (default 0.95). Multi-level outputs have shape ``(L, P')``.
        hdi_level : float, sequence, or None
            Highest-density interval level(s). If ``None``, uses
            ``eti_level`` so ETI and HDI summaries are returned for the same
            requested levels.
        theta_grid_size : int
            Number of theta grid points of the fixed grid on ``theta_prior``
            +/- 6 prior scales. With ``adaptive_grid=True`` it is the size of
            the coarse first-pass grid only; the second-pass grids use
            ``adaptive_grid_size``.
        n_plausible_values : int or None
            Number of deterministic inverse-CDF plausible values to return
            from the marginal theta posterior. If ``None``, no plausible
            values are returned.
        n_cores : int or None
            Number of CPU worker threads to use for posterior-draw chunks.
            ``None`` uses serial execution. Values greater than 1 parallelize
            existing chunked score computation without changing output
            semantics.
        item_conditions : mapping or None
            Additional raw condition keys required by the approximator's
            adapter when drawing item posteriors, for example variable-prior
            hyperparameters such as ``sigma_b``. ``n_items`` and ``n_persons``
            are inferred from ``responses`` automatically.
        item_ids : array-like or None
            Optional stable identifiers for response columns. If cached
            posteriors include identifiers, these must match exactly.
        adaptive_grid : bool
            If ``True`` (default), run a second quadrature pass on a
            per-person grid of ``adaptive_grid_size`` points spanning the
            coarse EAP +/- ``adaptive_span`` coarse SDs (SD floored at half
            the coarse spacing and the span clipped to the coarse bounds), and
            compute all summaries
            from it. Centring and scaling the quadrature on the posterior
            follows the idea of adaptive quadrature (Naylor & Smith, 1982;
            Schilling & Bock, 2005), here with a uniform rather than
            Gauss-Hermite rule. If ``False``, summaries come from the single
            fixed grid on ``theta_prior`` +/- 6 prior scales.
        adaptive_span : float
            Half-width of the second-pass grid in coarse SDs. Default 4.
        adaptive_grid_size : int
            Number of points of each second-pass grid. Default 21. The
            span and size defaults were chosen by a sweep as the cheapest
            setting that
            matched 6 SD x 31 points on calibration and on deviation from a
            4001-point fixed grid, for tests of 5 to 100 items. Persons whose
            second-pass spacing exceeds 0.5 pass-2 SD (long tests, posterior
            SD well below the coarse spacing) are re-centred on the pass-2
            EAP/SD and re-run, up to 6 times; this keeps interval bounds
            within about 0.025 SD of a 20001-point fixed grid at 50 to 3000
            items.
            Because pass 2 renormalizes each item draw inside the window, a
            containment guard checks on the coarse grid that no single draw
            puts more than 1e-3 of its posterior mass outside a person's
            window; if one does (widely separated item draws), that person's
            window is widened to the union of the draws' own windows (mean
            +/- ``adaptive_span`` SD per draw) and is not refined further.
        device : {None, 'auto', 'cpu', 'cuda'}
            ``None`` (default) uses the NumPy path. ``'cpu'`` or ``'cuda'``
            computes the draw-averaged grid posterior (both passes) with
            PyTorch in float64 on that device, chunked over item draws and
            persons to bound memory; the summaries (mean, SD, MAP, median,
            ETI, HDI, plausible values) are then computed from those masses
            by the same NumPy code, so they are identical in definition and
            agree with the NumPy path to floating-point rounding. ``'auto'``
            uses CUDA when available and the NumPy path otherwise, so one
            call works on machines with and without a GPU. Only the built-in
            1PL, 2PL, PCM and GPCM likelihoods are supported; custom
            likelihoods raise ``NotImplementedError``. ``n_cores`` is ignored
            on the torch path. PyTorch is imported lazily and is optional.
        torch_dtype : {'float64', 'float32'}
            Precision of the torch path (default ``'float64'``). ``'float32'``
            is several times faster on GPUs with reduced float64 throughput;
            mean and SD then agree with the NumPy path to about 1e-6 instead
            of rounding. Ignored on the NumPy path.
        return_posterior : str, collection of str, or None
            Extra posterior representations to attach to the result. ``None``
            (default) attaches nothing. Allowed values: ``{"grid",
            "draws"}``, alone or together. A single string or a
            tuple/list/set of strings is accepted. ``"grid"``
            fills ``PersonScore.theta_grid`` and
            ``PersonScore.theta_posterior_mass`` (both ``(P', Q)``) with the
            final per-person grid and the normalized draw-averaged masses
            from which all summaries are computed. ``"draws"`` runs one extra
            pass over the final grids and fills ``PersonScore.theta_draws``
            ``(S, P')`` with one theta draw per item posterior draw and
            person from ``p(theta | y_p, items_s)`` (inverse cell-uniform CDF
            sampling), plus ``item_draws``; see
            :class:`PersonScore` for truncation and memory notes (the
            output is ``S * P'`` float64). Both fill ``theta_window_uncontained``. Any
            other value raises ``ValueError``.
        draw_moments : bool
            If ``True`` (requires ``"draws"``), also return the per-draw
            conditional mean and SD on the grid as ``theta_draw_mean`` and
            ``theta_draw_sd`` (both ``(S, P')``).
        seed : int or None
            Seed of the uniform variates used for ``"draws"`` (a
            ``(S, P')`` array drawn on the host with
            ``numpy.random.default_rng(seed)``, so results do not depend on
            ``device``, ``n_cores`` or chunking). Only allowed with
            ``"draws"``; ``None`` uses fresh entropy. It does not seed the
            item draws: with an approximator they come from its stochastic
            ``sample()``, so ``theta_draws`` are reproducible only given the
            same item draws (cached ``item_posteriors``, or the same scorer
            whose draws are already cached).
        intervals : str, collection of str, or None
            Which credible intervals to compute: ``"eti"``, ``"hdi"``, both
            (default) or none (``None`` or an empty collection). A skipped
            interval's bounds are ``None`` in the result and its levels
            empty; its level argument is still validated. Skipping the HDI
            saves its per-person search when only ETIs (or none) are used.

        Returns
        -------
        PersonScore
            Structured output with point estimates, uncertainty measures,
            and optional theta plausible values.

        Notes
        -----
        The theta posterior is evaluated on a grid (Bock & Mislevy, 1982) as
        the average over item draws of each draw's normalized conditional
        posterior. Median, ETI, HDI and plausible values treat each grid mass
        as uniform on its cell and invert the resulting piecewise-linear CDF
        (Devroye, 1986). Taking quantiles from the CDF *at* the grid points
        instead would bias them low by up to half a grid cell; with the
        default 31-point grid on [-6, 6] (spacing 0.4) that is
        material when the posterior SD (about 0.3 at 50 items) is below the
        spacing.

        Raises
        ------
        ValueError
            If response dimensions don't match cached posteriors, or if
            ``return_posterior`` contains a value other than ``"grid"`` or
            ``"draws"``, or if ``seed`` or ``draw_moments`` is set without
            ``"draws"``.
        """
        requested = _validate_return_posterior(return_posterior)
        intervals = _validate_intervals(intervals)
        seed = _validate_seed(seed, requested)
        if draw_moments and "draws" not in requested:
            raise ValueError(
                "draw_moments=True requires return_posterior to include 'draws'."
            )
        _validate_probability_levels("eti_level", eti_level)
        if hdi_level is None:
            hdi_level = eti_level
        _validate_probability_levels("hdi_level", hdi_level)

        grid_size = _validate_positive_int("theta_grid_size", theta_grid_size)
        if not np.isfinite(adaptive_span) or adaptive_span <= 0:
            raise ValueError(f"adaptive_span must be positive, got {adaptive_span!r}.")
        pass2_size = _validate_positive_int("adaptive_grid_size", adaptive_grid_size)
        if pass2_size < 3:
            raise ValueError(
                f"adaptive_grid_size must be at least 3, got {pass2_size}."
            )
        if grid_size < 3:
            raise ValueError(f"theta_grid_size must be at least 3, got {grid_size}.")

        if n_plausible_values is not None:
            n_plausible_values = _validate_positive_int(
                "n_plausible_values",
                n_plausible_values,
            )
        n_cores = _validate_n_cores(n_cores)
        torch_device = _resolve_device(device)
        if torch_dtype not in ("float64", "float32"):
            raise ValueError(
                f"torch_dtype must be 'float64' or 'float32'; got {torch_dtype!r}."
            )

        # Validate responses
        responses = np.asarray(responses, dtype=np.float64)
        if responses.ndim != 2:
            raise ValueError(
                f"responses must be 2-D (P', I), got shape {responses.shape}"
            )

        # NaN marks missing/not-administered responses and is ignored in the
        # person likelihood. Infinite values are always invalid.
        _validate_responses_for_model(responses, self.model_type, allow_nan=True)

        n_persons, n_items_response = responses.shape
        if n_persons < 1 or n_items_response < 1:
            raise ValueError(
                "responses must have at least one person and one item, "
                f"got {responses.shape}."
            )

        # Check or draw item posteriors
        if self._item_posteriors is None:
            # Draw posteriors from approximator
            self._item_posteriors = self._draw_item_posteriors(
                responses,
                n_item_samples,
                item_conditions=item_conditions,
                item_ids=item_ids,
            )

            # Extract metadata
            n_samples, n_items = self._validate_or_redetect_posteriors(
                self._item_posteriors,
            )
            if n_items != n_items_response:
                raise ValueError(
                    "Approximator returned posteriors for "
                    f"{n_items} items, but responses have {n_items_response} items."
                )
            self._n_samples = n_samples
            self._n_items = n_items
            self._item_ids = _posterior_item_ids(
                self._item_posteriors,
                n_items,
                explicit_item_ids=item_ids,
            )
            if self._item_ids is not None:
                self._item_posteriors.pop("item_ids", None)
        else:
            # Validate item count matches cached posteriors
            if n_items_response != self._n_items:
                raise ValueError(
                    f"Response data has {n_items_response} items, but cached "
                    f"posteriors have {self._n_items} items. PersonScorer is "
                    "item-specific — you cannot score responses to different items."
                )
            if self._item_ids is not None:
                score_item_ids = _validate_item_ids(item_ids, n_items_response)
                if score_item_ids is None:
                    raise ValueError(
                        "Cached posteriors include item_ids. Pass matching "
                        "item_ids to score() so item order can be verified."
                    )
                if not np.array_equal(score_item_ids, self._item_ids):
                    raise ValueError(
                        "Response item_ids do not match cached posterior item_ids. "
                        "PersonScorer is item-order specific."
                    )
            elif item_ids is not None:
                raise ValueError(
                    "Cached posteriors were created without item_ids, so the "
                    "provided item_ids cannot be verified. Recreate the scorer "
                    "with item_ids in the constructor or redraw posteriors with "
                    "draw_item_posteriors(..., item_ids=...)."
                )
            n_samples = self._n_samples

        if n_samples is None or n_samples < 1:
            raise ValueError("Item posteriors must contain at least one sample.")

        # Checked here, after drawing and model-type redetection, so that the
        # error reflects the model type that will actually be scored.
        if torch_device is not None and (
            self.custom_log_likelihood is not None
            or self.model_type not in ("1PL", "2PL", "PCM", "GPCM")
        ):
            raise NotImplementedError(
                "score(device=...) supports only the built-in 1PL, 2PL, PCM and "
                "GPCM likelihoods; use device=None for "
                f"model_type={self.model_type!r} or custom likelihoods."
            )

        # Setup quadrature grid
        theta_grid, log_prior_weights = self._get_quadrature_grid(
            grid_size,
        )

        # Get log-likelihood function
        log_likelihood_fn = self._get_log_likelihood_fn()

        # Build item parameter samples dict based on model type
        param_keys = _posterior_parameter_keys(
            self._item_posteriors,
            self.model_type,
        )

        # Extract parameter samples for the model type
        param_samples = {key: self._item_posteriors[key] for key in param_keys}

        # Validate polytomous upper category bound once delta is available.
        if self.model_type in ("PCM", "GPCM"):
            max_category = int(param_samples["delta"].shape[-1])
            _validate_responses_for_model(
                responses,
                self.model_type,
                max_category=max_category,
                allow_nan=True,
            )

        # Check for edge cases (all-correct or all-incorrect)
        observed = ~np.isnan(responses)
        has_observed = np.any(observed, axis=1)
        edge_cases = None
        edge_label = ""
        if self.model_type in ("1PL", "2PL"):
            all_high = has_observed & np.all((responses == 1) | ~observed, axis=1)
            all_low = has_observed & np.all((responses == 0) | ~observed, axis=1)
            edge_label = "all-correct or all-incorrect"
            edge_cases = all_high | all_low
        elif self.model_type in ("PCM", "GPCM"):
            max_category = int(param_samples["delta"].shape[-1])
            all_high = has_observed & np.all(
                (responses == max_category) | ~observed,
                axis=1,
            )
            all_low = has_observed & np.all((responses == 0) | ~observed, axis=1)
            edge_label = "all-highest-category or all-lowest-category"
            edge_cases = all_high | all_low

        if edge_cases is not None and np.any(edge_cases):
            warnings.warn(
                f"{np.sum(edge_cases)} response pattern(s) are {edge_label}. "
                "Posterior summaries are bounded by quadrature grid "
                f"bounds [{theta_grid[0]:.2f}, {theta_grid[-1]:.2f}].",
                UserWarning,
                stacklevel=2,
            )

        # Mixture over item posterior draws. For each draw, compute the
        # conditional theta posterior on the grid and average the normalized
        # posterior masses. This preserves within-person theta uncertainty.
        use_binary_chunk = self.custom_log_likelihood is None and self.model_type in (
            "1PL",
            "2PL",
        )
        chunk_size = (
            _BINARY_POSTERIOR_CHUNK_SIZE
            if use_binary_chunk or torch_device is not None
            else _GENERAL_POSTERIOR_CHUNK_SIZE
        )

        chunk_bounds = [
            (chunk_start, min(chunk_start + chunk_size, n_samples))
            for chunk_start in range(0, n_samples, chunk_size)
        ]
        custom_log_likelihood = self.custom_log_likelihood is not None

        def mixture_on_grid(
            grid: np.ndarray,
            grid_log_prior_weights: np.ndarray,
            resp: np.ndarray | None = None,
            window: tuple | None = None,
            on_post: Callable | None = None,
        ):
            """Draw-averaged posterior masses on ``grid`` ``(Q,)`` or ``(P, Q)``.

            ``on_post(draw_start, draw_stop, post, person_slice)`` receives
            each normalized per-draw block ``(n_draws, n_persons_block, Q)``
            (see :func:`_mixture_weights_torch`); it must be
            thread-safe on the NumPy path (``n_cores > 1``).

            ``resp`` restricts scoring to a subset of persons (rows of
            ``responses``); ``grid`` must then have one row per such person.
            ``window = (lower, upper, span, sd_floor)`` (shared grid only)
            additionally returns ``(weights, (outside, lo, hi))``, see
            :func:`_draw_window_stats`.
            """
            resp = responses if resp is None else resp
            if torch_device is not None:
                return _mixture_weights_torch(
                    grid,
                    grid_log_prior_weights,
                    resp,
                    param_samples,
                    self.model_type,
                    device=torch_device,
                    draw_chunk_size=chunk_size,
                    window=window,
                    on_post=on_post,
                    dtype_name=torch_dtype,
                )

            def process_chunk(bounds: tuple[int, int]):
                chunk_start, chunk_stop = bounds
                log_likelihood_chunk = _compute_log_likelihood_chunk(
                    chunk_start,
                    chunk_stop,
                    theta_grid=grid,
                    responses=resp,
                    param_samples=param_samples,
                    param_keys=param_keys,
                    model_type=self.model_type,
                    log_likelihood_fn=log_likelihood_fn,
                    custom_log_likelihood=custom_log_likelihood,
                )
                # (Q,) or (P, Q) prior weights broadcast against (S, P, Q).
                log_mass = log_likelihood_chunk + grid_log_prior_weights
                log_norm = logsumexp(log_mass, axis=-1, keepdims=True)
                post = np.exp(log_mass - log_norm)
                weights = post.sum(axis=0) / n_samples
                if on_post is not None:
                    on_post(chunk_start, chunk_stop, post, slice(None))
                if window is None:
                    return weights
                return weights, _draw_window_stats(post, grid, *window)

            if n_cores == 1 or len(chunk_bounds) == 1:
                chunk_results = [process_chunk(bounds) for bounds in chunk_bounds]
            else:
                max_workers = min(n_cores, len(chunk_bounds))
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    chunk_results = list(executor.map(process_chunk, chunk_bounds))
            if window is None:
                return np.sum(chunk_results, axis=0)
            return (
                np.sum([r[0] for r in chunk_results], axis=0),
                _combine_window_stats([r[1] for r in chunk_results]),
            )

        mixture_weights = mixture_on_grid(theta_grid, log_prior_weights)
        uncontained = np.zeros(n_persons, dtype=bool)

        if adaptive_grid:
            # Pass 2: re-evaluate on a per-person grid centred on the coarse
            # EAP and spanning +/- k coarse SDs (clipped to the coarse bounds).
            lo_bound, hi_bound = float(theta_grid[0]), float(theta_grid[-1])
            spacing = float(theta_grid[1] - theta_grid[0])
            mean_c, sd_c = _grid_mean_sd(theta_grid, mixture_weights)
            half = adaptive_span * np.maximum(
                sd_c, _ADAPTIVE_GRID_MIN_SD_SPACINGS * spacing
            )

            def person_grids(mean, half_width):
                width = np.minimum(2.0 * half_width, hi_bound - lo_bound)
                lower = np.clip(mean - 0.5 * width, lo_bound, hi_bound - width)
                return _uniform_person_grids(
                    lower,
                    lower + width,
                    pass2_size,
                    prior_mean=self.theta_prior_mean,
                    prior_scale=self.theta_prior_scale,
                )

            # Containment guard: a draw whose posterior lies partly outside
            # the window would be renormalized onto the window edge. Check
            # the (cheap) coarse grid per draw and, where needed, widen the
            # window to the union of the draws' own windows.
            coarse_lo = np.clip(mean_c - half, lo_bound, hi_bound)
            coarse_hi = np.clip(mean_c + half, lo_bound, hi_bound)
            _, (outside, union_lo, union_hi) = mixture_on_grid(
                theta_grid,
                log_prior_weights,
                window=(
                    coarse_lo,
                    coarse_hi,
                    adaptive_span,
                    _ADAPTIVE_GRID_MIN_SD_SPACINGS * spacing,
                ),
            )
            uncontained = outside > _ADAPTIVE_GRID_MAX_OUTSIDE_MASS
            if np.any(uncontained):
                lo_w = np.where(uncontained, np.minimum(mean_c - half, union_lo), 0.0)
                hi_w = np.where(uncontained, np.maximum(mean_c + half, union_hi), 0.0)
                half = np.where(
                    uncontained, np.maximum(half, 0.5 * (hi_w - lo_w)), half
                )
                mean_c = np.where(uncontained, 0.5 * (lo_w + hi_w), mean_c)

            theta_grid, log_prior_weights = person_grids(mean_c, half)
            mixture_weights = mixture_on_grid(theta_grid, log_prior_weights)

            # The coarse SD (floored) can leave the window far wider than the
            # posterior on long tests, so the pass-2 spacing no longer
            # resolves it. Re-centre on the pass-2 EAP/SD and re-run for those
            # persons only, until the spacing is small relative to the SD.
            for _ in range(_ADAPTIVE_GRID_MAX_REFINEMENTS):
                mean_p, sd_p = _grid_mean_sd(theta_grid, mixture_weights)
                grid_spacing = theta_grid[:, 1] - theta_grid[:, 0]
                todo = np.flatnonzero(
                    (grid_spacing > _ADAPTIVE_GRID_MAX_SPACING_SD * sd_p) & ~uncontained
                )
                if todo.size == 0:
                    break
                new_half = np.maximum(
                    adaptive_span * sd_p[todo],
                    _ADAPTIVE_GRID_MIN_HALF_WIDTH_SPACINGS * grid_spacing[todo],
                )
                new_grid, new_log_w = person_grids(mean_p[todo], new_half)
                new_weights = mixture_on_grid(new_grid, new_log_w, responses[todo])
                theta_grid[todo] = new_grid
                mixture_weights[todo] = new_weights
                # Keep the prior weights matched to the final grid (used by
                # the "draws" pass below).
                log_prior_weights[todo] = new_log_w

        theta_draws = theta_draw_mean = theta_draw_sd = None
        if "draws" in requested:
            n_grid = theta_grid.shape[-1]
            grid_pg = np.broadcast_to(theta_grid, (n_persons, n_grid))
            edges_pg = np.broadcast_to(_cell_edges(theta_grid), (n_persons, n_grid + 1))
            # One (S, P') uniform array drawn on the host, so the result does
            # not depend on device, n_cores or draw chunking.
            u_all = np.random.default_rng(seed).random((n_samples, n_persons))
            theta_draws = np.empty((n_samples, n_persons), dtype=np.float64)
            if draw_moments:
                theta_draw_mean = np.empty_like(theta_draws)
                theta_draw_sd = np.empty_like(theta_draws)

            def sample_block(s0: int, s1: int, post: np.ndarray, ps: slice) -> None:
                theta_draws[s0:s1, ps] = _invert_cell_uniform(
                    post, edges_pg[ps], u_all[s0:s1, ps]
                )
                if draw_moments:
                    g = grid_pg[ps]
                    m = np.einsum("spq,pq->sp", post, g)
                    # centred, as in the marginal SD (raw moments cancel badly
                    # when |theta| >> sd)
                    v = np.sum(post * (g[None] - m[..., None]) ** 2, axis=-1)
                    theta_draw_mean[s0:s1, ps] = m
                    theta_draw_sd[s0:s1, ps] = np.sqrt(v)

            mixture_on_grid(theta_grid, log_prior_weights, on_post=sample_block)

        aggregated = _aggregate_grid_posterior(
            theta_grid,
            mixture_weights,
            None,
            eti_level=eti_level if "eti" in intervals else None,
            hdi_level=hdi_level if "hdi" in intervals else None,
            n_plausible_values=n_plausible_values,
        )

        out_grid = None
        out_mass = None
        if "grid" in requested:
            out_grid = np.array(
                np.broadcast_to(theta_grid, mixture_weights.shape),
                dtype=np.float64,
            )
            out_mass = np.asarray(mixture_weights, dtype=np.float64)
            out_mass = out_mass / out_mass.sum(axis=-1, keepdims=True)

        return PersonScore(
            mean=aggregated["mean"],
            map=aggregated["map"],
            median=aggregated["median"],
            sd=aggregated["sd"],
            eti_lower=aggregated["eti_lower"],
            eti_upper=aggregated["eti_upper"],
            hdi_lower=aggregated["hdi_lower"],
            hdi_upper=aggregated["hdi_upper"],
            plausible_values=aggregated["plausible_values"],
            n_items=self._n_items,
            eti_levels=aggregated["eti_levels"],
            hdi_levels=aggregated["hdi_levels"],
            theta_grid=out_grid,
            theta_posterior_mass=out_mass,
            theta_window_uncontained=(uncontained.copy() if requested else None),
            theta_draws=theta_draws,
            item_draws=(
                dict(self._item_posteriors) if theta_draws is not None else None
            ),
            theta_draw_mean=theta_draw_mean,
            theta_draw_sd=theta_draw_sd,
        )
