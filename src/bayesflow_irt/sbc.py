"""Simulation-based calibration helpers for pooled condition diagnostics.

This module provides reusable utilities for SBC workflows where conditions
(``n_items``, ``n_persons``) are sampled from configured ranges, one posterior
is computed per simulated dataset, and rank statistics are pooled across all
sampled conditions.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _edge_corners_from_simulator(simulator: Any) -> list[tuple[int, int]]:
    """Infer unique edge-corner ``(n_items, n_persons)`` pairs from a simulator."""
    try:
        items_lo, items_hi = int(simulator.n_items[0]), int(simulator.n_items[1])
        persons_lo, persons_hi = (
            int(simulator.n_persons[0]),
            int(simulator.n_persons[1]),
        )
    except Exception as exc:
        raise TypeError("simulator must expose n_items and n_persons ranges.") from exc

    corners: list[tuple[int, int]] = []
    for corner in (
        (items_lo, persons_lo),
        (items_lo, persons_hi),
        (items_hi, persons_lo),
        (items_hi, persons_hi),
    ):
        if corner not in corners:
            corners.append(corner)
    return corners


def _collect_edge_corner_batches(
    simulator: Any,
    approximator: Any,
    *,
    corners: Sequence[tuple[int, int]] | None,
    samples_per_condition: int,
    n_posterior_draws: int,
    sample_chunk_size: int | None,
    clear_cuda_cache: bool,
    flow_sample_kwargs: Mapping[str, Any] | None,
) -> tuple[
    list[tuple[int, int]],
    list[dict[str, np.ndarray]],
    list[dict[str, np.ndarray]],
]:
    """Sample conditions and posterior draws for each requested edge corner."""
    if not hasattr(simulator, "clone_with_sizes"):
        raise TypeError("simulator must provide clone_with_sizes().")

    if corners is None:
        corner_list = _edge_corners_from_simulator(simulator)
    else:
        corner_list = [(int(n_items), int(n_persons)) for n_items, n_persons in corners]
        if not corner_list:
            raise ValueError("corners must be non-empty when provided.")

    condition_batches: list[dict[str, np.ndarray]] = []
    posterior_batches: list[dict[str, np.ndarray]] = []
    for n_items_case, n_persons_case in corner_list:
        sim_case = simulator.clone_with_sizes(
            n_items=(n_items_case, n_items_case),
            n_persons=(n_persons_case, n_persons_case),
        )
        conditions = sim_case.sample((samples_per_condition,))
        if sample_chunk_size is None:
            post = approximator.sample(
                num_samples=n_posterior_draws,
                conditions=conditions,
                flow_sample_kwargs=flow_sample_kwargs,
            )
            posterior = {k: np.asarray(v) for k, v in post.items()}
        else:
            posterior = chunked_posterior_sample(
                approximator,
                conditions,
                num_samples=n_posterior_draws,
                chunk_size=sample_chunk_size,
                clear_cuda_cache=clear_cuda_cache,
                flow_sample_kwargs=flow_sample_kwargs,
            )

        condition_batches.append({k: np.asarray(v) for k, v in conditions.items()})
        posterior_batches.append(posterior)

    return corner_list, condition_batches, posterior_batches


def sample_edge_corner_batches(
    simulator: Any,
    approximator: Any,
    *,
    samples_per_condition: int = 20,
    n_posterior_draws: int = 500,
    sample_chunk_size: int | None = None,
    clear_cuda_cache: bool = False,
    corners: Sequence[tuple[int, int]] | None = None,
    flow_sample_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Sample and cache edge-corner condition/posterior batches for reuse.

    This helper is intended for notebook and script workflows which need to
    render diagnostics for several parameters on the same four edge corners.
    """
    if samples_per_condition < 1:
        raise ValueError("samples_per_condition must be >= 1.")
    if n_posterior_draws < 1:
        raise ValueError("n_posterior_draws must be >= 1.")
    if sample_chunk_size is not None and sample_chunk_size < 1:
        raise ValueError("sample_chunk_size must be >= 1 when provided.")

    corner_list, condition_batches, posterior_batches = _collect_edge_corner_batches(
        simulator,
        approximator,
        corners=corners,
        samples_per_condition=samples_per_condition,
        n_posterior_draws=n_posterior_draws,
        sample_chunk_size=sample_chunk_size,
        clear_cuda_cache=clear_cuda_cache,
        flow_sample_kwargs=flow_sample_kwargs,
    )
    return {
        "corners": corner_list,
        "conditions": condition_batches,
        "posteriors": posterior_batches,
        "samples_per_condition": int(samples_per_condition),
        "n_posterior_draws": int(n_posterior_draws),
    }


def _flatten_parameter_draws(
    true_vals: np.ndarray,
    post_vals: np.ndarray,
    *,
    param_key: str,
    n_posterior_draws: int,
) -> dict[str, np.ndarray]:
    """Validate and flatten one parameter's true values and posterior draws."""
    if post_vals.ndim != true_vals.ndim + 1:
        raise ValueError(
            f"Posterior shape for {param_key!r} must insert sample axis at 1: "
            f"true={true_vals.shape}, posterior={post_vals.shape}."
        )
    if post_vals.shape[0] != true_vals.shape[0]:
        raise ValueError(
            f"Posterior/true batch mismatch for {param_key!r}: "
            f"{post_vals.shape[0]} vs {true_vals.shape[0]}."
        )
    if post_vals.shape[1] != n_posterior_draws:
        raise ValueError(
            f"Posterior sample axis mismatch for {param_key!r}: "
            f"{post_vals.shape[1]} vs {n_posterior_draws}."
        )
    if post_vals.shape[2:] != true_vals.shape[1:]:
        raise ValueError(
            f"Posterior trailing dims mismatch for {param_key!r}: "
            f"{post_vals.shape[2:]} vs {true_vals.shape[1:]}."
        )

    true_flat = true_vals.reshape(-1)
    post_flat = np.moveaxis(post_vals, 1, -1).reshape(-1, n_posterior_draws)
    point_estimate = post_flat.mean(axis=1)
    ranks = np.sum(post_vals < np.expand_dims(true_vals, axis=1), axis=1).reshape(-1)
    return {
        "true_flat": true_flat,
        "posterior_flat": post_flat,
        "point_estimate": point_estimate,
        "ranks": ranks,
    }


def _subsample_flattened_parameter_draws(
    flattened: Mapping[str, np.ndarray],
    *,
    target_count: int | None,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Optionally subsample flattened parameter draws to a common count."""
    out = {key: np.array(val, copy=True) for key, val in flattened.items()}
    n_obs = out["true_flat"].shape[0]
    if target_count is None or n_obs <= target_count:
        return out

    keep_idx = np.sort(rng.choice(n_obs, size=target_count, replace=False))
    out["true_flat"] = out["true_flat"][keep_idx]
    out["posterior_flat"] = out["posterior_flat"][keep_idx]
    out["point_estimate"] = out["point_estimate"][keep_idx]
    out["ranks"] = out["ranks"][keep_idx]
    return out


def _compute_recovery_metrics(
    true_flat: np.ndarray,
    point_estimate: np.ndarray,
) -> dict[str, float]:
    """Compute correlation, RMSE, normalized RMSE, and a scale diagnostic.

    ``nrmse_ideal`` is the NRMSE that an exact posterior mean would achieve at
    the observed correlation, and ``nrmse_excess`` is how far the estimator
    falls short of it. Both follow from the standard decomposition

    .. code-block:: text

        MSE = bias^2 + (sd(m) - sd(t))^2 + 2 sd(m) sd(t) (1 - r)

    for an estimator ``m`` of a truth ``t`` with correlation ``r``. The exact
    posterior mean is unbiased and shrunk by exactly the correlation factor,
    ``sd(m) = r sd(t)`` -- which also makes it the correlation-maximizing
    estimator, since for any ``g`` the tower rule gives ``Cov(g, t) =
    Cov(g, m)`` and Cauchy-Schwarz then bounds ``r`` by ``sd(m) / sd(t)``.
    Substituting collapses the decomposition to

    .. code-block:: text

        NRMSE = sqrt(1 - r^2)

    so at the ideal the two metrics are redundant and carry no independent
    information. A positive ``nrmse_excess`` therefore measures mis-scaling
    rather than mis-ranking: the estimator orders items as well as its
    correlation implies but spreads them wrongly, most often under-shrinking
    (``sd(m) > r sd(t)``, an over-confident posterior). This separates "wrong
    ordering" from "right ordering, wrong spread", which correlation and NRMSE
    cannot distinguish on their own.
    """
    rmse = float(np.sqrt(np.mean((point_estimate - true_flat) ** 2)))
    scale = float(np.std(true_flat))
    nrmse = float(rmse / scale) if scale > 0.0 else float("nan")
    if true_flat.size > 1 and scale > 0.0 and np.std(point_estimate) > 0.0:
        corr = float(np.corrcoef(true_flat, point_estimate)[0, 1])
    else:
        corr = float("nan")
    if np.isfinite(corr) and abs(corr) <= 1.0:
        nrmse_ideal = float(np.sqrt(1.0 - corr * corr))
        nrmse_excess = (
            float(nrmse - nrmse_ideal) if np.isfinite(nrmse) else float("nan")
        )
    else:
        nrmse_ideal = float("nan")
        nrmse_excess = float("nan")
    return {
        "corr": corr,
        "rmse": rmse,
        "nrmse": nrmse,
        "nrmse_ideal": nrmse_ideal,
        "nrmse_excess": nrmse_excess,
    }


def compute_calibration_error(
    posterior_flat: np.ndarray,
    true_flat: np.ndarray,
    *,
    resolution: int = 20,
    min_level: float = 0.005,
    max_level: float = 0.995,
    aggregation: str = "median",
) -> dict[str, float]:
    """Compute the marginal calibration error for flattened posterior draws.

    Follows the definition implemented by
    ``bayesflow.diagnostics.metrics.calibration_error``: for ``resolution``
    central credibility levels ``alpha`` evenly spaced in
    ``[min_level, max_level]``, the empirical coverage is the fraction of
    observations whose true value falls inside the central posterior interval
    bounded by the ``(1 - alpha) / 2`` and ``(1 + alpha) / 2`` quantiles of
    the draws; the calibration error is the median over levels of
    ``|empirical - alpha|``. Values near 0 indicate well-calibrated
    posteriors. This is the scalar summary of the coverage-profile
    diagnostic rendered by :func:`plot_edge_parameter_diagnostics`, rooted
    in the simulation-based calibration methodology of Talts et al. (2018;
    see ``references.md``).

    Because quantiles are equivariant under strictly monotone transforms,
    the result is identical for raw and log-transformed draws, so no
    ``log_scale`` handling is needed.

    ``aggregation="mean"`` averages the per-level deviations instead, as
    ``bayesflow-hpo``'s ``mean_calibration_error``: the median responds only
    to the middle of the coverage curve, so a posterior miscalibrated in the
    tails still scores well. The mean is that
    package's design choice, not a published standard; the median is the
    default.

    Parameters
    ----------
    posterior_flat : np.ndarray
        Posterior draws with shape ``(n_observations, n_draws)``.
    true_flat : np.ndarray
        True parameter values with shape ``(n_observations,)``.
    resolution : int
        Number of central credibility levels to evaluate.
    min_level, max_level : float
        Smallest and largest central credibility level in ``(0, 1)``.
    aggregation : {"median", "mean"}
        How the per-level absolute deviations are combined.

    Returns
    -------
    dict
        ``{"calibration_error": float}`` for the median, or
        ``{"mean_calibration_error": float}`` for the mean, so the two are
        never confused in a stored record.
    """
    posterior_flat = np.asarray(posterior_flat)
    true_flat = np.asarray(true_flat)
    if posterior_flat.ndim != 2:
        raise ValueError("posterior_flat must be 2D (n_observations, n_draws).")
    if true_flat.shape != (posterior_flat.shape[0],):
        raise ValueError(
            "true_flat must be 1D with one entry per posterior_flat row: "
            f"{true_flat.shape} vs {posterior_flat.shape}."
        )
    if resolution < 1:
        raise ValueError("resolution must be >= 1.")
    if not 0.0 < min_level <= max_level < 1.0:
        raise ValueError("Require 0 < min_level <= max_level < 1.")
    if aggregation not in ("median", "mean"):
        raise ValueError(
            f"aggregation must be 'median' or 'mean'; got {aggregation!r}."
        )

    alphas = np.linspace(min_level, max_level, num=resolution)
    lowers = (1.0 - alphas) / 2.0
    uppers = 1.0 - lowers
    # Shape: (resolution, n_observations) each.
    lower_bounds = np.quantile(posterior_flat, lowers, axis=1)
    upper_bounds = np.quantile(posterior_flat, uppers, axis=1)
    inlier = (lower_bounds <= true_flat[None, :]) & (upper_bounds >= true_flat[None, :])
    empirical = inlier.mean(axis=1)
    deviation = np.abs(empirical - alphas)
    if aggregation == "mean":
        return {"mean_calibration_error": float(np.mean(deviation))}
    return {"calibration_error": float(np.median(deviation))}


# Element budget for one chunk of the TARP distance loop. Module-level so
# tests can shrink it and actually exercise the multi-chunk path: at the
# default value every realistic call fits in a single chunk, which is exactly
# how a chunking bug would go unnoticed.
_TARP_CHUNK_ELEMENTS = 2e7

#: Mixed into the reference-point seed so that passing the data's own seed
#: is safe. Any fixed value would have worked; this one is arbitrary. It
#: must not change once results are committed, because it fixes the draw --
#: ``test_tarp_reference_draw_is_pinned_to_the_salt`` enforces that rather
#: than leaving it to this comment.
_TARP_REFERENCE_SALT = 0x7A297


def compute_tarp_coverage(
    posterior_draws: np.ndarray,
    true_values: np.ndarray,
    *,
    reference_points: np.ndarray | None = None,
    resolution: int = 20,
    metric: str = "euclidean",
    standardize: bool = True,
    seed: int | None = None,
) -> dict[str, Any]:
    """Expected coverage of TARP regions, a *joint* accuracy test.

    Implements Algorithm 2 of Lemos, Coogan, Hezaveh and Perreault-Levasseur
    (2023), "Sampling-Based Accuracy Testing of Posterior Estimators for
    General Inference" (see ``references.md``). For each simulation ``i`` with
    true parameters ``theta*_i`` and posterior draws ``theta_ij``, a reference
    point ``theta_r`` is drawn and

    ``f_i = (1/n) * sum_j 1[d(theta_ij, theta_r) < d(theta*_i, theta_r)]``

    is the fraction of draws lying closer to the reference point than the
    truth does. The expected coverage probability at credibility level
    ``1 - alpha`` is ``mean_i 1[f_i < 1 - alpha]`` -- the ECDF of the ``f_i``.
    For an exact posterior the ``f_i`` are uniform, so the curve is the
    diagonal.

    **Why this and not marginal coverage.** TARP regions are balls centred on
    a freely chosen reference point, which makes the region generator
    *positionable*. Lemos et al.'s Theorem 3 shows that correct expected
    coverage over every position function implies the estimator equals the
    true posterior -- a conclusion HPD coverage cannot reach, because HPD
    regions are not positionable. Concretely, an estimator that ignores the
    data entirely (``p_hat(theta|x) = p(theta)``) has *perfect* HPD coverage
    at every level, and marginal per-parameter checks share the blind spot
    (Modrak et al. 2025). TARP also tests the joint rather than the
    marginals, so it can see wrong correlation structure.

    **The reference distribution is not a free choice.** Section 4.3 of the
    paper is explicit that TARP with an ``x``-*independent* reference point is
    **also blind** to the ``p_hat(theta|x) = p(theta)`` case: the authors
    compute that estimator's coverage three ways and state they "expect the
    first two methods" -- HPD, and TARP with ``theta_r ~ U(0, 1)`` -- "to have
    ECP equal to ``1 - alpha``, but not for the third", the third being a
    reference point that depends on the data. So a TARP run with the default
    random reference points returns a *clean* result for exactly this
    failure. Pass
    ``reference_points`` computed from the data to get the version with teeth.
    Running both and comparing is what makes the diagnosis; a clean
    random-reference curve on its own must never be reported as an all-clear.

    Parameters
    ----------
    posterior_draws : np.ndarray
        Draws with shape ``(n_simulations, n_draws, n_parameters)``.
    true_values : np.ndarray
        True parameters with shape ``(n_simulations, n_parameters)``.
    reference_points : np.ndarray, optional
        One reference point per simulation, shape
        ``(n_simulations, n_parameters)``, in the same units as
        ``true_values``. When ``None``, points are drawn uniformly over the
        hypercube spanned by the standardized truths, reproducing the paper's
        default -- which carries the blind spot described above.
    resolution : int
        Number of credibility levels at which the curve is evaluated.
    metric : {"euclidean", "manhattan"}
        Distance on parameter space. Section 4.2 reports the method is robust
        to this choice.
    standardize : bool
        Standardize each parameter dimension before computing distances.
        Required whenever dimensions have different scales -- ``log a`` and
        ``b`` do -- because otherwise the balls are dominated by whichever
        parameter has the larger variance and the distance is not meaningful.
        Raises if any dimension's truth is constant, since that cannot be
        standardized; slice such dimensions out yourself rather than letting
        them be silently ignored. Centring and scaling use the **true** values
        only, so the metric does
        not depend on the estimator under test and two checkpoints stay
        comparable. Taking the scale from the draws instead would let a wider
        (worse) posterior score better by shrinking its own distances.
    seed : int, optional
        Seed for the reference-point draw. Ignored when ``reference_points``
        is supplied. Safe to set to the same seed that generated the data:
        the value is mixed with a fixed salt into a ``SeedSequence`` before
        use, so the reference stream is very probably far from the one that
        drew ``true_values`` rather than a replay of it.

        NumPy documents this ``[varying, fixed]`` sequence-of-integer-seeds
        form as carrying "about the same" collision safety as spawning, by
        successions of integer hashes whose avalanche property makes two
        close input seeds "produce initial states that are very far from
        each other (with very high probability)" (NumPy Developers, 2026).
        That is a probabilistic guarantee, not an absolute one, and it is
        the one claimed here.

    Returns
    -------
    dict
        ``credibility_levels``, ``expected_coverage``, the raw ``f_i`` as
        ``coverage_fractions``, ``tarp_error`` (median over levels of
        ``|ECP - level|``, the same *aggregation* as
        :func:`compute_calibration_error` though on a different level grid, so
        the two read side by side but are not the same statistic),
        ``max_deviation`` and ``reference_mode`` (``"provided"`` when
        ``reference_points`` was passed, ``"random"`` when drawn here).

        ``reference_mode`` deliberately does not claim ``"data_dependent"``.
        The function cannot tell how a supplied array was built, and the
        distinction is load-bearing: only an x-dependent reference detects a
        posterior that ignores the data. Record that property alongside the
        result yourself if you need it asserted.

    Notes
    -----
    ``f_i`` is supported on ``{0, 1/n_draws, ..., 1}``, so its ECDF is a
    staircase and ``tarp_error`` cannot reach zero. At ``n_draws = 5`` the
    floor is exactly ``1 / (resolution + 1)`` and dominates everything else;
    by ``n_draws = 100`` it has fallen below the Monte Carlo noise from a few
    thousand simulations, so beyond that point the residual is set by
    ``n_simulations`` rather than by discreteness. Use at least 100 draws. Any
    quoted floor figure is only meaningful alongside the ``n_simulations`` and
    ``resolution`` it was measured at, since those fix the noise it is being
    compared against.

    With ``reference_points=None`` and ``seed=None`` the reference draw is
    non-deterministic, so two identical calls return slightly different
    numbers. Pass ``seed`` for anything quoted in a report or committed.
    """
    # float32 input stays float32 here; each chunk is converted to float64
    # and standardized inside the distance loop, so the full draws array is
    # never copied (a float64 copy plus its standardized copy was ~2.2 GB at
    # 1000 x 2000 x 70).
    posterior_draws = np.asarray(posterior_draws)
    if not np.issubdtype(posterior_draws.dtype, np.floating):
        posterior_draws = posterior_draws.astype(float)
    true_values = np.asarray(true_values, dtype=float)
    if posterior_draws.ndim != 3:
        raise ValueError(
            "posterior_draws must be 3D (n_simulations, n_draws, n_parameters); "
            f"got shape {posterior_draws.shape}."
        )
    n_sims, n_draws, n_params = posterior_draws.shape
    if true_values.shape != (n_sims, n_params):
        raise ValueError(
            "true_values must have shape (n_simulations, n_parameters) matching "
            f"posterior_draws: {true_values.shape} vs {(n_sims, n_params)}."
        )
    if n_sims < 1:
        raise ValueError("posterior_draws must contain at least one simulation.")
    if n_draws < 2:
        raise ValueError("posterior_draws needs at least 2 draws per simulation.")
    if metric not in {"euclidean", "manhattan"}:
        raise ValueError("metric must be 'euclidean' or 'manhattan'.")
    if resolution < 1:
        raise ValueError("resolution must be >= 1.")

    # Non-finite values do not raise here by accident: `d_draws < d_truth` is
    # False for NaN, so NaN draws would be silently counted as "not closer",
    # biasing f_i downward with no error and no warning. A partially diverged
    # checkpoint would then produce a mildly-off but entirely plausible curve
    # instead of announcing that it diverged.
    n_bad_draws = int(np.count_nonzero(~np.isfinite(posterior_draws)))
    n_bad_truth = int(np.count_nonzero(~np.isfinite(true_values)))
    if n_bad_draws or n_bad_truth:
        raise ValueError(
            "compute_tarp_coverage requires finite inputs: "
            f"{n_bad_draws} non-finite posterior draw(s) and {n_bad_truth} "
            "non-finite true value(s). Drop or impute the affected "
            "simulations deliberately rather than letting them bias the "
            "coverage fractions."
        )

    # A parameter whose truth is constant across simulations cannot be
    # standardized, and the caller has to say what it means. Silently
    # dropping such dimensions, on the reasoning that a constant dimension
    # left in raw units would dominate every distance and wreck a calibrated
    # posterior's score, is wrong, and the drop would hide real failures:
    #
    #   * Keeping the dimension costs a *correct* posterior nothing. If the
    #     draws are also constant there, the same term is added to both sides
    #     of ``d(draw, ref) < d(truth, ref)`` -- ``sqrt(S + c^2)`` for
    #     euclidean, ``+|c|`` for manhattan -- so every ``f_i`` is unchanged.
    #     Verified numerically for both metrics and several reference values.
    #   * Dropping it hides an *incorrect* one. A posterior placing sd-50
    #     mass on a point-mass parameter would score as calibrated with the
    #     drop and near the 0.5 ceiling without it -- a true positive, not a
    #     scaling artifact.
    #
    # So there is no upside and a serious downside. Raise, and let the caller
    # slice out dimensions that are fixed settings rather than parameters
    # under test. Only ``standardize=True`` is affected: without
    # standardization nothing is divided by the zero scale, every dimension
    # stays in the same raw units, and the dominance problem cannot arise.
    if standardize:
        centre = true_values.mean(axis=0)
        scale = true_values.std(axis=0)
        if not (np.all(np.isfinite(scale)) and np.all(np.isfinite(centre))):
            bad = [
                int(i)
                for i in np.flatnonzero(~(np.isfinite(scale) & np.isfinite(centre)))
            ]
            raise ValueError(
                f"Parameter dimension(s) {bad} have a non-finite mean or "
                "standard deviation, so they cannot be standardized. "
                "np.std squares before averaging, so magnitudes above about "
                "1e154 overflow to inf; the z-scores would collapse to zero "
                "and the dimension would drop out of every distance without "
                "any sign in the result. Rescale the parameters first."
            )
        constant = np.flatnonzero(scale <= 0)
        if constant.size:
            raise ValueError(
                "Parameter dimension(s) "
                f"{[int(i) for i in constant]} are constant across "
                "simulations and cannot be standardized. Slice them out "
                "if they are fixed settings rather than parameters under "
                "test, or pass standardize=False if every dimension is "
                "already on a common scale. They are not dropped "
                "automatically: a posterior with spread on a point-mass "
                "parameter is genuinely miscalibrated, and dropping the "
                "dimension would report it as clean."
            )
    else:
        centre = np.zeros(n_params)
        scale = np.ones(n_params)

    truth_z = (true_values - centre) / scale
    n_kept = n_params

    if reference_points is None:
        # Domain-separated, NOT ``default_rng(seed)``. The docstring tells the
        # caller to pass a seed for anything committed, and the seed nearest
        # to hand is the one that generated the data. That is exactly the seed
        # that must not be used here: a fresh ``default_rng(s)`` replays the
        # stream that produced ``true_values``, and where the two consumptions
        # align -- truths drawn by ``uniform`` of this shape -- the references
        # land on the truths. Then ``dist_truth`` is zero, no draw is strictly
        # closer, every ``f_i`` is zero and ECP is 1.0 at every level, so
        # ``tarp_error`` pins at exactly 0.5 for a posterior that is in fact
        # calibrated.
        #
        # Measured at 1500 sims x 200 draws on two fixtures, each drawing
        # from its model's *exact* posterior so that "calibrated" is a
        # property of the fixture rather than a label on it:
        #
        #   theta ~ U(-3, 3), x | theta ~ N(theta, 1), posterior N(x, 1)
        #   truncated to [-3, 3] -- truths drawn by ``uniform``, so the
        #   streams align: 0.5000 at the colliding seed for seeds 0, 1 and 7
        #   alike, against 0.005-0.015 at an independent one.
        #
        #   theta ~ N(0, 1), x | theta ~ N(theta, 1), posterior N(x/2, 0.5)
        #   -- truths drawn by ``normal``, so the Ziggurat's variable uniform
        #   consumption desynchronises the streams: 0.0069 against 0.0046.
        #
        # The second is the case worth fixing for. The first is loud enough
        # that nobody would ship it; a 50% inflation of a number that small
        # is invisible.
        #
        # Mixing a fixed salt into the SeedSequence entropy makes
        # ``seed=<the data seed>`` safe, so the documented advice stops being
        # conditional on the caller knowing this. ``seed=None`` still means
        # non-deterministic, and a given ``seed`` still reproduces.
        rng = np.random.default_rng(
            None if seed is None else [int(seed), _TARP_REFERENCE_SALT]
        )
        # Percentiles, not min/max: the extremes of n_simulations draws grow
        # like sqrt(2 log n_simulations), so a min/max box would silently
        # widen with sample size and make tarp_error magnitudes incomparable
        # between runs of different length.
        lo = np.percentile(truth_z, 1.0, axis=0)
        hi = np.percentile(truth_z, 99.0, axis=0)
        hi = np.where(hi > lo, hi, lo + 1.0)
        refs_z = rng.uniform(lo, hi, size=(n_sims, n_kept))
        reference_mode = "random"
    else:
        reference_points = np.asarray(reference_points, dtype=float)
        if reference_points.shape != (n_sims, n_params):
            raise ValueError(
                "reference_points must have shape (n_simulations, n_parameters): "
                f"{reference_points.shape} vs {(n_sims, n_params)}."
            )
        if not np.all(np.isfinite(reference_points)):
            raise ValueError("reference_points must be finite.")
        refs_z = (reference_points - centre) / scale
        # "provided", NOT "data_dependent". Whether the reference depends on x
        # is exactly what decides if this diagnostic has teeth (Lemos et al.
        # 2023, Sec. 4.3: an x-independent reference reports perfect coverage
        # for a posterior that ignores the data), and passing an array here
        # says nothing about how it was built -- externally generated random
        # references, supplied only for reproducibility, arrive by the same
        # route. Claiming the safer mode on that basis would let a persisted
        # result assert a guarantee it does not have.
        reference_mode = "provided"

    if metric == "euclidean":
        # einsum here too, matching the draw-distance branch below: different
        # summation orders leave the two sides of the strict `<` inconsistent
        # at the 1e-16 level, which only matters for a draw sitting exactly on
        # the truth but costs nothing to avoid.
        _dt = truth_z - refs_z
        dist_truth = np.sqrt(np.einsum("ij,ij->i", _dt, _dt))
    else:
        dist_truth = np.sum(np.abs(truth_z - refs_z), axis=-1)

    # Chunked over simulations: the difference tensor is
    # (n_simulations, n_draws, n_parameters) float64 and the squaring
    # allocates a second one before reducing, so the peak is twice its size.
    # At 5000 x 2000 x 200 that is ~32 GB materialized all at once.
    fractions = np.empty(n_sims, dtype=float)
    chunk = max(1, int(_TARP_CHUNK_ELEMENTS // max(1, n_draws * n_kept)))
    for start in range(0, n_sims, chunk):
        stop = min(start + chunk, n_sims)
        draws_z = (posterior_draws[start:stop].astype(float) - centre) / scale
        diff = draws_z - refs_z[start:stop, None, :]
        if metric == "euclidean":
            dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
        else:
            dist = np.sum(np.abs(diff), axis=-1)
        fractions[start:stop] = np.mean(dist < dist_truth[start:stop, None], axis=1)

    levels = np.linspace(0.0, 1.0, num=resolution + 2)[1:-1]
    expected = np.array([np.mean(fractions < level) for level in levels])
    deviation = np.abs(expected - levels)
    return {
        "credibility_levels": levels,
        "expected_coverage": expected,
        "coverage_fractions": fractions,
        "tarp_error": float(np.median(deviation)),
        "max_deviation": float(np.max(deviation)),
        "reference_mode": reference_mode,
        "n_simulations": int(n_sims),
        "n_draws": int(n_draws),
        "n_parameters": int(n_params),
    }


def classical_item_statistics(responses: np.ndarray) -> dict[str, np.ndarray]:
    """Classical item indices, for use as TARP reference coordinates.

    Returns, per item, the proportion-correct ``p_value``, its normal deviate
    ``z_difficulty = Phi^-1(1 - p)``, and the ``point_biserial`` correlation
    of the item with the total score over the remaining items.

    **These are deliberately the unconverted indices.** Lord and Novick
    (1968), as reported by Embretson and Reise (2013, ch. 4), relate the
    normal-ogive item parameters to classical indices by ``b_i ~ z_i / r_bis``
    and a corresponding expression for ``a_i`` in terms of the biserial
    correlation -- so applying those conversions would turn these into
    approximate *estimates* of the item parameters. That is precisely what a
    TARP reference point must not be.

    The reason is measurable. TARP asks what fraction of posterior draws lie
    closer to the reference point than the truth does, so a reference sitting
    near ``theta*`` leaves almost no draws closer, collapses every ``f_i``
    toward zero, and reports a *correct* posterior as maximally miscalibrated.
    On an exactly calibrated conjugate Gaussian, a reference placed at the
    truth plus noise scores ``tarp_error``:

    ==================  ====================
    noise scale         tarp_error (correct)
    ==================  ====================
    0.001 - 0.1         0.500 (worst possible)
    0.5                 0.408
    1.0                 0.251
    5.0                 0.055
    ==================  ====================

    Compared across reference schemes on the same data, where a usable scheme
    needs the calibrated case near zero *and* the ignored-data case large:

    ===========================  ==========  =================
    reference scheme             calibrated  posterior = prior
    ===========================  ==========  =================
    data-scale (paper's x + u)   0.006       0.294
    unbiased estimate of theta   0.256       0.336
    far-offset estimate          0.050       0.081
    ===========================  ==========  =================

    Only the data-scale reference separates the two. In the paper's conjugate
    example the posterior mean is ``x / 2``, so ``theta_r = x_0 + u`` sits
    systematically off ``theta*`` while still moving with the data -- and that
    offset is what makes it work. ``z_difficulty`` reproduces the same
    relationship for IRT: it is *proportional to* ``b_j * rho_j`` under the
    normal-ogive correspondence, so it is monotone in ``b_j`` but
    systematically attenuated.

    "Proportional to", not "equals". The underlying result is Lord and
    Novick's (1968, Chapter 16), reported by Martinkova and Hladka (2023,
    Sec. 7.3.1): under a standard normal latent trait and the **normal-ogive**
    2PL, ``rho_j = a_j / sqrt(1 + a_j^2)`` and ``b_j = Phi^-1(pi_j) / rho_j``.
    This package uses a **logistic** link, so the correspondence is exact only
    up to the scaling between the two metrics. Measured over 200 items and
    4000 persons under this package's default priors, ``z_difficulty``
    against ``b_j * rho_j`` gives a correlation of 0.994 with a slope of
    0.72 -- the monotone relationship the reference construction relies on,
    but not an identity.

    A note on the sign, since the textbook formula cannot be transcribed
    directly. ``b_j = Phi^-1(pi_j) / rho_j`` holds with ``pi_j`` the
    proportion *incorrect* under this package's ``P(correct) =
    logistic(a_j (theta - b_j))``. Measured on the same simulation,
    ``Phi^-1(1 - p) / rho`` correlates about +0.995 with ``b`` while
    ``Phi^-1(p) / rho`` correlates about -0.995 (0.994-0.995 across seeds;
    the magnitude is seed-dependent, the sign is not, and the sign is the
    point). ``z_difficulty = Phi^-1(1 - p)`` therefore has
    the correct orientation; copying the formula with ``p`` as written would
    invert it.

    The point-biserial is used rather than the biserial for the same reason:
    it is the attenuated index, it is an ordinary Pearson correlation with no
    conversion step, and correcting it toward the biserial would move the
    reference closer to the parameter it is supposed to stand apart from.

    Parameters
    ----------
    responses : np.ndarray
        Binary responses, ``(n_persons, n_items)`` or a batch
        ``(n_simulations, n_persons, n_items)``. Values must be 0 or 1;
        anything else raises, because these indices are defined only for
        dichotomous items and would otherwise return plausible nonsense.

    Returns
    -------
    dict
        ``p_value``, ``z_difficulty`` and ``point_biserial``, each shaped like
        ``responses`` with the person axis removed.

    Notes
    -----
    The total score excludes the item itself (a corrected item-total
    correlation); including it correlates the item with a total it
    contributes to, which inflates the index most exactly where there are
    fewest items -- the low-``I`` corners.

    An item with no response variance has an undefined correlation and gets
    ``point_biserial = 0``; ``p_value`` of exactly 0 or 1 makes
    ``z_difficulty`` infinite, so it is clipped to the open interval implied
    by one half-response.
    """
    from scipy.special import ndtri

    responses = np.asarray(responses, dtype=float)
    if responses.ndim == 2:
        responses = responses[None, ...]
        squeeze = True
    elif responses.ndim == 3:
        squeeze = False
    else:
        raise ValueError(
            "responses must be (n_persons, n_items) or "
            f"(n_simulations, n_persons, n_items); got {responses.shape}."
        )
    if not np.all(np.isfinite(responses)):
        raise ValueError("responses must be finite.")
    # Every index below is defined only for 0/1 data, and none of them fails
    # loudly on anything else. A polytomous matrix makes `p_value` a mean
    # CATEGORY SCORE rather than a proportion correct; the clip then drags it
    # into [0, 1], collapsing many items onto the same extreme
    # `z_difficulty`. The result looks like an ordinary set of item
    # statistics and would silently corrupt a TARP reference built from it,
    # so reject the input instead of returning plausible nonsense.
    unique = np.unique(responses)
    if unique.size and not np.all(np.isin(unique, (0.0, 1.0))):
        raise ValueError(
            "classical_item_statistics needs binary (0/1) responses; got "
            f"values {np.sort(unique)[:8].tolist()}. These indices are "
            "defined for dichotomous items only -- for polytomous data "
            "(PCM/GPCM) derive a reference from category-appropriate "
            "statistics instead."
        )

    n_persons = responses.shape[1]
    if n_persons < 2:
        raise ValueError("responses need at least 2 persons.")

    p_value = responses.mean(axis=1)
    # Clip by half a response so a ceiling/floor item stays on the real line.
    half = 0.5 / n_persons
    p_clipped = np.clip(p_value, half, 1.0 - half)
    z_difficulty = -ndtri(p_clipped)

    total = responses.sum(axis=2, keepdims=True)
    rest = total - responses
    item_c = responses - responses.mean(axis=1, keepdims=True)
    rest_c = rest - rest.mean(axis=1, keepdims=True)
    num = (item_c * rest_c).sum(axis=1)
    den = np.sqrt((item_c**2).sum(axis=1) * (rest_c**2).sum(axis=1))
    point_biserial = np.divide(num, den, out=np.zeros_like(num), where=den > 0)

    out = {
        "p_value": p_value,
        "z_difficulty": z_difficulty,
        "point_biserial": point_biserial,
    }
    if squeeze:
        out = {k: v[0] for k, v in out.items()}
    return out


#: Classical index standing in for each item parameter as a data-dependent
#: TARP reference coordinate: ``point_biserial`` for discrimination and
#: ``z_difficulty`` for difficulty. Both are dichotomous indices, which
#: confines the scheme to 1PL/2PL; PCM/GPCM have none yet. See
#: :func:`classical_item_statistics` for why the indices are left unconverted.
CLASSICAL_REFERENCE_COORDS: dict[str, str] = {
    "a": "point_biserial",
    "b": "z_difficulty",
}


def _require_classical_coords(param_keys: Sequence[str]) -> None:
    """Raise unless every key has a classical reference coordinate."""
    unsupported = [k for k in param_keys if k not in CLASSICAL_REFERENCE_COORDS]
    if unsupported:
        raise ValueError(
            f"No classical reference coordinate for parameter(s) "
            f"{unsupported}. `classical_item_statistics` returns dichotomous "
            "indices (proportion correct and item-rest correlation), so this "
            "reference scheme covers 1PL and 2PL only. PCM/GPCM need "
            "category-appropriate reference coordinates, which do not exist "
            "yet. Running TARP here would need a "
            "reference this package cannot honestly supply."
        )


def classical_reference_points(
    responses: np.ndarray,
    param_keys: Sequence[str],
    *,
    jitter: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Data-dependent TARP reference points from classical item statistics.

    Lemos et al. (2023, Sec. 4.3) show that TARP with reference points
    independent of the data is blind to a posterior that ignores the data;
    these coordinates move with the responses and stay offset from the true
    parameters, the construction :func:`classical_item_statistics` documents.
    The scheme stacks ``a`` on the log scale, so pair them with ``log a``
    draws.

    Parameters
    ----------
    responses : np.ndarray
        Binary responses ``(n_simulations, n_persons, n_items)``.
    param_keys : sequence of str
        Parameter blocks in column order; each must be a key of
        :data:`CLASSICAL_REFERENCE_COORDS`.
    jitter : float
        Width of uncentred ``U(0, jitter)`` noise added to the coordinates
        (Lemos et al., 2023, Sec. 4.3). Default 0: a deterministic reference.
    rng : numpy.random.Generator, optional
        Source of the jitter; required when ``jitter`` is nonzero.

    Returns
    -------
    np.ndarray
        ``(n_simulations, len(param_keys) * n_items)``, blocks in
        *param_keys* order with items inside each block.

    Raises
    ------
    ValueError
        If a key has no classical coordinate (e.g. PCM/GPCM ``delta``).
    """
    _require_classical_coords(param_keys)
    stats = classical_item_statistics(np.asarray(responses, dtype=float))
    refs = np.concatenate(
        [stats[CLASSICAL_REFERENCE_COORDS[k]] for k in param_keys], axis=1
    )
    if jitter:
        if rng is None:
            raise ValueError("rng is required when jitter is nonzero.")
        refs = refs + jitter * rng.uniform(size=refs.shape)
    return refs


def _compute_log_recovery_metrics(
    true_flat: np.ndarray,
    posterior_flat: np.ndarray,
) -> dict[str, float]:
    """Compute recovery metrics from log true values and log posterior means."""
    log_true = np.log(true_flat)
    log_point_estimate = np.mean(np.log(posterior_flat), axis=1)
    return _compute_recovery_metrics(log_true, log_point_estimate)


def _format_corner_annotation(
    corner: tuple[int, int],
    metrics: Mapping[str, float],
    *,
    n_observations: int,
) -> str:
    """Format the per-corner condition and recovery metrics."""
    corr_txt = f"{metrics['corr']:.3f}" if np.isfinite(metrics["corr"]) else "nan"
    nrmse_txt = f"{metrics['nrmse']:.3f}" if np.isfinite(metrics["nrmse"]) else "nan"
    return (
        f"I={corner[0]}, P={corner[1]}\n"
        f"n={n_observations}, r={corr_txt}, NRMSE={nrmse_txt}"
    )


def _compute_coverage_profile_difference(
    posterior_flat: np.ndarray,
    true_flat: np.ndarray,
    *,
    prob: float = 0.95,
) -> dict[str, np.ndarray]:
    """Compute coverage-profile differences while sorting only once."""
    from scipy.stats import beta

    num_datasets, num_draws = posterior_flat.shape
    widths = np.arange(0, num_draws + 2) / (num_draws + 1)
    sorted_samples = np.sort(posterior_flat, axis=1)

    coverage_estimates = np.zeros_like(widths)
    coverage_lower = np.zeros_like(widths)
    coverage_upper = np.zeros_like(widths)

    for width_idx, width in enumerate(widths):
        n_ranks_covered = round((num_draws + 1) * width)
        low_rank = round(num_draws / 2 - n_ranks_covered / 2)
        high_rank = low_rank + n_ranks_covered - 1
        low_rank = max(0, low_rank)
        high_rank = min(num_draws - 1, high_rank)

        actual_width = (high_rank - low_rank + 1) / (num_draws + 1)
        covered = (true_flat >= sorted_samples[:, low_rank]) & (
            true_flat <= sorted_samples[:, high_rank]
        )
        num_covered = int(np.sum(covered))
        coverage_est = num_covered / num_datasets

        if actual_width == 0 or actual_width == 1:
            ci_low = actual_width
            ci_high = actual_width
        else:
            alpha_post = num_covered + 1
            beta_post = num_datasets - num_covered + 1
            ci_low = beta.ppf((1 - prob) / 2, alpha_post, beta_post)
            ci_high = beta.ppf((1 + prob) / 2, alpha_post, beta_post)

        coverage_estimates[width_idx] = coverage_est
        coverage_lower[width_idx] = ci_low
        coverage_upper[width_idx] = ci_high
        widths[width_idx] = actual_width

    return {
        "widths": widths,
        "coverage_diff": coverage_estimates - widths,
        "coverage_diff_low": coverage_lower - widths,
        "coverage_diff_high": coverage_upper - widths,
    }


def _free_shared_axis(ax: Any, axis: str) -> None:
    """Detach one Matplotlib axis from shared-axis groups if present."""
    if axis == "x":
        shared_axes = ax.get_shared_x_axes()
    elif axis == "y":
        shared_axes = ax.get_shared_y_axes()
    else:  # pragma: no cover - internal guard
        raise ValueError("axis must be 'x' or 'y'.")

    if len(shared_axes.get_siblings(ax)) <= 1:
        return

    grouper = getattr(shared_axes, "_grouper", None)
    if grouper is not None and hasattr(grouper, "remove"):
        grouper.remove(ax)


def _set_recovery_identity_axes(
    ax: Any,
    true_flat: np.ndarray,
    point_estimate: np.ndarray,
    intervals: np.ndarray,
    *,
    log_scale: bool,
) -> None:
    """Set recovery panel limits and draw the identity line.

    Degenerate panels (all-NaN posteriors, or nothing positive to show on a
    log axis) leave the limits to matplotlib's autoscale rather than raising:
    a diagnostic figure exists to reveal a model that misbehaves, so it must
    survive one. This can happen when a size-pinned model is scored far out
    of distribution.
    """
    if log_scale:
        candidates = np.concatenate(
            [
                true_flat.ravel(),
                point_estimate.ravel(),
                intervals[0].ravel(),
                intervals[1].ravel(),
            ]
        )
        # Must be *positive*, not merely finite: a non-positive value makes
        # the log10 ratio below NaN, and cannot be shown on a log axis.
        positive_values = candidates[np.isfinite(candidates) & (candidates > 0.0)]
        if positive_values.size == 0:
            return
        lower = float(np.min(positive_values))
        upper = float(np.max(positive_values))
        if lower == upper:
            lower /= 10.0
            upper *= 10.0
        else:
            pad_factor = 10.0 ** (0.05 * np.log10(upper / lower))
            lower /= pad_factor
            upper *= pad_factor
    else:
        candidates = np.concatenate([true_flat.ravel(), point_estimate.ravel()])
        candidates = candidates[np.isfinite(candidates)]
        if candidates.size == 0:
            return
        lower = float(np.min(candidates))
        upper = float(np.max(candidates))
        eps = (upper - lower) * 0.1
        lower -= eps
        upper += eps

    ax.set_xlim((lower, upper))
    ax.set_ylim((lower, upper))
    ax.plot(
        [lower, upper],
        [lower, upper],
        color="black",
        alpha=0.9,
        linestyle="dashed",
    )


def plot_edge_parameter_diagnostics(
    simulator: Any | None = None,
    approximator: Any | None = None,
    *,
    param_key: str,
    samples_per_condition: int = 20,
    n_posterior_draws: int = 500,
    sample_chunk_size: int | None = None,
    ci_level: float = 0.90,
    color: str = "#4C72B0",
    figsize: tuple[float, float] | None = None,
    clear_cuda_cache: bool = False,
    corners: Sequence[tuple[int, int]] | None = None,
    title_prefix: str | None = None,
    edge_data: Mapping[str, Any] | None = None,
    equalize_observations: bool = True,
    equalize_seed: int = 123,
    log_scale: bool = False,
    flow_sample_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Plot a 4-row edge-corner diagnostic figure for a single parameter.

    Each column corresponds to one corner, so the figure is ``4 x
    len(corners)`` -- 4x4 for the usual four edge corners induced by the
    simulator's ``(n_items, n_persons)`` ranges, and wider when extra
    conditions are scored alongside them (for example a pinned training size
    that is not itself an edge corner). Rows show, from top to bottom:
    recovery with posterior standard deviation bands, SBC rank histograms,
    calibration ECDF differences, and coverage-profile differences.

    The recovery row and calibration summaries follow the BayesFlow
    diagnostics implementations in ``bayesflow.diagnostics.plots`` and
    ``bayesflow.utils`` (Talts et al., 2018; Saeilynoja et al., 2022).
    Set ``log_scale=True`` to render the recovery panel on log-log axes for
    strictly positive parameters such as 2PL discrimination ``a``.
    """
    if samples_per_condition < 1:
        raise ValueError("samples_per_condition must be >= 1.")
    if n_posterior_draws < 1:
        raise ValueError("n_posterior_draws must be >= 1.")
    if sample_chunk_size is not None and sample_chunk_size < 1:
        raise ValueError("sample_chunk_size must be >= 1 when provided.")
    if not 0.0 < ci_level < 1.0:
        raise ValueError("ci_level must be in (0, 1).")

    from bayesflow.utils.ecdf import simultaneous_ecdf_bands
    from bayesflow.utils.plot_utils import prettify_subplots
    from matplotlib.ticker import PercentFormatter
    from scipy.stats import binom

    if edge_data is not None:
        corner_list = [
            (int(n_items), int(n_persons))
            for n_items, n_persons in edge_data["corners"]
        ]
        condition_batches = list(edge_data["conditions"])
        posterior_batches = list(edge_data["posteriors"])
        n_posterior_draws = int(edge_data["n_posterior_draws"])
    else:
        if simulator is None or approximator is None:
            raise ValueError(
                "Pass either edge_data or both simulator and approximator."
            )
        corner_list, condition_batches, posterior_batches = (
            _collect_edge_corner_batches(
                simulator,
                approximator,
                corners=corners,
                samples_per_condition=samples_per_condition,
                n_posterior_draws=n_posterior_draws,
                sample_chunk_size=sample_chunk_size,
                clear_cuda_cache=clear_cuda_cache,
                flow_sample_kwargs=flow_sample_kwargs,
            )
        )

    import matplotlib.pyplot as plt

    n_cols = len(corner_list)
    if figsize is None:
        # 4.0 in per corner column keeps panel aspect constant as columns are
        # added; 13.5 in is the height of the original four-column figure.
        figsize = (4.0 * n_cols, 13.5)
    fig, axes = plt.subplots(4, n_cols, figsize=figsize, squeeze=False)

    corner_results: dict[tuple[int, int], dict[str, Any]] = {}
    band_cache: dict[int, tuple[float, np.ndarray, np.ndarray, np.ndarray]] = {}
    rng = np.random.default_rng(equalize_seed)

    target_count: int | None = None
    if equalize_observations:
        obs_counts = []
        for col, corner in enumerate(corner_list):
            condition_batch = condition_batches[col]
            posterior_batch = posterior_batches[col]
            if param_key not in condition_batch or param_key not in posterior_batch:
                continue
            obs_counts.append(np.asarray(condition_batch[param_key]).size)
        if obs_counts:
            target_count = int(min(obs_counts))

    for col, corner in enumerate(corner_list):
        condition_batch = condition_batches[col]
        posterior_batch = posterior_batches[col]
        if param_key not in condition_batch:
            raise KeyError(
                f"Missing true parameter key {param_key!r} in condition batch "
                f"for corner {corner}."
            )
        if param_key not in posterior_batch:
            raise KeyError(
                f"Missing posterior parameter key {param_key!r} in sample output "
                f"for corner {corner}."
            )

        true_vals = np.asarray(condition_batch[param_key])
        post_vals = np.asarray(posterior_batch[param_key])
        flattened = _flatten_parameter_draws(
            true_vals,
            post_vals,
            param_key=param_key,
            n_posterior_draws=n_posterior_draws,
        )
        flattened = _subsample_flattened_parameter_draws(
            flattened,
            target_count=target_count,
            rng=rng,
        )
        n_observations = int(flattened["true_flat"].size)
        if log_scale and (
            np.any(flattened["true_flat"] <= 0.0)
            or np.any(flattened["point_estimate"] <= 0.0)
            or np.any(flattened["posterior_flat"] <= 0.0)
        ):
            raise ValueError(
                "log_scale=True requires strictly positive true values and "
                "posterior draws."
            )
        if log_scale:
            metrics = _compute_log_recovery_metrics(
                flattened["true_flat"],
                flattened["posterior_flat"],
            )
        else:
            metrics = _compute_recovery_metrics(
                flattened["true_flat"],
                flattened["point_estimate"],
            )
        _format_corner_annotation(
            corner,
            metrics,
            n_observations=n_observations,
        )
        corner_results[corner] = {
            **flattened,
            **metrics,
            "condition": corner,
            "n_observations": n_observations,
        }

        # Row 1: recovery scatter with posterior standard deviation bands.
        ax = axes[0, col]
        point_estimate = flattened["point_estimate"]
        posterior_sd = np.std(flattened["posterior_flat"], axis=1)
        ax.errorbar(
            flattened["true_flat"],
            point_estimate,
            yerr=posterior_sd,
            fmt="o",
            alpha=0.45,
            color=color,
            markersize=3,
            elinewidth=0.5,
        )
        if log_scale:
            _free_shared_axis(ax, "x")
            _free_shared_axis(ax, "y")
            ax.set_xscale("log")
            ax.set_yscale("log")
        _set_recovery_identity_axes(
            ax,
            flattened["true_flat"],
            point_estimate,
            np.vstack([point_estimate - posterior_sd, point_estimate + posterior_sd]),
            log_scale=log_scale,
        )
        ax.set_xlabel(f"True {param_key}")
        ax.set_ylabel(f"Posterior mean {param_key}")

        # Row 2: rank histogram.
        ax = axes[1, col]
        num_bins = min(20, max(5, int(n_posterior_draws // 20)))
        endpoints = binom.interval(0.99, flattened["ranks"].size, 1 / num_bins)
        expected = flattened["ranks"].size / num_bins
        ax.axhspan(endpoints[0], endpoints[1], facecolor="gray", alpha=0.25)
        ax.axhline(expected, color="gray", alpha=0.9, lw=1.0)
        ax.hist(
            flattened["ranks"],
            bins=num_bins,
            range=(-0.5, n_posterior_draws + 0.5),
            color=color,
            alpha=0.9,
        )
        ax.get_yaxis().set_ticks([])
        ax.set_xlabel("Rank statistic")
        ax.set_ylabel("")

        # Row 3: calibration ECDF difference with simultaneous bands.
        ax = axes[2, col]
        ranks_frac = flattened["ranks"] / n_posterior_draws
        xx = np.repeat(np.sort(ranks_frac), 2)
        xx = np.pad(xx, (1, 1), constant_values=(0, 1))
        yy = np.linspace(0, 1, num=xx.shape[-1] // 2)
        yy = np.repeat(yy, 2)
        yy_diff = yy - xx
        if n_observations not in band_cache:
            band_cache[n_observations] = simultaneous_ecdf_bands(n_observations)
        alpha_band, z_band, lower_band, upper_band = band_cache[n_observations]
        lower_band = lower_band - z_band
        upper_band = upper_band - z_band
        ax.fill_between(
            z_band,
            lower_band,
            upper_band,
            color="grey",
            alpha=0.2,
            label=rf"{int((1 - alpha_band) * 100)}% band" if col == 0 else None,
        )
        ax.plot(xx, yy_diff, color=color, lw=1.2, alpha=0.95)
        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.6)
        ax.set_xlabel("Fractional rank")
        ax.set_ylabel("ECDF - Uniform")
        ax.set_xlim(0, 1)

        # Row 4: coverage difference.
        ax = axes[3, col]
        coverage_data = _compute_coverage_profile_difference(
            flattened["posterior_flat"],
            flattened["true_flat"],
            prob=0.95,
        )
        ax.fill_between(
            coverage_data["widths"],
            coverage_data["coverage_diff_low"],
            coverage_data["coverage_diff_high"],
            color="grey",
            alpha=0.25,
            label="95% interval" if col == 0 else None,
        )
        ax.plot(
            coverage_data["widths"],
            coverage_data["coverage_diff"],
            color=color,
            lw=1.5,
        )
        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.6)
        ax.set_xlabel("Central interval width")
        ax.set_ylabel("Observed - Nominal")
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    prettify_subplots(axes, num_subplots=4 * n_cols, tick_fontsize=10)

    row_titles = [
        "Recovery",
        "Rank Histogram",
        "Calibration ECDF",
        "Coverage Profile",
    ]
    row_y = [0.86, 0.62, 0.385, 0.15]
    for label, y_pos in zip(row_titles, row_y):
        fig.text(
            0.015,
            y_pos,
            label,
            rotation=90,
            va="center",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )
    for col, corner in enumerate(corner_list):
        axes[0, col].set_title(
            _format_corner_annotation(
                corner,
                corner_results[corner],
                n_observations=corner_results[corner]["n_observations"],
            ),
            fontsize=11,
            fontweight="bold",
            pad=18,
        )

    figure_title = title_prefix or f"Edge-Corner Diagnostics for {param_key}"
    fig.suptitle(
        figure_title,
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0.03, 0, 1, 0.965))
    return {
        "figure": fig,
        "axes": axes,
        "corners": corner_list,
        "param_key": param_key,
        "samples_per_condition": int(samples_per_condition),
        "n_posterior_draws": int(n_posterior_draws),
        "corner_results": corner_results,
        "equalize_observations": bool(equalize_observations),
        "log_scale": bool(log_scale),
        "plot_n_observations": target_count,
    }


def summarize_edge_parameter_diagnostics(
    diagnostics: Mapping[str, Any],
    *,
    calibration_resolution: int = 20,
    print_summary: bool = False,
) -> list[dict[str, Any]]:
    """Numeric per-corner summary (correlation, NRMSE, calibration error).

    Consumes the dict returned by :func:`plot_edge_parameter_diagnostics`
    directly. No extra posterior sampling is needed: that dict already
    retains each corner's flattened true values and posterior draws in
    ``corner_results``, which is exactly what :func:`compute_calibration_error`
    needs.

    Parameters
    ----------
    diagnostics : Mapping
        The dict returned by :func:`plot_edge_parameter_diagnostics`.
    calibration_resolution : int
        Forwarded to :func:`compute_calibration_error`.
    print_summary : bool
        If True, also print one formatted line per corner.

    Returns
    -------
    list of dict
        One entry per edge corner, each with ``param_key``, ``n_items``,
        ``n_persons``, ``n_observations``, ``corr``, ``nrmse``,
        ``nrmse_ideal``, ``nrmse_excess``, and ``calibration_error``.
        ``nrmse_excess`` is the shortfall against the NRMSE an exact posterior
        mean would achieve at the same correlation, so values near zero mean
        the estimator is scaled like a posterior mean and positive values
        indicate mis-scaling (usually under-shrinkage). When the source
        diagnostic was produced with ``log_scale=True``, ``corr`` and the
        NRMSE columns are in log space (matching what the plot displays);
        ``calibration_error`` is invariant to the transform.
    """
    param_key = diagnostics["param_key"]
    rows: list[dict[str, Any]] = []
    for corner, result in diagnostics["corner_results"].items():
        calibration_error = compute_calibration_error(
            result["posterior_flat"],
            result["true_flat"],
            resolution=calibration_resolution,
        )["calibration_error"]
        row = {
            "param_key": param_key,
            "n_items": int(corner[0]),
            "n_persons": int(corner[1]),
            "n_observations": int(result["n_observations"]),
            "corr": float(result["corr"]),
            "nrmse": float(result["nrmse"]),
            "nrmse_ideal": float(result["nrmse_ideal"]),
            "nrmse_excess": float(result["nrmse_excess"]),
            "calibration_error": float(calibration_error),
        }
        rows.append(row)
        if print_summary:
            print(
                f"{param_key}: I={row['n_items']:>3}, P={row['n_persons']:>4}  "
                f"NRMSE={row['nrmse']:.3f}  r={row['corr']:.3f}  "
                f"dNRMSE={row['nrmse_excess']:+.3f}  "
                f"CalErr={row['calibration_error']:.3f}"
            )
    return rows


def chunked_posterior_sample(
    approximator: Any,
    conditions: Mapping[str, np.ndarray],
    *,
    num_samples: int,
    chunk_size: int,
    clear_cuda_cache: bool = False,
    flow_sample_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Sample posterior draws in condition chunks and return NumPy arrays."""
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1.")

    n = len(next(iter(conditions.values())))
    parts: list[dict[str, np.ndarray]] = []

    torch = None
    if clear_cuda_cache:
        try:
            import torch as _torch  # type: ignore

            torch = _torch
        except Exception:  # pragma: no cover - optional dependency path
            torch = None

    for lo in range(0, n, chunk_size):
        hi = min(lo + chunk_size, n)
        chunk_cond = {k: v[lo:hi] for k, v in conditions.items()}
        if clear_cuda_cache and torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        chunk_out = approximator.sample(
            num_samples=num_samples,
            conditions=chunk_cond,
            flow_sample_kwargs=flow_sample_kwargs,
        )
        parts.append({k: np.asarray(v) for k, v in chunk_out.items()})

    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}


def sample_random_size_conditions(
    simulator: Any,
    *,
    n_conditions: int,
    datasets_per_condition: int = 1,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Sample condition batches from random ``(n_items, n_persons)`` draws."""
    if n_conditions < 1:
        raise ValueError("n_conditions must be >= 1.")
    if datasets_per_condition < 1:
        raise ValueError("datasets_per_condition must be >= 1.")

    if not hasattr(simulator, "clone_with_sizes"):
        raise TypeError(
            "simulator must provide clone_with_sizes for random-size SBC sampling."
        )

    try:
        items_lo, items_hi = int(simulator.n_items[0]), int(simulator.n_items[1])
        persons_lo, persons_hi = (
            int(simulator.n_persons[0]),
            int(simulator.n_persons[1]),
        )
    except Exception as exc:
        raise TypeError("simulator must expose n_items and n_persons ranges.") from exc

    rng = np.random.default_rng(seed)
    # Cloned simulators restart from the original seed, so each condition gets
    # its own stream derived from this helper's seed.
    condition_streams = np.random.SeedSequence(seed).spawn(n_conditions)
    sampled: list[dict[str, Any]] = []
    for cond_idx in range(n_conditions):
        n_items_case = int(rng.integers(items_lo, items_hi + 1))
        n_persons_case = int(rng.integers(persons_lo, persons_hi + 1))
        sim_case = simulator.clone_with_sizes(
            n_items=(n_items_case, n_items_case),
            n_persons=(n_persons_case, n_persons_case),
        )
        if hasattr(sim_case, "_rng"):
            sim_case._rng = np.random.default_rng(condition_streams[cond_idx])
        for draw_idx in range(datasets_per_condition):
            cond_case = sim_case.sample((1,))
            sampled.append(
                {
                    "condition_index": cond_idx,
                    "draw_index": draw_idx,
                    "N_items": n_items_case,
                    "N_persons": n_persons_case,
                    "conditions": {
                        k: np.array(v, copy=True) for k, v in cond_case.items()
                    },
                }
            )
    return sampled


def pooled_sbc_for_parameter(
    approximator: Any,
    *,
    condition_batches: Sequence[Mapping[str, np.ndarray] | Mapping[str, Any]],
    param_key: str = "b",
    n_posterior_samples: int = 500,
    sample_chunk_size: int = 1,
    clear_cuda_cache: bool = False,
    flow_sample_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute pooled SBC ranks and flattened posterior arrays for one parameter."""
    if n_posterior_samples < 1:
        raise ValueError("n_posterior_samples must be >= 1.")
    if sample_chunk_size < 1:
        raise ValueError("sample_chunk_size must be >= 1.")
    if not condition_batches:
        raise ValueError("condition_batches must be non-empty.")

    all_ranks: list[np.ndarray] = []
    all_post_flat: list[np.ndarray] = []
    all_true_flat: list[np.ndarray] = []

    for case in condition_batches:
        case_conditions = case.get("conditions", case)  # support metadata wrappers
        if param_key not in case_conditions:
            raise KeyError(f"Missing true parameter key {param_key!r} in conditions.")

        posterior = chunked_posterior_sample(
            approximator,
            case_conditions,
            num_samples=n_posterior_samples,
            chunk_size=sample_chunk_size,
            clear_cuda_cache=clear_cuda_cache,
            flow_sample_kwargs=flow_sample_kwargs,
        )
        if param_key not in posterior:
            raise KeyError(
                f"Missing posterior parameter key {param_key!r} in sample output."
            )

        true_vals = np.asarray(case_conditions[param_key])
        post_vals = np.asarray(posterior[param_key])
        if post_vals.ndim != true_vals.ndim + 1:
            raise ValueError(
                "Posterior rank for param "
                f"{param_key!r} must have sample axis at position 1: "
                f"true shape={true_vals.shape}, posterior shape={post_vals.shape}."
            )
        if post_vals.shape[0] != true_vals.shape[0]:
            raise ValueError(
                "Posterior and true batch axes do not match for param "
                f"{param_key!r}: {post_vals.shape[0]} vs {true_vals.shape[0]}."
            )
        if post_vals.shape[1] != n_posterior_samples:
            raise ValueError(
                "Posterior sample axis does not match n_posterior_samples for param "
                f"{param_key!r}: {post_vals.shape[1]} vs {n_posterior_samples}."
            )
        if post_vals.shape[2:] != true_vals.shape[1:]:
            raise ValueError(
                "Posterior trailing shape does not match true parameter shape "
                "for param "
                f"{param_key!r}: {post_vals.shape[2:]} vs {true_vals.shape[1:]}."
            )

        ranks_case = np.sum(
            post_vals < np.expand_dims(true_vals, axis=1), axis=1
        ).reshape(-1)
        all_ranks.append(ranks_case)
        all_post_flat.append(
            np.moveaxis(post_vals, 1, -1).reshape(-1, n_posterior_samples)
        )
        all_true_flat.append(true_vals.reshape(-1))

    return {
        "ranks": np.concatenate(all_ranks),
        "posterior_flat": np.vstack(all_post_flat),
        "true_flat": np.concatenate(all_true_flat),
        "n_posterior_samples": int(n_posterior_samples),
        "n_condition_batches": int(len(condition_batches)),
    }


def pooled_sbc_random_conditions(
    simulator: Any,
    approximator: Any,
    *,
    param_key: str = "b",
    n_conditions: int = 100,
    datasets_per_condition: int = 1,
    n_posterior_samples: int = 500,
    sample_chunk_size: int = 1,
    seed: int | None = None,
    clear_cuda_cache: bool = False,
    flow_sample_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run pooled SBC on random size conditions sampled from simulator ranges."""
    sampled_conditions = sample_random_size_conditions(
        simulator,
        n_conditions=n_conditions,
        datasets_per_condition=datasets_per_condition,
        seed=seed,
    )
    pooled = pooled_sbc_for_parameter(
        approximator,
        condition_batches=sampled_conditions,
        param_key=param_key,
        n_posterior_samples=n_posterior_samples,
        sample_chunk_size=sample_chunk_size,
        clear_cuda_cache=clear_cuda_cache,
        flow_sample_kwargs=flow_sample_kwargs,
    )
    pooled["sampled_conditions"] = sampled_conditions
    pooled["n_conditions"] = int(n_conditions)
    pooled["datasets_per_condition"] = int(datasets_per_condition)
    return pooled


def run_sbc_diagnostics(
    all_ranks_arr: np.ndarray,
    all_post_flat_arr: np.ndarray,
    all_true_flat_arr: np.ndarray,
    *,
    title_prefix: str = "SBC Diagnostics",
    show_plots: bool = True,
    color: str = "#4C72B0",
) -> dict[str, float]:
    """Run pooled SBC diagnostics and optionally render the standard 4-panel plot."""
    from matplotlib.ticker import PercentFormatter
    from scipy.stats import chisquare, kstest
    from scipy.stats import norm as sp_norm

    all_ranks_arr = np.asarray(all_ranks_arr)
    all_post_flat_arr = np.asarray(all_post_flat_arr)
    all_true_flat_arr = np.asarray(all_true_flat_arr)

    n_posterior_draws = int(all_post_flat_arr.shape[1])
    n_total = int(len(all_ranks_arr))
    frac_r = all_ranks_arr / n_posterior_draws

    ks_stat, ks_pval = kstest(
        (all_ranks_arr + 0.5) / (n_posterior_draws + 1), "uniform"
    )
    n_bins_chi = 20
    hist_counts, _ = np.histogram(
        all_ranks_arr, bins=n_bins_chi, range=(-0.5, n_posterior_draws + 0.5)
    )
    chi2_stat, chi2_pval = chisquare(
        hist_counts, f_exp=[n_total / n_bins_chi] * n_bins_chi
    )

    mean_all = all_post_flat_arr.mean(axis=1)
    r_global = float(np.corrcoef(all_true_flat_arr, mean_all)[0, 1])
    rmse_global = float(np.sqrt(np.mean((mean_all - all_true_flat_arr) ** 2)))

    print(f"{title_prefix}  ({n_total:,} rank statistics,  M = {n_posterior_draws})")
    print(f"  KS test:      stat = {ks_stat:.4f},  p = {ks_pval:.4f}")
    print(f"  Chi-squared:  stat = {chi2_stat:.2f},  p = {chi2_pval:.4f}")
    print("  (p > 0.05 -> cannot reject calibration)")
    print(f"\nGlobal recovery:  r = {r_global:.4f},  RMSE = {rmse_global:.4f}")

    if show_plots:
        import matplotlib.pyplot as plt

        cov_levels = np.linspace(0.01, 0.99, 99)
        emp_cov = np.array(
            [
                np.mean((frac_r > (1 - lev) / 2) & (frac_r <= 1 - (1 - lev) / 2))
                for lev in cov_levels
            ]
        )
        cov_diff = emp_cov - cov_levels

        z95 = sp_norm.ppf(0.975)
        denom = 1 + z95**2 / n_total
        ci_lo_cov = np.clip(
            (emp_cov + z95**2 / (2 * n_total)) / denom
            - z95
            * np.sqrt(emp_cov * (1 - emp_cov) / n_total + z95**2 / (4 * n_total**2))
            / denom,
            0,
            1,
        )
        ci_hi_cov = np.clip(
            (emp_cov + z95**2 / (2 * n_total)) / denom
            + z95
            * np.sqrt(emp_cov * (1 - emp_cov) / n_total + z95**2 / (4 * n_total**2))
            / denom,
            0,
            1,
        )
        diff_lo = ci_lo_cov - cov_levels
        diff_hi = ci_hi_cov - cov_levels

        fig, axes = plt.subplots(2, 2, figsize=(13, 10))
        axes = axes.flatten()

        ax = axes[0]
        n_hist = 20
        exp_bin = n_total / n_hist
        z99 = sp_norm.ppf(0.995)
        p_b = 1 / n_hist
        ax.hist(
            frac_r,
            bins=n_hist,
            range=(0, 1),
            color=color,
            edgecolor="white",
            alpha=0.9,
            label="Observed",
        )
        ax.axhspan(
            exp_bin - z99 * np.sqrt(n_total * p_b * (1 - p_b)),
            exp_bin + z99 * np.sqrt(n_total * p_b * (1 - p_b)),
            alpha=0.25,
            facecolor="grey",
            label="99% CI (uniform)",
        )
        ax.axhline(exp_bin, color="grey", lw=1.0, zorder=2)
        ax.get_yaxis().set_ticks([])
        ax.set_xlim(0, 1)
        ax.set_xlabel("Fractional rank r/M", fontsize=12)
        ax.set_title("Rank Histogram", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.text(
            0.03,
            0.97,
            "flat = well calibrated",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            color="#666",
        )

        ax = axes[1]
        sorted_frac = np.sort(frac_r)
        xx = np.repeat(sorted_frac, 2)
        xx = np.pad(xx, (1, 1), constant_values=(0, 1))
        yy = np.linspace(0, 1, num=xx.shape[-1] // 2)
        yy = np.repeat(yy, 2)
        yy_diff = yy - xx
        eps_band = np.sqrt(np.log(2 / 0.05) / (2 * n_total))
        z_axis = np.linspace(0, 1, 200)
        scale = np.sqrt(z_axis * (1 - z_axis)) / 0.5
        ax.fill_between(
            z_axis,
            -eps_band * scale,
            eps_band * scale,
            color="grey",
            alpha=0.2,
            label="95% Conf. Band",
        )
        ax.plot(xx, yy_diff, color=color, alpha=0.9, lw=1.0, label="Rank ECDF")
        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.6)
        ax.set_xlabel("Fractional rank statistic", fontsize=12)
        ax.set_ylabel("ECDF - Uniform", fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_title("Calibration ECDF (difference)", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.text(
            0.03,
            0.97,
            "curve near 0 = well calibrated",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            color="#666",
        )

        ax = axes[2]
        ax.fill_between(
            cov_levels, diff_lo, diff_hi, alpha=0.2, color="grey", label="95% CI"
        )
        ax.plot(cov_levels, cov_diff, color=color, lw=1.5, label="Coverage diff")
        ax.axhline(0, color="black", lw=1.0, ls="--", alpha=0.7)
        ax.set_xlabel("Nominal Coverage", fontsize=12)
        ax.set_ylabel("Observed - Nominal", fontsize=12)
        ax.set_title("Coverage Profile", fontsize=13, fontweight="bold")
        ax.set_xlim(0, 1)
        max_abs = max(np.max(np.abs(diff_lo)), np.max(np.abs(diff_hi)), 0.02)
        ax.set_ylim(-max_abs * 1.1, max_abs * 1.1)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.legend(fontsize=10, loc="upper left")
        ax.text(
            0.03,
            0.03,
            "flat near 0 = well calibrated",
            transform=ax.transAxes,
            va="bottom",
            fontsize=9,
            color="#666",
        )

        ax = axes[3]
        rng_s = np.random.default_rng(0)
        idx = rng_s.choice(
            len(all_true_flat_arr),
            size=min(5000, len(all_true_flat_arr)),
            replace=False,
        )
        bt, bm = all_true_flat_arr[idx], mean_all[idx]
        ax.scatter(bt, bm, alpha=0.15, s=6, color=color, edgecolors="none")
        lims = [min(bt.min(), bm.min()) - 0.15, max(bt.max(), bm.max()) + 0.15]
        ax.plot(lims, lims, "k--", lw=1.5, alpha=0.8)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
        ax.set_xlabel("Ground truth", fontsize=12)
        ax.set_ylabel("Posterior mean", fontsize=12)
        ax.set_title(
            "Recovery (all sampled conditions)", fontsize=13, fontweight="bold"
        )
        ax.text(
            0.05,
            0.95,
            f"r = {r_global:.3f}\nRMSE = {rmse_global:.3f}",
            transform=ax.transAxes,
            fontsize=11,
            va="top",
        )

        pooled_title = (
            f"{title_prefix} - Pooled over sampled conditions  "
            f"({n_total:,} rank observations,  KS p = {ks_pval:.3f})"
        )
        plt.suptitle(
            pooled_title,
            fontsize=13,
            fontweight="bold",
            y=1.01,
        )
        plt.tight_layout()
        if "agg" in plt.get_backend().lower():
            plt.close(fig)
        else:  # pragma: no cover - interactive notebook path
            plt.show()

    return {
        "ks_stat": float(ks_stat),
        "ks_pval": float(ks_pval),
        "chi2_stat": float(chi2_stat),
        "chi2_pval": float(chi2_pval),
        "r_global": float(r_global),
        "rmse_global": float(rmse_global),
    }


def _generic_g_required_n(
    g_values: Sequence[float], mcse_target: float, n_pilot: int
) -> tuple[float, int]:
    """``n_sim = S_G^2 / MCSE_target^2`` (Siepe et al. 2024).

    Table 3, read in the 2023 preprint (see ``references.md``).

    Drops non-finite G-values (e.g. a zero-variance EAP vector at small
    ``n_persons`` can make ``corrcoef`` return NaN) rather than letting them
    silently propagate into the variance/MCSE.
    """
    finite = np.asarray(g_values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    s2 = float(np.var(finite, ddof=1)) if len(finite) > 1 else 0.0
    required = int(np.ceil(s2 / mcse_target**2)) if s2 > 0 else n_pilot
    return s2, required


def run_person_theta_sbc(
    simulator_factory: Any,
    approximator: Any,
    *,
    conditions: Sequence[tuple[int, int]],
    n_pilot: int = 8,
    adaptive: bool = True,
    target_mcse_corr: float = 0.01,
    target_mcse_coverage: float = 0.01,
    max_replications: int = 400,
    n_item_samples: int = 500,
    n_plausible_values: int = 300,
    theta_grid_size: int = 31,
    eti_level: float = 0.95,
    seed: int | None = None,
    flow_sample_kwargs: Mapping[str, Any] | None = None,
    model_type: str | None = None,
) -> list[dict[str, Any]]:
    """Person-theta SBC across ``(n_items, n_persons)`` conditions via ``PersonScorer``.

    Replication sizing respects the cluster structure: persons scored within
    one replication share the same simulated item set and item-posterior
    draw, so they are not independent observations. Correlation, coverage,
    precision (mean posterior SD), and RMSE are computed as **per-replication
    statistics** (``G_r``, one value per replication), matching the "generic
    statistic G" design of simulation studies (Siepe, Bartos, Morris,
    Boulesteix, Heck & Pawel 2024; Table 3, read in the 2023 preprint).

    Two sizing modes, selected by ``adaptive``:

    - ``adaptive=True`` (default): a ``n_pilot``-replication pilot estimates
      each statistic's between-replication variance ``S_G^2``, then tops up
      to ``required_n = clip(n_sim, n_pilot, max_replications)`` so the
      achieved MCSE meets ``target_mcse_corr``/``target_mcse_coverage``. This
      is the full validation design -- expensive (up to
      ``max_replications`` per condition) and meant to run once per
      finalized model, not routinely.
    - ``adaptive=False``: exactly ``n_pilot`` replications per condition, no
      top-up. This is the fast "quick sweep" mode for demonstrations,
      trading MCSE guarantees for wall-clock time.

    A condition that hits ``max_replications`` without reaching its target
    MCSE is reported via ``hit_cap=True`` rather than silently accepted.

    Parameters
    ----------
    simulator_factory : Callable[[int, int, int], Any]
        Called as ``simulator_factory(n_items, n_persons, seed)`` to build a
        fixed-size, seeded simulator for one replication.
    approximator : Any
        A trained ``EquivariantIRTApproximator`` (or subclass).
    conditions : Sequence[tuple[int, int]]
        ``(n_items, n_persons)`` pairs to evaluate.
    n_pilot : int
        Pilot replication count (adaptive mode) or fixed replication count
        (non-adaptive mode).
    adaptive : bool
        See above.
    target_mcse_corr, target_mcse_coverage : float
        MCSE targets for correlation/coverage (adaptive mode only).
    max_replications : int
        Upper cap on replications per condition (adaptive mode only).
    n_item_samples, n_plausible_values, theta_grid_size, eti_level :
        Forwarded to ``PersonScorer.score()``.
    seed : int, optional
        Seeds the per-replication simulator RNG.
    flow_sample_kwargs : Mapping, optional
        Forwarded to the approximator's ``sample()`` call inside
        ``PersonScorer`` (e.g. ``{"method": "euler", "steps": 16}`` for the
        a fast, diagnostic-quality solve).
    model_type : str, optional
        Forwarded to ``PersonScorer(approximator, model_type=...)``. PCM and
        GPCM require this explicitly (``"PCM"``/``"GPCM"``); 1PL/2PL can
        leave it ``None`` for auto-detection from the adapter config.

    Returns
    -------
    list of dict
        One entry per condition with ``label``, ``n_items``, ``n_persons``,
        ``n_replications``, ``hit_cap``, per-statistic ``*_g_values`` arrays,
        ``mean_*``, and (adaptive mode only) ``achieved_mcse_*``.
    """
    from .scoring import PersonScorer

    if n_pilot < 1:
        raise ValueError("n_pilot must be >= 1.")
    if max_replications < n_pilot:
        raise ValueError("max_replications must be >= n_pilot.")

    rng = np.random.default_rng(seed)

    # PersonScorer.score() has no flow_sample_kwargs passthrough (it calls
    # approximator.sample(num_samples=..., conditions=...) with no **kwargs),
    # so injecting a fast/diagnostic ODE-solver setting requires temporarily
    # monkeypatching the approximator's own sample(), restored in `finally`
    # even if a replication raises.
    _original_sample = approximator.sample

    def _patched_sample(*args, **kwargs):
        if flow_sample_kwargs is not None:
            kwargs.setdefault("flow_sample_kwargs", flow_sample_kwargs)
        return _original_sample(*args, **kwargs)

    def _run_one_replication(
        n_items_rep: int, n_persons_rep: int
    ) -> tuple[float, float, float, float]:
        rep_seed = int(rng.integers(0, 2**31 - 1))
        rep_simulator = simulator_factory(n_items_rep, n_persons_rep, rep_seed)
        rep_sample = rep_simulator.sample((1,))
        rep_responses = np.asarray(rep_sample["responses"][0], dtype=np.float64)
        rep_true_theta = np.asarray(rep_sample["theta"][0], dtype=np.float64)

        rep_scorer = PersonScorer(approximator, model_type=model_type)
        rep_scores = rep_scorer.score(
            rep_responses,
            n_item_samples=n_item_samples,
            theta_grid_size=theta_grid_size,
            eti_level=eti_level,
            n_plausible_values=n_plausible_values,
        )
        eap = np.asarray(rep_scores.mean, dtype=np.float64)
        corr_g = float(np.corrcoef(rep_true_theta, eap)[0, 1])
        covered = (rep_scores.eti_lower <= rep_true_theta) & (
            rep_true_theta <= rep_scores.eti_upper
        )
        coverage_g = float(np.mean(covered))
        precision_g = float(np.mean(rep_scores.sd))
        rmse_g = float(np.sqrt(np.mean((eap - rep_true_theta) ** 2)))
        return corr_g, coverage_g, precision_g, rmse_g

    condition_results: list[dict[str, Any]] = []
    approximator.sample = _patched_sample
    try:
        for n_items_c, n_persons_c in conditions:
            label = f"I{n_items_c}_P{n_persons_c}"
            corr_vals: list[float] = []
            cov_vals: list[float] = []
            prec_vals: list[float] = []
            rmse_vals: list[float] = []

            for _ in range(n_pilot):
                corr_g, coverage_g, precision_g, rmse_g = _run_one_replication(
                    n_items_c, n_persons_c
                )
                corr_vals.append(corr_g)
                cov_vals.append(coverage_g)
                prec_vals.append(precision_g)
                rmse_vals.append(rmse_g)

            if adaptive:
                _, required_n_corr = _generic_g_required_n(
                    corr_vals, target_mcse_corr, n_pilot
                )
                _, required_n_cov = _generic_g_required_n(
                    cov_vals, target_mcse_coverage, n_pilot
                )
                _, required_n_prec = _generic_g_required_n(
                    prec_vals, target_mcse_corr, n_pilot
                )
                required_n = int(
                    np.clip(
                        max(required_n_corr, required_n_cov, required_n_prec),
                        n_pilot,
                        max_replications,
                    )
                )
                n_topup = max(0, required_n - n_pilot)
                for _ in range(n_topup):
                    corr_g, coverage_g, precision_g, rmse_g = _run_one_replication(
                        n_items_c, n_persons_c
                    )
                    corr_vals.append(corr_g)
                    cov_vals.append(coverage_g)
                    prec_vals.append(precision_g)
                    rmse_vals.append(rmse_g)
                hit_cap = required_n >= max_replications
            else:
                hit_cap = False

            n_used = len(corr_vals)
            finite_corr = np.asarray(corr_vals, dtype=np.float64)
            finite_corr = finite_corr[np.isfinite(finite_corr)]
            finite_cov = np.asarray(cov_vals, dtype=np.float64)
            finite_cov = finite_cov[np.isfinite(finite_cov)]
            finite_prec = np.asarray(prec_vals, dtype=np.float64)
            finite_prec = finite_prec[np.isfinite(finite_prec)]
            finite_rmse = np.asarray(rmse_vals, dtype=np.float64)
            finite_rmse = finite_rmse[np.isfinite(finite_rmse)]

            entry: dict[str, Any] = {
                "label": label,
                "n_items": n_items_c,
                "n_persons": n_persons_c,
                "n_replications": n_used,
                "hit_cap": hit_cap,
                "corr_g_values": finite_corr,
                "coverage_g_values": finite_cov,
                "precision_g_values": finite_prec,
                "rmse_g_values": finite_rmse,
                "mean_corr": float(np.mean(finite_corr))
                if len(finite_corr)
                else float("nan"),
                "mean_coverage": float(np.mean(finite_cov))
                if len(finite_cov)
                else float("nan"),
                "mean_precision_sd": float(np.mean(finite_prec))
                if len(finite_prec)
                else float("nan"),
                "mean_rmse": float(np.mean(finite_rmse))
                if len(finite_rmse)
                else float("nan"),
            }
            if adaptive:
                s2_corr_final = (
                    float(np.var(finite_corr, ddof=1)) if len(finite_corr) > 1 else 0.0
                )
                s2_cov_final = (
                    float(np.var(finite_cov, ddof=1)) if len(finite_cov) > 1 else 0.0
                )
                s2_prec_final = (
                    float(np.var(finite_prec, ddof=1)) if len(finite_prec) > 1 else 0.0
                )
                s2_rmse_final = (
                    float(np.var(finite_rmse, ddof=1)) if len(finite_rmse) > 1 else 0.0
                )
                entry["achieved_mcse_corr"] = math.sqrt(
                    s2_corr_final / max(1, len(finite_corr))
                )
                entry["achieved_mcse_coverage"] = math.sqrt(
                    s2_cov_final / max(1, len(finite_cov))
                )
                entry["achieved_mcse_precision"] = math.sqrt(
                    s2_prec_final / max(1, len(finite_prec))
                )
                entry["achieved_mcse_rmse"] = math.sqrt(
                    s2_rmse_final / max(1, len(finite_rmse))
                )

            condition_results.append(entry)
    finally:
        approximator.sample = _original_sample

    return condition_results


def sample_random_theta_sbc_conditions(
    n_items_range: tuple[int, int],
    n_persons_range: tuple[int, int],
    *,
    n_conditions: int,
    seed: int | None = None,
) -> list[tuple[int, int]]:
    """Draw ``n_conditions`` random ``(n_items, n_persons)`` pairs within range.

    Used by a quick sweep to pick a handful of conditions to spot-check,
    as a cheap alternative to
    ``run_person_theta_sbc``'s full deterministic grid.
    """
    if n_conditions < 1:
        raise ValueError("n_conditions must be >= 1.")
    rng = np.random.default_rng(seed)
    items_lo, items_hi = n_items_range
    persons_lo, persons_hi = n_persons_range
    return [
        (
            int(rng.integers(items_lo, items_hi + 1)),
            int(rng.integers(persons_lo, persons_hi + 1)),
        )
        for _ in range(n_conditions)
    ]
