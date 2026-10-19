"""Per-parameter model-card metrics with Monte Carlo standard errors.

One observation per simulated dataset: the posterior draws and the true value
of one randomly chosen item parameter (or person), so observations are
independent across datasets.

MCSEs follow Siepe et al. (2024; Table 3, read in the 2023 preprint, adapted
from Morris et al., 2019, Table 6): analytic where the table gives a formula
(bias, RMSE, coverage, mean interval width), and a delete-one jackknife over
datasets otherwise (correlation, NRMSE, the calibration errors and the SBC
deviation), which the same text names as the more accurate general choice.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .sbc import _compute_recovery_metrics

#: Central-interval levels reported individually on the card.
CARD_LEVELS: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95, 0.99, 0.995)

#: BayesFlow's calibration-error levels (``compute_calibration_error``
#: defaults: 20 levels evenly spaced in [0.005, 0.995]).
CALIBRATION_LEVELS = np.linspace(0.005, 0.995, 20)

#: Levels at which the SBC rank ECDF is compared with the uniform CDF.
SBC_GRID = np.arange(1, 40) / 40

Metric = tuple[float, float]


def jackknife_se(loo: np.ndarray) -> float:
    """Delete-one jackknife standard error from leave-one-out estimates."""
    loo = np.asarray(loo, dtype=float)
    n = loo.shape[0]
    return float(np.sqrt((n - 1) / n * np.sum((loo - loo.mean()) ** 2)))


def _loo_means(ind: np.ndarray) -> np.ndarray:
    """Leave-one-out column means of ``(n, L)`` values, shape ``(n, L)``."""
    n = ind.shape[0]
    return (ind.sum(axis=0, keepdims=True) - ind) / (n - 1)


def _central_intervals(
    draws: np.ndarray, levels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper bounds ``(n, L)`` of central posterior intervals."""
    lo = np.quantile(draws, (1 - levels) / 2, axis=1).T
    hi = np.quantile(draws, (1 + levels) / 2, axis=1).T
    return lo, hi


def _mean_with_se(values: np.ndarray) -> Metric:
    n = values.shape[0]
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(n))


def _jackknife(values: np.ndarray, stat: Any) -> Metric:
    """``stat(values)`` with its delete-one jackknife SE over rows."""
    n = values.shape[0]
    keep = np.ones(n, dtype=bool)
    loo = np.empty(n)
    for i in range(n):
        keep[i] = False
        loo[i] = stat(values[keep])
        keep[i] = True
    return float(stat(values)), jackknife_se(loo)


def _contraction(post_var: np.ndarray, truth: np.ndarray) -> Metric:
    """``1 - mean(post_var) / var(truth)`` with a delete-one jackknife SE.

    The prior variance is estimated from the same datasets' true values, so
    the jackknife leaves a dataset out of both the numerator and the
    denominator.
    """
    n = truth.shape[0]

    def stat(keep: np.ndarray) -> float:
        prior = truth[keep].var(ddof=1)
        return 1.0 - float(post_var[keep].mean()) / prior if prior > 0 else np.nan

    full = stat(np.ones(n, dtype=bool))
    keep = np.ones(n, dtype=bool)
    loo = np.empty(n)
    for i in range(n):
        keep[i] = False
        loo[i] = stat(keep)
        keep[i] = True
    if not (np.isfinite(full) and np.all(np.isfinite(loo))):
        return (float("nan"), float("nan"))
    return (float(full), jackknife_se(loo))


def parameter_metrics(draws: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    """All per-parameter card metrics, as ``(value, mcse)`` plus the SBC ECDF.

    Parameters
    ----------
    draws : np.ndarray
        Posterior draws ``(n_datasets, n_draws)``, one row per dataset.
    truth : np.ndarray
        True values ``(n_datasets,)``.

    Returns
    -------
    dict
        ``bias``, ``rmse``, ``corr``, ``nrmse``, ``nrmse_excess`` (point
        estimate: the posterior mean; definitions as
        :func:`bayesflow_irt.sbc._compute_recovery_metrics`),
        ``coverage_<level>`` for each of :data:`CARD_LEVELS`,
        ``width_0.95`` (mean width of the central 95% interval),
        ``width_0.95_sd`` (its standard deviation over datasets, jackknife
        SE; recorded from evaluation format 3 on, not yet shown),
        ``posterior_variance`` (mean posterior variance) and
        ``truth_variance`` (variance of the true values, jackknife SE), the
        two parts of ``contraction``,
        ``z_mean`` and ``z_sd`` (mean and standard deviation over datasets
        of the posterior z-score ``(posterior mean - truth) / posterior
        sd``, as BayesFlow's ``z_score_contraction`` defines it after Schad,
        Betancourt & Vasishth, 2021; calibrated: mean 0, sd 1),
        ``contraction`` (``1 - mean posterior variance / variance of the
        true values``: BayesFlow's ``posterior_contraction`` with the
        dataset-wise ratios averaged instead of medianed and not clipped to
        [0, 1], so the value has a jackknife MCSE and an over-dispersed
        posterior shows as negative),
        ``mean_calibration_error`` and ``calibration_error`` (mean and median
        over :data:`CALIBRATION_LEVELS` of ``|coverage - level|``, as
        :func:`bayesflow_irt.sbc.compute_calibration_error`) and
        ``sbc_ecdf_max_dev`` (``max |ECDF(u) - alpha|`` over :data:`SBC_GRID`
        of the posterior fractions ``u`` below the truth; Talts et al., 2018),
        each ``(value, mcse)``; and ``sbc_ecdf``, the ECDF of ``u`` on
        :data:`SBC_GRID` (a list), for the ECDF-difference plot;
        ``calibration_curve``, the coverage and its MCSE at each of
        :data:`CALIBRATION_LEVELS` (``levels``, ``coverage``, ``mcse``
        lists), for the calibration detail plot; and
        ``pairs``, one row per dataset (``truth``, ``mean``, ``sd``,
        ``q025``, ``q975``, rounded to 4 significant digits), for
        true-against-estimate plots. A metric
        that is undefined on the data (for example a correlation of
        constant values) is ``(nan, nan)``.

    Notes
    -----
    Finite draws make the central intervals slightly too narrow, about in
    proportion to ``1 / n_draws``, so tail coverage is biased low.
    """
    draws = np.asarray(draws, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if draws.ndim != 2 or truth.shape != (draws.shape[0],):
        raise ValueError(
            "draws must be (n_datasets, n_draws) and truth (n_datasets,); got "
            f"{draws.shape} and {truth.shape}."
        )
    n = draws.shape[0]
    if n < 3:
        raise ValueError(f"Need at least 3 datasets for an MCSE; got {n}.")

    out: dict[str, Any] = {}
    estimate = draws.mean(axis=1)
    error = estimate - truth
    out["bias"] = _mean_with_se(error)
    sq = error**2
    mse = float(sq.mean())
    rmse = float(np.sqrt(mse))
    rmse_se = (
        float(np.sqrt(sq.var(ddof=1) / (4 * n * mse))) if mse > 0 else float("nan")
    )
    out["rmse"] = (rmse, rmse_se)

    full = _compute_recovery_metrics(truth, estimate)
    keep = np.ones(n, dtype=bool)
    loo = {k: np.empty(n) for k in ("corr", "nrmse", "nrmse_excess")}
    for i in range(n):
        keep[i] = False
        rec = _compute_recovery_metrics(truth[keep], estimate[keep])
        keep[i] = True
        for k in loo:
            loo[k][i] = rec[k]
    for k, values in loo.items():
        ok = np.isfinite(full[k]) and np.all(np.isfinite(values))
        out[k] = (full[k], jackknife_se(values)) if ok else (float("nan"),) * 2

    levels = np.asarray(CARD_LEVELS)
    lo, hi = _central_intervals(draws, levels)
    covered = ((lo <= truth[:, None]) & (truth[:, None] <= hi)).astype(float)
    for j, level in enumerate(CARD_LEVELS):
        c = float(covered[:, j].mean())
        out[f"coverage_{level}"] = (c, float(np.sqrt(c * (1 - c) / n)))
    width = (hi - lo)[:, CARD_LEVELS.index(0.95)]
    out["width_0.95"] = _mean_with_se(width)
    # Spread of the width over datasets, kept for later use (not shown).
    keep = np.ones(n, dtype=bool)
    sd_loo = np.empty(n)
    for i in range(n):
        keep[i] = False
        sd_loo[i] = width[keep].std(ddof=1)
        keep[i] = True
    out["width_0.95_sd"] = (float(width.std(ddof=1)), jackknife_se(sd_loo))
    post_var = draws.var(axis=1, ddof=1)
    out["contraction"] = _contraction(post_var, truth)
    out["posterior_variance"] = _mean_with_se(post_var)
    out["truth_variance"] = _jackknife(truth, lambda t: float(t.var(ddof=1)))
    sd = np.sqrt(post_var)
    z = np.where(sd > 0, (estimate - truth) / np.where(sd > 0, sd, 1.0), np.nan)
    if np.all(np.isfinite(z)):
        out["z_mean"] = _mean_with_se(z)
        out["z_sd"] = _jackknife(z, lambda v: float(v.std(ddof=1)))
    else:
        out["z_mean"] = out["z_sd"] = (float("nan"), float("nan"))
    q = np.quantile(draws, [0.025, 0.975], axis=1)
    out["pairs"] = {k: [float(f"{x:.4g}") for x in v] for k, v in (
        ("truth", truth), ("mean", estimate), ("sd", sd),
        ("q025", q[0]), ("q975", q[1]))}

    lo, hi = _central_intervals(draws, CALIBRATION_LEVELS)
    cal = ((lo <= truth[:, None]) & (truth[:, None] <= hi)).astype(float)
    dev = np.abs(cal.mean(axis=0) - CALIBRATION_LEVELS)
    dev_loo = np.abs(_loo_means(cal) - CALIBRATION_LEVELS)
    out["mean_calibration_error"] = (
        float(dev.mean()), jackknife_se(dev_loo.mean(axis=1))
    )
    out["calibration_error"] = (
        float(np.median(dev)), jackknife_se(np.median(dev_loo, axis=1))
    )
    # Coverage at each of those levels, for the detail plot; the MCSE of a
    # mean over datasets (equal to its jackknife SE).
    out["calibration_curve"] = {
        "levels": [round(float(v), 6) for v in CALIBRATION_LEVELS],
        "coverage": cal.mean(axis=0).tolist(),
        "mcse": (cal.std(axis=0, ddof=1) / np.sqrt(cal.shape[0])).tolist(),
    }

    u = (draws < truth[:, None]).mean(axis=1)
    below = (u[:, None] <= SBC_GRID).astype(float)
    ecdf = below.mean(axis=0)
    out["sbc_ecdf_max_dev"] = (
        float(np.max(np.abs(ecdf - SBC_GRID))),
        jackknife_se(np.max(np.abs(_loo_means(below) - SBC_GRID), axis=1)),
    )
    # The curve itself, on SBC_GRID, so the ECDF-difference plot can be
    # drawn from the stored record without resampling.
    out["sbc_ecdf"] = ecdf.tolist()
    return out
