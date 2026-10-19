"""The HTML dashboard of the model card.

One self-contained file, ``model_card.html``: the record's cells are inlined
as JSON and plain JavaScript (``_card_assets/``) draw the
point plots and per-cell panels from them. No CDN, so the card works offline.
The Markdown card stays the source of truth; every number shown here is in
it too.

Python computes what the browser should not: the simultaneous SBC ECDF band
(Säilynoja, Bürkner & Vehtari, 2022, §2.3, computed exactly) and the
Wilson intervals of the theta bins. Output is deterministic (sorted JSON
keys, no simulation, no render timestamp).
"""

from __future__ import annotations

import base64
import json
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

from .card_render import (
    _FORMULAS,
    MIN_BIN_CLUSTERS,
    MIN_BIN_N,
    REF_LABELS,
    RELIABILITY_LABEL,
    CardInputs,
    _envelope,
    _identity,
    _limitations,
    _or_unrecorded,
    _priors,
    _usage,
    bin_coverage,
    card_metrics_for,
    status_banners,
    verbal_summary,
)

#: The dashboard is refused above this size.
MAX_HTML_BYTES = 2_000_000


def _band_limits(n: int, z: np.ndarray, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    """Scaled band limits ``Bin^-1(gamma/2)`` and ``Bin^-1(1 - gamma/2)``."""
    from scipy.stats import binom

    return binom(n, z).ppf(gamma / 2), binom(n, z).ppf(1 - gamma / 2)


def _band_coverage(n: int, z: np.ndarray, gamma: float) -> float:
    """Probability that a uniform ECDF trajectory stays inside the band.

    The forward recursion of Säilynoja, Bürkner and Vehtari (2022, §2.3):
    ``r_0 = 0`` at ``z_0 = 0``; given ``r_i``, the remaining ``n - r_i``
    ranks are uniform on ``[z_i, 1]``, so ``r_{i+1} - r_i`` is binomial with
    ``n - r_i`` trials and success probability ``(z_{i+1} - z_i) /
    (1 - z_i)``. Mass outside a point's interior (borders count as inside)
    is dropped; what is left at the last point is the coverage.
    """
    from scipy.stats import binom

    lo, hi = _band_limits(n, z, gamma)
    # Only ranks inside the previous interior carry mass: the state is that
    # interior, starting from r_0 = 0.
    ranks, prob = np.array([0]), np.array([1.0])
    prev = 0.0
    for i, zi in enumerate(z):
        p = (zi - prev) / (1.0 - prev)
        nxt = np.arange(int(lo[i]), int(hi[i]) + 1)
        # step[j, m]: from rank ranks[j] to nxt[m], a Binomial(n - r, p) increment.
        step = binom.pmf(nxt[None, :] - ranks[:, None], (n - ranks)[:, None], p)
        ranks, prob = nxt, prob @ step
        prev = zi
    return float(prob.sum())


def sbc_band(num_estimates: int, confidence: float = 0.95) -> dict[str, list[float]]:
    """Simultaneous ECDF band for ``num_estimates`` uniform ranks on SBC_GRID.

    The optimization method of Säilynoja, Bürkner and Vehtari (2022, §2.3),
    evaluated on :data:`card_metrics.SBC_GRID`, the grid the record stores
    the ECDF on. The pointwise level ``gamma`` sets the band
    ``[Bin^-1(gamma/2 | N, z), Bin^-1(1 - gamma/2 | N, z)] / N``; the
    simultaneous coverage of a band is computed exactly by
    :func:`_band_coverage`. Coverage falls in steps as ``gamma`` grows (the
    binomial is discrete), so ``confidence`` is in general not met exactly:
    as in the paper, the ``gamma`` whose coverage is closest to it is used,
    found by bisection on the step. No simulation, so no seed.
    """
    from .card_metrics import SBC_GRID

    n = int(num_estimates)
    z = np.asarray(SBC_GRID, dtype=float)
    # Coverage decreases in gamma: bisect for the step where it crosses.
    a, b = 0.0, 1.0 - confidence
    for _ in range(60):
        mid = (a + b) / 2
        if _band_coverage(n, z, mid) >= confidence:
            a = mid
        else:
            b = mid
    gamma = min((a, b), key=lambda g: abs(_band_coverage(n, z, g) - confidence))
    lo, hi = _band_limits(n, z, gamma)
    return {"grid": z.tolist(), "lower": (lo / n).tolist(), "upper": (hi / n).tolist()}


def _metric(
    key,
    tab,
    label,
    path,
    kind,
    limits,
    guide,
    *,
    offset=0.0,
    index=None,
    se_path=None,
    floor_path=None,
) -> dict[str, Any]:
    return {
        "key": key,
        "tab": tab,
        "label": label,
        "path": path,
        "kind": kind,
        "limits": list(limits),
        "guide": guide,
        "offset": offset,
        "index": index,
        "se_path": se_path,
        "floor_path": floor_path,
    }


#: Note every colour limit carries: a display range, not a threshold.
LIMITS_NOTE = (
    "Colour limits are a display convention chosen for legibility, not a "
    "threshold for acceptable values."
)


#: APA 7 entries (from references.md) cited on the dashboard pages.
_REFS = {
    "birnbaum": (
        "Birnbaum, A. (1968). Some latent trait models and their use in "
        "inferring an examinee's ability. In F. M. Lord & M. R. Novick (Eds.), "
        "*Statistical theories of mental test scores* (pp. 395-479). "
        "Addison-Wesley"
    ),
    "masters": (
        "Masters, G. N. (1982). A Rasch model for partial credit scoring. "
        "*Psychometrika, 47*(2), 149-174. https://doi.org/10.1007/BF02296272"
    ),
    "muraki": (
        "Muraki, E. (1992). A generalized partial credit model: Application of "
        "an EM algorithm. *Applied Psychological Measurement, 16*(2), 159-176. "
        "https://doi.org/10.1177/014662169201600206"
    ),
    "bayesflow": (
        "Kühmichel, L., Huang, J. M., Pratz, V., Arruda, J., Olischläger, H., "
        "Habermann, D., Kucharsky, S., Elsemüller, L., Mishra, A., Bracher, N., "
        "Jedhoff, S., Schmitt, M., Bürkner, P.-C., & Radev, S. T. (2026). "
        "*BayesFlow 2: Multi-backend amortized Bayesian inference in Python*. "
        "arXiv. https://doi.org/10.48550/arXiv.2602.07098"
    ),
    "morris": (
        "Morris, T. P., White, I. R., & Crowther, M. J. (2019). Using "
        "simulation studies to evaluate statistical methods. *Statistics in "
        "Medicine, 38*(11), 2074–2102. https://doi.org/10.1002/sim.8086"
    ),
    "cameron": (
        "Cameron, A. C., & Miller, D. L. (2015). A practitioner's guide to "
        "cluster-robust inference. *Journal of Human Resources, 50*(2), "
        "317–372. https://doi.org/10.3368/jhr.50.2.317"
    ),
    "talts": (
        "Talts, S., Betancourt, M., Simpson, D., Vehtari, A., & Gelman, A. "
        "(2018). *Validating Bayesian inference algorithms with "
        "simulation-based calibration*. arXiv. https://arxiv.org/abs/1804.06788"
    ),
    "sailynoja": (
        "Säilynoja, T., Bürkner, P.-C., & Vehtari, A. (2022). Graphical test "
        "for discrete uniformity and its applications in goodness-of-fit "
        "evaluation and multiple sample comparison. *Statistics and Computing, "
        "32*(2), Article 32. https://doi.org/10.1007/s11222-022-10090-6"
    ),
    "lemos": (
        "Lemos, P., Coogan, A., Hezaveh, Y., & Perreault-Levasseur, L. (2023). "
        "Sampling-based accuracy testing of posterior estimators for general "
        "inference. In *Proceedings of the 40th International Conference on "
        "Machine Learning* (PMLR 202, pp. 19256–19273). "
        "https://proceedings.mlr.press/v202/lemos23a.html"
    ),
    "wilson": (
        "Wilson, E. B. (1927). Probable inference, the law of succession, and "
        "statistical inference. *Journal of the American Statistical "
        "Association, 22*(158), 209–212. "
        "https://doi.org/10.1080/01621459.1927.10502953"
    ),
    "okabe": (
        "Okabe, M., & Ito, K. (2008). *Color universal design (CUD): How to "
        "make figures and presentations that are friendly to colorblind "
        "people*. https://jfly.uni-koeln.de/color/"
    ),
}
#: What each page cites, as (topic, entry); the model page adds its
#: family's source.
_PAGE_REFS = {
    "overview": [
        ("Monte Carlo standard errors", "morris"),
        ("Colour palette", "okabe"),
    ],
    "model": [("Neural posterior estimation (BayesFlow)", "bayesflow")],
    "recovery": [
        ("Monte Carlo standard errors", "morris"),
        ("Posterior contraction", "bayesflow"),
        ("Cluster-robust MCSE of the theta bins", "cameron"),
    ],
    "calibration": [
        ("Simulation-based calibration (SBC)", "talts"),
        ("SBC ECDF difference and its band", "sailynoja"),
        ("TARP", "lemos"),
        ("Monte Carlo standard errors", "morris"),
        ("Wilson intervals of the theta bins", "wilson"),
        ("Cluster-robust intervals of the theta bins", "cameron"),
    ],
}
_FAMILY_REFS = {
    "1pl": "birnbaum",
    "2pl": "birnbaum",
    "pcm": "masters",
    "gpcm": "muraki",
}


def page_references(family: str) -> dict[str, list[dict[str, str]]]:
    """The references at the end of each dashboard page, with their topic."""
    out = {
        page: [{"topic": t, "text": _REFS[k]} for t, k in pairs]
        for page, pairs in _PAGE_REFS.items()
    }
    if family in _FAMILY_REFS:
        out["model"].insert(
            0,
            {
                "topic": f"{family.upper()} response model",
                "text": _REFS[_FAMILY_REFS[family]],
            },
        )
    return out


def prior_width(inp: CardInputs, param: str) -> float | None:
    """Width of the prior's central 95% interval for a parameter.

    The upper colour limit for posterior interval width: a posterior as wide
    as its prior learned nothing from the data. Computed from
    :data:`card_render.PRIOR_DENSITY_DRAWS` seeded draws; ``log_a`` uses the
    log of the ``a`` prior. ``None`` when the prior is not recorded.
    """
    from .card_render import PRIOR_DENSITY_DRAWS
    from .priors import resolve_prior
    from .simulators import _prior_from_record

    role = "a" if param == "log_a" else param
    rec = (inp.spec.get("priors") or {}).get(role)
    if not isinstance(rec, dict):
        return None
    sampler = resolve_prior(_prior_from_record(rec, role), role)
    x = np.asarray(
        sampler((PRIOR_DENSITY_DRAWS,), np.random.default_rng(0)), dtype=float
    ).ravel()
    if param == "log_a":
        x = np.log(x)
    lo, hi = np.quantile(x, [0.025, 0.975])
    return round(float(hi - lo), 2)


def _extreme_share(inp: CardInputs, c: dict[str, Any]) -> list[float | None]:
    """Share of scored persons with an all-extreme pattern, ``[value, None]``."""
    prot = inp.record["protocol"]
    scored = prot["datasets_per_cell"] * min(
        prot["persons_scored"], c["cell"]["n_persons"]
    )
    count = c.get("persons", {}).get("extreme_patterns")
    return [count / scored if count is not None and scored else None, None]


def dashboard_metrics(inp: CardInputs) -> list[dict[str, Any]]:
    """Every heatmap metric, grouped by tab.

    The Markdown card metrics keep their limits; ``sbc_ecdf_max_dev``, the
    reliability variants, TARP and TCC are added. Every limit is fixed and
    independent of the record's values, so cards stay comparable: width
    runs to the prior's own 95% width (:func:`prior_width`), extreme
    patterns are a share of scored persons in [0, 1], and the rest are the
    display conventions of :data:`LIMITS_NOTE`. Coverage uses one symmetric
    scale for every level, so levels compare directly.
    """
    from .card_metrics import CARD_LEVELS

    base = {m.key: m for m in card_metrics_for(inp.parameters)}
    out = []
    for param in [*inp.parameters, "theta"]:
        tab = "persons" if param == "theta" else "items"
        group = ["persons", "theta"] if param == "theta" else ["items", param]
        for level in CARD_LEVELS:
            m = base[f"{param}.coverage_dev_0.95"]
            out.append(
                _metric(
                    f"{param}.coverage_dev_{level}",
                    tab,
                    f"{param}: coverage at {level} minus {level}",
                    [*group, f"coverage_{level}"],
                    "deviation",
                    m.limits,
                    m.guide.replace("95%", f"{level * 100:g}%").replace(
                        "0.95", str(level)
                    ),
                    offset=level,
                )
            )
        for k in ("mean_calibration_error", "nrmse_excess", "corr", "contraction"):
            m = base[f"{param}.{k}"]
            out.append(
                _metric(m.key, tab, m.label, [*group, k], m.kind, m.limits, m.guide)
            )
        out.append(
            _metric(
                f"{param}.sbc_ecdf_max_dev",
                tab,
                f"{param}: SBC ECDF maximum deviation",
                [*group, "sbc_ecdf_max_dev"],
                "magnitude",
                (0.0, 0.25),
                "Largest gap between the ECDF of the posterior ranks and the "
                "uniform CDF on the 39-point grid (Talts et al., 2018). Zero is "
                "uniform.",
            )
        )
        width = prior_width(inp, param)
        out.append(
            _metric(
                f"{param}.width_0.95",
                tab,
                f"{param}: width of the 95% interval",
                [*group, "width_0.95"],
                "magnitude",
                (0.0, width or 1.0),
                "Mean width of the central 95% interval, in the parameter's units. "
                "Narrower is more informative only when coverage is nominal."
                + (f" The prior's own 95% width is {width}." if width else ""),
            )
        )
    rel_guides = {
        "posterior_variance": (
            "magnitude",
            (0.0, 1.0),
            "1 - mean posterior variance / prior variance.",
        ),
        "squared_correlation": (
            "magnitude",
            (0.0, 1.0),
            "Squared correlation of posterior mean and true "
            "theta over the scored persons.",
        ),
        "difference": (
            "deviation",
            (-0.5, 0.5),
            "Posterior-variance reliability minus squared correlation; "
            "zero when the two agree.",
        ),
        "information": (
            "magnitude",
            (0.0, 1.0),
            "Information-based marginal reliability "
            "sigma^2 / (sigma^2 + MSEM), MSEM = E[1 / J(theta)] "
            "over the theta prior (Lee, 2026, Eq. A.5), from the "
            "generating items.",
        ),
    }
    for name, (kind, limits, guide) in rel_guides.items():
        out.append(
            _metric(
                f"theta.reliability_{name}",
                "persons",
                f"theta: reliability, {name} ({RELIABILITY_LABEL})",
                ["persons", "reliability", name, "mean"],
                kind,
                limits,
                f"{guide} {RELIABILITY_LABEL}.",
            )
        )
    out.append(
        _metric(
            "theta.extreme_share",
            "persons",
            "theta: share of all-extreme response patterns",
            ["derived", "extreme_share"],
            "magnitude",
            (0.0, 1.0),
            "Share of scored persons whose responses are all lowest or all highest.",
        )
    )
    if not inp.joint_omitted:
        for ref in ("random", "classical"):
            out.append(
                _metric(
                    f"joint.tarp_{ref}",
                    "joint",
                    f"joint: TARP error, {REF_LABELS[ref]} references",
                    ["joint", "tarp", ref, "tarp_error"],
                    "magnitude",
                    (0.0, 0.25),
                    "Deviation of the TARP coverage curve from the diagonal, jointly "
                    "over all items. An exact posterior still scores about the "
                    "cell's null floor (finite draws and datasets), shown with each "
                    "value, not zero."
                    + (
                        " Random references cannot detect an estimator that returns "
                        "the prior (Lemos et al., 2023, Sec. 4.3); read it with the "
                        "CTT-statistic-reference error."
                        if ref == "random"
                        else " Data-dependent references from classical test "
                        "theory item "
                        "statistics detect a posterior "
                        "that ignores the data (Lemos et al., 2023, Sec. 4.3)."
                    ),
                    floor_path=["joint", "tarp", ref, "null_floor"],
                )
            )
        thetas = next(
            (
                c["joint"]["tcc"]["thetas"]
                for c in inp.evaluated()
                if c.get("joint", {}).get("tcc")
            ),
            [],
        )
        for k, theta in enumerate(thetas):
            out.append(
                _metric(
                    f"joint.tcc_bias_{theta:g}",
                    "joint",
                    f"joint: TCC bias at theta = {theta:g}",
                    ["joint", "tcc", "d_mean"],
                    "deviation",
                    (-0.05, 0.05),
                    "Posterior-mean test characteristic curve minus the true one at "
                    "this theta, divided by the number of items (proportion-correct "
                    "scale; shown as a point, not a curve). Zero is unbiased.",
                    index=k,
                    se_path=["joint", "tcc", "d_mean_se"],
                )
            )
    calibration = ("coverage_dev_", "mean_calibration_error", "sbc_ecdf", "tarp_")
    for m in out:
        # Dashboard page: does the posterior's uncertainty match its error
        # (calibration), or how close and how precise is it (recovery)?
        m["axis"] = (
            "calibration" if any(k in m["key"] for k in calibration) else "recovery"
        )
    return out


def _bins(c: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for b in c.get("persons", {}).get("conditional") or []:
        out.append(
            {
                "edges": b["edges"],
                "n": int(b.get("n", 0)),
                "clusters": b.get("clusters"),
                "bias": b.get("bias"),
                **bin_coverage(b),
            }
        )
    return out


def _png(path: Path) -> str | None:
    if not path.is_file():
        return None
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def _md_table_rows(table: str) -> list[list[str]]:
    rows = [r.strip().strip("|").split(" | ") for r in table.splitlines()[2:]]
    return [[cell.strip().strip("`") for cell in r] for r in rows]


#: Bookkeeping in a serialized Keras config that says nothing about the
#: architecture.
_CONFIG_SKIP = {
    "name",
    "trainable",
    "dtype",
    "seed_generator",
    "build_config",
    "compile_config",
}


def _config_rows(cfg: dict[str, Any], prefix: str = "") -> list[list[str]]:
    """``[key, value]`` rows of one serialized object's config, flattened.

    A nested Keras object shows as its class name, followed by its own
    settings under a dotted prefix; a list of objects as their class names,
    then each one's settings under ``key[n].``;
    tensors (``__tensor__``) are left out.
    """
    rows = []
    for k, v in cfg.items():
        if k in _CONFIG_SKIP:
            continue
        key = prefix + k
        if isinstance(v, dict) and "class_name" in v:
            if v["class_name"] == "__tensor__":
                continue
            rows.append([key, str(v["class_name"])])
            rows += _config_rows(v.get("config") or {}, key + ".")
        elif isinstance(v, dict):
            rows += _config_rows(v, key + ".")
        elif (
            isinstance(v, list)
            and v
            and all(isinstance(x, dict) and "class_name" in x for x in v)
        ):
            rows.append([key, ", ".join(str(x["class_name"]) for x in v)])
            for n, x in enumerate(v):
                rows += _config_rows(x.get("config") or {}, f"{key}[{n}].")
        else:
            text = (
                json.dumps(v)
                if isinstance(v, (list, dict, bool))
                else "None"
                if v is None
                else str(v)
            )
            rows.append([key, text])
    return rows


def network_groups(record: dict[str, Any]) -> list[dict[str, Any]] | None:
    """The record's full network config as titled tables, or None if absent.

    Groups: the approximator's own settings, the summary network, the
    inference network and the adapter.
    """
    net = record.get("network")
    if not net or not isinstance(net.get("config"), dict):
        return None
    top = net["config"]
    cfg = top.get("config") or {}
    parts = {
        "summary_network": "Summary network",
        "inference_network": "Inference network",
        "adapter": "Adapter",
    }
    own = {k: v for k, v in cfg.items() if k not in parts}
    rows = _config_rows(own)
    if net.get("n_parameters") is not None:
        rows.insert(0, ["n_parameters", f"{net['n_parameters']:,}"])
    out = [{"title": f"Approximator ({top.get('class_name')})", "rows": rows}]
    for k, title in parts.items():
        sub = cfg.get(k)
        if isinstance(sub, dict) and "class_name" in sub:
            out.append(
                {
                    "title": f"{title} ({sub['class_name']})",
                    "rows": _config_rows(sub.get("config") or {}),
                }
            )
    out.append({"source": net.get("source")})
    return out


def prior_density_curves(inp: CardInputs, bins: int = 80) -> list[dict[str, Any]]:
    """Histogram densities of each recorded prior, for the page to draw.

    The same seeded draws as the PNG figure (``PRIOR_DENSITY_DRAWS``, seed 0)
    over the central 99.8%, so any prior the simulators resolve works, not
    only ones with a closed-form density. Also the central 95% interval.
    """
    from .card_render import PRIOR_DENSITY_DRAWS
    from .priors import resolve_prior
    from .simulators import _prior_from_record

    out = []
    for role, rec in (inp.spec.get("priors") or {}).items():
        if not isinstance(rec, dict) or role not in ("a", "b", "theta", "delta"):
            continue
        sampler = resolve_prior(_prior_from_record(rec, role), role)
        x = np.asarray(sampler((PRIOR_DENSITY_DRAWS,), np.random.default_rng(0)))
        lo, hi = np.quantile(x, [0.001, 0.999])
        dens, edges = np.histogram(x, bins=bins, range=(lo, hi), density=True)
        out.append(
            {
                "role": role,
                "label": f"{role} ~ {rec['name']}("
                + ", ".join(f"{k} = {v}" for k, v in rec.items() if k != "name")
                + ")",
                "x": ((edges[:-1] + edges[1:]) / 2).round(5).tolist(),
                "y": dens.round(5).tolist(),
                "interval": np.quantile(x, [0.025, 0.975]).round(4).tolist(),
            }
        )
    return out


def _size_bins(values: Any, probs: Any, max_bins: int) -> dict[str, list]:
    """One axis of the training-size map, in at most ``max_bins`` log bins.

    Each bin keeps its sizes' range and their mean probability (per size,
    so the map's unit stays probability per size pair); edges are geometric
    midpoints between neighbouring bins.
    """
    v = np.asarray(values, dtype=float)
    p = np.asarray(probs, dtype=float)
    lv = np.log(v)
    if v.size > max_bins:
        cuts = np.linspace(lv[0], lv[-1], max_bins + 1)
        idx = np.clip(np.searchsorted(cuts, lv, side="right") - 1, 0, max_bins - 1)
        groups = [
            g for g in (np.flatnonzero(idx == b) for b in range(max_bins)) if g.size
        ]
    else:
        groups = [np.array([i]) for i in range(v.size)]
    lo = np.array([lv[g[0]] for g in groups])
    hi = np.array([lv[g[-1]] for g in groups])
    mids = (hi[:-1] + lo[1:]) / 2
    pad = (mids[0] - lo[0]) if mids.size else 0.1
    pad_hi = (hi[-1] - mids[-1]) if mids.size else 0.1
    edges = np.concatenate(
        [[lo[0] - max(pad, 0.05)], mids, [hi[-1] + max(pad_hi, 0.05)]]
    )
    return {
        "edges": np.exp(edges).round(3).tolist(),
        "lo": [int(v[g[0]]) for g in groups],
        "hi": [int(v[g[-1]]) for g in groups],
        "p": [float(p[g].mean()) for g in groups],
    }


def trained_size_map(inp: CardInputs, max_bins: int = 60) -> dict[str, Any] | None:
    """The training-size map for the page to draw, or ``None``.

    Outer product of the per-axis training distributions
    (:func:`bayesflow_irt.card_render.trained_marginals`), each axis in at
    most ``max_bins`` log bins, plus the card's evaluation grid.
    """
    from .card_render import trained_marginals

    marginals = trained_marginals(inp)
    if marginals is None:
        return None
    return {
        "distribution": marginals["distribution"],
        "items": _size_bins(*marginals["items"], max_bins),
        "persons": _size_bins(*marginals["persons"], max_bins),
        "grid": {
            "items": [int(i) for i in inp.items_axis],
            "persons": [int(p) for p in inp.persons_axis],
        },
    }


def dashboard_payload(inp: CardInputs, fig_dir: Path | None = None) -> dict[str, Any]:
    """Everything the page script reads, as one JSON-serializable dict."""
    from .card_metrics import CARD_LEVELS

    rec, md, spec = inp.record, inp.metadata, inp.spec
    family = str(spec.get("model", "")).lower()
    ck = rec.get("checkpoint") or {}
    cells = []
    for c in rec["cells"]:
        entry = {k: v for k, v in c.items() if k not in ("seed",)}
        # Per-dataset pairs stay in evaluation.json until a plot uses them.
        for group in ("items", "persons"):
            if isinstance(entry.get(group), dict):
                entry[group] = {
                    p: (
                        {k: x for k, x in g.items() if k != "pairs"}
                        if isinstance(g, dict)
                        else g
                    )
                    for p, g in entry[group].items()
                }
        entry["bins"] = _bins(c)
        if "failed" not in c:
            entry["derived"] = {"extreme_share": _extreme_share(inp, c)}
        cells.append(entry)
    figures = {}
    if fig_dir is not None:
        for name in ("priors", "coverage_curves", "tarp_curves"):
            uri = _png(fig_dir / f"{name}.png")
            if uri:
                figures[name] = uri
    return {
        "title": f"Model card: {_or_unrecorded(spec.get('model'))} "
        f"({_or_unrecorded(spec.get('variant'))})",
        "identity": _md_table_rows(_identity(inp)),
        "pill": _or_unrecorded(ck.get("match")),
        "banners": status_banners(inp),
        "joint_omitted": inp.joint_omitted,
        "strip": {
            "preset": rec["protocol"].get("preset_name") or "custom",
            "datasets_per_cell": rec["protocol"]["datasets_per_cell"],
            "posterior_draws": rec["protocol"]["posterior_draws"],
            "draws_validated_for": rec.get("draws_validated_for") or [],
            "evaluated": sum(1 for c in rec["cells"] if "failed" not in c),
            "failed": sum(1 for c in rec["cells"] if "failed" in c),
        },
        "summary": verbal_summary(inp),
        "envelope": _md_table_rows(_envelope(inp)),
        "priors": _md_table_rows(_priors(inp)),
        "formula": _FORMULAS.get(family, "not recorded"),
        "family": family,
        "references": page_references(family),
        "prior_densities": prior_density_curves(inp),
        "trained_sizes": trained_size_map(inp),
        "network": network_groups(rec),
        "parameterization": _or_unrecorded(spec.get("internal_parameterization")),
        "limitations": _limitations(inp),
        "usage": "\n".join(_usage(inp).splitlines()[1:-1]),
        "notes": (md.get("intended_use") or {}).get("notes"),
        "training": md.get("training") or {},
        "architecture": md.get("architecture") or {},
        "provenance": {
            "protocol": rec["protocol"],
            "session_info": rec.get("session_info"),
            "sidecar_provenance": md.get("provenance") or {},
            "seconds": rec.get("seconds"),
        },
        "axes": {
            "items": inp.items_axis,
            "persons": inp.persons_axis,
            "in_items": inp.in_items,
            "in_persons": inp.in_persons,
        },
        "include_out_of_range": bool(rec["protocol"].get("include_out_of_range")),
        "parameters": [*inp.parameters, "theta"],
        "levels": list(CARD_LEVELS),
        "metrics": dashboard_metrics(inp),
        "band": sbc_band(rec["protocol"]["datasets_per_cell"]),
        "min_bin_n": MIN_BIN_N,
        "min_bin_clusters": MIN_BIN_CLUSTERS,
        "reliability_label": RELIABILITY_LABEL,
        "comparisons": [r.to_payload() for r in inp.references],
        "figures": figures,
        "cells": cells,
    }


def _asset(name: str) -> str:
    return (
        resources.files("bayesflow_irt")
        .joinpath("_card_assets", name)
        .read_text(encoding="utf-8")
    )


def render_html(inp: CardInputs, fig_dir: Path | None = None) -> str:
    """The dashboard page as a string."""
    payload = json.dumps(
        _finite(dashboard_payload(inp, fig_dir)),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    )
    # Escape every <, > and & as JSON unicode escapes: "</script" would end
    # the element and "<!--<script" would keep it from ending (WHATWG HTML,
    # script data escaped states). JSON.parse restores the characters.
    payload = (
        payload.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    )
    head_title = f"{_or_unrecorded(inp.spec.get('model'))} model card"
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{_escape(head_title)}</title>\n<style>\n{_asset('dashboard.css')}"
        '</style>\n</head>\n<body>\n<div id="app"></div>\n'
        f'<script type="application/json" id="card-data">{payload}</script>\n'
        f"<script>\n{_asset('dashboard.js')}</script>\n</body>\n</html>\n"
    )


def _finite(v: Any) -> Any:
    """NaN and infinity become null: JSON has no literal for them."""
    if isinstance(v, float):
        return v if np.isfinite(v) else None
    if isinstance(v, dict):
        return {k: _finite(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_finite(x) for x in v]
    return v


def _escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_html(inp: CardInputs, card_dir: Path, fig_dir: Path | None) -> Path:
    """Write ``model_card.html``; refuse a file over :data:`MAX_HTML_BYTES`."""
    text = render_html(inp, fig_dir)
    size = len(text.encode("utf-8"))
    if size > MAX_HTML_BYTES:
        raise ValueError(
            f"model_card.html would be {size / 1e6:.2f} MB, over the "
            f"{MAX_HTML_BYTES / 1e6:.0f} MB cap."
        )
    out = Path(card_dir) / "model_card.html"
    out.write_text(text, encoding="utf-8")
    return out
