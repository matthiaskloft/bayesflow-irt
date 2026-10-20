"""Render a model card from ``evaluation.json`` and the metadata sidecar.

The Markdown
card, the source of truth, and its PNG figures. The Markdown holds every
number of the record that the card reports, including per-cell appendix
tables; the HTML dashboard adds navigation over the same record.

Figures need matplotlib (the ``[card]`` extra). Without it the Markdown is
written with tables only and a warning says the figures were skipped.

Colour encodes magnitude, never quality: coverage is shown
as the signed deviation from the nominal level, on a symmetric scale fixed
across levels and cells; other metrics use a single-hue scale with fixed
limits. Every limit is printed in the figure's colour bar.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np

#: Format of ``evaluation.json`` this renderer reads.
READS_EVALUATION_FORMAT = (1, 2, 3)

#: Display name of each TARP reference rule; the record keys stay
#: ``random`` and ``classical``. The classical references are classical
#: test theory item statistics (:func:`bayesflow_irt.sbc.classical_reference_points`).
REF_LABELS = {"random": "random", "classical": "CTT-statistic"}

#: Draws used to plot each prior density (seeded, so figures are stable).
PRIOR_DENSITY_DRAWS = 200_000

#: Theta bins with fewer persons are marked; a display rule, not a flag.
MIN_BIN_N = 10

#: Format-2 theta bins with fewer contributing datasets are marked: the
#: cluster-robust SE needs many clusters, about 30 as a guideline (Bland &
#: Kanaan, 2021, Sec. 10.6; Cameron & Miller, 2015, Sec. VI). A display rule.
MIN_BIN_CLUSTERS = 30

#: Label every reliability value carries until the card is compared with a
#: reference estimator.
RELIABILITY_LABEL = "context, not yet compared with a reference"

#: Finite-draw residual, stated next to tail coverage.
TAIL_COVERAGE_NOTE = (
    "With finite posterior draws, ETIs come out slightly too narrow, so "
    "coverage at the extreme levels (0.99, 0.995) is biased low in every cell."
)

Pair = Sequence[float | None] | None


@dataclass(frozen=True)
class CardMetric:
    """One heatmap metric: where it lives in a cell and how it is coloured."""

    key: str
    label: str
    getter: Callable[[dict[str, Any]], Pair]
    kind: str  # "deviation" (symmetric around 0) or "magnitude"
    limits: tuple[float, float]
    guide: str


def _param_getter(param: str, key: str) -> Callable[[dict[str, Any]], Pair]:
    def get(cell: dict[str, Any]) -> Pair:
        group = cell.get("persons", {}) if param == "theta" else cell.get("items", {})
        return group.get(param, {}).get(key)

    return get


def _coverage_deviation(param: str, level: float) -> Callable[[dict[str, Any]], Pair]:
    get = _param_getter(param, f"coverage_{level}")

    def dev(cell: dict[str, Any]) -> Pair:
        pair = get(cell)
        if pair is None or pair[0] is None:
            return None
        return (pair[0] - level, pair[1])

    return dev


def _reliability(cell: dict[str, Any]) -> Pair:
    rel = cell.get("persons", {}).get("reliability", {})
    return rel.get("posterior_variance", {}).get("mean")


def _tarp(ref: str) -> Callable[[dict[str, Any]], Pair]:
    def get(cell: dict[str, Any]) -> Pair:
        return cell.get("joint", {}).get("tarp", {}).get(ref, {}).get("tarp_error")

    return get


def card_metrics_for(
    parameters: Sequence[str], *, joint: bool = True
) -> list[CardMetric]:
    """The heatmap metrics of the Markdown card, for the model's parameters.

    ``joint=False`` drops the TARP metrics, for records whose joint metrics
    were omitted (PCM/GPCM).
    """
    out = []
    for param in [*parameters, "theta"]:
        out += [
            CardMetric(
                f"{param}.coverage_dev_0.95",
                f"{param}: coverage at 0.95 minus 0.95",
                _coverage_deviation(param, 0.95),
                "deviation",
                (-0.15, 0.15),
                "Share of true values inside the central 95% interval, minus "
                "0.95. Zero is nominal; below zero, fewer true values fall "
                "inside the interval than nominal.",
            ),
            CardMetric(
                f"{param}.mean_calibration_error",
                f"{param}: mean calibration error",
                _param_getter(param, "mean_calibration_error"),
                "magnitude",
                (0.0, 0.25),
                "Mean absolute gap between nominal and observed coverage of "
                "central intervals, over 20 levels from 0.005 to 0.995. Zero "
                "is nominal.",
            ),
            CardMetric(
                f"{param}.nrmse_excess",
                f"{param}: NRMSE excess",
                _param_getter(param, "nrmse_excess"),
                "deviation",
                (-0.25, 0.25),
                "NRMSE of the posterior mean minus sqrt(1 - r^2), the NRMSE an "
                "exact posterior mean reaches at the observed correlation r. "
                "Zero matches that ideal; a positive value is spread "
                "(mis-scaling) beyond what r implies, not mis-ranking.",
            ),
            CardMetric(
                f"{param}.corr",
                f"{param}: correlation",
                _param_getter(param, "corr"),
                "magnitude",
                (0.0, 1.0),
                "Pearson correlation between the posterior mean and the true "
                "value across datasets. One is a perfect linear ordering.",
            ),
            CardMetric(
                f"{param}.contraction",
                f"{param}: posterior contraction",
                _param_getter(param, "contraction"),
                "magnitude",
                (0.0, 1.0),
                "One minus the mean posterior variance over the variance of the "
                "true values (BayesFlow's posterior_contraction, averaged over "
                "datasets). One: the data pin the parameter down; zero: the "
                "posterior is as wide as the prior. Recorded from evaluation "
                "format 3 on.",
            ),
        ]
    out += [
        CardMetric(
            "theta.reliability",
            f"theta: reliability, posterior variance ({RELIABILITY_LABEL})",
            _reliability,
            "magnitude",
            (0.0, 1.0),
            "1 - mean posterior variance / prior variance, over the scored "
            f"persons; {RELIABILITY_LABEL}.",
        ),
    ]
    if joint:
        out += [
            CardMetric(
                "joint.tarp_error",
                "joint: TARP error, random references",
                _tarp("random"),
                "magnitude",
                (0.0, 0.25),
                "Deviation of the TARP coverage curve from the diagonal, jointly "
                "over all items. Zero is nominal. Random references cannot "
                "detect an estimator that returns the prior (Lemos et al., "
                "2023, Sec. 4.3); read it with the CTT-statistic-reference error "
                "below.",
            ),
            CardMetric(
                "joint.tarp_error_classical",
                "joint: TARP error, CTT-statistic references",
                _tarp("classical"),
                "magnitude",
                (0.0, 0.25),
                "As above, with data-dependent classical test theory item "
                "statistics as references, which detect a posterior that "
                "ignores the data.",
            ),
        ]
    return out


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------


def _num(v: Any, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "–"
    if isinstance(v, bool) or isinstance(v, int):
        return str(v)
    return f"{v:.{digits}f}"


def _pm(pair: Pair, digits: int = 3) -> str:
    """``value (MCSE)``, or a dash."""
    if pair is None or pair[0] is None:
        return "–"
    se = pair[1] if len(pair) > 1 else None
    return f"{_num(pair[0], digits)} ({_num(se, digits)})"


def _or_unrecorded(v: Any) -> str:
    return "not recorded" if v in (None, "") else str(v)


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def wilson_interval(hits: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Uses :meth:`scipy.stats.binomtest` ``.proportion_ci(method="wilson")``,
    which follows Wilson (1927) without continuity correction (SciPy
    documentation; Newcombe, 1998). Unlike ``p ± z SE`` it stays inside
    [0, 1] and has non-zero width at 0 or ``n`` hits.
    """
    from scipy.stats import binomtest

    ci = binomtest(int(hits), int(n)).proportion_ci(level, method="wilson")
    return float(ci.low), float(ci.high)


def bin_coverage(b: Mapping[str, Any]) -> dict[str, Any]:
    """Coverage of one theta bin with its 95% interval and small flag.

    Format 1 (one random person per dataset, independent): Wilson interval,
    small below :data:`MIN_BIN_N` persons. Format 2 (all scored persons,
    ``clusters`` recorded): ``p ± 1.96 SE`` with the record's cluster-robust
    SE, clipped to [0, 1], small below :data:`MIN_BIN_CLUSTERS` datasets;
    Wilson assumes independent persons, which no longer holds.
    """
    n = int(b.get("n", 0))
    cov = b.get("coverage_0.95")
    clustered = "clusters" in b
    small = (b["clusters"] < MIN_BIN_CLUSTERS) if clustered else n < MIN_BIN_N
    out = {
        "coverage": None,
        "interval": None,
        "small": small,
        "kind": "cluster-robust" if clustered else "Wilson",
    }
    if cov is None or cov[0] is None or not n:
        return out
    out["coverage"] = cov[0]
    if clustered:
        if cov[1] is not None:
            out["interval"] = [
                max(0.0, cov[0] - 1.96 * cov[1]),
                min(1.0, cov[0] + 1.96 * cov[1]),
            ]
    else:
        out["interval"] = list(wilson_interval(round(cov[0] * n), n))
    return out


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


@dataclass
class CardInputs:
    """``evaluation.json``, the sidecar and the grid split derived from them."""

    record: dict[str, Any]
    metadata: dict[str, Any]
    items_axis: list[int]
    persons_axis: list[int]
    trained_items: tuple[int, int] | None
    trained_persons: tuple[int, int] | None
    cells: dict[tuple[int, int], dict[str, Any]]
    #: Reference estimators from ``<card_dir>/references/``.
    references: list[Any] = field(default_factory=list)

    @property
    def spec(self) -> dict[str, Any]:
        return self.metadata.get("model_spec", {}) or {}

    @property
    def parameters(self) -> list[str]:
        from .card import ITEM_PARAMETERS

        family = str(self.spec.get("model", "")).lower()
        if family not in ITEM_PARAMETERS:
            raise ValueError(
                f"Unsupported model family {self.spec.get('model')!r} in the "
                f"sidecar; the card renders {sorted(ITEM_PARAMETERS)}."
            )
        return ["log_a" if k == "a" else k for k in ITEM_PARAMETERS[family]]

    @property
    def joint_omitted(self) -> str | None:
        """The recorded reason when no evaluated cell has joint metrics."""
        cells = self.evaluated()
        reasons = {c.get("joint", {}).get("omitted") for c in cells}
        if cells and None not in reasons:
            return "; ".join(sorted(reasons))
        return None

    @property
    def metrics(self) -> list[CardMetric]:
        """The card metrics for this record."""
        return card_metrics_for(self.parameters, joint=self.joint_omitted is None)

    def in_range(self, i: int, p: int) -> bool:
        if self.trained_items is None or self.trained_persons is None:
            return False
        lo_i, hi_i = self.trained_items
        lo_p, hi_p = self.trained_persons
        return lo_i <= i <= hi_i and lo_p <= p <= hi_p

    @property
    def in_items(self) -> list[int]:
        """Grid item counts inside the trained range."""
        lo, hi = self.trained_items or (math.inf, -math.inf)
        return [i for i in self.items_axis if lo <= i <= hi]

    @property
    def in_persons(self) -> list[int]:
        """Grid person counts inside the trained range."""
        lo, hi = self.trained_persons or (math.inf, -math.inf)
        return [p for p in self.persons_axis if lo <= p <= hi]

    def evaluated(self, *, in_range: bool | None = None) -> list[dict[str, Any]]:
        """Evaluated cells without a failure, in grid order."""
        out = []
        for (i, p), c in sorted(self.cells.items()):
            if "failed" in c:
                continue
            if in_range is None or self.in_range(i, p) == in_range:
                out.append(c)
        return out


def load_card_inputs(
    card_dir: str | Path, metadata: str | Path | None = None
) -> CardInputs:
    """Read ``evaluation.json`` and its sidecar from a card directory.

    The sidecar is ``metadata`` when given, else the file named in the
    record's ``metadata.filename``, looked up in the card directory and then
    next to it (beside the checkpoint).
    """
    from .metadata import MetadataTracker

    card_dir = Path(card_dir)
    record = json.loads((card_dir / "evaluation.json").read_text(encoding="utf-8"))
    fmt = record.get("evaluation_format")
    if fmt not in READS_EVALUATION_FORMAT:
        raise ValueError(
            f"evaluation_format {fmt!r} is not one this renderer reads "
            f"({READS_EVALUATION_FORMAT})."
        )
    if metadata is None:
        name = (record.get("metadata") or {}).get("filename")
        # The name comes from the record, so only a bare file name is looked
        # up; any other location must be passed explicitly.
        bare = (
            isinstance(name, str)
            and name not in (".", "..")
            and PureWindowsPath(name).name == name
            and PurePosixPath(name).name == name
        )
        if name and not bare:
            raise FileNotFoundError(
                f"The record's metadata.filename {name!r} is not a bare file "
                "name, so it is not looked up automatically; pass the sidecar "
                "explicitly (metadata=..., or --metadata)."
            )
        candidates = [card_dir / name, card_dir.parent / name] if name else []
        found = next((c for c in candidates if c.is_file()), None)
        if found is None:
            raise FileNotFoundError(
                f"Sidecar {name!r} not found in {card_dir} or {card_dir.parent}; "
                "pass it explicitly (metadata=..., or --metadata)."
            )
        metadata = found
    md = MetadataTracker.load(Path(metadata)).to_dict()
    # The card takes text from the sidecar and numbers from the record: they
    # must describe the same model.
    rec_uuid = (record.get("metadata") or {}).get("metadata_uuid")
    if rec_uuid and md.get("metadata_uuid") and rec_uuid != md["metadata_uuid"]:
        raise ValueError(
            f"Sidecar metadata_uuid {md['metadata_uuid']!r} differs from the "
            f"{rec_uuid!r} the record was evaluated with."
        )
    rec_spec = record.get("model_spec")
    if rec_spec is not None and json.dumps(rec_spec, sort_keys=True) != json.dumps(
        md.get("model_spec"), sort_keys=True
    ):
        raise ValueError(
            "The sidecar's model_spec differs from the one stored in the record; "
            "is this the sidecar the record was evaluated with?"
        )
    ranges = (md.get("model_spec") or {}).get("size_ranges") or {}
    grid = record["protocol"]["grid"]
    cells = {(c["cell"]["n_items"], c["cell"]["n_persons"]): c for c in record["cells"]}
    # plan_cells adds the exact trained bounds to the grid: keep them.
    items_axis = sorted({int(v) for v in grid["n_items"]} | {i for i, _ in cells})
    persons_axis = sorted({int(v) for v in grid["n_persons"]} | {p for _, p in cells})

    def _range(key: str) -> tuple[int, int] | None:
        r = ranges.get(key)
        return (int(r[0]), int(r[1])) if r else None

    inp = CardInputs(
        record=record,
        metadata=md,
        items_axis=items_axis,
        persons_axis=persons_axis,
        trained_items=_range("n_items"),
        trained_persons=_range("n_persons"),
        cells=cells,
    )
    inp.parameters  # noqa: B018 -- reject an unsupported family up front
    # One source of truth for "in range": the record's flags must agree with
    # the sidecar's size_ranges, or the card would contradict itself.
    wrong = [
        f"{i}x{p}"
        for (i, p), c in sorted(cells.items())
        if bool(c["cell"].get("in_range")) != inp.in_range(i, p)
    ]
    if wrong:
        raise ValueError(
            f"Cells {', '.join(wrong)} are classified differently by the record "
            "and by the sidecar's size_ranges; is this the sidecar the record "
            "was evaluated with?"
        )
    from .card_reference import load_references

    inp.references = load_references(card_dir, inp)
    return inp


# --------------------------------------------------------------------------
# Text sections
# --------------------------------------------------------------------------

_FORMULAS = {
    "1pl": "P(y = 1) = sigmoid(theta - b)",
    "2pl": "P(y = 1) = sigmoid(a (theta - b))",
    "pcm": (
        "P(y = k) proportional to exp(sum_{j<=k} (theta - delta_j)), "
        "k = 0..K-1 (Masters, 1982)"
    ),
    "gpcm": (
        "P(y = k) proportional to exp(sum_{j<=k} a (theta - delta_j)), "
        "k = 0..K-1 (Muraki, 1992)"
    ),
}


def _cell_name(c: dict[str, Any]) -> str:
    return f"{c['cell']['n_items']} x {c['cell']['n_persons']}"


def verbal_summary(inp: CardInputs) -> list[str]:
    """Template sentences from the numbers: no judgments."""
    rec = inp.record
    cells = inp.evaluated(in_range=True)
    lines = []
    n_eval = sum(1 for c in rec["cells"] if "failed" not in c)
    n_failed = sum(1 for c in rec["cells"] if "failed" in c)
    lines.append(
        f"{n_eval} cells evaluated ({len(cells)} inside the trained range), "
        f"{n_failed} failed; {rec['protocol']['datasets_per_cell']} datasets "
        f"per cell, {rec['protocol']['posterior_draws']} posterior draws "
        f"(preset {rec['protocol'].get('preset_name') or 'custom'})."
    )
    for metric in inp.metrics:
        vals = [(metric.getter(c), c) for c in cells]
        vals = [(v, c) for v, c in vals if v is not None and v[0] is not None]
        if not vals:
            continue
        lo = min(vals, key=lambda t: t[0][0])
        hi = max(vals, key=lambda t: t[0][0])
        lines.append(
            f"{metric.label}: from {_pm(lo[0])} at {_cell_name(lo[1])} to "
            f"{_pm(hi[0])} at {_cell_name(hi[1])} (value (MCSE), in-range cells)."
        )
    extreme = [c["persons"].get("extreme_patterns", 0) for c in cells]
    if extreme:
        lines.append(
            f"Scored persons with an all-extreme response pattern: "
            f"{min(extreme)} to {max(extreme)} per cell."
        )
    return lines


def status_banners(inp: CardInputs) -> list[str]:
    """Warnings for the card header: a development run, an incomplete record."""
    rec = inp.record
    out = []
    if rec["protocol"].get("preset_name") == "development":
        out.append(
            "Development run, not a validation report: "
            f"{rec['protocol']['datasets_per_cell']} datasets per cell."
        )
    if not rec.get("complete", False):
        out.append(
            f"Incomplete evaluation: the record holds {len(rec['cells'])} cells "
            "and was not finished; planned cells may be missing."
        )
    return out


def _identity(inp: CardInputs) -> str:
    spec, md, rec = inp.spec, inp.metadata, inp.record
    prov = md.get("provenance") or {}
    ck = rec.get("checkpoint") or {}
    si = rec.get("session_info") or {}
    rows = [
        [
            "Model",
            f"{_or_unrecorded(spec.get('model'))} "
            f"({_or_unrecorded(spec.get('variant'))})",
        ],
        ["Package version", _or_unrecorded(spec.get("version"))],
        ["Trained", _or_unrecorded(prov.get("timestamp"))],
        ["Git commit (training)", _or_unrecorded(prov.get("git_commit"))],
        ["metadata_uuid", _or_unrecorded(md.get("metadata_uuid"))],
        ["Checkpoint", f"`{_or_unrecorded(ck.get('filename'))}`"],
        ["Checkpoint SHA-256", f"`{_or_unrecorded(ck.get('sha256'))}`"],
        ["Sidecar match", _or_unrecorded(ck.get("match"))],
        [
            "Evaluated",
            f"{_or_unrecorded(si.get('timestamp'))}, "
            f"commit {_or_unrecorded(si.get('git_commit'))}",
        ],
    ]
    return _table(["Field", "Value"], rows)


def _envelope(inp: CardInputs) -> str:
    from .envelope import data_contract

    spec = inp.spec
    evaluated = inp.evaluated(in_range=True)
    if evaluated:
        vi = [c["cell"]["n_items"] for c in evaluated]
        vp = [c["cell"]["n_persons"] for c in evaluated]
        validated = f"items {min(vi)}-{max(vi)}, persons {min(vp)}-{max(vp)}"
    else:
        validated = "none"
    failed = [_cell_name(c) for (_, _), c in sorted(inp.cells.items()) if "failed" in c]
    if failed:
        validated += f"; failed cells: {', '.join(failed)}"
    ti, tp = inp.trained_items, inp.trained_persons
    trained = (
        f"items {ti[0]}-{ti[1]}, persons {tp[0]}-{tp[1]}"
        if ti and tp
        else "not recorded"
    )
    contract = data_contract(inp.metadata)
    miss = contract["missing_responses"]
    rows = [
        ["Trained range", trained],
        ["Validated range (evaluated in-range cells)", validated],
        ["Categories K", _or_unrecorded(spec.get("n_categories"))],
        [
            "Missing responses",
            miss.get("note") or ("supported" if miss["supported"] else "not supported"),
        ],
        ["Response coding", contract["response_coding"]],
        ["Item orientation", contract["orientation"]],
        ["Score scale", contract["score_scale"]],
    ]
    return _table(["", ""], rows)


def _priors(inp: CardInputs) -> str:
    priors = inp.spec.get("priors") or {}
    rows = []
    for name, rec in priors.items():
        if isinstance(rec, dict):
            hyper = ", ".join(f"{k} = {v}" for k, v in rec.items() if k != "name")
            rows.append([f"`{name}`", str(rec.get("name")), hyper])
        else:
            rows.append([f"`{name}`", "custom", "not recorded"])
    return _table(["Parameter", "Distribution", "Hyperparameters"], rows)


def _limitations(inp: CardInputs) -> list[str]:
    rec, md = inp.record, inp.metadata
    family = str(inp.spec.get("model", "")).lower()
    out = [
        "All metrics are computed on data simulated from the training prior. "
        "They describe the model where its assumptions hold, not its fit to "
        "real data.",
        "Reliability is shown for context; it has not yet been compared with "
        "a reference.",
    ]
    out.append(TAIL_COVERAGE_NOTE)
    if family not in rec.get("draws_validated_for", []):
        out.append(
            f"The posterior-draw count ({rec['protocol']['posterior_draws']}) was "
            f"validated for {', '.join(rec.get('draws_validated_for', [])) or 'no'} "
            f"models only, not for {family.upper()}."
        )
    item_posterior = (md.get("architecture") or {}).get("item_posterior")
    if item_posterior is None:
        out.append(
            "The item-posterior type is not recorded, so whether items are "
            "drawn jointly or independently given the summary is unknown; "
            "independent draws understate the uncertainty all items share."
        )
    elif item_posterior == "flow_matching" and family == "2pl":
        out.append(
            "Item parameters are drawn independently given the summary, so the "
            "uncertainty all items share (the location and scale of theta) is "
            "missing: joint statements across items are overconfident; "
            "per-item marginals are not affected."
        )
    elif item_posterior == "flow_matching":
        out.append(
            "Item parameters are drawn independently given the summary, so "
            "uncertainty all items share can be missing and joint statements "
            f"across items for the {family.upper()} may be overconfident."
        )
    if not rec["protocol"].get("include_out_of_range"):
        out.append(
            "Cells outside the trained range were not evaluated; behaviour "
            "there is unknown."
        )
    else:
        out.append(
            "Cells outside the trained range are extrapolation and are reported "
            "separately."
        )
    return out


def _usage(inp: CardInputs) -> str:
    ck = (inp.record.get("checkpoint") or {}).get("filename") or "model.keras"
    side = (inp.record.get("metadata") or {}).get("filename") or "model_metadata.json"
    return "\n".join(
        [
            "```python",
            "import os",
            'os.environ.setdefault("KERAS_BACKEND", "torch")  # before keras',
            "",
            "import keras",
            "from bayesflow_irt import PersonScorer, check_data",
            "",
            f'approximator = keras.saving.load_model("{ck}", compile=False)',
            f'check_data(responses, metadata="{side}")  # (persons, items)',
            "scorer = PersonScorer(approximator)",
            "results = scorer.score(responses, n_item_samples=1000)",
            "```",
        ]
    )


# --------------------------------------------------------------------------
# Grid tables and the appendix
# --------------------------------------------------------------------------


def _grid_table(inp: CardInputs, metric: CardMetric, *, in_range: bool) -> str:
    items = inp.in_items if in_range else inp.items_axis
    persons = inp.in_persons if in_range else inp.persons_axis
    rows = []
    for i in items:
        row = [f"**{i}**"]
        for p in persons:
            c = inp.cells.get((i, p))
            if in_range != inp.in_range(i, p):
                row.append("")
            elif c is None:
                row.append("not evaluated")
            elif "failed" in c:
                row.append(f"failed ({c['failed'].get('stage')})")
            else:
                row.append(_pm(metric.getter(c)))
        rows.append(row)
    return _table(["items \\ persons", *map(str, persons)], rows)


def _cell_appendix(inp: CardInputs, c: dict[str, Any]) -> list[str]:
    from .card_metrics import CARD_LEVELS

    out = [
        f"### Cell {_cell_name(c)}"
        + ("" if c["cell"]["in_range"] else " (extrapolation)"),
        "",
    ]
    if "failed" in c:
        f = c["failed"]
        return [*out, f"Failed at stage `{f.get('stage')}`: {f.get('message')}", ""]
    params = [*inp.parameters, "theta"]
    keys = [
        "bias",
        "rmse",
        "corr",
        "nrmse_excess",
        *(f"coverage_{lv}" for lv in CARD_LEVELS),
        "width_0.95",
        "mean_calibration_error",
        "calibration_error",
        "sbc_ecdf_max_dev",
    ]
    rows = []
    for k in keys:
        rows.append([k, *(_pm(_param_getter(p, k)(c)) for p in params)])
    out += [_table(["metric", *params], rows), "", TAIL_COVERAGE_NOTE, ""]

    bins = c.get("persons", {}).get("conditional") or []
    if bins:
        clustered = "clusters" in bins[0]
        brow = []
        for b in bins:
            lo, hi = b["edges"]
            edge = f"[{_num(lo, 1) if lo is not None else '-inf'}, "
            edge += f"{_num(hi, 1) if hi is not None else 'inf'})"
            n = b.get("n", 0)
            bc = bin_coverage(b)
            if bc["interval"] is not None:
                wl, wh = bc["interval"]
                cov_s = f"{_num(bc['coverage'])} [{_num(wl)}, {_num(wh)}]"
            elif bc["coverage"] is not None:
                cov_s = _num(bc["coverage"])
            else:
                cov_s = "–"
            if clustered:
                g = b["clusters"]
                mark = f"{n} / {g}" + (
                    f" (datasets < {MIN_BIN_CLUSTERS})" if bc["small"] else ""
                )
            else:
                mark = f"{n} (n < {MIN_BIN_N})" if n < MIN_BIN_N else str(n)
            brow.append(
                [
                    edge,
                    mark,
                    _pm(b.get("bias")),
                    cov_s,
                    _pm(b.get("width_0.95")),
                    _pm(b.get("posterior_sd")),
                    _pm(b.get("conditional_se")),
                ]
            )
        basis = c.get("persons", {}).get("conditional_se_basis")
        intro = (
            "Theta by equal-probability bin of the true theta (prior z-scale; "
            "all scored persons; n / contributing datasets). MCSEs are "
            "cluster-robust by dataset (Cameron & Miller, 2015); coverage "
            "with a 95% interval p ± 1.96 SE."
            if clustered
            else "Theta by bin of the true theta (prior z-scale; one random person "
            "per dataset). Coverage with a 95% Wilson interval."
        )
        out += [
            intro + " Conditional "
            "SE: 1 / sqrt(test information) at the true theta"
            + (f" ({basis})." if basis else "; not computed for this family."),
            "",
            _table(
                [
                    "bin",
                    "n / datasets" if clustered else "n",
                    "bias",
                    "coverage 0.95",
                    "width 0.95",
                    "posterior SD",
                    "conditional SE",
                ],
                brow,
            ),
            "",
        ]
    rel = c.get("persons", {}).get("reliability") or {}
    jt = c.get("joint", {})
    jrows = []
    for name in (
        "posterior_variance",
        "squared_correlation",
        "difference",
        "information",
    ):
        if name not in rel:
            continue
        label = f"reliability, {name} ({RELIABILITY_LABEL})"
        if "omitted" in rel[name]:
            jrows.append([label, "omitted", str(rel[name]["omitted"])])
            continue
        rr = rel[name].get("range_95") or [None, None]
        note = f"{_num(rr[0])} to {_num(rr[1])}"
        if rel[name].get("basis"):
            note += f"; {rel[name]['basis']}"
        jrows.append([label, _pm(rel[name].get("mean")), note])
    if "omitted" in jt:
        jrows.append(["Joint metrics (TARP, TCC)", "omitted", str(jt["omitted"])])
    for ref in ("random", "classical"):
        t = jt.get("tarp", {}).get(ref)
        if t:
            label = f"TARP error, {REF_LABELS[ref]} references"
            jrows.append(
                [
                    label,
                    _pm(t.get("tarp_error")),
                    f"null floor {_num(t.get('null_floor'))}",
                ]
            )
    tcc = jt.get("tcc")
    if tcc:
        jrows.append(
            [
                "TCC bias (score units)",
                _pm(tcc.get("tcc_bias")),
                f"at theta {tcc.get('thetas')}",
            ]
        )
    rimse = c.get("persons", {}).get("rimse")
    if rimse:
        rr = rimse.get("range_95") or [None, None]
        jrows.append(
            [
                "RIMSE, random item",
                _pm(rimse.get("mean")),
                f"{_num(rr[0])} to {_num(rr[1])}",
            ]
        )
    jrows.append(
        [
            "Extreme response patterns",
            str(c.get("persons", {}).get("extreme_patterns", "–")),
            "",
        ]
    )
    s = c.get("sampling", {})
    jrows.append(
        [
            "Sampling seconds",
            _num(s.get("seconds"), 1),
            f"{len(s.get('retries') or [])} out-of-memory retries",
        ]
    )
    out += [_table(["quantity", "value (MCSE)", "range / note"], jrows), ""]
    return out


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def _have_matplotlib() -> bool:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return False
    return True


def _save(fig: Any, path: Path) -> None:
    fig.savefig(path, dpi=110, bbox_inches="tight", metadata={"Software": None})


def _heatmap(inp: CardInputs, metric: CardMetric, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    cmap = plt.get_cmap("PuOr" if metric.kind == "deviation" else "Blues")
    norm = Normalize(*metric.limits)
    panels = [("inside the trained range", inp.in_items, inp.in_persons, True)]
    panels.append(
        (
            "outside the trained range (extrapolation)",
            inp.items_axis,
            inp.persons_axis,
            False,
        )
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.2),
        gridspec_kw={"width_ratios": [len(inp.in_persons) or 1, len(inp.persons_axis)]},
    )
    for ax, (title, items, persons, in_range) in zip(axes, panels):
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(len(persons)), [str(p) for p in persons])
        ax.set_yticks(range(len(items)), [str(i) for i in items])
        ax.set_xlabel("persons")
        ax.set_ylabel("items")
        ax.set_xlim(-0.5, len(persons) - 0.5)
        ax.set_ylim(-0.5, len(items) - 0.5)
        if not in_range:
            ys = [y for y, i in enumerate(items) if i in inp.in_items]
            xs = [x for x, p in enumerate(persons) if p in inp.in_persons]
            if ys and xs:
                ax.text(
                    np.mean(xs),
                    np.mean(ys) + 0.6,
                    "trained range\n(left panel)",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="0.45",
                )
            if not inp.record["protocol"].get("include_out_of_range"):
                ax.text(
                    0.5,
                    0.12,
                    "not evaluated; run with --include-out-of-range",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=9,
                )
        for y, i in enumerate(items):
            for x, p in enumerate(persons):
                if inp.in_range(i, p) != in_range:
                    if not in_range:
                        ax.add_patch(
                            plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color="0.93", lw=0)
                        )
                    continue
                c = inp.cells.get((i, p))
                if c is None:
                    continue
                if "failed" in c:
                    ax.text(x, y, "failed", ha="center", va="center", fontsize=7)
                    continue
                v = metric.getter(c)
                if v is None or v[0] is None:
                    continue
                ax.add_patch(
                    plt.Rectangle(
                        (x - 0.5, y - 0.5), 1, 1, color=cmap(norm(v[0])), lw=0
                    )
                )
                shade = float(np.clip(norm(v[0]), 0, 1))
                dark = (
                    abs(shade - 0.5) > 0.35
                    if metric.kind == "deviation"
                    else shade > 0.7
                )
                ax.text(
                    x,
                    y,
                    f"{v[0]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if dark else "black",
                )
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    bar = fig.colorbar(sm, ax=axes, shrink=0.9)
    bar.set_label(
        ("deviation, clipped at " if metric.kind == "deviation" else "value, ")
        + f"[{metric.limits[0]}, {metric.limits[1]}]"
    )
    fig.suptitle(metric.label, fontsize=11)
    _save(fig, path)
    plt.close(fig)


def _coverage_curves(inp: CardInputs, path: Path) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .card_metrics import CARD_LEVELS

    params = [*inp.parameters, "theta"]
    cells = inp.evaluated(in_range=True)
    if not cells:
        return False
    sizes = [math.log(c["cell"]["n_items"] * c["cell"]["n_persons"]) for c in cells]
    lo, hi = min(sizes), max(sizes)
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, len(params), figsize=(4 * len(params), 3.6))
    for n, (ax, param) in enumerate(zip(np.atleast_1d(axes), params)):
        ax.plot([0.45, 1], [0.45, 1], color="0.5", lw=0.8, ls="--")
        for c, s in zip(cells, sizes):
            ys = [
                (_param_getter(param, f"coverage_{lv}")(c) or [None])[0]
                for lv in CARD_LEVELS
            ]
            if any(y is None for y in ys):
                continue
            ax.plot(CARD_LEVELS, ys, color=cmap((s - lo) / (hi - lo or 1)), lw=1)
        ax.set_title(param, fontsize=10)
        ax.set_xlabel("nominal level")
        if n == 0:
            ax.set_ylabel("observed coverage")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(lo, hi))
    fig.colorbar(sm, ax=axes, label="log(items x persons)")
    fig.suptitle("Central-interval coverage, one line per in-range cell", fontsize=11)
    _save(fig, path)
    plt.close(fig)
    return True


def _tarp_curves(inp: CardInputs, path: Path) -> bool:
    """Expected coverage against credibility level, both reference rules.

    Random references alone pass an estimator that returns the prior (Lemos
    et al., 2023, Sec. 4.3), so the two are drawn side by side.
    """
    cells = [c for c in inp.evaluated(in_range=True) if c.get("joint", {}).get("tarp")]
    if not cells:
        return False
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = [math.log(c["cell"]["n_items"] * c["cell"]["n_persons"]) for c in cells]
    lo, hi = min(sizes), max(sizes)
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), sharey=True)
    for ax, ref in zip(axes, ("random", "classical")):
        ax.plot([0, 1], [0, 1], color="0.5", lw=0.8, ls="--")
        for c, s in zip(cells, sizes):
            t = c["joint"]["tarp"].get(ref)
            if t:
                ax.plot(
                    t["levels"],
                    t["expected_coverage"],
                    color=cmap((s - lo) / (hi - lo or 1)),
                    lw=1,
                )
        ax.set_title(f"{REF_LABELS[ref]} references", fontsize=10)
        ax.set_xlabel("credibility level")
    axes[0].set_ylabel("expected coverage")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(lo, hi))
    fig.colorbar(sm, ax=axes, label="log(items x persons)")
    fig.suptitle("TARP coverage, one line per in-range cell", fontsize=11)
    _save(fig, path)
    plt.close(fig)
    return True


def _prior_densities(inp: CardInputs, path: Path) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .priors import resolve_prior
    from .simulators import _prior_from_record

    priors = {
        k: v
        for k, v in (inp.spec.get("priors") or {}).items()
        if isinstance(v, dict) and k in ("a", "b", "theta", "delta")
    }
    if not priors:
        return False
    fig, axes = plt.subplots(1, len(priors), figsize=(3.6 * len(priors), 2.8))
    for ax, (role, rec) in zip(np.atleast_1d(axes), priors.items()):
        sampler = resolve_prior(_prior_from_record(rec, role), role)
        x = np.asarray(sampler((PRIOR_DENSITY_DRAWS,), np.random.default_rng(0)))
        lo, hi = np.quantile(x, [0.001, 0.999])
        ax.hist(
            x, bins=120, range=(lo, hi), density=True, histtype="step", color="#185FA5"
        )
        ax.set_title(f"{role} ~ {rec['name']}", fontsize=10)
        ax.set_yticks([])
    fig.suptitle(f"Prior densities ({PRIOR_DENSITY_DRAWS:,} seeded draws)", fontsize=10)
    _save(fig, path)
    plt.close(fig)
    return True


def trained_marginals(inp: CardInputs) -> dict[str, Any] | None:
    """The training size distribution of each axis, or ``None``.

    Returns ``{"distribution", "items": (values, probabilities), "persons":
    (values, probabilities)}``. The joint is their outer product, valid
    because ``_draw_sizes`` draws the two axes independently.
    """
    from .simulators import DEFAULT_LOG_UNIFORM_SHARE, _size_distribution

    sampling = inp.spec.get("size_sampling")
    if not sampling or inp.trained_items is None or inp.trained_persons is None:
        return None
    dist = sampling.get("size_distribution", "uniform")
    share = sampling.get("log_uniform_share", DEFAULT_LOG_UNIFORM_SHARE)

    def axis(key: str, legacy_bucket: str, bounds: tuple[int, int]) -> Any:
        # describe_size_sampling records each axis with its support; use it.
        rec = sampling.get(key) or {}
        if rec.get("values") is not None and rec.get("probabilities") is not None:
            return np.asarray(rec["values"]), np.asarray(rec["probabilities"])
        bucket = rec.get("bucket", sampling.get(legacy_bucket))
        return _size_distribution(bounds, bucket, dist, share)

    return {
        "distribution": dist,
        "items": axis("n_items", "item_bucket", inp.trained_items),
        "persons": axis("n_persons", "person_bucket", inp.trained_persons),
    }


def _trained_map(inp: CardInputs, path: Path) -> bool:
    """Persons x items training distribution: outer product of the marginals."""
    marginals = trained_marginals(inp)
    if marginals is None:
        return False
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dist = marginals["distribution"]
    vi, pi = marginals["items"]
    vp, pp = marginals["persons"]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    joint = np.outer(pi, pp)
    mesh = ax.pcolormesh(vp, vi, joint, cmap="Blues", shading="nearest")
    ax.set_xscale("log")
    ax.set_yscale("log")
    gp, gi = inp.persons_axis, inp.items_axis
    ax.set_xticks(gp, [str(p) for p in gp])
    ax.set_yticks(gi, [str(i) for i in gi])
    ax.minorticks_off()
    ax.scatter(
        np.repeat(gp, len(gi)),
        np.tile(gi, len(gp)),
        s=10,
        color="#D85A30",
        marker="x",
        label="card grid",
    )
    ax.set_xlabel("persons (log scale)")
    ax.set_ylabel("items (log scale)")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(mesh, ax=ax, label="probability per size pair")
    ax.set_title(f"Training sizes ({dist})", fontsize=10)
    _save(fig, path)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def _reference_section(inp: CardInputs) -> list[str]:
    """One table per reference: card, reference and difference per cell."""
    if not inp.references:
        return []
    from .card_html import _extreme_share, dashboard_metrics
    from .card_reference import card_value, difference

    metrics = {m["key"]: m for m in dashboard_metrics(inp)}
    md = [
        "## Reference estimators",
        "",
        "Other estimators at the card's cells, value (MCSE). Difference: card "
        "minus reference, MCSE as for independent estimates. For a paired "
        "reference the covariance is unknown, so this MCSE is an "
        "approximation that can be too large or too small.",
        "",
    ]
    for ref in inp.references:
        priors = {
            True: "same as the card's",
            False: "differ from the card's",
            None: "not compared",
        }[ref.priors_match]
        md += [
            f"### {ref.name}",
            "",
            f"{ref.estimator} Datasets: {ref.datasets}. Priors: {priors}.",
            "",
        ]
        md += [f"- `{k}`: {v}" for k, v in sorted(ref.notes.items())]
        if ref.notes:
            md += [""]
        rows = []
        for (i, p), vals in sorted(ref.cells.items()):
            cell = inp.cells[(i, p)]
            if "failed" not in cell:
                # As in dashboard_payload, so both cards show the same values.
                cell = {**cell, "derived": {"extreme_share": _extreme_share(inp, cell)}}
            name = f"{i} x {p}" + ("" if inp.in_range(i, p) else " (extrapolation)")
            for key, rv in sorted(vals.items()):
                cv = None if "failed" in cell else card_value(cell, metrics[key])
                rows.append(
                    [name, f"`{key}`", _pm(cv), _pm(rv), _pm(difference(cv, rv))]
                )
        md += [_table(["Cell", "Metric", "Card", ref.name, "Difference"], rows), ""]
        if ref.skipped_cells:
            md += [
                f"Cells not on this card, left out: {', '.join(ref.skipped_cells)}.",
                "",
            ]
    return md


def _variant_label(variant: Any) -> str:
    """The prior variant for a title: ``"fixed"`` reads ``"fixed prior"``."""
    if variant in ("fixed", "variable"):
        return f"{variant} prior"
    return _or_unrecorded(variant)


def render_markdown(inp: CardInputs, figures: dict[str, str] | None = None) -> str:
    """The Markdown card; ``figures`` maps figure keys to relative paths."""
    figures = figures or {}
    spec = inp.spec
    family = str(spec.get("model", "")).lower()
    md: list[str] = [
        f"# Model card: {_or_unrecorded(spec.get('model'))} "
        f"({_variant_label(spec.get('variant'))})",
        "",
        *(f"> **{b}**\n" for b in status_banners(inp)),
        _identity(inp),
        "",
        "## Summary",
        "",
        *(f"- {line}" for line in verbal_summary(inp)),
        "",
        "## The model",
        "",
        f"`{_FORMULAS.get(family, 'not recorded')}`; internal parameterization "
        f"`{_or_unrecorded(spec.get('internal_parameterization'))}`. "
        "Identification: theta follows its training prior, which fixes the "
        "location and scale of the latent trait.",
        "",
        "## Supported data",
        "",
        _envelope(inp),
        "",
    ]
    if "trained_map" in figures:
        md += [f"![Training sizes]({figures['trained_map']})", ""]
    elif spec.get("size_sampling"):
        md += [
            "Map of trained sizes not drawn (needs matplotlib, the [card] "
            "extra); the sampling rule is in the sidecar's "
            "`model_spec.size_sampling`.",
            "",
        ]
    else:
        md += ["Training size sampling not recorded; no map of trained cells.", ""]
    md += ["## Priors", "", _priors(inp), ""]
    if "priors" in figures:
        md += [f"![Prior densities]({figures['priors']})", ""]

    notes = (inp.metadata.get("intended_use") or {}).get("notes")
    md += ["## Key metrics across cells", ""]
    if notes:
        md += [
            f"> **Author's note** (written by the model's author, not "
            f"generated): {notes}",
            "",
        ]
    if inp.joint_omitted:
        md += [f"Joint metrics (TARP, TCC) not computed: {inp.joint_omitted}.", ""]
    for metric in inp.metrics:
        md += [
            f"### {metric.label}",
            "",
            metric.guide + f" Colour limits: [{metric.limits[0]}, {metric.limits[1]}].",
            "",
        ]
        key = f"heatmap_{metric.key}"
        if key in figures:
            md += [f"![{metric.label}]({figures[key]})", ""]
        md += [
            "Inside the trained range, value (MCSE):",
            "",
            _grid_table(inp, metric, in_range=True),
            "",
        ]
        if inp.evaluated(in_range=False):
            md += [
                "Outside the trained range (extrapolation):",
                "",
                _grid_table(inp, metric, in_range=False),
                "",
            ]
    if "coverage" in figures:
        md += [f"![Coverage curves]({figures['coverage']})", ""]
    if "tarp" in figures:
        md += [f"![TARP coverage, both reference rules]({figures['tarp']})", ""]
    if not inp.record["protocol"].get("include_out_of_range"):
        md += [
            "Cells outside the trained range: not evaluated; run with "
            "`--include-out-of-range`.",
            "",
        ]

    md += _reference_section(inp)
    md += [
        "## Limitations",
        "",
        *(f"- {s}" for s in _limitations(inp)),
        "",
        "## Usage",
        "",
        _usage(inp),
        "",
    ]

    md += ["## Appendix: per-cell results", ""]
    for c in sorted(
        inp.record["cells"],
        key=lambda c: (
            not c["cell"]["in_range"],
            c["cell"]["n_items"],
            c["cell"]["n_persons"],
        ),
    ):
        md += _cell_appendix(inp, c)

    rec = inp.record
    prot = {k: v for k, v in rec["protocol"].items() if k != "seed_rule"}
    md += [
        "## Provenance",
        "",
        f"Protocol: `{json.dumps(prot, sort_keys=True)}`",
        "",
        f"Seed rule: {rec['protocol'].get('seed_rule')}",
        "",
        f"Evaluation wall time: {_num((rec.get('seconds') or 0) / 3600, 2)} h.",
        "",
        "```json",
        json.dumps(rec.get("session_info"), indent=1, sort_keys=True),
        "```",
        "",
    ]
    return "\n".join(md)


def render_model_card(card_dir: str | Path, metadata: str | Path | None = None) -> Path:
    """Write ``model_card.md``, ``model_card.html`` and ``figures/*.png``.

    Parameters
    ----------
    card_dir : path
        Directory holding ``evaluation.json`` (from
        :func:`bayesflow_irt.evaluate_for_card`).
    metadata : path, optional
        The sidecar; default: the file the record names, in ``card_dir`` or
        next to it.

    Returns
    -------
    Path
        The written ``model_card.md``.
    """
    card_dir = Path(card_dir)
    inp = load_card_inputs(card_dir, metadata)
    figures: dict[str, str] = {}
    if _have_matplotlib():
        fig_dir = card_dir / "figures"
        fig_dir.mkdir(exist_ok=True)
        for metric in inp.metrics:
            name = f"heatmap_{metric.key.replace('.', '_')}.png"
            _heatmap(inp, metric, fig_dir / name)
            figures[f"heatmap_{metric.key}"] = f"figures/{name}"
        if _coverage_curves(inp, fig_dir / "coverage_curves.png"):
            figures["coverage"] = "figures/coverage_curves.png"
        if _tarp_curves(inp, fig_dir / "tarp_curves.png"):
            figures["tarp"] = "figures/tarp_curves.png"
        if _prior_densities(inp, fig_dir / "priors.png"):
            figures["priors"] = "figures/priors.png"
        if _trained_map(inp, fig_dir / "trained_sizes.png"):
            figures["trained_map"] = "figures/trained_sizes.png"
    else:
        warnings.warn(
            "matplotlib is not installed (pip install 'bayesflow-irt[card]'); "
            "the model card is written with tables only, without figures.",
            stacklevel=2,
        )
    out = card_dir / "model_card.md"
    out.write_text(render_markdown(inp, figures), encoding="utf-8")
    from .card_html import write_html

    write_html(inp, card_dir, card_dir / "figures" if figures else None)
    return out


def main(argv: Sequence[str] | None = None) -> None:
    """``python -m bayesflow_irt.card_render <card_dir> [--metadata PATH]``."""
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("card_dir", type=Path)
    p.add_argument("--metadata", type=Path, default=None)
    args = p.parse_args(argv)
    print(render_model_card(args.card_dir, args.metadata))


if __name__ == "__main__":
    main()
