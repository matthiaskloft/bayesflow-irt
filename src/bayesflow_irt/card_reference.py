"""Reference estimators on the model card.

A reference is another estimator's results at the card's cells, for
example Stan/NUTS (the exact-posterior benchmark) or mirt EAP. Each one is
a JSON file in ``<card_dir>/references/``; the card reads and shows them,
and never runs R, Stan or anything else itself. Converters for other
estimators' outputs are not part of the package.

File format (``reference_format`` 1)::

    {
      "reference_format": 1,
      "name": "Stan NUTS",                 # short label on the card
      "estimator": "...",                  # one sentence: what was run
      "family": "2pl",                     # must match the card's model
      "datasets": {"kind": "card", "seed": 0, "protocol_version": 1}
                | {"kind": "own", "description": "..."},
      "priors": {...} | null,              # as the sidecar's model_spec.priors
      "provenance": {...},                 # script, commit, settings
      "notes": {"<metric key>": "..."},    # where a definition differs
      "cells": [{"n_items": 5, "n_persons": 500,
                 "metrics": {"b.corr": [0.93, 0.01], ...}}]
    }

Metric keys are the dashboard's (:func:`card_html.dashboard_metrics`), and
values are on the scale the card shows: coverage keys hold coverage minus
the level. Each value is ``[value, mcse]``; the MCSE may be null.

``datasets.kind == "card"`` declares a paired reference, run on the card's
own datasets (rebuilt from the record's seed rule); its seed and protocol
version must match the record. ``"own"`` is unpaired: same sizes and
model, different simulated datasets, so a difference to the card includes
both sides' Monte Carlo error.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .card_render import CardInputs

#: The reference file formats this card reads.
READS_REFERENCE_FORMAT = (1,)

#: Directory inside a card directory that holds the reference files.
REFERENCE_DIR = "references"


@dataclass
class Reference:
    """One reference estimator, checked against a card's record."""

    name: str
    estimator: str
    paired: bool
    datasets: str
    priors_match: bool | None
    notes: dict[str, str]
    provenance: dict[str, Any]
    cells: dict[tuple[int, int], dict[str, list[float | None]]]
    skipped_cells: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        """The dashboard's view: cells keyed ``"<items>x<persons>"``."""
        return {
            "name": self.name, "estimator": self.estimator, "paired": self.paired,
            "datasets": self.datasets, "priors_match": self.priors_match,
            "notes": self.notes, "provenance": self.provenance,
            "skipped_cells": self.skipped_cells,
            "cells": {f"{i}x{p}": m for (i, p), m in sorted(self.cells.items())},
        }


def _datasets_text(ds: dict[str, Any]) -> str:
    if ds.get("kind") == "card":
        return "the card's own datasets (paired)"
    return "own simulated datasets, not the card's (unpaired): " + str(
        ds.get("description") or "no description recorded")


def load_reference(path: str | Path, inp: CardInputs) -> Reference:
    """Read one reference file and check it against the card.

    Raises
    ------
    ValueError
        On an unknown format, another model family, a paired reference
        whose seed or protocol version differs from the record, an unknown
        metric key, a malformed value, or no cell in common with the card.
    """
    from .card_html import dashboard_metrics

    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    where = f"Reference {path.name}"
    fmt = raw.get("reference_format")
    if fmt not in READS_REFERENCE_FORMAT:
        raise ValueError(f"{where}: reference_format {fmt!r} is not one this "
                         f"card reads ({READS_REFERENCE_FORMAT}).")
    for key in ("name", "estimator", "family", "datasets", "cells"):
        if key not in raw:
            raise ValueError(f"{where}: missing {key!r}.")
    family = str(inp.spec.get("model", "")).lower()
    if str(raw["family"]).lower() != family:
        raise ValueError(f"{where}: family {raw['family']!r} differs from the "
                         f"card's {family!r}.")
    ds = raw["datasets"]
    if ds.get("kind") not in ("card", "own"):
        raise ValueError(f"{where}: datasets.kind must be 'card' or 'own'; got "
                         f"{ds.get('kind')!r}.")
    if ds["kind"] == "card":
        proto = inp.record["protocol"]
        for key in ("seed", "protocol_version"):
            if ds.get(key) != proto.get(key):
                raise ValueError(
                    f"{where}: a paired reference must use the card's datasets, "
                    f"but its {key} {ds.get(key)!r} differs from the record's "
                    f"{proto.get(key)!r}.")
    known = {m["key"] for m in dashboard_metrics(inp)}
    cells: dict[tuple[int, int], dict[str, list[float | None]]] = {}
    skipped = []
    for c in raw["cells"]:
        i, p = int(c["n_items"]), int(c["n_persons"])
        metrics = c.get("metrics") or {}
        unknown = sorted(set(metrics) - known)
        if unknown:
            raise ValueError(f"{where}: cell {i}x{p} has metric keys the card "
                             f"does not show: {', '.join(unknown)}.")
        for k, v in metrics.items():
            if not (isinstance(v, list) and len(v) == 2
                    and all(x is None or isinstance(x, (int, float)) for x in v)):
                raise ValueError(f"{where}: cell {i}x{p}, {k}: expected "
                                 f"[value, mcse]; got {v!r}.")
        if (i, p) in inp.cells:
            cells[(i, p)] = {k: [None if x is None else float(x) for x in v]
                             for k, v in metrics.items()}
        else:
            skipped.append(f"{i}x{p}")
    if not cells:
        raise ValueError(f"{where}: none of its cells is a cell of this card.")
    priors = raw.get("priors")
    card_priors = inp.spec.get("priors")
    priors_match = (None if priors is None or card_priors is None else
                    json.dumps(priors, sort_keys=True)
                    == json.dumps(card_priors, sort_keys=True))
    return Reference(
        name=str(raw["name"]), estimator=str(raw["estimator"]),
        paired=ds["kind"] == "card", datasets=_datasets_text(ds),
        priors_match=priors_match, notes=dict(raw.get("notes") or {}),
        provenance=dict(raw.get("provenance") or {}), cells=cells,
        skipped_cells=skipped)


def card_value(cell: dict[str, Any], m: dict[str, Any]) -> list[float | None] | None:
    """A dashboard metric's ``[value, mcse]`` in a record cell.

    The Python twin of ``value()`` in ``dashboard.js``.
    """
    v: Any = cell
    for k in m["path"]:
        v = v.get(k) if isinstance(v, dict) else None
    se = None
    if m["index"] is not None:
        v = v[m["index"]] if isinstance(v, list) else None
        s: Any = cell
        for k in m["se_path"] or []:
            s = s.get(k) if isinstance(s, dict) else None
        se = s[m["index"]] if isinstance(s, list) else None
    elif isinstance(v, list):
        v, se = v[0], v[1]
    if not isinstance(v, (int, float)) or v != v:
        return None
    return [float(v) - m["offset"], se]


def difference(card: list[float | None] | None,
               ref: list[float | None] | None) -> list[float | None] | None:
    """Card minus reference, MCSE as for independent estimates.

    For a paired reference (same datasets) the covariance of the two
    estimates is unknown, so this MCSE can be too large or too small.
    """
    if card is None or ref is None or card[0] is None or ref[0] is None:
        return None
    se = (None if card[1] is None or ref[1] is None
          else (card[1] ** 2 + ref[1] ** 2) ** 0.5)
    return [card[0] - ref[0], se]


def load_references(card_dir: str | Path, inp: CardInputs) -> list[Reference]:
    """Every ``*.json`` in ``<card_dir>/references/``, sorted by file name."""
    folder = Path(card_dir) / REFERENCE_DIR
    if not folder.is_dir():
        return []
    refs = [load_reference(f, inp) for f in sorted(folder.glob("*.json"))]
    names = [r.name for r in refs]
    if len(set(names)) != len(names):
        raise ValueError(f"Two references share a name: {names}.")
    return refs
