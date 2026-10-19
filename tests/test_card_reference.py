"""Tests for reference estimators on the model card (``card_reference``)."""

import json
import re

import pytest
from test_card_render import _copy, card_dir  # noqa: F401 -- module fixture

import bayesflow_irt.card_html as ch
import bayesflow_irt.card_render as cr
from bayesflow_irt.card_reference import card_value, difference, load_reference


def _ref(inp, **over):
    (i, p) = sorted(inp.cells)[0]
    raw = {
        "reference_format": 1,
        "name": "Stan NUTS",
        "estimator": "NUTS, 4 chains.",
        "family": inp.spec["model"],
        "datasets": {"kind": "own", "description": "25 datasets per cell"},
        "priors": None,
        "provenance": {"script": "x.py"},
        "notes": {"theta.corr": "pooled over persons"},
        "cells": [
            {
                "n_items": i,
                "n_persons": p,
                "metrics": {
                    "theta.corr": [0.8, 0.02],
                    "theta.coverage_dev_0.95": [-0.01, None],
                },
            },
            {"n_items": 999, "n_persons": 7, "metrics": {}},
        ],
    }
    raw.update(over)
    return raw


def _write(card, raw, name="stan.json"):
    folder = card / "references"
    folder.mkdir(exist_ok=True)
    (folder / name).write_text(json.dumps(raw), encoding="utf-8")


def test_reference_reaches_markdown_and_payload(card_dir, tmp_path):  # noqa: F811
    card = _copy(card_dir, tmp_path)
    _write(card, _ref(cr.load_card_inputs(card)))
    inp = cr.load_card_inputs(card)
    (ref,) = inp.references
    assert not ref.paired and ref.skipped_cells == ["999x7"]
    md = cr.render_markdown(inp)
    assert "## Reference estimators" in md and "pooled over persons" in md
    html = ch.render_html(inp)
    data = json.loads(re.search(r'id="card-data">(.*?)</script>', html, re.S).group(1))
    (comp,) = data["comparisons"]
    assert comp["name"] == "Stan NUTS"
    assert list(comp["cells"].values())[0]["theta.corr"] == [0.8, 0.02]


@pytest.mark.parametrize(
    "over, match",
    [
        ({"reference_format": 9}, "reference_format"),
        ({"family": "gpcm_x"}, "family"),
        ({"datasets": {"kind": "card", "seed": 12345, "protocol_version": 1}}, "seed"),
        ({"datasets": {"kind": "maybe"}}, "datasets.kind"),
        (
            {"cells": [{"n_items": 999, "n_persons": 7, "metrics": {}}]},
            "none of its cells",
        ),
    ],
)
def test_bad_references_are_refused(card_dir, tmp_path, over, match):  # noqa: F811
    card = _copy(card_dir, tmp_path)
    inp = cr.load_card_inputs(card)
    path = tmp_path / "r.json"
    path.write_text(json.dumps(_ref(inp, **over)), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_reference(path, inp)


def test_unknown_metric_and_malformed_value_are_refused(card_dir, tmp_path):  # noqa: F811
    card = _copy(card_dir, tmp_path)
    inp = cr.load_card_inputs(card)
    (i, p) = sorted(inp.cells)[0]
    for metrics, match in [
        ({"theta.made_up": [1.0, 0.1]}, "does not show"),
        ({"theta.corr": 0.8}, r"\[value, mcse\]"),
    ]:
        raw = _ref(inp, cells=[{"n_items": i, "n_persons": p, "metrics": metrics}])
        path = tmp_path / "r.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            load_reference(path, inp)


def test_paired_reference_with_the_card_seed_loads(card_dir, tmp_path):  # noqa: F811
    card = _copy(card_dir, tmp_path)
    inp = cr.load_card_inputs(card)
    proto = inp.record["protocol"]
    raw = _ref(
        inp,
        datasets={
            "kind": "card",
            "seed": proto["seed"],
            "protocol_version": proto["protocol_version"],
        },
        priors=inp.spec.get("priors"),
    )
    path = tmp_path / "r.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    ref = load_reference(path, inp)
    assert ref.paired and ref.priors_match is True


def test_markdown_extreme_share_matches_the_dashboard(card_dir, tmp_path):  # noqa: F811
    card = _copy(card_dir, tmp_path)
    inp = cr.load_card_inputs(card)
    (i, p) = sorted(inp.cells)[0]
    _write(
        card,
        _ref(
            inp,
            cells=[
                {
                    "n_items": i,
                    "n_persons": p,
                    "metrics": {"theta.extreme_share": [0.1, 0.01]},
                }
            ],
        ),
    )
    inp = cr.load_card_inputs(card)
    md = cr.render_markdown(inp)
    (row,) = [r for r in md.splitlines() if "`theta.extreme_share`" in r]
    html = ch.render_html(inp)
    data = json.loads(re.search(r'id="card-data">(.*?)</script>', html, re.S).group(1))
    (cell,) = [
        c
        for c in data["cells"]
        if (c["cell"]["n_items"], c["cell"]["n_persons"]) == (i, p)
    ]
    share = cell["derived"]["extreme_share"][0]
    assert share is not None and cr._pm([share, None]).split(" ")[0] in row
    # The paired MCSE is an approximation, not a bound.
    assert "conservative" not in md


def test_card_value_matches_the_record(card_dir):  # noqa: F811
    inp = cr.load_card_inputs(card_dir)
    cell = inp.evaluated()[0]
    ms = {m["key"]: m for m in ch.dashboard_metrics(inp)}
    cov = cell["persons"]["theta"]["coverage_0.95"]
    got = card_value(cell, ms["theta.coverage_dev_0.95"])
    assert got[0] == pytest.approx(cov[0] - 0.95) and got[1] == cov[1]
    assert difference([1.0, 0.3], [0.5, 0.4]) == pytest.approx([0.5, 0.5])
    assert difference([1.0, None], [0.5, 0.4]) == [0.5, None]
    assert difference(None, [0.5, 0.4]) is None
