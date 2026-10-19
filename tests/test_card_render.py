"""Tests for the Markdown model card (``card_render.render_model_card``)."""

import copy
import json
import re

import pytest
from test_card_evaluate import SIDECARS, SMALL, _protocol, _setup
from test_card_sampling import FakeApproximator

import bayesflow_irt.card_render as cr
from bayesflow_irt import evaluate_for_card, render_model_card
from bayesflow_irt.metadata import MetadataTracker

PCM_SIDECAR = SIDECARS / "pcm_equivariant_metadata.json"


@pytest.fixture(scope="module")
def card_dir(tmp_path_factory):
    """A small evaluated card: two in-range cells, sidecar next to the card."""
    root = tmp_path_factory.mktemp("card")
    ckpt = _setup(root, SMALL)
    evaluate_for_card(ckpt, protocol=_protocol(), out=root / "m_card",
                      approximator=FakeApproximator(sd=0.1))
    return root / "m_card"


def _copy(card_dir, tmp_path, edit=None):
    """Copy a card (record plus the sidecar beside it), optionally editing it."""
    out = tmp_path / "card"
    out.mkdir()
    rec = json.loads((card_dir / "evaluation.json").read_text("utf-8"))
    if edit:
        edit(rec)
    (out / "evaluation.json").write_text(json.dumps(rec), encoding="utf-8")
    side = card_dir.parent / "m_metadata.json"
    (tmp_path / "m_metadata.json").write_bytes(side.read_bytes())
    return out


def test_writes_the_card_and_its_figures(card_dir):
    path = render_model_card(card_dir)
    text = path.read_text("utf-8")
    for heading in ("## Summary", "## The model", "## Supported data",
                    "## Priors", "## Key metrics across cells", "## Limitations",
                    "## Usage", "## Appendix: per-cell results", "## Provenance"):
        assert heading in text
    assert "P(y = 1) = sigmoid(a (theta - b))" in text
    assert cr.RELIABILITY_LABEL in text
    assert "not evaluated; run with `--include-out-of-range`" in text
    assert (card_dir / "figures/priors.png").is_file()
    assert (card_dir / "figures/heatmap_b_coverage_dev_0_95.png").is_file()
    assert not re.search(r"\bnan\b|\bnone\b", text.lower())
    assert "Colour limits: [-0.15, 0.15]." in text
    assert "**Author's note**" not in text  # the fixture sidecar has no notes


def test_card_numbers_match_the_record(card_dir):
    inp = cr.load_card_inputs(card_dir)
    text = cr.render_markdown(inp)
    cell = inp.record["cells"][0]
    cov = cell["items"]["b"]["coverage_0.95"]
    rel = cell["persons"]["reliability"]["posterior_variance"]["mean"]
    tarp = cell["joint"]["tarp"]["random"]["tarp_error"]
    assert f"{cov[0] - 0.95:.3f} ({cov[1]:.3f})" in text
    assert f"{rel[0]:.3f} ({rel[1]:.3f})" in text
    assert f"{tarp[0]:.3f} ({tarp[1]:.3f})" in text


def test_nan_metric_renders_as_a_dash(card_dir, tmp_path):
    def edit(rec):
        rec["cells"][0]["items"]["b"]["coverage_0.95"] = [None, None]

    text = render_model_card(_copy(card_dir, tmp_path, edit)).read_text("utf-8")
    assert "| coverage_0.95 | " in text and "– |" in text
    assert "nan" not in text.lower().split()


def test_author_note_is_labelled(card_dir, tmp_path):
    out = _copy(card_dir, tmp_path)
    side = tmp_path / "m_metadata.json"
    md = json.loads(side.read_text("utf-8"))
    md.setdefault("intended_use", {})["notes"] = "Example note."
    side.write_text(json.dumps(md), encoding="utf-8")
    text = cr.render_markdown(cr.load_card_inputs(out))
    assert "> **Author's note** (written by the model's author, not generated): " \
        "Example note." in text


def test_figures_are_byte_identical_across_renders(card_dir, tmp_path):
    first = _copy(card_dir, tmp_path)
    render_model_card(first)
    a = {p.name: p.read_bytes() for p in (first / "figures").iterdir()}
    render_model_card(first)
    b = {p.name: p.read_bytes() for p in (first / "figures").iterdir()}
    assert a == b


def _edit_spec(card_dir, tmp_path, fn):
    """Copy the card and apply ``fn`` to the model_spec of record and sidecar."""
    out = _copy(card_dir, tmp_path, lambda rec: fn(rec["model_spec"]))
    side = tmp_path / "m_metadata.json"
    md = MetadataTracker.load(side).to_dict()
    fn(md["model_spec"])
    side.write_text(json.dumps(md), encoding="utf-8")
    return out


@pytest.mark.parametrize("kind", ["traversal", "absolute", "drive", ".."])
def test_sidecar_name_must_be_a_bare_file_name(card_dir, tmp_path, kind):
    """metadata.filename comes from the record, so it cannot select a file
    outside the card directory and its parent."""
    outside = tmp_path / "private" / "sidecar.json"
    outside.parent.mkdir()
    outside.write_bytes((card_dir.parent / "m_metadata.json").read_bytes())
    name = {
        "traversal": "../private/sidecar.json",
        "absolute": str(outside),
        "drive": "C:sidecar.json",
        "..": "..",
    }[kind]
    work = tmp_path / "w"
    work.mkdir()
    out = _copy(card_dir, work, lambda r: r["metadata"].update(filename=name))
    with pytest.raises(FileNotFoundError, match="not a bare file name"):
        cr.load_card_inputs(out)
    # An explicit sidecar stays possible.
    assert cr.load_card_inputs(out, metadata=outside) is not None


def test_sidecar_of_another_model_is_rejected(card_dir, tmp_path):
    out = _copy(card_dir, tmp_path)
    side = tmp_path / "m_metadata.json"
    md = MetadataTracker.load(side).to_dict()
    md["model_spec"]["priors"] = {"theta": {"name": "normal", "loc": 1.0,
                                            "scale": 1.0}}
    side.write_text(json.dumps(md), encoding="utf-8")
    with pytest.raises(ValueError, match="model_spec differs"):
        cr.load_card_inputs(out)
    (tmp_path / "u").mkdir()
    out = _copy(card_dir, tmp_path / "u", lambda r: r["metadata"].update(
        metadata_uuid="someone-else"))
    md = MetadataTracker.load(card_dir.parent / "m_metadata.json").to_dict()
    md["metadata_uuid"] = "mine"
    (tmp_path / "u" / "m_metadata.json").write_text(json.dumps(md), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata_uuid"):
        cr.load_card_inputs(out)


def test_sidecar_disagreeing_with_the_record_is_rejected(card_dir, tmp_path):
    out = _edit_spec(card_dir, tmp_path, lambda s: s.update(
        size_ranges={"n_items": [10, 50], "n_persons": [10, 500]}))
    with pytest.raises(ValueError, match="classified differently"):
        cr.load_card_inputs(out)


def test_unknown_family_is_rejected(card_dir, tmp_path):
    out = _edit_spec(card_dir, tmp_path, lambda s: s.update(model="3PL"))
    with pytest.raises(ValueError, match="Unsupported model family"):
        cr.load_card_inputs(out)


def test_exact_trained_bounds_stay_on_the_axes(card_dir, tmp_path):
    def edit(rec):
        far = copy.deepcopy(rec["cells"][0])
        far["cell"].update(n_persons=7, in_range=False)  # off the standard grid
        rec["cells"].append(far)
        rec["protocol"]["include_out_of_range"] = True

    inp = cr.load_card_inputs(_copy(card_dir, tmp_path, edit))
    assert 7 in inp.persons_axis


def test_development_and_incomplete_are_stamped(card_dir, tmp_path):
    def edit(rec):
        rec["protocol"]["preset_name"] = "development"
        rec["complete"] = False
        rec["cells"][0] = {"cell": rec["cells"][0]["cell"],
                           "failed": {"stage": "sampling", "message": "OOM"}}
        rec.pop("failed_cells", None)

    text = cr.render_markdown(cr.load_card_inputs(_copy(card_dir, tmp_path, edit)))
    assert "> **Development run, not a validation report" in text
    assert "> **Incomplete evaluation" in text
    assert ", 1 failed;" in text  # counted from the cells, not failed_cells


def test_tail_coverage_note_and_backend_in_snippet(card_dir):
    text = cr.render_markdown(cr.load_card_inputs(card_dir))
    assert text.count(cr.TAIL_COVERAGE_NOTE) >= 2  # limitations and appendix
    snippet = text.split("## Usage")[1]
    assert snippet.index("KERAS_BACKEND") < snippet.index("import keras")


def test_both_tarp_references_are_rendered(card_dir):
    render_model_card(card_dir)
    text = (card_dir / "model_card.md").read_text("utf-8")
    assert "joint: TARP error, CTT-statistic references" in text
    assert (card_dir / "figures/tarp_curves.png").is_file()


def test_trained_map_reads_the_nested_axis_records(card_dir, tmp_path):
    sampling = {
        "size_distribution": "uniform", "rule": "log_spaced_buckets+uniform",
        "n_items": {"range": [5, 5], "bucket": None, "values": [5],
                    "probabilities": [1.0]},
        "n_persons": {"range": [10, 25], "bucket": 3, "values": None,
                      "probabilities": None},
    }
    out = _edit_spec(card_dir, tmp_path, lambda s: s.update(size_sampling=sampling))
    inp = cr.load_card_inputs(out)
    assert "needs matplotlib" in cr.render_markdown(inp)  # no figure passed
    assert "Training size sampling not recorded" not in cr.render_markdown(inp)
    assert cr._trained_map(inp, tmp_path / "map.png")
def test_all_failed_record_renders(card_dir, tmp_path):
    def edit(rec):
        rec["cells"] = [{"cell": c["cell"], "failed": {"stage": "sampling",
                                                       "message": "OOM"}}
                        for c in rec["cells"]]

    out = _copy(card_dir, tmp_path, edit)
    text = render_model_card(out).read_text("utf-8")
    assert "Validated range (evaluated in-range cells) | none; failed cells:" in text
    assert not (out / "figures/coverage_curves.png").exists()


def test_item_posterior_limitation(card_dir, tmp_path):
    out = _copy(card_dir, tmp_path)
    side = tmp_path / "m_metadata.json"
    md = json.loads(side.read_text("utf-8"))
    for value, expected in ((None, "item-posterior type is not recorded"),
                            ("flow_matching", "joint statements across items are "
                                              "overconfident"),
                            ("autoregressive_coupling_flow", None)):
        md.setdefault("architecture", {})["item_posterior"] = value
        side.write_text(json.dumps(md), encoding="utf-8")
        text = cr.render_markdown(cr.load_card_inputs(out))
        if expected:
            assert expected in text
        else:
            assert "drawn independently" not in text
            assert "not recorded, so whether items" not in text


def test_pcm_card(tmp_path):
    md = MetadataTracker.load(PCM_SIDECAR).to_dict()
    md["model_spec"]["size_ranges"] = {"n_items": [5, 5], "n_persons": [10, 10]}
    ckpt = tmp_path / "p.keras"
    ckpt.write_bytes(b"not a real checkpoint")
    (tmp_path / "p_metadata.json").write_text(json.dumps(md), encoding="utf-8")
    evaluate_for_card(ckpt, protocol=_protocol(), out=tmp_path / "p_card",
                      approximator=FakeApproximator(sd=0.1))
    text = render_model_card(tmp_path / "p_card").read_text("utf-8")
    assert "delta: coverage at 0.95 minus 0.95" in text
    assert "log_a" not in text
    assert "omitted | no PCM/GPCM information source" in text
    assert "not computed for this family" in text
    assert "Joint metrics (TARP, TCC) not computed:" in text
    assert "### joint: TARP error" not in text


def test_usage_snippet_check_data_line_runs(card_dir):
    import numpy as np

    from bayesflow_irt import check_data

    side = card_dir.parent / "m_metadata.json"
    responses = np.random.default_rng(0).integers(0, 2, size=(10, 5))
    check_data(responses, metadata=str(side))  # the snippet's call, verbatim API


def test_markdown_is_deterministic(card_dir):
    inp = cr.load_card_inputs(card_dir)
    assert cr.render_markdown(inp) == cr.render_markdown(cr.load_card_inputs(card_dir))


def test_without_matplotlib_tables_only(card_dir, tmp_path, monkeypatch):
    out = _copy(card_dir, tmp_path)
    monkeypatch.setattr(cr, "_have_matplotlib", lambda: False)
    with pytest.warns(UserWarning, match="matplotlib is not installed"):
        path = render_model_card(out)
    assert not (out / "figures").exists()
    assert "![" not in path.read_text("utf-8")


def test_failed_and_extrapolated_cells_are_shown(card_dir, tmp_path):
    def edit(rec):
        rec["cells"][1] = {"cell": rec["cells"][1]["cell"],
                           "failed": {"stage": "sampling", "message": "OOM"}}
        far = copy.deepcopy(rec["cells"][0])
        far["cell"] = {"n_items": 3, "n_persons": 10, "in_range": False}
        rec["cells"].append(far)
        rec["protocol"]["include_out_of_range"] = True

    text = render_model_card(_copy(card_dir, tmp_path, edit)).read_text("utf-8")
    assert "failed (sampling)" in text
    assert "Failed at stage `sampling`: OOM" in text
    assert "Outside the trained range (extrapolation):" in text
    assert "### Cell 3 x 10 (extrapolation)" in text


def test_null_identity_fields_read_not_recorded(card_dir, tmp_path):
    out = _copy(card_dir, tmp_path)
    side = tmp_path / "m_metadata.json"
    md = json.loads(side.read_text("utf-8"))
    md["metadata_uuid"] = None
    md.setdefault("provenance", {})["git_commit"] = None
    side.write_text(json.dumps(md), encoding="utf-8")
    text = cr.render_markdown(cr.load_card_inputs(out))
    assert "| Git commit (training) | not recorded |" in text
    assert "| metadata_uuid | not recorded |" in text


def test_small_theta_bins_are_marked(card_dir):
    text = cr.render_markdown(cr.load_card_inputs(card_dir))
    # 20 datasets: every bin has fewer than MIN_BIN_CLUSTERS.
    assert f"(datasets < {cr.MIN_BIN_CLUSTERS})" in text
    assert "cluster-robust by dataset" in text


def test_format1_bins_still_render_with_wilson(card_dir, tmp_path):
    def edit(rec):
        rec["evaluation_format"] = 1
        for c in rec["cells"]:
            for b in c.get("persons", {}).get("conditional", []):
                b.pop("clusters", None)
                b["n"] = 5

    text = cr.render_markdown(cr.load_card_inputs(_copy(card_dir, tmp_path, edit)))
    assert f"(n < {cr.MIN_BIN_N})" in text
    assert "95% Wilson interval" in text


def test_wilson_interval():
    # Closed form of the score interval (Newcombe, 1998, method 3).
    k, n, z = 7, 50, 1.959963984540054
    centre = (k + z**2 / 2) / (n + z**2)
    half = z / (n + z**2) * (k * (n - k) / n + z**2 / 4) ** 0.5
    lo, hi = cr.wilson_interval(k, n)
    assert lo == pytest.approx(centre - half, abs=1e-9)
    assert hi == pytest.approx(centre + half, abs=1e-9)
    lo, hi = cr.wilson_interval(35, 35)
    assert hi == pytest.approx(1.0) and 0.85 < lo < 1.0  # non-zero width at n of n


def test_unknown_format_and_missing_sidecar_are_rejected(card_dir, tmp_path):
    out = _copy(card_dir, tmp_path, lambda r: r.update(evaluation_format=99))
    with pytest.raises(ValueError, match="evaluation_format"):
        render_model_card(out)
    lonely = tmp_path / "elsewhere" / "lonely"
    lonely.mkdir(parents=True)
    record = (card_dir / "evaluation.json").read_bytes()
    (lonely / "evaluation.json").write_bytes(record)
    with pytest.raises(FileNotFoundError, match="pass it explicitly"):
        render_model_card(lonely)
