"""Tests for the HTML dashboard (``card_html``)."""

import json
import re
import shutil
import subprocess

import numpy as np
import pytest
from test_card_render import _copy, card_dir  # noqa: F401 -- module fixture

import bayesflow_irt.card_html as ch
import bayesflow_irt.card_render as cr
from bayesflow_irt import render_model_card


def _payload(html: str) -> dict:
    m = re.search(
        r'<script type="application/json" id="card-data">(.*?)</script>', html, re.S
    )
    return json.loads(m.group(1))


def test_html_is_written_self_contained_and_parses(card_dir):  # noqa: F811
    render_model_card(card_dir)
    html = (card_dir / "model_card.html").read_text("utf-8")
    assert len(html.encode()) < ch.MAX_HTML_BYTES
    assert not re.search(r"<(script|link)[^>]+(src|href)=", html)  # no CDN
    data = _payload(html)
    assert data["parameters"] == [*cr.load_card_inputs(card_dir).parameters, "theta"]
    assert data["figures"]["priors"].startswith("data:image/png;base64,")
    assert data["reliability_label"] == cr.RELIABILITY_LABEL


def test_extrapolated_cells_reach_the_payload(card_dir, tmp_path):  # noqa: F811
    """The extrapolation switch reads out-of-range cells and the full axes."""
    import copy

    def edit(rec):
        far = copy.deepcopy(rec["cells"][0])
        far["cell"] = {"n_items": 3, "n_persons": 10, "in_range": False}
        rec["cells"].append(far)
        rec["protocol"]["include_out_of_range"] = True

    data = _payload(
        ch.render_html(cr.load_card_inputs(_copy(card_dir, tmp_path, edit)))
    )
    out = [c for c in data["cells"] if not c["cell"]["in_range"]]
    assert [(c["cell"]["n_items"], c["cell"]["n_persons"]) for c in out] == [(3, 10)]
    assert 3 in data["axes"]["items"] and 3 not in data["axes"]["in_items"]
    assert data["include_out_of_range"] is True


def test_banners_and_omitted_joint_reach_the_payload(card_dir, tmp_path):  # noqa: F811
    def edit(rec):
        rec["protocol"]["preset_name"] = "development"
        for c in rec["cells"]:
            c["joint"] = {"omitted": "not for this family"}

    data = _payload(
        ch.render_html(cr.load_card_inputs(_copy(card_dir, tmp_path, edit)))
    )
    assert data["banners"][0].startswith("Development run, not a validation")
    assert data["joint_omitted"] == "not for this family"
    assert not [m for m in data["metrics"] if m["tab"] == "joint"]


def test_html_is_deterministic(card_dir, tmp_path):  # noqa: F811
    out = _copy(card_dir, tmp_path)
    render_model_card(out)
    first = (out / "model_card.html").read_bytes()
    render_model_card(out)
    assert (out / "model_card.html").read_bytes() == first


def test_every_metric_resolves_in_some_cell(card_dir):  # noqa: F811
    inp = cr.load_card_inputs(card_dir)
    for m in ch.dashboard_metrics(inp):
        if "information" in m["key"] and "theta.reliability" in m["key"]:
            continue  # omitted for some families
        found = False
        for c in [c for c in ch.dashboard_payload(inp)["cells"] if "failed" not in c]:
            v = c
            for k in m["path"]:
                v = v.get(k) if isinstance(v, dict) else None
            if m["index"] is not None:
                v = v[m["index"]] if isinstance(v, list) else None
            found |= v is not None
        assert found, m["key"]


def test_author_text_cannot_close_the_script(card_dir, tmp_path):  # noqa: F811
    out = _copy(card_dir, tmp_path)
    side = tmp_path / "m_metadata.json"
    md = json.loads(side.read_text("utf-8"))
    md.setdefault("intended_use", {})["notes"] = "</script><script>alert(1)</script>"
    side.write_text(json.dumps(md), encoding="utf-8")
    html = ch.render_html(cr.load_card_inputs(out))
    assert html.count("</script>") == 2  # the data block and the page script
    assert _payload(html)["notes"] == "</script><script>alert(1)</script>"


def test_comment_opener_in_author_text_is_escaped(card_dir, tmp_path):  # noqa: F811
    out = _copy(card_dir, tmp_path)
    side = tmp_path / "m_metadata.json"
    md = json.loads(side.read_text("utf-8"))
    md.setdefault("intended_use", {})["notes"] = "a <!--<script> b & c"
    side.write_text(json.dumps(md), encoding="utf-8")
    html = ch.render_html(cr.load_card_inputs(out))
    block = re.search(r'id="card-data">(.*?)</script>', html, re.S).group(1)
    assert not set("<>&") & set(block)  # nothing the HTML tokenizer reads
    assert _payload(html)["notes"] == "a <!--<script> b & c"


def test_limits_do_not_depend_on_the_values(card_dir, tmp_path):  # noqa: F811
    def edit(rec):
        for c in rec["cells"]:
            c["items"]["b"]["width_0.95"] = [99.0, 0.1]
            c["persons"]["extreme_patterns"] = 10**6

    a = {
        m["key"]: m["limits"]
        for m in ch.dashboard_metrics(cr.load_card_inputs(card_dir))
    }
    b = {
        m["key"]: m["limits"]
        for m in ch.dashboard_metrics(
            cr.load_card_inputs(_copy(card_dir, tmp_path, edit))
        )
    }
    assert a == b
    inp = cr.load_card_inputs(card_dir)
    # b ~ normal(0, 2) in the fixture: 95% width 2 * 1.96 * 2.
    assert a["b.width_0.95"][1] == pytest.approx(7.84, abs=0.05)
    assert a["theta.extreme_share"] == [0.0, 1.0]
    assert inp  # fixture loads


def test_level_guides_and_tcc_thetas(card_dir):  # noqa: F811
    inp = cr.load_card_inputs(card_dir)
    ms = {m["key"]: m for m in ch.dashboard_metrics(inp)}
    assert "central 50% interval" in ms["b.coverage_dev_0.5"]["guide"]
    thetas = inp.evaluated()[0]["joint"]["tcc"]["thetas"]
    assert [k for k in ms if k.startswith("joint.tcc_bias_")] == [
        f"joint.tcc_bias_{t:g}" for t in thetas
    ]


def test_incomplete_record_counts_failures_from_cells(card_dir, tmp_path):  # noqa: F811
    def edit(rec):
        rec["complete"] = False
        rec.pop("failed_cells", None)
        rec["cells"][0] = {
            "cell": rec["cells"][0]["cell"],
            "failed": {"stage": "sampling", "message": "OOM"},
        }

    data = _payload(
        ch.render_html(cr.load_card_inputs(_copy(card_dir, tmp_path, edit)))
    )
    assert data["strip"]["failed"] == 1


def test_tarp_metrics_carry_their_null_floor(card_dir):  # noqa: F811
    ms = {m["key"]: m for m in ch.dashboard_metrics(cr.load_card_inputs(card_dir))}
    for ref in ("random", "classical"):
        assert ms[f"joint.tarp_{ref}"]["floor_path"][-1] == "null_floor"
        assert "null floor" in ms[f"joint.tarp_{ref}"]["guide"]
    assert "returns the prior" in ms["joint.tarp_random"]["guide"]


def test_assets_ship_as_package_data():
    from importlib import resources

    root = resources.files("bayesflow_irt").joinpath("_card_assets")
    assert root.joinpath("dashboard.js").is_file()
    assert root.joinpath("dashboard.css").is_file()


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_javascript_values_match_the_record(card_dir, tmp_path):  # noqa: F811
    """Run the page's own value() under Node on the payload."""
    from importlib import resources

    inp = cr.load_card_inputs(card_dir)
    data = ch.dashboard_payload(inp)
    (tmp_path / "payload.json").write_text(json.dumps(ch._finite(data)), "utf-8")
    js = resources.files("bayesflow_irt").joinpath("_card_assets", "dashboard.js")
    driver = tmp_path / "driver.js"
    driver.write_text(
        "const f = require(process.argv[2]);\n"
        "const D = JSON.parse(require('fs').readFileSync(process.argv[3]));\n"
        "const out = {};\n"
        "for (const c of D.cells) for (const m of D.metrics) {\n"
        "  const v = c.failed ? null : f.value(c, m);\n"
        "  out[c.cell.n_items + 'x' + c.cell.n_persons + '|' + m.key] =\n"
        "    v && { v: v.v, se: v.se };\n"
        "}\n"
        "console.log(JSON.stringify(out));\n",
        encoding="utf-8",
    )
    res = subprocess.run(
        ["node", str(driver), str(js), str(tmp_path / "payload.json")],
        capture_output=True,
        text=True,
        check=True,
    )
    got = json.loads(res.stdout)
    c = inp.evaluated()[0]
    name = f"{c['cell']['n_items']}x{c['cell']['n_persons']}"
    cov = c["items"]["b"]["coverage_0.95"]
    assert got[f"{name}|b.coverage_dev_0.95"]["v"] == pytest.approx(cov[0] - 0.95)
    assert got[f"{name}|b.coverage_dev_0.95"]["se"] == pytest.approx(cov[1])
    d, dse = c["joint"]["tcc"]["d_mean"], c["joint"]["tcc"]["d_mean_se"]
    t1 = f"joint.tcc_bias_{c['joint']['tcc']['thetas'][1]:g}"
    assert got[f"{name}|{t1}"] == {
        **got[f"{name}|{t1}"],
        "v": pytest.approx(d[1]),
        "se": pytest.approx(dse[1]),
    }
    assert got[f"{name}|theta.extreme_share"]["se"] is None


def test_non_finite_values_become_null(card_dir, tmp_path):  # noqa: F811
    def edit(rec):
        rec["cells"][0]["items"]["b"]["bias"] = [float("nan"), float("inf")]

    out = _copy(card_dir, tmp_path, edit)
    data = _payload(ch.render_html(cr.load_card_inputs(out)))
    assert data["cells"][0]["items"]["b"]["bias"] == [None, None]


def test_size_cap(card_dir, tmp_path, monkeypatch):  # noqa: F811
    out = _copy(card_dir, tmp_path)
    monkeypatch.setattr(ch, "MAX_HTML_BYTES", 1000)
    with pytest.raises(ValueError, match="MB cap"):
        ch.write_html(cr.load_card_inputs(out), out, None)


def test_bins_carry_wilson_intervals_and_small_n(card_dir):  # noqa: F811
    inp = cr.load_card_inputs(card_dir)
    bins = [b for c in inp.evaluated() for b in ch._bins(c)]
    assert any(b["small"] for b in bins)
    for b in bins:
        if b["coverage"] is not None:
            lo, hi = b["interval"]
            assert lo <= b["coverage"] <= hi


def test_format2_bins_use_the_clustered_se():
    b = {"edges": [0.0, 0.5], "n": 400, "clusters": 25, "coverage_0.95": [0.9, 0.02]}
    bc = cr.bin_coverage(b)
    assert bc["kind"] == "cluster-robust"
    assert bc["interval"] == pytest.approx([0.9 - 0.0392, 0.9 + 0.0392])
    assert bc["small"]  # 25 datasets < MIN_BIN_CLUSTERS
    top = cr.bin_coverage({**b, "clusters": 80, "coverage_0.95": [1.0, 0.0]})
    assert top["interval"] == [1.0, 1.0] and not top["small"]


def test_sbc_band_is_simultaneous_at_its_level():
    # Säilynoja et al. (2022): the band holds the whole uniform ECDF on the
    # grid with probability `confidence`. Check by fresh simulation.
    from bayesflow_irt.card_metrics import SBC_GRID

    n = 100
    band = ch.sbc_band(n)
    lo, hi = np.asarray(band["lower"]), np.asarray(band["upper"])
    assert band["grid"] == list(SBC_GRID)
    assert np.all(lo <= SBC_GRID) and np.all(SBC_GRID <= hi)
    u = np.random.default_rng(123).uniform(size=(4000, n))
    ecdf = (u[:, :, None] <= SBC_GRID).mean(axis=1)
    inside = np.all((ecdf >= lo) & (ecdf <= hi), axis=1).mean()
    assert 0.92 <= inside <= 0.98
    assert ch.sbc_band(n) == band  # deterministic


def test_band_coverage_recursion_is_exact():
    # §2.3 recursion against brute force: n = 3 ranks, two grid points, all
    # 3-point uniform samples enumerated via the multinomial over the three
    # intervals [0, z1], (z1, z2], (z2, 1].
    from math import factorial

    n, z = 3, np.array([0.3, 0.7])
    gamma = 0.4
    lo, hi = ch._band_limits(n, z, gamma)
    widths = np.diff(np.r_[0.0, z, 1.0])
    exact = 0.0
    for a in range(n + 1):
        for b in range(n + 1 - a):
            c = n - a - b
            if lo[0] <= a <= hi[0] and lo[1] <= a + b <= hi[1]:
                exact += (
                    factorial(n)
                    / (factorial(a) * factorial(b) * factorial(c))
                    * widths[0] ** a
                    * widths[1] ** b
                    * widths[2] ** c
                )
    assert np.isclose(ch._band_coverage(n, z, gamma), exact)


def test_every_metric_has_a_page_axis(card_dir):  # noqa: F811
    metrics = ch.dashboard_metrics(cr.load_card_inputs(card_dir))
    axis = {m["key"]: m["axis"] for m in metrics}
    assert set(axis.values()) <= {"recovery", "calibration"}
    assert axis["theta.coverage_dev_0.95"] == "calibration"
    assert axis["theta.mean_calibration_error"] == "calibration"
    assert axis["theta.nrmse_excess"] == "recovery"
    assert axis["theta.width_0.95"] == "recovery"


def test_prior_density_curves_integrate_to_about_one(card_dir):  # noqa: F811
    curves = ch.prior_density_curves(cr.load_card_inputs(card_dir))
    assert curves
    for c in curves:
        x, y = np.asarray(c["x"]), np.asarray(c["y"])
        assert len(x) == len(y) == 80
        # The curve covers the central 99.8% of the draws.
        assert np.sum(y) * (x[1] - x[0]) == pytest.approx(0.998, abs=0.01)
        assert x[0] < c["interval"][0] < c["interval"][1] < x[-1]


def test_network_groups_flatten_a_serialized_config():
    record = {
        "network": {
            "n_parameters": 1234,
            "source": "test",
            "config": {
                "class_name": "Approx",
                "config": {
                    "name": "approx",
                    "flow_chunk_size": None,
                    "item_posterior": "flow",
                    "summary_network": {
                        "class_name": "Summary",
                        "config": {
                            "name": "s",
                            "dtype": {"class_name": "DTypePolicy"},
                            "embed_dim": 32,
                            "raw_feature_channels": ["a", "b"],
                        },
                    },
                    "inference_network": {
                        "class_name": "CouplingFlow",
                        "config": {
                            "depth": 6,
                            "base_distribution": {
                                "class_name": "DiagonalNormal",
                                "config": {
                                    "mean": {"class_name": "__tensor__"},
                                    "trainable_parameters": False,
                                },
                            },
                        },
                    },
                },
            },
        }
    }
    groups = ch.network_groups(record)
    tables = {g["title"]: dict(g["rows"]) for g in groups if "rows" in g}
    assert tables["Approximator (Approx)"] == {
        "n_parameters": "1,234",
        "flow_chunk_size": "None",
        "item_posterior": "flow",
    }
    assert tables["Summary network (Summary)"] == {
        "embed_dim": "32",
        "raw_feature_channels": '["a", "b"]',
    }
    assert tables["Inference network (CouplingFlow)"] == {
        "depth": "6",
        "base_distribution": "DiagonalNormal",
        "base_distribution.trainable_parameters": "false",
    }
    assert groups[-1] == {"source": "test"}
    assert ch.network_groups({}) is None


def test_network_config_matches_what_save_writes(tmp_path):
    import zipfile

    import keras

    from bayesflow_irt.card import network_config

    model = keras.Sequential([keras.Input((3,)), keras.layers.Dense(2)])
    net = network_config(model)
    assert net["n_parameters"] == 8
    model.save(tmp_path / "m.keras")
    with zipfile.ZipFile(tmp_path / "m.keras") as z:
        saved = json.loads(z.read("config.json"))
    assert net["config"]["class_name"] == saved["class_name"] == "Sequential"
    assert net["config"]["config"]["layers"] == saved["config"]["layers"]
