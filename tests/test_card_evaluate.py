"""Tests for the model-card orchestrator (``card.evaluate_for_card``)."""

import gc
import json
import shutil
import weakref
from pathlib import Path

import pytest
import torch
from test_card_sampling import SPEC_2PL, FakeApproximator

from bayesflow_irt.card import (
    Cell,
    EvaluationProtocol,
    evaluate_cell,
    evaluate_for_card,
    main,
)
from bayesflow_irt.metadata import MetadataTracker, _sha256_file

SIDECARS = Path(__file__).resolve().parent / "fixtures" / "sidecars"
SIDECAR = SIDECARS / "2pl_equivariant_metadata.json"


def _setup(tmp_path, ranges=None, sha=None):
    """A dummy checkpoint next to a copy of the 2PL sidecar."""
    ckpt = tmp_path / "m.keras"
    ckpt.write_bytes(b"not a real checkpoint")
    md = json.loads(SIDECAR.read_text(encoding="utf-8"))
    md.setdefault("training_config", {}).setdefault("size_ranges", {})
    if ranges is not None:
        md["training_config"]["size_ranges"] = ranges
    if sha is not None:
        md.setdefault("provenance", {})["checkpoint_sha256"] = sha
    (tmp_path / "m_metadata.json").write_text(json.dumps(md), encoding="utf-8")
    return ckpt


def _protocol(**kw):
    kw.setdefault("datasets_per_cell", 20)
    kw.setdefault("posterior_draws", 30)
    kw.setdefault("persons_scored", 5)
    return EvaluationProtocol.preset("development", **kw)


SMALL = {"n_items": [5, 5], "n_persons": [10, 25]}


def test_writes_a_strict_complete_record(tmp_path):
    ckpt = _setup(tmp_path, SMALL)
    rec = evaluate_for_card(ckpt, protocol=_protocol(), out=tmp_path / "card",
                            approximator=FakeApproximator(sd=0.1))
    disk = json.loads((tmp_path / "card/evaluation.json").read_text("utf-8"))
    assert disk["complete"] and rec["complete"]
    assert [c["cell"]["n_persons"] for c in disk["cells"]] == [10, 25]
    assert disk["checkpoint"] == {"filename": "m.keras",
                                  "sha256": _sha256_file(ckpt),
                                  "match": "filename only"}
    assert disk["protocol"]["datasets_per_cell"] == 20
    assert "git_commit" in json.dumps(disk["session_info"])
    cell = disk["cells"][0]
    assert {"items", "joint", "persons", "sampling", "seed"} <= set(cell)
    assert "log_a" in cell["items"]
    assert disk["cells"][0]["persons"]["extreme_patterns"] > 0  # 5 items


def test_extreme_pattern_warning_is_counted_not_shown(tmp_path):
    import warnings

    ckpt = _setup(tmp_path, {"n_items": [5, 5], "n_persons": [10, 10]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        evaluate_for_card(ckpt, protocol=_protocol(),
                          approximator=FakeApproximator(sd=0.1))


def test_pcm_extreme_patterns_are_counted(tmp_path):
    import warnings

    pcm = SIDECARS / "pcm_equivariant_metadata.json"
    md = MetadataTracker.load(pcm).to_dict()
    proto = _protocol(datasets_per_cell=10)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        entry = evaluate_cell(FakeApproximator(), md["model_spec"],
                              Cell(3, 10, True), proto,
                              trained_at=md["provenance"]["timestamp"])
    assert "failed" not in entry
    assert entry["persons"]["extreme_patterns"] > 0  # 3 items


def test_failed_metrics_are_recorded_and_the_run_goes_on(tmp_path, monkeypatch):
    ckpt = _setup(tmp_path, SMALL)

    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr("bayesflow_irt.card.joint_metrics", boom)
    rec = evaluate_for_card(ckpt, protocol=_protocol(), out=tmp_path,
                            approximator=FakeApproximator())
    disk = json.loads((tmp_path / "evaluation.json").read_text("utf-8"))
    assert disk["complete"] and len(disk["failed_cells"]) == 2
    first = rec["cells"][0]
    assert first["failed"] == {"stage": "metrics", "message": "RuntimeError: boom"}
    assert "items" in first


def test_failed_oom_attempt_is_released_before_returning():
    """The last failed attempt's allocations do not outlive the cell."""

    class Holder:
        pass

    class AlwaysOOM:
        def __init__(self):
            self.refs = []

        def sample(self, conditions, num_samples):
            held = Holder()  # a local the traceback keeps alive
            self.refs.append(weakref.ref(held))
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (fake)")

    fake = AlwaysOOM()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        entry = evaluate_cell(fake, SPEC_2PL, Cell(5, 10, True), _protocol(),
                              chunk_size=4)
        assert entry["failed"]["stage"] == "sampling"
        assert len(fake.refs) == 4  # sizes 4, 3, 2 and 1
        assert all(ref() is None for ref in fake.refs)
    finally:
        if was_enabled:
            gc.enable()


def test_chunk_size_zero_is_rejected_up_front(tmp_path):
    with pytest.raises(ValueError, match="chunk_size"):
        evaluate_for_card(_setup(tmp_path, SMALL), protocol=_protocol(),
                          approximator=FakeApproximator(), chunk_size=0)


def test_recorded_hash_must_match(tmp_path):
    ckpt = _setup(tmp_path, SMALL, sha="0" * 64)
    with pytest.raises(ValueError, match="does not match"):
        evaluate_for_card(ckpt, protocol=_protocol(),
                          approximator=FakeApproximator())


def test_matching_hash_is_recorded(tmp_path):
    ckpt = _setup(tmp_path, SMALL)
    md_path = tmp_path / "m_metadata.json"
    md = json.loads(md_path.read_text("utf-8"))
    md["provenance"]["checkpoint_sha256"] = _sha256_file(ckpt)
    md_path.write_text(json.dumps(md), encoding="utf-8")
    rec = evaluate_for_card(ckpt, protocol=_protocol(datasets_per_cell=10),
                            approximator=FakeApproximator())
    assert rec["checkpoint"]["match"] == "sha256"


def _strip_timing(rec):
    rec.pop("seconds")
    rec["session_info"].pop("timestamp")
    for c in rec["cells"]:
        c.pop("metrics_seconds")
        c["sampling"].pop("seconds")
    return rec


def test_reproducible(tmp_path):
    ckpt = _setup(tmp_path, {"n_items": [5, 5], "n_persons": [10, 10]})
    one, two = (_strip_timing(evaluate_for_card(
        ckpt, protocol=_protocol(), approximator=FakeApproximator(sd=0.1)))
        for _ in range(2))
    assert one == two
    assert one["numba_kernel"] is False
    assert one["draws_validated_for"] == ["2pl"]


def test_out_of_range_cells_run_end_to_end(tmp_path):
    ckpt = _setup(tmp_path, {"n_items": [5, 5], "n_persons": [10, 10]})
    rec = evaluate_for_card(ckpt, protocol=_protocol(include_out_of_range=True,
                                                     datasets_per_cell=5),
                            approximator=FakeApproximator(sd=0.1))
    flags = {(c["cell"]["n_items"], c["cell"]["n_persons"]): c["cell"]["in_range"]
             for c in rec["cells"]}
    assert flags[(5, 10)] and not flags[(10, 25)] and not flags[(3, 10)]
    assert rec["failed_cells"] == []


def test_too_few_datasets_or_draws_are_rejected_up_front():
    with pytest.raises(ValueError, match="datasets_per_cell must be >= 3"):
        _protocol(datasets_per_cell=2)
    with pytest.raises(ValueError, match="posterior_draws must be >= 4"):
        _protocol(posterior_draws=3)


def test_hashless_sidecar_naming_another_checkpoint_is_rejected(tmp_path):
    ckpt = _setup(tmp_path, SMALL)
    md_path = tmp_path / "m_metadata.json"
    md = json.loads(md_path.read_text("utf-8"))
    md["provenance"]["checkpoint_filename"] = "other.keras"
    md_path.write_text(json.dumps(md), encoding="utf-8")
    with pytest.raises(ValueError, match="names checkpoint"):
        evaluate_for_card(ckpt, protocol=_protocol(),
                          approximator=FakeApproximator())


def test_unconventional_hashless_sidecar_is_unverified(tmp_path):
    ckpt = _setup(tmp_path, {"n_items": [5, 5], "n_persons": [10, 10]})
    other = tmp_path / "elsewhere.json"
    (tmp_path / "m_metadata.json").rename(other)
    rec = evaluate_for_card(ckpt, other, _protocol(datasets_per_cell=3),
                            out=tmp_path / "new/dir/e.json",
                            approximator=FakeApproximator())
    assert rec["checkpoint"]["match"] == "unverified"
    assert (tmp_path / "new/dir/e.json").exists()
    assert len(rec["cells"][0]["items"]["b"]["sbc_ecdf"]) == 39


def test_direct_construction_types_are_checked():
    kw = dict(preset_name="validation", datasets_per_cell=5, posterior_draws=100,
              include_out_of_range=False, persons_scored=10)
    with pytest.raises(TypeError, match="datasets_per_cell"):
        EvaluationProtocol(**{**kw, "datasets_per_cell": 3.5})
    with pytest.raises(TypeError, match="include_out_of_range"):
        EvaluationProtocol(**{**kw, "include_out_of_range": 1})


def test_direct_construction_is_checked_too():
    with pytest.raises(ValueError, match="datasets_per_cell"):
        EvaluationProtocol(preset_name="validation", datasets_per_cell=2,
                           posterior_draws=100, include_out_of_range=False,
                           persons_scored=10)


def test_public_entry_points_are_exported():
    import bayesflow_irt

    for name in ("evaluate_for_card", "evaluate_cell", "Cell", "CellFailedError"):
        assert name in bayesflow_irt.__all__ and hasattr(bayesflow_irt, name)


def test_cli_overrides_reach_the_record(tmp_path, monkeypatch):
    ckpt = _setup(tmp_path, {"n_items": [5, 5], "n_persons": [10, 10]})
    monkeypatch.setattr("keras.saving.load_model",
                        lambda *a, **k: FakeApproximator())
    main([str(ckpt), "--preset", "development", "--datasets-per-cell", "10",
          "--posterior-draws", "20", "--persons-scored", "4"])
    disk = json.loads((tmp_path / "m_card/evaluation.json").read_text("utf-8"))
    assert disk["protocol"]["overrides"] == [
        "datasets_per_cell", "persons_scored", "posterior_draws"]
    assert (tmp_path / "m_card/model_card.md").is_file()  # rendered by default
    shutil.rmtree(tmp_path / "m_card")
    main([str(ckpt), "--preset", "development", "--datasets-per-cell", "10",
          "--posterior-draws", "20", "--persons-scored", "4", "--no-render"])
    assert not (tmp_path / "m_card/model_card.md").exists()
    shutil.rmtree(tmp_path / "m_card")


def test_cli_render_with_a_json_out_path(tmp_path, monkeypatch, capsys):
    ckpt = _setup(tmp_path, {"n_items": [5, 5], "n_persons": [10, 10]})
    monkeypatch.setattr("keras.saving.load_model",
                        lambda *a, **k: FakeApproximator())
    small = ["--preset", "development", "--datasets-per-cell", "10",
             "--posterior-draws", "20", "--persons-scored", "4"]
    main([str(ckpt), *small, "--out", str(tmp_path / "x/evaluation.json")])
    assert (tmp_path / "x/model_card.md").is_file()
    main([str(ckpt), *small, "--out", str(tmp_path / "y/run.json")])
    assert not (tmp_path / "y/model_card.md").exists()
    assert "Not rendering" in capsys.readouterr().out


def test_cli_render_failure_keeps_the_evaluation(tmp_path, monkeypatch):
    ckpt = _setup(tmp_path, {"n_items": [5, 5], "n_persons": [10, 10]})
    monkeypatch.setattr("keras.saving.load_model",
                        lambda *a, **k: FakeApproximator())

    def broken(*a, **k):
        raise RuntimeError("renderer broke")

    monkeypatch.setattr("bayesflow_irt.card_render.render_model_card", broken)
    with pytest.warns(UserWarning, match="rendering the card failed"):
        main([str(ckpt), "--preset", "development", "--datasets-per-cell", "10",
              "--posterior-draws", "20", "--persons-scored", "4"])
    assert (tmp_path / "m_card/evaluation.json").is_file()
