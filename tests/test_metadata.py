"""Tests for bayesflow_irt.metadata."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from bayesflow_irt.metadata import MetadataTracker

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tracker():
    """Standard MetadataTracker instance for testing."""
    return MetadataTracker(model="1PL", variant="fixed", version="0.1.0")


@pytest.fixture
def tracker_with_hpo(tracker):
    """Tracker with HPO results already added."""
    tracker.add_hpo_results(num_trials=100, best_params={"lr": 0.001})
    return tracker


# ---------------------------------------------------------------------------
# _sanitize_for_json
# ---------------------------------------------------------------------------


class TestSanitizeForJson:
    """Test the internal _sanitize_for_json function."""

    def test_numpy_int(self):
        from bayesflow_irt.metadata import _sanitize_for_json

        assert _sanitize_for_json(np.int64(42)) == 42
        assert isinstance(_sanitize_for_json(np.int64(42)), int)

    def test_numpy_float(self):
        from bayesflow_irt.metadata import _sanitize_for_json

        result = _sanitize_for_json(np.float64(3.14))
        assert abs(result - 3.14) < 1e-10
        assert isinstance(result, float)

    def test_numpy_array(self):
        from bayesflow_irt.metadata import _sanitize_for_json

        arr = np.array([1, 2, 3])
        result = _sanitize_for_json(arr)
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_nested_dict(self):
        from bayesflow_irt.metadata import _sanitize_for_json

        data = {"a": np.int32(1), "b": {"c": np.float64(2.0)}}
        result = _sanitize_for_json(data)
        assert result == {"a": 1, "b": {"c": 2.0}}
        assert isinstance(result["a"], int)

    def test_list_with_numpy(self):
        from bayesflow_irt.metadata import _sanitize_for_json

        data = [np.int64(1), "hello", np.float32(2.5)]
        result = _sanitize_for_json(data)
        assert result == [1, "hello", pytest.approx(2.5)]

    def test_plain_types_pass_through(self):
        from bayesflow_irt.metadata import _sanitize_for_json

        assert _sanitize_for_json("hello") == "hello"
        assert _sanitize_for_json(42) == 42
        assert _sanitize_for_json(3.14) == 3.14
        assert _sanitize_for_json(True) is True
        assert _sanitize_for_json(None) is None

    def test_empty_structures(self):
        from bayesflow_irt.metadata import _sanitize_for_json

        assert _sanitize_for_json({}) == {}
        assert _sanitize_for_json([]) == []


# ---------------------------------------------------------------------------
# MetadataTracker.__init__
# ---------------------------------------------------------------------------


class TestMetadataTrackerInit:
    def test_stores_model_spec(self, tracker):
        data = tracker.to_dict()
        assert data["schema_version"] == 2
        assert data["model_spec"] == {
            "model": "1PL",
            "variant": "fixed",
            "version": "0.1.0",
        }

    def test_no_uuid_before_attach(self, tracker):
        assert tracker._metadata_uuid is None


# ---------------------------------------------------------------------------
# Phase builders
# ---------------------------------------------------------------------------


class TestPhaseBuilders:
    def test_add_training_config(self, tracker):
        tracker.add_training_config(
            priors={"b": "N(0,1)"},
            size_ranges={"I": [5, 15], "P": [50, 200]},
            hparams={"lr": 0.001},
        )
        data = tracker.to_dict()
        assert data["model_spec"]["priors"] == {"b": "N(0,1)"}
        assert data["model_spec"]["size_ranges"]["I"] == [5, 15]
        assert data["training"]["lr"] == 0.001

    def test_hparams_are_split_into_sections(self, tracker):
        tracker.add_training_config(
            size_ranges={"n_items": [3, 9], "n_categories": 4},
            hparams={
                "summary": {"embed_dim": 32},
                "flow": {"widths": [128]},
                "base_checkpoint": "base.keras",
                "refine_epochs": 2,
                "epochs": 8,
            },
            seed=7,
            internal_parameterization="loading",
        )
        data = tracker.to_dict()
        assert data["architecture"] == {
            "summary": {"embed_dim": 32},
            "flow": {"widths": [128]},
        }
        assert data["training"]["epochs"] == 8
        assert data["training"]["seed"] == 7
        assert data["training"]["refinement"] == {
            "base_checkpoint": "base.keras",
            "refine_epochs": 2,
        }
        assert data["model_spec"]["n_categories"] == 4
        assert data["model_spec"]["size_ranges"] == {"n_items": [3, 9]}
        assert data["model_spec"]["internal_parameterization"] == "loading"

    def test_repeated_training_config_merges(self, tracker):
        tracker.add_training_config(priors={"b": "N(0,1)"}, hparams={"epochs": 8})
        tracker.add_training_config(seed=3)
        data = tracker.to_dict()
        assert data["model_spec"]["priors"] == {"b": "N(0,1)"}
        assert data["training"] == {"epochs": 8, "seed": 3}

    def test_add_training_config_partial(self, tracker):
        tracker.add_training_config(priors={"b": "N(0,1)"})
        spec = tracker.to_dict()["model_spec"]
        assert "priors" in spec
        assert "size_ranges" not in spec
        assert "size_sampling" not in spec

    def test_add_training_config_records_size_sampling(self, tracker):
        sampling = {"rule": "log_uniform", "n_items": {"values": [5, 6]}}
        tracker.add_training_config(size_sampling=sampling)
        assert tracker.to_dict()["model_spec"]["size_sampling"] == sampling

    def test_add_hpo_results(self, tracker):
        tracker.add_hpo_results(num_trials=100, best_params={"lr": 0.001})
        results = tracker.to_dict()["hpo_results"]
        assert results["num_trials"] == 100
        assert results["best_params"] == {"lr": 0.001}

    def test_add_phase2_metrics(self, tracker):
        tracker.add_phase2_metrics(
            epochs=150,
            final_sbc_ecdf=0.058,
            final_coverage=0.947,
            final_nrmse=0.23,
        )
        training = tracker.to_dict()["training"]
        assert training["epochs_completed"] == 150
        assert training["final_metrics"] == {
            "sbc_ecdf": 0.058,
            "coverage": 0.947,
            "nrmse": 0.23,
        }

    def test_add_phase3_metrics_applied(self, tracker):
        tracker.add_phase3_metrics(applied=True, epochs=30)
        metrics = tracker.to_dict()["phase3_metrics"]
        assert metrics["applied"] is True
        assert metrics["epochs"] == 30

    def test_add_phase3_metrics_not_applied(self, tracker):
        tracker.add_phase3_metrics(applied=False)
        metrics = tracker.to_dict()["phase3_metrics"]
        assert metrics["applied"] is False
        assert "epochs" not in metrics

    def test_add_phase4_metrics(self, tracker):
        tracker.add_phase4_metrics(
            ood_validation={"corner": (5, 5), "nrmse": 0.3},
            robustness_analysis={"max_degradation": 1.8},
        )
        metrics = tracker.to_dict()["phase4_metrics"]
        assert metrics["ood_validation"]["corner"] == (5, 5)


# ---------------------------------------------------------------------------
# UUID key-lock
# ---------------------------------------------------------------------------


class TestAttachVerify:
    def test_attach_sets_uuid(self, tracker):
        model = MagicMock()
        uid = tracker.attach(model)
        assert isinstance(uid, str)
        assert len(uid) == 36  # UUID4 format

    def test_attach_stores_uuid_in_data(self, tracker):
        model = MagicMock()
        uid = tracker.attach(model)
        assert tracker.to_dict()["metadata_uuid"] == uid

    def test_attach_sets_model_attributes(self, tracker):
        model = MagicMock()
        uid = tracker.attach(model)
        assert model.metadata_uuid == uid
        assert model.metadata["metadata_uuid"] == uid

    def test_verify_match(self, tracker):
        model = MagicMock()
        tracker.attach(model)
        assert tracker.verify(model) is True

    def test_verify_mismatch(self, tracker):
        model = MagicMock()
        tracker.attach(model)
        model.metadata_uuid = "wrong-uuid"
        assert tracker.verify(model) is False

    def test_verify_no_uuid(self, tracker):
        model = MagicMock(spec=[])  # no attributes
        assert tracker.verify(model) is False

    def test_reattach_overwrites(self, tracker):
        model = MagicMock()
        uid1 = tracker.attach(model)
        uid2 = tracker.attach(model)
        assert uid1 != uid2
        assert tracker.verify(model) is True


# ---------------------------------------------------------------------------
# validate()
# ---------------------------------------------------------------------------


class TestValidate:
    def test_empty_tracker_valid(self, tracker):
        """model_info is always present; no provenance yet, no phase data."""
        errors = tracker.validate()
        assert errors == []

    def test_after_save_valid(self, tracker, tmp_path):
        tracker.add_hpo_results(num_trials=50)
        model = MagicMock()
        tracker.attach(model)
        tracker.save(tmp_path / "test_metadata.json")
        errors = tracker.validate()
        assert errors == []

    def test_provenance_missing_required_key(self, tracker):
        # Manually set incomplete provenance
        tracker._data["provenance"] = {"git_commit": "abc"}  # missing timestamp
        errors = tracker.validate()
        assert any("provenance" in e for e in errors)


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_round_trip(self, tracker_with_hpo, tmp_path):
        path = tmp_path / "meta.json"
        tracker_with_hpo.add_phase2_metrics(epochs=150, final_sbc_ecdf=0.058)
        model = MagicMock()
        tracker_with_hpo.attach(model)
        tracker_with_hpo.save(path)

        loaded = MetadataTracker.load(path)
        assert loaded.to_dict()["model_spec"]["model"] == "1PL"
        assert loaded.to_dict()["hpo_results"]["num_trials"] == 100
        assert loaded.to_dict()["training"]["epochs_completed"] == 150
        assert loaded._metadata_uuid == tracker_with_hpo._metadata_uuid

    def test_round_trip_preserves_uuid_in_data(self, tracker, tmp_path):
        path = tmp_path / "meta.json"
        tracker.add_hpo_results(num_trials=10)
        model = MagicMock()
        uid = tracker.attach(model)
        tracker.save(path)

        loaded = MetadataTracker.load(path)
        assert loaded.to_dict()["metadata_uuid"] == uid

    def test_provenance_auto_filled(self, tracker, tmp_path):
        path = tmp_path / "meta.json"
        tracker.add_hpo_results(num_trials=10)
        model = MagicMock()
        tracker.attach(model)
        tracker.save(path)

        with open(path) as f:
            data = json.load(f)
        prov = data["provenance"]
        assert "git_commit" in prov
        assert "timestamp" in prov
        assert "python_version" in prov
        assert "keras_version" in prov

    def test_git_commit_is_full_hash(self, tracker, tmp_path):
        path = tmp_path / "meta.json"
        tracker.add_hpo_results(num_trials=10)
        model = MagicMock()
        tracker.attach(model)
        tracker.save(path)

        with open(path) as f:
            data = json.load(f)
        commit = data["provenance"]["git_commit"]
        if commit == "unknown":
            pytest.skip("no git metadata available (not a git checkout)")
        assert len(commit) == 40  # full SHA-1 hash

    def test_numpy_serialization(self, tracker, tmp_path):
        path = tmp_path / "meta.json"
        tracker.add_phase2_metrics(
            epochs=int(np.int64(150)),
            final_sbc_ecdf=float(np.float64(0.058)),
        )
        model = MagicMock()
        tracker.attach(model)
        tracker.save(path)

        # Verify the JSON file has plain Python types
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data["training"]["epochs_completed"], int)
        assert isinstance(data["training"]["final_metrics"]["sbc_ecdf"], float)

    def test_saved_file_is_strictly_valid_json(self, tracker, tmp_path):
        """A sidecar carrying `-inf` must still parse outside Python.

        `json.dump` writes non-finite floats as bare `Infinity` / `NaN`
        literals. RFC 8259 has no token for them, so R's `jsonlite` refuses
        the file -- and a plain `json.load` here would **not** catch it,
        because Python accepts those literals on read. Hence the raising
        `parse_constant` hook: it is the only way this test can fail.

        Not hypothetical. `add_hpo_results(pareto_front=...)` stores trial
        records verbatim and those carry `log_gamma`, which can be `-inf`.
        """
        path = tmp_path / "meta.json"
        tracker.add_hpo_results(
            num_trials=2,
            pareto_front=[
                {"log_gamma": float("-inf"), "nrmse": 0.2},
                {"log_gamma": np.float32("-inf"), "nrmse": np.float64("nan")},
            ],
        )
        model = MagicMock()
        tracker.attach(model)
        tracker.save(path)

        def refuse(token):
            raise AssertionError(f"non-JSON literal in sidecar: {token}")

        with open(path) as f:
            data = json.load(f, parse_constant=refuse)

        front = data["hpo_results"]["pareto_front"]
        # `None`, not a dropped key: "underflowed" stays distinguishable
        # from "not computed", and reads back as NA in R.
        assert front[0]["log_gamma"] is None
        assert front[1]["log_gamma"] is None
        assert front[1]["nrmse"] is None
        assert front[0]["nrmse"] == pytest.approx(0.2)

    def test_float32_is_sanitized_like_float64(self, tracker, tmp_path):
        """`np.float32` is not a `float` subclass; `np.float64` is.

        A sanitizer written against `float` alone silently passes `float32`
        straight through to the writer. This pins both branches.
        """
        from bayesflow_irt.metadata import _sanitize_for_json

        for value in (
            np.float32("-inf"),
            np.float64("-inf"),
            float("-inf"),
            np.float32("nan"),
            float("nan"),
        ):
            assert _sanitize_for_json({"x": value})["x"] is None, value

        # Finite values are untouched, and still plain Python floats.
        for value in (np.float32(1.5), np.float64(1.5), 1.5):
            out = _sanitize_for_json({"x": value})["x"]
            assert out == pytest.approx(1.5)
            assert isinstance(out, float)

    def test_save_creates_parent_dirs(self, tracker, tmp_path):
        path = tmp_path / "deep" / "nested" / "meta.json"
        tracker.add_hpo_results(num_trials=10)
        model = MagicMock()
        tracker.attach(model)
        tracker.save(path)
        assert path.exists()

    def test_save_without_attach_raises(self, tracker, tmp_path):
        with pytest.raises(ValueError, match="attach"):
            tracker.save(tmp_path / "meta.json")

    def test_load_missing_model_info_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text('{"training_config": {}}')
        with pytest.raises(KeyError, match="model_info"):
            MetadataTracker.load(path)


class TestLoadMutable:
    def test_can_add_phases_after_load(self, tracker, tmp_path):
        path = tmp_path / "meta.json"
        tracker.add_phase2_metrics(epochs=100)
        model = MagicMock()
        tracker.attach(model)
        tracker.save(path)

        loaded = MetadataTracker.load(path)
        loaded.add_phase3_metrics(applied=True, epochs=30)
        assert loaded.to_dict()["phase3_metrics"]["applied"] is True
        assert loaded.to_dict()["phase3_metrics"]["epochs"] == 30


# ---------------------------------------------------------------------------
# verify_file
# ---------------------------------------------------------------------------


class TestVerifyFile:
    def test_matching_uuid(self, tracker, tmp_path):
        path = tmp_path / "meta.json"
        tracker.add_hpo_results(num_trials=10)
        model = MagicMock()
        uid = tracker.attach(model)
        tracker.save(path)

        model.metadata_uuid = uid
        assert MetadataTracker.verify_file(model, path) is True

    def test_mismatching_uuid(self, tracker, tmp_path):
        path = tmp_path / "meta.json"
        tracker.add_hpo_results(num_trials=10)
        model = MagicMock()
        tracker.attach(model)
        tracker.save(path)

        model.metadata_uuid = "wrong-uuid"
        assert MetadataTracker.verify_file(model, path) is False


# ---------------------------------------------------------------------------
# to_dict isolation
# ---------------------------------------------------------------------------


class TestToDict:
    def test_returns_deep_copy(self, tracker):
        tracker.add_hpo_results(num_trials=10)
        d = tracker.to_dict()
        d["hpo_results"]["num_trials"] = 999
        assert tracker.to_dict()["hpo_results"]["num_trials"] == 10


# ---------------------------------------------------------------------------
# Schema v2: checkpoint binding and version handling
# ---------------------------------------------------------------------------


class TestSchemaV2:
    def test_save_records_checkpoint_sha256(self, tracker, tmp_path):
        import hashlib

        ckpt = tmp_path / "m.keras"
        ckpt.write_bytes(b"weights" * 1000)
        tracker.add_hpo_results(num_trials=1)
        tracker.attach(MagicMock())
        tracker.save(tmp_path / "m_metadata.json", checkpoint_path=ckpt)

        prov = json.loads((tmp_path / "m_metadata.json").read_text())["provenance"]
        assert prov["checkpoint_filename"] == "m.keras"
        expected = hashlib.sha256(ckpt.read_bytes()).hexdigest()
        assert prov["checkpoint_sha256"] == expected

    def test_save_without_checkpoint_path_has_no_hash(self, tracker, tmp_path):
        tracker.add_hpo_results(num_trials=1)
        tracker.attach(MagicMock())
        tracker.save(tmp_path / "m.json")
        prov = json.loads((tmp_path / "m.json").read_text())["provenance"]
        assert "checkpoint_sha256" not in prov

    def test_save_writes_schema_version(self, tracker, tmp_path):
        tracker.add_hpo_results(num_trials=1)
        tracker.attach(MagicMock())
        tracker.save(tmp_path / "m.json")
        assert json.loads((tmp_path / "m.json").read_text())["schema_version"] == 2

    def test_newer_schema_version_is_rejected(self, tmp_path):
        path = tmp_path / "m.json"
        path.write_text('{"schema_version": 3, "model_spec": {}}')
        with pytest.raises(ValueError, match="schema_version"):
            MetadataTracker.load(path)

    def test_unknown_top_level_keys_survive_a_round_trip(self, tracker, tmp_path):
        tracker.add_hpo_results(num_trials=1)
        tracker.attach(MagicMock())
        tracker.save(tmp_path / "m.json")
        data = json.loads((tmp_path / "m.json").read_text())
        data["intended_use"] = {"note": "kept"}
        (tmp_path / "m.json").write_text(json.dumps(data))
        loaded = MetadataTracker.load(tmp_path / "m.json").to_dict()
        assert loaded["intended_use"] == {"note": "kept"}


# ---------------------------------------------------------------------------
# v1 -> v2 shim, golden tests over the fixture sidecars
# ---------------------------------------------------------------------------

#: Version-1 sidecars: one per family, a calibrated 2PL one and an HPO-only one.
_SIDECAR_DIR = Path(__file__).resolve().parent / "fixtures" / "sidecars"
_V1_SIDECARS = sorted(_SIDECAR_DIR.glob("*_metadata.json"))


def _v1(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_v1_upgrade(path):
    """A v1 sidecar loads as v2, keeps its facts and is never rewritten."""
    before = path.read_bytes()
    old = _v1(path)
    assert "schema_version" not in old

    new = MetadataTracker.load(path).to_dict()

    assert path.read_bytes() == before  # never rewritten
    assert new["schema_version"] == 2
    spec = new["model_spec"]
    assert {k: spec[k] for k in ("model", "variant", "version")} == old["model_info"]
    assert new.get("metadata_uuid") == old.get("metadata_uuid")
    assert new["provenance"]["git_commit"] == old["provenance"]["git_commit"]
    assert new["provenance"]["checkpoint_sha256"] is None

    config = old.get("training_config")
    if config is None:  # HPO-only file
        assert new["hpo_results"] == old["hpo_results"]
        assert spec["priors"] is None and spec["size_ranges"] is None
        return
    assert spec["priors"] == config["priors"]
    sizes = {k: v for k, v in config["size_ranges"].items() if k != "n_categories"}
    assert spec["size_ranges"] == sizes
    hp = config["hparams"]
    assert new["architecture"]["summary"] == hp.get("summary")
    assert new["architecture"]["flow"] == hp.get("flow")
    for key in ("optimizer", "base_lr", "device", "wall_clock_hours", "epochs"):
        assert new["training"][key] == hp.get(key)
    assert new["training"]["epochs_completed"] == old["phase2_metrics"]["epochs"]
    assert (
        new["training"]["per_epoch_metrics"]
        == old["phase2_metrics"]["per_epoch_metrics"]
    )


class TestUpgradeV1ToV2Golden:
    """Every fixture sidecar must load, and loading never rewrites it."""

    @pytest.mark.parametrize("path", _V1_SIDECARS, ids=lambda p: p.name)
    def test_loads_and_preserves_the_v1_facts(self, path):
        check_v1_upgrade(path)

    def test_fixtures_cover_every_kind(self):
        loaded = [_v1(p) for p in _V1_SIDECARS]
        models = {d["model_info"]["model"] for d in loaded}
        assert {"1PL", "2PL", "PCM", "GPCM"} <= models
        assert any("training_config" not in d for d in loaded)  # HPO-only

    def test_values_v1_never_held_are_explicit_none(self):
        d = MetadataTracker.load(_V1_SIDECARS[0]).to_dict()
        assert d["model_spec"]["size_sampling"] is None
        assert d["training"]["final_metrics"] is None
        for key in ("seed", "total_steps", "batches_per_epoch"):
            assert key in d["training"]

    def test_refinement_hparams_go_to_training_refinement(self):
        refined = [
            p
            for p in _V1_SIDECARS
            if "base_checkpoint"
            in ((_v1(p).get("training_config") or {}).get("hparams") or {})
        ]
        assert refined
        d = MetadataTracker.load(refined[0]).to_dict()
        assert "base_checkpoint" in d["training"]["refinement"]
        assert d["architecture"]["summary"] is None


class TestUpgradeV1ToV2Unit:
    def test_pure_function_does_not_mutate_input(self):
        from bayesflow_irt.metadata import upgrade_v1_to_v2

        old = {
            "model_info": {"model": "PCM", "variant": "fixed", "version": "0.1.0"},
            "training_config": {
                "size_ranges": {"n_items": [3, 5], "n_categories": 4},
                "hparams": {"summary": {"d": 1}},
            },
        }
        snapshot = json.loads(json.dumps(old))
        new = upgrade_v1_to_v2(old)
        assert old == snapshot
        assert new["model_spec"]["n_categories"] == 4
        assert new["model_spec"]["size_ranges"] == {"n_items": [3, 5]}

    def test_n_categories_read_from_hparams(self):
        from bayesflow_irt.metadata import upgrade_v1_to_v2

        old = {
            "model_info": {"model": "PCM", "variant": "fixed", "version": "0.1.0"},
            "training_config": {"hparams": {"n_categories": 5}},
        }
        assert upgrade_v1_to_v2(old)["model_spec"]["n_categories"] == 5

    def test_seed_and_parameterization_from_sibling_run_json(self, tmp_path):
        old = {
            "model_info": {"model": "2PL", "variant": "fixed", "version": "0.1.0"},
            "metadata_uuid": "u",
            "provenance": {"git_commit": "abc", "timestamp": "t"},
        }
        (tmp_path / "t1_metadata.json").write_text(json.dumps(old))
        run = {"config": {"seed": 5, "internal_parameterization": "loading"}}
        (tmp_path / "t1_run.json").write_text(json.dumps(run))
        d = MetadataTracker.load(tmp_path / "t1_metadata.json").to_dict()
        assert d["training"]["seed"] == 5
        assert d["model_spec"]["internal_parameterization"] == "loading"

    def test_missing_model_info_raises(self):
        from bayesflow_irt.metadata import upgrade_v1_to_v2

        with pytest.raises(KeyError, match="model_info"):
            upgrade_v1_to_v2({"training_config": {}})


class TestRefinementRecords:
    def test_refinement_merges_across_calls(self, tracker):
        tracker.add_training_config(hparams={"base_checkpoint": "base.keras"})
        tracker.add_training_config(hparams={"refine_epochs": 2})
        assert tracker.to_dict()["training"]["refinement"] == {
            "base_checkpoint": "base.keras",
            "refine_epochs": 2,
        }

    def test_phase2_call_replaces_its_own_keys(self, tracker):
        tracker.add_phase2_metrics(epochs=5, final_coverage=0.9, per_epoch_metrics=[{}])
        tracker.add_phase2_metrics(epochs=6)
        training = tracker.to_dict()["training"]
        assert training == {"epochs_completed": 6}

    def test_config_alone_is_not_phase_data(self, tracker, tmp_path):
        tracker.add_training_config(priors={"b": "N(0,1)"})
        tracker.attach(MagicMock())
        tracker.save(tmp_path / "m.json")
        assert any("required" in e for e in tracker.validate())

    def test_resave_keeps_the_checkpoint_binding(self, tracker, tmp_path):
        ckpt = tmp_path / "m.keras"
        ckpt.write_bytes(b"w")
        tracker.add_phase2_metrics(epochs=1)
        tracker.attach(MagicMock())
        tracker.save(tmp_path / "m.json", checkpoint_path=ckpt)
        loaded = MetadataTracker.load(tmp_path / "m.json")
        loaded.add_phase3_metrics(applied=True)
        loaded.save(tmp_path / "m.json")
        prov = json.loads((tmp_path / "m.json").read_text())["provenance"]
        assert prov["checkpoint_filename"] == "m.keras"
        assert prov["checkpoint_sha256"]


def test_add_intended_use_stores_notes_and_survives_round_trip(tracker, tmp_path):
    tracker.add_hpo_results(num_trials=1)
    tracker.add_intended_use(notes="Adult cognitive screening, 20 items.")
    tracker.attach(MagicMock())
    tracker.save(tmp_path / "m.json")
    loaded = MetadataTracker.load(tmp_path / "m.json").to_dict()
    assert loaded["intended_use"] == {"notes": "Adult cognitive screening, 20 items."}


class TestRefinementAfterV1Upgrade:
    def test_refinement_merges_after_v1_upgrade(self, tracker):
        tracker._data.setdefault("training", {})["refinement"] = None
        tracker.add_training_config(hparams={"base_checkpoint": "base.keras"})
        assert tracker.to_dict()["training"]["refinement"] == {
            "base_checkpoint": "base.keras"
        }

    def test_resave_keeps_extra_provenance_fields(self, tracker_with_hpo, tmp_path):
        tracker_with_hpo.attach(MagicMock())
        path = tmp_path / "m_metadata.json"
        tracker_with_hpo.save(path)
        loaded = MetadataTracker.load(path)
        loaded._data["provenance"]["hpo_source"] = {"trial": 7}
        loaded.save(path)
        prov = MetadataTracker.load(path).to_dict()["provenance"]
        assert prov["hpo_source"] == {"trial": 7}
        assert "timestamp" in prov
