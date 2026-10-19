"""Tests for bayesflow_irt.envelope (the runtime guard and the data contract)."""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from bayesflow_irt import (
    DataContractError,
    MetadataTracker,
    check_data,
    data_contract,
)
from bayesflow_irt.approximators import EquivariantIRTApproximator


def _spec(model="2PL", n_categories=None, ranges="default"):
    if ranges == "default":
        ranges = {"n_items": [5, 50], "n_persons": [10, 500]}
    spec = {"model": model, "variant": "fixed", "version": "0.1.0"}
    if n_categories is not None:
        spec["n_categories"] = n_categories
    if ranges is not None:
        spec["size_ranges"] = ranges
    return {"schema_version": 2, "model_spec": spec}


def _responses(p=100, i=20, high=1, seed=0):
    return np.random.default_rng(seed).integers(0, high + 1, size=(p, i))


class TestSizes:
    def test_inside(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = check_data(_responses(), metadata=_spec())
        assert out.status == "inside"
        assert out.axes == {"n_persons": "inside", "n_items": "inside"}
        assert out.trained_ranges["n_items"] == [5, 50]

    def test_bounds_are_inclusive(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = check_data(n_persons=10, n_items=50, metadata=_spec())
            assert out.status == "inside"

    def test_outside_warns_and_names_the_range(self):
        with pytest.warns(UserWarning, match=r"n_persons=1000.*\[10, 500\]"):
            out = check_data(n_persons=1000, n_items=20, metadata=_spec())
        assert out.status == "outside"
        assert out.axes["n_items"] == "inside"

    def test_outside_raises_when_strict(self):
        with pytest.raises(ValueError, match="n_items=3"):
            check_data(n_persons=100, n_items=3, metadata=_spec(), strict=True)

    def test_batch_axis_is_ignored(self):
        batch = np.zeros((4, 100, 20), dtype=int)
        assert check_data(batch, metadata=_spec()).status == "inside"

    def test_missing_ranges_is_unknown_and_never_blocks(self):
        with pytest.warns(UserWarning, match="no trained range"):
            out = check_data(
                n_persons=10**6, n_items=1, metadata=_spec(ranges=None), strict=True
            )
        assert out.status == "unknown"

    def test_sizes_or_responses_required(self):
        with pytest.raises(TypeError):
            check_data(n_persons=10, metadata=_spec())


class TestDataContract:
    def test_k_mismatch_raises_even_when_not_strict(self):
        with pytest.raises(DataContractError, match="K=4"):
            check_data(
                n_persons=100,
                n_items=20,
                n_categories=3,
                metadata=_spec("PCM", 4),
                strict=False,
            )

    def test_polytomous_responses_above_k_raise(self):
        with pytest.raises(DataContractError, match=r"0\.\.3"):
            check_data(_responses(high=4), metadata=_spec("PCM", 4))

    def test_binary_model_rejects_category_two(self):
        with pytest.raises(DataContractError, match=r"0\.\.1"):
            check_data(_responses(high=2), metadata=_spec("2PL"))

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_non_finite_responses_raise(self, bad):
        x = _responses().astype(float)
        x[0, 0] = bad
        with pytest.raises(DataContractError, match="Missing responses"):
            check_data(x, metadata=_spec())

    def test_negative_sentinel_raises(self):
        x = _responses()
        x[3, 3] = -1
        with pytest.raises(DataContractError):
            check_data(x, metadata=_spec())

    def test_non_integer_raises(self):
        with pytest.raises(DataContractError, match="integer"):
            check_data(_responses().astype(float) + 0.5, metadata=_spec())

    def test_float_encoded_integers_are_accepted(self):
        out = check_data(_responses().astype(float), metadata=_spec())
        assert out.status == "inside"

    def test_wrong_rank_raises(self):
        with pytest.raises(DataContractError, match="shape"):
            check_data(np.zeros(10), metadata=_spec())

    @pytest.mark.parametrize(
        ("model", "k", "expected"),
        [
            ("1PL", None, "0/1"),
            ("2PL", None, "0/1"),
            ("PCM", 4, "0..3"),
            ("GPCM", 5, "0..4"),
            ("mystery", None, "Not recorded"),
        ],
    )
    def test_contract_text_follows_the_family(self, model, k, expected):
        contract = data_contract(_spec(model, k))
        assert expected in contract["response_coding"]
        assert set(contract) == {
            "missing_responses",
            "response_coding",
            "orientation",
            "score_scale",
        }
        missing = contract["missing_responses"]
        assert missing["supported"] is False
        assert missing["mechanisms"] == [] and missing["max_missing_rate"] is None
        assert "Not supported" in missing["note"]

    def test_missing_responses_record_is_read_from_model_spec(self):
        spec = _spec("2PL")
        spec["model_spec"]["missing_responses"] = {
            "supported": True,
            "coding": "nan",
            "mechanisms": ["mcar"],
            "max_missing_rate": 0.2,
            "future_key": 1,
        }
        missing = data_contract(spec)["missing_responses"]
        assert missing["mechanisms"] == ["mcar"]
        assert missing["max_missing_rate"] == 0.2
        assert missing["future_key"] == 1  # unknown keys carried through
        # check_data rejects missing values, so the contract never claims them.
        assert missing["supported"] is False
        assert "Not supported" in missing["note"]


class TestMetadataInputs:
    def test_tracker_and_path_and_v1_dict_agree(self, tmp_path):
        tracker = MetadataTracker(model="2PL", variant="fixed", version="0.1.0")
        tracker.add_training_config(
            size_ranges={"n_items": [5, 50], "n_persons": [10, 500]}
        )
        v1 = {
            "model_info": {"model": "2PL", "variant": "fixed", "version": "0.1.0"},
            "training_config": {
                "size_ranges": {"n_items": [5, 50], "n_persons": [10, 500]}
            },
        }
        path = tmp_path / "m_metadata.json"
        path.write_text(json.dumps(v1))
        for meta in (tracker, v1, path, str(path)):
            out = check_data(n_persons=100, n_items=20, metadata=meta)
            assert out.status == "inside"

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError, match="metadata"):
            check_data(n_persons=1, n_items=1, metadata=42)

    def test_real_polytomous_sidecar(self):
        root = Path(__file__).resolve().parent / "fixtures" / "sidecars"
        found = [p for p in root.rglob("*_metadata.json") if "gpcm" in p.name.lower()]
        if not found:
            pytest.skip("no GPCM sidecar present")
        spec = MetadataTracker.load(found[0]).to_dict()["model_spec"]
        assert spec["n_categories"]
        too_high = _responses(high=int(spec["n_categories"]))
        with pytest.raises(DataContractError):
            check_data(too_high, metadata=found[0])


class TestSampleWiring:
    """`sample(metadata=...)` runs the guard before touching the model."""

    def test_strict_outside_raises_before_sampling(self):
        conditions = {"responses": np.zeros((1, 1000, 20), dtype=int)}
        with pytest.raises(ValueError, match="n_persons=1000"):
            EquivariantIRTApproximator.sample(
                object(),
                conditions,
                num_samples=2,
                metadata=_spec(),
                strict=True,
            )

    def test_contract_violation_raises_before_sampling(self):
        x = np.zeros((1, 100, 20))
        x[0, 0, 0] = np.nan
        with pytest.raises(DataContractError):
            EquivariantIRTApproximator.sample(
                object(), {"responses": x}, num_samples=2, metadata=_spec()
            )


class TestCheckDataSizes:
    def test_sizes_that_disagree_with_the_matrix_raise(self):
        with pytest.raises(DataContractError, match="n_persons=100"):
            check_data(_responses(1000, 20), n_persons=100, metadata=_spec())

    def test_matching_explicit_sizes_are_accepted(self):
        out = check_data(_responses(100, 20), n_persons=100, metadata=_spec())
        assert out.status == "inside"

    def test_mixed_status_warning_names_the_unknown_axis(self):
        spec = _spec(ranges={"n_persons": [10, 500]})
        with pytest.warns(UserWarning) as rec:
            out = check_data(n_persons=1000, n_items=20, metadata=spec)
        assert out.status == "outside"
        text = str(rec[0].message)
        assert "n_persons=1000" in text and "n_items: no trained range" in text

    def test_masked_cells_raise(self):
        x = np.ma.masked_array(_responses(), mask=np.zeros((100, 20), dtype=bool))
        x[0, 0] = np.ma.masked
        with pytest.raises(DataContractError, match="masked"):
            check_data(x, metadata=_spec())

    def test_unmasked_masked_array_is_accepted(self):
        x = np.ma.masked_array(_responses())
        assert check_data(x, metadata=_spec()).status == "inside"

    @pytest.mark.parametrize("shape", [(0, 5), (100, 0), (0, 100, 20)])
    def test_empty_matrices_raise_the_contract_error(self, shape):
        with pytest.raises(DataContractError, match="zero-size"):
            check_data(np.zeros(shape, dtype=int), metadata=_spec())

    def test_dict_with_unsupported_schema_version_raises(self):
        spec = _spec()
        spec["schema_version"] = 999
        with pytest.raises(ValueError, match="schema_version"):
            check_data(n_persons=100, n_items=20, metadata=spec)

    def test_binary_model_rejects_other_category_counts(self):
        with pytest.raises(DataContractError, match="K=2"):
            check_data(n_persons=100, n_items=20, n_categories=5, metadata=_spec())

    @pytest.mark.parametrize("dtype", [bool, np.uint8, np.int64])
    def test_integer_dtypes_validate_without_float_copy(self, dtype):
        x = _responses().astype(dtype)
        assert check_data(x, metadata=_spec()).status == "inside"

    def test_score_scale_states_the_recorded_theta_prior(self):
        spec = _spec()
        spec["model_spec"]["priors"] = {
            "theta": {"name": "student_t", "loc": 0.0, "scale": 1.0, "df": 5}
        }
        text = data_contract(spec)["score_scale"]
        assert "student_t" in text and "N(0, 1)" not in text

    def test_score_scale_for_the_default_prior_and_for_none(self):
        spec = _spec()
        spec["model_spec"]["priors"] = {
            "theta": {"name": "normal", "loc": 0.0, "scale": 1.0}
        }
        assert "N(0, 1)" in data_contract(spec)["score_scale"]
        assert "not recorded" in data_contract(_spec())["score_scale"]

    @pytest.mark.parametrize(
        ("model", "needle"),
        [
            ("2PL", "prior is positive"),
            ("GPCM", "prior is positive"),
            ("1PL", "fixed at 1"),
            ("PCM", "fixed at 1"),
            ("mystery", "recoded"),
        ],
    )
    def test_orientation_reason_follows_the_family(self, model, needle):
        assert needle in data_contract(_spec(model, 3))["orientation"]
        if model == "mystery":
            assert "discrimination" not in data_contract(_spec(model))["orientation"]

    def test_warning_points_at_the_caller_with_stacklevel(self):
        with pytest.warns(UserWarning) as rec:
            check_data(n_persons=1000, n_items=20, metadata=_spec())
        assert rec[0].filename.endswith("test_envelope.py")

    def test_sample_warns_when_the_responses_key_is_missing(self):
        with pytest.warns(UserWarning, match="no 'responses'"):
            with pytest.raises(Exception):  # object() has no adapter afterwards
                EquivariantIRTApproximator.sample(
                    object(), {"x": 1}, num_samples=2, metadata=_spec()
                )


def test_add_intended_use_preserves_other_fields(tmp_path):
    from unittest.mock import MagicMock

    tracker = MetadataTracker(model="2PL", variant="fixed", version="0.1.0")
    tracker.add_hpo_results(num_trials=1)
    tracker.attach(MagicMock())
    tracker.save(tmp_path / "m.json")
    data = json.loads((tmp_path / "m.json").read_text())
    data["intended_use"] = {"population": "adults"}
    (tmp_path / "m.json").write_text(json.dumps(data))
    loaded = MetadataTracker.load(tmp_path / "m.json")
    loaded.add_intended_use(notes="n")
    assert loaded.to_dict()["intended_use"] == {"population": "adults", "notes": "n"}


class TestDataContractEdgeCases:
    def test_supplied_k_bounds_responses_when_sidecar_lacks_it(self):
        with pytest.raises(DataContractError, match=r"0\.\.3"):
            check_data(_responses(high=4), n_categories=4, metadata=_spec("PCM"))

    @pytest.mark.parametrize("sizes", [(0, 10), (100, 0), (-5, 10), (2.5, 10)])
    def test_nonpositive_or_fractional_sizes_raise(self, sizes):
        with pytest.raises(DataContractError, match="positive integer"):
            check_data(n_persons=sizes[0], n_items=sizes[1], metadata=_spec())

    def test_sizes_checked_without_ranges(self):
        with pytest.raises(DataContractError):
            check_data(n_persons=0, n_items=5, metadata=_spec(ranges=None))
