"""Tests for optional calibrated IRT composition."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import bayesflow as bf
import keras
import numpy as np
import pytest
from conftest import strip_registered_name_prefixes

from bayesflow_irt import calibration as calibration_module
from bayesflow_irt.adapters import make_2pl_adapter
from bayesflow_irt.approximators import EquivariantIRTApproximator
from bayesflow_irt.calibration import CalibratedIRTApproximator
from bayesflow_irt.networks import MatrixEquivariantIRTSummary
from bayesflow_irt.priors import Prior
from bayesflow_irt.simulators import make_2pl_simulator


class _RecordingSummaryNetwork:
    def __init__(self):
        self.trainable = True
        self.training_calls = []

    def __call__(self, summary_variables, training):
        self.training_calls.append(training)
        batch_size = keras.ops.shape(summary_variables)[0]
        num_items = keras.ops.shape(summary_variables)[2]
        return keras.ops.zeros((batch_size, num_items, 5))


class TestCalibratedIRTApproximator:
    """Unit tests for optional composed calibrated IRT class."""

    def test_raises_helpful_error_when_dependency_missing(self):
        if calibration_module._CALIBRATION_LOSS_IMPORT_ERROR is None:
            pytest.skip("bayesflow-calibration-loss available in environment")

        with pytest.raises(ImportError, match=r"bayesflow-irt\[calibration\]"):
            CalibratedIRTApproximator()

    def test_defaults_to_low_clutter_metric_reporting(self):
        if calibration_module._CALIBRATION_LOSS_IMPORT_ERROR is not None:
            pytest.skip("bayesflow-calibration-loss not available")

        with patch.object(EquivariantIRTApproximator, "__init__", return_value=None):
            approx = CalibratedIRTApproximator()

        assert approx.report_calibration_metrics is False

    def test_flattens_itemwise_calibration_tensors(self):
        if calibration_module._CALIBRATION_LOSS_IMPORT_ERROR is not None:
            pytest.skip("bayesflow-calibration-loss not available")

        with patch.object(EquivariantIRTApproximator, "__init__", return_value=None):
            approx = CalibratedIRTApproximator(report_calibration_metrics=True)

        summary_outputs = keras.ops.convert_to_tensor(
            np.zeros((2, 3, 5), dtype=np.float32)
        )
        approx.summary_network = lambda _, training: summary_outputs

        inference_variables = keras.ops.convert_to_tensor(
            np.zeros((2, 3, 2), dtype=np.float32)
        )
        inference_conditions = keras.ops.convert_to_tensor(
            np.zeros((2, 4), dtype=np.float32)
        )
        summary_variables = keras.ops.convert_to_tensor(
            np.zeros((2, 7, 3), dtype=np.float32)
        )

        theta, conditions = approx._get_calibration_tensors(
            inference_variables=inference_variables,
            inference_conditions=inference_conditions,
            summary_variables=summary_variables,
            stage="training",
        )

        theta_shape = tuple(keras.ops.convert_to_numpy(keras.ops.shape(theta)))
        cond_shape = tuple(keras.ops.convert_to_numpy(keras.ops.shape(conditions)))

        assert theta_shape == (6, 2)
        assert cond_shape == (6, 9)

    def test_freezes_summary_embeddings_for_calibration_refinement(self):
        approx = object.__new__(CalibratedIRTApproximator)
        approx.freeze_summary_network = True
        approx.summary_network = _RecordingSummaryNetwork()

        inference_variables = keras.ops.convert_to_tensor(
            np.zeros((2, 3, 2), dtype=np.float32)
        )
        summary_variables = keras.ops.convert_to_tensor(
            np.zeros((2, 7, 3), dtype=np.float32)
        )

        _, conditions = approx._get_calibration_tensors(
            inference_variables=inference_variables,
            summary_variables=summary_variables,
            stage="training",
        )

        cond_shape = tuple(keras.ops.convert_to_numpy(keras.ops.shape(conditions)))
        assert cond_shape == (6, 5)
        assert approx.summary_network.training_calls == [False]

    def test_can_opt_into_joint_summary_refinement(self):
        approx = object.__new__(CalibratedIRTApproximator)
        approx.freeze_summary_network = False
        approx.summary_network = _RecordingSummaryNetwork()

        inference_variables = keras.ops.convert_to_tensor(
            np.zeros((2, 3, 2), dtype=np.float32)
        )
        summary_variables = keras.ops.convert_to_tensor(
            np.zeros((2, 7, 3), dtype=np.float32)
        )

        approx._get_calibration_tensors(
            inference_variables=inference_variables,
            summary_variables=summary_variables,
            stage="training",
        )

        assert approx.summary_network.training_calls == [True]

    def test_sample_uses_calibration_defaults_when_kwargs_omitted(self):
        approx = object.__new__(CalibratedIRTApproximator)
        approx.flow_matching_sample_kwargs = {
            "sample_chunk_size": 8,
            "flow_sample_kwargs": {"method": "euler", "steps": 32},
        }

        with patch.object(
            EquivariantIRTApproximator,
            "sample",
            return_value={"b": np.zeros((1, 2, 3))},
        ) as sample:
            out = approx.sample(
                num_samples=2,
                conditions={"responses": np.zeros((1, 4, 3))},
                sample_chunk_size=4,
            )

        assert "b" in out
        _, kwargs = sample.call_args
        assert kwargs["sample_chunk_size"] == 4
        assert kwargs["flow_sample_kwargs"] == {"method": "euler", "steps": 32}

    def test_from_config_accepts_and_merges_custom_objects(self):
        custom_summary = object()
        custom_log_transform = object()
        serialized_summary = {"class_name": "CustomSummary", "config": {}}
        config = {
            "adapter": MagicMock(),
            "inference_network": MagicMock(),
            "summary_network": serialized_summary,
        }

        with (
            patch.object(calibration_module, "_CALIBRATION_LOSS_IMPORT_ERROR", None),
            patch.object(EquivariantIRTApproximator, "__init__", return_value=None),
            patch.object(
                keras.saving,
                "deserialize_keras_object",
                return_value=custom_summary,
            ) as deserialize,
        ):
            restored = CalibratedIRTApproximator.from_config(
                config,
                custom_objects={
                    "CustomSummary": custom_summary,
                    "LogTransform": custom_log_transform,
                },
            )

        assert isinstance(restored, CalibratedIRTApproximator)
        deserialize.assert_called_once()
        assert (
            deserialize.call_args.kwargs["custom_objects"]["CustomSummary"]
            is custom_summary
        )
        assert (
            deserialize.call_args.kwargs["custom_objects"]["LogTransform"]
            is custom_log_transform
        )

    def test_save_load_round_trip(self, tmp_path):
        if calibration_module._CALIBRATION_LOSS_IMPORT_ERROR is not None:
            pytest.skip("bayesflow-calibration-loss not available")

        prior_a = Prior("lognormal", loc=0.0, scale=0.5)
        prior_b = Prior("normal", loc=0.0, scale=2.0)
        prior_theta = Prior("normal", loc=0.0, scale=1.0)
        simulator = make_2pl_simulator(
            n_items=(3, 3),
            n_persons=(5, 5),
            prior_a=prior_a,
            prior_b=prior_b,
            prior_theta=prior_theta,
        )
        adapter = make_2pl_adapter(simulator)
        summary = MatrixEquivariantIRTSummary(
            embed_dim=8,
            matrix_hidden_dim=8,
            num_matrix_blocks=1,
            num_outer_sabs=1,
            outer_embed_dim=8,
            num_outer_heads=2,
            outer_mlp_depth=1,
            outer_mlp_width=16,
            dropout=0.0,
        )
        flow = bf.networks.FlowMatching(
            subnet_kwargs={"widths": (16,), "time_embedding_dim": 8},
            integrate_kwargs={"method": "euler", "steps": 2},
        )
        approximator = CalibratedIRTApproximator(
            adapter=adapter,
            inference_network=flow,
            summary_network=summary,
            n_samples=4,
        )

        batch = adapter(simulator.sample((2,)))
        approximator.build({key: value.shape for key, value in batch.items()})
        model_path = tmp_path / "calibrated_2pl.keras"
        approximator.save(model_path)

        # No custom_objects: both classes are registered Keras serializables,
        # so importing the package is enough to locate them.
        restored = keras.saving.load_model(model_path)

        assert isinstance(restored, CalibratedIRTApproximator)
        assert restored.inference_network.base_distribution.dim == 2

        # Checkpoints predating registration name the class without the
        # "bayesflow_irt>" prefix; the bare-name aliases must keep those
        # loadable with no custom_objects either.
        legacy_path = tmp_path / "calibrated_2pl_legacy.keras"
        strip_registered_name_prefixes(model_path, legacy_path)
        legacy_restored = keras.saving.load_model(legacy_path)

        assert isinstance(legacy_restored, CalibratedIRTApproximator)
        assert legacy_restored.inference_network.base_distribution.dim == 2
