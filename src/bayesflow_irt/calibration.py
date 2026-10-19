"""Optional calibrated IRT approximators.

This module composes `bayesflow-calibration-loss` with `bayesflow-irt`.
It remains import-safe when calibration extras are not installed.
"""

from __future__ import annotations

import keras
import keras.ops as ops

from .approximators import EquivariantIRTApproximator

try:
    from bayesflow_calibration_loss.mixin import CalibrationMixin
except ImportError as exc:  # pragma: no cover - import-guard path
    _CALIBRATION_LOSS_IMPORT_ERROR = exc

    class _CalibrationMixinPlaceholder:
        pass

    _CalibrationMixinBase = _CalibrationMixinPlaceholder
else:
    _CALIBRATION_LOSS_IMPORT_ERROR = None
    _CalibrationMixinBase = CalibrationMixin


@keras.saving.register_keras_serializable(package="bayesflow_irt")
class CalibratedIRTApproximator(_CalibrationMixinBase, EquivariantIRTApproximator):
    """Equivariant IRT approximator with Lagrangian calibration loss.

    Parameters
    ----------
    freeze_summary_network : bool, optional
        If ``True`` (default), calibration-loss refinement optimizes only the
        inference network. The summary network is marked non-trainable and its
        embeddings are evaluated in inference mode with stopped gradients during
        calibration tensor construction. Set to ``False`` to retain joint
        summary/inference training.
    flow_matching_sample_kwargs : dict or None, optional
        Default keyword arguments used when posterior sampling is requested from
        the calibrated approximator without explicit sampling options. This is
        useful for calibration-monitor callbacks which call ``model.sample`` and
        should use the same FlowMatching integration settings as diagnostics.
        Its nested ``flow_sample_kwargs`` entry (e.g.
        ``{"flow_sample_kwargs": {"method": "euler", "steps": 32}}``) is also
        forwarded as ``CalibrationMixin``'s ``sample_kwargs``, so the same fast
        integrator is used for the internal posterior sampling that computes
        the calibration loss on every training step -- without this, that call
        falls back to the inference network's default sampler (e.g.
        ``FlowMatching``'s adaptive-step ODE solve), which is far slower.
    report_calibration_metrics : bool, optional
        If ``False`` (default), keep training console/history concise while still
        applying calibrated loss. Set to ``True`` to log full calibration metrics
        (`calibration_error`, `calibration_loss`, `lambda`, `gamma`,
        `effective_gamma`).
    **kwargs
        Forwarded to `CalibrationMixin` and `EquivariantIRTApproximator`.
    """

    def __init__(
        self,
        *,
        freeze_summary_network: bool = True,
        flow_matching_sample_kwargs: dict | None = None,
        report_calibration_metrics: bool = False,
        **kwargs,
    ):
        if _CALIBRATION_LOSS_IMPORT_ERROR is not None:
            raise ImportError(
                "CalibratedIRTApproximator requires bayesflow-calibration-loss. "
                "Install with 'pip install bayesflow-irt[calibration]' "
                "or install bayesflow-calibration-loss directly."
            ) from _CALIBRATION_LOSS_IMPORT_ERROR

        kwargs.setdefault("report_calibration_metrics", report_calibration_metrics)
        flow_matching_sample_kwargs = dict(flow_matching_sample_kwargs or {})
        kwargs.setdefault(
            "sample_kwargs", flow_matching_sample_kwargs.get("flow_sample_kwargs", {})
        )
        super().__init__(**kwargs)
        if getattr(self, "item_posterior", "flow_matching") != "flow_matching":
            raise NotImplementedError(
                "CalibratedIRTApproximator supports only "
                "item_posterior='flow_matching'."
            )
        self.freeze_summary_network = bool(freeze_summary_network)
        self.flow_matching_sample_kwargs = flow_matching_sample_kwargs
        summary_network = getattr(self, "summary_network", None)
        if self.freeze_summary_network and summary_network is not None:
            summary_network.trainable = False

    def _get_calibration_tensors(
        self,
        inference_variables,
        inference_conditions=None,
        summary_variables=None,
        stage: str = "training",
    ):
        """Return flattened per-item tensors for calibration computation."""
        if summary_variables is None:
            raise ValueError(
                "CalibratedIRTApproximator requires summary_variables for "
                "per-item conditioning."
            )

        summary_outputs = self._compute_summary_outputs(summary_variables, stage=stage)
        batch_size = ops.shape(inference_variables)[0]
        num_items = ops.shape(inference_variables)[1]

        flat_params = ops.reshape(inference_variables, (batch_size * num_items, -1))
        flat_conditions = self._flatten_conditions(
            summary_outputs, inference_conditions, batch_size, num_items
        )
        return flat_params, flat_conditions

    def _compute_summary_outputs(self, summary_variables, *, stage: str):
        """Return fixed or trainable summary embeddings for refinement."""
        freeze_summary_network = getattr(self, "freeze_summary_network", True)
        summary_outputs = self.summary_network(
            summary_variables,
            training=(stage == "training" and not freeze_summary_network),
        )
        if freeze_summary_network:
            summary_outputs = ops.stop_gradient(summary_outputs)
        return summary_outputs

    def get_config(self):
        """Serialize calibrated refinement options."""
        config = super().get_config()
        config["freeze_summary_network"] = getattr(self, "freeze_summary_network", True)
        config["flow_matching_sample_kwargs"] = getattr(
            self, "flow_matching_sample_kwargs", {}
        )
        return config

    def sample(self, *args, **kwargs):
        """Sample posteriors, applying calibration defaults when omitted."""
        flow_matching_sample_kwargs = getattr(self, "flow_matching_sample_kwargs", {})
        if flow_matching_sample_kwargs:
            sample_kwargs = dict(flow_matching_sample_kwargs)
            sample_kwargs.update(kwargs)
            kwargs = sample_kwargs
        return super().sample(*args, **kwargs)


# Checkpoints written before this class was registered carry the bare class name
# as their ``registered_name``, which the Keras registry (keyed only by
# ``"bayesflow_irt>Name"``) would not resolve. Register the bare name as an alias
# so those checkpoints keep loading without an explicit ``custom_objects`` dict.
keras.saving.get_custom_objects()["CalibratedIRTApproximator"] = (
    CalibratedIRTApproximator
)
