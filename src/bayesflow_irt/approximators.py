"""Custom approximator for equivariant (per-item) IRT inference.

``EquivariantIRTApproximator`` works with summary networks which output
``(batch, num_items, embed_dim)`` — one embedding per item. The approximator
merges ``batch × num_items`` before feeding the flow so that a **shared**
``FlowMatching`` network can operate on individual ``(a_i, b_i)`` pairs
conditioned on per-item embeddings.
"""

from __future__ import annotations

import logging
import warnings

import keras
import keras.ops as ops
import numpy as np
from bayesflow.adapters import Adapter
from bayesflow.approximators.continuous_approximator import ContinuousApproximator
from bayesflow.networks import CouplingFlow, InferenceNetwork

try:
    from bayesflow.networks.standardization import Standardization
except ModuleNotFoundError:
    from bayesflow.networks.helpers import Standardization

from .draws_summary import DrawsSummary

ITEM_POSTERIORS = ("flow_matching", "autoregressive_coupling_flow")

logger = logging.getLogger(__name__)


def _collect_seed_generators(root) -> list:
    """Return every Keras ``SeedGenerator`` reachable from ``root``.

    Covers generators tracked by ``root`` and its sublayers (``_seed_generators``,
    e.g. dropout) and generators held by plain helper objects one level below a
    layer attribute (e.g. a base distribution's ``seed_generator``).
    """
    found: list = []
    seen: set[int] = set()

    def _add(candidate) -> None:
        if isinstance(candidate, keras.random.SeedGenerator) and (
            id(candidate) not in seen
        ):
            seen.add(id(candidate))
            found.append(candidate)

    flatten = getattr(root, "_flatten_layers", None)
    layers = list(flatten(include_self=True, recursive=True)) if flatten else [root]
    for layer in layers:
        for generator in getattr(layer, "_seed_generators", ()):
            _add(generator)
        for value in vars(layer).values():
            _add(value)
            if not isinstance(value, keras.layers.Layer) and hasattr(value, "__dict__"):
                for inner in vars(value).values():
                    _add(inner)
    return found


def _restore_parameterization_flag(adapter) -> None:
    """Re-derive the ``loading`` flag on an adapter that lost it.

    The adapter factories tag the adapter with
    ``_bayesflow_irt_internal_parameterization``, but a plain attribute is not
    serialized, so a reloaded adapter has none.  The loading reparameterization
    is itself a serialized transform, so its presence identifies the
    parameterization; consumers such as :meth:`EquivariantIRTApproximator.sample`
    then derive the same ``d`` output as before saving.
    """
    from .adapters import _BaseLoadingReparamTransform

    if getattr(adapter, "_bayesflow_irt_internal_parameterization", None):
        return
    transforms = getattr(adapter, "transforms", None)
    if isinstance(transforms, (list, tuple)) and any(
        isinstance(t, _BaseLoadingReparamTransform) for t in transforms
    ):
        adapter._bayesflow_irt_internal_parameterization = "loading"


class _AccumulationEpochReset(keras.callbacks.Callback):
    """Discard a partial gradient accumulation window at each epoch end."""

    def __init__(self, approximator) -> None:
        super().__init__()
        self._approximator = approximator

    def on_epoch_end(self, epoch, logs=None) -> None:
        self._approximator._flush_partial_accumulation(epoch)


@keras.saving.register_keras_serializable(package="bayesflow_irt")
class EquivariantIRTApproximator(ContinuousApproximator):
    """Approximator for per-item IRT parameter estimation.

    Expects:

    * ``summary_variables``: ``(batch, num_persons, num_items)`` — response
      matrices.
    * ``inference_variables``: ``(batch, num_items, param_dim)`` — per-item
      parameters (e.g. ``param_dim=2`` for 2PL ``[a_i, b_i]``).

    The summary network must produce ``(batch, num_items, embed_dim)``
    per-item embeddings. This approximator merges the ``batch × num_items``
    dimensions so the inference network sees
    ``(batch*num_items, param_dim)`` targets with
    ``(batch*num_items, embed_dim)`` conditions.

    Parameters
    ----------
    adapter : bayesflow.adapters.Adapter
        Adapter for data processing.
    inference_network : InferenceNetwork
        A ``FlowMatching`` network operating on per-item parameters.
    summary_network : keras.Layer
        Per-item summary network with output shape ``(batch, num_items, D)``.
    standardize_params : bool
        If ``True``, apply online standardization to the flattened
        ``(batch*items, param_dim)`` inference variables before the
        flow.  Default ``False`` — prefer passing prior specs to the
        adapter factories instead, which gives deterministic static
        standardization with no warm-up.  Enable this as a fallback
        when the adapter does not standardize.
    flow_chunk_size : int or None
        Optional chunk size for computing inference-network loss over the
        flattened ``batch*num_items`` axis. ``None`` disables chunking.
    flow_chunk_backward : {None, "checkpoint"}
        Backward strategy for chunked loss computation. ``None`` disables
        special backward handling and ``"checkpoint"`` enables activation
        checkpointing for chunked training passes on the torch backend.
    gradient_accumulation_steps : int
        Number of micro-steps to accumulate before one optimizer update.
        Must be positive. ``standardize_params=True`` is incompatible with
        values greater than 1 because running-stat updates would be applied
        multiple times per logical optimizer step.
    item_posterior : {"flow_matching", "autoregressive_coupling_flow"}
        Factorization of the item posterior.

        * ``"flow_matching"`` (default): items are conditionally independent
          given their embeddings, ``prod_i q(item_i | h_i)``; one shared flow
          (typically ``FlowMatching``) over each item's parameters.
        * ``"autoregressive_coupling_flow"``: the autoregressive item factor
          ``prod_i q(item_i | item_<i, Y)`` of Habermann et al. (2026,
          Sec. 2.5, Eq. 3). The flow for item ``i`` is conditioned on
          ``[h_i, s_i, log(i + 1)]`` plus the global
          ``inference_conditions``, where ``s_i`` is the *draws summary*: a
          set summary of the earlier draws paired with their embeddings,
          ``(x_j, h_j), j < i``. This pairing simplifies the extended
          observations of Sec. 2.6 (which pair parameters with raw
          observations before the encoding), so the response representation
          does not depend on earlier draws; the log count is appended as in
          Sec. 2.6. The log item and person counts the paper also asks for
          arrive through the adapter's ``n_items``/``n_persons`` conditions,
          so the adapter must emit them (all built-in adapters do). Training
          is teacher-forced; sampling loops over items. Pass a
          ``CouplingFlow`` as ``inference_network``: sampling calls the flow
          once per item, and a coupling flow inverts in one pass where flow
          matching integrates an ODE. Draws are joint across items, so
          functionals of several items (test characteristic curves, sum
          scores) get the right spread.
    draws_summary_pooling : {"mean", "mean_std", "mean_max", "attention", "pma"}
        How the draws summary pools the earlier draws
        (``autoregressive_coupling_flow`` only): the DeepSet mean of
        Habermann et al. (2026), the mean plus a second aggregator (Corso et
        al., 2020), attention with the current item's embedding as query, or
        pooling by multihead attention with one learned seed (both Lee et
        al., 2019). See ``bayesflow_irt.draws_summary.DrawsSummary``.
    draws_summary_hidden_dim, draws_summary_dim : int
        Hidden and output widths of the draws summary.
    draws_summary_num_heads : int
        Attention heads for ``"attention"`` and ``"pma"``; must divide
        ``draws_summary_hidden_dim``.

    References
    ----------
    Habermann, D., Bulling, A., Radev, S. T., & Bürkner, P.-C. (2026).
    Amortized Bayesian inference on multilevel models of arbitrary structure.
    arXiv. https://doi.org/10.48550/arXiv.2609.40024
    """

    def __init__(
        self,
        *,
        adapter: Adapter,
        inference_network: InferenceNetwork,
        summary_network: keras.Layer,
        standardize_params: bool = False,
        flow_chunk_size: int | None = None,
        flow_chunk_backward: str | None = None,
        gradient_accumulation_steps: int = 1,
        item_posterior: str = "flow_matching",
        draws_summary_pooling: str = "mean",
        draws_summary_hidden_dim: int = 32,
        draws_summary_dim: int = 32,
        draws_summary_num_heads: int = 4,
        **kwargs,
    ):
        if item_posterior not in ITEM_POSTERIORS:
            raise ValueError(
                f"item_posterior must be one of {ITEM_POSTERIORS}, "
                f"got {item_posterior!r}."
            )
        if item_posterior == "autoregressive_coupling_flow" and not isinstance(
            inference_network, CouplingFlow
        ):
            raise TypeError(
                "item_posterior='autoregressive_coupling_flow' requires a "
                "CouplingFlow inference_network, got "
                f"{type(inference_network).__name__}."
            )
        if item_posterior != "flow_matching" and standardize_params:
            raise ValueError(
                "standardize_params=True requires item_posterior='flow_matching'; "
                "standardize item parameters in the adapter instead."
            )
        # Disable the parent's standardization (it can't handle 3-D shapes);
        # we optionally do our own in the flattened 2-D space inside compute_metrics.
        # Backward/forward compatibility:
        # Serialized models may carry a `standardize` key from the base class
        # config. We always control standardization here (via
        # `standardize_params` + flattened standardizer), so drop that key to
        # avoid passing duplicate keyword arguments to ContinuousApproximator.
        kwargs.pop("standardize", None)
        super().__init__(
            adapter=adapter,
            inference_network=inference_network,
            summary_network=summary_network,
            standardize=[],
            **kwargs,
        )
        _restore_parameterization_flag(adapter)
        if flow_chunk_size is not None and int(flow_chunk_size) <= 0:
            raise ValueError("flow_chunk_size must be a positive integer or None.")
        if flow_chunk_backward == "none":
            flow_chunk_backward = None
        if flow_chunk_backward not in {None, "checkpoint"}:
            raise ValueError(
                "flow_chunk_backward must be None or 'checkpoint'. "
                "'per_chunk' is reserved for a future release."
            )
        if int(gradient_accumulation_steps) <= 0:
            raise ValueError("gradient_accumulation_steps must be a positive integer.")
        if standardize_params and int(gradient_accumulation_steps) > 1:
            raise ValueError(
                "standardize_params=True cannot be combined with "
                "gradient_accumulation_steps > 1 due to running-stat updates."
            )
        self.standardize_params = standardize_params
        self.flow_chunk_size = (
            int(flow_chunk_size) if flow_chunk_size is not None else None
        )
        self.flow_chunk_backward = flow_chunk_backward
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        if standardize_params:
            self._flat_standardize = Standardization(trainable=False)
        self._global_condition_dim = 0
        self._accum_step = 0
        self.item_posterior = item_posterior
        self.draws_summary_pooling = draws_summary_pooling
        self.draws_summary_hidden_dim = int(draws_summary_hidden_dim)
        self.draws_summary_dim = int(draws_summary_dim)
        self.draws_summary_num_heads = int(draws_summary_num_heads)
        self.draws_summary = (
            DrawsSummary(
                self.draws_summary_hidden_dim,
                self.draws_summary_dim,
                pooling=self.draws_summary_pooling,
                num_heads=self.draws_summary_num_heads,
            )
            if self.autoregressive
            else None
        )

    @property
    def autoregressive(self) -> bool:
        return self.item_posterior == "autoregressive_coupling_flow"

    def _item_conditions(self, summary_outputs, items):
        """Per-item conditions ``(B, I, ·)`` before global conditions.

        For ``flow_matching`` these are the embeddings ``h``; for the
        autoregressive factor ``[h, s]`` with teacher-forced draws summaries
        ``s`` of the true ``items``.
        """
        if not self.autoregressive:
            return summary_outputs
        s = self.draws_summary(items, summary_outputs)
        return ops.concatenate([summary_outputs, s], axis=-1)

    @staticmethod
    def _static_dim(x, axis: int, name: str) -> int:
        dim = x.shape[axis]
        if dim is None:
            raise ValueError(f"{name} must have a static shape for chunked execution.")
        return int(dim)

    def _replayable_chunk_loss(self, chunk_loss, weight_chunk):
        """Bind one chunk's weights and make its recomputation deterministic.

        ``torch.utils.checkpoint`` re-runs the chunk function during backward.
        Its RNG preservation covers torch's generators only, whereas the flow's
        noise, time and dropout draws come from Keras ``SeedGenerator`` state,
        which advances on every call.  The seed state seen by the first
        (forward) call is therefore snapshotted and restored for any later
        call, and the state reached after the forward pass is put back
        afterwards so checkpointing consumes exactly as many draws as plain
        chunking.

        Parameters
        ----------
        chunk_loss : callable
            ``(params_chunk, cond_chunk, weight_chunk) -> loss``.
        weight_chunk : Tensor or None
            This chunk's sample weights; bound here rather than passed through
            ``checkpoint`` (which needs tensor arguments) and rather than
            captured from the caller's loop variable.
        """
        generators = _collect_seed_generators(self.inference_network)
        snapshot: list[np.ndarray] | None = None

        def _read():
            return [ops.convert_to_numpy(g.state).copy() for g in generators]

        def _write(values):
            for g, v in zip(generators, values):
                g.state.assign(v)

        def run(params_chunk, cond_chunk):
            nonlocal snapshot
            if snapshot is None:
                snapshot = _read()
                return chunk_loss(params_chunk, cond_chunk, weight_chunk)
            after_forward = _read()
            _write(snapshot)
            try:
                return chunk_loss(params_chunk, cond_chunk, weight_chunk)
            finally:
                _write(after_forward)

        return run

    def _chunked_inference_loss(
        self,
        flat_params,
        flat_conditions,
        *,
        sample_weight=None,
        stage: str,
    ):
        """Compute flow loss with optional chunking over the flattened item axis.

        Notes
        -----
        BayesFlow reduces the per-row loss as ``mean(loss * sample_weight)``
        over rows (``bayesflow.utils.weighted_mean``), so each chunk's loss is
        already a row mean.  Chunks are therefore combined by row count,
        weighted or not, to reproduce the unchunked value exactly.
        """
        flat_n = self._static_dim(flat_params, 0, "inference_variables")
        if self.flow_chunk_size is None or self.flow_chunk_size >= flat_n:
            return self.inference_network.compute_metrics(
                flat_params,
                conditions=flat_conditions,
                sample_weight=sample_weight,
                stage=stage,
            )["loss"]

        if self.flow_chunk_backward is not None and keras.backend.backend() != "torch":
            raise NotImplementedError(
                "flow_chunk_backward is currently supported only on the torch backend."
            )

        chunk = self.flow_chunk_size
        weighted = []

        def _chunk_loss(params_chunk, cond_chunk, weight_chunk):
            return self.inference_network.compute_metrics(
                params_chunk,
                conditions=cond_chunk,
                sample_weight=weight_chunk,
                stage=stage,
            )["loss"]

        for start in range(0, flat_n, chunk):
            end = min(start + chunk, flat_n)
            params_chunk = flat_params[start:end]
            cond_chunk = flat_conditions[start:end]
            weight_chunk = None if sample_weight is None else sample_weight[start:end]

            if self.flow_chunk_backward == "checkpoint" and stage == "training":
                import torch.utils.checkpoint as cp

                # Keep the autograd graph connection: requires_grad_() in-place
                # does NOT detach. Using detach() here would sever the gradient
                # path to the summary network, causing it to receive zero
                # gradient while the flow still trains (silent correctness bug).
                if not getattr(cond_chunk, "requires_grad", False):
                    cond_chunk = cond_chunk.requires_grad_()
                loss_chunk = cp.checkpoint(
                    self._replayable_chunk_loss(_chunk_loss, weight_chunk),
                    params_chunk,
                    cond_chunk,
                    use_reentrant=False,
                )
            else:
                loss_chunk = self.inference_network.compute_metrics(
                    params_chunk,
                    conditions=cond_chunk,
                    sample_weight=weight_chunk,
                    stage=stage,
                )["loss"]

            n_chunk = end - start
            weighted.append(loss_chunk * n_chunk)

        return ops.sum(ops.stack(weighted, axis=0), axis=0) / float(flat_n)

    # ------------------------------------------------------------------
    # Shared helper: flatten per-item embeddings + optional global conds
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_conditions(
        summary_outputs, inference_conditions, batch_size, num_items
    ):
        """Merge per-item embeddings with optional global conditions, then flatten.

        Parameters
        ----------
        summary_outputs : Tensor
            ``(batch, num_items, embed_dim)`` from the summary network.
        inference_conditions : Tensor or None
            ``(batch, cond_dim)`` global conditions, or ``None``.
        batch_size, num_items : int or symbolic
            Batch and item dimensions.

        Returns
        -------
        Tensor
            ``(batch * num_items, total_cond_dim)`` flattened conditions.
        """
        flat_n = batch_size * num_items
        flat_conditions = ops.reshape(summary_outputs, (flat_n, -1))
        if inference_conditions is not None:
            expanded = ops.expand_dims(inference_conditions, axis=1)
            expanded = ops.broadcast_to(
                expanded,
                (batch_size, num_items, ops.shape(inference_conditions)[-1]),
            )
            flat_global = ops.reshape(expanded, (flat_n, -1))
            flat_conditions = ops.concatenate([flat_conditions, flat_global], axis=-1)
        return flat_conditions

    # ------------------------------------------------------------------
    # Override build to handle 3-D shapes
    # ------------------------------------------------------------------

    def build(self, data_shapes: dict[str, tuple[int] | dict[str, dict]]) -> None:
        """Build all sub-networks for 3-D ``(batch, I, param_dim)`` inputs.

        Computes the flattened condition dimension
        (``embed_dim + cond_dim``) from the summary network's output shape
        and builds the inference network with placeholder 2-D shapes.
        """
        data_shapes = self._normalize_build_shapes(data_shapes)
        self._build_data_shapes = data_shapes

        # Build summary network
        if not self.summary_network.built:
            self.summary_network.build(data_shapes["summary_variables"])

        # For the inference network: it sees (*, param_dim) with
        # conditions (*, embed_dim).
        # We just need the last dimensions
        iv_shape = data_shapes["inference_variables"]  # (batch, I, param_dim)
        param_dim = iv_shape[-1]

        # Build flat standardization layer for (*, param_dim) data
        if self.standardize_params and not self._flat_standardize.built:
            self._flat_standardize.build((2, param_dim))

        # Compute summary output embed_dim by doing a dummy forward
        summary_out_shape = self.summary_network.compute_output_shape(
            data_shapes["summary_variables"]
        )
        embed_dim = summary_out_shape[-1]

        # Combine with any inference_conditions
        cond_dim = embed_dim
        if self.autoregressive:
            if not self.draws_summary.built:
                self.draws_summary.build(
                    (None, None, param_dim), (None, None, embed_dim)
                )
            cond_dim += self.draws_summary.output_dim
        if "inference_conditions" in data_shapes:
            cond_dim += data_shapes["inference_conditions"][-1]
            self._global_condition_dim = int(data_shapes["inference_conditions"][-1])
        else:
            self._global_condition_dim = 0

        # Build inference network with flattened rank-2 placeholder shapes.
        # BayesFlow coupling subnets expect at least 2-D tensors during build;
        # passing rank-1 feature-only shapes can fail in residual projectors.
        if not self.inference_network.built:
            self.inference_network.build(
                (2, param_dim),
                (2, cond_dim),
            )

        self.built = True

    @staticmethod
    def _normalize_build_shapes(data_shapes):
        """Return build shapes as tuples after Keras JSON deserialization."""
        if not isinstance(data_shapes, dict):
            raise TypeError("data_shapes must be a dictionary of input shapes.")
        return {
            key: tuple(value) if isinstance(value, (list, tuple)) else value
            for key, value in data_shapes.items()
        }

    def get_build_config(self):
        """Serialize custom 3-D input shapes for Keras model loading."""
        data_shapes = getattr(self, "_build_data_shapes", None)
        if data_shapes is None:
            return {}
        return {"input_shape": data_shapes}

    def build_from_config(self, config):
        """Rebuild variables from Keras' saved ``input_shape`` dictionary."""
        data_shapes = config.get("input_shape") or config.get("data_shapes")
        if data_shapes is None:
            return
        self.build(data_shapes)

    # ------------------------------------------------------------------
    # Override call() to handle dict inputs from Keras's symbolic build
    # ------------------------------------------------------------------

    def call(self, data, training: bool = False, **kwargs):
        """Keras-compatible forward pass.

        During normal training Keras calls ``train_step`` → ``compute_metrics``
        directly. However Keras's ``_symbolic_build`` (run once before the
        first optimizer step to create Adam slots) uses ``compute_output_spec``
        which invokes ``call(batch_dict, training=False)``.  Without this
        override the dict lands entirely in the ``inference_variables``
        positional argument, leaving ``summary_variables=None`` and crashing.
        """
        if isinstance(data, dict):
            return self.compute_metrics(
                inference_variables=data.get("inference_variables"),
                inference_conditions=data.get("inference_conditions"),
                summary_variables=data.get("summary_variables"),
                stage="training" if training else "validation",
            )
        return super().call(data, training=training, **kwargs)

    # ------------------------------------------------------------------
    # Override compute_metrics for the per-item merge logic
    # ------------------------------------------------------------------

    def compute_metrics(
        self,
        inference_variables,
        inference_conditions=None,
        summary_variables=None,
        sample_weight=None,
        stage: str = "training",
    ) -> dict[str, keras.KerasTensor]:
        """Compute loss with batch × item merging.

        Parameters
        ----------
        inference_variables : Tensor
            Shape ``(batch, num_items, param_dim)``.
        summary_variables : Tensor
            Shape ``(batch, num_persons, num_items)``.
        inference_conditions : Tensor or None
            Optional global conditions ``(batch, cond_dim)``.
        sample_weight : Tensor or None
            Optional sample weights.
        stage : str
            ``"training"`` or ``"validation"``.

        Returns
        -------
        dict with ``"loss"`` key.
        """
        # Get per-item embeddings from summary network
        summary_outputs = self._compute_summary_outputs(summary_variables, stage=stage)
        # summary_outputs: (batch, num_items, embed_dim)

        batch_size = ops.shape(inference_variables)[0]
        num_items = ops.shape(inference_variables)[1]

        # Flatten batch × items dimensions
        flat_params = ops.reshape(inference_variables, (batch_size * num_items, -1))
        flat_conditions = self._flatten_conditions(
            self._item_conditions(summary_outputs, inference_variables),
            inference_conditions,
            batch_size,
            num_items,
        )

        # Online standardization in the flattened 2-D space
        if self.standardize_params:
            flat_params = self._flat_standardize(flat_params, stage=stage)

        if sample_weight is not None:
            # Per-dataset weights (batch,) or per-item weights (batch, items)
            # are broadcast to the flattened row order of ``flat_params``.
            sample_weight = ops.reshape(sample_weight, (batch_size, -1))
            sample_weight = ops.broadcast_to(sample_weight, (batch_size, num_items))
            sample_weight = ops.reshape(sample_weight, (-1,))

        loss = self._chunked_inference_loss(
            flat_params,
            flat_conditions,
            sample_weight=sample_weight,
            stage=stage,
        )

        return {"loss": loss}

    def _compute_summary_outputs(self, summary_variables, *, stage: str):
        """Return summary-network embeddings for metric computation."""
        return self.summary_network(summary_variables, training=(stage == "training"))

    def train_step(self, data):
        """Custom train step for gradient accumulation (torch backend).

        Uses PyTorch's native gradient accumulation: ``loss.backward()`` is
        called on each micro-step *without* zeroing gradients in between.
        After ``gradient_accumulation_steps`` micro-steps the accumulated
        ``param.grad`` tensors are passed to the optimizer and then cleared
        via ``zero_grad()``.  This mirrors
        :meth:`ContinuousApproximator.train_step` which also uses
        ``torch.enable_grad()`` / ``loss.backward()`` rather than
        ``keras.GradientTape`` (which is not available on the torch backend).
        """
        if self.gradient_accumulation_steps == 1:
            return super().train_step(data)

        backend = keras.backend.backend()
        if backend != "torch":
            raise NotImplementedError(
                f"Gradient accumulation in {self.__class__.__name__}.train_step() "
                "is only implemented for the torch Keras backend, but the active "
                f"backend is {backend!r}. Set gradient_accumulation_steps=1 or "
                "switch to the torch backend."
            )

        import torch

        ga = self.gradient_accumulation_steps

        # BayesFlow datasets yield plain dicts; handle both dict and
        # (x, y, w) tuple inputs for compatibility.  Extract sample_weight
        # when present so per-sample weighting is preserved across micro-steps.
        if isinstance(data, dict):
            x = data
            sample_weight = None
        else:
            x = data[0]
            sample_weight = data[2] if len(data) > 2 else None

        try:
            with torch.enable_grad():
                metrics = self.compute_metrics(
                    inference_variables=x.get("inference_variables"),
                    inference_conditions=x.get("inference_conditions"),
                    summary_variables=x.get("summary_variables"),
                    sample_weight=sample_weight,
                    stage="training",
                )
                loss = metrics["loss"] / float(ga)
                if not bool(np.all(ops.convert_to_numpy(ops.isfinite(loss)))):
                    raise RuntimeError(
                        "Encountered non-finite loss in gradient accumulation "
                        f"micro-step {self._accum_step + 1}/{ga}."
                    )
            loss.backward()
        except Exception:
            # Clear any partial .grad accumulations so the next call starts
            # clean.  Without this, a retry (e.g. via fit_with_oom_retry)
            # would accumulate fresh gradients on top of stale partials from
            # the failed step, silently corrupting the optimisation trajectory.
            self.zero_grad()
            self._accum_step = 0
            raise

        self._accum_step += 1

        if self._accum_step >= ga:
            trainable_weights = self.trainable_weights[:]
            gradients = [v.value.grad for v in trainable_weights]

            if all(g is None for g in gradients):
                self.zero_grad()
                self._accum_step = 0
                raise RuntimeError(
                    "All gradients are None at accumulation step "
                    f"{ga}/{ga}. The loss appears disconnected from trainable "
                    "parameters."
                )
            else:
                with torch.no_grad():
                    self.optimizer.apply(gradients, trainable_weights)

            self.zero_grad()
            self._accum_step = 0

        self._update_metrics(metrics, self._batch_size_from_data(data))
        return {k: v for k, v in metrics.items()}

    def fit(self, *args, **kwargs):
        """Train the approximator; see :meth:`ContinuousApproximator.fit`.

        Keras never calls a model's own ``on_epoch_end``; only callbacks
        receive epoch events.  When gradient accumulation is active, a callback
        that discards a partial accumulation window at each epoch end is
        therefore added to ``callbacks``.
        """
        if self.gradient_accumulation_steps > 1:
            callbacks = kwargs.get("callbacks")
            reset = _AccumulationEpochReset(self)
            if callbacks is None:
                callbacks = [reset]
            elif isinstance(callbacks, keras.callbacks.CallbackList):
                callbacks.callbacks.insert(0, reset)
            else:
                callbacks = [reset, *callbacks]
            kwargs["callbacks"] = callbacks
            self._discard_accumulation()
            try:
                return super().fit(*args, **kwargs)
            finally:
                self._discard_accumulation()
        return super().fit(*args, **kwargs)

    def _discard_accumulation(self) -> None:
        """Drop accumulated gradients and the micro-step counter."""
        self.zero_grad()
        self._accum_step = 0

    def _flush_partial_accumulation(self, epoch: int) -> None:
        """Discard a partial gradient accumulation window at an epoch boundary.

        If ``num_batches % gradient_accumulation_steps != 0``, the last
        micro-steps of an epoch would silently carry over into the next epoch,
        mixing gradients from two epochs in a single optimizer update.  Any such
        partial accumulation is discarded and a warning is logged so users can
        adjust ``num_batches`` to be divisible by
        ``gradient_accumulation_steps``.
        """
        if self.gradient_accumulation_steps > 1 and self._accum_step > 0:
            logger.warning(
                "Epoch %d ended with %d accumulated micro-steps that did not "
                "complete a full accumulation window "
                "(gradient_accumulation_steps=%d). "
                "Discarding partial gradients to prevent cross-epoch "
                "gradient contamination. Consider setting num_batches to a "
                "multiple of gradient_accumulation_steps.",
                epoch,
                self._accum_step,
                self.gradient_accumulation_steps,
            )
            self.zero_grad()
            self._accum_step = 0

    # ------------------------------------------------------------------
    # Override _sample for posterior sampling
    # ------------------------------------------------------------------

    def _sample(
        self,
        num_samples: int,
        inference_conditions=None,
        summary_variables=None,
        **kwargs,
    ):
        """Sample from the per-item posterior.

        Chunks over both the condition axis (``batch * num_items``) and
        the sample axis to keep peak GPU memory bounded.  Chunk sizes
        are configurable via ``kwargs``:

        Parameters
        ----------
        num_samples : int
            Number of posterior draws per item.
        inference_conditions : Tensor or None
            Optional global conditions ``(batch, cond_dim)``.  If ``None``
            and the flow was built with global conditions, zero-valued
            placeholders are injected automatically.
        summary_variables : Tensor
            Response matrix ``(batch, P, I)`` or ``(batch, P, I, K)``.
        sample_chunk_size : int
            Max posterior draws per inner loop iteration (default 64).
            Unused for ``autoregressive_coupling_flow``.
        condition_chunk_size : int
            Max item-conditions per outer loop iteration (default 256).
            For ``autoregressive_coupling_flow`` it counts ``(dataset, draw)``
            rows instead, each of which makes ``num_items`` sequential flow
            calls.
        flow_sample_kwargs : dict
            Extra keyword arguments forwarded to
            ``inference_network.sample(...)``. Use this for FlowMatching
            solver overrides such as ``{"method": "euler", "steps": 24}``.

        Returns
        -------
        Tensor
            Shape ``(batch, num_samples, num_items, param_dim)`` — kept as
            4-D so the adapter inverse correctly splits the last axis back
            into the original per-item variables (e.g. ``a`` and ``b``).
        """
        summary_batch_size = kwargs.pop("summary_batch_size", None)
        if summary_batch_size is not None:
            summary_batch_size = int(summary_batch_size)
            if summary_batch_size <= 0:
                raise ValueError("summary_batch_size must be a positive integer.")

        if summary_batch_size is None:
            summary_outputs = self.summary_network(summary_variables)
        else:
            total_batch = self._static_dim(summary_variables, 0, "summary_variables")
            outs = []
            for start in range(0, total_batch, summary_batch_size):
                end = min(start + summary_batch_size, total_batch)
                outs.append(self.summary_network(summary_variables[start:end]))
            summary_outputs = ops.concatenate(outs, axis=0)
        # summary_outputs: (batch, num_items, embed_dim)

        batch_size = self._static_dim(summary_outputs, 0, "summary_outputs")
        num_items = self._static_dim(summary_outputs, 1, "summary_outputs")
        flat_n = batch_size * num_items

        # Some sampling paths can provide only summary variables. If the flow
        # was built with global conditions, inject zero-valued placeholders to
        # keep the expected condition dimensionality consistent.
        if inference_conditions is None and self._global_condition_dim > 0:
            inference_conditions = ops.zeros(
                (batch_size, self._global_condition_dim),
                dtype=keras.ops.dtype(summary_outputs),
            )

        flat_conditions = self._flatten_conditions(
            summary_outputs,
            inference_conditions,
            batch_size,
            num_items,
        )

        # Sampling all trajectories at once can be memory-intensive for large
        # validation batches. Chunk over both item-conditions and sample axis
        # to keep peak memory bounded while preserving output semantics.
        sample_chunk_size = int(kwargs.pop("sample_chunk_size", 64))
        condition_chunk_size = int(kwargs.pop("condition_chunk_size", 256))
        flow_sample_kwargs = kwargs.pop("flow_sample_kwargs", None)
        if flow_sample_kwargs is None:
            flow_sample_kwargs = {}
        elif not isinstance(flow_sample_kwargs, dict):
            raise TypeError("flow_sample_kwargs must be a dict or None.")
        if sample_chunk_size <= 0:
            raise ValueError("sample_chunk_size must be a positive integer.")
        if condition_chunk_size <= 0:
            raise ValueError("condition_chunk_size must be a positive integer.")

        if self.autoregressive:
            return self._sample_autoregressive(
                num_samples,
                summary_outputs,
                inference_conditions,
                row_chunk_size=condition_chunk_size,
                flow_sample_kwargs=flow_sample_kwargs,
            )

        condition_chunks = []
        for cond_start in range(0, flat_n, condition_chunk_size):
            n_cond = min(condition_chunk_size, flat_n - cond_start)
            flat_conditions_cond = flat_conditions[cond_start : cond_start + n_cond]

            sample_chunks = []
            for sample_start in range(0, num_samples, sample_chunk_size):
                n_sample = min(sample_chunk_size, num_samples - sample_start)

                flat_conditions_chunk = ops.expand_dims(flat_conditions_cond, axis=1)
                cond_dim = self._static_dim(flat_conditions_cond, -1, "flat_conditions")
                flat_conditions_chunk = ops.broadcast_to(
                    flat_conditions_chunk,
                    (n_cond, n_sample, cond_dim),
                )
                chunk = self.inference_network.sample(
                    (n_cond, n_sample),
                    conditions=flat_conditions_chunk,
                    **flow_sample_kwargs,
                )
                sample_chunks.append(chunk)

            condition_chunks.append(ops.concatenate(sample_chunks, axis=1))

        samples = ops.concatenate(condition_chunks, axis=0)
        param_dim = self._static_dim(samples, -1, "samples")

        # Inverse standardization: map from N(0,1) space back to original scale
        if self.standardize_params:
            flat_samples = ops.reshape(samples, (-1, param_dim))
            flat_samples = self._flat_standardize(
                flat_samples, stage="inference", forward=False
            )
            samples = ops.reshape(flat_samples, (flat_n, num_samples, param_dim))

        # Reshape: (batch, num_items, num_samples, param_dim)
        samples = ops.reshape(samples, (batch_size, num_items, num_samples, param_dim))

        # Transpose to (batch, num_samples, num_items, param_dim)
        samples = ops.transpose(samples, axes=(0, 2, 1, 3))

        # Keep 4-D shape so the adapter inverse correctly splits the last
        # axis (param_dim=2) back into the original per-item variables.
        return samples

    def _sample_autoregressive(
        self,
        num_samples,
        summary_outputs,
        inference_conditions,
        *,
        row_chunk_size,
        flow_sample_kwargs,
    ):
        """Draw items one at a time, each conditioned on the earlier draws.

        Rows are ``(dataset, draw)`` pairs, processed ``row_chunk_size`` at a
        time. Returns ``(batch, num_samples, num_items, param_dim)``.
        """
        batch_size = self._static_dim(summary_outputs, 0, "summary_outputs")
        num_items = self._static_dim(summary_outputs, 1, "summary_outputs")
        total = batch_size * num_samples
        # An integer seed would restart the same stream in every per-item
        # call (bayesflow versions that honour it), so draw all items and
        # chunks from one generator instead.
        seed = flow_sample_kwargs.get("seed")
        if isinstance(seed, (int, np.integer)) and not isinstance(seed, bool):
            flow_sample_kwargs = {
                **flow_sample_kwargs,
                "seed": keras.random.SeedGenerator(int(seed)),
            }
        out = []
        for start in range(0, total, row_chunk_size):
            rows = np.arange(start, min(start + row_chunk_size, total))
            n = len(rows)
            b_idx = ops.convert_to_tensor(rows // num_samples, dtype="int32")
            h = ops.take(summary_outputs, b_idx, axis=0)  # (n, I, D)
            fixed = (
                []
                if inference_conditions is None
                else [ops.take(inference_conditions, b_idx, axis=0)]
            )
            summary = self.draws_summary
            # "mean" keeps a running sum (O(I)); the other poolings need the
            # earlier phi values themselves, kept stacked.
            run_sum = ops.zeros((n, summary.hidden_dim), dtype=h.dtype)
            prev = []
            drawn = []
            for i in range(num_items):
                h_i = h[:, i]
                if summary.pooling == "mean":
                    count = ops.full((n, 1), float(i), dtype=h.dtype)
                    s = summary.from_prefix_sum(run_sum, count)
                else:
                    stacked = ops.stack(prev, axis=1) if prev else None
                    s = summary.from_prefix(stacked, h_i, i)
                cond = ops.concatenate([h_i, s, *fixed], axis=-1)
                x_i = self.inference_network.sample(
                    (n, 1), conditions=ops.expand_dims(cond, 1), **flow_sample_kwargs
                )[:, 0, :]
                phi = summary.phi_values(x_i, h_i)
                if summary.pooling == "mean":
                    run_sum = run_sum + phi
                else:
                    prev.append(phi)
                drawn.append(x_i)
            out.append(ops.stack(drawn, axis=1))  # (n, I, P)
        samples = ops.concatenate(out, axis=0)
        return ops.reshape(
            samples, (batch_size, num_samples, num_items, samples.shape[-1])
        )

    def sample(
        self,
        *args,
        num_samples: int | None = None,
        n_samples: int | None = None,
        conditions=None,
        split: bool = False,
        batch_size: int | None = None,
        sample_shape: str | tuple[int] | int = "infer",
        metadata=None,
        strict: bool = False,
        **kwargs,
    ):
        """Sample posteriors with per-item flattening and optional loading key.

        If ``metadata`` (the model's sidecar: a path, dict or
        :class:`~bayesflow_irt.metadata.MetadataTracker`) is given, the
        ``responses`` in ``conditions`` are first checked against the trained
        envelope with :func:`bayesflow_irt.envelope.check_data`: a size
        outside the trained range warns (``strict=True`` raises), and data the
        model cannot run on raises
        :class:`~bayesflow_irt.envelope.DataContractError`.  A loaded
        checkpoint carries no metadata, so it is never looked up implicitly.

        Bypasses the parent's ``sample()`` because the standard
        ``_prepare_conditions`` pipeline cannot handle the 3-D summary
        outputs ``(batch, num_items, embed_dim)`` produced by
        :class:`MatrixEquivariantIRTSummary`.  Instead, this method adapts
        the input, calls :meth:`_sample` (which flattens
        ``batch × num_items`` internally), then inverse-adapts the result.

        If the adapter uses the loading parameterization, the derived key
        ``d = -a*b`` is added to the output.

        Extra ``**kwargs`` (e.g., ``sample_chunk_size``,
        ``condition_chunk_size``, ``flow_sample_kwargs``) are forwarded to
        ``_sample()``.
        """
        # Compatibility shim:
        # - allow `conditions` as first positional argument
        # - allow legacy alias `n_samples`
        if len(args) > 1:
            raise TypeError(
                "sample() accepts at most one positional argument (conditions). "
                f"Received {len(args)} positional arguments."
            )
        if len(args) == 1:
            if conditions is not None:
                raise TypeError("conditions provided both positionally and by keyword.")
            conditions = args[0]
        if num_samples is None:
            num_samples = n_samples
        if num_samples is None:
            raise TypeError(
                "sample() missing required argument: provide `num_samples` "
                "(or legacy alias `n_samples`)."
            )
        if conditions is None:
            raise TypeError("sample() missing required argument: `conditions`.")

        if metadata is not None:
            if isinstance(conditions, dict) and "responses" in conditions:
                from .envelope import check_data

                check_data(
                    conditions["responses"],
                    metadata=metadata,
                    strict=strict,
                    stacklevel=3,
                )
            else:
                warnings.warn(
                    "metadata was given but conditions has no 'responses'; the "
                    "data was not checked against the trained envelope.",
                    UserWarning,
                    stacklevel=2,
                )

        # 1. Adapt raw conditions (drops inference_variables, keeps summary_variables)
        adapted = self.adapter(conditions, strict=False)
        adapted = {k: ops.convert_to_tensor(v) for k, v in adapted.items()}

        summary_variables = adapted.get("summary_variables")
        inference_conditions = adapted.get("inference_conditions")

        # 2. Sample via the custom per-item pipeline
        samples = self._sample(
            num_samples=num_samples,
            inference_conditions=inference_conditions,
            summary_variables=summary_variables,
            **kwargs,
        )

        # 3. Inverse-adapt: (batch, num_samples, num_items, param_dim) → dict
        samples_np = ops.convert_to_numpy(samples)
        out = self.adapter(
            {"inference_variables": samples_np},
            inverse=True,
            strict=False,
        )

        # 4. Derive loading-space key d = -a*b when applicable
        is_loading = (
            getattr(self.adapter, "_bayesflow_irt_internal_parameterization", None)
            == "loading"
        )
        if (
            is_loading
            and isinstance(out, dict)
            and "d" not in out
            and "a" in out
            and "b" in out
        ):
            out = dict(out)
            out["d"] = -(np.asarray(out["a"]) * np.asarray(out["b"]))

        if split:
            from bayesflow.utils.numpy_utils import split_arrays

            out = split_arrays(out, axis=-1)

        return out

    def get_config(self):
        base = super().get_config()
        base.update(
            {
                "standardize_params": self.standardize_params,
                "flow_chunk_size": self.flow_chunk_size,
                "flow_chunk_backward": self.flow_chunk_backward,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "item_posterior": self.item_posterior,
                "draws_summary_pooling": self.draws_summary_pooling,
                "draws_summary_hidden_dim": self.draws_summary_hidden_dim,
                "draws_summary_dim": self.draws_summary_dim,
                "draws_summary_num_heads": self.draws_summary_num_heads,
            }
        )
        return base

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config = dict(config)
        from .adapters import (
            LogTransform,
            OneHotEncode,
            _GPCMLoadingReparamTransform,
            _LoadingReparamTransform,
        )

        builtin_custom_objects = {
            "LogTransform": LogTransform,
            "OneHotEncode": OneHotEncode,
            "_LoadingReparamTransform": _LoadingReparamTransform,
            "_GPCMLoadingReparamTransform": _GPCMLoadingReparamTransform,
            "bayesflow_irt>LogTransform": LogTransform,
            "bayesflow_irt>OneHotEncode": OneHotEncode,
            "bayesflow_irt>_LoadingReparamTransform": _LoadingReparamTransform,
            "bayesflow_irt>_GPCMLoadingReparamTransform": _GPCMLoadingReparamTransform,
        }
        custom_objects = {
            **builtin_custom_objects,
            **(custom_objects or {}),
        }
        for key in ("adapter", "inference_network", "summary_network"):
            value = config.get(key)
            if isinstance(value, dict) and "class_name" in value:
                config[key] = keras.saving.deserialize_keras_object(
                    value,
                    custom_objects=custom_objects,
                )
        return cls(**config)


# Checkpoints written before this class was registered carry the bare class name
# as their ``registered_name``, which the Keras registry (keyed only by
# ``"bayesflow_irt>Name"``) would not resolve. Register the bare name as an alias
# so those checkpoints keep loading without an explicit ``custom_objects`` dict.
keras.saving.get_custom_objects()["EquivariantIRTApproximator"] = (
    EquivariantIRTApproximator
)
