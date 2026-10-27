"""HPO hooks for IRT models using ``bayesflow_hpo``.

Provides reusable search spaces, validation helpers, and hook factories
for running Optuna-based hyperparameter optimization on IRT approximators
via :func:`bayesflow_hpo.optimize`.

Requires the ``hpo`` extra: ``pip install bayesflow-irt[hpo]``. This module
always imports cleanly (mirroring :mod:`bayesflow_irt.calibration`'s
import-safe pattern for the optional calibration extra); the clear
``ImportError`` is raised lazily, at the first call to :func:`make_irt_hooks`
or :func:`make_irt_validation_dataset`, not at module-import time.

Example
-------
>>> from bayesflow_irt import make_2pl_simulator, make_2pl_adapter, Prior
>>> from bayesflow_irt.hpo import make_irt_hooks
>>> import bayesflow_hpo as hpo
>>>
>>> simulator = make_2pl_simulator(
...     n_items=(5, 10),
...     n_persons=(50, 100),
...     prior_a=Prior("lognormal", loc=0, scale=0.5),
...     prior_b=Prior("normal", loc=0, scale=1.5),
... )
>>> adapter = make_2pl_adapter(simulator)
>>> hooks = make_irt_hooks(
...     simulator, adapter,
...     corners=[(5, 50), (5, 100), (10, 50), (10, 100)],
...     summary_api="matrix",
... )
>>> study = hpo.optimize(
...     simulator=simulator,
...     adapter=adapter,
...     search_space=hooks.search_space,
...     build_approximator_fn=hooks.build_fn,
...     train_fn=hooks.train_fn,
...     validate_fn=hooks.validate_fn,
...     validation_conditions=None,
... )
"""

from __future__ import annotations

import logging
import math
import re
import weakref
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, NamedTuple

import keras
import numpy as np

from .evaluation import sample_posterior
from .sbc import CLASSICAL_REFERENCE_COORDS
from .utils import (
    _PeriodicCudaCacheRelease,
    fit_with_oom_retry,
    set_summary_memory_config,
)

try:
    from bayesflow_hpo import (
        # A study-level misconfiguration must ESCAPE `GenericObjective`. Its
        # catch-all converts anything else raised from `validate_fn` into a
        # training-loss fallback, so a plain ValueError would leave the study
        # scoring every trial on a fabricated number behind a warning line
        # instead of stopping on the first one. Only
        # `JointMetricConfigurationError` and `AggregationConfigError` are
        # re-raised there.
        #
        # From the TOP LEVEL, not `validation.metrics`. Both resolve to the
        # same class, and both are public, but 0.5.0 promoted it here
        # precisely because the deep module "is not where a reader would look
        # for a catchable type" -- that promotion is what the release was
        # cut for. It is also the only one of the two that bayesflow-hpo's
        # `docs/api_reference.md` documents.
        #
        # Imported plainly rather than guarded: the `hpo` extra floors on
        # 0.5.0, the release that defines it. A fallback to `ValueError`
        # would be silently wrong rather than merely degraded -- that is
        # exactly the exception the catch-all swallows, so such a guard
        # would reinstate the failure it is meant to prevent.
        AggregationConfigError,
        CompositeSearchSpace,
        CouplingFlowSpace,
        FlowMatchingSpace,
        TrainingSpace,
        ValidationDataset,
    )
    from bayesflow_hpo.objectives import register_metric_direction

    try:
        # Public accessor since bayesflow-hpo#126 (matthiaskloft/bayesflow-hpo#142).
        from bayesflow_hpo.objectives import metric_direction as _hpo_metric_direction
    except ImportError:  # bayesflow-hpo <= 0.5.3
        # Same resolution the objective uses: unlike `METRIC_DIRECTIONS.get`,
        # it honours `HIGHER_IS_BETTER` mutations, so the worst corner is
        # taken by the convention the study actually minimizes.
        from bayesflow_hpo.objectives import _direction_for as _hpo_metric_direction
    from bayesflow_hpo.pipeline import check_pipeline
    from bayesflow_hpo.registration import register_custom_summary_network
    from bayesflow_hpo.search_spaces import BaseSearchSpace
    from bayesflow_hpo.search_spaces.base import (
        CategoricalDimension,
        FloatDimension,
        IntDimension,
    )
    from bayesflow_hpo.search_spaces.registry import (
        SUMMARY_SPACE_FACTORIES,
        get_summary_space,
    )

    # These five are UNDOCUMENTED upstream: none appears in bayesflow-hpo's
    # `docs/api_reference.md`, and none is listed in a package `__all__`.
    # (`validation/registry.py` defines no `__all__` at all, so absence from
    # that one is not evidence either way -- the API reference is.) They are
    # depended on anyway, because the metric registry is the only thing that
    # can answer which columns a metric emits and which metrics may not be
    # objectives; re-deriving either here would mean maintaining a second
    # copy of upstream's registry, which would go stale silently.
    #
    # The exposure is that an upstream refactor of this module breaks this
    # package at import, and since 0.5.0 there is no fallback to soften it.
    # Export requested in bayesflow-hpo#122; until that is answered, treat a
    # change here as a breaking change and pin accordingly.
    from bayesflow_hpo.validation.registry import (
        JointMetricFn,
        JointMetricInputs,
        is_diagnostic_metric,
        is_joint_metric,
        output_keys_for,
        producer_for_key,
        resolve_joint_metrics,
        resolve_metrics,
    )
except ImportError as exc:
    _BAYESFLOW_HPO_IMPORT_ERROR: ImportError | None = exc

    class _BaseSearchSpacePlaceholder:
        """Stand-in base class so this module still imports without
        ``bayesflow_hpo`` installed (see module docstring)."""

    BaseSearchSpace = _BaseSearchSpacePlaceholder
    CouplingFlowSpace = _BaseSearchSpacePlaceholder
else:
    _BAYESFLOW_HPO_IMPORT_ERROR = None
    # Registered here, matching how `bayesflow_hpo` registers `tarp_error`
    # itself. Without this, `_lower_is_better("tarp_error_item")` returns
    # None and the key cannot be used as an objective -- the failure arrives
    # at `optimize()`, after the search is already configured.
    #
    # `worst_raw=1.0` mirrors the sibling key rather than the 0.5 the
    # statistic actually reaches: a worst case that is merely conservative
    # costs nothing, and two different bounds for the same family of
    # statistic would be the more confusing thing to explain.
    register_metric_direction("tarp_error_item", higher_is_better=False, worst_raw=1.0)
    # Unbounded above, so the failure penalty is +inf.
    register_metric_direction("tcc_z_bias", higher_is_better=False, worst_raw=math.inf)
    # |mean of (TCC error / n_items)| with each term in [-1, 1], so 1.0 bounds it.
    register_metric_direction("tcc_bias", higher_is_better=False, worst_raw=1.0)

from .approximators import EquivariantIRTApproximator
from .networks import MatrixEquivariantIRTSummary

logger = logging.getLogger(__name__)

#: Memory controls applied to every trial, held constant across the search.
#:
#: These do not change a model's numerical output -- they trade compute for
#: peak memory -- so holding them fixed leaves architecture comparisons intact
#: while making a trial trainable at the sizes a variable-size IRT simulator
#: draws.
#:
#: Without them a search on a GPU scores memory feasibility as model
#: quality: an OOM is recorded with the failure penalty, which is
#: indistinguishable from a genuinely bad architecture.
DEFAULT_TRIAL_MEMORY_CONFIG: dict[str, Any] = {
    "gradient_checkpointing": True,
    "person_chunk_size": None,
    "flow_chunk_size": 1024,
    "flow_chunk_backward": "checkpoint",
}


def _require_bayesflow_hpo() -> None:
    """Refuse with a message that distinguishes absent from too old.

    Both reach this function as an ``ImportError``, and saying only
    "requires bayesflow-hpo" sends someone to reinstall a package that is
    already there: a name missing from an older release raises the same
    ``ImportError`` on a machine where the dependency is installed.
    """
    if _BAYESFLOW_HPO_IMPORT_ERROR is None:
        return
    # Report the MODULE's own version and location before falling back to
    # distribution metadata. The two disagree exactly where this message
    # matters most: an editable install pointed at a git worktree, or a
    # `PYTHONPATH` that shadows the installed copy, leaves `.dist-info`
    # describing a different tree than the one Python imported. Reporting
    # metadata alone can therefore assert "0.5.0 is installed" while
    # refusing to import a name 0.5.0 defines, which sends the reader
    # further down the wrong path than saying nothing would.
    module_version: str | None = None
    module_path: str | None = None
    try:
        import bayesflow_hpo as _found

        module_version = getattr(_found, "__version__", None)
        module_path = getattr(_found, "__file__", None)
    except Exception:  # noqa: BLE001 - genuinely absent
        module_version = None
    if module_version is None:
        try:
            from importlib.metadata import version

            module_version = version("bayesflow-hpo")
        except Exception:  # noqa: BLE001 - not installed, or no metadata
            module_version = None

    install_hint = "pip install 'bayesflow-hpo>=0.5.2'"
    if module_version is None:
        detail = (
            "bayesflow_irt.hpo requires bayesflow-hpo>=0.5.0, which is not "
            f"installed. Install it with: {install_hint}"
        )
    else:
        where = f" (imported from {module_path})" if module_path else ""
        detail = (
            "bayesflow_irt.hpo requires bayesflow-hpo>=0.5.0. The version "
            f"Python imported reports {module_version}{where}, and the "
            f"import failed on a name a supported version provides: "
            f"{_BAYESFLOW_HPO_IMPORT_ERROR}. If that version looks new "
            "enough, the imported copy is not the installed one -- check "
            f"PYTHONPATH and editable installs. Otherwise: {install_hint}"
        )
    raise ImportError(detail) from _BAYESFLOW_HPO_IMPORT_ERROR


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------


#: Values of the retired ``irt_matrix_raw_feature_preset`` dimension, which
#: legacy params dicts hold in place of
#: ``irt_matrix_include_rest_mean_score``: whether each stored preset turned
#: the ``rest_mean_score`` channel on. ``"means"`` / ``"means_rest"`` predate
#: the removal of the ``person_mean_score`` / ``item_mean_score`` channels
#: (the Hartford EMN basis computes those means internally) and map to their
#: closest surviving setting, so stored trials still rebuild.
_LEGACY_RAW_FEATURE_PRESETS: dict[str, bool] = {
    "none": False,
    "rest": True,
    "means": False,
    "means_rest": True,
}


@dataclass
class MatrixIRTSummarySpace(BaseSearchSpace):
    """Search space for :class:`MatrixEquivariantIRTSummary` hyperparameters.

    Dimension names are prefixed with ``irt_matrix_`` to avoid collisions
    with other merged search-space keys.

    Two independent switches tune ``raw_feature_channels``, one per
    channel, named after it:

    * ``irt_matrix_include_rest_mean_score``: ``False`` (default) or
      ``True``, adding the cell-level ``rest_mean_score`` channel.
    * ``irt_matrix_include_item_rest_correlation``: ``False`` (default) or
      ``True``, adding the late-fused ``item_rest_correlation`` channel.

    Together they give 4 raw-feature configurations. Legacy params
    stored the first switch as ``irt_matrix_raw_feature_preset``
    (``"none"`` / ``"rest"``, and earlier ``"means"`` / ``"means_rest"``);
    :meth:`build` still reads that key via ``_LEGACY_RAW_FEATURE_PRESETS``
    so stored trials rebuild. The ``irt_matrix_person_context`` dimension
    was removed as redundant with the Hartford EMN basis; any stored key is
    ignored.

    Fixed (not tuned):

    * ``num_outer_heads = 4``
    * ``outer_mlp_depth = 2``
    * ``outer_mlp_width = irt_matrix_outer_embed_dim * 2``
    * ``use_outer_isab = True``, ``num_outer_inducing_points = 16``

    ``item_chunk_size``, ``person_chunk_size``, and ``gradient_checkpointing``
    are intentionally **not** search dimensions here: :func:`build` never sets
    them, so they stay at their constructor defaults (``None``/``False``,
    i.e. dense, unchunked). These are deployment-time memory knobs, not
    modeling hyperparameters — unlike every other dimension in this class,
    they do not change the model's numerical output. Use
    :func:`bayesflow_irt.utils.probe_memory_config` or
    :func:`bayesflow_irt.utils.fit_with_oom_retry` to choose them after a
    trial's architecture is fixed, not as part of the search itself.

    **A caller must actually do that.** Leaving this class's defaults in place
    for a real search is not a neutral choice: on a variable-size simulator the
    dense path allocates the full ``(batch, persons, items, hidden)``
    activation grid and a plausible architecture OOMs part-way through the
    first epoch. :func:`make_irt_hooks` therefore applies
    :data:`DEFAULT_TRIAL_MEMORY_CONFIG` to every trial it builds, and retries
    through ``fit_with_oom_retry`` if that is still not enough.
    """

    irt_matrix_hidden_dim: IntDimension = field(
        default_factory=lambda: IntDimension("irt_matrix_hidden_dim", 16, 64, step=16)
    )
    irt_matrix_num_blocks: IntDimension = field(
        default_factory=lambda: IntDimension("irt_matrix_num_blocks", 1, 4)
    )
    irt_matrix_person_pooling: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "irt_matrix_person_pooling", ["mean", "pma", "mean_pma"]
        )
    )
    irt_matrix_num_pma_seeds: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "irt_matrix_num_pma_seeds", [1, 2, 4]
        )
    )
    irt_matrix_include_rest_mean_score: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "irt_matrix_include_rest_mean_score", [False, True]
        )
    )
    irt_matrix_include_item_rest_correlation: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "irt_matrix_include_item_rest_correlation", [False, True]
        )
    )
    irt_matrix_num_outer_sabs: IntDimension = field(
        default_factory=lambda: IntDimension("irt_matrix_num_outer_sabs", 1, 3)
    )
    irt_matrix_outer_embed_dim: IntDimension = field(
        default_factory=lambda: IntDimension(
            "irt_matrix_outer_embed_dim", 32, 96, step=16
        )
    )
    #: Summary-network-wide dropout: maps to ``MatrixEquivariantIRTSummary``'s
    #: ``dropout``, which feeds the matrix blocks, the PMA pooling *and* the
    #: outer item stack. Not to be confused with the block-only
    #: ``matrix_block_dropout`` override on that class. As with every other
    #: dimension here, the ``irt_matrix_`` prefix names the summary family and
    #: is remapped, not preserved (cf. ``irt_matrix_embed_dim`` ->
    #: ``embed_dim``); renaming it would invalidate saved Optuna studies.
    irt_matrix_dropout: FloatDimension = field(
        default_factory=lambda: FloatDimension("irt_matrix_dropout", 0.0, 0.2)
    )
    irt_matrix_embed_dim: IntDimension = field(
        default_factory=lambda: IntDimension("irt_matrix_embed_dim", 16, 64, step=16)
    )

    def build(self, params: dict[str, Any]) -> MatrixEquivariantIRTSummary:
        """Build a :class:`MatrixEquivariantIRTSummary` from sampled params."""
        outer_embed = int(params["irt_matrix_outer_embed_dim"])
        if "irt_matrix_include_rest_mean_score" in params:
            include_rest = bool(params["irt_matrix_include_rest_mean_score"])
        else:
            legacy = params.get("irt_matrix_raw_feature_preset", "none")
            include_rest = _LEGACY_RAW_FEATURE_PRESETS[legacy]
        raw_feature_channels: tuple[str, ...] = (
            ("rest_mean_score",) if include_rest else ()
        )
        if params.get("irt_matrix_include_item_rest_correlation", False):
            raw_feature_channels = (*raw_feature_channels, "item_rest_correlation")
        return MatrixEquivariantIRTSummary(
            embed_dim=int(params["irt_matrix_embed_dim"]),
            matrix_hidden_dim=int(params["irt_matrix_hidden_dim"]),
            num_matrix_blocks=int(params["irt_matrix_num_blocks"]),
            person_pooling=params["irt_matrix_person_pooling"],
            num_pma_seeds=int(params.get("irt_matrix_num_pma_seeds", 1)),
            raw_feature_channels=raw_feature_channels,
            num_outer_sabs=int(params["irt_matrix_num_outer_sabs"]),
            outer_embed_dim=outer_embed,
            outer_mlp_depth=2,
            outer_mlp_width=outer_embed * 2,
            num_outer_heads=4,
            use_outer_isab=True,
            num_outer_inducing_points=16,
            dropout=float(params["irt_matrix_dropout"]),
        )


# ---------------------------------------------------------------------------
# Registration with bayesflow_hpo's search-space registry
# ---------------------------------------------------------------------------


def _build_irt_matrix_summary(params: dict[str, Any]) -> MatrixEquivariantIRTSummary:
    """``params -> network`` builder registered under ``"irt_matrix"``."""
    return MatrixIRTSummarySpace().build(params)


if _BAYESFLOW_HPO_IMPORT_ERROR is None:
    # Idempotent: guard against re-registration on module reload (e.g. under
    # pytest's importlib reload or repeated notebook re-execution), since
    # bayesflow_hpo.register_custom_summary_network() raises KeyError if the
    # name is already registered and overwrite=False.
    if "irt_matrix" not in SUMMARY_SPACE_FACTORIES:
        register_custom_summary_network(
            name="irt_matrix",
            space_factory=MatrixIRTSummarySpace,
            builder=_build_irt_matrix_summary,
            aliases=["matrix"],
        )


def _dimension_values(dim: Any) -> list[Any]:
    """Every value a categorical or integer dimension can take."""
    # Upstream makes `constant` exclusive with `choices` and `low`/`high`,
    # so an unset range means a pinned dimension.
    if isinstance(dim, CategoricalDimension):
        return [dim.constant] if dim.choices is None else list(dim.choices)
    if dim.low is None:
        return [dim.constant]
    return list(range(dim.low, dim.high + 1, dim.step or 1))


#: Item posteriors an approximator can be built with.
ITEM_POSTERIORS = ("flow_matching", "autoregressive_coupling_flow")


def normalize_item_posterior(item_posterior: str | Sequence[str]) -> tuple[str, ...]:
    """Return the item-posterior families an ``item_posterior`` value names.

    A string pins one family; a list of two or more is searched as a
    categorical ``item_posterior`` dimension (a one-element list pins its
    family, like the bare string).

    Raises
    ------
    ValueError
        If the value is empty, repeats a family, or names an unknown one.
    """
    families = (
        (item_posterior,) if isinstance(item_posterior, str) else tuple(item_posterior)
    )
    if (
        not families
        or len(set(families)) != len(families)
        or not set(families) <= set(ITEM_POSTERIORS)
    ):
        raise ValueError(
            f"item_posterior must be one of {ITEM_POSTERIORS} or a non-empty "
            f"list of distinct ones, got {item_posterior!r}."
        )
    return families


@dataclass
class AutoregressiveCouplingFlowSpace(CouplingFlowSpace):
    """Search space for ``item_posterior="autoregressive_coupling_flow"``.

    The ``cf_*`` dimensions of ``bayesflow_hpo``'s ``CouplingFlowSpace``
    (the per-item flow, built by :meth:`build`) plus the draws summary that
    conditions item ``i`` on the earlier draws (see
    :class:`~bayesflow_irt.EquivariantIRTApproximator`). The draws-summary
    values are read by :meth:`draws_summary_kwargs`, which
    :func:`make_irt_hooks` passes to the approximator.

    Dimensions (in addition to ``CouplingFlowSpace``'s)
    ---------------------------------------------------
    draws_summary_pooling : str
        ``"mean"`` (Habermann et al., 2026), ``"mean_std"``, ``"mean_max"``
        (Corso et al., 2020), ``"attention"`` or ``"pma"`` (Lee et al., 2019).
    draws_summary_hidden_dim : int
        16--64, step 16 (a multiple of the 4 heads).
    draws_summary_dim : int
        16--64, step 16.
    draws_summary_num_heads : int
        Fixed at 4; used only by ``"attention"`` and ``"pma"``.
    """

    draws_summary_pooling: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "draws_summary_pooling",
            choices=["mean", "mean_std", "mean_max", "attention", "pma"],
        )
    )
    draws_summary_hidden_dim: IntDimension = field(
        default_factory=lambda: IntDimension(
            "draws_summary_hidden_dim", low=16, high=64, step=16
        )
    )
    draws_summary_dim: IntDimension = field(
        default_factory=lambda: IntDimension(
            "draws_summary_dim", low=16, high=64, step=16
        )
    )
    draws_summary_num_heads: IntDimension = field(
        default_factory=lambda: IntDimension("draws_summary_num_heads", constant=4)
    )

    def __post_init__(self) -> None:
        # Multi-head attention splits the hidden width across the heads, so
        # an attention pooling with a width the heads do not divide fails in
        # build -- per trial, scored as a bad architecture. Refuse up front.
        poolings = _dimension_values(self.draws_summary_pooling)
        if not {"attention", "pma"} & set(poolings):
            return
        bad = [
            (width, heads)
            for width in _dimension_values(self.draws_summary_hidden_dim)
            for heads in _dimension_values(self.draws_summary_num_heads)
            if width % heads
        ]
        if bad:
            raise ValueError(
                "draws_summary_hidden_dim must be a multiple of "
                "draws_summary_num_heads for the attention poolings; "
                f"(hidden_dim, num_heads) {bad[:3]} are not."
            )

    @staticmethod
    def draws_summary_kwargs(params: dict[str, Any]) -> dict[str, Any]:
        """Approximator kwargs for the draws summary from sampled ``params``."""
        return {
            "draws_summary_pooling": str(params["draws_summary_pooling"]),
            "draws_summary_hidden_dim": int(params["draws_summary_hidden_dim"]),
            "draws_summary_dim": int(params["draws_summary_dim"]),
            "draws_summary_num_heads": int(params["draws_summary_num_heads"]),
        }


@dataclass
class ItemPosteriorChoiceSpace:
    """Search the item posterior itself over the listed families.

    Samples a categorical ``item_posterior`` dimension over ``families``
    first, then only the chosen family's dimensions (Optuna's define-by-run
    sampling), so a trial's parameters name one family: ``fm_*`` or
    ``cf_*`` plus ``draws_summary_*``. One study then holds one Pareto front
    over the listed families, trained and validated on the same data.

    The parameter is named ``item_posterior`` rather than bayesflow-hpo's
    ``NetworkSelectionSpace`` key ``inference_network_type`` because it
    selects the approximator's factorization, not only the network.

    Parameters
    ----------
    flow_matching : FlowMatchingSpace
        Space searched when the trial picks ``"flow_matching"``.
    autoregressive_coupling_flow : AutoregressiveCouplingFlowSpace
        Space searched when the trial picks ``"autoregressive_coupling_flow"``.
    families : tuple of str
        Families the categorical chooses from, in this order (any distinct
        subset of :data:`ITEM_POSTERIORS`). Defaults to all of them. A single
        family pins the choice (used to dry-run each family in the pipeline check).
    """

    flow_matching: Any = field(default_factory=lambda: FlowMatchingSpace())
    autoregressive_coupling_flow: Any = field(
        default_factory=lambda: AutoregressiveCouplingFlowSpace()
    )
    families: tuple[str, ...] = ITEM_POSTERIORS

    def __post_init__(self) -> None:
        self.families = normalize_item_posterior(self.families)
        if not isinstance(
            self.autoregressive_coupling_flow, AutoregressiveCouplingFlowSpace
        ):
            raise TypeError(
                "autoregressive_coupling_flow must be an "
                "AutoregressiveCouplingFlowSpace, got "
                f"{type(self.autoregressive_coupling_flow).__name__}."
            )
        if isinstance(self.flow_matching, AutoregressiveCouplingFlowSpace):
            raise TypeError(
                "flow_matching must not be an AutoregressiveCouplingFlowSpace."
            )

    def _space(self, item_posterior: str) -> Any:
        if item_posterior not in self.families:
            raise ValueError(
                f"item_posterior must be one of {self.families}, "
                f"got {item_posterior!r}."
            )
        return getattr(self, item_posterior)

    @property
    def dimensions(self) -> list[Any]:
        """The choice dimension plus the listed families' dimensions."""
        return [
            CategoricalDimension("item_posterior", choices=list(self.families)),
            *(d for fam in self.families for d in getattr(self, fam).dimensions),
        ]

    @property
    def constants(self) -> dict[str, Any]:
        """The listed families' pinned dimensions (their names do not overlap)."""
        return {
            k: v
            for fam in self.families
            for k, v in getattr(getattr(self, fam), "constants", {}).items()
        }

    def sample(self, trial: Any) -> dict[str, Any]:
        """Sample the family, then that family's dimensions."""
        choice = trial.suggest_categorical("item_posterior", list(self.families))
        return {"item_posterior": choice, **self._space(choice).sample(trial)}

    def build(self, params: dict[str, Any]) -> Any:
        """Build the inference network of ``params["item_posterior"]``."""
        return self._space(params["item_posterior"]).build(params)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class IRTHooks(NamedTuple):
    """Bundle returned by :func:`make_irt_hooks`.

    Attributes
    ----------
    search_space : CompositeSearchSpace
        Combined search space (inference + IRT summary + training).
    validation_data : ValidationDataset
        Pre-generated edge-corner validation dataset.
    build_fn : callable
        ``build_approximator_fn`` hook for ``optimize()``.
    train_fn : callable
        ``train_fn`` hook for ``optimize()``.
    validate_fn : callable
        ``validate_fn`` hook for ``optimize()``.  **Ignores** the
        ``validation_data`` argument passed by ``optimize()`` and uses
        the captured edge-corner dataset instead.
    """

    search_space: CompositeSearchSpace
    validation_data: ValidationDataset
    build_fn: Callable
    train_fn: Callable
    validate_fn: Callable


# ---------------------------------------------------------------------------
# Validation dataset helper
# ---------------------------------------------------------------------------

_DEFAULT_SIMS_PER_CORNER = 25
_DEFAULT_SEED = 42

#: Accepted forms of a validation budget: one count shared by every corner,
#: one count per corner positionally, or a mapping keyed by the corner's own
#: ``(n_items, n_persons)`` pair.
SimsPerCorner = int | Sequence[int] | Mapping[tuple[int, int], int]


def _resolve_sims_per_corner(
    sims_per_corner: SimsPerCorner,
    corners: Sequence[tuple[int, int]],
) -> list[int]:
    """Resolve a validation budget into one simulation count per corner.

    A scalar is broadcast, a sequence is taken positionally against
    *corners*, and a mapping is looked up by each corner's
    ``(n_items, n_persons)`` pair.

    The mapping form exists because the positional one can misalign
    silently: reordering ``corners`` without reordering the counts is a
    valid-looking config that validates a different grid than intended.
    A mapping is keyed by the thing it describes, so the same reordering
    is a no-op. Both are therefore checked against *corners* here rather
    than trusted -- a sequence of the wrong length and a mapping that
    misses or oversupplies a corner both raise.

    Parameters
    ----------
    sims_per_corner : int or sequence of int or mapping
        The budget, in any of the three accepted forms.
    corners : sequence of (int, int)
        The corners it must cover, as ``(n_items, n_persons)`` pairs.

    Returns
    -------
    list of int
        One positive count per element of *corners*, in that order.
    """
    if not corners:
        # Nothing to allocate over. Left to the caller to reject, so the
        # empty-`corners` error stays the one that names `corners` rather
        # than a confusing length mismatch against the budget.
        return []

    keys = [(int(n_items), int(n_persons)) for n_items, n_persons in corners]

    if isinstance(sims_per_corner, bool):
        # `bool` is an `int` subclass, so it would otherwise broadcast as 0
        # or 1 rather than being caught as the mistake it is.
        raise TypeError("sims_per_corner must be an int, not a bool.")
    if isinstance(sims_per_corner, (int, np.integer)):
        counts = [int(sims_per_corner)] * len(keys)
    elif isinstance(sims_per_corner, Mapping):
        missing = [k for k in keys if k not in sims_per_corner]
        if missing:
            raise ValueError(
                "sims_per_corner is missing a count for "
                f"{missing}. A mapping must cover every corner; "
                f"corners are {keys}."
            )
        # Compared as the keys are, without coercing them: every hashable
        # key already compares equal to the corner tuple it means -- a
        # float or `np.integer` tuple included -- and coercing instead
        # turned an unhashable or mixed-type stray key into a bare
        # `TypeError` from `tuple()` or `sorted()`, losing the message
        # that says which entry is wrong.
        known = set(keys)
        unknown = [k for k in sims_per_corner if k not in known]
        if unknown:
            raise ValueError(
                f"sims_per_corner has entries for {unknown}, which are not "
                f"corners of this dataset ({keys}). Note the key order is "
                "(n_items, n_persons)."
            )
        counts = [_as_count(sims_per_corner[k], f"sims_per_corner[{k}]") for k in keys]
    elif isinstance(sims_per_corner, np.ndarray) or (
        # An ndarray is not a `Sequence`, but a budget computed from
        # measured per-corner SDs arrives as one; both are ordered, which
        # is all the positional form needs. A `set` is not, and stays
        # rejected below rather than being allocated in arbitrary order.
        isinstance(sims_per_corner, Sequence)
        and not isinstance(sims_per_corner, (str, bytes))
    ):
        values = list(sims_per_corner)
        if len(values) != len(keys):
            raise ValueError(
                f"sims_per_corner has {len(values)} entries but there "
                f"{'is' if len(keys) == 1 else 'are'} {len(keys)} "
                f"corner{'' if len(keys) == 1 else 's'}. A sequence is "
                "positional against `corners`; pass a mapping keyed by "
                "(n_items, n_persons) to be independent of their order."
            )
        counts = [_as_count(v, f"sims_per_corner[{i}]") for i, v in enumerate(values)]
    else:
        raise TypeError(
            "sims_per_corner must be an int, a sequence of int, or a mapping "
            f"from (n_items, n_persons) to int, got {type(sims_per_corner).__name__}."
        )

    for key, count in zip(keys, counts):
        if count < 1:
            raise ValueError(
                f"sims_per_corner must be >= 1 for every corner, got {count} for {key}."
            )
    return counts


def _as_count(value: Any, what: str) -> int:
    """Coerce one budget entry to ``int``, rejecting non-integral values."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{what} must be an int, got {type(value).__name__}.")
    return int(value)


def make_irt_validation_dataset(
    simulator: Any,
    corners: list[tuple[int, int]],
    param_keys: list[str],
    *,
    sims_per_corner: SimsPerCorner = _DEFAULT_SIMS_PER_CORNER,
    seed: int = _DEFAULT_SEED,
) -> ValidationDataset:
    """Build a :class:`ValidationDataset` from edge-condition corners.

    For each ``(n_items, n_persons)`` pair in *corners*, a fixed-size
    simulator clone is created via ``clone_with_sizes()`` and sampled.
    This bypasses ``bayesflow_hpo.generate_validation_dataset()`` because
    IRT simulators do not accept ``conditions=`` keyword arguments: variable
    matrix shape is controlled via ``clone_with_sizes(n_items=, n_persons=)``,
    not a ``conditions`` dict forwarded to ``.sample()``.

    .. note::
        This is a deliberate, documented workaround rather than a shared
        code path. A future ``bayesflow_hpo`` release may add a generic
        extension point (e.g. a ``sampler_fn`` override on
        ``generate_validation_dataset()``) so simulators with a
        ``clone_with_sizes``-style API can plug into the shared helper
        directly; until then, this function stays IRT-local. See the
        matching note on ``bayesflow_hpo.validation.data.generate_validation_dataset``.

    Parameters
    ----------
    simulator : bayesflow_irt simulator
        A simulator returned by ``make_1pl_simulator`` or
        ``make_2pl_simulator``.
    corners : list of (int, int)
        Each element is ``(n_items, n_persons)`` pinned at both ends
        (``lo == hi``).
    param_keys : list of str
        Parameter names to validate (e.g. ``["a", "b"]`` for 2PL).
    sims_per_corner : int or sequence of int or mapping
        Validation budget. An ``int`` is used at every corner. A sequence
        is positional against *corners*. A mapping is keyed by the corner's
        ``(n_items, n_persons)`` pair, which cannot misalign if *corners*
        is reordered.

        A per-corner budget exists because corners can differ in
        measurement SD -- and, because item-pooled statistics score
        ``n_sims * n_items`` rows, few-item corners tend to be noisier,
        while also being the cheapest to simulate. Equalising
        fidelity through a scalar raises the whole grid to the worst
        corner's requirement; allocating per corner buys the same
        worst-corner precision for a fraction of the seconds.

        Note what an unequal budget does NOT change: the reduction over
        corners is unweighted, so ``"<metric>/mean"`` becomes a plain mean
        over estimates of unequal precision. ``"<metric>/worst"`` is
        unaffected. See :func:`make_irt_hooks`, which warns about the
        combination.
    seed : int
        Random seed (stored in the dataset for reproducibility metadata).

        This is metadata, not the draw's seed: each corner is sampled from
        a clone produced by ``clone_with_sizes()``, which carries the
        *simulator's* own ``seed``. A clone starts a fresh generator from
        that seed, so a corner's draw depends on its own size and on
        nothing an earlier corner did.

        **For a simulator built with a seed** -- which is the only case
        where reproducibility is on offer at all -- that makes changing
        one corner's count leave every other corner's batch
        bit-identical, and it is what makes a per-corner budget safe to
        tune. Verified across 1PL, 2PL, PCM and GPCM, on both the numpy
        and numba response paths. The package default is ``seed=None``,
        where each clone takes fresh entropy and no batch is reproducible
        in the first place.

        The corners do share a stream seed with one another, so their
        draws are correlated; that is pre-existing behaviour, unchanged
        here because altering it would move every metric value recorded
        with earlier versions.

    Returns
    -------
    ValidationDataset
    """
    _require_bayesflow_hpo()
    if not corners:
        raise ValueError(
            "corners must be a non-empty list of (n_items, n_persons) tuples."
        )
    if not param_keys:
        raise ValueError("param_keys must be a non-empty list of parameter names.")
    if not hasattr(simulator, "clone_with_sizes"):
        raise TypeError(
            f"Simulator {type(simulator).__name__} does not "
            "support clone_with_sizes(). "
            "Use an IRT simulator from bayesflow_irt."
        )

    counts = _resolve_sims_per_corner(sims_per_corner, corners)

    simulations: list[dict[str, np.ndarray]] = []
    condition_labels: list[dict[str, float | int]] = []

    for (n_items, n_persons), n_sims in zip(corners, counts):
        corner_sim = simulator.clone_with_sizes(
            n_items=(n_items, n_items),
            n_persons=(n_persons, n_persons),
        )
        batch = corner_sim.sample((n_sims,))
        simulations.append(batch)
        condition_labels.append({"n_items": n_items, "n_persons": n_persons})

    return ValidationDataset(
        simulations=simulations,
        condition_labels=condition_labels,
        param_keys=param_keys,
        data_keys=["responses"],
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Hook factory
# ---------------------------------------------------------------------------

#: Metrics computed at every trial. Computing one is cheap once the posterior
#: draws exist, so this is deliberately wider than the set worth *optimizing*
#: -- see :data:`DEFAULT_OBJECTIVE_METRICS`.
#:
#: ``log_gamma`` replaces ``calibration_error`` as the calibration term.
#: BayesFlow documents its ``calibration_log_gamma`` as "typically more
#: sensitive than the Kolmogorov-Smirnoff test or ChiSq test", and unlike
#: ``calibration_error`` it is a hypothesis-test statistic with a defined null
#: rather than a median absolute deviation.
#:
#: ``calibration_error`` is still computed, for comparability with results
#: recorded with earlier versions. It is simply no longer
#: recommended as a search objective.
_DEFAULT_METRICS = ["log_gamma", "nrmse", "calibration_error"]

#: The subset to pass to ``bayesflow_hpo.optimize(objective_metrics=...)``.
#:
#: **`nrmse` is load-bearing and must not be dropped.** Both calibration terms
#: are rank statistics over *marginal* parameters, and Modrák et al. (2025)
#: show that such a statistic returns a clean result for a posterior that
#: ignores the data entirely -- as does expected HPD coverage (Lemos et al.
#: 2023, Sec. 3.1). A search optimizing only on calibration could therefore be
#: rewarded for finding exactly that degenerate posterior. `nrmse` is what
#: makes it unprofitable: a posterior that ignores the data has terrible point
#: recovery, so it cannot reach the useful part of the Pareto front.
#:
#: **Switching to `log_gamma` does NOT close that blind spot**, and it would
#: be wrong to record it as if it had. The blind spot is about *which quantity
#: is ranked*, not which test is applied to the ranks --
#: ``bayesflow_hpo``'s ``_bf_log_gamma`` calls ``calibration_log_gamma``
#: without ``test_quantities``, so it ranks marginal parameters exactly as
#: before. Closing it requires a test quantity that is a function of data
#: *and* parameters, i.e. the joint log-likelihood. Currently,
#: `nrmse` is the guard, not the calibration term.
#:
#: **Leave ``directions=None``.** `log_gamma` is higher-is-better while
#: `nrmse` is lower-is-better, so this list looks like it needs an explicit
#: ``directions=["maximize", "minimize", ...]``. It does not, and passing one
#: is an error: from ``bayesflow-hpo`` 0.2.0 the objective converts every
#: metric into minimize space itself through
#: ``bayesflow_hpo.objectives.METRIC_DIRECTIONS`` (`log_gamma` is registered
#: with ``higher_is_better=True`` and ``to_minimize=lambda v: -v``), and
#: ``optimize`` rejects any non-``minimize`` entry because applying one would
#: invert that conversion a second time. That floor is why
#: a direction for `log_gamma` must be registered upstream: without one,
#: this default would make the search prefer the *most strongly rejected*
#: calibration result.
DEFAULT_OBJECTIVE_METRICS = ["log_gamma", "nrmse"]

# Which way is "better" for a metric comes from `bayesflow_hpo`'s direction
# registry -- the same resolution the objective minimizes -- not from a local
# table. A second hand-maintained copy drifts silently: a direction read from
# the wrong place does not raise, it ranks backwards.
#
# Aggregating across corners requires knowing the sign, and guessing it is
# strictly worse than refusing: a metric silently aggregated the wrong way
# would report the *best* corner as the worst-case guarantee, which is the
# failure this aggregation exists to prevent. Upstream records these facts;
# why they matter for corner aggregation:
#
# - ``calibration_error`` (ECE), ``rmse``, ``nrmse`` are errors: 0 is perfect.
# - ``sbc_ks`` / ``sbc_chi2`` are registered in `bayesflow_hpo` as
#   "minimize -> 0 = uniform ranks".
# - ``contraction`` is registered as "1 = strong learning, 0 = no narrowing".
# - ``log_gamma`` runs the OTHER way. BayesFlow's `calibration_log_gamma`
#   documents it as log(gamma/gamma_null) where "log_gamma < 0 implies a
#   rejection of the hypothesis of uniform ranks at the 5% level", so a
#   larger value is better and taking the max would report the best corner.
#
# Metrics this package owns (`tarp_error_item`, `tcc_bias`, `tcc_z_bias`)
# are registered upstream at import, above.


def _joint_block(
    values: np.ndarray, key: str, *, sample_axis: int | None
) -> np.ndarray:
    """Flatten one parameter's item axes into columns, keeping simulations.

    *values* is ``(B, I, ...)`` for truths and ``(B, S, I, ...)`` for draws.
    The item axis (and any trailing category axis) collapses into the
    parameter-column axis; the simulation axis never does. That is the whole
    difference from the marginal path, which folds items into ``B``.
    """
    transform = _JOINT_PARAM_TRANSFORMS.get(key)
    if transform is not None:
        finite_before = np.isfinite(values)
        with np.errstate(divide="ignore", invalid="ignore"):
            values = transform(values)
        if not np.all(np.isfinite(values[finite_before])):
            # Almost always one cause: the adapter was built with
            # `log_transform_a=False`, so its inverse does not exponentiate
            # and `a` draws arrive on a scale that reaches zero and below.
            # `np.log` then yields NaN/-inf, the joint metric raises on
            # non-finite draws, and the trial reports as a *failure* rather
            # than as the configuration mismatch it is. Say which it is.
            n_bad = int(np.sum(~np.isfinite(values[finite_before])))
            raise ValueError(
                f"Transforming parameter {key!r} into the joint metric space "
                f"produced {n_bad} non-finite value(s) from finite input. "
                f"`_JOINT_PARAM_TRANSFORMS` applies {transform.__name__} to "
                f"{key!r}, which assumes it is strictly positive on the scale "
                "the approximator returns. An adapter built with "
                "`log_transform_a=False` returns `a` on a scale that reaches "
                "zero, which is the usual cause. Joint metrics here assume "
                "the default adapter; see `_JOINT_PARAM_TRANSFORMS`."
            )
    if sample_axis is None:
        return values.reshape(values.shape[0], -1)
    return values.reshape(values.shape[0], values.shape[sample_axis], -1)


def _joint_row(
    joint_metric_fns: Mapping[str, Any],
    *,
    corner_batch: Mapping[str, np.ndarray],
    posterior: Mapping[str, np.ndarray],
    param_keys: Sequence[str],
    data_keys: Sequence[str],
    approximator: Any,
    cond_id: int,
    n_conditions: int,
    failed_joint: dict[str, str],
    emitted_keys: dict[str, set[str]],
) -> dict[str, float]:
    """Score every joint metric on one condition.

    Assembles the ``(n_sims, n_samples, n_params)`` view
    :class:`~bayesflow_hpo.validation.registry.JointMetricInputs` requires.
    ``n_params`` is ``n_items`` times the number of parameter keys: for 2PL
    the columns are ``[log a_1 .. log a_I, b_1 .. b_I]``, blocked by
    parameter and then by item, which is the layout the classical
    reference provider expects.

    The column order is passed explicitly as ``param_keys`` rather than left
    implied by iteration order, because nothing else records it and a
    reference provider slices positionally.

    Failure handling mirrors ``bayesflow_hpo.validation.pipeline``'s joint
    guard, deliberately rather than coincidentally -- the reasoning there
    applies unchanged here, and two different answers to the same question
    would be worse than either:

    - **A failure anywhere invalidates the metric for the whole trial**, via
      *failed_joint*, and the aggregation then drops its keys entirely
      instead of averaging the corners that did work. A worst case taken
      over the corners that happened to succeed is not a worst case over
      the grid, and it reads as a finite, flattering number precisely when
      the measurement is broken.
    - **A NaN takes the same path as a raise.** A metric returning a number
      on one corner and NaN on another would otherwise be reduced to the
      number.
    - **Keys are recorded as *emitted*, not as registered.** A metric passed
      through ``joint_metrics=`` carries a caller-chosen mapping key that
      need not match what the callable returns -- ``make_tarp_joint_metric``
      emits ``tarp_error_random`` whenever it has no reference provider, no
      matter which key it is registered under -- so dropping by mapping key
      alone would leave the computed value in place under a name nothing
      expects.
    """
    true_values = np.concatenate(
        [
            _joint_block(np.asarray(corner_batch[k], dtype=float), k, sample_axis=None)
            for k in param_keys
        ],
        axis=1,
    )
    draws = np.concatenate(
        [
            _joint_block(np.asarray(posterior[k], dtype=float), k, sample_axis=1)
            for k in param_keys
        ],
        axis=2,
    )
    inputs = JointMetricInputs(
        draws=draws,
        true_values=true_values,
        param_keys=tuple(param_keys),
        sim_batch=corner_batch,
        data_keys=tuple(data_keys),
        approximator=approximator,
        cond_id=cond_id,
        n_conditions=n_conditions,
    )
    row: dict[str, float] = {}
    for name, fn in joint_metric_fns.items():
        if name in failed_joint:
            # Already invalidated for this trial; recomputing it would pay
            # the full per-condition price for a value that gets dropped.
            continue
        try:
            result = fn(inputs)
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            failed_joint[name] = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "Joint metric %r failed on condition %d and is invalidated "
                "for this trial: %s",
                name,
                cond_id,
                exc,
            )
            continue
        if not isinstance(result, Mapping):
            # `result.items()` would raise AttributeError, which escapes to
            # the corner loop and discards every marginal result the trial
            # has already computed. One metric returning the wrong shape
            # must not have that blast radius.
            failed_joint[name] = f"returned {type(result).__name__}, expected a mapping"
            logger.warning(
                "Joint metric %r returned %s instead of a mapping and is "
                "invalidated for this trial.",
                name,
                type(result).__name__,
            )
            continue
        # What this metric emitted on EARLIER conditions, captured before
        # the update below folds this condition in -- the completeness check
        # at the bottom compares against it, and would always see an empty
        # difference if it read the merged set.
        previously_emitted = set(emitted_keys.get(name, ()))
        # Recorded before the NaN check, so a metric that fails on a LATER
        # condition still drops what it contributed on this one.
        emitted_keys.setdefault(name, set()).update(result)
        try:
            converted = {key: float(value) for key, value in result.items()}
        except (TypeError, ValueError) as exc:
            failed_joint[name] = (
                f"returned a non-numeric value on condition {cond_id}: {exc}"
            )
            logger.warning(
                "Joint metric %r returned a non-numeric value on condition "
                "%d and is invalidated for this trial: %s",
                name,
                cond_id,
                exc,
            )
            continue
        # Converted first, then inspected: `np.float32` does not subclass
        # Python float, so testing the raw value would miss a NaN coming
        # out of a float32 reduction. An infinity is left alone -- a metric
        # can mean it.
        nan_keys = sorted(k for k, v in converted.items() if math.isnan(v))
        if nan_keys:
            failed_joint[name] = f"returned NaN for {nan_keys} on condition {cond_id}"
            logger.warning(
                "Joint metric %r returned NaN for %s on condition %d and is "
                "invalidated for this trial.",
                name,
                nan_keys,
                cond_id,
            )
            continue
        # A key this metric produced on an EARLIER condition but not on this
        # one is the same failure as a NaN, and just as quiet: aggregation
        # skips the missing value and the worst case is taken over the
        # corners that happened to report. A metric returning
        # `{"tarp_error": 0.01}` on one corner and `{}` on the next reported
        # `tarp_error/worst == 0.01` -- a clean worst-case guarantee derived
        # from half the grid.
        #
        # Checked against what the metric has been OBSERVED to emit, not
        # against the registry: an ad hoc joint metric declares nothing, and
        # the first condition is what establishes the expected key set.
        missing = sorted(previously_emitted - converted.keys())
        if missing:
            failed_joint[name] = (
                f"omitted {missing} on condition {cond_id} after emitting "
                "them on an earlier one"
            )
            logger.warning(
                "Joint metric %r omitted %s on condition %d after emitting "
                "them earlier, and is invalidated for this trial.",
                name,
                missing,
                cond_id,
            )
            continue
        if not converted:
            # Nothing on the very first condition either: there is no
            # measurement, so there is nothing to aggregate and nothing that
            # should look like a score.
            failed_joint[name] = f"returned no keys on condition {cond_id}"
            logger.warning(
                "Joint metric %r returned no keys on condition %d and is "
                "invalidated for this trial.",
                name,
                cond_id,
            )
            continue
        row.update(converted)
    return row


def _dropped_joint_keys(
    failed_joint: Mapping[str, str], emitted_keys: Mapping[str, set[str]]
) -> set[str]:
    """Summary keys belonging to joint metrics invalidated this trial.

    Both sources are needed and neither covers the other: the *observed*
    keys are empty when a metric failed on the very first condition, and the
    *declared* ones are only meaningful for a name the registry knows.
    """
    dropped: set[str] = set()
    for name in failed_joint:
        dropped.update(emitted_keys.get(name, ()))
        dropped.update(output_keys_for(name))
    return dropped


def _lower_is_better(base: str) -> bool | None:
    """Return whether lower is better for *base*, or ``None`` if unknown.

    Reads ``bayesflow_hpo``'s direction registry through
    ``bayesflow_hpo.metric_direction`` (``_direction_for`` on older
    releases), i.e. the direction the objective actually minimizes. Not
    ``METRIC_DIRECTIONS.get``: that ignores ``HIGHER_IS_BETTER`` mutations
    and can disagree with the objective. ``None`` means no recorded
    direction, which callers refuse on rather than guess.
    """
    if _BAYESFLOW_HPO_IMPORT_ERROR is not None:
        return None
    direction = _hpo_metric_direction(base)
    if direction is None:
        return None
    return not direction.higher_is_better


def _worst(metric_name: str, values: list[float]) -> float:
    """Reduce per-corner values to the worst one, by the metric's direction."""
    base = metric_name.split("/")[0]
    lower_is_better = _lower_is_better(base)
    if lower_is_better is None:
        raise ValueError(
            f"cannot take the worst case over corners for metric "
            f"{metric_name!r}: its direction is unknown. Register it with "
            "bayesflow_hpo.objectives.register_metric_direction(), which "
            "this module reads, with a source for which way is better. "
            "Or pass "
            "aggregate='mean' and accept that corners cancel. "
            "Guessing the direction would silently report the best corner as "
            "the worst-case guarantee."
        )
    return max(values) if lower_is_better else min(values)


#: Reductions `aggregate` accepts, matching ``bayesflow_hpo``'s own names.
_AGGREGATIONS = ("mean", "worst", "geometric")


#: The key shape ``_validate_fn`` synthesizes for its per-corner values.
#:
#: Matched against the keys a trial actually reduces, to keep a metric from
#: occupying the namespace this module writes into. A joint metric emits
#: whatever its callable returns, so without this a metric emitting the
#: literal key ``"nrmse/corner_0/mean"`` overwrites the summary synthesized
#: for ``nrmse`` -- two different measurements under one key, indistinguishable
#: once stored.
_RESERVED_CORNER_KEY = re.compile(r"/corner_\d+/(?:mean|worst)$")


def _geometric(values: list[float]) -> float:
    """Geometric mean, or NaN when the values leave its domain.

    NaN rather than an exception, deliberately. ``correlation`` and
    ``contraction`` are legitimately non-positive for an undertrained
    approximator, so raising would let one poor trial end an otherwise
    healthy study -- the same distinction ``bayesflow_hpo`` draws between its
    ``AggregationConfigError`` and ``AggregationDomainError``. A NaN here
    reaches Optuna as a failed trial, which is what a value outside the
    reduction's domain means for that trial alone.

    Computed in log space: a product over eight grid cells of numbers around
    1e-2 underflows a float64 before the root is taken.
    """
    if any(v <= 0.0 for v in values):
        return float("nan")
    return float(np.exp(np.mean(np.log(values))))


#: What an unnamed metric falls back to when `aggregate` is a mapping.
#:
#: NOT ``bayesflow_hpo``'s "mean", and the difference is deliberate: this
#: factory's documented default is the worst case, because a mean over the
#: grid lets opposite failures cancel (Modrák et al., 2025). Naming one
#: metric's reduction must not quietly move every other metric onto the
#: reduction the default exists to avoid.
_DEFAULT_AGGREGATE = "worst"


def _resolve_aggregate(aggregate: str | Mapping[str, str], metric_name: str) -> str:
    """Which reduction applies to *metric_name*.

    Reduction happens per EMITTED key, so *metric_name* is a column such as
    ``"coverage_90"`` rather than the metric the caller configured. Both
    spellings are honoured: an exact key wins, and otherwise the producing
    metric's entry applies to every column it emits. Naming ``"coverage"``
    therefore sets the reduction for all five of its columns at once, which
    is what a caller who listed ``"coverage"`` in ``metrics`` means, while
    naming ``"coverage_90"`` still overrides that single column.
    """
    if isinstance(aggregate, str):
        return aggregate
    key = metric_name.split("/")[0]
    if key in aggregate:
        return aggregate[key]
    return aggregate.get(_producer_of(key), _DEFAULT_AGGREGATE)


def _is_diagnostic(key: str) -> bool:
    """Whether *key* comes from a metric that may not be an objective.

    Asked of the PRODUCER, because the registry records the kind against
    the metric rather than against each column it emits:
    ``is_diagnostic_metric("coverage")`` is True while
    ``is_diagnostic_metric("coverage_90")`` is False, and the second answer
    is about a name the registry has no kind for, not a licence to optimize
    the column.

    An unregistered key -- a joint metric's own emitted key, say -- has no
    kind, and ``is_diagnostic_metric`` returns ``False`` for it rather than
    raising (verified). That is the right answer here: absent a declared
    kind, nothing licenses skipping the worst case.
    """
    return bool(is_diagnostic_metric(_producer_of(key)))


def _producer_of(key: str) -> str:
    """The metric that emits *key*, or *key* itself when unknown.

    ``producer_for_key`` is declared ``str | None`` and **returns** ``None``
    for a key the registry never registered; it does not raise. That case is
    normal rather than exceptional here, because a joint metric's emitted key
    comes from a caller's callable and was never registered.

    Never returns ``None``. Guarding with ``except Exception`` would be
    dead code against a function that returns instead of raising, and a
    ``None`` would matter: the inert-key check below puts this result in a
    set it then sorts, and ``sorted({"nrmse", None})`` raises
    ``TypeError`` -- which ``GenericObjective`` does NOT re-raise, so the
    one path written to stop a misconfigured study would turn into a
    training-loss fallback on every trial instead.
    """
    return producer_for_key(key) or key


# ---------------------------------------------------------------------------
# Joint metrics: the data-dependent reference point
# ---------------------------------------------------------------------------

#: Scale each parameter is carried in when draws are assembled jointly.
#:
#: ``a`` is logged. Two independent reasons agree on this, and both matter:
#: it is the space the flow is trained in (:func:`make_2pl_adapter` applies
#: ``log`` to ``a`` before standardizing **at its ``log_transform_a=True``
#: default** -- an adapter built with ``log_transform_a=False`` breaks that
#: assumption, and :func:`_joint_block` raises rather than emitting NaN),
#: and it is the space the classical reference scheme stacks ``log(a)``
#: against ``b`` in. TARP standardizes each dimension before computing
#: distances, but standardizing does not undo a log, so a joint metric run
#: on natural-scale ``a`` measures a different geometry.
#:
#: A joint metric receives draws and truths already in these coordinates.
_JOINT_PARAM_TRANSFORMS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "a": np.log,
}

#: Classical index standing in for each item parameter, as a TARP reference.
#:
#: Deliberately the *unconverted* indices -- see
#: :func:`~bayesflow_irt.sbc.classical_item_statistics`, which carries the
#: attribution and documents why applying the normal-ogive conversions would
#: turn these into approximate *estimates* of the item parameters and destroy
#: the offset that makes the reference work.
#:
#: ``point_biserial`` stands for discrimination and ``z_difficulty`` for
#: difficulty. Both are defined for dichotomous items only, which is what
#: confines this scheme to 1PL/2PL.
# Alias of sbc.CLASSICAL_REFERENCE_COORDS.
_CLASSICAL_REFERENCE_COORDS = CLASSICAL_REFERENCE_COORDS

#: Jitter added to the classical reference coordinates, by default.
#:
#: **Zero.** The reference is then a deterministic function of the
#: responses, which is the arm with the most sensitivity and the only one
#: with no scale to get wrong.
#:
#: Jitter is not outside the method -- Lemos et al. (2023, Sec. 4.3) use
#: ``theta_r = x_0 + u`` with ``u ~ U(0, 1)``, uncentred too, and that is
#: exactly what a nonzero ``jitter`` still applies. But
#: ``make_tarp_joint_metric(standardize=True)`` divides each reference
#: column by the *truth's* sd -- ``sd(log a)`` for the discrimination
#: block, since :data:`_JOINT_PARAM_TRANSFORMS` applies ``np.log`` to
#: ``a`` -- and the two classical indices have very different spreads. A
#: fixed jitter therefore makes up a very different share of each block's
#: reference variance, and that share moves with the corner. A joint
#: coverage curve averages the blocks and cannot show that one of them is
#: mostly noise. Reshaping the jitter -- centring it, or scaling it per
#: block -- would be a noise law no source here describes, so the default
#: is 0 instead.
#:
#: On calibrated posteriors the jitter does not move the null, which is
#: what the theorem guarantees; for a posterior of the wrong *width*,
#: raising the jitter costs sensitivity.
#:
#: Sharpen a result with more simulations instead. One ``theta_r`` per
#: simulation is what Algorithm 2 does, so the number of reference points
#: grows with the number of simulations, one each, and every new simulation
#: brings a fresh ``x`` -- a genuinely new place to check coverage rather
#: than a wider smear around the same ones.
#:
#: What neither dial fixes: a deterministic reference is a single position
#: function, while Theorem 3's sufficiency is stated over all of them. For a
#: reference ``f(x)``, a true posterior ``N(f(x) + a, 1)`` and an estimate
#: ``N(f(x) - a, 1)`` put identical distributions on the distance from the
#: reference, so no number of simulations separates them.
#:
#: Jitter *does* separate them, because ``U(0, jitter)`` is not symmetric
#: about zero and so breaks the mirror symmetry the blindness rests on. At
#: ``a = 1``, raising the jitter raises ``tarp_error`` -- the opposite
#: ordering to the width case above. See
#: ``test_jitter_detects_the_mirror_case_the_default_misses``.
#:
#: So the two dials trade against each other, and 0 is a default rather than
#: a dominant choice: it maximises sensitivity to an estimator
#: *shrunk toward the prior* -- which a zero-jitter
#: reference does see, since such an estimator sits closer to the reference
#: than the truth does -- at the cost of being blind to an exact mirror. A
#: second, different data-dependent reference addresses the mirror case
#: without giving up width sensitivity; a nonzero jitter addresses it by
#: trading one for the other.
DEFAULT_REFERENCE_JITTER = 0.0


def _fold_items_into_rows(values: np.ndarray, n_keys: int) -> np.ndarray:
    """Joint layout -> item pooling. One row per ``(simulation, item)``.

    The joint column order set by :func:`_joint_row` is blocked by parameter
    and then by item -- for 2PL, ``[log a_1 .. log a_I, b_1 .. b_I]`` -- so
    the ``k``-th block is ``values[..., k*I:(k+1)*I]`` and item ``j`` is
    column ``j`` within every block. Pairing is positional in both
    directions, which is why *n_keys* is passed rather than inferred.

    Accepts ``(n_sims, n_keys*I)`` truths or references and returns
    ``(n_sims*I, n_keys)``; accepts ``(n_sims, n_draws, n_keys*I)`` draws and
    returns ``(n_sims*I, n_draws, n_keys)``.
    """
    n_cols = values.shape[-1]
    if n_cols % n_keys:
        raise ValueError(
            f"{n_cols} joint columns is not divisible by {n_keys} parameter "
            "key(s); the column order must be blocked by parameter."
        )
    n_items = n_cols // n_keys
    blocks = [values[..., k * n_items : (k + 1) * n_items] for k in range(n_keys)]
    stacked = np.stack(blocks, axis=-1)
    if values.ndim == 2:  # (S, I, K) -> (S*I, K)
        return stacked.reshape(-1, n_keys)
    stacked = np.transpose(stacked, (0, 2, 1, 3))  # (S, I, D, K)
    return stacked.reshape(-1, stacked.shape[2], n_keys)


def _check_item_fold_is_valid(inputs: Any, param_keys: Sequence[str]) -> None:
    """Refuse a joint layout the item fold cannot address positionally.

    :func:`_fold_items_into_rows` slices the joint row into ``len(param_keys)``
    equal blocks and reads column ``j`` of each as item ``j``. Two things have
    to hold for that to mean what it says, and neither is checked by the
    arithmetic -- a wrong layout that happens to divide evenly produces a
    plausible number rather than an error:

    - **The key order has to be the one the row was built in.** *param_keys*
      is a caller argument; ``inputs.param_keys`` is what :func:`_joint_row`
      actually concatenated. A reordered list (``["b", "a"]``) gives matching
      shapes while pairing every reference coordinate with the wrong
      posterior parameter, and a subset (``["b"]``) silently rescales the
      item count.
    - **Every key has to contribute one column per item.** True for scalar
      1PL/2PL parameters, false for the polytomous layout ``["a", "delta"]``
      that :func:`make_irt_hooks` resolves for GPCM, where ``delta`` carries
      ``n_items * n_categories`` columns. The fold would then cut through the
      middle of the real blocks.

    Raising here routes through :func:`_joint_row`'s guard, which invalidates
    the metric for the whole trial rather than letting one corner report a
    number the others do not mean.
    """
    expected = tuple(param_keys)
    actual = tuple(getattr(inputs, "param_keys", ()) or ())
    if not actual:
        raise ValueError(
            "Item-pooled TARP cannot verify the joint column order: the "
            "metric inputs carry no `param_keys`. Skipping the check here "
            "would pass exactly when the evidence for the fold being valid "
            "is missing."
        )
    if actual != expected:
        raise ValueError(
            f"Item-pooled TARP was built for param_keys {list(expected)} but "
            f"the joint row was assembled in the order {list(actual)}. The "
            "fold and the reference provider both slice positionally, so a "
            "different order or subset scores the wrong parameter against "
            "each coordinate. Pass the same `param_keys` to `make_irt_hooks`."
        )
    sim_batch = getattr(inputs, "sim_batch", None)
    if sim_batch is None or not hasattr(sim_batch, "get"):
        raise ValueError(
            "Item-pooled TARP cannot verify the per-key block widths: the "
            "metric inputs carry no `sim_batch`."
        )
    shapes, widths = {}, {}
    for k in expected:
        block = sim_batch.get(k)
        if block is None:
            raise ValueError(
                f"Item-pooled TARP cannot verify parameter {k!r}: it is in "
                "`param_keys` but absent from `sim_batch`."
            )
        arr = np.asarray(block)
        shapes[k] = arr.shape
        widths[k] = int(np.prod(arr.shape[1:])) if arr.ndim > 1 else 1
    # Positive, not relative. Equal widths alone pass trivially for a single
    # vector-valued key -- `param_keys=["delta"]` on GPCM has nothing to be
    # unequal to -- so each block is required to be `(n_sims, n_items)`
    # itself. A scalar item parameter is 2-D; one carrying a category axis
    # is not.
    vector_valued = {k: v for k, v in shapes.items() if len(v) != 2}
    if vector_valued:
        raise ValueError(
            "Item-pooled TARP needs every parameter key to contribute one "
            f"column per item, but {vector_valued} carries more than an item "
            "axis. A vector-valued parameter (such as GPCM `delta`, which "
            "carries one column per item *and* category) cannot be folded "
            "into `(item, parameter)` rows positionally. Use the joint "
            "metric, or restrict `param_keys` to the scalar item parameters."
        )
    if len(set(widths.values())) > 1:
        raise ValueError(
            "Item-pooled TARP needs every parameter key to cover the same "
            f"items, but the joint blocks have widths {widths}."
        )


def make_tarp_item_metric(
    param_keys: Sequence[str],
    *,
    reference_points: Any,
    resolution: int = 20,
    metric: str = "euclidean",
    standardize: bool = True,
    key: str = "tarp_error_item",
) -> Callable[[Any], dict[str, float]]:
    """TARP on ``(simulation, item)`` rows instead of one row per simulation.

    :func:`_joint_row` pools every item into a single row, so at the 50-item
    corner the joint metric sees one 100-dimensional vector per simulation.
    This scores the same draws at ``d = len(param_keys)`` -- for 2PL the
    2-vector ``[log a_j, b_j]`` -- with ``n_simulations * n_items`` rows.

    Both are TARP; they differ in what a "point" is, and the difference is
    not only one of resolution. Theorem 3 of Lemos et al. (2023) states
    sufficiency for the *joint* posterior: TARP on a joint row can detect
    dependence between coordinates that per-coordinate checks cannot.
    Folding to ``(simulation, item)`` rows gives that up for the item axis
    -- it is TARP on each item's marginal, blind to cross-item dependence in
    exactly the way Modrák et al. (2025) argue marginal checks are -- while
    keeping it *within* the ``len(param_keys)`` coordinates of one item. It
    buys resolution at the unidentified corners by narrowing what is being
    certified. Use it alongside the joint metric, not in place of it.

    Which is the better *objective* is an open question. When the posterior
    width is scaled away from the calibrated one, the joint metric can
    saturate at its ceiling while item pooling stays graded. A censored
    metric still ranks the calibrated point first; it cannot say how far
    from it anything else is. A TARP magnitude is not comparable across
    ``n_simulations``, ``n_draws`` or ``resolution``.

    Registering both under different keys is what makes them comparable
    within one study rather than across two.

    Parameters
    ----------
    param_keys : sequence of str
        Column order of the joint parameter block, exactly as passed to
        :func:`make_irt_hooks`. Only its *length* is used here -- it fixes
        how many columns per item the joint block carries -- but it is taken
        as the sequence so a caller cannot pass the two inconsistently.
    reference_points : callable or sequence of arrays
        Resolved per condition in the joint layout and folded to item rows
        alongside the draws, so a provider written for the joint metric works
        here unchanged. A single shared ``(n_sims, n_params)`` array is
        rejected upstream, for the reason given in
        ``bayesflow_hpo.validation.tarp._references_for_condition``.
    resolution, metric, standardize
        Passed through to
        :func:`~bayesflow_hpo.validation.tarp.compute_tarp_coverage`.
    key : str
        Result key. Distinct from ``"tarp_error"`` by default: reporting a
        different statistic under the same name would make the two
        indistinguishable in a results file.

    Returns
    -------
    callable
        ``(JointMetricInputs) -> {key: float}``.

    Warnings
    --------
    The ``n_sims * n_items`` coverage fractions are **not independent** --
    rows from one simulation share that simulation's person abilities -- and
    this does reach ``tarp_error``.

    Clustering does not *bias* the ECP estimate: each ``f_ij`` is marginally
    uniform under a correct posterior, so ``ECP_hat(l)`` is unbiased at every
    level. It inflates the estimate's variance. But ``tarp_error`` is
    ``median_l |ECP_hat(l) - l|``, a non-linear functional of that estimate,
    and under a calibrated posterior ``ECP_hat(l) - l`` is noise about zero,
    so the median absolute deviation is proportional to its spread. A noisier
    ECP therefore reports a *larger* ``tarp_error`` at the same true
    calibration. The floor is a variance floor, not only an SE.

    The practical consequence is that ``n_sims * n_items`` rows do **not**
    buy ``sqrt(n_items)`` precision over ``n_sims`` rows; the effective
    sample size lies somewhere between the two and depends on how strongly
    the shared abilities couple the items. Size a comparison from the
    spread of ``tarp_error`` measured at a calibrated posterior rather than
    from the row count. Anything computing an explicit SE over the fractions needs a
    cluster-aware bootstrap.

    References
    ----------
    Lemos, P., Coogan, A., Hezaveh, Y., & Perreault-Levasseur, L. (2023).
    Sampling-based accuracy testing of posterior estimators for general
    inference. *Proceedings of the 40th International Conference on Machine
    Learning*, 202, 19256-19273.

    Modrák, M., Moon, A. H., Kim, S., Bürkner, P., Huurre, N.,
    Faltejsková, K., Gelman, A., & Vehtari, A. (2025). Simulation-based
    calibration checking for Bayesian computation: The choice of test
    quantities shapes sensitivity. *Bayesian Analysis, 20*(2), 461-488.
    """
    from bayesflow_hpo.validation.tarp import (
        _references_for_condition,
        compute_tarp_coverage,
    )

    n_keys = len(param_keys)
    if n_keys < 1:
        raise ValueError("param_keys must not be empty.")
    if resolution < 1:
        raise ValueError(f"resolution must be at least 1, got {resolution}.")

    def _item_tarp(inputs: Any) -> dict[str, float]:
        _check_item_fold_is_valid(inputs, param_keys)
        refs = _references_for_condition(reference_points, inputs)
        result = compute_tarp_coverage(
            _fold_items_into_rows(np.asarray(inputs.draws, dtype=float), n_keys),
            _fold_items_into_rows(np.asarray(inputs.true_values, dtype=float), n_keys),
            reference_points=_fold_items_into_rows(
                np.asarray(refs, dtype=float), n_keys
            ),
            resolution=resolution,
            metric=metric,
            # No `seed`: `compute_tarp_coverage` consumes it only in its
            # `reference_points is None` branch, and `reference_points` is
            # required here. `make_tarp_joint_metric` does pass one, because
            # its own `reference_points` defaults to `None` and it can
            # therefore reach that branch; copying the argument across would
            # be a knob with no effect.
        )
        return {key: float(result["tarp_error"])}

    return _item_tarp


#: Abilities at which :func:`make_tcc_bias_metric` evaluates the test
#: characteristic curve by default.
DEFAULT_TCC_THETAS = (-1.0, 0.0, 1.0)


def make_tcc_bias_metric(
    thetas: Sequence[float] = DEFAULT_TCC_THETAS,
    *,
    key: str | None = None,
    model: Literal["2pl", "1pl"] = "2pl",
    scale: Literal["z", "tcc"] = "z",
) -> Callable[[Any], dict[str, float]]:
    """Bias of the test characteristic curve, as a joint metric.

    The provenance of each part differs, so they are stated separately.

    **TCC definition (literature).** The test characteristic curve is the
    sum of the item response functions, ``TCC(theta) = sum_i P_i(theta)``
    with the 2PL ``P_i(theta) = logistic(a_i (theta - b_i))`` (Birnbaum,
    1968).

    **Checking a function of all items (literature).** Simulation-based
    checks apply to any scalar function of the parameters, not only to the
    parameters themselves (Cook et al., 2006, as summarized in Talts et al.,
    2018, Sec. 3). Modrák et al. (2025) show that such derived test
    quantities detect problems the individual parameters miss, including
    incorrect posterior correlations (Secs. 4 and 6.1). The TCC is such a
    derived quantity.

    **The z statistic and its null property.** Per simulation and
    ``theta``, ``z = (mean_s TCC_s - TCC_true) / sd_s TCC_s``: the posterior
    z-score of Schad et al. (2021, Eq. 4), applied to the TCC, whose sign
    gives the direction of a bias. The premise is the self-consistency of the
    Bayesian joint distribution (Talts et al., 2018, Sec. 2): when
    ``(params_true, y)`` is drawn from the joint, ``params_true`` given
    ``y`` is distributed as the posterior ``p(. | y)``. The steps from it
    are this package's. Computed from the same draws, the sample mean and
    sample sd are dependent for a skewed TCC posterior, so ``E[z]`` is not
    zero even for an exact posterior and more simulations do not remove it.
    The draws of each simulation are therefore split into two disjoint
    halves (even and odd draw indices): the numerator uses the mean of half
    A and the denominator the sd of half B. Given ``y`` the halves are
    independent, so ``E[z | y] = E[mean_A - TCC_true | y] * E[1 / sd_B | y]``
    and the first factor is zero when the draws come from the posterior
    (its mean is ``E[TCC_true | y]``). By total expectation ``E[z] = 0``.
    A bias in the posterior makes the mean of ``z`` differ from zero. Needs
    at least 4 posterior draws.

    **Noise floor (measured, not derived).** The floor was measured by
    simulation on calibrated synthetic posteriors (20 items, default
    ``thetas``): the metric averaged 0.25 at ``n = 25`` simulations and
    0.13 at ``n = 100``, and the worst of 4 corners 0.38 and 0.20. A
    bias must clear that floor to be seen; size ``sims_per_corner`` from it.

    **Aggregation (a design choice of this package).** The metric is
    ``max_theta |mean_sims z|``: mean over simulations, absolute value, max
    over ``thetas``. Lower is better. Nothing source-backs this particular
    choice; it is one reasonable summary of the per-theta z means.

    A shared per-item shift adds up in the TCC while a per-item check sees
    only the per-item shift, which is the failure the marginal objectives
    can miss.

    **The TCC-unit variant (``scale="tcc"``, key ``tcc_bias``).** Per
    simulation and ``theta``, ``d = (mean_s TCC_s - TCC_true) / n_items``,
    with the posterior mean taken over all draws. The metric is
    ``max_theta |mean_sims d|``. It has no posterior-spread denominator.
    The reason: a posterior that samples the items independently (flow
    matching over the item vector) understates the spread of the *sum* of
    correlated item parameters, which makes the TCC posterior too narrow and
    inflates ``z`` for the same bias in the mean, while an autoregressive
    posterior keeps the item dependence. ``z`` then scores the two families
    differently for the same error in the TCC itself; ``d`` scores the error
    itself. Dividing by ``n_items`` expresses the TCC as the expected
    proportion correct, so corners with different item counts are
    comparable; ``|d| <= 1``. The TCC definition (Birnbaum, 1968) and the use
    of a derived test quantity (Modrák et al., 2025) are source-backed. The
    normalization by ``n_items``, the mean-over-simulations bias and the
    max over ``thetas`` are choices of this package. Since ``d`` is not
    studentized, ``E[d]`` is zero for a calibrated posterior only in the
    sense that ``E[TCC_true | y]`` equals the posterior mean, so no
    half-splitting is needed.

    The ``tcc_bias`` noise floor was measured the same way (calibrated
    synthetic posteriors with a per-coordinate posterior sd of 0.3, 20 items,
    default ``thetas``; mean over repeated draws of the metric, "worst of 4"
    the mean of the max over 4 independent corners). Unlike ``z`` it is not
    scale-free: it scales with the posterior spread of the TCC, so a wider
    posterior raises it.

    ========  ==========  ==============
    sims      mean        worst of 4
    ========  ==========  ==============
    25        0.0048      0.0073
    100       0.0023      0.0036
    600       0.0010      0.0015
    ========  ==========  ==============

    A true-item "oracle" with near-zero posterior spread is not this floor;
    it is not a calibrated posterior.

    Parameters
    ----------
    thetas : sequence of float
        Abilities at which the TCC is evaluated.
    key : str, optional
        Result key; defaults to ``"tcc_z_bias"`` for ``scale="z"`` and
        ``"tcc_bias"`` for ``scale="tcc"``. Any other key is registered as
        lower-is-better (when ``bayesflow_hpo`` is installed) so it can be
        used as an objective, with ``worst_raw=inf`` for ``"z"`` and ``1.0``
        for ``"tcc"``.
    model : {"2pl", "1pl"}
        ``"2pl"`` requires ``param_keys`` to contain both ``a`` (log scale)
        and ``b``, in either order. ``"1pl"`` requires ``("b",)`` and takes
        ``a = 1``. A mismatch raises ``ValueError`` naming the keys, rather
        than silently using the other model's branch.
    scale : {"z", "tcc"}
        ``"z"`` is the posterior z-score metric; ``"tcc"`` the TCC-unit bias
        per item described above.

    Returns
    -------
    callable
        ``(JointMetricInputs) -> {key: float}``. Reads the joint layout of
        :func:`_joint_row`.

    References
    ----------
    Birnbaum, A. (1968). Some latent trait models and their use in inferring
    an examinee's ability. In F. M. Lord & M. R. Novick (Eds.), *Statistical
    theories of mental test scores* (pp. 395-479). Addison-Wesley.

    Modrák, M., Moon, A. H., Kim, S., Bürkner, P., Huurre, N.,
    Faltejsková, K., Gelman, A., & Vehtari, A. (2025). Simulation-based
    calibration checking for Bayesian computation: The choice of test
    quantities shapes sensitivity. *Bayesian Analysis, 20*(2), 461-488.
    https://doi.org/10.1214/23-BA1404

    Schad, D. J., Betancourt, M., & Vasishth, S. (2021). Toward a principled
    Bayesian workflow in cognitive science. *Psychological Methods, 26*(1),
    103-126. https://doi.org/10.1037/met0000275 (read in the arXiv preprint,
    https://arxiv.org/abs/1904.12765v3)

    Talts, S., Betancourt, M., Simpson, D., Vehtari, A., & Gelman, A.
    (2018). *Validating Bayesian inference algorithms with simulation-based
    calibration*. arXiv. https://arxiv.org/abs/1804.06788
    """
    if model not in ("2pl", "1pl"):
        raise ValueError(f"model must be '2pl' or '1pl', got {model!r}.")
    theta = np.asarray(thetas, dtype=float)
    if theta.ndim != 1 or theta.size == 0:
        raise ValueError("thetas must be a non-empty sequence of numbers.")
    if scale not in ("z", "tcc"):
        raise ValueError(f"scale must be 'z' or 'tcc', got {scale!r}.")
    default_key = "tcc_z_bias" if scale == "z" else "tcc_bias"
    if key is None:
        key = default_key
    elif key != default_key and _BAYESFLOW_HPO_IMPORT_ERROR is None:
        register_metric_direction(
            key,
            higher_is_better=False,
            worst_raw=math.inf if scale == "z" else 1.0,
        )

    def _tcc(log_a: np.ndarray | None, b: np.ndarray) -> np.ndarray:
        # (..., I) -> (..., n_theta)
        a = 1.0 if log_a is None else np.exp(log_a)[..., None]
        logits = a * (theta - b[..., None])
        return (1.0 / (1.0 + np.exp(-logits))).sum(axis=-2)

    def _tcc_bias(inputs: Any) -> dict[str, float]:
        keys = tuple(inputs.param_keys)
        draws = np.asarray(inputs.draws, dtype=float)
        truth = np.asarray(inputs.true_values, dtype=float)
        if model == "2pl":
            if sorted(keys) != ["a", "b"]:
                raise ValueError(
                    f"{key} with model='2pl' needs param_keys containing both "
                    f"'a' and 'b', got {keys}."
                )
            # Blocked by parameter in `keys` order (see `_joint_row`).
            n_items = draws.shape[-1] // 2
            ia, ib = (
                slice(i * n_items, (i + 1) * n_items) for i in map(keys.index, "ab")
            )
            n_items_total = n_items
            tcc_draws = _tcc(draws[..., ia], draws[..., ib])
            tcc_true = _tcc(truth[..., ia], truth[..., ib])
        else:
            if keys != ("b",):
                raise ValueError(
                    f"{key} with model='1pl' needs param_keys ('b',), got {keys}."
                )
            n_items_total = draws.shape[-1]
            tcc_draws = _tcc(None, draws)
            tcc_true = _tcc(None, truth)
        if tcc_draws.shape[1] < 4:
            raise ValueError(f"{key} needs at least 4 posterior draws.")
        if scale == "tcc":
            d = (tcc_draws.mean(axis=1) - tcc_true) / n_items_total
            return {key: float(np.max(np.abs(d.mean(axis=0))))}
        # Disjoint halves: mean from one, sd from the other (see docstring).
        sd = tcc_draws[:, 1::2].std(axis=1, ddof=1)
        if np.any(sd <= 0):
            raise ValueError(f"{key}: a posterior TCC has zero spread.")
        z = (tcc_draws[:, 0::2].mean(axis=1) - tcc_true) / sd
        return {key: float(np.max(np.abs(z.mean(axis=0))))}

    return _tcc_bias


def make_classical_reference_provider(
    param_keys: Sequence[str],
    *,
    responses_key: str = "responses",
    jitter: float = DEFAULT_REFERENCE_JITTER,
    seed: int = 20260915,
) -> Callable[[Any], np.ndarray]:
    """Build TARP reference points from classical item statistics.

    ``bayesflow_hpo`` deliberately computes no reference points of its own:
    it accepts an array or a provider callable, because the reference is the
    part of TARP that has to know what the data *is*. This is the IRT
    provider, and the reason it lives in this package.

    Why it matters: Lemos et al. (2023, Sec. 4.3) show TARP with an
    ``x``-independent reference is blind to ``p_hat(theta|x) = p(theta)``
    -- a posterior that ignores its data. A reference that depends on the
    data, such as these classical coordinates, is not.

    Parameters
    ----------
    param_keys : sequence of str
        Column order of the joint parameter block, as the validation closure
        assembles it. Every key must appear in
        :data:`_CLASSICAL_REFERENCE_COORDS`; ``"delta"`` (PCM/GPCM step
        difficulties) does not, and raises here.
    responses_key : str, default "responses"
        Key of the response matrix in the condition batch.
    jitter : float, default :data:`DEFAULT_REFERENCE_JITTER`
        Width of the uniform noise added to the raw reference coordinates,
        as ``jitter * U(0, 1)``. **Zero by default.**

        A nonzero value is the construction of Lemos et al. (2023,
        Sec. 4.3), applied unchanged. It is also not scale-free: TARP
        divides each reference column by the *truth's* sd, and the
        point-biserial's spread is much smaller than the difficulty
        deviate's, so one additive width lands very differently in the two
        blocks and the balance moves across the validation grid. See
        :data:`DEFAULT_REFERENCE_JITTER`. That is
        the reason the default is 0 rather than a reason to reshape the
        noise: a centred, per-block-scaled jitter would be a noise law no
        source here describes.
    seed : int
        Base seed for the jitter. The per-condition stream is derived from
        this and the condition index, so corners do not share noise and a
        rerun reproduces it. A fixed value rather than ``None``: as an HPO
        objective, two identical trials must not score differently.

    Returns
    -------
    callable
        ``(JointMetricInputs) -> (n_simulations, n_parameters)``, suitable as
        the ``reference_points`` argument of
        :func:`bayesflow_hpo.validation.tarp.make_tarp_joint_metric`.

    Raises
    ------
    ValueError
        If any key in *param_keys* has no classical reference coordinate.
        Raised at construction, not per condition: a polytomous model is a
        scoping decision the caller can fix in one line, and discovering it
        inside the condition loop would cost a trained trial first.

    Warnings
    --------
    The provider reads only ``sim_batch[responses_key]``. It never touches
    ``inputs.approximator`` or ``inputs.draws``, and must not: a reference
    derived from the estimator under test reintroduces the blind spot the
    metric exists to detect, is reported under the key that claims teeth,
    and nothing downstream can detect it.

    See Also
    --------
    bayesflow_irt.sbc.classical_item_statistics : The indices themselves,
        and why they are left unconverted.

    References
    ----------
    Lemos, P., Coogan, A., Hezaveh, Y., & Perreault-Levasseur, L. (2023).
    Sampling-based accuracy testing of posterior estimators for general
    inference. *Proceedings of the 40th International Conference on Machine
    Learning*, 202, 19256-19273.
    """
    from .sbc import _require_classical_coords, classical_reference_points

    _require_classical_coords(param_keys)  # at construction (see Raises)

    def provider(inputs: Any) -> np.ndarray:
        # Seeded per condition rather than once: a single stream reused
        # across corners correlates their noise, and `cond_id` is in the
        # upstream contract precisely so a metric need not do that.
        rng = np.random.default_rng([seed, inputs.cond_id]) if jitter else None
        return classical_reference_points(
            inputs.sim_batch[responses_key], param_keys, jitter=jitter, rng=rng
        )

    return provider


def make_irt_hooks(
    simulator: Any,
    adapter: Any,
    *,
    corners: list[tuple[int, int]],
    param_keys: list[str] | None = None,
    metrics: list[str] | None = None,
    inference_space: Any | None = None,
    summary_api: str = "matrix",
    summary_space: BaseSearchSpace | None = None,
    training_space: TrainingSpace | None = None,
    sims_per_corner: SimsPerCorner = _DEFAULT_SIMS_PER_CORNER,
    seed: int = _DEFAULT_SEED,
    aggregate: Literal["worst", "mean", "geometric"] | Mapping[str, str] = "worst",
    memory_config: dict[str, Any] | None = None,
    joint_metrics: Mapping[str, JointMetricFn] | None = None,
    flow_sample_kwargs: Mapping[str, Any] | None = None,
    validation_chunk_size: int | Literal["auto"] = "auto",
    run_pipeline_check: bool = False,
    item_posterior: str | Sequence[str] = "autoregressive_coupling_flow",
) -> IRTHooks:
    """Create HPO hooks for an IRT model.

    Returns an :class:`IRTHooks` NamedTuple containing the search space,
    validation dataset, and three hook callables ready to pass to
    :func:`bayesflow_hpo.optimize`.

    Parameters
    ----------
    simulator : bayesflow_irt simulator
        Training simulator (``make_1pl_simulator``,
        ``make_2pl_simulator``, or ``make_gpcm_simulator``).
    adapter : bayesflow.adapters.Adapter
        Adapter matching *simulator*.
    corners : list of (int, int)
        Edge-condition corners as ``(n_items, n_persons)`` pairs.
    param_keys : list of str or None
        Parameter names to validate.  Auto-detected from *simulator*
        if ``None``: ``["a", "delta"]`` for GPCM,
        ``["a", "b"]`` for 2PL, ``["b"]`` for 1PL.  For PCM
        simulators, pass ``param_keys=["delta"]`` explicitly.
    metrics : list of str or None
        Metric names for ``resolve_metrics()``.  Defaults to
        ``["calibration_error", "nrmse"]``. Names registered upstream as
        *joint* (``tarp_error``, ``tarp_error_random``) are routed to the
        joint dispatch instead of the per-parameter one.
    joint_metrics : mapping of str to JointMetricFn or None
        Joint metrics as ``{name: fn}``, merged over anything resolved from
        *metrics* and, like the upstream pipeline, merged
        **unconditionally** -- a metric given here runs whether or not its
        name appears in *metrics*, and is aggregated across corners either
        way.

        This is how ``tarp_error`` becomes reachable, since it cannot run at
        its registered default (it has no reference points)::

            from bayesflow_hpo.validation.tarp import make_tarp_joint_metric
            from bayesflow_irt.hpo import (
                make_classical_reference_provider, make_irt_hooks,
            )

            param_keys = ["a", "b"]
            hooks = make_irt_hooks(
                simulator, adapter, corners=[(5, 50), (10, 100)],
                param_keys=param_keys,
                joint_metrics={
                    "tarp_error": make_tarp_joint_metric(
                        make_classical_reference_provider(param_keys),
                        reference_id="irt-classical-jitter0",
                    )
                },
            )

        A joint metric receives draws already shaped
        ``(n_sims, n_samples, n_params)``, with items blocked inside each
        parameter and ``a`` on the **log** scale -- see
        :data:`_JOINT_PARAM_TRANSFORMS` and :func:`_joint_row`.

        .. warning::
            **These hooks have no resume guard.** ``reference_id`` and the
            rest of a joint metric's ``joint_metric_settings`` are pinned by
            ``bayesflow_hpo`` only on its own validation pipeline;
            ``_planned_joint_settings`` returns an empty mapping whenever a
            custom ``validate_fn`` is supplied, which is exactly what this
            factory produces. Changing the reference provider, the
            ``reference_id``, the jitter or TARP's ``resolution`` between
            runs of one study therefore mixes incomparable scores with
            nothing to stop it. Start a new study when any of those change.

        Nothing joint runs unless asked for, so a study that does not ask
        scores exactly as without joint metrics,
        and leaves the per-corner cost with the caller who wanted it.

        Results are reported under the key the callable **emits**, which is
        not necessarily the key it is registered under here. A joint metric
        that fails, returns a non-mapping, or returns NaN on *any* corner is
        invalidated for the whole trial and its keys are omitted entirely,
        rather than averaged over the corners that worked — the objective
        then substitutes its registered worst case once, instead of reading
        a flattering partial mean. This mirrors
        ``bayesflow_hpo.validation.pipeline``'s joint guard deliberately.

        Under the default ``aggregate="worst"``, a joint metric emitting a
        key with no registered direction raises when its per-corner values
        are reduced -- at the end of the first trial, not at construction,
        because the direction is only consulted once there is something to
        reduce. ``tarp_error`` and ``tarp_error_random`` register theirs
        upstream, so this affects custom metrics only; either register a
        direction or pass ``aggregate="mean"``.
    inference_space : BaseSearchSpace or None
        Inference network search space.  Defaults to
        ``FlowMatchingSpace()``, or to
        :class:`AutoregressiveCouplingFlowSpace` when ``item_posterior`` is
        ``"autoregressive_coupling_flow"``, or to
        :class:`ItemPosteriorChoiceSpace` over the listed families when it
        is a list of two or more.
    summary_api : str
        Summary search-space selector, resolved through
        ``bayesflow_hpo``'s shared search-space registry (this module
        registers ``"irt_matrix"``/``"matrix"`` (default) at import time
        via ``register_custom_summary_network``). Ignored when
        ``summary_space`` is provided.
    summary_space : BaseSearchSpace or None
        Explicit summary search-space override. If provided, this object
        is used directly and ``summary_api`` is ignored.
    training_space : TrainingSpace or None
        Training search-space override. Defaults to ``TrainingSpace()``,
        which searches ``initial_lr`` over ``[1e-4, 1e-2]`` (log) and
        ``batch_size`` over ``[32, 256]`` in steps of 32.

        The reason to pass one is to **pin** a dimension rather than to
        rebound it: an ``IntDimension`` carrying ``constant=`` is injected
        without being suggested, so the sampler never sees it at all::

            from bayesflow_hpo.search_spaces.base import IntDimension

            training_space = TrainingSpace(
                batch_size=IntDimension("batch_size", constant=192)
            )

        Pinning ``batch_size`` makes sense when its effect saturates:
        if the best achievable objectives are flat above some value,
        sampling it further only spends trials.

        Note this is not a cost-free reduction in either direction, because
        ``epochs`` and ``num_batches`` are fixed under
        ``training_mode="fixed_budget"``: simulations per trial are
        ``batch_size * epochs * num_batches``, so the training budget scales
        linearly in whatever value is pinned.

        Do not use this to couple ``initial_lr`` to ``batch_size`` via
        ``lr_reference_batch_size``. It would not reduce the search --
        ``batch_size`` stays a sampled dimension and only ``initial_lr``
        becomes derived.
    sims_per_corner : int or sequence of int or mapping
        Simulations per validation corner. An ``int`` applies to every
        corner; a sequence is positional against *corners*; a mapping is
        keyed by ``(n_items, n_persons)``. See
        :func:`make_irt_validation_dataset` for why the corners deserve
        unequal budgets and what an unequal budget costs.

        **An unequal budget makes** ``"<metric>/mean"`` **an unweighted
        mean over estimates of unequal precision.** That reduction is
        deliberately left alone rather than silently turned into a
        weighted one: ``"<metric>/mean"`` keeps a fixed definition. Passing a
        heterogeneous budget together with a reduction that averages
        therefore logs a warning -- both for ``aggregate="mean"`` and for a
        mapping that selects
        ``"mean"`` on any metric, since the per-metric form averages the
        same unequally-precise corners on the metrics it names.
        ``aggregate="worst"`` -- the default -- is unaffected, because a
        worst case selects a corner rather than averaging over them.
    seed : int
        Seed for validation dataset generation.
    aggregate : {"worst", "mean", "geometric"} or mapping
        Which reduction Optuna **optimizes**. Default ``"worst"``.

        A mapping selects the reduction per metric, so one search can hold
        different axes to different standards::

            aggregate={"nrmse": "geometric"}

        Metrics not named keep ``"worst"`` -- NOT ``bayesflow_hpo``'s
        ``"mean"`` default. Naming one metric must not quietly move every
        other one onto the reduction this factory's default exists to avoid.

        Reduction happens per **emitted column**, so a multi-output metric
        may be named either way. ``coverage`` emits ``coverage_90`` and four
        siblings; ``{"coverage": "mean"}`` sets the reduction for all five,
        and ``{"coverage_90": "geometric"}`` overrides that one column. An
        exact column name wins over its producer's entry.

        For a **marginal** metric the reduction runs over the **parameter
        type x corner grid**, not over corners alone. This factory builds one
        row per ``(corner, param_key)`` and never collapses the parameter
        axis, so ``"worst"`` already reports the worst cell of that grid --
        unlike ``bayesflow_hpo``'s own pipeline, which arithmetic-means over
        parameter types before reducing conditions. A search running these
        hooks is not exposed to that dilution.

        A **joint** metric has no parameter axis to keep: it scores the
        parameter vector as a whole, so it contributes one row per corner and
        its reduction is over corners alone. With two parameters and four
        corners, ``calibration_error`` therefore reduces over eight cells
        while ``tarp_error`` reduces over four. Both are reduced the same
        way, over every cell that metric has; they simply do not have the
        same number of them, which matters when comparing how exposed two
        metrics are to a single bad condition.

        ``"geometric"`` exists for the recovery axis. ``worst`` on ``nrmse``
        can select the smallest-data corner, which reports that
        corner's information floor rather than the architecture; a
        geometric mean compares each cell against its own achievable floor.
        Recovery is information-limited and calibration is not, which is why
        the two axes take different reductions rather than one setting.

        A non-positive value has no geometric mean. That is treated as a
        DOMAIN failure of the trial -- the metric becomes NaN and Optuna
        reads a failed trial -- rather than as a configuration error, because
        ``correlation`` and ``contraction`` are legitimately negative for an
        undertrained approximator and one poor trial must not end the study.
        The same distinction ``bayesflow_hpo`` draws between
        ``AggregationConfigError`` and ``AggregationDomainError``.

        .. note::
            This is **this factory's own** reduction, not
            ``bayesflow_hpo``'s. Do not also pass ``aggregate=`` to
            ``optimize()``: these hooks supply a custom ``validate_fn``, and
            ``require_pipeline_aggregate`` refuses a non-default reduction
            alongside one, on the grounds that a custom ``validate_fn``
            returns scores that are already reduced.

        Both reductions are always *reported*, under explicit
        ``"<metric>/mean"`` and ``"<metric>/worst"`` keys. They answer
        different questions and both are cheap, so neither is conditional on
        this setting; ``aggregate`` only decides which one additionally
        occupies the bare ``"<metric>"`` key that the search optimizes.

        ``"mean"`` is the reason this parameter exists. Averaging across
        corners lets opposite failures cancel: a
        model that under-covers at one corner and over-covers at another
        averages to an acceptable-looking value, and the search then selects
        it — the cancellation mechanism described by Modrák et al. (2025) for
        rank-based calibration statistics. Use it only when you specifically
        want typical-corner behaviour and have some other guard on the worst
        corner.

        ``"worst"`` uses the metric's direction from ``bayesflow_hpo``'s
        registry. For a metric whose direction is not
        recorded, ``"<metric>/worst"`` is simply omitted; the call raises only
        if that metric was also meant to be the objective. Guessing the
        direction would report the *best* corner as the worst-case guarantee,
        which is worse than refusing.

        **Which corner won is recorded**, as ``"<metric>/worst_corner"`` (the
        index into *validation_data*'s conditions) plus
        ``"<metric>/worst_n_items"`` and ``"<metric>/worst_n_persons"`` when
        the dataset labels its conditions. When several corners share a
        non-finite worst value (e.g. ``log_gamma`` at ``-inf`` everywhere) no
        corner is identified and the three keys are NaN. This is not
        bookkeeping. ``"worst"``
        does not shift a scale — it **selects which corner is reported** — so
        whenever one corner dominates, the others stop contributing to that
        objective axis and the search ranks on that corner alone while
        appearing to rank on the grid.

        That matters most for metrics with a corner-dependent *noise floor*.
        ``tarp_error`` is ``median_l |ECP_hat(l) - l|``, a median absolute
        deviation of noise about zero under calibration, so it scales with the
        ECP's spread: a noisier corner reports a **larger** value at identical
        true calibration. Comparing two trials' ``"<metric>"`` is only a
        comparison of calibration if they share a worst corner — which these
        keys are what let you check.

        Fewer rows make a corner's estimate noisier, but that alone does not
        say which corner wins. The keys matter for a plainer reason: the
        argmax over corners may be noise when corners score close, so
        check its stability before trusting it. That makes this reduction
        unsuitable as a ranking axis, which consumes the argmax, and sound as
        a threshold, which consumes only the value.

        For a metric whose value is ``-inf`` at every corner, such as
        ``log_gamma``, the argmax is meaningless and the keys are NaN. A
        stable corner label is also not evidence that the metric separates
        trials: check the metric's measurement noise against the gaps
        between the trials it is asked to rank.

        Without them the reduction is one-way and the check is unanswerable
        after the fact, since re-running means repeating the search.

        On a tie the first corner in *corners* order is reported, matching
        ``max``/``min``. For marginal metrics the reduction runs over corner
        **and** parameter, so the reported corner is the one containing the
        worst ``(corner, parameter)`` cell.

        **The per-corner values behind the reduction are recorded too**, as
        ``"<metric>/corner_<i>/mean"`` and ``"<metric>/corner_<i>/worst"``
        for every corner index *i*. Both are reduced over that corner's
        cells — one per parameter for a marginal metric, a single cell for a
        joint metric, in which case the two keys carry the same value.

        This makes ``"<metric>/worst"`` reconstructible exactly, and with it
        the margin by which ``worst_corner`` beat the runner-up: the worst
        over the grid is the worst over the per-corner worsts, whatever the
        corners contribute. It does **not** make ``"<metric>/geometric"``
        reconstructible, and it reconstructs ``"<metric>/mean"`` only while
        every corner contributes the same number of valid cells — the grid
        mean is cell-weighted, these are corner-weighted, and one NaN cell
        separates them. Both of those reductions are stored in full anyway.

        A corner that produced no value gets ``NaN`` rather than a missing
        key, so an analysis can filter on the value rather than on whether
        the key is there. The key set is therefore fixed across the trials
        on which the metric *ran*; a metric that failed on a trial has no
        corner keys there, just as it has no ``"<metric>/worst_corner"``.
        ``"/worst"`` is omitted for a metric of unknown direction, exactly as
        ``"<metric>/worst"`` is.
    memory_config : dict or None
        Memory controls applied to **every** trial, merged over
        :data:`DEFAULT_TRIAL_MEMORY_CONFIG`. Keys:
        ``gradient_checkpointing``, ``person_chunk_size``,
        ``flow_chunk_size``, ``flow_chunk_backward``.

        Held constant across the search on purpose. These knobs trade compute
        for peak memory without changing a model's numerical output, so a
        fixed value leaves architecture comparisons intact — while the
        defaults (dense and unchunked, as ``MatrixIRTSummarySpace.build``
        leaves them) make any large draw OOM on a variable-size simulator.
        A failed trial is scored with the failure penalty, so leaving them off
        makes the search rank memory feasibility as model quality.

        ``fit_with_oom_retry`` in the train hook tightens them further for a
        trial that still does not fit.
    flow_sample_kwargs : Mapping or None
        Per-call integrator override for every posterior draw the validation
        hook takes, e.g. ``{"method": "euler", "steps": 16}``. Forwarded to
        ``EquivariantIRTApproximator.sample``, which hands it to
        ``FlowMatching._inverse``, where it is merged *over* the network's own
        ``integrate_kwargs`` (``self.integrate_kwargs | kwargs``). The model is
        therefore still **built** at whatever fidelity the search space
        sampled; only the draws taken for scoring are cheapened. Nothing is
        rebuilt and no second checkpoint is kept.

        Defaults to ``None`` -- no override, every trial scores at the
        integrator its ``inference_space`` chose.

        Fixed-step ``euler``/16 samples much faster than ``tsit5``/adaptive.

        .. warning::
            That does **not** license this as a default, for two
            reasons. ``nrmse`` is a posterior *mean*,
            while ``calibration_error`` -- the usual search axis here -- is a
            rank statistic over the whole posterior shape, which is where
            integrator error would surface first; and HPO's job is ranking
            bad against good, where an under-trained trial may carry a
            stiffer velocity field that fixed-step Euler handles worse.
            Measure your own metric, on an under-trained architecture too,
            before turning this on for a study.

        .. note::
            This applies to **all** validation the hook performs -- the
            intermediate passes feeding dominance pruning as well as the
            end-of-trial score -- because ``bayesflow_hpo`` calls the same
            ``validate_fn`` from ``validation_callback`` and from ``objective``
            with an identical signature and no flag distinguishing them. There
            is no way for this closure to score the two differently. Full
            fidelity remains what you get everywhere outside the search: plain
            ``sample()``, the ``sbc`` module, and any post-study scoring of the
            winning checkpoint.

            It also does not reach a *joint* metric that draws its own samples
            from the ``approximator`` handed to it in
            :class:`JointMetricInputs`; such a metric owns its own sampling
            call and must take its own override.

        Setting this changes what a trial's score means, so a study run with it
        is not comparable with one run without it. Start a new study.
    validation_chunk_size : int or "auto"
        Datasets per ``sample`` call during validation, through
        :func:`bayesflow_irt.evaluation.sample_posterior`, the driver the
        model card uses. Each chunk goes to the item posterior as one batch
        of rows (``batch_flow_rows=True``), so the GPU sees a few large
        calls instead of one call per 256 rows. ``"auto"`` (default) sizes
        chunks from free device memory with a two-probe measurement; an
        out-of-memory chunk shrinks and is retried.
    run_pipeline_check : bool
        If ``True``, run ``bayesflow_hpo.pipeline.check_pipeline()`` on the
        constructed hooks before returning — a fast (1 epoch, 1 batch)
        dry run of build -> compile -> train -> validate that raises
        ``PipelineError`` immediately on an interface mismatch, instead of
        failing deep into a full Optuna study. Defaults to ``False`` since
        the dry run performs a real training step and posterior sample,
        which is unnecessary overhead when hooks are already known-good
        (e.g. in a test suite calling ``make_irt_hooks`` repeatedly).
    item_posterior : str or sequence of str
        Item-posterior factorization of the built approximators (see
        :class:`~bayesflow_irt.EquivariantIRTApproximator`). Defaults to
        ``"autoregressive_coupling_flow"``; ``"flow_matching"`` is
        deprecated. With ``"autoregressive_coupling_flow"`` the inference
        space must be an :class:`AutoregressiveCouplingFlowSpace` (the
        default then), whose
        draws-summary dimensions are searched alongside the coupling flow's,
        and ``flow_sample_kwargs`` must be unset. A list of two or more
        families searches the factorization as a categorical
        ``item_posterior`` dimension through an
        :class:`ItemPosteriorChoiceSpace` over exactly those families (the
        default then; a one-element list pins its family like the string);
        ``flow_sample_kwargs`` then applies to the flow-matching trials only.

    Returns
    -------
    IRTHooks
    """
    _require_bayesflow_hpo()
    if validation_chunk_size != "auto" and (
        isinstance(validation_chunk_size, bool)
        or not isinstance(validation_chunk_size, int)
        or validation_chunk_size < 1
    ):
        raise ValueError(
            "validation_chunk_size must be a positive int or 'auto'; got "
            f"{validation_chunk_size!r}."
        )

    memory_config = {**DEFAULT_TRIAL_MEMORY_CONFIG, **(memory_config or {})}
    unknown = set(memory_config) - set(DEFAULT_TRIAL_MEMORY_CONFIG)
    if unknown:
        raise ValueError(
            f"memory_config has unknown keys: {sorted(unknown)}. "
            f"Known keys: {sorted(DEFAULT_TRIAL_MEMORY_CONFIG)}."
        )

    # Checked here rather than at the first `sample()` call: a bad value would
    # otherwise raise inside `_validate_fn`, after a trial had already been
    # trained, and `GenericObjective` converts a validation exception into a
    # training-loss fallback -- so the study would silently score every trial
    # on the wrong axis instead of failing.
    if flow_sample_kwargs is None:
        flow_sample_kwargs = {}
    elif not isinstance(flow_sample_kwargs, Mapping):
        raise TypeError(
            "flow_sample_kwargs must be a mapping or None, got "
            f"{type(flow_sample_kwargs).__name__}."
        )
    else:
        flow_sample_kwargs = dict(flow_sample_kwargs)

    # The annotation is not enforced at runtime, and the aggregation closure
    # below falls through to the worst case for anything it does not
    # recognise. Without these checks a typo, or a plausible-but-unsupported
    # value such as "median", silently runs a search the caller did not ask
    # for.
    if isinstance(aggregate, str):
        if aggregate not in _AGGREGATIONS:
            raise ValueError(
                f"aggregate must be one of {_AGGREGATIONS} or a mapping from "
                f"metric name to one of them, got {aggregate!r}."
            )
    elif isinstance(aggregate, Mapping):
        for key, reduction in aggregate.items():
            if reduction not in _AGGREGATIONS:
                raise ValueError(
                    f"aggregate[{key!r}] must be one of {_AGGREGATIONS}, got "
                    f"{reduction!r}."
                )
        # Key validation needs `metrics` resolved and the joint metrics
        # merged, neither of which has happened yet, so it runs further down.
        aggregate = dict(aggregate)
    else:
        raise ValueError(
            "aggregate must be a string or a mapping from metric name to a "
            f"reduction, got {type(aggregate).__name__}."
        )

    # A per-corner budget and a mean over corners are individually fine and
    # jointly questionable: the mean weights every corner equally while their
    # estimates no longer carry equal precision, so the corner with the
    # smallest budget moves the objective as much as the one paid for most.
    # Warned rather than forbidden -- a caller may want exactly that -- and
    # warned rather than silently reweighted, because results recorded with
    # earlier versions use the unweighted definition of `"<metric>/mean"`.
    #
    # The condition is the BUDGET alone, not which reduction `aggregate`
    # selects. Tying the warning to the objective goes wrong two ways:
    #
    # * Testing `aggregate == "mean"` misses the mapping form entirely,
    #   which is the form a caller reaches for when they want the mean on
    #   one axis only.
    # * Testing for "mean" at all misses `"geometric"`, which is an
    #   unweighted average in log space and has exactly the same problem.
    #
    # More basically, `"<metric>/mean"` and `"<metric>/geometric"` are
    # recorded for EVERY metric regardless of `aggregate`, and the
    # surrounding comments actively invite reading them out of a finished
    # study later. So an unequal budget makes averaged numbers that nobody
    # warned about, whatever the objective happens to reduce by. The worst
    # case is unaffected either way, because it selects a corner rather
    # than averaging over them.
    corner_budget = _resolve_sims_per_corner(sims_per_corner, corners)
    if len(set(corner_budget)) > 1:
        logger.warning(
            "unequal sims_per_corner (%s): every '<metric>/mean' and "
            "'<metric>/geometric' recorded by this study is an UNWEIGHTED "
            "average over corners whose estimates differ in precision, "
            "whatever 'aggregate' selects. '<metric>/worst' is unaffected, "
            "because it selects a corner rather than averaging over them. "
            "Give every corner the same budget if that was not intended.",
            corner_budget,
        )

    # --- defaults ---
    if param_keys is None:
        if hasattr(simulator, "prior_a") and hasattr(simulator, "prior_delta"):
            # GPCM: both discrimination and step difficulty
            param_keys = ["a", "delta"]
        elif hasattr(simulator, "prior_a"):
            # 2PL: discrimination and difficulty
            param_keys = ["a", "b"]
        elif hasattr(simulator, "prior_b"):
            # 1PL: difficulty only
            param_keys = ["b"]
        else:
            raise ValueError(
                "Cannot auto-detect param_keys for this simulator "
                "type. Pass param_keys explicitly (e.g., "
                "param_keys=['delta'] for PCM simulators)."
            )

    if metrics is None:
        metrics = list(_DEFAULT_METRICS)

    # Resolve joint metrics HERE, not inside the validation closure: an
    # unknown name, or `tarp_error` requested without a reference provider,
    # must raise while building the hooks. Deferred to the closure it would
    # raise per trial, after a model had already been trained -- and
    # `GenericObjective` converts a validation exception into a
    # training-loss fallback, so the study would burn its whole budget
    # scoring trials on a metric that never ran.
    #
    # Override names are passed in rather than merged after, so a name the
    # caller supplies is not resolved from the registry at all: the
    # registered `tarp_error` is a placeholder that refuses at resolve time.
    #
    # Only names the registry actually claims as joint are handed to
    # `resolve_joint_metrics`. Upstream deliberately raises `KeyError` on
    # any name it does not know, which is right for a caller using both
    # halves of its pipeline -- but here `resolve_metrics` is the other
    # half and is called in the closure, so pre-filtering keeps an unknown
    # name that resolver's business. Nothing is dropped: a typo still
    # raises, from the resolver that owns it.
    joint_metrics = dict(joint_metrics or {})
    joint_names = [n for n in metrics if is_joint_metric(n)]
    joint_metric_fns: dict[str, Any] = {
        **resolve_joint_metrics(joint_names, overridden=joint_metrics.keys()),
        **joint_metrics,
    }
    # A name supplied through `joint_metrics=` is never also a marginal
    # metric, even when it appears in `metrics`. `is_joint_metric` only
    # consults the registry, so an ad hoc joint metric under an unregistered
    # name would stay in the marginal list and `resolve_metrics` would raise
    # `KeyError` for it inside the closure -- contradicting this factory's
    # documented promise that a metric passed here runs whether or not its
    # name appears in `metrics`, and doing so per trial rather than at
    # construction.
    marginal_metrics = [m for m in metrics if m not in joint_metric_fns]

    # Resolve the marginal names NOW and throw the result away. `_validate_fn`
    # resolves them again when it runs, and a name the registry does not know
    # raises `KeyError` there -- inside the objective's catch-all, which
    # converts it into a training-loss fallback. So a single typo in `metrics`
    # costs the entire study budget and reports a fabricated number for every
    # trial, with nothing to show which name was wrong.
    #
    # The joint names were already checked this way by `resolve_joint_metrics`
    # above; this closes the same gap on the marginal side. It is easy to
    # hit: results are recorded under EMITTED column names, and
    # `aggregate` deliberately accepts those, so naming a column such as
    # `mean_cal_error` in `metrics` is the natural mistake -- and upstream's
    # kind validator will not catch it, because a column has no registered
    # kind.
    resolve_metrics(marginal_metrics)

    # `aggregate`'s KEYS are checked here rather than with its values above,
    # because what this call computes is only settled now: `metrics` has been
    # defaulted, and the joint metrics have been merged.
    #
    # A key matching nothing is not inert -- it leaves that metric on the
    # default reduction with nothing to show the setting never took effect,
    # which is the failure `bayesflow_hpo` fixed in its own mapping
    # validator. Matched against the keys metrics EMIT, not the names they
    # are registered under: a multi-output metric such as `coverage` emits
    # `coverage_90` and friends and is never itself a column, and a joint
    # metric is reduced under the key it emits, which for a TARP metric
    # built without a reference provider is `tarp_error_random` whatever key
    # it was registered as.
    #
    # A joint metric's emitted key is NOT derivable here. `output_keys_for`
    # answers from the registry, and a callable passed through
    # `joint_metrics=` emits whatever it emits -- a TARP metric built without
    # a reference provider emits `tarp_error_random` under any registration
    # key. So once any joint metric is present the set of valid names is open
    # and nothing can be rejected on this evidence; `_validate_fn` checks it
    # at the end of the first trial instead, where the emitted keys are known.
    # With no joint metric every name is knowable, and a typo should not cost
    # a trained trial to discover.
    if not isinstance(aggregate, str) and not joint_metric_fns:
        known: set[str] = set()
        for name in marginal_metrics:
            known.add(name)
            known.update(output_keys_for(name))
        unknown = sorted(set(aggregate) - known)
        if unknown:
            raise ValueError(
                f"aggregate names {unknown}, which this call does not "
                f"compute. Available: {sorted(known)}."
            )

    families = normalize_item_posterior(item_posterior)
    searched = len(families) > 1
    item_posterior = None if searched else families[0]
    autoregressive = item_posterior == "autoregressive_coupling_flow"
    if inference_space is None:
        if searched:
            inference_space = ItemPosteriorChoiceSpace(families=families)
        elif autoregressive:
            inference_space = AutoregressiveCouplingFlowSpace()
        else:
            inference_space = FlowMatchingSpace()
    if searched:
        if not isinstance(inference_space, ItemPosteriorChoiceSpace):
            raise TypeError(
                f"item_posterior={list(families)!r} needs an "
                "ItemPosteriorChoiceSpace inference_space, got "
                f"{type(inference_space).__name__}."
            )
        if tuple(inference_space.families) != families:
            raise ValueError(
                f"item_posterior={list(families)!r} does not match the "
                f"inference space's families {list(inference_space.families)!r}."
            )
    elif isinstance(inference_space, ItemPosteriorChoiceSpace):
        raise TypeError(
            "ItemPosteriorChoiceSpace needs item_posterior to list two or "
            f"more families, got {item_posterior!r}."
        )
    elif autoregressive:
        if not isinstance(inference_space, AutoregressiveCouplingFlowSpace):
            # The draws-summary dimensions live on this space; any other
            # space would build the approximator without them.
            raise TypeError(
                "item_posterior='autoregressive_coupling_flow' needs an "
                "AutoregressiveCouplingFlowSpace inference_space, got "
                f"{type(inference_space).__name__}."
            )
        if flow_sample_kwargs:
            raise ValueError(
                "flow_sample_kwargs sets an ODE integrator, but the "
                "autoregressive item posterior samples with a coupling flow, "
                "which takes none."
            )
    elif isinstance(inference_space, AutoregressiveCouplingFlowSpace):
        # Its draws-summary dimensions would be sampled and recorded with
        # every trial while `_build_fn` ignores them.
        raise TypeError(
            "AutoregressiveCouplingFlowSpace needs "
            "item_posterior='autoregressive_coupling_flow', got "
            f"{item_posterior!r}."
        )

    if summary_space is None:
        try:
            summary_space = get_summary_space(summary_api)
        except KeyError as exc:
            raise ValueError(
                "summary_api must be 'matrix' "
                f"or pass summary_space explicitly, got {summary_api!r}."
            ) from exc

    # --- validation dataset ---
    validation_data = make_irt_validation_dataset(
        simulator,
        corners=corners,
        param_keys=param_keys,
        # The already-resolved per-corner list, not the caller's form: the
        # warning above and the dataset must agree on what was resolved.
        sims_per_corner=corner_budget,
        seed=seed,
    )

    # --- search space ---
    if training_space is None:
        training_space = TrainingSpace()

    search_space = CompositeSearchSpace(
        inference_space=inference_space,
        summary_space=summary_space,
        training_space=training_space,
    )

    # --- closures ---

    def _build_fn(hparams: dict[str, Any]) -> EquivariantIRTApproximator:
        summary_net = search_space.summary_space.build(hparams)
        flow = search_space.inference_space.build(hparams)

        # Turn on the package's default memory controls.
        # `MatrixIRTSummarySpace.build` deliberately leaves these at their
        # dense, unchunked defaults, because they are deployment knobs rather
        # than modeling hyperparameters -- they do not change the model's
        # numerical output. That is the right call for the *search space*; it
        # is the wrong default for the *trial*, which then allocates the full
        # (batch, persons, items, hidden) activation grid and OOMs on any
        # large draw.
        #
        # Setting them here rather than in `build` keeps them out of the
        # search -- every trial gets the same value, so nothing about the
        # comparison between architectures changes -- while making a trial
        # trainable at the sizes this simulator actually draws.
        # `fit_with_oom_retry` in `_train_fn` tightens them further if a
        # particular architecture still does not fit.
        set_summary_memory_config(
            summary_net,
            item_chunk_size=None,
            gradient_checkpointing=memory_config["gradient_checkpointing"],
            person_chunk_size=memory_config["person_chunk_size"],
        )
        trial_posterior = hparams["item_posterior"] if searched else item_posterior
        item_kwargs: dict[str, Any] = {"item_posterior": trial_posterior}
        if trial_posterior == "autoregressive_coupling_flow":
            item_kwargs.update(
                AutoregressiveCouplingFlowSpace.draws_summary_kwargs(hparams)
            )
        return EquivariantIRTApproximator(
            adapter=adapter,
            inference_network=flow,
            summary_network=summary_net,
            flow_chunk_size=memory_config["flow_chunk_size"],
            flow_chunk_backward=memory_config["flow_chunk_backward"],
            **item_kwargs,
        )

    def _train_fn(
        approximator: Any,
        sim: Any,
        hparams: dict[str, Any],
        callbacks: list[Any],
    ) -> None:
        # Each trial samples a different summary/flow architecture and draws
        # a different (n_items, n_persons) shape per batch, so PyTorch's CUDA
        # caching allocator sees many distinct tensor sizes across a search --
        # the same variable-shape fragmentation that motivated
        # ``_PeriodicCudaCacheRelease`` in ``fit_with_oom_retry``, but across
        # trials too since the allocator's reserved pool persists for the
        # life of the process. See that callback's docstring.
        if keras.backend.backend() == "torch":
            callbacks = [*callbacks, _PeriodicCudaCacheRelease()]

        # `fit_with_oom_retry`, not a bare `fit`. A trial's memory demand is
        # dominated by activations over the (batch, persons, items, hidden)
        # grid, and the search samples batch size (32-256) independently of
        # the summary width -- so on a variable-size simulator drawing up to
        # `n_persons` a plausible architecture can allocate tens of GB and die
        # part-way through epoch 1.
        #
        # A bare `fit` turns that into a FAILED trial, scored with the failure
        # penalty -- which is indistinguishable from a genuinely bad
        # architecture. The search would then be ranking memory feasibility as
        # model quality, and nothing in the stored study would say so.
        #
        # This is the mechanism MatrixIRTSummarySpace's own docstring
        # prescribes: chunking and gradient checkpointing are deployment knobs
        # to be chosen after a trial's architecture is fixed, not search
        # dimensions.
        #
        # Only the numerically neutral steps of the ladder (item then person
        # chunking) are allowed: `min_batch_size` equal to the sampled batch
        # size stops it before any batch-size cut, and `fit_with_oom_retry`
        # then raises, so the trial FAILS instead of training a configuration
        # the sampler did not propose. A cut is not neutral -- batch_size is a
        # searched dimension, and under a simulation budget a smaller batch at
        # the same num_batches also trains on fewer simulations.
        requested = int(hparams.get("batch_size", 256))
        result = fit_with_oom_retry(
            approximator,
            sim,
            start_batch_size=requested,
            min_batch_size=requested,
            fit_kwargs={
                "epochs": int(hparams["epochs"]),
                "num_batches": int(hparams["num_batches"]),
                "callbacks": callbacks,
            },
        )

        if result.attempts > 1:
            logger.info(
                "Trial fit after %d attempts at the sampled batch_size=%d "
                "(item_chunk_size=%s, person_chunk_size=%s); chunking is "
                "numerically neutral, so the sampled architecture is what "
                "was measured.",
                result.attempts,
                requested,
                result.item_chunk_size,
                result.person_chunk_size,
            )

    # Chunk size per model and corner, chosen on the model's first validation.
    _probed_chunks: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

    def _validate_fn(
        approximator: Any,
        _validation_data_unused: Any,
        n_posterior_samples: int,
    ) -> dict[str, float]:
        """Evaluate per-item posteriors across edge-condition corners.

        .. note::
            The *_validation_data_unused* argument is **ignored**.
            This function uses the captured ``validation_data`` from the
            enclosing :func:`make_irt_hooks` call instead, because
            ``optimize()`` passes its own internally-generated dataset
            which is incompatible with the IRT edge-corner pattern.
        """
        metric_fns = resolve_metrics(marginal_metrics)
        all_rows: list[dict[str, float]] = []
        # Which corner each row came from, parallel to `all_rows`.
        #
        # Without this the reduction below is one-way: the per-corner values
        # are computed, reduced and dropped, so a finished study cannot say
        # WHICH corner produced `{metric}/worst`. That matters because
        # `aggregate="worst"` selects a corner rather than shifting a scale --
        # see the `{metric}/worst_corner` note in the docstring -- and the
        # question is unanswerable after the fact, when re-running means
        # repeating the search.
        row_conditions: list[int] = []
        # Optional: a dataset built outside `make_irt_validation_data` may not
        # label its conditions, in which case the corner index is still
        # emitted and only the size keys are omitted.
        condition_labels = list(
            getattr(validation_data, "condition_labels", None) or []
        )
        n_conditions = len(validation_data.simulations)
        # Per-trial, not per-condition: a joint metric that fails anywhere is
        # invalidated everywhere. See `_joint_row`.
        failed_joint: dict[str, str] = {}
        emitted_keys: dict[str, set[str]] = {}

        for cond_id, corner_batch in enumerate(validation_data.simulations):
            # Passed only when non-empty so the default path calls `sample`
            # with its default arguments.
            # Under the item-posterior search, only flow-matching trials take
            # the ODE override; a coupling flow has no integrator.
            sample_kwargs: dict[str, Any] = {}
            if flow_sample_kwargs and (
                getattr(approximator, "item_posterior", None)
                != "autoregressive_coupling_flow"
            ):
                sample_kwargs["flow_sample_kwargs"] = flow_sample_kwargs
            # The memory probe runs once per corner and model: later
            # validations of the same trial reuse the size it chose (after
            # any out-of-memory shrink).
            sizes = _probed_chunks.setdefault(approximator, {})
            result = sample_posterior(
                approximator,
                corner_batch,
                n_posterior_samples,
                param_keys,
                chunk_size=sizes.get(cond_id, validation_chunk_size),
                batch_flow_rows=True,
                sample_kwargs=sample_kwargs,
            )
            sizes[cond_id] = result.final_chunk_size
            posterior = result.draws

            for key in param_keys:
                true_vals = np.asarray(corner_batch[key])
                draws = np.asarray(posterior[key])

                # Flatten works for both 2D (B, I) and 3D
                # (B, I, K-1) params.  Axis 1 is the sample axis.
                true_flat = true_vals.reshape(-1)
                draws_flat = np.moveaxis(draws, 1, -1).reshape(-1, draws.shape[1])

                row: dict[str, float] = {}
                for fn in metric_fns.values():
                    row.update(fn(draws_flat, true_flat))
                all_rows.append(row)
                row_conditions.append(cond_id)

            # --- joint metrics ---
            #
            # The loop above is marginal twice over: it reduces per parameter
            # AND treats each item as its own simulation. A joint metric
            # needs neither reduction, so it gets its own assembly rather
            # than a reshape of that one. Skipped entirely when nothing asked
            # for it -- the stacking is cheap but not free, and a study
            # optimizing marginals should not pay for it.
            if joint_metric_fns:
                all_rows.append(
                    _joint_row(
                        joint_metric_fns,
                        corner_batch=corner_batch,
                        posterior=posterior,
                        param_keys=param_keys,
                        data_keys=list(validation_data.data_keys),
                        approximator=approximator,
                        cond_id=cond_id,
                        n_conditions=n_conditions,
                        failed_joint=failed_joint,
                        emitted_keys=emitted_keys,
                    )
                )
                row_conditions.append(cond_id)

        # Aggregate across corners and parameters.
        #
        # An unconditional mean would be the exact
        # cancellation mechanism Modrák et al. (2025) describe: a model that
        # under-covers at one corner and over-covers at another averages to
        # an acceptable-looking number, and HPO then selects it. The mean is
        # a statement about the typical corner; a search that is meant to
        # produce a model usable across the whole grid needs a statement
        # about the worst one.
        #
        # `aggregate` is therefore "worst" by default. Do not "simplify" this
        # back to a mean.
        result: dict[str, float] = {}
        # Joint names are appended rather than assumed present in `metrics`:
        # a joint metric supplied through `joint_metrics=` runs whether or
        # not its name appears there (mirroring the upstream pipeline's
        # unconditional merge), and a metric computed but never aggregated
        # would be paid for and then dropped.
        # Keys of a joint metric invalidated this trial. Dropped OUTRIGHT
        # rather than averaged over the corners that worked: the objective
        # substitutes a registered worst case for an absent metric exactly
        # once, whereas a partial mean reaches it as a real, flattering
        # number that never triggers the penalty.
        dropped = _dropped_joint_keys(failed_joint, emitted_keys)
        # Aggregated by EMITTED key, never by the `joint_metrics=` mapping
        # key. The two need not match -- `make_tarp_joint_metric()` built
        # without a reference provider emits `tarp_error_random` whatever
        # key it is registered under -- and aggregating the mapping key
        # would invent an all-NaN metric that was never computed. Optuna
        # reads that NaN as a failed trial, so a naming slip would present
        # as a broken model. What ran is reported under the name it
        # actually produced; what did not run is simply absent, and the
        # objective substitutes its registered worst case once.
        # Marginal metrics are reduced under the keys they EMIT, for the same
        # reason the joint path is. A multi-output metric writes its own
        # columns into every row -- `coverage` writes `coverage_90` and four
        # others, never `coverage` -- so reducing by the configured name
        # would find nothing, report NaN under a name no cell ever produced,
        # and silently discard every value the metric did compute. The NaN
        # is the worse half: the loop would return it so that Optuna reads a
        # failed trial, which is right for a metric that could not be
        # computed and wrong here, where the metric ran on every cell. A
        # study configured with `metrics=["coverage"]` would record a
        # failed-looking trial for its entire budget.
        #
        # `output_keys_for` is the registry's own answer and is the identity
        # for the single-output case, so this changes nothing for the
        # metrics this package actually searches on.
        marginal_keys: list[str] = []
        for name in marginal_metrics:
            for key in output_keys_for(name):
                if key not in marginal_keys:
                    marginal_keys.append(key)
        joint_only = [
            k
            for fn_keys in emitted_keys.values()
            for k in sorted(fn_keys)
            if k not in marginal_keys and k not in dropped
        ]
        seen: set[str] = set()
        joint_only = [k for k in joint_only if not (k in seen or seen.add(k))]
        for name, fn_keys in emitted_keys.items():
            if name not in fn_keys and name not in failed_joint:
                logger.warning(
                    "Joint metric registered as %r emits %s, not %r. Its "
                    "results are reported under the emitted key(s); nothing "
                    "is reported under %r. If %r was meant to be the "
                    "objective, the callable is not the one that produces "
                    "it -- a TARP metric built without a reference provider "
                    "emits 'tarp_error_random', never 'tarp_error'.",
                    name,
                    sorted(fn_keys),
                    name,
                    name,
                    name,
                )
        # The per-corner keys below are SYNTHESIZED into `result` under
        # `{metric}/corner_{i}/mean` and `.../worst`. A joint metric emits
        # whatever its callable returns, so a caller can land inside that
        # namespace: a metric emitting the literal key
        # `"nrmse/corner_0/mean"` is reduced like any other emitted key, and
        # the `result[metric_name] = ...` write below then replaces the
        # summary synthesized for `nrmse` with a number that means something
        # else. Nothing downstream can tell the two apart afterwards, and a
        # study cannot be un-run -- so this is refused rather than renamed
        # or merged.
        #
        # Checked against the keys actually reduced, not just the emitted
        # ones, so a registered metric whose column took the same shape is
        # caught too. `AggregationConfigError` for the same reason as the
        # inert-key check at the end: an emitted key is a property of the
        # callable and identical on every trial, so it has to escape
        # `GenericObjective` rather than become a training-loss fallback on
        # all of them.
        reserved = [
            key
            for key in [*marginal_keys, *joint_only]
            if _RESERVED_CORNER_KEY.search(key)
        ]
        if reserved:
            raise AggregationConfigError(
                f"metric key(s) {reserved} use the '/corner_<i>/mean' or "
                "'/corner_<i>/worst' suffix, which this module synthesizes "
                "for the per-corner values behind each reduction. A metric "
                "emitting one of these would silently overwrite the corner "
                "summary of the metric it names. Rename the emitted key -- "
                "a joint metric is reported under the key it EMITS, so "
                "rename it in the callable rather than at registration."
            )

        for metric_name in [*marginal_keys, *joint_only]:
            if metric_name in dropped:
                continue
            # Kept paired with the corner index through the NaN filter: a
            # dropped row shifts positions, so reducing first and looking up
            # the index afterwards would report the wrong corner.
            pairs = [
                (r.get(metric_name, float("nan")), cid)
                for r, cid in zip(all_rows, row_conditions)
            ]
            valid_pairs = [(v, cid) for v, cid in pairs if not np.isnan(v)]
            valid = [v for v, _ in valid_pairs]

            # The rows the reductions below consume, kept alongside them.
            # `mean` and `worst` are one number each; the cells they came
            # from are discarded, and with them every question about the
            # margin between corners. A `worst_corner` index cannot then be
            # told apart from a coin flip: `log_gamma` can be `-inf` on every
            # corner, and then its recorded corner is an arbitrary
            # tie-break and nothing in the stored keys said so. The
            # values cost four floats per corner and recovering them later
            # costs GPU hours -- or is impossible, once a trial's checkpoint
            # is gone.
            #
            # Two keys per corner rather than one, because a corner is not
            # one cell: the loop above emits a row per `param_keys` entry
            # per corner for a marginal metric, so the grid is parameter x
            # corner and a corner holds `len(param_keys)` cells. Reducing
            # *within* a corner both ways is unconditional for the same
            # reason the grid-wide pair is. For a joint metric, whose corner
            # holds a single cell, the two keys carry that one value twice.
            #
            # What that does and does NOT let a reader rebuild, exactly:
            #
            # * `{metric}/worst` is the worst of the per-corner worsts,
            #   always -- max/min distributes over a partition whatever the
            #   block sizes -- so `worst_corner`'s margin over the runner-up
            #   is now readable. That is the reduction this factory defaults
            #   to.
            # * `{metric}/mean` is the mean of the per-corner means ONLY
            #   while every corner contributes the same number of valid
            #   cells. The grid mean is cell-weighted and these are
            #   corner-weighted, so one NaN cell on one corner separates
            #   them. Equal budgets do not help: `sims_per_corner` changes
            #   the rows a cell is estimated FROM, not how many cells a
            #   corner has.
            # * `{metric}/geometric` is not recoverable from arithmetic
            #   per-corner means at all. It is stored in full below; these
            #   keys are not a substitute for it.
            #
            # Emitted for EVERY corner of a metric that ran, NaN where the
            # corner produced no valid value, so an analysis filters on the
            # value rather than on whether the key is present -- filtering
            # on key presence hides exactly this. "Of a metric
            # that ran" is the real guarantee and not a weaker restatement:
            # a metric in `dropped` skipped this loop body entirely and has
            # no corner keys on that trial, exactly as it has no
            # `{metric}/worst_corner`. A joint metric that raises on
            # non-finite draws does that routinely on undertrained early
            # trials.
            #
            # NaN rather than `None` because these pass through
            # `bayesflow_hpo`'s `round(float(val), 6)`, which raises on
            # `None`. A results file written with `allow_nan=False` after
            # mapping non-finite values to `None` (as `MetadataTracker.save()`
            # does) still reads as JSON `null`.
            by_corner: dict[int, list[float]] = {}
            for value, cid in valid_pairs:
                by_corner.setdefault(cid, []).append(value)
            # Asked once for the metric, not once per corner: the direction
            # is a property of the metric, so an unknown one omits the
            # `/worst` keys identically on every trial -- which is what
            # keeps the key set stable -- and matches `{metric}/worst`
            # being omitted below for that same reason.
            has_direction = _lower_is_better(metric_name.split("/")[0]) is not None
            for cid in range(n_conditions):
                corner_values = by_corner.get(cid, [])
                result[f"{metric_name}/corner_{cid}/mean"] = (
                    float(np.mean(corner_values)) if corner_values else float("nan")
                )
                if has_direction:
                    result[f"{metric_name}/corner_{cid}/worst"] = (
                        _worst(metric_name, corner_values)
                        if corner_values
                        else float("nan")
                    )

            if not valid:
                # Return NaN so Optuna treats this trial as failed rather
                # than selecting it as the best (0.0 would be misleading).
                result[metric_name] = float("nan")
                continue

            # Both reductions are ALWAYS reported, under their own explicit
            # keys. They answer different questions -- "how does this
            # configuration do typically" and "how bad is its worst corner" --
            # and both are cheap once the per-corner rows exist. Making either
            # one conditional on `aggregate` would mean the common setup
            # silently loses a number worth having, and would leave a study's
            # stored trials unable to answer a question its own data contains.
            mean_value = float(np.mean(valid))
            result[f"{metric_name}/mean"] = mean_value
            # Reported alongside the other two whenever it is computable, on
            # the same reasoning: the grid values exist, the reduction is
            # free, and a study that later wants it should not have to re-run
            # the search. Omitted rather than NaN when the values leave its
            # domain, so an absent key means "not applicable here" and never
            # "this trial failed".
            geometric_value = _geometric(valid)
            if not np.isnan(geometric_value):
                result[f"{metric_name}/geometric"] = geometric_value
            worst_value: float | None
            try:
                worst_value = _worst(metric_name, valid)
            except ValueError:
                # Direction unknown: the worst case is not computable, but the
                # mean still is. Only raise if the worst case was asked to be
                # the objective, which happens below.
                worst_value = None
            else:
                result[f"{metric_name}/worst"] = worst_value
                # Which corner that worst case came from. Emitted only
                # alongside a computable worst value, so the two keys are
                # never out of step. On a finite tie the first corner in
                # `corners` order wins, matching `max`/`min`.
                worst_cid = next(cid for v, cid in valid_pairs if v == worst_value)
                tied_corners = {cid for v, cid in valid_pairs if v == worst_value}
                # A non-finite worst shared by several corners identifies no
                # corner: `log_gamma` can be `-inf` on every corner, and the
                # tie-break index then read as
                # a structured measurement. NaN, not `None`, for the
                # same `round(float(val), 6)` reason as the per-corner keys;
                # it still lands in the results file as JSON `null`. A single
                # corner at `-inf` is a genuine worst and is still reported.
                ambiguous = not np.isfinite(worst_value) and len(tied_corners) > 1
                result[f"{metric_name}/worst_corner"] = (
                    float("nan") if ambiguous else float(worst_cid)
                )
                label = (
                    condition_labels[worst_cid]
                    if worst_cid < len(condition_labels)
                    else {}
                )
                for size_key in ("n_items", "n_persons"):
                    if size_key in label:
                        result[f"{metric_name}/worst_{size_key}"] = (
                            float("nan") if ambiguous else float(label[size_key])
                        )

            # `aggregate` chooses only which of the reductions occupies the
            # bare metric key -- the one Optuna optimizes. A mapping resolves
            # per metric, so calibration can keep a worst-case guarantee while
            # recovery is compared on a reduction that does not simply report
            # the smallest-data corner's information floor.
            reduction = _resolve_aggregate(aggregate, metric_name)
            if reduction == "mean":
                result[metric_name] = mean_value
            elif reduction == "geometric":
                # NaN when a value was <= 0: a domain failure for this trial,
                # not a configuration error for the study. See `_geometric`.
                result[metric_name] = geometric_value
            else:
                if worst_value is None and _is_diagnostic(metric_name):
                    # A diagnostic metric cannot be the objective --
                    # `validate_objective_metric_kinds` refuses it upstream --
                    # so an unknown direction here cannot silently become a
                    # penalised objective, and there is nothing to stop the
                    # study for. `coverage` is the case in hand: none of its
                    # five columns has a registered direction, and demanding
                    # one would turn the ordinary `metrics=["coverage"]` into
                    # a study-stopping error, which is the same config broken
                    # a different way. The mean and geometric reductions are
                    # already recorded above; only the bare key is omitted,
                    # and an absent key reads as "not applicable" rather than
                    # as a failed trial.
                    continue
                if worst_value is None:
                    # Also `AggregationConfigError`, and for the same reason
                    # as the inert-key check below: an unknown direction is a
                    # property of the study's settings, identical on every
                    # trial, so it has to escape `GenericObjective` rather
                    # than become a training-loss fallback; a plain
                    # ValueError would be swallowed.
                    raise AggregationConfigError(
                        f"aggregate={aggregate!r} needs the worst case over "
                        f"the parameter x corner grid for metric "
                        f"{metric_name!r}, whose direction "
                        "is unknown. Register it with "
                        "bayesflow_hpo.objectives.register_metric_direction(), "
                        "which this module reads, with a source for which way "
                        "is better. Or give this metric 'mean' "
                        "or 'geometric' and accept that cells cancel."
                    )
                result[metric_name] = worst_value

        # The construction-time check above accepts a joint metric under its
        # registration key OR its emitted key, because a callable passed
        # through `joint_metrics=` does not reveal which it emits until it
        # runs. Now it has. A key that reduced nothing would silently leave
        # its metric on the default reduction, which is exactly what naming
        # it was meant to change, so it fails here -- at the end of the first
        # trial, before the study spends its budget ranking on a setting that
        # never took effect.
        #
        # Every trial would hit this identically, so it is a configuration
        # error, and it is raised as `AggregationConfigError` specifically:
        # `GenericObjective` re-raises that and converts anything else into a
        # training-loss fallback, which would spend the whole budget on
        # fabricated values rather than stopping. It subclasses ValueError,
        # so a caller catching that still catches this.
        if not isinstance(aggregate, str):
            reduced = {m.split("/")[0] for m in result}
            # A producer name counts as used when any column it emits was
            # reduced, because `_resolve_aggregate` honours it for all of
            # them. Without this, the documented `{"coverage": "mean"}`
            # would take effect on all five columns and then be reported as
            # naming nothing -- failing the study on the strength of a
            # setting that did work.
            reduced |= {_producer_of(key) for key in reduced}
            # A joint metric that failed or returned NaN on THIS trial had
            # its keys dropped above, so they reduced nothing -- but that is
            # a property of the trial, not of the study. `_joint_block`
            # raises on non-finite draws, which an undertrained early trial
            # produces routinely, and a failing joint metric is deliberately
            # reported as absent rather than zero. Because
            # `AggregationConfigError` IS re-raised upstream, counting these
            # as inert would let one bad trial abort the whole study, which
            # is the opposite of what a per-trial failure should cost.
            #
            # They are exempted rather than reported, and the naming-slip
            # case this check exists for is unaffected: a key that names a
            # metric which never ran at all is still absent from both sets.
            reduced |= set(dropped)
            reduced |= {_producer_of(key) for key in dropped}
            reduced |= set(failed_joint)
            inert = sorted(set(aggregate) - reduced)
            if inert:
                raise AggregationConfigError(
                    f"aggregate names {inert}, which produced no value this "
                    f"trial. Reduced: {sorted(reduced)}. A joint metric is "
                    "reduced under the key it EMITS -- a TARP metric built "
                    "without a reference provider emits 'tarp_error_random' "
                    "whatever key it is registered under -- so name the "
                    "emitted key here."
                )

        return result

    hooks = IRTHooks(
        search_space=search_space,
        validation_data=validation_data,
        build_fn=_build_fn,
        train_fn=_train_fn,
        validate_fn=_validate_fn,
    )

    if run_pipeline_check:
        # Dry-run build -> compile -> train -> validate (1 epoch, 1 batch) so
        # interface mismatches surface immediately as PipelineError, instead
        # of deep into a full Optuna study. validation_conditions is left at
        # its None default: check_pipeline's own tiny validation dataset is
        # only used as an argument to our validate_fn, which ignores it in
        # favor of the captured edge-corner `validation_data` above (see
        # _validate_fn's docstring) — so the conditions= incompatibility
        # documented on make_irt_validation_dataset never comes into play
        # here.
        # A searched study would otherwise dry-run flow matching only (the mock
        # trial takes the first choice), so each family is checked in turn.
        if isinstance(inference_space, ItemPosteriorChoiceSpace):
            _check_spaces = [
                CompositeSearchSpace(
                    inference_space=replace(inference_space, families=(fam,)),
                    summary_space=summary_space,
                    training_space=training_space,
                )
                for fam in inference_space.families
            ]
        else:
            _check_spaces = [search_space]
        for _check_space in _check_spaces:
            check_pipeline(
                simulator=simulator,
                adapter=adapter,
                search_space=_check_space,
                build_approximator_fn=hooks.build_fn,
                train_fn=hooks.train_fn,
                validate_fn=hooks.validate_fn,
                # Joint metric names included, not just `metrics`. With a custom
                # `validate_fn`, `check_pipeline` requires every name here to
                # appear in the validation output -- and the joint guard omits a
                # metric that failed, so a broken reference provider would
                # otherwise sail through the preflight on the strength of the
                # marginal metrics alone, which is the opposite of what this
                # check is for.
                #
                # A name whose callable emits something else also fails here,
                # deliberately: that is the forgotten-provider misconfiguration,
                # caught before a trial is trained rather than after.
                #
                # REGISTERED DIAGNOSTIC metrics are filtered out. Since
                # bayesflow-hpo v0.5.0 `check_pipeline` runs
                # `validate_objective_metric_kinds` over this argument and
                # refuses a diagnostic outright, and there is no separate
                # "keys that must be present" parameter to use instead. A config
                # RECORDING `correlation` or `contraction` -- which is the only
                # way they may appear at all, since the same upstream rule bars
                # them from `objective_metrics` -- would otherwise fail the
                # preflight it is meant to pass. Unregistered names are NOT
                # filtered: upstream leaves those valid on purpose, because a
                # custom `validate_fn` may emit objective values the built-in
                # registry has never heard of, which is exactly what `log_gamma`
                # and the two TARP metrics are here.
                #
                # What this costs, stated rather than glossed: the preflight no
                # longer asserts that a diagnostic's key reaches the validation
                # output, so a diagnostic with a broken producer surfaces on the
                # first real trial instead of before training. That is a genuine
                # reduction in coverage, forced by upstream -- and a narrower
                # loss than skipping the preflight entirely.
                objective_metrics=[
                    name
                    for name in (
                        *metrics,
                        *(k for k in joint_metric_fns if k not in metrics),
                    )
                    if not _is_diagnostic(name)
                ],
            )

    return hooks
