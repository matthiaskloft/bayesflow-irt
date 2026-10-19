"""Regression test guarding the core public API against ``__all__`` drift.

This does not assert set equality: ``__all__`` legitimately contains more
exports (SBC/HPO helpers) than the core lists below. It only asserts that
every core symbol is actually exported.
"""

import bayesflow_irt

# Core public API: Simulators.
CORE_SIMULATORS = [
    "make_1pl_simulator",
    "make_2pl_simulator",
    "make_pcm_simulator",
    "make_gpcm_simulator",
]

# Core public API: Adapters.
CORE_ADAPTERS = [
    "make_1pl_adapter",
    "make_2pl_adapter",
    "make_pcm_adapter",
    "make_gpcm_adapter",
    "LogTransform",
    "OneHotEncode",
]

# Core public API: Approximators and Networks.
CORE_APPROXIMATORS_AND_NETWORKS = [
    "MatrixEquivariantIRTSummary",
    "EquivariantIRTApproximator",
    "CalibratedIRTApproximator",
]

# Core public API: Priors and Utilities.
CORE_PRIORS_AND_UTILITIES = [
    "Prior",
    "PriorSpec",
    "VariablePrior",
    "make_incremental_prior",
    "make_variable_incremental_prior",
    "prior_moments",
    "sample_prior_predictive",
    "MetadataTracker",
    "PersonScorer",
    "PersonScore",
    "configure_large_scale_training",
    "probe_memory_config",
    "fit_with_oom_retry",
    "set_summary_memory_config",
    "safe_empty_cache",
    "is_oom_error",
    "build_multicondition_val_dataset",
    "PrecomputedDataset",
    "TrainingMemoryConfig",
    "ProbeResult",
    "MemoryProbeReport",
    "FitResult",
]

CORE_DOCUMENTED_NAMES = (
    CORE_SIMULATORS
    + CORE_ADAPTERS
    + CORE_APPROXIMATORS_AND_NETWORKS
    + CORE_PRIORS_AND_UTILITIES
)


def test_core_api_is_subset_of_all():
    """Every core public symbol must be exported in __all__.

    This is a subset check, not a set-equality check: __all__ may contain
    additional exports (e.g. SBC/HPO helpers).
    """
    assert set(CORE_DOCUMENTED_NAMES) <= set(bayesflow_irt.__all__)


def test_core_api_is_importable():
    """Every core public symbol must be a real package attribute."""
    for name in CORE_DOCUMENTED_NAMES:
        assert hasattr(bayesflow_irt, name), (
            f"{name!r} is listed as core public API but is "
            "not importable as an attribute of bayesflow_irt."
        )
