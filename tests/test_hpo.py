"""Tests for bayesflow_irt.hpo module."""

import logging
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from bayesflow_irt import (
    Prior,
    make_1pl_adapter,
    make_1pl_simulator,
    make_2pl_adapter,
    make_2pl_simulator,
    make_gpcm_adapter,
    make_gpcm_simulator,
    make_pcm_simulator,
)

pytest.importorskip("bayesflow_hpo", reason="bayesflow_hpo not installed")

from bayesflow_hpo import AggregationConfigError, ValidationDataset
from bayesflow_hpo.validation.registry import output_keys_for

import bayesflow_irt.hpo as hpo_mod
from bayesflow_irt.hpo import (
    IRTHooks,
    MatrixIRTSummarySpace,
    make_irt_hooks,
    make_irt_validation_dataset,
)
from bayesflow_irt.networks import MatrixEquivariantIRTSummary
from bayesflow_irt.utils import FitResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CORNERS_SMALL = [(3, 20), (5, 30)]


def _low_end_hparams(hooks) -> dict:
    """Every search dimension at its smallest value.

    Enough to build an approximator without sampling a trial, and cheap: the
    tests using it care about the memory knobs the hooks apply, not about the
    architecture they are applied to.
    """
    params: dict = {}
    for sub in ("summary_space", "inference_space", "training_space"):
        space = getattr(hooks.search_space, sub)
        for dim in space.dimensions:
            choices = getattr(dim, "choices", None)
            low = getattr(dim, "low", None)
            constant = getattr(dim, "constant", None)
            if choices:
                params[dim.name] = list(choices)[0]
            elif isinstance(low, (int, float)):
                params[dim.name] = low
            elif isinstance(constant, (int, float, str, bool)):
                # A dimension pinned to one value still has to appear in the
                # params dict: `build` reads it by name.
                params[dim.name] = constant
            else:
                raise AssertionError(f"cannot pick a value for {dim.name!r}: {dim!r}")
    return params


@pytest.fixture()
def sim_2pl():
    return make_2pl_simulator(
        n_items=(3, 5),
        n_persons=(20, 30),
        prior_a=Prior("lognormal", loc=0, scale=0.5),
        prior_b=Prior("normal", loc=0, scale=2.0),
    )


@pytest.fixture()
def adapter_2pl(sim_2pl):
    return make_2pl_adapter(sim_2pl)


@pytest.fixture()
def sim_1pl():
    return make_1pl_simulator(
        n_items=(3, 5),
        n_persons=(20, 30),
        prior_b=Prior("normal", loc=0, scale=2.0),
    )


@pytest.fixture()
def adapter_1pl(sim_1pl):
    return make_1pl_adapter(sim_1pl)


@pytest.fixture()
def sim_gpcm():
    return make_gpcm_simulator(
        n_items=(3, 5),
        n_persons=(20, 30),
        n_categories=4,
        prior_a=Prior("lognormal", loc=0, scale=0.5),
        prior_delta=Prior("normal", loc=0, scale=2.0),
    )


@pytest.fixture()
def adapter_gpcm(sim_gpcm):
    return make_gpcm_adapter(sim_gpcm)


# ---------------------------------------------------------------------------
# MatrixIRTSummarySpace
# ---------------------------------------------------------------------------


class TestMatrixIRTSummarySpace:
    def test_build_returns_matrix_summary_net(self):
        space = MatrixIRTSummarySpace()
        params = {
            "irt_matrix_hidden_dim": 32,
            "irt_matrix_num_blocks": 2,
            "irt_matrix_person_pooling": "mean_pma",
            "irt_matrix_num_pma_seeds": 2,
            "irt_matrix_num_outer_sabs": 2,
            "irt_matrix_outer_embed_dim": 64,
            "irt_matrix_dropout": 0.05,
            "irt_matrix_embed_dim": 32,
        }
        net = space.build(params)
        assert isinstance(net, MatrixEquivariantIRTSummary)
        assert net.person_pooling == "mean_pma"
        assert net.num_pma_seeds == 2

    def test_dimensions_prefixed(self):
        space = MatrixIRTSummarySpace()
        dim_names = [d.name for d in space.dimensions]
        for name in dim_names:
            assert name.startswith("irt_matrix_")

    def test_person_pooling_choices_include_mean_pma(self):
        space = MatrixIRTSummarySpace()
        pooling = next(
            d for d in space.dimensions if d.name == "irt_matrix_person_pooling"
        )
        assert set(pooling.choices) == {"mean", "pma", "mean_pma"}

    def test_num_pma_seeds_choices(self):
        space = MatrixIRTSummarySpace()
        seeds = next(
            d for d in space.dimensions if d.name == "irt_matrix_num_pma_seeds"
        )
        assert set(seeds.choices) == {1, 2, 4}

    def test_person_context_dimension_removed(self):
        """person_context is not a search dimension (redundant with the
        EMN row-mean term)."""
        space = MatrixIRTSummarySpace()
        assert not any(d.name == "irt_matrix_person_context" for d in space.dimensions)

    def test_include_rest_mean_score_choices(self):
        space = MatrixIRTSummarySpace()
        rest = next(
            d
            for d in space.dimensions
            if d.name == "irt_matrix_include_rest_mean_score"
        )
        assert set(rest.choices) == {False, True}
        assert not any(
            d.name == "irt_matrix_raw_feature_preset" for d in space.dimensions
        )

    def test_include_item_rest_correlation_choices(self):
        space = MatrixIRTSummarySpace()
        corr = next(
            d
            for d in space.dimensions
            if d.name == "irt_matrix_include_item_rest_correlation"
        )
        assert set(corr.choices) == {False, True}

    def test_raw_feature_dimensions_default_to_none_and_no_correlation(self):
        """A params dict without either key must fall back to
        raw_feature_channels=()."""
        space = MatrixIRTSummarySpace()
        params = {
            "irt_matrix_hidden_dim": 32,
            "irt_matrix_num_blocks": 2,
            "irt_matrix_person_pooling": "mean",
            "irt_matrix_num_pma_seeds": 1,
            "irt_matrix_num_outer_sabs": 2,
            "irt_matrix_outer_embed_dim": 64,
            "irt_matrix_dropout": 0.05,
            "irt_matrix_embed_dim": 32,
        }
        net = space.build(params)
        assert net.raw_feature_channels == ()

    @pytest.mark.parametrize(
        "include_rest,include_corr,expected",
        [
            (False, False, ()),
            (False, True, ("item_rest_correlation",)),
            (True, False, ("rest_mean_score",)),
            (True, True, ("rest_mean_score", "item_rest_correlation")),
        ],
    )
    def test_build_maps_channel_flags_to_channels(
        self, include_rest, include_corr, expected
    ):
        """Each channel must be tunable independently of the other -- e.g.
        correlation alone or rest without correlation."""
        space = MatrixIRTSummarySpace()
        params = {
            "irt_matrix_hidden_dim": 32,
            "irt_matrix_num_blocks": 2,
            "irt_matrix_person_pooling": "mean",
            "irt_matrix_num_pma_seeds": 1,
            "irt_matrix_include_rest_mean_score": include_rest,
            "irt_matrix_include_item_rest_correlation": include_corr,
            "irt_matrix_num_outer_sabs": 2,
            "irt_matrix_outer_embed_dim": 64,
            "irt_matrix_dropout": 0.05,
            "irt_matrix_embed_dim": 32,
        }
        net = space.build(params)
        assert net.raw_feature_channels == expected

    @pytest.mark.parametrize(
        "legacy_preset,expected",
        [
            ("none", ()),
            ("rest", ("rest_mean_score",)),
            ("means", ()),
            ("means_rest", ("rest_mean_score",)),
        ],
    )
    def test_build_maps_legacy_presets(self, legacy_preset, expected):
        """Params dicts using the legacy irt_matrix_raw_feature_preset key
        (including the means presets) must still build."""
        space = MatrixIRTSummarySpace()
        params = {
            "irt_matrix_hidden_dim": 32,
            "irt_matrix_num_blocks": 2,
            "irt_matrix_person_pooling": "mean",
            "irt_matrix_num_pma_seeds": 1,
            "irt_matrix_raw_feature_preset": legacy_preset,
            "irt_matrix_num_outer_sabs": 2,
            "irt_matrix_outer_embed_dim": 64,
            "irt_matrix_dropout": 0.05,
            "irt_matrix_embed_dim": 32,
        }
        net = space.build(params)
        assert net.raw_feature_channels == expected

    def test_build_ignores_stored_person_context_key(self):
        """Params including the legacy irt_matrix_person_context key must
        still build."""
        space = MatrixIRTSummarySpace()
        params = {
            "irt_matrix_hidden_dim": 32,
            "irt_matrix_num_blocks": 2,
            "irt_matrix_person_pooling": "mean",
            "irt_matrix_num_pma_seeds": 1,
            "irt_matrix_person_context": "item_mean",
            "irt_matrix_num_outer_sabs": 2,
            "irt_matrix_outer_embed_dim": 64,
            "irt_matrix_dropout": 0.05,
            "irt_matrix_embed_dim": 32,
        }
        net = space.build(params)
        assert isinstance(net, MatrixEquivariantIRTSummary)


# ---------------------------------------------------------------------------
# make_irt_validation_dataset
# ---------------------------------------------------------------------------


class TestMakeIRTValidationDataset:
    def test_basic(self, sim_2pl):
        ds = make_irt_validation_dataset(
            sim_2pl,
            corners=CORNERS_SMALL,
            param_keys=["a", "b"],
            sims_per_corner=5,
        )
        assert isinstance(ds, ValidationDataset)
        assert len(ds.simulations) == 2
        assert len(ds.condition_labels) == 2
        assert ds.param_keys == ["a", "b"]
        assert ds.data_keys == ["responses"]

    def test_shapes(self, sim_2pl):
        ds = make_irt_validation_dataset(
            sim_2pl,
            corners=[(3, 20)],
            param_keys=["a", "b"],
            sims_per_corner=4,
        )
        batch = ds.simulations[0]
        assert batch["a"].shape == (4, 3)
        assert batch["b"].shape == (4, 3)
        assert batch["responses"].shape == (4, 20, 3)

    def test_condition_labels(self, sim_2pl):
        ds = make_irt_validation_dataset(
            sim_2pl,
            corners=[(3, 20), (5, 30)],
            param_keys=["a", "b"],
            sims_per_corner=2,
        )
        assert ds.condition_labels[0] == {"n_items": 3, "n_persons": 20}
        assert ds.condition_labels[1] == {"n_items": 5, "n_persons": 30}


# ---------------------------------------------------------------------------
# Per-corner validation budget
# ---------------------------------------------------------------------------


class TestPerCornerSimsBudget:
    """``sims_per_corner`` as a sequence or an ``(n_items, n_persons)`` map.

    The corners differ by several times in measurement SD while the noisiest
    of them are the cheapest to simulate, so a scalar budget overpays for a
    given worst-corner fidelity.
    """

    def test_sequence_is_positional(self, sim_2pl):
        ds = make_irt_validation_dataset(
            sim_2pl,
            corners=[(3, 20), (5, 30)],
            param_keys=["a", "b"],
            sims_per_corner=[7, 2],
        )
        assert ds.simulations[0]["responses"].shape == (7, 20, 3)
        assert ds.simulations[1]["responses"].shape == (2, 30, 5)

    def test_mapping_is_keyed_by_corner(self, sim_2pl):
        ds = make_irt_validation_dataset(
            sim_2pl,
            corners=[(3, 20), (5, 30)],
            param_keys=["a", "b"],
            sims_per_corner={(3, 20): 7, (5, 30): 2},
        )
        assert ds.simulations[0]["responses"].shape == (7, 20, 3)
        assert ds.simulations[1]["responses"].shape == (2, 30, 5)

    def test_mapping_survives_reordered_corners(self, sim_2pl):
        """The reason the mapping form exists: reordering cannot misalign."""
        budget = {(3, 20): 7, (5, 30): 2}
        ds = make_irt_validation_dataset(
            sim_2pl,
            corners=[(5, 30), (3, 20)],
            param_keys=["a", "b"],
            sims_per_corner=budget,
        )
        assert ds.simulations[0]["responses"].shape == (2, 30, 5)
        assert ds.simulations[1]["responses"].shape == (7, 20, 3)

    def test_scalar_unchanged(self, sim_2pl):
        ds = make_irt_validation_dataset(
            sim_2pl,
            corners=CORNERS_SMALL,
            param_keys=["a", "b"],
            sims_per_corner=3,
        )
        assert [s["responses"].shape[0] for s in ds.simulations] == [3, 3]

    #: ``(factory kwargs, param_keys)`` per simulator family. All four are
    #: covered because the seeding guarantee below is a property of
    #: ``clone_with_sizes()``, which each family implements separately -- a
    #: family that stopped propagating ``seed=`` would break the guarantee
    #: while every 2PL test kept passing.
    _FAMILIES = {
        "2pl": (
            dict(
                prior_a=Prior("lognormal", loc=0, scale=0.5),
                prior_b=Prior("normal", loc=0, scale=2.0),
            ),
            ["a", "b"],
        ),
        "1pl": (dict(prior_b=Prior("normal", loc=0, scale=2.0)), ["b"]),
        "pcm": (
            dict(n_categories=4, prior_delta=Prior("normal", loc=0, scale=2.0)),
            ["delta"],
        ),
        "gpcm": (
            dict(
                n_categories=4,
                prior_a=Prior("lognormal", loc=0, scale=0.5),
                prior_delta=Prior("normal", loc=0, scale=2.0),
            ),
            ["a", "delta"],
        ),
    }

    @pytest.mark.parametrize("family", sorted(_FAMILIES))
    def test_other_corners_are_unaffected_by_one_corner_s_count(self, family):
        """Retuning one corner's budget must not redraw the others.

        Each corner is sampled from its own ``clone_with_sizes()`` clone,
        which restarts the generator from the simulator's seed, so a
        corner's batch depends on its own size and count and on nothing an
        earlier corner did. That is what makes a per-corner budget safe to
        tune: raising the first corner leaves the second bit-identical.

        Holds for a *seeded* simulator, which is the only case where any
        batch is reproducible -- see the unseeded companion test.
        """
        corners = [(3, 20), (5, 30)]
        factories = {
            "2pl": make_2pl_simulator,
            "1pl": make_1pl_simulator,
            "pcm": make_pcm_simulator,
            "gpcm": make_gpcm_simulator,
        }
        kwargs, param_keys = self._FAMILIES[family]

        def build(counts):
            sim = factories[family](
                n_items=(3, 5), n_persons=(20, 30), seed=7, **kwargs
            )
            return make_irt_validation_dataset(
                sim,
                corners=corners,
                param_keys=param_keys,
                sims_per_corner=counts,
            )

        few = build([2, 5])
        many = build([9, 5])
        assert few.simulations[0]["responses"].shape[0] == 2
        assert many.simulations[0]["responses"].shape[0] == 9
        np.testing.assert_array_equal(
            few.simulations[1]["responses"], many.simulations[1]["responses"]
        )
        for key in param_keys:
            np.testing.assert_array_equal(
                few.simulations[1][key], many.simulations[1][key]
            )

    def test_an_unseeded_simulator_offers_no_such_guarantee(self):
        """The docs qualify the claim; this is what they are qualifying.

        ``seed=None`` is the package default, and each clone then takes
        fresh entropy -- so no batch is reproducible in the first place,
        with or without a per-corner budget. Pinned so the unqualified
        version of the claim cannot creep back into the docstring.
        """

        def build():
            sim = make_2pl_simulator(
                n_items=(3, 5),
                n_persons=(20, 30),
                prior_a=Prior("lognormal", loc=0, scale=0.5),
                prior_b=Prior("normal", loc=0, scale=2.0),
            )
            return make_irt_validation_dataset(
                sim,
                corners=[(3, 20), (5, 30)],
                param_keys=["a", "b"],
                sims_per_corner=[2, 5],
            )

        assert not np.array_equal(
            build().simulations[1]["b"], build().simulations[1]["b"]
        )

    def test_a_numpy_array_budget_is_accepted(self):
        """A budget computed from measured per-corner SDs arrives as one."""
        sim = make_2pl_simulator(
            n_items=(3, 5),
            n_persons=(20, 30),
            prior_a=Prior("lognormal", loc=0, scale=0.5),
            prior_b=Prior("normal", loc=0, scale=2.0),
        )
        ds = make_irt_validation_dataset(
            sim,
            corners=CORNERS_SMALL,
            param_keys=["a", "b"],
            sims_per_corner=np.array([7, 2]),
        )
        assert [s["responses"].shape[0] for s in ds.simulations] == [7, 2]

    @pytest.mark.parametrize("budget", [{2, 5}, iter([2, 5])])
    def test_an_unordered_or_one_shot_budget_is_refused(self, sim_2pl, budget):
        """A set has no order to be positional against; a generator is not one."""
        with pytest.raises(TypeError, match="sims_per_corner"):
            make_irt_validation_dataset(
                sim_2pl,
                corners=CORNERS_SMALL,
                param_keys=["a", "b"],
                sims_per_corner=budget,
            )

    def test_a_stray_mapping_key_is_named_not_crashed_on(self, sim_2pl):
        """An unhashable-to-sort stray key must still reach the real message."""
        with pytest.raises(ValueError, match="not corners of this dataset"):
            make_irt_validation_dataset(
                sim_2pl,
                corners=CORNERS_SMALL,
                param_keys=["a", "b"],
                sims_per_corner={(3, 20): 2, (5, 30): 2, "5x10": 2},
            )

    def test_sequence_length_mismatch_raises(self, sim_2pl):
        with pytest.raises(ValueError, match="3 entries but"):
            make_irt_validation_dataset(
                sim_2pl,
                corners=CORNERS_SMALL,
                param_keys=["a", "b"],
                sims_per_corner=[1, 2, 3],
            )

    def test_mapping_missing_corner_raises(self, sim_2pl):
        with pytest.raises(ValueError, match="missing a count"):
            make_irt_validation_dataset(
                sim_2pl,
                corners=CORNERS_SMALL,
                param_keys=["a", "b"],
                sims_per_corner={(3, 20): 2},
            )

    def test_mapping_reversed_key_raises(self, sim_2pl):
        """``(n_persons, n_items)`` is a plausible slip and must not pass."""
        with pytest.raises(ValueError, match="missing a count"):
            make_irt_validation_dataset(
                sim_2pl,
                corners=CORNERS_SMALL,
                param_keys=["a", "b"],
                sims_per_corner={(20, 3): 2, (30, 5): 2},
            )

    def test_mapping_unknown_corner_raises(self, sim_2pl):
        with pytest.raises(ValueError, match="not corners of this dataset"):
            make_irt_validation_dataset(
                sim_2pl,
                corners=CORNERS_SMALL,
                param_keys=["a", "b"],
                sims_per_corner={(3, 20): 2, (5, 30): 2, (9, 90): 2},
            )

    @pytest.mark.parametrize("budget", [0, [2, 0], {(3, 20): 2, (5, 30): -1}])
    def test_non_positive_count_raises(self, sim_2pl, budget):
        with pytest.raises(ValueError, match=">= 1"):
            make_irt_validation_dataset(
                sim_2pl,
                corners=CORNERS_SMALL,
                param_keys=["a", "b"],
                sims_per_corner=budget,
            )

    @pytest.mark.parametrize("budget", [2.5, "4", True, [2, 2.5]])
    def test_non_integer_budget_raises(self, sim_2pl, budget):
        with pytest.raises(TypeError, match="sims_per_corner"):
            make_irt_validation_dataset(
                sim_2pl,
                corners=CORNERS_SMALL,
                param_keys=["a", "b"],
                sims_per_corner=budget,
            )


# ---------------------------------------------------------------------------
# make_irt_hooks
# ---------------------------------------------------------------------------


class TestMakeIRTHooks:
    def test_returns_named_tuple(self, sim_2pl, adapter_2pl):
        hooks = make_irt_hooks(
            sim_2pl,
            adapter_2pl,
            corners=CORNERS_SMALL,
            sims_per_corner=2,
        )
        assert isinstance(hooks, IRTHooks)
        assert callable(hooks.build_fn)
        assert callable(hooks.train_fn)
        assert callable(hooks.validate_fn)
        assert isinstance(hooks.validation_data, ValidationDataset)
        assert isinstance(hooks.search_space.summary_space, MatrixIRTSummarySpace)

    def test_param_keys_default_2pl(self, sim_2pl, adapter_2pl):
        hooks = make_irt_hooks(
            sim_2pl,
            adapter_2pl,
            corners=CORNERS_SMALL,
            sims_per_corner=2,
        )
        assert hooks.validation_data.param_keys == ["a", "b"]

    def test_param_keys_default_1pl(self, sim_1pl, adapter_1pl):
        hooks = make_irt_hooks(
            sim_1pl,
            adapter_1pl,
            corners=CORNERS_SMALL,
            sims_per_corner=2,
        )
        assert hooks.validation_data.param_keys == ["b"]

    def test_param_keys_explicit(self, sim_2pl, adapter_2pl):
        hooks = make_irt_hooks(
            sim_2pl,
            adapter_2pl,
            corners=CORNERS_SMALL,
            param_keys=["b"],
            sims_per_corner=2,
        )
        assert hooks.validation_data.param_keys == ["b"]

    def test_summary_api_matrix(self, sim_2pl, adapter_2pl):
        hooks = make_irt_hooks(
            sim_2pl,
            adapter_2pl,
            corners=CORNERS_SMALL,
            summary_api="matrix",
            sims_per_corner=2,
        )
        assert isinstance(hooks.search_space.summary_space, MatrixIRTSummarySpace)

    def test_summary_space_override(self, sim_2pl, adapter_2pl):
        override_space = MatrixIRTSummarySpace()
        hooks = make_irt_hooks(
            sim_2pl,
            adapter_2pl,
            corners=CORNERS_SMALL,
            summary_api="invalid_but_ignored",
            summary_space=override_space,
            sims_per_corner=2,
        )
        assert hooks.search_space.summary_space is override_space

    def test_training_space_defaults_to_a_searched_batch_size(
        self, sim_2pl, adapter_2pl
    ):
        hooks = make_irt_hooks(
            sim_2pl, adapter_2pl, corners=CORNERS_SMALL, sims_per_corner=2
        )
        batch = next(
            d
            for d in hooks.search_space.training_space.dimensions
            if d.name == "batch_size"
        )
        assert batch.low is not None and batch.high is not None

    def test_training_space_override_pins_the_dimension(self, sim_2pl, adapter_2pl):
        """A pinned dimension must leave the sampler, not merely be bounded.

        `constant=` is injected by `sample()` without reaching Optuna, so the
        value never appears in `trial.params`. That is the whole point of
        pinning `batch_size` in a search -- a degenerate range would still
        cost TPE a dimension of its model.
        """
        optuna = pytest.importorskip("optuna")
        from bayesflow_hpo import TrainingSpace
        from bayesflow_hpo.search_spaces.base import IntDimension

        override = TrainingSpace(batch_size=IntDimension("batch_size", constant=192))
        hooks = make_irt_hooks(
            sim_2pl,
            adapter_2pl,
            corners=CORNERS_SMALL,
            training_space=override,
            sims_per_corner=2,
        )
        assert hooks.search_space.training_space is override

        seen: list[dict] = []
        study = optuna.create_study()
        study.optimize(
            lambda t: (seen.append(hooks.search_space.sample(t)), 0.0)[1],
            n_trials=1,
        )
        assert seen[0]["batch_size"] == 192
        assert "batch_size" not in study.trials[0].params

    def test_invalid_summary_api_raises(self, sim_2pl, adapter_2pl):
        with pytest.raises(ValueError, match="summary_api"):
            make_irt_hooks(
                sim_2pl,
                adapter_2pl,
                corners=CORNERS_SMALL,
                summary_api="not_a_valid_api",
                sims_per_corner=2,
            )

    def test_invalid_aggregate_raises(self, sim_2pl, adapter_2pl):
        """An unsupported ``aggregate`` must raise, not fall through.

        The ``Literal`` annotation is not enforced at runtime and the
        aggregation closure treats anything that is not ``"mean"`` as the
        worst case, so without the guard ``aggregate="median"`` silently ran
        a different search than the caller requested.
        """
        with pytest.raises(ValueError, match="aggregate"):
            make_irt_hooks(
                sim_2pl,
                adapter_2pl,
                corners=CORNERS_SMALL,
                aggregate="median",  # type: ignore[arg-type]
                sims_per_corner=2,
            )

    def _mean_budget_warnings(self, caplog):
        return [
            r.getMessage()
            for r in caplog.records
            if "UNWEIGHTED average over corners" in r.getMessage()
        ]

    def test_unequal_budget_with_mean_warns(self, sim_2pl, adapter_2pl, caplog):
        """An unequal budget makes ``"<metric>/mean"`` an odd estimator.

        The reduction is left unweighted on purpose -- redefining it would
        change the meaning of ``"<metric>/mean"`` for existing callers, and
        selectable
        reductions are bayesflow-hpo#118's -- so the combination is warned
        about rather than silently redefined.
        """
        with caplog.at_level(logging.WARNING, logger="bayesflow_irt.hpo"):
            make_irt_hooks(
                sim_2pl,
                adapter_2pl,
                corners=CORNERS_SMALL,
                aggregate="mean",
                sims_per_corner=[2, 5],
            )
        assert self._mean_budget_warnings(caplog)

    def test_unequal_budget_with_mapped_mean_warns(self, sim_2pl, adapter_2pl, caplog):
        """A per-metric mapping must reach the same warning as the string."""
        with caplog.at_level(logging.WARNING, logger="bayesflow_irt.hpo"):
            make_irt_hooks(
                sim_2pl,
                adapter_2pl,
                corners=CORNERS_SMALL,
                metrics=["nrmse", "calibration_error"],
                aggregate={"nrmse": "mean"},
                sims_per_corner=[2, 5],
            )
        assert self._mean_budget_warnings(caplog)

    def test_unequal_budget_with_geometric_warns(self, sim_2pl, adapter_2pl, caplog):
        """``geometric`` averages in log space, so it has the same problem.

        A config selecting ``{"nrmse": "geometric"}`` together with a
        per-corner budget averages unweighted over unequally-precise corners.
        """
        with caplog.at_level(logging.WARNING, logger="bayesflow_irt.hpo"):
            make_irt_hooks(
                sim_2pl,
                adapter_2pl,
                corners=CORNERS_SMALL,
                metrics=["nrmse", "calibration_error"],
                aggregate={"nrmse": "geometric"},
                sims_per_corner=[2, 5],
            )
        assert self._mean_budget_warnings(caplog)

    def test_unequal_budget_with_worst_still_warns(self, sim_2pl, adapter_2pl, caplog):
        """The objective does not decide whether averaged keys are recorded.

        ``aggregate="worst"`` leaves the OBJECTIVE unaffected -- a worst case
        selects a corner rather than averaging over them -- but
        ``"<metric>/mean"`` and ``"<metric>/geometric"`` are written for every
        metric regardless, and callers may read them from a finished study.
        An unequal budget makes those numbers unweighted averages over
        unequally-precise corners whatever the objective reduces by, so the
        warning is about the budget.
        """
        with caplog.at_level(logging.WARNING, logger="bayesflow_irt.hpo"):
            make_irt_hooks(
                sim_2pl,
                adapter_2pl,
                corners=CORNERS_SMALL,
                aggregate="worst",
                sims_per_corner=[2, 5],
            )
        assert self._mean_budget_warnings(caplog)

    def test_equal_budget_with_mean_does_not_warn(self, sim_2pl, adapter_2pl, caplog):
        """The budget is the whole condition, so an equal one must be silent."""
        with caplog.at_level(logging.WARNING, logger="bayesflow_irt.hpo"):
            make_irt_hooks(
                sim_2pl,
                adapter_2pl,
                corners=CORNERS_SMALL,
                aggregate="mean",
                sims_per_corner=[2, 2],
            )
        assert not self._mean_budget_warnings(caplog)

    def test_a_scalar_budget_does_not_warn(self, sim_2pl, adapter_2pl, caplog):
        """A scalar is equal by construction, so it must not warn."""
        with caplog.at_level(logging.WARNING, logger="bayesflow_irt.hpo"):
            make_irt_hooks(
                sim_2pl,
                adapter_2pl,
                corners=CORNERS_SMALL,
                aggregate="mean",
                sims_per_corner=2,
            )
        assert not self._mean_budget_warnings(caplog)

    def test_hooks_pass_per_corner_budget_through(self, sim_2pl, adapter_2pl):
        hooks = make_irt_hooks(
            sim_2pl,
            adapter_2pl,
            corners=[(3, 20), (5, 30)],
            sims_per_corner={(3, 20): 4, (5, 30): 2},
        )
        sims = hooks.validation_data.simulations
        assert [s["responses"].shape[0] for s in sims] == [4, 2]

    def test_train_fn_injects_periodic_cuda_cache_release(self, sim_2pl, adapter_2pl):
        """train_fn should append the cache-release callback on the torch
        backend, since each trial's different summary/flow architecture and
        variable-shape batches can ratchet up CUDA reserved memory across
        trials the same way a single fit_with_oom_retry run can (see
        bayesflow_irt.utils._PeriodicCudaCacheRelease)."""
        hooks = make_irt_hooks(
            sim_2pl,
            adapter_2pl,
            corners=CORNERS_SMALL,
            sims_per_corner=2,
        )

        seen_kwargs = {}

        class FakeApprox:
            def fit(self, **kwargs):
                seen_kwargs.update(kwargs)

        user_callback = object()
        hooks.train_fn(
            FakeApprox(),
            sim_2pl,
            {"epochs": 1, "num_batches": 1, "batch_size": 8},
            [user_callback],
        )

        callbacks = seen_kwargs["callbacks"]
        assert user_callback in callbacks
        assert any(type(cb).__name__ == "_PeriodicCudaCacheRelease" for cb in callbacks)

    def test_train_fn_routes_through_the_oom_retry(self, sim_2pl, adapter_2pl):
        """train_fn must retry on OOM rather than let the trial fail.

        A trial's memory is dominated by activations over the
        (batch, persons, items, hidden) grid, and batch size is searched
        independently of the summary width — so a plausible architecture can
        OOM mid-epoch. A bare ``fit`` turns that into a failed trial scored
        with the failure penalty, which is indistinguishable from a genuinely
        bad architecture: the search would rank memory feasibility as model
        quality.
        """
        hooks = make_irt_hooks(
            sim_2pl,
            adapter_2pl,
            corners=CORNERS_SMALL,
            sims_per_corner=2,
        )

        calls = {}

        def fake_retry(approximator, simulator, **kwargs):
            calls.update(kwargs)
            return FitResult(
                history=None,
                batch_size=kwargs["start_batch_size"],
                item_chunk_size=None,
                attempts=1,
                config=None,
            )

        original = hpo_mod.fit_with_oom_retry
        hpo_mod.fit_with_oom_retry = fake_retry
        try:
            hooks.train_fn(
                object(),
                sim_2pl,
                {"epochs": 1, "num_batches": 1, "batch_size": 8},
                [],
            )
        finally:
            hpo_mod.fit_with_oom_retry = original

        assert calls["start_batch_size"] == 8
        # No batch-size cut: chunking only, then the trial fails.
        assert calls["min_batch_size"] == 8
        assert calls["fit_kwargs"]["epochs"] == 1
        assert calls["fit_kwargs"]["num_batches"] == 1

    def test_build_fn_applies_the_trial_memory_config(self, sim_2pl, adapter_2pl):
        """Every trial is built with the memory controls, not the dense default.

        ``MatrixIRTSummarySpace.build`` deliberately leaves these unset, so the
        hooks have to apply them; otherwise a real search OOMs on any large
        draw.
        """
        hooks = make_irt_hooks(
            sim_2pl,
            adapter_2pl,
            corners=CORNERS_SMALL,
            sims_per_corner=2,
        )
        approximator = hooks.build_fn(_low_end_hparams(hooks))

        assert approximator.summary_network.gradient_checkpointing is True
        assert (
            approximator.flow_chunk_size
            == hpo_mod.DEFAULT_TRIAL_MEMORY_CONFIG["flow_chunk_size"]
        )

    def test_memory_config_override_is_applied(self, sim_2pl, adapter_2pl):
        hooks = make_irt_hooks(
            sim_2pl,
            adapter_2pl,
            corners=CORNERS_SMALL,
            sims_per_corner=2,
            memory_config={"gradient_checkpointing": False, "flow_chunk_size": 64},
        )
        approximator = hooks.build_fn(_low_end_hparams(hooks))

        assert approximator.summary_network.gradient_checkpointing is False
        assert approximator.flow_chunk_size == 64

    def test_unknown_memory_config_key_raises(self, sim_2pl, adapter_2pl):
        """A typo must not silently leave the dense default in place."""
        with pytest.raises(ValueError, match="memory_config"):
            make_irt_hooks(
                sim_2pl,
                adapter_2pl,
                corners=CORNERS_SMALL,
                sims_per_corner=2,
                memory_config={"gradient_checkpoint": True},
            )

    @pytest.mark.slow
    def test_run_pipeline_check_passes(self, sim_1pl, adapter_1pl):
        """run_pipeline_check=True dry-runs build->compile->train->validate."""
        hooks = make_irt_hooks(
            sim_1pl,
            adapter_1pl,
            corners=[(3, 20)],
            sims_per_corner=2,
            run_pipeline_check=True,
        )
        assert isinstance(hooks, IRTHooks)

    @pytest.mark.slow
    def test_pipeline_check_rejects_a_broken_joint_metric(self, sim_2pl, adapter_2pl):
        """The preflight must see joint metrics, not just marginal ones.

        The joint guard omits a metric that failed, so a broken reference
        provider would otherwise pass the build/train/validate check on the
        strength of the marginal metrics alone -- the opposite of what the
        check exists for.
        """

        def _broken(inputs):
            raise RuntimeError("provider is broken")

        with pytest.raises(Exception, match="tarp_error"):
            make_irt_hooks(
                sim_2pl,
                adapter_2pl,
                corners=[(3, 20)],
                param_keys=["a", "b"],
                sims_per_corner=2,
                joint_metrics={"tarp_error": _broken},
                run_pipeline_check=True,
            )

    def test_a_recorded_diagnostic_is_kept_out_of_the_preflight(
        self, sim_2pl, adapter_2pl, monkeypatch
    ):
        """A diagnostic in `metrics` must not reach `objective_metrics`.

        Since bayesflow-hpo v0.5.0 `check_pipeline` runs
        `validate_objective_metric_kinds` over that argument and refuses a
        registered diagnostic outright. `correlation` and `contraction` may
        only ever be RECORDED -- the same upstream rule bars them from
        `objective_metrics` -- so forwarding them here made a legitimate
        config fail the preflight it is meant to pass. Pinned here rather
        than left to the slow tests.

        Unregistered names must still be forwarded: upstream leaves those
        valid deliberately, and `log_gamma` plus the TARP metrics depend on
        it.
        """
        seen: dict[str, list[str]] = {}

        def _capture(**kwargs):
            seen["names"] = list(kwargs["objective_metrics"])

        monkeypatch.setattr(hpo_mod, "check_pipeline", _capture)
        make_irt_hooks(
            sim_2pl,
            adapter_2pl,
            corners=CORNERS_SMALL,
            param_keys=["a", "b"],
            sims_per_corner=2,
            metrics=["calibration_error", "nrmse", "correlation", "log_gamma"],
            run_pipeline_check=True,
        )
        assert "correlation" not in seen["names"], (
            "a registered diagnostic reached check_pipeline's "
            "objective_metrics, which v0.5.0 refuses"
        )
        # The objective-ready and the unregistered names both stay.
        assert "calibration_error" in seen["names"]
        assert "nrmse" in seen["names"]
        assert "log_gamma" in seen["names"]


# ---------------------------------------------------------------------------
# Search-space registration with bayesflow_hpo's shared registry
# ---------------------------------------------------------------------------


class TestSearchSpaceRegistration:
    def test_irt_spaces_registered(self):
        from bayesflow_hpo import list_registered_network_spaces

        registered = list_registered_network_spaces()["summary"]
        assert "irt_matrix" in registered

    def test_get_summary_space_resolves_aliases(self):
        from bayesflow_hpo.search_spaces.registry import get_summary_space

        assert isinstance(get_summary_space("matrix"), MatrixIRTSummarySpace)
        assert isinstance(get_summary_space("irt_matrix"), MatrixIRTSummarySpace)


class TestImportSafety:
    """bayesflow_irt.hpo must import cleanly even without bayesflow_hpo.

    Simulates a missing ``bayesflow_hpo`` by blocking it in ``sys.modules``
    (the standard trick: a ``None`` entry makes ``import`` raise
    ``ImportError``) and reloading ``bayesflow_irt.hpo``, mirroring the
    import-safe placeholder pattern used by ``bayesflow_irt.calibration``.
    """

    def test_module_reloads_without_bayesflow_hpo(self):
        import importlib
        import sys

        import bayesflow_irt.hpo as hpo_mod

        saved_modules = {
            name: mod
            for name, mod in sys.modules.items()
            if name == "bayesflow_hpo" or name.startswith("bayesflow_hpo.")
        }
        for name in saved_modules:
            del sys.modules[name]
        sys.modules["bayesflow_hpo"] = None  # forces ImportError on import

        try:
            importlib.reload(hpo_mod)

            assert hpo_mod._BAYESFLOW_HPO_IMPORT_ERROR is not None
            assert hpo_mod.MatrixIRTSummarySpace is not None

            with pytest.raises(ImportError, match="bayesflow-hpo"):
                hpo_mod.make_irt_hooks(None, None, corners=[(3, 20)])
            with pytest.raises(ImportError, match="bayesflow-hpo"):
                hpo_mod.make_irt_validation_dataset(
                    None, corners=[(3, 20)], param_keys=["a"]
                )
        finally:
            del sys.modules["bayesflow_hpo"]
            sys.modules.update(saved_modules)
            importlib.reload(hpo_mod)


# ---------------------------------------------------------------------------
# validate_fn flatten logic
# ---------------------------------------------------------------------------


class TestValidateFnFlatten:
    def test_flatten_shapes(self, sim_2pl, adapter_2pl):
        """Verify that validate_fn flattens (B, n_samples, I) correctly."""
        hooks = make_irt_hooks(
            sim_2pl,
            adapter_2pl,
            corners=[(3, 20)],
            param_keys=["a", "b"],
            sims_per_corner=4,
        )

        batch_size, n_samples, n_items = 4, 10, 3

        # Mock approximator.sample to return known shapes.
        mock_approx = MagicMock()
        mock_approx.sample.return_value = {
            "a": np.random.randn(batch_size, n_samples, n_items),
            "b": np.random.randn(batch_size, n_samples, n_items),
        }

        result = hooks.validate_fn(mock_approx, None, n_samples)

        assert "calibration_error" in result
        assert "nrmse" in result
        assert isinstance(result["calibration_error"], float)
        assert isinstance(result["nrmse"], float)

        # Verify sample was called with the right n_samples.
        call_kwargs = mock_approx.sample.call_args[1]
        assert call_kwargs["num_samples"] == n_samples


class TestMultiOutputMarginalMetrics:
    """A metric emitting columns must be reduced by those columns.

    `coverage` writes `coverage_90` and four siblings into each
    `(corner, param_key)` row; it never writes `coverage`. Reducing by the
    configured name found nothing in any row, so every computed value was
    discarded and the group name was reported as NaN -- which the reduction
    returns so that Optuna reads a FAILED trial. Nothing had failed, so a
    study configured with `metrics=["coverage"]` would have recorded a
    failed-looking trial for its entire budget.
    """

    def _mock(self, n_items=3, n_samples=10, batch=4):
        approx = MagicMock()
        approx.sample.return_value = {
            "a": np.random.rand(batch, n_samples, n_items) + 0.5,
            "b": np.random.randn(batch, n_samples, n_items),
        }
        return approx

    def _result(self, sim, adapter, **kwargs):
        hooks = make_irt_hooks(
            sim,
            adapter,
            corners=[(3, 20)],
            param_keys=["a", "b"],
            sims_per_corner=4,
            **kwargs,
        )
        return hooks.validate_fn(self._mock(), None, 10)

    def test_every_emitted_column_is_reduced(self, sim_2pl, adapter_2pl):
        result = self._result(sim_2pl, adapter_2pl, metrics=["coverage"])
        for key in output_keys_for("coverage"):
            assert f"{key}/mean" in result, f"{key} was never reduced"
            assert not np.isnan(result[f"{key}/mean"])

    def test_the_group_name_is_not_reported_as_a_failed_metric(
        self, sim_2pl, adapter_2pl
    ):
        """The NaN is the harmful half: Optuna reads it as a failed trial."""
        result = self._result(sim_2pl, adapter_2pl, metrics=["coverage"])
        assert not ("coverage" in result and np.isnan(result["coverage"])), (
            "the group name is back, and NaN"
        )

    def test_a_single_output_metric_is_unchanged(self, sim_2pl, adapter_2pl):
        """`output_keys_for` is the identity here, so nothing may move."""
        result = self._result(sim_2pl, adapter_2pl, metrics=["calibration_error"])
        assert isinstance(result["calibration_error"], float)
        assert not np.isnan(result["calibration_error"])
        assert "calibration_error/worst" in result

    def test_a_mapping_keyed_by_the_producer_reaches_every_column(
        self, sim_2pl, adapter_2pl
    ):
        """`{"coverage": ...}` is what a caller who listed `coverage` means."""
        result = self._result(
            sim_2pl,
            adapter_2pl,
            metrics=["coverage"],
            aggregate={"coverage": "mean"},
        )
        for key in output_keys_for("coverage"):
            assert result[key] == result[f"{key}/mean"]

    def test_a_mapping_keyed_by_one_column_overrides_only_it(
        self, sim_2pl, adapter_2pl
    ):
        """An exact column key must win over its producer's entry."""
        result = self._result(
            sim_2pl,
            adapter_2pl,
            metrics=["coverage"],
            aggregate={"coverage": "mean", "coverage_90": "geometric"},
        )
        assert result["coverage_90"] == result["coverage_90/geometric"]
        assert result["coverage_95"] == result["coverage_95/mean"]

    def test_a_producer_mapping_is_not_reported_as_inert(self, sim_2pl, adapter_2pl):
        """The inert-key check must credit the producer for its columns.

        It raises `AggregationConfigError` to stop a study whose mapping
        never took effect. A producer name takes effect on all five of its
        columns, so counting only column names would fail the study on a
        setting that did work.
        """
        self._result(
            sim_2pl,
            adapter_2pl,
            metrics=["coverage"],
            aggregate={"coverage": "geometric"},
        )

    def test_a_diagnostic_column_without_a_direction_omits_the_bare_key(
        self, sim_2pl, adapter_2pl
    ):
        """No direction is registered for any `coverage_*` column.

        `aggregate="worst"` is the default, so reducing these columns at all
        reaches the worst case. A diagnostic metric cannot be an objective
        (`validate_objective_metric_kinds` refuses it upstream), so there is
        no objective to protect and raising would break `metrics=["coverage"]`
        a second way. The bare key is omitted instead, which reads as "not
        applicable" rather than as a failed trial.
        """
        result = self._result(sim_2pl, adapter_2pl, metrics=["coverage"])
        assert "coverage_90" not in result
        assert "coverage_90/worst" not in result
        assert "coverage_90/mean" in result


class TestValidateFnFlowSampleKwargs:
    """Search-time integrator override.

    A fixed-step Euler solver samples much faster than the adaptive
    default. The model is still *built* at
    whatever fidelity the search space chose; only the scoring draws are
    cheapened, via ``FlowMatching._inverse``'s ``self.integrate_kwargs |
    kwargs`` merge.
    """

    @staticmethod
    def _mock_approx():
        mock_approx = MagicMock()
        mock_approx.sample.return_value = {
            "a": np.abs(np.random.randn(4, 10, 3)) + 0.1,
            "b": np.random.randn(4, 10, 3),
        }
        return mock_approx

    def _hooks(self, sim_2pl, adapter_2pl, **kwargs):
        return make_irt_hooks(
            sim_2pl,
            adapter_2pl,
            corners=[(3, 20)],
            param_keys=["a", "b"],
            sims_per_corner=4,
            **kwargs,
        )

    def test_the_override_reaches_every_scoring_draw(self, sim_2pl, adapter_2pl):
        override = {"method": "euler", "steps": 16}
        hooks = self._hooks(sim_2pl, adapter_2pl, flow_sample_kwargs=override)
        mock_approx = self._mock_approx()

        hooks.validate_fn(mock_approx, None, 10)

        assert mock_approx.sample.call_count > 0
        for call in mock_approx.sample.call_args_list:
            assert call[1]["flow_sample_kwargs"] == override

    def test_the_default_leaves_the_sample_call_untouched(self, sim_2pl, adapter_2pl):
        """Not merely absent-and-empty: the kwarg must not be passed at all.

        Omitting it must leave the call identical to the default.
        """
        hooks = self._hooks(sim_2pl, adapter_2pl)
        mock_approx = self._mock_approx()

        hooks.validate_fn(mock_approx, None, 10)

        for call in mock_approx.sample.call_args_list:
            assert "flow_sample_kwargs" not in call[1]

    def test_an_empty_mapping_is_treated_as_no_override(self, sim_2pl, adapter_2pl):
        hooks = self._hooks(sim_2pl, adapter_2pl, flow_sample_kwargs={})
        mock_approx = self._mock_approx()

        hooks.validate_fn(mock_approx, None, 10)

        for call in mock_approx.sample.call_args_list:
            assert "flow_sample_kwargs" not in call[1]

    def test_a_non_mapping_raises_at_construction_not_mid_study(
        self, sim_2pl, adapter_2pl
    ):
        """Deferred to the closure this would raise after a trial had trained,
        and ``GenericObjective`` converts a validation exception into a
        training-loss fallback — so the study would quietly score every trial
        on the wrong axis instead of failing."""
        with pytest.raises(TypeError, match="flow_sample_kwargs"):
            self._hooks(sim_2pl, adapter_2pl, flow_sample_kwargs="euler")

    def test_the_caller_mapping_is_copied_not_captured(self, sim_2pl, adapter_2pl):
        """Mutating the dict after building the hooks must not silently change
        what later trials are scored under."""
        override = {"method": "euler", "steps": 16}
        hooks = self._hooks(sim_2pl, adapter_2pl, flow_sample_kwargs=override)
        override["steps"] = 2
        mock_approx = self._mock_approx()

        hooks.validate_fn(mock_approx, None, 10)

        assert mock_approx.sample.call_args[1]["flow_sample_kwargs"] == {
            "method": "euler",
            "steps": 16,
        }


class TestGPCMHooks:
    def test_param_keys_default_gpcm(self, sim_gpcm, adapter_gpcm):
        hooks = make_irt_hooks(
            sim_gpcm,
            adapter_gpcm,
            corners=CORNERS_SMALL,
            sims_per_corner=2,
        )
        assert hooks.validation_data.param_keys == ["a", "delta"]

    def test_2pl_detection_unaffected(self, sim_2pl, adapter_2pl):
        hooks = make_irt_hooks(
            sim_2pl,
            adapter_2pl,
            corners=CORNERS_SMALL,
            sims_per_corner=2,
        )
        assert hooks.validation_data.param_keys == ["a", "b"]

    def test_validate_fn_multidim_delta(
        self,
        sim_gpcm,
        adapter_gpcm,
    ):
        """validate_fn handles 3D delta (B, n_s, I, K-1)."""
        hooks = make_irt_hooks(
            sim_gpcm,
            adapter_gpcm,
            corners=[(3, 20)],
            param_keys=["a", "delta"],
            sims_per_corner=4,
        )
        batch_size, n_samples, n_items, n_steps = 4, 10, 3, 3
        mock_approx = MagicMock()
        mock_approx.sample.return_value = {
            "a": np.random.randn(batch_size, n_samples, n_items),
            "delta": np.random.randn(batch_size, n_samples, n_items, n_steps),
        }
        result = hooks.validate_fn(mock_approx, None, n_samples)
        assert "calibration_error" in result
        assert "nrmse" in result


class TestCornerAggregation:
    """Averaging a metric across corners lets opposite failures cancel.

    Modrak et al. (2025) describe this as the mechanism by which a
    calibration statistic reports a clean result for a model that is
    miscalibrated in opposite directions at different points. HPO
    *optimizes* on these numbers, so an averaged objective can actively
    reward a configuration whose worst corner is bad.
    """

    def _hooks_with_metric(self, sim, adapter, monkeypatch, per_corner, metric):
        """Build hooks whose metric returns a scripted value per corner."""
        calls = {"n": 0}

        def fake_resolve(names):
            def fn(_draws, _true):
                # Two param_keys per corner, so each corner is visited twice;
                # index by corner rather than by call.
                value = per_corner[calls["n"] // 1 % len(per_corner)]
                calls["n"] += 1
                return {metric: value}

            return {metric: fn}

        monkeypatch.setattr(hpo_mod, "resolve_metrics", fake_resolve)
        return calls

    def test_worst_uses_the_metrics_direction(self):
        """`max` is "worst" only for lower-is-better metrics."""
        # Errors: bigger is worse.
        assert hpo_mod._worst("nrmse", [0.1, 0.9]) == 0.9
        assert hpo_mod._worst("calibration_error", [0.02, 0.30]) == 0.30
        # log_gamma runs the other way: BayesFlow documents log_gamma < 0 as
        # rejecting uniformity, so the SMALLEST value is the worst corner.
        assert hpo_mod._worst("log_gamma", [-2.0, 0.5]) == -2.0
        assert hpo_mod._worst("contraction", [0.1, 0.9]) == 0.1

    def test_worst_refuses_an_unknown_metric_rather_than_guessing(self):
        """A wrongly-signed reduction reports the BEST corner as the worst."""
        with pytest.raises(ValueError, match="direction is unknown"):
            hpo_mod._worst("some_custom_metric", [0.1, 0.9])

    def test_worst_is_the_default_and_does_not_average_corners(
        self, sim_2pl, adapter_2pl, monkeypatch
    ):
        """A good corner must not be allowed to hide a bad one."""
        self._hooks_with_metric(sim_2pl, adapter_2pl, monkeypatch, [0.1, 0.9], "nrmse")
        hooks = make_irt_hooks(
            sim_2pl,
            adapter_2pl,
            corners=[(3, 20), (5, 30)],
            param_keys=["a"],
            metrics=["nrmse"],
            sims_per_corner=2,
        )
        mock_approx = MagicMock()
        mock_approx.sample.return_value = {
            "a": np.random.randn(2, 5, 3),
            "b": np.random.randn(2, 5, 3),
        }
        result = hooks.validate_fn(mock_approx, None, 5)
        # The mean would be 0.5 and would look acceptable; the worst is 0.9.
        assert result["nrmse"] == pytest.approx(0.9)

    def _worst_corner_hooks(self, sim, adapter, monkeypatch, per_corner):
        """Hooks over two labelled corners with scripted per-corner values."""
        self._hooks_with_metric(sim, adapter, monkeypatch, per_corner, "nrmse")
        hooks = make_irt_hooks(
            sim,
            adapter,
            corners=[(3, 20), (5, 30)],
            param_keys=["a"],
            metrics=["nrmse"],
            sims_per_corner=2,
        )
        mock_approx = MagicMock()
        mock_approx.sample.return_value = {
            "a": np.random.randn(2, 5, 3),
            "b": np.random.randn(2, 5, 3),
        }
        return hooks.validate_fn(mock_approx, None, 5)

    def test_worst_records_which_corner_produced_it(
        self, sim_2pl, adapter_2pl, monkeypatch
    ):
        """`worst` selects a corner, so say which one it selected.

        Without this the per-corner rows are computed, reduced and dropped,
        and a finished study cannot tell whether two trials' `nrmse` values
        even describe the same corner.
        """
        result = self._worst_corner_hooks(sim_2pl, adapter_2pl, monkeypatch, [0.1, 0.9])
        assert result["nrmse"] == pytest.approx(0.9)
        # 0.9 came from the second corner, (5, 30).
        assert result["nrmse/worst_corner"] == pytest.approx(1.0)
        assert result["nrmse/worst_n_items"] == pytest.approx(5.0)
        assert result["nrmse/worst_n_persons"] == pytest.approx(30.0)

    def test_worst_corner_follows_the_value_not_the_position(
        self, sim_2pl, adapter_2pl, monkeypatch
    ):
        """Reversing the corners must move the reported corner with them."""
        result = self._worst_corner_hooks(sim_2pl, adapter_2pl, monkeypatch, [0.9, 0.1])
        assert result["nrmse"] == pytest.approx(0.9)
        assert result["nrmse/worst_corner"] == pytest.approx(0.0)
        assert result["nrmse/worst_n_items"] == pytest.approx(3.0)
        assert result["nrmse/worst_n_persons"] == pytest.approx(20.0)

    def test_worst_corner_survives_a_dropped_nan_row(
        self, sim_2pl, adapter_2pl, monkeypatch
    ):
        """A NaN corner shifts list positions; the label must not shift.

        Reducing first and looking the index up afterwards would report
        corner 0 here, because filtering NaN moves the surviving value to
        position 0.
        """
        result = self._worst_corner_hooks(
            sim_2pl, adapter_2pl, monkeypatch, [float("nan"), 0.9]
        )
        assert result["nrmse"] == pytest.approx(0.9)
        assert result["nrmse/worst_corner"] == pytest.approx(1.0)
        assert result["nrmse/worst_n_items"] == pytest.approx(5.0)

    def test_worst_corner_breaks_a_tie_toward_the_first_corner(
        self, sim_2pl, adapter_2pl, monkeypatch
    ):
        """The docstring promises `max`/`min`'s rule; hold it to that.

        Two corners at the same value leave the reported corner genuinely
        underdetermined, so the choice has to be documented *and* pinned.
        Otherwise a later refactor could flip it silently and every
        cross-trial corner comparison in a finished study would shift
        without any value changing.
        """
        result = self._worst_corner_hooks(sim_2pl, adapter_2pl, monkeypatch, [0.5, 0.5])
        assert result["nrmse"] == pytest.approx(0.5)
        assert result["nrmse/worst_corner"] == pytest.approx(0.0)
        assert result["nrmse/worst_n_items"] == pytest.approx(3.0)

    def test_worst_corner_is_nan_when_every_corner_is_minus_inf(
        self, sim_2pl, adapter_2pl, monkeypatch
    ):
        """An all-`-inf` row identifies no corner.

        `log_gamma` can underflow to `-inf` on every corner; the tie-break
        index would then read as a measured corner.
        """
        monkeypatch.setattr(hpo_mod, "_lower_is_better", lambda _b: False)
        result = self._worst_corner_hooks(
            sim_2pl, adapter_2pl, monkeypatch, [-np.inf, -np.inf]
        )
        assert result["nrmse/worst"] == -np.inf
        assert np.isnan(result["nrmse/worst_corner"])
        assert np.isnan(result["nrmse/worst_n_items"])
        assert np.isnan(result["nrmse/worst_n_persons"])

    def test_worst_corner_kept_when_one_corner_is_minus_inf(
        self, sim_2pl, adapter_2pl, monkeypatch
    ):
        """A single non-finite corner is a genuine worst, not a tie."""
        monkeypatch.setattr(hpo_mod, "_lower_is_better", lambda _b: False)
        result = self._worst_corner_hooks(
            sim_2pl, adapter_2pl, monkeypatch, [0.3, -np.inf]
        )
        assert result["nrmse/worst_corner"] == pytest.approx(1.0)
        assert result["nrmse/worst_n_items"] == pytest.approx(5.0)

    def _scripted_hooks(self, sim, adapter, monkeypatch, per_call, param_keys):
        """Hooks whose metric returns `per_call[i]` on its i-th call.

        The per-corner helper above indexes by corner, which cannot express
        two cells of one corner differing -- the case the per-corner keys
        exist to resolve.
        """
        calls = {"n": 0}

        def fake_resolve(_names):
            def fn(_draws, _true):
                value = per_call[calls["n"]]
                calls["n"] += 1
                return {"nrmse": value}

            return {"nrmse": fn}

        monkeypatch.setattr(hpo_mod, "resolve_metrics", fake_resolve)
        hooks = make_irt_hooks(
            sim,
            adapter,
            corners=[(3, 20), (5, 30)],
            param_keys=param_keys,
            metrics=["nrmse"],
            sims_per_corner=2,
        )
        mock_approx = MagicMock()
        mock_approx.sample.return_value = {
            "a": np.random.randn(2, 5, 3),
            "b": np.random.randn(2, 5, 3),
        }
        return hooks.validate_fn(mock_approx, None, 5)

    def test_the_per_corner_values_are_recorded_not_only_their_reduction(
        self, sim_2pl, adapter_2pl, monkeypatch
    ):
        """A `worst_corner` index is uninterpretable without the margin.

        The reduction says corner 1 was worst; only the values say whether
        it won by 0.8 or by 0.001 -- that is, whether "worst" was a
        decision or a coin flip.
        """
        result = self._worst_corner_hooks(sim_2pl, adapter_2pl, monkeypatch, [0.1, 0.9])
        assert result["nrmse/worst_corner"] == pytest.approx(1.0)
        assert result["nrmse/corner_0/worst"] == pytest.approx(0.1)
        assert result["nrmse/corner_1/worst"] == pytest.approx(0.9)
        # One cell per corner here, so both reductions see the same value.
        assert result["nrmse/corner_0/mean"] == pytest.approx(0.1)
        assert result["nrmse/corner_1/mean"] == pytest.approx(0.9)

    def test_the_grid_reductions_are_reconstructible_from_the_corners(
        self, sim_2pl, adapter_2pl, monkeypatch
    ):
        """Both keys per corner, because a corner holds several cells.

        With two `param_keys` the grid is parameter x corner, and the two
        reductions disagree about which corner is worst: corner 0 holds the
        single worst cell (0.3) while corner 1 has the higher mean. One key
        per corner could record only one of those.
        """
        result = self._scripted_hooks(
            sim_2pl,
            adapter_2pl,
            monkeypatch,
            [0.1, 0.3, 0.2, 0.25],
            ["a", "b"],
        )
        assert result["nrmse/corner_0/worst"] == pytest.approx(0.3)
        assert result["nrmse/corner_0/mean"] == pytest.approx(0.2)
        assert result["nrmse/corner_1/worst"] == pytest.approx(0.25)
        assert result["nrmse/corner_1/mean"] == pytest.approx(0.225)
        # The grid-wide numbers follow from the per-corner ones.
        assert result["nrmse/worst"] == pytest.approx(
            max(result["nrmse/corner_0/worst"], result["nrmse/corner_1/worst"])
        )
        assert result["nrmse/worst_corner"] == pytest.approx(0.0)
        assert result["nrmse/mean"] == pytest.approx(
            (result["nrmse/corner_0/mean"] + result["nrmse/corner_1/mean"]) / 2
        )

    def test_one_nan_cell_separates_the_grid_mean_from_the_corner_means(
        self, sim_2pl, adapter_2pl, monkeypatch
    ):
        """The documented limit of the per-corner keys, pinned.

        The grid mean is over *cells* and the per-corner means are over
        *corners*, so they coincide only while every corner contributes the
        same number of valid cells. Corner 0 loses one of its two here, and
        the two averages part company: 0.36667 against 0.5. The docstring
        says so; this holds the code to the weaker claim, so a later change
        cannot quietly make the stronger one true-looking.

        `{metric}/worst` is unaffected -- max distributes over a partition
        whatever the block sizes -- which is why that one is documented
        without a caveat.
        """
        result = self._scripted_hooks(
            sim_2pl,
            adapter_2pl,
            monkeypatch,
            [float("nan"), 0.9, 0.1, 0.1],
            ["a", "b"],
        )
        assert result["nrmse/corner_0/mean"] == pytest.approx(0.9)
        assert result["nrmse/corner_1/mean"] == pytest.approx(0.1)
        # Cell-weighted, over the three cells that survived.
        assert result["nrmse/mean"] == pytest.approx((0.9 + 0.1 + 0.1) / 3)
        assert result["nrmse/mean"] != pytest.approx(
            (result["nrmse/corner_0/mean"] + result["nrmse/corner_1/mean"]) / 2
        )
        # The worst case still reconstructs exactly.
        assert result["nrmse/worst"] == pytest.approx(
            max(result["nrmse/corner_0/worst"], result["nrmse/corner_1/worst"])
        )
        assert result["nrmse/worst_corner"] == pytest.approx(0.0)

    def test_a_corner_with_no_value_is_nan_rather_than_a_missing_key(
        self, sim_2pl, adapter_2pl, monkeypatch
    ):
        """The key set must not vary from trial to trial.

        An analysis that filters on key presence rather than on value is
        a mistake, so a corner that computed nothing still
        gets its keys.
        """
        result = self._worst_corner_hooks(
            sim_2pl, adapter_2pl, monkeypatch, [float("nan"), 0.9]
        )
        assert "nrmse/corner_0/worst" in result
        assert np.isnan(result["nrmse/corner_0/worst"])
        assert np.isnan(result["nrmse/corner_0/mean"])
        assert result["nrmse/corner_1/worst"] == pytest.approx(0.9)

    def test_an_emitted_key_may_not_occupy_the_corner_namespace(
        self, sim_2pl, adapter_2pl
    ):
        """A joint metric must not be able to overwrite a corner summary.

        `joint_metrics` emits whatever its callable returns, so the literal
        key `nrmse/corner_0/mean` is reduced like any other emitted key and
        its reduction lands on the key this module synthesized for
        `nrmse`, so the summary would come back as the joint metric's
        value. Nothing downstream can tell the two
        apart afterwards, so it is refused at the end of the first trial
        rather than written.
        """
        hooks = make_irt_hooks(
            sim_2pl,
            adapter_2pl,
            corners=[(3, 20), (5, 30)],
            param_keys=["a", "b"],
            metrics=["nrmse"],
            sims_per_corner=4,
            joint_metrics={"probe": lambda inputs: {"nrmse/corner_0/mean": 999.0}},
        )
        with pytest.raises(hpo_mod.AggregationConfigError, match="corner_"):
            hooks.validate_fn(_mock_2pl_approx(), None, 8)

    def test_an_unknown_direction_omits_the_per_corner_worst_only(
        self, sim_2pl, adapter_2pl
    ):
        """Same rule as `{metric}/worst`: refuse rather than guess.

        The direction is a property of the metric, so this omits the same
        keys on every trial of a study and leaves the key set stable.
        """
        hooks = make_irt_hooks(
            sim_2pl,
            adapter_2pl,
            corners=[(3, 20)],
            param_keys=["a", "b"],
            sims_per_corner=4,
            aggregate="mean",
            joint_metrics={"probe": lambda inputs: {"probe": 0.5}},
        )
        result = hooks.validate_fn(_mock_2pl_approx(), None, 8)
        assert result["probe/corner_0/mean"] == pytest.approx(0.5)
        assert "probe/corner_0/worst" not in result
        assert "probe/worst" not in result

    def test_mean_is_still_available_and_reproduces_the_old_number(
        self, sim_2pl, adapter_2pl, monkeypatch
    ):
        self._hooks_with_metric(sim_2pl, adapter_2pl, monkeypatch, [0.1, 0.9], "nrmse")
        hooks = make_irt_hooks(
            sim_2pl,
            adapter_2pl,
            corners=[(3, 20), (5, 30)],
            param_keys=["a"],
            metrics=["nrmse"],
            sims_per_corner=2,
            aggregate="mean",
        )
        mock_approx = MagicMock()
        mock_approx.sample.return_value = {"a": np.random.randn(2, 5, 3)}
        result = hooks.validate_fn(mock_approx, None, 5)
        assert result["nrmse"] == pytest.approx(0.5)

    @pytest.mark.parametrize("aggregate", ["worst", "mean"])
    def test_both_reductions_are_always_reported(
        self, sim_2pl, adapter_2pl, monkeypatch, aggregate
    ):
        """Neither number is conditional on which one is the objective.

        They answer different questions and both are cheap once the
        per-corner rows exist, so a study's stored trials should never be
        unable to answer a question its own data contains.
        """
        self._hooks_with_metric(sim_2pl, adapter_2pl, monkeypatch, [0.1, 0.9], "nrmse")
        hooks = make_irt_hooks(
            sim_2pl,
            adapter_2pl,
            corners=[(3, 20), (5, 30)],
            param_keys=["a"],
            metrics=["nrmse"],
            sims_per_corner=2,
            aggregate=aggregate,
        )
        mock_approx = MagicMock()
        mock_approx.sample.return_value = {"a": np.random.randn(2, 5, 3)}
        result = hooks.validate_fn(mock_approx, None, 5)
        assert result["nrmse/mean"] == pytest.approx(0.5)
        assert result["nrmse/worst"] == pytest.approx(0.9)
        # `aggregate` chooses only which one is ALSO the bare objective key.
        expected = 0.9 if aggregate == "worst" else 0.5
        assert result["nrmse"] == pytest.approx(expected)

    def test_unknown_direction_still_reports_the_mean(
        self, sim_2pl, adapter_2pl, monkeypatch
    ):
        """A metric with no recorded direction loses only the worst case."""
        self._hooks_with_metric(
            sim_2pl, adapter_2pl, monkeypatch, [0.1, 0.9], "mystery_metric"
        )
        hooks = make_irt_hooks(
            sim_2pl,
            adapter_2pl,
            corners=[(3, 20), (5, 30)],
            param_keys=["a"],
            metrics=["mystery_metric"],
            sims_per_corner=2,
            aggregate="mean",
        )
        mock_approx = MagicMock()
        mock_approx.sample.return_value = {"a": np.random.randn(2, 5, 3)}
        result = hooks.validate_fn(mock_approx, None, 5)
        assert result["mystery_metric/mean"] == pytest.approx(0.5)
        assert "mystery_metric/worst" not in result
        assert result["mystery_metric"] == pytest.approx(0.5)

    def test_unknown_direction_raises_only_when_it_is_the_objective(
        self, sim_2pl, adapter_2pl, monkeypatch
    ):
        self._hooks_with_metric(
            sim_2pl, adapter_2pl, monkeypatch, [0.1, 0.9], "mystery_metric"
        )
        hooks = make_irt_hooks(
            sim_2pl,
            adapter_2pl,
            corners=[(3, 20), (5, 30)],
            param_keys=["a"],
            metrics=["mystery_metric"],
            sims_per_corner=2,
        )
        mock_approx = MagicMock()
        mock_approx.sample.return_value = {"a": np.random.randn(2, 5, 3)}
        with pytest.raises(ValueError, match="direction.*is unknown"):
            hooks.validate_fn(mock_approx, None, 5)


class TestDefaultObjectiveMetrics:
    """The search optimizes on these, so their blind spots are load-bearing."""

    def test_objectives_are_accepted_as_objectives_by_the_registry(self):
        """`coverage_*` are registered diagnostic and would be rejected."""
        from bayesflow_hpo.validation.registry import (
            validate_objective_metric_kinds,
        )

        validate_objective_metric_kinds(hpo_mod.DEFAULT_OBJECTIVE_METRICS)

    def test_objectives_are_a_subset_of_what_is_computed(self):
        assert set(hpo_mod.DEFAULT_OBJECTIVE_METRICS) <= set(hpo_mod._DEFAULT_METRICS)

    def test_nrmse_is_an_objective(self):
        """The guard against a posterior that ignores the data.

        Both calibration terms are rank statistics over marginal parameters,
        and Modrak et al. (2025) show such a statistic is clean for a
        posterior that ignores the data. `nrmse` is what makes that posterior
        unprofitable to the search; dropping it re-opens the hole.
        """
        assert "nrmse" in hpo_mod.DEFAULT_OBJECTIVE_METRICS

    def test_every_objective_has_a_recorded_direction(self):
        """`_worst` raises on an unknown direction, so a default that lacked
        one would break every search at the first validation call."""
        for name in hpo_mod._DEFAULT_METRICS:
            assert hpo_mod._lower_is_better(name) is not None, name

    def test_calibration_error_is_still_computed_for_comparability(self):
        """A stable member of the default metric set."""
        assert "calibration_error" in hpo_mod._DEFAULT_METRICS

    def test_log_gamma_worst_case_is_the_minimum(self):
        """Larger log_gamma is better, so `max` would report the BEST corner.

        BayesFlow's `calibration_log_gamma` documents log_gamma as
        log(gamma/gamma_null), where "log_gamma < 0 implies a rejection of the
        hypothesis of uniform ranks at the 5% level".
        """
        assert hpo_mod._lower_is_better("log_gamma") is False
        assert hpo_mod._worst("log_gamma", [-8.0, -0.2, 1.5]) == -8.0


# ---------------------------------------------------------------------------
# Joint metrics: reaching TARP from the validation closure
# ---------------------------------------------------------------------------


def _mock_2pl_approx(n_sims=4, n_samples=8, n_items=3):
    """An approximator whose draws match whatever conditions it is given."""
    mock_approx = MagicMock()

    def _sample(conditions, num_samples, **_):
        b, i = np.asarray(conditions["a"]).shape
        rng = np.random.default_rng(0)
        return {
            "a": np.abs(rng.normal(size=(b, num_samples, i))) + 0.1,
            "b": rng.normal(size=(b, num_samples, i)),
        }

    mock_approx.sample.side_effect = _sample
    return mock_approx


class TestJointBlock:
    """The un-flattening itself: items become columns, simulations do not."""

    def test_truths_keep_the_simulation_axis(self):
        # The marginal path folds items INTO the batch, giving (B*I,). A
        # joint metric needs the opposite: (B, I).
        values = np.arange(2 * 3, dtype=float).reshape(2, 3)
        out = hpo_mod._joint_block(values, "b", sample_axis=None)
        assert out.shape == (2, 3)
        assert np.array_equal(out, values)

    def test_draws_keep_simulation_and_sample_axes(self):
        values = np.random.randn(4, 10, 3)
        out = hpo_mod._joint_block(values, "b", sample_axis=1)
        assert out.shape == (4, 10, 3)

    def test_trailing_category_axis_folds_into_columns(self):
        # 3-D delta, (B, I, K-1): the step axis is a parameter axis, not a
        # simulation axis, so it joins the columns.
        values = np.random.randn(4, 3, 2)
        assert hpo_mod._joint_block(values, "delta", sample_axis=None).shape == (4, 6)

    def test_a_is_logged_and_b_is_not(self):
        # Not cosmetic: TARP standardizes each dimension, but standardizing
        # does not undo a log, so the geometry the classical reference was
        # certified in (log(a) stacked against b) is the log one.
        a = np.abs(np.random.randn(4, 3)) + 0.5
        assert np.allclose(hpo_mod._joint_block(a, "a", sample_axis=None), np.log(a))
        b = np.random.randn(4, 3)
        assert np.allclose(hpo_mod._joint_block(b, "b", sample_axis=None), b)


class TestClassicalReferenceProvider:
    def test_shape_is_n_sims_by_n_params(self, sim_2pl):
        batch = sim_2pl.sample((6,))
        n_items = np.asarray(batch["a"]).shape[1]
        provider = hpo_mod.make_classical_reference_provider(["a", "b"])
        refs = provider(SimpleNamespace(sim_batch=batch, cond_id=0, n_conditions=1))
        assert refs.shape == (6, 2 * n_items)

    def test_column_order_follows_param_keys(self, sim_2pl):
        batch = sim_2pl.sample((6,))
        n_items = np.asarray(batch["a"]).shape[1]
        inputs = SimpleNamespace(sim_batch=batch, cond_id=0, n_conditions=1)
        ab = hpo_mod.make_classical_reference_provider(["a", "b"], jitter=0.0)(inputs)
        ba = hpo_mod.make_classical_reference_provider(["b", "a"], jitter=0.0)(inputs)
        # Swapping the keys swaps the blocks. Column order is stated, not
        # implied -- the provider slices positionally, so a silent reorder
        # would pair discrimination draws with difficulty reference points.
        assert np.allclose(ab[:, :n_items], ba[:, n_items:])
        assert np.allclose(ab[:, n_items:], ba[:, :n_items])

    def test_uses_the_unconverted_classical_indices(self, sim_2pl):
        from bayesflow_irt import classical_item_statistics

        batch = sim_2pl.sample((6,))
        stats = classical_item_statistics(np.asarray(batch["responses"], dtype=float))
        refs = hpo_mod.make_classical_reference_provider(["a", "b"], jitter=0.0)(
            SimpleNamespace(sim_batch=batch, cond_id=0, n_conditions=1)
        )
        n_items = stats["point_biserial"].shape[1]
        assert np.allclose(refs[:, :n_items], stats["point_biserial"])
        assert np.allclose(refs[:, n_items:], stats["z_difficulty"])

    def test_jitter_is_off_by_default(self):
        # The reference is a deterministic function of the responses. See
        # DEFAULT_REFERENCE_JITTER.
        assert hpo_mod.DEFAULT_REFERENCE_JITTER == 0.0

    def test_default_provider_is_a_pure_function_of_the_responses(self, sim_2pl):
        batch = sim_2pl.sample((6,))
        inputs = SimpleNamespace(sim_batch=batch, cond_id=0, n_conditions=1)
        n_items = np.asarray(batch["a"]).shape[1]
        from bayesflow_irt import classical_item_statistics

        stats = classical_item_statistics(batch["responses"])

        refs = hpo_mod.make_classical_reference_provider(["a", "b"])(inputs)

        assert np.allclose(refs[:, :n_items], stats["point_biserial"])
        assert np.allclose(refs[:, n_items:], stats["z_difficulty"])

    def test_jitter_applies_the_papers_construction(self, sim_2pl):
        """``U(0, jitter)`` on the raw statistic, exactly as Sec. 4.3 has it.

        Pinned because it is tempting to "fix": the noise is uncentred and
        unscaled, so it lands very differently in the two blocks (see
        DEFAULT_REFERENCE_JITTER). Centring or rescaling it would be a
        noise law no source here describes, which is why the default is 0
        instead.
        """
        batch = sim_2pl.sample((400,))
        inputs = SimpleNamespace(sim_batch=batch, cond_id=0, n_conditions=1)
        clean = hpo_mod.make_classical_reference_provider(["a", "b"], jitter=0.0)(
            inputs
        )
        noisy = hpo_mod.make_classical_reference_provider(["a", "b"], jitter=1.0)(
            inputs
        )
        added = noisy - clean

        assert added.min() >= 0.0
        assert added.max() <= 1.0
        # U(0, 1): mean 1/2, sd 1/sqrt(12).
        assert float(added.mean()) == pytest.approx(0.5, abs=0.02)
        assert float(added.std()) == pytest.approx(12.0**-0.5, abs=0.02)

    def test_jitter_detects_the_mirror_case_the_default_misses(self):
        """The trade-off behind ``DEFAULT_REFERENCE_JITTER``, pinned.

        A zero-jitter reference is a single position function, and
        Theorem 3's sufficiency is stated over all of them. Against a
        truth ``N(f(x) + a, 1)`` and an estimate ``N(f(x) - a, 1)`` the
        distance from ``f(x)`` has the same distribution either way, so the
        default is blind. ``U(0, jitter)`` is not symmetric about zero, so
        it breaks that symmetry and the error becomes visible.

        This is the exact reverse of the width corruption the docstring
        also reports, where jitter *costs* sensitivity. Both directions are
        pinned so that neither gets written up as a general rule.
        """
        tarp = pytest.importorskip("bayesflow_hpo.validation.tarp")

        n_sims, n_draws, a = 4000, 200, 1.0
        errors = []
        for jitter in (0.0, 1.0):
            rng = np.random.default_rng(10)
            fx = rng.normal(size=(n_sims, 1))
            truth = fx + a + rng.normal(size=(n_sims, 1))
            draws = fx[:, None, :] - a + rng.normal(size=(n_sims, n_draws, 1))
            refs = fx + jitter * rng.uniform(size=fx.shape)
            out = tarp.compute_tarp_coverage(
                draws, truth, reference_points=refs, standardize=True
            )
            errors.append(float(out["tarp_error"]))

        blind, seen = errors
        # Near the floor: the mirror is invisible without jitter.
        assert blind < 0.05
        # Unmistakable with it -- an order of magnitude, not a nudge.
        assert seen > 0.15

    def test_jitter_is_deterministic(self, sim_2pl):
        batch = sim_2pl.sample((6,))
        inputs = SimpleNamespace(sim_batch=batch, cond_id=0, n_conditions=2)
        # Explicitly jittered: with the default of zero there is no noise
        # to be deterministic about, and the test would pass vacuously.
        first = hpo_mod.make_classical_reference_provider(["a", "b"], jitter=1.0)(
            inputs
        )
        second = hpo_mod.make_classical_reference_provider(["a", "b"], jitter=1.0)(
            inputs
        )
        # As an HPO objective, two identical trials must not score
        # differently.
        assert np.allclose(first, second)

    def test_jitter_differs_across_conditions(self, sim_2pl):
        batch = sim_2pl.sample((6,))
        provider = hpo_mod.make_classical_reference_provider(["a", "b"], jitter=1.0)
        first = provider(SimpleNamespace(sim_batch=batch, cond_id=0, n_conditions=2))
        second = provider(SimpleNamespace(sim_batch=batch, cond_id=1, n_conditions=2))
        # Same data, different corner: reusing one stream would correlate
        # the corners' noise, which is why `cond_id` is in the contract.
        assert not np.allclose(first, second)

    def test_polytomous_keys_rejected_at_construction(self):
        # Scoped deliberately, not discovered at run time.
        with pytest.raises(ValueError, match="1PL and 2PL only"):
            hpo_mod.make_classical_reference_provider(["a", "delta"])

    def test_provider_never_reads_the_posterior(self, sim_2pl):
        # A reference derived from the estimator under test reintroduces
        # exactly the blind spot the metric exists to detect, is reported
        # under the key that claims teeth, and nothing downstream can
        # detect it.
        batch = sim_2pl.sample((6,))

        def _boom(self):
            raise AssertionError("the provider must not read the posterior")

        class _NoPosterior:
            sim_batch = batch
            cond_id = 0
            n_conditions = 1
            draws = property(_boom)
            approximator = property(_boom)

        hpo_mod.make_classical_reference_provider(["a", "b"])(_NoPosterior())


class TestValidateFnJointDispatch:
    def test_tarp_reaches_the_closure(self, sim_2pl, adapter_2pl):
        """TARP is reachable from an IRT HPO study."""
        from bayesflow_hpo.validation.tarp import make_tarp_joint_metric

        hooks = make_irt_hooks(
            sim_2pl,
            adapter_2pl,
            corners=[(3, 20), (5, 30)],
            param_keys=["a", "b"],
            sims_per_corner=4,
            joint_metrics={
                "tarp_error": make_tarp_joint_metric(
                    hpo_mod.make_classical_reference_provider(["a", "b"]),
                    reference_id="test-classical",
                )
            },
        )
        result = hooks.validate_fn(_mock_2pl_approx(), None, 8)

        assert "tarp_error" in result
        assert np.isfinite(result["tarp_error"])
        # Aggregated across corners like any other metric.
        assert "tarp_error/mean" in result
        assert "tarp_error/worst" in result
        # The marginal metrics still work: the joint path is additive.
        assert "nrmse" in result
        assert "calibration_error" in result

    def test_worst_case_uses_the_upstream_direction(self):
        # `tarp_error` is registered lower-is-better upstream, so the worst
        # corner is the max. This module keeps no local copy.
        assert not hasattr(hpo_mod, "_METRIC_LOWER_IS_BETTER")
        assert hpo_mod._lower_is_better("tarp_error") is True
        assert hpo_mod._worst("tarp_error", [0.1, 0.4, 0.2]) == 0.4

    def test_direction_follows_what_the_objective_minimizes(self):
        """A `HIGHER_IS_BETTER` mutation changes the objective's direction;
        the corner reduction must follow it, not the raw dict."""
        from bayesflow_hpo.objectives import HIGHER_IS_BETTER

        HIGHER_IS_BETTER.discard("contraction")
        try:
            assert hpo_mod._lower_is_better("contraction") is None
        finally:
            HIGHER_IS_BETTER.add("contraction")
        assert hpo_mod._lower_is_better("contraction") is False

    def test_nothing_joint_runs_unless_asked(self, sim_2pl, adapter_2pl):
        # Opt-in, so a study that does not ask for joint metrics scores
        # exactly as without them.
        hooks = make_irt_hooks(
            sim_2pl,
            adapter_2pl,
            corners=[(3, 20)],
            param_keys=["a", "b"],
            sims_per_corner=4,
        )
        result = hooks.validate_fn(_mock_2pl_approx(), None, 8)
        assert not any(k.startswith("tarp") for k in result)

    def test_tarp_without_a_reference_raises_at_construction(
        self, sim_2pl, adapter_2pl
    ):
        # Not per trial: a model would already have been trained, and
        # `GenericObjective` converts a validation exception into a
        # training-loss fallback, so the study would burn its whole budget
        # scoring trials on a metric that never ran.
        with pytest.raises(Exception, match="(?i)reference|configur"):
            make_irt_hooks(
                sim_2pl,
                adapter_2pl,
                corners=[(3, 20)],
                param_keys=["a", "b"],
                sims_per_corner=4,
                metrics=["nrmse", "tarp_error"],
            )

    def test_random_reference_keeps_its_own_key(self, sim_2pl, adapter_2pl):
        """The reference mode must not be laundered.

        Lemos et al. (2023, Sec. 4.3): a TARP run with an x-independent
        reference is blind to a posterior that ignores its data, so its
        clean curve must never be reportable under the key that claims
        teeth.
        """
        hooks = make_irt_hooks(
            sim_2pl,
            adapter_2pl,
            corners=[(3, 20)],
            param_keys=["a", "b"],
            sims_per_corner=4,
            metrics=["nrmse", "tarp_error_random"],
        )
        result = hooks.validate_fn(_mock_2pl_approx(), None, 8)
        assert "tarp_error_random" in result
        assert "tarp_error" not in result

    def test_a_failing_joint_metric_is_absent_not_zero(self, sim_2pl, adapter_2pl):
        """A failed joint metric must not reach the objective as a number.

        Absent rather than NaN, which is where joint metrics deliberately
        differ from the marginal ones above: the objective substitutes a
        metric's registered worst case exactly once for an absent key, and
        `tarp_error` registers `worst_raw=1.0`. 0.0 would make a broken
        measurement the best trial in the study.
        """

        def _explode(inputs):
            raise RuntimeError("boom")

        hooks = make_irt_hooks(
            sim_2pl,
            adapter_2pl,
            corners=[(3, 20)],
            param_keys=["a", "b"],
            sims_per_corner=4,
            joint_metrics={"tarp_error": _explode},
        )
        result = hooks.validate_fn(_mock_2pl_approx(), None, 8)
        assert "tarp_error" not in result
        assert not any(k.startswith("tarp_error") for k in result)

    def test_joint_inputs_carry_the_stated_column_order(self, sim_2pl, adapter_2pl):
        captured = {}

        def _capture(inputs):
            captured["param_keys"] = inputs.param_keys
            captured["draws"] = inputs.draws.shape
            captured["true_values"] = inputs.true_values.shape
            captured["data_keys"] = inputs.data_keys
            return {"probe": 0.0}

        hooks = make_irt_hooks(
            sim_2pl,
            adapter_2pl,
            corners=[(3, 20)],
            param_keys=["a", "b"],
            sims_per_corner=4,
            # "probe" has no registered direction, so the worst-case
            # aggregation would refuse it -- see the `joint_metrics`
            # docstring. That refusal is the subject of its own test.
            aggregate="mean",
            joint_metrics={"probe": _capture},
        )
        hooks.validate_fn(_mock_2pl_approx(), None, 8)

        # Stated explicitly, not implied by iteration order.
        assert captured["param_keys"] == ("a", "b")
        # (n_sims, n_samples, n_params) with n_params = 2 * n_items -- NOT
        # the (B*I, n_draws) the marginal path builds.
        assert captured["draws"] == (4, 8, 6)
        assert captured["true_values"] == (4, 6)
        assert captured["data_keys"] == ("responses",)

    def test_custom_joint_name_needs_a_direction_to_be_worst_aggregated(
        self, sim_2pl, adapter_2pl
    ):
        # A joint metric this package does not own has no registered
        # direction, so the worst corner is not computable. Refusing is
        # right -- guessing would report the BEST corner as the worst-case
        # guarantee -- but it happens at the end of the first trial rather
        # than at construction, because the direction is only consulted once
        # there are per-corner values to reduce.
        hooks = make_irt_hooks(
            sim_2pl,
            adapter_2pl,
            corners=[(3, 20)],
            param_keys=["a", "b"],
            sims_per_corner=4,
            joint_metrics={"probe": lambda inputs: {"probe": 0.5}},
        )
        with pytest.raises(ValueError, match="direction"):
            hooks.validate_fn(_mock_2pl_approx(), None, 8)

        # ...and `aggregate="mean"` is the stated way through.
        hooks_mean = make_irt_hooks(
            sim_2pl,
            adapter_2pl,
            corners=[(3, 20)],
            param_keys=["a", "b"],
            sims_per_corner=4,
            aggregate="mean",
            joint_metrics={"probe": lambda inputs: {"probe": 0.5}},
        )
        assert hooks_mean.validate_fn(_mock_2pl_approx(), None, 8)["probe"] == 0.5


class TestJointColumnCorrespondence:
    """Column *j* of draws, truths and references must be one parameter.

    The failure this guards is the one that still produces a plausible
    number: a transposed or misordered block pairs discrimination draws with
    difficulty reference points, and TARP returns a perfectly ordinary
    coverage error from it.
    """

    def _row(self, corner_batch, posterior, param_keys):
        captured = {}

        def _capture(inputs):
            captured["inputs"] = inputs
            return {"probe": 0.0}

        hpo_mod._joint_row(
            {"probe": _capture},
            corner_batch=corner_batch,
            posterior=posterior,
            param_keys=param_keys,
            data_keys=["responses"],
            approximator=None,
            cond_id=0,
            n_conditions=1,
            failed_joint={},
            emitted_keys={},
        )
        return captured["inputs"]

    def test_draws_and_truths_agree_column_by_column(self):
        n_sims, n_samples, n_items = 3, 4, 5
        # Distinct sentinels: value encodes (param, sim, item), so any
        # misordering shows up as an inequality rather than as noise.
        a_true = np.exp(np.arange(n_sims * n_items).reshape(n_sims, n_items) * 0.1)
        b_true = -np.arange(n_sims * n_items).reshape(n_sims, n_items) * 1.0
        batch = {"a": a_true, "b": b_true, "responses": np.zeros((n_sims, 2, n_items))}
        # Every posterior sample equals the truth, so draws and truths must
        # match column for column if the assembly is consistent.
        post = {
            "a": np.repeat(a_true[:, None, :], n_samples, axis=1),
            "b": np.repeat(b_true[:, None, :], n_samples, axis=1),
        }
        inputs = self._row(batch, post, ["a", "b"])

        assert inputs.true_values.shape == (n_sims, 2 * n_items)
        assert inputs.draws.shape == (n_sims, n_samples, 2 * n_items)
        for s in range(n_samples):
            assert np.allclose(inputs.draws[:, s, :], inputs.true_values)
        # And the blocks are the parameters they claim to be.
        assert np.allclose(inputs.true_values[:, :n_items], np.log(a_true))
        assert np.allclose(inputs.true_values[:, n_items:], b_true)

    def test_three_dimensional_param_agrees_too(self):
        # delta is (B, I, K-1); its draws are (B, S, I, K-1). The item and
        # category axes must fold into columns in the SAME order on both
        # paths, or a step difficulty is compared against a different one.
        n_sims, n_samples, n_items, n_steps = 3, 4, 5, 2
        delta = np.arange(n_sims * n_items * n_steps, dtype=float).reshape(
            n_sims, n_items, n_steps
        )
        batch = {"delta": delta, "responses": np.zeros((n_sims, 2, n_items))}
        post = {"delta": np.repeat(delta[:, None, ...], n_samples, axis=1)}
        inputs = self._row(batch, post, ["delta"])

        assert inputs.true_values.shape == (n_sims, n_items * n_steps)
        assert inputs.draws.shape == (n_sims, n_samples, n_items * n_steps)
        for s in range(n_samples):
            assert np.allclose(inputs.draws[:, s, :], inputs.true_values)

    def test_reference_columns_line_up_with_draw_columns(self, sim_2pl):
        from bayesflow_irt import classical_item_statistics

        batch = sim_2pl.sample((6,))
        n_items = np.asarray(batch["a"]).shape[1]
        post = {
            k: np.repeat(np.asarray(batch[k], dtype=float)[:, None, :], 4, axis=1)
            for k in ("a", "b")
        }
        inputs = self._row(batch, post, ["a", "b"])
        refs = hpo_mod.make_classical_reference_provider(["a", "b"], jitter=0.0)(inputs)

        assert refs.shape == inputs.true_values.shape
        stats = classical_item_statistics(np.asarray(batch["responses"], dtype=float))
        # Column block 0 is discrimination on both sides; block 1 difficulty.
        assert np.allclose(inputs.true_values[:, :n_items], np.log(batch["a"]))
        assert np.allclose(refs[:, :n_items], stats["point_biserial"])
        assert np.allclose(inputs.true_values[:, n_items:], batch["b"])
        assert np.allclose(refs[:, n_items:], stats["z_difficulty"])


class TestJointFailureInvalidation:
    """A joint metric that fails anywhere is invalidated everywhere.

    Mirrors `bayesflow_hpo.validation.pipeline`'s guard. Averaging the
    corners that happened to work would report a finite, flattering number
    exactly when the measurement is broken -- and a worst case over a
    silently smaller grid is not a worst case over the grid.
    """

    def _hooks(self, sim, adapter, fn, **kw):
        return make_irt_hooks(
            sim,
            adapter,
            corners=[(3, 20), (5, 30)],
            param_keys=["a", "b"],
            sims_per_corner=4,
            joint_metrics={"tarp_error": fn},
            **kw,
        )

    def test_failure_on_one_corner_drops_the_metric_entirely(
        self, sim_2pl, adapter_2pl
    ):
        calls = {"n": 0}

        def _fail_second(inputs):
            calls["n"] += 1
            if inputs.cond_id == 1:
                raise RuntimeError("boom")
            return {"tarp_error": 0.01}

        result = self._hooks(sim_2pl, adapter_2pl, _fail_second).validate_fn(
            _mock_2pl_approx(), None, 8
        )
        # Absent, not 0.01: the objective substitutes its registered worst
        # case once, rather than reading a good score off the one corner
        # that worked.
        assert "tarp_error" not in result
        assert "tarp_error/mean" not in result
        assert "tarp_error/worst" not in result

    def test_a_nan_is_treated_as_a_failure_not_averaged_away(
        self, sim_2pl, adapter_2pl
    ):
        def _nan_second(inputs):
            return {"tarp_error": 0.01 if inputs.cond_id == 0 else float("nan")}

        result = self._hooks(sim_2pl, adapter_2pl, _nan_second).validate_fn(
            _mock_2pl_approx(), None, 8
        )
        # Without this, nanmean reports 0.01 -- a clean score from a metric
        # that failed half the grid.
        assert "tarp_error" not in result

    def test_an_invalidated_metric_is_not_recomputed(self, sim_2pl, adapter_2pl):
        calls = {"n": 0}

        def _always_fail(inputs):
            calls["n"] += 1
            raise RuntimeError("boom")

        self._hooks(sim_2pl, adapter_2pl, _always_fail).validate_fn(
            _mock_2pl_approx(), None, 8
        )
        # Two corners, but the second must not pay for a value already known
        # to be discarded.
        assert calls["n"] == 1

    def test_a_non_mapping_return_does_not_take_the_trial_down(
        self, sim_2pl, adapter_2pl
    ):
        result = self._hooks(sim_2pl, adapter_2pl, lambda inputs: 0.5).validate_fn(
            _mock_2pl_approx(), None, 8
        )
        # The marginal metrics the trial already computed survive.
        assert "tarp_error" not in result
        assert "nrmse" in result
        assert "calibration_error" in result

    def test_the_emitted_key_is_dropped_not_the_mapping_key(self, sim_2pl, adapter_2pl):
        """A joint metric's emitted key need not be its mapping key.

        `make_tarp_joint_metric()` with the provider forgotten emits
        `tarp_error_random` no matter which key it is registered under. If
        invalidation keyed on the mapping key alone, the computed value
        would survive under a name nothing asked for.
        """

        def _mismatched(inputs):
            if inputs.cond_id == 1:
                raise RuntimeError("boom")
            return {"tarp_error_random": 0.02}

        result = self._hooks(sim_2pl, adapter_2pl, _mismatched).validate_fn(
            _mock_2pl_approx(), None, 8
        )
        assert "tarp_error_random" not in result
        assert "tarp_error" not in result

    def test_an_emitted_key_is_reported_even_when_unlike_the_mapping_key(
        self, sim_2pl, adapter_2pl
    ):
        # The success side of the same mismatch: a value that was computed
        # must not be paid for and then dropped.
        result = self._hooks(
            sim_2pl,
            adapter_2pl,
            lambda inputs: {"tarp_error_random": 0.02},
            aggregate="mean",
        ).validate_fn(_mock_2pl_approx(), None, 8)
        assert result["tarp_error_random"] == pytest.approx(0.02)

    def test_mapping_key_never_emitted_is_not_fabricated(
        self, sim_2pl, adapter_2pl, caplog
    ):
        """The forgotten-provider case, end to end.

        `make_tarp_joint_metric()` without a provider emits
        `tarp_error_random` no matter which key it is registered under.
        Aggregating the mapping key would invent an all-NaN `tarp_error`,
        which Optuna reads as a failed trial -- so a naming slip would
        present as a broken model while the value that WAS computed got
        paid for and dropped.
        """
        from bayesflow_hpo.validation.tarp import make_tarp_joint_metric

        hooks = make_irt_hooks(
            sim_2pl,
            adapter_2pl,
            corners=[(3, 20)],
            param_keys=["a", "b"],
            sims_per_corner=4,
            aggregate="mean",
            joint_metrics={"tarp_error": make_tarp_joint_metric()},
        )
        with caplog.at_level(logging.WARNING, logger="bayesflow_irt.hpo"):
            result = hooks.validate_fn(_mock_2pl_approx(), None, 8)

        # Not fabricated as NaN...
        assert "tarp_error" not in result
        # ...and what actually ran is reported under the key it produced.
        assert "tarp_error_random" in result
        assert np.isfinite(result["tarp_error_random"])
        # And the mismatch is not silent.
        assert any(
            "emits ['tarp_error_random'], not 'tarp_error'" in r.message
            for r in caplog.records
        )

    def test_omitting_a_key_on_a_later_corner_invalidates(self, sim_2pl, adapter_2pl):
        """An omitted key is as quiet as a NaN, and just as wrong.

        Returning `{"tarp_error": 0.01}` on one corner and `{}` on the next
        reported `tarp_error/worst == 0.01` -- a clean worst-case guarantee
        derived from half the grid.
        """

        def _partial(inputs):
            return {"tarp_error": 0.01} if inputs.cond_id == 0 else {}

        result = self._hooks(sim_2pl, adapter_2pl, _partial).validate_fn(
            _mock_2pl_approx(), None, 8
        )
        assert "tarp_error" not in result
        assert "tarp_error/worst" not in result

    def test_returning_nothing_on_the_first_corner_invalidates(
        self, sim_2pl, adapter_2pl
    ):
        # No earlier condition to compare against, so the completeness check
        # cannot fire -- an empty mapping is simply no measurement.
        def _empty_first(inputs):
            return {} if inputs.cond_id == 0 else {"tarp_error": 0.01}

        result = self._hooks(sim_2pl, adapter_2pl, _empty_first).validate_fn(
            _mock_2pl_approx(), None, 8
        )
        assert "tarp_error" not in result

    def test_a_metric_emitting_every_key_on_every_corner_survives(
        self, sim_2pl, adapter_2pl
    ):
        # The completeness check must not fire on the ordinary case, nor on
        # a multi-output metric that reports all of its keys each time.
        result = self._hooks(
            sim_2pl,
            adapter_2pl,
            lambda inputs: {"tarp_error": 0.01, "tarp_error_random": 0.02},
            aggregate="mean",
        ).validate_fn(_mock_2pl_approx(), None, 8)
        assert result["tarp_error"] == pytest.approx(0.01)
        assert result["tarp_error_random"] == pytest.approx(0.02)


class TestJointMetricsOverrideMarginalResolution:
    """A name in `joint_metrics=` is never also resolved as a marginal one.

    `is_joint_metric` consults the registry only, so an ad hoc joint metric
    under an unregistered name stayed in the marginal list and
    `resolve_metrics` raised `KeyError` for it inside the closure --
    contradicting the documented promise that a metric passed through
    `joint_metrics=` runs whether or not its name appears in `metrics`.
    """

    def test_unregistered_name_in_both_lists_runs_as_joint(self, sim_2pl, adapter_2pl):
        hooks = make_irt_hooks(
            sim_2pl,
            adapter_2pl,
            corners=[(3, 20)],
            param_keys=["a", "b"],
            sims_per_corner=4,
            aggregate="mean",
            metrics=["nrmse", "probe"],
            joint_metrics={"probe": lambda inputs: {"probe": 0.5}},
        )
        result = hooks.validate_fn(_mock_2pl_approx(), None, 8)
        assert result["probe"] == pytest.approx(0.5)
        # The genuine marginal metric is unaffected.
        assert "nrmse" in result

    def test_an_unknown_name_that_is_not_overridden_still_raises(
        self, sim_2pl, adapter_2pl
    ):
        # The narrowing must not turn every typo into a silent no-op: a name
        # nothing supplies a callable for is still the marginal resolver's
        # business, and still raises.
        #
        # At CONSTRUCTION rather than on the first validate: raised inside
        # `GenericObjective`'s catch-all, the `KeyError` would become a
        # training-loss fallback, costing the whole study budget and
        # reporting a fabricated number for every trial instead of failing.
        with pytest.raises(KeyError, match="definitely_not_a_metric"):
            make_irt_hooks(
                sim_2pl,
                adapter_2pl,
                corners=[(3, 20)],
                param_keys=["a", "b"],
                sims_per_corner=4,
                metrics=["nrmse", "definitely_not_a_metric"],
            )


class TestJointTransformGuard:
    def test_non_positive_a_raises_naming_the_adapter(self):
        # With `log_transform_a=False` the adapter's inverse does not
        # exponentiate, so `a` draws reach zero and below. Emitting NaN here
        # would report a configuration mismatch as a failed trial.
        bad = np.array([[1.0, -0.5, 2.0]])
        with pytest.raises(ValueError, match="log_transform_a"):
            hpo_mod._joint_block(bad, "a", sample_axis=None)

    def test_finite_positive_a_is_unaffected(self):
        good = np.array([[1.0, 0.5, 2.0]])
        assert np.allclose(
            hpo_mod._joint_block(good, "a", sample_axis=None), np.log(good)
        )

    def test_already_non_finite_input_is_not_blamed_on_the_transform(self):
        # A NaN that arrived NaN is the metric's problem, not the
        # transform's; the guard compares against the finite mask.
        already = np.array([[1.0, np.nan, 2.0]])
        out = hpo_mod._joint_block(already, "a", sample_axis=None)
        assert np.isnan(out[0, 1])


class TestItemPooledTarp:
    """``tarp_error_item`` scores the same draws at ``d = len(param_keys)``."""

    def _inputs(self, n_sims=80, n_draws=60, n_items=4, seed=0):
        rng = np.random.default_rng(seed)
        truth = rng.normal(size=(n_sims, 2 * n_items))
        draws = truth[:, None, :] + rng.normal(size=(n_sims, n_draws, 2 * n_items))
        refs = rng.normal(size=(n_sims, 2 * n_items))
        return (
            SimpleNamespace(
                draws=draws,
                true_values=truth,
                cond_id=0,
                n_conditions=1,
                param_keys=("a", "b"),
                sim_batch={
                    "a": np.zeros((n_sims, n_items)),
                    "b": np.zeros((n_sims, n_items)),
                },
            ),
            refs,
        )

    def test_the_fold_pairs_each_item_across_the_parameter_blocks(self):
        """Positional pairing, in the layout ``_joint_row`` documents.

        Hand-encoded rather than round-tripped: a fold that mixed items or
        swapped the blocks would still produce the right *shape*, and every
        downstream number would look plausible.
        """
        truth = np.array(
            [[0, 1, 2, 10, 11, 12], [100, 101, 102, 110, 111, 112]], dtype=float
        )
        folded = hpo_mod._fold_items_into_rows(truth, 2)
        assert folded.shape == (6, 2)
        # Row 1 is simulation 0, item 1: [log a_1, b_1].
        assert np.array_equal(folded[1], [1.0, 11.0])
        # Row 3 is simulation 1, item 0.
        assert np.array_equal(folded[3], [100.0, 110.0])

        draws = np.arange(2 * 4 * 6, dtype=float).reshape(2, 4, 6)
        fd = hpo_mod._fold_items_into_rows(draws, 2)
        assert fd.shape == (6, 4, 2)
        assert np.array_equal(
            fd[1], np.stack([draws[0, :, 1], draws[0, :, 4]], axis=-1)
        )

    def test_a_column_count_that_does_not_divide_is_refused(self):
        with pytest.raises(ValueError, match="divisible"):
            hpo_mod._fold_items_into_rows(np.zeros((3, 7)), 2)

    def test_a_reordered_joint_row_is_refused_not_silently_mispaired(self):
        """``["b", "a"]`` has the right shape and the wrong meaning.

        The fold and the reference provider both slice positionally, so a
        swapped order scores every reference coordinate against the wrong
        posterior parameter and still returns a plausible number.
        """
        inputs, refs = self._inputs()
        inputs.param_keys = ("b", "a")
        fn = hpo_mod.make_tarp_item_metric(["a", "b"], reference_points=refs)
        with pytest.raises(ValueError, match="assembled in the order"):
            fn(inputs)

    def test_a_vector_valued_parameter_block_is_refused(self):
        """GPCM ``delta`` carries item *and* category columns.

        The total can still divide by ``n_keys``, in which case the fold cuts
        through the middle of the real blocks.
        """
        n_sims, n_items, n_cat = 6, 2, 3
        inputs, _ = self._inputs(n_sims=n_sims, n_items=n_items)
        inputs.param_keys = ("a", "delta")
        inputs.sim_batch = {
            "a": np.zeros((n_sims, n_items)),
            "delta": np.zeros((n_sims, n_items, n_cat)),
        }
        refs = np.zeros((n_sims, n_items + n_items * n_cat))
        fn = hpo_mod.make_tarp_item_metric(["a", "delta"], reference_points=refs)
        with pytest.raises(ValueError, match="more than an item axis"):
            fn(inputs)

    def test_absent_evidence_refuses_rather_than_skips_the_check(self):
        """Passing when the evidence is missing is the wrong failure mode."""
        inputs, refs = self._inputs()
        inputs.param_keys = ()
        fn = hpo_mod.make_tarp_item_metric(["a", "b"], reference_points=[refs])
        with pytest.raises(ValueError, match="carry no `param_keys`"):
            fn(inputs)

    def test_a_single_vector_valued_key_is_refused(self):
        """Equal widths pass trivially when there is nothing to compare to."""
        n_sims, n_items, n_cat = 6, 2, 3
        inputs, _ = self._inputs(n_sims=n_sims, n_items=n_items)
        inputs.param_keys = ("delta",)
        inputs.sim_batch = {"delta": np.zeros((n_sims, n_items, n_cat))}
        refs = np.zeros((n_sims, n_items * n_cat))
        fn = hpo_mod.make_tarp_item_metric(["delta"], reference_points=[refs])
        with pytest.raises(ValueError, match="more than an item axis"):
            fn(inputs)

    def test_matching_scalar_blocks_are_accepted(self):
        """The guard must not reject the layout it was built for."""
        n_sims, n_items = 40, 4
        inputs, refs = self._inputs(n_sims=n_sims, n_items=n_items)
        inputs.param_keys = ("a", "b")
        inputs.sim_batch = {
            "a": np.zeros((n_sims, n_items)),
            "b": np.zeros((n_sims, n_items)),
        }
        fn = hpo_mod.make_tarp_item_metric(["a", "b"], reference_points=[refs])
        assert "tarp_error_item" in fn(inputs)

    def test_it_emits_its_own_key(self):
        """Not ``tarp_error``: two statistics under one name are unreadable."""
        inputs, refs = self._inputs()
        fn = hpo_mod.make_tarp_item_metric(["a", "b"], reference_points=lambda _: refs)
        assert set(fn(inputs)) == {"tarp_error_item"}

    def test_it_differs_from_the_joint_metric_on_the_same_draws(self):
        """Same draws, same reference -- only the pooling differs.

        Pinned because the cheapest wrong implementation is one that folds
        nothing and quietly re-reports the joint number under a new key.
        """
        tarp = pytest.importorskip("bayesflow_hpo.validation.tarp")
        inputs, refs = self._inputs()
        joint = tarp.make_tarp_joint_metric(reference_points=lambda _: refs)(inputs)
        item = hpo_mod.make_tarp_item_metric(
            ["a", "b"], reference_points=lambda _: refs
        )(inputs)
        assert item["tarp_error_item"] != joint["tarp_error"]

    def test_an_empty_param_key_list_is_refused(self):
        with pytest.raises(ValueError, match="param_keys"):
            hpo_mod.make_tarp_item_metric([], reference_points=lambda _: None)


class TestPerMetricAggregation:
    """`aggregate` as a mapping: one reduction per metric.

    Different metrics can need different reductions: a worst case suits a
    target that is attainable at every data condition, while a metric
    bounded by the information in the data may be better averaged.
    """

    def _hooks(self, sim_2pl, adapter_2pl, aggregate):
        return make_irt_hooks(
            sim_2pl,
            adapter_2pl,
            corners=[(3, 20), (5, 30)],
            param_keys=["a", "b"],
            metrics=["calibration_error", "nrmse"],
            sims_per_corner=4,
            aggregate=aggregate,
        )

    def test_all_three_reductions_are_always_reported(self, sim_2pl, adapter_2pl):
        hooks = self._hooks(sim_2pl, adapter_2pl, "worst")
        result = hooks.validate_fn(_mock_2pl_approx(), None, 8)
        for suffix in ("mean", "worst", "geometric"):
            assert f"nrmse/{suffix}" in result, suffix

    def test_a_mapping_selects_per_metric(self, sim_2pl, adapter_2pl):
        hooks = self._hooks(sim_2pl, adapter_2pl, {"nrmse": "geometric"})
        result = hooks.validate_fn(_mock_2pl_approx(), None, 8)
        assert result["nrmse"] == result["nrmse/geometric"]
        # Unnamed metrics keep "worst", NOT bayesflow_hpo's "mean" default.
        assert result["calibration_error"] == result["calibration_error/worst"]

    def test_the_reduction_runs_over_the_parameter_by_corner_grid(
        self, sim_2pl, adapter_2pl
    ):
        """Not over corners alone.

        This factory builds one row per (corner, param_key) and never
        collapses the parameter axis, so `worst` is the worst CELL. Pinned
        because `bayesflow_hpo`'s own pipeline does average parameter types
        first, and a future refactor toward it would silently reintroduce
        the dilution -- a broken parameter hidden by a healthy one.
        """
        hooks = self._hooks(sim_2pl, adapter_2pl, "worst")
        result = hooks.validate_fn(_mock_2pl_approx(), None, 8)
        # 2 params x 2 corners. If the parameter axis were averaged away
        # first, the worst could not exceed the larger per-corner MEAN.
        assert result["nrmse/worst"] >= result["nrmse/mean"]
        assert result["nrmse/geometric"] <= result["nrmse/mean"]

    def test_geometric_mean_of_a_known_grid(self):
        """The reduction itself, where the answer is checkable by hand."""
        assert hpo_mod._geometric([1.0, 4.0]) == pytest.approx(2.0)
        assert hpo_mod._geometric([2.0, 2.0, 2.0]) == pytest.approx(2.0)

    def test_geometric_underflow_is_avoided(self):
        """A product of realistic metric values underflows float64."""
        values = [1e-40] * 8
        assert hpo_mod._geometric(values) == pytest.approx(1e-40)

    def test_a_non_positive_value_is_a_domain_failure_not_an_error(self):
        """NaN, not an exception.

        `correlation` and `contraction` are legitimately negative for an
        undertrained approximator, so raising would let one poor trial end an
        otherwise healthy study.
        """
        assert np.isnan(hpo_mod._geometric([1.0, -0.5]))
        assert np.isnan(hpo_mod._geometric([1.0, 0.0]))

    def test_an_unknown_reduction_is_refused(self, sim_2pl, adapter_2pl):
        with pytest.raises(ValueError, match="aggregate must be one of"):
            self._hooks(sim_2pl, adapter_2pl, "median")
        with pytest.raises(ValueError, match="must be one of"):
            self._hooks(sim_2pl, adapter_2pl, {"nrmse": "median"})

    def test_a_key_that_is_never_computed_is_refused(self, sim_2pl, adapter_2pl):
        """A silently-inert key leaves the metric on the default reduction."""
        with pytest.raises(ValueError, match="does not compute"):
            self._hooks(sim_2pl, adapter_2pl, {"nrsme": "geometric"})

    def test_a_non_mapping_non_string_is_refused(self, sim_2pl, adapter_2pl):
        with pytest.raises(ValueError, match="string or a mapping"):
            self._hooks(sim_2pl, adapter_2pl, ["worst"])

    def test_an_unregistered_key_does_not_break_the_inert_report(
        self, sim_2pl, adapter_2pl
    ):
        """The inert-key path must raise its OWN error, not `TypeError`.

        `producer_for_key` returns `None` for a key the registry never
        registered -- a joint metric's emitted key is exactly that -- and
        feeding that `None` into a set that is then sorted makes
        `sorted({"nrmse", None})` raise `TypeError`. `GenericObjective`
        re-raises only `JointMetricConfigurationError` and
        `AggregationConfigError`; a `TypeError` reaches its catch-all and
        becomes a training-loss fallback. So the one path written to STOP a
        misconfigured study instead scored every trial on a fabricated
        number.

        The typo must still be reported, and as the catchable type.
        """
        hooks = make_irt_hooks(
            sim_2pl,
            adapter_2pl,
            corners=CORNERS_SMALL,
            param_keys=["a", "b"],
            sims_per_corner=2,
            joint_metrics={"probe": lambda inputs: {"probe": 0.5}},
            aggregate={"probe": "mean", "nrsme": "geometric"},
        )
        with pytest.raises(AggregationConfigError, match="nrsme"):
            hooks.validate_fn(_mock_2pl_approx(), None, 6)

    def test_a_joint_metric_failing_this_trial_is_not_reported_as_inert(
        self, sim_2pl, adapter_2pl
    ):
        """A per-trial failure must not abort the whole study.

        A joint metric that raises or returns NaN has its keys dropped, so
        the mapping key naming it reduced nothing this trial. That is a
        property of the TRIAL -- `_joint_block` raises on non-finite draws,
        which an undertrained early trial produces routinely. Because
        `AggregationConfigError` is re-raised upstream, counting it as inert
        would let one bad trial kill the search.
        """

        def _fails(inputs):
            raise RuntimeError("this trial's draws were not finite")

        hooks = make_irt_hooks(
            sim_2pl,
            adapter_2pl,
            corners=CORNERS_SMALL,
            param_keys=["a", "b"],
            sims_per_corner=2,
            joint_metrics={"probe": _fails},
            aggregate={"probe": "mean"},
        )
        result = hooks.validate_fn(_mock_2pl_approx(), None, 6)
        assert "probe" not in result

    def test_an_unknown_marginal_metric_is_refused_at_construction(
        self, sim_2pl, adapter_2pl
    ):
        """Otherwise the `KeyError` lands inside the objective's catch-all.

        `_validate_fn` resolves the marginal names when it RUNS, and an
        unknown name raises `KeyError` there -- which `GenericObjective`
        converts into a training-loss fallback, so one typo costs the whole
        study budget and reports a fabricated number for every trial. The
        joint names were already checked at construction; this closes the
        same gap on the marginal side.

        `mean_cal_error` is chosen deliberately: it is a real EMITTED column
        of `coverage`, and since results are recorded under emitted names
        it is the natural mistake to make.
        """
        with pytest.raises(KeyError, match="mean_cal_error"):
            make_irt_hooks(
                sim_2pl,
                adapter_2pl,
                corners=CORNERS_SMALL,
                metrics=["nrmse", "mean_cal_error"],
                sims_per_corner=2,
            )

    def test_a_mapping_works_with_the_default_metric_list(self, sim_2pl, adapter_2pl):
        """`metrics=None` must not make every mapping key look unknown.

        The key check needs the DEFAULTED metric list. Validating before the
        default was applied rejected this factory's own documented call,
        `make_irt_hooks(..., aggregate={"nrmse": "geometric"})` with
        `metrics` omitted, even though nrmse is in the default set and is
        computed.
        """
        hooks = make_irt_hooks(
            sim_2pl,
            adapter_2pl,
            corners=CORNERS_SMALL,
            param_keys=["a", "b"],
            sims_per_corner=2,
            aggregate={"nrmse": "geometric"},
        )
        assert hooks.validate_fn is not None

    def test_a_joint_metric_may_be_named_by_either_spelling(self, sim_2pl, adapter_2pl):
        """Registration key or emitted key -- both pass construction.

        A callable given through `joint_metrics=` does not reveal which key
        it emits until it runs, so once any joint metric is present the set
        of valid names is open and construction cannot reject on it.
        """
        tarp = pytest.importorskip("bayesflow_hpo.validation.tarp")
        for named in ("tarp_error", "tarp_error_random"):
            make_irt_hooks(
                sim_2pl,
                adapter_2pl,
                corners=CORNERS_SMALL,
                param_keys=["a", "b"],
                sims_per_corner=2,
                joint_metrics={
                    "tarp_error": tarp.make_tarp_joint_metric(
                        hpo_mod.make_classical_reference_provider(["a", "b"]),
                        reference_id="test-classical",
                    )
                },
                aggregate={named: "mean"},
            )

    def test_naming_the_registration_key_of_a_differently_emitting_metric_raises(
        self, sim_2pl, adapter_2pl
    ):
        """The mismatch case, which construction cannot catch.

        A TARP metric built without a reference provider emits
        `tarp_error_random` whatever key it is registered under. Naming
        `tarp_error` would otherwise be accepted and then reduce nothing,
        leaving the metric on the default reduction with nothing to show for
        it -- so the first trial fails instead.
        """
        tarp = pytest.importorskip("bayesflow_hpo.validation.tarp")
        hooks = make_irt_hooks(
            sim_2pl,
            adapter_2pl,
            corners=CORNERS_SMALL,
            param_keys=["a", "b"],
            sims_per_corner=2,
            joint_metrics={"tarp_error": tarp.make_tarp_joint_metric()},
            aggregate={"tarp_error": "mean"},
        )
        with pytest.raises(ValueError, match="produced no value"):
            hooks.validate_fn(_mock_2pl_approx(), None, 6)

    def test_a_study_level_misconfiguration_escapes_the_objective(
        self, sim_2pl, adapter_2pl
    ):
        """It must be `AggregationConfigError`, not a bare ValueError.

        `GenericObjective` re-raises `JointMetricConfigurationError` and
        `AggregationConfigError` from `validate_fn` and converts everything
        else into a training-loss fallback. A plain ValueError here would
        therefore not stop the study -- it would let every trial record a
        fabricated value behind a warning line, which is the failure this
        check exists to prevent, one layer deeper.

        Imported outright rather than skipped when absent. The `hpo` extra
        floors on bayesflow-hpo 0.5.0, which defines the class, so a version
        without it is not a supported install and a skip there would hide
        the very gap this test exists to close -- it would report green on
        exactly the version where the guarantee does not hold.
        """
        from bayesflow_hpo.validation.metrics import AggregationConfigError

        config_error = AggregationConfigError
        tarp = pytest.importorskip("bayesflow_hpo.validation.tarp")
        hooks = make_irt_hooks(
            sim_2pl,
            adapter_2pl,
            corners=CORNERS_SMALL,
            param_keys=["a", "b"],
            sims_per_corner=2,
            joint_metrics={"tarp_error": tarp.make_tarp_joint_metric()},
            aggregate={"tarp_error": "mean"},
        )
        with pytest.raises(config_error):
            hooks.validate_fn(_mock_2pl_approx(), None, 6)
