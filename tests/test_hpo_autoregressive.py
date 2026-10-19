"""HPO hooks for ``item_posterior="autoregressive_coupling_flow"``."""

import numpy as np
import pytest

from bayesflow_irt import Prior, make_2pl_adapter, make_2pl_simulator

pytest.importorskip("bayesflow_hpo", reason="bayesflow_hpo not installed")

import bayesflow as bf  # noqa: E402
from bayesflow_hpo import FlowMatchingSpace  # noqa: E402

from bayesflow_irt import hpo  # noqa: E402

# Access names through the module: tests/test_hpo.py reloads bayesflow_irt.hpo,
# which would leave directly imported classes stale.

_AR = "autoregressive_coupling_flow"
CORNERS = [(3, 20), (5, 30)]


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


def _hparams(hooks, **override) -> dict:
    """Every dimension at its first choice / low end / constant, then overrides."""
    params: dict = {}
    for sub in ("summary_space", "inference_space", "training_space"):
        for dim in getattr(hooks.search_space, sub).dimensions:
            choices = getattr(dim, "choices", None)
            low = getattr(dim, "low", None)
            constant = getattr(dim, "constant", None)
            if choices:
                params[dim.name] = list(choices)[0]
            elif isinstance(low, (int, float)):
                params[dim.name] = low
            else:
                params[dim.name] = constant
    params.update(override)
    return params


def test_space_dimensions():
    names = {d.name for d in hpo.AutoregressiveCouplingFlowSpace().dimensions}
    assert {
        "cf_depth",
        "draws_summary_pooling",
        "draws_summary_hidden_dim",
        "draws_summary_dim",
        "draws_summary_num_heads",
    } <= names
    params = {
        "draws_summary_pooling": "attention",
        "draws_summary_hidden_dim": 32,
        "draws_summary_dim": 16,
        "draws_summary_num_heads": 4,
    }
    assert hpo.AutoregressiveCouplingFlowSpace.draws_summary_kwargs(params) == params


def test_default_space_and_build(sim_2pl, adapter_2pl):
    hooks = hpo.make_irt_hooks(
        sim_2pl, adapter_2pl, corners=CORNERS, sims_per_corner=2, item_posterior=_AR
    )
    assert isinstance(
        hooks.search_space.inference_space, hpo.AutoregressiveCouplingFlowSpace
    )
    model = hooks.build_fn(
        _hparams(hooks, draws_summary_pooling="attention", draws_summary_hidden_dim=32)
    )
    assert model.item_posterior == _AR
    assert isinstance(model.inference_network, bf.networks.CouplingFlow)
    assert model.draws_summary_pooling == "attention"
    assert model.draws_summary_hidden_dim == 32

    data = sim_2pl.sample((2,))
    model.build_from_data(adapter_2pl(data))
    out = model.sample(num_samples=3, conditions=data)
    assert out["a"].shape == (2, 3, data["a"].shape[-1])
    assert np.all(np.isfinite(out["a"]))


def test_default_stays_flow_matching(sim_2pl, adapter_2pl):
    hooks = hpo.make_irt_hooks(sim_2pl, adapter_2pl, corners=CORNERS, sims_per_corner=2)
    assert isinstance(hooks.search_space.inference_space, FlowMatchingSpace)
    assert hooks.build_fn(_hparams(hooks)).item_posterior == "flow_matching"


def test_rejects_wrong_space(sim_2pl, adapter_2pl):
    with pytest.raises(TypeError, match="AutoregressiveCouplingFlowSpace"):
        hpo.make_irt_hooks(
            sim_2pl,
            adapter_2pl,
            corners=CORNERS,
            sims_per_corner=2,
            item_posterior=_AR,
            inference_space=FlowMatchingSpace(),
        )


def test_rejects_ode_kwargs(sim_2pl, adapter_2pl):
    with pytest.raises(ValueError, match="flow_sample_kwargs"):
        hpo.make_irt_hooks(
            sim_2pl,
            adapter_2pl,
            corners=CORNERS,
            sims_per_corner=2,
            item_posterior=_AR,
            flow_sample_kwargs={"method": "euler", "steps": 8},
        )


def test_rejects_unknown_item_posterior(sim_2pl, adapter_2pl):
    with pytest.raises(ValueError, match="item_posterior"):
        hpo.make_irt_hooks(
            sim_2pl,
            adapter_2pl,
            corners=CORNERS,
            sims_per_corner=2,
            item_posterior="autoregressive",
        )


def test_rejects_ar_space_with_flow_matching(sim_2pl, adapter_2pl):
    with pytest.raises(TypeError, match="autoregressive_coupling_flow"):
        hpo.make_irt_hooks(
            sim_2pl,
            adapter_2pl,
            corners=CORNERS,
            sims_per_corner=2,
            inference_space=hpo.AutoregressiveCouplingFlowSpace(),
        )


@pytest.mark.parametrize(
    "pooling", ["mean", "mean_std", "mean_max", "attention", "pma"]
)
@pytest.mark.parametrize("hidden", [16, 64])
def test_build_every_pooling_at_the_width_extremes(
    sim_2pl, adapter_2pl, pooling, hidden
):
    hooks = hpo.make_irt_hooks(
        sim_2pl, adapter_2pl, corners=CORNERS, sims_per_corner=2, item_posterior=_AR
    )
    model = hooks.build_fn(
        _hparams(hooks, draws_summary_pooling=pooling, draws_summary_hidden_dim=hidden)
    )
    data = sim_2pl.sample((2,))
    model.build_from_data(adapter_2pl(data))
    out = model.sample(num_samples=2, conditions=data)
    assert np.all(np.isfinite(out["b"]))


def test_heads_must_divide_the_attention_width():
    from bayesflow_hpo.search_spaces.base import IntDimension

    with pytest.raises(ValueError, match="multiple of"):
        hpo.AutoregressiveCouplingFlowSpace(
            draws_summary_num_heads=IntDimension("draws_summary_num_heads", constant=3)
        )
    # no attention pooling, no constraint
    from bayesflow_hpo.search_spaces.base import CategoricalDimension

    hpo.AutoregressiveCouplingFlowSpace(
        draws_summary_pooling=CategoricalDimension(
            "draws_summary_pooling", constant="mean"
        ),
        draws_summary_num_heads=IntDimension("draws_summary_num_heads", constant=3),
    )


@pytest.mark.slow
def test_pipeline_check(sim_2pl, adapter_2pl):
    """Dry run build -> compile -> train -> validate on the AR path."""
    hpo.make_irt_hooks(
        sim_2pl,
        adapter_2pl,
        corners=CORNERS,
        sims_per_corner=2,
        item_posterior=_AR,
        run_pipeline_check=True,
    )


# --- item_posterior=[...]: the factorization as a search dimension ---------


def _choice_trial_params(choice: str) -> dict:
    """Params Optuna samples from `ItemPosteriorChoiceSpace` for one choice."""
    import optuna

    space = hpo.ItemPosteriorChoiceSpace()
    study = optuna.create_study()
    study.enqueue_trial({"item_posterior": choice})
    return space.sample(study.ask())


def test_choice_space_samples_only_the_chosen_family():
    fm = _choice_trial_params("flow_matching")
    ar = _choice_trial_params(_AR)
    assert fm["item_posterior"] == "flow_matching"
    assert ar["item_posterior"] == _AR
    assert any(k.startswith("fm_") for k in fm)
    assert not any(k.startswith(("cf_", "draws_summary_")) for k in fm)
    assert any(k.startswith("draws_summary_") for k in ar)
    assert not any(k.startswith("fm_") for k in ar)


BOTH = ["flow_matching", _AR]


def test_choice_space_searches_only_the_listed_families():
    space = hpo.ItemPosteriorChoiceSpace(families=("flow_matching",))
    names = {d.name for d in space.dimensions}
    assert not any(n.startswith(("cf_", "draws_summary_")) for n in names)
    assert _choice_trial_params("flow_matching")["item_posterior"] == "flow_matching"


@pytest.mark.parametrize("bad", [[], ["flow_matching", "flow_matching"], ["nope"]])
def test_item_posterior_list_is_validated(bad):
    with pytest.raises(ValueError, match="item_posterior"):
        hpo.normalize_item_posterior(bad)


def test_one_element_list_pins_the_family(sim_2pl, adapter_2pl):
    hooks = hpo.make_irt_hooks(
        sim_2pl, adapter_2pl, corners=CORNERS, sims_per_corner=2, item_posterior=[_AR]
    )
    assert isinstance(
        hooks.search_space.inference_space, hpo.AutoregressiveCouplingFlowSpace
    )


def test_list_must_match_the_choice_space_families(sim_2pl, adapter_2pl):
    with pytest.raises(ValueError, match="does not match"):
        hpo.make_irt_hooks(
            sim_2pl,
            adapter_2pl,
            corners=CORNERS,
            sims_per_corner=2,
            item_posterior=BOTH,
            inference_space=hpo.ItemPosteriorChoiceSpace(families=("flow_matching",)),
        )
    with pytest.raises(ValueError, match="does not match"):
        hpo.make_irt_hooks(
            sim_2pl,
            adapter_2pl,
            corners=CORNERS,
            sims_per_corner=2,
            item_posterior=BOTH,
            inference_space=hpo.ItemPosteriorChoiceSpace(families=tuple(BOTH[::-1])),
        )


def test_choice_space_rejects_a_non_ar_space():
    with pytest.raises(TypeError, match="AutoregressiveCouplingFlowSpace"):
        hpo.ItemPosteriorChoiceSpace(autoregressive_coupling_flow=FlowMatchingSpace())


def test_list_builds_either_family(sim_2pl, adapter_2pl):
    hooks = hpo.make_irt_hooks(
        sim_2pl, adapter_2pl, corners=CORNERS, sims_per_corner=2, item_posterior=BOTH
    )
    assert isinstance(hooks.search_space.inference_space, hpo.ItemPosteriorChoiceSpace)
    for choice, net in (
        ("flow_matching", bf.networks.FlowMatching),
        (_AR, bf.networks.CouplingFlow),
    ):
        model = hooks.build_fn(_hparams(hooks, item_posterior=choice))
        assert model.item_posterior == choice
        assert isinstance(model.inference_network, net)


def test_list_applies_ode_kwargs_to_flow_matching_only(sim_2pl, adapter_2pl):
    ode = {"method": "euler", "steps": 4}
    hooks = hpo.make_irt_hooks(
        sim_2pl,
        adapter_2pl,
        corners=CORNERS,
        sims_per_corner=2,
        item_posterior=BOTH,
        flow_sample_kwargs=ode,
        metrics=["calibration_error"],
    )
    seen = {}
    for choice in ("flow_matching", _AR):
        model = hooks.build_fn(_hparams(hooks, item_posterior=choice))
        model.build_from_data(adapter_2pl(sim_2pl.sample((2,))))
        original = model.sample

        def recording(*args, _choice=choice, _orig=original, **kwargs):
            seen[_choice] = kwargs.get("flow_sample_kwargs")
            return _orig(*args, **kwargs)

        model.sample = recording
        hooks.validate_fn(model, None, 4)
    assert seen == {"flow_matching": ode, _AR: None}


def test_list_rejects_a_single_family_space(sim_2pl, adapter_2pl):
    with pytest.raises(TypeError, match="ItemPosteriorChoiceSpace"):
        hpo.make_irt_hooks(
            sim_2pl,
            adapter_2pl,
            corners=CORNERS,
            sims_per_corner=2,
            item_posterior=BOTH,
            inference_space=FlowMatchingSpace(),
        )
    with pytest.raises(TypeError, match="two or more"):
        hpo.make_irt_hooks(
            sim_2pl,
            adapter_2pl,
            corners=CORNERS,
            sims_per_corner=2,
            inference_space=hpo.ItemPosteriorChoiceSpace(),
        )


def test_pipeline_check_dry_runs_each_family(sim_2pl, adapter_2pl, monkeypatch):
    """`check_pipeline` samples with a first-choice mock, so each family is pinned."""
    import optuna

    seen = []

    def _capture(**kwargs):
        space = kwargs["search_space"]
        params = space.sample(optuna.create_study().ask())
        seen.append(params["item_posterior"])

    monkeypatch.setattr(hpo, "check_pipeline", _capture)
    hpo.make_irt_hooks(
        sim_2pl,
        adapter_2pl,
        corners=CORNERS,
        sims_per_corner=2,
        item_posterior=BOTH,
        run_pipeline_check=True,
    )
    assert seen == ["flow_matching", _AR]


def test_pipeline_check_runs_once_for_a_single_family(
    sim_2pl, adapter_2pl, monkeypatch
):
    calls = []
    monkeypatch.setattr(hpo, "check_pipeline", lambda **kw: calls.append(kw))
    hpo.make_irt_hooks(
        sim_2pl,
        adapter_2pl,
        corners=CORNERS,
        sims_per_corner=2,
        item_posterior=_AR,
        run_pipeline_check=True,
    )
    assert len(calls) == 1


@pytest.mark.slow
def test_pipeline_check_trains_the_autoregressive_family(sim_2pl, adapter_2pl):
    hooks = hpo.make_irt_hooks(
        sim_2pl,
        adapter_2pl,
        corners=[(3, 20)],
        sims_per_corner=2,
        item_posterior=BOTH,
        run_pipeline_check=True,
    )
    assert isinstance(hooks, hpo.IRTHooks)
