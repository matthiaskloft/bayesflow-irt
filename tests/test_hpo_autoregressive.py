"""HPO hooks for the autoregressive item posterior and its per-item flow."""

import numpy as np
import pytest

from bayesflow_irt import Prior, make_2pl_adapter, make_2pl_simulator

pytest.importorskip("bayesflow_hpo", reason="bayesflow_hpo not installed")

import bayesflow as bf  # noqa: E402
from bayesflow_hpo import FlowMatchingSpace  # noqa: E402

from bayesflow_irt import hpo  # noqa: E402

# Access names through the module: tests/test_hpo.py reloads bayesflow_irt.hpo,
# which would leave directly imported classes stale.

CORNERS = [(3, 20), (5, 30)]
BOTH = ["coupling_flow", "flow_matching"]
_NETWORKS = {
    "coupling_flow": bf.networks.CouplingFlow,
    "flow_matching": bf.networks.FlowMatching,
}


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


def _hooks(sim, adapter, **kwargs):
    return hpo.make_irt_hooks(
        sim, adapter, corners=CORNERS, sims_per_corner=2, **kwargs
    )


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


@pytest.mark.parametrize(
    ("space", "prefix"),
    [
        ("AutoregressiveCouplingFlowSpace", "cf_"),
        ("AutoregressiveFlowMatchingSpace", "fm_"),
    ],
)
def test_space_dimensions(space, prefix):
    names = {d.name for d in getattr(hpo, space)().dimensions}
    assert any(n.startswith(prefix) for n in names)
    assert {
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
    assert getattr(hpo, space).draws_summary_kwargs(params) == params


def test_default_is_coupling_flow(sim_2pl, adapter_2pl):
    hooks = _hooks(sim_2pl, adapter_2pl)
    assert isinstance(
        hooks.search_space.inference_space, hpo.AutoregressiveCouplingFlowSpace
    )
    model = hooks.build_fn(
        _hparams(hooks, draws_summary_pooling="attention", draws_summary_hidden_dim=32)
    )
    assert isinstance(model.inference_network, bf.networks.CouplingFlow)
    assert model.draws_summary_pooling == "attention"
    assert model.draws_summary_hidden_dim == 32

    data = sim_2pl.sample((2,))
    model.build_from_data(adapter_2pl(data))
    out = model.sample(num_samples=3, conditions=data)
    assert out["a"].shape == (2, 3, data["a"].shape[-1])
    assert np.all(np.isfinite(out["a"]))


def test_flow_matching_builds_and_samples(sim_2pl, adapter_2pl):
    hooks = _hooks(sim_2pl, adapter_2pl, flow="flow_matching")
    assert isinstance(
        hooks.search_space.inference_space, hpo.AutoregressiveFlowMatchingSpace
    )
    model = hooks.build_fn(_hparams(hooks, draws_summary_pooling="mean_std"))
    assert isinstance(model.inference_network, bf.networks.FlowMatching)
    assert model.draws_summary_pooling == "mean_std"

    data = sim_2pl.sample((2,))
    model.build_from_data(adapter_2pl(data))
    out = model.sample(
        num_samples=2,
        conditions=data,
        flow_sample_kwargs={"method": "euler", "steps": 2},
    )
    assert np.all(np.isfinite(out["b"]))


@pytest.mark.parametrize(
    ("flow", "space"),
    [
        ("coupling_flow", FlowMatchingSpace),
        ("flow_matching", FlowMatchingSpace),
        ("flow_matching", lambda: hpo.AutoregressiveCouplingFlowSpace()),
    ],
)
def test_rejects_a_space_without_the_flow(sim_2pl, adapter_2pl, flow, space):
    with pytest.raises(TypeError, match="needs an Autoregressive"):
        _hooks(sim_2pl, adapter_2pl, flow=flow, inference_space=space())


def test_rejects_ode_kwargs_for_coupling_flow(sim_2pl, adapter_2pl):
    with pytest.raises(ValueError, match="flow_sample_kwargs"):
        _hooks(sim_2pl, adapter_2pl, flow_sample_kwargs={"method": "euler", "steps": 8})


@pytest.mark.parametrize("bad", ["nope", [], BOTH[:1] * 2])
def test_flow_is_validated(sim_2pl, adapter_2pl, bad):
    with pytest.raises(ValueError, match="flow must be"):
        _hooks(sim_2pl, adapter_2pl, flow=bad)


@pytest.mark.parametrize(
    "pooling", ["mean", "mean_std", "mean_max", "attention", "pma"]
)
@pytest.mark.parametrize("hidden", [16, 64])
def test_build_every_pooling_at_the_width_extremes(
    sim_2pl, adapter_2pl, pooling, hidden
):
    hooks = _hooks(sim_2pl, adapter_2pl)
    model = hooks.build_fn(
        _hparams(hooks, draws_summary_pooling=pooling, draws_summary_hidden_dim=hidden)
    )
    data = sim_2pl.sample((2,))
    model.build_from_data(adapter_2pl(data))
    out = model.sample(num_samples=2, conditions=data)
    assert np.all(np.isfinite(out["b"]))


@pytest.mark.parametrize(
    "space", ["AutoregressiveCouplingFlowSpace", "AutoregressiveFlowMatchingSpace"]
)
def test_heads_must_divide_the_attention_width(space):
    from bayesflow_hpo.search_spaces.base import CategoricalDimension, IntDimension

    cls = getattr(hpo, space)
    with pytest.raises(ValueError, match="multiple of"):
        cls(draws_summary_num_heads=IntDimension("draws_summary_num_heads", constant=3))
    # no attention pooling, no constraint
    cls(
        draws_summary_pooling=CategoricalDimension(
            "draws_summary_pooling", constant="mean"
        ),
        draws_summary_num_heads=IntDimension("draws_summary_num_heads", constant=3),
    )


@pytest.mark.slow
@pytest.mark.parametrize("flow", BOTH)
def test_pipeline_check(sim_2pl, adapter_2pl, flow):
    """Dry run build -> compile -> train -> validate for each flow."""
    _hooks(sim_2pl, adapter_2pl, flow=flow, run_pipeline_check=True)


# --- flow=[...]: the per-item flow as a search dimension --------------------


def _choice_trial_params(choice: str, **space_kwargs) -> dict:
    """Params Optuna samples from `FlowChoiceSpace` for one choice."""
    import optuna

    space = hpo.FlowChoiceSpace(**space_kwargs)
    study = optuna.create_study()
    study.enqueue_trial({"flow": choice})
    return space.sample(study.ask())


def test_choice_space_samples_only_the_chosen_flow():
    cf = _choice_trial_params("coupling_flow")
    fm = _choice_trial_params("flow_matching")
    assert cf["flow"] == "coupling_flow"
    assert fm["flow"] == "flow_matching"
    assert any(k.startswith("cf_") for k in cf)
    assert not any(k.startswith("fm_") for k in cf)
    assert any(k.startswith("fm_") for k in fm)
    assert not any(k.startswith("cf_") for k in fm)
    for params in (cf, fm):
        assert any(k.startswith("draws_summary_") for k in params)


def test_choice_space_lists_shared_dimensions_once():
    names = [d.name for d in hpo.FlowChoiceSpace().dimensions]
    assert len(names) == len(set(names))
    assert names[0] == "flow"


def test_choice_space_searches_only_the_listed_flows():
    space = hpo.FlowChoiceSpace(flows=("flow_matching",))
    names = {d.name for d in space.dimensions}
    assert not any(n.startswith("cf_") for n in names)
    params = _choice_trial_params("flow_matching", flows=("flow_matching",))
    assert params["flow"] == "flow_matching"


@pytest.mark.parametrize("bad", [[], ["flow_matching", "flow_matching"], ["nope"]])
def test_flow_list_is_validated(bad):
    with pytest.raises(ValueError, match="flow must be"):
        hpo.normalize_flow(bad)


def test_one_element_list_pins_the_flow(sim_2pl, adapter_2pl):
    hooks = _hooks(sim_2pl, adapter_2pl, flow=["flow_matching"])
    assert isinstance(
        hooks.search_space.inference_space, hpo.AutoregressiveFlowMatchingSpace
    )


def test_list_must_match_the_choice_space_flows(sim_2pl, adapter_2pl):
    for flows in (("flow_matching",), tuple(BOTH[::-1])):
        with pytest.raises(ValueError, match="does not match"):
            _hooks(
                sim_2pl,
                adapter_2pl,
                flow=BOTH,
                inference_space=hpo.FlowChoiceSpace(flows=flows),
            )


def test_choice_space_rejects_a_space_without_draws_summary():
    with pytest.raises(TypeError, match="AutoregressiveFlowMatchingSpace"):
        hpo.FlowChoiceSpace(flow_matching=FlowMatchingSpace())


def test_list_builds_either_flow(sim_2pl, adapter_2pl):
    hooks = _hooks(sim_2pl, adapter_2pl, flow=BOTH)
    assert isinstance(hooks.search_space.inference_space, hpo.FlowChoiceSpace)
    for choice in BOTH:
        model = hooks.build_fn(_hparams(hooks, flow=choice))
        assert isinstance(model.inference_network, _NETWORKS[choice])


def test_list_applies_ode_kwargs_to_flow_matching_only(sim_2pl, adapter_2pl):
    ode = {"method": "euler", "steps": 2}
    hooks = _hooks(
        sim_2pl,
        adapter_2pl,
        flow=BOTH,
        flow_sample_kwargs=ode,
        metrics=["calibration_error"],
    )
    seen = {}
    for choice in BOTH:
        model = hooks.build_fn(_hparams(hooks, flow=choice))
        model.build_from_data(adapter_2pl(sim_2pl.sample((2,))))
        original = model.sample

        def recording(*args, _choice=choice, _orig=original, **kwargs):
            seen[_choice] = kwargs.get("flow_sample_kwargs")
            return _orig(*args, **kwargs)

        model.sample = recording
        hooks.validate_fn(model, None, 4)
    assert seen == {"coupling_flow": None, "flow_matching": ode}


def test_list_rejects_a_single_flow_space(sim_2pl, adapter_2pl):
    with pytest.raises(TypeError, match="FlowChoiceSpace"):
        _hooks(
            sim_2pl,
            adapter_2pl,
            flow=BOTH,
            inference_space=hpo.AutoregressiveFlowMatchingSpace(),
        )
    with pytest.raises(TypeError, match="needs an Autoregressive"):
        _hooks(sim_2pl, adapter_2pl, inference_space=hpo.FlowChoiceSpace())


def test_pipeline_check_dry_runs_each_flow(sim_2pl, adapter_2pl, monkeypatch):
    """`check_pipeline` samples with a first-choice mock, so each flow is pinned."""
    import optuna

    seen = []

    def _capture(**kwargs):
        space = kwargs["search_space"]
        params = space.sample(optuna.create_study().ask())
        seen.append(params["flow"])

    monkeypatch.setattr(hpo, "check_pipeline", _capture)
    _hooks(sim_2pl, adapter_2pl, flow=BOTH, run_pipeline_check=True)
    assert seen == BOTH


def test_pipeline_check_runs_once_for_a_single_flow(sim_2pl, adapter_2pl, monkeypatch):
    calls = []
    monkeypatch.setattr(hpo, "check_pipeline", lambda **kw: calls.append(kw))
    _hooks(sim_2pl, adapter_2pl, run_pipeline_check=True)
    assert len(calls) == 1


@pytest.mark.slow
def test_pipeline_check_trains_both_flows(sim_2pl, adapter_2pl):
    hooks = hpo.make_irt_hooks(
        sim_2pl,
        adapter_2pl,
        corners=[(3, 20)],
        sims_per_corner=2,
        flow=BOTH,
        run_pipeline_check=True,
    )
    assert isinstance(hooks, hpo.IRTHooks)
