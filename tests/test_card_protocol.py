"""Tests for the model-card protocol and cell planning (``card.py``)."""

import json

import numpy as np
import pytest

from bayesflow_irt.card import (
    PROTOCOL_VERSION,
    STANDARD_GRID,
    Cell,
    EvaluationProtocol,
    cell_seed,
    plan_cells,
)

RANGES_2PL = {"n_items": (5, 35), "n_persons": (5, 1000)}


def test_presets():
    val = EvaluationProtocol.preset()
    dev = EvaluationProtocol.preset("development")
    assert (val.preset_name, val.datasets_per_cell, val.include_out_of_range) == (
        "validation",
        1000,
        True,
    )
    assert (dev.datasets_per_cell, dev.posterior_draws) == (100, 500)
    assert not dev.include_out_of_range
    assert val.overrides == dev.overrides == ()


def test_overrides_are_recorded_only_when_they_change_a_value():
    p = EvaluationProtocol.preset(
        "validation", datasets_per_cell=1000, posterior_draws=300, seed=7
    )
    assert p.overrides == ("posterior_draws", "seed")
    assert p.posterior_draws == 300


def test_preset_validation():
    with pytest.raises(ValueError, match="Unknown preset"):
        EvaluationProtocol.preset("card")
    with pytest.raises(TypeError, match="Unknown protocol fields"):
        EvaluationProtocol.preset(n_draws=10)
    with pytest.raises(ValueError, match="datasets_per_cell"):
        EvaluationProtocol.preset(datasets_per_cell=0)


def test_to_dict_is_json_and_versioned():
    d = EvaluationProtocol.preset("development", seed=3).to_dict()
    assert d["protocol_version"] == PROTOCOL_VERSION
    assert d["grid"] == {k: list(v) for k, v in STANDARD_GRID.items()}
    assert d["overrides"] == ["seed"]
    assert json.loads(json.dumps(d)) == d


def test_plan_cells_2pl_example():
    cells = plan_cells(RANGES_2PL, EvaluationProtocol.preset())
    inside = [(c.n_items, c.n_persons) for c in cells if c.in_range]
    outside = {(c.n_items, c.n_persons) for c in cells if not c.in_range}
    # Grid values inside the range plus the exact bounds (5 persons).
    assert sorted({i for i, _ in inside}) == [5, 10, 20, 35]
    assert sorted({p for _, p in inside}) == [5, 10, 25, 50, 100, 250, 500, 1000]
    assert len(inside) == 32
    # Edges (one axis out) plus corners (both out): 2*8 + 4*1 + 2*1.
    assert len(outside) == 22
    assert {(3, 5000), (50, 5000)} <= outside
    assert (3, 10) in outside and (20, 5000) in outside
    assert all(c.in_range for c in cells[:32])


def test_out_of_range_switch():
    cells = plan_cells(RANGES_2PL, EvaluationProtocol.preset("development"))
    assert len(cells) == 32 and all(c.in_range for c in cells)


def test_range_at_the_grid_edge_has_no_cell_beyond_it():
    cells = plan_cells(
        {"n_items": (3, 50), "n_persons": (10, 5000)}, EvaluationProtocol.preset()
    )
    assert all(c.in_range for c in cells)
    assert len(cells) == 6 * 8


def test_bounds_off_the_grid_are_cells():
    cells = plan_cells(
        {"n_items": (7, 7), "n_persons": (60, 60)},
        EvaluationProtocol.preset(include_out_of_range=False),
    )
    assert cells == [Cell(7, 60, True)]


def test_inverted_range_raises():
    with pytest.raises(ValueError, match="lower bound"):
        plan_cells(
            {"n_items": (10, 5), "n_persons": (5, 10)}, EvaluationProtocol.preset()
        )


def test_cell_seed_is_keyed_by_sizes_not_position():
    full = plan_cells(RANGES_2PL, EvaluationProtocol.preset())
    trained = plan_cells(RANGES_2PL, EvaluationProtocol.preset("development"))
    seeds_full = {(c.n_items, c.n_persons): cell_seed(0, c) for c in full}
    for c in trained:
        assert cell_seed(0, c) == seeds_full[(c.n_items, c.n_persons)]
    assert len(set(seeds_full.values())) == len(seeds_full)
    assert cell_seed(1, full[0]) != cell_seed(0, full[0])


def test_override_types_are_checked():
    with pytest.raises(TypeError, match="posterior_draws must be an int"):
        EvaluationProtocol.preset("development", posterior_draws=500.0)
    with pytest.raises(TypeError, match="must be a bool"):
        EvaluationProtocol.preset(include_out_of_range=1)
    with pytest.raises(TypeError, match="seed must be an int"):
        EvaluationProtocol.preset(seed=True)
    with pytest.raises(ValueError, match="seed must be >= 0"):
        EvaluationProtocol.preset(seed=-1)
    p = EvaluationProtocol.preset(seed=np.int64(4))
    assert type(p.seed) is int and p.overrides == ("seed",)


def test_seed_rule_is_stored():
    assert (
        "SeedSequence([seed, n_items, n_persons])"
        in (EvaluationProtocol.preset().to_dict()["seed_rule"])
    )


def test_equal_bounds_and_bounds_beyond_the_grid():
    proto = EvaluationProtocol.preset()
    cells = plan_cells({"n_items": (5, 5), "n_persons": (2, 6000)}, proto)
    inside = {(c.n_items, c.n_persons) for c in cells if c.in_range}
    outside = {(c.n_items, c.n_persons) for c in cells if not c.in_range}
    assert {i for i, _ in inside} == {5}
    assert {p for _, p in inside} == {2, 10, 25, 50, 100, 250, 500, 1000, 5000, 6000}
    # Items 3 and 10 are the next grid values out; no persons value is out.
    assert {i for i, _ in outside} == {3, 10}
    assert {p for _, p in outside} == {p for _, p in inside}
