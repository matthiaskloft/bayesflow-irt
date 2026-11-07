"""Tests for the size-condition moment helpers in ``bayesflow_irt.compat``."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from bayesflow_irt.compat import (
    SIZE_MOMENTS_CHANGED_AT,
    adapter_size_moments,
    apply_checkpoint_size_moments,
    legacy_log_size_moments,
    legacy_size_moments,
    resolve_checkpoint_size_moments,
    set_adapter_size_moments,
)


def _bucketed_sim():
    from bayesflow_irt import make_2pl_simulator
    from bayesflow_irt.priors import Prior

    return make_2pl_simulator(
        n_items=(5, 50),
        n_persons=(10, 500),
        item_bucket=5,
        person_bucket=50,
        prior_a=Prior("lognormal", loc=0, scale=0.5),
        prior_b=Prior("normal", loc=0, scale=2.0),
        prior_theta=Prior("normal", loc=0, scale=1.0),
    )


def _assert_moments_close(actual, expected):
    assert set(actual) == set(expected)
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-6)


class TestLegacySizeMoments:
    def test_reference_config_values(self):
        m_items = legacy_log_size_moments((5, 50), 5)
        m_persons = legacy_log_size_moments((10, 500), 50)
        np.testing.assert_allclose(m_items, (3.3104026, 0.5422526), atol=1e-6)
        np.testing.assert_allclose(m_persons, (5.471099, 0.65226936), atol=1e-6)

    def test_unbucketed_uses_every_integer(self):
        mean, std = legacy_log_size_moments((2, 5), None)
        logs = np.log([2.0, 3.0, 4.0, 5.0])
        np.testing.assert_allclose((mean, std), (logs.mean(), logs.std()))

    def test_pinned_range_has_unit_std(self):
        assert legacy_log_size_moments((7, 7), None)[1] == 1.0

    def test_simulator_helper_uses_buckets(self):
        moments = legacy_size_moments(_bucketed_sim())
        np.testing.assert_allclose(
            moments["n_items"], (3.3104026, 0.5422526), atol=1e-6
        )
        np.testing.assert_allclose(
            moments["n_persons"], (5.471099, 0.65226936), atol=1e-6
        )


class TestAdapterSizeMoments:
    def test_round_trip_and_application(self):
        from bayesflow_irt import make_2pl_adapter

        sim = _bucketed_sim()
        adapter = make_2pl_adapter(sim)
        before = adapter_size_moments(adapter)
        assert set(before) == {"n_items", "n_persons"}
        assert all(isinstance(v, float) for pair in before.values() for v in pair)

        # Fill the transform_map cache so the reset is exercised.
        batch = sim.sample(4)
        adapter(batch)

        new = {"n_items": [1.5, 2.0], "n_persons": [4.0, 0.25]}
        set_adapter_size_moments(adapter, new)
        _assert_moments_close(adapter_size_moments(adapter), new)

        out = adapter(batch)
        cond = np.asarray(out["inference_conditions"])
        n_items = float(np.asarray(batch["n_items"]).ravel()[0])
        n_persons = float(np.asarray(batch["n_persons"]).ravel()[0])
        # Condition layout: n_items, then n_persons, are the last two columns.
        np.testing.assert_allclose(
            cond[0, -2], (np.log(n_items) - 1.5) / 2.0, rtol=1e-4
        )
        np.testing.assert_allclose(
            cond[0, -1], (np.log(n_persons) - 4.0) / 0.25, rtol=1e-4
        )

    def test_missing_transform_raises(self):
        from bayesflow_irt import make_2pl_adapter

        adapter = make_2pl_adapter(_bucketed_sim())
        with pytest.raises(ValueError, match="n_nothing"):
            set_adapter_size_moments(adapter, {"n_nothing": [0.0, 1.0]})


class TestResolveCheckpointSizeMoments:
    sim = None

    @classmethod
    def setup_class(cls):
        cls.sim = _bucketed_sim()

    def test_record_wins(self):
        record = {"adapter_size_moments": {"n_items": [1.0, 2.0]}}
        before = SIZE_MOMENTS_CHANGED_AT - timedelta(days=30)
        assert (
            resolve_checkpoint_size_moments(record, self.sim, before)
            == record["adapter_size_moments"]
        )

    def test_pre_cutoff_is_legacy(self):
        before = SIZE_MOMENTS_CHANGED_AT - timedelta(seconds=1)
        assert resolve_checkpoint_size_moments({}, self.sim, before) == (
            legacy_size_moments(self.sim)
        )

    def test_post_cutoff_or_unknown_is_none(self):
        after = SIZE_MOMENTS_CHANGED_AT + timedelta(seconds=1)
        assert resolve_checkpoint_size_moments({}, self.sim, after) is None
        assert resolve_checkpoint_size_moments({}, self.sim, None) is None

    def test_apply_modes(self):
        from bayesflow_irt import make_2pl_adapter

        adapter = make_2pl_adapter(self.sim)
        current = adapter_size_moments(adapter)
        old = datetime(2026, 10, 1, 12, 0)  # naive: read in the cutover's UTC offset
        assert (
            apply_checkpoint_size_moments(adapter, self.sim, "current", {}, old)
            == "current code"
        )
        assert adapter_size_moments(adapter) == current
        msg = apply_checkpoint_size_moments(adapter, self.sim, "auto", {}, old)
        assert msg.startswith("legacy size moments")
        _assert_moments_close(
            adapter_size_moments(adapter), legacy_size_moments(self.sim)
        )
        with pytest.raises(ValueError):
            apply_checkpoint_size_moments(adapter, self.sim, "bogus")
