"""Loading checkpoints across the bayesflow attention rename.

The rename cannot be reproduced here by installing a second bayesflow, so the
tests build a real checkpoint with the installed version and then rewrite its
layer paths to the *other* spelling. That is exactly the file a differently
versioned machine would have written, because the two implementations hold the
same variables in the same order.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import h5py
import numpy as np
import pytest

from bayesflow_irt.compat import (
    _NEW_TO_OLD,
    _OLD_TO_NEW,
    SIZE_MOMENTS_CHANGED_AT,
    _rename_component,
    _rewrite,
    adapter_size_moments,
    apply_checkpoint_size_moments,
    legacy_log_size_moments,
    legacy_size_moments,
    load_weights_compat,
    resolve_checkpoint_size_moments,
    set_adapter_size_moments,
    weights_digest,
)
from bayesflow_irt.networks import MatrixEquivariantIRTSummary

pytest.importorskip("keras")
import keras  # noqa: E402

BATCH, N_PERSONS, N_ITEMS = 4, 12, 6

#: Moves the attention stack to a name no spelling recovers, whichever
#: spelling the installed bayesflow wrote.
_BOGUS = {
    "induced_set_attention_block": "definitely_not_an_attention_layer",
    "induced_set_attention": "definitely_not_an_attention_layer",
    "set_attention_block": "definitely_not_an_attention_layer",
    "set_attention": "definitely_not_an_attention_layer",
}


def _responses(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.binomial(1, 0.5, size=(BATCH, N_PERSONS, N_ITEMS)).astype(np.float32)


def _model(use_isab: bool = True) -> keras.Model:
    net = MatrixEquivariantIRTSummary(
        embed_dim=8,
        matrix_hidden_dim=8,
        num_matrix_blocks=1,
        num_outer_sabs=1,
        use_outer_isab=use_isab,
        num_outer_inducing_points=4,
        dropout=0.0,
    )
    model = keras.Sequential([net])
    model(_responses(), training=False)
    return model


def _paths(path) -> list[str]:
    out: list[str] = []
    with h5py.File(path, "r") as f:
        f.visititems(
            lambda n, o: out.append(n) if isinstance(o, h5py.Dataset) else None
        )
    return out


class TestRenameComponent:
    def test_maps_a_bare_name(self):
        assert (
            _rename_component("induced_set_attention_block", _OLD_TO_NEW)
            == "induced_set_attention"
        )

    def test_preserves_the_uniquifying_suffix(self):
        """Keras appends `_1`, `_2` for repeats within one container."""
        assert (
            _rename_component("set_attention_block_2", _OLD_TO_NEW) == "set_attention_2"
        )

    def test_leaves_unrelated_names_alone(self):
        for name in ("mab0", "output_projector", "matrix_blocks", "vars"):
            assert _rename_component(name, _OLD_TO_NEW) == name

    def test_round_trips(self):
        original = "induced_set_attention_block_3"
        moved = _rename_component(original, _OLD_TO_NEW)
        assert _rename_component(moved, _NEW_TO_OLD) == original


class TestLoadWeightsCompat:
    def test_a_matching_file_loads_directly(self, tmp_path):
        source = _model()
        path = tmp_path / "w.weights.h5"
        source.save_weights(str(path))

        target = _model()
        assert weights_digest(target) != weights_digest(source)
        assert load_weights_compat(target, path) == "direct"
        assert weights_digest(target) == weights_digest(source)

    @pytest.mark.parametrize("use_isab", [True, False])
    def test_a_file_in_the_other_spelling_still_loads(self, tmp_path, use_isab):
        """The case this module exists for: a checkpoint from the other version."""
        source = _model(use_isab=use_isab)
        written = tmp_path / "written.weights.h5"
        source.save_weights(str(written))

        # Rewrite to whichever spelling the installed bayesflow did NOT use.
        foreign = tmp_path / "foreign.weights.h5"
        for mapping in (_OLD_TO_NEW, _NEW_TO_OLD):
            if _rewrite(written, foreign, mapping):
                break
        else:  # pragma: no cover - would mean no attention stack was built
            pytest.fail("no attention layer found to rename")
        assert _paths(foreign) != _paths(written)

        target = _model(use_isab=use_isab)
        assert load_weights_compat(target, foreign).startswith("renamed:")
        assert weights_digest(target) == weights_digest(source)

    def test_an_unloadable_file_raises_rather_than_returning_quietly(self, tmp_path):
        """A swallowed failure would leave an untrained network reporting metrics."""
        path = tmp_path / "empty.weights.h5"
        with h5py.File(path, "w"):
            pass

        target = _model()
        before = weights_digest(target)
        with pytest.raises(RuntimeError, match="could not restore weights"):
            load_weights_compat(target, path)
        assert weights_digest(target) == before

    def test_reloading_the_same_weights_is_not_treated_as_a_failure(self, tmp_path):
        """An idempotent restore changes nothing and is still a restore.

        Success has to mean "`load_weights` did not raise", not "the values
        moved" -- otherwise saving and immediately reloading, or simply loading
        twice, would be reported as an unloadable checkpoint.
        """
        model = _model()
        path = tmp_path / "w.weights.h5"
        model.save_weights(str(path))
        expected = weights_digest(model)

        assert load_weights_compat(model, path) == "direct"
        assert weights_digest(model) == expected
        # And again, now that nothing can possibly change.
        assert load_weights_compat(model, path) == "direct"
        assert weights_digest(model) == expected

    def test_a_partially_matching_file_leaves_the_model_as_it_was(self, tmp_path):
        """Keras assigns as it walks and raises only at the end.

        So a checkpoint that matches the outer layers but not the attention
        subtree overwrites some weights before failing. The helper must hand
        back the model it was given, not a half-restored one.
        """
        source = _model()
        written = tmp_path / "written.weights.h5"
        source.save_weights(str(written))

        # Matches everything except the attention stack, whose paths are moved
        # to a name no spelling will recover.
        broken = tmp_path / "broken.weights.h5"
        assert _rewrite(written, broken, _BOGUS)

        target = _model()
        before = weights_digest(target)
        with pytest.raises(RuntimeError, match="rolled back"):
            load_weights_compat(target, broken)
        assert weights_digest(target) == before


class TestExternalStorageRejected:
    SENTINEL = np.array([1234.5, -6789.25, 42.0, 7.0], dtype="<f4")

    def _hostile(self, tmp_path):
        outside = tmp_path / "private.bin"
        outside.write_bytes(self.SENTINEL.tobytes())
        path = tmp_path / "hostile.weights.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset(
                "layers/set_attention_block/vars/0",
                shape=(4,),
                dtype="<f4",
                external=[(str(outside), 0, self.SENTINEL.nbytes)],
            )
        return path

    def test_rewrite_refuses_before_reading(self, tmp_path):
        path = self._hostile(tmp_path)
        destination = tmp_path / "out.weights.h5"
        with pytest.raises(ValueError, match="external"):
            _rewrite(path, destination, _OLD_TO_NEW)
        assert not destination.exists()

    def test_load_weights_compat_does_not_materialize_external_bytes(self, tmp_path):
        path = self._hostile(tmp_path)
        target = _model()
        before = weights_digest(target)
        with pytest.raises(ValueError, match="external"):
            load_weights_compat(target, path)
        assert weights_digest(target) == before
        assert not any(
            np.array_equal(keras.ops.convert_to_numpy(w).ravel()[:4], self.SENTINEL)
            for w in target.weights
        )

    def test_external_links_are_refused(self, tmp_path):
        other = tmp_path / "other.h5"
        with h5py.File(other, "w") as f:
            f.create_dataset("x", data=self.SENTINEL)
        path = tmp_path / "linked.weights.h5"
        with h5py.File(path, "w") as f:
            f["layers/set_attention_block"] = h5py.ExternalLink(str(other), "/")
        with pytest.raises(ValueError, match="external link"):
            _rewrite(path, tmp_path / "out.weights.h5", _OLD_TO_NEW)


class TestRewrite:
    def test_reports_when_nothing_changed(self, tmp_path):
        """A copy that renamed nothing is not worth retrying with."""
        model = _model(use_isab=True)
        source = tmp_path / "w.weights.h5"
        model.save_weights(str(source))

        # Renaming in the direction the file is already in is a no-op.
        forward = tmp_path / "a.weights.h5"
        backward = tmp_path / "b.weights.h5"
        changed_forward = _rewrite(source, forward, _OLD_TO_NEW)
        changed_backward = _rewrite(source, backward, _NEW_TO_OLD)
        assert changed_forward != changed_backward

    def test_preserves_every_dataset_value(self, tmp_path):
        model = _model()
        source = tmp_path / "w.weights.h5"
        model.save_weights(str(source))
        moved = tmp_path / "m.weights.h5"
        mapping = _OLD_TO_NEW if _rewrite(source, moved, _OLD_TO_NEW) else _NEW_TO_OLD
        if mapping is _NEW_TO_OLD:
            _rewrite(source, moved, _NEW_TO_OLD)

        with h5py.File(source, "r") as a, h5py.File(moved, "r") as b:
            originals = sorted(_paths(source))
            assert len(originals) == len(sorted(_paths(moved)))
            for name in originals:
                renamed = "/".join(
                    _rename_component(part, mapping) for part in name.split("/")
                )
                np.testing.assert_array_equal(a[name][()], b[renamed][()])


# ---------------------------------------------------------------------------
# Size-condition moments
# ---------------------------------------------------------------------------


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
