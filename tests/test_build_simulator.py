"""Tests for :func:`bayesflow_irt.build_simulator`.

The builder rebuilds a model's training simulator from the metadata sidecar's
``model_spec``; a model card is only valid if that rebuild matches what the
model was trained on.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from bayesflow_irt import (
    MetadataTracker,
    build_simulator,
    describe_size_sampling,
    make_2pl_simulator,
)
from bayesflow_irt.priors import Prior

N01 = {"name": "normal", "loc": 0.0, "scale": 1.0}
LOGN = {"name": "lognormal", "loc": 0.0, "scale": 0.5}
SIZES = {"n_items": [5, 8], "n_persons": [20, 30]}

SPECS = {
    "1pl": {"model": "1PL", "priors": {"b": N01, "theta": N01}},
    "2pl": {"model": "2PL", "priors": {"a": LOGN, "b": N01, "theta": N01}},
    "pcm": {"model": "PCM", "n_categories": 4, "priors": {"delta": N01}},
    "gpcm": {
        "model": "GPCM",
        "n_categories": 3,
        "priors": {
            "a": LOGN,
            "delta_mean": N01,
            "delta_cat_sep": {"name": "normal", "loc": 2.0, "scale": 0.5},
            "theta": N01,
        },
    },
}


AFTER = "2026-08-01T00:00:00+00:00"  # a training time after the scaling change


def _spec(family: str, **extra) -> dict:
    return {**SPECS[family], "size_ranges": SIZES, **extra}


def _draws_equal(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)


class TestBuild:
    @pytest.mark.parametrize("family", sorted(SPECS))
    def test_each_family_builds_and_samples(self, family):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            draws = build_simulator(_spec(family), seed=0, trained_at=AFTER).sample(3)
        assert draws["responses"].shape[0] == 3

    def test_seed_fixes_the_draws(self):
        a = build_simulator(_spec("2pl"), seed=5).sample(4)
        b = build_simulator(_spec("2pl"), seed=5).sample(4)
        assert _draws_equal(a, b)

    def test_size_override(self):
        sim = build_simulator(_spec("2pl"), n_items=(7, 7), n_persons=(40, 40))
        assert sim.sample(2)["responses"].shape[1:3] in {(40, 7), (7, 40)}

    def test_matches_direct_factory(self):
        direct = make_2pl_simulator(
            n_items=(5, 8),
            n_persons=(20, 30),
            prior_a=Prior("lognormal", loc=0.0, scale=0.5),
            prior_b=Prior("normal", loc=0.0, scale=1.0),
            prior_theta=Prior("normal", loc=0.0, scale=1.0),
            seed=3,
        )
        built = build_simulator(_spec("2pl"), seed=3)
        assert _draws_equal(direct.sample(4), built.sample(4))

    def test_recorded_size_sampling_round_trips(self):
        trained = make_2pl_simulator(
            n_items=(5, 40),
            n_persons=(10, 500),
            prior_a=Prior("lognormal", loc=0.0, scale=0.5),
            prior_b=Prior("normal", loc=0.0, scale=1.0),
            item_bucket=5,
            person_bucket=50,
        )
        spec = {
            **SPECS["2pl"],
            "size_ranges": {"n_items": [5, 40], "n_persons": [10, 500]},
            "size_sampling": describe_size_sampling(trained),
        }
        rebuilt = build_simulator(spec)
        assert describe_size_sampling(rebuilt) == describe_size_sampling(trained)
        # Explicit sizes drop the training buckets unless asked for.
        assert (
            build_simulator(spec, n_items=(5, 40), n_persons=(10, 500)).item_bucket
            is None
        )
        assert (
            build_simulator(
                spec, n_items=(5, 40), n_persons=(10, 500), apply_size_sampling=True
            ).item_bucket
            == 5
        )


class TestRefusals:
    def test_custom_callable_prior(self):
        spec = _spec("2pl")
        spec["priors"] = {**spec["priors"], "a": "<function custom>"}
        with pytest.raises(ValueError, match="custom prior"):
            build_simulator(spec)

    def test_missing_prior(self):
        with pytest.raises(ValueError, match=r"no record of \['a'\]"):
            build_simulator({**_spec("2pl"), "priors": {"b": N01}})

    def test_unknown_prior_key(self):
        spec = _spec("1pl")
        spec["priors"] = {**spec["priors"], "a": LOGN}
        with pytest.raises(ValueError, match="does not take"):
            build_simulator(spec)

    def test_unknown_family(self):
        with pytest.raises(ValueError, match="expected one of"):
            build_simulator({**_spec("2pl"), "model": "3PL"})

    def test_polytomous_needs_categories(self):
        with pytest.raises(ValueError, match="n_categories"):
            build_simulator({**_spec("pcm"), "n_categories": None})

    @pytest.mark.parametrize("trained_at", [None, "2026-07-01T00:00:00+00:00", "?"])
    def test_incremental_prior_needs_a_recent_training_time(self, trained_at):
        with pytest.raises(ValueError, match="scaling change"):
            build_simulator(_spec("gpcm"), trained_at=trained_at)

    def test_single_delta_prior_needs_no_training_time(self):
        build_simulator(_spec("pcm"))

    def test_missing_theta_warns(self):
        spec = _spec("1pl")
        spec["priors"] = {"b": N01}
        with pytest.warns(UserWarning, match="Normal\\(0, 1\\)"):
            build_simulator(spec)


# --- Fixture sidecars --------------------------------------------------------

#: One v1 sidecar per family, a calibrated 2PL one and an HPO-only one.
SIDECARS = sorted(
    (Path(__file__).resolve().parent / "fixtures" / "sidecars").glob("*_metadata.json")
)
# HPO study sidecars (named ``hpo_*``) record no priors; they describe a
# search, not a model.
NO_PRIORS_PREFIX = "hpo_"


def check_rebuild(path):
    """The sidecar's ``model_spec`` rebuilds a simulator that samples."""
    data = MetadataTracker.load(path).to_dict()
    spec = data["model_spec"]
    trained_at = (data.get("provenance") or {}).get("timestamp")
    if Path(path).name.startswith(NO_PRIORS_PREFIX):
        with pytest.raises(ValueError, match="no record of"):
            build_simulator(spec)
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        draws = build_simulator(spec, seed=0, trained_at=trained_at).sample(2)
    assert draws["responses"].shape[0] == 2


@pytest.mark.parametrize("path", SIDECARS, ids=lambda p: Path(p).name)
def test_committed_sidecar_rebuilds(path):
    check_rebuild(path)
