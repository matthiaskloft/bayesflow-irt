"""Tests for bayesflow_irt.simulators."""

import numpy as np
import pytest

from bayesflow_irt.priors import Prior, VariablePrior, make_variable_incremental_prior
from bayesflow_irt.simulators import (
    make_1pl_simulator,
    make_2pl_simulator,
    make_gpcm_simulator,
    make_pcm_simulator,
)

# Standard priors used across tests — must be specified explicitly by all callers.
_PRIOR_A = Prior("lognormal", loc=0, scale=0.5)
_PRIOR_B = Prior("normal", loc=0, scale=2.0)
_PRIOR_THETA = Prior("normal", loc=0, scale=1.0)
_PRIOR_DELTA = Prior("normal", loc=0, scale=2.0)


class TestMake2PLSimulator:
    def test_variable_size(self):
        sim = make_2pl_simulator(
            n_items=(10, 14),
            n_persons=(30, 35),
            prior_a=_PRIOR_A,
            prior_b=_PRIOR_B,
            prior_theta=_PRIOR_THETA,
        )
        out = sim.sample((5,))

        n_items = int(out["n_items"][0, 0])
        n_persons = int(out["n_persons"][0, 0])

        assert out["a"].shape == (5, n_items)
        assert out["b"].shape == (5, n_items)
        assert out["theta"].shape == (5, n_persons)
        assert out["responses"].shape == (5, n_persons, n_items)
        assert out["n_items"].shape == (5, 1)
        assert out["n_persons"].shape == (5, 1)
        assert 10 <= n_items <= 14
        assert 30 <= n_persons <= 35

        # Responses should be binary
        assert set(np.unique(out["responses"])).issubset({0.0, 1.0})

    def test_discriminations_positive(self):
        sim = make_2pl_simulator(
            n_items=(18, 20),
            n_persons=(45, 50),
            prior_a=_PRIOR_A,
            prior_b=_PRIOR_B,
            prior_theta=_PRIOR_THETA,
        )
        out = sim.sample((100,))
        assert np.all(out["a"] > 0), "Discriminations should be positive"

    def test_loading_parameterization(self):
        sim = make_2pl_simulator(
            n_items=(10, 14),
            n_persons=(30, 35),
            parameterization="loading",
            prior_a=_PRIOR_A,
            prior_b=_PRIOR_B,
            prior_theta=_PRIOR_THETA,
        )
        out = sim.sample((5,))
        assert out["responses"].shape[0] == 5

    def test_response_dtype_uint8(self):
        sim = make_2pl_simulator(
            n_items=(10, 10),
            n_persons=(30, 30),
            prior_a=_PRIOR_A,
            prior_b=_PRIOR_B,
            prior_theta=_PRIOR_THETA,
            response_dtype="uint8",
        )
        out = sim.sample((3,))
        assert out["responses"].dtype == np.uint8

    def test_prior_a_options(self):
        for prior in (
            Prior("lognormal", loc=0, scale=1),
            Prior("halfnorm"),
            Prior("truncnorm"),
        ):
            sim = make_2pl_simulator(
                n_items=(5, 7),
                n_persons=(10, 13),
                prior_a=prior,
                prior_b=_PRIOR_B,
                prior_theta=_PRIOR_THETA,
            )
            out = sim.sample((3,))
            assert np.all(out["a"] > 0)

    def test_rejects_scalar_size_specs(self):
        with pytest.raises(ValueError, match="n_items must be a tuple/list"):
            make_2pl_simulator(
                n_items=10,
                n_persons=(20, 30),
                prior_a=_PRIOR_A,
                prior_b=_PRIOR_B,
                prior_theta=_PRIOR_THETA,
            )
        with pytest.raises(ValueError, match="n_persons must be a tuple/list"):
            make_2pl_simulator(
                n_items=(5, 10),
                n_persons=30,
                prior_a=_PRIOR_A,
                prior_b=_PRIOR_B,
                prior_theta=_PRIOR_THETA,
            )

    def test_fixed_size_via_clone(self):
        """clone_with_sizes with hi == lo should pin to exact dimensions."""
        sim = make_2pl_simulator(
            n_items=(5, 10),
            n_persons=(50, 100),
            prior_a=_PRIOR_A,
            prior_b=_PRIOR_B,
            prior_theta=_PRIOR_THETA,
        )
        pinned = sim.clone_with_sizes(n_items=(5, 5), n_persons=(50, 50))
        for _ in range(5):
            out = pinned.sample((4,))
            assert out["a"].shape == (4, 5)
            assert out["b"].shape == (4, 5)
            assert out["responses"].shape == (4, 50, 5)

    # -- Prior validation at construction time ---------------------------------

    def test_rejects_invalid_prior_a(self):
        with pytest.raises(ValueError, match="prior_a"):
            make_2pl_simulator(
                n_items=(5, 10),
                n_persons=(10, 20),
                prior_a=Prior("invalid"),
                prior_b=_PRIOR_B,
                prior_theta=_PRIOR_THETA,
            )

    def test_rejects_invalid_prior_b(self):
        with pytest.raises(ValueError, match="prior_b"):
            make_2pl_simulator(
                n_items=(5, 10),
                n_persons=(10, 20),
                prior_a=_PRIOR_A,
                prior_b=Prior("laplace"),
                prior_theta=_PRIOR_THETA,
            )

    def test_rejects_invalid_prior_theta(self):
        with pytest.raises(ValueError, match="prior_theta"):
            make_2pl_simulator(
                n_items=(5, 10),
                n_persons=(10, 20),
                prior_a=_PRIOR_A,
                prior_b=_PRIOR_B,
                prior_theta=Prior("beta"),
            )

    def test_rejects_string_prior(self):
        """Strings are not accepted as PriorSpec."""
        with pytest.raises(TypeError, match="prior_a"):
            make_2pl_simulator(
                n_items=(5, 10),
                n_persons=(10, 20),
                prior_a="lognormal",
                prior_b=_PRIOR_B,
                prior_theta=_PRIOR_THETA,
            )

    def test_rejects_invalid_parameterization(self):
        with pytest.raises(ValueError, match="parameterization"):
            make_2pl_simulator(
                n_items=(5, 10),
                n_persons=(10, 20),
                parameterization="unknown",
                prior_a=_PRIOR_A,
                prior_b=_PRIOR_B,
                prior_theta=_PRIOR_THETA,
            )


class TestMake1PLSimulator:
    def test_variable_size(self):
        sim = make_1pl_simulator(
            n_items=(8, 12),
            n_persons=(25, 30),
            prior_b=_PRIOR_B,
            prior_theta=_PRIOR_THETA,
        )
        out = sim.sample((4,))

        n_items = int(out["n_items"][0, 0])
        n_persons = int(out["n_persons"][0, 0])

        assert out["b"].shape == (4, n_items)
        assert out["theta"].shape == (4, n_persons)
        assert out["responses"].shape == (4, n_persons, n_items)
        assert "a" not in out  # 1PL has no discrimination parameter

    def test_variable_size_2pl_batched(self):
        """Variable-size 2PL should produce clean batches."""
        sim = make_2pl_simulator(
            n_items=(5, 20),
            n_persons=(10, 50),
            prior_a=_PRIOR_A,
            prior_b=_PRIOR_B,
            prior_theta=_PRIOR_THETA,
        )
        out = sim.sample((8,))

        n_items = int(out["n_items"][0, 0])
        n_persons = int(out["n_persons"][0, 0])
        assert out["a"].shape == (8, n_items)
        assert out["b"].shape == (8, n_items)
        assert out["theta"].shape == (8, n_persons)
        assert out["responses"].shape == (8, n_persons, n_items)
        assert "responses" in out
        assert "n_items" in out

    def test_rejects_inverted_ranges(self):
        """hi < lo should raise; hi == lo (fixed size) is allowed."""
        with pytest.raises(ValueError, match="hi >= lo"):
            make_1pl_simulator(
                n_items=(10, 5),
                n_persons=(20, 30),
                prior_b=_PRIOR_B,
                prior_theta=_PRIOR_THETA,
            )
        with pytest.raises(ValueError, match="hi >= lo"):
            make_1pl_simulator(
                n_items=(10, 12),
                n_persons=(30, 20),
                prior_b=_PRIOR_B,
                prior_theta=_PRIOR_THETA,
            )

    def test_fixed_size_ranges(self):
        """hi == lo should produce constant-size batches."""
        sim = make_1pl_simulator(
            n_items=(5, 5),
            n_persons=(50, 50),
            prior_b=_PRIOR_B,
            prior_theta=_PRIOR_THETA,
        )
        out = sim.sample((4,))
        assert out["b"].shape == (4, 5)
        assert out["responses"].shape == (4, 50, 5)

    def test_rejects_invalid_prior_b(self):
        with pytest.raises(ValueError, match="prior_b"):
            make_1pl_simulator(
                n_items=(5, 10),
                n_persons=(10, 20),
                prior_b=Prior("laplace"),
                prior_theta=_PRIOR_THETA,
            )

    def test_rejects_invalid_prior_theta(self):
        with pytest.raises(ValueError, match="prior_theta"):
            make_1pl_simulator(
                n_items=(5, 10),
                n_persons=(10, 20),
                prior_b=_PRIOR_B,
                prior_theta=Prior("cauchy"),
            )

    def test_response_dtype_uint8(self):
        sim = make_1pl_simulator(
            n_items=(8, 8),
            n_persons=(25, 25),
            prior_b=_PRIOR_B,
            prior_theta=_PRIOR_THETA,
            response_dtype="uint8",
        )
        out = sim.sample((4,))
        assert out["responses"].dtype == np.uint8


class TestSimulatorWithPriorObjects:
    """Test simulators with Prior dataclass and callable inputs."""

    def test_2pl_prior_dataclass(self):
        sim = make_2pl_simulator(
            n_items=(5, 7),
            n_persons=(10, 13),
            prior_a=Prior("lognormal", loc=0.3, scale=0.4),
            prior_b=Prior("normal", loc=0, scale=1.5),
            prior_theta=_PRIOR_THETA,
        )
        out = sim.sample((3,))
        assert np.all(out["a"] > 0)
        assert out["b"].shape[0] == 3

    def test_2pl_callable_prior(self):
        def custom_a(shape, rng):
            return rng.exponential(1.0, size=shape)

        sim = make_2pl_simulator(
            n_items=(5, 7),
            n_persons=(10, 13),
            prior_a=custom_a,
            prior_b=_PRIOR_B,
            prior_theta=_PRIOR_THETA,
        )
        out = sim.sample((3,))
        assert np.all(out["a"] > 0)

    def test_2pl_new_distributions(self):
        sim = make_2pl_simulator(
            n_items=(5, 7),
            n_persons=(10, 13),
            prior_a=Prior("gamma", concentration=4, scale=0.25),
            prior_b=Prior("student_t", df=3, loc=0, scale=1),
            prior_theta=Prior("student_t", df=5, loc=0, scale=1),
        )
        out = sim.sample((5,))
        assert np.all(out["a"] > 0)
        assert out["b"].shape[0] == 5

    def test_2pl_mixed_spec_types(self):
        """Prior and callable can be mixed."""
        sim = make_2pl_simulator(
            n_items=(5, 7),
            n_persons=(10, 13),
            prior_a=Prior("lognormal", loc=0, scale=1),
            prior_b=Prior("normal", loc=0, scale=1.0),
            prior_theta=lambda shape, rng: rng.normal(0, 1, size=shape),
        )
        out = sim.sample((3,))
        assert out["a"].shape[0] == 3

    def test_1pl_prior_dataclass(self):
        sim = make_1pl_simulator(
            n_items=(5, 7),
            n_persons=(10, 13),
            prior_b=Prior("uniform", low=-4, high=4),
            prior_theta=Prior("student_t", df=5, loc=0, scale=1),
        )
        out = sim.sample((4,))
        assert out["b"].shape[0] == 4

    def test_1pl_callable_prior(self):
        sim = make_1pl_simulator(
            n_items=(5, 7),
            n_persons=(10, 13),
            prior_b=lambda shape, rng: rng.normal(0, 2, size=shape),
            prior_theta=_PRIOR_THETA,
        )
        out = sim.sample((3,))
        assert out["b"].shape[0] == 3


# ---------------------------------------------------------------------------
# PCM Simulator
# ---------------------------------------------------------------------------


class TestMakePCMSimulator:
    def test_basic_output_keys(self):
        """PCM simulator should produce all expected keys."""
        sim = make_pcm_simulator(
            n_items=(8, 12),
            n_persons=(20, 30),
            n_categories=5,
            prior_delta=_PRIOR_DELTA,
        )
        out = sim.sample((4,))
        assert set(out.keys()) == {
            "delta",
            "theta",
            "responses",
            "n_items",
            "n_persons",
        }

    def test_output_shapes(self):
        """PCM simulator outputs have the expected batch shapes."""
        n_categories = 5
        sim = make_pcm_simulator(
            n_items=(8, 12),
            n_persons=(20, 30),
            n_categories=n_categories,
            prior_delta=_PRIOR_DELTA,
        )
        out = sim.sample((4,))

        n_items = int(out["n_items"][0, 0])
        n_persons = int(out["n_persons"][0, 0])

        assert out["delta"].shape == (4, n_items, n_categories - 1)
        assert out["theta"].shape == (4, n_persons)
        assert out["responses"].shape == (4, n_persons, n_items)
        assert out["n_items"].shape == (4, 1)
        assert out["n_persons"].shape == (4, 1)

    def test_variable_size(self):
        """Item and person counts vary; category count stays fixed."""
        n_categories = 4
        sim = make_pcm_simulator(
            n_items=(5, 20),
            n_persons=(10, 50),
            n_categories=n_categories,
            prior_delta=_PRIOR_DELTA,
        )

        item_sizes, person_sizes = set(), set()
        for _ in range(20):
            out = sim.sample((2,))
            item_sizes.add(int(out["n_items"][0, 0]))
            person_sizes.add(int(out["n_persons"][0, 0]))
            # n_categories-1 step diffs per item, always
            assert out["delta"].shape[2] == n_categories - 1

        # With wide ranges and 20 draws, we should see variation
        assert len(item_sizes) > 1, "n_items should vary across batches"
        assert len(person_sizes) > 1, "n_persons should vary across batches"

    def test_responses_in_range(self):
        """Responses should be integers in {0, ..., n_categories-1}."""
        n_categories = 7
        sim = make_pcm_simulator(
            n_items=(8, 12),
            n_persons=(50, 60),
            n_categories=n_categories,
            prior_delta=_PRIOR_DELTA,
        )
        out = sim.sample((10,))
        unique = np.unique(out["responses"])
        assert np.all(unique >= 0)
        assert np.all(unique <= n_categories - 1)
        # With enough data, should see more than just 0/1
        assert len(unique) > 2, f"Expected multiple categories, got {unique}"

    def test_step_difficulties_shape(self):
        """delta has exactly n_categories-1 steps per item."""
        for n_categories in (2, 3, 5, 7):
            sim = make_pcm_simulator(
                n_items=(5, 8),
                n_persons=(10, 15),
                n_categories=n_categories,
                prior_delta=_PRIOR_DELTA,
            )
            out = sim.sample((3,))
            assert out["delta"].shape[2] == n_categories - 1

    def test_prior_delta_options(self):
        """Multiple distributions should work for delta."""
        for prior in (
            Prior("normal", loc=0, scale=1),
            Prior("uniform", low=-3, high=3),
            Prior("student_t", df=5, loc=0, scale=1),
        ):
            sim = make_pcm_simulator(
                n_items=(5, 7),
                n_persons=(10, 13),
                n_categories=4,
                prior_delta=prior,
            )
            out = sim.sample((3,))
            assert out["delta"].shape[0] == 3

    def test_rejects_invalid_n_categories(self):
        """n_categories must be int >= 2."""
        with pytest.raises(ValueError, match="n_categories"):
            make_pcm_simulator(
                n_items=(5, 10),
                n_persons=(10, 20),
                n_categories=1,
                prior_delta=_PRIOR_DELTA,
            )
        with pytest.raises(ValueError, match="n_categories"):
            make_pcm_simulator(
                n_items=(5, 10),
                n_persons=(10, 20),
                n_categories=0,
                prior_delta=_PRIOR_DELTA,
            )
        with pytest.raises((ValueError, TypeError), match="n_categories"):
            make_pcm_simulator(
                n_items=(5, 10),
                n_persons=(10, 20),
                n_categories=3.5,
                prior_delta=_PRIOR_DELTA,
            )

    def test_pcm_k2_single_step(self):
        """n_categories=2 gives 1 step difficulty per item (binary-like)."""
        sim = make_pcm_simulator(
            n_items=(5, 8), n_persons=(10, 15), n_categories=2, prior_delta=_PRIOR_DELTA
        )
        out = sim.sample((4,))
        assert out["delta"].shape[2] == 1
        unique = np.unique(out["responses"])
        assert set(unique).issubset({0.0, 1.0})

    def test_rejects_invalid_prior_delta(self):
        with pytest.raises(ValueError, match="prior_delta"):
            make_pcm_simulator(
                n_items=(5, 10),
                n_persons=(10, 20),
                n_categories=3,
                prior_delta=Prior("lognormal"),
            )

    def test_rejects_invalid_prior_theta(self):
        with pytest.raises(ValueError, match="prior_theta"):
            make_pcm_simulator(
                n_items=(5, 10),
                n_persons=(10, 20),
                n_categories=3,
                prior_delta=_PRIOR_DELTA,
                prior_theta=Prior("lognormal"),
            )

    def test_callable_prior_delta(self):
        sim = make_pcm_simulator(
            n_items=(5, 7),
            n_persons=(10, 13),
            n_categories=4,
            prior_delta=lambda shape, rng: rng.normal(0, 1.5, size=shape),
        )
        out = sim.sample((3,))
        assert out["delta"].shape[0] == 3
        assert out["delta"].shape[2] == 3  # n_categories-1 = 3

    def test_dtype_float32(self):
        """All outputs should be float32."""
        sim = make_pcm_simulator(
            n_items=(5, 8), n_persons=(10, 15), n_categories=4, prior_delta=_PRIOR_DELTA
        )
        out = sim.sample((3,))
        for v in out.values():
            assert v.dtype == np.float32

    def test_response_dtype_uint8(self):
        sim = make_pcm_simulator(
            n_items=(5, 8),
            n_persons=(10, 15),
            n_categories=4,
            prior_delta=_PRIOR_DELTA,
            response_dtype="uint8",
        )
        out = sim.sample((3,))
        assert out["responses"].dtype == np.uint8


# ---------------------------------------------------------------------------
# GPCM Simulator
# ---------------------------------------------------------------------------


class TestMakeGPCMSimulator:
    def test_basic_output_keys(self):
        sim = make_gpcm_simulator(
            n_items=(8, 12),
            n_persons=(20, 30),
            n_categories=5,
            prior_a=_PRIOR_A,
            prior_delta=_PRIOR_DELTA,
        )
        out = sim.sample((4,))
        assert set(out.keys()) == {
            "a",
            "delta",
            "theta",
            "responses",
            "n_items",
            "n_persons",
        }

    def test_output_shapes(self):
        n_categories = 5
        sim = make_gpcm_simulator(
            n_items=(8, 12),
            n_persons=(20, 30),
            n_categories=n_categories,
            prior_a=_PRIOR_A,
            prior_delta=_PRIOR_DELTA,
        )
        out = sim.sample((4,))
        n_items = int(out["n_items"][0, 0])
        n_persons = int(out["n_persons"][0, 0])
        assert out["a"].shape == (4, n_items)
        assert out["delta"].shape == (4, n_items, n_categories - 1)
        assert out["theta"].shape == (4, n_persons)
        assert out["responses"].shape == (4, n_persons, n_items)

    def test_variable_size(self):
        n_categories = 4
        sim = make_gpcm_simulator(
            n_items=(5, 20),
            n_persons=(10, 50),
            n_categories=n_categories,
            prior_a=_PRIOR_A,
            prior_delta=_PRIOR_DELTA,
        )
        item_sizes = set()
        for _ in range(20):
            out = sim.sample((2,))
            item_sizes.add(int(out["n_items"][0, 0]))
            assert out["delta"].shape[2] == n_categories - 1
        assert len(item_sizes) > 1

    def test_responses_in_range(self):
        n_categories = 7
        sim = make_gpcm_simulator(
            n_items=(8, 12),
            n_persons=(50, 60),
            n_categories=n_categories,
            prior_a=_PRIOR_A,
            prior_delta=_PRIOR_DELTA,
        )
        out = sim.sample((10,))
        unique = np.unique(out["responses"])
        assert np.all(unique >= 0)
        assert np.all(unique <= n_categories - 1)
        assert len(unique) > 2

    def test_discriminations_positive(self):
        sim = make_gpcm_simulator(
            n_items=(10, 15),
            n_persons=(30, 40),
            n_categories=5,
            prior_a=_PRIOR_A,
            prior_delta=_PRIOR_DELTA,
        )
        out = sim.sample((50,))
        assert np.all(out["a"] > 0)

    def test_rejects_invalid_n_categories(self):
        with pytest.raises(ValueError, match="n_categories"):
            make_gpcm_simulator(
                n_items=(5, 10),
                n_persons=(10, 20),
                n_categories=1,
                prior_a=_PRIOR_A,
                prior_delta=_PRIOR_DELTA,
            )

    def test_rejects_invalid_parameterization(self):
        with pytest.raises(ValueError, match="parameterization"):
            make_gpcm_simulator(
                n_items=(5, 10),
                n_persons=(10, 20),
                n_categories=5,
                prior_a=_PRIOR_A,
                prior_delta=_PRIOR_DELTA,
                parameterization="unknown",
            )

    def test_clone_with_sizes(self):
        sim = make_gpcm_simulator(
            n_items=(5, 10),
            n_persons=(50, 100),
            n_categories=4,
            prior_a=_PRIOR_A,
            prior_delta=_PRIOR_DELTA,
        )
        pinned = sim.clone_with_sizes(n_items=(7, 7), n_persons=(40, 40))
        for _ in range(5):
            out = pinned.sample((3,))
            assert out["a"].shape == (3, 7)
            assert out["delta"].shape == (3, 7, 3)
            assert out["responses"].shape == (3, 40, 7)

    def test_loading_parameterization(self):
        sim = make_gpcm_simulator(
            n_items=(8, 12),
            n_persons=(20, 30),
            n_categories=5,
            prior_a=_PRIOR_A,
            prior_delta=_PRIOR_DELTA,
            parameterization="loading",
        )
        out = sim.sample((4,))
        assert out["responses"].shape[0] == 4

    def test_gpcm_k2_binary_responses(self):
        """n_categories=2 GPCM should use the binary response support cleanly."""
        sim = make_gpcm_simulator(
            n_items=(5, 8),
            n_persons=(10, 15),
            n_categories=2,
            prior_a=_PRIOR_A,
            prior_delta=_PRIOR_DELTA,
            response_dtype="uint8",
        )
        out = sim.sample((4,))
        assert out["delta"].shape[2] == 1
        assert out["responses"].dtype == np.uint8
        assert set(np.unique(out["responses"])).issubset({0, 1})

    def test_pcm_equivalence_when_a_is_one(self):
        """When a_i = 1 for all items, GPCM must match PCM."""
        n_categories = 5

        def constant_one(shape, rng):
            return np.ones(shape)

        gpcm_sim = make_gpcm_simulator(
            n_items=(10, 10),
            n_persons=(30, 30),
            n_categories=n_categories,
            prior_a=constant_one,
            prior_delta=_PRIOR_DELTA,
        )
        pcm_sim = make_pcm_simulator(
            n_items=(10, 10),
            n_persons=(30, 30),
            n_categories=n_categories,
            prior_delta=_PRIOR_DELTA,
        )
        seed = 42
        gpcm_sim._rng = np.random.default_rng(seed)
        pcm_sim._rng = np.random.default_rng(seed)
        gpcm_out = gpcm_sim.sample((8,))
        pcm_out = pcm_sim.sample((8,))
        np.testing.assert_array_equal(gpcm_out["delta"], pcm_out["delta"])
        np.testing.assert_array_equal(gpcm_out["theta"], pcm_out["theta"])
        np.testing.assert_array_equal(
            gpcm_out["responses"],
            pcm_out["responses"],
        )

    def test_dtype_float32(self):
        sim = make_gpcm_simulator(
            n_items=(5, 8),
            n_persons=(10, 15),
            n_categories=4,
            prior_a=_PRIOR_A,
            prior_delta=_PRIOR_DELTA,
        )
        out = sim.sample((3,))
        for v in out.values():
            assert v.dtype == np.float32

    def test_response_dtype_uint8(self):
        sim = make_gpcm_simulator(
            n_items=(5, 8),
            n_persons=(10, 15),
            n_categories=4,
            prior_a=_PRIOR_A,
            prior_delta=_PRIOR_DELTA,
            response_dtype="uint8",
        )
        out = sim.sample((3,))
        assert out["responses"].dtype == np.uint8

    def test_rejects_invalid_response_dtype(self):
        with pytest.raises(ValueError, match="response_dtype"):
            make_gpcm_simulator(
                n_items=(5, 8),
                n_persons=(10, 15),
                n_categories=4,
                prior_a=_PRIOR_A,
                prior_delta=_PRIOR_DELTA,
                response_dtype="int16",
            )


# ---------------------------------------------------------------------------
# Variable prior integration — 1PL
# ---------------------------------------------------------------------------


class Test1PLSimulatorVariable:
    def test_sigma_b_in_output(self):
        vp = VariablePrior("normal", role="b", scale_range=(0.5, 3.0))
        sim = make_1pl_simulator(n_items=(5, 8), n_persons=(10, 15), prior_b=vp)
        out = sim.sample((4,))
        assert "sigma_b" in out
        assert out["sigma_b"].shape == (4, 1)
        assert out["sigma_b"].dtype == np.float32

    def test_sigma_b_range(self):
        vp = VariablePrior("normal", role="b", scale_range=(0.5, 3.0))
        sim = make_1pl_simulator(n_items=(5, 8), n_persons=(10, 15), prior_b=vp)
        for _ in range(10):
            out = sim.sample((4,))
            assert np.all(out["sigma_b"] >= 0.5)
            assert np.all(out["sigma_b"] <= 3.0)

    def test_no_sigma_with_fixed_prior(self):
        sim = make_1pl_simulator(n_items=(5, 8), n_persons=(10, 15), prior_b=_PRIOR_B)
        out = sim.sample((4,))
        assert "sigma_b" not in out


# ---------------------------------------------------------------------------
# Variable prior integration — 2PL
# ---------------------------------------------------------------------------


class Test2PLSimulatorVariable:
    def test_sigma_keys_in_output(self):
        vp_a = VariablePrior("lognormal", role="a", scale_range=(0.1, 1.0))
        vp_b = VariablePrior("normal", role="b", scale_range=(0.5, 3.0))
        sim = make_2pl_simulator(
            n_items=(5, 8),
            n_persons=(10, 15),
            prior_a=vp_a,
            prior_b=vp_b,
        )
        out = sim.sample((4,))
        assert "sigma_a" in out
        assert "sigma_b" in out
        assert out["sigma_a"].shape == (4, 1)
        assert out["sigma_b"].shape == (4, 1)

    def test_mixed_fixed_variable(self):
        vp_a = VariablePrior("lognormal", role="a", scale_range=(0.1, 1.0))
        sim = make_2pl_simulator(
            n_items=(5, 8),
            n_persons=(10, 15),
            prior_a=vp_a,
            prior_b=_PRIOR_B,
        )
        out = sim.sample((4,))
        assert "sigma_a" in out
        assert "sigma_b" not in out


class TestSizeBucketing:
    """`item_bucket`/`person_bucket` quantize drawn sizes to a small,
    repeating set of values -- see `_draw_bucketed_size`'s docstring for why
    (CUDA allocator fragmentation from near-unique per-batch tensor shapes)."""

    def test_none_preserves_original_per_integer_draw(self):
        # No bucketing (default): many distinct sizes should appear across
        # enough draws.
        sim = make_2pl_simulator(
            n_items=(5, 50), n_persons=(5, 1000), prior_a=_PRIOR_A, prior_b=_PRIOR_B
        )
        seen = {int(sim.sample((1,))["n_persons"][0, 0]) for _ in range(60)}
        assert len(seen) > 20

    def test_bucketing_limits_distinct_sizes(self):
        sim = make_2pl_simulator(
            n_items=(5, 50),
            n_persons=(5, 1000),
            prior_a=_PRIOR_A,
            prior_b=_PRIOR_B,
            item_bucket=5,
            person_bucket=50,
        )
        seen_items = set()
        seen_persons = set()
        for _ in range(200):
            out = sim.sample((1,))
            seen_items.add(int(out["n_items"][0, 0]))
            seen_persons.add(int(out["n_persons"][0, 0]))
        # n_items=(5,50) bucketed by 5 -> at most 10 windows;
        # n_persons=(5,1000) bucketed by 50 -> at most 20 windows.
        assert len(seen_items) <= 10
        assert len(seen_persons) <= 20

    def test_bucketed_sizes_stay_within_requested_range(self):
        sim = make_2pl_simulator(
            n_items=(5, 50),
            n_persons=(5, 1000),
            prior_a=_PRIOR_A,
            prior_b=_PRIOR_B,
            item_bucket=7,  # does not evenly divide the range
            person_bucket=300,  # coarser than the range itself
        )
        for _ in range(50):
            out = sim.sample((1,))
            n_items = int(out["n_items"][0, 0])
            n_persons = int(out["n_persons"][0, 0])
            assert 5 <= n_items <= 50
            assert 5 <= n_persons <= 1000

    def test_bucket_covers_full_range_including_edges(self):
        # A bucket coarser than the range still yields both ends.
        sim = make_2pl_simulator(
            n_items=(5, 50),
            n_persons=(5, 1000),
            prior_a=_PRIOR_A,
            prior_b=_PRIOR_B,
            person_bucket=2000,
        )
        seen = {int(sim.sample((1,))["n_persons"][0, 0]) for _ in range(100)}
        assert seen == {5, 1000}

    @pytest.mark.parametrize(
        "items,persons,item_bucket,person_bucket",
        [((5, 50), (10, 500), 5, 50)],
    )
    def test_bucketed_draws_include_both_ends(
        self, items, persons, item_bucket, person_bucket
    ):
        from bayesflow_irt.simulators import _bucket_representatives

        sim = make_2pl_simulator(
            n_items=items,
            n_persons=persons,
            prior_a=_PRIOR_A,
            prior_b=_PRIOR_B,
            item_bucket=item_bucket,
            person_bucket=person_bucket,
        )
        item_reps = set(_bucket_representatives(*items, item_bucket).tolist())
        person_reps = set(_bucket_representatives(*persons, person_bucket).tolist())
        seen_items, seen_persons = set(), set()
        for _ in range(600):
            out = sim.sample((1,))
            seen_items.add(int(out["n_items"][0, 0]))
            seen_persons.add(int(out["n_persons"][0, 0]))
        assert seen_items <= item_reps
        assert seen_persons <= person_reps
        assert {5, 50} <= seen_items
        assert {10, 500} <= seen_persons

    @pytest.mark.parametrize(
        "lo,hi,bucket", [(5, 50, 5), (10, 500, 50), (5, 1000, 50), (3, 4, 10)]
    )
    def test_representatives_respect_memory_bound(self, lo, hi, bucket):
        from bayesflow_irt.simulators import _bucket_representatives

        reps = _bucket_representatives(lo, hi, bucket)
        assert reps.size <= max(2, -(-(hi - lo + 1) // bucket))
        assert reps[0] == lo and reps[-1] == hi
        assert np.all(np.diff(reps) > 0)

    def test_representatives_pinned_range(self):
        from bayesflow_irt.simulators import _bucket_representatives

        np.testing.assert_array_equal(_bucket_representatives(7, 7, 3), [7])

    def test_clone_with_sizes_preserves_buckets(self):
        sim = make_2pl_simulator(
            n_items=(5, 50),
            n_persons=(5, 1000),
            prior_a=_PRIOR_A,
            prior_b=_PRIOR_B,
            item_bucket=5,
            person_bucket=50,
        )
        cloned = sim.clone_with_sizes(n_persons=(100, 200))
        assert cloned.item_bucket == 5
        assert cloned.person_bucket == 50

    def test_rejects_non_positive_bucket(self):
        with pytest.raises(ValueError):
            make_2pl_simulator(
                n_items=(5, 50),
                n_persons=(5, 1000),
                prior_a=_PRIOR_A,
                prior_b=_PRIOR_B,
                item_bucket=0,
            )
        with pytest.raises(ValueError):
            make_2pl_simulator(
                n_items=(5, 50),
                n_persons=(5, 1000),
                prior_a=_PRIOR_A,
                prior_b=_PRIOR_B,
                person_bucket=-3,
            )


class TestSizeDistribution:
    """`size_distribution` for unbucketed draws, and the metadata record of
    which sizes a simulator trains on."""

    def _sim(self, **kw):
        return make_2pl_simulator(
            n_items=(5, 50),
            n_persons=(10, 500),
            prior_a=_PRIOR_A,
            prior_b=_PRIOR_B,
            seed=0,
            **kw,
        )

    def test_log_uniform_draws_match_stated_probabilities(self):
        from bayesflow_irt.simulators import _draw_bucketed_size, _size_distribution

        values, probs = _size_distribution((10, 500), None, "log_uniform")
        assert probs.sum() == pytest.approx(1.0)
        rng = np.random.default_rng(1)
        draws = np.array(
            [
                _draw_bucketed_size(rng, (10, 500), None, "log_uniform")
                for _ in range(20000)
            ]
        )
        assert draws.min() >= 10 and draws.max() <= 500
        # Share of small datasets: stated vs drawn, and far above uniform.
        stated = probs[values <= 24].sum()
        assert np.mean(draws <= 24) == pytest.approx(stated, abs=0.01)
        assert stated > 0.2 > 15 / 491

    def test_uniform_default_is_unchanged(self):
        from bayesflow_irt.simulators import _size_distribution

        assert self._sim().size_distribution == "uniform"
        values, probs = _size_distribution((10, 500), None)
        np.testing.assert_array_equal(values, np.arange(10, 501))
        assert np.allclose(probs, 1 / 491)

    def test_mixed_is_the_stated_mixture_and_draws_match_it(self):
        from bayesflow_irt.simulators import _draw_bucketed_size, _size_distribution

        values, probs = _size_distribution((10, 500), None, "mixed", 0.7)
        _, logp = _size_distribution((10, 500), None, "log_uniform")
        np.testing.assert_allclose(probs, 0.7 * logp + 0.3 / values.size)
        rng = np.random.default_rng(2)
        draws = np.array(
            [
                _draw_bucketed_size(rng, (10, 500), None, "mixed", 0.7)
                for _ in range(20000)
            ]
        )
        assert np.mean(draws <= 24) == pytest.approx(
            probs[values <= 24].sum(), abs=0.01
        )

    def test_pinned_range_consumes_the_same_randomness_under_every_rule(self):
        """Validation clones pin each corner; their data must not depend on
        the training simulator's size distribution."""
        from bayesflow_irt.simulators import _draw_bucketed_size

        states = []
        for rule in ("uniform", "log_uniform", "mixed"):
            rng = np.random.default_rng(5)
            _draw_bucketed_size(rng, (10, 10), None, rule)
            states.append(rng.random())
        assert states[0] == states[1] == states[2]

    def test_invalid_log_uniform_share_raises(self):
        with pytest.raises(ValueError, match="log_uniform_share"):
            self._sim(size_distribution="mixed", log_uniform_share=1.5)

    def test_clone_preserves_size_distribution(self):
        sim = self._sim(size_distribution="mixed", log_uniform_share=0.6)
        clone = sim.clone_with_sizes(n_items=(5, 5))
        assert clone.size_distribution == "mixed"
        assert clone.log_uniform_share == 0.6

    def test_invalid_size_distribution_raises(self):
        with pytest.raises(ValueError, match="size_distribution"):
            self._sim(size_distribution="loguniform")

    def test_describe_size_sampling_records_buckets_and_both_ends(self):
        from bayesflow_irt.simulators import describe_size_sampling

        rec = describe_size_sampling(self._sim(item_bucket=5, person_bucket=50))
        assert rec["rule"] == "log_spaced_buckets"
        for key, (lo, hi) in (("n_items", (5, 50)), ("n_persons", (10, 500))):
            axis = rec[key]
            assert axis["range"] == [lo, hi]
            assert axis["values"][0] == lo and axis["values"][-1] == hi
            assert sum(axis["probabilities"]) == pytest.approx(1.0)

    def test_describe_size_sampling_is_json_serializable(self):
        import json

        from bayesflow_irt.simulators import describe_size_sampling

        rec = describe_size_sampling(self._sim(size_distribution="log_uniform"))
        assert rec["rule"] == "log_uniform"
        json.dumps(rec)

    def test_describe_size_sampling_keeps_full_precision(self):
        from bayesflow_irt.simulators import _size_distribution, describe_size_sampling

        rec = describe_size_sampling(self._sim(size_distribution="log_uniform"))
        _, probs = _size_distribution((10, 500), None, "log_uniform")
        assert rec["n_persons"]["probabilities"] == probs.tolist()

    def test_describe_size_sampling_omits_very_wide_supports(self):
        from bayesflow_irt.simulators import describe_size_sampling

        sim = make_2pl_simulator(
            n_items=(5, 50),
            n_persons=(10, 1_000_000),
            prior_a=_PRIOR_A,
            prior_b=_PRIOR_B,
            size_distribution="log_uniform",
        )
        axis = describe_size_sampling(sim)["n_persons"]
        assert axis["values"] is None and axis["probabilities"] is None
        assert axis["range"] == [10, 1_000_000]


# ---------------------------------------------------------------------------
# Variable prior integration — PCM
# ---------------------------------------------------------------------------


class TestPCMSimulatorVariable:
    def test_sigma_keys_in_output(self):
        vip = make_variable_incremental_prior((0.5, 3.0), (0.3, 2.0), 5)
        sim = make_pcm_simulator(
            n_items=(5, 8),
            n_persons=(10, 15),
            n_categories=5,
            prior_delta=vip,
        )
        out = sim.sample((4,))
        assert "sigma_mu" in out
        assert "sigma_eps" in out
        assert out["sigma_mu"].shape == (4, 1)
        assert out["sigma_eps"].shape == (4, 1)

    def test_no_sigma_with_fixed_prior(self):
        sim = make_pcm_simulator(
            n_items=(5, 8),
            n_persons=(10, 15),
            n_categories=5,
            prior_delta=_PRIOR_DELTA,
        )
        out = sim.sample((4,))
        assert "sigma_mu" not in out


# ---------------------------------------------------------------------------
# Variable prior integration — GPCM
# ---------------------------------------------------------------------------


class TestGPCMSimulatorVariable:
    def test_sigma_keys_in_output(self):
        vp_a = VariablePrior("lognormal", role="a", scale_range=(0.1, 1.0))
        vip = make_variable_incremental_prior((0.5, 3.0), (0.3, 2.0), 5)
        sim = make_gpcm_simulator(
            n_items=(5, 8),
            n_persons=(10, 15),
            n_categories=5,
            prior_a=vp_a,
            prior_delta=vip,
        )
        out = sim.sample((4,))
        assert "sigma_a" in out
        assert "sigma_mu" in out
        assert "sigma_eps" in out
        assert out["sigma_a"].shape == (4, 1)
        assert out["sigma_mu"].shape == (4, 1)
        assert out["sigma_eps"].shape == (4, 1)

    def test_mixed_fixed_variable(self):
        vip = make_variable_incremental_prior((0.5, 3.0), (0.3, 2.0), 5)
        sim = make_gpcm_simulator(
            n_items=(5, 8),
            n_persons=(10, 15),
            n_categories=5,
            prior_a=_PRIOR_A,
            prior_delta=vip,
        )
        out = sim.sample((4,))
        assert "sigma_a" not in out
        assert "sigma_mu" in out
        assert "sigma_eps" in out


# ---------------------------------------------------------------------------
# response_dtype — shared validation helper (_normalize_response_dtype)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory,kwargs",
    [
        (
            make_2pl_simulator,
            {"prior_a": _PRIOR_A, "prior_b": _PRIOR_B, "prior_theta": _PRIOR_THETA},
        ),
        (
            make_1pl_simulator,
            {"prior_b": _PRIOR_B, "prior_theta": _PRIOR_THETA},
        ),
        (
            make_pcm_simulator,
            {"n_categories": 3, "prior_delta": _PRIOR_DELTA},
        ),
    ],
)
def test_rejects_invalid_response_dtype_all_models(factory, kwargs):
    """_normalize_response_dtype must reject unsupported dtypes for every simulator."""
    with pytest.raises(ValueError, match="response_dtype"):
        factory(n_items=(5, 8), n_persons=(10, 15), response_dtype="int16", **kwargs)


@pytest.mark.parametrize(
    "factory,kwargs",
    [
        (
            make_2pl_simulator,
            {"prior_a": _PRIOR_A, "prior_b": _PRIOR_B, "prior_theta": _PRIOR_THETA},
        ),
        (
            make_1pl_simulator,
            {"prior_b": _PRIOR_B, "prior_theta": _PRIOR_THETA},
        ),
        (
            make_pcm_simulator,
            {"n_categories": 3, "prior_delta": _PRIOR_DELTA},
        ),
        (
            make_gpcm_simulator,
            {"n_categories": 3, "prior_a": _PRIOR_A, "prior_delta": _PRIOR_DELTA},
        ),
    ],
)
def test_clone_with_sizes_preserves_response_dtype(factory, kwargs):
    """clone_with_sizes must propagate response_dtype to the new simulator."""
    sim = factory(n_items=(5, 8), n_persons=(10, 15), response_dtype="uint8", **kwargs)
    cloned = sim.clone_with_sizes(n_items=(5, 5), n_persons=(10, 10))
    out = cloned.sample((2,))
    assert out["responses"].dtype == np.uint8, (
        f"Expected uint8 after clone_with_sizes; got {out['responses'].dtype}"
    )


# ---------------------------------------------------------------------------
# seed kwarg — additive, reproducible RNG across all four factories
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory,kwargs",
    [
        (
            make_2pl_simulator,
            {"prior_a": _PRIOR_A, "prior_b": _PRIOR_B, "prior_theta": _PRIOR_THETA},
        ),
        (
            make_1pl_simulator,
            {"prior_b": _PRIOR_B, "prior_theta": _PRIOR_THETA},
        ),
        (
            make_pcm_simulator,
            {"n_categories": 3, "prior_delta": _PRIOR_DELTA},
        ),
        (
            make_gpcm_simulator,
            {"n_categories": 3, "prior_a": _PRIOR_A, "prior_delta": _PRIOR_DELTA},
        ),
    ],
)
def test_seed_makes_sampling_reproducible(factory, kwargs):
    """seed=<int> must produce identical draws across independent simulators."""
    sim_a = factory(n_items=(5, 5), n_persons=(10, 10), seed=123, **kwargs)
    sim_b = factory(n_items=(5, 5), n_persons=(10, 10), seed=123, **kwargs)
    out_a = sim_a.sample((4,))
    out_b = sim_b.sample((4,))
    for key in ("responses", "theta"):
        np.testing.assert_array_equal(out_a[key], out_b[key])


@pytest.mark.parametrize(
    "factory,kwargs",
    [
        (
            make_2pl_simulator,
            {"prior_a": _PRIOR_A, "prior_b": _PRIOR_B, "prior_theta": _PRIOR_THETA},
        ),
        (
            make_gpcm_simulator,
            {"n_categories": 3, "prior_a": _PRIOR_A, "prior_delta": _PRIOR_DELTA},
        ),
    ],
)
def test_clone_with_sizes_preserves_seed(factory, kwargs):
    """clone_with_sizes must propagate the seed so clones stay reproducible."""
    sim = factory(n_items=(5, 5), n_persons=(10, 10), seed=7, **kwargs)
    cloned_a = sim.clone_with_sizes(n_items=(6, 6), n_persons=(12, 12))
    cloned_b = sim.clone_with_sizes(n_items=(6, 6), n_persons=(12, 12))
    out_a = cloned_a.sample((3,))
    out_b = cloned_b.sample((3,))
    np.testing.assert_array_equal(out_a["responses"], out_b["responses"])


# ---------------------------------------------------------------------------
# use_numba — opt-in accelerated response sampling
# ---------------------------------------------------------------------------

from bayesflow_irt.simulators import _HAS_NUMBA  # noqa: E402


@pytest.mark.skipif(_HAS_NUMBA, reason="only meaningful when numba is NOT installed")
@pytest.mark.parametrize(
    "factory,kwargs",
    [
        (
            make_2pl_simulator,
            {"prior_a": _PRIOR_A, "prior_b": _PRIOR_B, "prior_theta": _PRIOR_THETA},
        ),
        (
            make_1pl_simulator,
            {"prior_b": _PRIOR_B, "prior_theta": _PRIOR_THETA},
        ),
        (
            make_pcm_simulator,
            {"n_categories": 3, "prior_delta": _PRIOR_DELTA},
        ),
        (
            make_gpcm_simulator,
            {"n_categories": 3, "prior_a": _PRIOR_A, "prior_delta": _PRIOR_DELTA},
        ),
    ],
)
def test_use_numba_without_numba_installed_raises(factory, kwargs):
    with pytest.raises(ValueError, match="numba"):
        factory(n_items=(5, 5), n_persons=(10, 10), use_numba=True, **kwargs)


def test_use_numba_defaults_to_false():
    """Base install behavior must be unaffected: use_numba defaults off."""
    sim = make_2pl_simulator(
        n_items=(5, 5),
        n_persons=(10, 10),
        prior_a=_PRIOR_A,
        prior_b=_PRIOR_B,
        prior_theta=_PRIOR_THETA,
    )
    assert sim.use_numba is False


# ---------------------------------------------------------------------------
# Variable prior is rejected for theta
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory, kwargs",
    [
        (make_1pl_simulator, {"prior_b": _PRIOR_B}),
        (make_2pl_simulator, {"prior_a": _PRIOR_A, "prior_b": _PRIOR_B}),
        (make_pcm_simulator, {"prior_delta": _PRIOR_DELTA, "n_categories": 4}),
        (
            make_gpcm_simulator,
            {"prior_a": _PRIOR_A, "prior_delta": _PRIOR_DELTA, "n_categories": 4},
        ),
    ],
)
def test_variable_prior_theta_is_rejected(factory, kwargs):
    vp = VariablePrior("normal", role="theta", scale_range=(0.5, 2.0))
    with pytest.raises(ValueError, match="prior_theta.*variable prior"):
        factory(n_items=(5, 8), n_persons=(10, 15), prior_theta=vp, **kwargs)
