"""Integration tests — full train loop for variable-size IRT models."""

import bayesflow as bf
import numpy as np
import pytest

from bayesflow_irt.adapters import (
    make_1pl_adapter,
    make_2pl_adapter,
    make_gpcm_adapter,
    make_pcm_adapter,
)
from bayesflow_irt.approximators import EquivariantIRTApproximator
from bayesflow_irt.networks import MatrixEquivariantIRTSummary
from bayesflow_irt.priors import Prior
from bayesflow_irt.simulators import (
    make_1pl_simulator,
    make_2pl_simulator,
    make_gpcm_simulator,
    make_pcm_simulator,
)
from bayesflow_irt.utils import (
    ChunkSearchOptions,
    FitResult,
    PrecomputedDataset,
    _build_probe_approximator,
    fit_with_oom_retry,
    probe_memory_config,
)

# Standard test priors — must be specified explicitly at every call site.
_PRIOR_A = Prior("lognormal", loc=0, scale=0.5)
_PRIOR_B = Prior("normal", loc=0, scale=2.0)
_PRIOR_THETA = Prior("normal", loc=0, scale=1.0)
_PRIOR_DELTA = Prior("normal", loc=0, scale=2.0)


def _make_flow():
    return bf.networks.FlowMatching(subnet_kwargs={"widths": (32, 32)})


class TestApproximatorWorkflow:
    """Smoke test: train and sample with a variable-size 2PL workflow."""

    def test_equivariant_train(self):
        """Build and train the equivariant pipeline for a few steps."""
        simulator = make_2pl_simulator(
            n_items=(5, 10),
            n_persons=(10, 20),
            prior_a=_PRIOR_A,
            prior_b=_PRIOR_B,
            prior_theta=_PRIOR_THETA,
        )
        adapter = make_2pl_adapter(simulator)

        summary_net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=16,
            num_matrix_blocks=1,
            num_outer_sabs=1,
            outer_embed_dim=16,
            outer_mlp_width=32,
        )

        approximator = EquivariantIRTApproximator(
            adapter=adapter,
            inference_network=_make_flow(),
            item_posterior="flow_matching",
            summary_network=summary_net,
        )
        approximator.compile(optimizer="adam")

        history = approximator.fit(
            simulator=simulator,
            epochs=2,
            num_batches=4,
            batch_size=4,
        )

        losses = history.history["loss"]
        assert len(losses) == 2
        assert all(np.isfinite(loss) for loss in losses), f"Non-finite losses: {losses}"

    def test_equivariant_sample(self):
        """After fitting, sample per-item posteriors."""
        simulator = make_2pl_simulator(
            n_items=(7, 10),
            n_persons=(15, 20),
            prior_a=_PRIOR_A,
            prior_b=_PRIOR_B,
            prior_theta=_PRIOR_THETA,
        )
        adapter = make_2pl_adapter(simulator)

        summary_net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=16,
            num_matrix_blocks=1,
            num_outer_sabs=1,
            outer_embed_dim=16,
            outer_mlp_width=32,
        )

        approximator = EquivariantIRTApproximator(
            adapter=adapter,
            inference_network=_make_flow(),
            item_posterior="flow_matching",
            summary_network=summary_net,
        )
        approximator.compile(optimizer="adam")

        approximator.fit(
            simulator=simulator,
            epochs=1,
            num_batches=2,
            batch_size=4,
        )

        test_data = simulator.sample((3,))
        samples = approximator.sample(num_samples=20, conditions=test_data)

        # Should have a and b keys from adapter inverse
        assert "a" in samples or "b" in samples
        # Per-item samples: (num_datasets, num_samples, N_ITEMS)
        if "a" in samples:
            assert samples["a"].shape[0] == 3
            assert samples["a"].shape[1] == 20
        if "b" in samples:
            assert samples["b"].shape[0] == 3
            assert samples["b"].shape[1] == 20

    def test_equivariant_sample_with_chunked_sampling(self):
        """Sampling with many draws should support chunking over sample axis."""
        simulator = make_2pl_simulator(
            n_items=(7, 10),
            n_persons=(15, 20),
            prior_a=_PRIOR_A,
            prior_b=_PRIOR_B,
            prior_theta=_PRIOR_THETA,
        )
        adapter = make_2pl_adapter(simulator)

        summary_net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=16,
            num_matrix_blocks=1,
            num_outer_sabs=1,
            outer_embed_dim=16,
            outer_mlp_width=32,
        )

        approximator = EquivariantIRTApproximator(
            adapter=adapter,
            inference_network=_make_flow(),
            item_posterior="flow_matching",
            summary_network=summary_net,
        )
        approximator.compile(optimizer="adam")

        approximator.fit(
            simulator=simulator,
            epochs=1,
            num_batches=2,
            batch_size=4,
        )

        test_data = simulator.sample((3,))
        samples = approximator.sample(
            num_samples=128,
            conditions=test_data,
            sample_chunk_size=32,
        )

        assert "a" in samples and "b" in samples
        assert samples["a"].shape[0] == 3
        assert samples["a"].shape[1] == 128
        assert np.all(np.isfinite(samples["a"]))
        assert samples["b"].shape[0] == 3
        assert samples["b"].shape[1] == 128
        assert np.all(np.isfinite(samples["b"]))

    def test_equivariant_sample_with_condition_and_sample_chunking(self):
        """Sampling should support chunking over both item-conditions and samples."""
        simulator = make_2pl_simulator(
            n_items=(8, 10),
            n_persons=(15, 20),
            prior_a=_PRIOR_A,
            prior_b=_PRIOR_B,
            prior_theta=_PRIOR_THETA,
        )
        adapter = make_2pl_adapter(simulator)

        summary_net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=16,
            num_matrix_blocks=1,
            num_outer_sabs=1,
            outer_embed_dim=16,
            outer_mlp_width=32,
        )

        approximator = EquivariantIRTApproximator(
            adapter=adapter,
            inference_network=_make_flow(),
            item_posterior="flow_matching",
            summary_network=summary_net,
        )
        approximator.compile(optimizer="adam")

        approximator.fit(
            simulator=simulator,
            epochs=1,
            num_batches=2,
            batch_size=4,
        )

        test_data = simulator.sample((4,))
        samples = approximator.sample(
            num_samples=96,
            conditions=test_data,
            sample_chunk_size=24,
            condition_chunk_size=16,
        )

        assert "a" in samples and "b" in samples
        assert samples["a"].shape[0] == 4
        assert samples["a"].shape[1] == 96
        assert np.all(np.isfinite(samples["a"]))
        assert samples["b"].shape[0] == 4
        assert samples["b"].shape[1] == 96
        assert np.all(np.isfinite(samples["b"]))

    def test_equivariant_sample_without_global_conditions(self):
        """Sampling should remain valid when inference_conditions are absent."""
        simulator = make_2pl_simulator(
            n_items=(8, 10),
            n_persons=(15, 20),
            prior_a=_PRIOR_A,
            prior_b=_PRIOR_B,
            prior_theta=_PRIOR_THETA,
        )
        adapter = make_2pl_adapter(simulator)

        summary_net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=16,
            num_matrix_blocks=1,
            num_outer_sabs=1,
            outer_embed_dim=16,
            outer_mlp_width=32,
        )

        approximator = EquivariantIRTApproximator(
            adapter=adapter,
            inference_network=_make_flow(),
            item_posterior="flow_matching",
            summary_network=summary_net,
        )
        approximator.compile(optimizer="adam")

        approximator.fit(
            simulator=simulator,
            epochs=1,
            num_batches=2,
            batch_size=4,
        )

        raw = simulator.sample((3,))
        conditions = {"responses": raw["responses"]}
        samples = approximator.sample(
            num_samples=32,
            conditions=conditions,
            sample_chunk_size=16,
            condition_chunk_size=32,
        )

        assert "a" in samples and "b" in samples
        assert samples["a"].shape[0] == 3
        assert samples["a"].shape[1] == 32
        assert np.all(np.isfinite(samples["a"]))
        assert samples["b"].shape[0] == 3
        assert samples["b"].shape[1] == 32
        assert np.all(np.isfinite(samples["b"]))

    def test_equivariant_loading_sample_exposes_d(self):
        """Loading-mode sampling should expose d = -a*b for HPO validators."""
        simulator = make_2pl_simulator(
            n_items=(7, 10),
            n_persons=(15, 20),
            prior_a=_PRIOR_A,
            prior_b=_PRIOR_B,
            prior_theta=_PRIOR_THETA,
        )
        adapter = make_2pl_adapter(simulator, internal_parameterization="loading")

        summary_net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=16,
            num_matrix_blocks=1,
            num_outer_sabs=1,
            outer_embed_dim=16,
            outer_mlp_width=32,
        )

        approximator = EquivariantIRTApproximator(
            adapter=adapter,
            inference_network=_make_flow(),
            item_posterior="flow_matching",
            summary_network=summary_net,
        )
        approximator.compile(optimizer="adam")

        approximator.fit(
            simulator=simulator,
            epochs=1,
            num_batches=2,
            batch_size=4,
        )

        raw = simulator.sample((3,))
        samples = approximator.sample(num_samples=16, conditions=raw)

        assert "a" in samples and "b" in samples and "d" in samples
        np.testing.assert_allclose(
            samples["d"],
            -(samples["a"] * samples["b"]),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_equivariant_train_with_memory_controls(self):
        """Training should run with chunking and checkpointing enabled."""
        simulator = make_2pl_simulator(
            n_items=(5, 8),
            n_persons=(10, 15),
            prior_a=_PRIOR_A,
            prior_b=_PRIOR_B,
            prior_theta=_PRIOR_THETA,
        )
        adapter = make_2pl_adapter(simulator)

        summary_net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=16,
            num_matrix_blocks=1,
            num_outer_sabs=1,
            outer_embed_dim=16,
            outer_mlp_width=32,
            gradient_checkpointing=True,
        )

        approximator = EquivariantIRTApproximator(
            adapter=adapter,
            inference_network=_make_flow(),
            item_posterior="flow_matching",
            summary_network=summary_net,
        )
        approximator.compile(optimizer="adam")

        history = approximator.fit(
            simulator=simulator,
            epochs=1,
            num_batches=2,
            batch_size=2,
        )

        losses = history.history["loss"]
        assert len(losses) == 1
        assert all(np.isfinite(loss) for loss in losses), f"Non-finite losses: {losses}"

    def test_standardized_adapter(self):
        """FlowMatching with prior-standardized adapter produces finite losses."""
        simulator = make_2pl_simulator(
            n_items=(5, 8),
            n_persons=(10, 15),
            prior_a=_PRIOR_A,
            prior_b=_PRIOR_B,
            prior_theta=_PRIOR_THETA,
        )
        adapter = make_2pl_adapter(simulator)

        summary_net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=16,
            num_matrix_blocks=1,
            num_outer_sabs=1,
            outer_embed_dim=16,
            outer_mlp_width=32,
        )

        approximator = EquivariantIRTApproximator(
            adapter=adapter,
            inference_network=_make_flow(),
            item_posterior="flow_matching",
            summary_network=summary_net,
        )
        approximator.compile(optimizer="adam")

        history = approximator.fit(
            simulator=simulator,
            epochs=2,
            num_batches=3,
            batch_size=4,
        )

        losses = history.history["loss"]
        assert len(losses) == 2
        assert all(np.isfinite(loss) for loss in losses), f"Non-finite losses: {losses}"


class TestMatrixEquivariantWorkflow:
    """Smoke tests for MatrixEquivariantIRTSummary across model families."""

    @staticmethod
    def _make_precomputed_dataset(simulator, adapter, *, batch_size=4, num_batches=2):
        batches = [adapter(simulator.sample((batch_size,))) for _ in range(num_batches)]
        return PrecomputedDataset(batches)

    def test_matrix_1pl_train_and_sample(self):
        simulator = make_1pl_simulator(
            n_items=(5, 8),
            n_persons=(10, 16),
            prior_b=_PRIOR_B,
            prior_theta=_PRIOR_THETA,
        )
        adapter = make_1pl_adapter(simulator)
        summary_net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=16,
            num_matrix_blocks=0,
            num_outer_sabs=1,
            outer_embed_dim=16,
            outer_mlp_width=32,
        )
        approximator = EquivariantIRTApproximator(
            adapter=adapter,
            inference_network=_make_flow(),
            item_posterior="flow_matching",
            summary_network=summary_net,
        )
        approximator.compile(optimizer="adam")
        dataset = self._make_precomputed_dataset(
            simulator, adapter, batch_size=4, num_batches=2
        )

        history = approximator.fit(
            dataset=dataset,
            epochs=1,
        )
        assert all(np.isfinite(v) for v in history.history["loss"])

        test_data = simulator.sample((2,))
        samples = approximator.sample(num_samples=12, conditions=test_data)
        assert "b" in samples
        assert samples["b"].shape[:2] == (2, 12)
        assert np.all(np.isfinite(samples["b"]))

    def test_matrix_1pl_fast_path_item_level_channel_train_and_sample(self):
        """L=0 + mean pooling + an item-level channel: the late-fused
        item_rest_correlation must not knock the summary off the
        pool-before-project fast path (a technical smoke for the relaxed
        guard)."""
        simulator = make_1pl_simulator(
            n_items=(5, 8),
            n_persons=(10, 16),
            prior_b=_PRIOR_B,
            prior_theta=_PRIOR_THETA,
        )
        adapter = make_1pl_adapter(simulator)
        summary_net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=16,
            num_matrix_blocks=0,
            raw_feature_channels=("item_rest_correlation",),
            num_outer_sabs=1,
            outer_embed_dim=16,
            outer_mlp_width=32,
        )
        approximator = EquivariantIRTApproximator(
            adapter=adapter,
            inference_network=_make_flow(),
            item_posterior="flow_matching",
            summary_network=summary_net,
        )
        approximator.compile(optimizer="adam")
        dataset = self._make_precomputed_dataset(
            simulator, adapter, batch_size=4, num_batches=2
        )

        history = approximator.fit(
            dataset=dataset,
            epochs=1,
        )
        assert all(np.isfinite(v) for v in history.history["loss"])

        test_data = simulator.sample((2,))
        samples = approximator.sample(num_samples=12, conditions=test_data)
        assert "b" in samples
        assert samples["b"].shape[:2] == (2, 12)
        assert np.all(np.isfinite(samples["b"]))

    def test_matrix_2pl_train_and_sample(self):
        simulator = make_2pl_simulator(
            n_items=(5, 9),
            n_persons=(10, 16),
            prior_a=_PRIOR_A,
            prior_b=_PRIOR_B,
            prior_theta=_PRIOR_THETA,
        )
        adapter = make_2pl_adapter(simulator)
        summary_net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=16,
            num_matrix_blocks=2,
            num_outer_sabs=1,
            outer_embed_dim=16,
            outer_mlp_width=32,
        )
        approximator = EquivariantIRTApproximator(
            adapter=adapter,
            inference_network=_make_flow(),
            item_posterior="flow_matching",
            summary_network=summary_net,
        )
        approximator.compile(optimizer="adam")
        dataset = self._make_precomputed_dataset(
            simulator, adapter, batch_size=4, num_batches=2
        )

        history = approximator.fit(
            dataset=dataset,
            epochs=1,
        )
        assert all(np.isfinite(v) for v in history.history["loss"])

        test_data = simulator.sample((2,))
        samples = approximator.sample(num_samples=12, conditions=test_data)
        assert "a" in samples and "b" in samples
        assert samples["a"].shape[:2] == (2, 12)
        assert np.all(np.isfinite(samples["a"]))

    def test_matrix_2pl_mean_pma_train_and_sample(self):
        simulator = make_2pl_simulator(
            n_items=(5, 9),
            n_persons=(10, 16),
            prior_a=_PRIOR_A,
            prior_b=_PRIOR_B,
            prior_theta=_PRIOR_THETA,
        )
        adapter = make_2pl_adapter(simulator)
        summary_net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=16,
            num_matrix_blocks=2,
            person_pooling="mean_pma",
            num_pma_seeds=2,
            num_outer_sabs=1,
            outer_embed_dim=16,
            outer_mlp_width=32,
        )
        approximator = EquivariantIRTApproximator(
            adapter=adapter,
            inference_network=_make_flow(),
            item_posterior="flow_matching",
            summary_network=summary_net,
        )
        approximator.compile(optimizer="adam")
        dataset = self._make_precomputed_dataset(
            simulator, adapter, batch_size=4, num_batches=2
        )

        history = approximator.fit(
            dataset=dataset,
            epochs=1,
        )
        assert all(np.isfinite(v) for v in history.history["loss"])

        test_data = simulator.sample((2,))
        samples = approximator.sample(num_samples=12, conditions=test_data)
        assert "a" in samples and "b" in samples
        assert samples["a"].shape[:2] == (2, 12)
        assert np.all(np.isfinite(samples["a"]))

    def test_matrix_2pl_raw_feature_channels_train_and_sample(self):
        simulator = make_2pl_simulator(
            n_items=(5, 9),
            n_persons=(10, 16),
            prior_a=_PRIOR_A,
            prior_b=_PRIOR_B,
            prior_theta=_PRIOR_THETA,
        )
        adapter = make_2pl_adapter(simulator)
        summary_net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=16,
            num_matrix_blocks=2,
            raw_feature_channels=("rest_mean_score", "item_rest_correlation"),
            num_outer_sabs=1,
            outer_embed_dim=16,
            outer_mlp_width=32,
        )
        approximator = EquivariantIRTApproximator(
            adapter=adapter,
            inference_network=_make_flow(),
            item_posterior="flow_matching",
            summary_network=summary_net,
        )
        approximator.compile(optimizer="adam")
        dataset = self._make_precomputed_dataset(
            simulator, adapter, batch_size=4, num_batches=2
        )

        history = approximator.fit(
            dataset=dataset,
            epochs=1,
        )
        assert all(np.isfinite(v) for v in history.history["loss"])

        test_data = simulator.sample((2,))
        samples = approximator.sample(num_samples=12, conditions=test_data)
        assert "a" in samples and "b" in samples
        assert samples["a"].shape[:2] == (2, 12)
        assert np.all(np.isfinite(samples["a"]))

    def test_matrix_2pl_parameter_heads_train_and_sample(self):
        simulator = make_2pl_simulator(
            n_items=(5, 9),
            n_persons=(10, 16),
            prior_a=_PRIOR_A,
            prior_b=_PRIOR_B,
            prior_theta=_PRIOR_THETA,
        )
        adapter = make_2pl_adapter(simulator)
        summary_net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=16,
            num_matrix_blocks=2,
            num_parameter_heads=2,
            num_outer_sabs=1,
            outer_embed_dim=16,
            outer_mlp_width=32,
        )
        approximator = EquivariantIRTApproximator(
            adapter=adapter,
            inference_network=_make_flow(),
            item_posterior="flow_matching",
            summary_network=summary_net,
        )
        approximator.compile(optimizer="adam")
        dataset = self._make_precomputed_dataset(
            simulator, adapter, batch_size=4, num_batches=2
        )

        history = approximator.fit(
            dataset=dataset,
            epochs=1,
        )
        assert all(np.isfinite(v) for v in history.history["loss"])

        test_data = simulator.sample((2,))
        samples = approximator.sample(num_samples=12, conditions=test_data)
        assert "a" in samples and "b" in samples
        assert samples["a"].shape[:2] == (2, 12)
        assert np.all(np.isfinite(samples["a"]))

    def test_matrix_pcm_train_and_sample(self):
        k = 4
        simulator = make_pcm_simulator(
            n_items=(5, 8),
            n_persons=(10, 15),
            n_categories=k,
            prior_delta=_PRIOR_DELTA,
        )
        adapter = make_pcm_adapter(simulator)
        summary_net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=16,
            num_matrix_blocks=1,
            num_outer_sabs=1,
            outer_embed_dim=16,
            outer_mlp_width=32,
        )
        approximator = EquivariantIRTApproximator(
            adapter=adapter,
            inference_network=bf.networks.FlowMatching(depth=2),
            item_posterior="flow_matching",
            summary_network=summary_net,
        )
        approximator.compile(optimizer="adam")
        dataset = self._make_precomputed_dataset(
            simulator, adapter, batch_size=4, num_batches=2
        )

        history = approximator.fit(
            dataset=dataset,
            epochs=1,
        )
        assert all(np.isfinite(v) for v in history.history["loss"])

        test_data = simulator.sample((2,))
        samples = approximator.sample(num_samples=8, conditions=test_data)
        assert "delta" in samples
        assert samples["delta"].shape[:2] == (2, 8)
        assert samples["delta"].shape[-1] == k - 1
        assert np.all(np.isfinite(samples["delta"]))

    def test_matrix_gpcm_train_and_sample(self):
        k = 4
        simulator = make_gpcm_simulator(
            n_items=(5, 8),
            n_persons=(10, 15),
            n_categories=k,
            prior_a=_PRIOR_A,
            prior_delta=_PRIOR_DELTA,
        )
        adapter = make_gpcm_adapter(simulator)
        summary_net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=16,
            num_matrix_blocks=1,
            num_outer_sabs=1,
            outer_embed_dim=16,
            outer_mlp_width=32,
        )
        approximator = EquivariantIRTApproximator(
            adapter=adapter,
            inference_network=bf.networks.FlowMatching(depth=2),
            item_posterior="flow_matching",
            summary_network=summary_net,
        )
        approximator.compile(optimizer="adam")
        dataset = self._make_precomputed_dataset(
            simulator, adapter, batch_size=4, num_batches=2
        )

        history = approximator.fit(
            dataset=dataset,
            epochs=1,
        )
        assert all(np.isfinite(v) for v in history.history["loss"])

        test_data = simulator.sample((2,))
        samples = approximator.sample(num_samples=8, conditions=test_data)
        assert "a" in samples
        assert "delta" in samples
        assert samples["a"].shape[:2] == (2, 8)
        assert samples["delta"].shape[:2] == (2, 8)
        assert samples["delta"].shape[-1] == k - 1
        assert np.all(np.isfinite(samples["a"]))
        assert np.all(np.isfinite(samples["delta"]))

    def test_matrix_gpcm_raw_feature_channels_train_and_sample(self):
        k = 4
        simulator = make_gpcm_simulator(
            n_items=(5, 8),
            n_persons=(10, 15),
            n_categories=k,
            prior_a=_PRIOR_A,
            prior_delta=_PRIOR_DELTA,
        )
        adapter = make_gpcm_adapter(simulator)
        summary_net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=16,
            num_matrix_blocks=1,
            raw_feature_channels=("rest_mean_score", "item_rest_correlation"),
            num_outer_sabs=1,
            outer_embed_dim=16,
            outer_mlp_width=32,
        )
        approximator = EquivariantIRTApproximator(
            adapter=adapter,
            inference_network=bf.networks.FlowMatching(depth=2),
            item_posterior="flow_matching",
            summary_network=summary_net,
        )
        approximator.compile(optimizer="adam")
        dataset = self._make_precomputed_dataset(
            simulator, adapter, batch_size=4, num_batches=2
        )

        history = approximator.fit(
            dataset=dataset,
            epochs=1,
        )
        assert all(np.isfinite(v) for v in history.history["loss"])

        test_data = simulator.sample((2,))
        samples = approximator.sample(num_samples=8, conditions=test_data)
        assert "a" in samples
        assert "delta" in samples
        assert samples["a"].shape[:2] == (2, 8)
        assert samples["delta"].shape[:2] == (2, 8)
        assert samples["delta"].shape[-1] == k - 1
        assert np.all(np.isfinite(samples["a"]))
        assert np.all(np.isfinite(samples["delta"]))


class TestTrainingHelpers:
    """Integration tests for probe_memory_config and fit_with_oom_retry."""

    @staticmethod
    def _make_small_approximator():
        simulator = make_2pl_simulator(
            n_items=(5, 8),
            n_persons=(10, 15),
            prior_a=_PRIOR_A,
            prior_b=_PRIOR_B,
            prior_theta=_PRIOR_THETA,
        )
        adapter = make_2pl_adapter(simulator)
        summary_net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=16,
            num_matrix_blocks=1,
            num_outer_sabs=1,
            outer_embed_dim=16,
            outer_mlp_width=32,
        )
        approximator = EquivariantIRTApproximator(
            adapter=adapter,
            inference_network=_make_flow(),
            item_posterior="flow_matching",
            summary_network=summary_net,
        )
        approximator.compile(optimizer="adam")
        return approximator, adapter, simulator

    def test_fit_with_oom_retry_succeeds(self):
        """fit_with_oom_retry should complete a real training loop."""
        approx, adapter, sim = self._make_small_approximator()
        result = fit_with_oom_retry(
            approx,
            sim,
            start_batch_size=4,
            min_batch_size=2,
            fit_kwargs=dict(epochs=1, num_batches=2),
        )
        assert isinstance(result, FitResult)
        assert result.attempts == 1
        assert result.batch_size == 4
        losses = result.history.history["loss"]
        assert all(np.isfinite(loss) for loss in losses)

    def test_approximator_build_uses_rank2_flow_shapes(self):
        """Explicit build must succeed for FlowMatching subnet construction."""
        approx, _adapter, _sim = self._make_small_approximator()

        data_shapes = {
            "summary_variables": (4, 12, 8),
            "inference_variables": (4, 8, 2),
        }
        approx.build(data_shapes)

        assert approx.built
        assert approx.inference_network.built

    def test_probe_memory_config_cpu(self):
        """On CPU the probe should return an empty report gracefully."""
        import torch

        if torch.cuda.is_available():
            pytest.skip("This test is for CPU-only environments.")

        approx, _adapter, sim = self._make_small_approximator()

        def make_sim(**kw):
            return make_2pl_simulator(
                **kw, prior_a=_PRIOR_A, prior_b=_PRIOR_B, prior_theta=_PRIOR_THETA
            )

        report = probe_memory_config(
            approx,
            make_sim,
            batch_size=4,
            n_items=8,
            n_persons=15,
        )
        assert len(report.trials) == 0
        assert report.best is None

    def test_probe_does_not_build_original(self):
        """Probing must never build (allocate weights for) the original approximator.

        The disposable-clone approach should only build a temporary clone on
        the GPU; the original stays unbuilt so its optimizer state and weights
        are pristine when real training begins.
        """
        import torch

        if not torch.cuda.is_available():
            pytest.skip("GPU required for memory probing.")

        approx, _adapter, sim = self._make_small_approximator()
        assert not approx.built, "Approximator should start unbuilt"

        def make_sim(**kw):
            return make_2pl_simulator(
                **kw, prior_a=_PRIOR_A, prior_b=_PRIOR_B, prior_theta=_PRIOR_THETA
            )

        report = probe_memory_config(
            approx,
            make_sim,
            batch_size=4,
            n_items=8,
            n_persons=15,
            search=ChunkSearchOptions(
                candidate_item_chunk_sizes=[None], checkpoint_options=[False]
            ),
            verbose=False,
        )

        # Original must still be unbuilt — only the clone was used.
        assert not approx.built, (
            "probe_memory_config built the original approximator; "
            "the disposable-clone approach is broken"
        )

        # Best config should still be applied to the summary network.
        if report.best is not None:
            assert approx.summary_network.item_chunk_size == report.best.item_chunk_size
            assert (
                approx.summary_network.gradient_checkpointing
                == report.best.gradient_checkpointing
            )

    def test_probe_clone_supports_matrix_summary(self):
        """Disposable probe clone should preserve MatrixEquivariantIRTSummary config."""
        simulator = make_2pl_simulator(
            n_items=(5, 8),
            n_persons=(10, 15),
            prior_a=_PRIOR_A,
            prior_b=_PRIOR_B,
            prior_theta=_PRIOR_THETA,
        )
        adapter = make_2pl_adapter(simulator)
        summary_net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=12,
            num_matrix_blocks=2,
            num_outer_sabs=1,
            outer_embed_dim=16,
        )
        approx = EquivariantIRTApproximator(
            adapter=adapter,
            inference_network=_make_flow(),
            item_posterior="flow_matching",
            summary_network=summary_net,
        )
        approx.compile(optimizer="adam")

        clone = _build_probe_approximator(approx)
        try:
            assert isinstance(clone.summary_network, MatrixEquivariantIRTSummary)
            cfg = clone.summary_network.get_config()
            assert cfg["matrix_hidden_dim"] == 12
            assert cfg["num_matrix_blocks"] == 2
        finally:
            del clone
