"""Tests for bayesflow_irt.networks — shape & permutation invariance/equivariance."""

import warnings
from unittest import mock

import keras
import keras.ops as ops
import numpy as np
import pytest

import bayesflow_irt.networks as bfi_networks
from bayesflow_irt.networks import (
    MatrixEquivariantIRTSummary,
    _get_chunked_trainable_matrix_pool_fn,
    _HartfordEMNBlock,
    _widened_person_mean,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BATCH, N_PERSONS, N_ITEMS = 4, 20, 10


def _random_responses(batch=BATCH, n_persons=N_PERSONS, n_items=N_ITEMS):
    return (
        np.random.default_rng(42)
        .binomial(1, 0.5, size=(batch, n_persons, n_items))
        .astype(np.float32)
    )


def _chunked_hartford_forward(block, x, person_chunk_size, training=False):
    """Reference two-pass person-chunked forward for ``_HartfordEMNBlock``.

    Proves that the two-pass person-chunked forward is exact, independent
    of the production implementation. Not the production API, which wires an
    equivalent computation into ``_HartfordEMNBlock`` itself behind a
    ``person_chunk_size`` constructor argument. Here ``x`` is still fully
    materialized (this verifies *numerical exactness*, not memory
    bounds).

    Pass 1 (stats): accumulate the person-axis reductions (``col_mean``,
    ``global_mean``) via chunked sums, since both are associative reductions
    over persons. Pass 2 (apply): recombine each chunk's own ``row_mean``
    (no cross-person dependency) with the finalized ``col_mean``/``global_mean``
    from pass 1, then run the same projections/LayerNorm/activation as
    ``block.call()``.
    """
    n_persons = x.shape[1]
    n_items = x.shape[2]
    starts = list(range(0, n_persons, person_chunk_size))

    col_sum = None
    global_sum = None
    for start in starts:
        end = min(start + person_chunk_size, n_persons)
        chunk = x[:, start:end]
        chunk_col_sum = ops.sum(chunk, axis=1, keepdims=True)
        chunk_global_sum = ops.sum(chunk, axis=(1, 2), keepdims=True)
        col_sum = chunk_col_sum if col_sum is None else col_sum + chunk_col_sum
        global_sum = (
            chunk_global_sum if global_sum is None else global_sum + chunk_global_sum
        )
    col_mean = col_sum / n_persons
    global_mean = global_sum / (n_persons * n_items)
    col_term = block.col_mean_proj(col_mean)
    global_term = block.global_proj(global_mean)

    outputs = []
    for start in starts:
        end = min(start + person_chunk_size, n_persons)
        chunk = x[:, start:end]
        row_mean = ops.mean(chunk, axis=2, keepdims=True)
        y = (
            block.self_proj(chunk)
            + block.row_mean_proj(row_mean)
            + col_term
            + global_term
            + block.bias
        )
        y = block.norm(y)
        y = block.activation_layer(y)
        if block.dropout_layer is not None:
            y = block.dropout_layer(y, training=training)
        outputs.append(y)
    return ops.concatenate(outputs, axis=1)


# ---------------------------------------------------------------------------
# MatrixEquivariantIRTSummary
# ---------------------------------------------------------------------------


class TestMatrixEquivariantIRTSummary:
    def test_item_chunk_size_warns_as_ignored(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            MatrixEquivariantIRTSummary(embed_dim=16, item_chunk_size=8)
            assert any("ignores item_chunk_size" in str(w.message) for w in caught)

    def test_output_shape(self):
        net = MatrixEquivariantIRTSummary(
            embed_dim=16, matrix_hidden_dim=16, num_matrix_blocks=2
        )
        out = net(_random_responses())
        assert ops.shape(out) == (BATCH, N_ITEMS, 16)

    def test_4d_output_shape(self):
        net = MatrixEquivariantIRTSummary(
            embed_dim=16, matrix_hidden_dim=16, num_matrix_blocks=1
        )
        out = net(_random_onehot_responses())
        assert ops.shape(out) == (BATCH, N_ITEMS, 16)

    def test_variable_num_persons_and_items(self):
        net = MatrixEquivariantIRTSummary(
            embed_dim=16, matrix_hidden_dim=12, num_matrix_blocks=1
        )
        out1 = net(_random_responses(n_persons=20, n_items=10))
        out2 = net(_random_responses(n_persons=45, n_items=13))
        assert ops.shape(out1) == (BATCH, 10, 16)
        assert ops.shape(out2) == (BATCH, 13, 16)

    def test_person_invariance(self):
        net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=16,
            num_matrix_blocks=2,
            num_outer_sabs=1,
        )
        rng = np.random.default_rng(13)
        x = _random_responses()
        out_orig = ops.convert_to_numpy(net(x, training=False))
        perm = rng.permutation(N_PERSONS)
        out_perm = ops.convert_to_numpy(net(x[:, perm, :], training=False))
        np.testing.assert_allclose(out_orig, out_perm, atol=1e-5)

    def test_item_equivariance(self):
        net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=16,
            num_matrix_blocks=2,
            num_outer_sabs=1,
        )
        rng = np.random.default_rng(14)
        x = _random_responses()
        out_orig = ops.convert_to_numpy(net(x, training=False))
        perm = rng.permutation(N_ITEMS)
        out_perm = ops.convert_to_numpy(net(x[:, :, perm], training=False))
        np.testing.assert_allclose(out_orig[:, perm, :], out_perm, atol=1e-5)

    def test_person_pooling_pma(self):
        net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=16,
            num_matrix_blocks=1,
            person_pooling="pma",
            num_pma_heads=2,
            num_pma_seeds=1,
            num_outer_sabs=1,
        )
        out = net(_random_responses())
        assert ops.shape(out) == (BATCH, N_ITEMS, 16)

    @pytest.mark.parametrize("person_pooling", ["pma", "mean_pma"])
    def test_pma_builds_with_unknown_person_count(self, person_pooling):
        net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=16,
            num_matrix_blocks=1,
            person_pooling=person_pooling,
            num_pma_heads=2,
            num_pma_seeds=1,
            num_outer_sabs=1,
        )
        net.build((None, None, N_ITEMS))
        assert net.person_pma.built
        out = net(_random_responses())
        assert ops.shape(out) == (BATCH, N_ITEMS, 16)

    def test_person_chunk_size_ignored_when_pma_pooling(self):
        """Mutating person_chunk_size post-construction (as
        bayesflow_irt.utils.set_summary_memory_config does during
        fit_with_oom_retry / probe_memory_config) must not enable chunking
        for a PMA-pooling instance. __init__ only validates
        person_pooling == "mean" against person_chunk_size when the latter
        is provided at construction time, not on later mutation -- PMA
        pooling does not decompose into the chunked-sum reduction that
        person chunking relies on, so taking the chunked path here would
        silently replace attention pooling with a mean pool. See
        MatrixEquivariantIRTSummary._person_chunking_config_is_eligible."""
        net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=16,
            num_matrix_blocks=1,
            person_pooling="pma",
            num_pma_heads=2,
            num_pma_seeds=1,
            num_outer_sabs=1,
            dropout=0.0,
        )
        x = _random_responses(n_persons=20)
        dense_out = net(x, training=False)

        # Mutate person_chunk_size the way set_summary_memory_config would;
        # n_persons=20 > person_chunk_size=4 so chunking would trigger if
        # the eligibility gate were missing.
        net.person_chunk_size = 4
        chunked_out = net(x, training=False)

        np.testing.assert_allclose(
            ops.convert_to_numpy(dense_out),
            ops.convert_to_numpy(chunked_out),
            atol=1e-6,
        )

    def test_person_pooling_pma_multi_seed(self):
        net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=16,
            num_matrix_blocks=1,
            person_pooling="pma",
            num_pma_heads=2,
            num_pma_seeds=3,
            num_outer_sabs=1,
        )
        out = net(_random_responses())
        assert ops.shape(out) == (BATCH, N_ITEMS, 16)

    def test_person_pooling_mean_pma_single_seed(self):
        net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=16,
            num_matrix_blocks=1,
            person_pooling="mean_pma",
            num_pma_heads=2,
            num_pma_seeds=1,
            num_outer_sabs=1,
        )
        out = net(_random_responses())
        assert ops.shape(out) == (BATCH, N_ITEMS, 16)

    def test_person_pooling_mean_pma_multi_seed(self):
        net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=16,
            num_matrix_blocks=1,
            person_pooling="mean_pma",
            num_pma_heads=2,
            num_pma_seeds=3,
            num_outer_sabs=1,
        )
        out = net(_random_responses())
        assert ops.shape(out) == (BATCH, N_ITEMS, 16)

    def test_person_pooling_mean_std(self):
        net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=16,
            num_matrix_blocks=1,
            person_pooling="mean_std",
            num_outer_sabs=1,
        )
        out = net(_random_responses())
        assert ops.shape(out) == (BATCH, N_ITEMS, 16)

    def test_person_pooling_mean_max(self):
        net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=16,
            num_matrix_blocks=1,
            person_pooling="mean_max",
            num_outer_sabs=1,
        )
        out = net(_random_responses())
        assert ops.shape(out) == (BATCH, N_ITEMS, 16)

    def test_mean_max_doubles_pooled_width_without_adding_parameters(self):
        """Like the std branch, max is a reduction rather than a layer."""
        kwargs = dict(
            embed_dim=16,
            matrix_hidden_dim=16,
            num_matrix_blocks=1,
            num_outer_sabs=1,
        )
        mean_net = MatrixEquivariantIRTSummary(person_pooling="mean", **kwargs)
        max_net = MatrixEquivariantIRTSummary(person_pooling="mean_max", **kwargs)
        mean_net(_random_responses())
        max_net(_random_responses())
        assert max_net._pooled_dim == 2 * mean_net._pooled_dim

    def test_person_invariance_mean_max(self):
        """Max is order-invariant, so the item embeddings must be too.

        Worth testing separately from mean/std: max is the one aggregator
        here whose output is decided by a single person per item, so an
        indexing error would survive a mean-based check.
        """
        import numpy as np

        net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=16,
            num_matrix_blocks=2,
            person_pooling="mean_max",
            num_outer_sabs=1,
        )
        x = _random_responses()
        first = ops.convert_to_numpy(net(x, training=False))
        perm = np.random.default_rng(0).permutation(ops.shape(x)[1])
        second = ops.convert_to_numpy(net(ops.take(x, perm, axis=1), training=False))
        assert np.allclose(first, second, atol=1e-5)

    def test_mean_max_rejects_person_chunking(self):
        """Same deferral as mean_std: the chunked path stages one accumulator.

        Max *is* associative and so decomposes in principle; the guard is an
        implementation boundary, not a mathematical one.
        """
        with pytest.raises(ValueError, match="person_pooling='mean'"):
            MatrixEquivariantIRTSummary(
                person_pooling="mean_max",
                person_chunk_size=4,
                num_matrix_blocks=1,
            )

    def test_mean_std_doubles_pooled_width_without_adding_parameters(self):
        """The std branch is a reduction, not a layer.

        If it ever gains parameters, the comparison against `mean` stops
        isolating the pooling operation and starts confounding it with
        capacity -- which is the entire point of preferring it to PMA.
        """
        kwargs = dict(
            embed_dim=16,
            matrix_hidden_dim=16,
            num_matrix_blocks=1,
            num_outer_sabs=1,
        )
        mean_net = MatrixEquivariantIRTSummary(person_pooling="mean", **kwargs)
        std_net = MatrixEquivariantIRTSummary(person_pooling="mean_std", **kwargs)

        assert mean_net._pooled_dim == 16
        assert std_net._pooled_dim == 32

        mean_net(_random_responses())
        std_net(_random_responses())

        def block_params(net):
            return sum(
                int(np.prod(w.shape)) for b in net.matrix_blocks for w in b.weights
            )

        # The wider pooled vector feeds the outer bridge, so `mean_std` carries
        # a few more weights there; the pooling itself must add none. Compare
        # the matrix-block stacks, which the pooling change must not touch.
        assert block_params(mean_net) == block_params(std_net)

    def test_mean_std_is_finite_and_differentiable_at_zero_variance(self):
        """A constant item has exactly zero person-axis variance.

        `d/dv sqrt(v)` is unbounded as v -> 0, so without the epsilon floor an
        item everyone answers identically emits nan gradients that propagate
        to every weight. This is not a hypothetical: all-correct and
        all-incorrect items occur routinely in simulated response matrices.
        """
        import numpy as np
        import torch

        net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=8,
            num_matrix_blocks=1,
            person_pooling="mean_std",
            num_outer_sabs=1,
            dropout=0.0,
        )
        # Every person answers every item identically -> zero variance in every
        # cell channel, the exact degenerate case.
        constant = ops.convert_to_tensor(np.ones((2, 6, 4), dtype="float32"))
        assert bool(ops.all(ops.isfinite(net(constant))))

        with torch.enable_grad():
            loss = ops.sum(ops.square(net(constant)))
            loss.backward()

        checked = 0
        for w in net.trainable_weights:
            grad = w.value.grad
            if grad is None:
                continue
            checked += 1
            assert bool(torch.all(torch.isfinite(grad))), w.path
        assert checked > 0, "no gradients reached the weights; test proves nothing"

    def test_mean_std_rejects_person_chunking(self):
        """Chunking would skip the std accumulator entirely, not just slow it."""
        with pytest.raises(ValueError, match="person_pooling='mean'"):
            MatrixEquivariantIRTSummary(
                person_pooling="mean_std",
                person_chunk_size=4,
                num_matrix_blocks=1,
            )

    def test_person_invariance_mean_std(self):
        """Pooling must stay invariant to person order, like mean and PMA."""
        import numpy as np

        net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=16,
            num_matrix_blocks=2,
            person_pooling="mean_std",
            num_outer_sabs=1,
            dropout=0.0,
        )
        x = _random_responses()
        out = net(x)
        perm = np.random.default_rng(0).permutation(ops.shape(x)[1])
        permuted = net(ops.take(x, ops.convert_to_tensor(perm), axis=1))

        assert np.allclose(
            ops.convert_to_numpy(out), ops.convert_to_numpy(permuted), atol=1e-4
        )

    def test_person_invariance_mean_pma(self):
        net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=16,
            num_matrix_blocks=2,
            person_pooling="mean_pma",
            num_pma_seeds=2,
            num_outer_sabs=1,
        )
        rng = np.random.default_rng(113)
        x = _random_responses()
        out_orig = ops.convert_to_numpy(net(x, training=False))
        perm = rng.permutation(N_PERSONS)
        out_perm = ops.convert_to_numpy(net(x[:, perm, :], training=False))
        np.testing.assert_allclose(out_orig, out_perm, atol=1e-5)

    def test_item_equivariance_mean_pma(self):
        net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=16,
            num_matrix_blocks=2,
            person_pooling="mean_pma",
            num_pma_seeds=2,
            num_outer_sabs=1,
        )
        rng = np.random.default_rng(114)
        x = _random_responses()
        out_orig = ops.convert_to_numpy(net(x, training=False))
        perm = rng.permutation(N_ITEMS)
        out_perm = ops.convert_to_numpy(net(x[:, :, perm], training=False))
        np.testing.assert_allclose(out_orig[:, perm, :], out_perm, atol=1e-5)

    def test_from_config_round_trip(self):
        original = MatrixEquivariantIRTSummary(
            embed_dim=24,
            matrix_hidden_dim=20,
            num_matrix_blocks=3,
            person_pooling="mean",
            num_pma_seeds=2,
            use_outer_isab=True,
            num_outer_inducing_points=8,
            item_chunk_size=16,
            gradient_checkpointing=True,
        )
        config = original.get_config()
        restored = MatrixEquivariantIRTSummary.from_config(config)
        for key in (
            "embed_dim",
            "matrix_hidden_dim",
            "num_matrix_blocks",
            "person_pooling",
            "num_pma_seeds",
            "num_outer_inducing_points",
            "item_chunk_size",
            "gradient_checkpointing",
        ):
            assert config[key] == restored.get_config()[key], f"Mismatch on {key}"
        out = restored(_random_responses())
        assert ops.shape(out) == (BATCH, N_ITEMS, 24)

    def test_compute_output_shape(self):
        net = MatrixEquivariantIRTSummary(embed_dim=19)
        assert net.compute_output_shape((None, 20, 11)) == (None, 11, 19)
        assert net.compute_output_shape((None, 20, 11, 5)) == (None, 11, 19)

    def test_num_matrix_blocks_zero_path(self):
        net = MatrixEquivariantIRTSummary(
            embed_dim=16, matrix_hidden_dim=12, num_matrix_blocks=0
        )
        out = net(_random_responses())
        assert ops.shape(out) == (BATCH, N_ITEMS, 16)

    def test_num_matrix_blocks_zero_reorder_exactness(self):
        """L=0 + mean pooling must equal mean-before-projection exactly."""
        net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=12,
            num_matrix_blocks=0,
            person_pooling="mean",
            num_outer_sabs=1,
            outer_embed_dim=16,
            dropout=0.0,
        )
        x = _random_responses()

        out_fast = ops.convert_to_numpy(net(x, training=False))

        x4 = ops.expand_dims(ops.convert_to_tensor(x), axis=-1)
        manual = ops.mean(x4, axis=1)
        manual = net.cell_proj(manual)
        if net.bridge is not None:
            manual = net.bridge(manual)
        for sab in net.outer_sabs:
            manual = sab(manual, training=False)
        manual = net.output_proj(manual)

        np.testing.assert_allclose(out_fast, ops.convert_to_numpy(manual), atol=1e-6)

    def test_uint8_input_matches_float32(self):
        net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=12,
            num_matrix_blocks=1,
            num_outer_sabs=1,
            dropout=0.0,
        )
        x_float = _random_responses()
        x_uint8 = x_float.astype(np.uint8)
        out_float = ops.convert_to_numpy(net(x_float, training=False))
        out_uint8 = ops.convert_to_numpy(net(x_uint8, training=False))
        np.testing.assert_allclose(out_float, out_uint8, atol=1e-5)

    def test_hartford_block_reduces_to_self_term_when_pools_zeroed(self):
        block = _HartfordEMNBlock(hidden_dim=7, activation="linear", dropout=0.0)
        x = np.random.default_rng(321).normal(size=(2, 5, 4, 3)).astype(np.float32)
        _ = block(x, training=False)  # build

        block.row_mean_proj.kernel.assign(
            np.zeros_like(ops.convert_to_numpy(block.row_mean_proj.kernel))
        )
        block.col_mean_proj.kernel.assign(
            np.zeros_like(ops.convert_to_numpy(block.col_mean_proj.kernel))
        )
        block.global_proj.kernel.assign(
            np.zeros_like(ops.convert_to_numpy(block.global_proj.kernel))
        )

        out = ops.convert_to_numpy(block(x, training=False))
        expected = block.self_proj(x) + block.bias
        expected = block.norm(expected)
        expected = block.activation_layer(expected)
        np.testing.assert_allclose(out, ops.convert_to_numpy(expected), atol=1e-6)

    def test_hartford_block_chunked_forward_matches_dense(self):
        """The two-pass person-chunked forward (`_chunked_hartford_forward`)
        must exactly reproduce `_HartfordEMNBlock.call()`'s dense output — the
        core correctness invariant of person chunking. Covers chunk_size=1
        (finest), a non-divisible chunk size, chunk_size == n_persons, and
        chunk_size > n_persons (single chunk covering everything).

        Scope note: this invariant is proven for `dropout=0.0` only (and,
        via `training=False` here, dropout would no-op anyway even if
        nonzero). It does NOT extend to `dropout > 0` with `training=True` —
        see `test_hartford_block_chunked_forward_dropout_is_distributional_not_bitwise`
        for why, and what invariant *does* hold in that case."""
        # Seed the weight init: it draws from the global Keras RNG, whose state
        # depends on which tests ran earlier, and float32 summation-order error
        # sits close enough to the 1e-6 tolerance for some inits to exceed it.
        keras.utils.set_random_seed(0)
        block = _HartfordEMNBlock(hidden_dim=6, activation="mish", dropout=0.0)
        n_persons = 17
        x = (
            np.random.default_rng(7)
            .normal(size=(2, n_persons, 5, 3))
            .astype(np.float32)
        )
        dense_out = ops.convert_to_numpy(block(x, training=False))

        for chunk_size in (1, 4, n_persons, n_persons * 2):
            chunked_out = ops.convert_to_numpy(
                _chunked_hartford_forward(block, x, person_chunk_size=chunk_size)
            )
            np.testing.assert_allclose(
                chunked_out,
                dense_out,
                atol=1e-6,
                err_msg=f"Mismatch at person_chunk_size={chunk_size}",
            )

    def test_hartford_block_chunked_forward_dropout_is_distributional_not_bitwise(self):
        """MatrixEquivariantIRTSummary's default
        `dropout=0.05` is passed into these blocks during training, but the
        exactness test above only covers `dropout=0.0`/`training=False`. This
        test documents and verifies the actual invariant under nonzero
        dropout + training=True:

        - Dense calls `dropout_layer` ONCE over the full `(B, P, I, H)`
          tensor; the chunked path calls it ONCE PER CHUNK. Both apply the
          same per-cell Bernoulli(rate) protocol, but consume RNG state
          differently, so the two are NOT expected to produce the same
          realized mask/output — unlike `gradient_checkpointing`, which
          recomputes deterministically and IS bit-exact even with dropout.
        - What chunking preserves under dropout is the *distribution*, not
          the *realization*: the empirical fraction of zeroed elements
          should be close to `rate` for both the dense call and the
          concatenated chunked calls, on a large enough tensor.

        This narrows the OOM backoff ladder's framing: person/item
        chunking is "no change to the modeling distribution" (safe to place
        before batch-size reduction), not "bit-identical to dense" in the
        way gradient_checkpointing is, whenever dropout is active.
        """
        rate = 0.3
        block = _HartfordEMNBlock(hidden_dim=4, activation="linear", dropout=rate)
        # Large enough that both the dense and chunked empirical zero-rates
        # are stable estimates of the true dropout rate.
        n_persons = 400
        x = (
            np.random.default_rng(11)
            .normal(size=(1, n_persons, 6, 3))
            .astype(np.float32)
        )

        keras.utils.set_random_seed(0)
        dense_out = ops.convert_to_numpy(block(x, training=True))
        keras.utils.set_random_seed(1)
        chunked_out = ops.convert_to_numpy(
            _chunked_hartford_forward(block, x, person_chunk_size=32, training=True)
        )

        # Not bit-identical: different RNG consumption per chunk-call vs.
        # one monolithic call. (Astronomically unlikely to coincide for a
        # tensor this size, so this is a meaningful assertion, not a flake.)
        assert not np.allclose(dense_out, chunked_out, atol=1e-6), (
            "Expected dense and chunked dropout masks to differ (different "
            "RNG call granularity) — if this now matches, dropout handling "
            "changed and the surrounding docstring needs revisiting."
        )

        # But both should realize approximately the same per-cell dropout
        # rate — same distribution, different draw.
        dense_zero_rate = float(np.mean(dense_out == 0.0))
        chunked_zero_rate = float(np.mean(chunked_out == 0.0))
        assert abs(dense_zero_rate - rate) < 0.05, dense_zero_rate
        assert abs(chunked_zero_rate - rate) < 0.05, chunked_zero_rate

    def test_hartford_block_person_chunk_size_matches_dense(self):
        """The production `person_chunk_size` constructor argument
        (not the standalone `_chunked_hartford_forward` helper) must exactly
        reproduce the dense forward, on the same weights, for multiple
        `H_in` values and both divisible and non-divisible chunk sizes.
        """
        n_persons = 17
        for n_items, h_in in ((5, 3), (9, 1)):
            block = _HartfordEMNBlock(hidden_dim=6, activation="mish", dropout=0.0)
            x = (
                np.random.default_rng(7)
                .normal(size=(2, n_persons, n_items, h_in))
                .astype(np.float32)
            )
            dense_out = ops.convert_to_numpy(block(x, training=False))  # builds

            # n_persons and n_persons * 2 are intentionally dense-vs-dense
            # sanity checks: person_chunk_size < n_persons is required to
            # actually engage the chunked path (see call()'s gate), so those
            # two entries confirm the >= n_persons case is a correct no-op,
            # not a check on _chunked_call itself. n_persons - 1 is the
            # smallest value that still engages chunking and is the
            # nearest-boundary case (two chunks, one nearly the full size).
            for chunk_size in (1, 4, n_persons - 1, n_persons, n_persons * 2):
                block.person_chunk_size = chunk_size
                chunked_out = ops.convert_to_numpy(block(x, training=False))
                np.testing.assert_allclose(
                    chunked_out,
                    dense_out,
                    atol=1e-6,
                    err_msg=(
                        f"Mismatch at person_chunk_size={chunk_size}, "
                        f"n_items={n_items}, h_in={h_in}"
                    ),
                )
            block.person_chunk_size = None

    def test_hartford_block_chunked_forward_is_permutation_equivariant(self):
        """Reshuffling persons across chunk boundaries must not change the
        per-person output beyond re-ordering it identically — i.e. chunking
        must not introduce order-dependent artifacts (e.g. from a running
        accumulator whose result depended on visitation order, which the
        stats-pass's plain summation should rule out by construction)."""
        n_persons = 23
        block = _HartfordEMNBlock(
            hidden_dim=5, activation="mish", dropout=0.0, person_chunk_size=6
        )
        x = (
            np.random.default_rng(99)
            .normal(size=(2, n_persons, 4, 3))
            .astype(np.float32)
        )
        out = ops.convert_to_numpy(block(x, training=False))

        perm = np.random.default_rng(100).permutation(n_persons)
        x_perm = x[:, perm]
        out_perm = ops.convert_to_numpy(block(x_perm, training=False))

        np.testing.assert_allclose(out_perm, out[:, perm], atol=1e-6)

    def test_hartford_block_person_chunk_size_config_roundtrip(self):
        block = _HartfordEMNBlock(
            hidden_dim=8, activation="mish", dropout=0.0, person_chunk_size=4
        )
        restored = _HartfordEMNBlock.from_config(block.get_config())
        assert restored.person_chunk_size == 4

    def test_hartford_block_person_chunk_size_rejects_non_positive(self):
        with pytest.raises(ValueError):
            _HartfordEMNBlock(hidden_dim=4, person_chunk_size=0)
        with pytest.raises(ValueError):
            _HartfordEMNBlock(hidden_dim=4, person_chunk_size=-1)

    def test_hartford_block_chunked_forward_float16_accumulates_safely(self):
        """The stats pass's chunked col_sum/
        global_sum must not accumulate in x.dtype directly: a float16
        input could overflow float16's ~65,504 max well before the (finite)
        mean is reached -- even though the dense `ops.mean` path is
        unaffected by the same data. Uses an all-ones input at a person
        count safely above that threshold and asserts the chunked path
        matches the dense path, with no inf/nan anywhere.
        """
        n_persons = 70_000  # sum of n_persons ones > float16 max (65,504)
        block = _HartfordEMNBlock(hidden_dim=3, activation="linear", dropout=0.0)
        x = np.ones((1, n_persons, 2, 2), dtype=np.float16)
        dense_out = ops.convert_to_numpy(block(x, training=False))
        assert np.all(np.isfinite(dense_out))

        block.person_chunk_size = 10_000
        chunked_out = ops.convert_to_numpy(block(x, training=False))
        assert np.all(np.isfinite(chunked_out)), (
            "chunked stats-pass accumulation overflowed float16"
        )
        np.testing.assert_allclose(chunked_out, dense_out, atol=1e-2, rtol=1e-2)

    def test_person_chunk_size_matches_dense_for_multiple_num_matrix_blocks(self):
        """Core correctness invariant: the CPU-staged, person-chunked
        cell_proj + matrix-block-stack + mean-pool pipeline
        (`_chunked_matrix_and_pool`) must exactly reproduce the dense path's
        output, on the same weights, for `num_matrix_blocks` in {1, 2, 3} and
        both divisible and non-divisible chunk sizes."""
        n_persons = 37
        for num_matrix_blocks in (1, 2, 3):
            net = MatrixEquivariantIRTSummary(
                embed_dim=8,
                matrix_hidden_dim=6,
                num_matrix_blocks=num_matrix_blocks,
                num_outer_sabs=1,
                dropout=0.0,
            )
            x = _random_responses(n_persons=n_persons)
            dense_out = ops.convert_to_numpy(net(x, training=False))  # builds

            # 37 (== n_persons) and 100 (> n_persons) are intentionally
            # dense-vs-dense sanity checks confirming the >= n_persons case
            # falls back to dense correctly -- they don't exercise
            # _chunked_matrix_and_pool itself. 36 (n_persons - 1) is the
            # nearest-boundary chunk size that does engage chunking.
            for chunk_size in (5, 36, 37, 100):
                net.person_chunk_size = chunk_size
                chunked_out = ops.convert_to_numpy(net(x, training=False))
                np.testing.assert_allclose(
                    chunked_out,
                    dense_out,
                    atol=1e-5,
                    err_msg=(
                        f"Mismatch at num_matrix_blocks={num_matrix_blocks}, "
                        f"person_chunk_size={chunk_size}"
                    ),
                )
            net.person_chunk_size = None

    def test_person_chunk_size_inert_when_num_matrix_blocks_zero(self):
        """person_chunk_size only guards the num_matrix_blocks>=1 path; with
        zero matrix blocks the already-cheap fast path runs unchanged."""
        net = MatrixEquivariantIRTSummary(
            embed_dim=8,
            matrix_hidden_dim=6,
            num_matrix_blocks=0,
            num_outer_sabs=1,
            dropout=0.0,
            person_chunk_size=4,
        )
        x = _random_responses(n_persons=20)
        out = ops.convert_to_numpy(net(x, training=False))
        assert ops.shape(out) == (BATCH, N_ITEMS, 8)

    def test_person_chunk_size_compatible_with_item_level_raw_features(self):
        """Item-level raw channels (computed from the raw, un-chunked
        response tensor) are orthogonal to the chunked matrix-block pipeline
        and must remain compatible with person_chunk_size."""
        n_persons = 29
        net = MatrixEquivariantIRTSummary(
            embed_dim=8,
            matrix_hidden_dim=6,
            num_matrix_blocks=2,
            num_outer_sabs=1,
            dropout=0.0,
            raw_feature_channels=("item_rest_correlation",),
        )
        x = _random_responses(n_persons=n_persons)
        dense_out = ops.convert_to_numpy(net(x, training=False))
        net.person_chunk_size = 6
        chunked_out = ops.convert_to_numpy(net(x, training=False))
        np.testing.assert_allclose(chunked_out, dense_out, atol=1e-5)

    def test_widened_person_mean_float16_accumulates_safely(self):
        """The final person-pool reduction of both
        `_chunked_matrix_and_pool` and `_ChunkedTrainableMatrixPool.forward`
        must not accumulate in the staged array's own dtype, or it hits the
        float16 overflow `_stats_terms` is widened to avoid. Checks the
        shared `_widened_person_mean` helper both call sites use: ten
        float16 chunks of 7,000 ones each (70,000 total) sum to `inf`
        without widening, while the dense `ops.mean` on the same
        data returns `1.0`.

        A full-network version of this test (feeding an all-ones response
        matrix through a real `MatrixEquivariantIRTSummary` and comparing
        chunked vs. dense) would not reliably catch an overflow: the matrix
        block's own LayerNorm normalizes a person-uniform input down to
        near-zero-magnitude activations, which don't overflow float16 no
        matter how many persons are summed. Testing the extracted helper
        directly avoids depending on random weight initialization
        happening to produce large enough post-LayerNorm activations.
        """
        n_persons = 70_000
        chunks = [np.ones((1, 7_000, 4), dtype=np.float16) for _ in range(10)]
        pooled = ops.convert_to_numpy(_widened_person_mean(chunks, n_persons))
        assert np.all(np.isfinite(pooled)), "_widened_person_mean overflowed float16"
        np.testing.assert_allclose(pooled, np.ones_like(pooled), atol=1e-3)

    def test_person_chunk_size_rejects_non_positive(self):
        with pytest.raises(ValueError):
            MatrixEquivariantIRTSummary(embed_dim=8, person_chunk_size=0)
        with pytest.raises(ValueError):
            MatrixEquivariantIRTSummary(embed_dim=8, person_chunk_size=-1)

    def test_person_chunk_size_rejects_non_mean_pooling(self):
        with pytest.raises(ValueError, match="person_pooling='mean'"):
            MatrixEquivariantIRTSummary(
                embed_dim=8,
                person_pooling="pma",
                num_pma_seeds=1,
                person_chunk_size=4,
            )

    def test_person_chunk_size_rejects_cell_level_raw_features(self):
        with pytest.raises(ValueError, match="cell-level raw_feature_channels"):
            MatrixEquivariantIRTSummary(
                embed_dim=8,
                raw_feature_channels=("rest_mean_score",),
                person_chunk_size=4,
            )

    def test_person_chunk_size_config_roundtrip(self):
        net = MatrixEquivariantIRTSummary(embed_dim=8, person_chunk_size=4)
        restored = MatrixEquivariantIRTSummary.from_config(net.get_config())
        assert restored.person_chunk_size == 4

    def test_gradient_checkpointing_with_person_chunk_size_warns_it_is_a_noop(self):
        """gradient_checkpointing has no effect whenever person-axis
        chunking is active -- call()'s dense-only
        `_maybe_checkpointed_matrix` call is skipped by both chunked
        branches -- so the user must be told when
        gradient_checkpointing=True stops doing anything. A
        warning is emitted at construction time whenever both are
        combined with num_matrix_blocks >= 1 (mirroring the existing
        item_chunk_size warning pattern for this class)."""
        with pytest.warns(UserWarning, match="gradient_checkpointing"):
            MatrixEquivariantIRTSummary(
                embed_dim=8,
                num_matrix_blocks=1,
                gradient_checkpointing=True,
                person_chunk_size=4,
            )

        # num_matrix_blocks == 0 never routes through person-chunking (the
        # fast path is already cheap), so no warning is warranted there.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            MatrixEquivariantIRTSummary(
                embed_dim=8,
                num_matrix_blocks=0,
                gradient_checkpointing=True,
                person_chunk_size=4,
            )

        # Either knob alone is fine.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            MatrixEquivariantIRTSummary(
                embed_dim=8, num_matrix_blocks=1, gradient_checkpointing=True
            )
            MatrixEquivariantIRTSummary(
                embed_dim=8, num_matrix_blocks=1, person_chunk_size=4
            )

    def test_chunked_matrix_and_pool_never_materializes_more_than_one_chunk(
        self, monkeypatch
    ):
        """The numerical-equivalence tests for
        `_chunked_matrix_and_pool` (inference path) prove the
        chunked output matches the dense output, but never check the actual
        mechanism its `O(person_chunk_size)` memory claim depends on -- that
        no call ever re-hydrates more than one chunk's worth of persons back
        onto the accelerator at once. `ops.convert_to_tensor` is the only
        call site in the module that re-hydrates a staged NumPy chunk (see
        `_chunked_matrix_and_pool` and `_widened_person_mean`), and every
        tensor it produces during this pipeline has the person axis at index
        1 -- so instrumenting it directly tests the invariant instead of
        proxying it through a noisy OS-level memory measurement.
        """
        n_persons = 37
        chunk_size = 6
        net = MatrixEquivariantIRTSummary(
            embed_dim=8,
            matrix_hidden_dim=6,
            num_matrix_blocks=2,
            num_outer_sabs=1,
            dropout=0.0,
        )
        x = _random_responses(n_persons=n_persons)
        net(x, training=False)  # build via the dense path first

        net.person_chunk_size = chunk_size
        max_person_axis_seen = 0
        n_calls = 0
        original_convert_to_tensor = bfi_networks.ops.convert_to_tensor

        def _spy_convert_to_tensor(arr, *args, **kwargs):
            nonlocal max_person_axis_seen, n_calls
            out = original_convert_to_tensor(arr, *args, **kwargs)
            shape = getattr(out, "shape", None)
            if shape is not None and len(shape) >= 2:
                n_calls += 1
                max_person_axis_seen = max(max_person_axis_seen, int(shape[1]))
            return out

        monkeypatch.setattr(
            bfi_networks.ops, "convert_to_tensor", _spy_convert_to_tensor
        )
        net(x, training=False)

        assert n_calls > 0, (
            "convert_to_tensor spy never fired -- test is not exercising "
            "the chunked path"
        )
        assert max_person_axis_seen <= chunk_size, (
            f"A tensor with person-axis size {max_person_axis_seen} was "
            f"materialized during a chunked call with "
            f"person_chunk_size={chunk_size} -- the chunked pipeline is "
            "holding more than one chunk's worth of persons on the "
            "accelerator at once."
        )

    def test_person_chunk_size_falls_back_to_dense_during_training(self):
        """`use_person_chunking` must respect `training`: a
        `training=True` call entering the CPU-staged (NumPy round-trip)
        path would detach from autograd and silently leave `cell_proj`/the
        matrix blocks with `None`
        gradients while the rest of the network still backpropagated fine.
        Uses a `person_chunk_size` smaller than `n_persons` (which WOULD
        engage chunking at inference time) and asserts every trainable
        weight still receives a real gradient during training -- i.e. the
        dense, differentiable path was used instead, per the same pattern
        as `test_gradient_checkpointing_produces_gradients`."""
        import torch

        net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=16,
            num_matrix_blocks=2,
            num_outer_sabs=1,
            person_chunk_size=5,
        )
        x = keras.ops.zeros((2, 30, 8, 1))
        with torch.enable_grad():
            out = net(x, training=True)
            out.sum().backward()
        none_grads = [w.path for w in net.trainable_weights if w.value.grad is None]
        assert none_grads == [], f"Weights with None gradients: {none_grads}"

    def test_can_use_trainable_chunking_requires_torch_only(self):
        """Unit test for the gate the trainable chunked path relies on.

        RNG state is preserved across the backward recompute (see
        `_get_chunked_trainable_matrix_pool_fn`'s docstring), so nonzero
        dropout does not disable this path on CPU/CUDA -- only the backend
        does. `x`'s device is irrelevant when no block has dropout, so a
        plain NumPy array (device-less) still passes.
        """
        is_torch = keras.backend.backend() == "torch"
        x = _random_responses(n_persons=8)

        net = MatrixEquivariantIRTSummary(
            embed_dim=8, matrix_hidden_dim=6, num_matrix_blocks=2, dropout=0.0
        )
        assert net._can_use_trainable_chunking(x) == is_torch

        net_dropout = MatrixEquivariantIRTSummary(
            embed_dim=8, matrix_hidden_dim=6, num_matrix_blocks=2, dropout=0.1
        )
        # With dropout the device matters: only CPU/CUDA snapshot the RNG the
        # backward recompute consumes, so pass an actual backend tensor whose
        # device the gate can inspect.
        assert (
            net_dropout._can_use_trainable_chunking(ops.convert_to_tensor(x))
            == is_torch
        )

    @pytest.mark.skipif(
        keras.backend.backend() != "torch",
        reason="Device gating only matters on the torch backend",
    )
    def test_can_use_trainable_chunking_falls_back_on_unsupported_device_with_dropout(
        self,
    ):
        """`_can_use_trainable_chunking` enables the
        RNG-preserving path for every torch device, but forward/backward only
        snapshot the CPU and CUDA RNG. A torch device with its own RNG that
        is neither (e.g. Apple MPS) would recompute dropout with a different
        mask than the forward pass used, silently corrupting gradients. With
        nonzero dropout the gate must therefore restrict itself to CPU/CUDA;
        with zero dropout there is no stochastic op and every device is fine.
        """
        import torch

        # Build the tensor directly on CPU rather than via
        # `ops.convert_to_tensor`, which honors the Keras default device and
        # would land on `cuda` on a GPU machine.
        x_cpu = torch.tensor(_random_responses(n_persons=8), device="cpu")
        assert x_cpu.device.type == "cpu"

        net_dropout = MatrixEquivariantIRTSummary(
            embed_dim=8, matrix_hidden_dim=6, num_matrix_blocks=2, dropout=0.1
        )
        net_zero = MatrixEquivariantIRTSummary(
            embed_dim=8, matrix_hidden_dim=6, num_matrix_blocks=2, dropout=0.0
        )

        # CPU is supported for both.
        assert net_dropout._can_use_trainable_chunking(x_cpu) is True
        assert net_zero._can_use_trainable_chunking(x_cpu) is True

        # Simulate a tensor on an unsupported device (its own RNG we do not
        # snapshot) without needing that hardware present: a lightweight stand
        # -in exposing only `.device.type`. Dropout -> dense fallback; zero
        # dropout -> still eligible (no stochastic op to reproduce).
        class _FakeDeviceTensor:
            device = torch.device("meta")  # type != "cpu"/"cuda"

        fake = _FakeDeviceTensor()
        assert fake.device.type not in ("cpu", "cuda")
        assert net_dropout._can_use_trainable_chunking(fake) is False
        assert net_zero._can_use_trainable_chunking(fake) is True

    @pytest.mark.skipif(
        keras.backend.backend() != "torch",
        reason="Trainable person-chunking is a torch.autograd.Function, torch-only",
    )
    def test_chunked_trainable_matrix_pool_does_not_retain_unused_final_stage(
        self, monkeypatch
    ):
        """`_ChunkedTrainableMatrixPool.forward` must not
        store every staged stage (cell_proj's output plus every block's
        output, including the last block's own output) in `ctx.staged`, but
        `backward`'s `for b in reversed(range(n_blocks))` loop only ever
        reads `staged[b]` for `b < n_blocks` -- the last block's own output
        is used only to compute `pooled_value` inside `forward` and is never
        read again during backward. Retaining it would hold an unused
        O(n_persons x n_items x H) CPU array alive for the whole backward
        pass. Asserts `ctx.staged` has exactly `num_matrix_blocks`
        entries, not `num_matrix_blocks + 1`.
        """
        import torch

        num_matrix_blocks = 3
        net = MatrixEquivariantIRTSummary(
            embed_dim=8,
            matrix_hidden_dim=6,
            num_matrix_blocks=num_matrix_blocks,
            num_outer_sabs=1,
            dropout=0.0,
            person_chunk_size=5,
        )
        x = _random_responses(n_persons=23)
        assert net._can_use_trainable_chunking(x) is True

        chunked_fn = _get_chunked_trainable_matrix_pool_fn()
        captured = {}
        original_forward = chunked_fn.forward

        def _spy_forward(ctx, *args, **kwargs):
            out = original_forward(ctx, *args, **kwargs)
            captured["n_staged"] = len(ctx.staged)
            return out

        monkeypatch.setattr(chunked_fn, "forward", staticmethod(_spy_forward))

        with torch.enable_grad():
            out = net(x, training=True)
            out.sum().backward()

        assert captured.get("n_staged") == num_matrix_blocks, (
            f"Expected ctx.staged to hold exactly {num_matrix_blocks} "
            "stages (cell_proj's output plus every block's output except "
            f"the last), got {captured.get('n_staged')} -- the unused "
            "final stage may be leaking into ctx."
        )

    @pytest.mark.skipif(
        keras.backend.backend() != "torch",
        reason="Trainable person-chunking is a torch.autograd.Function, torch-only",
    )
    def test_person_chunk_size_training_propagates_gradient_into_upstream_layer(self):
        """Regression test:
        `_ChunkedTrainableMatrixPool.backward` must not return `None` for
        `x`'s gradient on the assumption `x` is always raw response data.
        If `x` is instead produced by some upstream trainable layer, that
        would silently drop gradient into it -- a
        correctness gap the dense path does not have, since it is a plain
        differentiable pipeline. Chains a trainable `Dense` layer before the
        summary network and asserts its gradient under `person_chunk_size`
        training matches the dense reference exactly, rather than being
        `None`.
        """
        import torch

        n_persons = 23
        upstream = keras.layers.Dense(1)
        net = MatrixEquivariantIRTSummary(
            embed_dim=8,
            matrix_hidden_dim=6,
            num_matrix_blocks=1,
            num_outer_sabs=1,
            dropout=0.0,
            person_chunk_size=5,
        )
        raw = ops.convert_to_tensor(
            np.random.default_rng(0)
            .normal(size=(2, n_persons, 4, 3))
            .astype(np.float32)
        )

        net.person_chunk_size = None
        with torch.enable_grad():
            x_dense = upstream(raw)
            out_dense = net(x_dense, training=True)
            out_dense.sum().backward()
        dense_upstream_grad = upstream.kernel.value.grad.clone()
        upstream.kernel.value.grad = None
        for w in net.trainable_weights:
            w.value.grad = None

        net.person_chunk_size = 5
        assert net._can_use_trainable_chunking(x_dense) is True
        with torch.enable_grad():
            x_chunked = upstream(raw)
            out_chunked = net(x_chunked, training=True)
            out_chunked.sum().backward()

        assert upstream.kernel.value.grad is not None, (
            "upstream layer received no gradient -- person_chunk_size "
            "training silently dropped it"
        )
        np.testing.assert_allclose(
            ops.convert_to_numpy(upstream.kernel.value.grad),
            ops.convert_to_numpy(dense_upstream_grad),
            atol=1e-4,
            rtol=1e-4,
        )

    @pytest.mark.skipif(
        keras.backend.backend() != "torch",
        reason="Trainable person-chunking is a torch.autograd.Function, torch-only",
    )
    def test_person_chunk_size_training_gradients_match_dense(self):
        """With dropout=0.0 (torch backend), training must use the
        gradient-aware chunked path (_chunked_matrix_and_pool_trainable) and
        its gradients must match the dense path's exactly (multi-block,
        trainable-input gradient equivalence).
        Also asserts the chunked path was actually exercised (not a silent
        fallback to dense), by spying on both call targets.

        Sweeps multiple chunk sizes per num_matrix_blocks -- a single fixed
        chunk size would leave the per-block retain_graph loop's
        chunk-count-dependent behavior mostly unexercised at other ratios.
        Includes chunk_size=1 (many tiny chunks -- the loop's opposite
        extreme from a single chunk_size=5) and chunk_size=n_persons - 1 (a
        near-boundary case: two chunks, one nearly the full person count).
        """
        import torch

        n_persons = 23
        for num_matrix_blocks in (1, 2, 3):
            net = MatrixEquivariantIRTSummary(
                embed_dim=8,
                matrix_hidden_dim=6,
                num_matrix_blocks=num_matrix_blocks,
                num_outer_sabs=1,
                dropout=0.0,
            )
            x = _random_responses(n_persons=n_persons)
            assert net._can_use_trainable_chunking(x) is True

            # Dense reference: same weights, chunking disabled entirely.
            # Computed once per num_matrix_blocks and reused across every
            # chunk_size below.
            net.person_chunk_size = None
            with torch.enable_grad():
                dense_out = net(x, training=True)
                dense_out.sum().backward()
            dense_grads = {
                w.path: w.value.grad.clone()
                for w in net.trainable_weights
                if w.value.grad is not None
            }
            for w in net.trainable_weights:
                w.value.grad = None

            original_numpy_staged = net._chunked_matrix_and_pool
            original_trainable = net._chunked_matrix_and_pool_trainable

            for chunk_size in (1, 5, n_persons - 1):
                net.person_chunk_size = chunk_size
                numpy_staged_spy = mock.MagicMock(wraps=original_numpy_staged)
                trainable_spy = mock.MagicMock(wraps=original_trainable)
                net._chunked_matrix_and_pool = numpy_staged_spy
                net._chunked_matrix_and_pool_trainable = trainable_spy
                with torch.enable_grad():
                    chunked_out = net(x, training=True)
                    chunked_out.sum().backward()

                ctx = f"num_matrix_blocks={num_matrix_blocks}, chunk_size={chunk_size}"
                assert trainable_spy.call_count == 1, (
                    f"{ctx}: trainable chunked path was not invoked during training"
                )
                assert numpy_staged_spy.call_count == 0, (
                    f"{ctx}: NumPy-staged (autograd-detaching) path was "
                    "invoked during training"
                )
                np.testing.assert_allclose(
                    ops.convert_to_numpy(chunked_out),
                    ops.convert_to_numpy(dense_out),
                    atol=1e-5,
                    err_msg=f"{ctx}: forward output mismatch",
                )
                missing = [
                    w.path
                    for w in net.trainable_weights
                    if w.path in dense_grads and w.value.grad is None
                ]
                assert missing == [], (
                    f"{ctx}: weights with None gradient under chunking: {missing}"
                )
                for w in net.trainable_weights:
                    if w.path not in dense_grads:
                        continue
                    np.testing.assert_allclose(
                        ops.convert_to_numpy(w.value.grad),
                        ops.convert_to_numpy(dense_grads[w.path]),
                        atol=1e-4,
                        rtol=1e-4,
                        err_msg=f"{ctx}: gradient mismatch for {w.path}",
                    )
                for w in net.trainable_weights:
                    w.value.grad = None

    @pytest.mark.skipif(
        keras.backend.backend() != "torch",
        reason="Trainable person-chunking is a torch.autograd.Function, torch-only",
    )
    def test_person_chunk_size_training_gradients_match_dense_with_dropout_single_chunk(
        self,
    ):
        """With `person_chunk_size == n_persons` (exactly one
        chunk), the trainable chunked path's per-block `_apply_chunk` call
        has the identical shape and RNG-consumption pattern as the dense
        path's own per-block call -- both call it exactly once, over the
        full person axis. This is the minimal case where dense and
        chunked-trainable are structurally identical computations, so
        seeding the same `torch.manual_seed` before each and comparing is an
        exact (not merely distributional) equivalence check -- unlike the
        multi-chunk case (see the test below), where dense draws one mask
        per block and chunked draws one per (block, chunk), an inherently
        different RNG draw pattern even when the RNG-preservation mechanism
        being tested here works correctly. This directly exercises
        `_ChunkedTrainableMatrixPool`'s RNG snapshot/restore (see
        `_get_chunked_trainable_matrix_pool_fn`'s docstring): if backward's
        recompute drew a different dropout mask than forward did, this
        would fail.
        """
        import torch

        n_persons = 23
        for num_matrix_blocks in (1, 2, 3):
            net = MatrixEquivariantIRTSummary(
                embed_dim=8,
                matrix_hidden_dim=6,
                num_matrix_blocks=num_matrix_blocks,
                num_outer_sabs=1,
                dropout=0.3,
            )
            x = _random_responses(n_persons=n_persons)
            assert net._can_use_trainable_chunking(ops.convert_to_tensor(x)) is True

            net.person_chunk_size = None
            torch.manual_seed(0)
            with torch.enable_grad():
                dense_out = net(x, training=True)
                dense_out.sum().backward()
            dense_grads = {
                w.path: w.value.grad.clone()
                for w in net.trainable_weights
                if w.value.grad is not None
            }
            for w in net.trainable_weights:
                w.value.grad = None

            net.person_chunk_size = n_persons  # exactly one chunk
            torch.manual_seed(0)
            with torch.enable_grad():
                chunked_out = net(x, training=True)
                chunked_out.sum().backward()

            ctx = f"num_matrix_blocks={num_matrix_blocks}"
            np.testing.assert_allclose(
                ops.convert_to_numpy(chunked_out),
                ops.convert_to_numpy(dense_out),
                atol=1e-5,
                err_msg=f"{ctx}: forward output mismatch under dropout",
            )
            for w in net.trainable_weights:
                if w.path not in dense_grads:
                    continue
                assert w.value.grad is not None, (
                    f"{ctx}: weight {w.path} has None gradient under chunking"
                )
                np.testing.assert_allclose(
                    ops.convert_to_numpy(w.value.grad),
                    ops.convert_to_numpy(dense_grads[w.path]),
                    atol=1e-4,
                    rtol=1e-4,
                    err_msg=f"{ctx}: gradient mismatch for {w.path} under dropout",
                )
            for w in net.trainable_weights:
                w.value.grad = None

    @pytest.mark.skipif(
        keras.backend.backend() != "torch",
        reason="Trainable person-chunking is a torch.autograd.Function, torch-only",
    )
    def test_person_chunk_size_training_with_dropout_multi_chunk_is_well_formed(self):
        """With multiple chunks, dense and chunked draw dropout
        masks in different shapes/counts (one big draw per block vs. one
        per (block, chunk)), so exact numerical equivalence to a literal
        dense run isn't the right invariant here -- that's what the
        single-chunk test above already proves exactly, by construction.
        This test instead checks the structural correctness properties that
        would break if RNG preservation were silently broken or partially
        applied: the trainable chunked path is actually invoked (not a
        silent dense fallback), every trainable weight receives a real
        (non-None, finite) gradient, and the result is reproducible given a
        fixed seed.
        """
        import torch

        n_persons = 23
        net = MatrixEquivariantIRTSummary(
            embed_dim=8,
            matrix_hidden_dim=6,
            num_matrix_blocks=2,
            num_outer_sabs=1,
            dropout=0.3,
        )
        x = _random_responses(n_persons=n_persons)
        assert net._can_use_trainable_chunking(ops.convert_to_tensor(x)) is True

        original_numpy_staged = net._chunked_matrix_and_pool
        original_trainable = net._chunked_matrix_and_pool_trainable

        for chunk_size in (1, 5, n_persons - 1):
            net.person_chunk_size = chunk_size
            trainable_spy = mock.MagicMock(wraps=original_trainable)
            numpy_staged_spy = mock.MagicMock(wraps=original_numpy_staged)
            net._chunked_matrix_and_pool_trainable = trainable_spy
            net._chunked_matrix_and_pool = numpy_staged_spy

            torch.manual_seed(0)
            with torch.enable_grad():
                out = net(x, training=True)
                out.sum().backward()

            ctx = f"chunk_size={chunk_size}"
            assert trainable_spy.call_count == 1, (
                f"{ctx}: trainable chunked path was not invoked under dropout"
            )
            assert numpy_staged_spy.call_count == 0, (
                f"{ctx}: NumPy-staged (autograd-detaching) path was invoked "
                "during training"
            )
            for w in net.trainable_weights:
                grad = w.value.grad
                assert grad is not None, f"{ctx}: weight {w.path} has None gradient"
                grad_np = ops.convert_to_numpy(grad)
                assert np.all(np.isfinite(grad_np)), (
                    f"{ctx}: weight {w.path} has non-finite gradient"
                )
                w.value.grad = None

    @pytest.mark.skipif(
        keras.backend.backend() != "torch",
        reason="Trainable person-chunking is a torch.autograd.Function, torch-only",
    )
    def test_person_chunk_size_training_backward_preserves_caller_rng_state(self):
        """The trainable chunked backward restores each
        saved forward RNG snapshot to reproduce the right dropout mask, but it
        must also leave torch's global RNG stream where the caller had it when
        backward began -- otherwise a later training step draws dropout masks
        from random numbers this backward already consumed, silently
        correlating masks across steps (the checkpoint-style invariant
        `torch.utils.checkpoint(preserve_rng_state=True)` guarantees).

        Asserts the checkpoint invariant directly: torch's RNG state
        immediately after `backward()` equals its state immediately before.
        Uses multiple chunks and >=2 matrix blocks so backward actually
        recomputes several dropout draws (the case that leaks without the
        fix), and a control run with dropout=0 (no stochastic op, so the
        state is trivially preserved) to show the assertion is meaningful.
        """
        import torch

        n_persons = 23

        def _rng_state_delta_after_backward(dropout):
            net = MatrixEquivariantIRTSummary(
                embed_dim=8,
                matrix_hidden_dim=6,
                num_matrix_blocks=2,
                num_outer_sabs=1,
                dropout=dropout,
                person_chunk_size=5,  # < n_persons => multiple chunks
            )
            x = _random_responses(n_persons=n_persons)
            with torch.enable_grad():
                out = net(x, training=True)
                # Advance the RNG some arbitrary amount so "before backward"
                # is not just the freshly seeded state.
                torch.rand(7)
                before = torch.get_rng_state()
                out.sum().backward()
                after = torch.get_rng_state()
            for w in net.trainable_weights:
                w.value.grad = None
            return torch.equal(before, after)

        # With dropout, backward recomputes dropout draws; without the caller
        # -state restore, `after` would differ from `before`.
        assert _rng_state_delta_after_backward(dropout=0.3) is True
        # Control: no stochastic op at all, so preservation is trivial.
        assert _rng_state_delta_after_backward(dropout=0.0) is True

    @pytest.mark.slow
    @pytest.mark.skipif(
        keras.backend.backend() != "torch",
        reason="Trainable person-chunking is a torch.autograd.Function, torch-only",
    )
    @pytest.mark.skipif(
        not (
            keras.backend.backend() == "torch"
            and __import__("torch").cuda.is_available()
        ),
        reason="Peak-memory measurement requires a CUDA device",
    )
    def test_person_chunk_size_training_memory_stays_flat_on_gpu(self):
        """GPU memory sweep: a real ``MatrixEquivariantIRTSummary``
        forward+backward (training=True) at increasing ``n_persons`` must keep
        peak CUDA memory roughly flat, not scaling with ``n_persons``, checked
        across >=3 points (not just one), since a single data point cannot
        show a trend. Scoped to the summary
        network's own forward+backward (matching this file's existing
        gradient tests), not a full ``approximator.fit`` call.
        """
        import gc

        import torch

        net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=16,
            num_matrix_blocks=2,
            num_outer_sabs=1,
            dropout=0.0,
            person_chunk_size=2_000,
        )
        # Trigger a small build first so weights land on CUDA (Keras's torch
        # backend places new layer weights on CUDA by default when
        # available) before the real, large-n_persons calls below.
        net(torch.zeros(1, 2, 8, 1, device="cuda"), training=False)

        peaks = []
        for n_persons in (20_000, 100_000, 300_000):
            x = torch.zeros(1, n_persons, 8, 1, device="cuda")
            net.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            with torch.enable_grad():
                out = net(x, training=True)
                out.sum().backward()
            torch.cuda.synchronize()
            peaks.append(torch.cuda.max_memory_allocated() / 2**20)

        assert peaks[-1] < peaks[0] * 3, (
            f"peak memory grew from {peaks[0]:.1f} to {peaks[-1]:.1f} MiB "
            f"across a 15x increase in n_persons: {peaks}"
        )

    def test_item_chunk_size_is_idempotent(self):
        net = MatrixEquivariantIRTSummary(
            embed_dim=16, matrix_hidden_dim=16, num_matrix_blocks=1
        )
        x = _random_responses()
        net.item_chunk_size = None
        out_full = ops.convert_to_numpy(net(x, training=False))
        net.item_chunk_size = 4
        out_flagged = ops.convert_to_numpy(net(x, training=False))
        np.testing.assert_allclose(out_full, out_flagged, atol=1e-5)

    def test_gradient_checkpointing_equivalence(self):
        net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=16,
            num_matrix_blocks=2,
            gradient_checkpointing=False,
        )
        x = _random_responses()
        out_no_cp = ops.convert_to_numpy(net(x, training=False))
        net.gradient_checkpointing = True
        out_cp = ops.convert_to_numpy(net(x, training=False))
        np.testing.assert_allclose(out_no_cp, out_cp, atol=1e-5)

    @pytest.mark.skipif(
        keras.backend.backend() != "torch",
        reason="Gradient checkpointing requires PyTorch backend",
    )
    def test_gradient_checkpointing_produces_gradients(self):
        import torch

        net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=16,
            num_matrix_blocks=2,
            gradient_checkpointing=True,
            num_outer_sabs=1,
        )
        x = keras.ops.zeros((2, 30, 8, 1))
        with torch.enable_grad():
            out = net(x, training=True)
            out.sum().backward()
        none_grads = [w.path for w in net.trainable_weights if w.value.grad is None]
        assert none_grads == [], f"Weights with None gradients: {none_grads}"

    def test_invalid_input_ndim_raises(self):
        net = MatrixEquivariantIRTSummary(embed_dim=16)
        with pytest.raises(ValueError, match="Expected 3D.*or 4D"):
            net(np.zeros((4, 10), dtype=np.float32))

    def test_invalid_num_matrix_blocks_raises(self):
        with pytest.raises(ValueError, match="num_matrix_blocks must be >= 0"):
            MatrixEquivariantIRTSummary(num_matrix_blocks=-1)

    def test_invalid_person_pooling_raises(self):
        with pytest.raises(ValueError, match="person_pooling must be"):
            MatrixEquivariantIRTSummary(person_pooling="invalid")

    def test_invalid_num_pma_seeds_raises(self):
        with pytest.raises(ValueError, match="num_pma_seeds must be"):
            MatrixEquivariantIRTSummary(num_pma_seeds=0)

    def test_person_context_removed_raises(self):
        """person_context is not supported (redundant with the EMN
        row-mean term) — passing it must
        fail loudly rather than being silently accepted."""
        with pytest.raises(
            (TypeError, ValueError), match="person_context|Unrecognized"
        ):
            MatrixEquivariantIRTSummary(person_context="item_mean")


# ---------------------------------------------------------------------------
# Raw IRT feature channels (split into cell-level and item-level fusion)
# ---------------------------------------------------------------------------


class TestMatrixEquivariantIRTSummaryRawFeatures:
    ALL_CHANNELS = (
        "rest_mean_score",
        "item_rest_correlation",
    )

    def test_default_no_features_unchanged_shape(self):
        net = MatrixEquivariantIRTSummary(
            embed_dim=16, matrix_hidden_dim=12, num_matrix_blocks=1
        )
        assert net.raw_feature_channels == ()
        out = net(_random_responses())
        assert ops.shape(out) == (BATCH, N_ITEMS, 16)

    @pytest.mark.parametrize("channel", ALL_CHANNELS)
    def test_each_channel_builds_for_binary_input(self, channel):
        net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=12,
            num_matrix_blocks=1,
            raw_feature_channels=(channel,),
        )
        out = net(_random_responses())
        assert ops.shape(out) == (BATCH, N_ITEMS, 16)

    def test_person_chunk_size_ignored_for_cell_level_raw_features(self):
        """Mutating person_chunk_size post-construction must not enable
        chunking for an instance with a cell-level raw_feature_channel
        (here "rest_mean_score"). __init__ only rejects person_chunk_size +
        cell-level channels together when person_chunk_size is set at
        construction time; on later mutation, the chunked path would call
        cell_proj directly on the un-augmented (B, P, I, F) slice instead of
        the augmented (B, P, I, F + C) tensor the dense path builds
        cell_proj against, which is a shape mismatch, not just a numeric
        difference."""
        net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=12,
            num_matrix_blocks=1,
            raw_feature_channels=("rest_mean_score",),
            dropout=0.0,
        )
        x = _random_responses(n_persons=20)
        dense_out = net(x, training=False)

        net.person_chunk_size = 4
        chunked_out = net(x, training=False)

        np.testing.assert_allclose(
            ops.convert_to_numpy(dense_out),
            ops.convert_to_numpy(chunked_out),
            atol=1e-6,
        )

    def test_all_channels_together_build_for_binary_input(self):
        net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=12,
            num_matrix_blocks=1,
            raw_feature_channels=self.ALL_CHANNELS,
        )
        out = net(_random_responses())
        assert ops.shape(out) == (BATCH, N_ITEMS, 16)

    def test_all_channels_build_for_onehot_input(self):
        net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=12,
            num_matrix_blocks=1,
            raw_feature_channels=self.ALL_CHANNELS,
        )
        out = net(_random_onehot_responses())
        assert ops.shape(out) == (BATCH, N_ITEMS, 16)

    def test_cell_level_channel_disables_num_matrix_blocks_zero_fast_path(self):
        """Cell-level channels must be materialized per (person, item) cell
        before the projection, so they skip the L=0 pool-before-project fast
        path; this checks the general path handles num_matrix_blocks=0
        correctly."""
        net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=12,
            num_matrix_blocks=0,
            raw_feature_channels=("rest_mean_score",),
        )
        out = net(_random_responses())
        assert ops.shape(out) == (BATCH, N_ITEMS, 16)

    def test_item_level_channel_keeps_fast_path_equivalence(self):
        """item_rest_correlation is late-fused after person pooling, so it is
        compatible with the L=0 mean-pooling fast path: the fast path plus
        late concat must equal the manually reordered general path exactly."""
        net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=12,
            num_matrix_blocks=0,
            person_pooling="mean",
            raw_feature_channels=("item_rest_correlation",),
            num_outer_sabs=1,
            outer_embed_dim=16,
            dropout=0.0,
        )
        x = _random_responses()
        out_fast = ops.convert_to_numpy(net(x, training=False))

        # General (pool-after-project) order with the same weights.
        x4 = ops.expand_dims(ops.convert_to_tensor(x), axis=-1)
        manual = net.cell_proj(x4)  # (B, P, I, H)
        manual = ops.mean(manual, axis=1)  # (B, I, H)
        item_features = net._compute_item_features(x4)  # (B, I, 1)
        manual = ops.concatenate((manual, item_features), axis=-1)
        if net.bridge is not None:
            manual = net.bridge(manual)
        for sab in net.outer_sabs:
            manual = sab(manual, training=False)
        manual = net.output_proj(manual)

        np.testing.assert_allclose(out_fast, ops.convert_to_numpy(manual), atol=1e-5)

    def test_channel_fusion_routing_widths(self):
        """Cell-level channels widen cell_proj; item-level channels widen the
        outer-stack input instead — each channel must be routed to exactly
        one fusion point."""
        net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=12,
            num_matrix_blocks=1,
            raw_feature_channels=("rest_mean_score", "item_rest_correlation"),
            num_outer_sabs=1,
            outer_embed_dim=24,
        )
        _ = net(_random_responses())
        # binary input F=1 plus one cell-level channel
        assert net.cell_proj.kernel.shape[0] == 2
        # pooled width (matrix_hidden_dim, mean pooling) plus one item-level
        # channel enters the outer stack via the bridge
        assert net._outer_input_dim == 12 + 1
        assert net.bridge is not None
        assert net.bridge.kernel.shape[0] == 13

    def test_removed_mean_channels_raise(self):
        """person_mean_score / item_mean_score are not supported (the EMN
        basis computes them internally)."""
        for channel in ("person_mean_score", "item_mean_score"):
            with pytest.raises(ValueError, match="Invalid raw_feature_channels"):
                MatrixEquivariantIRTSummary(raw_feature_channels=(channel,))

    def test_item_rest_correlation_matches_manual_reference(self):
        rng = np.random.default_rng(7)
        b, p, n_items = 2, 9, 5
        x = rng.integers(0, 2, size=(b, p, n_items)).astype(np.float32)

        net = MatrixEquivariantIRTSummary(
            embed_dim=8,
            matrix_hidden_dim=8,
            num_matrix_blocks=1,
            raw_feature_channels=("item_rest_correlation",),
        )
        score = ops.convert_to_numpy(
            net._cell_score(ops.convert_to_tensor(x[..., None]))
        )
        item_total = score.sum(axis=2, keepdims=True)
        rest = (item_total - score) / max(n_items - 1, 1)
        corr = ops.convert_to_numpy(
            net._item_rest_correlation(
                ops.convert_to_tensor(score), ops.convert_to_tensor(rest)
            )
        )

        for bi in range(b):
            for ii in range(n_items):
                s = x[bi, :, ii]
                r = (x[bi].sum(axis=1) - s) / max(n_items - 1, 1)
                d1, d2 = s - s.mean(), r - r.mean()
                expected = (d1 * d2).mean() / (
                    np.sqrt((d1**2).mean()) * np.sqrt((d2**2).mean()) + 1e-6
                )
                np.testing.assert_allclose(corr[bi, 0, ii], expected, atol=1e-5)

    def test_item_rest_correlation_zero_variance_item_is_zero_not_nan(self):
        x = np.zeros((2, 6, 3), dtype=np.float32)
        net = MatrixEquivariantIRTSummary(
            embed_dim=8,
            matrix_hidden_dim=8,
            num_matrix_blocks=1,
            raw_feature_channels=("item_rest_correlation",),
        )
        out = ops.convert_to_numpy(net(x, training=False))
        assert not np.isnan(out).any()

    def test_shuffled_channel_is_a_permutation_of_the_real_one(self):
        """The shuffled-pairing control channel must preserve the multiset
        and destroy the pairing.

        Changing the values would add a distribution shift; preserving the
        order would not shuffle at all.
        """
        rng = np.random.default_rng(11)
        x = rng.integers(0, 2, size=(3, 40, 12)).astype(np.float32)
        net = MatrixEquivariantIRTSummary(
            embed_dim=8,
            matrix_hidden_dim=8,
            num_matrix_blocks=1,
            raw_feature_channels=("item_rest_correlation",),
        )
        xt = ops.convert_to_tensor(x[..., None])
        real = ops.convert_to_numpy(net._compute_item_features(xt))  # (B, I, 1)
        shuffled = ops.convert_to_numpy(
            net._shuffle_over_items(ops.convert_to_tensor(real))
        )

        for bi in range(x.shape[0]):
            np.testing.assert_allclose(
                np.sort(shuffled[bi, :, 0]), np.sort(real[bi, :, 0]), atol=1e-6
            )
        # With 12 distinct items the chance of drawing the identity for every
        # batch element is negligible, so an unchanged order means the shuffle
        # never happened.
        assert not np.allclose(shuffled, real)

    def test_shuffled_channel_still_shuffles_outside_training(self):
        """The shuffled channel must shuffle outside training too.

        Otherwise it would be trained on noise and evaluated on the real
        statistic, a train/eval mismatch.
        """
        rng = np.random.default_rng(12)
        x = rng.integers(0, 2, size=(2, 40, 10)).astype(np.float32)
        net = MatrixEquivariantIRTSummary(
            embed_dim=8,
            matrix_hidden_dim=8,
            num_matrix_blocks=1,
            raw_feature_channels=("item_rest_correlation_shuffled",),
        )
        first = ops.convert_to_numpy(net(x, training=False))
        second = ops.convert_to_numpy(net(x, training=False))
        assert not np.allclose(first, second)

    def test_shuffled_channel_is_accepted_and_widens_the_outer_input(self):
        net = MatrixEquivariantIRTSummary(
            embed_dim=8,
            matrix_hidden_dim=8,
            num_matrix_blocks=1,
            raw_feature_channels=(
                "item_rest_correlation",
                "item_rest_correlation_shuffled",
            ),
        )
        x = np.random.default_rng(13).integers(0, 2, size=(2, 20, 6)).astype(np.float32)
        out = ops.convert_to_numpy(net(x, training=False))
        assert np.isfinite(out).all()
        assert net._outer_input_dim == net._pooled_dim + 2

    def test_invalid_raw_feature_channel_raises(self):
        with pytest.raises(ValueError, match="Invalid raw_feature_channels"):
            MatrixEquivariantIRTSummary(raw_feature_channels=("bogus_feature",))

    def test_person_invariance_with_raw_features(self):
        net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=16,
            num_matrix_blocks=2,
            raw_feature_channels=self.ALL_CHANNELS,
            num_outer_sabs=1,
        )
        rng = np.random.default_rng(211)
        x = _random_responses()
        out_orig = ops.convert_to_numpy(net(x, training=False))
        perm = rng.permutation(N_PERSONS)
        out_perm = ops.convert_to_numpy(net(x[:, perm, :], training=False))
        np.testing.assert_allclose(out_orig, out_perm, atol=1e-4)

    def test_item_equivariance_with_raw_features(self):
        net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=16,
            num_matrix_blocks=2,
            raw_feature_channels=self.ALL_CHANNELS,
            num_outer_sabs=1,
        )
        rng = np.random.default_rng(212)
        x = _random_responses()
        out_orig = ops.convert_to_numpy(net(x, training=False))
        perm = rng.permutation(N_ITEMS)
        out_perm = ops.convert_to_numpy(net(x[:, :, perm], training=False))
        np.testing.assert_allclose(out_orig[:, perm, :], out_perm, atol=1e-4)

    def test_variable_num_persons_and_items(self):
        net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=12,
            num_matrix_blocks=1,
            raw_feature_channels=self.ALL_CHANNELS,
        )
        out1 = net(_random_responses(n_persons=15, n_items=8))
        assert ops.shape(out1) == (BATCH, 8, 16)
        out2 = net(_random_responses(n_persons=30, n_items=12))
        assert ops.shape(out2) == (BATCH, 12, 16)

    def test_config_round_trip_preserves_raw_feature_channels(self):
        original = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=12,
            num_matrix_blocks=1,
            raw_feature_channels=("rest_mean_score", "item_rest_correlation"),
        )
        config = original.get_config()
        restored = MatrixEquivariantIRTSummary.from_config(config)
        assert restored.raw_feature_channels == (
            "rest_mean_score",
            "item_rest_correlation",
        )
        out = restored(_random_responses())
        assert ops.shape(out) == (BATCH, N_ITEMS, 16)


class TestMatrixEquivariantIRTSummaryParameterHeads:
    """``num_parameter_heads`` adds undifferentiated parallel capacity before
    ``output_proj``; it is not a per-parameter routing mechanism (see the
    docstring). These tests only check shape/config/symmetry properties, not
    any claim about what a given head "means"."""

    def test_default_zero_heads_matches_pre_change_path(self):
        net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=12,
            num_matrix_blocks=1,
            num_outer_sabs=1,
            outer_embed_dim=16,
        )
        assert net.num_parameter_heads == 0
        assert net.parameter_head_layers is None
        x = _random_responses()
        out_default = ops.convert_to_numpy(net(x, training=False))

        # Manual reference: bridge/SABs then output_proj directly, no heads.
        x4 = ops.expand_dims(ops.convert_to_tensor(x), axis=-1)
        manual = net.cell_proj(x4)
        for block in net.matrix_blocks:
            manual = block(manual, training=False)
        manual = ops.mean(manual, axis=1)
        if net.bridge is not None:
            manual = net.bridge(manual)
        for sab in net.outer_sabs:
            manual = sab(manual, training=False)
        manual = net.output_proj(manual)

        np.testing.assert_allclose(out_default, ops.convert_to_numpy(manual), atol=1e-5)

    @pytest.mark.parametrize("num_heads", [1, 2, 3])
    def test_head_counts_build_and_call(self, num_heads):
        net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=12,
            num_matrix_blocks=1,
            num_outer_sabs=1,
            outer_embed_dim=16,
            num_parameter_heads=num_heads,
        )
        assert net.num_parameter_heads == num_heads
        assert len(net.parameter_head_layers) == num_heads
        out = net(_random_responses())
        assert ops.shape(out) == (BATCH, N_ITEMS, 16)

    def test_negative_num_parameter_heads_raises(self):
        with pytest.raises(ValueError, match="num_parameter_heads"):
            MatrixEquivariantIRTSummary(num_parameter_heads=-1)

    def test_invalid_parameter_head_dim_raises(self):
        with pytest.raises(ValueError, match="parameter_head_dim"):
            MatrixEquivariantIRTSummary(num_parameter_heads=2, parameter_head_dim=0)

    def test_parameter_head_dim_defaults_to_embed_dim(self):
        net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=12,
            num_matrix_blocks=1,
            num_parameter_heads=2,
        )
        net(_random_responses())
        for head_layer in net.parameter_head_layers:
            assert head_layer.kernel.shape[-1] == 16

    def test_parameter_head_dim_explicit_override(self):
        net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=12,
            num_matrix_blocks=1,
            num_parameter_heads=2,
            parameter_head_dim=5,
        )
        net(_random_responses())
        for head_layer in net.parameter_head_layers:
            assert head_layer.kernel.shape[-1] == 5

    def test_config_round_trip_preserves_parameter_heads(self):
        original = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=12,
            num_matrix_blocks=1,
            num_parameter_heads=2,
            parameter_head_dim=7,
        )
        config = original.get_config()
        restored = MatrixEquivariantIRTSummary.from_config(config)
        assert restored.num_parameter_heads == 2
        assert restored._parameter_head_dim == 7
        out = restored(_random_responses())
        assert ops.shape(out) == (BATCH, N_ITEMS, 16)

    def test_variable_num_persons_and_items_with_heads(self):
        net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=12,
            num_matrix_blocks=1,
            num_parameter_heads=2,
        )
        out1 = net(_random_responses(n_persons=20, n_items=10))
        out2 = net(_random_responses(n_persons=45, n_items=13))
        assert ops.shape(out1) == (BATCH, 10, 16)
        assert ops.shape(out2) == (BATCH, 13, 16)

    def test_person_invariance_with_heads(self):
        net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=16,
            num_matrix_blocks=2,
            num_outer_sabs=1,
            num_parameter_heads=2,
        )
        rng = np.random.default_rng(21)
        x = _random_responses()
        out_orig = ops.convert_to_numpy(net(x, training=False))
        perm = rng.permutation(N_PERSONS)
        out_perm = ops.convert_to_numpy(net(x[:, perm, :], training=False))
        np.testing.assert_allclose(out_orig, out_perm, atol=1e-5)

    def test_item_equivariance_with_heads(self):
        net = MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=16,
            num_matrix_blocks=2,
            num_outer_sabs=1,
            num_parameter_heads=2,
        )
        rng = np.random.default_rng(22)
        x = _random_responses()
        out_orig = ops.convert_to_numpy(net(x, training=False))
        perm = rng.permutation(N_ITEMS)
        out_perm = ops.convert_to_numpy(net(x[:, :, perm], training=False))
        np.testing.assert_allclose(out_orig[:, perm, :], out_perm, atol=1e-5)


# ---------------------------------------------------------------------------
# 4D input support (PCM one-hot encoded responses)
# ---------------------------------------------------------------------------

K = 5  # number of categories for PCM tests


def _random_onehot_responses(
    batch=BATCH, n_persons=N_PERSONS, n_items=N_ITEMS, n_categories=K
):
    """Generate random one-hot encoded responses: (B, N_PERSONS, N_ITEMS, K)."""
    rng = np.random.default_rng(42)
    cats = rng.integers(0, n_categories, size=(batch, n_persons, n_items))
    return np.eye(n_categories, dtype=np.float32)[cats]


# ---------------------------------------------------------------------------
# Hartford EMN block: normalization placement (experimental option)
# ---------------------------------------------------------------------------


CHUNK_SAFE_NORM_PLACEMENTS = (
    "pre_activation",
    "post_activation",
    "none",
    "residual_post_norm",
    "residual_no_norm",
    "per_branch",
)
ALL_NORM_PLACEMENTS = CHUNK_SAFE_NORM_PLACEMENTS + ("residual_pre_norm",)


class TestHartfordNormPlacement:
    """``norm_placement`` variants (experimental option).

    Every placement must stay permutation-equivariant (the norm acts per
    ``(person, item)`` cell on the channel axis) and the default must be
    numerically identical to the shipped behavior.
    """

    def test_default_placement_is_pre_activation(self):
        block = _HartfordEMNBlock(hidden_dim=4)
        assert block.norm_placement == "pre_activation"

    def test_residual_no_norm_is_the_none_placement_plus_a_skip(self):
        """The reference architecture: ``x + act(sum)``, no normalization.

        Pinned against ``none`` so the variant cannot silently acquire a norm or
        move the skip to the wrong side of the activation.
        """
        x = np.random.default_rng(21).normal(size=(2, 6, 5, 8)).astype(np.float32)
        plain = _HartfordEMNBlock(
            hidden_dim=8, activation="mish", dropout=0.0, norm_placement="none"
        )
        residual = _HartfordEMNBlock(
            hidden_dim=8,
            activation="mish",
            dropout=0.0,
            norm_placement="residual_no_norm",
        )
        plain.build(x.shape)
        residual.build(x.shape)
        residual.set_weights(plain.get_weights())
        np.testing.assert_allclose(
            ops.convert_to_numpy(residual(x, training=False)),
            ops.convert_to_numpy(plain(x, training=False)) + x,
            atol=1e-5,
        )

    def test_residual_no_norm_owns_no_normalization_layers(self):
        block = _HartfordEMNBlock(hidden_dim=8, norm_placement="residual_no_norm")
        block.build((None, 6, 5, 8))
        assert block.norm is None
        assert block.branch_norms is None

    def test_explicit_default_matches_implicit_default_bitwise(self):
        """The option must not perturb the default configuration."""
        x = np.random.default_rng(3).normal(size=(2, 6, 5, 4)).astype(np.float32)
        keras.utils.set_random_seed(11)
        implicit = _HartfordEMNBlock(hidden_dim=4, activation="mish", dropout=0.0)
        implicit_out = ops.convert_to_numpy(implicit(x, training=False))
        keras.utils.set_random_seed(11)
        explicit = _HartfordEMNBlock(
            hidden_dim=4,
            activation="mish",
            dropout=0.0,
            norm_placement="pre_activation",
        )
        explicit_out = ops.convert_to_numpy(explicit(x, training=False))
        np.testing.assert_array_equal(implicit_out, explicit_out)

    @pytest.mark.parametrize("placement", ALL_NORM_PLACEMENTS)
    def test_placement_preserves_permutation_equivariance(self, placement):
        rng = np.random.default_rng(5)
        n_persons, n_items = 9, 6
        x = rng.normal(size=(2, n_persons, n_items, 4)).astype(np.float32)
        block = _HartfordEMNBlock(
            hidden_dim=4, activation="mish", dropout=0.0, norm_placement=placement
        )
        out = ops.convert_to_numpy(block(x, training=False))
        person_perm = rng.permutation(n_persons)
        item_perm = rng.permutation(n_items)
        permuted = ops.convert_to_numpy(
            block(x[:, person_perm][:, :, item_perm], training=False)
        )
        np.testing.assert_allclose(
            out[:, person_perm][:, :, item_perm], permuted, atol=1e-5
        )

    @pytest.mark.parametrize("placement", CHUNK_SAFE_NORM_PLACEMENTS)
    def test_chunked_forward_matches_dense_for_every_placement(self, placement):
        """Person chunking must stay exact under each supported placement.

        The epilogue is shared between ``call`` and ``_apply_chunk``, so a
        placement that broke that sharing would show up here.
        """
        n_persons = 11
        x = (
            np.random.default_rng(9)
            .normal(size=(2, n_persons, 4, 5))
            .astype(np.float32)
        )
        keras.utils.set_random_seed(0)  # see the dense-match test above
        block = _HartfordEMNBlock(
            hidden_dim=5, activation="mish", dropout=0.0, norm_placement=placement
        )
        dense_out = ops.convert_to_numpy(block(x, training=False))
        for chunk_size in (1, 3, n_persons - 1):
            block.person_chunk_size = chunk_size
            np.testing.assert_allclose(
                ops.convert_to_numpy(block(x, training=False)),
                dense_out,
                atol=1e-6,
                err_msg=f"{placement} mismatch at chunk_size={chunk_size}",
            )
        block.person_chunk_size = None

    def test_residual_pre_norm_rejects_person_chunking(self):
        with pytest.raises(ValueError, match="residual_pre_norm"):
            _HartfordEMNBlock(
                hidden_dim=4, norm_placement="residual_pre_norm", person_chunk_size=2
            )

    def test_invalid_placement_raises(self):
        with pytest.raises(ValueError, match="norm_placement must be one of"):
            _HartfordEMNBlock(hidden_dim=4, norm_placement="sandwich")

    def test_residual_placement_requires_matching_width(self):
        block = _HartfordEMNBlock(hidden_dim=4, norm_placement="residual_post_norm")
        with pytest.raises(ValueError, match="input_dim == hidden_dim"):
            block.build((None, 5, 3, 7))

    @pytest.mark.parametrize("placement", ALL_NORM_PLACEMENTS)
    def test_get_config_round_trip(self, placement):
        block = _HartfordEMNBlock(hidden_dim=4, norm_placement=placement)
        assert block.get_config()["norm_placement"] == placement
        assert _HartfordEMNBlock.from_config(block.get_config()).norm_placement == (
            placement
        )

    @pytest.mark.parametrize("placement", ALL_NORM_PLACEMENTS)
    def test_summary_forwards_placement_to_blocks(self, placement):
        net = MatrixEquivariantIRTSummary(
            embed_dim=8,
            matrix_hidden_dim=8,
            num_matrix_blocks=2,
            matrix_norm_placement=placement,
        )
        assert [b.norm_placement for b in net.matrix_blocks] == [placement] * 2
        out = net(_random_responses())
        assert out.shape == (BATCH, N_ITEMS, 8)

    def test_summary_rejects_residual_pre_norm_with_person_chunking(self):
        with pytest.raises(ValueError, match="residual_pre_norm"):
            MatrixEquivariantIRTSummary(
                embed_dim=8,
                matrix_hidden_dim=8,
                num_matrix_blocks=2,
                matrix_norm_placement="residual_pre_norm",
                person_chunk_size=4,
            )


# ---------------------------------------------------------------------------
# Hartford EMN block: dropout granularity (experimental option)
# ---------------------------------------------------------------------------


class TestHartfordDropoutKind:
    """``dropout_kind`` variants (experimental option).

    Hartford et al. (2018) §3.2 prescribe channel dropout ("we therefore
    randomly drop out whole channels during training (as opposed to dropping
    out individual elements)"); the shipped default is element-wise.
    """

    def test_default_kind_is_element(self):
        assert _HartfordEMNBlock(hidden_dim=4, dropout=0.1).dropout_kind == "element"

    def test_zero_rate_creates_no_dropout_layer_for_any_kind(self):
        for kind in ("element", "channel", "row_col"):
            block = _HartfordEMNBlock(hidden_dim=4, dropout=0.0, dropout_kind=kind)
            assert block.dropout_layer is None
            assert block.item_dropout_layer is None

    def test_only_row_col_builds_a_second_dropout_layer(self):
        for kind in ("element", "channel"):
            block = _HartfordEMNBlock(hidden_dim=4, dropout=0.1, dropout_kind=kind)
            assert block.item_dropout_layer is None
        block = _HartfordEMNBlock(hidden_dim=4, dropout=0.1, dropout_kind="row_col")
        assert block.item_dropout_layer is not None

    def test_channel_mask_is_constant_across_persons_and_items(self):
        """The defining property: a whole channel is dropped, everywhere."""
        block = _HartfordEMNBlock(
            hidden_dim=16,
            activation="linear",
            dropout=0.5,
            dropout_kind="channel",
            norm_placement="none",
        )
        x = np.random.default_rng(2).normal(size=(3, 7, 5, 16)).astype(np.float32)
        keras.utils.set_random_seed(4)
        out = ops.convert_to_numpy(block(x, training=True))
        zeroed = out == 0.0
        # A dropped channel is zero at every (person, item) cell; a surviving
        # one is zero nowhere (up to the measure-zero chance of an exact 0).
        per_channel = zeroed.all(axis=(1, 2)) | (~zeroed).all(axis=(1, 2))
        assert per_channel.all()
        assert zeroed.any(), "expected at least one dropped channel at rate 0.5"

    def test_element_mask_is_not_constant_across_cells(self):
        """Contrast case: the shipped default drops individual entries."""
        block = _HartfordEMNBlock(
            hidden_dim=16,
            activation="linear",
            dropout=0.5,
            dropout_kind="element",
            norm_placement="none",
        )
        x = np.random.default_rng(2).normal(size=(3, 7, 5, 16)).astype(np.float32)
        keras.utils.set_random_seed(4)
        out = ops.convert_to_numpy(block(x, training=True))
        zeroed = out == 0.0
        assert zeroed.any()
        # Some channel must be dropped at some cells but not all of them.
        assert not (zeroed.all(axis=(1, 2)) | (~zeroed).all(axis=(1, 2))).all()

    def test_channel_dropout_is_exactly_equivariant_during_training(self):
        """A channel mask is constant across (P, I), so it commutes with
        row/column permutations *per draw* — unlike an element mask, which
        only preserves equivariance in distribution.
        """
        rng = np.random.default_rng(6)
        n_persons, n_items = 8, 6
        x = rng.normal(size=(2, n_persons, n_items, 8)).astype(np.float32)
        person_perm = rng.permutation(n_persons)
        item_perm = rng.permutation(n_items)
        block = _HartfordEMNBlock(
            hidden_dim=8, activation="mish", dropout=0.5, dropout_kind="channel"
        )
        block(x, training=False)  # build, so the seed generator exists

        # Keras' Dropout owns a SeedGenerator whose state advances per call
        # and is *not* reset by set_random_seed, so the mask must be pinned by
        # restoring that state — the same snapshot/restore the chunked
        # backward relies on. Without this the two calls draw different masks
        # and the comparison tests the RNG, not equivariance.
        seed_state = block.dropout_layer.seed_generator.state
        pinned = ops.convert_to_numpy(seed_state)
        out = ops.convert_to_numpy(block(x, training=True))
        seed_state.assign(pinned)
        permuted = ops.convert_to_numpy(
            block(x[:, person_perm][:, :, item_perm], training=True)
        )
        np.testing.assert_allclose(
            out[:, person_perm][:, :, item_perm], permuted, atol=1e-5
        )

    def test_row_col_zeros_are_exactly_whole_persons_and_whole_items(self):
        """The defining property of ``matrix_dropout`` in the reference TF
        implementation: ``noise_shape=[N,1,1]`` then ``[1,M,1]`` drops whole
        rows and whole columns, tied across channels.
        """
        block = _HartfordEMNBlock(
            hidden_dim=8,
            activation="linear",
            dropout=0.5,
            dropout_kind="row_col",
            norm_placement="none",
        )
        x = np.random.default_rng(12).normal(size=(3, 9, 7, 8)).astype(np.float32)
        keras.utils.set_random_seed(5)
        out = ops.convert_to_numpy(block(x, training=True))
        zeroed = out == 0.0
        dropped_person = zeroed.all(axis=(2, 3))  # (B, P)
        dropped_item = zeroed.all(axis=(1, 3))  # (B, I)
        assert dropped_person.any(), "expected a dropped person at rate 0.5"
        assert dropped_item.any(), "expected a dropped item at rate 0.5"
        # Every zero is explained by its person or its item being dropped --
        # nothing is zeroed cell-wise or channel-wise.
        expected = np.broadcast_to(
            dropped_person[:, :, None, None] | dropped_item[:, None, :, None],
            zeroed.shape,
        )
        np.testing.assert_array_equal(zeroed, expected)

    def test_row_col_dropout_is_not_exactly_equivariant_per_draw(self):
        """Contrast with channel dropout: a row/column mask is *indexed by*
        person and item, so it does not commute with a permutation for a fixed
        draw. Equivariance holds only in distribution. Documented so the
        weaker guarantee is not mistaken for a bug.
        """
        rng = np.random.default_rng(13)
        n_persons, n_items = 8, 6
        x = rng.normal(size=(2, n_persons, n_items, 8)).astype(np.float32)
        person_perm = rng.permutation(n_persons)
        block = _HartfordEMNBlock(
            hidden_dim=8, activation="mish", dropout=0.5, dropout_kind="row_col"
        )
        block(x, training=False)
        pinned = ops.convert_to_numpy(block.dropout_layer.seed_generator.state)
        item_pinned = ops.convert_to_numpy(
            block.item_dropout_layer.seed_generator.state
        )
        out = ops.convert_to_numpy(block(x, training=True))
        block.dropout_layer.seed_generator.state.assign(pinned)
        block.item_dropout_layer.seed_generator.state.assign(item_pinned)
        permuted = ops.convert_to_numpy(block(x[:, person_perm], training=True))
        assert not np.allclose(out[:, person_perm], permuted, atol=1e-5)

    @pytest.mark.parametrize("kind", ["channel", "row_col"])
    def test_axis_tied_kinds_reject_person_chunking(self, kind):
        with pytest.raises(ValueError, match=f"dropout_kind={kind!r}"):
            _HartfordEMNBlock(
                hidden_dim=4, dropout=0.1, dropout_kind=kind, person_chunk_size=2
            )

    def test_invalid_kind_raises(self):
        with pytest.raises(ValueError, match="dropout_kind must be one of"):
            _HartfordEMNBlock(hidden_dim=4, dropout_kind="spatial")

    @pytest.mark.parametrize("kind", ["element", "channel", "row_col"])
    def test_get_config_round_trip(self, kind):
        block = _HartfordEMNBlock(hidden_dim=4, dropout=0.1, dropout_kind=kind)
        assert block.get_config()["dropout_kind"] == kind
        restored = _HartfordEMNBlock.from_config(block.get_config())
        assert restored.dropout_kind == kind
        assert (restored.item_dropout_layer is not None) == (kind == "row_col")


class TestMatrixSummaryDropoutOverride:
    """``matrix_block_dropout`` / ``matrix_block_dropout_kind`` plumbing."""

    def test_matrix_dropout_defaults_to_shared_rate(self):
        net = MatrixEquivariantIRTSummary(
            embed_dim=8, matrix_hidden_dim=8, num_matrix_blocks=2, dropout=0.2
        )
        assert [b.dropout for b in net.matrix_blocks] == [0.2, 0.2]

    def test_matrix_dropout_overrides_only_the_blocks(self):
        """The override must not disturb the outer stack's dropout, so it
        affects only the matrix blocks.
        """
        net = MatrixEquivariantIRTSummary(
            embed_dim=8,
            matrix_hidden_dim=8,
            num_matrix_blocks=2,
            dropout=0.2,
            matrix_block_dropout=0.0,
        )
        assert [b.dropout for b in net.matrix_blocks] == [0.0, 0.0]
        assert net.matrix_blocks[0].dropout_layer is None
        assert net._config["dropout"] == 0.2

    def test_matrix_dropout_kind_forwarded_and_forward_pass_runs(self):
        net = MatrixEquivariantIRTSummary(
            embed_dim=8,
            matrix_hidden_dim=8,
            num_matrix_blocks=2,
            dropout=0.1,
            matrix_block_dropout_kind="channel",
        )
        assert [b.dropout_kind for b in net.matrix_blocks] == ["channel"] * 2
        assert net(_random_responses(), training=True).shape == (BATCH, N_ITEMS, 8)

    def test_summary_rejects_channel_dropout_with_person_chunking(self):
        with pytest.raises(ValueError, match="matrix_block_dropout_kind='channel'"):
            MatrixEquivariantIRTSummary(
                embed_dim=8,
                matrix_hidden_dim=8,
                num_matrix_blocks=2,
                dropout=0.1,
                matrix_block_dropout_kind="channel",
                person_chunk_size=4,
            )


# ---------------------------------------------------------------------------
# Definition 2.1 (full form): the layer must NOT be equivariant to permutations
# other than row/column permutations
# ---------------------------------------------------------------------------


def _shuffle_persons_within_each_item(x, rng):
    """Independently permute the person axis *within each item column*.

    This is a permutation of the elements of ``X`` that is **not** a row
    permutation: a legal row permutation applies the *same* person ordering to
    every column, whereas this applies a different one per column. It leaves
    every column mean (and the global mean) untouched while changing which
    persons share a row.
    """
    out = np.empty_like(x)
    n_persons, n_items = x.shape[1], x.shape[2]
    for b in range(x.shape[0]):
        for i in range(n_items):
            out[b, :, i] = x[b, rng.permutation(n_persons), i]
    return out


class TestIllegalPermutationSensitivity:
    """The "should *not* hold" side of the equivariance guarantee.

    Hartford et al. (2018) Def. 2.1's footnote notes that the simplified
    definition alone would admit "a trivial layer with a constant weight
    matrix"; the full definition (their Sec. 5) therefore additionally
    requires that the layer **not** be equivariant with respect to any other
    permutation of ``X``'s elements. The rest of this module tests the
    "must hold" side (person invariance, item equivariance); these tests pin
    the complementary side, so a degenerate layer that ignored the matrix
    structure entirely would fail here rather than pass everything.
    """

    def test_block_is_not_equivariant_to_within_column_shuffles(self):
        """The Hartford basis must actually use the row structure.

        A within-column shuffle leaves ``col_mean`` and ``global_mean``
        unchanged but changes ``row_mean``, so the block's output must differ
        from simply applying the same shuffle to the original output.
        """
        rng = np.random.default_rng(17)
        x = rng.normal(size=(2, 9, 5, 3)).astype(np.float32)
        block = _HartfordEMNBlock(hidden_dim=6, activation="mish", dropout=0.0)
        out = ops.convert_to_numpy(block(x, training=False))

        shuffled_x = _shuffle_persons_within_each_item(x, np.random.default_rng(18))
        shuffled_out = ops.convert_to_numpy(block(shuffled_x, training=False))
        # If the block were (wrongly) equivariant to this permutation, applying
        # the shuffle to the *input* would equal applying it to the *output*.
        out_then_shuffled = _shuffle_persons_within_each_item(
            out, np.random.default_rng(18)
        )
        assert not np.allclose(shuffled_out, out_then_shuffled, atol=1e-4)

    def test_summary_with_blocks_is_sensitive_to_within_column_shuffles(self):
        """With >=1 matrix block the summary must see more than column means."""
        rng = np.random.default_rng(19)
        x = rng.binomial(1, 0.5, size=(3, 12, 6)).astype(np.float32)
        net = MatrixEquivariantIRTSummary(
            embed_dim=8, matrix_hidden_dim=8, num_matrix_blocks=2, dropout=0.0
        )
        base = ops.convert_to_numpy(net(x, training=False))
        shuffled = ops.convert_to_numpy(
            net(_shuffle_persons_within_each_item(x, rng), training=False)
        )
        assert not np.allclose(base, shuffled, atol=1e-4)

    def test_summary_without_blocks_collapses_to_column_marginals(self):
        """``num_matrix_blocks=0`` is *intentionally* insensitive here.

        With no matrix blocks the summary is a function of the column means
        alone (a column-marginal DeepSet, sufficient for Rasch), so it is
        invariant to within-column shuffles by construction. Pinning this
        documents the collapse as a deliberate property of the L=0 fast path
        rather than letting it look like the bug the test above rules out.
        """
        rng = np.random.default_rng(20)
        x = rng.binomial(1, 0.5, size=(3, 12, 6)).astype(np.float32)
        net = MatrixEquivariantIRTSummary(
            embed_dim=8, matrix_hidden_dim=8, num_matrix_blocks=0, dropout=0.0
        )
        base = ops.convert_to_numpy(net(x, training=False))
        shuffled = ops.convert_to_numpy(
            net(_shuffle_persons_within_each_item(x, rng), training=False)
        )
        np.testing.assert_allclose(base, shuffled, atol=1e-5)


@pytest.mark.parametrize("num_items", [1, 2, 5, 11])
def test_matrix_rest_score_matches_other_item_mean(num_items):
    """Rest scores retain the leave-one-item-out mean for varying sizes."""
    score = np.random.default_rng(71).random((2, 3, num_items)).astype("float32")
    net = MatrixEquivariantIRTSummary()
    actual = ops.convert_to_numpy(net._rest_score(ops.convert_to_tensor(score)))
    expected = np.zeros_like(score)
    if num_items > 1:
        for item in range(num_items):
            expected[..., item] = np.delete(score, item, axis=2).mean(axis=2)
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_matrix_pma_build_initializes_nested_layers_and_restores_weights(tmp_path):
    """Building the summary must create PMA state before checkpoint loading."""
    net = MatrixEquivariantIRTSummary(
        embed_dim=8,
        matrix_hidden_dim=8,
        num_matrix_blocks=0,
        person_pooling="pma",
        num_pma_seeds=2,
        num_outer_sabs=0,
        outer_embed_dim=8,
        dropout=0.0,
    )
    net.build((2, 12, 5))
    # A seed vector alone is insufficient: nested attention/MLP weights
    # must already exist, including on BayesFlow versions without PMA.build.
    assert len(net.person_pma.weights) > 1
    model = keras.Sequential([net])
    responses = _random_responses(batch=2, n_persons=12, n_items=5)
    expected = ops.convert_to_numpy(model(responses, training=False))
    path = tmp_path / "pma.keras"
    model.save(path)
    restored = keras.saving.load_model(
        path,
        compile=False,
        custom_objects={"MatrixEquivariantIRTSummary": MatrixEquivariantIRTSummary},
    )
    assert len(model.weights) == len(restored.weights)
    for original, recovered in zip(model.weights, restored.weights, strict=True):
        np.testing.assert_array_equal(
            ops.convert_to_numpy(original), ops.convert_to_numpy(recovered)
        )
    np.testing.assert_allclose(
        ops.convert_to_numpy(restored(responses, training=False)), expected, atol=1e-6
    )
