"""Tests for the optional numba-accelerated response-sampling kernels.

Skipped entirely if numba isn't installed (``pip install bayesflow-irt[numba]``).
"""

from __future__ import annotations

import numpy as np
import pytest

numba = pytest.importorskip("numba", reason="numba not installed")

from bayesflow_irt._numba_kernels import (  # noqa: E402
    binary_response_kernel,
    categorical_response_kernel,
)


class TestBinaryResponseKernel:
    def test_shape_and_dtype(self):
        rng = np.random.default_rng(0)
        batch_size, n_persons, n_items = 8, 50, 10
        logits = rng.normal(size=(batch_size, n_persons, n_items)).astype(np.float32)
        out = np.empty((batch_size, n_persons, n_items), dtype=np.uint8)
        binary_response_kernel(logits, out, 0)
        assert out.shape == (batch_size, n_persons, n_items)
        assert set(np.unique(out).tolist()) <= {0, 1}

    def test_float32_output_dtype(self):
        rng = np.random.default_rng(0)
        logits = rng.normal(size=(4, 10, 5)).astype(np.float32)
        out = np.empty((4, 10, 5), dtype=np.float32)
        binary_response_kernel(logits, out, 0)
        assert set(np.unique(out).tolist()) <= {0.0, 1.0}

    def test_reproducible_same_seed(self):
        rng = np.random.default_rng(0)
        logits = rng.normal(size=(8, 50, 10)).astype(np.float32)
        out_a = np.empty((8, 50, 10), dtype=np.uint8)
        out_b = np.empty((8, 50, 10), dtype=np.uint8)
        binary_response_kernel(logits, out_a, 42)
        binary_response_kernel(logits, out_b, 42)
        assert np.array_equal(out_a, out_b)

    def test_thread_count_invariant(self):
        """Regression test for a seeding pitfall:
        seeding numba's RNG once *before* the parallel loop makes output
        depend on thread count. Reseeding per batch-row (as the kernel does)
        must give identical output regardless of thread count."""
        rng = np.random.default_rng(0)
        logits = rng.normal(size=(8, 50, 10)).astype(np.float32)

        original_num_threads = numba.get_num_threads()
        try:
            numba.set_num_threads(1)
            out_1t = np.empty((8, 50, 10), dtype=np.uint8)
            binary_response_kernel(logits, out_1t, 7)

            numba.set_num_threads(min(4, numba.config.NUMBA_DEFAULT_NUM_THREADS))
            out_4t = np.empty((8, 50, 10), dtype=np.uint8)
            binary_response_kernel(logits, out_4t, 7)
        finally:
            numba.set_num_threads(original_num_threads)

        assert np.array_equal(out_1t, out_4t)

    def test_statistical_equivalence_to_sigmoid(self):
        """Not bit-identical to the numpy path (different RNG algorithm),
        but must draw from the same Bernoulli(sigmoid(logit)) distribution."""
        n_samples = 100_000
        fixed_logit = np.float32(0.7)
        logits = np.full((n_samples, 1, 1), fixed_logit, dtype=np.float32)
        out = np.empty((n_samples, 1, 1), dtype=np.uint8)
        binary_response_kernel(logits, out, 55)

        empirical_rate = out.mean()
        true_rate = 1.0 / (1.0 + np.exp(-fixed_logit))
        se = np.sqrt(true_rate * (1 - true_rate) / n_samples)
        assert abs(empirical_rate - true_rate) < 5 * se


class TestCategoricalResponseKernel:
    def test_shape_and_range(self):
        rng = np.random.default_rng(0)
        batch_size, n_persons, n_items, n_categories = 8, 50, 10, 5
        logits = rng.normal(size=(batch_size, n_persons, n_items, n_categories)).astype(
            np.float32
        )
        out = np.empty((batch_size, n_persons, n_items), dtype=np.uint8)
        categorical_response_kernel(logits, out, 0)
        assert out.shape == (batch_size, n_persons, n_items)
        assert out.min() >= 0
        assert out.max() < n_categories

    def test_reproducible_same_seed(self):
        rng = np.random.default_rng(0)
        logits = rng.normal(size=(8, 50, 10, 5)).astype(np.float32)
        out_a = np.empty((8, 50, 10), dtype=np.uint8)
        out_b = np.empty((8, 50, 10), dtype=np.uint8)
        categorical_response_kernel(logits, out_a, 42)
        categorical_response_kernel(logits, out_b, 42)
        assert np.array_equal(out_a, out_b)

    def test_thread_count_invariant(self):
        """Same regression test as the binary kernel: reseeding per batch-row
        must make output independent of thread count."""
        rng = np.random.default_rng(0)
        logits = rng.normal(size=(8, 50, 10, 5)).astype(np.float32)

        original_num_threads = numba.get_num_threads()
        try:
            numba.set_num_threads(1)
            out_1t = np.empty((8, 50, 10), dtype=np.uint8)
            categorical_response_kernel(logits, out_1t, 7)

            numba.set_num_threads(min(4, numba.config.NUMBA_DEFAULT_NUM_THREADS))
            out_4t = np.empty((8, 50, 10), dtype=np.uint8)
            categorical_response_kernel(logits, out_4t, 7)
        finally:
            numba.set_num_threads(original_num_threads)

        assert np.array_equal(out_1t, out_4t)

    def test_statistical_equivalence_to_softmax(self):
        """Not bit-identical to the numpy Gumbel-max path (different RNG
        algorithm), but must draw from Categorical(softmax(logits))."""
        n_samples = 100_000
        n_categories = 5
        rng = np.random.default_rng(0)
        fixed_logits = rng.normal(size=(1, 1, 1, n_categories)).astype(np.float32)
        logits = np.tile(fixed_logits, (n_samples, 1, 1, 1))
        out = np.empty((n_samples, 1, 1), dtype=np.uint8)
        categorical_response_kernel(logits, out, 123)

        empirical = np.bincount(out.ravel(), minlength=n_categories) / n_samples
        shifted = fixed_logits[0, 0, 0] - fixed_logits[0, 0, 0].max()
        softmax = np.exp(shifted) / np.exp(shifted).sum()
        se = np.sqrt(softmax * (1 - softmax) / n_samples)
        assert np.all(np.abs(empirical - softmax) < 5 * se)
