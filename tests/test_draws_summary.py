"""Tests for :class:`bayesflow_irt.draws_summary.DrawsSummary`."""

import keras
import numpy as np
import pytest

from bayesflow_irt.draws_summary import DrawsSummary


def _built(pooling="mean"):
    layer = DrawsSummary(8, 6, pooling=pooling)
    layer.build((None, None, 2), (None, None, 5))
    return layer


@pytest.mark.parametrize("pooling", DrawsSummary.POOLINGS)
def test_fresh_layer_callable_without_manual_build(pooling):
    """Keras auto-builds from the call arguments (`build(x_shape, h_shape)`)."""
    layer = DrawsSummary(hidden_dim=8, summary_dim=8, pooling=pooling)
    out = layer(keras.ops.zeros((2, 3, 2)), keras.ops.zeros((2, 3, 8)))
    assert tuple(out.shape) == (2, 3, 9)
    assert layer.built


def test_mean_prefix_matches_naive_loop():
    """The cumulative-sum path equals the mean over each prefix, with the
    learned constant at position 0 and the log count appended."""
    layer = _built()
    rng = np.random.default_rng(1)
    x = keras.ops.convert_to_tensor(rng.normal(size=(2, 7, 2)).astype("float32"))
    h = keras.ops.convert_to_tensor(rng.normal(size=(2, 7, 5)).astype("float32"))
    fast = keras.ops.convert_to_numpy(layer(x, h))

    phi = keras.ops.convert_to_numpy(layer.phi_values(x, h))
    naive = np.zeros_like(fast)
    for i in range(7):
        if i == 0:
            c = keras.ops.convert_to_numpy(layer.ctx0)
        else:
            mean = phi[:, :i].mean(axis=1)
            c = keras.ops.convert_to_numpy(layer.rho_2(layer.rho_1(mean)))
        naive[:, i, : layer.summary_dim] = c
        naive[:, i, -1] = np.log(i + 1.0)
    np.testing.assert_allclose(fast, naive, atol=1e-5)


@pytest.mark.parametrize("pooling", DrawsSummary.POOLINGS)
def test_teacher_forcing_matches_per_position_path(pooling):
    """`call` (all positions at once) and `from_prefix` (one position, used
    when sampling) give the same summary."""
    layer = _built(pooling)
    rng = np.random.default_rng(3)
    x = keras.ops.convert_to_tensor(rng.normal(size=(2, 6, 2)).astype("float32"))
    h = keras.ops.convert_to_tensor(rng.normal(size=(2, 6, 5)).astype("float32"))
    full = keras.ops.convert_to_numpy(layer(x, h))
    phi = layer.phi_values(x, h)
    for i in range(6):
        prefix = None if i == 0 else phi[:, :i]
        one = keras.ops.convert_to_numpy(layer.from_prefix(prefix, h[:, i], i))
        np.testing.assert_allclose(one, full[:, i], atol=1e-5)


def test_draws_summary_is_public():
    from bayesflow_irt import DrawsSummary, __all__

    assert "DrawsSummary" in __all__
    assert DrawsSummary.__name__ == "DrawsSummary"
