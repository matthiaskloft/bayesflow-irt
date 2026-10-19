"""Tests for the TCC bias joint HPO metric."""

import numpy as np
import pytest

pytest.importorskip("bayesflow_hpo", reason="bayesflow_hpo not installed")

# Access names through the module: tests/test_hpo.py reloads bayesflow_irt.hpo,
# which would leave directly imported callables stale.
from bayesflow_irt import hpo  # noqa: E402


class TestTccBiasMetric:
    """``make_tcc_bias_metric`` on hand-built joint inputs."""

    @staticmethod
    def _inputs(draws, truth, keys):
        from types import SimpleNamespace

        return SimpleNamespace(draws=draws, true_values=truth, param_keys=keys)

    @staticmethod
    def _calibrated(rng, n_sims, n_items, shift=0.0):
        # posterior N(m, 0.3^2) per coordinate with truth ~ that posterior:
        # calibrated by construction; `shift` moves every draw of b by the
        # same amount, the shared bias the metric is for
        m = rng.normal(size=(n_sims, 2 * n_items))
        truth = m + 0.3 * rng.normal(size=m.shape)
        draws = m[:, None] + 0.3 * rng.normal(size=(n_sims, 400, 2 * n_items))
        draws[..., n_items:] += shift
        return draws, truth

    def test_calibrated_is_near_its_floor_and_shift_is_detected(self):
        make_tcc_bias_metric = hpo.make_tcc_bias_metric

        metric = make_tcc_bias_metric()
        rng = np.random.default_rng(0)
        n_sims = 400
        ok = metric(self._inputs(*self._calibrated(rng, n_sims, 20), ("a", "b")))
        # floor ~ sqrt(2/pi)/sqrt(n) per theta; max over 3 thetas
        assert ok["tcc_z_bias"] < 4 / np.sqrt(n_sims)
        biased = metric(
            self._inputs(*self._calibrated(rng, n_sims, 20, shift=-0.1), ("a", "b"))
        )
        assert biased["tcc_z_bias"] > 0.5

    def test_skewed_tcc_posterior_has_no_mean_sd_offset(self):
        # With mean and sd from the same draws, a skewed TCC
        # posterior (1PL TCC = logistic(-b), saturated for b ~ N(10, 3))
        # gives mean z != 0 even for an exact posterior; the split estimator
        # must not.
        rng = np.random.default_rng(23)
        truth = rng.normal(10, 3, size=(20000, 1))
        draws = rng.normal(10, 3, size=(20000, 200, 1))
        metric = hpo.make_tcc_bias_metric(thetas=(0.0,), model="1pl")
        got = metric(self._inputs(draws, truth, ("b",)))["tcc_z_bias"]

        tcc = 1 / (1 + np.exp(draws))
        tcc_true = 1 / (1 + np.exp(truth))
        z_same = (tcc.mean(1) - tcc_true) / tcc.std(1, ddof=1)
        z_split = (tcc[:, 0::2].mean(1) - tcc_true) / tcc[:, 1::2].std(1, ddof=1)
        se_same = z_same.std() / np.sqrt(z_same.size)
        se_split = z_split.std() / np.sqrt(z_split.size)
        # the same-draws estimator is clearly off, the split one is not
        assert abs(z_same.mean()) > 4 * se_same
        assert got == pytest.approx(abs(z_split.mean()))
        assert got < 3 * se_split

    def test_too_few_draws_raises(self):
        metric = hpo.make_tcc_bias_metric(model="1pl")
        rng = np.random.default_rng(5)
        draws, truth = rng.normal(size=(5, 3, 4)), rng.normal(size=(5, 4))
        with pytest.raises(ValueError, match="at least 4"):
            metric(self._inputs(draws, truth, ("b",)))

    def test_one_pl_and_unknown_keys(self):
        make_tcc_bias_metric = hpo.make_tcc_bias_metric

        metric = make_tcc_bias_metric(key="k", model="1pl")
        rng = np.random.default_rng(1)
        draws, truth = self._calibrated(rng, 50, 4)
        out = metric(self._inputs(draws[..., 4:], truth[..., 4:], ("b",)))
        assert np.isfinite(out["k"])
        with pytest.raises(ValueError, match="param_keys"):
            metric(self._inputs(draws, truth, ("a", "delta")))

    def test_model_must_match_param_keys(self):
        rng = np.random.default_rng(3)
        draws, truth = self._calibrated(rng, 20, 4)
        two_pl = hpo.make_tcc_bias_metric(model="2pl")
        with pytest.raises(ValueError, match=r"'a' and 'b'.*\('b',\)"):
            two_pl(self._inputs(draws[..., 4:], truth[..., 4:], ("b",)))
        one_pl = hpo.make_tcc_bias_metric(model="1pl")
        with pytest.raises(ValueError, match=r"\('b',\).*\('a', 'b'\)"):
            one_pl(self._inputs(draws, truth, ("a", "b")))
        with pytest.raises(ValueError, match="model must be"):
            hpo.make_tcc_bias_metric(model="gpcm")

    def test_column_order_follows_param_keys(self):
        make_tcc_bias_metric = hpo.make_tcc_bias_metric

        metric = make_tcc_bias_metric()
        rng = np.random.default_rng(2)
        draws, truth = self._calibrated(rng, 50, 6, shift=-0.1)
        ab = metric(self._inputs(draws, truth, ("a", "b")))
        swap = np.r_[6:12, 0:6]
        ba = metric(self._inputs(draws[..., swap], truth[..., swap], ("b", "a")))
        assert ab == pytest.approx(ba)

    def test_direction_is_registered(self):
        from bayesflow_hpo.objectives import METRIC_DIRECTIONS

        assert METRIC_DIRECTIONS["tcc_z_bias"].higher_is_better is False

    def test_custom_key_direction_is_registered(self):
        from bayesflow_hpo.objectives import METRIC_DIRECTIONS

        hpo.make_tcc_bias_metric(key="tcc_custom_key")
        direction = METRIC_DIRECTIONS["tcc_custom_key"]
        assert direction.higher_is_better is False
        assert direction.worst_raw == float("inf")


def _logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


class TestTccBiasInTccUnits:
    """``make_tcc_bias_metric(scale="tcc")``: bias per item, no z denominator."""

    _inputs = staticmethod(TestTccBiasMetric._inputs)

    def test_1pl_known_answer(self):
        # Every draw has b = 0.5 and the truth b = 0 for 4 items; at theta = 0
        # each item's P is logistic(-0.5) against 0.5, so the per-item TCC
        # error is the same for every item and simulation.
        draws = np.full((3, 6, 4), 0.5)
        truth = np.zeros((3, 4))
        metric = hpo.make_tcc_bias_metric(thetas=(0.0,), model="1pl", scale="tcc")
        got = metric(self._inputs(draws, truth, ("b",)))
        assert got == {"tcc_bias": pytest.approx(0.5 - _logistic(-0.5))}
        # opposite sign, same magnitude (absolute value)
        flipped = metric(self._inputs(truth[:, None] + 0 * draws, draws[:, 0], ("b",)))
        assert flipped["tcc_bias"] == pytest.approx(0.5 - _logistic(-0.5))

    def test_2pl_known_answer_uses_log_a(self):
        # keys ("b", "a"): blocks are b first, then log a; a = exp(log 2) = 2.
        n_items = 3
        draws = np.concatenate(
            [np.full((2, 8, n_items), 1.0), np.full((2, 8, n_items), np.log(2.0))],
            axis=-1,
        )
        truth = np.concatenate(
            [np.zeros((2, n_items)), np.full((2, n_items), np.log(2.0))], axis=-1
        )
        metric = hpo.make_tcc_bias_metric(thetas=(0.0,), scale="tcc")
        got = metric(self._inputs(draws, truth, ("b", "a")))["tcc_bias"]
        assert got == pytest.approx(0.5 - _logistic(-2.0))

    def test_max_over_thetas_and_mean_over_simulations(self):
        # Two simulations with opposite errors cancel in the mean over sims.
        draws = np.stack([np.full((6, 2), 0.5), np.full((6, 2), -0.5)])
        truth = np.zeros((2, 2))
        at_zero = hpo.make_tcc_bias_metric(thetas=(0.0,), model="1pl", scale="tcc")
        assert at_zero(self._inputs(draws, truth, ("b",)))["tcc_bias"] == (
            pytest.approx(0.0, abs=1e-12)
        )
        # one-sided error: the larger |d| of the two thetas is returned
        metric = hpo.make_tcc_bias_metric(thetas=(0.0, 1.0), model="1pl", scale="tcc")
        draws = np.full((2, 6, 2), 0.5)
        got = metric(self._inputs(draws, truth, ("b",)))["tcc_bias"]
        d = [abs(_logistic(t - 0.5) - _logistic(t)) for t in (0.0, 1.0)]
        assert got == pytest.approx(max(d))

    def test_comparable_across_item_counts(self):
        # The same per-item shift gives the same value for 2 and 20 items.
        metric = hpo.make_tcc_bias_metric(thetas=(0.0,), model="1pl", scale="tcc")
        vals = [
            metric(
                self._inputs(np.full((2, 6, n), 0.5), np.zeros((2, n)), ("b",))
            )["tcc_bias"]
            for n in (2, 20)
        ]
        assert vals[0] == pytest.approx(vals[1])

    def test_no_spread_denominator(self):
        # Zero posterior spread is fine here; the z scale refuses it.
        draws = np.full((2, 6, 3), 0.5)
        inputs = self._inputs(draws, np.zeros((2, 3)), ("b",))
        assert np.isfinite(
            hpo.make_tcc_bias_metric(model="1pl", scale="tcc")(inputs)["tcc_bias"]
        )
        with pytest.raises(ValueError, match="zero spread"):
            hpo.make_tcc_bias_metric(model="1pl")(inputs)

    def test_calibrated_is_near_zero_and_shift_is_detected(self):
        metric = hpo.make_tcc_bias_metric(scale="tcc")
        rng = np.random.default_rng(0)
        make = TestTccBiasMetric._calibrated
        ok = metric(self._inputs(*make(rng, 600, 20), ("a", "b")))["tcc_bias"]
        assert ok < 0.01
        biased = metric(
            self._inputs(*make(rng, 600, 20, shift=-0.1), ("a", "b"))
        )["tcc_bias"]
        assert biased > 0.01

    def test_validation(self):
        with pytest.raises(ValueError, match="scale must be"):
            hpo.make_tcc_bias_metric(scale="raw")
        metric = hpo.make_tcc_bias_metric(scale="tcc")
        with pytest.raises(ValueError, match="'a' and 'b'"):
            metric(self._inputs(np.zeros((2, 6, 3)), np.zeros((2, 3)), ("b",)))
        with pytest.raises(ValueError, match="at least 4"):
            hpo.make_tcc_bias_metric(model="1pl", scale="tcc")(
                self._inputs(np.zeros((2, 3, 3)), np.zeros((2, 3)), ("b",))
            )

    def test_direction_is_registered(self):
        from bayesflow_hpo.objectives import METRIC_DIRECTIONS

        direction = METRIC_DIRECTIONS["tcc_bias"]
        assert direction.higher_is_better is False
        assert direction.worst_raw == 1.0

    def test_custom_key_direction_follows_the_scale(self):
        from bayesflow_hpo.objectives import METRIC_DIRECTIONS

        hpo.make_tcc_bias_metric(key="tcc_custom_tcc_key", scale="tcc")
        assert METRIC_DIRECTIONS["tcc_custom_tcc_key"].worst_raw == 1.0
        hpo.make_tcc_bias_metric(key="tcc_custom_z_key", scale="z")
        assert METRIC_DIRECTIONS["tcc_custom_z_key"].worst_raw == float("inf")
