"""Validation-manifest resume guard for ``make_irt_hooks`` (D03)."""

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("bayesflow_hpo", reason="bayesflow_hpo not installed")
optuna = pytest.importorskip("optuna", reason="optuna not installed")

import bayesflow_irt.hpo as hpo_mod  # noqa: E402
from bayesflow_irt import Prior, make_2pl_adapter, make_2pl_simulator  # noqa: E402

# Attributes are read off the module at call time, not imported by name: other
# HPO tests reload ``bayesflow_irt.hpo``, which rebinds its classes, and a name
# imported earlier would then no longer match the exception actually raised.
VALIDATION_MANIFEST_ATTR = hpo_mod.VALIDATION_MANIFEST_ATTR


def Err():  # noqa: N802
    return hpo_mod.ValidationManifestError


def check(*args, **kwargs):
    return hpo_mod.check_validation_manifest(*args, **kwargs)

CORNERS = [(3, 20), (5, 30)]


def _sim(scale_a=0.5):
    return make_2pl_simulator(
        n_items=(3, 5),
        n_persons=(20, 30),
        prior_a=Prior("lognormal", loc=0, scale=scale_a),
        prior_b=Prior("normal", loc=0, scale=2.0),
        seed=1,
    )


def _hooks(scale_a=0.5, **kw):
    sim = _sim(scale_a)
    kw.setdefault("sims_per_corner", 3)
    kw.setdefault("metrics", ["nrmse", "calibration_error"])
    return hpo_mod.make_irt_hooks(sim, make_2pl_adapter(sim), corners=CORNERS, **kw)


def _study(tmp_path=None):
    storage = None if tmp_path is None else f"sqlite:///{tmp_path / 's.db'}"
    return optuna.create_study(study_name="t", storage=storage, load_if_exists=True)


def _finish_trial(study):
    study.add_trial(optuna.trial.create_trial(value=0.5, distributions={}, params={}))


class TestManifest:
    def test_manifest_is_json_and_deterministic(self):
        m1, m2 = _hooks().manifest, _hooks().manifest
        assert m1 == m2
        assert json.loads(json.dumps(m1, sort_keys=True)) == m1
        scoring = m1["scoring"]
        for key in (
            "param_keys", "metrics", "metric_directions", "aggregate",
            "corners", "sims_per_corner", "seed", "validation_data_sha256",
        ):
            assert key in scoring
        assert set(m1["versions"]) == {"bayesflow", "bayesflow_hpo", "bayesflow_irt"}

    def test_fresh_study_is_stamped(self):
        hooks, study = _hooks(), _study()
        assert check(study, hooks.manifest) == "stamped"
        assert study.user_attrs[VALIDATION_MANIFEST_ATTR] == hooks.manifest

    def test_identical_resume_passes(self, tmp_path):
        study = _study(tmp_path)
        check(study, _hooks().manifest)
        _finish_trial(study)
        resumed = _study(tmp_path)
        assert check(resumed, _hooks().manifest) == "matched"

    def test_patch_version_change_is_tolerated(self):
        hooks, study = _hooks(), _study()
        stored = json.loads(json.dumps(hooks.manifest))
        stored["versions"]["bayesflow_irt"] = "0.5.9"
        cur = json.loads(json.dumps(hooks.manifest))
        cur["versions"]["bayesflow_irt"] = "0.5.0"
        study.set_user_attr(VALIDATION_MANIFEST_ATTR, stored)
        assert check(study, cur) == "matched"
        cur["versions"]["bayesflow_irt"] = "0.6.0"
        with pytest.raises(Err(), match="versions.bayesflow_irt"):
            check(study, cur)

    @pytest.mark.parametrize(
        ("change", "key"),
        [
            ({"metrics": ["nrmse"]}, "scoring.metrics"),
            ({"aggregate": "mean"}, "scoring.aggregate"),
            ({"seed": 7}, "scoring.seed"),
            ({"sims_per_corner": 4}, "scoring.sims_per_corner"),
            (
                {"manifest_extra": {"reference": {"jitter": 0.1}}},
                "scoring.extra.reference.jitter",
            ),
        ],
    )
    def test_changed_setting_fails_naming_key(self, change, key):
        study = _study()
        check(study, _hooks().manifest)
        _finish_trial(study)
        with pytest.raises(Err(), match="new study") as exc:
            check(study, _hooks(**change).manifest)
        assert key in str(exc.value)

    def test_changed_reference_id_and_resolution_fail(self):
        tarp = pytest.importorskip("bayesflow_hpo.validation.tarp")
        provider = hpo_mod.make_classical_reference_provider(["a", "b"])

        def hooks_with(ref_id, resolution=10):
            return _hooks(
                joint_metrics={
                    "tarp_error": tarp.make_tarp_joint_metric(
                        provider, reference_id=ref_id, resolution=resolution
                    )
                },
            )

        study = _study()
        check(study, hooks_with("ref-a").manifest)
        with pytest.raises(Err(), match="reference_id"):
            check(study, hooks_with("ref-b").manifest)
        with pytest.raises(Err(), match="resolution"):
            check(study, hooks_with("ref-a", 20).manifest)

    def test_changed_simulator_prior_fails_via_data_digest(self):
        study = _study()
        check(study, _hooks().manifest)
        with pytest.raises(Err(), match="validation_data_sha256"):
            check(study, _hooks(scale_a=0.9).manifest)

    def test_legacy_study_refused_by_default(self):
        study = _study()
        _finish_trial(study)
        with pytest.raises(Err(), match="no stored validation"):
            check(study, _hooks().manifest)
        assert VALIDATION_MANIFEST_ATTR not in study.user_attrs

    def test_legacy_study_adopted_with_override(self, caplog):
        hooks, study = _hooks(), _study()
        _finish_trial(study)
        with caplog.at_level("WARNING"):
            result = check(
                study, hooks.manifest, allow_unversioned_study=True
            )
        assert result == "adopted"
        assert study.user_attrs[VALIDATION_MANIFEST_ATTR] == hooks.manifest
        assert "unversioned" in caplog.text

    def test_failed_trial_does_not_make_study_legacy(self):
        study = _study()
        study.add_trial(optuna.trial.create_trial(state=optuna.trial.TrialState.FAIL))
        assert check(study, _hooks().manifest) == "stamped"

    def test_unserializable_extra_is_rejected(self):
        with pytest.raises(TypeError, match="manifest path"):
            _hooks(manifest_extra={"x": object()})


class TestTrainFnGuard:
    def test_error_is_the_class_the_objective_re_raises(self):
        from bayesflow_hpo import JointMetricConfigurationError

        assert issubclass(Err(), JointMetricConfigurationError)

    def test_train_fn_stamps_then_refuses_before_training(self):
        study = _study()
        hooks = _hooks()
        # The guard runs first and needs no approximator; training itself then
        # fails on the None placeholder, which is not the guard's error.
        trial = study.ask()
        with pytest.raises(Exception) as first:
            hooks.train_fn(None, None, {}, [SimpleNamespace(trial=trial)])
        assert not isinstance(first.value, Err())
        assert study.user_attrs[VALIDATION_MANIFEST_ATTR] == hooks.manifest
        _finish_trial(study)
        trial2 = study.ask()
        with pytest.raises(Err(), match="scoring.seed"):
            _hooks(seed=99).train_fn(None, None, {}, [SimpleNamespace(trial=trial2)])

    def test_manifest_overwritten_during_training_refuses(self, monkeypatch):
        # Two workers stamping a fresh study at once: the later write wins
        # while this worker trains, so it must refuse before being scored.
        study = _study()
        hooks = _hooks()
        other = _hooks(seed=99).manifest

        def fake_fit(*args, **kwargs):
            study.set_user_attr(VALIDATION_MANIFEST_ATTR, other)
            return SimpleNamespace(attempts=1)

        monkeypatch.setattr(hpo_mod, "fit_with_oom_retry", fake_fit)
        trial = study.ask()
        with pytest.raises(Err(), match="scoring.seed"):
            hooks.train_fn(
                SimpleNamespace(),
                None,
                {"epochs": 1, "num_batches": 1},
                [SimpleNamespace(trial=trial)],
            )

    def test_declared_output_key_directions_are_recorded(self, monkeypatch):
        monkeypatch.setattr(
            hpo_mod, "output_keys_for", lambda name: [f"{name}_out"]
        )
        hooks = _hooks(joint_metrics={"jm": lambda inputs: {}})
        assert "jm_out" in hooks.manifest["scoring"]["metric_directions"]

    def test_callbacks_without_a_trial_skip_the_guard(self):
        hpo_mod._guard_from_callbacks([SimpleNamespace()], _hooks().manifest, False)


def test_marginal_metric_output_key_directions_are_recorded():
    hooks = _hooks(metrics=["coverage"])
    assert "coverage_90" in hooks.manifest["scoring"]["metric_directions"]
