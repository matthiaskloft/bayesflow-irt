### Added

- HPO: new categorical search dimension `irt_matrix_outer_attention`
  (`"isab16"`, `"isab32"`, `"full"`) on `MatrixIRTSummarySpace` makes the outer
  item attention searchable (ISAB with 16 or 32 inducing points, or full
  self-attention). Default `"isab16"` is the previously hard-coded setting:
  params without the key rebuild unchanged. Pin it with
  `CategoricalDimension("irt_matrix_outer_attention", constant=...)`.
  Trials of studies started before this change have no value for it and ran
  with `"isab16"`; read a missing value as `"isab16"` when comparing modes.
  Seeded reruns of a study may propose different values than before, since
  the new dimension is sampled among the others.
- HPO resume guard: `make_irt_hooks(...)` now builds a normalized validation
  manifest (`hooks.manifest`: parameters, metrics and directions, aggregation,
  joint-metric settings incl. `reference_id`/`resolution`, corners, per-corner
  budget, seed, a digest of the validation data, flows and sampling kwargs,
  optional `manifest_extra`, package versions). The first trial stores it in
  the Optuna study's user attributes; a resumed study is compared before any
  training and a mismatch raises `ValidationManifestError` naming the
  differing keys (start a new study). Versions compare at major.minor. A
  study with finished trials but no manifest is refused unless
  `allow_unversioned_study=True`. `make_classical_reference_provider` exposes
  `.reference_settings` for `manifest_extra`. Previously the custom
  `validate_fn` bypassed upstream's resume guard silently.

### Fixed

- `WarmupCosine` now ships in the installed package
  (`bayesflow_irt.schedules`, also exported from the package root). It used to
  live only in `scripts/train_irt.py`, so a fresh process could not load a
  compiled checkpoint (`load_model(..., compile=True)` raised "Could not
  locate class 'WarmupCosine'"). The registered name is unchanged
  (`bayesflow_irt_scripts>WarmupCosine`), so existing checkpoints load without
  `custom_objects`.
- `fit_with_oom_retry`: after a CUDA OOM following a completed epoch the
  optimizer step counter was rewound while the weights and Adam moment slots
  kept the failed partial epoch's updates, skewing Adam bias correction and
  LR schedules. Every applied update is now retained as partial progress and
  `optimizer.iterations` stays monotonic; remaining epochs are tracked
  separately.
- Model card: the item-posterior type was reported as "not recorded" for
  models whose hyperparameters arrive flat (e.g. from HPO trials), because the
  metadata tracker filed `item_posterior` under `training` while the card read
  it from `architecture`. The tracker now files it under `architecture`, the
  card falls back to `training` for existing sidecars, and autoregressive
  coupling flows get their own limitation note (items drawn jointly).
