### Removed

- `load_weights_compat` (also exported from `bayesflow_irt`) and the
  `compat.py` attention-rename machinery for BayesFlow 2.0.8 vs 2.0.10
  checkpoints. It could not load 2.0.x files on 2.1 and 2.1 needs no key
  renaming. `weights_digest` and the size-moment helpers remain.

### Changed

- **BREAKING:** requires `bayesflow>=2.1.0`; BayesFlow 2.0.x is no longer
  supported. 2.1 builds a different ISAB/SAB/PMA from the same settings
  (pre-LN RMSNorm, gated residuals starting at 0, GLU feed-forward, no bias,
  orthogonal init; about half the parameters at 16 inducing points, width 64,
  4 heads) and moves the coupling-flow subnet initializer from
  `glorot_uniform` to `orthogonal`. Checkpoints, weights files and HPO
  studies from 2.0.x cannot be loaded or resumed; retrain. See
  `docs/architecture/bayesflow-2.1-audit.md` and
  `docs/architecture/bayesflow-2.1-checkpoints.md`.
- Dependency floors raised to match BayesFlow 2.1: `keras>=3.15.0`,
  `numpy>=2.5.1`, `scipy>=1.18.0`; backend extras to the Keras 3.15 minimums (`torch>=2.1`,
  `tensorflow>=2.16.1`, `jax>=0.4.20`); `card` extra `matplotlib>=3.11.1`. The `hpo` extra now
  requires `bayesflow-hpo>=0.8.0`, the first release that supports BayesFlow
  2.1 (matthiaskloft/bayesflow-hpo#152).
- Requires Python >= 3.12 (and < 3.14 in practice), as BayesFlow 2.1.0 does;
  Python 3.11 support dropped.
- `Adapter.add_transform` (removed in 2.1) replaced by `Adapter.append`;
  custom adapter transforms accept backend tensors;
  `EquivariantIRTApproximator.sample` still returns NumPy arrays. Direct
  `Adapter` calls (forward or `inverse=True`) now return backend tensors
  (BayesFlow 2.1 behaviour); use `keras.ops.convert_to_numpy`.

### Removed

- `outer_mlp_depth` and `outer_mlp_width` from `MatrixEquivariantIRTSummary`
  and the HPO builder: BayesFlow 2.1 attention blocks have no such settings
  and silently ignored them. Passing them now raises `TypeError`.
