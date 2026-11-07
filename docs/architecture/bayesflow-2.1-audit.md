# BayesFlow 2.0.8 -> 2.1.0 audit

`bayesflow-irt` requires `bayesflow>=2.1.0` (decision 2026-11-07; 2.0.x is not
supported). This note records what the package builds from BayesFlow layers,
what changed between 2.0.8 and 2.1.0 in each, and what that does to capacity
and results. Both versions were read from the installed source. Paths are
relative to `site-packages/bayesflow/networks/`; 2.0.8 line numbers refer to
the 2.0.8 wheel, 2.1.0 to the 2.1.0 wheel. No separate release notes were used.

## Summary

| Layer built by the package | Capacity / results change? | Package action |
|---|---|---|
| Outer `InducedSetAttention` (ISAB) / `SetAttention` | **Yes**: different block, fewer parameters, `mlp_width`/`mlp_depth` gone | Removed dead `outer_mlp_depth`/`outer_mlp_width` |
| Inner matrix blocks (package code, plain Keras) | No | none |
| Draws-summary / person `PoolingByMultiHeadAttention` (PMA) | **Yes**: new pre-FFN, gated FFN, orthogonal init | none (defaults used); documented here |
| `CouplingFlow` + subnet | Same computation; initializer `glorot_uniform` -> `orthogonal` | none |
| `Adapter` | API change (`add_transform` removed, tensors in/out) | done in the earlier commit on this branch |

Every 2.0.x `.keras` / `.weights.h5` file is incompatible; see
`bayesflow-2.1-checkpoints.md`. Retraining is needed, not a re-load.

## 1. ISAB / SAB (outer item attention)

Built in `src/bayesflow_irt/networks.py` (`_make_sab_stack`) for
`MatrixEquivariantIRTSummary` (`use_outer_isab`, `num_outer_inducing_points`,
`num_outer_heads`, `outer_embed_dim`, `dropout`).

2.0.8 (`transformers/mab.py`, `isab.py`):

- MAB is post-LN: `input_projector(x) + attention`, `LayerNormalization`, then
  `h + output_projector(MLP(h))`, `LayerNormalization` (mab.py:113-121).
- Attention `key_dim=embed_dim` (mab.py:72), i.e. full width per head.
- FFN is `MLP(widths=(mlp_width,)*mlp_depth, activation="gelu")`
  (mab.py:80-85); defaults `mlp_depth=2`, `mlp_width=128`, `use_bias=True`,
  `kernel_initializer="lecun_normal"`.
- Inducing points `glorot_uniform` (isab.py:66-68).

2.1.0 (`summary/transformers/attention/multihead_attention.py`,
`induced_set_attention.py`, `helpers/blocks/transformer_feedforward.py`):

- MAB is pre-LN with `RMSNormalization` before attention (queries and
  key/values separately) and before the FFN (multihead_attention.py:105-107,
  149-169).
- Attention `key_dim = embed_dim // num_heads` (:88).
- FFN is a gated GLU (`swiglu` default) with intermediate width
  `ceil64(int(embed_dim * expansion_factor * 2/3))`
  (transformer_feedforward.py:83-86), e.g. 64 -> 192; `use_bias=False`.
- Both residual branches are multiplied by a trainable scalar initialized to
  zero (`gate_attention`, `gate_ffn`; :110-115), so each block starts as the
  identity.
- Init is `orthogonal` for all kernels and the inducing points
  (induced_set_attention.py:53, 74-78). The docstrings still say
  `glorot_uniform`; the signature defaults are authoritative.
- `input_projector` is `Identity` when the input width equals `embed_dim`,
  else a bias-free `Dense` (:236-246). 2.0.8 always had a `Dense`.
- Signature: `num_inducing_points, embed_dim, num_heads, dropout,
  expansion_factor, glu_variant, kernel_initializer, use_bias, layer_norm,
  gate_attention, gate_ffn`. There is **no** `mlp_depth`, `mlp_width` or
  `mlp_activation`. Extra kwargs go through `layer_kwargs`
  (`utils/dict_utils.py:83-86`), which keeps only `dtype`, `name` and
  `trainable` and drops the rest **silently**. This is why the old call kept
  "working" on 2.1 with no effect.

What the package now builds: `InducedSetAttention(num_inducing_points,
embed_dim=outer_embed_dim, num_heads=num_outer_heads, dropout=dropout)` and
`SetAttention(embed_dim, num_heads, dropout)`. The dead
`outer_mlp_depth`/`outer_mlp_width` arguments were removed from
`MatrixEquivariantIRTSummary`, its helpers, `MatrixIRTSummarySpace` (`hpo.py`)
and the example; passing them now raises `TypeError`. The FFN width follows
from `outer_embed_dim` (factor 4 * 2/3, rounded up to a multiple of 64).

Capacity (ISAB, 16 inducing points, width 64, 4 heads; figures from the
requesting session, not re-measured here): 208,768 trainable parameters on
2.0.8 vs 107,908 on 2.1.0. The outer encoder is a different, smaller model.
Zero-initialized gates and orthogonal init also change early training. HPO
results and benchmark numbers from 2.0.x are not comparable.

The package passes no `attention_mask` to ISAB, SAB or PMA, so the changed mask
conventions in 2.1 do not matter here.

## 2. Inner blocks

The matrix blocks and projection heads are package code on plain Keras layers
and are unchanged. Only the Keras version moved (3.12.1 -> 3.15.1 in the test
environment).

## 3. Draws summary and person pooling: PMA

`networks.py` builds `PoolingByMultiHeadAttention(num_seeds, embed_dim,
num_heads, dropout)` (`person_pma`) for `person_pooling` in `"pma"` /
`"mean_pma"`; all other arguments use defaults.

2.0.8 (`transformers/pma.py`): post-LN `MultiHeadAttentionBlock`, seed vector
`glorot_uniform` (:89), `MLP(widths=(128,128))` applied to the set before
pooling (:93).

2.1.0 (`pooling_by_multihead_attention.py`): MAB with `gate_attention=False`,
`gate_ffn=True`; seed vector `orthogonal`; a new pre-FFN (RMSNorm, GLU FFN)
whose output is `skip + pre_ffn_scale * ffn_out` with `pre_ffn_scale`
zero-initialized, so the set passes through unchanged at the start; a bias-free
skip projector only if widths differ. Output shape
`(batch, num_seeds * embed_dim)` is unchanged. `build` needs a concrete shape;
the package's eager build of `person_pma` still applies.

`compat.py` (rename shim for `*_block` vs unsuffixed layer names) addressed
2.0.8 vs 2.0.10 key names. It cannot make a 2.0.x checkpoint load on 2.1,
because the variable sets differ. It stays harmless for weights-only files
written by 2.1.

## 4. Coupling flow and subnet

Used by `examples/` (`bf.networks.CouplingFlow()`), tests, and the HPO
`AutoregressiveCouplingFlowSpace` through `bayesflow_hpo`.

Compared `coupling_flow/` (2.0.8) with `inference/coupling/` (2.1.0):

- Computation: same architecture (ActNorm, permutation, `DualCoupling`, affine
  transform with `arcsinh` clamp and shifted softplus). The `log_det` sum moved
  out of the transform (`transforms/affine_transform.py` now returns
  per-element `log_jac`); same total.
- New optional `mask_handler` / `target_mask` plumbing; unused by the package.
- Subnet defaults: 2.0.8 `SingleCoupling.MLP_DEFAULT_CONFIG`
  (single_coupling.py:19-26) is widths (128,128), `hard_silu`,
  `glorot_uniform`, no residual, dropout 0.05. 2.1.0 `defaults.py`
  `COUPLING_MLP_DEFAULTS` is identical except `kernel_initializer="orthogonal"`
  (merged at `single_coupling.py:53`); the 2.0.8 `spectral_normalization`
  option is replaced by `norm=None`.
- Output projector is still zero-initialized.

Effect: same capacity, different initialization, so different seeded
trajectories and possibly different training speed. An explicit
`kernel_initializer` in `subnet_kwargs` overrides the default and keeps the
2.0.8 behaviour. Not checked: whether `bayesflow_hpo` 0.7.0 passes one.

## 5. Adapters

`Adapter.add_transform` was removed; the package uses `append`
(`adapters.py`). Adapters pass backend tensors between transforms and return
tensors from `inverse` (`adapters/adapter.py`, `device_handoff` near :377).
The custom transforms accept tensors and `EquivariantIRTApproximator.sample`
converts back to NumPy. Done in the earlier commit on this branch.

## Open points for review

1. ISAB capacity roughly halves for the same settings; the search dimensions
   `irt_matrix_outer_embed_dim` and `irt_matrix_outer_attention` now span
   different models. Re-run, do not resume, 2.0.x studies.
2. Seeded results do not reproduce across the boundary (init, extra RNG draws
   in 2.1).
3. The removed fixed MLP width (`2 * outer_embed`) has no 2.1 equivalent; the
   package does not expose `expansion_factor`.
