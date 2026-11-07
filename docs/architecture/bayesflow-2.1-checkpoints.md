# 2.0.x checkpoints and BayesFlow 2.1

Nothing in the repository ships trained weights: `git ls-files` lists no
`.keras`, `.weights.h5`, `.pt` or `.safetensors` file, and `.gitignore`
excludes `/checkpoints/` and local run output. The impact is on checkpoints
held elsewhere and on code and docs that read them.

A `.keras` archive written under 2.0.x stores 2.0.x layer configs
(`mlp_depth`, `mlp_width`, post-LN MAB) and a different variable set (see
`bayesflow-2.1-audit.md`). It cannot be loaded into 2.1 layers, and the
removed `outer_mlp_*` arguments would reject the old config as well.
Weights-only files cannot be remapped: shapes and variable counts differ
(208,768 vs 107,908 parameters at the reference ISAB), so
the former `compat.load_weights_compat` rename shim would not have helped
(it is removed).

| Reference | What it assumes | Proposal |
|---|---|---|
| `README.md` "Planned: pretrained checkpoints", adaptation section (`parent.keras`) | Checkpoints exist; none distributed | README now states checkpoints must be trained on 2.1 |
| `examples/adapt_1pl.py` | User-supplied pretrained `.keras` | Add a docstring note that the parent must come from 2.1 (e.g. `train_1pl_minimal.py`) |
| `examples/train_1pl_minimal.py` | Writes `models_trained/examples/1pl_minimal_example.keras` | Retrain on 2.1 if an old local copy exists; the file is untracked |
| `scripts/test_onnx_export.py`, `scripts/test_onnx_recovery.py` | `--model` / `--weights` input files | Regenerate inputs on 2.1 |
| `scripts/configs/2pl_fixed.json`, `smoke_1pl.json` | Training configs, no weights | Reusable; they contain no `outer_mlp_*` |
| `tests/fixtures/sidecars/2pl_equivariant_calibrated_metadata.json` | JSON sidecar fixture | Sidecar only; unchanged |
| `src/bayesflow_irt/compat.py`, `tests/test_compat.py` | Formerly a 2.0.8 vs 2.0.10 key-rename shim | Shim removed (could not rescue 2.0.x files); `weights_digest` and size-moment helpers kept |
| `src/bayesflow_irt/adaptation.py`, `card.py` | `keras.saving.load_model(...)` on a user checkpoint | Loading a 2.0.x file fails loudly; add a clearer error only if wanted |
| Research-repo HPO checkpoints and Optuna studies | Built on 2.0.x | Retrain; do not resume studies |

Nothing was deleted or retrained.
