# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Changed

- Model card dashboard: the calibration view shows one coverage plot instead
  of two. The across-cells coverage plot now uses the 20 levels the
  calibration error averages over (records without them keep the six summary
  levels), and the separate per-cell plot is gone. Its caption notes that
  coverage misses a shifted posterior, which the SBC ECDF difference shows.
- Model card dashboard: the SBC ECDF difference plot draws every in-range
  cell faded behind the chosen one, with their median, as the coverage plot
  does. The sidebar header is set apart from the page tabs.
- Model card dashboard: with "Show extrapolation cells" on, the coverage and
  SBC ECDF plots add the out-of-range cells as dashed lines, outside the
  median and shading. The across-cell line plots share one width and one
  set of line weights.
- Model card dashboard: Recovery and Calibration keep the same parameter tab
  chosen, as they already did the cell.

### Fixed

- Model card dashboard: "Copy as CSV" no longer just reports a failure when
  the browser blocks the clipboard (embedded viewers, `file://` pages). It
  falls back to the copy command, then shows the CSV to select with a
  "Download CSV" link.

## [0.1.0] - 2026-10-19

First public release.

### Added

- Simulators and adapters for the 1PL, 2PL, PCM, and GPCM models over
  variable-size response matrices (`make_*_simulator`, `make_*_adapter`),
  with configurable and variable priors (`Prior`, `VariablePrior`,
  incremental priors for PCM/GPCM) and optional numba response sampling.
- `MatrixEquivariantIRTSummary`, a summary network exchangeable over items
  and persons, and `EquivariantIRTApproximator` for per-item posterior
  inference, including an autoregressive item posterior.
- `CalibratedIRTApproximator` for calibration-aware training (optional,
  via `bayesflow-calibration-loss`).
- Person scoring by grid quadrature (`PersonScorer`).
- Diagnostics: simulation-based calibration, TARP coverage, and
  calibration error (`run_sbc_diagnostics`, `compute_tarp_coverage`,
  `compute_calibration_error`, and related helpers).
- Model cards: `evaluate_for_card` evaluates a trained model on a grid of
  data sizes and `render_model_card` writes a Markdown card and an HTML
  dashboard.
- Metadata sidecars for trained models (`MetadataTracker`), simulator
  rebuilds from a sidecar (`build_simulator`), and data checks against the
  trained range (`check_data`).
- Memory controls for large `n_persons`: person-axis chunking, memory
  probing, and out-of-memory retry (`fit_with_oom_retry`).
- Hyperparameter-optimization hooks for `bayesflow-hpo` (optional,
  `bayesflow_irt.hpo`).
- `scripts/train_irt.py` for config-driven training, and a minimal training
  example (`examples/train_1pl_minimal.py`).
