### Added

- `bayesflow_irt.evaluation.sample_posterior`, one chunked posterior-sampling
  driver shared by the model card and HPO validation. With
  `batch_flow_rows=True` a flow-matching chunk is passed as one ODE batch.
  HPO validation on an RTX 5090 is about 22x (flow matching) to 25x
  (autoregressive) faster per 50x500 corner.

### Changed

- HPO validation samples through `bayesflow_irt.evaluation.sample_posterior`,
  the chunked driver the model card uses: datasets per call sized from free
  device memory (`validation_chunk_size="auto"`, new `make_irt_hooks`
  argument), out-of-memory shrink and retry, and each chunk passed to the
  item posterior as one batch of rows instead of many 256-row calls.
  Validation draws therefore differ from earlier releases;
  scores from before and after are not draw-for-draw comparable.
- `card.reset_posterior_seed` and the chunk-size helpers moved to
  `bayesflow_irt.evaluation`; `card` re-exports them. Card draws are
  unchanged.

### Fixed

- Model card dashboard: colours are kept when extrapolation is shown, and
  plot sizing and layout are improved.
