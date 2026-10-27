### Changed

- **Breaking:** `EquivariantIRTApproximator` always uses the autoregressive
  item posterior; the `item_posterior` argument is gone. The per-item flow
  is the `inference_network`, a `CouplingFlow` or now also a `FlowMatching`
  network (one ODE solve per item when sampling, so slower). Pass
  `FlowMatching` solver settings as `flow_sample_kwargs`.
- **Breaking:** `make_irt_hooks` takes `flow="coupling_flow"` (default),
  `"flow_matching"` or a list of both instead of `item_posterior`. The new
  `AutoregressiveFlowMatchingSpace` adds the draws-summary dimensions to
  `FlowMatchingSpace`, and `FlowChoiceSpace` (searched as a `flow`
  dimension) replaces `ItemPosteriorChoiceSpace`; `normalize_flow` replaces
  `normalize_item_posterior`. `flow_sample_kwargs` applies to flow-matching
  trials.
- `sample_posterior` no longer takes `batch_flow_rows` or `responses_key`:
  every chunk's rows go to the flow as one batch.
- Autoregressive sampling sends up to 16384 `(dataset, draw)` rows through
  each flow call by default (`condition_chunk_size`, previously 256). A flow
  call costs about the same for 1 or tens of thousands of rows, so sampling
  1000 draws for 16 datasets of 50 items went from 186 s to 3 s on a GTX 1660
  SUPER. Pass `condition_chunk_size` to lower it if GPU memory runs out.

### Removed

- **Breaking:** the independent item posterior (`item_posterior="flow_matching"`,
  items sampled independently given the summary). In a 2PL comparison it was
  less calibrated than the autoregressive coupling flow, its draws made
  functionals across items (test characteristic curves) too narrow, and it
  was slower and used more memory to sample (#20). Models saved with it no longer
  load; `from_config` says so and points to bayesflow-irt 0.2.
- With it: `CalibratedIRTApproximator` and the `calibration` extra (the
  calibration loss supported only the independent item posterior),
  `standardize_params` (standardize item parameters in the adapter, as the
  built-in adapters do), `profile_posterior_inference`, `InferenceProfile`,
  and the `sample_chunk_size` argument of `EquivariantIRTApproximator.sample`.

### Fixed

- `sample_posterior` frees a failed attempt's device memory before it retries
  after running out of memory: it strips the error's tracebacks and runs the
  cycle collector. Before, the retries could shrink to one dataset and still
  fail (#21).
- `sample_posterior` runs the cycle collector between chunks and around the
  `chunk_size="auto"` probe. Each flow-matching chunk left about 1 GiB of
  device tensors in reference cycles. Over 24 chunks of a 50x500 corner that
  added up to 22 GiB, and the auto chunk size was computed from memory still
  holding that garbage (#21). Each collection costs about 0.13 s with a large
  model loaded, so a GPU call with `chunk_size="auto"` takes about 0.26 s
  longer.
