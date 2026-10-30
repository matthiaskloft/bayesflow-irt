### Added

- `adapt_and_fit` and `AdaptationResult`: explicit target-size adaptation.
  Fine-tunes a pretrained checkpoint on fresh simulations at the observed
  matrix's dimensions using the recorded priors, saves a distinct checkpoint
  and metadata sidecar (published only after save, reload and a weight-digest
  check), then draws item posteriors and optionally scores persons.
  `prior_overrides` accepts `Prior` or `VariablePrior` objects; observed
  responses are never training targets. The parent's gradient accumulation is
  not inherited, and masked response cells are rejected.
- Recorded `VariablePrior` specifications can be rebuilt from metadata.
