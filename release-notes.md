### Fixed

- `is_oom_error` now treats `RuntimeError("CUDA driver error: device not
  ready")` as an out-of-memory error when `PYTORCH_CUDA_ALLOC_CONF` contains
  `expandable_segments:True` (seen on WSL2), so `fit_with_oom_retry` and
  `adapt_and_fit` run their recovery ladder instead of re-raising (#42).
