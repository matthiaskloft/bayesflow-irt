### Changed

- `condition_chunk_size` for autoregressive sampling defaults to `"auto"`.
  On PyTorch with CUDA the first chunk (at most 16384 rows) measures the
  peak device memory per `(dataset, draw)` row once per approximator; every
  later chunk and call, such as a loop of SBC or scoring calls, fills 80% of
  the free device memory from that measurement, scaled up for more items. A
  chunk that runs out of memory shrinks by a quarter and is retried. Without
  CUDA it stays 16384 rows. With `"auto"`, chunking depends on free memory,
  so pass an integer `condition_chunk_size` to reproduce seeded draws
  exactly; an integer is used as given and out-of-memory errors propagate.
