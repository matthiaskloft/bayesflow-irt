# Future R wrapper package

## Python bridge candidate: mizu (2026-11-05)

Consider **mizu/pymizu alongside reticulate** when designing the future R
wrapper for bayesflow-irt. This is an option to investigate, not a dependency
or an architecture decision. No existing R-package plan was found in this
checkout when this note was added.

The relevant feature is R–Python interoperability, rather than faster Python
training. mizu can launch Python workers from R and exchange supported arrays
and tables through shared memory. A possible wrapper would expose operations
such as loading a checkpoint, inferring item parameters, and scoring persons,
with a persistent Python worker retaining the model and scorer between calls.
Worker-local persistence must be verified before adopting this design.

Compared with reticulate's in-process Python interface, a worker bridge offers
process isolation and potential asynchronous execution, but requires an
explicit data-oriented interface and worker lifecycle management. Large
response matrices could benefit from shared-memory exchange; this does not
imply faster GPU computation or zero-copy transfer for every payload.

Before choosing a bridge, prototype and compare:

- Repeated inference/scoring calls with a model retained in one Python worker.
- Matrix shapes, memory order, dtypes, missing values, and posterior result
  structures; verify which transfers actually avoid copies or conversions.
- Python environment selection, backend initialization, error propagation,
  worker restart, and cleanup on R-session exit.
- Reproducible simulator and posterior RNG state across calls and workers.
- End-to-end latency and peak memory against reticulate on realistic workloads.
- Windows installation and packaging: current pymizu documentation requires
  clang-cl for a Windows build. The API is experimental.

Sources:

- [Release announcement](https://opensource.posit.co/blog/2026-11-05_mizu-0-1-0/)
- [R-to-Python interoperability](https://shikokuchuo.net/mizu/articles/interop.html)
- [Python-side interoperability and dtype rules](https://shikokuchuo.net/pymizu/user-guide/interop.html)
- [Platform and installation requirements](https://shikokuchuo.net/pymizu/user-guide/installation.html)
