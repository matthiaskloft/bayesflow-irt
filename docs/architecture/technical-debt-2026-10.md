# bayesflow-irt: technical debt and architectural direction

Review date: 20 October 2026. Source snapshot: commit `2a39c5b`.
Status: proposed architectural direction; standards are not yet enforced. Source locations
refer to the reviewed snapshot.

## Assessment and decision context

The package has a substantial working foundation: a src layout, many behavioral tests, serializable custom layers, explicit data-envelope checks, scientific references, distribution validation, and release automation. It does not need a rewrite of its numerical core.

The accumulated debt is mainly **implicit contracts and intertwined responsibilities**. Model identity is reconstructed from adapters, posterior keys, timestamps, and loosely structured metadata. Training recovery owns optimizer semantics as well as resource policy. Diagnostics, HPO, and model cards each implement part of the evaluation workflow. Optional integrations are exposed through the package root and reach into upstream internals. These patterns make small changes require knowledge of distant modules.

The agreed direction for the refactor is:

- A stable public research library maintained by a small team.
- PyTorch is supported; JAX and TensorFlow are best effort.
- A clean Python API break is acceptable. Provide explicit artifact migration where practical rather than retaining every historical runtime path.

Stability should mean documented semantics, trustworthy saved artifacts, and reproducible validation within a declared environment. It should not mean indefinite support for every historical configuration.

## Scope and confidence

The review covered package structure and dependencies; training and inference control flow; simulator/adapter/model contracts; scoring and diagnostic interfaces; metadata and checkpoint compatibility; HPO integration; tests; CI; packaging; documentation; and the available public repository history. Targeted executable probes verified two specific defects. Static findings have source locations below.

This is an architectural and engineering review, not a new validation of every statistical formula or an empirical assessment of posterior quality. No new model was trained for scientific benchmarking. Existing references are useful evidence of intent, but are not treated as proof that every implementation matches its cited method. Local test results are distinguished from clean-install and hosted CI results.

### Structural measurements

Counts include comments/docstrings, and function spans include nested functions. They indicate concentration of responsibilities, not an automatic quality score.

| Measurement | Result |
|---|---:|
| Package Python modules, including `__init__` | 21 |
| Package Python source lines | 25,036 |
| Names exported by root `__all__` | 92 |
| Test modules | 34 |
| AST `test_*` function/method definitions, before parameter expansion | 1,227 |
| Modules exempted from mypy errors | 7 |
| Source lines in those exempted modules | 14,151, or 56.5% |
| Dashboard JavaScript | 1,185 lines |

The five largest modules are `hpo.py` (3,287 lines), `scoring.py` (3,033), `utils.py` (2,766), `networks.py` (2,365), and `sbc.py` (2,208). Together they contain about 55% of package Python source.

Examples of responsibility concentration:

| Function | Source span | Observation |
|---|---:|---|
| `make_irt_hooks` | 1,322 lines, including 382 docstring lines | Builds search spaces and training/validation closures; owns metric routing, aggregation, and compatibility concerns |
| `fit_with_oom_retry` | 707 lines, including 168 docstring lines | Resource search, training progress, optimizer counter changes, callbacks, history merging, interruption handling |
| `PersonScorer.score` | 591 lines, including 152 docstring lines | Validation, grid policy, execution backend, chunking, aggregation, returned summaries |
| `probe_memory_config` | 470 lines, including 123 docstring lines | Cloning, probing, search policy, resource measurement, applying selected configuration |

## Prioritized debt register

P1 means address before relying on the affected workflow or making a stabilization release. P2 means include in the structural refactor. P3 means follow-up consolidation. Effort is relative: S is localized, M crosses a few modules, L changes a contract and its consumers. These are not calendar estimates.

### D01 — OOM recovery can retain optimizer state while rewinding its counter

**P1 / correctness / S–M / reproduced.**

[utils.py:2410](../../src/bayesflow_irt/utils.py#L2410) rewinds `optimizer.iterations` to the baseline plus updates attributed to completed epochs after an OOM. It does not restore the corresponding weights and Adam moment slots. The special case for an OOM before the first completed epoch already recognizes this consistency requirement, but the branch after a completed epoch still violates it.

A probe applied six real Adam updates, recorded one completed four-batch epoch, raised a simulated OOM, and applied one more update through the retry path. The next fit started at counter 4; the final counter was 5 although seven updates remained represented in the weights/moments. The parameter became `-0.0070771724`, versus `-0.0069999355` for seven consistent updates. The injected OOM exercises recovery control flow; it does not claim to reproduce hardware allocation behavior.

Keras 3.12.1 Adam uses the iteration-derived step for moment bias correction; this is more than a learning-rate display issue. See the [versioned Adam implementation](https://github.com/keras-team/keras/blob/v3.12.1/keras/src/optimizers/adam.py).

**Action:** choose one recovery invariant: retain every completed update and keep a monotonic optimizer counter, or restore the entire model/optimizer state to a known boundary. Track reporting epochs separately. Prefer the former initially because it avoids large snapshots; describe the failed partial epoch as retained partial progress. Add the missing real-Adam regression for an OOM after a completed epoch. Existing tests at [test_utils.py:2600](../../tests/test_utils.py#L2600) mainly assert the counter policy, while the real-Adam consistency test at line 2657 covers the before-first-epoch case.

### D02 — Script-defined serializable objects make training artifacts depend on a checkout

**P1 / artifact portability / S / reproduced.**

`WarmupCosine` is defined and registered in [scripts/train_irt.py:91](../../scripts/train_irt.py#L91), then placed in the saved optimizer configuration. The distributed package contains `src/bayesflow_irt`, not an importable `train_irt` module. The source itself documents the need for a supplied custom class.

A small model compiled with this exact schedule saved successfully. In a fresh process that imported Keras and this package, `load_model(..., compile=True)` failed with `Could not locate class 'WarmupCosine'`; `compile=False` loaded. This isolates the schedule defect without claiming all trained models fail to load.

**Action:** every serializable class emitted by supported workflows must live in the installed package at a stable registered identity. Move the schedule into the training implementation, or replace it with an equivalent documented Keras schedule only after checking numerical equivalence. Add fresh-process wheel tests for both inference loading and continued training. The [Keras serialization guide](https://keras.io/guides/serialization_and_saving/) explains that saved models do not contain custom Python class definitions.

### D03 — HPO objective identity is not protected on resume

**P1 / scientific comparability / M / explicit source warning.**

[hpo.py:2037](../../src/bayesflow_irt/hpo.py#L2037) documents that the custom `validate_fn` bypasses the upstream resume guard. Changing references, jitter, or resolution can therefore mix incomparable scores within one study. Additional configuration checks avoid some upstream training-loss fallbacks, but depend on upstream exception behavior; see [hpo.py:61](../../src/bayesflow_irt/hpo.py#L61) and [hpo.py:2380](../../src/bayesflow_irt/hpo.py#L2380).

**Action:** persist a normalized validation manifest on study creation and compare it before any resumed trial. Include model/parameter coordinates, metrics and versions, aggregation rules, references, simulation budget/corners, seeds, and sampling settings. A mismatch must stop the study or require a new study. Unknown configuration must not be converted into a plausible objective. Make the upstream contract explicit through the HPO integration boundary rather than relying on unrelated exception classes throughout IRT code.

### D04 — Supported dependency ranges exceed demonstrated compatibility

**P1 / release confidence / M / observed configuration and local failures.**

[pyproject.toml:26](../../pyproject.toml#L26) declares broad lower bounds for BayesFlow, Keras, NumPy, and other runtime dependencies. CI resolves a current stack plus HPO, rather than testing a minimum supported stack. It covers Python 3.11–3.13 on Linux/PyTorch, but not independent dependency boundaries.

The local environment contains BayesFlow 2.0.8 and Keras 3.12.1, but an editable HPO checkout reporting 0.1.0. Full pytest collection fails because that HPO lacks `AggregationConfigError`. This does not establish a defect in the declared `hpo>=0.5.2` extra; it demonstrates that the local environment is not an acceptable HPO validation environment. The installed torch 2.5.1 CPU build also lacks `torch.cuda.get_per_process_memory_fraction`, although tests assume that attribute can be patched. Production code catches missing attributes in this helper, so this observation establishes a support/test gap rather than proving all training fails on that torch version. Non-HPO baseline results are recorded in the verification appendix.

**Action:** keep one known-good development constraints file and test two deliberate dependency sets: the lowest jointly supported set, and current supported releases. A lock/constraints file supports reproducible development; published library requirements should describe actual compatibility rather than freezing every transitive dependency. Raise unsupported floors based on evidence. Add upper bounds where a specific internal dependency contract makes them necessary, rather than speculative caps everywhere.

### D05 — Upstream internals and optional integrations leak into the package root

**P2 / dependency architecture / M.**

[__init__.py:49](../../src/bayesflow_irt/__init__.py#L49) eagerly imports HPO along with core inference, cards, and diagnostics. The HPO module registers metric directions in an external registry during import. It also imports upstream private or undocumented helpers, including `_direction_for` at [hpo.py:99](../../src/bayesflow_irt/hpo.py#L99) and `_references_for_condition` around line 1622. Network/approximator modules carry BayesFlow import-layout fallbacks. Keras RNG inspection uses `_flatten_layers` and `_seed_generators` in [approximators.py:51](../../src/bayesflow_irt/approximators.py#L51).

The current missing-extra guards are valuable. However, they make the whole root import inspect optional installations, and broad import guards can conflate missing extras with incompatible or broken installations.

**Action:** expose optional HPO/calibration through explicit integration namespaces. Concentrate unavoidable upstream internals in small bridge modules with version-scoped contract tests and removal conditions. Separate integration initialization from importing the numerical core. Reduce root exports to the intended stable user workflow. Do not build a generic plugin framework for two integrations.

### D06 — Model identity and tensor semantics are inferred in several places

**P2 / correctness and extensibility / L.**

Adapters stamp a plain `_bayesflow_irt_internal_parameterization` attribute at [adapters.py:487](../../src/bayesflow_irt/adapters.py#L487) and line 740. [approximators.py:64](../../src/bayesflow_irt/approximators.py#L64) restores it by scanning serialized transforms. [scoring.py:2164](../../src/bayesflow_irt/scoring.py#L2164) infers model family by inspecting adapters, potentially drawing a dummy posterior, then falling back after a broad exception. Posterior key requirements are repeated at lines 1658 and 1698; family maps also exist in simulation and cards.

This is architectural debt even where a later validation path catches an incorrect inference. Construction can perform sampling, and adding a model requires updating distributed knowledge.

**Action:** define one small, serializable `ModelSpec` for family, categories, parameter coordinates/roles, priors, and response conventions. Keep learned normalization values in an `AdapterSpec` or an equally explicit serialized adapter record. Pass explicit model identity to scoring. Define shape contracts at boundaries: responses `(B,P,I)`, per-item targets `(B,I,D)`, draws `(B,S,I,D)`, and category/threshold axes where applicable. Use a small family descriptor table for shared facts; do not introduce an inheritance hierarchy merely to replace four dispatch branches.

### D07 — Checkpoint semantics depend on dates and scattered companion files

**P2 / artifact compatibility / L.**

[compat.py:59](../../src/bayesflow_irt/compat.py#L59) defines an exact historical timestamp for adapter size moments. [compat.py:430](../../src/bayesflow_irt/compat.py#L430) chooses semantics from explicit moments when available, otherwise from training time. [simulators.py:1344](../../src/bayesflow_irt/simulators.py#L1344) refuses ambiguous historical incremental priors. V1 metadata may read another `<tag>_run.json` to recover seed/parameterization at [metadata.py:841](../../src/bayesflow_irt/metadata.py#L841).

These safeguards are preferable to silent scientific changes. Their debt is that historical research context is now part of everyday library execution.

**Action:** use explicit versioned semantics for all new artifacts: model spec, adapter spec and fitted moments, architecture identity, prior-construction version, dependency versions, and checkpoint digest. Under the clean-break policy, move date inference and old-layout recovery to an opt-in migration command. If historical information is insufficient, require a supplied value or declare the artifact unmigratable; do not invent it. Preserve original artifacts during migration and write a migration record.

### D08 — Metadata is versioned but only weakly validated, and inference checks are optional

**P2 / boundary contracts / M–L.**

[MetadataTracker.validate](../../src/bayesflow_irt/metadata.py#L746) checks a few required keys and the presence of training/HPO results. It does not define a complete validated model/architecture/training schema; `save()` does not call it. [approximators.py:997](../../src/bayesflow_irt/approximators.py#L997) only checks the data envelope when the caller supplies metadata. The function explicitly notes that loaded models do not automatically retrieve it. Model-card loading already verifies a recorded checkpoint digest, a useful foundation at [card.py:1423](../../src/bayesflow_irt/card.py#L1423).

**Action:** distinguish a partial metadata builder from a complete validated artifact manifest. Validate required field types and semantic combinations on load/save of the final artifact. Offer a high-level `load_artifact` object pairing model, manifest, and digest verification, so normal prediction uses the recorded contract. Keep low-level approximator use available for experiments. Separate impossible data errors from extrapolation warnings. Never imply that an inside-envelope result establishes statistical validity.

### D09 — Training machinery owns too many independent policies

**P2 / maintainability / L.**

The span of `utils.py` includes resource configuration, CUDA process behavior, disposable model cloning, probing/search, profiling, training retries, history merging, callbacks, and validation datasets. `approximators.py` combines posterior factorization, adaptation, seed replay, custom accumulation, training, sampling, and serialization. `networks.py` includes numerical layers and a substantial custom torch backward implementation at [networks.py:856](../../src/bayesflow_irt/networks.py#L856).

**Action:** extract according to actual responsibilities: memory probe/search; retry policy and progress; datasets; training schedules; backend-specific chunked autograd. Keep the public approximator as a small BayesFlow integration facade. Preserve the tested numerical implementation while moving it. Represent retry attempts and outcomes explicitly rather than hiding progress bookkeeping in nested loops. Make process-wide resource changes a documented execution policy, not an incidental consequence of model construction.

Do not replace the custom backward merely because it is long. Existing tests compare outputs, gradients, dropout replay, and memory behavior; these protect valuable behavior and should drive extraction.

### D10 — Diagnostics and model-card presentation have reverse dependencies

**P2 / architectural boundaries / M.**

There is a runtime dependency cycle through `card_render`, `card_html`, and `card_reference`, using local imports to defer execution. For example, [card_reference.py:100](../../src/bayesflow_irt/card_reference.py#L100) imports dashboard metrics from HTML rendering to validate reference metric keys. [card_html.py:25](../../src/bayesflow_irt/card_html.py#L25) imports many presentation helpers from the Markdown renderer. [card_render.py:312](../../src/bayesflow_irt/card_render.py#L312) imports model-family parameter definitions from the evaluator.

**Action:** extract a presentation-neutral evaluation schema and metric catalog. Reference loading validates against that catalog. Both Markdown and HTML consume the same read model; neither defines scientific metric identity for the other. Evaluation writes records and does not depend on rendering. Remove runtime cycles; retain `TYPE_CHECKING` imports where useful.

### D11 — Three evaluation workflows can drift semantically

**P2 / scientific consistency / M–L.**

`sbc.py`, `hpo.py`, and `card.py` each own parts of sampling, reshaping, metric selection, and aggregation. The model card's [joint_metrics](../../src/bayesflow_irt/card.py#L761) claims the same TCC definitions as the HPO metric factory at [hpo.py:1661](../../src/bayesflow_irt/hpo.py#L1661), but has its own implementation and uncertainty calculations. `card_metrics.py` imports an underscore recovery helper from `sbc.py`, which also contains plotting and orchestration.

Not all apparent duplication should be unified: pooled item summaries, independent dataset-level estimates, objective aggregation, and MCSE calculations can intentionally differ.

**Action:** share pure mathematical kernels and explicit array-coordinate conversions. Keep HPO aggregation, diagnostic sampling, and uncertainty estimation as separate policies. Define metric identity with coordinates, sampling unit, estimand, reduction, and version. Use the same synthetic draws to check that matching definitions agree across entry points. Preserve intentional differences in the manifest and documentation instead of forcing one generic `evaluate()` function to do everything.

### D12 — Type checking gives uneven and environment-dependent assurance

**P2 / verification / M.**

[pyproject.toml:73](../../pyproject.toml#L73) globally ignores missing imports, and seven central modules suppress all type errors. The package ships `py.typed`. The lint CI job installs ruff/mypy, not the runtime dependencies, so missing dependency types can turn important values into `Any`.

Locally, normal `mypy --no-incremental` found five errors in `_numba_kernels.py` and `sbc.py`. Running `mypy --no-site-packages --no-incremental` passed. The latter is a diagnostic approximation of dependency visibility, not a reproduction of the complete hosted job. It demonstrates that a green result can depend on not seeing dependency types.

**Action:** run mypy against the declared development/runtime stack. Scope missing-import exemptions to the third-party packages that need them. Type the new specs, artifact records, posterior contracts, and integration protocols first. Remove whole-module ignores as each module is extracted. Ratchet typed coverage rather than flipping strict mode across 25,000 lines at once; do not repair errors by replacing useful types with `Any`.

### D13 — CI misses the combinations with the greatest maintenance risk

**P2 / verification / M.**

[tests.yml](../../.github/workflows/tests.yml#L1) tests Linux/PyTorch with HPO and default pytest selection. It does not explicitly install calibration or numba. Slow tests are excluded by default in [pyproject.toml:65](../../pyproject.toml#L65); the CUDA memory scaling check is both slow and GPU-gated at [test_networks.py:1484](../../tests/test_networks.py#L1484). Those tests exist, but this workflow does not exercise them. Transitive installation of an extra is not reliable evidence of a tested configuration.

[check_distribution.py](../../scripts/check_distribution.py#L1) already performs useful isolated wheel import/version/asset and `pip check` validation. It stops short of a behavioral simulator → adapter → sample/save/load workflow. JavaScript has a useful conditional Node value test at [test_card_html.py:180](../../tests/test_card_html.py#L180), but no located browser interaction/rendering suite; the CI file does not explicitly provision Node.

The local run also exposed resource-test isolation debt: multiple tests mock only `cuda.is_available()` to true, then reach real `current_device()` on a CPU-only build. Their failure is not evidence of actual GPU memory defects. Mock the resource boundary as a unit, or supply a complete fake device implementation, and reserve real CUDA calls for GPU tests.

**Action:** use the bounded test matrix below. Add checks for extra presence and expected test execution, so missing optional dependencies cannot make a named integration job pass by skipping its tests. Keep expensive GPU measurements out of every ordinary PR, but make them an explicit release condition for memory-path changes.

### D14 — Reproducibility policy is scattered and some explanations are stale

**P2 / reproducibility / M.**

Seed collection/reset exists independently in [approximators.py:34](../../src/bayesflow_irt/approximators.py#L34) and [card.py:273](../../src/bayesflow_irt/card.py#L273). The card records a detailed seed rule and chunk behavior, which is good. However, [card.py:310](../../src/bayesflow_irt/card.py#L310) says the numba kernel ignores the seed, while [simulators.py:418](../../src/bayesflow_irt/simulators.py#L418) supplies one and [_numba_kernels.py:122](../../src/bayesflow_irt/_numba_kernels.py#L122) reseeds per batch row. Tests explicitly check seeded repeatability and thread-count behavior. This is documentation drift; it is not evidence that the kernel is currently unseeded.

[sbc.py:2034](../../src/bayesflow_irt/sbc.py#L2034) temporarily replaces an approximator's `sample` method to inject options. It restores the method in a finally path, but the operation still mutates a shared object for the duration of the diagnostic run.

**Action:** define seed ownership and documented reproducibility scope: simulator, posterior sampler, diagnostics/references, and performance optimizations. Pass sampling options explicitly rather than monkeypatching model methods. Test replay of each supported execution path within the supported stack. Do not promise identical draws across NumPy/numba, different chunk partitions, hardware, or dependency versions without evidence. PyTorch documents checkpoint recomputation and RNG-state limitations in its [checkpoint reference](https://docs.pytorch.org/docs/stable/checkpoint.html).

### D15 — Persistence and provenance have inconsistent guarantees

**P2 / artifact integrity / S–M.**

[card.py:1261](../../src/bayesflow_irt/card.py#L1261) writes evaluation JSON through a temporary file and replace. [metadata.py:825](../../src/bayesflow_irt/metadata.py#L825) truncates/writes metadata directly. Checkpoint and manifest publication are separate operations. `_get_provenance` at [metadata.py:132](../../src/bayesflow_irt/metadata.py#L132) runs git in the caller's current directory, while `session_info` separately distinguishes working-directory and package repository information. A lone `git_commit` is therefore ambiguous when users train from another project.

**Action:** use one strict, atomic JSON writer for important records, unique temporary names, and an explicit publication sequence: write checkpoint, hash it, write validated manifest last. Treat the manifest as the completed artifact indicator; this is not a claim of a multi-file filesystem transaction. Name provenance fields by origin: library version/commit versus caller training-code commit, and record dirty state when available. Keep absence explicit for installed wheels.

### D16 — Public API and configuration outgrew the documented workflow

**P3 / user and maintainer cost / M.**

The root exports 92 names, including compatibility internals, diagnostic helpers, and integration search-space types. The [public API test](../../tests/test_public_api.py#L1) checks a subset is present; it is not an API stability/behavior contract. The README delegates API documentation to docstrings. `MetadataTracker.add_phase2/3/4_metrics` retains research pipeline numbering. The training CLI reads nested dictionaries directly at [train_irt.py:358](../../scripts/train_irt.py#L358), and forwards selected fields, without a single versioned config schema. CONTRIBUTING has duplicate release sections and omits some checks that CI enforces. Version data is manually repeated in packaging, fallback code, citation, and changelog; release checks mitigate but do not eliminate maintenance work.

**Action:** define stable, advanced, and experimental namespaces; document one end-to-end installed-package workflow and a support/capability table. Use semantic training/evaluation/refinement section names. Parse CLI configuration into typed validated records, rejecting unknown fields at the user-input boundary. Put callable training orchestration inside the package and keep the CLI thin; add an installed entry point if the CLI is a supported product surface. Adopt one version authority with generated/checked derivative metadata. Preserve useful scientific docstrings while moving long workflow explanations into task documentation. Keep adopted architectural rules in this public repository and CI; parent-directory agent instructions are not available to external contributors.

## Proposed architecture

Use one distribution and a modest set of responsibility-based packages. The following is a target map, not a requirement to move every file in one commit or to add abstractions for single-use code.

```text
bayesflow_irt/
  __init__.py              small intentional stable facade
  spec.py                  model, parameter, response and shape contracts
  priors.py                prior specifications and sampling
  simulation/              four model simulators, optional numba kernels
  inference/               adapters, approximator, summary/draw networks
    _torch/                chunked autograd and backend execution details
  scoring/                 likelihood kernels, quadrature, scorer/results
  training/                schedules, datasets, probing, retry/progress
  artifacts/               validated manifests, load/save, migration
  diagnostics/             pure metrics, SBC/TARP, evaluation protocols
  cards/                   read model, references, Markdown/HTML/assets
  integrations/            HPO and calibration bridges
  cli/                     argument/config parsing and thin entry points
```

Allowed dependency direction:

```text
cli / integrations / cards
          ↓
training / diagnostics / artifacts
          ↓
inference / scoring / simulation
          ↓
specifications / priors / pure numerical kernels
```

This is a conceptual layering rule with explicit edges, not an assertion that every package in a row may import every peer. Examples of necessary edges: scoring may call an inference interface; artifacts may deserialize inference objects; diagnostic evaluation may use scoring and simulation. Define those edges once as the files move. Pure metric kernels should not depend on artifact loading or plotting. Neither core inference nor artifact validation should import HPO or cards. Specifications must not import the high-level layers that consume them.

Start by extracting shared records/catalogs and the torch execution implementation. Split other modules when a responsibility has independent tests and a clear caller interface. Avoid arbitrary file-size limits, a general dependency-injection framework, a repository-wide class hierarchy, or another package release cycle. File relocation alone will not repay the debt.

## Standards to adopt and enforce

These standards are proposals. None has been installed or enforced by this analysis.

| Standard | Concrete rule | Enforcement |
|---|---|---|
| Explicit domain contract | One serialized model identity and explicit parameter/draw axes; no model-family discovery by dummy sampling | Typed records; boundary tests across 1PL/2PL/PCM/GPCM; reject inconsistent specs |
| Directed dependencies | Lower layers cannot import cards, CLI, or optional integrations; no runtime package cycles | Small architecture test over import edges, including function-local imports; exclude type-only edges; adopt a dedicated import-contract tool only if needed |
| Dependency bridges | Third-party private API use allowed only in named bridge modules, with version rationale and removal condition | Import/attribute checks for known internal APIs plus focused bridge tests; code review for reflective access |
| Artifact ownership | Every saved custom object is installed with the package; checkpoint and manifest are explicitly bound | Wheel-only fresh-process save/load/sample/continue-training tests; digest mismatch tests |
| Versioned semantics | Model, adapter, prior construction, evaluation format, protocol and metric definitions have explicit identifiers | Schema fixtures, migration tests and version-change review; no new timestamp-derived semantics |
| Recovery consistency | Retained weights, moments, counters, accumulation, and RNG state describe the same update history | Real-optimizer fault-injection tests before/after epoch and accumulation boundaries |
| Scientific change discipline | Formula/estimand changes are distinct from structural moves; cite the method and name coordinates, assumptions and sampling unit | Focused reference examples/invariants; shared-input metric parity tests; human scientific review |
| Backend policy | PyTorch CPU is required; CUDA memory features have explicit validation; JAX/TensorFlow are best effort with a capability table | Required PyTorch jobs; selected secondary-backend smoke tests in separate processes, nonblocking initially |
| Optional dependencies | Missing extra gives an actionable error only when used; installed-but-broken integration remains diagnosable | Base-only and extras jobs; assertions that integration tests actually execute |
| Reproducibility | Seed streams and scope are documented; no mutation of another object's methods for configuration | Replay tests and explicit sampler arguments; versioned manifests |
| Types | New public boundaries fully annotated; do not add new whole-module ignores; type-check against real runtime dependencies | Required mypy, scoped third-party exemptions, gradual removal of current ignores; see [mypy configuration semantics](https://mypy.readthedocs.io/en/stable/config_file.html) |
| Failure semantics | Configuration errors stop early; failed evaluations are structured; no silent switch to a different objective | Fault-injection tests and error-type contract tests for HPO and artifact loading |
| Persistence | Important JSON is strict and atomically replaced; manifests are published after checkpoint completion | Writer failure tests; round-trip/schema/digest tests |
| API/configuration | Stable workflow surface is intentional; unknown user config keys fail; experimental features are labeled | Export manifest plus installed examples; versioned config parsing; migration notes |
| Formatting and review | Ruff lint and format checks use the same documented commands locally/CI | Add `ruff format --check .`; pin a development tool set; keep PR changes scoped |

The API/config rule applies validation at actual user/file boundaries. It is not a recommendation to repeat checks inside every numerical helper.

### A manageable verification matrix

| Cadence | Required scope |
|---|---|
| Each PR | Ruff lint/format; mypy in the declared runtime stack; PyTorch CPU tests on Python 3.11–3.13; architecture and artifact-schema checks |
| Each PR, one Python version | Base installation without optional integrations; explicit HPO/numba/calibration integration jobs where available; installed-wheel simulator/adapter/inference/artifact smoke |
| Relevant changes and before release | Slow training/recovery checks; fresh-process continued training; CUDA output/gradient/RNG equivalence and bounded memory behavior |
| Periodic and before release | Lowest jointly supported dependencies and current supported dependencies; Windows smoke for supported PyTorch workflow; JAX/TensorFlow smoke as best effort |
| Card changes | Python payload/metric checks and the existing Node value test; explicit Node provisioning; a small browser smoke for navigation, plots and CSV fallback |

Do not multiply every Python version by every dependency set, backend, extra, and OS. That would create a maintenance burden larger than the team. Cover orthogonal risks with selected jobs. A calibration job cannot establish integration correctness until that dependency is available in the tested environment; until then, mark that feature experimental rather than suggesting a green absent-extra test validates it.

Prefer tests against installed artifacts for release guarantees; pytest's [integration guidance](https://pytest.org/en/stable/explanation/goodpractices.html) distinguishes checkout tests from installed-package tests. The current src layout is already the right foundation.

## Refactoring sequence and exit criteria

### Stage 1: establish a trustworthy baseline and fix isolated defects

1. Create a repository-local environment from a declared constraints set; ensure HPO comes from the required public release rather than an older editable sibling.
2. Reconcile test/type failures without hiding them through imports, skips, or broad ignores. Capture an explicit baseline by feature.
3. Fix D01 and D02 independently, with targeted regressions. Add the HPO study manifest/resume check from D03.
4. Adopt the written backend/API break policy and identify which existing artifacts actually need migration.

**Exit:** supported environment runs the required suite; verified defects have regressions; study resumes cannot change objective identity unnoticed; a script-produced training artifact reloads from the installed package.

### Stage 2: make model and artifact contracts explicit

1. Add the minimal model/adapter/posterior records and artifact manifest.
2. Convert model construction, scoring, and diagnostics to consume those records.
3. Implement normal load/save with digest and schema checks.
4. Move required historical behavior to an explicit migration path. Do not move every historical workaround blindly into the new design.

**Exit:** no dummy sampling for family detection; new artifacts do not depend on timestamps or unrecorded plain attributes; ambiguous old artifacts are refused with actionable migration requirements.

### Stage 3: extract execution and evaluation responsibilities

1. Extract torch-specific chunking/autograd without changing the algorithm.
2. Separate probing/search, retry/progress, schedules, and datasets.
3. Extract shared metric kernels and the evaluation catalog; break card cycles.
4. Narrow HPO/calibration bridge modules and root imports.

**Exit:** architectural dependency tests pass; matching metric definitions agree on shared inputs; output/gradient/seed tests pass before and after extraction; inference can be used without initializing HPO.

### Stage 4: publish the new contract

1. Document stable versus experimental APIs and model/feature/backend capabilities.
2. Package thin training/evaluation entry points with validated configuration.
3. Add user migration guidance and validated examples from an installed wheel.
4. Finish the typing ratchet and dependency test lanes.

**Exit:** an external user can install, simulate, train, save, reload, score, and render diagnostics without sibling repositories or checkout-only definitions. Release checks prove that workflow on the stated supported stack.

Ship these as focused changes with their own evidence. Keep correctness fixes separate from file moves and from scientific algorithm changes. The clean break makes it reasonable to remove legacy call aliases and research-specific phase APIs; it does not justify silently changing posterior coordinates or the meaning of existing reported metrics.

## What to preserve

- Existing output/gradient equivalence tests around person chunking, including dropout/RNG replay.
- Explicit unsupported missing-data behavior and data-envelope checks.
- Checkpoint digest binding and guarded HDF5 rewriting; do not weaken the external-storage checks while migrating compatibility code.
- Source-backed scientific explanations, with workflow material moved rather than deleted.
- Versioned evaluation records and recorded Monte Carlo uncertainty.
- Offline self-contained cards and their packaged assets.
- The separate public repository and its own release cycle. No hardcoded sibling checkout paths were located in package source/tests/scripts; the major remaining coupling is semantic and environmental.

## Article evaluation deliverables (2026-10-28)

Follow-up requirements, separate from the 20 October review findings:
the article focuses exclusively on 2PL, with item calibration and person
scoring both compared against Stan/NUTS and mirt. The model-card evaluation
is the simulation-study backbone. The reference-comparison study plan
specifies the research deliverables.

R/Stan fitting, orchestration and estimator-specific converters stay in the
research repository. The package should provide these reusable pieces:

- [ ] Supported evaluation-input export/reconstruction: responses, truths,
  dataset IDs, selected item/person indices, cell/protocol specification and
  simulator provenance. Research runners should not duplicate private seed
  or selection logic.
- [ ] Stronger reference pairing contract: dataset identity/hash manifests,
  replication counts and per-metric subsets, beyond seed/protocol checks.
  Detect incompatible manifests and expose missing/failed pairs and prior
  mismatches in the card.
- [ ] Supplied paired differences and MCSEs in reference records/rendering.
  card_reference.difference() currently combines marginal MCSEs as independent
  estimates even for a paired reference. Do not present this approximation
  as a paired MCSE; distinguish supplied paired uncertainty, independent
  uncertainty and unavailable uncertainty.
- [ ] Reusable metric computation for reference outputs covering items,
  persons and conditional ability summaries, with explicit coordinates,
  selections and aggregation. Preserve omissions/notes for diagnostics that
  an estimator does not support.
- [ ] TCC interval coverage and width alongside existing bias diagnostics,
  with MCSEs treating datasets as independent units. Verify definitions
  against scientific sources before implementation.
- [ ] Per-dataset metric contributions or an export sufficient for paired
  resampling by the research runner. Keep large raw-draw archives optional.
- [ ] Reference failure/usable counts and timing summaries with explicit
  timing boundaries. Show item and person results and complete workflow
  timing, distinguishing single-dataset latency and batched throughput.
- [ ] Focused tests for manifest mismatches, paired-difference ingestion,
  unavailable paired MCSEs, failed-fit subsets, metric parity and rendering.
  Document the reference/export contract alongside implementation.

Completion: research converters can populate existing cards with traceable
Stan and mirt item/person comparisons and valid paired uncertainty. Extend
existing reference loading/display rather than rebuilding it. Article
tables and figures remain research outputs.

## Verification note

The review used Python 3.11 with Keras 3.12.1, BayesFlow 2.0.8, and a torch
2.5.1 CPU build. An older editable HPO installation prevented full-suite
collection; calibration was absent. These observations are a local diagnostic
baseline, not a clean-install certification or a report of hosted CI status.

- Ruff lint and formatting checks passed.
- Mypy reported five errors in two nonexempt modules. A diagnostic run without
  installed dependency types passed, illustrating the visibility issue in D12.
- Excluding the three HPO test modules, 1,135 tests passed. The remaining
  failures comprised card writes blocked by sandbox temporary-directory
  permissions and 20 CUDA-cap/probe failures involving incomplete device mocks
  or an API absent from the installed torch build.
- All 82 tests in the four affected card modules passed when rerun with an
  explicitly writable temporary directory. Their earlier failures and cascading
  fixture errors are not independently established product defects.
- Targeted probes reproduced D01 with real Adam updates and an injected OOM,
  and D02 with a fresh-process load of the script's serialized schedule.

No clean public-dependency installation or empirical posterior-quality benchmark
was performed. Coverage from the incomplete diagnostic run is not used as a
quality target. Machine-local logs, coverage data, temporary model files, and the
one-off probe script are intentionally excluded from this document.
