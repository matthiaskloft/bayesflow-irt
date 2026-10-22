# MCAR-capable 2PL model plan

Updated: 2026-10-22. Status: planning only; model implementation is future work.

## Scope and agreed design

- Target sizes: 5–50 items and 10–1000 persons.
- Version 1 uses a maximum requested missingness rate of 50% for every size.
  No size-dependent rate caps are needed in the first implementation.
- Missingness masks depend only on dataset size, the requested rate, and random
  draws; never on responses or model parameters. These are structured MCAR masks,
  not independent Bernoulli draws per cell.
- For N persons, J items, and requested rate rho, set K=round(rho*N*J),
  k=K//N, r=K-N*k. Uniformly select r persons to have k+1 missing cells;
  the others have k. Select missing positions uniformly and independently per row.
- Require at least one observed response per person. Reject and redraw the entire
  mask if any item is entirely missing. Keep N, J, and K fixed during retries.
- Validate the integer bound K <= N*J-max(N,J) before drawing masks. Specify the
  rounding convention; the experiment uses nearest integer, ties to even.
- External NaNs become zero-filled responses plus an observed/missing indicator.
- Preserve optional hand-built response statistics, computed from observed cells
  only. Compute only enabled features and their required dependencies.
- Inject log observed counts late in the summary network.

## Feasibility and rejection cost

The bound is necessary and sufficient even with balanced row allocations.
Let L=NJ-K be the number of observations. Then L>=max(N,J), and every balanced
observed row degree lies in [1,J]. Assign each row its observed cells consecutively
around a cyclic item list, continuing across rows: no row repeats a column, and
all columns are covered because L>=J. This proves existence; it is not a substitute
for the requested random sampler.

Existence does not imply efficient rejection. For fixed missing row counts m_i,
the expected number of empty items is J*product_i(m_i/J). The union bound implies
acceptance >= 1-E[empty items]. Exact acceptance is also available by
inclusion-exclusion over empty items, as implemented in the standalone probe.

For simple rejection, provisionally prefer acceptance >=0.9 and require >=0.5
for the coarse allowed-rate grid below (at most two attempts on average).
These are computational policy choices, not inferential guarantees. Add a finite
retry budget and an explicit failure outcome; do not silently change rho after
failure, since that would change the intended training distribution.

## Exploratory maximum-rate grid (deferred beyond version 1)

Highest TESTED rate with exact single-proposal acceptance >=0.5:

| Persons / Items | 5 | 10 | 20 | 50 |
|---|---:|---:|---:|---:|
| 10 | 80% | 50% | 50% | 60% |
| 20 | 80% | 80% | 80% | 80% |
| 50 | 80% | 90% | 90% | 90% |
| 100 | 80% | 90% | 95% | 95% |
| 1000 | 80% | 90% | 95% | 98% |

This is a coarse computational grid, not a finalized training policy or an exact
maximum. In particular, 60% was tested at 10 persons / 50 items but not at
10 persons / 10 or 20 items; do not interpret that irregularity as a real benefit
of having more items. Refine the rate grid before implementing interpolation or
a continuous size-dependent cap. Always enforce the integer feasibility bound.

Version 1 restricts rho to [0,0.5] across all sizes. Higher rates and size-dependent
caps are deferred until recovery/calibration experiments justify extending this
range. Choose the complete-dataset mixture probability explicitly; the mask
simulation provides no evidence for a particular frequency. The distribution of
positive rates and final size-dependent inferential caps remain open decisions.

## Evidence and reproducibility

- `mcar_mask_probe.py`: standalone NumPy experiment, no production imports.
- `mcar_mask_probe_results.csv`: 116 cases, including 93 in the target size range
  (75 feasible); 2,000–20,000 independent proposals per feasible case.
- `mcar_mask_probe_results.json`: seed, environment, trial budget, timing details.
- Reproduce from the repository root:
  `python dev/mcar_mask_probe.py --fresh --trials 20000 --seed 20261021`.
- Exact integer inclusion-exclusion avoids cancellation at very low acceptance.
  Exhaustive small-mask checks verify it. Construction checks cover 988 feasible
  small rectangles/allocations. Empirical empty-item means provide another check.
- Per-cell 95% Wilson intervals use the NIST formula:
  https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm.
  They are not simultaneous intervals; some discrepancies across many cells are
  expected. Zero accepted proposals is not proof of impossibility.
- Measured CPU batched proposal/check cost: about 8 microseconds for 10x50,
  0.96 milliseconds for 1000x50. At 10x50 / 80% missingness, exact expected
  attempts are about 956, giving projected accepted-mask cost about 8.6 ms.
  At 10x50 / 90%, expected attempts are about 3.7e19 despite feasibility.
  These timings are not production training throughput measurements.

## Implementation follow-up

Problem: support incomplete response matrices across 5–50 items and 10–1000
persons while retaining observed-only optional statistics and explicit counts.

- [ ] Apply the fixed 50% maximum requested missingness rate across all version 1
      conditions; retain the integer feasibility check after rounding K.
- [ ] Choose the positive-rate distribution, complete-dataset mixture frequency,
      and bounded retry policy explicitly.
- [ ] Implement response-independent balanced masks with exact K, nonempty rows,
      whole-mask rejection for empty items, and pre-draw feasibility validation.
- [ ] Implement NaN-to-zero-plus-indicator input handling, observed-only optional
      statistics with enabled dependencies only, and late log-count injection.
- [ ] Check invariants, deterministic seeds, and complete-data behavior.
- [ ] Benchmark mask generation in the actual training pipeline.
- [ ] Run recovery and posterior calibration experiments across sizes and rates
      up to 50%; adjust the training range if these checks require it.
- [ ] Consider rates above 50% and refine the exploratory size-dependent grid
      only as a later extension supported by recovery/calibration evidence.

This plan and its simulation artifacts are a planning deliverable. They do not
implement or establish posterior calibration for an MCAR-capable model.
