# Idea: parallel (Jacobi) sampling of the autoregressive item posterior

Status: parked (2026-10-28). Not implemented, not measured on a trained model.

## Why it could matter

Autoregressive sampling runs one flow call per item. On GPU, a call costs about
55–70 ms whether it gets 1 row or ~65k rows (Python/Keras layer overhead), so
time is roughly `n_items x per-call cost x row chunks`. With
`condition_chunk_size="auto"` there is usually one chunk, so a call takes about
`n_items x 60 ms` (16 datasets x 50 items x 1000 draws: ~3 s on a GTX 1660 SUPER).

For many datasets (SBC, model cards) that is fine. For an **empirical dataset
where only one posterior is needed** the rows per call are just the draws
(e.g. 1000). Then each flow call is far below the overhead floor, and nearly
all the time is the `n_items` sequential calls. Jacobi sampling replaces those
with a few calls over `n_items x draws` rows, which is where it would help most.

## Method

Song, Meng, Liao, and Ermon (2021), "Accelerating feedforward computation via
parallel nonlinear equation solving", ICML, PMLR 139, pp. 9791-9800 (see
`references.md`). Algorithm 1 is the plain Jacobi iteration below;
Proposition 1 guarantees the sequential result after at most T (here
`n_items`) iterations from any initialization. Algorithms 2 and 3 (Jacobi-GS,
GS-Jacobi) work on blocks of items and also converge in finitely many steps.
For ancestral sampling they report 26x for MADE and 2.1-6.5x for PixelCNN++
(Sec. 5.3).

### The paper's formulation (Eq. 5, Algorithms 1-3)

Equations below are transcribed from the paper's equation and algorithm images
(the converted text has them only as images).

Feedforward computation of states $\mathbf{s}_1, \dots, \mathbf{s}_T$ from an
input $\mathbf{u}$ solves the triangular system (Eq. 5)

$$
h_t(\mathbf{u}, \mathbf{s}_{1:t-1}) - \mathbf{s}_t = 0, \qquad t = 1, \dots, T .
$$

Sequential computation is one Gauss-Seidel sweep over it. **Algorithm 1
(nonlinear Jacobi)** instead updates all states at once from the previous
iterate:

$$
\mathbf{s}_t^{k} \leftarrow h_t\!\left(\mathbf{u}, \mathbf{s}_{1:t-1}^{k-1}\right)
\quad \text{for } t = 1, \dots, T \text{ in parallel},
$$

starting from any $\mathbf{s}^0_{1:T}$, until $k = T$ or
$\lVert \mathbf{s}^{k}_{1:T} - \mathbf{s}^{k-1}_{1:T} \rVert \le \epsilon$.
With $\epsilon = 0$ it returns the sequential result after at most $T$
iterations (Proposition 1): after pass $k$ the first $k$ states are exact.

For blocks $\mathcal{B}_1, \dots, \mathcal{B}_M$ partitioning $1, \dots, T$,
with $\mathcal{B}_i = [\![a, b]\!]$:

- **Algorithm 2 (Jacobi-GS).** All blocks in parallel, states inside a block
  sequentially:
  $\mathbf{s}_j^{k} \leftarrow h_j\!\left(\mathbf{u}, \mathbf{s}^{k-1}_{1:a-1},
  \mathbf{s}^{k}_{a:j-1}\right)$ for $j \in \mathcal{B}_i$, until $k = M$ or the
  change is at most $\epsilon$.
- **Algorithm 3 (GS-Jacobi).** Blocks in sequence, Jacobi inside each block:
  $\mathbf{s}_j^{k} \leftarrow h_j\!\left(\mathbf{u}, \mathbf{s}_{1:a-1},
  \mathbf{s}^{k-1}_{a:j-1}\right)$ for $j \in \mathcal{B}_i$ in parallel, until
  $k = \lvert \mathcal{B}_i \rvert$ or
  $\lVert \mathbf{s}^{k}_{\mathcal{B}_i} - \mathbf{s}^{k-1}_{\mathcal{B}_i} \rVert
  \le \epsilon$. Earlier blocks $\mathbf{s}_{1:a-1}$ are final.

Both converge in at most $M$ block iterations with $\epsilon = 0$
(Proposition 2).

### Mapping onto the autoregressive item posterior

In `_sample_rows` (`approximators.py`) item $t$'s draw is

$$
\boldsymbol{\theta}_t = F^{-1}\!\left(\mathbf{z}_t;\;
\mathbf{c}_t\right), \qquad
\mathbf{c}_t = \left[\mathbf{h}_t,\;
\mathrm{pool}\big(\varphi(\boldsymbol{\theta}_1, \mathbf{h}_1), \dots,
\varphi(\boldsymbol{\theta}_{t-1}, \mathbf{h}_{t-1})\big),\;
\mathbf{x}\right],
$$

with $F$ the per-item flow, $\mathbf{z}_t$ its base noise, $\mathbf{h}_t$ the
summary-network output for item $t$, $\varphi$ and $\mathrm{pool}$ the draws
summary (`phi_values`, prefix pooling) and $\mathbf{x}$ the optional
`inference_conditions`. So $\mathbf{s}_t = \boldsymbol{\theta}_t$,
$\mathbf{u} = (\mathbf{z}_{1:T}, \mathbf{h}_{1:T}, \mathbf{x})$, $T$ = number
of items, and $h_t$ is "pool the prefix, then invert the flow".

1. Draw $\mathbf{z}_{1:T}$ once and keep it fixed. `inference_network.sample`
   draws its own noise on every call, so a pass must call the flow's inverse
   on given $\mathbf{z}$ instead; otherwise the iteration has no fixed point.
2. Initialize $\boldsymbol{\theta}^0_{1:T}$, e.g. the flow output with the
   prefix pooling of zero items for every $t$.
3. Each pass: compute all $\mathbf{c}_t$ from $\boldsymbol{\theta}^{k-1}$ (for
   `"mean"` a cumulative sum over items), invert the flow for all $T$ items as
   one batch of $T \times$ rows.
4. Stop at $\lVert \boldsymbol{\theta}^{k} - \boldsymbol{\theta}^{k-1} \rVert
   \le \epsilon$ or after $T$ passes; at $T$ passes the draws equal sequential
   sampling with the same $\mathbf{z}$.

## What the paper says about our case

- **Cost per pass (Sec. 5.3.1).** MADE is their best case: one sequential step
  costs a full network pass, the same as one parallel Jacobi pass, so Jacobi
  needs no extra parallel hardware (26x on MNIST, 25x on CIFAR-10, measured
  wall-clock on one GPU). One-dataset sampling here is close to that: each
  flow call is overhead-bound, so one pass over all items costs about one
  sequential item call.
- **Passes to converge (Sec. 4.1).** The number of passes is bounded by the
  critical path of the dependency graph, but in practice it depends on how
  strongly states influence each other. With fully independent states one
  pass is exact (Example 1), and DenseNets converge fast although every layer
  is connected, because most connections matter little. Here every item
  depends on all earlier items through the prefix poolings. If item
  parameters are only weakly dependent given the summary, convergence should
  take few passes; that is the open question.
- **Stopping rule (Secs. 5.2, 5.3.1).** The change between passes tracked the
  true error closely for DenseNets and MADE, so it worked as a stopping rule
  there.
- **Fallback when memory runs out (Sec. 5.3.2, Table 1).** GS-Jacobi runs
  blocks of items in sequence, Jacobi inside each block. For PixelCNN++ on
  CIFAR-10 plain Jacobi gained only 1.18x and GS-Jacobi 2.09x, because the
  sequential sampler could reuse a cache that plain Jacobi cannot.

## What is known, what to measure first

- One parallel pass: 54–71 ms for 80k rows, 0.5–1.1 s for 800k rows (GTX 1660
  SUPER). For one dataset (1000 draws x 50 items = 50k rows) a pass is about
  one sequential call, so even 10 passes would be ~5x faster than 50.
- Unknown: passes to convergence for a *trained* model (untrained flows are
  not representative). Measure that first: one trained 2PL model, count passes
  to tolerances 1e-3 / 1e-5, compare draws to sequential with the same `z`.
- Stopping at a tolerance gives slightly different draws than sequential
  sampling; check calibration (SBC) if it is ever used outside one-off fits.
- Memory scales with `n_items x rows` per pass, so it needs the row chunking.

## Related cheaper idea

For 1PL, BayesFlow `DualCoupling` has `pivot = dim // 2 = 0`, so one coupling
per pair is a no-op that still runs an MLP. Skipping it would roughly halve 1PL
flow cost exactly (from reading the code, untested).
