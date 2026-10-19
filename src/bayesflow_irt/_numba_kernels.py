"""Optional numba-accelerated response-sampling kernels.

Requires the ``numba`` extra: ``pip install bayesflow-irt[numba]``. This
module is imported lazily by :mod:`bayesflow_irt.simulators` and must never
be imported at package top level, so a missing numba dependency never
breaks the base install.

These kernels fuse the multi-pass numpy operations used in
``simulators.py`` (noise draw, clip, log, subtract, argmax / compare) into a
single per-element pass, avoiding the several full-array temporaries the
numpy path allocates.

Why these kernels are single-threaded (not ``parallel=True``/``prange``)
--------------------------------------------------------------------------
Parallelizing the batch axis with ``numba.prange`` is tempting. In a
tight loop of back-to-back ``simulator.sample()`` calls (nothing else
competing for CPU), that ``parallel=True`` kernel is genuinely faster than
both the numpy path and the single-threaded version below -- confirmed by
benchmark.

But that isn't the only place these kernels run. Two call sites both put a
``parallel=True`` kernel in contention with other CPU-bound work on the
*same* threads:

1. ``Approximator.build_dataset`` defaults to ``workers="auto"`` ==
   ``os.cpu_count()``. With ``use_multiprocessing=False`` (also the
   default), Keras's ``OrderedEnqueuer`` runs that many *Python threads*
   (via ``multiprocessing.dummy.Pool``) that each independently call
   ``simulator.sample()`` to prefetch batches. Each of those threads then
   also spins up numba's *own* internal thread pool for the ``prange``
   loop, so up to ``cpu_count() ** 2`` OS threads end up contending for
   ``cpu_count()`` physical cores -- the oversubscription scenario
   numba's own threading-layer docs warn about.
2. Even with ``workers=0`` (no prefetch pool at all, which keeps GPU
   memory usage predictable), data
   generation and the GPU training step run serially on the *same*
   thread. Every call to a ``parallel=True`` kernel there still spins up
   and tears down numba's internal thread pool, momentarily competing for
   CPU with PyTorch's own host-side orchestration (issuing CUDA kernels,
   autograd bookkeeping, gradient-checkpointing recomputation) for that
   same step, with no prefetching to overlap and hide the cost. Benchmarked
   end-to-end against a small ``EquivariantIRTApproximator`` this showed a
   measurable (if variable, roughly 5-35% depending on run length and
   model size) per-step slowdown relative to both the numpy path and the
   single-threaded kernel below -- nowhere near the pathological
   ``cpu_count() ** 2`` case above, but consistently never a win either.

Removing ``prange``/``parallel=True`` avoids both failure modes: whatever
outer concurrency exists (Keras's thread pool, or none at all) already
determines how many cores are in play, so each kernel call itself just
needs to run its fused single-pass computation without also trying to
grab every core for itself. This keeps the fusion speedup (no per-op
numpy temporaries) without numba ever contending with something else for
CPU.

RNG note
--------
These kernels use numba's own internal RNG (a legacy MT19937-based
implementation), which is a **separate stream** from the
``numpy.random.Generator`` (PCG64) the rest of the package uses. Enabling
``use_numba=True`` on a simulator therefore changes the exact simulated
values for a given ``seed=`` (same distributions, different draws) -- it is
opt-in for this reason, see ``simulators.py``.

This is a deliberate choice, not an oversight to "modernize" later. Numba
(>=0.56) can accept a ``numpy.random.Generator`` object passed in from
regular Python and call its methods (e.g. ``rng.standard_exponential()``)
inside ``@njit`` code, which would let these kernels share the exact same
RNG stream as the rest of the simulator. But numba's own documentation
states this Generator support is **not thread-safe**, and this was
confirmed empirically: passing a ``Generator`` into an
``@njit(parallel=True)`` function and calling its methods from inside
``prange`` compiles and runs without any error, yet gives a *different*
result across repeated runs with the same seed (the same Generator object
used serially, with no ``prange``, is bit-identical to plain numpy and
reproducible run-to-run -- the non-reproducibility is specifically a
``prange`` parallel-scheduling hazard). The legacy ``np.random.*``
global-state API used here has been thread-safe and fork-safe since numba
0.28, which is why it is used instead despite requiring a separate RNG
stream.

Each kernel reseeds numba's RNG **per batch row** -- e.g.
``np.random.seed(seed + b)`` before the per-item work for row ``b`` --
rather than seeding once for the whole call. Under ``prange``, seeding once
before the loop makes output depend on the number of threads, because
numba's RNG state is thread-local and advances independently of loop order
under parallel scheduling. Per-row reseeding is the pattern that is both
reproducible (same seed -> bit-identical output across repeated runs) and
thread-count-invariant, so it is kept although the loop is serial, which
also preserves bit-identical output for existing ``use_numba=True`` seeds.
Do not "simplify" this by
hoisting the seed call out of the loop -- that would change existing
users' simulated values for the same ``seed=``.
"""

from __future__ import annotations

import numpy as np
from numba import njit

_TINY_F32 = np.float32(1.1754944e-38)  # np.finfo(np.float32).tiny


@njit(fastmath=True, cache=True)
def binary_response_kernel(logits: np.ndarray, out: np.ndarray, seed: int) -> None:
    """Fused sigmoid + uniform draw + compare, written into ``out`` in place.

    Parameters
    ----------
    logits : np.ndarray, shape (batch_size, n_persons, n_items), float32
        Item logits (read-only).
    out : np.ndarray, shape (batch_size, n_persons, n_items)
        Preallocated output buffer (``uint8`` or ``float32``) to write
        binary responses into.
    seed : int
        Seed for numba's internal RNG (independent of the simulator's own
        ``np.random.Generator``; see module docstring).
    """
    batch_size, n_persons, n_items = logits.shape
    for b in range(batch_size):
        np.random.seed(seed + b)
        for p in range(n_persons):
            for i in range(n_items):
                x = logits[b, p, i]
                sig = np.float32(1.0) / (np.float32(1.0) + np.exp(-x))
                u = np.random.random()
                out[b, p, i] = 1 if u < sig else 0


@njit(fastmath=True, cache=True)
def categorical_response_kernel(logits: np.ndarray, out: np.ndarray, seed: int) -> None:
    """Fused exponential-noise Gumbel-max + argmax, written into ``out``.

    Uses the exponential-noise form of Gumbel-max described by Maddison et
    al. (2014): if E ~ Exponential(1), then -log(E) is standard Gumbel
    noise. See ``references.md`` and ``simulators.py``'s
    ``_sample_categorical_responses_from_logits`` for the numpy equivalent
    of this same construction.

    Parameters
    ----------
    logits : np.ndarray, shape (batch_size, n_persons, n_items, n_categories), float32
        Category logits (read-only).
    out : np.ndarray, shape (batch_size, n_persons, n_items)
        Preallocated output buffer (``uint8`` or ``float32``) to write
        category indices into.
    seed : int
        Seed for numba's internal RNG (independent of the simulator's own
        ``np.random.Generator``; see module docstring).
    """
    batch_size, n_persons, n_items, n_categories = logits.shape
    for b in range(batch_size):
        np.random.seed(seed + b)
        for p in range(n_persons):
            for i in range(n_items):
                # k=0 seeds best_val/best_idx directly, avoiding a -inf
                # sentinel (which fastmath's no-infinity assumption could
                # otherwise mishandle).
                e0 = np.random.exponential(1.0)
                if e0 < _TINY_F32:
                    e0 = _TINY_F32
                # Gumbel = -log(E), and argmax(logits + Gumbel) samples
                # Categorical(softmax(logits)); logits - log(E) is the
                # same thing written without the double negative.
                best_val = logits[b, p, i, 0] - np.float32(np.log(e0))
                best_idx = 0
                for k in range(1, n_categories):
                    e = np.random.exponential(1.0)
                    if e < _TINY_F32:
                        e = _TINY_F32
                    v = logits[b, p, i, k] - np.float32(np.log(e))
                    if v > best_val:
                        best_val = v
                        best_idx = k
                out[b, p, i] = best_idx
