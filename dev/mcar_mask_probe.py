"""Standalone planning experiment; no BayesFlow imports or production changes.

Run: python dev/mcar_mask_probe.py --trials 20000 --seed 20261021
Outputs CSV plus environment/method metadata next to this script.
Only NumPy is required. Timings measure batched CPU proposal generation/checks,
not model training or a production retry implementation.

Wilson interval formula: https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm
Independent slice permutations: https://numpy.org/doc/2.4/reference/random/generated/numpy.random.Generator.permuted.html

Derivations specific to this experiment:
For fixed observed row degrees d_i, a specified set of s items is empty with
probability product_i C(J-s,d_i)/C(J,d_i). Inclusion-exclusion over the J items
therefore gives exact acceptance. Integer numerator/denominator arithmetic
avoids cancellation even when acceptance is extremely small.
Expected empty items = J * product_i(m_i/J), by linearity of expectation.

Feasibility: L=NJ-K >= max(N,J) is necessary and sufficient, also for balanced
rows. Each balanced observed degree is in [1,J]. Assign each row d_i successive
columns in a cyclic list of all J columns, continuing where the last row ended.
No row repeats a column, and L>=J covers every column. This is a proof witness,
not a replacement for the requested random sampler.
"""

import argparse
import csv
import itertools
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np


def exact_acceptance(n, j, k_total):
    k, r = divmod(k_total, n)
    groups = [(j - k, n - r), (j - k - 1, r)]
    denominator = math.prod(math.comb(j, d) ** count for d, count in groups)
    numerator = 0
    for s in range(j + 1):
        ways = math.prod(
            (math.comb(j - s, d) if d <= j - s else 0) ** count
            for d, count in groups
        )
        numerator += (-1) ** s * math.comb(j, s) * ways
    return numerator / denominator


def wilson(success, trials):
    z = 1.959963984540054
    p = success / trials
    center = (p + z * z / (2 * trials)) / (1 + z * z / trials)
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials**2))
    half /= 1 + z * z / trials
    return max(0.0, center - half), min(1.0, center + half)


def draw_masks(rng, n, j, k_total, batch):
    k, r = divmod(k_total, n)
    counts = np.full(n, k, dtype=np.int64)
    counts[:r] += 1
    counts = rng.permuted(np.broadcast_to(counts, (batch, n)), axis=1)
    return rng.permuted(np.arange(j) < counts[:, :, None], axis=2)


def verify_edges():
    # Exhaust all feasible K for small rectangles and construct proof witnesses.
    checked = 0
    for n in range(1, 9):
        for j in range(1, 9):
            for k_total in range(n * j - max(n, j) + 1):
                k, r = divmod(k_total, n)
                observed = np.zeros((n, j), dtype=bool)
                cursor = 0
                for i in range(n):
                    degree = j - k - (i < r)
                    observed[i, np.arange(cursor, cursor + degree) % j] = True
                    cursor += degree
                assert observed.any(axis=0).all() and observed.any(axis=1).all()
                assert observed.sum() == n * j - k_total
                checked += 1
    rng = np.random.default_rng(234)
    masks = draw_masks(rng, 7, 11, 31, 1000)
    assert (masks.sum(axis=(1, 2)) == 31).all()
    assert (np.sort(masks.sum(axis=2), axis=1) == [4, 4, 4, 4, 5, 5, 5]).all()
    assert (~masks).any(axis=2).all()
    assert exact_acceptance(2, 2, 2) == 0.5
    assert math.isclose(exact_acceptance(10, 10, 90), math.factorial(10) / 10**10)
    # Independent exhaustive check of the inclusion-exclusion implementation.
    for n in range(1, 4):
        for j in range(1, 5):
            for k_total in range(n * j - max(n, j) + 1):
                k, r = divmod(k_total, n)
                row_options = [list(itertools.combinations(range(j), j-k-(i < r)))
                               for i in range(n)]
                total = accepted = 0
                for rows in itertools.product(*row_options):
                    total += 1
                    accepted += len(set().union(*map(set, rows))) == j
                assert math.isclose(exact_acceptance(n, j, k_total), accepted/total)
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=20261021)
    parser.add_argument(
        '--fresh', action='store_true', help='Ignore saved same-seed results'
    )
    args = parser.parse_args()
    checked = verify_edges()
    output = Path(__file__).resolve().with_name('mcar_mask_probe_results.csv')
    cache = {}
    metadata_path = output.with_suffix('.json')
    if not args.fresh and output.exists() and metadata_path.exists():
        old_metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        if old_metadata['seed'] == args.seed:
            with output.open(newline='', encoding='utf-8') as handle:
                for row in csv.DictReader(handle):
                    if row['feasible'] == 'True':
                        converted = {}
                        for key, value in row.items():
                            if key == 'feasible':
                                converted[key] = True
                            elif not value:
                                converted[key] = None
                            elif key in {'n', 'j', 'k_total', 'trials', 'accepted'}:
                                converted[key] = int(value)
                            else:
                                converted[key] = float(value)
                        cache[
                            (int(row['n']), int(row['j']), int(row['k_total']))
                        ] = converted
    cases = []
    # User's relevant training-size range; cross all representative sizes.
    for n in (10, 20, 50, 100, 1000):
        for j in (5, 10, 20, 50):
            boundary = (n*j-max(n, j))/(n*j)
            for rho in sorted({.5, .8, .9, .95, boundary}):
                cases.append((n, j, rho))
    for n, j, rhos in [
        (100, 10, [0, .5, .8, .9]),
        (100, 20, [.8, .9, .95]),
        (200, 20, [.9, .95]),
        (50, 15, [.8, .9]),
        (10, 50, [.2, .4, .5, .6, .7, .8, .9]),
        (5, 100, [.2, .4, .5, .6, .7, .8]),
        (10, 100, [.5, .7, .8, .9]),
        (20, 100, [.8, .9, .95]),
        (100, 100, [.9, .95, .99]),
        (1, 20, [0, .05]),
        (20, 1, [0, .05]),
        (2, 2, [.5, .75]),
        (10, 10, [.9, .91]),
        (3, 5, [8 / 15]),
    ]:
        for rho in rhos:
            cases.append((n, j, rho))
    records = []
    seen = set()
    for n, j, rho in cases:
        k_total = round(rho * n * j)  # Python ties-to-even, explicitly reproducible.
        key = (n, j, k_total)
        if key in seen:
            continue
        seen.add(key)
        record = dict(n=n, j=j, rho_requested=rho, k_total=k_total,
                      rho_actual=k_total / (n * j), feasible=k_total <= n*j-max(n,j))
        if record['feasible']:
            trials = min(args.trials, max(2000, 100_000_000 // (n*j)))
            if key in cache and cache[key]['trials'] >= trials:
                records.append(cache[key])
                print(
                    f'{n}x{j} rho={rho:.4f}: reused {cache[key]["trials"]} trials',
                    flush=True,
                )
                continue
            k, r = divmod(k_total, n)
            expected_empty = j * (k / j) ** (n - r) * ((k + 1) / j) ** r
            p_exact = exact_acceptance(n, j, k_total)
            rng = np.random.default_rng(
                np.random.SeedSequence([args.seed, n, j, k_total])
            )
            success = empty_sum = 0
            batch_max = min(512, max(1, 2_000_000 // (n * j)))
            start = time.perf_counter()
            for offset in range(0, trials, batch_max):
                batch = min(batch_max, trials - offset)
                masks = draw_masks(rng, n, j, k_total, batch)
                empty = masks.all(axis=1).sum(axis=1)
                success += int((empty == 0).sum())
                empty_sum += int(empty.sum())
            elapsed = time.perf_counter() - start
            lo, hi = wilson(success, trials)
            proposal_us = elapsed / trials * 1e6
            record.update(
                trials=trials, accepted=success, p_hat=success/trials,
                p_ci95_low=lo, p_ci95_high=hi, p_exact=p_exact,
                attempts_exact=1/p_exact, attempts_ci95_low=1/hi,
                attempts_ci95_high=(1/lo if success else None),
                empty_expected=expected_empty, empty_empirical=empty_sum/trials,
                proposal_us=proposal_us, accepted_us_projected=proposal_us/p_exact,
                experiment_seconds=elapsed,
            )
            print(f'{n:3}x{j:<3} rho={rho:.4f} p={success/trials:.6f} '
                  f'[{lo:.6f},{hi:.6f}] exact={p_exact:.6g} '
                  f'attempts={1/p_exact:.5g} proposal={proposal_us:.2f}us', flush=True)
        else:
            print(f'{n}x{j} rho={rho:.4f}: infeasible', flush=True)
        records.append(record)
    fields = list(dict.fromkeys(key for rec in records for key in rec))
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    metadata = dict(seed=args.seed, max_trials_per_feasible_case=args.trials,
                    trial_budget='min(max_trials, max(2000, 100_000_000 // (N*J)))',
                    cache=('Reuse matching-seed cells with at least the requested '
                           'trial budget'),
                    python=sys.version, numpy=np.__version__,
                    platform=platform.platform(),
                    processor=platform.processor(),
                    witness_cases_checked=checked,
                    total_simulation_seconds=sum(
                        r.get('experiment_seconds', 0) for r in records
                    ),
                    rounding='Python round: nearest, ties to even',
                    interval='95% Wilson, per cell, not simultaneous',
                    timing=('Batched CPU NumPy full-mask proposal + coverage check; '
                            'no retries; projected accepted cost = measured '
                            'proposal cost / exact acceptance'),
                    row_assignment=('Uniform independent permutation of balanced '
                                    'row counts per mask'),
                    mask_independence=('No responses or model parameters are '
                                       'generated or read'),
                    analytic=('Integer inclusion-exclusion exact rational converted '
                              'to float'))
    output.with_suffix('.json').write_text(
        json.dumps(metadata, indent=2), encoding='utf-8'
    )
    print(output)


if __name__ == '__main__':
    main()
