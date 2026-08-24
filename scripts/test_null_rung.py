#!/usr/bin/env python3
"""Does the null rung actually behave like a null?

    python scripts/test_null_rung.py --latents cache/latents/n1_R1_cubes

The N1b registration makes rung N an invalidation gate: two halves of ONE
dataset must come out close to zero, or the metric cannot tell identical data
from two different laboratories and no other rung is readable.

That gate is only worth having if it works, and it runs at the END of a
two-hour pipeline. This exercises the same code path on a cache that already
exists, so a bug surfaces now rather than after the compute is spent.

Three checks, in increasing strictness:

  split-half     halves of one dataset, split BY EPISODE. Should be small but
                 not zero: different episodes really are different data.
  shuffled       the same latents split at random, ignoring episode boundaries.
                 Near-duplicate neighbouring frames land on both sides, so this
                 should be SMALLER than split-half. If it is not, the pooling or
                 the metric is wrong.
  identical      a half against itself. Must be essentially exactly zero. Any
                 meaningful value here means the metric is broken outright.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from measure_domain_gap import centroid_distance, frechet, mmd2  # noqa: E402


def pool(files):
    out = []
    for f in files:
        z = np.load(f).astype(np.float32)
        out.append(z.reshape(z.shape[0], -1, z.shape[-1]).mean(axis=1))
    return np.concatenate(out, axis=0)


def gap(a, b, n, dim, seed):
    rng = np.random.default_rng(seed)
    a = a[rng.choice(len(a), n, replace=False)]
    b = b[rng.choice(len(b), n, replace=False)]
    mu, sd = a.mean(0), a.std(0) + 1e-6
    an, bn = (a - mu) / sd, (b - mu) / sd
    w, v = np.linalg.eigh(np.cov(an, rowvar=False))
    basis = v[:, np.argsort(w)[::-1][:dim]]
    ap, bp = an @ basis, bn @ basis
    m, _ = mmd2(ap, bp)
    return centroid_distance(ap, bp), m, frechet(ap, bp)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", default="cache/latents/n1_R1_cubes")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = sorted(Path(args.latents).glob("episode_*.npy"))
    if len(files) < 4:
        print(f"{args.latents}: need >=4 episodes, found {len(files)}")
        return 1
    mid = len(files) // 2
    A, B = pool(files[:mid]), pool(files[mid:])
    allz = np.concatenate([A, B], axis=0)

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(allz))
    half = len(allz) // 2
    SA, SB = allz[idx[:half]], allz[idx[half:2 * half]]

    n = min(len(A), len(B), len(SA), len(SB))
    print(f"{args.latents}: {len(files)} episodes, n={n} per side, PCA {args.dim}\n")

    rows = [
        ("split-half (by episode)", gap(A, B, n, args.dim, args.seed)),
        ("shuffled (ignores episodes)", gap(SA, SB, n, args.dim, args.seed)),
        ("identical (half vs itself)", gap(A, A.copy(), n, args.dim, args.seed)),
    ]
    print(f"{'condition':32s} {'centroid':>10} {'MMD^2':>10} {'Frechet':>10}")
    print("-" * 66)
    for name, (c, m, f) in rows:
        print(f"{name:32s} {c:>10.4f} {m:>10.5f} {f:>10.4f}")

    sh_f, id_f = rows[1][1][2], rows[2][1][2]
    sp_f = rows[0][1][2]

    print()
    ok = True
    if abs(id_f) > 1e-3:
        print(f"  BROKEN: identical data gives Frechet {id_f:.4f}, not ~0.")
        ok = False
    else:
        print(f"  identical -> {id_f:.2e}. The metric zeroes on identical input.")

    if sh_f >= sp_f:
        print(f"  SUSPECT: shuffled ({sh_f:.2f}) >= split-half ({sp_f:.2f}).")
        print("    Shuffling puts neighbouring frames on both sides, so it should")
        print("    be the EASIER case. Check pooling and the episode split.")
        ok = False
    else:
        print(f"  shuffled {sh_f:.2f} < split-half {sp_f:.2f}, as expected:")
        print("    episode boundaries carry real structure, frame adjacency leaks.")

    print(f"\n  Rung N will report roughly {sp_f:.1f} on this dataset.")
    print("  It is the floor every other rung is read against, and it is NOT zero:")
    print("  two halves of one recording session are genuinely different data.")
    print(f"\n{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
