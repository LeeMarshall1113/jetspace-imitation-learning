#!/usr/bin/env python3
"""How far apart do two datasets land in the same frozen latent space?

    python scripts/measure_domain_gap.py --reference cache/latents/n1_R1_cubes \\
                                         --other     cache/latents/push_s1n60

Implements the N1 measurement specified in docs/prereg-n1.md. Three metrics,
always reported together, because each is blind to something the others catch:

  centroid   mean shift only. Comparable with the existing precedent (Domain
             Invariance Score, arXiv:2501.16389) and blind to spread -- domain
             randomisation can recentre a distribution without covering the
             target and score perfectly here.
  MMD        nonparametric two-sample distance (Gretton et al., JMLR 2012).
             Sees any distributional difference, including spread and shape.
             Reported with a permutation p-value, so "is this gap real at all"
             has an answer rather than a vibe.
  Frechet    mean AND covariance (Heusel et al., NeurIPS 2017). The FID
             construction, applied to latents rather than InceptionV3 features.

**Disagreement between them is a result.** If centroid says "close" and MMD
says "far", the distributions share a centre and differ in shape, which is
worth more than either number alone. Nothing here picks a favourite.

---

Two implementation choices were made after the pre-registration was committed
and before any result was seen. Both are recorded here rather than folded in
silently:

1. **Spatial mean-pooling to 1024 dimensions.** Latents are 4x4x1024 = 16384-d.
   Frechet needs a covariance estimate, and covariance in 16384 dimensions from
   a few thousand samples is not an estimate.

2. **PCA to `--dim` (default 64), fit on the REFERENCE side only.** Same reason:
   Frechet wants d << n. Fitting on the reference and applying to both follows
   the pre-registered normalisation rule -- simulation never gets to define the
   coordinate system it is being measured in.

The centroid and MMD numbers are reported at BOTH full 1024-d and reduced
dimensionality, so it is visible whether the reduction changed anything. Given
that a PCA step is what turned a harmless artifact into a load-bearing one
earlier in this project (ledger L7), that check is not optional.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_pooled(d: Path, cap: int | None, seed: int) -> np.ndarray:
    """Every latent from a cache, spatially mean-pooled to (N, hidden)."""
    files = sorted(d.glob("episode_*.npy"))
    if not files:
        raise SystemExit(f"no latents in {d}")
    out = []
    for f in files:
        z = np.load(f).astype(np.float32)
        z = z.reshape(z.shape[0], -1, z.shape[-1])   # (T, grid, hidden)
        out.append(z.mean(axis=1))                    # (T, hidden)
    x = np.concatenate(out, axis=0)
    if cap and len(x) > cap:
        rng = np.random.default_rng(seed)
        x = x[rng.choice(len(x), cap, replace=False)]
    return x


def centroid_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.mean(0) - b.mean(0)))


def _sq_dists(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.maximum(
        (x * x).sum(1)[:, None] + (y * y).sum(1)[None, :] - 2.0 * x @ y.T, 0.0
    )


def mmd2(a: np.ndarray, b: np.ndarray, sigma: float | None = None) -> tuple[float, float]:
    """Unbiased MMD^2 with an RBF kernel; returns (mmd2, sigma)."""
    aa, bb, ab = _sq_dists(a, a), _sq_dists(b, b), _sq_dists(a, b)
    if sigma is None:
        # Median heuristic over the pooled pairwise distances.
        pooled = np.concatenate([aa[np.triu_indices_from(aa, 1)],
                                 bb[np.triu_indices_from(bb, 1)],
                                 ab.ravel()])
        med = np.median(pooled)
        sigma = float(np.sqrt(max(med, 1e-12) / 2.0))
    g = 1.0 / (2.0 * sigma * sigma)
    m, n = len(a), len(b)
    kaa, kbb, kab = np.exp(-g * aa), np.exp(-g * bb), np.exp(-g * ab)
    np.fill_diagonal(kaa, 0.0)
    np.fill_diagonal(kbb, 0.0)
    val = (kaa.sum() / (m * (m - 1)) + kbb.sum() / (n * (n - 1))
           - 2.0 * kab.mean())
    return float(val), float(sigma)


def mmd_pvalue(a: np.ndarray, b: np.ndarray, observed: float, sigma: float,
               n_perm: int, seed: int) -> float:
    """Permutation test: how often does shuffled labelling beat the observed gap?"""
    rng = np.random.default_rng(seed)
    pooled = np.concatenate([a, b], axis=0)
    m = len(a)
    hits = 0
    for _ in range(n_perm):
        idx = rng.permutation(len(pooled))
        v, _ = mmd2(pooled[idx[:m]], pooled[idx[m:]], sigma=sigma)
        hits += v >= observed
    return (hits + 1) / (n_perm + 1)


def frechet(a: np.ndarray, b: np.ndarray) -> float:
    """||mu_a - mu_b||^2 + Tr(Ca + Cb - 2 (Ca Cb)^1/2)."""
    mu_a, mu_b = a.mean(0), b.mean(0)
    ca = np.cov(a, rowvar=False)
    cb = np.cov(b, rowvar=False)
    diff = mu_a - mu_b

    # Tr((Ca Cb)^1/2) via the symmetric form Ca^1/2 Cb Ca^1/2, which is PSD and
    # avoids the complex eigenvalues sqrtm returns on a non-symmetric product.
    wa, va = np.linalg.eigh(ca)
    wa = np.clip(wa, 0.0, None)
    ca_half = (va * np.sqrt(wa)) @ va.T
    mid = ca_half @ cb @ ca_half
    wm = np.clip(np.linalg.eigvalsh(mid), 0.0, None)
    tr_covmean = float(np.sqrt(wm).sum())

    return float(diff @ diff + np.trace(ca) + np.trace(cb) - 2.0 * tr_covmean)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True,
                    help="the side whose statistics define the space (always the real one)")
    ap.add_argument("--other", required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--dim", type=int, default=64, help="PCA components for Frechet")
    ap.add_argument("--cap", type=int, default=2500, help="max latents per side")
    ap.add_argument("--permutations", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ref_dir, oth_dir = Path(args.reference), Path(args.other)
    label = args.label or f"{ref_dir.name} vs {oth_dir.name}"

    a = load_pooled(ref_dir, args.cap, args.seed)
    b = load_pooled(oth_dir, args.cap, args.seed)

    # Equal sample counts. MMD and Frechet are both biased by sample size, and
    # comparing a 2500-sample gap against a 900-sample gap would rank the
    # datasets by how much data they happen to have.
    n = min(len(a), len(b))
    rng = np.random.default_rng(args.seed)
    if len(a) > n:
        a = a[rng.choice(len(a), n, replace=False)]
    if len(b) > n:
        b = b[rng.choice(len(b), n, replace=False)]

    print(f"{label}")
    print(f"  reference {ref_dir.name}   other {oth_dir.name}")
    print(f"  {n} latents per side, {a.shape[1]}-d pooled\n")

    # Standardise on the reference side only.
    mu, sd = a.mean(0), a.std(0) + 1e-6
    an, bn = (a - mu) / sd, (b - mu) / sd

    cen_full = centroid_distance(an, bn)
    mmd_full, sigma_full = mmd2(an, bn)

    # PCA, also fit on the reference side only.
    cov = np.cov(an, rowvar=False)
    w, v = np.linalg.eigh(cov)
    basis = v[:, np.argsort(w)[::-1][: args.dim]]
    ap_, bp_ = an @ basis, bn @ basis

    cen_pca = centroid_distance(ap_, bp_)
    mmd_pca, sigma_pca = mmd2(ap_, bp_)
    fre_pca = frechet(ap_, bp_)
    p = mmd_pvalue(ap_, bp_, mmd_pca, sigma_pca, args.permutations, args.seed)

    print(f"{'':16s} {'full 1024-d':>14} {f'PCA {args.dim}-d':>14}")
    print("-" * 46)
    print(f"{'centroid':16s} {cen_full:>14.4f} {cen_pca:>14.4f}")
    print(f"{'MMD^2':16s} {mmd_full:>14.5f} {mmd_pca:>14.5f}")
    print(f"{'Frechet':16s} {'-':>14} {fre_pca:>14.4f}")
    print(f"{'MMD p-value':16s} {'-':>14} {p:>14.4f}")

    print()
    if p > 0.05:
        print("  The two samples are not distinguishable at p<0.05. Whatever gap")
        print("  the point estimates show is within permutation noise.")
    else:
        print(f"  Distributions differ (p={p:.4f}). The magnitude is only")
        print("  interpretable against the other rungs of the ladder.")

    if (cen_full > 0) and abs(cen_pca / max(cen_full, 1e-9) - 1.0) > 0.5:
        print("\n  NOTE: PCA changed the centroid distance by more than 50%.")
        print("  The reduction is not neutral here -- report both.")

    rec = {
        "label": label, "reference": str(ref_dir), "other": str(oth_dir),
        "n_per_side": int(n), "pca_dim": args.dim,
        "centroid_full": cen_full, "centroid_pca": cen_pca,
        "mmd2_full": mmd_full, "mmd2_pca": mmd_pca,
        "frechet_pca": fre_pca, "mmd_pvalue": p,
        "permutations": args.permutations, "seed": args.seed,
    }
    out = Path(args.out) if args.out else None
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=2))
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
