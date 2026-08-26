#!/usr/bin/env python3
"""H1: is the gap→degradation relationship predictive, reproducible, and real?

    python scripts/harden_gap_prediction.py --task push --seeds 0 1 2

Implements docs/prereg-h1.md. The published result -- rho = -0.921 between
latent gap and direction-cosine degradation -- is one task, one seed, a
post-hoc correlation over 22 non-independent poses, entirely in simulation.
Literature review found the claim shape partially taken (arXiv:2604.13645
correlates Wasserstein against policy success at 0.6-0.8), so what has to be
defended is not the correlation but the three things it lacks:

  H1a  PREDICTION      fit on half the poses, predict the held-out half
  H1b  reproducibility three world-model seeds
  H1e  honest interval bootstrap by pose FAMILY, not by pose

H1a is the primary claim. A correlation says two quantities move together in a
set already measured; a prediction says a degradation can be estimated before
it is measured. That is the difference between an observation and an
instrument, and it is the one thing none of the prior work attempts.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


#: Poses share episodes by construction, so they are not independent draws.
#: Resampling individual poses overstates the effective sample size; these are
#: the blocks H1e resamples instead.
FAMILY = {"az": "azimuth", "el": "elevation", "d": "distance", "a": "off_axis"}


def family_of(pose: str) -> str:
    body = pose[len("r1_"):]
    for prefix in ("az", "el"):
        if body.startswith(prefix):
            return FAMILY[prefix]
    if re.match(r"^a\d{2}e\d{2}$", body):
        return FAMILY["a"]
    if body.startswith("d"):
        return FAMILY["d"]
    return "other"


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 1e-12 else float("nan")


def oos_prediction(gap: np.ndarray, deg: np.ndarray, repeats: int, seed: int):
    """H1a: fit on half the poses, predict the other half.

    A monotone relationship is fitted in RANK space rather than assumed linear
    -- the curve saturates, and forcing a straight line through it would report
    a prediction failure that belongs to the model, not to the relationship.
    """
    rng = np.random.default_rng(seed)
    n = len(gap)
    r2s, maes = [], []
    for _ in range(repeats):
        idx = rng.permutation(n)
        tr, te = idx[: n // 2], idx[n // 2:]
        if len(tr) < 4 or len(te) < 4:
            continue
        # Isotonic-flavoured fit: monotone interpolation over the training
        # points, extrapolating flat beyond their range.
        order = np.argsort(gap[tr])
        gx, gy = gap[tr][order], deg[tr][order]
        gx_u, inv = np.unique(gx, return_inverse=True)
        gy_u = np.array([gy[inv == i].mean() for i in range(len(gx_u))])
        pred = np.interp(gap[te], gx_u, gy_u)
        resid = deg[te] - pred
        ss_res = float((resid ** 2).sum())
        ss_tot = float(((deg[te] - deg[te].mean()) ** 2).sum())
        r2s.append(1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan)
        maes.append(float(np.median(np.abs(resid))))
    return np.array(r2s, dtype=float), np.array(maes, dtype=float)


def family_bootstrap(gap, deg, fams, n_boot: int, seed: int):
    """H1e: resample whole pose families, not individual poses."""
    rng = np.random.default_rng(seed)
    uniq = sorted(set(fams))
    by_fam = {f: np.where(np.array(fams) == f)[0] for f in uniq}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_fam[f] for f in pick])
        if len(set(idx.tolist())) < 4:
            continue
        out.append(spearman(gap[idx], deg[idx]))
    return np.array(out, dtype=float)


def collect(task: str, seed: int) -> list[dict]:
    """Gap and degradation per pose, for one task and one world-model seed."""
    ruler = Path(f"cache/r1_ruler_{task}.json")
    if not ruler.exists():
        ruler = Path("cache/r1_ruler.json")
    if not ruler.exists():
        return []
    gaps = {r["pose"]: r for r in json.loads(ruler.read_text())["poses"]}

    rows = []
    for pose, g in gaps.items():
        c = Path(f"cache/conservatism_h1_{task}_s{seed}_{pose}.json")
        if not c.exists():
            continue
        d = json.loads(c.read_text())
        rows.append({
            "pose": pose, "family": family_of(pose),
            "frechet": g["frechet"], "mmd2": g.get("mmd2"),
            "centroid": g.get("centroid"),
            "cosine": d["mean_cosine"], "ratio": d["mean_ratio"],
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="push")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print("=" * 74)
    print(f"H1 -- hardening gap->degradation, task {args.task}")
    print("=" * 74)

    per_seed = {}
    for s in args.seeds:
        rows = collect(args.task, s)
        if len(rows) < 8:
            print(f"  seed {s}: only {len(rows)} poses scored -- skipping")
            continue
        per_seed[s] = rows

    if not per_seed:
        print("no scored poses. Run scripts/run_h1.sh first.")
        return 1

    results = {}
    print(f"\n{'seed':>5} {'poses':>6} {'rho_cos':>9} {'rho_ratio':>10} "
          f"{'OOS R2':>9} {'OOS MAE':>9}")
    print("-" * 56)
    for s, rows in per_seed.items():
        gap = np.array([r["frechet"] for r in rows])
        cos = np.array([r["cosine"] for r in rows])
        rat = np.array([r["ratio"] for r in rows])
        rho_c = spearman(gap, cos)
        rho_r = spearman(gap, rat)
        r2s, maes = oos_prediction(gap, cos, args.repeats, s)
        results[s] = {
            "n": len(rows), "rho_cosine": rho_c, "rho_ratio": rho_r,
            "oos_r2_median": float(np.nanmedian(r2s)),
            "oos_mae_median": float(np.nanmedian(maes)),
            "rows": rows,
        }
        print(f"{s:>5} {len(rows):>6} {rho_c:>9.3f} {rho_r:>10.3f} "
              f"{np.nanmedian(r2s):>9.3f} {np.nanmedian(maes):>9.4f}")

    rhos = np.array([r["rho_cosine"] for r in results.values()])
    r2s = np.array([r["oos_r2_median"] for r in results.values()])
    maes = np.array([r["oos_mae_median"] for r in results.values()])

    # ---- pooled, with the family-level interval ------------------------
    pooled = [r for rows in per_seed.values() for r in rows]
    gap = np.array([r["frechet"] for r in pooled])
    cos = np.array([r["cosine"] for r in pooled])
    fams = [r["family"] for r in pooled]
    boot = family_bootstrap(gap, cos, fams, args.boot, 0)
    lo, hi = (np.percentile(boot, [2.5, 97.5]) if len(boot) else (np.nan, np.nan))

    print("\n" + "=" * 74)
    print("REGISTERED PREDICTIONS")
    print("=" * 74)

    h1a = np.nanmedian(r2s) >= 0.5 and np.nanmedian(maes) <= 0.05
    print(f"H1a prediction (R2>=0.5, MAE<=0.05): {'HOLDS' if h1a else 'FAILS'}"
          f"   R2 {np.nanmedian(r2s):.3f}, MAE {np.nanmedian(maes):.4f}")
    if not h1a:
        print("     A correlation without out-of-sample prediction is an")
        print("     observation, not an instrument. This is the primary claim.")

    spread = float(rhos.max() - rhos.min()) if len(rhos) > 1 else 0.0
    h1b = bool((rhos <= -0.6).all() and spread <= 0.15)
    print(f"H1b every seed rho<=-0.6, spread<=0.15: {'HOLDS' if h1b else 'FAILS'}"
          f"   rhos {np.round(rhos, 3).tolist()}, spread {spread:.3f}")

    h1e = bool(hi <= -0.6)
    print(f"H1e family-bootstrap 95% excludes -0.6: {'HOLDS' if h1e else 'FAILS'}"
          f"   [{lo:.3f}, {hi:.3f}]")
    if not h1e:
        print("     Resampling whole pose families rather than poses is the")
        print("     honest interval; the published one overstated n.")

    # ---- H1f: is it Frechet specifically? ------------------------------
    print("\nH1f metric ablation (pooled)")
    for key in ("frechet", "mmd2", "centroid"):
        v = [r.get(key) for r in pooled]
        if any(x is None for x in v):
            print(f"  {key:9s} not recorded in the ruler")
            continue
        print(f"  {key:9s} rho {spearman(np.array(v, dtype=float), cos):+.3f}")

    out = Path(args.out or f"cache/h1_{args.task}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "task": args.task, "per_seed": results,
        "family_ci95": [float(lo), float(hi)],
        "verdicts": {"H1a": bool(h1a), "H1b": h1b, "H1e": h1e},
    }, indent=2, default=float))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
