#!/usr/bin/env python3
"""Is the effect of one action even visible in the latent?

    python scripts/probe_action_signal.py --task pickplace

E3 produced a world model that ignores its actions, and conditioning on
displacements rather than absolute targets did not fix it. Before changing the
model again, measure whether the signal it is being asked to learn is present in
the data at all.

Two questions, both answered with linear probes rather than trained models,
because a linear probe is a *lower bound*: if ridge regression can recover the
action from the latent change, a transformer certainly can, and the failure is
the model's. If a linear probe cannot, the information is not there and no
architecture will find it.

  1. **Forward:** how much does the latent move per step, against how much it
     moves over longer intervals? If a single step is lost in the noise floor,
     the target contains no action effect to learn.
  2. **Inverse:** can the executed action be recovered from (z_t, z_t+k)? This
     is the inverse-dynamics probe, and it is the cleanest possible test of
     whether action information survives into the representation.

Reported per interval k, because the plausible failure is temporal: at 25 Hz
with tubelet 2, one latent step spans 80 ms, in which the arm moves a couple of
millimetres — plausibly below what a 4x4 pooled grid of a 224 px frame can
resolve.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jetspace.data.episode import EpisodeDataset  # noqa: E402


def ridge_r2(X: np.ndarray, Y: np.ndarray, lam: float = 1.0, folds: int = 4) -> float:
    """Cross-validated R^2 of a ridge fit Y ~ X. Chance is 0."""
    n = len(X)
    if n < folds * 4:
        return float("nan")
    idx = np.random.default_rng(0).permutation(n)
    X, Y = X[idx], Y[idx]
    scores = []
    for f in range(folds):
        va = slice(f * n // folds, (f + 1) * n // folds)
        mask = np.ones(n, bool)
        mask[va] = False
        Xtr, Ytr, Xva, Yva = X[mask], Y[mask], X[va], Y[va]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
        ym = Ytr.mean(0)
        w = np.linalg.solve(
            Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ (Ytr - ym)
        )
        pred = Xva @ w + ym
        ss_res = ((Yva - pred) ** 2).sum()
        ss_tot = ((Yva - Ytr.mean(0)) ** 2).sum()
        scores.append(1 - ss_res / max(ss_tot, 1e-12))
    return float(np.mean(scores))


def main() -> int:
    ap = argparse.ArgumentParser()
    # No `choices` here. These scripts take an arbitrary name plus explicit
    # --data/--latents paths, and a fixed choice list has already blocked
    # real-robot data once; E6 adds e6_* arms that would hit it again.
    ap.add_argument("--task", default="pickplace")
    ap.add_argument("--data", default=None)
    ap.add_argument("--latents", default=None)
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--intervals", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--pca-dim", type=int, default=128,
                    help="reduce latents before probing, so the probe is not just memorising")
    args = ap.parse_args()

    data = Path(args.data or f"data/episodes/{args.task}")
    lat_dir = Path(args.latents or f"cache/latents/{args.task}")
    fpl = json.loads((lat_dir / "info.json").read_text())["frames_per_latent"]
    ds = EpisodeDataset(data)

    eps = []
    for i in range(len(ds)):
        f = lat_dir / f"episode_{ds.records[i]['index']:06d}.npy"
        if not f.exists():
            continue
        z = np.load(f).astype(np.float32)
        z = z.reshape(z.shape[0], -1)                       # flatten tokens
        ep = ds[i]
        raw_a = ep["action"].astype(np.float32)
        qpos = ep["proprio"].astype(np.float32)[:, : raw_a.shape[1]]
        delta = raw_a - qpos
        usable = (len(delta) // fpl) * fpl
        a = delta[:usable].reshape(-1, fpl, delta.shape[1]).sum(axis=1)[: len(z)]
        if len(z) > max(args.intervals) + 1:
            eps.append((z, a))
        if len(eps) >= args.episodes:
            break
    if len(eps) < 5:
        print(f"Only {len(eps)} usable episodes; need more or shorter intervals.")
        return 1

    allz = np.concatenate([z for z, _ in eps])
    print(f"{args.task}: {len(eps)} episodes, {len(allz)} latents of dim {allz.shape[1]}")

    # PCA once, on the pooled latents, so probes operate in a compact space.
    mu = allz.mean(0)
    U, S, Vt = np.linalg.svd(allz[:: max(1, len(allz) // 3000)] - mu, full_matrices=False)
    basis = Vt[: args.pca_dim].T
    var_kept = (S[: args.pca_dim] ** 2).sum() / (S**2).sum()
    print(f"PCA to {args.pca_dim} dims keeps {var_kept:.1%} of variance\n")

    step_norm = np.mean([
        np.linalg.norm(np.diff(z, axis=0), axis=1).mean() for z, _ in eps
    ])
    print(f"{'k':>3} {'|z_t+k - z_t|':>15} {'vs 1-step':>10} {'inverse-dynamics R^2':>22}")
    print("-" * 56)

    results = {}
    for k in args.intervals:
        pairs_x, pairs_y, dists = [], [], []
        for z, a in eps:
            zp = (z - mu) @ basis
            for t in range(0, len(z) - k, max(1, k)):
                pairs_x.append(np.concatenate([zp[t], zp[t + k] - zp[t]]))
                pairs_y.append(a[t : t + k].sum(axis=0))
                dists.append(np.linalg.norm(z[t + k] - z[t]))
        X, Y = np.stack(pairs_x), np.stack(pairs_y)
        r2 = ridge_r2(X, Y)
        d = float(np.mean(dists))
        results[k] = {"latent_move": d, "inverse_r2": r2, "n": len(X)}
        print(f"{k:>3} {d:>15.3f} {d/max(step_norm,1e-9):>9.2f}x {r2:>21.3f}")

    print("\n" + "=" * 56)
    best_k = max(results, key=lambda k: results[k]["inverse_r2"])
    best = results[best_k]["inverse_r2"]
    print(f"best inverse-dynamics R^2 = {best:.3f} at interval k={best_k}")
    if best < 0.1:
        print("VERDICT: action effects are NOT linearly recoverable at any interval.")
        print("  The latent does not register what the arm was commanded to do.")
    elif best < 0.4:
        print("VERDICT: weak action signal. Recoverable but faint.")
    else:
        print("VERDICT: action signal IS present. The world model's blindness is")
        print("  a modelling failure, not a data limitation.")
    print("=" * 56)

    out = Path(f"cache/probe_action_{args.task}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"task": args.task, "one_step_norm": float(step_norm),
                               "results": {str(k): v for k, v in results.items()}}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
