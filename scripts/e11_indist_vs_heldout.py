#!/usr/bin/env python3
"""Separate "good features" from "viewpoint-robust features".

    python scripts/e11_indist_vs_heldout.py push

VC-1 scores 1.147 on held-out viewpoints -- last of nine, worse than predicting
the mean action -- yet a ridge probe on its features at the REFERENCE pose
recovers actions at R2 = 0.55. Both can be true: features that carry plenty of
action signal in-distribution can still fall apart when the camera moves.

That distinction is invisible in the E11 table, which reports only the held-out
number. This measures both for every arm from the already-cached latents:

  in-distribution   ridge probe trained and tested at the reference pose,
                    split by episode
  held-out          the E11 multiview number, at 8 poses never trained on

An encoder that is strong in-distribution and weak held-out is viewpoint-
brittle, which is a different and more interesting failure than being weak
everywhere. Burns et al. (CoRL 2024) found manipulation-tuned representations
do not reliably transfer robustness, and this is the measurement that would
show it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ARMS = [
    ("vjepa2", "r1", "2025-06", "video SSL"),
    ("dinov2", "dino", "2023-04", "image SSL"),
    ("siglip2", "siglip2", "2025-02", "image-text"),
    ("dinov3", "dinov3", "2025-08", "image SSL"),
    ("vit-in1k", "vitin1k", "2020-10", "supervised"),
    ("aimv2", "aimv2", "2024-11", "autoregressive"),
    ("vc1", "vc1", "2023-06", "robot MAE"),
    ("clip", "clip", "2021-01", "image-text"),
    ("random", "r1cnn", "none", "none"),
]


def load(prefix: str, task: str, pose: str = "r1_ref"):
    d = Path("cache/latents") / f"{prefix}_{task}__{pose}"
    files = sorted(d.glob("episode_*.npy"))
    if not files:
        return None
    return [np.load(f).astype(np.float32).reshape(np.load(f).shape[0], -1)
            for f in files]


def actions(task: str):
    files = sorted((Path("data/episodes") / f"r1_{task}").glob("episode_*.npz"))
    return [np.load(f)["action"].astype(np.float32) for f in files]


def ridge_r2(Xtr, Ytr, Xte, Yte, lam: float = 10.0) -> float:
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ay, asd = Ytr.mean(0), Ytr.std(0) + 1e-6
    Ytr, Yte = (Ytr - ay) / asd, (Yte - ay) / asd
    d = Xtr.shape[1]
    if d <= len(Xtr):
        W = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(d), Xtr.T @ Ytr)
    else:
        K = Xtr @ Xtr.T + lam * np.eye(len(Xtr))
        W = Xtr.T @ np.linalg.solve(K, Ytr)
    pred = Xte @ W
    ss_res = float(((Yte - pred) ** 2).sum())
    ss_tot = float(((Yte - Yte.mean(0)) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def main() -> int:
    task = sys.argv[1] if len(sys.argv) > 1 else "push"
    acts = actions(task)

    held = {}
    for f in Path("cache").glob("e11_*.json"):
        b = json.loads(f.read_text())
        if b.get("multiview"):
            held[f.stem[4:]] = float(np.mean(b["multiview"]))

    print(f"{'encoder':10s} {'released':9s} {'kind':15s} "
          f"{'in-dist R2':>11} {'held-out':>10}  {'drop'}")
    print("-" * 72)
    rows = []
    for name, prefix, rel, kind in ARMS:
        feats = load(prefix, task)
        if feats is None:
            continue
        # Split BY EPISODE: frames within an episode are strongly correlated,
        # so a frame-wise split would leak near-duplicates across the boundary.
        cut = max(1, int(0.8 * len(feats)))
        pairs = []
        for f, a in zip(feats, acts):
            n = min(len(f), len(a) // 2)
            if n >= 4:
                pairs.append((f[:n], a[: 2 * n : 2]))
        if len(pairs) < 4:
            continue
        Xtr = np.concatenate([p[0] for p in pairs[:cut]])
        Ytr = np.concatenate([p[1] for p in pairs[:cut]])
        Xte = np.concatenate([p[0] for p in pairs[cut:]])
        Yte = np.concatenate([p[1] for p in pairs[cut:]])
        if len(Xte) < 10:
            continue
        r2 = ridge_r2(Xtr, Ytr, Xte, Yte)
        h = held.get(name)
        rows.append((name, rel, kind, r2, h))
        hs = f"{h:.3f}" if h is not None else "  --  "
        print(f"{name:10s} {rel:9s} {kind:15s} {r2:>11.3f} {hs:>10}")

    usable = [r for r in rows if r[4] is not None]
    if len(usable) < 3:
        return 0

    print()
    print("Ranked by in-distribution strength (does the feature carry the "
          "action at all):")
    for name, _, _, r2, h in sorted(usable, key=lambda r: -r[3]):
        print(f"  {name:10s} in-dist {r2:+.3f}   held-out {h:.3f}")

    r2s = np.array([r[3] for r in usable])
    hs = np.array([r[4] for r in usable])
    ra = np.argsort(np.argsort(-r2s)).astype(float)
    rb = np.argsort(np.argsort(hs)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    rho = float((ra * rb).sum() / den) if den > 1e-9 else float("nan")
    print(f"\n  Spearman(in-distribution rank, held-out rank) = {rho:+.3f}")
    if rho < 0.5:
        print("  In-distribution strength does NOT predict held-out ranking.")
        print("  Being a good feature and being viewpoint-robust are different")
        print("  properties, and the E11 table measures only the second.")
    else:
        print("  The two rankings largely agree: held-out performance mostly")
        print("  reflects raw feature quality rather than viewpoint robustness.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
