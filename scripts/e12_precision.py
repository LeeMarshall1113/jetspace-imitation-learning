#!/usr/bin/env python3
"""Where is E12's uncertainty coming from, and can it be bought down?

    python scripts/e12_precision.py push lighting

Registered in docs/prereg-e12-stage3.md S6a (A1, A2), written before this ran.

Every cell in the main analysis is one 80/20 split: ten episodes, eight fitted,
two held out. Each probe R^2 and each held-out MSE therefore rests on a single
split of 300 samples, and that noise lands in the encoder ranking that rho is
computed from. Two questions follow.

A1  Does averaging over ten leave-one-episode-out folds, instead of one split,
    narrow the interval on rho? Same latents, same actions, no new compute.

A2  Is there encoder-level signal above episode noise at all? If held-out error
    varies more between episodes than between encoders, then ranking encoders
    is mostly ranking which episodes landed in the test fold, and buying more
    encoders cannot fix that -- only more episodes can.

A2 is the one that decides what to run next, which is why it is registered.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

LAT = Path("cache/latents")
BOOT = 8000
RNG = np.random.default_rng(0)

ENCODERS = [
    ("vjepa2", "r1"), ("dinov3", "dinov3"), ("siglip2", "siglip2"),
    ("aimv2", "aimv2"), ("dinov2", "dino"), ("clip", "clip"),
    ("vit-in1k", "vitin1k"), ("vc1", "vc1"), ("random", "r1cnn"),
    ("dinov2-large", "dinov2l"), ("dinov3-large", "dinov3l"),
    ("siglip1", "siglip1"), ("vit-large", "vitlarge"),
    ("clip-large", "cliplarge"), ("vc1-large", "vc1large"),
]
LEVELS = {
    "lighting": ["lighting_0p3", "lighting_0p45", "lighting_0p55",
                 "lighting_0p62"],
    "texture": ["texture_0p06", "texture_0p1", "texture_0p16", "texture_0p24"],
    "clutter": ["clutter_1", "clutter_2", "clutter_3", "clutter_4"],
    "noise": ["noise_4p0", "noise_10p0", "noise_20p0", "noise_35p0"],
    "defocus": ["defocus_1", "defocus_2", "defocus_4", "defocus_7"],
    "compress": ["compress_4", "compress_8", "compress_14", "compress_22"],
    "exposure": ["exposure_0p65", "exposure_0p80", "exposure_1p25",
                 "exposure_1p55"],
    "lowres": ["lowres_2", "lowres_3", "lowres_5", "lowres_8"],
}
IMAGE_AXES = ("noise", "defocus", "compress", "exposure", "lowres")


def spearman(a, b) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 1e-12 else float("nan")


def ridge(Xtr, Ytr, Xte, lam: float = 10.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    d = Xtr.shape[1]
    if d <= len(Xtr):
        W = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(d), Xtr.T @ Ytr)
    else:
        K = Xtr @ Xtr.T + lam * np.eye(len(Xtr))
        W = Xtr.T @ np.linalg.solve(K, Ytr)
    return Xte @ W


def latents(prefix, tag, task):
    d = LAT / f"{prefix}_e12_{task}__{tag}"
    fs = sorted(d.glob("episode_*.npy"))
    if not fs:
        return None
    return [np.load(f).astype(np.float32).reshape(np.load(f).shape[0], -1)
            for f in fs]


def acts(tag, task):
    d = Path("data/episodes") / f"e12_{task}__{tag}"
    if not d.is_dir() and tag.startswith(IMAGE_AXES):
        d = Path("data/episodes") / f"e12_{task}__ref"
    return [np.load(f)["action"].astype(np.float32)
            for f in sorted(d.glob("episode_*.npz"))]


def pair(feats, actions):
    X, Y = [], []
    for f, a in zip(feats, actions):
        n = min(len(f), len(a) // 2)
        if n >= 4:
            X.append(f[:n])
            Y.append(a[: 2 * n : 2])
    return X, Y


def per_episode(prefix, axis, task):
    """Held-out error for EVERY (fold, level) rather than one split.

    Returns (probe_cv, rel_cv, per_ep) where per_ep[i][j] is the relative
    degradation at level j when episode i was the held-out fold. Keeping the
    grid un-averaged is what makes the variance decomposition possible.
    """
    ref_f = latents(prefix, "ref", task)
    if ref_f is None:
        return None
    Xs, Ys = pair(ref_f, acts("ref", task))
    n_ep = len(Xs)
    if n_ep < 5:
        return None
    levels = [lv for lv in LEVELS[axis] if latents(prefix, lv, task)]
    if len(levels) != len(LEVELS[axis]):
        return None

    lvl_f = {lv: pair(latents(prefix, lv, task), acts(lv, task))
             for lv in levels}

    probes, refs, grid = [], [], []
    for i in range(n_ep):
        tr = [j for j in range(n_ep) if j != i]
        Xtr = np.concatenate([Xs[j] for j in tr])
        Ytr = np.concatenate([Ys[j] for j in tr])
        ay, asd = Ytr.mean(0), Ytr.std(0) + 1e-6
        live = Ytr.std(0) > 1e-6
        Ytr_n = ((Ytr - ay) / asd)[:, live]

        Xte, Yte = Xs[i], Ys[i]
        Yte_n = ((Yte - ay) / asd)[:, live]
        pred = ridge(Xtr, Ytr_n, Xte)
        ss_res = float(((Yte_n - pred) ** 2).sum())
        ss_tot = float(((Yte_n - Yte_n.mean(0)) ** 2).sum())
        probes.append(1.0 - ss_res / max(ss_tot, 1e-12))
        ref_mse = float(((Yte_n - pred) ** 2).mean())
        refs.append(ref_mse)

        row = []
        for lv in levels:
            Xl, Yl = lvl_f[lv]
            if i >= len(Xl):
                row.append(np.nan)
                continue
            Yn = ((Yl[i] - ay) / asd)[:, live]
            m = float(((Yn - ridge(Xtr, Ytr_n, Xl[i])) ** 2).mean())
            row.append(m / max(ref_mse, 1e-9))
        grid.append(row)

    return (float(np.mean(probes)),
            float(np.nanmean(grid)),
            np.array(grid, dtype=float))


def main() -> int:
    task = sys.argv[1] if len(sys.argv) > 1 else "push"
    axis = sys.argv[2] if len(sys.argv) > 2 else "lighting"
    print(f"task={task}  axis={axis}   (registered: prereg-e12-stage3 S6a)\n")

    rows, grids = {}, {}
    for name, prefix in ENCODERS:
        r = per_episode(prefix, axis, task)
        if r is None:
            continue
        rows[name] = (r[0], r[1])
        grids[name] = r[2]
    if len(rows) < 5:
        print(f"only {len(rows)} encoders with full coverage -- nothing to do")
        return 1

    names = sorted(rows)
    probe = np.array([rows[n][0] for n in names])
    rel = np.array([rows[n][1] for n in names])

    print("A1  leave-one-episode-out CV (10 folds) vs the registered 80/20\n")
    print(f"  {'encoder':14s} {'probe(CV)':>10s} {'rel degr(CV)':>13s}")
    for n in names:
        print(f"  {n:14s} {rows[n][0]:10.3f} {rows[n][1]:13.3f}")

    rho_cv = spearman(-probe, rel)
    boot = np.empty(BOOT)
    k = len(names)
    for i in range(BOOT):
        idx = RNG.integers(0, k, k)
        boot[i] = spearman(-probe[idx], rel[idx])
    boot = boot[np.isfinite(boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\n  rho(probe, relative degradation) under CV = {rho_cv:+.3f}")
    print(f"  95% CI [{lo:+.3f}, {hi:+.3f}]   width {hi - lo:.3f}")

    ref = json.loads(Path(f"cache/e12_{task}.json").read_text()) \
        if Path(f"cache/e12_{task}.json").exists() else {}
    if axis in ref and ref[axis].get("rho") is not None:
        print(f"  registered 80/20 rho was {ref[axis]['rho']:+.3f}"
              f"   sign {'AGREES' if ref[axis]['rho'] * rho_cv > 0 else 'DIFFERS'}")

    print("\nA2  variance decomposition of relative degradation\n")
    # grid[enc][episode, level]
    M = np.stack([grids[n] for n in names])          # (enc, ep, level)
    M = M[:, :, ~np.all(np.isnan(M), axis=(0, 1))]
    grand = np.nanmean(M)
    v_enc = float(np.nanvar(np.nanmean(M, axis=(1, 2))))
    v_ep = float(np.nanvar(np.nanmean(M, axis=(0, 2))))
    v_lvl = float(np.nanvar(np.nanmean(M, axis=(0, 1))))
    resid = float(np.nanvar(M)) - (v_enc + v_ep + v_lvl)
    tot = max(v_enc + v_ep + v_lvl + max(resid, 0.0), 1e-12)
    print(f"  grand mean relative degradation {grand:.3f}")
    print(f"  {'component':12s} {'variance':>12s} {'share':>8s}")
    for lab, v in (("encoder", v_enc), ("episode", v_ep),
                   ("level", v_lvl), ("residual", max(resid, 0.0))):
        print(f"  {lab:12s} {v:12.5f} {100 * v / tot:7.1f}%")

    print()
    if v_enc > v_ep:
        print("  Encoder variance exceeds episode variance: encoders are")
        print("  genuinely separable here, and buying more encoders is the")
        print("  efficient way to narrow rho.")
    else:
        print("  Episode variance exceeds encoder variance: much of the")
        print("  ranking reflects which episodes landed in the fold, not which")
        print("  encoder was used. More encoders will NOT fix that -- more")
        print("  episodes per condition is the only route.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
