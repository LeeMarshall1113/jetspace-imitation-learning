#!/usr/bin/env python3
"""E12: does probe accuracy predict robustness, per nuisance axis?

    python scripts/e12_analyze.py push

Implements docs/prereg-e12.md. For every (encoder, axis) cell:

  probe R^2     ridge from frozen features to actions at the REFERENCE
                condition, split BY EPISODE -- what the field reports
  robustness    a head trained at the reference, evaluated at the held-out
                displaced conditions, in normalised action MSE -- what the
                field wants to know

Then correlates the two rankings per axis. E12a registers that the mean |rho|
across axes is <= 0.4; a high rho would mean probes DO predict robustness and
E11's viewpoint-only finding was axis-specific, which is reported as the result
rather than treated as a failure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ENCODERS = [
    ("vjepa2", "r1"), ("dinov3", "dinov3"), ("siglip2", "siglip2"),
    ("aimv2", "aimv2"), ("dinov2", "dino"), ("clip", "clip"),
    ("vit-in1k", "vitin1k"), ("vc1", "vc1"), ("random", "r1cnn"),
]
AXES = ["lighting", "texture", "clutter"]
FLOOR = 0.9          # prereg S3.1: a head above this cannot be compared
MIN_DEGRADE = 0.10   # prereg S3.2: an axis this weak cannot rank anything


def load(prefix: str, tag: str, task: str):
    d = Path("cache/latents") / f"{prefix}_e12_{task}__{tag}"
    files = sorted(d.glob("episode_*.npy"))
    if not files:
        return None
    return [np.load(f).astype(np.float32).reshape(np.load(f).shape[0], -1)
            for f in files]


def actions(tag: str, task: str):
    d = Path("data/episodes") / f"e12_{task}__{tag}"
    return [np.load(f)["action"].astype(np.float32)
            for f in sorted(d.glob("episode_*.npz"))]


def pair(feats, acts):
    X, Y = [], []
    for f, a in zip(feats, acts):
        n = min(len(f), len(a) // 2)
        if n >= 4:
            X.append(f[:n])
            Y.append(a[: 2 * n : 2])
    return X, Y


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


def spearman(a, b) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 1e-12 else float("nan")


def cell(prefix: str, axis: str, task: str, levels: list[str]):
    """(probe R^2 at reference, mean normalised MSE at held-out levels)."""
    ref_f = load(prefix, "ref", task)
    if ref_f is None:
        return None
    ra = actions("ref", task)
    Xs, Ys = pair(ref_f, ra)
    if len(Xs) < 5:
        return None

    cut = max(1, int(0.8 * len(Xs)))
    Xtr, Ytr = np.concatenate(Xs[:cut]), np.concatenate(Ys[:cut])
    Xte, Yte = np.concatenate(Xs[cut:]), np.concatenate(Ys[cut:])

    ay, asd = Ytr.mean(0), Ytr.std(0) + 1e-6
    live = Ytr.std(0) > 1e-6
    Ytr_n = ((Ytr - ay) / asd)[:, live]
    Yte_n = ((Yte - ay) / asd)[:, live]

    pred = ridge(Xtr, Ytr_n, Xte)
    ss_res = float(((Yte_n - pred) ** 2).sum())
    ss_tot = float(((Yte_n - Yte_n.mean(0)) ** 2).sum())
    probe = 1.0 - ss_res / max(ss_tot, 1e-12)

    # Reference-fit MSE, the denominator for degradation and the floor check.
    ref_mse = float(((Yte_n - pred) ** 2).mean())

    # Robustness: same ridge map, at the SAME held-out episodes.
    #
    # Episode seeds are shared across conditions by design, so episode i at a
    # displaced condition is the same trajectory as episode i at the reference.
    # Evaluating on all ten would therefore score 80% of the displaced set on
    # trajectories the ridge was fitted to, and `ref_mse` -- which uses only
    # the held-out 20% -- would not be a comparable denominator.
    #
    # That leak is invisible wherever the nuisance is strong enough to swamp
    # it, and dominant wherever it is not: the first run of this analysis
    # reported clutter degrading performance by MINUS 69.6%, i.e. distractors
    # apparently making the task easier, which is what sent me looking.
    mses = []
    for lv in levels:
        f = load(prefix, lv, task)
        if f is None:
            continue
        Xl, Yl = pair(f, actions(lv, task))
        if len(Xl) <= cut:
            continue
        X, Y = np.concatenate(Xl[cut:]), np.concatenate(Yl[cut:])
        Yn = ((Y - ay) / asd)[:, live]
        mses.append(float(((Yn - ridge(Xtr, Ytr_n, X)) ** 2).mean()))
    if not mses:
        return None
    return {"probe": probe, "ref_mse": ref_mse, "held": float(np.mean(mses)),
            "n_levels": len(mses)}


def main() -> int:
    task = sys.argv[1] if len(sys.argv) > 1 else "push"
    levels_for = {
        "lighting": ["lighting_0p3", "lighting_0p45", "lighting_0p55",
                     "lighting_0p62"],
        "texture": ["texture_0p06", "texture_0p1", "texture_0p16",
                    "texture_0p24"],
        "clutter": ["clutter_1", "clutter_2", "clutter_3", "clutter_4"],
    }

    out: dict = {}
    rhos = {}
    for axis in AXES:
        rows = []
        for name, prefix in ENCODERS:
            c = cell(prefix, axis, task, levels_for[axis])
            if c is None:
                continue
            rows.append((name, c))
        if len(rows) < 5:
            print(f"\n{axis}: only {len(rows)} encoders cached -- skipping")
            continue

        # Invalidation 1: a head that cannot fit the reference is not comparable.
        bad = [n for n, c in rows if c["ref_mse"] >= FLOOR]
        usable = [(n, c) for n, c in rows if c["ref_mse"] < FLOOR]

        print(f"\n=== {axis} ===")
        print(f"  {'encoder':10s} {'probe R2':>9} {'ref MSE':>9} {'held-out':>9}")
        for n, c in sorted(rows, key=lambda r: r[1]["held"]):
            flag = "  EXCLUDED (ref >= floor)" if c["ref_mse"] >= FLOOR else ""
            print(f"  {n:10s} {c['probe']:>9.3f} {c['ref_mse']:>9.3f} "
                  f"{c['held']:>9.3f}{flag}")
        if bad:
            print(f"  excluded by the reference floor: {bad}")

        if len(usable) < 5:
            print("  too few usable encoders to rank; axis reported, not analysed")
            out[axis] = {"rows": {n: c for n, c in rows}, "rho": None,
                         "excluded": bad}
            continue

        # Invalidation 2: an axis that degrades nobody cannot rank anyone.
        degrade = np.mean([c["held"] / max(c["ref_mse"], 1e-9) - 1.0
                           for _, c in usable])
        if degrade < MIN_DEGRADE:
            print(f"  AXIS TOO WEAK: mean degradation {degrade:+.1%} < "
                  f"{MIN_DEGRADE:.0%}. Reported, excluded from E12a/E12b.")
            out[axis] = {"rows": {n: c for n, c in usable}, "rho": None,
                         "too_weak": True, "degradation": degrade}
            continue

        # E12d, the registered control on ourselves. Frozen random features
        # should be near-useless on any axis that genuinely separates
        # encoders. If they are competitive, the axis is not discriminating
        # and ranking encoders on it measures noise -- so its rows are
        # reported and then excluded, exactly as registered. This check was
        # written into the pre-registration and then omitted from the first
        # version of this script.
        order = [n for n, _ in sorted(usable, key=lambda r: r[1]["held"])]
        if "random" in order:
            pos = order.index("random") + 1
            third = len(order) / 3.0
            print(f"  E12d control: random ranks {pos}/{len(order)}")
            if pos <= 2 * third:
                print(f"  AXIS DOES NOT DISCRIMINATE: random features are not "
                      f"in the bottom third.")
                print(f"  Ranking encoders here measures noise. Reported, "
                      f"excluded from E12a/E12b.")
                out[axis] = {"rows": {n: c for n, c in usable}, "rho": None,
                             "e12d_fired": True, "random_rank": pos,
                             "degradation": degrade}
                continue

        # Higher probe = better; lower held-out MSE = better. Negate the probe
        # so both rank "better" the same way before correlating.
        rho = spearman([-c["probe"] for _, c in usable],
                       [c["held"] for _, c in usable])
        rhos[axis] = rho
        print(f"  mean degradation {degrade:+.1%}")
        print(f"  Spearman(probe rank, robustness rank) = {rho:+.3f}")
        out[axis] = {"rows": {n: c for n, c in usable}, "rho": rho,
                     "degradation": degrade, "excluded": bad}

    if rhos:
        vals = np.array(list(rhos.values()))
        mean_abs = float(np.abs(vals).mean())
        below = int((vals < 0.5).sum())
        print("\n" + "=" * 62)
        print("E12a -- does probe accuracy predict robustness?")
        print("=" * 62)
        for a, r in rhos.items():
            print(f"  {a:10s} rho {r:+.3f}")
        print(f"  mean |rho| {mean_abs:.3f} over {len(vals)} axes; "
              f"{below}/{len(vals)} below 0.5")
        holds = mean_abs <= 0.4 and below >= min(3, len(vals))
        print(f"\n  registered (mean |rho| <= 0.4 and rho < 0.5 on >= 3 axes): "
              f"{'HOLDS' if holds else 'FAILS'}")
        if not holds:
            print("  Probes DO predict robustness on these axes. E11's")
            print("  viewpoint-only anti-correlation is then axis-specific,")
            print("  and that scoping is the result.")
        out["e12a"] = {"rhos": rhos, "mean_abs": mean_abs, "holds": bool(holds)}

    Path("cache").mkdir(exist_ok=True)
    Path(f"cache/e12_{task}.json").write_text(json.dumps(out, indent=2,
                                                         default=float))
    print(f"\nwrote cache/e12_{task}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
