#!/usr/bin/env python3
"""H1d: does gap -> degradation survive on real robot video?

    python scripts/analyze_h1d.py --seeds 0 1 2

Registered in docs/prereg-h1.md as the differentiator: rho <= -0.5 between
latent gap and degradation, where the world model is trained on one real
laboratory and evaluated on the others. Everything upstream of this is
simulation, and arXiv:2604.13645 already correlates Wasserstein against policy
success in simulation. Real video is the part nobody has shown.

Three things this has to survive that the simulated version did not:

**Task is not held constant.** Lab A stacks cubes, lab D rolls a ball, lab G
sorts a bin. A cross-lab gap therefore mixes visual domain with task, which is
why the registered threshold is -0.5 rather than -0.6.

**Action spaces are not interchangeable** (ledger L8). All eight labs are 6-dim
on roughly [-100, 100], but per-dimension spread differs by up to 5x. The world
model is action-conditioned, so part of any degradation is action mismatch
rather than visual shift. This script measures action distance per pair and
reports the PARTIAL correlation controlling for it. If gap only predicts
degradation because both track action mismatch, the control kills it.

**Training lab is a cluster, not a sample.** Each training lab contributes
seven pairs that share one checkpoint. Bootstrapping over pairs would treat
them as independent; the interval here resamples whole training labs.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from align_simulator import gap_between  # noqa: E402
from harden_gap_prediction import oos_prediction, spearman  # noqa: E402

LABS = {
    "A_cubes": "n1b_A_cubes__ego",
    "B_svla": "n1b_B_svla__side",
    "C_tape": "n1b_C_tape__birdEye",
    "D_ball": "n1b_D_ball__front",
    "E_summer": "n1b_E_summer__front",
    "F_cup": "n1b_F_cup__cam_front",
    "G_bin": "n1b_G_bin__front",
    "H_penmug1": "n1b_H_penmug1__camera_2",
}


def load_latents(name: str) -> np.ndarray:
    files = sorted((Path("cache/latents") / name).glob("episode_*.npy"))
    out = []
    for f in files:
        z = np.load(f).astype(np.float32)
        out.append(z.reshape(z.shape[0], -1, z.shape[-1]).mean(axis=1))
    return np.concatenate(out, axis=0)


def load_actions(name: str) -> np.ndarray:
    files = sorted((Path("data/episodes") / name).glob("episode_*.npz"))
    return np.concatenate([np.load(f)["action"].astype(np.float32) for f in files])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--out", default="cache/h1d.json")
    args = ap.parse_args()

    print("=" * 74)
    print("H1d -- gap -> degradation on REAL robot video")
    print("=" * 74)

    lat = {k: load_latents(v) for k, v in LABS.items()}
    act = {k: load_actions(v) for k, v in LABS.items()}

    # Visual gap and action gap, both as Frechet, both at the same dim so the
    # control is on the same footing as the thing it controls for.
    vis: dict[tuple[str, str], float] = {}
    acts: dict[tuple[str, str], float] = {}
    for a, b in itertools.permutations(LABS, 2):
        vis[(a, b)] = gap_between(lat[a], lat[b], args.dim, 0)
        acts[(a, b)] = gap_between(act[a], act[b], min(6, args.dim) - 1, 0)

    rows = []
    missing = []
    for train in LABS:
        for s in args.seeds:
            base = Path(f"cache/conservatism_h1d_{train}_s{s}_{train}.json")
            if not base.exists():
                missing.append(f"{train}/s{s} in-domain")
                continue
            ref = json.loads(base.read_text())["mean_cosine"]
            for ev in LABS:
                if ev == train:
                    continue
                c = Path(f"cache/conservatism_h1d_{train}_s{s}_{ev}.json")
                if not c.exists():
                    missing.append(f"{train}/s{s}->{ev}")
                    continue
                d = json.loads(c.read_text())
                rows.append({
                    "train": train, "eval": ev, "seed": s,
                    "gap": float(vis[(train, ev)]),
                    "action_gap": float(acts[(train, ev)]),
                    "cosine": d["mean_cosine"],
                    # Degradation relative to this model's OWN in-domain score.
                    # Absolute cosine would confound "this lab is hard" with
                    # "this lab is far", and only the second is the claim.
                    "degradation": float(ref - d["mean_cosine"]),
                })

    if missing:
        print(f"  missing {len(missing)} cells, e.g. {missing[:3]}")
    if len(rows) < 8:
        print(f"\nonly {len(rows)} pairs scored -- run scripts/run_h1d.sh first")
        return 1

    gap = np.array([r["gap"] for r in rows])
    deg = np.array([r["degradation"] for r in rows])
    agap = np.array([r["action_gap"] for r in rows])
    trains = np.array([r["train"] for r in rows])

    print(f"\n{len(rows)} (train, eval, seed) cells across "
          f"{len(set(trains))} training labs\n")

    # ---- per training lab ------------------------------------------------
    print(f"  {'train lab':12s} {'n':>3} {'rho':>8} {'mean deg':>10}")
    per = {}
    for t in sorted(set(trains)):
        m = trains == t
        # Sign convention: gap up, cosine down, so degradation UP. rho should
        # be POSITIVE here -- the simulated arms correlated gap against cosine
        # itself, where it is negative. Same relationship, opposite sign.
        r = spearman(gap[m], deg[m]) if m.sum() >= 4 else float("nan")
        per[t] = float(r)
        print(f"  {t:12s} {int(m.sum()):3d} {r:8.3f} {deg[m].mean():10.4f}")

    rho = spearman(gap, deg)
    rho_a = spearman(agap, deg)
    rho_ga = spearman(gap, agap)

    # ---- partial correlation, the control that matters -------------------
    # Spearman partial: correlate the two rank vectors after removing their
    # linear dependence on the control's ranks.
    def ranks(x):
        r = np.argsort(np.argsort(x)).astype(float)
        return (r - r.mean()) / (r.std() + 1e-12)

    rg, rd, ra = ranks(gap), ranks(deg), ranks(agap)
    resid_g = rg - (rg @ ra) / len(ra) * ra
    resid_d = rd - (rd @ ra) / len(ra) * ra
    denom = np.linalg.norm(resid_g) * np.linalg.norm(resid_d)
    partial = float(resid_g @ resid_d / denom) if denom > 1e-12 else float("nan")

    # ---- cluster bootstrap over training labs ----------------------------
    rng = np.random.default_rng(0)
    labs = sorted(set(trains))
    by = {t: np.where(trains == t)[0] for t in labs}
    boots = []
    for _ in range(args.boot):
        pick = rng.choice(labs, size=len(labs), replace=True)
        idx = np.concatenate([by[t] for t in pick])
        if len(set(idx.tolist())) < 6:
            continue
        boots.append(spearman(gap[idx], deg[idx]))
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5])

    r2s, maes = oos_prediction(gap, deg, args.repeats, 0)

    print("\n" + "=" * 74)
    print("REGISTERED PREDICTION")
    print("=" * 74)
    verdict = "HOLDS" if rho >= 0.5 else "FAILS"
    print(f"H1d rho >= +0.5 (gap vs degradation): {verdict}   rho {rho:+.3f}")
    print(f"     lab-cluster 95% CI  [{lo:+.3f}, {hi:+.3f}]")
    print(f"     per-lab rho range   [{min(per.values()):+.3f}, "
          f"{max(per.values()):+.3f}]")

    print("\n" + "=" * 74)
    print("ACTION-SPACE CONTROL (ledger L8)")
    print("=" * 74)
    print(f"  gap        vs degradation   rho {rho:+.3f}")
    print(f"  action gap vs degradation   rho {rho_a:+.3f}")
    print(f"  gap        vs action gap    rho {rho_ga:+.3f}")
    print(f"  PARTIAL gap vs degradation, action gap held  rho {partial:+.3f}")
    if not np.isnan(partial):
        if abs(partial) < 0.3 <= abs(rho):
            print("  The relationship does NOT survive the control. Latent gap")
            print("  and degradation both track action mismatch; the visual")
            print("  claim is not supported on real data. Report this.")
        elif abs(partial) >= 0.5 * abs(rho):
            print("  Survives the control: visual gap predicts degradation")
            print("  beyond what action-space mismatch explains.")
        else:
            print("  Partially attenuated -- some but not all of the")
            print("  relationship is action mismatch. Report both numbers.")

    print("\n  out-of-sample: R2 "
          f"{np.nanmedian(r2s):.3f}, MAE {np.nanmedian(maes):.4f} "
          "(not registered for H1d)")

    Path(args.out).write_text(json.dumps(
        {"rows": rows, "rho": float(rho), "ci": [float(lo), float(hi)],
         "per_lab": per, "rho_action": float(rho_a),
         "rho_gap_action": float(rho_ga), "partial": partial,
         "oos_r2": float(np.nanmedian(r2s)), "oos_mae": float(np.nanmedian(maes)),
         "dim": args.dim, "seeds": args.seeds}, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
