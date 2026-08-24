#!/usr/bin/env python3
"""Measure every N1b rung as a distribution, per docs/prereg-n1b.md.

    python scripts/run_n1b_measure.py

N1 compared single numbers and one unrepresentative viewpoint control was
enough to invalidate it. Here every rung is a set of independent pairings, so a
rung is a spread rather than a point that one bad choice can move.

Rungs, all on scene cameras:

    N       one dataset split by episode        sampling noise -- the null
    Vsim    simulated camera poses, pairwise    viewpoint ALONE, all else pinned
    Vreal   one real dataset, its two cameras   viewpoint + real-camera differences
    S       same lab, same task, two sessions   session noise
    X       different laboratories              domain, as the field meets it
    SIM     simulation vs each real dataset     the measurement
    SIM_DR  randomised simulation vs each real  does DR close the gap

Frechet is the primary metric, named in advance. Centroid is reported but
decides nothing: in N1 it ranked simulation below the session floor, which is
the blindness to spread the first registration predicted for it.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from measure_domain_gap import (  # noqa: E402
    centroid_distance,
    frechet,
    load_pooled,
    mmd2,
)

L = Path("cache/latents")

#: lab -> (dataset stem, camera A, camera B). Camera A is the primary, used for
#: every cross-lab pairing; B exists only so Vreal has a within-dataset pair.
REAL = {
    "A_cubes":   ("n1b_A_cubes",   "ego",       "external"),
    "B_svla":    ("n1b_B_svla",    "up",        "side"),
    "C_tape":    ("n1b_C_tape",    "birdEye",   "thirdPerson"),
    "D_ball":    ("n1b_D_ball",    "front",     "side"),
    "E_summer":  ("n1b_E_summer",  "front",     "side"),
    "F_cup":     ("n1b_F_cup",     "cam_front", "cam_top"),
    "G_bin":     ("n1b_G_bin",     "front",     "top"),
    "H_penmug":  ("n1b_H_penmug1", "camera_2",  "camera_4"),
}

#: Datasets whose gaps were already seen during N1. Any pairing touching these
#: is reported separately and excluded from the headline.
SEEN = {"A_cubes", "H_penmug"}

#: Same lab, same task, four different recording sessions -> rung S.
SESSIONS = ["n1b_H_penmug1", "n1b_H_penmug2", "n1b_H_penmug3", "n1b_H_penmug4"]

SIM_CAMERAS = ["front", "front_high", "side", "side_high", "top"]


def d(stem: str, cam: str) -> Path:
    return L / f"{stem}__{cam}"


def measure(a_dir: Path, b_dir: Path, n: int, seed: int, dim: int,
            a: np.ndarray | None = None, b: np.ndarray | None = None) -> dict | None:
    """Centroid, MMD and Frechet for one pairing, at a fixed sample count."""
    try:
        A = a if a is not None else load_pooled(a_dir, None, seed)
        B = b if b is not None else load_pooled(b_dir, None, seed)
    except SystemExit:
        return None
    if len(A) < n or len(B) < n:
        return None

    rng = np.random.default_rng(seed)
    A = A[rng.choice(len(A), n, replace=False)]
    B = B[rng.choice(len(B), n, replace=False)]

    # Statistics come from the reference (first) side, always the real one for
    # SIM pairings, so simulation never defines the space it is judged in.
    mu, sd = A.mean(0), A.std(0) + 1e-6
    An, Bn = (A - mu) / sd, (B - mu) / sd

    cov = np.cov(An, rowvar=False)
    w, v = np.linalg.eigh(cov)
    basis = v[:, np.argsort(w)[::-1][:dim]]
    Ap, Bp = An @ basis, Bn @ basis

    m, _ = mmd2(Ap, Bp)
    return {
        "centroid": centroid_distance(Ap, Bp),
        "mmd2": m,
        "frechet": frechet(Ap, Bp),
    }


def split_halves(stem_cam: Path, seed: int):
    """Two disjoint halves of one dataset, split by EPISODE not by frame.

    Splitting by frame would put neighbouring frames of the same episode on both
    sides, which are near-duplicates -- the null would come out at zero for a
    trivial reason and would not test the metric at all.
    """
    files = sorted(stem_cam.glob("episode_*.npy"))
    if len(files) < 4:
        return None, None
    mid = len(files) // 2

    def pool(fs):
        out = []
        for f in fs:
            z = np.load(f).astype(np.float32)
            out.append(z.reshape(z.shape[0], -1, z.shape[-1]).mean(axis=1))
        return np.concatenate(out, axis=0)

    return pool(files[:mid]), pool(files[mid:])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=None, help="latents per side; default = smallest")
    ap.add_argument("--out", default="cache/n1b_rungs.json")
    args = ap.parse_args()

    have = {p.name for p in L.glob("n1b_*") if (p / "info.json").exists()}
    if not have:
        print("no n1b latent caches found; run scripts/run_n1b.sh first")
        return 1

    counts = []
    for name in have:
        fs = glob.glob(str(L / name / "episode_*.npy"))
        counts.append(sum(np.load(f).shape[0] for f in fs))
    n = args.n or max(200, min(counts))
    print(f"{len(have)} caches, sample count fixed at n={n} per side\n")

    rungs: dict[str, list[dict]] = {}

    def add(rung: str, label: str, a: Path, b: Path, seen: bool,
            arrs=(None, None)) -> None:
        r = measure(a, b, n, args.seed, args.dim, *arrs)
        if r is None:
            print(f"  {rung:8s} {label:44s} skipped (too few latents)")
            return
        r.update({"pair": label, "seen": seen})
        rungs.setdefault(rung, []).append(r)
        mark = " [seen]" if seen else ""
        print(f"  {rung:8s} {label:44s} frechet {r['frechet']:8.1f}{mark}")

    # ---- N: the null. If this is not near zero the instrument is broken. ----
    print("rung N - one dataset split by episode (null)")
    for lab, (stem, cam, _) in REAL.items():
        if d(stem, cam).name not in have:
            continue
        A, B = split_halves(d(stem, cam), args.seed)
        if A is None:
            continue
        add("N", f"{lab} first half vs second half", d(stem, cam), d(stem, cam),
            lab in SEEN, arrs=(A, B))

    # ---- Vsim: viewpoint with everything else pinned ----
    print("\nrung Vsim - simulated camera poses, identical episodes")
    for c1, c2 in itertools.combinations(SIM_CAMERAS, 2):
        a, b = d("n1b_sim_push", c1), d("n1b_sim_push", c2)
        if a.name in have and b.name in have:
            add("Vsim", f"sim {c1} vs {c2}", a, b, False)

    # ---- Vreal: two scene cameras inside one real dataset ----
    print("\nrung Vreal - two scene cameras, same real dataset")
    for lab, (stem, ca, cb) in REAL.items():
        a, b = d(stem, ca), d(stem, cb)
        if a.name in have and b.name in have:
            add("Vreal", f"{lab} {ca} vs {cb}", a, b, lab in SEEN)

    # ---- S: same lab, same task, different sessions ----
    print("\nrung S - same lab and task, different sessions")
    cam = REAL["H_penmug"][1]
    for s1, s2 in itertools.combinations(SESSIONS, 2):
        a, b = d(s1, cam), d(s2, cam)
        if a.name in have and b.name in have:
            add("S", f"{s1[-8:]} vs {s2[-8:]}", a, b, True)

    # ---- X: across laboratories ----
    print("\nrung X - different laboratories")
    labs = [(k, v) for k, v in REAL.items() if d(v[0], v[1]).name in have]
    for (l1, v1), (l2, v2) in itertools.combinations(labs, 2):
        add("X", f"{l1} vs {l2}", d(v1[0], v1[1]), d(v2[0], v2[1]),
            l1 in SEEN or l2 in SEEN)

    # ---- SIM and SIM_DR ----
    for rung, stem in (("SIM", "n1b_sim_push"), ("SIM_DR", "n1b_sim_push_dr")):
        print(f"\nrung {rung} - simulation vs each real dataset (front camera)")
        for lab, (rs, rc, _) in REAL.items():
            a, b = d(rs, rc), d(stem, "front")
            if a.name in have and b.name in have:
                add(rung, f"{lab} vs {stem[-4:]}", a, b, lab in SEEN)

    # ---- SIM_min: best simulated viewpoint per real dataset ----
    print("\nrung SIM_min - closest simulated camera pose to each real dataset")
    for lab, (rs, rc, _) in REAL.items():
        if d(rs, rc).name not in have:
            continue
        best, which = None, None
        for c in SIM_CAMERAS:
            if d("n1b_sim_push", c).name not in have:
                continue
            r = measure(d(rs, rc), d("n1b_sim_push", c), n, args.seed, args.dim)
            if r and (best is None or r["frechet"] < best["frechet"]):
                best, which = r, c
        if best:
            best.update({"pair": f"{lab} vs sim {which}", "seen": lab in SEEN,
                         "best_camera": which})
            rungs.setdefault("SIM_min", []).append(best)
            print(f"  SIM_min  {lab:10s} best pose {which:11s} "
                  f"frechet {best['frechet']:8.1f}")

    # ------------------------------------------------------------ summary --
    def stats(rows, key="frechet"):
        v = np.array([r[key] for r in rows])
        return v.mean(), v.std(), v.min(), v.max(), len(v)

    print("\n" + "=" * 78)
    print("N1b RUNGS (Frechet, primary metric)")
    print("=" * 78)
    print(f"{'rung':9s} {'n':>3} {'mean':>9} {'sd':>8} {'min':>9} {'max':>9}   unseen-only mean")
    print("-" * 78)
    order = ["N", "Vsim", "Vreal", "S", "X", "SIM", "SIM_min", "SIM_DR"]
    summary = {}
    for k in order:
        rows = rungs.get(k)
        if not rows:
            continue
        m, s, lo, hi, c = stats(rows)
        unseen = [r for r in rows if not r["seen"]]
        um = np.mean([r["frechet"] for r in unseen]) if unseen else float("nan")
        summary[k] = {"mean": m, "sd": s, "min": lo, "max": hi, "n": c,
                      "unseen_mean": um, "pairs": rows}
        us = f"{um:9.1f}" if unseen else "        -"
        print(f"{k:9s} {c:>3} {m:>9.1f} {s:>8.1f} {lo:>9.1f} {hi:>9.1f}   {us}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {k: {kk: (vv if kk != "pairs" else vv) for kk, vv in v.items()}
         for k, v in summary.items()}, indent=2, default=float))
    print(f"\nwrote {args.out}")

    # ------------------------------------------------- registered verdict --
    print("\n" + "=" * 78)
    print("VERDICT, by the rule registered in docs/prereg-n1b.md")
    print("=" * 78)
    if "N" not in summary or "X" not in summary or "SIM" not in summary:
        print("Rungs missing; cannot apply the rule.")
        return 0

    null_m = summary["N"]["mean"]
    x_all = np.array([r["frechet"] for r in summary["X"]["pairs"]])
    x_p10 = float(np.percentile(x_all, 10))
    if null_m > x_p10:
        print(f"INVALIDATED: the null rung is {null_m:.1f}, above the 10th percentile")
        print(f"  of cross-lab gaps ({x_p10:.1f}). The metric cannot tell two halves of")
        print("  ONE dataset apart from two different labs. Nothing else is readable.")
        return 0
    print(f"null N = {null_m:.1f} vs 10th pct of X = {x_p10:.1f}  -> instrument OK")

    vreal_m = summary.get("Vreal", {}).get("mean", float("nan"))
    x_mean = summary["X"]["mean"]
    if vreal_m >= x_mean:
        print(f"\nNOTE: Vreal ({vreal_m:.1f}) >= X mean ({x_mean:.1f}). Viewpoint still")
        print("  rivals domain among real datasets, so X is partly a viewpoint")
        print("  measurement. Vsim and SIM_min remain valid -- both pin or sweep")
        print("  viewpoint deliberately.")

    s_mean = summary.get("S", {}).get("mean", float("nan"))
    sim_m = summary["SIM"]["mean"]
    x_max = summary["X"]["max"]
    print(f"\nS={s_mean:.1f}   X mean={x_mean:.1f}   X max={x_max:.1f}   SIM={sim_m:.1f}")
    if sim_m <= s_mean:
        print("\n  SIM <= S: simulation closer to real than two sessions of one lab.")
        print("  Registered as NOT CREDIBLE -- evidence the instrument is still")
        print("  wrong, reported as such rather than as a finding.")
    elif sim_m <= x_mean:
        print("\n  S < SIM <= X mean: SIMULATION IS WITHIN NORMAL CROSS-LAB VARIATION.")
        print("  The frozen-encoder alignment assumption is SUPPORTED.")
    elif sim_m <= x_max:
        print("\n  X mean < SIM <= X max: further than a typical lab, still inside")
        print("  the range real labs span. WEAK SUPPORT.")
    else:
        print("\n  SIM > X max: simulation falls OUTSIDE the real manifold.")
        print("  The alignment assumption FAILS. That is the result.")

    if "SIM_DR" in summary:
        dr = summary["SIM_DR"]["mean"]
        arrow = "INCREASED" if dr > sim_m else "DECREASED"
        print(f"\n  Domain randomisation {arrow} the gap: {sim_m:.1f} -> {dr:.1f}")
        print("  N1 predicted an increase, on one reference dataset; this tests it")
        print(f"  against {summary['SIM_DR']['n']}.")

    if "SIM_min" in summary:
        mn = summary["SIM_min"]["mean"]
        print(f"\n  SIM_min = {mn:.1f} (best simulated viewpoint per real dataset)")
        print(f"  vs SIM = {sim_m:.1f} on the default 'front' pose. The difference is")
        print("  what matching the camera is worth, which is the number someone")
        print("  building a simulator actually wants.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
