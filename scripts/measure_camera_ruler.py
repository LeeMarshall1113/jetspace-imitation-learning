#!/usr/bin/env python3
"""R1: gap as a function of camera displacement, and the conversion it implies.

    python scripts/measure_camera_ruler.py

Implements docs/prereg-camera-ruler.md. Every simulated pose is compared
against the reference pose using the N1b instrument -- same metric, same
sample count, same comb-free encoding, statistics fit on the reference side.
Because all poses render from ONE rollout, the only thing differing between any
pair is where the camera stood.

The deliverable is a conversion: read a measured gap off the curve and state it
in degrees of camera rotation. Registered explicitly as an equivalence in
MAGNITUDE, never a causal claim -- cross-lab gaps contain task, operator,
lighting and hardware differences at once, and this design separates none of
them.

Checks every registered prediction and both invalidation conditions, and prints
which ones failed rather than only the ones that passed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jetspace.envs.so101_env import R1_POSES, r1_displacement  # noqa: E402
from measure_domain_gap import centroid_distance, frechet, mmd2  # noqa: E402

# Reference values from docs/n1b-results.md, measured before this experiment.
N1B = {"null": 208.5, "S": 305.5, "Vsim": 812.0, "X": 1429.8,
       "Vreal": 1437.6, "SIM_min": 1565.5, "SIM_DR": 1677.5, "SIM": 1839.3,
       "X_sd": 303.3}


def pool(files) -> np.ndarray:
    out = []
    for f in files:
        z = np.load(f).astype(np.float32)
        out.append(z.reshape(z.shape[0], -1, z.shape[-1]).mean(axis=1))
    return np.concatenate(out, axis=0)


def gap(a: np.ndarray, b: np.ndarray, n: int, dim: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    a = a[rng.choice(len(a), n, replace=False)]
    b = b[rng.choice(len(b), n, replace=False)]
    mu, sd = a.mean(0), a.std(0) + 1e-6
    an, bn = (a - mu) / sd, (b - mu) / sd
    w, v = np.linalg.eigh(np.cov(an, rowvar=False))
    basis = v[:, np.argsort(w)[::-1][:dim]]
    ap, bp = an @ basis, bn @ basis
    m, _ = mmd2(ap, bp)
    return {"frechet": frechet(ap, bp), "mmd2": m,
            "centroid": centroid_distance(ap, bp)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", default="cache/latents")
    ap.add_argument("--prefix", default="r1_push")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="cache/r1_ruler.json")
    args = ap.parse_args()

    L = Path(args.latents)
    have = {}
    for name in R1_POSES:
        d = L / f"{args.prefix}__{name}"
        if (d / "info.json").exists():
            have[name] = sorted(d.glob("episode_*.npy"))
    if "r1_ref" not in have:
        print(f"no reference cache at {L}/{args.prefix}__r1_ref")
        return 1

    counts = {k: sum(np.load(f).shape[0] for f in v) for k, v in have.items()}
    n = min(counts.values())
    print(f"{len(have)} poses cached, n={n} latents per side, PCA {args.dim}\n")

    pools = {k: pool(v) for k, v in have.items()}
    ref = pools["r1_ref"]

    rows = []
    for name in R1_POSES:
        if name not in pools or name == "r1_ref":
            continue
        d = r1_displacement(name)
        g = gap(ref, pools[name], n, args.dim, args.seed)
        rows.append({"pose": name, **d, **g})

    # ---- invalidation 1: the zero-displacement control -------------------
    files = have["r1_ref"]
    mid = len(files) // 2
    null = None
    if len(files) >= 4:
        # Each half holds roughly half the episodes, so it cannot supply the
        # full-set sample count n. Sizing this from n crashed on the first run.
        a, b = pool(files[:mid]), pool(files[mid:])
        null = gap(a, b, min(len(a), len(b)), args.dim, args.seed)
        null["n_per_side"] = int(min(len(a), len(b)))

    print("=" * 74)
    print("R1 CAMERA RULER")
    print("=" * 74)
    if null:
        print(f"0-displacement control (r1_ref split by episode): "
              f"frechet {null['frechet']:.1f}  [n={null['n_per_side']}]")
        if null["n_per_side"] < n:
            print(f"  NOTE: the null uses n={null['n_per_side']} against the poses'"
                  f" n={n}. Frechet grows as n falls, so this is an UPPER bound on"
                  " the true null and the curve's floor is if anything lower.")
        print(f"  N1b null for comparison: {N1B['null']:.1f}")
        if null["frechet"] > 3 * N1B["null"]:
            print("  INVALIDATED: the same pose does not reproduce the null. The")
            print("  sweep is measuring something other than displacement.")
            return 0
        print("  -> instrument OK\n")

    def show(title, sel):
        sub = sorted([r for r in rows if sel(r)], key=lambda r: r["angle"] + r["dist_ratio"])
        if not sub:
            return
        print(title)
        print(f"  {'pose':14s} {'angle':>7} {'dist':>6} {'frechet':>9} {'MMD^2':>9}")
        for r in sub:
            print(f"  {r['pose']:14s} {r['angle']:>6.1f}° {r['dist_ratio']:>5.2f}x "
                  f"{r['frechet']:>9.1f} {r['mmd2']:>9.5f}")
        print()

    show("AZIMUTH sweep", lambda r: r["pose"].startswith("r1_az"))
    show("ELEVATION sweep", lambda r: r["pose"].startswith("r1_el"))
    show("DISTANCE sweep", lambda r: r["pose"].startswith("r1_d"))
    show("OFF-AXIS", lambda r: r["pose"].startswith("r1_a") and "e" in r["pose"][4:])

    # ---- predictions -----------------------------------------------------
    print("=" * 74)
    print("REGISTERED PREDICTIONS")
    print("=" * 74)
    verdicts = {}

    az = sorted([r for r in rows if r["pose"].startswith("r1_az")],
                key=lambda r: r["angle"])
    el = sorted([r for r in rows if r["pose"].startswith("r1_el")],
                key=lambda r: r["angle"])
    mono_az = all(az[i]["frechet"] <= az[i + 1]["frechet"] * 1.05
                  for i in range(len(az) - 1))
    verdicts["1 monotonic in angle"] = mono_az
    print(f"1 monotonic in azimuth: {'HOLDS' if mono_az else 'FAILS'}")
    if not mono_az:
        print("    Gap is not monotonic in angle. Viewpoint is not a single axis")
        print("    and no scalar ruler exists. This invalidates the conversion.")

    dist = [r for r in rows if r["pose"].startswith("r1_d")]
    far = max((r["frechet"] for r in dist if r["dist_ratio"] > 1.5), default=0.0)
    rot45 = next((r["frechet"] for r in az if abs(r["angle"] - 45) < 1), None)
    if rot45 is not None:
        p2 = far < rot45
        verdicts["2 angle dominates distance"] = p2
        print(f"2 angle beats distance: {'HOLDS' if p2 else 'FAILS'}  "
              f"(1.8x distance {far:.1f} vs 45° rotation {rot45:.1f})")

    # ---- the conversion --------------------------------------------------
    curve = sorted(az, key=lambda r: r["angle"])
    angles = np.array([0.0] + [r["angle"] for r in curve])
    gaps = np.array([null["frechet"] if null else 0.0] + [r["frechet"] for r in curve])

    def to_degrees(target: float) -> float | None:
        """Invert the azimuth curve by linear interpolation."""
        if target <= gaps[0]:
            return 0.0
        for i in range(len(gaps) - 1):
            if gaps[i] <= target <= gaps[i + 1]:
                span = gaps[i + 1] - gaps[i]
                f = 0.0 if span < 1e-9 else (target - gaps[i]) / span
                return float(angles[i] + f * (angles[i + 1] - angles[i]))
        return None

    print()
    print("=" * 74)
    print("CONVERSION -- measured gaps in units of camera rotation")
    print("=" * 74)
    print(f"azimuth curve spans frechet {gaps.min():.1f} .. {gaps.max():.1f} "
          f"over 0..{angles.max():.0f}°\n")
    print(f"  {'N1b rung':12s} {'frechet':>9}   equivalent rotation")
    for k in ("S", "Vsim", "X", "Vreal", "SIM_min", "SIM_DR", "SIM"):
        deg = to_degrees(N1B[k])
        txt = (f"{deg:.1f}°" if deg is not None
               else f"BEYOND the sweep (>{angles.max():.0f}°)")
        print(f"  {k:12s} {N1B[k]:>9.1f}   {txt}")

    theta_x = to_degrees(N1B["X"])
    print()
    p3 = theta_x is not None and theta_x > 30
    verdicts["3 theta_X > 30 deg"] = bool(p3)
    if theta_x is None:
        print("3 theta_X > 30°: FAILS -- cross-lab gaps EXCEED anything camera")
        print("    movement produces. N1b's Vreal ~ X was then a coincidence of two")
        print("    unrelated quantities, and this is a correction to our own result.")
    else:
        print(f"3 theta_X > 30°: {'HOLDS' if p3 else 'FAILS'}  (theta_X = {theta_x:.1f}°)")

    sim_deg = to_degrees(N1B["SIM"])
    p5 = sim_deg is None
    verdicts["5 SIM exceeds the ruler"] = bool(p5)
    print(f"5 SIM beyond the sweep: {'HOLDS' if p5 else 'FAILS'}"
          + ("" if p5 else f"  (SIM = {sim_deg:.1f}°, inside the range -- viewpoint"
                           " alone could account for the sim-to-real gap)"))

    # ---- invalidation 2: saturation -------------------------------------
    if len(curve) >= 3:
        g10 = next((r["frechet"] for r in curve if abs(r["angle"] - 10) < 1), None)
        gmax = gaps.max()
        if g10 and gmax > 0 and g10 / gmax > 0.9:
            print("\nINVALIDATED: the curve saturates by 10°. The ruler has no")
            print("  resolution in the range that matters and converts nothing.")

    # ---- off-axis composition -------------------------------------------
    off = [r for r in rows if r["pose"].startswith("r1_a") and "e" in r["pose"][4:]]
    if off and az and el:
        sub = 0
        for r in off:
            a_part = next((x["frechet"] for x in az
                           if abs(x["angle"] - r["azim"]) < 1), None)
            e_part = next((x["frechet"] for x in el
                           if abs(x["angle"] - abs(r["elev"])) < 1), None)
            if a_part and e_part and r["frechet"] < a_part + e_part:
                sub += 1
        p4 = sub >= len(off) * 0.7
        verdicts["4 sub-additive"] = bool(p4)
        print(f"\n4 sub-additive composition: {'HOLDS' if p4 else 'FAILS'}  "
              f"({sub}/{len(off)} off-axis gaps below the sum of their parts)")

    print()
    print("=" * 74)
    failed = [k for k, v in verdicts.items() if not v]
    if failed:
        print(f"PREDICTIONS THAT FAILED: {', '.join(failed)}")
        print("Reported because they were registered, not because they are welcome.")
    else:
        print("All checked predictions held.")
    print("=" * 74)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "n_per_side": int(n), "pca_dim": args.dim,
        "null": null, "poses": rows, "verdicts": verdicts,
        "theta_X": theta_x, "n1b_reference": N1B,
    }, indent=2, default=float))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
