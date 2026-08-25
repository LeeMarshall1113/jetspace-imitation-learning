#!/usr/bin/env python3
"""E2: every distribution-shift rung, measured in ONE space.

    python scripts/measure_rungs.py --dim 32

The numbers this project has been quoting for session drift, cross-lab shift and
sim-to-real came from different runs at different `pca_dim` settings. Ledger L7
records what that costs: two runs differing only in `pca_dim` and `hidden`
produced an apparent finding ("real beats sim") that was an artefact of the
mismatch. Frechet distance has no absolute scale -- it is only meaningful
against another Frechet computed the same way -- so a table assembled from
separate runs is not a table, it is a list of incomparable numbers.

This script recomputes every rung from cached latents with one `dim`, one
pooling, one estimator, so the rows can actually be compared to each other and,
later, to the simulator sweep in `exchange_rate.py`.

The rung that matters most is SESSION: lab H recorded four separate sessions
(penmug1-4) on the same camera with the same setup. Nothing changed except the
day. That quantity -- how far a robot dataset drifts from itself between
sessions -- appears to be unmeasured in the literature, and it is the floor
under every cross-domain number anyone reports: a cross-lab gap only means
something relative to how far one lab drifts from itself.

NULL is the estimator's own noise: one condition split into two halves by
episode. Any rung not clearly above NULL is not a measurement.
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

LAT = Path("cache/latents")

#: Lab H's four sessions: same lab, same camera, same task, different days.
SESSIONS_H = [f"n1b_H_penmug{i}__camera_2" for i in (1, 2, 3, 4)]

#: Each real lab's two viewpoints. Within a lab the session is fixed, so these
#: isolate viewpoint from everything else.
CAMERA_PAIRS = [
    ("n1b_A_cubes__ego", "n1b_A_cubes__external_D455"),
    ("n1b_B_svla__side", "n1b_B_svla__up"),
    ("n1b_C_tape__birdEye", "n1b_C_tape__thirdPerson"),
    ("n1b_D_ball__front", "n1b_D_ball__side"),
    ("n1b_E_summer__front", "n1b_E_summer__side"),
    ("n1b_F_cup__cam_front", "n1b_F_cup__cam_top"),
    ("n1b_G_bin__front", "n1b_G_bin__top"),
    ("n1b_H_penmug1__camera_2", "n1b_H_penmug1__camera_4"),
]

#: One canonical viewpoint per lab, for cross-lab pairs.
LABS = ["n1b_A_cubes__ego", "n1b_B_svla__side", "n1b_C_tape__birdEye",
        "n1b_D_ball__front", "n1b_E_summer__front", "n1b_F_cup__cam_front",
        "n1b_G_bin__front", "n1b_H_penmug1__camera_2"]

SIM = "n1b_sim_push__front"
SIM_DR = "n1b_sim_push_dr__front"
SIM_CAMS = ["n1b_sim_push__front", "n1b_sim_push__side", "n1b_sim_push__top",
            "n1b_sim_push__front_high", "n1b_sim_push__side_high"]


def short(name: str) -> str:
    """Strip the n1b_ prefix so the printed tables stay readable."""
    return name[4:] if name.startswith("n1b_") else name


def load(name: str, half: str | None = None) -> np.ndarray:
    """Pooled latents for one condition.

    `half` splits BY EPISODE, never by frame -- frames within an episode are
    strongly correlated, so a frame-wise split would put near-duplicate frames
    on both sides and report a noise floor far below the truth.
    """
    files = sorted((LAT / name).glob("episode_*.npy"))
    if not files:
        raise SystemExit(f"no latents in {LAT / name}")
    if half == "a":
        files = files[: len(files) // 2]
    elif half == "b":
        files = files[len(files) // 2:]
    out = []
    for f in files:
        z = np.load(f).astype(np.float32)
        out.append(z.reshape(z.shape[0], -1, z.shape[-1]).mean(axis=1))
    return np.concatenate(out, axis=0)


def sym_gap(a: np.ndarray, b: np.ndarray, dim: int, seed: int) -> tuple[float, float]:
    """Frechet both ways.

    `gap_between` fits its whitening statistics and PCA basis on the first
    argument, so it is not symmetric. Reporting the mean without the spread
    would hide how much that choice matters.
    """
    ab = gap_between(a, b, dim, seed)
    ba = gap_between(b, a, dim, seed)
    return 0.5 * (ab + ba), abs(ab - ba)


def summarise(label: str, vals: list[float], extra: str = "") -> dict:
    v = np.asarray(vals, dtype=float)
    lo, hi = float(v.min()), float(v.max())
    print(f"  {label:34s} n={len(v):<3d} mean {v.mean():8.1f}   "
          f"range [{lo:7.1f}, {hi:7.1f}]  {extra}")
    return {"n": int(len(v)), "mean": float(v.mean()), "sd": float(v.std()),
            "min": lo, "max": hi, "values": [float(x) for x in v]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="cache/e2_rungs.json")
    args = ap.parse_args()

    print("=" * 78)
    print(f"E2 -- all rungs in one space (Frechet, PCA dim {args.dim}, "
          f"seed {args.seed})")
    print("=" * 78)

    cache: dict[str, np.ndarray] = {}

    def get(name: str, half: str | None = None) -> np.ndarray:
        key = f"{name}|{half}"
        if key not in cache:
            cache[key] = load(name, half)
        return cache[key]

    res: dict[str, dict] = {}
    asym: list[float] = []

    # ---- NULL: the estimator's own noise --------------------------------
    print("\nNULL -- same condition, episodes split in half")
    nulls = []
    for name in SESSIONS_H + [SIM, "n1b_A_cubes__ego", "n1b_D_ball__front"]:
        try:
            g, d = sym_gap(get(name, "a"), get(name, "b"), args.dim, args.seed)
        except ValueError as exc:
            print(f"    {short(name)}: skipped ({exc})")
            continue
        nulls.append(g)
        asym.append(d)
        print(f"    {short(name):>26s}  {g:8.1f}")
    if not nulls:
        raise SystemExit("no null rung could be computed -- everything below "
                         "would be uninterpretable, so stopping here.")
    res["null"] = summarise("null (self, split by episode)", nulls)
    floor = res["null"]["mean"]

    def vs_floor(vals: list[float]) -> str:
        return f"{float(np.mean(vals)) / max(floor, 1e-9):.1f}x null"

    # ---- SESSION: the unclaimed rung ------------------------------------
    print("\nSESSION -- lab H, camera_2, four sessions, nothing else changed")
    vals = []
    for a, b in itertools.combinations(SESSIONS_H, 2):
        g, d = sym_gap(get(a), get(b), args.dim, args.seed)
        vals.append(g)
        asym.append(d)
        print(f"    {short(a).split('__')[0]:>14s} vs "
              f"{short(b).split('__')[0]:<14s} {g:8.1f}")
    res["session"] = summarise("session (same lab, same camera)", vals, vs_floor(vals))

    # ---- CAMERA: viewpoint, session held fixed --------------------------
    print("\nCAMERA -- same lab, same session, two viewpoints")
    vals = []
    for a, b in CAMERA_PAIRS:
        g, d = sym_gap(get(a), get(b), args.dim, args.seed)
        vals.append(g)
        asym.append(d)
        print(f"    {short(a).split('__')[0]:>14s}  {g:8.1f}")
    res["camera"] = summarise("camera (within lab)", vals, vs_floor(vals))

    # ---- SIM CAMERA: the same knob, in simulation ------------------------
    print("\nSIM CAMERA -- simulator, five viewpoints, everything else identical")
    vals = []
    for a, b in itertools.combinations(SIM_CAMS, 2):
        g, d = sym_gap(get(a), get(b), args.dim, args.seed)
        vals.append(g)
        asym.append(d)
    res["sim_camera"] = summarise("sim camera", vals, vs_floor(vals))

    # ---- CROSS-LAB -------------------------------------------------------
    print("\nCROSS-LAB -- different lab, different robot, different everything")
    vals = []
    for a, b in itertools.combinations(LABS, 2):
        g, d = sym_gap(get(a), get(b), args.dim, args.seed)
        vals.append(g)
        asym.append(d)
    res["cross_lab"] = summarise("cross-lab", vals, vs_floor(vals))

    # ---- SIM-TO-REAL -----------------------------------------------------
    print("\nSIM-TO-REAL -- simulator against each real lab")
    for tag, sim in (("sim2real", SIM), ("sim2real_dr", SIM_DR)):
        vals = []
        for lab in LABS:
            g, d = sym_gap(get(sim), get(lab), args.dim, args.seed)
            vals.append(g)
            asym.append(d)
        res[tag] = summarise(tag, vals, vs_floor(vals))

    # ---- the ladder ------------------------------------------------------
    print("\n" + "=" * 78)
    print("THE LADDER (one space, so these rows are comparable)")
    print("=" * 78)
    order = [k for k in ["null", "session", "sim_camera", "camera", "cross_lab",
                         "sim2real_dr", "sim2real"] if k in res]
    sess = res["session"]["mean"]
    print(f"  {'rung':14s} {'mean':>9} {'x null':>8} {'x session':>10}   range")
    for k in order:
        r = res[k]
        print(f"  {k:14s} {r['mean']:9.1f} {r['mean'] / max(floor, 1e-9):8.1f} "
              f"{r['mean'] / max(sess, 1e-9):10.1f}   "
              f"[{r['min']:.0f}, {r['max']:.0f}]")

    print(f"\n  estimator asymmetry (|A->B minus B->A|): median "
          f"{np.median(asym):.1f}, max {np.max(asym):.1f}")

    # Monotonicity is the one thing a ladder has to satisfy to be a ladder.
    means = [res[k]["mean"] for k in order]
    if all(x < y for x, y in zip(means, means[1:])):
        print("  ladder is monotone: null < session < ... < sim-to-real")
    else:
        bad = [f"{order[i]} >= {order[i + 1]}"
               for i in range(len(means) - 1) if means[i] >= means[i + 1]]
        print(f"  LADDER NOT MONOTONE at: {bad}")
        print("  A rung out of order is a result, not a bug -- report it.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"dim": args.dim, "seed": args.seed, "rungs": res,
         "asymmetry_median": float(np.median(asym))}, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
