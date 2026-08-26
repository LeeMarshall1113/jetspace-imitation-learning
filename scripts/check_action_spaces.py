#!/usr/bin/env python3
"""Are the simulated and real action spaces actually interchangeable?

    python scripts/check_action_spaces.py

**Blocking check B1.** I have repeatedly described our simulated SO-101 and the
public SO-101 datasets as sharing a "byte-identical" action space, on the
grounds that both are six joints in the same order with the same names. That
was an assumption stated as a measurement, and it was downgraded to "nominally
identical" once the adversarial audit pointed out that matching names is not
matching semantics.

This measures it. No hardware required: if two datasets encode the same joint
in different units, their recorded value ranges cannot agree.

What to look for, in order of how badly it breaks things:

  units      MuJoCo joints are radians, so a shoulder sweep spans about +-3.14.
             LeRobot commonly records servo positions in DEGREES or in a
             normalised [-100, 100] scale. A 57.3x mismatch would make any
             cross-domain policy or world model transfer meaningless while
             leaving every within-domain result intact -- which is exactly the
             kind of defect that survives until someone tries to deploy.
  offsets    Same units, shifted zero. Shows up as a displaced midpoint with a
             comparable span.
  gripper    Frequently parametrised differently from the arm joints -- degrees
             for the revolute joints and a 0-1 or 0-100 opening fraction for
             the gripper is a common combination.

Prints per-joint range, midpoint and span for every episode dataset on disk,
then flags the pairs whose spans differ by more than 3x.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jetspace.data.episode import EpisodeDataset  # noqa: E402

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex",
          "wrist_flex", "wrist_roll", "gripper"]


def summarise(root: Path, max_eps: int = 10) -> dict | None:
    try:
        ds = EpisodeDataset(root)
    except Exception:  # noqa: BLE001
        return None
    if not len(ds):
        return None
    acts = []
    for i in range(min(max_eps, len(ds))):
        acts.append(ds[i]["action"])
    a = np.concatenate(acts, axis=0)
    return {
        "name": root.name,
        "real": bool(ds.info.get("real_robot", False)),
        "source": ds.info.get("source", "simulation"),
        "n": len(a),
        "min": a.min(axis=0),
        "max": a.max(axis=0),
    }


def guess_units(span: np.ndarray) -> str:
    """Crude but decisive: what scale are these numbers on?"""
    arm = float(np.median(span[:5]))
    if arm < 0.5:
        return "small (normalised?)"
    if arm <= 7.0:
        return "RADIANS"
    if arm <= 400.0:
        return "DEGREES or [-100,100]"
    return "large (raw ticks?)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/episodes")
    ap.add_argument("--episodes", type=int, default=10)
    args = ap.parse_args()

    roots = sorted(Path(p).parent for p in glob.glob(f"{args.root}/*/info.json"))
    rows = [r for r in (summarise(p, args.episodes) for p in roots) if r]
    if not rows:
        print(f"no episode datasets under {args.root}")
        return 1

    print(f"{'dataset':30s} {'kind':5s} {'units':22s}  per-joint span (max-min)")
    print("-" * 110)
    for r in rows:
        span = r["max"] - r["min"]
        kind = "real" if r["real"] else "sim"
        spans = "  ".join(f"{s:7.2f}" for s in span[:6])
        print(f"{r['name'][:30]:30s} {kind:5s} {guess_units(span):22s}  {spans}")

    print(f"\n{'dataset':30s}  per-joint midpoint (zero offset)")
    print("-" * 110)
    for r in rows:
        mid = (r["max"] + r["min"]) / 2.0
        print(f"{r['name'][:30]:30s}  " + "  ".join(f"{m:7.2f}" for m in mid[:6]))

    # --- the verdict --------------------------------------------------------
    sims = [r for r in rows if not r["real"]]
    reals = [r for r in rows if r["real"]]
    print("\n" + "=" * 70)
    if not sims or not reals:
        print("Need both simulated and real datasets to compare. Nothing to say.")
        return 0

    s_span = np.median([r["max"] - r["min"] for r in sims], axis=0)
    r_span = np.median([r["max"] - r["min"] for r in reals], axis=0)
    ratio = r_span / np.maximum(s_span, 1e-9)

    print("real span / sim span, per joint:")
    for j, name in enumerate(JOINTS[: len(ratio)]):
        flag = "  <-- MISMATCH" if (ratio[j] > 3.0 or ratio[j] < 1 / 3.0) else ""
        print(f"  {name:16s} {ratio[j]:8.2f}x{flag}")

    arm_ratio = float(np.median(ratio[:5]))
    print()
    if 0.75 <= arm_ratio <= 1.33:
        print("VERDICT: the arm joints share a scale. Units are consistent, and")
        print("  the interchangeability claim survives this test (offsets and")
        print("  sign conventions are still unchecked).")
    else:
        print("VERDICT: THE ACTION SPACES ARE NOT INTERCHANGEABLE. Arm joints")
        print(f"  differ by {arm_ratio:.1f}x.")
        if 40.0 <= arm_ratio <= 75.0:
            print(f"  {arm_ratio:.1f} is close to 180/pi = 57.3, so this is radians")
            print("  versus degrees. Conversion is mechanical, but nothing that")
            print("  crosses the domains is valid until it is applied.")
        print()
        print("  Within-domain results are unaffected: E2, E3 and the horizon")
        print("  numbers each train and evaluate inside one domain, so a global")
        print("  scale factor cancels. What breaks is anything CROSSING them --")
        print("  N2's sim-trained world model on real video, and any policy")
        print("  trained on public data and deployed on our own arm.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
