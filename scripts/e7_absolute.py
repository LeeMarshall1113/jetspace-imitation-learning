#!/usr/bin/env python3
"""E7 pilot diagnostic: retention is a ratio, and the ratio hides the answer.

    python scripts/e7_absolute.py

`retention = ref_mse / pose_mse` was registered as the measure. The pilot shows
it is the wrong one, for a reason that is arithmetic rather than empirical: it
divides by how well an arm fitted its own training viewpoint. An arm that
memorises the reference harder gets a SMALLER retention for the same displaced
error. The metric therefore penalises exactly the behaviour it was meant to
detect, and two arms with identical displaced-pose error can differ in
retention purely because one fitted the reference better.

Absolute normalised MSE at the displaced poses has no such dependence. 1.0
means no better than predicting the mean action; lower is better.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "cache/e7_push.json")
    b = json.loads(path.read_text())

    print("=" * 70)
    print(f"E7 pilot -- absolute vs ratio ({path})")
    print("=" * 70)
    print(f"{'arm':6s} {'ref MSE':>9} {'displaced MSE':>16} {'retention':>11}")
    print("-" * 46)

    out = {}
    for arm in ("vjepa", "rand"):
        refs, disp = [], []
        for sd in b["results"][arm]["seeds"]:
            refs.append(sd["ref_mse"])
            disp.append(float(np.mean([v["mse"] for p, v in sd["poses"].items()
                                       if p != "r1_ref"])))
        refs, disp = np.array(refs), np.array(disp)
        out[arm] = (refs, disp)
        print(f"{arm:6s} {refs.mean():9.4f} {disp.mean():11.3f} +-{disp.std():.3f} "
              f"{np.mean(refs / disp):11.4f}")
        print(f"       per-seed displaced: {np.round(disp, 3)}")

    rv, rr = out["vjepa"][1], out["rand"][1]
    print()
    print("normalised MSE 1.0 = no better than predicting the mean action")
    print(f"\nON ABSOLUTE DISPLACED ERROR: vjepa {rv.mean():.3f} vs "
          f"rand {rr.mean():.3f}   ({'vjepa' if rv.mean() < rr.mean() else 'rand'} better "
          f"by {abs(rv.mean() - rr.mean()):.3f})")
    sep = (rv.mean() + 1.96 * rv.std()) < (rr.mean() - 1.96 * rr.std())
    print(f"  intervals separate across 3 seeds: {sep}")

    print(f"\nON THE REGISTERED RATIO: vjepa {np.mean(out['vjepa'][0] / rv):.4f} vs "
          f"rand {np.mean(out['rand'][0] / rr):.4f}")
    print("  The two measures do not disagree about the arms -- they disagree")
    print("  about the margin, because rand fitted the reference pose better")
    print(f"  ({out['rand'][0].mean():.4f} vs {out['vjepa'][0].mean():.4f}) and the "
          "ratio divides that in.")

    # The memorisation problem, which is separate from the metric problem.
    print("\nMEMORISATION CHECK")
    for arm in ("vjepa", "rand"):
        r = out[arm][0].mean()
        print(f"  {arm:6s} reference MSE {r:.4f} -- explains "
              f"{100 * (1 - r):.1f}% of action variance at the training viewpoint")
    print("  Both arms fit the reference pose almost perfectly on 398 samples.")
    print("  That is memorisation, and it is why displaced error is so high for")
    print("  both. More data is necessary but may not be sufficient; the head")
    print("  needs regularisation or the comparison stays in this regime.")
    return 0




def growth(path: str = "cache/e7_push.json") -> None:
    """Does the advantage grow with camera displacement?

    This is the difference between two very different claims. If V-JEPA's
    advantage is flat across displacement, it is simply a better feature space
    and the viewpoint framing is wrong. If the advantage widens as the camera
    moves further, that is viewpoint robustness specifically -- which is the
    claim this project was founded on.
    """
    import sys as _sys
    _sys.path.insert(0, "src")
    from jetspace.envs.so101_env import r1_displacement

    b = json.loads(Path(path).read_text())
    per = {}
    for arm in ("vjepa", "rand"):
        acc: dict[str, list] = {}
        for sd in b["results"][arm]["seeds"]:
            for p, v in sd["poses"].items():
                if p != "r1_ref":
                    acc.setdefault(p, []).append(v["mse"])
        per[arm] = {p: float(np.mean(x)) for p, x in acc.items()}

    poses = sorted(per["vjepa"], key=lambda p: r1_displacement(p)["angle"])
    n = len(poses) // 3
    bands = [("near", poses[:n]), ("mid", poses[n:2 * n]), ("far", poses[2 * n:])]

    print("\n" + "=" * 70)
    print("DOES THE ADVANTAGE GROW WITH DISPLACEMENT?")
    print("=" * 70)
    print(f"  {'band':6s} {'poses':>6} {'angle':>12} {'vjepa':>8} {'rand':>8} {'advantage':>10}")
    advs = []
    for name, ps in bands:
        a = np.mean([per["vjepa"][p] for p in ps])
        r = np.mean([per["rand"][p] for p in ps])
        angs = [r1_displacement(p)["angle"] for p in ps]
        advs.append(r - a)
        print(f"  {name:6s} {len(ps):6d} {min(angs):5.0f}-{max(angs):<5.0f}° "
              f"{a:8.3f} {r:8.3f} {r - a:+10.3f}")
    print(f"\n  near -> far advantage: {advs[0]:+.3f} -> {advs[-1]:+.3f} "
          f"(growth {advs[-1] - advs[0]:+.3f})")
    if advs[-1] > advs[0]:
        print("  The advantage WIDENS with displacement: consistent with")
        print("  viewpoint robustness rather than a constant offset in fit.")
    else:
        print("  The advantage does NOT widen. V-JEPA is better here, but this")
        print("  looks like a better feature space generally rather than")
        print("  viewpoint robustness specifically. The distinction matters:")
        print("  only the second supports the premise this project was built on.")


if __name__ == "__main__":
    rc = main()
    growth()
    raise SystemExit(rc)
