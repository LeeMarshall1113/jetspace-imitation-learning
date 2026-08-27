#!/usr/bin/env python3
"""Does the winning encoder actually separate from the runner-up?

    python scripts/e11_compare.py

The E11 table sorts eight encoders by held-out viewpoint error. A ranking is not
a result: with three seeds, the top two can differ by less than their own spread
and the order would flip on a re-run. This prints the per-seed values, tests
whether the leader's interval clears the runner-up's, and counts per-seed wins,
which is the honest form of "X is better than Y" at n=3.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RELEASED = {
    "vjepa2": ("2025-06", "video SSL", 326),
    "dinov3": ("2025-08", "image SSL", 86),
    "siglip2": ("2025-02", "image-text", 93),
    "aimv2": ("2024-11", "autoregressive", 309),
    "dinov2": ("2023-04", "image SSL", 87),
    "clip": ("2021-01", "image-text", 86),
    "vit-in1k": ("2020-10", "supervised", 86),
    "vc1": ("2023-06", "robot MAE", 86),
    "random": ("none", "none", 7),
}


def main() -> int:
    arms = {}
    for f in sorted(Path("cache").glob("e11_*.json")):
        blob = json.loads(f.read_text())
        if blob.get("multiview"):
            arms[f.stem[4:]] = np.asarray(blob["multiview"], dtype=float)
    if len(arms) < 2:
        print("need at least two evaluated arms")
        return 1

    order = sorted(arms, key=lambda k: arms[k].mean())
    print(f"{'encoder':10s} {'released':9s} {'kind':15s} {'M':>4}  "
          f"{'per-seed':28s} mean")
    print("-" * 82)
    for k in order:
        rel, kind, params = RELEASED.get(k, ("?", "?", 0))
        vals = np.round(arms[k], 3)
        print(f"{k:10s} {rel:9s} {kind:15s} {params:>4}  "
              f"{str(vals):28s} {arms[k].mean():.3f}")

    top, second = order[0], order[1]
    a, b = arms[top], arms[second]
    print()
    print(f"leader {top} vs runner-up {second}")
    print(f"  difference          {b.mean() - a.mean():+.4f}")
    print(f"  {top:8s} interval  [{a.mean() - 1.96 * a.std():.3f}, "
          f"{a.mean() + 1.96 * a.std():.3f}]")
    print(f"  {second:8s} interval  [{b.mean() - 1.96 * b.std():.3f}, "
          f"{b.mean() + 1.96 * b.std():.3f}]")
    sep = a.mean() + 1.96 * a.std() < b.mean() - 1.96 * b.std()
    print(f"  intervals separate  {sep}")
    print(f"  per-seed wins       {int((a < b).sum())}/3")
    if not sep:
        print("\n  The leader does NOT separate from the runner-up. The ranking")
        print("  is a ranking, not a demonstrated difference.")

    # Does pretraining beat no pretraining at all?
    if "random" in arms:
        r = arms["random"]
        beaten = [k for k in order if k != "random"
                  and arms[k].mean() + 1.96 * arms[k].std() < r.mean() - 1.96 * r.std()]
        worse = [k for k in order if k != "random" and arms[k].mean() > r.mean()]
        print(f"\n  beat random with separated intervals: {len(beaten)}/"
              f"{len(arms) - 1}  {beaten}")
        if worse:
            print(f"  WORSE than random features: {worse}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
