#!/usr/bin/env python3
"""Do the encoder arms emit comparable feature caches?

    python scripts/check_encoder_parity.py r1_push dino_push clip_push

E8 attributes its result to the encoder. That attribution only holds if the arms
differ in nothing else -- same episodes, same timesteps, same spatial grid.
`zip()` over mismatched caches truncates silently and `check_arm_parity.py`
already had to be written once for exactly this, so it is checked rather than
assumed.

Feature DIMENSION is allowed to differ (V-JEPA 2 is 1024-wide, DINOv2-base 768)
because every arm is PCA-projected to a common dimension downstream. Episode
count, per-episode timesteps and spatial grid are not allowed to differ.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

POSE = "r1_ref"


def describe(prefix: str):
    d = Path("cache/latents") / f"{prefix}__{POSE}"
    files = sorted(d.glob("episode_*.npy"))
    if not files:
        return None
    shapes = [np.load(f, mmap_mode="r").shape for f in files]
    return {"n": len(files), "steps": [s[0] for s in shapes],
            "grid": shapes[0][1:3], "dim": shapes[0][-1]}


def main() -> int:
    prefixes = sys.argv[1:] or ["r1_push"]
    info = {}
    print(f"{'prefix':22s} {'eps':>4} {'grid':>8} {'dim':>6}   timesteps")
    for p in prefixes:
        d = describe(p)
        if d is None:
            print(f"{p:22s}   no cache at cache/latents/{p}__{POSE}")
            continue
        info[p] = d
        print(f"{p:22s} {d['n']:>4} {str(d['grid']):>8} {d['dim']:>6}   "
              f"{d['steps']}")

    if len(info) < 2:
        return 0

    ref_name, ref = next(iter(info.items()))
    bad = []
    for name, d in info.items():
        if name == ref_name:
            continue
        if d["n"] != ref["n"]:
            bad.append(f"{name}: {d['n']} episodes vs {ref['n']}")
        if d["steps"] != ref["steps"]:
            bad.append(f"{name}: timesteps differ from {ref_name}")
        if d["grid"] != ref["grid"]:
            bad.append(f"{name}: grid {d['grid']} vs {ref['grid']}")

    print()
    if bad:
        print("NOT COMPARABLE:")
        for b in bad:
            print(f"  {b}")
        print("Any encoder difference measured across these arms is "
              "confounded with the mismatch above.")
        return 1
    dims = {n: d["dim"] for n, d in info.items()}
    print(f"arms are comparable (feature widths {dims} differ but are "
          f"PCA-projected to a common dimension downstream)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
