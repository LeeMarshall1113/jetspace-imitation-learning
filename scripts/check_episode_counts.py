#!/usr/bin/env python3
"""Per-pose episode counts for every encoder arm.

    python scripts/check_episode_counts.py push

e11_status.py reported the V-JEPA arm as partially cached: some poses hold 10
episodes and others still hold the 5 from the original R1 sweep, because the
expanded collection was killed by an OOM partway through re-encoding. An arm
that is 10 episodes on some viewpoints and 5 on others is not one arm, and a
comparison against it measures episode count as much as encoder.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ARMS = ["r1", "dinov3", "siglip2", "aimv2", "dino", "clip", "vitin1k", "r1cnn"]


def main() -> int:
    task = sys.argv[1] if len(sys.argv) > 1 else "push"
    lat = Path("cache/latents")

    print(f"{'arm':10s} {'poses':>6}  episode-count distribution across poses")
    print("-" * 66)
    short: dict[str, list[str]] = {}
    for arm in ARMS:
        dirs = sorted(lat.glob(f"{arm}_{task}__*"))
        if not dirs:
            print(f"{arm:10s} {'0':>6}  (not cached)")
            continue
        counts = Counter(len(list(d.glob("episode_*.npy"))) for d in dirs)
        dist = "  ".join(f"{n} eps x{k}" for n, k in sorted(counts.items()))
        print(f"{arm:10s} {len(dirs):>6}  {dist}")
        if len(counts) > 1:
            top = max(counts)
            short[arm] = sorted(d.name.split("__")[1] for d in dirs
                                if len(list(d.glob("episode_*.npy"))) < top)

    if short:
        print()
        for arm, poses in short.items():
            print(f"{arm}: {len(poses)} poses below the arm's own maximum")
            print(f"  {' '.join(poses)}")
        print("\nThese must be levelled before any cross-encoder comparison.")
        return 1
    print("\nevery arm is internally uniform")
    return 0


if __name__ == "__main__":
    sys.exit(main())
