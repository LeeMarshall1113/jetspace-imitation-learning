#!/usr/bin/env python3
"""Where has the E11 encoder comparison got to?

    python scripts/e11_status.py [task]

Prints per-encoder caching progress and, for arms that have finished, the
held-out viewpoint numbers from cache/e11_*.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# prefix, display name, release date
ARMS = [
    ("r1", "vjepa2", "2025-06"),
    ("dinov3", "dinov3", "2025-08"),
    ("siglip2", "siglip2", "2025-02"),
    ("aimv2", "aimv2", "2024-11"),
    ("dino", "dinov2", "2023-04"),
    ("clip", "clip", "2021-01"),
    ("vitin1k", "vit-in1k", "2020-10"),
    ("r1cnn", "random", "none"),
]


def main() -> int:
    task = sys.argv[1] if len(sys.argv) > 1 else "push"
    lat = Path("cache/latents")

    ref_n = len(list((lat / f"r1_{task}__r1_ref").glob("episode_*.npy")))
    print(f"task {task}; V-JEPA arm holds {ref_n} episodes (every arm must match)\n")

    print(f"  {'encoder':10s} {'released':9s} {'poses':>7} {'eps':>5}  status")
    print("  " + "-" * 52)
    for prefix, name, rel in ARMS:
        dirs = sorted(lat.glob(f"{prefix}_{task}__*"))
        done = [d for d in dirs
                if len(list(d.glob("episode_*.npy"))) >= max(ref_n, 1)]
        n_ep = len(list((lat / f"{prefix}_{task}__r1_ref").glob("episode_*.npy")))
        if not dirs:
            status = "not started"
        elif len(done) >= 23:
            status = "complete"
        else:
            status = f"caching ({len(done)}/23 done)"
        print(f"  {name:10s} {rel:9s} {len(dirs):>4}/23 {n_ep:>5}  {status}")

    print()
    results = sorted(Path("cache").glob("e11_*.json"))
    if not results:
        print("  no arms evaluated yet")
        return 0

    print(f"  {'encoder':10s} {'baseline':>18} {'multiview':>18}")
    print("  " + "-" * 48)
    rows = []
    for f in results:
        try:
            b = json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            continue
        name = f.stem.replace("e11_", "")
        base = b.get("baseline")
        multi = b.get("multiview")
        if not base or not multi:
            continue
        import statistics as st
        rows.append((name, st.mean(base), st.pstdev(base),
                     st.mean(multi), st.pstdev(multi)))
    for name, bm, bs, mm, ms in sorted(rows, key=lambda r: r[3]):
        print(f"  {name:10s} {bm:>10.3f} +-{bs:.3f} {mm:>10.3f} +-{ms:.3f}")
    if rows:
        print("\n  1.0 = no better than predicting the mean action; lower is better")
        print("  sorted by multiview (the held-out viewpoint number that matters)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
