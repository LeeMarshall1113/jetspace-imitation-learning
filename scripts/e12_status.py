#!/usr/bin/env python3
"""How far has E12 got?

    python scripts/e12_status.py [task]

Collection progress per condition, then encoder coverage per condition.
"""

from __future__ import annotations

import sys
from pathlib import Path

CONDS = ["ref",
         "lighting_0p3", "lighting_0p45", "lighting_0p55", "lighting_0p62",
         "texture_0p06", "texture_0p1", "texture_0p16", "texture_0p24",
         "clutter_1", "clutter_2", "clutter_3", "clutter_4"]

ARMS = [("vjepa2", "r1"), ("dinov3", "dinov3"), ("siglip2", "siglip2"),
        ("aimv2", "aimv2"), ("dinov2", "dino"), ("clip", "clip"),
        ("vit-in1k", "vitin1k"), ("vc1", "vc1"), ("random", "r1cnn")]


def main() -> int:
    task = sys.argv[1] if len(sys.argv) > 1 else "push"
    print(f"{'condition':16s} {'episodes':>9}   encoders cached")
    print("-" * 62)
    done_conds = 0
    for c in CONDS:
        eps = len(list(Path("data/episodes", f"e12_{task}__{c}")
                       .glob("episode_*.npz"))) if Path(
            "data/episodes", f"e12_{task}__{c}").exists() else 0
        got = []
        for name, prefix in ARMS:
            d = Path("cache/latents", f"{prefix}_e12_{task}__{c}")
            if d.exists() and len(list(d.glob("episode_*.npy"))) >= max(eps, 1):
                got.append(name)
        if eps:
            done_conds += 1
        print(f"{c:16s} {eps:>9}   {len(got)}/9" +
              (f"  {' '.join(got)}" if got else ""))

    print(f"\n{done_conds}/{len(CONDS)} conditions collected")
    total = len(CONDS) * len(ARMS)
    have = sum(
        1 for c in CONDS for _, p in ARMS
        if Path("cache/latents", f"{p}_e12_{task}__{c}").exists()
        and list(Path("cache/latents", f"{p}_e12_{task}__{c}").glob("episode_*.npy")))
    print(f"{have}/{total} encoder-condition cells cached")
    return 0


if __name__ == "__main__":
    sys.exit(main())
