#!/usr/bin/env python3
"""How far do episodes of ONE task sit from each other?

    python scripts/check_within_task_spread.py

E10's arms both landed above the mean-action floor on the simulated tasks while
the same code reached 0.416 on the real laboratories. `info.json` does not
record the randomisation config for any dataset, so the regime cannot be read
off metadata -- but it can be measured.

Split each task's episodes in half and compute the same Frechet gap E2 used for
its null rung. A real lab bolts its camera down for a session, so its halves
should sit close together. A domain-randomised simulator resamples viewpoint,
lighting and clutter every episode, so its halves should sit far apart -- and
if they do, the same action corresponds to very different images, action
prediction is a harder problem there, and both E10 arms failing says something
about the data rather than about transfer.

E2 measured the real-lab null at 34-214 and a five-viewpoint sim camera sweep at
531.9, so those are the reference points.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from align_simulator import gap_between  # noqa: E402

SETS = [
    ("push (sim)", "push"),
    ("pickplace (sim)", "pickplace"),
    ("reach (sim)", "reach"),
    ("r1_push ref pose", "r1_push__r1_ref"),
    ("A_cubes (real)", "n1b_A_cubes__ego"),
    ("H_penmug (real)", "n1b_H_penmug1__camera_2"),
]


def halves(name: str, dim: int = 32):
    files = sorted((Path("cache/latents") / name).glob("episode_*.npy"))
    if len(files) < 4:
        return None
    def pool(fs):
        out = []
        for f in fs:
            z = np.load(f).astype(np.float32)
            out.append(z.reshape(z.shape[0], -1, z.shape[-1]).mean(axis=1))
        return np.concatenate(out)
    a, b = pool(files[: len(files) // 2]), pool(files[len(files) // 2:])
    if min(len(a), len(b)) < dim + 3:
        return None
    return 0.5 * (gap_between(a, b, dim, 0) + gap_between(b, a, dim, 0))


def main() -> int:
    print(f"{'dataset':22s} {'episodes':>9} {'within-task gap':>16}")
    print("-" * 50)
    for label, name in SETS:
        d = Path("cache/latents") / name
        n = len(list(d.glob("episode_*.npy"))) if d.exists() else 0
        if not n:
            print(f"{label:22s} {'--':>9}   no latents")
            continue
        g = halves(name)
        print(f"{label:22s} {n:>9} {('n/a' if g is None else f'{g:.1f}'):>16}")

    print("\nE2 reference points, same estimator and dim:")
    print("  real-lab null (self, split by episode)   34 - 214")
    print("  sim camera sweep across 5 viewpoints          531.9")
    print("  cross-laboratory                            1228.5")
    print("\nA simulated task whose own episodes sit near or above the")
    print("five-viewpoint sweep is being resampled per episode, and is not")
    print("comparable to a fixed-camera real lab.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
