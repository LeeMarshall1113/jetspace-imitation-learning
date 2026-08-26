#!/usr/bin/env python3
"""Assert the two encoder arms cover the same episodes.

    python scripts/check_arm_parity.py n1b n1bcnn

E9 compares a V-JEPA arm against a random-CNN arm and attributes the difference
to pretraining. That attribution only holds if both arms saw the same data. The
caching run reported "9 episodes" for some labs where the V-JEPA cache holds 8,
which would silently give one arm more data than the other -- and since
`zip(latents, actions)` truncates to the shorter list without complaining, the
mismatch would never surface as an error.

Prints a table and exits non-zero if any lab disagrees.
"""

from __future__ import annotations

import sys
from pathlib import Path

LABS = ["A_cubes__ego", "B_svla__side", "C_tape__birdEye", "D_ball__front",
        "E_summer__front", "F_cup__cam_front", "G_bin__front",
        "H_penmug1__camera_2"]


def main() -> int:
    a = sys.argv[1] if len(sys.argv) > 1 else "n1b"
    b = sys.argv[2] if len(sys.argv) > 2 else "n1bcnn"

    print(f"{'lab':<22}{'npz':>5}{a:>9}{b:>9}")
    bad = []
    for lab in LABS:
        n = len(list(Path("data/episodes", f"n1b_{lab}").glob("episode_*.npz")))
        va = len(list(Path("cache/latents", f"{a}_{lab}").glob("episode_*.npy")))
        vb = len(list(Path("cache/latents", f"{b}_{lab}").glob("episode_*.npy")))
        flag = ""
        if va and vb and va != vb:
            flag = "  <-- MISMATCH"
            bad.append(lab)
        elif va and vb and (va != n or vb != n):
            flag = "  <-- differs from npz count"
        print(f"{lab:<22}{n:>5}{va:>9}{vb:>9}{flag}")

    if bad:
        print(f"\n{len(bad)} labs disagree between arms: {bad}")
        print("The encoder comparison is not matched until these agree.")
        return 1
    print("\narms are matched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
