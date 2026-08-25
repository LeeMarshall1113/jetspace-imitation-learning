#!/usr/bin/env python3
"""Refuse a rollout horizon that would silently score zero episodes.

    python scripts/check_horizon.py cache/latents/r1_push__r1_ref auto
    python scripts/check_horizon.py cache/latents/r1_push__r1_ref 48

Prints one line: `OK <h>`, `AUTO <h>`, `LOW <message>` or `ERR <message>`.

This trap has fired four times in this project. h=145 kept 2 of 60 push
episodes; reach at h=32 scored zero; pickplace at h=48 kept 1 of 4 while the
world models trained perfectly well and the run printed "0 poses scored" three
times in a row before anyone noticed.

The failure is silent by construction. A horizon longer than an episode does
not raise -- it simply yields no scorable windows, and a loop over 23 poses
produces 23 empty results and a summary saying there is nothing to correlate.
Every one of those runs looked like a completed experiment. So the check has to
happen before anything trains, and it has to refuse rather than warn.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

#: The horizons the evaluators actually use. Picking from a fixed ladder keeps
#: results comparable across tasks instead of every task getting its own
#: arbitrary maximum.
LADDER = (8, 16, 24, 32, 48, 64, 96, 128)


def main() -> int:
    if len(sys.argv) < 3:
        print("ERR usage: check_horizon.py <latent-dir> <horizon|auto>")
        return 1
    ref, want = sys.argv[1], sys.argv[2]

    lengths = sorted(int(np.load(f, mmap_mode="r").shape[0])
                     for f in Path(ref).glob("episode_*.npy"))
    if not lengths:
        print(f"ERR no episodes in {ref}")
        return 1

    # An episode contributes a scorable window only if it is strictly longer
    # than the horizon, so the binding constraint is the SHORTEST episode, not
    # the mean. Reporting the mean is how a set like [29, 36, 43, 130] looks
    # comfortable at h=48.
    best = max((h for h in LADDER if h < lengths[0]), default=0)

    if want == "auto":
        if best:
            print(f"AUTO {best}")
        else:
            print(f"ERR shortest episode is {lengths[0]} latents; "
                  f"no horizon on the ladder fits")
        return 0

    h = int(want)
    keep = sum(1 for x in lengths if x > h)
    if keep == len(lengths):
        print(f"OK {h}")
    elif best:
        print(f"LOW only {keep}/{len(lengths)} episodes survive h={h} "
              f"(shortest is {lengths[0]}); largest safe horizon is {best}")
    else:
        print(f"LOW only {keep}/{len(lengths)} episodes survive h={h} "
              f"and no ladder horizon fits (shortest is {lengths[0]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
