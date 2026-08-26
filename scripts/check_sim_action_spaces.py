#!/usr/bin/env python3
"""Are the simulated tasks actually on one action space?

    python scripts/check_sim_action_spaces.py

E9 found cross-task transfer fails on eight real laboratories, but those labs
differ in BOTH task and action convention (ledger L8: per-dimension spread
differing up to 5x, joint zero-offsets by ~140 units). That confound means E9
cannot say whether representation transfer failed or whether the action spaces
simply disagreed.

push, pickplace and reach all run on the same simulated SO-101 through the same
env, so they should share an action space exactly. This asserts that rather
than assuming it -- the whole point of the follow-up experiment is that one
variable is held fixed, and "should" is how ledger L8 got written in the first
place, when a byte-identical action space was asserted for weeks and turned out
false when measured.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TASKS = ["push", "pickplace", "reach"]


def main() -> int:
    print(f"{'task':12s} {'eps':>4} {'dim':>4} {'min':>9} {'max':>9}   per-dim std")
    stats = {}
    for t in TASKS:
        files = sorted(Path("data/episodes", t).glob("episode_*.npz"))
        if not files:
            print(f"{t:12s} no episodes")
            continue
        a = np.concatenate([np.load(f)["action"] for f in files])
        stats[t] = a
        print(f"{t:12s} {len(files):>4} {a.shape[1]:>4} {a.min():9.3f} "
              f"{a.max():9.3f}   {np.round(a.std(0), 3)}")

    if len(stats) < 2:
        return 1

    print()
    dims = {t: a.shape[1] for t, a in stats.items()}
    if len(set(dims.values())) != 1:
        print(f"DIMENSIONS DIFFER: {dims} -- not one action space.")
        return 1
    print(f"all tasks are {next(iter(dims.values()))}-dimensional")

    # Spread ratio is what mattered on the real labs: a 5x difference meant a
    # pretrained head was dominated by the loudest lab. Report the worst ratio
    # per dimension across tasks.
    mat = np.stack([a.std(0) for a in stats.values()])
    ratio = mat.max(0) / np.maximum(mat.min(0), 1e-9)
    print(f"per-dimension spread ratio across tasks: {np.round(ratio, 2)}")
    print(f"worst ratio {ratio.max():.2f}x   "
          f"(the eight real labs reached ~5x, which E9 could not separate "
          f"from task difference)")
    if ratio.max() < 2.0:
        print("\nThese tasks are close enough to one convention that a "
              "cross-task\nresult here is about REPRESENTATION, not action "
              "rescaling.")
    else:
        print("\nSpread still differs enough to be a partial confound. Report "
              "it\nrather than claiming the axis is clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
