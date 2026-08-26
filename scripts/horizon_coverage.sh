#!/usr/bin/env bash
# How many episodes actually contribute at each horizon?
#
# eval_horizon needs episodes holding H+1 latents, so raising the horizon
# silently restricts the evaluation to the LONGEST episodes -- which for a
# scripted task are the atypical ones where the expert took the longest route.
# A horizon measured at the data ceiling can rest on two episodes and still
# print a confident number.
cd "$(dirname "$0")/.."
python3 - <<'PY'
import glob
import numpy as np

for name, lat in [("push", "push_decombed"), ("pickplace", "pickplace_decombed"),
                  ("real_cubes", "real_cubes")]:
    fs = glob.glob(f"cache/latents/{lat}/episode_*.npy")
    if not fs:
        continue
    L = np.array(sorted(np.load(f).shape[0] for f in fs))
    print(f"{name:12s} {len(L):3d} episodes  latents: med {int(np.median(L))}  max {L[-1]}")
    for h in (48, 96, 145, 174, 246):
        n = int((L >= h + 1).sum())
        if n:
            flag = "   <-- too few to trust" if n < 5 else ""
            print(f"     h={h:4d}: {n:3d} episodes qualify ({100*n/len(L):3.0f}%){flag}")
    print()
PY
