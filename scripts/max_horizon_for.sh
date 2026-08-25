#!/usr/bin/env bash
# Largest horizon a latent cache can actually support, per task.
#
# check_conservatism needs >=2 episodes holding H+1 latents. Passing a horizon
# the episodes cannot support returns "found 0" and scores nothing -- which is
# how the reach arm of H1 trained three world models and then measured nothing.
cd "$(dirname "$0")/.."
python3 - <<'PY'
import glob
import numpy as np
for task in ("push", "pickplace", "reach"):
    fs = glob.glob(f"cache/latents/r1_{task}__r1_ref/episode_*.npy")
    if not fs:
        print(f"{task:12s} no r1 reference cache")
        continue
    L = sorted(np.load(f).shape[0] for f in fs)
    ceiling = L[-2] - 1 if len(L) > 1 else 0
    # Leave headroom so most episodes contribute, not just the two longest --
    # the coverage lesson from eval_horizon.
    safe = max(8, int(np.percentile(L, 25)) - 1)
    print(f"{task:12s} {len(L):2d} eps  latents med {int(np.median(L)):3d} "
          f"max {L[-1]:3d}  ceiling {ceiling:3d}  coverage-safe {safe:3d}")
PY
