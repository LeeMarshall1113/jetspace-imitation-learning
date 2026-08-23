#!/usr/bin/env bash
# Re-measure the trustworthy horizon without a ceiling.
#
# The first pass reported "32 steps" for real and ">48" for sim. Both were the
# largest horizon *tested*, not the horizon at which the model failed -- the
# error curve never crossed its threshold inside either range. Comparing two
# right-censored bounds says nothing about which horizon is shorter, and N3 is
# exactly that comparison, so both have to be pushed until they actually break.
#
# The step counts are also not comparable across domains: simulation runs at
# 25 Hz and the imported real data at 15 Hz, so one "step" is 80 ms in sim and
# 133 ms on real video. eval_horizon.py now reads fps from the dataset and
# reports seconds; seconds are what the comparison should be made in.
set -euo pipefail
cd "$(dirname "$0")/.."

H=${1:-96}
echo "max horizon: $H"

echo
echo "=== how many latents per episode do we actually have? ==="
python - <<PY
import glob
import numpy as np
for d in ["real_so101_teleop_cubes", "push", "pickplace", "reach"]:
    fs = sorted(glob.glob(f"cache/latents/{d}/episode_*.npy"))
    if not fs:
        print(f"{d:28s} no latents cached")
        continue
    L = [np.load(f).shape[0] for f in fs]
    fps = 15 if d.startswith("real") else 25
    print(f"{d:28s} n={len(L):3d}  latents min={min(L)} med={int(np.median(L))} "
          f"max={max(L)}  ({max(L) / (fps / 2):.1f}s at {fps}Hz)")
PY

for spec in "real_so101_teleop_cubes:real_cubes:checkpoints/real/predictor_real_cubes_seed0.pt" \
            "push:push:checkpoints/predictor_push_seed0.pt" \
            "pickplace:pickplace:checkpoints/predictor_pickplace_seed0.pt"; do
    data="${spec%%:*}"; rest="${spec#*:}"; name="${rest%%:*}"; ckpt="${rest#*:}"
    [ -f "$ckpt" ] || { echo; echo "=== $name: no checkpoint at $ckpt, skipping ==="; continue; }
    echo
    echo "=== E3 uncensored: $name ==="
    python scripts/eval_horizon.py \
        --task "$name" \
        --data "data/episodes/$data" \
        --latents "cache/latents/$data" \
        --checkpoint "$ckpt" \
        --max-horizon "$H" \
        --out "cache/e3_uncensored_${name}.json" || echo "  ($name failed at H=$H)"
done

echo
echo "=== comparison in SECONDS, not steps ==="
python - <<PY
import glob
import json
rows = []
for f in sorted(glob.glob("cache/e3_uncensored_*.json")):
    d = json.load(open(f))
    rows.append((
        d["task"], d["fps"], d["useful_horizon"], d.get("useful_horizon_seconds", 0.0),
        d["action_aware_horizon"], d.get("censored", False), d["max_horizon_tested"],
    ))
if not rows:
    print("no results")
else:
    print(f"{'task':22s} {'fps':>4} {'useful':>8} {'seconds':>9} {'aware':>7}  censored")
    for t, fps, u, s, a, c, H in rows:
        print(f"{t:22s} {fps:>4} {u:>8} {s:>9.2f} {a:>7}  {'YES (>= bound)' if c else 'no'}")
    print()
    print("A censored row is a lower bound. Do not compare it against an")
    print("uncensored row and call the difference a result.")
PY
