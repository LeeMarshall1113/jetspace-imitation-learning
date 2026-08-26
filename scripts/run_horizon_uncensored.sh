#!/usr/bin/env bash
# Re-measure the trustworthy horizon without a ceiling, then attack the result.
#
# The first pass reported "32 steps" for real and ">48" for sim. Both were the
# largest horizon *tested*, not the horizon at which the model failed -- the
# error curve never crossed its threshold in either range. Comparing two
# right-censored bounds says nothing about which horizon is shorter, and N3 is
# exactly that comparison, so both have to be pushed until they actually break.
#
# The step counts are also not comparable across domains: simulation runs at
# 25 Hz and the imported real data at 15 Hz, so one "step" is 80 ms in sim and
# 133 ms on real video. eval_horizon.py now reads fps from the dataset and
# reports seconds; seconds are what the comparison should be made in.
#
# The 96-step run then produced a number too good to accept on sight: model
# error stayed near 0.65 while the do-nothing baseline sat between 2 and 8, out
# to 7.68 s. A world model that flat over that horizon is either a real result
# or a model that has collapsed onto something cheap -- the mean latent, say --
# which would also produce a low, flat error. So conservatism is re-checked at
# the SAME horizon rather than at the h=24 where it originally passed.
set -euo pipefail
cd "$(dirname "$0")/.."

H=${1:-96}
echo "max horizon: $H"

echo
echo "=== latents available per episode ==="
python - <<PY
import glob
import json
import numpy as np
for d in ["real_so101_teleop_cubes", "push", "pickplace", "reach"]:
    fs = sorted(glob.glob(f"cache/latents/{d}/episode_*.npy"))
    if not fs:
        print(f"{d:28s} no latents cached")
        continue
    L = [np.load(f).shape[0] for f in fs]
    try:
        fps = json.load(open(f"data/episodes/{d}/info.json"))["fps"]
    except Exception:
        fps = 25
    print(f"{d:28s} n={len(L):3d}  latents min={min(L)} med={int(np.median(L))} "
          f"max={max(L)}  (median {int(np.median(L)) / (fps / 2):.1f}s at {fps}Hz)")
    print(f"{'':28s} usable max-horizon (>=2 eps): "
          f"{sorted(L)[-2] - 1 if len(L) > 1 else 0}")
PY

SPECS=(
  "real_so101_teleop_cubes:real_cubes:checkpoints/real/predictor_real_cubes_seed0.pt"
  "push:push:checkpoints/predictor_push_seed0.pt"
  "pickplace:pickplace:checkpoints/predictor_pickplace_seed0.pt"
)

for spec in "${SPECS[@]}"; do
    data="${spec%%:*}"; rest="${spec#*:}"; name="${rest%%:*}"; ckpt="${rest#*:}"
    [ -f "$ckpt" ] || { echo; echo "=== $name: no checkpoint at $ckpt, skipping ==="; continue; }

    # Back off until the episodes are actually long enough. A silent skip here
    # is how the real-data horizon went unmeasured in the first place.
    for h in "$H" 64 48 32 24; do
        [ "$h" -le "$H" ] || continue
        echo
        echo "=== E3 uncensored: $name (H=$h) ==="
        if python scripts/eval_horizon.py \
            --task "$name" \
            --data "data/episodes/$data" \
            --latents "cache/latents/$data" \
            --checkpoint "$ckpt" \
            --max-horizon "$h" \
            --out "cache/e3_uncensored_${name}.json"; then
            break
        fi
        echo "  H=$h did not fit; backing off"
    done

    echo
    echo "=== conservatism at the SAME horizon: $name ==="
    python scripts/check_conservatism.py \
        --task "$name" \
        --data "data/episodes/$data" \
        --latents "cache/latents/$data" \
        --checkpoint "$ckpt" \
        --max-horizon "$H" 2>&1 | tail -22 || echo "  (conservatism failed for $name)"
done

echo
echo "================= SUMMARY: seconds, not steps ================="
python - <<PY
import glob
import json
rows = []
for f in sorted(glob.glob("cache/e3_uncensored_*.json")):
    d = json.load(open(f))
    rows.append((
        d["task"], d["fps"], d["useful_horizon"], d.get("useful_horizon_seconds", 0.0),
        d["action_aware_horizon"], d.get("censored", False),
    ))
if not rows:
    print("no results")
else:
    print(f"{'task':22s} {'fps':>4} {'useful':>8} {'seconds':>9} {'aware':>7}  censored")
    for t, fps, u, s, a, c in rows:
        print(f"{t:22s} {fps:>4} {u:>8} {s:>9.2f} {a:>7}  {'YES (lower bound)' if c else 'no'}")
    print()
    print("A censored row is a lower bound. Do not compare it against an")
    print("uncensored row and call the difference a result.")
PY
