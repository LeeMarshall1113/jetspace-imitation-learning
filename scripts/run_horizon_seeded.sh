#!/usr/bin/env bash
# Seeded horizon curves at the largest horizon the DATA can support.
#
# Batch 1 asked for horizon 256 and every task returned:
#
#     Need >=2 episodes with at least 257 latents; found 0
#
# which is not a failure of the model. Our episodes are 8-10 seconds; 257
# latents is 514 frames, over 20 seconds at 25 Hz. The horizon cannot be
# measured beyond the length of the episodes it is measured on.
#
# That reframes N3 rather than blocking it. "The trustworthy horizon exceeds
# every episode we have" is a real statement with a stated cause, and it is
# more honest than a number produced by padding episodes with idle frames --
# once the task finishes the scene is static, prediction is trivial, and the
# horizon measured on that tail would be an artifact of nothing happening.
#
# So: find the largest horizon each dataset actually supports, run there across
# seeds, and report with error bars.
set -uo pipefail
cd "$(dirname "$0")/.."

SEEDS=${1:-"0 1 2"}

echo "=============== largest horizon each dataset supports ==============="
python3 - <<'PY'
import glob
import json
import os
import numpy as np

SETS = {
    "push": "cache/latents/push_decombed",
    "pickplace": "cache/latents/pickplace_decombed",
    "real_cubes": "cache/latents/real_cubes",
}
out = {}
for task, lat in SETS.items():
    if not os.path.isdir(lat):
        lat = lat.replace("_decombed", "")
    fs = glob.glob(f"{lat}/episode_*.npy")
    if not fs:
        print(f"{task:12s} no latents")
        continue
    L = sorted(np.load(f).shape[0] for f in fs)
    # eval_horizon needs >=2 episodes holding H+1 latents, so the second
    # longest episode sets the ceiling.
    ceiling = L[-2] - 1
    try:
        fps = json.load(open(f"data/episodes/{task}/info.json"))["fps"]
    except Exception:
        fps = json.load(open("data/episodes/real_so101_teleop_cubes/info.json"))["fps"]
    print(f"{task:12s} {len(L):3d} episodes  latents min {L[0]} med {int(np.median(L))} "
          f"max {L[-1]}  -> max horizon {ceiling}  ({ceiling/(fps/2):.1f}s)")
    out[task] = ceiling
json.dump(out, open("cache/horizon_ceilings.json", "w"))
PY

CEIL=cache/horizon_ceilings.json
run() {
    task="$1"; data="$2"; lat="$3"; s="$4"
    h=$(python3 -c "import json;print(json.load(open('$CEIL')).get('$task',96))")
    ckpt="checkpoints/seeds/predictor_${task}_seed${s}.pt"
    [ -f "$ckpt" ] || { echo "  no checkpoint: $ckpt"; return; }
    python scripts/eval_horizon.py --task "$task" --data "$data" --latents "$lat" \
        --checkpoint "$ckpt" --max-horizon "$h" \
        --out "cache/b1_horizon_${task}_seed${s}.json" \
        2>&1 | grep -E "USEFUL|ACTION-AWARE|CENSORED" | head -3
}

echo
echo "=============== horizons, seeded, at the data ceiling ==============="
for s in $SEEDS; do
    for task in push pickplace; do
        lat="cache/latents/${task}_decombed"
        [ -d "$lat" ] || lat="cache/latents/${task}"
        echo "--- $task seed $s ---"
        run "$task" "data/episodes/${task}" "$lat" "$s"
    done
    echo "--- real_cubes seed $s ---"
    run real_cubes data/episodes/real_so101_teleop_cubes cache/latents/real_cubes "$s"
done

echo
echo "=============== summary, mean +- sd across seeds ==============="
python3 - <<'PY'
import glob
import json
import re
from collections import defaultdict

runs = defaultdict(list)
for f in glob.glob("cache/b1_horizon_*.json"):
    m = re.match(r".*b1_horizon_(.+)_seed(\d+)\.json", f)
    if m:
        runs[m.group(1)].append(json.load(open(f)))

if not runs:
    print("no horizon results")
else:
    print(f"{'task':14s} {'seeds':>5} {'useful':>14} {'seconds':>14} "
          f"{'aware':>14} {'ceiling?':>9}")
    print("-" * 76)
    for task, ds in sorted(runs.items()):
        def ms(key):
            v = [d[key] for d in ds]
            mu = sum(v) / len(v)
            sd = (sum((x - mu) ** 2 for x in v) / max(len(v) - 1, 1)) ** 0.5
            return mu, sd
        u, us = ms("useful_horizon")
        a, as_ = ms("action_aware_horizon")
        sec, secs = ms("useful_horizon_seconds")
        cens = any(d.get("censored") for d in ds)
        print(f"{task:14s} {len(ds):>5} {u:>7.1f} +-{us:<5.1f} {sec:>7.2f} +-{secs:<5.2f} "
              f"{a:>7.1f} +-{as_:<5.1f} {'YES' if cens else 'no':>9}")
    print()
    print("'ceiling? YES' means the curve never crossed inside the horizons the")
    print("EPISODES could support -- the model outlasts the data. That is a lower")
    print("bound with a stated cause, not a measurement, and N3 cannot compare")
    print("two such bounds. Longer episodes are the only fix, and padding with")
    print("idle frames after the task ends would not be one.")
PY
