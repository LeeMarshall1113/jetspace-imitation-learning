#!/usr/bin/env bash
# Seeded horizons at COVERAGE-SAFE lengths, not at the data ceiling.
#
# The ceiling run was wrong. eval_horizon needs episodes holding H+1 latents,
# so pushing the horizon to each dataset's maximum silently reduced the
# evaluation to the longest recordings:
#
#     push       h=145 ->  2 of 60 episodes (3%)
#     pickplace  h=174 ->  4 of 80 episodes (5%)
#     real_cubes h=246 ->  2 of 20 episodes (10%)
#
# "push is action-blind, 0.0 +- 0.0 across seeds" was computed on two episodes.
# The seeds agreed perfectly because they were all looking at the same two
# atypical recordings, which is what perfect agreement across seeds on a
# 2-sample statistic looks like.
#
# So: the largest horizon at which at least 60% of episodes still qualify.
set -uo pipefail
cd "$(dirname "$0")/.."

SEEDS=${1:-"0 1 2"}
MINCOV=${2:-0.6}

python3 - <<PY > cache/horizon_covered.json
import glob, json
import numpy as np
out = {}
for name, lat in [("push","push_decombed"),("pickplace","pickplace_decombed"),
                  ("real_cubes","real_cubes")]:
    fs = glob.glob(f"cache/latents/{lat}/episode_*.npy")
    if not fs: continue
    L = np.array(sorted(np.load(f).shape[0] for f in fs))
    best = 8
    for h in range(8, int(L[-1])):
        if (L >= h + 1).sum() / len(L) >= $MINCOV:
            best = h
    out[name] = int(best)
json.dump(out, __import__("sys").stdout)
PY
echo "coverage-safe horizons (>=${MINCOV} of episodes):"
cat cache/horizon_covered.json; echo; echo

for s in $SEEDS; do
    for spec in "push:data/episodes/push:cache/latents/push_decombed" \
                "pickplace:data/episodes/pickplace:cache/latents/pickplace_decombed" \
                "real_cubes:data/episodes/real_so101_teleop_cubes:cache/latents/real_cubes"; do
        task="${spec%%:*}"; rest="${spec#*:}"; data="${rest%%:*}"; lat="${rest#*:}"
        [ -d "$lat" ] || lat="cache/latents/${task}"
        h=$(python3 -c "import json;print(json.load(open('cache/horizon_covered.json'))['$task'])")
        ck="checkpoints/seeds/predictor_${task}_seed${s}.pt"
        [ -f "$ck" ] || continue
        echo "--- $task seed $s  (h=$h) ---"
        python scripts/eval_horizon.py --task "$task" --data "$data" --latents "$lat" \
            --checkpoint "$ck" --max-horizon "$h" \
            --out "cache/cov_horizon_${task}_seed${s}.json" \
            2>&1 | grep -E "EPISODES USED|USEFUL|ACTION-AWARE|TOO FEW|LOW COVERAGE" | head -4
    done
done

echo
echo "=============== summary, mean +- sd ==============="
python3 - <<'PY'
import glob, json, re
from collections import defaultdict
runs = defaultdict(list)
for f in glob.glob("cache/cov_horizon_*.json"):
    m = re.match(r".*cov_horizon_(.+)_seed(\d+)\.json", f)
    if m: runs[m.group(1)].append(json.load(open(f)))
print(f"{'task':13s} {'eps':>7} {'cov':>5} {'useful':>13} {'seconds':>13} {'aware':>13}")
print("-" * 70)
for task, ds in sorted(runs.items()):
    def ms(k):
        v=[d[k] for d in ds]; mu=sum(v)/len(v)
        return mu, (sum((x-mu)**2 for x in v)/max(len(v)-1,1))**0.5
    u,us = ms("useful_horizon"); a,as_ = ms("action_aware_horizon")
    sec,ss = ms("useful_horizon_seconds")
    d0 = ds[0]
    print(f"{task:13s} {d0['episodes']:>7} {100*d0.get('coverage',0):>4.0f}% "
          f"{u:>6.1f} +-{us:<5.1f} {sec:>6.2f} +-{ss:<5.2f} {a:>6.1f} +-{as_:<5.1f}")
PY
