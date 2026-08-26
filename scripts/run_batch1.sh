#!/usr/bin/env bash
# Batch 1: everything that blocks writing the paper.
#
# Three gaps, in order of how badly each would hurt at review:
#
#   1. EVERY number so far is seed 0. No result has uncertainty attached, and a
#      paper where nothing has error bars is rejected on sight regardless of how
#      good the point estimates are. Three seeds minimum, everywhere.
#
#   2. The horizon is right-censored. ">=96 steps" is the largest horizon
#      TESTED, not where the model failed, in both sim and real. N3 asks which
#      horizon is shorter and two lower bounds cannot answer it. Push until the
#      curve actually crosses.
#
#   3. The pooling ablation ran on combed latents at a censored horizon, so it
#      is currently unusable. Redone comb-free.
#
# Written to run overnight beside the N1b pipeline. Both fit in VRAM together
# (the encoder peaks at 0.79 GB of 15.9); they contend for CPU, which costs
# wall-clock and nothing else.
set -uo pipefail
cd "$(dirname "$0")/.."

SEEDS=${1:-"0 1 2"}
HMAX=${2:-256}

echo "=============================================================="
echo "  BATCH 1   seeds: $SEEDS   max horizon: $HMAX"
echo "=============================================================="

# --------------------------------------------------------------- 1. seeds --
# train_predictor already takes --seed; the gap was that nothing ever varied it.
for task in push pickplace; do
    lat="cache/latents/${task}_decombed"
    [ -d "$lat" ] || lat="cache/latents/${task}"
    for s in $SEEDS; do
        ckpt="checkpoints/seeds/predictor_${task}_seed${s}.pt"
        if [ -f "$ckpt" ]; then
            echo "### $task seed $s already trained"
            continue
        fi
        echo "### training $task seed $s  (latents: $(basename $lat))"
        python scripts/train_predictor.py --task "$task" \
            --data "data/episodes/${task}" --latents "$lat" \
            --out checkpoints/seeds --epochs 30 --seed "$s" --pca-dim 128 \
            2>&1 | grep -vE "UserWarning|self.blocks" | tail -2
    done
done

# real data too -- it is the domain the paper leans on hardest
for s in $SEEDS; do
    ckpt="checkpoints/seeds/predictor_real_cubes_seed${s}.pt"
    if [ -f "$ckpt" ]; then
        echo "### real_cubes seed $s already trained"
        continue
    fi
    echo "### training real_cubes seed $s"
    python scripts/train_predictor.py --task real_cubes \
        --data data/episodes/real_so101_teleop_cubes \
        --latents cache/latents/real_cubes \
        --out checkpoints/seeds --epochs 30 --seed "$s" --pca-dim 128 \
        2>&1 | grep -vE "UserWarning|self.blocks" | tail -2
done

# ------------------------------------------------- 2. uncensored horizons --
echo
echo "=============== horizons, pushed until they break ==============="
run_h() {
    task="$1"; data="$2"; lat="$3"; s="$4"
    ckpt="checkpoints/seeds/predictor_${task}_seed${s}.pt"
    [ -f "$ckpt" ] || { echo "  no checkpoint for $task seed $s"; return; }
    python scripts/eval_horizon.py --task "$task" --data "$data" --latents "$lat" \
        --checkpoint "$ckpt" --max-horizon "$HMAX" \
        --out "cache/b1_horizon_${task}_seed${s}.json" \
        2>&1 | grep -E "USEFUL|ACTION-AWARE|CENSORED|episodes with" | head -4
}
for s in $SEEDS; do
    for task in push pickplace; do
        lat="cache/latents/${task}_decombed"
        [ -d "$lat" ] || lat="cache/latents/${task}"
        echo "--- $task seed $s ---"
        run_h "$task" "data/episodes/${task}" "$lat" "$s"
    done
    echo "--- real_cubes seed $s ---"
    run_h real_cubes data/episodes/real_so101_teleop_cubes cache/latents/real_cubes "$s"
done

# ------------------------------------------------- 3. conservatism, seeded --
echo
echo "=============== conservatism across seeds ==============="
for s in $SEEDS; do
    for task in push pickplace; do
        lat="cache/latents/${task}_decombed"
        [ -d "$lat" ] || lat="cache/latents/${task}"
        ckpt="checkpoints/seeds/predictor_${task}_seed${s}.pt"
        [ -f "$ckpt" ] || continue
        echo -n "  $task seed $s: "
        python scripts/check_conservatism.py --task "$task" \
            --data "data/episodes/${task}" --latents "$lat" --checkpoint "$ckpt" \
            --max-horizon 96 2>&1 | grep -E "mean displacement|mean direction" | tr '\n' ' '
        echo
    done
done

# ------------------------------------------- 4. pooling ablation, comb-free --
echo
echo "=============== pooling ablation, comb-free ==============="
for g in 1 2 4 8; do
    lat="cache/latents/b1_push_g${g}"
    if [ ! -f "$lat/info.json" ]; then
        echo "### encoding push at pool_grid $g (comb-free)"
        python scripts/cache_latents.py --task "b1_push_g${g}" --data data/episodes/push \
            --out "$lat" --chunk 32 --margin 15 --pool-grid "$g" --limit 30 \
            2>&1 | grep -vE "UserWarning|self.blocks|Loading weights" | tail -1
    fi
    ckpt="checkpoints/pool/predictor_b1_push_g${g}_seed0.pt"
    if [ ! -f "$ckpt" ]; then
        python scripts/train_predictor.py --task "b1_push_g${g}" --data data/episodes/push \
            --latents "$lat" --out checkpoints/pool --epochs 30 --seed 0 --pca-dim 128 \
            2>&1 | grep -vE "UserWarning|self.blocks" | tail -1
    fi
    echo -n "  pool_grid $g: "
    python scripts/eval_horizon.py --task "b1_push_g${g}" --data data/episodes/push \
        --latents "$lat" --checkpoint "$ckpt" --max-horizon "$HMAX" \
        --out "cache/b1_pool_g${g}.json" 2>&1 | grep -E "USEFUL HORIZON" | head -1
done

# ------------------------------------------------------------- 5. summary --
echo
echo "=============================================================="
echo "  BATCH 1 SUMMARY"
echo "=============================================================="
python3 - <<'PY'
import glob
import json
import re
from collections import defaultdict

runs = defaultdict(list)
for f in glob.glob("cache/b1_horizon_*.json"):
    d = json.load(open(f))
    m = re.match(r".*b1_horizon_(.+)_seed(\d+)\.json", f)
    if m:
        runs[m.group(1)].append(d)

if runs:
    print("HORIZON, mean +- sd across seeds")
    print(f"{'task':16s} {'seeds':>6} {'useful':>16} {'seconds':>16} {'censored':>9}")
    print("-" * 70)
    for task, ds in sorted(runs.items()):
        u = [d["useful_horizon"] for d in ds]
        sec = [d.get("useful_horizon_seconds", 0.0) for d in ds]
        cens = any(d.get("censored") for d in ds)
        mu = sum(u) / len(u)
        sd = (sum((x - mu) ** 2 for x in u) / max(len(u) - 1, 1)) ** 0.5
        smu = sum(sec) / len(sec)
        ssd = (sum((x - smu) ** 2 for x in sec) / max(len(sec) - 1, 1)) ** 0.5
        print(f"{task:16s} {len(ds):>6} {mu:>8.1f} +-{sd:<5.1f} "
              f"{smu:>8.2f} +-{ssd:<5.2f} {'YES' if cens else 'no':>9}")
    print()
    if any(d.get("censored") for ds in runs.values() for d in ds):
        print("  STILL CENSORED somewhere. N3 needs a larger --max-horizon or")
        print("  longer episodes; do not compare a bound against a measurement.")
    else:
        print("  No censoring. Every horizon is a measurement, and the sim-vs-real")
        print("  comparison N3 asks for is finally available.")

pool = []
for f in sorted(glob.glob("cache/b1_pool_g*.json")):
    d = json.load(open(f))
    g = re.search(r"g(\d+)", f).group(1)
    pool.append((int(g), d["useful_horizon"], d.get("censored", False)))
if pool:
    print("\nPOOLING (comb-free)")
    print(f"{'pool_grid':>10} {'useful horizon':>16} {'censored':>9}")
    for g, u, c in sorted(pool):
        print(f"{g:>10} {u:>16} {'YES' if c else 'no':>9}")
PY
