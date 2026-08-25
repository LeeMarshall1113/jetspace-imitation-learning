#!/usr/bin/env bash
# H1: score every R1 pose across seeds, so the gap->degradation result can be
# hardened per docs/prereg-h1.md.
#
# Train the world model at the reference pose only, then evaluate that same
# model at every displaced pose. Normalisation and PCA basis travel with the
# checkpoint, so each displaced pose is genuinely out of distribution, and all
# poses share episodes and rollouts -- only the camera differs.
set -uo pipefail
cd "$(dirname "$0")/.."

TASK=${1:-push}
SEEDS=${2:-"0 1 2"}
HMAX=${3:-64}
DATA="data/episodes/r1_${TASK}"
CK=checkpoints/h1
mkdir -p logs "$CK"

REF="cache/latents/r1_${TASK}__r1_ref"
[ -d "$REF" ] || { echo "no reference latents at $REF -- run run_r1.sh $TASK"; exit 1; }

# ---- horizon coverage, checked BEFORE anything trains ---------------------
# Four silent zero-score runs came from horizons longer than the episodes.
# check_horizon.py refuses rather than warns; pass HMAX=auto to let it pick.
COVER=$(python3 scripts/check_horizon.py "$REF" "$HMAX")
case "$COVER" in
    "OK "*)   HMAX=${COVER#OK } ;;
    "AUTO "*) HMAX=${COVER#AUTO }; echo "horizon auto-selected: $HMAX" ;;
    "LOW "*)  echo "REFUSING: ${COVER#LOW }"
              echo "  Scoring here would return 0 poses without erroring."
              echo "  Re-run with HMAX=auto, or collect longer episodes."
              exit 1 ;;
    *)        echo "coverage check failed: $COVER"; exit 1 ;;
esac
echo "horizon $HMAX covers all $(ls "$REF"/episode_*.npy | wc -l) episodes

POSES=$(python3 -c "
import sys; sys.path.insert(0,'src')
from jetspace.envs.so101_env import R1_POSES
print(' '.join(R1_POSES))")

for s in $SEEDS; do
    ck="$CK/predictor_h1_${TASK}_s${s}.pt"
    if [ ! -f "$ck" ]; then
        echo "### training world model on the reference pose, seed $s"
        python scripts/train_predictor.py --task "h1_${TASK}_s${s}" --data "$DATA" \
            --latents "$REF" --out "$CK" --epochs 30 --seed "$s" --pca-dim 128 \
            > "logs/h1_train_${TASK}_${s}.log" 2>&1
        gain=$(grep -aoE "\([0-9.]+x better" "logs/h1_train_${TASK}_${s}.log" | tail -1)
        echo "    ${gain:-no gain line}"
        # Registered invalidation: a degradation curve measured from a model
        # that never worked is the R2 failure repeated.
        if grep -aqE "\(0\.[0-9]+x better" "logs/h1_train_${TASK}_${s}.log"; then
            echo "    INVALID: gain ratio below 1.0, this model never worked"
        fi
        [ -f "$CK/predictor_h1_${TASK}_s${s}_seed${s}.pt" ] && \
            mv "$CK/predictor_h1_${TASK}_s${s}_seed${s}.pt" "$ck"
    fi
    [ -f "$ck" ] || { echo "    no checkpoint, skipping seed $s"; continue; }

    echo "### scoring poses, seed $s"
    for p in $POSES; do
        outc="cache/conservatism_h1_${TASK}_s${s}_${p}.json"
        [ -f "$outc" ] && continue
        lat="cache/latents/r1_${TASK}__${p}"
        [ -d "$lat" ] || continue
        python scripts/check_conservatism.py --task "h1_${TASK}_s${s}_${p}" \
            --data "$DATA" --latents "$lat" --checkpoint "$ck" --max-horizon "$HMAX" \
            > "logs/h1_cons_${TASK}_${s}_${p}.log" 2>&1
    done
    n=$(ls cache/conservatism_h1_${TASK}_s${s}_*.json 2>/dev/null | wc -l)
    echo "    ${n} poses scored"
done

echo
python scripts/harden_gap_prediction.py --task "$TASK" --seeds $SEEDS
