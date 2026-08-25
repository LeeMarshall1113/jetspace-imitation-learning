#!/usr/bin/env bash
# E7: cache the random-CNN arm for every R1 pose, then compare arms.
#
#   bash scripts/run_e7.sh reach "0 1 2"
#
# The V-JEPA arm is already cached from R1/H1 as cache/latents/r1_<task>__<pose>.
# The random arm does not exist yet, so it is generated here at settings matched
# to the V-JEPA side: pool_grid 4, hidden 1024, frames_per_latent 2. Those three
# must match or the arms are not comparable and E7's invalidation 3 fires
# (ledger L7 is what happens when they silently do not).
#
# Encoder weights are seeded, because for the random arm the weights ARE the
# experiment.
set -uo pipefail
cd "$(dirname "$0")/.."

TASK=${1:-reach}
SEEDS=${2:-"0 1 2"}
PILOT=${3:-}
DATA="data/episodes/r1_${TASK}"
mkdir -p logs

[ -d "$DATA" ] || { echo "no $DATA -- run scripts/run_r1.sh $TASK first"; exit 1; }

POSES=$(python3 -c "
import sys; sys.path.insert(0,'src')
from jetspace.envs.so101_env import R1_POSES
print(' '.join(R1_POSES))")

echo "### caching random-CNN features for $(echo $POSES | wc -w) poses"
for p in $POSES; do
    out="cache/latents/r1cnn_${TASK}__${p}"
    if [ -d "$out" ] && [ -n "$(ls -A "$out" 2>/dev/null)" ]; then
        continue
    fi
    printf "  %-14s " "$p"
    # Log to a file rather than piping: a pipe into grep swallowed every error
    # the first time R2 ran, and 23 poses reported nothing for an hour.
    python scripts/cache_latents_cnn.py --data "$DATA" --camera "$p" \
        --out "$out" --pool-grid 4 --hidden 1024 --frames-per-latent 2 \
        --seed 0 > "logs/e7_cnn_${p}.log" 2>&1
    if [ -d "$out" ] && [ -n "$(ls -A "$out" 2>/dev/null)" ]; then
        echo "ok ($(ls "$out" | wc -l) episodes)"
    else
        echo "FAILED -- logs/e7_cnn_${p}.log"
        tail -3 "logs/e7_cnn_${p}.log" | sed 's/^/      /'
    fi
done

echo
python scripts/e7_encoder_transfer.py --task "$TASK" --seeds $SEEDS --pca-dim 128 ${PILOT:+--pilot}
