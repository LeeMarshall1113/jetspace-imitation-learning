#!/usr/bin/env bash
# Retrain on decombed latents, then re-measure. This is the test that counts.
#
# Evaluating the EXISTING checkpoint on decombed latents was not a fair test and
# its result should not be quoted: that predictor was trained on combed latents,
# so feeding it decombed ones is a distribution shift, and the collapse it
# produced (96 steps -> 6 steps, action-awareness to zero) partly measures the
# shift rather than the artifact.
#
# What it does establish is that the collapse is large enough to matter. If the
# comb were incidental, a model trained on it would degrade gently without it.
# Total failure means the comb was load-bearing -- which is exactly the thing
# that has to be ruled out before the horizon number can be published.
#
# So: train from scratch on decombed latents, evaluate on decombed latents, and
# compare like with like.
set -uo pipefail
cd "$(dirname "$0")/.."

for t in push pickplace; do
    lat="cache/latents/${t}_decombed"
    [ -d "$lat" ] || { echo "### $t: no decombed cache"; continue; }
    echo "================ $t: train on decombed ================"
    python scripts/train_predictor.py --task "${t}_decombed" \
        --data "data/episodes/$t" --latents "$lat" \
        --out checkpoints/decombed --epochs 30 --seed 0 \
        2>&1 | grep -vE "UserWarning|self.blocks" | tail -6
    echo
    echo "---------------- $t: E3 on decombed ----------------"
    python scripts/eval_horizon.py --task "${t}_decombed" \
        --data "data/episodes/$t" --latents "$lat" \
        --checkpoint "checkpoints/decombed/predictor_${t}_decombed_seed0.pt" \
        --max-horizon 96 --out "cache/e3_decombed_retrained_${t}.json" \
        2>&1 | grep -vE "UserWarning|self.blocks" | tail -12
    echo
    echo "---------------- $t: conservatism ----------------"
    python scripts/check_conservatism.py --task "${t}_decombed" \
        --data "data/episodes/$t" --latents "$lat" \
        --checkpoint "checkpoints/decombed/predictor_${t}_decombed_seed0.pt" \
        --max-horizon 96 2>&1 | grep -vE "UserWarning|self.blocks" | tail -8
    echo
done
