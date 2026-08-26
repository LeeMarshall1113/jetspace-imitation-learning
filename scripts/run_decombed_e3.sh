#!/usr/bin/env bash
# Does the horizon result survive removing the encoder comb?
#
# E3's do-nothing baseline collapsed at every lag that was a multiple of the
# encode stride, which made the gain figures oscillate between 3.4x and 13x for
# reasons that had nothing to do with the robot. The qualitative claim looked
# safe -- even the worst phase beat the baseline 3.4x -- but "looked safe" is
# not a check. This runs the same evaluation against decombed latents.
#
# real_cubes is deliberately run on the ORIGINAL cache: its comb measured
# 1.014x, so there is no artifact to remove and subtracting phase means would
# only delete signal.
set -uo pipefail
cd "$(dirname "$0")/.."

run() {
    name="$1"; lat="$2"; data="$3"; ckpt="$4"; H="$5"
    [ -d "$lat" ] || { echo "### $name: no cache at $lat"; return; }
    [ -f "$ckpt" ] || { echo "### $name: no checkpoint at $ckpt"; return; }
    echo "### $name  (latents: $lat, H=$H)"
    python scripts/eval_horizon.py --task "$name" --data "$data" --latents "$lat" \
        --checkpoint "$ckpt" --max-horizon "$H" --out "cache/e3_decombed_${name}.json" \
        2>&1 | grep -vE "UserWarning|self.blocks" | tail -14
    echo
    python scripts/check_conservatism.py --task "$name" --data "$data" --latents "$lat" \
        --checkpoint "$ckpt" --max-horizon "$H" 2>&1 | grep -vE "UserWarning|self.blocks" | tail -10
    echo
}

run push      cache/latents/push_decombed      data/episodes/push      checkpoints/predictor_push_seed0.pt      96
run pickplace cache/latents/pickplace_decombed data/episodes/pickplace checkpoints/predictor_pickplace_seed0.pt 96
run real_cubes cache/latents/real_cubes data/episodes/real_so101_teleop_cubes \
    checkpoints/real/predictor_real_cubes_seed0.pt 64
