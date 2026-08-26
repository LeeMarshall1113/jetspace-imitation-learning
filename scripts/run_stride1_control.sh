#!/usr/bin/env bash
# Settle whether simulation is genuinely worse than real, or only looked better.
#
# Where this stands. The window-phase comb inflated simulation's apparent
# quality; removing it by subtracting per-phase means dropped push's direction
# cosine from 0.902 to 0.668 and pickplace's to 0.622, while real teleoperation
# video -- which never had a comb -- sits at 0.847. That would mean the world
# model is BETTER on real video than in simulation, which is worth saying only
# if it is true.
#
# It is not yet established, because neither sim number is unbiased:
#
#   combed      cosine 0.902   inflated by the artifact
#   decombed    cosine 0.668   deflated by overcorrection past flat (0.94, 0.85)
#
# The truth is somewhere between, and a claim spanning 0.67-0.90 cannot be
# compared against real's 0.847 -- the interval contains it.
#
# The fix that does not estimate anything. margin=15 gives
# stride = 32 - 30 = 2 frames = ONE latent, so every latent comes from a window
# at a different phase and the comb cannot form. No subtraction, no bias, about
# 8x the encoding cost.
#
# The control matters as much as the treatment. Encoding 20 episodes at stride 1
# and comparing against 60 episodes at stride 8 would confound the artifact with
# dataset size, so the control is the SAME 20 episodes at stride 8.
set -uo pipefail
cd "$(dirname "$0")/.."

TASK=${1:-push}
N=${2:-20}

echo "=============================================================="
echo "  $TASK: $N episodes, stride-1 treatment vs stride-8 control"
echo "=============================================================="

echo
echo "--- encode: stride 8 latents (control, chunk 32 margin 8) ---"
python scripts/cache_latents.py --task "$TASK" --data "data/episodes/$TASK" \
    --out "cache/latents/${TASK}_s8n${N}" --chunk 32 --margin 8 --limit "$N" \
    2>&1 | grep -vE "UserWarning|self.blocks|Loading weights" | tail -4

echo
echo "--- encode: stride 1 latent (treatment, chunk 32 margin 15) ---"
echo "    roughly 8x the work; this is the slow step"
python scripts/cache_latents.py --task "$TASK" --data "data/episodes/$TASK" \
    --out "cache/latents/${TASK}_s1n${N}" --chunk 32 --margin 15 --limit "$N" \
    2>&1 | grep -vE "UserWarning|self.blocks|Loading weights" | tail -4

echo
echo "--- comb present in each? (expect ~1.5x control, ~1.0x treatment) ---"
for s in s8 s1; do
    echo -n "  $s: "
    python scripts/decomb_latents.py --latents "cache/latents/${TASK}_${s}n${N}" --dry-run \
        2>&1 | grep "comb ratio before" || echo "(failed)"
done

for s in s8 s1; do
    lat="cache/latents/${TASK}_${s}n${N}"
    name="${TASK}_${s}n${N}"
    echo
    echo "=============== $name ==============="
    python scripts/train_predictor.py --task "$name" --data "data/episodes/$TASK" \
        --latents "$lat" --out checkpoints/stride --epochs 30 --seed 0 \
        2>&1 | grep -vE "UserWarning|self.blocks" | tail -3
    echo
    python scripts/eval_horizon.py --task "$name" --data "data/episodes/$TASK" \
        --latents "$lat" --checkpoint "checkpoints/stride/predictor_${name}_seed0.pt" \
        --max-horizon 96 --out "cache/e3_stride_${name}.json" \
        2>&1 | grep -vE "UserWarning|self.blocks" | tail -11
    echo
    python scripts/check_conservatism.py --task "$name" --data "data/episodes/$TASK" \
        --latents "$lat" --checkpoint "checkpoints/stride/predictor_${name}_seed0.pt" \
        --max-horizon 96 2>&1 | grep -vE "UserWarning|self.blocks" | tail -8
done

echo
echo "=============================================================="
echo "  Compare the stride-1 cosine against real_cubes' 0.847."
echo "  Stride-1 is the unbiased simulation number: if it lands near"
echo "  0.67 the real-beats-sim result stands; near 0.85 the gap was"
echo "  the artifact all along and there is no such result."
echo "=============================================================="
