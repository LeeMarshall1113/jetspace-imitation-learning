#!/usr/bin/env bash
# Why is the push world model action-blind, and only under one encoding?
#
# The seeded horizon run returned, identically across all three seeds:
#
#     push        useful >=145   ACTION-AWARE 0.0 +- 0.0
#     pickplace   useful >=174   ACTION-AWARE >=174
#
# Zero action-awareness means shuffling the actions changes the rollout error
# by less than 2% -- the model ignores which action was taken, from the very
# first step. Identical across seeds, so systematic rather than noise, and it
# contradicts earlier push runs that were action-aware to >=96.
#
# THE SUSPECT: those seeded checkpoints trained on `push_decombed`, the
# phase-mean-subtracted cache. Subtracting a per-phase mean was always a
# stopgap, and its own docstring says each phase holds only ~n/period samples
# so the estimate "partly fits real content and removes it along with the
# artifact" -- it overcorrects push from 1.669 to 0.941, past flat. If what it
# removed was action-relevant, this is that bill arriving.
#
# THE TEST: the same predictor configuration on two comb-free caches that were
# made comb-free by DIFFERENT means.
#
#   push_decombed   default encoding, phase means subtracted (overcorrects)
#   push_s1n60      stride-1 encoding, comb-free by construction
#
# If stride-1 is action-aware and decombed is not, the subtraction is
# destroying action signal and every decombed result has to be rebuilt on
# stride-1 latents. If both are blind, the cause is elsewhere and the earlier
# action-aware push numbers need re-examining instead.
set -uo pipefail
cd "$(dirname "$0")/.."

H=${1:-96}
DATA=data/episodes/push

check() {
    name="$1"; lat="$2"; ckpt="$3"
    [ -d "$lat" ] || { echo "  $name: no latents at $lat"; return; }
    [ -f "$ckpt" ] || { echo "  $name: no checkpoint at $ckpt"; return; }
    echo "--- $name  (latents $(basename "$lat")) ---"
    python scripts/eval_horizon.py --task "diag_$name" --data "$DATA" --latents "$lat" \
        --checkpoint "$ckpt" --max-horizon "$H" --out "cache/diag_${name}.json" \
        2>&1 | grep -E "USEFUL|ACTION-AWARE" | head -2
    echo -n "  inverse-dynamics probe: "
    python scripts/probe_action_signal.py --task "diag_$name" --data "$DATA" \
        --latents "$lat" --episodes 30 2>&1 | grep -iE "R\^?2" | head -2 | tr '\n' ' '
    echo
}

echo "=============================================================="
echo "  Action-blindness: which comb-free cache keeps action signal?"
echo "=============================================================="

# Same predictor recipe (PCA-128) on both caches.
for lat_name in push_decombed push_s1n60; do
    lat="cache/latents/$lat_name"
    ck="checkpoints/diag/predictor_diag_${lat_name}_seed0.pt"
    if [ ! -f "$ck" ]; then
        echo "### training on $lat_name"
        python scripts/train_predictor.py --task "diag_${lat_name}" --data "$DATA" \
            --latents "$lat" --out checkpoints/diag --epochs 30 --seed 0 --pca-dim 128 \
            2>&1 | grep -vE "UserWarning|self.blocks" | tail -2
    fi
    check "$lat_name" "$lat" "$ck"
done

echo
echo "=============================================================="
echo "  The comb ratio each cache actually carries"
echo "=============================================================="
for lat_name in push push_decombed push_s1n60; do
    [ -d "cache/latents/$lat_name" ] || continue
    echo -n "  $lat_name: "
    python scripts/decomb_latents.py --latents "cache/latents/$lat_name" --period 8 \
        --dry-run 2>&1 | grep "comb ratio before" | head -1
done

echo
echo "Read: 1.000 is flat. Above is a comb; BELOW is an overcorrection, which"
echo "means real structure was subtracted away along with the artifact."
