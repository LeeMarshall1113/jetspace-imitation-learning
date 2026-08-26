#!/usr/bin/env bash
# H1d: does gap -> degradation survive on REAL robot video?
#
#   bash scripts/run_h1d.sh "0 1 2" 24
#
# Registered in docs/prereg-h1.md as "THE DIFFERENTIATOR": train a world model
# on one real laboratory's video, evaluate it on every other real set, and
# correlate latent gap against degradation. Registered threshold rho <= -0.5,
# weaker than the simulated -0.6 because task is not held constant across labs.
#
# This is the prediction that separates the claim from arXiv:2604.13645, which
# correlates Wasserstein distance against policy success in simulation only.
#
# Every lab takes a turn as the training domain, because with eight labs a
# single choice of training lab is one draw, and the R1 sweep already taught
# this project that a single seed's rho (-0.921) was the best of three, not the
# number.
#
# **Known confound, measured rather than assumed.** Ledger L8 recorded that
# action spaces are not interchangeable across labs. All eight are 6-dim on a
# roughly [-100, 100] scale, but per-dimension spread differs by up to 5x. The
# world model is conditioned on actions, so some of any cross-lab degradation is
# action mismatch rather than visual shift. `harden_gap_prediction.py --h1d`
# therefore also computes the action-distribution distance per pair and reports
# the partial correlation of gap against degradation controlling for it. If the
# relationship only exists because both track action mismatch, that control
# removes it, and that outcome is reported.
set -uo pipefail
cd "$(dirname "$0")/.."

SEEDS=${1:-"0 1 2"}
HMAX=${2:-24}
CK=checkpoints/h1d
mkdir -p logs "$CK" cache

LABS="n1b_A_cubes__ego n1b_B_svla__side n1b_C_tape__birdEye n1b_D_ball__front \
n1b_E_summer__front n1b_F_cup__cam_front n1b_G_bin__front n1b_H_penmug1__camera_2"

for train in $LABS; do
    tag=$(echo "$train" | sed 's/^n1b_//; s/__.*//')
    for s in $SEEDS; do
        ck="$CK/predictor_h1d_${tag}_s${s}.pt"
        if [ ! -f "$ck" ]; then
            echo "### training on ${tag}, seed ${s}"
            python scripts/train_predictor.py --task "h1d_${tag}_s${s}" \
                --data "data/episodes/${train}" \
                --latents "cache/latents/${train}" \
                --out "$CK" --epochs 30 --seed "$s" --pca-dim 128 \
                > "logs/h1d_train_${tag}_${s}.log" 2>&1
            gain=$(grep -aoE "\([0-9.]+x better" "logs/h1d_train_${tag}_${s}.log" | tail -1)
            echo "    ${gain:-no gain line}"
            # Same registered invalidation as H1: a degradation curve measured
            # from a model that never worked is the R2 failure repeated.
            if grep -aqE "\(0\.[0-9]+x better" "logs/h1d_train_${tag}_${s}.log"; then
                echo "    INVALID: gain below 1.0, this model never worked -- skipping"
                continue
            fi
            [ -f "$CK/predictor_h1d_${tag}_s${s}_seed${s}.pt" ] && \
                mv "$CK/predictor_h1d_${tag}_s${s}_seed${s}.pt" "$ck"
        fi
        [ -f "$ck" ] || { echo "    no checkpoint for ${tag} s${s}"; continue; }

        # Score on every lab INCLUDING the training lab: the in-domain score is
        # the zero point, and without it a degradation is a number with no
        # baseline to degrade from.
        for ev in $LABS; do
            etag=$(echo "$ev" | sed 's/^n1b_//; s/__.*//')
            outc="cache/conservatism_h1d_${tag}_s${s}_${etag}.json"
            [ -f "$outc" ] && continue
            python scripts/check_conservatism.py --task "h1d_${tag}_s${s}_${etag}" \
                --data "data/episodes/${ev}" --latents "cache/latents/${ev}" \
                --checkpoint "$ck" --max-horizon "$HMAX" \
                > "logs/h1d_cons_${tag}_${s}_${etag}.log" 2>&1
        done
        n=$(ls cache/conservatism_h1d_${tag}_s${s}_*.json 2>/dev/null | wc -l)
        echo "    ${tag} s${s}: ${n}/8 labs scored"
    done
done

echo
python scripts/analyze_h1d.py --seeds $SEEDS
