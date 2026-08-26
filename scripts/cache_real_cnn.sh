#!/usr/bin/env bash
# Random-CNN features for the eight real laboratories, matched to the V-JEPA
# side at pool_grid 4, hidden 1024, frames_per_latent 2.
#
# Without this the E9 encoder ablation cannot run, and without the ablation E9
# only shows that transfer happens -- not that pretraining is what carries it,
# which is the claim.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

DC="docker compose -f docker/compose.yaml --profile wsl2 run --rm -T dev-wsl"
export DOCKER_UID=1000 DOCKER_GID=1000

LABS="A_cubes__ego B_svla__side C_tape__birdEye D_ball__front \
E_summer__front F_cup__cam_front G_bin__front H_penmug1__camera_2"

for L in $LABS; do
    cam="${L#*__}"
    out="cache/latents/n1bcnn_${L}"
    if [ -d "$out" ] && [ -n "$(ls -A "$out" 2>/dev/null)" ]; then
        echo "  ${L}: cached"; continue
    fi
    printf "  %-26s " "$L"
    $DC python scripts/cache_latents_cnn.py --data "data/episodes/n1b_${L}" \
        --camera "$cam" --out "$out" --pool-grid 4 --hidden 1024 \
        --frames-per-latent 2 --seed 0 > "logs/e9_cnn_${L}.log" 2>&1
    if [ -d "$out" ] && [ -n "$(ls -A "$out" 2>/dev/null)" ]; then
        echo "ok ($(ls "$out" | wc -l) episodes)"
    else
        echo "FAILED -- logs/e9_cnn_${L}.log"
        tail -3 "logs/e9_cnn_${L}.log" | sed 's/^/      /'
    fi
done
