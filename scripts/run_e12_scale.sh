#!/usr/bin/env bash
# Scale E12 from nine encoders to fifteen.
#
#   bash scripts/run_e12_scale.sh push pickplace
#
# CortexBench compares ~10 encoders and Burns et al. 15; that count is the first
# number a reviewer of a benchmark paper looks at, and the caching path is
# generic, so additional arms are the cheapest available scale -- roughly twenty
# minutes each and no new code.
#
# The six additions each change ONE thing relative to an arm already present,
# rather than adding a near-duplicate:
#
#   dinov2-large   capacity within DINOv2, which leads on viewpoint at base
#   dinov3-large   capacity within DINOv3, which LOSES to DINOv2 at base --
#                  does that survive scaling?
#   siglip         SigLIP 1 against SigLIP 2, a generation control within one
#                  objective
#   vit-large      capacity within supervised ImageNet
#   clip-large     capacity within CLIP, whose axis-dependence is the sharpest
#                  result E12 has
#   vc1-large      capacity within the one robotics-specific arm that loads
#
# Waits for any running E12 pass rather than competing for the GPU: the last
# collision on this box was an OOM that killed a collection mid-run.
set -uo pipefail
cd "$(dirname "$0")/.."

TASKS=${@:-push}
DCP="docker compose -f docker/compose.yaml --profile wsl2 run --rm -T -e PYTHONPATH=/workspace/.pydeps dev-wsl"
export DOCKER_UID=1000 DOCKER_GID=1000
mkdir -p logs

CONDS="ref lighting_0p3 lighting_0p45 lighting_0p55 lighting_0p62 \
texture_0p06 texture_0p1 texture_0p16 texture_0p24 \
clutter_1 clutter_2 clutter_3 clutter_4"

# model:cache-prefix
NEW="dinov2-large:dinov2l dinov3-large:dinov3l siglip:siglip1 \
vit-large:vitlarge clip-large:cliplarge vc1-large:vc1large"

while pgrep -f "run_e12.sh|collect_e12" >/dev/null; do
    echo "  waiting for the running E12 pass ($(date +%T))"
    sleep 180
done
echo "starting the scale-up at $(date +%T)"

for TASK in $TASKS; do
    EP=$(ls "data/episodes/e12_${TASK}__ref"/episode_*.npz 2>/dev/null | wc -l)
    if [ "$EP" -eq 0 ]; then
        echo "### ${TASK}: no conditions collected, skipping"
        continue
    fi
    echo
    echo "### ${TASK}: 6 new arms x 13 conditions at ${EP} episodes"
    for spec in $NEW; do
        model="${spec%%:*}"; pre="${spec##*:}"
        printf "  %-14s " "$model"
        for c in $CONDS; do
            data="data/episodes/e12_${TASK}__${c}"
            [ -d "$data" ] || continue
            out="cache/latents/${pre}_e12_${TASK}__${c}"
            n=$(ls "$out"/episode_*.npy 2>/dev/null | wc -l)
            [ "$n" -ge "$EP" ] && continue
            $DCP python scripts/cache_latents_hf.py --model "$model" \
                --data "$data" --out "$out" --limit "$EP" --pool-grid 4 \
                --frames-per-latent 2 > "logs/e12s_${pre}_${TASK}_${c}.log" 2>&1
        done
        done_n=$(ls -d cache/latents/${pre}_e12_${TASK}__* 2>/dev/null | wc -l)
        echo "${done_n}/13 conditions"
    done

    echo
    echo "### ${TASK} with 15 arms"
    $DCP python scripts/e12_analyze.py "$TASK" 2>&1 \
        | grep -avE "Container |UserWarning|warnings.warn" | tail -60
done
