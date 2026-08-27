#!/usr/bin/env bash
# E12: nine encoders crossed with three new nuisance axes.
#
#   bash scripts/run_e12.sh push 10
#
# Per docs/prereg-e12.md. Three stages:
#
#   1. collect  one episode set per condition -- a reference plus four
#               displaced levels for each of lighting, texture and clutter.
#               Everything but the axis under test is pinned, dynamics
#               included, and episode seeds are shared so the arm does the
#               same thing in every condition and only the rendering differs.
#   2. encode   all nine encoders over all thirteen conditions, at matched
#               pool_grid and frames_per_latent.
#   3. analyse  probe R^2 at the reference against robustness at the held-out
#               levels, per axis, with the registered invalidation checks.
#
# Viewpoint is already measured by E11 and is not recollected here.
set -uo pipefail
cd "$(dirname "$0")/.."

TASK=${1:-push}
EPISODES=${2:-10}
DC="docker compose -f docker/compose.yaml --profile wsl2 run --rm -T dev-wsl"
DCP="docker compose -f docker/compose.yaml --profile wsl2 run --rm -T -e PYTHONPATH=/workspace/.pydeps dev-wsl"
export DOCKER_UID=1000 DOCKER_GID=1000
mkdir -p logs

# tag:axis:level, reference first so it exists before anything needs it.
CONDS="ref:reference: \
lighting_0p3:lighting:0.30 lighting_0p45:lighting:0.45 \
lighting_0p55:lighting:0.55 lighting_0p62:lighting:0.62 \
texture_0p06:texture:0.06 texture_0p1:texture:0.10 \
texture_0p16:texture:0.16 texture_0p24:texture:0.24 \
clutter_1:clutter:1 clutter_2:clutter:2 clutter_3:clutter:3 clutter_4:clutter:4"

echo "### stage 1: collecting $(echo $CONDS | wc -w) conditions x ${EPISODES} episodes"
for spec in $CONDS; do
    tag="${spec%%:*}"; rest="${spec#*:}"; axis="${rest%%:*}"; level="${rest#*:}"
    out="data/episodes/e12_${TASK}__${tag}"
    n=$(ls "$out"/episode_*.npz 2>/dev/null | wc -l)
    [ "$n" -ge "$EPISODES" ] && { echo "  ${tag}: cached (${n})"; continue; }
    printf "  %-16s " "$tag"
    if [ "$axis" = "reference" ]; then
        $DC python scripts/collect_e12.py --axis reference --task "$TASK" \
            --episodes "$EPISODES" --out "$out" > "logs/e12_collect_${tag}.log" 2>&1
    else
        $DC python scripts/collect_e12.py --axis "$axis" --level "$level" \
            --task "$TASK" --episodes "$EPISODES" --out "$out" \
            > "logs/e12_collect_${tag}.log" 2>&1
    fi
    n=$(ls "$out"/episode_*.npz 2>/dev/null | wc -l)
    if [ "$n" -ge "$EPISODES" ]; then echo "ok (${n})"; else
        echo "FAILED -- logs/e12_collect_${tag}.log"
        grep -avE "EGL|OpenGL|glCheckError|__del__|self.free" \
            "logs/e12_collect_${tag}.log" | tail -3 | sed 's/^/      /'
    fi
done

echo
echo "### stage 2: encoding nine arms over every condition"
for spec in $CONDS; do
    tag="${spec%%:*}"
    data="data/episodes/e12_${TASK}__${tag}"
    [ -d "$data" ] || continue
    printf "  %-16s " "$tag"

    # V-JEPA 2 -- the project's own encoder, its own script.
    out="cache/latents/r1_e12_${TASK}__${tag}"
    if [ ! -d "$out" ] || [ "$(ls "$out"/episode_*.npy 2>/dev/null | wc -l)" -lt "$EPISODES" ]; then
        $DC python scripts/cache_latents.py --task "e12_${TASK}__${tag}" \
            --data "$data" --out "$out" --pool-grid 4 --limit "$EPISODES" \
            > "logs/e12_enc_vjepa_${tag}.log" 2>&1
    fi
    printf "vjepa "

    # Random CNN control.
    out="cache/latents/r1cnn_e12_${TASK}__${tag}"
    if [ ! -d "$out" ] || [ "$(ls "$out"/episode_*.npy 2>/dev/null | wc -l)" -lt "$EPISODES" ]; then
        $DC python scripts/cache_latents_cnn.py --data "$data" --out "$out" \
            --pool-grid 4 --hidden 1024 --frames-per-latent 2 \
            --limit "$EPISODES" --seed 0 > "logs/e12_enc_rand_${tag}.log" 2>&1
    fi
    printf "rand "

    # The HuggingFace and timm arms.
    for pair in "dinov3:dinov3" "siglip2:siglip2" "aimv2:aimv2" \
                "dinov2:dino" "clip:clip" "vit-in1k:vitin1k" "vc1:vc1"; do
        model="${pair%%:*}"; pre="${pair##*:}"
        out="cache/latents/${pre}_e12_${TASK}__${tag}"
        if [ ! -d "$out" ] || [ "$(ls "$out"/episode_*.npy 2>/dev/null | wc -l)" -lt "$EPISODES" ]; then
            $DCP python scripts/cache_latents_hf.py --model "$model" \
                --data "$data" --out "$out" --limit "$EPISODES" \
                --pool-grid 4 --frames-per-latent 2 \
                > "logs/e12_enc_${pre}_${tag}.log" 2>&1
        fi
        printf "%s " "$pre"
    done
    echo
done

echo
echo "### stage 3: analysis"
$DC python scripts/e12_analyze.py "$TASK"
