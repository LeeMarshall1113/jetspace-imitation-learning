#!/usr/bin/env bash
# Five image-space nuisance axes, across every encoder arm.
#
#   bash scripts/run_e12_image_axes.sh push pickplace
#
# E12's rendered axes cost a collection pass each and only two of four survived
# their controls, which makes scaling the axis count slow. Sensor noise,
# defocus, compression, exposure drift and resolution loss are transforms of
# frames already on disk, so they are applied at encode time: no rendering, no
# second copy of any dataset, and every encoder sees identical corrupted pixels
# because the per-frame noise is seeded.
#
# That takes the benchmark from two valid axes to as many as seven, which is the
# dimension that was furthest behind -- COLOSSEUM reports fourteen.
#
# These are NOT substitutes for rendered nuisances. A box blur is not a real
# defocus and a block average is not a codec; the write-up says so. They are the
# same KIND of perturbation a deployed policy meets, and any axis that fails to
# discriminate gets excluded by E12d exactly as clutter did.
set -uo pipefail
cd "$(dirname "$0")/.."

TASKS=${@:-push}
DC="docker compose -f docker/compose.yaml --profile wsl2 run --rm -T dev-wsl"
DCP="docker compose -f docker/compose.yaml --profile wsl2 run --rm -T -e PYTHONPATH=/workspace/.pydeps dev-wsl"
export DOCKER_UID=1000 DOCKER_GID=1000
mkdir -p logs

# axis:level pairs, mild to severe.
CONDS="noise:4.0 noise:10.0 noise:20.0 noise:35.0 \
defocus:1 defocus:2 defocus:4 defocus:7 \
compress:4 compress:8 compress:14 compress:22 \
exposure:0.65 exposure:0.80 exposure:1.25 exposure:1.55 \
lowres:2 lowres:3 lowres:5 lowres:8"

HF_ARMS="dinov3:dinov3 siglip2:siglip2 aimv2:aimv2 dinov2:dino clip:clip \
vit-in1k:vitin1k vc1:vc1 dinov2-large:dinov2l dinov3-large:dinov3l \
siglip:siglip1 vit-large:vitlarge clip-large:cliplarge vc1-large:vc1large"

# Never compete with a running pass; the last collision here was an OOM.
while pgrep -f "run_e12.sh|run_e12_scale|collect_e12" >/dev/null; do
    echo "  waiting for the running E12 pass ($(date +%T))"
    sleep 180
done
echo "starting image axes at $(date +%T)"

for TASK in $TASKS; do
    SRC="data/episodes/e12_${TASK}__ref"
    EP=$(ls "$SRC"/episode_*.npz 2>/dev/null | wc -l)
    [ "$EP" -eq 0 ] && { echo "### ${TASK}: no reference episodes, skipping"; continue; }
    echo
    echo "### ${TASK}: 20 conditions x 15 arms at ${EP} episodes"

    for spec in $CONDS; do
        axis="${spec%%:*}"; level="${spec##*:}"
        tag="${axis}_${level//./p}"
        printf "  %-16s " "$tag"

        out="cache/latents/r1_e12_${TASK}__${tag}"
        if [ "$(ls "$out"/episode_*.npy 2>/dev/null | wc -l)" -lt "$EP" ]; then
            $DC python scripts/cache_latents.py --task "e12_${TASK}__${tag}" \
                --data "$SRC" --out "$out" --pool-grid 4 --limit "$EP" \
                --nuisance "$axis" --nuisance-level "$level" \
                > "logs/e12i_vjepa_${TASK}_${tag}.log" 2>&1
        fi

        out="cache/latents/r1cnn_e12_${TASK}__${tag}"
        if [ "$(ls "$out"/episode_*.npy 2>/dev/null | wc -l)" -lt "$EP" ]; then
            $DC python scripts/cache_latents_cnn.py --data "$SRC" --out "$out" \
                --pool-grid 4 --hidden 1024 --frames-per-latent 2 --limit "$EP" \
                --seed 0 --nuisance "$axis" --nuisance-level "$level" \
                > "logs/e12i_rand_${TASK}_${tag}.log" 2>&1
        fi

        for pair in $HF_ARMS; do
            model="${pair%%:*}"; pre="${pair##*:}"
            out="cache/latents/${pre}_e12_${TASK}__${tag}"
            [ "$(ls "$out"/episode_*.npy 2>/dev/null | wc -l)" -ge "$EP" ] && continue
            $DCP python scripts/cache_latents_hf.py --model "$model" \
                --data "$SRC" --out "$out" --limit "$EP" --pool-grid 4 \
                --frames-per-latent 2 --nuisance "$axis" \
                --nuisance-level "$level" \
                > "logs/e12i_${pre}_${TASK}_${tag}.log" 2>&1
        done

        n=$(ls -d cache/latents/*_e12_${TASK}__${tag} 2>/dev/null | wc -l)
        echo "${n}/15 arms"
    done
done

echo
echo "### analysis across every axis"
for TASK in $TASKS; do
    echo "--- ${TASK} ---"
    $DCP python scripts/e12_analyze.py "$TASK" 2>&1 \
        | grep -avE "Container |UserWarning|warnings.warn" | tail -70
done
