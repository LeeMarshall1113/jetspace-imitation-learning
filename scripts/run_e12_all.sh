#!/usr/bin/env bash
# The whole E12 programme, in one sequence.
#
#   nohup bash scripts/run_e12_all.sh > logs/e12_all.log 2>&1 &
#
# Previously this was three scripts each waiting on the others via pgrep. When
# the session ended, two of them were still waiting for a process that had
# already died, and neither ever started. One sequence with no inter-process
# waiting cannot fail that way.
#
# Ordered by value, so a run cut short still leaves the most useful state:
#
#   1. finish pickplace on the original nine arms   -- a SECOND TASK, which is
#      the single most quotable weakness in the paper
#   2. scale to fifteen arms on both tasks          -- the encoder count a
#      benchmark reviewer counts first
#   3. image-space axes                              -- axis count, the
#      dimension furthest behind
#
# Every stage skips work already cached, so this is safe to re-run after any
# interruption.
set -uo pipefail
cd "$(dirname "$0")/.."
export DOCKER_UID=1000 DOCKER_GID=1000

DC="docker compose -f docker/compose.yaml --profile wsl2 run --rm -T dev-wsl"
DCP="docker compose -f docker/compose.yaml --profile wsl2 run --rm -T -e PYTHONPATH=/workspace/.pydeps dev-wsl"
mkdir -p logs

# Yield politely if the compute broker has asked for this machine back. A cell
# is the natural checkpoint: everything already encoded is cached, so stopping
# between cells costs nothing and re-running resumes. Stop the whole run rather
# than skipping cells -- a skipped cell looks like a finished one, and the
# analysis at the end would then be computed over incomplete arms.
preempt_check() {
    if [ -n "${BROKER_PREEMPT_FILE:-}" ] && [ -f "$BROKER_PREEMPT_FILE" ]; then
        echo
        echo "########## yielding to compute broker at $(date +%T)"
        echo "cached work is intact; re-run this script to resume"
        exit 0
    fi
}

# Docker is the backend for every encode. If it is down, encode_cell fails
# instantly for each cell in turn and the run marches to the end having done
# nothing, then analyses whatever partial arms exist. Fail loudly instead.
if ! timeout 60 docker info >/dev/null 2>&1; then
    echo "docker is not responding -- refusing to start" >&2
    exit 69
fi

RENDERED="ref lighting_0p3 lighting_0p45 lighting_0p55 lighting_0p62 \
texture_0p06 texture_0p1 texture_0p16 texture_0p24 \
clutter_1 clutter_2 clutter_3 clutter_4"

IMAGE_CONDS="noise:4.0 noise:10.0 noise:20.0 noise:35.0 \
defocus:1 defocus:2 defocus:4 defocus:7 \
compress:4 compress:8 compress:14 compress:22 \
exposure:0.65 exposure:0.80 exposure:1.25 exposure:1.55 \
lowres:2 lowres:3 lowres:5 lowres:8"

BASE_HF="dinov3:dinov3 siglip2:siglip2 aimv2:aimv2 dinov2:dino clip:clip \
vit-in1k:vitin1k vc1:vc1"
SCALE_HF="dinov2-large:dinov2l dinov3-large:dinov3l siglip:siglip1 \
vit-large:vitlarge clip-large:cliplarge vc1-large:vc1large"

# Expansion to 22. Kept as its own list so the fifteen-arm results stay
# reproducible: run with EXPAND=1 to include these.
EXPAND_HF="convnext:convnext convnext-large:convnextl resnet50:resnet50 \
ijepa:ijepa mae:mae beit:beit swin:swin"
[ "${EXPAND:-0}" = "1" ] && SCALE_HF="$SCALE_HF $EXPAND_HF"

# encode <task> <condition-tag> <data-dir> <hf-arm-list> [nuisance] [level]
encode_cell() {
    local task="$1" tag="$2" data="$3" arms="$4" nz="${5:-}" lvl="${6:-}"
    local ep n out
    ep=$(ls "$data"/episode_*.npz 2>/dev/null | wc -l)
    [ "$ep" -eq 0 ] && return 0
    local nzargs=""
    [ -n "$nz" ] && nzargs="--nuisance $nz --nuisance-level $lvl"

    out="cache/latents/r1_e12_${task}__${tag}"
    n=$(ls "$out"/episode_*.npy 2>/dev/null | wc -l)
    if [ "$n" -lt "$ep" ]; then
        $DC python scripts/cache_latents.py --task "e12_${task}__${tag}" \
            --data "$data" --out "$out" --pool-grid 4 --limit "$ep" $nzargs \
            > "logs/e12a_vjepa_${task}_${tag}.log" 2>&1
    fi
    out="cache/latents/r1cnn_e12_${task}__${tag}"
    n=$(ls "$out"/episode_*.npy 2>/dev/null | wc -l)
    if [ "$n" -lt "$ep" ]; then
        $DC python scripts/cache_latents_cnn.py --data "$data" --out "$out" \
            --pool-grid 4 --hidden 1024 --frames-per-latent 2 --limit "$ep" \
            --seed 0 $nzargs > "logs/e12a_rand_${task}_${tag}.log" 2>&1
    fi
    for pair in $arms; do
        local model="${pair%%:*}" pre="${pair##*:}"
        out="cache/latents/${pre}_e12_${task}__${tag}"
        n=$(ls "$out"/episode_*.npy 2>/dev/null | wc -l)
        [ "$n" -ge "$ep" ] && continue
        $DCP python scripts/cache_latents_hf.py --model "$model" --data "$data" \
            --out "$out" --limit "$ep" --pool-grid 4 --frames-per-latent 2 \
            $nzargs > "logs/e12a_${pre}_${task}_${tag}.log" 2>&1
    done
}

echo "########## stage 1: finish pickplace, nine arms  $(date +%T)"
for c in $RENDERED; do
    preempt_check
    printf "  %-16s " "$c"
    encode_cell pickplace "$c" "data/episodes/e12_pickplace__${c}" "$BASE_HF"
    echo "$(ls -d cache/latents/*_e12_pickplace__${c} 2>/dev/null | wc -l)/9"
done

echo
echo "########## stage 2: scale to fifteen arms  $(date +%T)"
for task in push pickplace; do
    echo "--- ${task} ---"
    for c in $RENDERED; do
        preempt_check
        printf "  %-16s " "$c"
        encode_cell "$task" "$c" "data/episodes/e12_${task}__${c}" "$SCALE_HF"
        echo "$(ls -d cache/latents/*_e12_${task}__${c} 2>/dev/null | wc -l)/15"
    done
done

echo
echo "########## stage 3: image-space axes  $(date +%T)"
for task in push pickplace; do
    echo "--- ${task} ---"
    src="data/episodes/e12_${task}__ref"
    [ -d "$src" ] || continue
    for spec in $IMAGE_CONDS; do
        preempt_check
        axis="${spec%%:*}"; level="${spec##*:}"; tag="${axis}_${level//./p}"
        printf "  %-16s " "$tag"
        encode_cell "$task" "$tag" "$src" "$BASE_HF $SCALE_HF" "$axis" "$level"
        echo "$(ls -d cache/latents/*_e12_${task}__${tag} 2>/dev/null | wc -l)/15"
    done
done

echo
echo "########## analysis  $(date +%T)"
for task in push pickplace; do
    echo "===== ${task} ====="
    $DCP python scripts/e12_analyze.py "$task" 2>&1 \
        | grep -avE "Container |UserWarning|warnings.warn" | tail -80
done

echo
echo "########## figures"
$DC python scripts/make_figures.py --out paper/figures 2>&1 \
    | grep -avE "Container |UserWarning" | tail -12
echo "done $(date +%T)"
