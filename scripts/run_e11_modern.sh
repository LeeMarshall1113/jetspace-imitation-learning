#!/usr/bin/env bash
# E11, modern half: V-JEPA 2 against its actual contemporaries.
#
#   bash scripts/run_e11_modern.sh push
#
# The first E11 pass cached DINOv2 (2023-04), CLIP (2021-01) and supervised ViT
# (2020-10). Beating those would only show that V-JEPA 2 (2025-06) is newer than
# they are. This pass adds encoders from the same generation, which is the
# comparison that actually tests the claim:
#
#   siglip2   2025-02  image-text        google/siglip2-base-patch16-224
#   aimv2     2024-11  autoregressive    apple/aimv2-large-patch14-224
#   dinov3    2025-08  image SSL         gated; skipped unless access is granted
#
# DINOv3 is the most important of the three -- it is the direct image-SSL
# contemporary of V-JEPA 2 -- and it is gated behind a licence only the repo
# owner can accept. This script probes for it and reports rather than failing,
# so the rest of the comparison is not blocked waiting on an approval.
#
# Parameter counts are NOT matched (SigLIP2 93M, AIMv2 309M, V-JEPA 2 326M) and
# are reported per arm rather than pretended away.
set -uo pipefail
cd "$(dirname "$0")/.."

TASK=${1:-push}
DATA="data/episodes/r1_${TASK}"
DC="docker compose -f docker/compose.yaml --profile wsl2 run --rm -T dev-wsl"
export DOCKER_UID=1000 DOCKER_GID=1000
mkdir -p logs

# Do not fight the first pass for the GPU.
while pgrep -f "run_e11.sh" >/dev/null; do sleep 60; done
echo "first E11 pass finished; starting modern encoders at $(date +%T)"

NEP=$(ls "cache/latents/r1_${TASK}__r1_ref"/episode_*.npy 2>/dev/null | wc -l)
[ "$NEP" -gt 0 ] || { echo "no V-JEPA cache to match"; exit 1; }
echo "matching the V-JEPA arm at ${NEP} episodes"

POSES=$(python3 -c "
import sys; sys.path.insert(0,'src')
from jetspace.envs.so101_env import R1_POSES
print(' '.join(R1_POSES))")

# Is DINOv3 reachable yet? Probed once, not per pose.
DINOV3=""
if $DC python3 -c "
from huggingface_hub import hf_hub_download, get_token
hf_hub_download('facebook/dinov3-vitb16-pretrain-lvd1689m','config.json',token=get_token())
" >/dev/null 2>&1; then
    DINOV3="dinov3:dinov3"
    echo "DINOv3 access GRANTED -- including it"
else
    echo "DINOv3 still gated -- skipping (accept the licence at"
    echo "  https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m"
    echo "  then re-run this script; cached arms are reused)"
fi

for spec in "siglip2:siglip2" "aimv2:aimv2" $DINOV3; do
    model="${spec%%:*}"; tag="${spec##*:}"
    echo
    echo "### caching ${model}"
    for p in $POSES; do
        out="cache/latents/${tag}_${TASK}__${p}"
        n=$(ls "$out"/episode_*.npy 2>/dev/null | wc -l)
        [ "$n" -ge "$NEP" ] && continue
        printf "  %-14s " "$p"
        $DC python scripts/cache_latents_hf.py --model "$model" \
            --data "$DATA" --camera "$p" --out "$out" --limit "$NEP" \
            --pool-grid 4 --frames-per-latent 2 \
            > "logs/e11_${tag}_${p}.log" 2>&1
        n=$(ls "$out"/episode_*.npy 2>/dev/null | wc -l)
        if [ "$n" -ge "$NEP" ]; then echo "ok (${n})"; else
            echo "FAILED -- logs/e11_${tag}_${p}.log"
            tail -3 "logs/e11_${tag}_${p}.log" | sed 's/^/      /'
        fi
    done
done

echo
echo "### parity across every arm"
ARMS="r1_${TASK}"
for t in dinov3 siglip2 aimv2 dino clip vitin1k r1cnn; do
    [ -d "cache/latents/${t}_${TASK}__r1_ref" ] && ARMS="$ARMS ${t}_${TASK}"
done
$DC python scripts/check_encoder_parity.py $ARMS || {
    echo "REFUSING to compare: arms are not matched."; exit 1; }

echo
echo "### held-out viewpoint generalization"
echo "    (1.0 = no better than predicting the mean action)"
printf "  %-10s %-9s %-18s %s\n" "encoder" "released" "baseline" "multiview"
for spec in "r1_${TASK}:vjepa2:2025-06" "dinov3_${TASK}:dinov3:2025-08" \
            "siglip2_${TASK}:siglip2:2025-02" "aimv2_${TASK}:aimv2:2024-11" \
            "dino_${TASK}:dinov2:2023-04" "clip_${TASK}:clip:2021-01" \
            "vitin1k_${TASK}:vit-in1k:2020-10" "r1cnn_${TASK}:random:none"; do
    prefix="${spec%%:*}"; rest="${spec#*:}"; name="${rest%%:*}"; rel="${rest##*:}"
    [ -d "cache/latents/${prefix}__r1_ref" ] || continue
    $DC python scripts/e8_canonicalize.py --task "$TASK" --seeds 0 1 2 \
        --prefix "$prefix" --out "cache/e11_${name}.json" \
        > "logs/e11_eval_${name}.log" 2>&1
    b=$(grep -aE "^  baseline "  "logs/e11_eval_${name}.log" | awk '{print $2, $3, $4}')
    m=$(grep -aE "^  multiview " "logs/e11_eval_${name}.log" | awk '{print $2, $3, $4}')
    printf "  %-10s %-9s %-18s %s\n" "$name" "$rel" "${b:-FAILED}" "${m:-FAILED}"
done
