#!/usr/bin/env bash
# E11: is V-JEPA 2 special, or does any strong pretraining do this?
#
#   bash scripts/run_e11.sh push
#
# E8 showed a frozen V-JEPA 2 head trained on 2 camera viewpoints beats a frozen
# random-CNN head trained on 14, evaluated on 8 held-out viewpoints. With one
# baseline that result cannot distinguish two very different claims:
#
#   "video pretraining specifically buys viewpoint generalization"   vs
#   "any strong visual pretraining beats random features"
#
# The literature check found this is the first question a reviewer asks --
# CortexBench compares ~10 encoders, Burns et al. compares 15, and a
# one-baseline result reads as uncontrolled next to those. So: fill in the
# middle of the range.
#
#   vjepa      V-JEPA 2 ViT-L      video SSL          already cached
#   dinov2     DINOv2 ViT-B        image SSL
#   clip       CLIP ViT-B/16       image-text contrastive
#   vit-in1k   ViT-B/16            supervised ImageNet
#   rand       scratch CNN         no pretraining      already cached
#
# Episode count is pinned to whatever the V-JEPA cache holds, because the arms
# must see identical episodes and V-JEPA is the slow one to re-cache. Parity is
# asserted before any comparison runs.
set -uo pipefail
cd "$(dirname "$0")/.."

TASK=${1:-push}
DATA="data/episodes/r1_${TASK}"
DC="docker compose -f docker/compose.yaml --profile wsl2 run --rm -T dev-wsl"
export DOCKER_UID=1000 DOCKER_GID=1000
mkdir -p logs

[ -d "$DATA" ] || { echo "no $DATA"; exit 1; }

# Match the V-JEPA arm's episode count exactly.
NEP=$(ls "cache/latents/r1_${TASK}__r1_ref"/episode_*.npy 2>/dev/null | wc -l)
[ "$NEP" -gt 0 ] || { echo "no V-JEPA cache for ${TASK}; nothing to match"; exit 1; }
echo "matching the V-JEPA arm at ${NEP} episodes"

POSES=$(python3 -c "
import sys; sys.path.insert(0,'src')
from jetspace.envs.so101_env import R1_POSES
print(' '.join(R1_POSES))")

for spec in "dinov2:dino" "clip:clip" "vit-in1k:vitin1k"; do
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
echo "### parity check"
$DC python scripts/check_encoder_parity.py \
    "r1_${TASK}" "dino_${TASK}" "clip_${TASK}" "vitin1k_${TASK}" "r1cnn_${TASK}"
if [ $? -ne 0 ]; then
    echo "REFUSING to compare: the arms are not matched (see above)."
    exit 1
fi

echo
echo "### held-out viewpoint generalization, per encoder"
for spec in "r1_${TASK}:vjepa" "dino_${TASK}:dinov2" "clip_${TASK}:clip" \
            "vitin1k_${TASK}:vit-in1k" "r1cnn_${TASK}:random"; do
    prefix="${spec%%:*}"; name="${spec##*:}"
    [ -d "cache/latents/${prefix}__r1_ref" ] || continue
    $DC python scripts/e8_canonicalize.py --task "$TASK" --seeds 0 1 2 \
        --prefix "$prefix" --out "cache/e11_${name}.json" \
        > "logs/e11_eval_${name}.log" 2>&1
    b=$(grep -aE "^  baseline "  "logs/e11_eval_${name}.log" | awk '{print $2, $3, $4}')
    m=$(grep -aE "^  multiview " "logs/e11_eval_${name}.log" | awk '{print $2, $3, $4}')
    printf "  %-10s baseline %-16s multiview %s\n" "$name" "${b:-FAILED}" "${m:-FAILED}"
done
