#!/usr/bin/env bash
# E11, final pass: level every arm to the same episodes, add DINOv3, compare.
#
#   bash scripts/run_e11_final.sh push
#
# check_episode_counts.py found the arms were not comparable. The V-JEPA arm
# held 10 episodes on 12 viewpoints, 5 on ten more and none on one, because the
# expanded R1 collection was killed by an OOM partway through re-encoding. Every
# other encoder had been pinned to 10 by sampling r1_ref, which happened to be
# one of the finished poses. Comparing against that would have measured episode
# count as much as encoder.
#
# So: level everything to TARGET episodes per pose, then cache DINOv3 (which
# needed a fix -- its 201 tokens are 196 patches plus CLS plus four register
# tokens, and the old CLS-or-nothing check rejected it), then compare.
set -uo pipefail
cd "$(dirname "$0")/.."

TASK=${1:-push}
TARGET=${2:-10}
DATA="data/episodes/r1_${TASK}"
DC="docker compose -f docker/compose.yaml --profile wsl2 run --rm -T dev-wsl"
export DOCKER_UID=1000 DOCKER_GID=1000
mkdir -p logs

# Never run two renderers at once on this box; the last collision was an OOM
# that killed the collection this script exists to repair.
while pgrep -f "run_e11_modern|cache_latents_hf|watch_dinov3|collect_demos" >/dev/null; do
    echo "  waiting for the running pass to finish ($(date +%T))"
    sleep 120
done

AVAIL=$(ls "$DATA"/episode_*.npz 2>/dev/null | wc -l)
[ "$AVAIL" -ge "$TARGET" ] || { echo "only $AVAIL episodes on disk, need $TARGET"; exit 1; }
echo "levelling every arm to ${TARGET} episodes across 23 poses"

POSES=$(python3 -c "
import sys; sys.path.insert(0,'src')
from jetspace.envs.so101_env import R1_POSES
print(' '.join(R1_POSES))")

# ---- 1. V-JEPA: the slow arm, re-encoded only where short ----------------
echo
echo "### levelling V-JEPA (only poses below ${TARGET})"
for p in $POSES; do
    out="cache/latents/r1_${TASK}__${p}"
    n=$(ls "$out"/episode_*.npy 2>/dev/null | wc -l)
    [ "$n" -ge "$TARGET" ] && continue
    printf "  %-14s %s -> %s " "$p" "$n" "$TARGET"
    $DC python scripts/cache_latents.py --task "r1_${TASK}__${p}" --data "$DATA" \
        --camera "$p" --limit "$TARGET" --pool-grid 4 \
        > "logs/e11_level_vjepa_${p}.log" 2>&1
    echo "-> $(ls "$out"/episode_*.npy 2>/dev/null | wc -l)"
done

# ---- 2. random CNN: fast, re-cache wholesale ------------------------------
echo
echo "### levelling random CNN"
for p in $POSES; do
    out="cache/latents/r1cnn_${TASK}__${p}"
    n=$(ls "$out"/episode_*.npy 2>/dev/null | wc -l)
    [ "$n" -ge "$TARGET" ] && continue
    printf "  %-14s " "$p"
    $DC python scripts/cache_latents_cnn.py --data "$DATA" --camera "$p" \
        --out "$out" --pool-grid 4 --hidden 1024 --frames-per-latent 2 \
        --limit "$TARGET" --seed 0 > "logs/e11_level_cnn_${p}.log" 2>&1
    echo "$(ls "$out"/episode_*.npy 2>/dev/null | wc -l)"
done

# ---- 3. DINOv3 and any other short HF arm ---------------------------------
for spec in "dinov3:dinov3" "aimv2:aimv2" "siglip2:siglip2"; do
    model="${spec%%:*}"; tag="${spec##*:}"
    echo
    echo "### ${model}"
    for p in $POSES; do
        out="cache/latents/${tag}_${TASK}__${p}"
        n=$(ls "$out"/episode_*.npy 2>/dev/null | wc -l)
        [ "$n" -ge "$TARGET" ] && continue
        printf "  %-14s " "$p"
        $DC python scripts/cache_latents_hf.py --model "$model" --data "$DATA" \
            --camera "$p" --out "$out" --limit "$TARGET" --pool-grid 4 \
            --frames-per-latent 2 > "logs/e11_${tag}_${p}.log" 2>&1
        n=$(ls "$out"/episode_*.npy 2>/dev/null | wc -l)
        if [ "$n" -ge "$TARGET" ]; then echo "ok (${n})"; else
            echo "FAILED -- logs/e11_${tag}_${p}.log"
            tail -3 "logs/e11_${tag}_${p}.log" | sed 's/^/      /'
        fi
    done
done

# ---- 4. parity, across every pose rather than just the reference ----------
echo
echo "### episode counts after levelling"
$DC python scripts/check_episode_counts.py "$TASK" || {
    echo "REFUSING to compare: arms are still uneven."; exit 1; }

# ---- 5. evaluate ---------------------------------------------------------
echo
echo "### held-out viewpoint generalization"
for spec in "r1_${TASK}:vjepa2" "dinov3_${TASK}:dinov3" "siglip2_${TASK}:siglip2" \
            "aimv2_${TASK}:aimv2" "dino_${TASK}:dinov2" "clip_${TASK}:clip" \
            "vitin1k_${TASK}:vit-in1k" "r1cnn_${TASK}:random"; do
    prefix="${spec%%:*}"; name="${spec##*:}"
    [ -d "cache/latents/${prefix}__r1_ref" ] || continue
    $DC python scripts/e8_canonicalize.py --task "$TASK" --seeds 0 1 2 \
        --prefix "$prefix" --out "cache/e11_${name}.json" \
        > "logs/e11_eval_${name}.log" 2>&1
    echo "  ${name}: exit $?"
done

echo
$DC python scripts/e11_status.py "$TASK"
