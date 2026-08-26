#!/usr/bin/env bash
# Cache VC-1 across the R1 sweep and fold it into the E11 comparison.
#
#   bash scripts/run_vc1.sh push 10
#
# VC-1 (NeurIPS 2023) is the robotics-specific encoder a reviewer looks for
# first in a manipulation benchmark, and the one the CortexBench results are
# built on. It is not a transformers model -- the Hub ships an MAE ViT-B/16
# state dict plus a hydra config -- so cache_latents_hf.py rebuilds it with
# timm, which lives in /workspace/.pydeps because the container cannot write to
# its own venv.
#
# Theia and R3M are NOT here and their absence is deliberate: Theia's remote
# modelling code predates transformers 5.x and fails on every loading path, and
# R3M is not on the Hub while its GitHub install is blocked by container
# permissions. Both are reported as unavailable rather than silently omitted.
set -uo pipefail
cd "$(dirname "$0")/.."

TASK=${1:-push}
TARGET=${2:-10}
DC="docker compose -f docker/compose.yaml --profile wsl2 run --rm -T -e PYTHONPATH=/workspace/.pydeps dev-wsl"
export DOCKER_UID=1000 DOCKER_GID=1000
mkdir -p logs

POSES=$(python3 -c "
import sys
sys.path.insert(0, 'src')
from jetspace.envs.so101_env import R1_POSES
print(' '.join(R1_POSES))")

echo "### caching VC-1 across $(echo "$POSES" | wc -w) poses at ${TARGET} episodes"
for p in $POSES; do
    out="cache/latents/vc1_${TASK}__${p}"
    n=$(ls "$out"/episode_*.npy 2>/dev/null | wc -l)
    [ "$n" -ge "$TARGET" ] && continue
    printf "  %-14s " "$p"
    $DC python scripts/cache_latents_hf.py --model vc1 \
        --data "data/episodes/r1_${TASK}" --camera "$p" --out "$out" \
        --limit "$TARGET" --pool-grid 4 --frames-per-latent 2 \
        > "logs/e11_vc1_${p}.log" 2>&1
    n=$(ls "$out"/episode_*.npy 2>/dev/null | wc -l)
    if [ "$n" -ge "$TARGET" ]; then echo "ok (${n})"; else
        echo "FAILED -- logs/e11_vc1_${p}.log"
        tail -3 "logs/e11_vc1_${p}.log" | sed 's/^/      /'
    fi
done

echo
echo "### episode counts across every arm"
$DC python scripts/check_episode_counts.py "$TASK" || {
    echo "REFUSING to compare: arms are uneven."; exit 1; }

echo
echo "### evaluating VC-1 on held-out viewpoints"
$DC python scripts/e8_canonicalize.py --task "$TASK" --seeds 0 1 2 \
    --prefix "vc1_${TASK}" --out cache/e11_vc1.json \
    > logs/e11_eval_vc1.log 2>&1
echo "  exit=$?"

echo
$DC python scripts/e11_compare.py
