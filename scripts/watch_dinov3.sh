#!/usr/bin/env bash
# Poll for DINOv3 access; cache and evaluate it the moment approval lands.
#
#   nohup bash scripts/watch_dinov3.sh push > logs/dinov3_watch.log 2>&1 &
#
# DINOv3 (2025-08) is the direct image-SSL contemporary of V-JEPA 2 (2025-06)
# and therefore the single most informative arm in E11: if a modern image
# encoder matches V-JEPA on held-out viewpoints, the claim collapses from "video
# pretraining buys viewpoint generalization" to "any modern pretraining does",
# which the literature already covers.
#
# Meta approves DINOv3 requests manually, so the wait is unbounded. Polling
# hourly rather than asking anyone to watch a web page.
set -uo pipefail
cd "$(dirname "$0")/.."

TASK=${1:-push}
INTERVAL=${2:-3600}
DC="docker compose -f docker/compose.yaml --profile wsl2 run --rm -T dev-wsl"
export DOCKER_UID=1000 DOCKER_GID=1000
mkdir -p logs

granted() {
    $DC python3 -c "
from huggingface_hub import hf_hub_download, get_token
hf_hub_download('facebook/dinov3-vitb16-pretrain-lvd1689m','config.json',token=get_token())
" >/dev/null 2>&1
}

echo "watching for DINOv3 approval, polling every ${INTERVAL}s from $(date +%T)"
tries=0
until granted; do
    tries=$((tries + 1))
    # One line per hour, not per poll, so the log stays readable over a long wait.
    echo "  still awaiting approval (check ${tries}, $(date +%T))"
    sleep "$INTERVAL"
done
echo "DINOv3 ACCESS GRANTED at $(date +%T) after ${tries} checks"

# Do not fight the other passes for the GPU.
while pgrep -f "run_e11" >/dev/null; do
    echo "  waiting for the running E11 pass to finish"
    sleep 120
done

NEP=$(ls "cache/latents/r1_${TASK}__r1_ref"/episode_*.npy 2>/dev/null | wc -l)
[ "$NEP" -gt 0 ] || { echo "no V-JEPA cache to match"; exit 1; }
echo "caching DINOv3 at ${NEP} episodes to match the V-JEPA arm"

POSES=$(python3 -c "
import sys; sys.path.insert(0,'src')
from jetspace.envs.so101_env import R1_POSES
print(' '.join(R1_POSES))")

for p in $POSES; do
    out="cache/latents/dinov3_${TASK}__${p}"
    n=$(ls "$out"/episode_*.npy 2>/dev/null | wc -l)
    [ "$n" -ge "$NEP" ] && continue
    printf "  %-14s " "$p"
    $DC python scripts/cache_latents_hf.py --model dinov3 --data "data/episodes/r1_${TASK}" \
        --camera "$p" --out "$out" --limit "$NEP" --pool-grid 4 \
        --frames-per-latent 2 > "logs/e11_dinov3_${p}.log" 2>&1
    n=$(ls "$out"/episode_*.npy 2>/dev/null | wc -l)
    if [ "$n" -ge "$NEP" ]; then echo "ok (${n})"; else
        echo "FAILED -- logs/e11_dinov3_${p}.log"
        tail -3 "logs/e11_dinov3_${p}.log" | sed 's/^/      /'
    fi
done

echo
echo "### parity against the V-JEPA arm"
$DC python scripts/check_encoder_parity.py "r1_${TASK}" "dinov3_${TASK}" || {
    echo "REFUSING to compare: arms are not matched."; exit 1; }

echo
echo "### DINOv3 on held-out viewpoints"
$DC python scripts/e8_canonicalize.py --task "$TASK" --seeds 0 1 2 \
    --prefix "dinov3_${TASK}" --out "cache/e11_dinov3.json" \
    > logs/e11_eval_dinov3.log 2>&1
grep -aE "^  (baseline|canonical|multiview) " logs/e11_eval_dinov3.log
echo
echo "compare against cache/e11_*.json for the other arms."
