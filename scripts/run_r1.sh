#!/usr/bin/env bash
# R1: collect the camera sweep, encode it, measure the ruler.
#
# Registered in docs/prereg-camera-ruler.md before anything here ran.
#
# All 23 poses render from ONE rollout per episode, so seeds, physics, actions
# and meshes are identical across viewpoints by construction rather than by
# assumption. That is the entire reason a simulated sweep can calibrate a ruler
# and a comparison across public datasets cannot: every real lab varies its
# camera together with its room, lighting, table and operator.
set -uo pipefail
cd "$(dirname "$0")/.."

TASK=${1:-push}
EPS=${2:-8}
STAGE=${3:-all}

OUT="data/episodes/r1_${TASK}"
CAMS=$(python3 -c "
import sys; sys.path.insert(0,'src')
from jetspace.envs.so101_env import R1_POSES
print(','.join(R1_POSES))")

if [ "$STAGE" = "all" ] || [ "$STAGE" = "collect" ]; then
    have=$(ls "$OUT"/episode_*.npz 2>/dev/null | wc -l)
    if [ "$have" -ge "$EPS" ]; then
        echo "### $OUT: $have episodes, complete"
    else
        [ "$have" -gt 0 ] && { echo "### partial ($have), rebuilding"; rm -rf "$OUT"; }
        echo "### collecting $EPS episodes x 23 poses, mesh-rendered"
        echo "    (the slow step: one render per pose per timestep)"
        python scripts/collect_demos.py --task "$TASK" --episodes "$EPS" \
            --pretty --cameras "$CAMS" --out "$OUT" --seed 0 2>&1 | tail -4
    fi
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "encode" ]; then
    for cam in $(echo "$CAMS" | tr ',' ' '); do
        dst="cache/latents/r1_${TASK}__${cam}"
        eps=$(ls "$OUT"/episode_*.npz 2>/dev/null | wc -l)
        lat=$(ls "$dst"/episode_*.npy 2>/dev/null | wc -l)
        if [ -f "$dst/info.json" ] && [ "$lat" -ge "$eps" ] && [ "$eps" -gt 0 ]; then
            echo "### ${cam} encoded ($lat/$eps)"
            continue
        fi
        [ "$lat" -gt 0 ] && rm -rf "$dst"
        echo "### encoding ${cam}"
        python scripts/cache_latents.py --task "r1_${TASK}__${cam}" --data "$OUT" \
            --out "$dst" --camera "$cam" --chunk 32 --margin 15 --pool-grid 4 \
            2>&1 | grep -vE "UserWarning|self.blocks|Loading weights" | tail -1
    done
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "measure" ]; then
    python scripts/measure_camera_ruler.py --prefix "r1_${TASK}" \
        2>&1 | grep -vE "UserWarning|self.blocks"
fi
