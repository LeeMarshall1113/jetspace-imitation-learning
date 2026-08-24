#!/usr/bin/env bash
# N1b: import, collect, encode. Measurement is run_n1b_measure.py.
#
# Registered in docs/prereg-n1b.md before any of this was run. Scene cameras
# only -- a wrist view rides the gripper and is a different sensing modality,
# not a viewpoint variant, and substituting one for the other is what
# invalidated N1.
#
# Stage guards check COMPLETENESS, not directory existence. A killed run
# previously left 7 of 30 episodes behind and the old guard would have accepted
# them; because the measurement equalises sample count using the smallest
# dataset, that would have silently capped every rung at a fraction of its data.
set -uo pipefail
cd "$(dirname "$0")/.."

STAGE=${1:-all}
REAL_EPS=${2:-8}
SIM_EPS=${3:-12}

# lab | out stem | repo | camera A | camera B
REALS=(
  "A_cubes|n1b_A_cubes|qb1t/so101_teleop_cubes|observation.images.ego|observation.images.external_D455"
  "B_svla|n1b_B_svla|lerobot/svla_so101_pickplace|observation.images.up|observation.images.side"
  "C_tape|n1b_C_tape|ReubenLim/so101_tape_in_square|observation.images.birdEye|observation.images.thirdPerson"
  "D_ball|n1b_D_ball|hellozjt/lerobot_so101_put_ball2cup|observation.images.front|observation.images.side"
  "E_summer|n1b_E_summer|SummerDrinks/LeRobot_SO101|observation.images.front|observation.images.side"
  "F_cup|n1b_F_cup|DecisionFacts/Physical_AI_SO101_Cup_Nesting_Task|observation.images.cam_front|observation.images.cam_top"
  "G_bin|n1b_G_bin|BrutalCaesar/phi_so101_8bin_v1|observation.images.front|observation.images.top"
  "H_penmug1|n1b_H_penmug1|bjb7/so101_pen_mug_10_1|observation.images.camera_2|observation.images.camera_4"
  "H_penmug2|n1b_H_penmug2|bjb7/so101_pen_mug_10_2|observation.images.camera_2|-"
  "H_penmug3|n1b_H_penmug3|bjb7/so101_pen_mug_10_3|observation.images.camera_2|-"
  "H_penmug4|n1b_H_penmug4|bjb7/so101_pen_mug_10_4|observation.images.camera_2|-"
)

import_one() {
    stem="$1"; repo="$2"; cam="$3"
    [ "$cam" = "-" ] && return
    short="${cam##*.}"
    out="data/episodes/${stem}__${short}"
    have=$(ls "$out"/episode_*.npz 2>/dev/null | wc -l)
    if [ "$have" -ge "$REAL_EPS" ]; then
        echo "### ${stem}__${short}: $have episodes, complete"
        return
    fi
    [ "$have" -gt 0 ] && { echo "### ${stem}__${short}: partial ($have), rebuilding"; rm -rf "$out"; }
    echo "### importing ${stem}__${short}  <-  $repo"
    python scripts/fetch_lerobot.py --repo "$repo" --camera "$cam" \
        --episodes "$REAL_EPS" --stride 2 --out "$out" 2>&1 | tail -3
}

if [ "$STAGE" = "all" ] || [ "$STAGE" = "import" ]; then
    for row in "${REALS[@]}"; do
        IFS='|' read -r _lab stem repo ca cb <<< "$row"
        import_one "$stem" "$repo" "$ca"
        import_one "$stem" "$repo" "$cb"
    done
fi

# ------------------------------------------------------------ sim collection
# All five poses render from ONE rollout, so physics, seeds and actions are
# identical across viewpoints by construction rather than by assumption. That
# is the whole reason the simulated sweep is a clean viewpoint reference and a
# comparison across public datasets is not.
CAMS="front,front_high,side,side_high,top"

collect_sim() {
    out="$1"; shift
    have=$(ls "$out"/episode_*.npz 2>/dev/null | wc -l)
    if [ "$have" -ge "$SIM_EPS" ]; then
        echo "### $(basename "$out"): $have episodes, complete"
        return
    fi
    [ "$have" -gt 0 ] && { echo "### $(basename "$out"): partial ($have), rebuilding"; rm -rf "$out"; }
    echo "### collecting $(basename "$out"): 5 poses x meshes"
    python scripts/collect_demos.py --task push --episodes "$SIM_EPS" \
        --pretty --cameras "$CAMS" --out "$out" "$@" 2>&1 | tail -4
}

if [ "$STAGE" = "all" ] || [ "$STAGE" = "collect" ]; then
    collect_sim data/episodes/n1b_sim_push    --seed 0
    collect_sim data/episodes/n1b_sim_push_dr --seed 1 --randomize
fi

# ----------------------------------------------------------------- encoding
# Simulation stores five cameras per episode, so it is encoded once per camera
# into its own cache, matching the one-camera-per-cache layout the real imports
# already use.
if [ "$STAGE" = "all" ] || [ "$STAGE" = "encode" ]; then
    for src in data/episodes/n1b_*; do
        [ -d "$src" ] || continue
        name=$(basename "$src")
        case "$name" in
            n1b_sim_*) cams="front front_high side side_high top" ;;
            *)         cams="" ;;
        esac
        if [ -z "$cams" ]; then
            out="cache/latents/$name"
            eps=$(ls "$src"/episode_*.npz 2>/dev/null | wc -l)
            lat=$(ls "$out"/episode_*.npy 2>/dev/null | wc -l)
            if [ -f "$out/info.json" ] && [ "$lat" -ge "$eps" ] && [ "$eps" -gt 0 ]; then
                echo "### $name encoded ($lat/$eps)"
                continue
            fi
            [ "$lat" -gt 0 ] && rm -rf "$out"
            echo "### encoding $name"
            python scripts/cache_latents.py --task "$name" --data "$src" --out "$out" \
                --chunk 32 --margin 15 --pool-grid 4 \
                2>&1 | grep -vE "UserWarning|self.blocks|Loading weights" | tail -2
        else
            for c in $cams; do
                out="cache/latents/${name}__${c}"
                eps=$(ls "$src"/episode_*.npz 2>/dev/null | wc -l)
                lat=$(ls "$out"/episode_*.npy 2>/dev/null | wc -l)
                if [ -f "$out/info.json" ] && [ "$lat" -ge "$eps" ] && [ "$eps" -gt 0 ]; then
                    echo "### ${name}__${c} encoded ($lat/$eps)"
                    continue
                fi
                [ "$lat" -gt 0 ] && rm -rf "$out"
                echo "### encoding ${name}__${c}"
                python scripts/cache_latents.py --task "${name}__${c}" --data "$src" \
                    --out "$out" --camera "$c" --chunk 32 --margin 15 --pool-grid 4 \
                    2>&1 | grep -vE "UserWarning|self.blocks|Loading weights" | tail -2
            done
        fi
    done
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "measure" ]; then
    python scripts/run_n1b_measure.py 2>&1 | grep -vE "UserWarning|self.blocks"
fi
