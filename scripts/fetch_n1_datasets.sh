#!/usr/bin/env bash
# Import the real datasets named in docs/prereg-n1.md.
#
# Episode counts are chosen to give each condition a comparable number of
# latents rather than a comparable number of episodes -- the datasets have very
# different episode lengths (R1 ~776 frames, R2 ~299), and a distribution
# comparison cares about sample count, not recording count.
#
# Every import uses --stride 2, taking 30 Hz sources to 15 Hz. Simulation runs
# at 25 Hz, and a frame-rate mismatch is indistinguishable from a domain gap in
# embedding space (audit item B4). Matching is not optional.
#
# Camera choice follows the pre-registration: scene-level cameras for the
# ladder, wrist only for the viewpoint confound check V.
set -uo pipefail
cd "$(dirname "$0")/.."

fetch() {
    name="$1"; repo="$2"; cam="$3"; eps="$4"
    out="data/episodes/$name"
    if [ -d "$out" ] && [ -f "$out/info.json" ]; then
        echo "### $name already imported, skipping"
        return
    fi
    echo "### $name  <-  $repo  [$cam]  x$eps"
    python scripts/fetch_lerobot.py --repo "$repo" --camera "$cam" \
        --episodes "$eps" --stride 2 --out "$out" 2>&1 | tail -6
    echo
}

# --- the ladder -----------------------------------------------------------
# R1 lab A, cubes -> bowl. Already imported at 20 eps for E2/E3; re-imported
# here at 10 so every rung contributes a similar number of latents.
fetch n1_R1_cubes      qb1t/so101_teleop_cubes                      observation.images.ego      10

# R2 / R3 lab B, pen -> mug, two different sessions. This pair is rung S, the
# floor: same lab, same task, same hardware, different day.
fetch n1_R2_penmug_s9  bjb7/so101_pen_mug_10_9                      observation.images.camera_2 10
fetch n1_R3_penmug_s12 bjb7/so101_pen_mug_10_12                     observation.images.camera_2 10

# R4 lab C, blocks. Rung L against R1 (different lab, both rigid pick-place).
fetch n1_R4_blocks_top HashtagRobotics/tic-tac-toe-so101-block-a-clean-v1 observation.images.top   8

# --- confound check V -----------------------------------------------------
# Same dataset, same episodes, same everything except which camera. If this
# rivals the cross-lab gap, viewpoint dominates and the ladder is inconclusive.
fetch n1_V_blocks_wrist HashtagRobotics/tic-tac-toe-so101-block-a-clean-v1 observation.images.wrist 8

echo "=============================================================="
python3 - <<'PY'
import glob
import json
import os
print(f"{'dataset':24s} {'eps':>4} {'frames':>8} {'fps':>4}  camera")
for p in sorted(glob.glob("data/episodes/n1_*/info.json")):
    i = json.load(open(p))
    n = len(glob.glob(f"{os.path.dirname(p)}/episode_*.npz"))
    fr = sum(r["length"] for r in i.get("episodes", [])) if i.get("episodes") else "?"
    print(f"{os.path.basename(os.path.dirname(p)):24s} {n:>4} {str(fr):>8} "
          f"{i.get('fps'):>4}  {i.get('lerobot_camera')}")
PY
