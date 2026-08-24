#!/usr/bin/env bash
# A second, separately labelled viewpoint reference. NOT an edit to the
# pre-registration -- docs/prereg-n1.md says explicitly that a design flaw is
# fixed by adding a labelled analysis, never by revising the registered one.
#
# WHY. The registered V rung used R4's two cameras: `wrist` and `top`. That
# came back at or above the cross-lab rung L on centroid and Frechet, which by
# the registered reading rule makes the ladder INCONCLUSIVE -- viewpoint
# dominating means no cross-dataset gap can be attributed to domain.
#
# That verdict stands. But V as instantiated is the most extreme viewpoint
# change available: a wrist camera rides the gripper and sees a close-up that
# moves with the arm, while a top camera is static and frames the whole scene.
# They do not observe the same thing in any useful sense. It is an upper bound
# on viewpoint effect, not a typical one, and the *severity* of the confound is
# therefore overstated by it.
#
# V2 is the milder and more relevant control: R1's two SCENE-level cameras,
# `ego` and `external_D455`. Same lab, same session, same episodes, same task,
# both static, both framing the workspace -- differing only in where the tripod
# sits. That is the kind of viewpoint variation that actually separates two
# labs recording the same task, and it is the right yardstick for asking
# whether L and T measure domain or furniture.
#
#   V2 small, well below L  -> ordinary viewpoint change is cheap; the original
#                              V was an outlier and the ladder is worth
#                              rebuilding on scene cameras only.
#   V2 comparable to L      -> even mild viewpoint rivals cross-lab domain, and
#                              the INCONCLUSIVE verdict is confirmed rather
#                              than being an artifact of a harsh control.
set -uo pipefail
cd "$(dirname "$0")/.."

D=data/episodes/n1_V2_cubes_external
if [ ! -d "$D" ]; then
    echo "### importing R1 second camera (external_D455)"
    python scripts/fetch_lerobot.py --repo qb1t/so101_teleop_cubes \
        --camera observation.images.external_D455 --episodes 10 --stride 2 \
        --out "$D" 2>&1 | tail -5
    echo
fi

if [ ! -f cache/latents/n1_V2_cubes_external/info.json ]; then
    echo "### encoding (comb-free, same settings as every other rung)"
    python scripts/cache_latents.py --task n1_V2_cubes_external --data "$D" \
        --out cache/latents/n1_V2_cubes_external \
        --chunk 32 --margin 15 --pool-grid 4 \
        2>&1 | grep -vE "UserWarning|self.blocks|Loading weights" | tail -3
    echo
fi

# Same n as every registered rung, so V2 slots into the same table.
N=$(python3 - <<'PY'
import glob
import json
ns = [json.load(open(f))["n_per_side"] for f in glob.glob("cache/n1_*.json")]
print(min(ns) if ns else 574)
PY
)
echo "### rung V2 at n=$N (matching the registered rungs)"
python scripts/measure_domain_gap.py \
    --reference cache/latents/n1_R1_cubes \
    --other     cache/latents/n1_V2_cubes_external \
    --label V2 --cap "$N" --out cache/n1_V2.json \
    2>&1 | grep -vE "UserWarning|self.blocks"

echo
echo "================== LADDER, WITH BOTH VIEWPOINT CONTROLS =================="
python3 - <<'PY'
import glob
import json
order = ["V", "V2", "S", "L", "T", "SIM_push", "SIM_pickplace", "SIM_push_DR"]
note = {
    "V": "viewpoint, EXTREME (wrist vs top)",
    "V2": "viewpoint, mild (two scene cameras)",
    "S": "session, same lab+task  [the floor]",
    "L": "different lab, similar task",
    "T": "different lab, different task",
    "SIM_push": "simulation",
    "SIM_pickplace": "simulation",
    "SIM_push_DR": "simulation + domain randomisation",
}
rows = {}
for f in glob.glob("cache/n1_*.json"):
    d = json.load(open(f))
    rows[d["label"]] = d
print(f"{'rung':16s} {'centroid':>10} {'MMD^2':>10} {'Frechet':>10}   what varies")
print("-" * 82)
for k in order:
    d = rows.get(k)
    if not d:
        continue
    print(f"{k:16s} {d['centroid_pca']:>10.3f} {d['mmd2_pca']:>10.5f} "
          f"{d['frechet_pca']:>10.1f}   {note.get(k, '')}")
PY
