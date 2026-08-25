#!/usr/bin/env bash
# G1: harden the feature-grid result, per docs/prereg-resolution.md.
#
# Five grids from 7x7 to 112x112, five seeds each, and the falsification test
# that the whole diagnosis rests on: score every arm at success radii of 4, 6
# and 8 cm on the SAME rollouts.
#
# If the spatial grid bounds achievable precision, a tolerance far coarser than
# the grid should not care about the grid. So the 14x14-vs-56x56 spread must
# shrink by at least half at 8 cm. If it does not, precision is not the
# mechanism and the explanation is wrong.
set -uo pipefail
cd "$(dirname "$0")/.."

SEEDS=${1:-"0 1 2 3 4"}
DATA=data/episodes/reach_v2
mkdir -p logs checkpoints/g1 cache

# grid = in_size / 2^stages
#   tag        in   stages   grid
ARMS="g007:112:4 g014:112:3 g028:112:2 g056:224:2 g112:224:1"

for spec in $ARMS; do
    tag="${spec%%:*}"; rest="${spec#*:}"; insize="${rest%%:*}"; stages="${rest#*:}"
    for s in $SEEDS; do
        ck="checkpoints/g1/bc_${tag}_seed${s}.pt"
        [ -f "$ck" ] && continue
        python scripts/train_bc.py --data "$DATA" --out checkpoints/g1 \
            --seed "$s" --epochs 50 --in-size "$insize" --stages "$stages" \
            > "logs/g1_train_${tag}_${s}.log" 2>&1
        if [ -f "checkpoints/g1/bc_seed${s}.pt" ]; then
            mv "checkpoints/g1/bc_seed${s}.pt" "$ck"
        else
            echo "  ${tag} seed ${s}: TRAINING FAILED -- logs/g1_train_${tag}_${s}.log"
        fi
    done
    n=$(ls checkpoints/g1/bc_${tag}_seed*.pt 2>/dev/null | wc -l)
    echo "### ${tag} (in ${insize}, ${stages} stages): ${n} checkpoints"
done

# ---- evaluate every arm at every radius, on identical rollouts ----------
echo
echo "### scoring each arm at 4 / 6 / 8 cm"
python3 - <<'PY'
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from jetspace.envs.registry import get_task
from jetspace.utils.device import get_device
from eval_policy import ERROR_KEYS, load_policy

import torch

device = get_device("auto")
seeds = json.loads(Path("configs/eval_seeds.json").read_text())["seeds"]
RADII = (0.04, 0.06, 0.08)

def rollout_closest(env, policy, mean, std, norm, cam, adim, seed):
    """Run once; return the closest approach reached. Success at any radius is
    then a threshold on this single number, so all three radii share rollouts."""
    obs = env.reset(seed=seed)
    best = float("inf")
    terminated = truncated = False
    while not (terminated or truncated):
        a = policy.act(obs.pixels[cam], (obs.proprio - mean) / std,
                       obs.proprio[:adim], norm, device=device)
        r = env.step(a)
        err = next((r.info[k] for k in ERROR_KEYS if k in r.info), None)
        if err is not None:
            best = min(best, float(err))
        obs = r.obs
        terminated, truncated = r.terminated, r.truncated
    return best

# Scored at the LOOSEST radius so no episode terminates early at a tighter one;
# terminating at 4 cm would truncate the trajectory and change what the 8 cm
# score sees.
env = get_task("reach")["env"](image_size=224, max_steps=400, success_radius=max(RADII))

rows = []
for ck in sorted(glob.glob("checkpoints/g1/bc_g*_seed*.pt")):
    m = re.match(r".*bc_(g\d+)_seed(\d+)\.pt", ck)
    if not m:
        continue
    tag, seed = m.group(1), int(m.group(2))
    c = torch.load(ck, map_location="cpu", weights_only=False)
    policy, mean, std, norm, cam, adim = load_policy(Path(ck), device)
    closest = np.array([rollout_closest(env, policy, mean, std, norm, cam, adim, s)
                        for s in seeds])
    rows.append({
        "tag": tag, "seed": seed,
        "grid": c.get("in_size", 112) // (2 ** c.get("stages", 3)),
        "val_loss": c.get("best_val_loss"),
        "median_cm": float(np.median(closest) * 100),
        **{f"succ_{int(r*100)}": float((closest < r).mean()) for r in RADII},
    })
    print(f"  {tag} seed {seed}: grid {rows[-1]['grid']:>3}  "
          f"median {rows[-1]['median_cm']:5.2f} cm  "
          + "  ".join(f"{int(r*100)}cm {rows[-1][f'succ_{int(r*100)}']:.1%}" for r in RADII))

json.dump(rows, open("cache/g1_hardening.json", "w"), indent=2)
print(f"\nwrote cache/g1_hardening.json ({len(rows)} runs)")
PY
