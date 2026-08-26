#!/usr/bin/env bash
# Is the M2 regression a RENDERING change?
#
# M2 was measured at 5cd3cb5. Commit a645541 then made collision primitives the
# default render for an 11x speedup. The BC policy consumes PIXELS. A policy
# trained and validated on mesh renders, then evaluated on blocky collision
# primitives, is being shown a different visual distribution than it ever saw.
#
# Torque is already ruled out (31% at both settings), as are the eval seeds
# (30.3% over all 100) and the success criterion (unchanged at 0.04 m). The
# checkpoints are the originals. Rendering is the remaining candidate that the
# policy can actually see.
cd "$(dirname "$0")/.."

python3 - <<'PY'
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from jetspace.envs.registry import get_task
from jetspace.utils.device import get_device
from eval_policy import load_policy, rollout

device = get_device("auto")
seeds = json.loads(Path("configs/eval_seeds.json").read_text())["seeds"]
ckpts = sorted(glob.glob("checkpoints/bc_seed*.pt"))

for pretty in (False, True):
    label = "MESHES (as at M2)" if pretty else "collision primitives (current default)"
    env = get_task("reach")["env"](image_size=224, max_steps=150, pretty=pretty)
    rates = []
    for p in ckpts:
        policy, mean, std, norm, cam, adim = load_policy(Path(p), device)
        n = sum(rollout(env, policy, mean, std, norm, cam, adim, s, device)[0]
                for s in seeds)
        rates.append(n / len(seeds))
    a = np.array(rates)
    per = "  ".join(f"{r:.1%}" for r in a)
    print(f"{label:42s} {a.mean():6.1%} +- {a.std():5.1%}   [{per}]")

print()
print("M2 reported 85.7% +- 2.6%.")
PY
