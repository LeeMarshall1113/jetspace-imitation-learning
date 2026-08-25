#!/usr/bin/env bash
# Did I break M2 by giving the policy less time than M2 gave it?
#
# Every diagnostic so far passed --max-steps 150, taken from the registry's
# default_steps for reach. eval_policy's OWN default is 400. If M2 was measured
# at the default, the policy had 2.7x longer to get inside the 4 cm radius, and
# every number I have reported as a "regression" was measured under a budget M2
# never used.
#
# Torque, eval seeds, success radius and rendering are all ruled out. This is
# the remaining difference between how M2 was run and how I have been running
# it, and it is a difference I introduced.
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
print(f"{len(ckpts)} checkpoints, {len(seeds)} eval seeds\n")

print(f"{'max_steps':>10} {'mean':>8} {'sd':>7}   per checkpoint")
print("-" * 52)
for ms in (150, 250, 400):
    env = get_task("reach")["env"](image_size=224, max_steps=ms)
    rates = []
    for p in ckpts:
        policy, mean, std, norm, cam, adim = load_policy(Path(p), device)
        n = sum(rollout(env, policy, mean, std, norm, cam, adim, s, device)[0]
                for s in seeds)
        rates.append(n / len(seeds))
    a = np.array(rates)
    per = "  ".join(f"{r:.0%}" for r in a)
    flag = "   <-- eval_policy default" if ms == 400 else ""
    print(f"{ms:>10} {a.mean():>7.1%} {a.std():>6.1%}   {per}{flag}")

print()
print("M2 reported 85.7% +- 2.6%.")
PY
