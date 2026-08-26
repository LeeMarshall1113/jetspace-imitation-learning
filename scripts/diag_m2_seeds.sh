#!/usr/bin/env bash
# Torque is ruled out. What else could turn 85.7% +- 2.6% into 31% +- 26%?
#
# The tell is the per-checkpoint spread: 6.7 / 16.7 / 70.0 at one torque and
# 10.0 / 16.7 / 66.7 at another. Torque barely moves it; the CHECKPOINTS differ
# enormously from each other. M2 reported a tight +-2.6 across the same three
# files, so either the eval set changed or the files did.
#
# Three checks:
#   1. all 100 eval seeds, not the first 30 -- if the first 30 are unlucky the
#      mean moves, though it cannot explain a spread this wide
#   2. the checkpoints' own recorded config and val loss, which says whether
#      these are the files M2 measured
#   3. what the reach dataset's info.json actually records
cd "$(dirname "$0")/.."

echo "=== 1. checkpoint provenance ==="
python3 - <<'PY'
import glob
from pathlib import Path
import torch
for p in sorted(glob.glob("checkpoints/bc_seed*.pt")):
    c = torch.load(p, map_location="cpu", weights_only=False)
    cfg = c.get("config", {})
    print(f"  {Path(p).name:16s} val {c.get('best_val_loss', float('nan')):.5f}  "
          f"data {cfg.get('data')}  epochs {cfg.get('epochs')}  seed {cfg.get('seed')}")
    print(f"  {'':16s} mtime {Path(p).stat().st_mtime:.0f}")
PY

echo
echo "=== 2. reach dataset info.json keys ==="
python3 - <<'PY'
import json
import os
for name in ("reach", "push", "pickplace"):
    f = f"data/episodes/{name}/info.json"
    if not os.path.exists(f):
        print(f"  {name}: no info.json")
        continue
    i = json.load(open(f))
    eps = i.get("episodes")
    print(f"  {name}: keys={sorted(k for k in i if k != 'episodes')}")
    if isinstance(eps, list) and eps:
        ok = sum(1 for e in eps if e.get("success"))
        print(f"     {len(eps)} episodes, expert success {ok/len(eps):.1%}, "
              f"servo={i.get('servo')}")
    else:
        print(f"     episodes field: {type(eps).__name__}")
PY

echo
echo "=== 3. M2 policy on ALL 100 eval seeds, BOM torque ==="
python3 - <<'PY'
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from jetspace.envs.registry import get_task
from jetspace.utils.device import get_device
from eval_policy import load_policy, rollout

device = get_device("auto")
seeds = json.loads(Path("configs/eval_seeds.json").read_text())["seeds"]
env = get_task("reach")["env"](image_size=224, max_steps=150)
rates = []
for p in sorted(glob.glob("checkpoints/bc_seed*.pt")):
    policy, mean, std, norm, cam, adim = load_policy(Path(p), device)
    n = sum(rollout(env, policy, mean, std, norm, cam, adim, s, device)[0] for s in seeds)
    rates.append(n / len(seeds))
    print(f"  {Path(p).name:16s} success {n/len(seeds):6.1%}  ({len(seeds)} seeds)")
a = np.array(rates)
print(f"  MEAN {a.mean():.1%} +- {a.std():.1%}")
print()
print("  M2 reported 85.7% +- 2.6%. A spread of this size across the SAME three")
print("  checkpoints cannot come from the eval set; it means the files, the env,")
print("  or the success criterion is not what M2 measured.")
PY
