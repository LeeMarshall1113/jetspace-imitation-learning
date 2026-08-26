#!/usr/bin/env bash
# Why does M2's 85.7% not reproduce?
#
# The M2 gate was measured at commit 5cd3cb5. Commit fb2d79c then clamped
# actuator_forcerange to the real Feetech stall torque, which is a correct
# change -- a policy trained against Menagerie's stronger default can learn
# motions the real servos cannot execute -- but M2 was never re-measured after
# it. If that is the cause, the published number describes an environment that
# no longer exists.
#
# Two checks, both cheap:
#   1. the scripted EXPERT's success rate on the current env. If the expert
#      cannot do the task at the real torque, no policy trained from its data
#      can either, and the task itself needs revisiting.
#   2. the M2 policy at the Menagerie torque vs the BOM torque, same seeds.
cd "$(dirname "$0")/.."

echo "=== what torque variants exist ==="
python3 - <<'PY'
import sys
sys.path.insert(0, "src")
from jetspace.envs.so101_env import SERVO_TORQUE_NM, DEFAULT_SERVO
for k, v in SERVO_TORQUE_NM.items():
    mark = "  <-- default" if k == DEFAULT_SERVO else ""
    print(f"  {k:28s} {v:.2f} N-m{mark}")
PY

echo
echo "=== reach dataset: what success rate did the EXPERT get? ==="
python3 - <<'PY'
import glob
import json
import os
for d in sorted(glob.glob("data/episodes/*/info.json")):
    name = os.path.basename(os.path.dirname(d))
    if name.split("_")[0] not in ("reach", "push", "pickplace", "r1"):
        continue
    i = json.load(open(d))
    eps = i.get("episodes", [])
    if not eps:
        continue
    ok = sum(1 for e in eps if e.get("success"))
    print(f"  {name:22s} {len(eps):4d} episodes   expert success {ok/len(eps):6.1%}"
          f"   servo {i.get('servo', '?')}")
PY

echo
echo "=== M2 policy at each torque, identical seeds ==="
for servo in sts3215_12v sts3215_7v4_c001; do
    echo "--- servo: $servo ---"
    SERVO="$servo" python3 - <<'PY'
import os
import sys
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import glob
import json
from pathlib import Path

import numpy as np
import torch

from jetspace.envs.registry import get_task
from jetspace.utils.device import get_device
from eval_policy import ERROR_KEYS, load_policy, rollout

servo = os.environ["SERVO"]
device = get_device("auto")
seeds = json.loads(Path("configs/eval_seeds.json").read_text())["seeds"][:30]
try:
    env = get_task("reach")["env"](image_size=224, max_steps=150, servo=servo)
except ValueError as exc:
    print(f"  {exc}")
    raise SystemExit
rates = []
for p in sorted(glob.glob("checkpoints/bc_seed*.pt")):
    policy, mean, std, norm, cam, adim = load_policy(Path(p), device)
    n = sum(rollout(env, policy, mean, std, norm, cam, adim, s, device)[0] for s in seeds)
    rates.append(n / len(seeds))
    print(f"  {Path(p).name:20s} success {n/len(seeds):6.1%}")
if rates:
    a = np.array(rates)
    print(f"  MEAN {a.mean():.1%} +- {a.std():.1%}")
PY
done
