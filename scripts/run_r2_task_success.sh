#!/usr/bin/env bash
# R2: does the latent gap predict TASK SUCCESS, not just rollout accuracy?
#
# The gap this closes. R1's secondary showed latent distance predicts how much
# world-model accuracy survives a camera move -- rho = -0.75 against horizon,
# -0.92 against direction cosine. Both are internal to the world model. Nobody
# outside this repository cares how well a latent rollout tracks; they care
# whether the robot completes the task.
#
# So: train a behaviour-cloning policy on the reference camera, then run it in
# the environment with the camera moved to each R1 pose. The policy is not told
# and is not adapted. It simply receives a displaced view where it expects its
# own. Success rate against Frechet gap is the correlation nobody in this line
# of work has published, ourselves included.
#
# Everything about the setup is already fixed by R1: the poses, the gap curve
# in cache/r1_ruler.json, and the fact that all poses render from one rollout.
# The only new thing is the policy.
#
# Registered prediction, written before running: Spearman rho <= -0.6 between
# gap and success rate. If it FAILS -- if a policy keeps working at gaps that
# wreck the world model -- then latent distance predicts world-model internals
# and not behaviour, and every latent-space evaluation in this literature,
# including N1b and R1, is measuring something narrower than it claims.
set -uo pipefail
cd "$(dirname "$0")/.."

TASK=${1:-push}
SEEDS=${2:-"0 1 2"}
STEPS=${3:-300}
NSEEDS=${4:-30}
mkdir -p logs checkpoints/r2

# ---- a policy on the same task R1 swept -----------------------------------
# The existing bc_seed*.pt were trained on reach; R1 swept push. Matching them
# matters more than reusing a checkpoint.
for s in $SEEDS; do
    ck="checkpoints/r2/bc_${TASK}_seed${s}.pt"
    if [ -f "$ck" ]; then
        echo "### BC $TASK seed $s already trained"
        continue
    fi
    echo "### training BC on $TASK, seed $s"
    python scripts/train_bc.py --data "data/episodes/${TASK}" --out checkpoints/r2 \
        --seed "$s" --epochs 40 2>&1 | grep -vE "UserWarning|self.blocks" | tail -2
    # train_bc names its output bc_seed{N}.pt; keep one per task.
    [ -f "checkpoints/r2/bc_seed${s}.pt" ] && \
        mv "checkpoints/r2/bc_seed${s}.pt" "$ck"
done

POSES=$(python3 -c "
import sys; sys.path.insert(0,'src')
from jetspace.envs.so101_env import R1_POSES
print(' '.join(R1_POSES))")

# ---- evaluate under every displaced camera --------------------------------
echo
echo "### evaluating the reference-trained policy under each pose"
for p in $POSES; do
    out="cache/r2_success_${TASK}__${p}.json"
    [ -f "$out" ] && { echo "  $p: cached"; continue; }
    printf "  %-14s " "$p"
    # Capture the whole run to a file. Piping straight into grep swallowed
    # every error the first time this ran: 23 poses reported nothing at all,
    # and the only symptom was "not enough results to correlate" after an hour
    # of BC training. The cause was a KeyError two frames down.
    python scripts/eval_policy.py --task "$TASK" \
        --checkpoints "checkpoints/r2/bc_${TASK}_seed*.pt" \
        --camera-override "$p" --max-steps "$STEPS" --eval-limit "$NSEEDS" \
        > "logs/r2_${p}.log" 2>&1
    line=$(grep -aE "checkpoint\(s\): success" "logs/r2_${p}.log" | tail -1)
    if [ -z "$line" ]; then
        echo "FAILED -- logs/r2_${p}.log"
        grep -aE "Error|error:" "logs/r2_${p}.log" \
            | grep -avE "EGL|OpenGL|glCheckError" | head -2 | sed 's/^/      /'
        continue
    fi
    echo "$line" | sed 's/^.*success/success/'
    POSE="$p" OUTF="$out" LINE="$line" python3 -c "
import json, os, re
m = re.search(r'success\s+([\d.]+)%\s*\+-\s*([\d.]+)%', os.environ['LINE'])
if m:
    json.dump({'pose': os.environ['POSE'], 'success': float(m.group(1)) / 100,
               'sd': float(m.group(2)) / 100}, open(os.environ['OUTF'], 'w'))
"
done

# ---- correlate against the R1 gap -----------------------------------------
echo
R2_TASK="$TASK" python3 - <<'PY'
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, "src")
from jetspace.envs.so101_env import r1_displacement  # noqa: E402

task = os.environ.get("R2_TASK", "push")
ruler = f"cache/r1_ruler_{task}.json"
if not os.path.exists(ruler):
    ruler = "cache/r1_ruler.json"          # the original push-only path
if not os.path.exists(ruler):
    print(f"no ruler for {task} -- run scripts/run_r1.sh {task} first")
    raise SystemExit
print(f"gap curve: {ruler}\n")

gaps = {r["pose"]: r["frechet"] for r in json.load(open(ruler))["poses"]}

rows = []
for f in glob.glob(f"cache/r2_success_{task}__*.json"):
    d = json.load(open(f))
    pose = d["pose"]
    if pose == "r1_ref":
        ref = d
        continue
    if pose in gaps:
        rows.append({**d, "gap": gaps[pose],
                     "angle": r1_displacement(pose)["angle"]})

ref = next((json.load(open(f))
            for f in glob.glob(f"cache/r2_success_{task}__r1_ref.json")), None)
if not rows or ref is None:
    print("not enough results to correlate")
    raise SystemExit

print("=" * 72)
print("R2 -- does the latent gap predict TASK SUCCESS?")
print("=" * 72)
print(f"reference pose success: {ref['success']:.1%} +- {ref['sd']:.1%}\n")
print(f"{'pose':14s} {'angle':>7} {'gap':>8} {'success':>9} {'retained':>9}")
print("-" * 52)
for r in sorted(rows, key=lambda x: x["gap"]):
    ret = r["success"] / max(ref["success"], 1e-9)
    print(f"{r['pose']:14s} {r['angle']:>6.1f}° {r['gap']:>8.1f} "
          f"{r['success']:>8.1%} {ret:>8.2f}x")

def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 1e-12 else float("nan")

g = np.array([r["gap"] for r in rows])
sc = np.array([r["success"] for r in rows])
rho = spearman(g, sc)

# Bootstrap the correlation. A single rho over 22 non-independent poses is a
# point estimate with no stated uncertainty, which is exactly the gap that
# capped this project's statistical-rigor grade.
rng = np.random.default_rng(0)
boot = []
for _ in range(2000):
    idx = rng.integers(0, len(rows), len(rows))
    if len(set(idx.tolist())) < 4:
        continue
    boot.append(spearman(g[idx], sc[idx]))
lo, hi = np.percentile(boot, [2.5, 97.5])

print()
print(f"Spearman rho, gap vs success : {rho:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]")
print("  (CI by bootstrap over poses, 2000 resamples)")
print()
print("=" * 72)
if hi <= -0.6:
    print("REGISTERED PREDICTION HOLDS, and the interval excludes -0.6.")
    print("  Latent distance predicts TASK SUCCESS, not just world-model")
    print("  internals. That is the link the latent-evaluation literature")
    print("  assumes and does not measure.")
elif rho <= -0.6:
    print("PREDICTION HOLDS on the point estimate; the CI includes -0.6.")
    print("  Directionally right, not yet tight. More poses or more eval seeds.")
else:
    print("REGISTERED PREDICTION FAILS.")
    print(f"  rho = {rho:+.3f}, CI [{lo:+.3f}, {hi:+.3f}].")
    print("  A policy keeps working at gaps that wreck the world model, so")
    print("  latent distance predicts world-model internals and NOT behaviour.")
    print("  Every latent-space evaluation here -- N1b, R1, and the wider")
    print("  literature -- is then measuring something narrower than it claims.")
    print("  Registered in advance as the more consequential outcome.")
print("=" * 72)

outf = f"cache/r2_task_success_{task}.json"
json.dump({"task": task, "reference": ref, "poses": rows, "rho": rho,
           "ci95": [float(lo), float(hi)]}, open(outf, "w"), indent=2, default=float)
print(f"\nwrote {outf}")
PY
