#!/usr/bin/env bash
# R1 secondary: does the latent gap predict anything about behaviour?
#
# Registered in docs/prereg-camera-ruler.md §7, before the ruler was measured.
#
# N1b measures DISTANCE. E3 measures USEFULNESS. Nothing connects them, and a
# representational gap that predicts nothing about behaviour is a curiosity
# rather than a finding -- including ours.
#
# The test: train the world model once on the reference pose, then evaluate that
# same model on every displaced pose. The predictor's normalisation and PCA
# basis travel with the checkpoint, so a displaced pose is genuinely out of
# distribution for it. Correlate each pose's Frechet gap against how much
# horizon and direction accuracy survive.
#
# Registered prediction: Spearman rho <= -0.6 between gap and retained horizon.
#
#   If it HOLDS  the ruler forecasts degradation and becomes a tool.
#   If it FAILS  a large latent gap costs nothing behaviourally, which means
#                latent distance is the wrong thing for the field to be
#                measuring -- ourselves, N1b, and the Domain Invariance Score
#                line of work included. That is the more valuable outcome and
#                it is why this was registered rather than hoped for.
#
# Every pose shares the SAME episodes and the same rollout; only the camera
# differs. So any degradation is attributable to viewpoint and to nothing else.
set -uo pipefail
cd "$(dirname "$0")/.."

TASK=${1:-push}
HMAX=${2:-64}
SEED=${3:-0}

DATA="data/episodes/r1_${TASK}"
CK="checkpoints/r1"
REF="cache/latents/r1_${TASK}__r1_ref"
mkdir -p logs "$CK"

[ -d "$REF" ] || { echo "no reference latents at $REF"; exit 1; }

# ---- train once, on the reference pose only ------------------------------
ckpt="$CK/predictor_r1_${TASK}_ref_seed${SEED}.pt"
if [ ! -f "$ckpt" ]; then
    echo "### training the world model on the REFERENCE pose only"
    python scripts/train_predictor.py --task "r1_${TASK}_ref" --data "$DATA" \
        --latents "$REF" --out "$CK" --epochs 30 --seed "$SEED" --pca-dim 128 \
        2>&1 | grep -vE "UserWarning|self.blocks" | tail -2
fi

# ---- evaluate that model on every pose -----------------------------------
POSES=$(python3 -c "
import sys; sys.path.insert(0,'src')
from jetspace.envs.so101_env import R1_POSES
print(' '.join(R1_POSES))")

echo
echo "### evaluating the reference-trained model on each displaced pose"
for p in $POSES; do
    lat="cache/latents/r1_${TASK}__${p}"
    [ -d "$lat" ] || continue
    printf "  %-14s " "$p"
    python scripts/eval_horizon.py --task "r1sec_${p}" --data "$DATA" --latents "$lat" \
        --checkpoint "$ckpt" --max-horizon "$HMAX" \
        --out "cache/r1sec_h_${p}.json" 2>&1 \
        | grep -E "USEFUL HORIZON" | sed 's/USEFUL HORIZON *: */h=/' | tr -d '\n'
    python scripts/check_conservatism.py --task "r1sec_${p}" --data "$DATA" \
        --latents "$lat" --checkpoint "$ckpt" --max-horizon "$HMAX" \
        2>&1 | grep "mean direction" | sed 's/.*cosine *//' | cut -c1-6 \
        | tr -d '\n' | sed 's/^/  cos=/'
    echo
done

# ---- correlate -----------------------------------------------------------
echo
python3 - <<'PY'
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, "src")
from jetspace.envs.so101_env import r1_displacement  # noqa: E402

ruler = json.load(open("cache/r1_ruler.json"))
gaps = {r["pose"]: r["frechet"] for r in ruler["poses"]}

rows = []
for f in glob.glob("cache/r1sec_h_*.json"):
    pose = os.path.basename(f)[len("r1sec_h_"):-len(".json")]
    h = json.load(open(f))
    c = f"cache/conservatism_r1sec_{pose}.json"
    cos = json.load(open(c))["mean_cosine"] if os.path.exists(c) else None
    rows.append({"pose": pose, "gap": gaps.get(pose, 0.0),
                 "useful": h["useful_horizon"], "cosine": cos,
                 "angle": r1_displacement(pose)["angle"]})

ref = next((r for r in rows if r["pose"] == "r1_ref"), None)
rows = [r for r in rows if r["pose"] != "r1_ref" and r["cosine"] is not None]
if not rows or ref is None:
    print("not enough results to correlate")
    raise SystemExit

print("=" * 74)
print("R1 SECONDARY -- does the gap predict behaviour?")
print("=" * 74)
print(f"reference pose: horizon {ref['useful']}, cosine {ref['cosine']:.3f}\n")
print(f"{'pose':14s} {'angle':>7} {'gap':>8} {'horizon':>8} {'retained':>9} {'cosine':>8}")
print("-" * 60)
for r in sorted(rows, key=lambda x: x["gap"]):
    ret = r["useful"] / max(ref["useful"], 1)
    print(f"{r['pose']:14s} {r['angle']:>6.1f}° {r['gap']:>8.1f} {r['useful']:>8} "
          f"{ret:>8.2f}x {r['cosine']:>8.3f}")

def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 1e-12 else float("nan")

g = np.array([r["gap"] for r in rows])
hh = np.array([r["useful"] for r in rows], dtype=float)
cc = np.array([r["cosine"] for r in rows])

rho_h = spearman(g, hh)
rho_c = spearman(g, cc)

print()
print(f"Spearman rho, gap vs retained horizon : {rho_h:+.3f}")
print(f"Spearman rho, gap vs direction cosine : {rho_c:+.3f}")
print()
print("=" * 74)
if rho_h <= -0.6:
    print("REGISTERED PREDICTION HOLDS (rho <= -0.6).")
    print("  Latent distance forecasts behavioural degradation. The ruler is a")
    print("  tool: measure the gap, predict how much horizon you lose.")
elif rho_c <= -0.6:
    print("PARTIAL: horizon is insensitive but DIRECTION tracks the gap.")
    print("  Beating a do-nothing baseline is easy and survives; predicting the")
    print("  right motion is what degrades. Report the cosine, not the horizon.")
else:
    print("REGISTERED PREDICTION FAILS.")
    print(f"  gap vs horizon rho = {rho_h:+.3f}, gap vs cosine rho = {rho_c:+.3f}.")
    print()
    print("  A large latent gap costs little or nothing behaviourally. Latent")
    print("  distance is then the wrong thing to be measuring -- and that")
    print("  indicts N1b, R1's own conversion, and the Domain Invariance Score")
    print("  line of work alike. Registered in advance as the more valuable")
    print("  outcome; it is not a disappointment, it is the result.")
print("=" * 74)

json.dump({"reference": ref, "poses": rows,
           "rho_gap_horizon": rho_h, "rho_gap_cosine": rho_c},
          open("cache/r1_secondary.json", "w"), indent=2, default=float)
print("\nwrote cache/r1_secondary.json")
PY
