#!/usr/bin/env bash
# E6: is a frozen 326M video foundation model worth it against a 7M CNN?
#
# The answer decides what the paper is about, so the experiment has to be built
# so it cannot hand a win to the wrong arm.
#
#   arm 1  frozen V-JEPA        the incumbent. Cannot cheat: it cannot move.
#   arm 2  frozen RANDOM CNN    same architecture as arm 3, never trained.
#                               Isolates what PRETRAINING buys from what
#                               ARCHITECTURE buys -- if random convolutional
#                               features match V-JEPA, 22M videos bought nothing.
#   arm 3  jointly trained CNN  free to move, and therefore free to cheat.
#
# THE TRAP. Nothing in a prediction loss forbids arm 3 collapsing its own latent
# space. Every latent becomes the same vector, prediction error goes to zero,
# and the world model is worthless. Arms 1 and 2 cannot do this. Comparing raw
# validation losses hands arm 3 the win by construction, every time, whichever
# representation is actually better.
#
# Three controls, all of which already exist in the repo and all of which run
# identically on all three arms:
#
#   gain ratio vs do-nothing   collapse shrinks the baseline as much as the
#                              model error, so the RATIO survives what the raw
#                              loss does not
#   inverse-dynamics probe     a collapsed space cannot recover which action
#                              was taken; R^2 goes to zero
#   shuffled-action test       already inside eval_horizon
#
# All three arms write the same cache layout, so nothing downstream can tell
# which encoder produced a cache -- which is what stops anything downstream
# treating them differently.
set -uo pipefail
cd "$(dirname "$0")/.."

TASK=${1:-push}
EPS=${2:-30}
SEED=${3:-0}
HMAX=${4:-96}
# Optional 5th argument: the episode directory, for when it is not named
# after the task. real_cubes lives in data/episodes/real_so101_teleop_cubes,
# so without this the real-data arm silently finds nothing and every arm
# reports "no latents" instead of failing loudly.
DATA=${5:-"data/episodes/${TASK}"}
VJEPA="cache/latents/${TASK}_s1n60"          # comb-free V-JEPA, already cached
[ -d "$VJEPA" ] || VJEPA="cache/latents/${TASK}_decombed"
[ -d "$VJEPA" ] || VJEPA="cache/latents/${TASK}"

RAND="cache/latents/e6_${TASK}_rand_s${SEED}"
JOINT="cache/latents/e6_${TASK}_joint_s${SEED}"
JOINTREG="cache/latents/e6_${TASK}_jointreg_s${SEED}"
CK="checkpoints/e6"

echo "=============================================================="
echo "  E6   task=$TASK  episodes=$EPS  seed=$SEED"
echo "  arm 1 V-JEPA latents: $VJEPA"
echo "  NOTE: seeding changes the predictor init for every arm, and the"
echo "  ENCODER too for arms 2 and 3. V-JEPA is frozen, so its across-seed"
echo "  spread is predictor-only and will legitimately be the smallest."
echo "=============================================================="

# ---------------------------------------------------------------- arm 2 ----
if [ ! -f "$RAND/info.json" ]; then
    echo
    echo "### arm 2: caching RANDOM CNN latents"
    python scripts/cache_latents_cnn.py --task "$TASK" --data "$DATA" --out "$RAND" \
        --limit "$EPS" --seed "$SEED" 2>&1 | tail -3
fi

# ---------------------------------------------------------------- arm 3 ----
# Arm 3a, unregularised. Kept because it demonstrates the trap is real and not
# a hypothetical: the first smoke run collapsed inside one epoch to a validation
# loss a thousand times below V-JEPA's, while being worse than predicting no
# change at all.
if [ ! -f "$CK/joint_${TASK}_naive_seed${SEED}.pt" ]; then
    echo
    echo "### arm 3a: joint CNN, NO regularisation (expected to collapse)"
    python scripts/train_joint_cnn.py --task "$TASK" --data "$DATA" --out "$CK" \
        --limit "$EPS" --seed "$SEED" --epochs 20 --var-reg 0.0 \
        2>&1 | grep -vE "UserWarning|self.blocks" | tail -12
fi
if [ ! -f "$JOINT/info.json" ]; then
    python scripts/cache_latents_cnn.py --task "$TASK" --data "$DATA" --out "$JOINT" \
        --encoder "$CK/joint_${TASK}_naive_seed${SEED}.pt" --limit "$EPS" --seed "$SEED" \
        2>&1 | tail -2
fi

# Arm 3b, VICReg variance hinge. This is the arm V-JEPA actually has to beat --
# nobody would ship a collapsed encoder, so the unregularised arm alone would be
# a strawman comparison.
if [ ! -f "$CK/joint_${TASK}_reg_seed${SEED}.pt" ]; then
    echo
    echo "### arm 3b: joint CNN + variance regularisation (the fair competitor)"
    python scripts/train_joint_cnn.py --task "$TASK" --data "$DATA" --out "$CK" \
        --limit "$EPS" --seed "$SEED" --epochs 20 --var-reg 1.0 \
        2>&1 | grep -vE "UserWarning|self.blocks" | tail -12
fi
if [ ! -f "$JOINTREG/info.json" ]; then
    python scripts/cache_latents_cnn.py --task "$TASK" --data "$DATA" --out "$JOINTREG" \
        --encoder "$CK/joint_${TASK}_reg_seed${SEED}.pt" --limit "$EPS" --seed "$SEED" \
        2>&1 | tail -2
fi

# ------------------------------------------- identical pipeline, all arms --
run_arm() {
    name="$1"; lat="$2"
    [ -f "$lat/info.json" ] || { echo "### $name: no latents at $lat"; return; }
    echo
    echo "########## arm: $name ##########"

    ckpt="$CK/predictor_${name}_seed${SEED}.pt"
    if [ ! -f "$ckpt" ]; then
        python scripts/train_predictor.py --task "$name" --data "$DATA" --latents "$lat" \
            --out "$CK" --epochs 30 --seed "$SEED" --pca-dim 128 \
            2>&1 | grep -vE "UserWarning|self.blocks" | tail -2
    fi

    echo "-- horizon --"
    python scripts/eval_horizon.py --task "$name" --data "$DATA" --latents "$lat" \
        --checkpoint "$ckpt" --max-horizon "$HMAX" --out "cache/e6_h_${name}.json" \
        2>&1 | grep -E "USEFUL|ACTION-AWARE|CENSORED" | head -3

    echo "-- conservatism --"
    python scripts/check_conservatism.py --task "$name" --data "$DATA" --latents "$lat" \
        --checkpoint "$ckpt" --max-horizon "$HMAX" \
        2>&1 | grep -E "mean displacement|mean direction" | head -2

    echo "-- inverse-dynamics probe (the collapse control) --"
    python scripts/probe_action_signal.py --task "$name" --data "$DATA" --latents "$lat" \
        --episodes "$EPS" 2>&1 | grep -iE "R\^?2|r2|verdict" | head -4

    echo "-- reward signal (E2) --"
    python scripts/eval_reward.py --task "$name" --data "$DATA" --latents "$lat" \
        --limit "$EPS" --out "cache/e6_e2_${name}.json" \
        2>&1 | grep -E "CROSS-episode|Encoder earns" | head -2
}

run_arm "e6_${TASK}_vjepa_s${SEED}"    "$VJEPA"
run_arm "e6_${TASK}_rand_s${SEED}"     "$RAND"
run_arm "e6_${TASK}_joint_s${SEED}"    "$JOINT"
run_arm "e6_${TASK}_jointreg_s${SEED}" "$JOINTREG"

# ---------------------------------------------------------------- summary --
echo
echo "=============================================================="
echo "  E6 SUMMARY"
echo "=============================================================="
python3 - <<'PY'
import glob
import json
import os

arms = []
for f in sorted(glob.glob("cache/e6_h_*.json")):
    name = os.path.basename(f)[len("e6_h_"):-len(".json")]
    h = json.load(open(f))
    row = {"arm": name, "useful": h["useful_horizon"],
           "aware": h["action_aware_horizon"], "censored": h.get("censored", False)}
    e2 = f"cache/e6_e2_{name}.json"
    if os.path.exists(e2):
        d = json.load(open(e2))
        for k in ("cross_latent", "cross_episode_latent", "latent_cross"):
            if k in d:
                row["e2_rho"] = d[k]
                break
    lat = None
    for cand in (f"cache/latents/{name}", ):
        if os.path.exists(cand + "/info.json"):
            lat = json.load(open(cand + "/info.json"))
    if lat:
        row["params"] = lat.get("n_params")
        row["encoder"] = lat.get("model_id", "?")
    arms.append(row)

if not arms:
    print("no arms completed")
else:
    print(f"{'arm':26s} {'useful h':>9} {'aware h':>8} {'censored':>9}")
    print("-" * 58)
    for a in arms:
        print(f"{a['arm']:26s} {a['useful']:>9} {a['aware']:>8} "
              f"{'YES' if a['censored'] else 'no':>9}")

    print()
    print("HOW TO READ THIS, and it is not by the horizon alone:")
    print("  A trained encoder can shrink its own latent space until prediction")
    print("  is trivial. Check the inverse-dynamics R^2 printed per arm above:")
    print("  if the joint arm has a long horizon AND a low probe R^2, it")
    print("  collapsed and its horizon is meaningless.")
    print()
    print("  The decisive comparison is arm 1 vs arm 2. Both are frozen, so")
    print("  neither can cheat, and the difference between them is exactly what")
    print("  22M videos of pretraining bought over random convolutional features.")
    print()
    print("  Arm 3a (naive) is expected to collapse and is kept to show the trap")
    print("  is real. Arm 3b (variance-regularised) is the arm V-JEPA has to")
    print("  beat; comparing against 3a alone would be a strawman.")
PY
