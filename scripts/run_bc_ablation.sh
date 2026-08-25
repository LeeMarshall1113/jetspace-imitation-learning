#!/usr/bin/env bash
# Why does reach BC stop at 79%, and which lever fixes it?
#
# The rebuilt baseline gets 79.3% +- 3.7% with a MEDIAN CLOSEST APPROACH OF
# 3.9 cm against a 4.0 cm success radius. The policy is not failing to find the
# target; it is landing just inside the boundary half the time. That is a
# precision limit.
#
# The encoder downsamples 224 to 112, then three stride-2 stages take it to a
# 14x14 spatial-softmax grid. Over roughly half a metre of workspace that is
# ~3.6 cm per cell -- the same number as the accuracy wall.
#
# Three candidate levers, run as separate arms rather than all at once, so the
# result says WHICH one mattered:
#
#   A  baseline                     112 in, 3 stages -> 14x14
#   B  full input resolution        224 in, 3 stages -> 28x28
#   C  one fewer downsample         112 in, 2 stages -> 28x28
#   D  both                         224 in, 2 stages -> 56x56
#   E  baseline architecture, 5x the data
#
# B and C reach the same grid size by different routes. If they agree, the grid
# is what matters and the route does not. If they disagree, something other than
# resolution is involved and the diagnosis is wrong.
#
# A cosine LR schedule already exists, so the across-seed spread in validation
# loss (0.112 / 0.636 / 0.277) is NOT a missing schedule. Arm E tests whether it
# is data starvation instead.
set -uo pipefail
cd "$(dirname "$0")/.."

SEEDS=${1:-"0 1 2"}
BIG=${2:-2000}
DATA=data/episodes/reach_v2
BIGDATA=data/episodes/reach_v2_big
mkdir -p logs checkpoints/abl

run_arm() {
    tag="$1"; data="$2"; insize="$3"; stages="$4"
    for s in $SEEDS; do
        ck="checkpoints/abl/bc_${tag}_seed${s}.pt"
        [ -f "$ck" ] && continue
        python scripts/train_bc.py --data "$data" --out checkpoints/abl \
            --seed "$s" --epochs 50 --in-size "$insize" --stages "$stages" \
            2>&1 | grep -vE "UserWarning|self.blocks" | tail -1
        [ -f "checkpoints/abl/bc_seed${s}.pt" ] && mv "checkpoints/abl/bc_seed${s}.pt" "$ck"
    done
    echo -n "  $tag: "
    python scripts/eval_policy.py --task reach \
        --checkpoints "checkpoints/abl/bc_${tag}_seed*.pt" --train-data "$data" \
        > "logs/abl_${tag}.log" 2>&1
    grep -aE "checkpoint\(s\): success" "logs/abl_${tag}.log" | tail -1 \
        | sed 's/^.*success/success/' || echo "FAILED (logs/abl_${tag}.log)"
    grep -aE "leak check" "logs/abl_${tag}.log" | tail -1 | sed 's/^/      /'
}

echo "=============================================================="
echo "  BC ablation on reach -- baseline is 79.3% +- 3.7%"
echo "=============================================================="

echo
echo "--- A: baseline, 14x14 grid ---"
run_arm A_base "$DATA" 112 3

echo
echo "--- B: 224 input, 28x28 grid ---"
run_arm B_in224 "$DATA" 224 3

echo
echo "--- C: 2 stages, 28x28 grid ---"
run_arm C_stage2 "$DATA" 112 2

echo
echo "--- D: 224 input + 2 stages, 56x56 grid ---"
run_arm D_both "$DATA" 224 2

# ---- E: more data, baseline architecture --------------------------------
have=$(ls "$BIGDATA"/episode_*.npz 2>/dev/null | wc -l)
if [ "$have" -lt "$BIG" ]; then
    [ "$have" -gt 0 ] && rm -rf "$BIGDATA"
    echo
    echo "### collecting $BIG reach episodes for arm E"
    python scripts/collect_demos.py --task reach --episodes "$BIG" \
        --out "$BIGDATA" --seed 1 2>&1 | tail -3
fi
echo
echo "--- E: baseline architecture, ${BIG} episodes ---"
run_arm E_data "$BIGDATA" 112 3

echo
echo "=============================================================="
echo "  Read: if B and C agree, the FEATURE GRID is the constraint and"
echo "  how you get there does not matter. If E alone lifts it, the"
echo "  policy was data-starved rather than precision-limited. Median"
echo "  closest approach per arm is in logs/abl_*.log -- watch that"
echo "  fall below 4.0 cm, which is what success actually requires."
echo "=============================================================="
