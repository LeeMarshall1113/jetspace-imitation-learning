#!/usr/bin/env bash
# Wait for the current GPU jobs to drain, then run the two things that need a
# free GPU: the g007 backfill and the E1 exchange-rate sweep.
#
# Why a queue rather than just launching: three jobs already hold the card
# (A1 at budget 200, the R1 pickplace collection, the G1 ladder). Adding a
# fourth renderer does not make anything finish sooner, it makes all four
# finish later, and MuJoCo mesh rendering is the bottleneck in two of them.
set -uo pipefail
cd "$(dirname "$0")/.."

DC="docker compose -f docker/compose.yaml --profile wsl2 run --rm -T dev-wsl"
export DOCKER_UID=1000 DOCKER_GID=1000

# ---- wait ---------------------------------------------------------------
# Poll for the named jobs rather than for GPU utilisation: utilisation dips to
# zero between phases of a run that has not finished, and a dip is not a free
# card.
echo "waiting for: align_simulator, collect_demos, run_g1_hardening"
while pgrep -f "align_simulator|collect_demos|run_g1_hardening|train_predictor" >/dev/null; do
    sleep 60
done
echo "GPU free at $(date +%T)"

# ---- g007 backfill ------------------------------------------------------
# The finest G1 arm failed all five seeds at 14:20 against a bc.py that was
# fixed at 14:32, and the ladder loop had already moved past it. Nothing to
# fix -- just re-run the arm.
echo
echo "### g007 backfill (in 112, 4 stages -> 7x7 grid)"
for s in 0 1 2 3 4; do
    ck="checkpoints/g1/bc_g007_seed${s}.pt"
    [ -f "$ck" ] && { echo "  seed ${s}: already present"; continue; }
    $DC python scripts/train_bc.py --data data/episodes/reach_v2 \
        --out checkpoints/g1 --seed "$s" --epochs 50 --in-size 112 --stages 4 \
        > "logs/g1_train_g007_${s}.log" 2>&1
    if [ -f "checkpoints/g1/bc_seed${s}.pt" ]; then
        mv "checkpoints/g1/bc_seed${s}.pt" "$ck"
        echo "  seed ${s}: ok"
    else
        # Do not let a silent failure look like a finished arm, the way the
        # first pass did.
        echo "  seed ${s}: FAILED -- see logs/g1_train_g007_${s}.log"
        tail -3 "logs/g1_train_g007_${s}.log"
    fi
done
echo "g007: $(ls checkpoints/g1/bc_g007_seed*.pt 2>/dev/null | wc -l)/5 checkpoints"

# ---- E1 exchange rate ---------------------------------------------------
# Needs E2's rungs at a matching dim; E1 refuses to convert otherwise.
echo
echo "### E1 exchange rate"
if [ ! -f cache/e2_rungs.json ]; then
    echo "  cache/e2_rungs.json missing -- running E2 first"
    $DC python scripts/measure_rungs.py --dim 32 > logs/e2_rungs.log 2>&1
fi
$DC python scripts/exchange_rate.py --task push --episodes 6 --frames 64 \
    --dim 32 > logs/e1_exchange.log 2>&1
echo "exit=$?"
grep -avE "UserWarning|self.blocks|it/s\]|Loading weights|torch_dtype" \
    logs/e1_exchange.log | tail -40
