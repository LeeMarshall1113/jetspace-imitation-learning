#!/usr/bin/env bash
# R2 on reach, with a policy that actually works.
#
# The first R2 run printed "REGISTERED PREDICTION FAILS -- latent distance
# predicts world-model internals and NOT behaviour" from a reference success
# rate of 3.3%. Every pose scored 2-8%. There was no dynamic range for a
# correlation to live in, so that verdict measured noise, and the FLOOR = 0.25
# precondition now in run_r2_task_success.sh exists to refuse it.
#
# checkpoints/r2/bc_reach_seed*.pt are now the m2v2 policies (90.7% on the
# reference viewpoint), so the test can finally be run as registered. The old
# 3.3% checkpoints are in checkpoints/r2_old/.
#
# This is the experiment that decides what the paper is about. R1 and H1 show
# latent gap tracks WORLD-MODEL degradation, which is internal. R2 asks whether
# it tracks whether the robot completes the task, which is what anyone outside
# this repository would want from it. Either answer is publishable and they
# point at different papers, so it runs before the framing is chosen, not after.
set -uo pipefail
cd "$(dirname "$0")/.."

DC="docker compose -f docker/compose.yaml --profile wsl2 run --rm -T dev-wsl"
export DOCKER_UID=1000 DOCKER_GID=1000

# Wait for the earlier queue and every current renderer. R2 evaluates 23 poses
# x 30 episodes x 300 steps of MuJoCo, so it is a renderer too and piling it on
# top of four other jobs helps nothing.
echo "waiting for the GPU queue to drain"
while pgrep -f "queue_after_gpu|align_simulator|collect_demos|run_g1_hardening|exchange_rate" >/dev/null; do
    sleep 60
done
echo "starting R2 reach at $(date +%T)"

$DC bash scripts/run_r2_task_success.sh reach "0 1 2" 300 30 > logs/r2_reach_v2.log 2>&1
echo "exit=$?"
grep -avE "UserWarning|self.blocks|it/s\]|Container |EGL|OpenGL" logs/r2_reach_v2.log | tail -45
