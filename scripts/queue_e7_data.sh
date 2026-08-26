#!/usr/bin/env bash
# Expand the R1 push sweep so E7's registered test has enough samples.
#
# The current push sweep is 5 episodes -> 398 latents at the reference pose.
# Against a 128-dimensional PCA input that is thin, and prereg-e7.md S6 commits
# the registered test to >= 1200. Push yields ~80 latents per episode, so 16
# episodes clears it with margin.
#
# Reach is NOT expanded here. Its episodes are ~11 latents long, so reaching
# 1200 would need ~110 episodes at 23 renders per timestep -- far more render
# time than push for the same statistical power. E7 runs on push.
set -uo pipefail
cd "$(dirname "$0")/.."

DC="docker compose -f docker/compose.yaml --profile wsl2 run --rm -T dev-wsl"
export DOCKER_UID=1000 DOCKER_GID=1000

echo "waiting for renderers and the R2 queue to finish"
while pgrep -f "collect_demos|align_simulator|run_g1_hardening|exchange_rate|queue_after_gpu|queue_r2_reach|eval_policy" >/dev/null; do
    sleep 60
done
echo "starting expanded R1 push collection at $(date +%T)"

# run_r1.sh collects, encodes and rebuilds the ruler. Existing episodes are
# kept; this adds up to 16.
$DC bash scripts/run_r1.sh push 16 all > logs/r1_push16.log 2>&1
echo "collection exit=$?"
grep -avE "UserWarning|self.blocks|it/s\]|Container " logs/r1_push16.log | tail -12

# The random-CNN arm has to be regenerated against the expanded episode set --
# the cached features cover only the first 5 episodes, and E7 asserts matched
# latent counts between arms, so a stale cache would fire invalidation 2 rather
# than silently comparing different amounts of data.
echo
echo "clearing stale random-CNN cache before the registered run"
rm -rf cache/latents/r1cnn_push__*

$DC bash scripts/run_e7.sh push "0 1 2" > logs/e7_registered.log 2>&1
echo "e7 exit=$?"
grep -avE "UserWarning|self.blocks|it/s\]|Container " logs/e7_registered.log | tail -40
