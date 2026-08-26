#!/usr/bin/env bash
# E9 both encoder arms. Logs straight to files -- piping through tail is what
# hid an hour of progress on the first attempt.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

DC="docker compose -f docker/compose.yaml --profile wsl2 run --rm -T dev-wsl"
export DOCKER_UID=1000 DOCKER_GID=1000

for pfx in n1b n1bcnn; do
    echo "### E9 ${pfx} starting $(date +%T)"
    $DC python -u scripts/e9_task_transfer.py --prefix "$pfx" \
        --shots 1 2 4 --seeds 0 1 2 > "logs/e9_${pfx}.log" 2>&1
    echo "### E9 ${pfx} exit=$? $(date +%T)"
    grep -avE "UserWarning|self.blocks|it/s\]|Container " "logs/e9_${pfx}.log" | tail -20
done
