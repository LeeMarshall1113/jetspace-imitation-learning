#!/usr/bin/env bash
# Block until both E9 arms finish, then print both summaries.
#
# Written as a file rather than passed inline to `wsl -- bash -lc '...'`:
# the outer shell expands $variables and $(...) before WSL ever sees the
# string, so inline loops silently lose their variables and bake their
# timestamps at launch. Three waiters died that way before this one.
#
# Polls pgrep, not `docker ps`, whose Command column is truncated to about
# twenty characters -- long enough to hide the script name being matched.
set -uo pipefail
cd "$(dirname "$0")/.."

while pgrep -f e9_task_transfer >/dev/null; do
    sleep 60
done

echo "=== E9 finished at $(date +%T) ==="
for arm in n1b n1bcnn; do
    echo
    echo "######## ${arm} ########"
    if [ -f "logs/e9_${arm}.log" ]; then
        grep -avE "UserWarning|self.blocks|it/s\]|Container " "logs/e9_${arm}.log" \
            | tail -24
    else
        echo "  (no log -- arm did not run)"
    fi
done
