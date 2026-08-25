#!/usr/bin/env bash
# Does the M2 policy still reproduce 85.7%, and on which dataset?
#
# R2 retrained BC on data/episodes/reach and got 3.3% at the reference pose.
# The M2 result was measured on data/episodes/so101_reach with a different
# checkpoint. Before reporting any R2 correlation, establish which policy and
# dataset actually reproduce the published number -- a correlation measured
# against a 3% baseline has no dynamic range and means nothing.
cd "$(dirname "$0")/.."

echo "=== datasets present ==="
for d in data/episodes/*/; do
    n=$(ls "$d"/episode_*.npz 2>/dev/null | wc -l)
    [ "$n" -gt 0 ] && printf "  %-34s %4s episodes\n" "$(basename "$d")" "$n"
done

echo
echo "=== M2 checkpoints ==="
ls -1 checkpoints/bc_seed*.pt 2>/dev/null || echo "  none"

echo
echo "=== M2 policy on so101_reach (the published 85.7% setting) ==="
python scripts/eval_policy.py --task reach \
    --checkpoints "checkpoints/bc_seed*.pt" \
    --train-data data/episodes/so101_reach \
    --max-steps 150 --eval-limit 30 2>&1 \
    | grep -aE "leak check|success"

echo
echo "=== R2's freshly-trained policy on data/episodes/reach ==="
python scripts/eval_policy.py --task reach \
    --checkpoints "checkpoints/r2/bc_reach_seed*.pt" \
    --train-data data/episodes/reach \
    --max-steps 150 --eval-limit 30 2>&1 \
    | grep -aE "leak check|success"
