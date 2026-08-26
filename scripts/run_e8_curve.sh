#!/usr/bin/env bash
# E8 data-efficiency curve: how few simulated viewpoints buy generalization to
# viewpoints never seen, and does the answer depend on the encoder?
#
# This is the "small dataset, broad transfer" claim stated as an experiment.
# Held-out poses are fixed across every row, so the only thing changing is how
# many training viewpoints the head saw.
set -uo pipefail
cd "$(dirname "$0")/.."

DC="docker compose -f docker/compose.yaml --profile wsl2 run --rm -T dev-wsl"
export DOCKER_UID=1000 DOCKER_GID=1000

echo "n_views encoder    baseline        multiview"
for n in 2 4 7 10 14; do
    for pfx in r1_push r1cnn_push; do
        out="cache/e8_n${n}_${pfx}.json"
        log="logs/e8_n${n}_${pfx}.log"
        $DC python scripts/e8_canonicalize.py --task push --seeds 0 1 2 \
            --prefix "$pfx" --n-train-poses "$n" --out "$out" > "$log" 2>&1
        b=$(grep -aE "^  baseline " "$log"  | awk '{print $2, $3, $4}')
        m=$(grep -aE "^  multiview " "$log" | awk '{print $2, $3, $4}')
        printf "%-7s %-10s %-15s %s\n" "$n" "${pfx#r1}" "${b:-FAILED}" "${m:-FAILED}"
    done
done
