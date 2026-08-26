#!/usr/bin/env bash
# Measure and remove the window-phase comb on every cached latent set.
set -uo pipefail
cd "$(dirname "$0")/.."
for t in push pickplace real_cubes reach; do
    [ -d "cache/latents/$t" ] || { echo "### $t: no cache"; continue; }
    echo "### $t"
    python scripts/decomb_latents.py --latents "cache/latents/$t" || true
    echo
done
