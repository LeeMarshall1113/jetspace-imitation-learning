#!/usr/bin/env bash
# Where is the N1 pipeline up to?
#
# Exists as a file rather than an inline one-liner because shell loop variables
# do not survive being passed through Git Bash into `wsl -- bash -lc`; `$d`
# arrives empty and every path silently becomes the same wrong thing. A script
# on disk sidesteps the quoting entirely.
cd "$(dirname "$0")/.."

python3 - <<'PY'
import glob
import json
import os

WANT = [
    "n1_R1_cubes", "n1_R2_penmug_s9", "n1_R3_penmug_s12",
    "n1_R4_blocks_top", "n1_V_blocks_wrist",
    "n1_sim_push_pretty", "n1_sim_pickplace_pretty", "n1_sim_push_pretty_dr",
]

print(f"{'dataset':28s} {'episodes':>9} {'latents':>9}  status")
print("-" * 68)
ready = 0
for name in WANT:
    eps = len(glob.glob(f"data/episodes/{name}/episode_*.npz"))
    lat_files = glob.glob(f"cache/latents/{name}/episode_*.npy")
    has_info = os.path.exists(f"cache/latents/{name}/info.json")
    lat = len(lat_files)
    if eps and lat and has_info:
        status = "ready"
        ready += 1
    elif eps:
        status = "collected, needs encoding"
    else:
        status = "MISSING"
    print(f"{name:28s} {eps:>9} {lat:>9}  {status}")

print(f"\n{ready}/{len(WANT)} datasets ready to measure")

rungs = sorted(glob.glob("cache/n1_*.json"))
if rungs:
    print(f"\nrungs measured: {[json.load(open(f))['label'] for f in rungs]}")
else:
    print("\nno rungs measured yet")
PY
