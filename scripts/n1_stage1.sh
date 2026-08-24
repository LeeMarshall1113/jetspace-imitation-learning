#!/usr/bin/env bash
# Re-import the two v3.0 datasets that died mid-write, then TIME a comb-free
# encode so the rest of the pipeline can be sized from a measurement instead of
# a guess. Comb-free encoding is roughly 8x the forward passes of the default,
# and 8x an unknown is still unknown.
set -uo pipefail
cd "$(dirname "$0")/.."

for d in n1_R4_blocks_top n1_V_blocks_wrist; do
    if [ -d "data/episodes/$d" ]; then
        n=$(ls "data/episodes/$d"/episode_*.npz 2>/dev/null | wc -l)
        echo "removing partial data/episodes/$d ($n episodes)"
        rm -rf "data/episodes/$d"
    fi
done

bash scripts/fetch_n1_datasets.sh 2>&1 | grep -avE "it/s\]|B/s\]" | tail -20

echo
echo "=============== timing a comb-free encode ==============="
start=$(date +%s)
python scripts/cache_latents.py --task n1_R2_penmug_s9 \
    --data data/episodes/n1_R2_penmug_s9 --out cache/latents/n1_R2_penmug_s9 \
    --chunk 32 --margin 15 --pool-grid 4 \
    2>&1 | grep -vE "UserWarning|self.blocks|Loading weights" | tail -3
end=$(date +%s)
echo "elapsed: $((end - start))s"

python3 - <<'PY'
import glob
import json
import numpy as np
fs = glob.glob("cache/latents/n1_R2_penmug_s9/episode_*.npy")
if fs:
    n = sum(np.load(f).shape[0] for f in fs)
    print(f"{len(fs)} episodes, {n} latents ({n*2} frames)")
PY

echo
echo "Remaining frames to encode at this rate:"
python3 - <<'PY'
import glob
import json
import os
todo = 0
for p in sorted(glob.glob("data/episodes/n1_*/info.json")):
    name = os.path.basename(os.path.dirname(p))
    if os.path.isdir(f"cache/latents/{name}"):
        continue
    i = json.load(open(p))
    fr = sum(r["length"] for r in i.get("episodes", []))
    todo += fr
    print(f"  {name:26s} {fr:>7} frames")
print(f"  {'TOTAL':26s} {todo:>7} frames  (sim not yet collected)")
PY
