#!/usr/bin/env bash
# E6 across seeds, then aggregate. Four arms x three seeds.
#
# The single-seed run said a random 7.5M CNN beats frozen V-JEPA on every
# representation metric. That is the most striking claim in the project and it
# rested on one run, so it gets replicated before it goes anywhere near a draft.
#
# What a seed changes differs by arm, and it matters for reading the spreads:
#
#   V-JEPA      predictor init only. The encoder is frozen, so its across-seed
#               variance is a floor -- the irreducible noise of the downstream
#               pipeline, not of the representation.
#   random CNN  the encoder weights THEMSELVES, plus the predictor. This is the
#               widest legitimate spread and the one that says whether "random
#               features work" is about architecture or about luck.
#   joint CNN   encoder init, predictor init, and the whole optimisation path.
set -uo pipefail
cd "$(dirname "$0")/.."

TASK=${1:-push}
EPS=${2:-30}
SEEDS=${3:-"0 1 2"}
HMAX=${4:-96}
DATA=${5:-"data/episodes/${TASK}"}

for s in $SEEDS; do
    echo
    echo "###################### SEED $s ######################"
    bash scripts/run_e6.sh "$TASK" "$EPS" "$s" "$HMAX" "$DATA" 2>&1 \
        | grep -vE "UserWarning|self.blocks|it/s\]"
done

echo
echo "=============================================================="
echo "  E6 ACROSS SEEDS"
echo "=============================================================="
python3 - <<'PY'
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

# arm -> metric -> [values across seeds]
arms = defaultdict(lambda: defaultdict(list))

for f in glob.glob("cache/e6_h_*.json"):
    name = os.path.basename(f)[len("e6_h_"):-len(".json")]
    m = re.match(r"(.+)_s(\d+)$", name)
    if not m:
        continue
    arm = m.group(1)
    d = json.load(open(f))
    arms[arm]["useful"].append(d["useful_horizon"])
    arms[arm]["aware"].append(d["action_aware_horizon"])
    c = f"cache/conservatism_{name}.json"
    if os.path.exists(c):
        cd = json.load(open(c))
        arms[arm]["ratio"].append(cd["mean_ratio"])
        arms[arm]["cosine"].append(cd["mean_cosine"])
    pr = f"cache/probe_action_{name}.json"
    if os.path.exists(pr):
        # The probe stores {"results": {interval: {"inverse_r2": ...}}} and
        # prints the best; it never writes a top-level scalar, so the max has
        # to be recomputed here rather than read off.
        res = json.load(open(pr)).get("results", {})
        r2s = [v["inverse_r2"] for v in res.values() if "inverse_r2" in v]
        if r2s:
            arms[arm]["probe_r2"].append(max(r2s))

if not arms:
    print("no seeded arms found")
else:
    def fmt(v):
        if not v:
            return "        -"
        mu = float(np.mean(v))
        sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        return f"{mu:6.3f}+-{sd:<5.3f}"

    print(f"{'arm':24s} {'n':>2} {'cosine':>13} {'ratio':>13} "
          f"{'probe R^2':>13} {'useful h':>9}")
    print("-" * 80)
    for arm in sorted(arms):
        a = arms[arm]
        n = len(a["useful"])
        uh = f"{np.mean(a['useful']):.0f}" if a["useful"] else "-"
        print(f"{arm:24s} {n:>2} {fmt(a['cosine']):>13} {fmt(a['ratio']):>13} "
              f"{fmt(a['probe_r2']):>13} {uh:>9}")

    print()
    print("READ: the decisive pair is vjepa vs rand -- both frozen, neither can")
    print("collapse. If their intervals overlap, 22M videos of pretraining bought")
    print("nothing measurable here and the honest claim is a TIE, not a win for")
    print("either. Overlapping intervals are a result; picking the higher mean")
    print("out of two overlapping distributions is not.")
PY
