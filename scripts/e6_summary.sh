#!/usr/bin/env bash
# Aggregate every completed E6 arm across seeds, whatever has finished so far.
cd "$(dirname "$0")/.."
python3 - <<'PY'
import glob, json, os, re
from collections import defaultdict
import numpy as np

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
        res = json.load(open(pr)).get("results", {})
        r2 = [v["inverse_r2"] for v in res.values() if "inverse_r2" in v]
        if r2:
            arms[arm]["probe_r2"].append(max(r2))

def fmt(v):
    if not v:
        return "        -"
    mu = float(np.mean(v))
    sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    return f"{mu:6.3f}+-{sd:<5.3f}"

print(f"{'arm':26s} {'n':>2} {'cosine':>13} {'ratio':>13} {'probe R^2':>13} {'h':>5}")
print("-" * 78)
for arm in sorted(arms):
    a = arms[arm]
    h = f"{np.mean(a['useful']):.0f}" if a["useful"] else "-"
    print(f"{arm:26s} {len(a['useful']):>2} {fmt(a['cosine']):>13} "
          f"{fmt(a['ratio']):>13} {fmt(a['probe_r2']):>13} {h:>5}")
PY
