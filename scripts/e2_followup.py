#!/usr/bin/env python3
"""E2 follow-up: three checks the ladder table cannot answer on its own.

    python scripts/e2_followup.py

The E2 run produced three things that need testing before any of them is
claimed, because each one, if real, changes a conclusion this project has
already published.

**1. Is session drift above the estimator's noise?** The pooled null is 82.7,
but it pools lab H (34-45) with D_ball (214). Session drift is measured inside
lab H, so the matched control is lab H's own null, not the pool. Comparing
against a pooled floor inflated by an unrelated lab would be the wrong test in
whichever direction it happened to point.

**2. Does within-lab camera change rival cross-lab shift?** The means are 1006
and 1229 and the ranges overlap heavily. N1b claimed "camera placement rivals
laboratory identity", R1 RETRACTED that claim on the basis of a simulated
camera ruler, and E2 now measures both quantities directly in real data. If
they are statistically indistinguishable, the retraction was itself wrong --
which is a thing to establish carefully rather than announce.

**3. How much does estimator asymmetry cost?** `gap_between` fits its
whitening and PCA basis on the first argument, so it is directional. The median
asymmetry is 216, larger than the entire session rung. If the direction of
comparison moves a number more than the effect being measured, the estimator is
not fit for the finer distinctions.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def mannwhitney_u(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """U statistic and a normal-approximation two-sided p, ties averaged.

    Non-parametric on purpose: these are 6 to 28 Frechet values with no reason
    to be normal, and a t-test on n=8 would be assuming what it cannot check.
    """
    n1, n2 = len(a), len(b)
    allv = np.concatenate([a, b])
    order = np.argsort(allv)
    ranks = np.empty(len(allv), dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    for i, c in enumerate(cnt):
        if c > 1:
            ranks[inv == i] = ranks[inv == i].mean()
    r1 = ranks[:n1].sum()
    u1 = r1 - n1 * (n1 + 1) / 2
    u = min(u1, n1 * n2 - u1)
    mu = n1 * n2 / 2
    sd = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (u - mu) / sd if sd > 0 else 0.0
    # two-sided normal approximation
    p = 2 * 0.5 * (1 + math.erf(-abs(z) / np.sqrt(2))) if sd > 0 else 1.0
    return float(u), float(min(1.0, p))


def boot_diff(a: np.ndarray, b: np.ndarray, n: int = 20000, seed: int = 0):
    rng = np.random.default_rng(seed)
    d = [rng.choice(b, len(b)).mean() - rng.choice(a, len(a)).mean()
         for _ in range(n)]
    return np.percentile(d, [2.5, 97.5])


def main() -> int:
    blob = json.loads(Path("cache/e2_rungs.json").read_text())
    r = blob["rungs"]
    v = {k: np.array(r[k]["values"]) for k in r}

    print("=" * 74)
    print("E2 FOLLOW-UP")
    print("=" * 74)

    # ---- 1. session against its MATCHED null ----------------------------
    # The null list order follows measure_rungs.py: the four lab-H sessions
    # first, then sim_push, A_cubes, D_ball.
    null_h = v["null"][:4]
    print("\n1. SESSION DRIFT vs ITS MATCHED CONTROL")
    print(f"   lab H null (4 self-splits)   mean {null_h.mean():7.1f}  "
          f"range [{null_h.min():.1f}, {null_h.max():.1f}]")
    print(f"   lab H session (6 pairs)      mean {v['session'].mean():7.1f}  "
          f"range [{v['session'].min():.1f}, {v['session'].max():.1f}]")
    print(f"   ratio against MATCHED null   {v['session'].mean() / null_h.mean():.1f}x"
          f"   (against the pooled null it was "
          f"{v['session'].mean() / v['null'].mean():.1f}x)")
    u, p = mannwhitney_u(null_h, v["session"])
    lo, hi = boot_diff(null_h, v["session"])
    print(f"   Mann-Whitney U {u:.1f}, p {p:.4f}; "
          f"bootstrap 95% CI on the difference [{lo:.1f}, {hi:.1f}]")
    sep = v["session"].min() > null_h.max()
    print(f"   ranges disjoint: {sep}  "
          f"({v['session'].min():.1f} > {null_h.max():.1f})")

    # ---- 2. camera vs cross-lab -----------------------------------------
    print("\n2. WITHIN-LAB CAMERA vs CROSS-LAB")
    print(f"   camera    (n={len(v['camera']):2d})  mean {v['camera'].mean():7.1f}")
    print(f"   cross_lab (n={len(v['cross_lab']):2d})  mean {v['cross_lab'].mean():7.1f}")
    u, p = mannwhitney_u(v["camera"], v["cross_lab"])
    lo, hi = boot_diff(v["camera"], v["cross_lab"])
    print(f"   difference {v['cross_lab'].mean() - v['camera'].mean():+.1f}   "
          f"bootstrap 95% CI [{lo:+.1f}, {hi:+.1f}]")
    print(f"   Mann-Whitney U {u:.1f}, p {p:.4f}")
    ov = (min(v["camera"].max(), v["cross_lab"].max())
          - max(v["camera"].min(), v["cross_lab"].min()))
    span = max(v["camera"].max(), v["cross_lab"].max()) - \
        min(v["camera"].min(), v["cross_lab"].min())
    print(f"   range overlap {ov:.0f} of {span:.0f} total span "
          f"({100 * ov / span:.0f}%)")
    if p > 0.05:
        print("   NOT separable at p<0.05: moving the camera within one lab")
        print("   shifts latents about as far as changing laboratory entirely.")
        print("   That is N1b's retracted headline, measured directly.")
    else:
        print("   Separable: cross-lab is reliably larger than camera change.")

    # ---- 3. sim camera vs real camera: the R1 ruler's scale -------------
    print("\n3. SIM CAMERA vs REAL CAMERA (the R1 ruler's transferability)")
    print(f"   sim camera  (n={len(v['sim_camera']):2d})  mean {v['sim_camera'].mean():7.1f}")
    print(f"   real camera (n={len(v['camera']):2d})  mean {v['camera'].mean():7.1f}")
    ratio = v["camera"].mean() / v["sim_camera"].mean()
    u, p = mannwhitney_u(v["sim_camera"], v["camera"])
    lo, hi = boot_diff(v["sim_camera"], v["camera"])
    print(f"   real / sim  {ratio:.2f}x   Mann-Whitney p {p:.4f}   "
          f"CI on difference [{lo:+.1f}, {hi:+.1f}]")
    if p < 0.05:
        print("   A camera move in simulation does NOT produce the same latent")
        print("   shift as a camera move in reality. R1 built its degrees->Frechet")
        print("   ruler in simulation and then read REAL rungs off it. That")
        print("   conversion is confounded by this scale difference, and every")
        print("   'equals N degrees' figure derived from it needs restating.")

    # ---- 4. what the asymmetry costs ------------------------------------
    print("\n4. ESTIMATOR ASYMMETRY")
    am = blob["asymmetry_median"]
    print(f"   median |A->B minus B->A|  {am:.1f}")
    for k in ["session", "camera", "cross_lab"]:
        print(f"   vs {k:10s} mean {v[k].mean():7.1f}   "
              f"asymmetry is {100 * am / v[k].mean():.0f}% of it")
    print("   Symmetrising (as E2 does) removes the direction dependence but")
    print("   not the uncertainty it implies. Distinctions smaller than ~216")
    print("   Frechet should not be claimed from this estimator.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
