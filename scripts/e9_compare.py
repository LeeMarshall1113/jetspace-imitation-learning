#!/usr/bin/env python3
"""E9 encoder comparison, paired fold by fold.

    python scripts/e9_compare.py

The per-K standard deviations in the E9 summaries are dominated by how much the
eight tasks differ from each other, not by how much the encoders differ. B_svla
sits at 0.12 and F_cup at 1.69, so an unpaired comparison drowns any encoder
effect in task heterogeneity.

Both arms ran the identical folds -- same tasks, same seeds, same episode
splits -- so the comparison should be paired. Each fold contributes one
difference, and the question is whether those differences are consistently
signed rather than whether two wide distributions overlap.

Reports three things:

  1. Does pretraining on other tasks help?      (transfer vs scratch, per arm)
  2. Which encoder learns a new task faster?    (scratch, V-JEPA vs random)
  3. Which encoder is less damaged by transfer? (gain, V-JEPA vs random)

Only (2) can be positive for the project's thesis on this data, and it is the
weaker "few-shot within one task" claim rather than the transfer claim.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def load(p: str) -> dict:
    rows = json.loads(Path(p).read_text())["rows"]
    return {(r["target"], r["K"], r["seed"]): r for r in rows}


def wilcoxon_sign(d: np.ndarray) -> tuple[int, int, float]:
    """Sign test: how many differences are negative, and a two-sided binomial p.

    A sign test rather than a t-test because these are MSE differences across
    heterogeneous tasks, with no reason to be symmetric or normal.
    """
    d = d[d != 0]
    n = len(d)
    neg = int((d < 0).sum())
    k = min(neg, n - neg)
    # two-sided binomial tail at p=0.5
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return neg, n, float(min(1.0, 2 * tail))


def main() -> int:
    v = load("cache/e9_n1b.json")
    c = load("cache/e9_n1bcnn.json")
    keys = sorted(set(v) & set(c))
    print("=" * 72)
    print(f"E9 paired encoder comparison -- {len(keys)} matched folds")
    print("=" * 72)

    Ks = sorted({k[1] for k in keys})

    print("\n1. DOES PRETRAINING ON OTHER TASKS HELP?  (transfer - scratch)")
    print("   negative = transfer is better")
    for arm, d in (("vjepa", v), ("rand", c)):
        for K in Ks:
            sel = [k for k in keys if k[1] == K]
            diff = np.array([d[k]["transfer"] - d[k]["scratch"] for k in sel])
            neg, n, p = wilcoxon_sign(diff)
            print(f"   {arm:6s} K={K}  mean {diff.mean():+.3f}   "
                  f"transfer better in {neg}/{n} folds   p {p:.2e}")

    print("\n2. WHICH ENCODER LEARNS A NEW TASK FASTER?  (scratch, vjepa - rand)")
    print("   negative = V-JEPA is better")
    for K in Ks:
        sel = [k for k in keys if k[1] == K]
        diff = np.array([v[k]["scratch"] - c[k]["scratch"] for k in sel])
        neg, n, p = wilcoxon_sign(diff)
        print(f"   K={K}  mean {diff.mean():+.3f}   "
              f"V-JEPA better in {neg}/{n} folds   p {p:.2e}")

    print("\n3. WHICH ENCODER IS LESS DAMAGED BY TRANSFER?")
    print("   gain = scratch - transfer; less negative is less damage")
    for K in Ks:
        sel = [k for k in keys if k[1] == K]
        gv = np.array([v[k]["scratch"] - v[k]["transfer"] for k in sel])
        gc = np.array([c[k]["scratch"] - c[k]["transfer"] for k in sel])
        diff = gv - gc
        neg, n, p = wilcoxon_sign(-diff)   # negative diff = vjepa damaged more
        print(f"   K={K}  vjepa {gv.mean():+.3f}  rand {gc.mean():+.3f}  "
              f"V-JEPA less damaged in {neg}/{n} folds   p {p:.2e}")

    # Restrict to folds where the scratch baseline actually works. Above 1.0 a
    # model is worse than predicting the mean action, so a comparison there is
    # between two non-functional models.
    print("\n4. RESTRICTED TO LEARNABLE FOLDS (both scratch arms < 1.0)")
    ok = [k for k in keys if v[k]["scratch"] < 1.0 and c[k]["scratch"] < 1.0]
    print(f"   {len(ok)}/{len(keys)} folds qualify")
    if ok:
        dv = np.array([v[k]["transfer"] - v[k]["scratch"] for k in ok])
        dc = np.array([c[k]["transfer"] - c[k]["scratch"] for k in ok])
        ds = np.array([v[k]["scratch"] - c[k]["scratch"] for k in ok])
        for name, d in (("vjepa transfer-scratch", dv),
                        ("rand  transfer-scratch", dc),
                        ("scratch vjepa-rand", ds)):
            neg, n, p = wilcoxon_sign(d)
            print(f"   {name:24s} mean {d.mean():+.3f}  "
                  f"negative in {neg}/{n}  p {p:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
