#!/usr/bin/env python3
"""How much of E12's rho is signal, and how much is fifteen encoders?

    python scripts/e12_uncertainty.py

Every rho E12 reports is a point estimate over ~15 encoders. A Spearman
correlation on fifteen points is a noisy statistic, and the headline claim --
that probe accuracy predicts robustness on pickplace (0.536) but much less on
push (0.273) -- is a comparison between two such estimates. Without an interval
there is no way to tell that claim from sampling noise, and it is the first
thing a reviewer will press on.

Two things are computed here:

  per-axis CI     bootstrap over ENCODERS (the sampling unit), resampling the
                  (probe, robustness) pairs with replacement

  task difference paired bootstrap on the same encoder set, giving a CI on
                  rho_pickplace - rho_push per axis. Paired because both tasks
                  are measured on the identical fifteen encoders, so the
                  difference is far better determined than the two marginals.

Reads the cached result JSONs, so it is cheap and needs no GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BOOT = 20000
RNG = np.random.default_rng(0)
AXES = ["lighting", "texture", "clutter", "noise", "defocus", "compress",
        "exposure", "lowres"]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 1e-12 else float("nan")


def load(task: str) -> dict:
    p = Path("cache") / f"e12_{task}.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def boot_ci(probe: np.ndarray, held: np.ndarray, n: int = BOOT):
    """Resample encoders with replacement. A degenerate draw (every encoder the
    same) yields nan and is dropped rather than counted as zero correlation."""
    k = len(probe)
    out = np.empty(n)
    for i in range(n):
        idx = RNG.integers(0, k, k)
        out[i] = spearman(probe[idx], held[idx])
    out = out[np.isfinite(out)]
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), out


def main() -> int:
    tasks = {t: load(t) for t in ("push", "pickplace")}
    tasks = {t: d for t, d in tasks.items() if d}
    if not tasks:
        print("no cached results; run scripts/e12_analyze.py first")
        return 1

    print("=" * 72)
    print("per-axis rho with 95% bootstrap CI over encoders")
    print("=" * 72)
    print(f"  {'task/axis':24s} {'n':>3s} {'rho':>7s} {'95% CI':>18s}  "
          f"excludes 0?")
    marg = {}
    for task, d in tasks.items():
        for ax in AXES:
            cell = d.get(ax)
            if not cell or not cell.get("rows"):
                continue
            rows = cell["rows"]
            names = sorted(rows)
            probe = np.array([rows[n]["probe"] for n in names])
            held = np.array([rows[n]["held"] for n in names])
            rho = spearman(probe, held)
            lo, hi, _ = boot_ci(probe, held)
            marg[(task, ax)] = (names, probe, held, rho, lo, hi)
            excl = "yes" if (lo > 0 or hi < 0) else "NO"
            note = "" if cell.get("rho") is not None else "   (excluded axis)"
            print(f"  {task + '/' + ax:24s} {len(names):3d} {rho:+7.3f} "
                  f"[{lo:+.3f}, {hi:+.3f}]  {excl}{note}")

    print()
    print("=" * 72)
    print("task difference, paired bootstrap on the shared encoder set")
    print("=" * 72)
    print("  rho_pickplace - rho_push, resampling encoders jointly")
    print()
    for ax in AXES:
        a = marg.get(("push", ax))
        b = marg.get(("pickplace", ax))
        if not a or not b:
            continue
        shared = sorted(set(a[0]) & set(b[0]))
        if len(shared) < 5:
            continue
        ia = [a[0].index(n) for n in shared]
        ib = [b[0].index(n) for n in shared]
        pa, ha = a[1][ia], a[2][ia]
        pb, hb = b[1][ib], b[2][ib]
        obs = spearman(pb, hb) - spearman(pa, ha)
        k = len(shared)
        diffs = np.empty(BOOT)
        for i in range(BOOT):
            idx = RNG.integers(0, k, k)
            diffs[i] = spearman(pb[idx], hb[idx]) - spearman(pa[idx], ha[idx])
        diffs = diffs[np.isfinite(diffs)]
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        # Two-sided bootstrap p: how often the resampled difference lands on
        # the other side of zero from the observed one.
        p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
        verdict = "DISTINGUISHABLE" if (lo > 0 or hi < 0) else "not distinguishable"
        print(f"  {ax:10s} n={k:2d}  diff {obs:+.3f}  "
              f"95% CI [{lo:+.3f}, {hi:+.3f}]  p={p:.3f}  {verdict}")

    print()
    print("Read this before quoting any single rho: with fifteen encoders the")
    print("interval on one axis is wide. The paired difference is the sharper")
    print("statistic, because the same encoders carry both tasks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
