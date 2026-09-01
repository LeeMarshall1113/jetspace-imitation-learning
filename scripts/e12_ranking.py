#!/usr/bin/env python3
"""Which frozen encoder is actually most robust, and is the ranking real?

    python scripts/e12_ranking.py

Ranks encoders by mean relative degradation over the axis-task cells that
SURVIVE the invalidation conditions in docs/prereg-e12-stage3.md S4. Cells where
the untrained control fails to land in the worst third are excluded: a ranking
computed there is noise, and averaging it in would launder that noise into the
headline table. Nine of sixteen cells go.

Uncertainty is a PAIRED bootstrap over cells. Every encoder is measured on the
identical set of cells, so resampling them jointly keeps the pairing and makes
"is A better than B" far better determined than either encoder's own interval
suggests -- the same reason the task-difference test in e12_uncertainty.py is
paired.

Two honest limits, stated because the numbers look tidier than they are:

  * Seven cells is a small resampling universe. The interval on any single
    encoder's mean is wide, and the CI on a MEAN is not a CI on its rank.
  * This resamples cells, not episodes. Within-cell measurement noise (each
    cell rests on two held-out episodes) is not represented, so these are
    lower bounds on the true uncertainty.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BOOT = 20000
RNG = np.random.default_rng(0)


def valid_cells() -> dict:
    """(task, axis) -> {encoder: relative degradation}, survivors only."""
    out = {}
    for task in ("push", "pickplace"):
        p = Path("cache") / f"e12_{task}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        for axis, cell in d.items():
            if not isinstance(cell, dict) or "rows" not in cell:
                continue
            if cell.get("rho") is None:          # excluded by I1-I4
                continue
            out[(task, axis)] = {
                e: c["held"] / max(c["ref_mse"], 1e-9)
                for e, c in cell["rows"].items()
            }
    return out


def main() -> int:
    cells = valid_cells()
    if not cells:
        print("no valid cells; run scripts/e12_analyze.py first")
        return 1
    keys = sorted(cells)
    encs = sorted(set.intersection(*(set(cells[k]) for k in keys)))
    M = np.array([[cells[k][e] for k in keys] for e in encs])   # (enc, cell)

    print(f"{len(encs)} encoders over {len(keys)} valid cells: "
          + ", ".join(f"{t}/{a}" for t, a in keys))
    print("relative degradation; 1.00 = no loss under shift, lower is better\n")

    means = M.mean(1)
    # Paired: one cell resample, applied to every encoder at once.
    idx = RNG.integers(0, len(keys), (BOOT, len(keys)))
    boot = np.stack([M[:, i].mean(1) for i in idx])             # (BOOT, enc)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    ranks = boot.argsort(1).argsort(1)                          # 0 = best
    p_top3 = (ranks < 3).mean(0)
    p_bot3 = (ranks >= len(encs) - 3).mean(0)

    order = np.argsort(means)
    print(f"  {'':2s} {'encoder':16s} {'mean':>7s} {'95% CI':>17s} "
          f"{'P(top3)':>8s} {'P(bot3)':>8s}")
    for r, i in enumerate(order, 1):
        print(f"  {r:2d} {encs[i]:16s} {means[i]:7.3f} "
              f"[{lo[i]:6.3f},{hi[i]:7.3f}] {p_top3[i]:8.2f} {p_bot3[i]:8.2f}")

    print("\npairwise, paired bootstrap: P(row degrades LESS than column)")
    named = [e for e in ("aimv2", "vjepa2", "vc1", "vc1-large", "random")
             if e in encs]
    ix = {e: encs.index(e) for e in named}
    print(f"  {'':16s}" + "".join(f"{e:>12s}" for e in named))
    for a in named:
        row = f"  {a:16s}"
        for b in named:
            if a == b:
                row += f"{'--':>12s}"
            else:
                row += f"{(boot[:, ix[a]] < boot[:, ix[b]]).mean():12.3f}"
        print(row)

    print("\nheadline comparisons")
    for a, b in (("aimv2", "vc1"), ("aimv2", "vc1-large"),
                 ("vjepa2", "vc1"), ("vc1", "random"), ("aimv2", "vjepa2")):
        if a not in ix or b not in ix:
            continue
        d = boot[:, ix[a]] - boot[:, ix[b]]
        lo, hi = np.percentile(d, [2.5, 97.5])
        sure = "DISTINGUISHABLE" if hi < 0 or lo > 0 else "not distinguishable"
        print(f"  {a:11s} vs {b:11s} diff {d.mean():+8.3f} "
              f"[{lo:+.3f}, {hi:+.3f}]  {sure}")

    print("\nA CI on a mean is not a CI on a rank, and seven cells is a small")
    print("universe to resample. Quote P(top3)/P(bot3) for ranking claims and")
    print("the pairwise columns for head-to-head ones.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
