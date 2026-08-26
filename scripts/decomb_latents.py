#!/usr/bin/env python3
"""Remove the encoder's window-phase signature from cached latents.

    python scripts/decomb_latents.py --latents cache/latents/push --period 8

`check_chunk_phase.py` established that overlapped-window encoding stamps a
comb into the latent sequence: latents `stride/TUBELET` apart occupy the same
offset inside their encoding window, share V-JEPA's temporal position
embedding, and sit artificially close together. Measured on push, latents 8
apart are 1.40x closer than their neighbours -- and the do-nothing baseline in
E3 collapses from ~8.0 to ~2.9 at exactly those lags.

**Why subtracting a per-phase mean is the right shape of fix.** The position
embedding enters additively and depends only on the offset within the window,
so its contribution to every latent at phase p is the same vector. Estimating
that vector as the mean of all phase-p latents and removing it takes out the
artifact while leaving anything that varies within a phase untouched.

    z'[t] = z[t] - (mean{z[s] : s = t mod P} - mean{z})

The global mean is added back so the latents keep their location; only the
*differences between phases* are removed.

**What this cannot fix, measured rather than guessed.** Each phase holds only
about n/period latents -- roughly 12 for a 100-latent episode at period 8 -- so
the phase mean is a noisy estimate that partly fits real content and removes it
along with the artifact. On our caches this overshoots: push lands at 0.94 and
reach at 0.85, i.e. on-phase lags end up *farther* apart than off-phase ones.
The overshoot is much smaller than the original comb (reach: 0.44 off flat
becomes 0.16) but it is not nothing, and it is a bias, not noise.

The script therefore refuses to recommend itself where there is no comb to
remove. Real-robot latents measure 1.014 -- essentially flat already -- and
subtracting phase means there makes them *worse*, so the original cache stands.

The principled fix is a stride of one tubelet, which removes the phase structure
by construction at roughly 8x the encoding cost. This subtraction is a
stopgap that makes existing caches usable, and it should be named as one.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np


def comb_ratio(z: np.ndarray, period: int, max_h: int = 32) -> float:
    """How much closer on-phase lags sit than off-phase ones. 1.0 = no comb."""
    f = z.reshape(z.shape[0], -1).astype(np.float64)
    f = (f - f.mean(0)) / (f.std(0) + 1e-6)
    max_h = min(max_h, len(f) - 2)
    if max_h < period * 2:
        return float("nan")
    d = np.array([np.linalg.norm(f[h:] - f[:-h], axis=1).mean() for h in range(1, max_h + 1)])
    on = np.array([d[h - 1] for h in range(1, max_h + 1) if h % period == 0])
    off = np.array([d[h - 1] for h in range(1, max_h + 1) if h % period != 0])
    return float(off.mean() / max(on.mean(), 1e-9))


def decomb(z: np.ndarray, period: int) -> np.ndarray:
    """Subtract the per-phase mean offset, keeping the global mean."""
    out = z.astype(np.float32).copy()
    flat = out.reshape(out.shape[0], -1)
    g = flat.mean(0, keepdims=True)
    for p in range(period):
        idx = np.arange(p, len(flat), period)
        if len(idx) < 2:
            continue
        flat[idx] -= flat[idx].mean(0, keepdims=True) - g
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True)
    ap.add_argument("--period", type=int, default=None,
                    help="defaults to stride/TUBELET from the cache's info.json")
    ap.add_argument("--out", default=None, help="defaults to <latents>_decombed")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = Path(args.latents)
    info = json.loads((src / "info.json").read_text())

    period = args.period
    if period is None:
        chunk, margin = info.get("chunk", 32), info.get("margin", 8)
        fpl = info.get("frames_per_latent", 2)
        period = max(fpl, chunk - 2 * margin) // fpl
        print(f"period inferred from chunk={chunk} margin={margin} "
              f"frames_per_latent={fpl}  ->  {period}")

    files = sorted(src.glob("episode_*.npy"))
    if not files:
        print(f"no latents in {src}")
        return 1

    before, after = [], []
    dst = Path(args.out or f"{src}_decombed")
    if not args.dry_run:
        dst.mkdir(parents=True, exist_ok=True)

    for f in files:
        z = np.load(f)
        b = comb_ratio(z, period)
        zd = decomb(z, period)
        a = comb_ratio(zd, period)
        if not np.isnan(b):
            before.append(b)
            after.append(a)
        if not args.dry_run:
            np.save(dst / f.name, zd.astype(z.dtype))

    if not before:
        print("every episode too short to measure the comb")
        return 1

    b, a = float(np.mean(before)), float(np.mean(after))
    # Distance from 1.0 in EITHER direction is the thing to minimise. Reporting
    # "% of the excess removed" flattered the fix: subtracting past 1.0 scores
    # over 100% while actively making the latents worse.
    db, da = abs(b - 1.0), abs(a - 1.0)
    print(f"\n{src}: {len(files)} episodes, period {period}")
    print(f"  comb ratio before  {b:.3f}x   (|1-r| = {db:.3f})")
    print(f"  comb ratio after   {a:.3f}x   (|1-r| = {da:.3f})")

    if db < 0.05:
        print("\n  NO COMB TO REMOVE. These latents were already flat, so the")
        print("  subtraction can only delete real signal. Do NOT use the decombed")
        print("  cache for this set -- keep the original.")
        verdict = "skip"
    elif da > db:
        print("\n  THE FIX MADE IT WORSE. Keep the original cache.")
        verdict = "worse"
    elif a < 0.95:
        print("\n  OVERCORRECTED past flat. Each phase holds only ~n/period")
        print("  samples, so the phase mean partly fits real content and takes it")
        print("  with it. Usable, but a smaller encode stride is the honest fix.")
        verdict = "overcorrected"
    elif da > 0.10:
        print("\n  RESIDUE REMAINS. The artifact is not purely additive.")
        print("  Re-encode with a smaller stride before trusting lag-sensitive numbers.")
        verdict = "residue"
    else:
        print("\n  Comb removed. Re-run E2/E3 against the decombed cache and")
        print("  confirm the conclusions are unchanged before quoting either.")
        verdict = "ok"

    if verdict in ("skip", "worse") and not args.dry_run:
        print("  (writing anyway for inspection; info.json records the verdict)")

    if not args.dry_run:
        shutil.copy(src / "info.json", dst / "info.json")
        d = json.loads((dst / "info.json").read_text())
        d.update({"decombed": True, "decomb_period": period,
                  "comb_ratio_before": b, "comb_ratio_after": a,
                  "verdict": verdict, "use_this_cache": verdict not in ("skip", "worse"),
                  "source_cache": str(src)})
        (dst / "info.json").write_text(json.dumps(d, indent=2))
        print(f"\nwrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
