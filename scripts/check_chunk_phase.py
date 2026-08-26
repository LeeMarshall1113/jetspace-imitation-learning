#!/usr/bin/env python3
"""Does the encoder's chunking leak a periodic signature into the latents?

    python scripts/check_chunk_phase.py --task push

E3's do-nothing baseline -- plain ||z_t+h - z_t||, no model involved -- turned
out to be strongly periodic: latents exactly 8 apart sit 2.8x closer together
than latents 7 apart. Real robot motion cannot do that. Something in the
encoding is putting a comb into the representation.

**The suspect.** `VJEPAEncoder.encode` walks the clip in overlapping windows:

    stride = max(TUBELET, chunk - 2 * margin)          # 32 - 16 = 16 frames

16 frames is 8 latents at tubelet 2, so each surviving window contributes
exactly 8 latents and consecutive latents 8 apart occupy the *same position
inside their respective windows*. V-JEPA is a video transformer with temporal
position embeddings, so same-position latents share an additive component and
are pulled artificially close.

**The test.** Vary the stride and see whether the period follows it. If the
comb sits at `stride / TUBELET` for every setting, the chunking causes it and
the margin fix -- which trimmed boundary latents but left interior phase intact
-- did not go far enough. If the period stays at 8 regardless, the cause is
something else and this hypothesis is wrong.

Reports the raw autodistance curve per phase, so the comb is visible without a
model in the loop.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jetspace.data.episode import EpisodeDataset  # noqa: E402
from jetspace.models.vjepa import TUBELET, VJEPAEncoder  # noqa: E402
from jetspace.utils.device import get_device  # noqa: E402


def comb_strength(z: np.ndarray, period: int, max_h: int) -> tuple[np.ndarray, float]:
    """Mean ||z_t+h - z_t|| per h, and how far the on-phase lags sit below the rest."""
    z = z.reshape(z.shape[0], -1)
    z = (z - z.mean(0)) / (z.std(0) + 1e-6)
    d = np.zeros(max_h)
    for h in range(1, max_h + 1):
        d[h - 1] = np.linalg.norm(z[h:] - z[:-h], axis=1).mean()
    on = np.array([d[h - 1] for h in range(1, max_h + 1) if h % period == 0])
    off = np.array([d[h - 1] for h in range(1, max_h + 1) if h % period != 0])
    if not len(on) or not len(off):
        return d, float("nan")
    return d, float(off.mean() / max(on.mean(), 1e-9))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="push")
    ap.add_argument("--data", default=None)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--max-h", type=int, default=32)
    ap.add_argument("--camera", default=None)
    args = ap.parse_args()

    data = Path(args.data or f"data/episodes/{args.task}")
    device = get_device("auto")
    enc = VJEPAEncoder(device=device)
    ds = EpisodeDataset(data)
    cam = args.camera or ds.info["cameras"][0]

    # (chunk, margin) -> stride in frames -> expected comb period in latents.
    settings = [(32, 8), (32, 12), (32, 4), (32, 15), (64, 8)]

    print(f"{args.task}: {min(args.episodes, len(ds))} episodes, camera {cam!r}, "
          f"tubelet {TUBELET}\n")
    print(f"{'chunk':>6} {'margin':>7} {'stride_f':>9} {'expect':>7} "
          f"{'measured':>9} {'comb ratio':>11}  verdict")
    print("-" * 72)

    for chunk, margin in settings:
        stride_f = max(TUBELET, chunk - 2 * margin)
        expect = stride_f // TUBELET
        curves = []
        for i in range(min(args.episodes, len(ds))):
            frames = ds[i][f"pixels_{cam}"]
            with torch.no_grad():
                z = enc.encode(frames, chunk=chunk, margin=margin).float().cpu().numpy()
            if z.shape[0] > args.max_h + 1:
                curves.append(z)
        if not curves:
            print(f"{chunk:>6} {margin:>7} {stride_f:>9} {expect:>7}  (episodes too short)")
            continue

        # Search every plausible period and report whichever comb is strongest,
        # rather than only checking the one we expect to find.
        best_p, best_r, curve = 0, 0.0, None
        for p in range(2, args.max_h // 2 + 1):
            rs, ds_ = [], []
            for z in curves:
                d, r = comb_strength(z, p, args.max_h)
                rs.append(r)
                ds_.append(d)
            r = float(np.nanmean(rs))
            if r > best_r:
                best_p, best_r, curve = p, r, np.mean(ds_, axis=0)

        hit = "MATCHES stride" if best_p == expect else "does not match"
        print(f"{chunk:>6} {margin:>7} {stride_f:>9} {expect:>7} {best_p:>9} "
              f"{best_r:>10.2f}x  {hit}")
        if curve is not None and chunk == 32 and margin == 8:
            print("\n   autodistance by lag (no model involved):")
            for h in range(1, min(17, args.max_h + 1)):
                bar = "#" * int(curve[h - 1] / max(curve.max(), 1e-9) * 40)
                mark = "  <-- on phase" if h % best_p == 0 else ""
                print(f"   h={h:>2} {curve[h-1]:>7.3f} {bar}{mark}")
            print()

    print("\n" + "=" * 72)
    print("If the measured period tracks stride/TUBELET across settings, the comb")
    print("is an artifact of overlapped-window encoding: latents at the same")
    print("offset within their window share V-JEPA's temporal position embedding")
    print("and are pulled together regardless of what the robot did.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
