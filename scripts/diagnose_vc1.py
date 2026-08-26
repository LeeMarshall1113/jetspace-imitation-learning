#!/usr/bin/env python3
"""Is VC-1 really the worst encoder, or did I load it wrong?

    PYTHONPATH=/workspace/.pydeps python scripts/diagnose_vc1.py

VC-1 came last of nine arms at 1.147 -- worse than random convolutional
features (0.673) and worse than predicting the mean action. It is a NeurIPS 2023
encoder MAE-pretrained specifically for embodied control, so "worst of nine" is
far more likely to be my integration than the model.

Three things that would produce this without VC-1 being bad:

  1. WRONG NORMALISATION. The wrapper applies ImageNet mean/std. VC-1 ships its
     own transform (vc_models.transforms.vit_transforms) which is not loaded
     here, and a ViT fed mis-normalised input degrades badly.
  2. DEGENERATE FEATURES. If the checkpoint did not really land on the timm
     architecture, the output could be near-constant across frames -- which no
     downstream head can use, and which looks exactly like "bad encoder".
  3. WRONG TOKENS. The config says use_cls: True, so VC-1's intended
     representation may be the CLS token rather than the patch grid every other
     arm is pooled from.

This measures feature health directly rather than arguing about it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def stats(name: str, prefix: str, pose: str = "r1_ref") -> dict | None:
    d = Path("cache/latents") / f"{prefix}__{pose}"
    files = sorted(d.glob("episode_*.npy"))
    if not files:
        print(f"{name:10s} no cache")
        return None
    z = np.load(files[0]).astype(np.float32)          # (T, g, g, D)
    flat = z.reshape(z.shape[0], -1)
    # Per-dimension spread across time: if the encoder returns near-identical
    # features for every frame, nothing downstream can separate the timesteps.
    per_dim = flat.std(axis=0)
    # Cosine similarity between consecutive frames: ~1.0 means the encoder is
    # not distinguishing them at all.
    a = flat[:-1] / (np.linalg.norm(flat[:-1], axis=1, keepdims=True) + 1e-9)
    b = flat[1:] / (np.linalg.norm(flat[1:], axis=1, keepdims=True) + 1e-9)
    cons = float((a * b).sum(axis=1).mean())
    # Effective rank via participation ratio of the covariance eigenvalues.
    c = np.cov(flat - flat.mean(0), rowvar=False)
    ev = np.linalg.eigvalsh(c).clip(min=0)
    eff_rank = float((ev.sum() ** 2) / ((ev ** 2).sum() + 1e-12))
    out = {"mean": float(flat.mean()), "std": float(flat.std()),
           "dim_std_median": float(np.median(per_dim)),
           "consecutive_cos": cons, "eff_rank": eff_rank, "dim": flat.shape[1]}
    print(f"{name:10s} mean {out['mean']:+8.3f}  std {out['std']:7.3f}  "
          f"median per-dim std {out['dim_std_median']:7.4f}  "
          f"consec cos {cons:.4f}  eff-rank {eff_rank:6.1f} / {flat.shape[1]}")
    return out


def main() -> int:
    task = sys.argv[1] if len(sys.argv) > 1 else "push"
    print("Feature health at the reference pose. A healthy encoder varies")
    print("across frames (consecutive cosine well below 1.0) and spreads its")
    print("variance over many directions (high effective rank).\n")
    arms = [("vjepa2", f"r1_{task}"), ("dinov2", f"dino_{task}"),
            ("vc1", f"vc1_{task}"), ("clip", f"clip_{task}"),
            ("random", f"r1cnn_{task}")]
    got = {n: stats(n, p) for n, p in arms}

    vc1 = got.get("vc1")
    ref = got.get("dinov2") or got.get("vjepa2")
    if not vc1 or not ref:
        return 0

    print()
    suspicious = []
    if vc1["consecutive_cos"] > 0.999:
        suspicious.append("consecutive frames are near-identical -- the "
                          "encoder is not distinguishing timesteps")
    if vc1["eff_rank"] < 0.1 * ref["eff_rank"]:
        suspicious.append(f"effective rank {vc1['eff_rank']:.1f} vs "
                          f"{ref['eff_rank']:.1f} for a working arm -- features "
                          f"are collapsed")
    if vc1["dim_std_median"] < 1e-4:
        suspicious.append("per-dimension spread is essentially zero")

    if suspicious:
        print("VC-1 features look BROKEN, not merely bad:")
        for s in suspicious:
            print(f"  - {s}")
        print("\nThe 1.147 result is most likely a loading or preprocessing "
              "fault\nand must not be reported as a property of VC-1.")
    else:
        print("VC-1 features look healthy on these measures: they vary across")
        print("frames and are not collapsed. That does not prove the")
        print("normalisation is right, but it rules out the loudest failure")
        print("mode, and the poor score may be genuine for this task.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
