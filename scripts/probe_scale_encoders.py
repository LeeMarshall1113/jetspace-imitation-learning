#!/usr/bin/env python3
"""Which additional encoders can be added cheaply?

    PYTHONPATH=/workspace/.pydeps python scripts/probe_scale_encoders.py

E12 runs nine arms. CortexBench compares ~10 encoders and Burns et al. 15, and
that count is the first number a reviewer of a benchmark paper looks at. The
caching path is generic, so each additional encoder is roughly twenty minutes of
compute and no new code -- the cheapest available scale.

What matters is that the additions are not redundant. A second ViT-B trained the
same way as one already present adds a row, not evidence. These candidates each
change one thing: capacity within a family (base -> large), an older generation
of the same objective, or an objective not yet represented.

Probes for gated access, transformers support and a poolable token layout, and
reports rather than assumes -- Theia and R3M both failed that check earlier.
"""

from __future__ import annotations

import sys

import numpy as np
import torch

#: name, HF id, why it is not redundant with what E12 already has
CANDIDATES = [
    ("dinov2-large", "facebook/dinov2-large",
     "capacity within DINOv2, which currently leads on viewpoint"),
    ("dinov3-large", "facebook/dinov3-vitl16-pretrain-lvd1689m",
     "capacity within DINOv3, which loses to DINOv2 at base size"),
    ("siglip", "google/siglip-base-patch16-224",
     "SigLIP 1 against SigLIP 2 -- generation within one objective"),
    ("vit-large", "google/vit-large-patch16-224",
     "capacity within supervised ImageNet"),
    ("aimv2-base", "apple/aimv2-base-patch14-224",
     "capacity within AIMv2, currently only present at large"),
    ("clip-large", "openai/clip-vit-large-patch14",
     "capacity within CLIP, whose axis-dependence is the sharpest result"),
    ("vc1-large", "vc1-large",
     "capacity within the one robotics-specific arm that loads"),
]


def probe(name: str, hf_id: str):
    sys.path.insert(0, "scripts")
    from cache_latents_hf import build

    proc, model = build(hf_id, "cpu")
    n = sum(p.numel() for p in model.parameters())
    with torch.no_grad():
        inputs = proc(images=[np.zeros((224, 224, 3), dtype=np.uint8)],
                      return_tensors="pt")
        out = model(**dict(inputs)).last_hidden_state
    t = out.shape[1]
    side = int(t ** 0.5)
    while side > 0 and side * side > t:
        side -= 1
    prefix = t - side * side
    if prefix > 8:
        raise ValueError(f"{t} tokens implies {prefix} prefix tokens")
    return n / 1e6, f"{side}x{side}+{prefix}"


def main() -> int:
    print(f"{'name':14s} {'params':>8}  {'grid':>10}  status")
    print("-" * 78)
    ok = []
    for name, hf_id, why in CANDIDATES:
        try:
            n, grid = probe(name, hf_id)
            print(f"{name:14s} {n:>7.0f}M  {grid:>10}  OK    {why}")
            ok.append(name)
        except Exception as e:  # noqa: BLE001
            msg = str(e).split("\n")[0][:60]
            print(f"{name:14s} {'':>8}  {'':>10}  FAIL  {msg}")
    print(f"\n{len(ok)}/{len(CANDIDATES)} usable: {ok}")
    print(f"adding these takes E12 from 9 to {9 + len(ok)} arms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
