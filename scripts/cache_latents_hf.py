#!/usr/bin/env python3
"""Cache frozen features from any HuggingFace vision backbone.

    python scripts/cache_latents_hf.py --model facebook/dinov2-base \
        --data data/episodes/r1_push --camera r1_ref --out cache/latents/dino_push__r1_ref

E8 compares frozen V-JEPA 2 against a frozen random CNN and finds a large gap.
That comparison cannot distinguish "V-JEPA 2 is special" from "any strong
pretraining beats noise", which is the first question a reviewer asks and the
one a single baseline can never answer. This script adds the middle of the
range: image-level self-supervised (DINOv2), image-text contrastive (CLIP), and
plain supervised ImageNet (ViT).

**The matched-comparison rules, which are the whole point.**

`pool_grid` and `frames_per_latent` must match the V-JEPA side exactly, or the
arms differ in more than the encoder and ledger L7 happens again. Patch tokens
are adaptively pooled to 4x4 and two consecutive frames are averaged into one
latent, so every arm emits `(T//2, 4, 4, D)` on identical timesteps.

**The asymmetry that cannot be removed, stated rather than hidden.** V-JEPA 2
is a *video* encoder: its latent sees two frames jointly and can represent
motion between them. The image encoders here see each frame independently and
have their features averaged. Averaging is the closest available match, but it
is not the same operation, and any V-JEPA advantage found here is partly an
advantage of video pretraining over image pretraining rather than of V-JEPA
specifically. Reporting an image-encoder baseline without this caveat would
overstate the result.

The CLS token is dropped: it is a global summary, and the downstream policy head
needs the spatial grid that survives pooling.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jetspace.data.episode import EpisodeDataset  # noqa: E402
from jetspace.utils.device import get_device  # noqa: E402

#: Short names so cache directories stay readable.
ALIASES = {
    # Contemporaries of V-JEPA 2 (2025-06). These are the comparison that
    # matters -- beating 2021-2023 encoders would only show V-JEPA is newer.
    "dinov3": "facebook/dinov3-vitb16-pretrain-lvd1689m",      # 2025-08, gated
    "dinov3-large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "siglip2": "google/siglip2-base-patch16-224",              # 2025-02
    "aimv2": "apple/aimv2-large-patch14-224",                  # 2024-11
    # Older, kept as a reference row for what the prior literature used.
    "dinov2": "facebook/dinov2-base",                          # 2023-04
    "dinov2-large": "facebook/dinov2-large",
    "siglip": "google/siglip-base-patch16-224",                # 2023-03
    "clip": "openai/clip-vit-base-patch16",                    # 2021-01
    "vit-in1k": "google/vit-base-patch16-224",                 # 2020-10
}


def build(model_id: str, device: str):
    from transformers import AutoImageProcessor, AutoModel

    proc = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    # CLIP and SigLIP wrap a vision tower alongside a text tower; only the
    # vision side is wanted and loading both wastes memory.
    if hasattr(model, "vision_model"):
        model = model.vision_model
    return proc, model.to(device).eval()


@torch.no_grad()
def encode(frames: np.ndarray, proc, model, device: str, grid: int,
           fpl: int, batch: int = 32) -> np.ndarray:
    """(T, H, W, 3) uint8 -> (T // fpl, grid, grid, D) float32."""
    usable = (len(frames) // fpl) * fpl
    frames = frames[:usable]

    feats = []
    for i in range(0, usable, batch):
        chunk = frames[i:i + batch]
        inputs = proc(images=list(chunk), return_tensors="pt").to(device)
        out = model(**inputs).last_hidden_state          # (B, prefix + P, D)
        n_tok = out.shape[1]
        # Backbones differ in what they put BEFORE the patch grid: SigLIP has
        # nothing, CLIP and DINOv2 have a CLS token, and DINOv3 has CLS plus
        # four register tokens (201 = 196 + 1 + 4), which an earlier
        # CLS-or-nothing check rejected outright. In every case the patch
        # tokens are the trailing square block, so find the largest square that
        # fits and take the tail.
        side = int(n_tok ** 0.5)
        while side > 0 and side * side > n_tok:
            side -= 1
        prefix = n_tok - side * side
        if side == 0 or prefix > 8:
            raise ValueError(
                f"{n_tok} tokens leaves no plausible square patch grid "
                f"(largest square {side}x{side} implies {prefix} prefix "
                f"tokens); this backbone's layout is not handled")
        tokens = out[:, prefix:] if prefix else out
        b, _, d = tokens.shape
        t = tokens.reshape(b, side, side, d).permute(0, 3, 1, 2)
        t = F.adaptive_avg_pool2d(t, (grid, grid))       # (B, D, g, g)
        feats.append(t.permute(0, 2, 3, 1).float().cpu())
    z = torch.cat(feats)                                  # (T, g, g, D)
    # Average consecutive frames into one latent so the timestep grid matches
    # the video encoder's tubelet exactly.
    z = z.reshape(usable // fpl, fpl, grid, grid, z.shape[-1]).mean(dim=1)
    return z.numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help=f"HF id or alias: {sorted(ALIASES)}")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--camera", default=None)
    ap.add_argument("--pool-grid", type=int, default=4)
    ap.add_argument("--frames-per-latent", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    args = ap.parse_args()

    model_id = ALIASES.get(args.model, args.model)
    device = get_device("auto")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds = EpisodeDataset(Path(args.data))
    camera = args.camera or ds.info["cameras"][0]
    n = min(args.limit or len(ds), len(ds))

    proc, model = build(model_id, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args.data}: {len(ds)} episodes, encoding {n} from camera {camera!r}")
    print(f"{model_id}  {n_params / 1e6:.0f}M params  grid "
          f"{args.pool_grid}x{args.pool_grid}  fpl {args.frames_per_latent}")

    store = np.float16 if args.dtype == "float16" else np.float32
    total = skipped = 0
    t0 = time.time()
    for i in range(n):
        record = ds.records[i]
        dest = out / f"episode_{record['index']:06d}.npy"
        if dest.exists():
            skipped += 1
            continue
        frames = ds[i][f"pixels_{camera}"]
        z = encode(frames, proc, model, device, args.pool_grid,
                   args.frames_per_latent)
        np.save(dest, z.astype(store))
        total += len(frames)
        if (i + 1) % 5 == 0 or i == n - 1:
            el = time.time() - t0
            print(f"  [{i + 1}/{n}] {total} frames  "
                  f"{total / max(el, 1e-6):.1f} fps")

    (out / "meta.json").write_text(json.dumps({
        "model": model_id, "camera": camera, "pool_grid": args.pool_grid,
        "frames_per_latent": args.frames_per_latent, "params": int(n_params),
        "source": str(args.data), "dtype": args.dtype,
        "note": "image encoder; consecutive frames averaged to match the "
                "video encoder's tubelet. Not equivalent to joint 2-frame "
                "encoding.",
    }, indent=2))
    print(f"wrote {n - skipped} episodes to {out} ({skipped} already cached)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
