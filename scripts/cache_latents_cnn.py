#!/usr/bin/env python3
"""Cache latents from a scratch CNN encoder — E6 arms 2 and 3.

    # arm 2: random weights, never trained
    python scripts/cache_latents_cnn.py --task push --out cache/latents/e6_push_rand

    # arm 3: encoder lifted from a jointly-trained checkpoint
    python scripts/cache_latents_cnn.py --task push --encoder checkpoints/e6/joint_push.pt \\
        --out cache/latents/e6_push_joint

Writes exactly the layout `cache_latents.py` writes — same `.npy` per episode,
same `info.json` keys — so every downstream script (E2, E3, conservatism, the
inverse-dynamics probe, the domain-gap metrics) runs unchanged on all three
arms. Making the arms interchangeable at the file level is what keeps the
comparison honest: nothing downstream can tell which encoder produced a cache,
so nothing downstream can treat them differently.

`frames_per_latent` is pinned to V-JEPA's tubelet of 2. A CNN at a different
temporal rate would confound representation quality with how much time each
latent covers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jetspace.data.episode import EpisodeDataset  # noqa: E402
from jetspace.models.cnn_encoder import ScratchCNNEncoder, build  # noqa: E402
from jetspace.utils.device import get_device  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="push")
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--camera", default=None)
    ap.add_argument("--encoder", default=None,
                    help="checkpoint holding a trained encoder; omit for random weights")
    ap.add_argument("--pool-grid", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--frames-per-latent", type=int, default=2)
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = Path(args.data or f"data/episodes/{args.task}")
    out = Path(args.out or f"cache/latents/e6_{args.task}_cnn")
    out.mkdir(parents=True, exist_ok=True)

    ds = EpisodeDataset(data)
    camera = args.camera or ds.info["cameras"][0]
    n = min(args.limit or len(ds), len(ds))
    device = get_device("auto")

    # Seed before construction: for the random arm the weights ARE the
    # experiment, so an unseeded init would make the result unreproducible.
    torch.manual_seed(args.seed)

    if args.encoder:
        ckpt = torch.load(args.encoder, map_location="cpu", weights_only=False)
        cfg = ckpt.get("encoder_config", {})
        enc = ScratchCNNEncoder(
            hidden=cfg.get("hidden", args.hidden),
            grid=cfg.get("grid", args.pool_grid),
            frames_per_latent=cfg.get("frames_per_latent", args.frames_per_latent),
            width=cfg.get("width", args.width),
        )
        enc.load_state_dict(ckpt["encoder"])
        kind = "trained"
    else:
        enc = build("random", hidden=args.hidden, grid=args.pool_grid,
                    frames_per_latent=args.frames_per_latent, width=args.width)
        kind = "random"

    enc = enc.to(device).eval()
    n_params = sum(p.numel() for p in enc.parameters())
    print(f"{data}: {len(ds)} episodes, encoding {n} from camera {camera!r}")
    print(f"scratch CNN [{kind}]  {n_params/1e6:.1f}M params  "
          f"grid {enc.grid}x{enc.grid}  hidden {enc.hidden}  fpl {enc.frames_per_latent}")

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
        z = enc.encode(torch.from_numpy(np.ascontiguousarray(frames))).numpy().astype(store)
        np.save(dest, z)
        total += len(frames)
        if (i + 1) % 10 == 0 or i == n - 1:
            el = time.time() - t0
            print(f"  [{i+1}/{n}] {total} frames  {total/max(el,1e-6):.1f} fps")

    meta = {
        "model_id": f"scratch_cnn_{kind}",
        "encoder_kind": kind,
        "encoder_checkpoint": args.encoder,
        "n_params": int(n_params),
        "pool_grid": enc.grid,
        "hidden": enc.hidden,
        "frames_per_latent": enc.frames_per_latent,
        "width": args.width,
        "dtype": args.dtype,
        "camera": camera,
        # No windowing: a CNN has no temporal position embedding, so it cannot
        # acquire the period-8 comb that overlapped-window transformer encoding
        # stamps into simulated latents (ledger L6). Recorded so nothing
        # downstream infers a chunk/margin that was never used.
        "chunk": None,
        "margin": None,
        "seed": args.seed,
        "source": str(data),
        "episodes": n,
    }
    (out / "info.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {out}  ({n - skipped} encoded, {skipped} already present)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
