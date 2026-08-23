#!/usr/bin/env python3
"""Encode a dataset once with the frozen V-JEPA 2 encoder, cache to disk.

    python scripts/cache_latents.py --task pickplace

This is the step that makes the rest of M3/M4 cheap. The encoder is frozen and
deterministic, so its outputs never change: run it once, then every downstream
experiment — action-conditioned predictor, latent RL, the whole data-efficiency
sweep in docs/task-hierarchy.md — reads latents instead of pixels.

Measured on the target machine (RX 9070 XT, ROCm 7.2.4, bf16):

    peak VRAM      0.79 GB of 15.9      <- memory is NOT the constraint
    throughput     ~5.2 frames/second   <- this is
    storage        64 KB per frame-pair at 4x4x1024 float32

So caching is a wall-clock cost paid once, not a memory problem. Roughly two
hours for 400 episodes. Re-running skips episodes already cached.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jetspace.data.episode import EpisodeDataset  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="pickplace", choices=["reach", "push", "pickplace"])
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--pool-grid", type=int, default=4,
                    help="spatial tokens per side; 16 keeps V-JEPA's native grid")
    ap.add_argument("--chunk", type=int, default=32, help="frames per forward pass")
    ap.add_argument("--margin", type=int, default=8,
                    help="context frames discarded at each window edge")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"],
                    help="storage dtype; float16 halves disk for no measurable loss")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    data = Path(args.data or f"data/episodes/{args.task}")
    out = Path(args.out or f"cache/latents/{args.task}")
    out.mkdir(parents=True, exist_ok=True)

    ds = EpisodeDataset(data)
    camera = ds.info["cameras"][0]
    n = min(args.limit or len(ds), len(ds))
    print(f"{data}: {len(ds)} episodes, encoding {n} from camera {camera!r}")

    from jetspace.models.vjepa import MODEL_ID, VJEPAEncoder

    t0 = time.time()
    enc = VJEPAEncoder(pool_grid=args.pool_grid)
    print(f"loaded {MODEL_ID} in {time.time() - t0:.1f}s  "
          f"(grid {args.pool_grid}x{args.pool_grid}, hidden {enc.hidden})")

    store_dtype = np.float16 if args.dtype == "float16" else np.float32
    total_frames = skipped = 0
    t0 = time.time()
    for i in range(n):
        record = ds.records[i]
        dest = out / f"episode_{record['index']:06d}.npy"
        if dest.exists():
            skipped += 1
            continue
        frames = ds[i][f"pixels_{camera}"]
        z = enc.encode(frames, chunk=args.chunk, margin=args.margin).numpy().astype(store_dtype)
        np.save(dest, z)
        total_frames += len(frames)

        if (i + 1) % 10 == 0 or i == n - 1:
            el = time.time() - t0
            fps = total_frames / max(el, 1e-6)
            remaining = (n - i - 1) * (total_frames / max(i + 1 - skipped, 1)) / max(fps, 1e-6)
            print(f"  [{i+1}/{n}] {total_frames} frames  {fps:.1f} fps  "
                  f"eta {remaining/60:.1f} min")

    # Record how the cache was produced. A latent cache with no provenance is
    # indistinguishable from a stale one, and silently training on latents from
    # a different encoder or pooling grid would be unrecoverable.
    meta = {
        "model_id": MODEL_ID,
        "pool_grid": args.pool_grid,
        "hidden": enc.hidden,
        "frames_per_latent": enc.spec.frames_per_latent,
        "dtype": args.dtype,
        "camera": camera,
        "chunk": args.chunk,
        "margin": args.margin,
        "source": str(data),
        "episodes": n,
    }
    (out / "info.json").write_text(json.dumps(meta, indent=2))

    el = time.time() - t0
    size = sum(f.stat().st_size for f in out.glob("*.npy")) / 1e9
    print(f"\nencoded {total_frames} frames in {el/60:.1f} min "
          f"({total_frames/max(el,1e-6):.1f} fps), skipped {skipped} already cached")
    print(f"cache: {out}  ({size:.2f} GB)")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
