#!/usr/bin/env python3
"""Turn episodes or policy rollouts into things you can actually look at.

    # video + contact sheet from recorded demonstrations
    python scripts/render.py --data data/episodes/so101_reach

    # watch a trained policy instead of the demos
    python scripts/render.py --checkpoint checkpoints/bc_seed0.pt

Outputs into --out (default `renders/`):

    episodes.mp4       several episodes concatenated, with a gap between each
    contact_sheet.png  grid: one row per episode, time running left to right

The contact sheet is the more useful of the two. A video shows you one run; the
grid shows twenty at once, which is how you actually notice that the arm always
drifts left, or that a whole cluster of targets never gets reached.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jetspace.data.episode import EpisodeDataset  # noqa: E402


def contact_sheet(episodes: list[np.ndarray], cols: int = 8, pad: int = 2) -> np.ndarray:
    """Grid of frames: one row per episode, `cols` evenly-spaced timesteps."""
    tiles = []
    for frames in episodes:
        # Sample evenly across the episode so the row spans start -> finish
        # regardless of how long that particular episode ran.
        idx = np.linspace(0, len(frames) - 1, cols).round().astype(int)
        tiles.append([frames[i] for i in idx])

    h, w = tiles[0][0].shape[:2]
    rows, cols_n = len(tiles), cols
    sheet = np.full(
        ((h + pad) * rows + pad, (w + pad) * cols_n + pad, 3), 32, dtype=np.uint8
    )
    for r, row in enumerate(tiles):
        for c, frame in enumerate(row):
            y, x = pad + r * (h + pad), pad + c * (w + pad)
            sheet[y : y + h, x : x + w] = frame
    return sheet


def rollout_frames(
    checkpoint: Path, seeds: list[int], max_steps: int, randomize: bool = False,
    task: str = "pickplace",
) -> list[np.ndarray]:
    import torch

    from jetspace.envs.registry import get_task
    from jetspace.policies.bc import BCPolicy, SimpleVisualEncoder
    from jetspace.utils.device import get_device

    device = get_device("auto")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    policy = BCPolicy(SimpleVisualEncoder(), ckpt["proprio_dim"], ckpt["action_dim"]).to(device)
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    mean = np.asarray(ckpt["norm"]["proprio_mean"], dtype=np.float32)
    std = np.asarray(ckpt["norm"]["proprio_std"], dtype=np.float32)
    camera = ckpt.get("camera", "front")
    norm, adim = ckpt["norm"], ckpt["action_dim"]

    env = get_task(task)["env"](image_size=224, max_steps=max_steps, randomize=randomize)
    out = []
    for seed in seeds:
        obs = env.reset(seed=seed)
        frames, done = [obs.pixels[camera]], False
        while not done:
            action = policy.act(
                obs.pixels[camera], (obs.proprio - mean) / std, obs.proprio[:adim], norm,
                device=device,
            )
            result = env.step(action)
            obs = result.obs
            frames.append(obs.pixels[camera])
            done = result.terminated or result.truncated
        out.append(np.stack(frames))
    env.close()
    return out


def expert_frames(n: int, max_steps: int, task: str = "pickplace") -> list[np.ndarray]:
    """Scripted-expert rollouts in the randomized world, for visual inspection."""
    from jetspace.envs.registry import get_task

    spec = get_task(task)
    env = spec["env"](image_size=224, max_steps=max_steps, randomize=True)
    expert = spec["expert"](env, np.random.default_rng(0))
    out = []
    for i in range(n):
        obs = env.reset(seed=900_000_000 + i)
        expert.reset(env)
        frames, done = [obs.pixels[env.camera_names[0]]], False
        while not done:
            result = env.step(expert.act(obs))
            obs = result.obs
            frames.append(obs.pixels[env.camera_names[0]])
            done = result.terminated or result.truncated
        out.append(np.stack(frames))
    env.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--checkpoint", default=None, help="render policy rollouts instead of demos")
    ap.add_argument("--out", default="renders")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--task", default="pickplace", choices=["reach", "pickplace"])
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--gif", action="store_true",
                   help="also write episodes.gif, sized for embedding in a README")
    ap.add_argument("--gif-size", type=int, default=200)
    ap.add_argument("--gif-stride", type=int, default=2, help="keep every Nth frame")
    ap.add_argument("--randomize", action="store_true",
                   help="render the randomized world instead of the clean one")
    args = ap.parse_args()
    if args.data is None:
        args.data = f"data/episodes/{args.task}"

    import imageio.v2 as imageio

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.checkpoint:
        seeds = [900_000_000 + i for i in range(args.episodes)]
        episodes = rollout_frames(
            Path(args.checkpoint), seeds, args.max_steps,
            randomize=args.randomize, task=args.task,
        )
        fps, label = 25, f"policy {Path(args.checkpoint).name}"
    elif args.randomize:
        # No checkpoint: show what the randomized world looks like, driven by
        # the scripted expert. This is the picture to compare against a photo
        # of the real setup.
        episodes = expert_frames(args.episodes, args.max_steps, task=args.task)
        fps, label = 25, "randomized world (scripted expert)"
    else:
        ds = EpisodeDataset(args.data)
        cam = ds.info["cameras"][0]
        n = min(args.episodes, len(ds))
        episodes = [ds[i][f"pixels_{cam}"] for i in range(n)]
        fps, label = ds.info["fps"], f"demos from {args.data}"

    sheet = contact_sheet(episodes, cols=args.cols)
    sheet_path = out / "contact_sheet.png"
    imageio.imwrite(sheet_path, sheet)

    # A short gap between episodes so the video reads as separate attempts
    # rather than one long continuous motion.
    gap = np.full_like(episodes[0][:1], 16)
    video = np.concatenate([np.concatenate([ep, np.repeat(gap, 5, 0)]) for ep in episodes])
    video_path = out / "episodes.mp4"
    imageio.mimsave(video_path, list(video), fps=fps, macro_block_size=1)

    print(f"{label}: {len(episodes)} episodes")
    print(f"  {sheet_path}  ({sheet.shape[1]}x{sheet.shape[0]})")
    print(f"  {video_path}  ({len(video)} frames @ {fps} fps)")

    if args.gif:
        # GitHub renders GIFs inline in a README but will not play a committed
        # mp4, so this is the format that actually shows up on the page.
        # Subsampled and downscaled hard, because an inline asset that takes a
        # second to load is worse than no asset.
        from PIL import Image

        frames = [
            Image.fromarray(f).resize((args.gif_size, args.gif_size), Image.BILINEAR)
            for f in video[:: args.gif_stride]
        ]
        gif_path = out / "episodes.gif"
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 * args.gif_stride / fps),
            loop=0,
            optimize=True,
        )
        print(f"  {gif_path}  ({len(frames)} frames, {gif_path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
