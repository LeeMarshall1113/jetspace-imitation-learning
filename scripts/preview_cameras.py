#!/usr/bin/env python3
"""Render one frame from every sweep camera into a contact sheet, and look at it.

    python scripts/preview_cameras.py --task push --pretty

The N1b viewpoint sweep is only meaningful if every camera actually frames the
workspace. A pose that is edge-on to the plane of motion, or that clips the arm
out of frame, would still produce latents, still produce a gap number, and still
look like a result.

This project has already made that exact mistake once: a camera was mounted
edge-on to the plane the arm moved in, every metric looked plausible, and it was
caught only by rendering a contact sheet and looking at it. Cheap insurance,
run before the sweep and not after it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jetspace.envs.registry import get_task  # noqa: E402
from jetspace.envs.so101_env import ALL_CAMERAS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="push")
    ap.add_argument("--pretty", action="store_true")
    ap.add_argument("--steps", type=int, default=25,
                    help="advance this many steps so the arm is mid-motion")
    ap.add_argument("--out", default="renders/camera_sweep.png")
    args = ap.parse_args()

    spec = get_task(args.task)
    env = spec["env"](image_size=224, pretty=args.pretty, cameras=ALL_CAMERAS)
    obs = env.reset(seed=0)
    expert = spec["expert"](env, np.random.default_rng(0))
    expert.reset(env)

    for _ in range(args.steps):
        step = env.step(expert.act(obs))
        obs = step.obs

    frames = [(c, obs.pixels[c]) for c in ALL_CAMERAS if c in obs.pixels]
    if not frames:
        print("no frames rendered")
        return 1

    h, w = frames[0][1].shape[:2]
    pad = 4
    sheet = np.full((h + 2 * pad, len(frames) * (w + pad) + pad, 3), 30, np.uint8)
    for i, (_, img) in enumerate(frames):
        x = pad + i * (w + pad)
        sheet[pad:pad + h, x:x + w] = img

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio

        imageio.imwrite(out, sheet)
    except Exception as exc:  # noqa: BLE001
        print(f"could not write {out}: {exc}")
        return 1

    print(f"cameras (left to right): {[c for c, _ in frames]}")
    for c, img in frames:
        # A view that is mostly one flat colour is framing the backdrop, not the
        # workspace. Not proof of a good pose, but it catches the worst ones
        # without a human in the loop.
        std = float(img.astype(np.float32).std())
        flag = "  <-- nearly uniform, check framing" if std < 12.0 else ""
        print(f"  {c:12s} pixel std {std:6.2f}{flag}")
    print(f"\nwrote {out}  ({sheet.shape[1]}x{sheet.shape[0]})")
    print("LOOK AT IT. Every camera must show the arm and the workspace.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
