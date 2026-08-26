#!/usr/bin/env python3
"""Render one frame from every camera into a contact sheet, and look at it.

    python scripts/preview_cameras.py --task push --pretty --cameras r1

A sweep is only meaningful if every camera actually frames the workspace. A
pose that is edge-on to the plane of motion, or that clips the arm out of
frame, still produces latents, still produces a gap number, and still looks
like a result.

This project has already made that exact mistake once: a camera mounted edge-on
to the plane the arm moved in, every metric plausible, caught only by rendering
a contact sheet and looking at it. `docs/prereg-camera-ruler.md` registers this
check as a precondition of R1 rather than a nicety.

Also reports, per pose, the fraction of the frame the ARM occupies. At high
azimuth the arm occludes the object -- a real consequence of viewpoint, but not
a smooth function of angle, so it has to be visible in the output rather than
folded silently into the curve.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jetspace.envs.registry import get_task  # noqa: E402
from jetspace.envs.so101_env import ALL_CAMERAS, R1_POSES, r1_displacement  # noqa: E402


def select(spec: str) -> tuple[str, ...]:
    if spec == "all":
        return ALL_CAMERAS
    if spec == "r1":
        return ("front",) + tuple(R1_POSES)
    if spec == "n1b":
        return tuple(c for c in ALL_CAMERAS if not c.startswith("r1_"))
    return tuple(c.strip() for c in spec.split(",") if c.strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="push")
    ap.add_argument("--pretty", action="store_true")
    ap.add_argument("--steps", type=int, default=25,
                    help="advance this many steps so the arm is mid-motion")
    ap.add_argument("--cameras", default="all", help="'all', 'r1', 'n1b', or a list")
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--out", default="renders/camera_sweep.png")
    args = ap.parse_args()

    cams = select(args.cameras)
    spec = get_task(args.task)
    env = spec["env"](image_size=224, pretty=args.pretty, cameras=cams)
    obs = env.reset(seed=0)
    expert = spec["expert"](env, np.random.default_rng(0))
    expert.reset(env)

    for _ in range(args.steps):
        step = env.step(expert.act(obs))
        obs = step.obs

    frames = [(c, obs.pixels[c]) for c in cams if c in obs.pixels]
    if not frames:
        print("no frames rendered")
        return 1

    h, w = frames[0][1].shape[:2]
    pad, cols = 4, max(1, args.cols)
    rows = (len(frames) + cols - 1) // cols
    sheet = np.full((rows * (h + pad) + pad, cols * (w + pad) + pad, 3), 30, np.uint8)
    for i, (_, img) in enumerate(frames):
        r, c = divmod(i, cols)
        y, x = pad + r * (h + pad), pad + c * (w + pad)
        sheet[y:y + h, x:x + w] = img

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio

    imageio.imwrite(out, sheet)

    print(f"{len(frames)} cameras, {rows}x{cols} sheet\n")
    print(f"{'camera':14s} {'angle':>7} {'dist':>6} {'pixel sd':>9} {'arm %':>7}  flags")
    print("-" * 62)
    for c, img in frames:
        f = img.astype(np.float32)
        sd = float(f.std())
        # The arm and gripper are the saturated yellow/red parts of the scene;
        # the table and backdrop are blue-grey. A red-dominant mask is a crude
        # but serviceable proxy for how much frame the robot fills.
        arm = float(((f[..., 0] > f[..., 2] + 25)).mean() * 100)
        d = r1_displacement(c) if c.startswith("r1_") else {"angle": 0.0,
                                                            "dist_ratio": 1.0}
        flags = []
        if sd < 12.0:
            flags.append("NEARLY UNIFORM - check framing")
        if arm > 22.0:
            flags.append("arm fills frame - occlusion likely")
        if arm < 1.5:
            flags.append("arm barely visible")
        print(f"{c:14s} {d['angle']:>6.1f}° {d['dist_ratio']:>5.2f}x {sd:>9.2f} "
              f"{arm:>6.1f}%  {'; '.join(flags)}")

    print(f"\nwrote {out}  ({sheet.shape[1]}x{sheet.shape[0]})")
    print("LOOK AT IT. Every pose must show the arm and the workspace.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
