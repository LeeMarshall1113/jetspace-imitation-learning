#!/usr/bin/env python3
"""Collect one E12 condition: everything pinned except the axis under test.

    python scripts/collect_e12.py --axis lighting --level 0.45 --episodes 10

E12 crosses nine encoders with four nuisance axes. `collect_demos.py --randomize`
is a boolean, so it cannot express "vary lighting and nothing else" -- it
randomises camera, lighting, materials, clutter AND dynamics together, which
would confound every axis with every other.

This pins the whole `RandomizationConfig` and moves exactly one parameter.
Dynamics randomisation is disabled outright: mass, damping, friction and
actuator gain change the TRAJECTORY, not just its appearance, and E12 needs the
underlying behaviour held fixed so that a change in head performance is
attributable to the visual nuisance rather than to a different rollout.

Episode seeds are shared across conditions for the same reason. With identical
seeds, identical dynamics and a deterministic expert, the arm does the same
thing in every condition and only the rendering differs -- which is what makes
"train at the reference, evaluate at the displaced condition" a measurement of
the nuisance rather than of two unrelated datasets.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jetspace.data.episode import EpisodeBuffer, EpisodeWriter  # noqa: E402
from jetspace.envs.registry import get_task  # noqa: E402

#: The reference condition. Every axis moves away from exactly this.
REF = {"lighting": 0.7, "texture": 0.0, "clutter": 0}

#: Displaced levels per axis, held out of training.
LEVELS = {
    "lighting": [0.30, 0.45, 0.55, 0.62],
    "texture": [0.06, 0.10, 0.16, 0.24],
    "clutter": [1, 2, 3, 4],
}


def build_config(axis: str, level):
    from jetspace.envs.randomization import RandomizationConfig

    light = level if axis == "lighting" else REF["lighting"]
    hue = level if axis == "texture" else REF["texture"]
    clutter = int(level) if axis == "clutter" else REF["clutter"]

    return RandomizationConfig(
        enabled=True,
        # Camera pinned to the E11 reference viewpoint so the viewpoint axis
        # does not leak into the other three.
        camera_mode="fixed",
        camera_azimuth_range=(0.0, 0.0),
        camera_elevation_range=(30.0, 30.0),
        camera_distance_range=(0.81, 0.81),
        camera_lookat=(0.30, 0.0, 0.10),
        camera_lookat_jitter=0.0,
        camera_pos_jitter=0.0,
        # The three axes under test.
        light_diffuse_range=(light, light),
        light_pos_jitter=0.0,
        material_hue_jitter=hue,
        n_distractors=(clutter, clutter),
        # Dynamics pinned: these change the trajectory, not its appearance.
        mass_scale_range=(1.0, 1.0),
        damping_scale_range=(1.0, 1.0),
        frictionloss_scale_range=(1.0, 1.0),
        actuator_gain_scale_range=(1.0, 1.0),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", required=True, choices=sorted(LEVELS) + ["reference"])
    ap.add_argument("--level", default=None,
                    help="displaced level; omit for the reference condition")
    ap.add_argument("--task", default="push")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.axis == "reference":
        axis, level, tag = "lighting", REF["lighting"], "ref"
    else:
        if args.level is None:
            print(f"--level required for axis {args.axis}")
            return 1
        axis = args.axis
        level = float(args.level) if axis != "clutter" else int(args.level)
        tag = f"{axis}_{str(level).replace('.', 'p')}"

    out = Path(args.out or f"data/episodes/e12_{args.task}__{tag}")
    cfg = build_config(axis, level)
    spec = get_task(args.task)
    env = spec["env"](image_size=args.image_size, pretty=True, randomize=cfg)
    camera = env.camera_names[0]

    writer = EpisodeWriter(
        out, task=args.task, fps=25, action_dim=env.action_dim,
        cameras=(camera,), image_size=args.image_size,
        extra_info={"e12_axis": axis, "e12_level": level, "e12_tag": tag},
    )
    expert = spec["expert"](env, np.random.default_rng(args.seed))

    print(f"{tag}: {args.episodes} episodes, camera {camera!r}, "
          f"{axis}={level}")
    written = skipped = 0
    for ep in range(args.episodes):
        # Shared across conditions on purpose -- same seed, same pinned
        # dynamics, same deterministic expert means the same trajectory, so
        # only the rendering differs between conditions.
        ep_seed = args.seed * 1000 + ep
        obs = env.reset(seed=ep_seed)
        expert.reset(env)
        buf = EpisodeBuffer()
        terminated = truncated = False
        while not (terminated or truncated):
            action = expert.act(obs)
            result = env.step(action)
            buf.add(pixels=obs.pixels, proprio=obs.proprio, action=action,
                    reward=result.reward, success=result.info["success"])
            obs = result.obs
            terminated, truncated = result.terminated, result.truncated
        if buf.buffer_size() == 0:
            skipped += 1
            continue
        writer.write(buf, metadata={"seed": ep_seed, "axis": axis,
                                    "level": level})
        written += 1

    env.close()
    print(f"  wrote {written} episodes to {out}" +
          (f" ({skipped} empty)" if skipped else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
