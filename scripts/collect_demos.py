#!/usr/bin/env python3
"""Collect demonstrations for the SO-101 reach task.

    # scripted expert -- runs headless, needs no human
    python scripts/collect_demos.py --policy scripted --episodes 200

    # human teleoperation -- needs a display
    python scripts/collect_demos.py --policy keyboard --episodes 20

    # inspect what was collected
    python scripts/collect_demos.py --summary-only

The scripted expert drives the gripper to the target with damped-least-squares
IK. That choice is deliberate and fixes the defect that sank the first M2 run:

  * The action is a function of **observable state** -- current joint
    configuration and target position -- not of elapsed episode time. The
    previous expert eased toward a goal on an internal counter, so early actions
    were nearly identical across every target and a behavior-cloned policy
    minimised its loss by ignoring the target entirely and predicting the mean
    trajectory.
  * Exploration noise is **constant**, not decaying. Noise that fades to zero
    near the goal leaves no off-path states in the dataset, so an imitator has
    no recovery behaviour to copy and small errors compound uncorrected.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jetspace.data.episode import EpisodeBuffer, EpisodeDataset, EpisodeWriter  # noqa: E402
from jetspace.envs.registry import get_task  # noqa: E402


class HumanTeleop:
    """Keyboard teleoperation via pygame. Requires a display."""

    def __init__(self, env: object, mode: str, step: float = 0.03):
        import pygame

        self.pygame = pygame
        self.env = env
        self.mode = mode
        self.step = step
        self.action_dim = env.action_dim
        pygame.init()
        self.screen = pygame.display.set_mode((360, 140))
        pygame.display.set_caption("JetSpace teleop - 1-5 select joint, arrows move, R restart, ESC quit")
        self.joystick = None
        if mode == "gamepad":
            pygame.joystick.init()
            if pygame.joystick.get_count() == 0:
                raise RuntimeError("No gamepad detected. Use --policy keyboard.")
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
        self._cmd = np.zeros(self.action_dim, dtype=np.float32)
        self._joint = 0
        self.quit = False
        self.restart = False

    def reset(self, env: object) -> bool:
        self._cmd = np.asarray(env.data.qpos[: self.action_dim], dtype=np.float32).copy()
        self.restart = False
        return True

    def act(self, obs) -> np.ndarray:  # noqa: ANN001
        pg = self.pygame
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.quit = True
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    self.quit = True
                elif event.key == pg.K_r:
                    self.restart = True
                elif pg.K_1 <= event.key <= pg.K_6:
                    self._joint = min(event.key - pg.K_1, self.action_dim - 1)

        delta = np.zeros(self.action_dim, dtype=np.float32)
        if self.mode == "gamepad" and self.joystick is not None:
            for i in range(min(self.action_dim, self.joystick.get_numaxes())):
                axis = self.joystick.get_axis(i)
                if abs(axis) > 0.15:  # deadzone
                    delta[i] = axis * self.step
        else:
            keys = pg.key.get_pressed()
            delta[self._joint] = (keys[pg.K_UP] - keys[pg.K_DOWN]) * self.step

        self._cmd = self._cmd + delta
        return self._cmd

    def close(self) -> None:
        self.pygame.quit()


def collect(args: argparse.Namespace) -> int:
    spec = get_task(args.task)
    steps = args.max_steps or spec["default_steps"]
    env = spec["env"](
        image_size=args.image_size, max_steps=steps,
        randomize=args.randomize, pretty=args.pretty,
    )
    rng = np.random.default_rng(args.seed)
    policy = (
        spec["expert"](env, rng, noise=args.noise)
        if args.policy == "scripted"
        else HumanTeleop(env, args.policy)
    )

    writer = EpisodeWriter(
        args.out,
        task=args.task,
        fps=env.control_hz,
        action_dim=env.action_dim,
        cameras=env.camera_names,
        image_size=args.image_size,
        extra_info={"policy": args.policy, "seed": args.seed, "robot": "so101",
                    "randomized": bool(args.randomize)},
    )
    print(f"Writing to {writer.root} (already contains {writer.num_episodes} episodes)")
    print(f"env: {env.action_dim} actuators @ {env.control_hz} Hz")

    written = skipped = 0
    for ep in range(args.episodes):
        # Record the reset seed: without it the target cannot be reproduced, and
        # an episode that cannot be replayed cannot be verified.
        
        ep_seed = int(rng.integers(2**31))
        obs = env.reset(seed=ep_seed)
        if not policy.reset(env):
            skipped += 1
            continue

        buffer = EpisodeBuffer()
        terminated = truncated = False

        while not (terminated or truncated):

            action = policy.act(obs)
            result = env.step(action)

            # Record what the expert MEANT, not what exploration executed.
            label = getattr(policy, "label", None)

            buffer.add(
                pixels=obs.pixels,
                proprio=obs.proprio,
                action=action if label is None else label,
                action_executed=action,
                reward=result.reward,
                success=result.info["success"],
            )

            obs = result.obs
            terminated, truncated = result.terminated, result.truncated

            if getattr(policy, "restart", False) or getattr(policy, "quit", False):
                break

        if getattr(policy, "quit", False):
            print("Aborted by user.")
            break
        if getattr(policy, "restart", False):
            print(f"  episode {ep}: discarded (restart)")
            continue

        # Only successful demos are kept: the space of failures is far larger
        # and less structured than the space of successes.

        if not buffer.Episode.success[-1] and not args.keep_failures:
            skipped += 1
            continue

        writer.write(buffer, metadata={"policy": args.policy, "seed": ep_seed})
        written += 1
        
        if ep % 20 == 0 or ep == args.episodes - 1:
            print(f"  episode {ep}: {buffer.buffer_size()} frames, success={buffer.Episode.success[-1]}")

    env.close()
    if hasattr(policy, "close"):
        policy.close()

    print(f"\nWrote {written} episodes, discarded {skipped}.")
    print(EpisodeDataset(args.out).summary())
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--policy", choices=["scripted", "keyboard", "gamepad"], default="scripted")
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--task", default="pickplace", choices=["reach", "push", "pickplace"])
    p.add_argument("--out", default=None)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--noise", type=float, default=0.015)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pretty", action="store_true",
                   help="render visual meshes (11x slower); for human-facing output")
    p.add_argument("--randomize", action="store_true",
                   help="domain randomization: lighting, camera, clutter, dynamics, latency")
    p.add_argument("--keep-failures", action="store_true")
    p.add_argument("--summary-only", action="store_true")
    args = p.parse_args()
    if args.out is None:
        args.out = f"data/episodes/{args.task}"

    if args.summary_only:
        print(EpisodeDataset(args.out).summary())
        return 0
    return collect(args)


if __name__ == "__main__":
    sys.exit(main())
