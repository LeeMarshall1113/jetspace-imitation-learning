#!/usr/bin/env python3
"""Collect demonstrations for the reach task.

    # scripted expert -- runs headless, needs no human
    python scripts/collect_demos.py --policy scripted --episodes 100

    # human teleoperation -- needs a display
    python scripts/collect_demos.py --policy keyboard --episodes 20
    python scripts/collect_demos.py --policy gamepad  --episodes 20

    # inspect what was collected
    python scripts/collect_demos.py --summary-only

The scripted policy exists so the recording pipeline, the dataset format and the
downstream loaders can be exercised end to end before any human sits down to
teleoperate, and so CI has demos to test against. It is a genuine expert for
`reach` (analytic 2-link IK), which makes it useful as a BC upper bound too --
but demos from it are NOT a substitute for human demos on tasks where the point
is to capture human strategy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jetspace.data.episode import EpisodeBuffer, EpisodeDataset, EpisodeWriter  # noqa: E402
from jetspace.envs.mujoco_env import MujocoReachEnv  # noqa: E402

L1 = L2 = 0.25  # link lengths, must match REACH_XML
BASE_XY = np.array([0.0, 0.0])


def solve_ik(target_xy: np.ndarray, elbow_up: bool = True) -> np.ndarray | None:
    """Analytic IK for the planar 2-link arm. Returns (q1, q2) or None."""
    dx, dy = target_xy - BASE_XY
    dist_sq = dx * dx + dy * dy
    cos_q2 = (dist_sq - L1**2 - L2**2) / (2 * L1 * L2)
    if not -1.0 <= cos_q2 <= 1.0:
        return None  # target outside the reachable annulus
    sin_q2 = np.sqrt(1.0 - cos_q2**2) * (1.0 if elbow_up else -1.0)
    q2 = np.arctan2(sin_q2, cos_q2)
    q1 = np.arctan2(dy, dx) - np.arctan2(L2 * sin_q2, L1 + L2 * cos_q2)
    q1 = np.arctan2(np.sin(q1), np.cos(q1))  # wrap to [-pi, pi]
    if abs(q2) > 2.5 or abs(q1) > 3.14:
        return None  # violates the joint limits declared in the MJCF
    return np.array([q1, q2], dtype=np.float32)


class ScriptedExpert:
    """Interpolates from the current pose to the IK solution for the target.

    Motion is eased rather than stepped so the recorded trajectory resembles a
    demonstration instead of a setpoint jump, and a little noise is injected so
    repeated episodes are not identical -- BC on a set of identical trajectories
    learns nothing about the neighbourhood of the demonstrated behaviour.
    """

    def __init__(self, rng: np.random.Generator, *, noise: float = 0.02, horizon: int = 40):
        self.rng = rng
        self.noise = noise
        self.horizon = horizon
        self._goal: np.ndarray | None = None
        self._start: np.ndarray | None = None
        self._t = 0

    def reset(self, env: MujocoReachEnv) -> bool:
        target_xy = np.asarray(env.data.site("target").xpos[:2], dtype=np.float64)
        elbow_up = bool(self.rng.integers(2))  # vary the solution branch
        goal = solve_ik(target_xy, elbow_up=elbow_up)
        if goal is None:
            goal = solve_ik(target_xy, elbow_up=not elbow_up)
        if goal is None:
            return False
        self._goal = goal
        self._start = np.asarray(env.data.qpos[:2], dtype=np.float32).copy()
        self._t = 0
        # Vary pacing between episodes: real demos are not all the same length.
        self._horizon = int(self.horizon * self.rng.uniform(0.8, 1.3))
        return True

    def act(self, obs) -> np.ndarray:  # noqa: ANN001 - Observation
        assert self._goal is not None and self._start is not None
        self._t += 1
        alpha = min(self._t / self._horizon, 1.0)
        eased = 0.5 - 0.5 * np.cos(np.pi * alpha)  # smoothstep
        action = self._start + eased * (self._goal - self._start)
        return action + self.rng.normal(0.0, self.noise * (1.0 - eased), size=action.shape)


class HumanTeleop:
    """Keyboard or gamepad teleoperation via pygame. Requires a display."""

    def __init__(self, mode: str, action_dim: int, step: float = 0.04):
        import pygame

        self.pygame = pygame
        self.mode = mode
        self.step = step
        self.action_dim = action_dim
        pygame.init()
        self.screen = pygame.display.set_mode((320, 120))
        pygame.display.set_caption("JetSpace teleop - R restart, ESC quit")
        self.joystick = None
        if mode == "gamepad":
            pygame.joystick.init()
            if pygame.joystick.get_count() == 0:
                raise RuntimeError("No gamepad detected. Use --policy keyboard.")
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
        self._cmd = np.zeros(action_dim, dtype=np.float32)
        self.quit = False
        self.restart = False

    def reset(self, env: MujocoReachEnv) -> bool:
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

        delta = np.zeros(self.action_dim, dtype=np.float32)
        if self.mode == "gamepad" and self.joystick is not None:
            for i in range(min(self.action_dim, self.joystick.get_numaxes())):
                axis = self.joystick.get_axis(i)
                if abs(axis) > 0.15:  # deadzone
                    delta[i] = axis * self.step
        else:
            keys = pg.key.get_pressed()
            if self.action_dim >= 1:
                delta[0] = (keys[pg.K_RIGHT] - keys[pg.K_LEFT]) * self.step
            if self.action_dim >= 2:
                delta[1] = (keys[pg.K_UP] - keys[pg.K_DOWN]) * self.step

        self._cmd = self._cmd + delta
        return self._cmd

    def close(self) -> None:
        self.pygame.quit()


def collect(args: argparse.Namespace) -> int:
    env = MujocoReachEnv(image_size=args.image_size, max_steps=args.max_steps)
    rng = np.random.default_rng(args.seed)

    if args.policy == "scripted":
        policy = ScriptedExpert(rng)
    else:
        policy = HumanTeleop(args.policy, env.action_dim)

    writer = EpisodeWriter(
        args.out,
        task="reach",
        fps=env.control_hz,
        action_dim=env.action_dim,
        cameras=env.camera_names,
        image_size=args.image_size,
        extra_info={"policy": args.policy, "seed": args.seed},
    )
    print(f"Writing to {writer.root} (already contains {writer.num_episodes} episodes)")

    written = skipped = 0
    for ep in range(args.episodes):
        # Record the reset seed: without it the target position cannot be
        # reproduced, and an episode that cannot be replayed cannot be verified.
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
            buffer.add(
                pixels=obs.pixels,
                proprio=obs.proprio,
                action=action,
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
        # Only successful demos are kept. Balaguer & Carpin make the same choice:
        # the space of failures is far larger and less structured than the space
        # of successes, and imitation learns nothing useful from it.
        if not buffer.success[-1] and not args.keep_failures:
            skipped += 1
            print(f"  episode {ep}: FAILED, discarded")
            continue

        writer.write(buffer, metadata={"policy": args.policy, "seed": ep_seed})
        written += 1
        print(f"  episode {ep}: {len(buffer)} frames, success={buffer.success[-1]}")

    env.close()
    if hasattr(policy, "close"):
        policy.close()

    print(f"\nWrote {written} episodes, discarded {skipped}.")
    print(EpisodeDataset(args.out).summary())
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", choices=["scripted", "keyboard", "gamepad"], default="scripted")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--out", default="data/episodes/reach")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--max-steps", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keep-failures", action="store_true", help="also record unsuccessful episodes")
    p.add_argument("--summary-only", action="store_true", help="print dataset stats and exit")
    args = p.parse_args()

    if args.summary_only:
        print(EpisodeDataset(args.out).summary())
        return 0
    return collect(args)


if __name__ == "__main__":
    sys.exit(main())
