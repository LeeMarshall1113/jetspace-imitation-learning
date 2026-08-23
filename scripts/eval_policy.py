#!/usr/bin/env python3
"""Evaluate policies on the frozen evaluation set.

    python scripts/eval_policy.py --checkpoints checkpoints/bc_seed*.pt

Rolls each checkpoint over every seed in configs/eval_seeds.json and reports
success rate as mean +- std across checkpoints. Every milestone is quoted
against this exact seed set so numbers stay comparable (REQUIREMENTS.md).

It also verifies the eval seeds never appear in the training data. A leaked
evaluation set does not fail loudly -- it just reports a number that is too good
and cannot be reproduced later.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jetspace.data.episode import EpisodeDataset  # noqa: E402
from jetspace.envs.mujoco_env import MujocoReachEnv  # noqa: E402
from jetspace.policies.bc import BCPolicy, SimpleVisualEncoder  # noqa: E402
from jetspace.utils.device import get_device  # noqa: E402


def load_policy(path: Path, device: str) -> tuple[BCPolicy, np.ndarray, np.ndarray]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    policy = BCPolicy(SimpleVisualEncoder(), ckpt["proprio_dim"], ckpt["action_dim"]).to(device)
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    mean = np.asarray(ckpt["norm"]["proprio_mean"], dtype=np.float32)
    std = np.asarray(ckpt["norm"]["proprio_std"], dtype=np.float32)
    return policy, mean, std


def rollout(env, policy, mean, std, camera, seed, device) -> tuple[bool, float]:  # noqa: ANN001
    obs = env.reset(seed=seed)
    terminated = truncated = False
    best = float("inf")
    while not (terminated or truncated):
        action = policy.act(obs.pixels[camera], (obs.proprio - mean) / std, device=device)
        result = env.step(action)
        best = min(best, result.info["dist"])
        obs = result.obs
        terminated, truncated = result.terminated, result.truncated
        if terminated:
            return True, best
    return False, best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", default=["checkpoints/bc_seed*.pt"])
    ap.add_argument("--eval-seeds", default="configs/eval_seeds.json")
    ap.add_argument("--train-data", default="data/episodes/reach")
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    spec = json.loads(Path(args.eval_seeds).read_text())
    eval_seeds = spec["seeds"]

    # Leak check: a contaminated eval set reports an inflated number silently.
    try:
        train_seeds = {r.get("seed") for r in EpisodeDataset(args.train_data).records}
        overlap = train_seeds & set(eval_seeds)
        if overlap:
            print(f"ERROR: {len(overlap)} eval seeds appear in the training data: "
                  f"{sorted(overlap)[:5]}")
            return 1
        print(f"leak check: OK, 0 of {len(eval_seeds)} eval seeds appear in training data")
    except FileNotFoundError:
        print("leak check: SKIPPED (training data not found)")

    paths = sorted({p for pattern in args.checkpoints for p in glob.glob(pattern)})
    if not paths:
        print(f"No checkpoints matched {args.checkpoints}")
        return 1

    device = get_device(args.device)
    env = MujocoReachEnv(image_size=224, max_steps=args.max_steps)
    rates = []

    for path in paths:
        policy, mean, std = load_policy(Path(path), device)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        camera = ckpt.get("camera", "front")
        successes, dists = 0, []
        for seed in eval_seeds:
            ok, best = rollout(env, policy, mean, std, camera, seed, device)
            successes += ok
            dists.append(best)
        rate = successes / len(eval_seeds)
        rates.append(rate)
        print(f"{Path(path).name:24s} success {rate:6.1%}   "
              f"median closest approach {np.median(dists) * 100:.1f} cm")

    env.close()
    arr = np.array(rates)
    print(f"\n{len(arr)} checkpoint(s): success {arr.mean():.1%} +- {arr.std():.1%}")
    gate = 0.70
    verdict = "PASS" if arr.mean() >= gate else "BELOW GATE"
    print(f"M2 gate is {gate:.0%} on held-out targets -> {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
