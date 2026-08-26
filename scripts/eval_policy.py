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
from jetspace.envs.registry import get_task  # noqa: E402
from jetspace.policies.bc import BCPolicy, SimpleVisualEncoder  # noqa: E402
from jetspace.utils.device import get_device  # noqa: E402


def load_policy(path: Path, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    # Rebuild the architecture the checkpoint recorded. Defaults match the
    # original so older checkpoints still load.
    enc = SimpleVisualEncoder(in_size=ckpt.get("in_size", 112),
                              stages=ckpt.get("stages", 3))
    policy = BCPolicy(enc, ckpt["proprio_dim"], ckpt["action_dim"]).to(device)
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    mean = np.asarray(ckpt["norm"]["proprio_mean"], dtype=np.float32)
    std = np.asarray(ckpt["norm"]["proprio_std"], dtype=np.float32)
    return policy, mean, std, ckpt["norm"], ckpt.get("camera", "front"), ckpt["action_dim"]


#: Per-task error terms, in preference order. A task whose info carries none
#: of these still reports success; only the auxiliary distance is lost.
ERROR_KEYS = ("dist", "goal_error", "place_error", "grasp_error")


def rollout(env, policy, mean, std, norm, camera, adim, seed, device) -> tuple[bool, float]:  # noqa: ANN001
    obs = env.reset(seed=seed)
    terminated = truncated = False
    best = float("inf")
    while not (terminated or truncated):
        action = policy.act(
            obs.pixels[camera], (obs.proprio - mean) / std, obs.proprio[:adim], norm, device=device
        )
        result = env.step(action)
        # Each env reports its own error term: reach "dist", push
        # "goal_error", pickplace its own. Hardcoding "dist" made this
        # script reach-only and it raised KeyError the first time it was
        # pointed at another task -- which had never happened, because
        # nothing had called it since M2.
        err = next((result.info[k] for k in ERROR_KEYS if k in result.info), None)
        if err is not None:
            best = min(best, float(err))
        obs = result.obs
        terminated, truncated = result.terminated, result.truncated
        if terminated:
            return True, best
    return False, best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", default=["checkpoints/bc_seed*.pt"])
    ap.add_argument("--eval-seeds", default="configs/eval_seeds.json")
    ap.add_argument("--task", default="pickplace", choices=["reach", "push", "pickplace"])
    ap.add_argument("--train-data", default=None)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--eval-limit", type=int, default=None,
                    help="use only the first N eval seeds. The full set is "
                         "100; a 23-pose sweep at 100 seeds x 3 checkpoints "
                         "is ~2M env steps. 30 keeps the standard error "
                         "near 9%% at p=0.5, which is enough to rank poses.")
    # Render the policy's observation from a DIFFERENT camera than the one it
    # trained on, feeding it through unchanged. The policy is not told; it just
    # receives a displaced view where it expects its own. That is the whole
    # test -- it turns "the latent gap is N" into "the policy succeeds X% of
    # the time at that gap".
    ap.add_argument("--camera-override", default=None,
                    help="evaluate under this camera instead of the training one")
    args = ap.parse_args()
    if args.train_data is None:
        args.train_data = f"data/episodes/{args.task}"

    spec = json.loads(Path(args.eval_seeds).read_text())
    eval_seeds = spec["seeds"]
    if args.eval_limit:
        eval_seeds = eval_seeds[: args.eval_limit]

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
    # Render ONLY the view the policy will be given. Rendering the training
    # camera alongside the override doubles the per-step cost and nothing reads
    # it; across 23 poses that is hours.
    cams = (args.camera_override,) if args.camera_override else ("front",)
    env = get_task(args.task)["env"](
        image_size=224, max_steps=args.max_steps, cameras=cams
    )
    rates = []

    for path in paths:
        policy, mean, std, norm, camera, adim = load_policy(Path(path), device)
        successes, dists = 0, []
        for seed in eval_seeds:
            view = args.camera_override or camera
            ok, best = rollout(env, policy, mean, std, norm, view, adim, seed, device)
            successes += ok
            dists.append(best)
        rate = successes / len(eval_seeds)
        rates.append(rate)
        print(f"{Path(path).name:24s} success {rate:6.1%}   "
              f"median closest approach {np.median(dists) * 100:.1f} cm"
              if np.isfinite(np.median(dists)) else "")

    env.close()
    arr = np.array(rates)
    view = args.camera_override or "front (training view)"
    print(f"\n{len(arr)} checkpoint(s): success {arr.mean():.1%} +- {arr.std():.1%}"
          f"   [camera: {view}]")
    gate = 0.70
    verdict = "PASS" if arr.mean() >= gate else "BELOW GATE"
    print(f"M2 gate is {gate:.0%} on held-out targets -> {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
