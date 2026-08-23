#!/usr/bin/env python3
"""Verify that recorded episodes replay deterministically.

    python scripts/verify_replay.py --data data/episodes/so101_reach

Replays each episode's recorded action sequence from its recorded seed and
compares the resulting proprioceptive trajectory against what was stored. This
is the "replay verified" half of the M1 exit gate: a dataset whose actions do
not reproduce its observations is not a usable demonstration set, and the defect
is invisible until something trains on it and quietly fails.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jetspace.data.episode import EpisodeDataset  # noqa: E402
from jetspace.envs.so101_env import SO101ReachEnv  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/episodes/so101_reach")
    ap.add_argument("--limit", type=int, default=0, help="check only the first N episodes")
    ap.add_argument("--tol", type=float, default=1e-4, help="max allowed proprio deviation")
    args = ap.parse_args()

    ds = EpisodeDataset(args.data)
    if len(ds) == 0:
        print(f"No episodes in {args.data}")
        return 1

    env = SO101ReachEnv(image_size=ds.info["image_size"], max_steps=10_000)
    n = len(ds) if args.limit <= 0 else min(args.limit, len(ds))

    worst = 0.0
    failures: list[str] = []
    no_seed = 0

    for i in range(n):
        ep = ds[i]
        seed = ep["meta"].get("seed")
        if seed is None:
            no_seed += 1
            continue

        env.reset(seed=int(seed))
        replayed = []
        for action in ep["action"]:
            result = env.step(action)
            replayed.append(result.obs.proprio)

        recorded = ep["proprio"][1:]  # obs[t+1] follows action[t]
        replay = np.stack(replayed)[: len(recorded)]
        if len(recorded) == 0:
            continue
        dev = float(np.abs(replay - recorded[: len(replay)]).max())
        worst = max(worst, dev)
        if dev > args.tol:
            failures.append(f"episode {ep['meta']['index']}: max deviation {dev:.2e}")

    env.close()

    print(f"Checked {n - no_seed}/{n} episodes in {args.data}")
    if no_seed:
        print(f"  {no_seed} skipped: no seed recorded (collected before seeds were stored)")
    print(f"  worst proprio deviation: {worst:.3e}  (tolerance {args.tol:.0e})")
    if failures:
        print(f"  FAILED: {len(failures)} episodes")
        for f in failures[:10]:
            print(f"    {f}")
        return 1
    if n - no_seed == 0:
        print("  nothing verified - re-collect so seeds are recorded")
        return 1
    print("  PASS: all replayed episodes match their recorded trajectories")
    return 0


if __name__ == "__main__":
    sys.exit(main())
