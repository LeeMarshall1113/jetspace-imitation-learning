#!/usr/bin/env python3
"""Import a real-robot LeRobot dataset into our episode format.

    python scripts/fetch_lerobot.py --repo qb1t/so101_teleop_cubes --episodes 20

This is decision D1 finally cashing in: source demonstrations from public data
rather than recording our own. It also moves the project off sim-only data,
which is the single most common objection to a robotics paper.

Why this dataset family specifically: these are **SO-101 follower** recordings,
the same arm whose MuJoCo model we simulate, with an identical 6-DoF action
space and identical joint names — `shoulder_pan`, `shoulder_lift`,
`elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`. Nothing has to be
remapped, so the predictor architecture, the latent cache and the horizon
evaluation all run unchanged on real video.

**What does NOT transfer, stated plainly:**

  * There is no simulator behind real episodes, so `verify_replay` cannot check
    them. Determinism was our strongest data-integrity gate and it does not
    exist here. Real data is trusted, not verified.
  * There are no reset seeds, for the same reason.
  * Actions are recorded joint positions from a leader arm, not commands we
    issued, so the action/observation relationship is subtly different from our
    scripted collection.

Each is a real limitation and each belongs in the paper rather than in a
footnote.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jetspace.data.episode import EpisodeBuffer, EpisodeDataset, EpisodeWriter  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="qb1t/so101_teleop_cubes")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--camera", default=None, help="defaults to the first camera in meta")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth frame")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import imageio.v2 as imageio
    import pandas as pd
    from huggingface_hub import hf_hub_download

    name = args.repo.split("/")[-1]
    out = Path(args.out or f"data/episodes/real_{name}")

    info = json.loads(
        Path(hf_hub_download(args.repo, "meta/info.json", repo_type="dataset")).read_text()
    )
    cams = [k for k in info["features"] if "image" in k]
    camera = args.camera or cams[0]
    fps = info["fps"] // args.stride
    adim = info["features"]["action"]["shape"][0]
    print(f"{args.repo}: {info['total_episodes']} episodes, {info['total_frames']} frames, "
          f"{info['fps']} fps")
    print(f"  robot   {info.get('robot_type')}")
    print(f"  cameras {cams}  -> using {camera!r}")
    print(f"  action  {info['features']['action'].get('names')}")

    tasks_path = hf_hub_download(args.repo, "meta/episodes.jsonl", repo_type="dataset")
    eps_meta = [json.loads(x) for x in Path(tasks_path).read_text().splitlines() if x.strip()]
    print(f"  task    {eps_meta[0].get('tasks')}\n")

    writer = EpisodeWriter(
        out,
        task=f"real_{name}",
        fps=fps,
        action_dim=adim,
        cameras=(camera.split(".")[-1],),
        image_size=args.image_size,
        extra_info={
            "source": args.repo, "real_robot": True, "robot": info.get("robot_type"),
            "original_fps": info["fps"], "stride": args.stride,
            "lerobot_camera": camera,
            # Flagged in metadata, not just prose: nothing downstream should
            # quietly treat these as replay-verified.
            "replay_verifiable": False,
        },
    )

    cam_short = camera.split(".")[-1]
    n = min(args.episodes, info["total_episodes"])
    written = 0
    for i in range(n):
        try:
            pq = hf_hub_download(
                args.repo, f"data/chunk-000/episode_{i:06d}.parquet", repo_type="dataset"
            )
            vid = hf_hub_download(
                args.repo, f"videos/chunk-000/{camera}/episode_{i:06d}.mp4", repo_type="dataset"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  episode {i}: unavailable ({type(exc).__name__})")
            continue

        df = pd.read_parquet(pq)
        actions = np.stack(df["action"].to_numpy()).astype(np.float64)
        states = np.stack(df["observation.state"].to_numpy()).astype(np.float32)

        reader = imageio.get_reader(vid)
        buf = EpisodeBuffer()
        kept = 0
        for t, frame in enumerate(reader):
            if t % args.stride or t >= len(actions):
                continue
            img = np.asarray(frame)
            if img.shape[0] != args.image_size or img.shape[1] != args.image_size:
                from PIL import Image

                img = np.asarray(
                    Image.fromarray(img).resize(
                        (args.image_size, args.image_size), Image.BILINEAR
                    )
                )
            # Proprio is [state, velocity] in our format; real data has no
            # velocity channel, so it is filled with the finite difference.
            vel = (
                states[t] - states[t - args.stride]
                if t >= args.stride
                else np.zeros_like(states[t])
            )
            buf.add(
                pixels={cam_short: img},
                proprio=np.concatenate([states[t], vel]).astype(np.float32),
                action=actions[t],
                action_executed=actions[t],
                reward=0.0,
                # Every episode in a demonstration set is a success by
                # construction; there is no reward signal to derive one from.
                success=True,
            )
            kept += 1
        reader.close()

        if kept == 0:
            print(f"  episode {i}: no frames decoded")
            continue
        writer.write(buf, metadata={"source_episode": i, "task_text": eps_meta[i].get("tasks")})
        written += 1
        print(f"  episode {i}: {kept} frames")

    print(f"\nwrote {written} episodes to {out}")
    print(EpisodeDataset(out).summary())
    print("\nNOTE: real episodes cannot be replay-verified — there is no simulator "
          "to replay them in. `replay_verifiable: false` is recorded in info.json.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
