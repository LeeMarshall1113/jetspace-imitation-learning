#!/usr/bin/env python3
"""Import a real-robot LeRobot dataset into our episode format.

    python scripts/fetch_lerobot.py --repo qb1t/so101_teleop_cubes --episodes 20

This is decision D1 finally cashing in: source demonstrations from public data
rather than recording our own. It also moves the project off sim-only data,
which is the single most common objection to a robotics paper.

Why the SO-101 dataset family specifically: these are **SO-101 follower**
recordings, the same arm whose MuJoCo model we simulate, with the same six
joints in the same order -- `shoulder_pan`, `shoulder_lift`, `elbow_flex`,
`wrist_flex`, `wrist_roll`, `gripper`.

**Nominally identical is not verified identical.** Labs differ in joint naming
(`shoulder_pan.pos` versus `main_shoulder_pan`), which indicates different
LeRobot versions and calibration procedures, and units are not guaranteed to
match. Anything depending on action-space interchangeability must verify it
first -- see docs/novelty-upgrade.md B1.

**Two on-disk formats, both supported.**

  v2.1  one parquet and one mp4 per episode, plus meta/episodes.jsonl:
          data/chunk-000/episode_000000.parquet
          videos/chunk-000/{camera}/episode_000000.mp4

  v3.0  episodes concatenated into shared files, boundaries recorded in
        meta/episodes/**.parquet instead of a jsonl:
          data/chunk-000/file-000.parquet          (many episodes)
          videos/{camera}/chunk-000/file-000.mp4   (many episodes)
        Note the video path puts the camera BEFORE the chunk, the reverse of
        v2.1. Assuming v2.1 fails with a bare 404 on meta/episodes.jsonl that
        says nothing about why.

**What does NOT transfer, stated plainly:**

  * There is no simulator behind real episodes, so `verify_replay` cannot check
    them. Determinism was our strongest data-integrity gate and it does not
    exist here. Real data is trusted, not verified.
  * There are no reset seeds, for the same reason.
  * Actions are recorded joint positions from a leader arm, not commands we
    issued, so the action/observation relationship is subtly different from our
    scripted collection.

Each is a real limitation and each belongs in the paper rather than a footnote.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jetspace.data.episode import EpisodeBuffer, EpisodeDataset, EpisodeWriter  # noqa: E402


def _jsonable(v):
    """Coerce parquet cell values into something json.dump accepts.

    pandas hands back a list-valued column as a numpy ndarray, and the v3.0
    `tasks` column is list-valued. Writing it straight into episode metadata
    raised `Object of type ndarray is not JSON serializable` -- after the frames
    had already been decoded, so the work was done and then thrown away.
    """
    if isinstance(v, np.ndarray):
        return [_jsonable(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    return v


def _episode_table_v30(repo: str):
    """Concatenate every meta/episodes/**.parquet into one episode index."""
    import pandas as pd
    from huggingface_hub import HfApi, hf_hub_download

    files = sorted(
        f for f in HfApi().list_repo_files(repo, repo_type="dataset")
        if f.startswith("meta/episodes/") and f.endswith(".parquet")
    )
    if not files:
        raise SystemExit(f"{repo}: v3.0 layout but no meta/episodes/*.parquet")
    frames = [
        pd.read_parquet(hf_hub_download(repo, f, repo_type="dataset")) for f in files
    ]
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values("episode_index").reset_index(drop=True)


def _load_v30(repo: str, info: dict, camera: str, n: int):
    """Yield (episode_index, actions, states, frames_factory, task) for v3.0."""
    import imageio.v2 as imageio
    import pandas as pd
    from huggingface_hub import hf_hub_download

    eps = _episode_table_v30(repo)
    fps = info["fps"]
    n = min(n, len(eps))
    data_cache: dict[str, pd.DataFrame] = {}

    for i in range(n):
        row = eps.iloc[i]
        dpath = info["data_path"].format(
            chunk_index=int(row["data/chunk_index"]),
            file_index=int(row["data/file_index"]),
        )
        if dpath not in data_cache:
            data_cache[dpath] = pd.read_parquet(
                hf_hub_download(repo, dpath, repo_type="dataset")
            )
        df = data_cache[dpath]
        # Select by episode_index rather than by the global dataset_from/to
        # offsets: the parquet carries the column, and using it avoids assuming
        # anything about where a shard's global numbering starts.
        ep = df[df["episode_index"] == int(row["episode_index"])]
        if ep.empty:
            print(f"  episode {i}: no rows in {dpath}")
            continue

        vpath = info["video_path"].format(
            video_key=camera,
            chunk_index=int(row[f"videos/{camera}/chunk_index"]),
            file_index=int(row[f"videos/{camera}/file_index"]),
        )
        try:
            vfile = hf_hub_download(repo, vpath, repo_type="dataset")
        except Exception as exc:  # noqa: BLE001
            print(f"  episode {i}: video unavailable ({type(exc).__name__})")
            continue

        start = int(round(float(row[f"videos/{camera}/from_timestamp"]) * fps))
        count = int(row["length"])

        def frames(vfile=vfile, start=start, count=count):
            r = imageio.get_reader(vfile)
            try:
                for t, frame in enumerate(r):
                    if t < start:
                        continue
                    if t >= start + count:
                        break
                    yield t - start, frame
            finally:
                r.close()

        yield (
            int(row["episode_index"]),
            np.stack(ep["action"].to_numpy()).astype(np.float64),
            np.stack(ep["observation.state"].to_numpy()).astype(np.float32),
            frames,
            row.get("tasks"),
        )


def _load_v21(repo: str, info: dict, camera: str, n: int):
    """Yield (episode_index, actions, states, frames_factory, task) for v2.1."""
    import imageio.v2 as imageio
    import pandas as pd
    from huggingface_hub import hf_hub_download

    tasks: dict[int, object] = {}
    try:
        p = hf_hub_download(repo, "meta/episodes.jsonl", repo_type="dataset")
        for line in Path(p).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                tasks[r.get("episode_index", len(tasks))] = r.get("tasks")
    except Exception:  # noqa: BLE001
        pass

    n = min(n, info["total_episodes"])
    for i in range(n):
        try:
            pq = hf_hub_download(
                repo, f"data/chunk-000/episode_{i:06d}.parquet", repo_type="dataset"
            )
            vid = hf_hub_download(
                repo, f"videos/chunk-000/{camera}/episode_{i:06d}.mp4",
                repo_type="dataset",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  episode {i}: unavailable ({type(exc).__name__})")
            continue

        df = pd.read_parquet(pq)

        def frames(vid=vid):
            r = imageio.get_reader(vid)
            try:
                for t, frame in enumerate(r):
                    yield t, frame
            finally:
                r.close()

        yield (
            i,
            np.stack(df["action"].to_numpy()).astype(np.float64),
            np.stack(df["observation.state"].to_numpy()).astype(np.float32),
            frames,
            tasks.get(i),
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="qb1t/so101_teleop_cubes")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--camera", default=None, help="defaults to the first camera in meta")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth frame")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    name = args.repo.split("/")[-1]
    out = Path(args.out or f"data/episodes/real_{name}")

    info = json.loads(
        Path(hf_hub_download(args.repo, "meta/info.json", repo_type="dataset")).read_text()
    )
    version = str(info.get("codebase_version", "v2.1"))
    cams = [k for k in info["features"] if "image" in k]
    camera = args.camera or cams[0]
    if camera not in cams:
        raise SystemExit(f"camera {camera!r} not in {cams}")
    fps = info["fps"] // args.stride
    adim = info["features"]["action"]["shape"][0]

    print(f"{args.repo}: {info['total_episodes']} episodes, {info['total_frames']} frames, "
          f"{info['fps']} fps  [{version}]")
    print(f"  robot   {info.get('robot_type')}")
    print(f"  cameras {cams}  -> using {camera!r}")
    print(f"  action  {info['features']['action'].get('names')}\n")

    loader = _load_v30 if version.startswith("v3") else _load_v21

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
            "lerobot_camera": camera, "codebase_version": version,
            # Flagged in metadata, not just prose: nothing downstream should
            # quietly treat these as replay-verified.
            "replay_verifiable": False,
        },
    )

    cam_short = camera.split(".")[-1]
    written = 0
    for idx, actions, states, frames, task in loader(args.repo, info, camera, args.episodes):
        buf = EpisodeBuffer()
        kept = 0
        for t, frame in frames():
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

        if kept == 0:
            print(f"  episode {idx}: no frames decoded")
            continue
        writer.write(buf, metadata={"source_episode": int(idx),
                                    "task_text": _jsonable(task)})
        written += 1
        print(f"  episode {idx}: {kept} frames")

    if written == 0:
        print("\nNOTHING WRITTEN. Check the camera name and the layout version.")
        return 1

    print(f"\nwrote {written} episodes to {out}")
    print(EpisodeDataset(out).summary())
    print("\nNOTE: real episodes cannot be replay-verified — there is no simulator "
          "to replay them in. `replay_verifiable: false` is recorded in info.json.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
