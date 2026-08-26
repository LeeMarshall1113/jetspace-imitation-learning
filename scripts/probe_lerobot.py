#!/usr/bin/env python3
"""Report what a public LeRobot dataset actually contains, before importing it.

    python scripts/probe_lerobot.py qb1t/so101_teleop_cubes bjb7/so101_pen_mug

Reads only meta/info.json and meta/episodes.jsonl, so it costs a few kilobytes
per dataset instead of gigabytes. Used to apply the real-vs-real control
selection rule in docs/prereg-n1.md without downloading candidates first --
the rule is fixed in advance, and this only checks which datasets can satisfy
it mechanically (right robot, right action dimension, enough episodes).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def probe(repo: str) -> dict | None:
    from huggingface_hub import hf_hub_download

    try:
        info = json.loads(
            Path(hf_hub_download(repo, "meta/info.json", repo_type="dataset")).read_text()
        )
    except Exception as exc:  # noqa: BLE001
        return {"repo": repo, "error": f"{type(exc).__name__}: {exc}"[:70]}

    feats = info.get("features", {})
    cams = [k for k in feats if "image" in k]
    task = None
    try:
        p = hf_hub_download(repo, "meta/episodes.jsonl", repo_type="dataset")
        rows = [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
        if rows:
            task = rows[0].get("tasks")
    except Exception:  # noqa: BLE001
        pass

    return {
        "repo": repo,
        "robot": info.get("robot_type"),
        "episodes": info.get("total_episodes"),
        "frames": info.get("total_frames"),
        "fps": info.get("fps"),
        "action_dim": feats.get("action", {}).get("shape", [None])[0],
        "names": feats.get("action", {}).get("names"),
        "cameras": cams,
        "task": task,
    }


def main() -> int:
    repos = sys.argv[1:]
    if not repos:
        print(__doc__)
        return 1

    for r in repos:
        d = probe(r)
        if d is None or "error" in d:
            print(f"\n{r}\n   UNAVAILABLE  {d.get('error') if d else ''}")
            continue
        print(f"\n{d['repo']}")
        print(f"   robot     {d['robot']}   fps {d['fps']}   action_dim {d['action_dim']}")
        print(f"   episodes  {d['episodes']}   frames {d['frames']}")
        print(f"   cameras   {d['cameras']}")
        print(f"   task      {d['task']}")
        names = d.get("names")
        if names:
            flat = names if isinstance(names, list) else names.get("motors", names)
            print(f"   joints    {flat}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
