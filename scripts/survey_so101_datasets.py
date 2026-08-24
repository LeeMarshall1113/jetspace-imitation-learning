#!/usr/bin/env python3
"""Survey public SO-101 datasets for ones that can support the N1b ladder.

    python scripts/survey_so101_datasets.py --limit 120

N1's first run was invalidated because its viewpoint control used a wrist
camera against a top camera -- a different sensing modality, not a viewpoint
variant. N1b requires SCENE-level cameras everywhere, so the first thing needed
is a census of which public datasets actually have them.

Reads meta/info.json only, a few kilobytes per dataset. Classifies each camera
as wrist-mounted or scene-level by name, and reports how many scene cameras
each dataset carries, since a dataset with two is the only kind that can
provide a within-dataset viewpoint reference.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request

# Wrist/gripper-mounted cameras move with the arm and frame a close-up. They are
# not a viewpoint variant of a static scene camera and must never stand in as
# one -- that substitution is what invalidated the first ladder.
WRIST_HINTS = ("wrist", "gripper", "hand", "eye_in_hand", "eih", "arm")


def is_wrist(cam: str) -> bool:
    c = cam.lower().rsplit(".", 1)[-1]
    return any(h in c for h in WRIST_HINTS)


def search(term: str, limit: int) -> list[dict]:
    q = urllib.parse.quote(term)
    url = f"https://huggingface.co/api/datasets?search={q}&limit={limit}&full=false"
    try:
        return json.load(urllib.request.urlopen(url, timeout=30))
    except Exception:  # noqa: BLE001
        return []


def probe(repo: str) -> dict | None:
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/meta/info.json"
    try:
        info = json.load(urllib.request.urlopen(url, timeout=30))
    except Exception:  # noqa: BLE001
        return None
    feats = info.get("features", {})
    if "action" not in feats:
        return None
    cams = [k for k in feats if "image" in k]
    scene = [c for c in cams if not is_wrist(c)]
    return {
        "repo": repo,
        "version": str(info.get("codebase_version", "v2.1")),
        "robot": info.get("robot_type", "?"),
        "episodes": info.get("total_episodes", 0),
        "frames": info.get("total_frames", 0),
        "fps": info.get("fps", 0),
        "adim": feats["action"].get("shape", [0])[0],
        "cameras": cams,
        "scene": scene,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--min-episodes", type=int, default=8)
    args = ap.parse_args()

    seen: dict[str, int] = {}
    for term in ("so101", "so-101", "so100", "lerobot so101"):
        for d in search(term, args.limit):
            seen[d["id"]] = d.get("downloads", 0)
    print(f"{len(seen)} candidates from search; probing...\n", file=sys.stderr)

    rows = []
    for repo in sorted(seen, key=lambda k: -seen[k]):
        r = probe(repo)
        if not r or r["adim"] != 6 or r["episodes"] < args.min_episodes:
            continue
        rows.append(r)

    multi = [r for r in rows if len(r["scene"]) >= 2]
    single = [r for r in rows if len(r["scene"]) == 1]

    print(f"=== {len(multi)} datasets with TWO OR MORE scene cameras ===")
    print("(these can supply a within-dataset viewpoint reference)\n")
    print(f"{'repo':52s} {'ver':5s} {'eps':>5} {'frames':>8} {'fps':>4}  scene cameras")
    print("-" * 120)
    for r in sorted(multi, key=lambda x: -x["episodes"]):
        cams = ", ".join(c.rsplit(".", 1)[-1] for c in r["scene"])
        print(f"{r['repo'][:52]:52s} {r['version']:5s} {r['episodes']:>5} "
              f"{r['frames']:>8} {r['fps']:>4}  {cams}")

    print(f"\n=== {len(single)} datasets with exactly one scene camera ===")
    print("(usable as ladder rungs, cannot supply a viewpoint reference)\n")
    print(f"{'repo':52s} {'ver':5s} {'eps':>5} {'frames':>8} {'fps':>4}  scene camera")
    print("-" * 120)
    for r in sorted(single, key=lambda x: -x["episodes"])[:25]:
        cams = ", ".join(c.rsplit(".", 1)[-1] for c in r["scene"])
        print(f"{r['repo'][:52]:52s} {r['version']:5s} {r['episodes']:>5} "
              f"{r['frames']:>8} {r['fps']:>4}  {cams}")

    print(f"\ntotal usable: {len(rows)}  "
          f"(6-DoF, >= {args.min_episodes} episodes, at least one scene camera)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
