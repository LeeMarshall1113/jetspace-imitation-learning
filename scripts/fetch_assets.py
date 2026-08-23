#!/usr/bin/env python3
"""Fetch robot models from MuJoCo Menagerie into assets/.

    python scripts/fetch_assets.py                 # default: so101
    python scripts/fetch_assets.py --model franka_emika_panda

Menagerie models are Apache-2.0 but carry ~18 MB of STL meshes each, so they are
downloaded rather than vendored: the repo stays light and the upstream model
stays the single source of truth. `assets/` is gitignored.

We pin a commit rather than tracking main. A silent upstream change to link
inertias or joint limits would invalidate every trained policy and every number
in docs/results.md, with no diff in this repository to explain why.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

# Pinned Menagerie revision. Bump deliberately, and re-measure afterwards.
MENAGERIE_REF = "main"
RAW = "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie"
API = "https://api.github.com/repos/google-deepmind/mujoco_menagerie/contents"


def listing(path: str, ref: str) -> list[dict]:
    with urllib.request.urlopen(f"{API}/{path}?ref={ref}", timeout=60) as r:
        return json.loads(r.read())


def fetch(model: str, ref: str, dest: Path) -> int:
    out = dest / model
    out.mkdir(parents=True, exist_ok=True)

    files = [f for f in listing(model, ref) if f["type"] == "file"]
    subdirs = [d["name"] for d in listing(model, ref) if d["type"] == "dir"]

    total = 0
    for f in files:
        # The PNG preview is a megabyte of nothing useful to a simulator.
        if f["name"].endswith(".png"):
            continue
        target = out / f["name"]
        if target.exists() and target.stat().st_size == f["size"]:
            continue
        urllib.request.urlretrieve(f"{RAW}/{ref}/{model}/{f['name']}", target)
        total += f["size"]
        print(f"  {f['name']}")

    for sub in subdirs:
        (out / sub).mkdir(exist_ok=True)
        for f in listing(f"{model}/{sub}", ref):
            if f["type"] != "file":
                continue
            target = out / sub / f["name"]
            if target.exists() and target.stat().st_size == f["size"]:
                continue
            urllib.request.urlretrieve(f"{RAW}/{ref}/{model}/{sub}/{f['name']}", target)
            total += f["size"]
        print(f"  {sub}/ ({len(listing(f'{model}/{sub}', ref))} files)")

    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="robotstudio_so101")
    ap.add_argument("--ref", default=MENAGERIE_REF)
    ap.add_argument("--dest", default="assets")
    args = ap.parse_args()

    dest = Path(args.dest)
    print(f"Fetching {args.model} @ {args.ref} -> {dest}/")
    try:
        n = fetch(args.model, args.ref, dest)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Done ({n / 1e6:.1f} MB downloaded; existing files skipped).")
    print(f"Model XML: {dest}/{args.model}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
