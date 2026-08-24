#!/usr/bin/env python3
"""Dump the metadata schema of a LeRobot v3.0 dataset.

    python scripts/inspect_v3_schema.py HashtagRobotics/tic-tac-toe-so101-block-a-clean-v1

v2.1 stores one parquet and one mp4 per episode plus a meta/episodes.jsonl.
v3.0 concatenates many episodes into shared files and keeps the boundaries in
meta/episodes/**.parquet instead. The column names for those boundaries are
what an importer needs and what this prints.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    repo = sys.argv[1] if len(sys.argv) > 1 else (
        "HashtagRobotics/tic-tac-toe-so101-block-a-clean-v1"
    )
    import pandas as pd
    from huggingface_hub import hf_hub_download

    info = json.loads(
        Path(hf_hub_download(repo, "meta/info.json", repo_type="dataset")).read_text()
    )
    print(f"{repo}")
    print(f"  codebase_version  {info.get('codebase_version')}")
    for k in ("total_episodes", "total_frames", "total_videos", "chunks_size",
              "data_files_size_in_mb", "video_files_size_in_mb", "fps"):
        if k in info:
            print(f"  {k:18s}{info[k]}")
    for k in ("data_path", "video_path"):
        if k in info:
            print(f"  {k:18s}{info[k]}")

    p = hf_hub_download(repo, "meta/episodes/chunk-000/file-000.parquet",
                        repo_type="dataset")
    df = pd.read_parquet(p)
    print(f"\nmeta/episodes parquet: {len(df)} rows")
    print(f"  columns: {list(df.columns)}")
    print("\n  first two rows:")
    for i in range(min(2, len(df))):
        for c in df.columns:
            v = df.iloc[i][c]
            s = str(v)
            print(f"    {c:38s} {s[:70]}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
