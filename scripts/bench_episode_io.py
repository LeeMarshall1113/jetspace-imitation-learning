#!/usr/bin/env python3
"""What does episode I/O actually cost?

    python scripts/bench_episode_io.py

Issue #10 proposes an LMDB storage layer that serialises episodes through
orjson. Before agreeing or disagreeing, measure: how large is an episode, how
long does it take to read, and is reading it anywhere near the critical path?

Opinions about storage layers are cheap. This prints the numbers that decide
whether the change would help.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/episodes/e12_push__ref")
    files = sorted(root.glob("episode_*.npz"))
    if not files:
        print(f"no episodes in {root}")
        return 1

    f = files[0]
    on_disk = f.stat().st_size

    t0 = time.time()
    d = np.load(f)
    keys = list(d.keys())
    arrays = {k: d[k] for k in keys}
    t_full = time.time() - t0

    t0 = time.time()
    d2 = np.load(f)
    _ = d2["action"]
    t_action = time.time() - t0

    raw = sum(a.nbytes for a in arrays.values())
    pix = {k: a for k, a in arrays.items() if k.startswith("pixels_")}

    print(f"episode: {f.name}")
    print(f"  on disk (npz, compressed) {on_disk / 1e6:8.2f} MB")
    print(f"  in memory (raw arrays)    {raw / 1e6:8.2f} MB   "
          f"({raw / max(on_disk, 1):.1f}x)")
    print(f"  full read                 {t_full * 1000:8.1f} ms")
    print(f"  action-only read          {t_action * 1000:8.1f} ms")
    print()
    print(f"  {'array':22s} {'shape':22s} {'MB':>8}")
    for k, a in sorted(arrays.items(), key=lambda kv: -kv[1].nbytes):
        print(f"  {k:22s} {str(a.shape):22s} {a.nbytes / 1e6:8.2f}")

    if pix:
        k, a = next(iter(pix.items()))
        n_int = a.size
        # orjson emits each uint8 as decimal text plus a separator: 2-4 bytes
        # per element against 1 byte raw. Numpy arrays are not JSON-native, so
        # they must become Python lists first, which is the expensive part.
        est_json = n_int * 3
        print()
        print(f"  If serialised through JSON, '{k}' alone:")
        print(f"    {n_int:,} integers -> roughly {est_json / 1e6:.0f} MB of text")
        print(f"    against {a.nbytes / 1e6:.1f} MB raw and "
              f"{on_disk / 1e6:.1f} MB for the whole compressed episode")

        t0 = time.time()
        _ = a[:4].tolist()          # the conversion any JSON encoder needs
        t_list = (time.time() - t0) * len(a) / 4
        print(f"    .tolist() on the full array: ~{t_list:.1f} s per episode")

    # Is reading anywhere near the critical path?
    total_read = 0.0
    for g in files[:5]:
        t0 = time.time()
        dd = np.load(g)
        for k in dd.keys():
            _ = dd[k]
        total_read += time.time() - t0
    per_ep = total_read / min(5, len(files))
    print()
    print(f"  mean full read over {min(5, len(files))} episodes: "
          f"{per_ep * 1000:.0f} ms")
    print(f"  a 10-episode condition therefore reads in ~{per_ep * 10:.1f} s")
    print()
    print("  For comparison, encoding those same 10 episodes takes minutes:")
    print("  V-JEPA 2 runs at ~5 fps and the image encoders at 20-40 fps, so a")
    print("  178-frame episode is 4-35 s of GPU per encoder. Reading is not the")
    print("  bottleneck and was not measured to be.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
