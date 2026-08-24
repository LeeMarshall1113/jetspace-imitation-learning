#!/usr/bin/env bash
# What is cached, how big, and how fast did it encode?
cd "$(dirname "$0")/.."
python3 - <<'PY'
import glob, json, os
rows = []
for p in sorted(glob.glob("cache/latents/*/info.json")):
    d = os.path.dirname(p)
    try:
        i = json.load(open(p))
    except Exception:
        continue
    fs = glob.glob(f"{d}/episode_*.npy")
    if not fs:
        continue
    mb = sum(os.path.getsize(f) for f in fs) / 1e6
    rows.append((os.path.basename(d), len(fs), i.get("chunk"), i.get("margin"),
                 i.get("pool_grid"), i.get("total_frames"), i.get("encode_seconds"), mb))
print(f"{'cache':26s} {'eps':>4} {'chunk':>6} {'marg':>5} {'pool':>5} "
      f"{'frames':>8} {'sec':>8} {'fps':>7} {'MB':>7}")
for n, e, c, m, pg, fr, sec, mb in rows:
    fps = (fr / sec) if (fr and sec) else None
    print(f"{n:26s} {e:>4} {str(c):>6} {str(m):>5} {str(pg):>5} "
          f"{str(fr):>8} {(f'{sec:.0f}' if sec else '-'):>8} "
          f"{(f'{fps:.2f}' if fps else '-'):>7} {mb:>7.0f}")
PY
