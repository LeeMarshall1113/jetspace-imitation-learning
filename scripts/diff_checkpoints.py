#!/usr/bin/env python3
"""Print the non-weight configuration of predictor checkpoints side by side.

    python scripts/diff_checkpoints.py checkpoints/a.pt checkpoints/b.pt

Written because two runs that should have matched did not. A predictor
retrained from scratch on a freshly encoded, identically configured 60-episode
cache scored direction cosine 0.692, while the original checkpoint on the
original cache scored 0.902. Same task, same episode count, same chunk and
margin -- so the gap belongs to something recorded in the checkpoint rather
than to the encoding, and attributing it to the window comb without checking
would have been a guess presented as a diagnosis.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


def main() -> int:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        print(__doc__)
        return 1

    rows = []
    for p in paths:
        if not p.exists():
            print(f"{p}: missing")
            continue
        c = torch.load(p, map_location="cpu", weights_only=False)
        cfg = {k: v for k, v in c.items()
               if k not in ("state_dict", "norm", "pca_basis")}
        pb = c.get("pca_basis")
        cfg["pca_basis"] = None if pb is None else tuple(getattr(pb, "shape", (len(pb),)))
        norm = c.get("norm", {})
        for k in ("mu", "sd", "a_mu", "a_sd"):
            v = norm.get(k)
            if v is not None:
                arr = torch.as_tensor(v).flatten()
                cfg[f"norm.{k}"] = f"n={arr.numel()} mean={arr.float().mean():.4f}"
        n_params = sum(v.numel() for v in c["state_dict"].values())
        cfg["n_params"] = n_params
        rows.append((p.name, cfg))

    if not rows:
        return 1

    keys = []
    for _, cfg in rows:
        for k in cfg:
            if k not in keys:
                keys.append(k)

    w = max(len(k) for k in keys) + 2
    print(f"{'':{w}}" + "".join(f"{n[:34]:>36}" for n, _ in rows))
    print("-" * (w + 36 * len(rows)))
    for k in keys:
        vals = [str(cfg.get(k, "-")) for _, cfg in rows]
        flag = "  <-- DIFFERS" if len(set(vals)) > 1 else ""
        print(f"{k:{w}}" + "".join(f"{v[:34]:>36}" for v in vals) + flag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
