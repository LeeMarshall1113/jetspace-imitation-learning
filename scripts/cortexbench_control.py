#!/usr/bin/env python3
"""Does CortexBench discriminate trained encoders from random features?

    python scripts/cortexbench_control.py --task pen-v0 --episodes 25

E12 found that five of eight nuisance axes in THIS project's benchmark cannot
separate a trained encoder from an untrained CNN. The obvious objection is that
those axes are ours: maybe JetSpace's axes are broken and the field's are fine.

This runs the same control on CortexBench (Majumdar et al., NeurIPS 2023), the
benchmark the field actually uses to rank frozen visual encoders for embodied
AI, on its own published demonstration data.

Protocol, matched to E12 so the two are comparable:

  * frozen encoder, 4x4 spatial pooling, no fine-tuning
  * ridge from features to the demonstrated action, split BY EPISODE
  * an untrained CNN carried through as a control arm
  * report where that control ranks

Scope, stated plainly: CortexBench's headline metric is rollout success after
behaviour cloning, not probe error. This is the probe-level analogue -- the
same frozen-features-rank-encoders question the benchmark is used to answer,
measured the way E12 measures it. A competitive random arm here does not prove
their rollout rankings are noise; it does show that this data, under the
frozen-feature protocol, cannot by itself separate learned representations
from untrained ones.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import os
# Host path by default; inside the container the dataset is bind-mounted,
# so CORTEXBENCH_DATA overrides it rather than the path being hardcoded twice.
DATA = Path(os.environ.get("CORTEXBENCH_DATA",
                          "/home/lee-m/cortexbench-data/adroit-expert-v1.0"))


def load_task(task: str, n_ep: int):
    """(list of image arrays, list of action arrays) for the first n episodes."""
    with open(DATA / f"{task}.pickle", "rb") as fh:
        eps = pickle.load(fh)
    imgs, acts = [], []
    for e in eps[:n_ep]:
        imgs.append(np.asarray(e["images"]))
        acts.append(np.asarray(e["actions"], dtype=np.float32))
    return imgs, acts


def ridge(Xtr, Ytr, Xte, lam: float = 10.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    d = Xtr.shape[1]
    if d <= len(Xtr):
        W = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(d), Xtr.T @ Ytr)
    else:
        K = Xtr @ Xtr.T + lam * np.eye(len(Xtr))
        W = Xtr.T @ np.linalg.solve(K, Ytr)
    return Xte @ W


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="pen-v0")
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--models", default="aimv2,vjepa2-proxy,dinov2,clip,siglip2,"
                                        "vit-in1k,vc1,convnext,RANDOM")
    ap.add_argument("--pool-grid", type=int, default=4)
    ap.add_argument("--out", default="cache/cortexbench_pen.json")
    a = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    import torch.nn.functional as F                     # noqa: E402
    from cache_latents_hf import ALIASES, build, encode   # noqa: E402
    # The same untrained CNN E12 carries as its control, so "random" means the
    # identical network in both experiments. It takes (T,3,224,224) and emits
    # the pooled grid directly, so it gets its own path rather than being
    # forced through the transformers processor wrapper.
    from jetspace.models.cnn_encoder import ScratchCNNEncoder  # noqa: E402

    @torch.no_grad()
    def encode_random(frames, device, grid, seed=0, batch=32):
        torch.manual_seed(seed)
        net = ScratchCNNEncoder(hidden=1024, grid=grid,
                                frames_per_latent=1).to(device).eval()
        out = []
        for i in range(0, len(frames), batch):
            x = torch.from_numpy(
                np.asarray(frames[i:i + batch], dtype=np.float32) / 255.0
            ).permute(0, 3, 1, 2).to(device)
            if x.shape[-1] != 224:
                x = F.interpolate(x, size=(224, 224), mode="bilinear",
                                  align_corners=False)
            out.append(net(x).float().cpu().numpy())
        return np.concatenate(out)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    imgs, acts = load_task(a.task, a.episodes)
    n_ep = len(imgs)
    print(f"{a.task}: {n_ep} episodes, {imgs[0].shape[0]} frames each, "
          f"action dim {acts[0].shape[1]}, device {device}")

    cut = max(1, int(0.8 * n_ep))
    rows = {}
    for name in [m.strip() for m in a.models.split(",") if m.strip()]:
        try:
            if name == "RANDOM":
                feats = [encode_random(im, device, a.pool_grid) for im in imgs]
            else:
                # build() wants a resolved HF id; the short-name mapping
                # lives in ALIASES and is applied in that script's main().
                proc, model = build(ALIASES.get(name, name), device)
                feats = [encode(im, proc, model, device, a.pool_grid, 1)
                         for im in imgs]
        except Exception as e:                             # noqa: BLE001
            print(f"  {name:14s} SKIPPED ({type(e).__name__}: {e})"[:110])
            continue

        X = [f.reshape(f.shape[0], -1).astype(np.float32) for f in feats]
        Y = [c[:len(x)] for c, x in zip(acts, X)]
        Xtr, Ytr = np.concatenate(X[:cut]), np.concatenate(Y[:cut])
        Xte, Yte = np.concatenate(X[cut:]), np.concatenate(Y[cut:])
        ay, asd = Ytr.mean(0), Ytr.std(0) + 1e-6
        live = Ytr.std(0) > 1e-6
        Ytr_n, Yte_n = ((Ytr - ay) / asd)[:, live], ((Yte - ay) / asd)[:, live]
        pred = ridge(Xtr, Ytr_n, Xte)
        ss_res = float(((Yte_n - pred) ** 2).sum())
        ss_tot = float(((Yte_n - Yte_n.mean(0)) ** 2).sum())
        rows[name] = {"probe": 1.0 - ss_res / max(ss_tot, 1e-12),
                      "mse": float(((Yte_n - pred) ** 2).mean())}
        print(f"  {name:14s} probe R2 {rows[name]['probe']:+.3f}  "
              f"mse {rows[name]['mse']:.3f}")

    if "RANDOM" not in rows or len(rows) < 4:
        print("\ntoo few arms to judge")
        return 1
    order = sorted(rows, key=lambda k: -rows[k]["probe"])
    pos = order.index("RANDOM") + 1
    n = len(order)
    print(f"\n{'=' * 62}\nDISCRIMINABILITY CONTROL on CortexBench/{a.task}")
    print(f"  ranking by probe R2: {' > '.join(order)}")
    print(f"  untrained control ranks {pos}/{n}")
    if pos <= 2 * n / 3.0:
        print("  FAILS the E12d criterion: random features are not in the")
        print("  bottom third. On this data, under the frozen-feature protocol,")
        print("  the benchmark does not separate learned from untrained")
        print("  representations.")
    else:
        print("  PASSES: random features rank in the bottom third, so this")
        print("  task does discriminate. JetSpace's failing axes are not")
        print("  representative of the field's benchmark.")

    import json
    Path("cache").mkdir(exist_ok=True)
    Path(a.out).write_text(json.dumps(
        {"task": a.task, "episodes": n_ep, "rows": rows,
         "random_rank": pos, "n_arms": n}, indent=1))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
