#!/usr/bin/env python3
"""Train the M2 behavior-cloning baseline.

    python scripts/train_bc.py --seed 0
    python scripts/train_bc.py --seed 1
    python scripts/train_bc.py --seed 2

Run it once per seed; REQUIREMENTS.md requires results reported as mean and
standard deviation over three training seeds, because a single-seed number on a
dataset this small is noise, not a result.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jetspace.data.torch_data import BCFrameDataset  # noqa: E402
from jetspace.policies.bc import BCPolicy, SimpleVisualEncoder  # noqa: E402
from jetspace.utils.device import describe, get_device  # noqa: E402


def episode_split(dataset: BCFrameDataset, val_frac: float, rng: np.random.Generator):
    """Split by EPISODE, never by frame.

    Consecutive frames within an episode are nearly identical, so a random frame
    split leaks almost every validation frame into training and reports a
    validation loss that means nothing.
    """
    lengths = [r["length"] for r in dataset.source.records]
    bounds, start = [], 0
    for n in lengths:
        bounds.append((start, start + n))
        start += n

    order = rng.permutation(len(bounds))
    n_val = max(1, int(round(val_frac * len(bounds))))
    val_eps, train_eps = order[:n_val], order[n_val:]
    idx = lambda eps: [i for e in eps for i in range(*bounds[e])]  # noqa: E731
    return idx(train_eps), idx(val_eps), len(train_eps), len(val_eps)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/episodes/so101_reach")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = get_device(args.device)
    print(f"device: {describe()}")

    ds = BCFrameDataset(args.data)
    tr_idx, va_idx, n_tr_ep, n_va_ep = episode_split(ds, args.val_frac, rng)
    print(f"dataset: {len(ds)} frames from {len(ds.source)} episodes")
    print(f"  train {len(tr_idx)} frames / {n_tr_ep} episodes")
    print(f"  val   {len(va_idx)} frames / {n_va_ep} episodes  (split by episode)")

    tr = DataLoader(Subset(ds, tr_idx), batch_size=args.batch_size, shuffle=True, num_workers=2)
    va = DataLoader(Subset(ds, va_idx), batch_size=args.batch_size, num_workers=2)

    policy = BCPolicy(SimpleVisualEncoder(), ds.proprio_dim, ds.action_dim).to(device)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"policy: {n_params/1e6:.2f}M parameters")

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.MSELoss()

    best_val, best_state = float("inf"), None
    t0 = time.time()
    for epoch in range(args.epochs):
        policy.train()
        tr_loss = 0.0
        for px, pr, ac in tr:
            px, pr, ac = px.to(device), pr.to(device), ac.to(device)
            loss = loss_fn(policy(px, pr), ac)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(ac)
        tr_loss /= len(tr_idx)

        policy.eval()
        va_loss = 0.0
        with torch.no_grad():
            for px, pr, ac in va:
                px, pr, ac = px.to(device), pr.to(device), ac.to(device)
                va_loss += loss_fn(policy(px, pr), ac).item() * len(ac)
        va_loss /= max(len(va_idx), 1)
        sched.step()

        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:3d}  train {tr_loss:.5f}  val {va_loss:.5f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"bc_seed{args.seed}.pt"
    torch.save(
        {
            "state_dict": best_state,
            "proprio_dim": ds.proprio_dim,
            "action_dim": ds.action_dim,
            "norm": ds.norm_stats(),
            "config": vars(args),
            "best_val_loss": best_val,
            "camera": ds.camera,
        },
        path,
    )
    print(f"\nbest val loss {best_val:.5f} in {time.time() - t0:.1f}s -> {path}")
    print(json.dumps({"seed": args.seed, "best_val_loss": best_val}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
