#!/usr/bin/env python3
"""E9: does seeing other TASKS reduce the demos a new task needs?

    python scripts/e9_task_transfer.py --shots 1 2 4 --seeds 0 1 2

This is the question the project was started to answer, stated as an
experiment: transfer a skill from a small dataset to a variety of areas. E8
answered it for camera viewpoints, which is a narrower axis than intended.
Here "area" means TASK.

Eight real laboratories, eight different tasks -- stacking cubes, a ball, tape,
a cup, a bin, a pen and mug. Leave one out, train on the other seven, then give
the model K episodes of the held-out task and ask how much better it does than
a model that saw only those K episodes.

**Zero-shot is not the question and is not claimed.** No model can perform an
unseen manipulation task with no demonstrations of it; there is no goal
conditioning here and nothing tells the policy what the new task is. The
claim under test is few-shot: whether having seen other tasks makes a new one
cheaper to learn. That is what "small dataset, broad transfer" means
operationally.

Three arms per fold:

  scratch    head trained on K target episodes only -- the honest baseline
  transfer   head pretrained on 7 source tasks, fine-tuned on the same K
  zeroshot   pretrained head applied to the target with no adaptation, which
             exists to show the floor, not to be competitive

Crossed with two encoders, because if random convolutional features transfer
between tasks as well as V-JEPA does, pretraining is not what is carrying it.

**Action normalisation uses the K adaptation episodes only.** Using the
evaluation episodes' statistics would leak the answer, and using a global scale
across labs would assume a shared action convention that ledger L8 showed does
not exist. With K=1 those statistics are noisy, which is a real property of
having one demonstration rather than a flaw in the measurement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jetspace.utils.device import get_device  # noqa: E402

TASKS = {
    "A_cubes": "A_cubes__ego",
    "B_svla": "B_svla__side",
    "C_tape": "C_tape__birdEye",
    "D_ball": "D_ball__front",
    "E_summer": "E_summer__front",
    "F_cup": "F_cup__cam_front",
    "G_bin": "G_bin__front",
    "H_penmug": "H_penmug1__camera_2",
}


class Head(nn.Module):
    def __init__(self, dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, z):
        return self.net(z)


def load_episodes(latent_dir: str, data_dir: str):
    """Per-episode (latents, actions), aligned. Latent t covers frames 2t,2t+1."""
    lf = sorted(Path("cache/latents", latent_dir).glob("episode_*.npy"))
    af = sorted(Path("data/episodes", data_dir).glob("episode_*.npz"))
    if not lf or not af:
        return []
    out = []
    for a, b in zip(lf, af):
        z = np.load(a).astype(np.float32)
        z = z.reshape(z.shape[0], -1)
        act = np.load(b)["action"].astype(np.float32)
        n = min(len(z), len(act) // 2)
        if n >= 4:
            out.append((z[:n], act[: 2 * n : 2]))
    return out


def fit(head, X, Y, device, epochs, lr, seed):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    lf = nn.MSELoss()
    Xt = torch.from_numpy(X).float().to(device)
    Yt = torch.from_numpy(Y).float().to(device)
    n = len(Xt)
    for _ in range(epochs):
        head.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 256):
            b = perm[i:i + 256]
            opt.zero_grad()
            lf(head(Xt[b]), Yt[b]).backward()
            opt.step()
    head.eval()
    return head


@torch.no_grad()
def score(head, X, Y, device):
    Xt = torch.from_numpy(X).float().to(device)
    Yt = torch.from_numpy(Y).float().to(device)
    return float(((head(Xt) - Yt) ** 2).mean().item())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="n1b", help="n1b or n1bcnn")
    ap.add_argument("--shots", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--pca-dim", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--ft-epochs", type=int, default=40)
    ap.add_argument("--pca-fit-rows", type=int, default=2000,
                    help="rows subsampled to fit the PCA basis")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = args.out or f"cache/e9_{args.prefix}.json"
    device = get_device("auto")

    print("=" * 76)
    print(f"E9 -- few-shot transfer to UNSEEN TASKS ({args.prefix})")
    print("=" * 76)

    data = {}
    for tag, name in TASKS.items():
        eps = load_episodes(f"{args.prefix}_{name}", f"n1b_{name}")
        if eps:
            data[tag] = eps
    print(f"  {len(data)}/8 tasks loaded: {sorted(data)}\n")
    if len(data) < 4:
        print("need at least 4 tasks; cache the missing encoders first")
        return 1

    rows = []
    for target in sorted(data):
        sources = [t for t in sorted(data) if t != target]
        # PCA on SOURCE tasks only -- the target is unseen at pretraining time,
        # so fitting the basis on it would be the leak this experiment exists
        # to avoid.
        src_raw = np.concatenate([z for t in sources for z, _ in data[t]])
        mu = src_raw.mean(0)
        k = min(args.pca_dim, src_raw.shape[0] - 1)
        # Fit the basis on a random subsample of rows. np.linalg.svd on an
        # (m, n) matrix with m < n costs O(m^2 n), and m here is ~8800 pooled
        # source latents against n = 16384 -- about 500x the cost of E8's
        # 398-row fit, which put the first attempt at over two hours of pure
        # SVD. 2000 rows is ample for a 128-dimensional basis and the mean is
        # still taken over everything.
        fit_rows = src_raw
        if len(src_raw) > args.pca_fit_rows:
            sel = np.random.default_rng(0).choice(
                len(src_raw), args.pca_fit_rows, replace=False)
            fit_rows = src_raw[sel]
        _, _, vt = np.linalg.svd(fit_rows - mu, full_matrices=False)
        basis = vt[:k].T
        sd = ((src_raw - mu) @ basis).std(0) + 1e-6

        def proj(z):
            return ((z - mu) @ basis) / sd

        Xs = np.concatenate([proj(z) for t in sources for z, _ in data[t]])
        Ys = []
        for t in sources:
            ta = np.concatenate([a for _, a in data[t]])
            tm, ts = ta.mean(0), ta.std(0) + 1e-6
            Ys.append(np.concatenate([(a - tm) / ts for _, a in data[t]]))
        Ys = np.concatenate(Ys)

        for seed in args.seeds:
            # Pretrain once per (target, seed). It does not depend on K, and
            # rebuilding it inside the K loop repeated the most expensive step
            # in the script three times for an identical result.
            pre = fit(Head(k, Ys.shape[1]).to(device), Xs, Ys,
                      device, args.epochs, 1e-3, seed)
            for K in args.shots:
                rng = np.random.default_rng(seed)
                eps = data[target]
                order = rng.permutation(len(eps))
                adapt = [eps[i] for i in order[:K]]
                evalep = [eps[i] for i in order[K:]]
                if not evalep:
                    continue

                # Action scale from the adaptation episodes only.
                ay = np.concatenate([a for _, a in adapt])
                amu, asd = ay.mean(0), ay.std(0) + 1e-6

                Xa = np.concatenate([proj(z) for z, _ in adapt])
                Ya = np.concatenate([(a - amu) / asd for _, a in adapt])
                Xe = np.concatenate([proj(z) for z, _ in evalep])
                Ye = np.concatenate([(a - amu) / asd for _, a in evalep])

                scratch = fit(Head(k, Ya.shape[1]).to(device), Xa, Ya,
                              device, args.epochs, 1e-3, seed)
                zs = score(pre, Xe, Ye, device)
                # Fine-tune a copy at a lower rate: the point is to adapt the
                # pretrained solution, not to overwrite it from K episodes.
                tr = Head(k, Ya.shape[1]).to(device)
                tr.load_state_dict(pre.state_dict())
                tr = fit(tr, Xa, Ya, device, args.ft_epochs, 3e-4, seed)

                rows.append({
                    "target": target, "K": K, "seed": seed,
                    "scratch": score(scratch, Xe, Ye, device),
                    "transfer": score(tr, Xe, Ye, device),
                    "zeroshot": zs,
                    "n_eval": int(len(Xe)),
                })
        done = [r for r in rows if r["target"] == target]
        print(f"  {target:10s} folds {len(done):2d}  "
              f"scratch {np.mean([r['scratch'] for r in done]):.3f}  "
              f"transfer {np.mean([r['transfer'] for r in done]):.3f}",
              flush=True)

    print("\n" + "=" * 76)
    print("FEW-SHOT TRANSFER TO AN UNSEEN TASK")
    print("=" * 76)
    print(f"  {'K':>3} {'scratch':>18} {'transfer':>18} {'zero-shot':>12} {'gain':>8}")
    summary = {}
    for K in args.shots:
        sel = [r for r in rows if r["K"] == K]
        if not sel:
            continue
        s = np.array([r["scratch"] for r in sel])
        t = np.array([r["transfer"] for r in sel])
        z = np.array([r["zeroshot"] for r in sel])
        summary[K] = {"scratch": float(s.mean()), "transfer": float(t.mean()),
                      "zeroshot": float(z.mean()),
                      "gain": float(s.mean() - t.mean()),
                      "wins": int((t < s).sum()), "n": len(sel)}
        print(f"  {K:>3} {s.mean():10.3f} +-{s.std():.3f} "
              f"{t.mean():10.3f} +-{t.std():.3f} {z.mean():12.3f} "
              f"{s.mean() - t.mean():+8.3f}")
        print(f"      transfer wins {int((t < s).sum())}/{len(sel)} folds")

    print("\n  1.0 = no better than predicting the mean action of the K "
          "adaptation episodes")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(
        {"prefix": args.prefix, "rows": rows, "summary": summary}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
