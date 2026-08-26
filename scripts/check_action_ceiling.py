#!/usr/bin/env python3
"""The control E10 needed first: can these latents predict actions AT ALL?

    python scripts/check_action_ceiling.py

E10 compared two ways of learning a new task and both landed above the
mean-action floor. Before concluding anything about transfer, the question is
whether the ceiling is high enough for a comparison to mean anything: train on
MOST of one task's episodes and test on held-out episodes of the SAME task, no
transfer involved, no few-shot constraint.

If a head with 80% of a task's data still cannot beat predicting the mean, then
latent-to-action prediction is not learnable on that dataset and every E10
number is measuring the ceiling rather than the treatment. That is the same
defect as R2's verdict off a 3.3%-success policy, arrived at from the other
direction.

Reported alongside the real laboratories, which reached 0.416 at K=4 in E9 and
so are known to be learnable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jetspace.utils.device import get_device  # noqa: E402

SETS = [
    ("push (sim)", "push", "push"),
    ("pickplace (sim)", "pickplace", "pickplace"),
    ("reach (sim)", "reach", "reach"),
    ("A_cubes (real)", "n1b_A_cubes__ego", "n1b_A_cubes__ego"),
    ("G_bin (real)", "n1b_G_bin__front", "n1b_G_bin__front"),
]


class Head(nn.Module):
    def __init__(self, dim: int, act: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, act))

    def forward(self, z):
        return self.net(z)


def load(lat: str, dat: str):
    lf = sorted((Path("cache/latents") / lat).glob("episode_*.npy"))
    af = sorted((Path("data/episodes") / dat).glob("episode_*.npz"))
    out = []
    for a, b in zip(lf, af):
        z = np.load(a).astype(np.float32)
        z = z.reshape(z.shape[0], -1)
        act = np.load(b)["action"].astype(np.float32)
        n = min(len(z), len(act) // 2)
        if n >= 4:
            out.append((z[:n], act[: 2 * n : 2]))
    return out


def run(label: str, lat: str, dat: str, device: str, dim: int = 128) -> None:
    eps = load(lat, dat)
    if len(eps) < 6:
        print(f"{label:20s} only {len(eps)} episodes")
        return
    rng = np.random.default_rng(0)
    order = rng.permutation(len(eps))
    cut = int(0.8 * len(eps))
    tr = [eps[i] for i in order[:cut]]
    te = [eps[i] for i in order[cut:]]

    Xtr_raw = np.concatenate([z for z, _ in tr])
    mu = Xtr_raw.mean(0)
    fit = Xtr_raw if len(Xtr_raw) <= 2000 else Xtr_raw[
        rng.choice(len(Xtr_raw), 2000, replace=False)]
    _, _, vt = np.linalg.svd(fit - mu, full_matrices=False)
    k = min(dim, vt.shape[0])
    basis = vt[:k].T
    sd = ((Xtr_raw - mu) @ basis).std(0) + 1e-6

    Ytr_raw = np.concatenate([a for _, a in tr])
    live = Ytr_raw.std(0) > 1e-6
    amu, asd = Ytr_raw.mean(0), Ytr_raw.std(0) + 1e-6

    def px(z):
        return ((z - mu) @ basis) / sd

    def py(a):
        return ((a - amu) / asd)[:, live]

    Xtr = np.concatenate([px(z) for z, _ in tr])
    Ytr = np.concatenate([py(a) for _, a in tr])
    Xte = np.concatenate([px(z) for z, _ in te])
    Yte = np.concatenate([py(a) for _, a in te])

    torch.manual_seed(0)
    head = Head(k, Ytr.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    lf = nn.MSELoss()
    Xt = torch.from_numpy(Xtr).float().to(device)
    Yt = torch.from_numpy(Ytr).float().to(device)
    for _ in range(60):
        head.train()
        perm = torch.randperm(len(Xt), device=device)
        for i in range(0, len(Xt), 256):
            b = perm[i:i + 256]
            opt.zero_grad()
            lf(head(Xt[b]), Yt[b]).backward()
            opt.step()
    head.eval()
    with torch.no_grad():
        pred = head(torch.from_numpy(Xte).float().to(device)).cpu().numpy()
    mse = float(((pred - Yte) ** 2).mean())
    # Floor: predicting the training mean, in the same standardised units.
    floor = float((Yte ** 2).mean())
    print(f"{label:20s} {len(tr):>4}/{len(te):<4} eps  "
          f"held-out MSE {mse:7.3f}   mean-predictor {floor:7.3f}   "
          f"{'LEARNABLE' if mse < 0.9 * floor else 'NOT LEARNABLE'}")


def main() -> int:
    device = get_device("auto")
    print("Train on 80% of ONE task, test on held-out episodes of the SAME "
          "task.\nNo transfer, no few-shot. This is the ceiling E10 was "
          "measuring against.\n")
    print(f"{'dataset':20s} {'train/test':>10}       {'result'}")
    print("-" * 78)
    for label, lat, dat in SETS:
        try:
            run(label, lat, dat, device)
        except Exception as exc:  # noqa: BLE001
            print(f"{label:20s} failed: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
