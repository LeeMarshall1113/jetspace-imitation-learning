#!/usr/bin/env python3
"""Train the action-conditioned latent predictor on cached latents.

    python scripts/train_predictor.py --task pickplace --seed 0

Reads latents, never pixels — the encoder ran once and its outputs are on disk.
This is what makes the whole M3/M4 programme cheap.

Training uses **multi-step rollout loss**, not single-step teacher forcing. A
model trained only to predict one step ahead is never asked to consume its own
output, so it is never penalised for errors that compound — and compounding is
the entire quantity E3 exists to measure. V-JEPA 2-AC makes the same choice for
the same reason.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jetspace.data.episode import EpisodeDataset  # noqa: E402
from jetspace.models.predictor import ActionConditionedPredictor  # noqa: E402
from jetspace.utils.device import describe, get_device  # noqa: E402


def load_pairs(task: str, data: Path, lat_dir: Path, horizon: int):
    """Build (z_t, a_t..a_t+h, z_t+1..z_t+h) windows from cached episodes."""
    ds = EpisodeDataset(data)
    info = json.loads((lat_dir / "info.json").read_text())
    fpl = info["frames_per_latent"]

    zs, acts, targets, episodes = [], [], [], 0
    for i in range(len(ds)):
        f = lat_dir / f"episode_{ds.records[i]['index']:06d}.npy"
        if not f.exists():
            continue
        z = np.load(f).astype(np.float32)               # (T', g, g, hidden)
        z = z.reshape(z.shape[0], -1, z.shape[-1])      # (T', tokens, hidden)
        ep = ds[i]
        raw_a = ep["action"].astype(np.float32)         # (T, adim) absolute targets
        qpos = ep["proprio"].astype(np.float32)[:, : raw_a.shape[1]]

        # Condition on the commanded DISPLACEMENT, not the absolute joint target.
        #
        # Measured, and this is the reason E3's first run produced an
        # action-blind world model: absolute targets are ~94% "where the arm
        # already is", and the arm's configuration is plainly visible in z_t. So
        # the action carries almost no information the observation does not
        # already have, and a shuffled action from another episode is still a
        # plausible-looking joint configuration. The model correctly learned to
        # ignore it -- shuffled actions scored within 1% of correct ones.
        #
        # This is ledger L3 for the third time, at a third level of the stack.
        delta = raw_a - qpos

        # One latent step spans `fpl` control steps, so sum the displacements
        # commanded across that window rather than discarding half of them.
        usable = (len(delta) // fpl) * fpl
        a = delta[:usable].reshape(-1, fpl, delta.shape[1]).sum(axis=1)[: len(z)]
        if len(z) < horizon + 1 or len(a) < horizon + 1:
            continue
        for t in range(len(z) - horizon):
            zs.append(z[t])
            acts.append(a[t : t + horizon])
            targets.append(z[t + 1 : t + 1 + horizon])
        episodes += 1

    if not zs:
        raise RuntimeError(f"No usable windows in {lat_dir} at horizon {horizon}")
    return (
        torch.from_numpy(np.stack(zs)),
        torch.from_numpy(np.stack(acts)),
        torch.from_numpy(np.stack(targets)),
        episodes,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="pickplace", choices=["reach", "push", "pickplace"])
    ap.add_argument("--data", default=None)
    ap.add_argument("--latents", default=None)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--horizon", type=int, default=4, help="training rollout length")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--pca-dim", type=int, default=0,
                    help="predict in a PCA subspace of this size; 0 uses raw latents")
    args = ap.parse_args()

    data = Path(args.data or f"data/episodes/{args.task}")
    lat_dir = Path(args.latents or f"cache/latents/{args.task}")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = get_device("auto")
    print(f"device: {describe()}")

    z0, acts, tgt, n_ep = load_pairs(args.task, data, lat_dir, args.horizon)
    n, tokens, hidden = z0.shape
    adim = acts.shape[-1]
    print(f"{n} windows from {n_ep} episodes  |  tokens {tokens}  hidden {hidden}  "
          f"action {adim}  horizon {args.horizon}")

    # Normalise: raw V-JEPA activations are not zero-centred and their scale is
    # arbitrary, so an unnormalised MSE is dominated by whichever channels
    # happen to be large.
    mu, sd = z0.mean((0, 1)), z0.std((0, 1)) + 1e-6
    z0n, tgtn = (z0 - mu) / sd, (tgt - mu) / sd

    basis = None
    if args.pca_dim > 0:
        # Predict in a PCA subspace rather than raw latent space.
        #
        # Measured: |z_t+1 - z_t| ~ 160 while |z_t+16 - z_t| ~ 110. The latent
        # moves further between adjacent frames than it drifts over sixteen, so
        # frame-to-frame change is dominated by high-frequency variation that no
        # model can predict. Forward prediction in that space spends its capacity
        # on noise -- which is why the world model looked action-blind while a
        # linear inverse-dynamics probe recovered the action at R^2 up to 0.74.
        #
        # Projecting to the top components keeps the structure and drops the
        # part that is not there to be learned.
        flat = z0n.reshape(-1, hidden)
        sub = flat[:: max(1, len(flat) // 20000)]
        _, S, Vt = torch.linalg.svd(sub - sub.mean(0), full_matrices=False)
        basis = Vt[: args.pca_dim].T.contiguous()
        kept = (S[: args.pca_dim] ** 2).sum() / (S**2).sum()
        print(f"PCA: {hidden} -> {args.pca_dim} dims, {kept:.1%} of variance kept")
        z0n = z0n @ basis
        tgtn = tgtn @ basis
        hidden = args.pca_dim

    idx = rng.permutation(n)
    n_val = max(1, int(args.val_frac * n))
    va, tr = idx[:n_val], idx[n_val:]

    grid = int(round(tokens**0.5))
    model = ActionConditionedPredictor(hidden=hidden, grid=grid, action_dim=adim).to(device)
    print(f"predictor: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.MSELoss()

    # The do-nothing baseline: predict z_t+k = z_t for every k. Any model that
    # cannot beat this has learned nothing, and reporting it alongside is the
    # standing lesson from ledger L3.
    with torch.no_grad():
        static = loss_fn(z0n[va].unsqueeze(1).expand_as(tgtn[va]), tgtn[va]).item()
    print(f"baseline (predict no change): {static:.5f}")

    best, best_state = float("inf"), None
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(tr))
        total = 0.0
        for b in range(0, len(tr), args.batch_size):
            sel = tr[perm[b : b + args.batch_size].numpy()]
            zb, ab, tb = z0n[sel].to(device), acts[sel].to(device), tgtn[sel].to(device)
            z = zb
            loss = 0.0
            for h in range(args.horizon):
                z = model(z, ab[:, h])          # feed prediction back in
                loss = loss + loss_fn(z, tb[:, h])
            loss = loss / args.horizon
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(sel)
        sched.step()

        model.eval()
        with torch.no_grad():
            zb, ab, tb = z0n[va].to(device), acts[va].to(device), tgtn[va].to(device)
            z, vloss = zb, 0.0
            for h in range(args.horizon):
                z = model(z, ab[:, h])
                vloss += loss_fn(z, tb[:, h]).item()
            vloss /= args.horizon
        if vloss < best:
            best = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:3d}  train {total/len(tr):.5f}  val {vloss:.5f}"
                  f"   ({static/max(vloss,1e-9):.2f}x better than do-nothing)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"predictor_{args.task}_seed{args.seed}.pt"
    torch.save(
        {
            "state_dict": best_state,
            "hidden": hidden, "grid": grid, "action_dim": adim,
            "norm": {"mu": mu.numpy().tolist(), "sd": sd.numpy().tolist()},
            "pca_basis": basis.numpy().tolist() if basis is not None else None,
            "config": vars(args),
            "val_loss": best, "static_baseline": static,
        },
        path,
    )
    print(f"\nbest val {best:.5f} vs do-nothing {static:.5f} "
          f"({static/max(best,1e-9):.2f}x) in {time.time()-t0:.1f}s -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
