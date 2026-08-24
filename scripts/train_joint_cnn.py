#!/usr/bin/env python3
"""Train a CNN encoder and the world model together — E6 arm 3.

    python scripts/train_joint_cnn.py --task push --epochs 20

Arms 1 and 2 use frozen encoders (V-JEPA, and a random CNN). This is the arm
where the encoder is free to move, which is both the point and the hazard.

**Why this arm needs watching.** Nothing in a prediction loss forbids the
encoder collapsing its own representation. If every latent becomes the same
vector, the predictor's job is trivial, the loss goes to zero, and the world
model is worthless. A frozen encoder cannot do this; a trained one can, and it
is the cheapest available minimum. So a raw validation-loss comparison hands
this arm the win by construction, no matter which representation is better.

The loss is therefore reported against **the do-nothing baseline computed in
this encoder's own latent space**. Collapse shrinks the baseline exactly as
much as it shrinks the model error, so the *ratio* stays honest where the raw
number does not. Two further checks live downstream and both run on the cached
latents afterwards: the inverse-dynamics probe (a collapsed space cannot
recover which action was taken) and the shuffled-action test.

`collapse_ratio` — the mean per-dimension standard deviation across the
batch, which is exactly the quantity VICReg regularises — is logged every
epoch. Near 1.0 is healthy against a target of 1.0; near 0 means the dimensions
carry no variation and the space has collapsed.

**It is not hypothetical.** The first smoke run collapsed inside one epoch:

    epoch 0  val 0.00053  gain 0.52x  collapse_ratio 0.0004
    epoch 1  val 0.00011  gain 0.70x  collapse_ratio 0.0004

A validation loss of 1e-4 is roughly a thousand times below V-JEPA's, so a raw
loss comparison would have declared the CNN the winner by three orders of
magnitude. The gain ratio said 0.70x — worse than predicting no change at all,
because in a collapsed space predicting no change is already almost perfect.

**Which makes the unregularised arm a strawman.** Nobody deploying a jointly
trained encoder would ship one that collapsed; they would add a variance term
and move on. So `--var-reg` supplies the standard VICReg hinge, penalising any
latent dimension whose spread across the batch falls under a target. Both
settings are run: the unregularised arm shows the trap is real, the regularised
arm is the honest competitor V-JEPA has to beat.
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
from jetspace.models.cnn_encoder import ScratchCNNEncoder  # noqa: E402
from jetspace.models.predictor import ActionConditionedPredictor  # noqa: E402
from jetspace.utils.device import get_device  # noqa: E402


def build_windows(ds: EpisodeDataset, camera: str, fpl: int, horizon: int,
                  limit: int | None):
    """Frame windows plus the summed actions spanning each latent step.

    Actions are the delta between commanded position and measured position,
    summed over the `fpl` frames a latent covers -- identical to the treatment
    in train_predictor.py. Any mismatch here silently trains the model on
    actions that do not correspond to the transition being predicted, which is
    a defect this project has already hit twice at other points in the stack.
    """
    frames, acts, idx = [], [], []
    n = min(limit or len(ds), len(ds))
    for i in range(n):
        ep = ds[i]
        vid = ep[f"pixels_{camera}"]
        raw_a = ep["action"].astype(np.float32)
        qpos = ep["proprio"].astype(np.float32)[:, : raw_a.shape[1]]
        d = raw_a - qpos
        usable = (min(len(vid), len(d)) // fpl) * fpl
        if usable < (horizon + 2) * fpl:
            continue
        a = d[:usable].reshape(-1, fpl, d.shape[1]).sum(axis=1)
        v = vid[:usable]
        base = len(frames)
        frames.append(v)
        acts.append(a)
        n_lat = usable // fpl
        for t in range(n_lat - horizon - 1):
            idx.append((len(frames) - 1, t))
        _ = base
    if not idx:
        raise RuntimeError("no usable windows")
    return frames, acts, idx


def variance_hinge(z: torch.Tensor, target: float = 1.0) -> torch.Tensor:
    """VICReg's variance term: penalise dimensions whose batch spread collapses.

    Hinged rather than maximised, so it stops pushing once a dimension is
    healthy and never fights the prediction objective for its own sake.
    Bardes, Ponce & LeCun, ICLR 2022.
    """
    f = z.reshape(-1, z.shape[-1])
    std = torch.sqrt(f.var(dim=0) + 1e-6)
    return torch.relu(target - std).mean()


def collapse_ratio(z: torch.Tensor) -> float:
    """Mean per-dimension spread across the batch. Falls to 0 on collapse.

    The first version of this measured between-latent spread over within-latent
    spread, and it was wrong in a way worth recording rather than quietly
    fixing.

    Its numerator was the standard deviation, across latents, of each latent's
    mean over 16384 dimensions. Averaging that many values makes the per-latent
    mean nearly constant whatever the encoder is doing, so the numerator stayed
    tiny for healthy and collapsed encoders alike. Worse, the window it saw
    spans five latents -- 0.4 seconds -- during which a real arm barely moves,
    so genuinely distinct latents SHOULD look similar there. It reported
    "COLLAPSING" for both the naive and the regularised arm, which is exactly
    the uninformative behaviour of a metric that is measuring the wrong thing.

    This measures what collapse actually is: whether individual dimensions vary
    at all across the sample. It is also the quantity VICReg regularises, so
    with `--var-reg` and a target of 1.0 a healthy encoder should sit near 1.0
    and a collapsed one near 0 -- directly comparable rather than needing
    interpretation.
    """
    f = z.reshape(-1, z.shape[-1])
    if len(f) < 2:
        return float("nan")
    return float(f.std(dim=0).mean().item())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="push")
    ap.add_argument("--data", default=None)
    ap.add_argument("--camera", default=None)
    ap.add_argument("--out", default="checkpoints/e6")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--pool-grid", type=int, default=4)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--frames-per-latent", type=int, default=2)
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--var-reg", type=float, default=0.0,
                    help="VICReg variance-hinge weight. 0 reproduces the naive "
                         "arm, which collapses inside one epoch; ~1.0 gives the "
                         "regularised arm that is a fair competitor.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device("auto")
    data = Path(args.data or f"data/episodes/{args.task}")
    ds = EpisodeDataset(data)
    camera = args.camera or ds.info["cameras"][0]
    fpl = args.frames_per_latent

    frames, acts, idx = build_windows(ds, camera, fpl, args.horizon, args.limit)
    adim = acts[0].shape[1]
    print(f"{data}: {len(frames)} episodes, {len(idx)} windows, "
          f"action {adim}, horizon {args.horizon}")

    enc = ScratchCNNEncoder(hidden=args.hidden, grid=args.pool_grid,
                            frames_per_latent=fpl, width=args.width).to(device)
    # grid is the SIDE length; the predictor squares it into n_tokens.
    pred = ActionConditionedPredictor(
        hidden=args.hidden, grid=args.pool_grid, action_dim=adim
    ).to(device)
    n_enc = sum(p.numel() for p in enc.parameters())
    n_pred = sum(p.numel() for p in pred.parameters())
    print(f"encoder {n_enc/1e6:.1f}M  predictor {n_pred/1e6:.1f}M  (both training)")

    all_a = np.concatenate(acts, axis=0)
    a_mu = torch.tensor(all_a.mean(0), device=device)
    a_sd = torch.tensor(all_a.std(0) + 1e-6, device=device)

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(idx))
    split = int(0.85 * len(order))
    tr, va = order[:split], order[split:]

    opt = torch.optim.AdamW(
        list(enc.parameters()) + list(pred.parameters()), lr=args.lr, weight_decay=1e-4
    )
    loss_fn = nn.MSELoss()

    def encode_window(ep_i: int, t: int, length: int) -> torch.Tensor:
        v = frames[ep_i][t * fpl : (t + length) * fpl]
        v = torch.from_numpy(np.ascontiguousarray(v)).to(device).float() / 255.0
        v = v.reshape(-1, fpl, *v.shape[1:])
        v = v.permute(0, 1, 4, 2, 3).reshape(v.shape[0], fpl * 3, *v.shape[2:4])
        return enc(v)

    def batch_loss(sel, train: bool):
        total, cr = 0.0, 0.0
        for j in sel:
            ep_i, t = idx[j]
            z = encode_window(ep_i, t, args.horizon + 1)      # (H+1, g, g, C)
            z = z.reshape(z.shape[0], -1, z.shape[-1])         # (H+1, g*g, C)
            a = torch.from_numpy(acts[ep_i][t : t + args.horizon]).to(device)
            a = (a - a_mu) / a_sd

            cur, l = z[:1], 0.0
            for h in range(args.horizon):
                cur = pred(cur, a[h : h + 1])
                l = l + loss_fn(cur, z[h + 1 : h + 2])
            l = l / args.horizon
            # Applied to the ENCODER's output, not the prediction: the thing
            # being prevented is the representation collapsing, not the
            # predictor's output distribution narrowing.
            if args.var_reg > 0:
                l = l + args.var_reg * variance_hinge(z)
            total = total + l
            cr += collapse_ratio(z.detach())
        return total / len(sel), cr / len(sel)

    best, t0 = float("inf"), time.time()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "reg" if args.var_reg > 0 else "naive"
    ckpt_path = out_dir / f"joint_{args.task}_{tag}_seed{args.seed}.pt"

    for epoch in range(args.epochs):
        enc.train()
        pred.train()
        perm = rng.permutation(tr)
        run = 0.0
        for i in range(0, len(perm), args.batch):
            sel = perm[i : i + args.batch]
            loss, _ = batch_loss(sel, True)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(enc.parameters()) + list(pred.parameters()), 1.0
            )
            opt.step()
            run += loss.item() * len(sel)

        enc.eval()
        pred.eval()
        with torch.no_grad():
            vsel = va[: min(len(va), 128)]
            vloss, cr = batch_loss(vsel, False)
            vloss = vloss.item()

            # The do-nothing baseline in THIS encoder's own space. Collapse
            # shrinks it exactly as much as it shrinks the model error, which
            # is why the ratio is the honest number and the raw loss is not.
            static = 0.0
            for j in vsel:
                ep_i, t = idx[j]
                z = encode_window(ep_i, t, args.horizon + 1)
                z = z.reshape(z.shape[0], -1, z.shape[-1])
                static += loss_fn(
                    z[:1].expand_as(z[1:]), z[1:]
                ).item()
            static /= len(vsel)

        if vloss < best:
            best = vloss
            torch.save({
                "encoder": enc.state_dict(),
                "state_dict": pred.state_dict(),
                "encoder_config": {"hidden": args.hidden, "grid": args.pool_grid,
                                   "frames_per_latent": fpl, "width": args.width},
                "hidden": args.hidden, "grid": args.pool_grid,
                "action_dim": adim,
                "norm": {"mu": [0.0], "sd": [1.0],
                         "a_mu": a_mu.cpu().numpy().tolist(),
                         "a_sd": a_sd.cpu().numpy().tolist()},
                "val_loss": best, "static_baseline": static,
                "collapse_ratio": cr,
                "config": vars(args),
            }, ckpt_path)

        if epoch % 2 == 0 or epoch == args.epochs - 1:
            gain = static / max(vloss, 1e-9)
            # Against the VICReg target of 1.0; below ~0.1 the dimensions
            # carry essentially no variation and the space has collapsed.
            warn = "  <-- COLLAPSING" if cr < 0.1 else ""
            print(f"  epoch {epoch:3d}  train {run/len(tr):.5f}  val {vloss:.5f}  "
                  f"gain {gain:.2f}x  collapse_ratio {cr:.4f}{warn}")

    el = time.time() - t0
    print(f"\nbest val {best:.5f} vs do-nothing {static:.5f} "
          f"({static/max(best,1e-9):.2f}x) in {el:.1f}s -> {ckpt_path}")
    print("\nThe gain ratio is the comparable number, not the raw loss: a")
    print("collapsed encoder drives BOTH toward zero. Confirm with the")
    print("inverse-dynamics probe on the cached latents before believing it.")

    (out_dir / f"joint_{args.task}_{tag}_seed{args.seed}.json").write_text(json.dumps({
        "task": args.task, "val_loss": best, "static_baseline": static,
        "gain": static / max(best, 1e-9), "collapse_ratio": cr,
        "encoder_params": int(n_enc), "predictor_params": int(n_pred),
        "epochs": args.epochs, "seed": args.seed, "var_reg": args.var_reg,
        "collapsed": bool(cr < 0.1),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
