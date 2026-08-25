#!/usr/bin/env python3
"""E8: latent viewpoint canonicalization.

    python scripts/e8_canonicalize.py --task push --seeds 0 1 2

Registered in docs/prereg-e8.md, committed before this ran.

Learn `g: z_viewpoint -> z_reference` from paired simulated renders, pose-blind,
then ask whether a policy trained at ONE viewpoint recovers accuracy at
viewpoints it never saw once its observations are canonicalized first.

The R1 sweep renders one rollout from 23 cameras, so `z_p[t]` and `z_ref[t]` are
the same instant from different viewpoints. That pairing is free in simulation
and impossible on a real robot, which is what makes the correction worth
learning here and applying elsewhere.

Three arms, because two would not settle it:

  baseline      head trained on the reference pose, evaluated raw
  canonical     same head, evaluated on g(z) instead of z
  multiview     head trained on all 15 training poses directly

`multiview` is the registered falsifier (E8c). It is the baseline every
reviewer asks for, and if it matches `canonical` on held-out poses then `g` adds
nothing and this script says so.
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

from jetspace.envs.so101_env import R1_POSES, r1_displacement  # noqa: E402
from jetspace.utils.device import get_device  # noqa: E402

REF = "r1_ref"

#: Held out from every training batch of g and of every head. Chosen to span
#: the displacement range rather than to be easy: two large azimuths, extreme
#: elevation, both distance extremes, and two off-axis compounds.
HELD_OUT = ["r1_az20", "r1_az60", "r1_el25", "r1_el75",
            "r1_d060", "r1_d180", "r1_a20e45", "r1_a60e45"]


def load_pose(prefix: str, pose: str) -> list[np.ndarray] | None:
    d = Path("cache/latents") / f"{prefix}__{pose}"
    files = sorted(d.glob("episode_*.npy"))
    if not files:
        return None
    return [np.load(f).astype(np.float32).reshape(np.load(f).shape[0], -1)
            for f in files]


def load_actions(task: str) -> list[np.ndarray]:
    files = sorted((Path("data/episodes") / f"r1_{task}").glob("episode_*.npz"))
    return [np.load(f)["action"].astype(np.float32) for f in files]


def pair_with_actions(feats: list[np.ndarray], acts: list[np.ndarray]):
    X, Y = [], []
    for f, a in zip(feats, acts):
        n = min(len(f), len(a) // 2)
        if n < 2:
            continue
        X.append(f[:n])
        Y.append(a[: 2 * n : 2])
    return np.concatenate(X), np.concatenate(Y)


class Residual(nn.Module):
    """g: z -> z + delta(z). Residual because the identity is a good prior --
    a displaced latent is already close to its canonical counterpart relative
    to the scale of the whole latent space, and predicting the correction is a
    smaller job than predicting the target."""

    def __init__(self, dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, z):
        return z + self.net(z)


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


def fit_head(X, Y, seed, device, epochs=40):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    cut = int(0.9 * len(X))
    tr, va = idx[:cut], idx[cut:]
    Xt = torch.from_numpy(X).float().to(device)
    Yt = torch.from_numpy(Y).float().to(device)
    head = Head(X.shape[1], Y.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    lf = nn.MSELoss()
    best, state = float("inf"), None
    tr_t = torch.as_tensor(tr, device=device)
    va_t = torch.as_tensor(va, device=device)
    for _ in range(epochs):
        head.train()
        perm = torch.randperm(len(tr), device=device)
        for i in range(0, len(tr), 256):
            b = tr_t[perm[i:i + 256]]
            opt.zero_grad()
            lf(head(Xt[b]), Yt[b]).backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            v = lf(head(Xt[va_t]), Yt[va_t]).item()
        if v < best:
            best, state = v, {k: t.clone() for k, t in head.state_dict().items()}
    head.load_state_dict(state)
    return head


@torch.no_grad()
def mse(head, X, Y, device, g=None):
    Xt = torch.from_numpy(X).float().to(device)
    if g is not None:
        Xt = g(Xt)
    Yt = torch.from_numpy(Y).float().to(device)
    return float(((head(Xt) - Yt) ** 2).mean().item())


def train_g(pairs, dim, seed, device, epochs=60):
    """pairs: list of (z_displaced, z_reference) arrays, already standardised."""
    torch.manual_seed(seed)
    Xs = np.concatenate([a for a, _ in pairs])
    Ys = np.concatenate([b for _, b in pairs])
    Xt = torch.from_numpy(Xs).float().to(device)
    Yt = torch.from_numpy(Ys).float().to(device)
    g = Residual(dim).to(device)
    opt = torch.optim.AdamW(g.parameters(), lr=1e-3, weight_decay=1e-5)
    lf = nn.MSELoss()
    n = len(Xt)
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 256):
            b = perm[i:i + 256]
            opt.zero_grad()
            lf(g(Xt[b]), Yt[b]).backward()
            opt.step()
    g.eval()
    return g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="push")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--pca-dim", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--g-epochs", type=int, default=60)
    ap.add_argument("--out", default=None)
    ap.add_argument("--prefix", default=None,
                    help="latent prefix; r1cnn_<task> selects the "
                         "random-CNN arm for the encoder ablation")
    ap.add_argument("--n-train-poses", type=int, default=0,
                    help="use only the first N training poses, "
                         "for the how-few-viewpoints curve")
    args = ap.parse_args()

    out = args.out or f"cache/e8_{args.task}.json"
    device = get_device("auto")
    prefix = args.prefix or f"r1_{args.task}"
    acts = load_actions(args.task)

    print("=" * 74)
    print(f"E8 -- latent viewpoint canonicalization, task {args.task}")
    print("=" * 74)

    raw = {}
    for p in R1_POSES:
        f = load_pose(prefix, p)
        if f is not None:
            raw[p] = f
    if REF not in raw:
        print(f"no reference latents for {prefix}")
        return 1

    # ---- invalidation 1: the pairing --------------------------------------
    ref_counts = [len(e) for e in raw[REF]]
    bad = [p for p, f in raw.items() if [len(e) for e in f] != ref_counts]
    if bad:
        print(f"INVALID: per-episode latent counts differ at {bad}. The poses "
              f"are not timestep-paired, so g would learn noise (prereg S4.1).")
        return 1
    print(f"  pairing verified: {len(raw)} poses, episodes {ref_counts}")

    train_poses = [p for p in raw if p not in HELD_OUT and p != REF]
    held = [p for p in HELD_OUT if p in raw]
    # ---- invalidation 3: held-out purity ---------------------------------
    assert not (set(train_poses) & set(held)), "held-out pose leaked into training"
    if args.n_train_poses:
        # Deterministic subset: sorted by displacement so a small budget
        # spans the range rather than clustering at whichever poses happen
        # to sort first alphabetically.
        train_poses = sorted(train_poses,
                             key=lambda q: r1_displacement(q)["angle"]
                             )[::max(1, len(train_poses) // args.n_train_poses)]
        train_poses = train_poses[:args.n_train_poses]
    print(f"  train poses {len(train_poses)}, held-out {len(held)}: {held}")

    # PCA fitted on the reference pose only, shared by every arm and by g.
    Xref_raw = np.concatenate(raw[REF])
    mu = Xref_raw.mean(0)
    k = min(args.pca_dim, Xref_raw.shape[0] - 1)
    _, _, vt = np.linalg.svd(Xref_raw - mu, full_matrices=False)
    basis = vt[:k].T
    sd = ((Xref_raw - mu) @ basis).std(0) + 1e-6

    def project(f):
        return [((e - mu) @ basis) / sd for e in f]

    z = {p: project(f) for p, f in raw.items()}
    print(f"  PCA {Xref_raw.shape[1]} -> {k} dims (reference basis)\n")

    ay = np.concatenate([a for a in acts]).mean(0)
    asd = np.concatenate([a for a in acts]).std(0) + 1e-6

    def xy(pose):
        X, Y = pair_with_actions(z[pose], acts)
        return X, (Y - ay) / asd

    results: dict[str, dict] = {"baseline": {}, "canonical": {},
                                "multiview": {}, "ref_check": {}}
    for s in args.seeds:
        Xr, Yr = xy(REF)
        base = fit_head(Xr, Yr, s, device, args.epochs)
        ref_mse = mse(base, Xr, Yr, device)
        if ref_mse >= 0.9:
            print(f"INVALID: reference MSE {ref_mse:.3f} >= 0.9, no dynamic "
                  f"range (prereg S4.2).")
            return 1

        # g trains ONLY on the 15 training poses, paired to the reference.
        pairs = []
        for p in train_poses:
            for ep_p, ep_r in zip(z[p], z[REF]):
                n = min(len(ep_p), len(ep_r))
                pairs.append((ep_p[:n], ep_r[:n]))
        g = train_g(pairs, k, s, device, args.g_epochs)

        # multiview head: same architecture, trained on all training poses.
        Xs, Ys = [], []
        for p in [REF] + train_poses:
            a, b = xy(p)
            Xs.append(a)
            Ys.append(b)
        mv = fit_head(np.concatenate(Xs), np.concatenate(Ys), s, device, args.epochs)

        for p in held:
            X, Y = xy(p)
            results["baseline"].setdefault(p, []).append(mse(base, X, Y, device))
            results["canonical"].setdefault(p, []).append(mse(base, X, Y, device, g))
            results["multiview"].setdefault(p, []).append(mse(mv, X, Y, device))
        # E8d: does g damage the reference itself?
        results["ref_check"].setdefault("raw", []).append(ref_mse)
        results["ref_check"].setdefault("canon", []).append(
            mse(base, Xr, Yr, device, g))
        print(f"  seed {s}: reference MSE {ref_mse:.4f}")

    print()
    print("=" * 74)
    print("HELD-OUT VIEWPOINTS (never seen by g or by any head)")
    print("=" * 74)
    print(f"  {'pose':12s} {'angle':>7} {'baseline':>10} {'canonical':>10} {'multiview':>10}")
    for p in sorted(held, key=lambda q: r1_displacement(q)["angle"]):
        b = np.mean(results["baseline"][p])
        c = np.mean(results["canonical"][p])
        m = np.mean(results["multiview"][p])
        print(f"  {p:12s} {r1_displacement(p)['angle']:6.1f}° "
              f"{b:10.3f} {c:10.3f} {m:10.3f}")

    def arm(name):
        return np.array([np.mean([results[name][p][i] for p in held])
                         for i in range(len(args.seeds))])

    b, c, m = arm("baseline"), arm("canonical"), arm("multiview")
    print(f"\n  {'baseline':12s} {b.mean():.3f} +- {b.std():.3f}")
    print(f"  {'canonical':12s} {c.mean():.3f} +- {c.std():.3f}")
    print(f"  {'multiview':12s} {m.mean():.3f} +- {m.std():.3f}")

    print("\n" + "=" * 74)
    print("REGISTERED PREDICTIONS")
    print("=" * 74)
    gain = b.mean() - c.mean()
    sep = (c.mean() + 1.96 * c.std()) < (b.mean() - 1.96 * b.std())
    e8a = gain >= 0.15 and sep
    print(f"E8a (canonical beats baseline by >= 0.15, separated): "
          f"{'HOLDS' if e8a else 'FAILS'}   gain {gain:+.3f}, "
          f"{'separated' if sep else 'overlapping'}")

    e8c = m.mean() <= c.mean()
    print(f"E8c (falsifier: multiview matches or beats canonical): "
          f"{'FIRES' if e8c else 'does not fire'}   "
          f"multiview {m.mean():.3f} vs canonical {c.mean():.3f}")
    if e8c:
        print("     Training on all viewpoints directly is as good. The")
        print("     canonicalizer adds nothing and is reported as adding")
        print("     nothing.")

    rr = np.array(results["ref_check"]["raw"])
    rc = np.array(results["ref_check"]["canon"])
    e8d = abs(rc.mean() - rr.mean()) <= 0.05
    print(f"E8d (g does not damage the reference, within 0.05): "
          f"{'HOLDS' if e8d else 'FAILS'}   "
          f"raw {rr.mean():.4f} -> canonicalized {rc.mean():.4f}")
    if not e8d:
        print("     g is not viewpoint-correcting, it is flattening latents")
        print("     toward a mean. Any held-out gain above is that, not")
        print("     canonicalization.")

    over = [p for p in held if np.mean(results["canonical"][p]) > 1.0]
    if over:
        print(f"\n  bands where canonical still exceeds 1.0 (worse than the "
              f"mean action): {over}")
        print("  Reported separately per prereg S4.4.")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(
        {"task": args.task, "seeds": args.seeds, "held_out": held,
         "train_poses": train_poses, "results": results,
         "baseline": b.tolist(), "canonical": c.tolist(), "multiview": m.tolist(),
         "e8a": bool(e8a), "e8c": bool(e8c), "e8d": bool(e8d),
         "gain": float(gain)}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
