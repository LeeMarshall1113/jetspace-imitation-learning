#!/usr/bin/env python3
"""E7: does video pretraining buy POLICY viewpoint generalization?

    python scripts/e7_encoder_transfer.py --task reach --seeds 0 1 2

Registered in docs/prereg-e7.md, committed before this ran.

E6 found a frozen V-JEPA 2 buys nothing consistent over a frozen random CNN --
on world-model metrics. Those are internal to the predictor. This asks the
different question: train a policy head on ONE camera viewpoint and measure how
much of it survives at 22 viewpoints it never saw. Pretraining could fail to
improve one-step latent prediction while still producing features that move less
under camera motion. Nothing here or in the literature has separated the two.

The only difference between arms is which frozen encoder produced the features.
Same head, same optimiser, same epochs, same seeds, same episodes, same poses.
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


def load_arm(prefix: str, pose: str) -> np.ndarray | None:
    """Flattened features per latent step for one pose, or None if absent."""
    d = Path("cache/latents") / f"{prefix}__{pose}"
    files = sorted(d.glob("episode_*.npy"))
    if not files:
        return None
    return [np.load(f).astype(np.float32).reshape(np.load(f).shape[0], -1)
            for f in files]


def load_actions(task: str) -> list[np.ndarray]:
    files = sorted((Path("data/episodes") / f"r1_{task}").glob("episode_*.npz"))
    return [np.load(f)["action"].astype(np.float32) for f in files]


def align(feats: list[np.ndarray], acts: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Pair each latent with the action at the frame it summarises.

    A latent at index t covers frames [2t, 2t+1]; the action taken from that
    observation is acts[2t]. Truncating to the shorter of the two is not
    optional -- encoders drop a trailing odd frame and episodes differ in
    length, so an unchecked zip would silently pair latents with actions from
    the wrong timestep.
    """
    X, Y = [], []
    for f, a in zip(feats, acts):
        n = min(len(f), len(a) // 2)
        if n < 2:
            continue
        X.append(f[:n])
        Y.append(a[: 2 * n : 2])
    return np.concatenate(X), np.concatenate(Y)


class Head(nn.Module):
    """The policy head. Identical across arms by construction."""

    def __init__(self, in_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_head(X: np.ndarray, Y: np.ndarray, seed: int, device: str,
               epochs: int, val_frac: float = 0.1):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # Standardise with TRAIN statistics only, and carry them to every displaced
    # pose. Re-standardising per pose would quietly remove the very shift the
    # experiment is trying to measure.
    mu, sd = X.mean(0), X.std(0) + 1e-6
    ay, asd = Y.mean(0), Y.std(0) + 1e-6

    idx = rng.permutation(len(X))
    ncut = int(len(X) * (1 - val_frac))
    tr, va = idx[:ncut], idx[ncut:]

    Xt = torch.from_numpy((X - mu) / sd).float().to(device)
    Yt = torch.from_numpy((Y - ay) / asd).float().to(device)

    head = Head(X.shape[1], Y.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.MSELoss()

    best, best_state = float("inf"), None
    for _ in range(epochs):
        head.train()
        perm = torch.randperm(len(tr), device=device)
        for i in range(0, len(tr), 256):
            b = torch.as_tensor(tr, device=device)[perm[i: i + 256]]
            opt.zero_grad()
            loss = lossf(head(Xt[b]), Yt[b])
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            v = lossf(head(Xt[torch.as_tensor(va, device=device)]),
                      Yt[torch.as_tensor(va, device=device)]).item()
        if v < best:
            best = v
            best_state = {k: t.clone() for k, t in head.state_dict().items()}
    head.load_state_dict(best_state)
    return head, (mu, sd, ay, asd), best


@torch.no_grad()
def eval_pose(head, stats, X: np.ndarray, Y: np.ndarray, device: str) -> float:
    """Normalised action MSE: 1.0 means no better than predicting the mean."""
    mu, sd, ay, asd = stats
    Xt = torch.from_numpy((X - mu) / sd).float().to(device)
    Yt = torch.from_numpy((Y - ay) / asd).float().to(device)
    pred = head(Xt)
    return float(((pred - Yt) ** 2).mean().item())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="reach")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--vjepa-prefix", default=None)
    ap.add_argument("--rand-prefix", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--pca-dim", type=int, default=128,
                    help="matched across arms; 0 disables (prereg S6)")
    ap.add_argument("--pilot", action="store_true",
                    help="label output as the disclosed pilot, not the "
                         "registered test")
    args = ap.parse_args()

    vp = args.vjepa_prefix or f"r1_{args.task}"
    rp = args.rand_prefix or f"r1cnn_{args.task}"
    out = args.out or f"cache/e7_{args.task}.json"
    device = get_device("auto")

    acts = load_actions(args.task)
    print("=" * 74)
    print(f"E7 -- policy viewpoint generalization, task {args.task}")
    print("=" * 74)
    print(f"{len(acts)} episodes, {len(R1_POSES)} poses, seeds {args.seeds}\n")

    arms = {"vjepa": vp, "rand": rp}
    data: dict[str, dict[str, tuple]] = {}
    for arm, prefix in arms.items():
        got = {}
        for pose in R1_POSES:
            f = load_arm(prefix, pose)
            if f is None:
                continue
            got[pose] = align(f, acts)
        data[arm] = got
        print(f"  {arm:6s} ({prefix}__*): {len(got)}/{len(R1_POSES)} poses")

    # ---- PCA, per prereg §6 ---------------------------------------------
    # Fitted on the REFERENCE pose only and carried unchanged to every
    # displaced pose. Refitting per pose would absorb exactly the shift this
    # experiment measures. Each arm gets its own basis -- the arms live in
    # different feature spaces -- but the same number of dimensions, which is
    # what invalidation 3 requires.
    if args.pca_dim and REF in data.get("vjepa", {}) and REF in data.get("rand", {}):
        for arm in arms:
            if REF not in data[arm]:
                continue
            Xr = data[arm][REF][0]
            k = min(args.pca_dim, Xr.shape[0] - 1, Xr.shape[1])
            mu = Xr.mean(0)
            # Economy SVD on the centred reference block; components are the
            # right singular vectors.
            _, _, vt = np.linalg.svd(Xr - mu, full_matrices=False)
            basis = vt[:k].T
            data[arm] = {p: ((X - mu) @ basis, Y) for p, (X, Y) in data[arm].items()}
            print(f"  {arm:6s} PCA {Xr.shape[1]} -> {k} dims "
                  f"(basis from {REF}, {Xr.shape[0]} samples)")
        print()

    missing = [a for a, g in data.items() if REF not in g]
    if missing:
        print(f"\nno reference-pose features for {missing}. "
              f"Run scripts/cache_latents_cnn.py for the r1 poses first.")
        return 1

    # Invalidation 2: matched feature counts, or the arms are not comparable.
    nv = len(data["vjepa"][REF][0])
    nr = len(data["rand"][REF][0])
    if nv != nr:
        print(f"\nINVALID: {nv} vjepa latents vs {nr} rand latents at the "
              f"reference pose. The latent/action alignment differs between "
              f"arms, so this is not a matched comparison (prereg §3.2).")
        return 1

    common = sorted(set(data["vjepa"]) & set(data["rand"]))
    print(f"  poses scored in both arms: {len(common)}\n")

    results: dict[str, dict] = {}
    for arm in arms:
        per_seed = []
        for s in args.seeds:
            Xr, Yr = data[arm][REF]
            head, stats, _ = train_head(Xr, Yr, s, device, args.epochs)
            ref_mse = eval_pose(head, stats, Xr, Yr, device)
            # Invalidation 1: a head that cannot fit its own training viewpoint
            # has no dynamic range to degrade from.
            if ref_mse >= 0.9:
                print(f"  INVALID: {arm} seed {s} reference MSE {ref_mse:.3f} "
                      f">= 0.9 -- barely better than the mean action "
                      f"(prereg §3.1). No verdict issued.")
                return 1
            row = {}
            for pose in common:
                X, Y = data[arm][pose]
                m = eval_pose(head, stats, X, Y, device)
                row[pose] = {"mse": m, "retention": ref_mse / max(m, 1e-9),
                             "gap_angle": r1_displacement(pose)["angle"]}
            ret_mean = np.mean([v["retention"] for q, v in row.items()
                                if q != REF])
            per_seed.append({"seed": s, "ref_mse": ref_mse, "poses": row})
            print(f"  {arm:6s} seed {s}: reference MSE {ref_mse:.4f}, "
                  f"mean retention {ret_mean:.3f}")
        results[arm] = {"seeds": per_seed}
        print()

    # ---- the comparison --------------------------------------------------
    def retentions(arm: str) -> np.ndarray:
        """Per-seed mean retention over displaced poses."""
        return np.array([
            np.mean([v["retention"] for p, v in sd["poses"].items() if p != REF])
            for sd in results[arm]["seeds"]])

    rv, rr = retentions("vjepa"), retentions("rand")
    gap = float(rv.mean() - rr.mean())

    print("=" * 74)
    if args.pilot:
        print("PILOT -- NOT THE REGISTERED TEST (prereg S6)")
        print("  Numbers below check that the pipeline runs. They are a")
        print("  disclosed prior look, not a result. The registered test runs")
        print("  on the expanded R1 collection.")
    else:
        print("REGISTERED PREDICTIONS")
    print("=" * 74)
    print(f"  vjepa retention  {rv.mean():.3f} +- {rv.std():.3f}   {np.round(rv, 3)}")
    print(f"  rand  retention  {rr.mean():.3f} +- {rr.std():.3f}   {np.round(rr, 3)}")

    # Non-overlap of +-1.96 sd intervals across three seeds.
    lo_v, hi_r = rv.mean() - 1.96 * rv.std(), rr.mean() + 1.96 * rr.std()
    sep = lo_v > hi_r
    e7a = gap >= 0.05 and sep
    print(f"\nE7a (gap >= 0.05, intervals separate): "
          f"{'HOLDS' if e7a else 'FAILS'}   gap {gap:+.3f}, "
          f"{'separate' if sep else 'overlapping'}")

    # E7b: advantage at the largest displacements vs the smallest.
    def adv_at(poses: list[str]) -> float:
        a = np.mean([[sd["poses"][p]["retention"] for p in poses]
                     for sd in results["vjepa"]["seeds"]])
        b = np.mean([[sd["poses"][p]["retention"] for p in poses]
                     for sd in results["rand"]["seeds"]])
        return float(a - b)

    disp = sorted((p for p in common if p != REF),
                  key=lambda p: r1_displacement(p)["angle"])
    small, large = disp[:6], disp[-6:]
    grow = adv_at(large) - adv_at(small)
    e7b = grow >= 0.03
    print(f"E7b (advantage grows with displacement, >= 0.03): "
          f"{'HOLDS' if e7b else 'FAILS'}   "
          f"small {adv_at(small):+.3f} -> large {adv_at(large):+.3f} "
          f"(growth {grow:+.3f})")

    if gap <= 0:
        print("\n" + "=" * 74)
        print("E7c FIRES -- random convolutional features retain as well or")
        print("better than a pretrained 326M video encoder. E6's negative")
        print("result extends from world models to policies. This is the")
        print("headline, registered as such in advance.")
        print("=" * 74)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(
        {"task": args.task, "seeds": args.seeds, "results": results,
         "retention_vjepa": rv.tolist(), "retention_rand": rr.tolist(),
         "gap": gap, "e7a": bool(e7a), "e7b": bool(e7b),
         "growth": float(grow)}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
