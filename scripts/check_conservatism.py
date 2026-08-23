#!/usr/bin/env python3
"""Is the world model's slow error growth real, or is it just not moving?

    python scripts/check_conservatism.py --task push

E3 reported that prediction error grows roughly linearly rather than
exponentially, which contradicts the standing theoretical expectation. Before
that number is published it has to survive an obvious objection.

**The objection.** Our predictor has a zero-initialised output projection and
trains with a multi-step rollout loss, both of which bias it toward small
updates. A model that systematically *under-predicts motion* stays near its
starting latent — which is exactly where the do-nothing baseline lives — so its
error and the baseline's grow together and the gain ratio stays flat. That would
look like a stable world model and be a model that has learned to sit still.

**The test.** Compare how far the model *moves* against how far reality moves:

    displacement ratio = || pred_t+h - z_t ||  /  || true_t+h - z_t ||

  ratio ~ 1.0   the model moves as far as reality; slow error growth is real
  ratio << 1.0  the model under-moves; the headline needs restating
  ratio >> 1.0  the model overshoots, a different failure

Reported per horizon, alongside directional agreement (cosine between predicted
and true displacement) — because moving the right distance in the wrong
direction is not prediction either.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jetspace.data.episode import EpisodeDataset  # noqa: E402
from jetspace.models.predictor import ActionConditionedPredictor  # noqa: E402
from jetspace.utils.device import get_device  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="push")
    ap.add_argument("--data", default=None)
    ap.add_argument("--latents", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--max-horizon", type=int, default=24)
    ap.add_argument("--episodes", type=int, default=30)
    args = ap.parse_args()

    data = Path(args.data or f"data/episodes/{args.task}")
    lat_dir = Path(args.latents or f"cache/latents/{args.task}")
    ckpt_path = Path(args.checkpoint or f"checkpoints/predictor_{args.task}_seed0.pt")
    device = get_device("auto")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ActionConditionedPredictor(
        hidden=ckpt["hidden"], grid=ckpt["grid"], action_dim=ckpt["action_dim"]
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    mu = torch.tensor(ckpt["norm"]["mu"], device=device)
    sd = torch.tensor(ckpt["norm"]["sd"], device=device)
    a_mu = torch.tensor(ckpt["norm"].get("a_mu", [0.0]), device=device)
    a_sd = torch.tensor(ckpt["norm"].get("a_sd", [1.0]), device=device)
    basis = ckpt.get("pca_basis")
    basis = torch.tensor(basis, device=device) if basis is not None else None

    ds = EpisodeDataset(data)
    fpl = json.loads((lat_dir / "info.json").read_text())["frames_per_latent"]
    H = args.max_horizon

    loaded = []
    for i in range(len(ds)):
        f = lat_dir / f"episode_{ds.records[i]['index']:06d}.npy"
        if not f.exists():
            continue
        z = np.load(f).astype(np.float32)
        z = z.reshape(z.shape[0], -1, z.shape[-1])
        ep = ds[i]
        raw_a = ep["action"].astype(np.float32)
        qpos = ep["proprio"].astype(np.float32)[:, : raw_a.shape[1]]
        d = raw_a - qpos
        usable = (len(d) // fpl) * fpl
        a = d[:usable].reshape(-1, fpl, d.shape[1]).sum(axis=1)[: len(z)]
        if len(z) >= H + 1 and len(a) >= H + 1:
            loaded.append((z, a))
        if len(loaded) >= args.episodes:
            break
    if len(loaded) < 2:
        print(f"Need >=2 episodes with {H+1} latents; found {len(loaded)}")
        return 1
    print(f"{args.task}: {len(loaded)} episodes\n")

    ratios, cosines = [], []
    with torch.no_grad():
        for z, a in loaded:
            zt = torch.from_numpy(z).to(device)
            at = (torch.from_numpy(a).to(device) - a_mu) / a_sd
            zn = (zt - mu) / sd
            if basis is not None:
                zn = zn @ basis
            for t in range(0, len(z) - H, max(1, (len(z) - H) // 4)):
                start = zn[t : t + 1]
                pred = model.rollout(start, at[t : t + H].unsqueeze(0))[0]
                truth = zn[t + 1 : t + 1 + H]
                pd = (pred - start).flatten(1)
                td = (truth - start).flatten(1)
                pn = pd.norm(dim=1)
                tn = td.norm(dim=1)
                ratios.append((pn / tn.clamp_min(1e-9)).cpu().numpy())
                cos = (pd * td).sum(1) / (pn.clamp_min(1e-9) * tn.clamp_min(1e-9))
                cosines.append(cos.cpu().numpy())

    R = np.stack(ratios).mean(0)
    C = np.stack(cosines).mean(0)

    print(f"{'h':>3} {'displacement ratio':>20} {'direction (cos)':>18}")
    print("-" * 45)
    for h in range(0, H, max(1, H // 12)):
        print(f"{h+1:>3} {R[h]:>20.3f} {C[h]:>18.3f}")
    print(f"{H:>3} {R[-1]:>20.3f} {C[-1]:>18.3f}")

    print("\n" + "=" * 60)
    print(f"mean displacement ratio  {R.mean():.3f}   (1.0 = moves as far as reality)")
    print(f"mean direction cosine    {C.mean():.3f}   (1.0 = moves the right way)")
    # The original test was one-sided and scored a model that overshot by 47%
    # with a near-random direction (ratio 1.475, cosine 0.459) as healthy.
    # Moving too far is a failure too, and direction has to gate the verdict:
    # travelling exactly the right distance the wrong way is not prediction.
    if C.mean() < 0.5:
        print("\nVERDICT: DIRECTION IS NEAR-RANDOM (cosine < 0.5). Whatever the")
        print("  displacement ratio says, this model is not predicting, and any")
        print("  horizon number derived from it is meaningless.")
    elif R.mean() > 1.25:
        print("\nVERDICT: THE MODEL OVERSHOOTS. It moves considerably farther")
        print("  than reality does, inflating apparent motion without tracking")
        print("  it. Not conservatism -- the opposite failure.")
    elif R.mean() < 0.5:
        print("\nVERDICT: THE MODEL UNDER-MOVES. Slow error growth is partly")
        print("  conservatism, not accuracy. The E3 headline must be restated.")
    elif R.mean() < 0.8:
        print("\nVERDICT: mildly conservative. Report the ratio alongside the")
        print("  horizon curve so the number is interpretable.")
    elif C.mean() < 0.8:
        print("\nVERDICT: right distance, wrong way. Displacement is healthy but")
        print("  direction is not; report both or the ratio flatters the model.")
    else:
        print("\nVERDICT: the model moves about as far as reality does, in the")
        print("  right direction. Slow error growth reflects accuracy.")
    print("=" * 60)

    out = Path(f"cache/conservatism_{args.task}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "task": args.task, "displacement_ratio": R.tolist(),
        "direction_cosine": C.tolist(), "mean_ratio": float(R.mean()),
        "mean_cosine": float(C.mean()),
        "overshoots": bool(R.mean() > 1.25),
        "direction_random": bool(C.mean() < 0.5),
    }, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
