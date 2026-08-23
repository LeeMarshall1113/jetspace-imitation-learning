#!/usr/bin/env python3
"""E3 — how far can V-JEPA latent imagination be trusted?

    python scripts/eval_horizon.py --task pickplace

**This is the headline measurement.** Terver et al. (2026) argue theoretically
that JEPA embedding errors grow exponentially with horizon; V-JEPA 2-AC notes
the degradation qualitatively and uses a fixed horizon anyway; AtomVLA reports
97% success on the substrate without characterising when it fails. Nobody has
published the measured curve for the released checkpoint everyone builds on.

Rolls the trained predictor open-loop from a real starting latent, feeding its
own output back in, and reports prediction error against ground truth at each
horizon — alongside two baselines that make the number interpretable:

  * **do-nothing**: predict z_t+k = z_t. Any horizon where the model is no
    better than this is a horizon where imagination is worthless.
  * **shuffled-action**: roll out with actions from a DIFFERENT episode. If the
    model scores the same, it is ignoring actions and predicting generic scene
    drift — which would look like a working world model and be useless for
    control.

The horizon at which the model stops beating do-nothing is the reportable
number: how far imagination stays worth having.
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
    ap.add_argument("--task", default="pickplace",
                    help="names the default data/cache paths; any string is valid")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--latents", default=None)
    ap.add_argument("--max-horizon", type=int, default=24)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--out", default=None)
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
    # Evaluate in whatever space the model was trained in, or the comparison is
    # against a different quantity entirely.
    basis = ckpt.get("pca_basis")
    basis = torch.tensor(basis, device=device) if basis is not None else None
    # Actions must be normalised exactly as in training.
    a_mu = torch.tensor(ckpt["norm"].get("a_mu", [0.0]), device=device)
    a_sd = torch.tensor(ckpt["norm"].get("a_sd", [1.0]), device=device)

    ds = EpisodeDataset(data)
    fpl = json.loads((lat_dir / "info.json").read_text())["frames_per_latent"]
    H = args.max_horizon

    # (n, H) error per horizon, for each condition
    err_model, err_static, err_shuffled = [], [], []
    rng = np.random.default_rng(0)

    loaded = []
    for i in range(min(args.episodes * 2, len(ds))):
        f = lat_dir / f"episode_{ds.records[i]['index']:06d}.npy"
        if not f.exists():
            continue
        z = np.load(f).astype(np.float32)
        z = z.reshape(z.shape[0], -1, z.shape[-1])
        # Must match train_predictor.load_pairs exactly: commanded displacement,
        # summed across the tubelet window. Any mismatch here silently evaluates
        # the model on inputs it was never trained on.
        ep = ds[i]
        raw_a = ep["action"].astype(np.float32)
        qpos = ep["proprio"].astype(np.float32)[:, : raw_a.shape[1]]
        delta = raw_a - qpos
        usable = (len(delta) // fpl) * fpl
        a = delta[:usable].reshape(-1, fpl, delta.shape[1]).sum(axis=1)[: len(z)]
        if len(z) >= H + 1 and len(a) >= H + 1:
            loaded.append((z, a))
        if len(loaded) >= args.episodes:
            break
    if len(loaded) < 2:
        print(f"Need >=2 episodes with at least {H+1} latents; found {len(loaded)}.")
        print("Try a smaller --max-horizon, or a task with longer episodes.")
        return 1
    print(f"{args.task}: {len(loaded)} episodes with >= {H+1} latents\n")

    with torch.no_grad():
        for k, (z, a) in enumerate(loaded):
            zt = torch.from_numpy(z).to(device)
            at = (torch.from_numpy(a).to(device) - a_mu) / a_sd
            zn = (zt - mu) / sd
            if basis is not None:
                zn = zn @ basis

            starts = range(0, len(z) - H, max(1, (len(z) - H) // 4))
            for t in starts:
                truth = zn[t + 1 : t + 1 + H]
                # 1. the model, open loop
                pred = model.rollout(zn[t : t + 1], at[t : t + H].unsqueeze(0))[0]
                err_model.append(((pred - truth) ** 2).mean(dim=(1, 2)).cpu().numpy())
                # 2. do nothing
                err_static.append(((zn[t] - truth) ** 2).mean(dim=(1, 2)).cpu().numpy())
                # 3. actions from a different episode
                other = loaded[(k + 1) % len(loaded)][1]
                if len(other) >= H:
                    s = int(rng.integers(0, max(1, len(other) - H)))
                    a_sh = (torch.from_numpy(other[s : s + H]).to(device) - a_mu) / a_sd
                    a_sh = a_sh.unsqueeze(0)
                    p_sh = model.rollout(zn[t : t + 1], a_sh)[0]
                    err_shuffled.append(((p_sh - truth) ** 2).mean(dim=(1, 2)).cpu().numpy())

    model_e = np.stack(err_model).mean(0)
    static_e = np.stack(err_static).mean(0)
    shuf_e = np.stack(err_shuffled).mean(0) if err_shuffled else np.full(H, np.nan)

    print(f"{'h':>3} {'model':>10} {'do-nothing':>12} {'shuffled-act':>13} "
          f"{'gain':>8} {'action-aware':>13}")
    print("-" * 68)
    breakeven = None
    action_blind = None
    for h in range(H):
        gain = static_e[h] / max(model_e[h], 1e-12)
        aware = shuf_e[h] / max(model_e[h], 1e-12)
        if breakeven is None and gain <= 1.0:
            breakeven = h + 1
        if action_blind is None and aware <= 1.02:
            action_blind = h + 1
        print(f"{h+1:>3} {model_e[h]:>10.5f} {static_e[h]:>12.5f} {shuf_e[h]:>13.5f} "
              f"{gain:>7.2f}x {aware:>12.2f}x")

    print("\n" + "=" * 68)
    print(f"USEFUL HORIZON     : {breakeven - 1 if breakeven else H} steps "
          f"({(breakeven - 1 if breakeven else H) / 12.5:.2f} s at 25 Hz / tubelet 2)")
    print("   (last horizon where imagination beats predicting no change)")
    print(f"ACTION-AWARE UNTIL : {action_blind - 1 if action_blind else H} steps")
    print("   (beyond this the rollout ignores which actions were taken)")
    print("=" * 68)

    out = Path(args.out or f"cache/e3_horizon_{args.task}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "task": args.task,
        "model_error": model_e.tolist(),
        "static_error": static_e.tolist(),
        "shuffled_error": shuf_e.tolist(),
        "useful_horizon": (breakeven - 1) if breakeven else H,
        "action_aware_horizon": (action_blind - 1) if action_blind else H,
        "episodes": len(loaded),
    }, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
