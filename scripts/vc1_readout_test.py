#!/usr/bin/env python3
"""Is VC-1 being read out the wrong way?

    PYTHONPATH=/workspace/.pydeps python scripts/vc1_readout_test.py

Every arm in E11 is pooled from PATCH tokens, so the comparison is matched.
But VC-1's own config says `use_cls: True`, meaning its intended representation
is the CLS token, not the patch grid. Reading a model through an interface it
was not trained to expose is a fairness problem, not a property of the model --
and VC-1 came last of nine.

This compares three readouts of the same VC-1 checkpoint on the same frames:

    patch     4x4-pooled patch tokens, as E11 currently does
    cls       the CLS token alone, as VC-1 intends
    both      CLS concatenated onto the pooled patches

Measured by how much of the ACTION each readout linearly explains -- a ridge
probe from features to actions on held-out frames. That is the quantity the
downstream policy head needs, so it separates "bad features" from "bad readout"
without training a full head.

If cls markedly beats patch, E11's VC-1 number is an artefact of the readout
and must be re-run or withdrawn rather than reported.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cache_latents_hf import build  # noqa: E402
from jetspace.data.episode import EpisodeDataset  # noqa: E402
from jetspace.utils.device import get_device  # noqa: E402


@torch.no_grad()
def readouts(frames: np.ndarray, proc, model, device: str, grid: int = 4):
    """Return {name: (T//2, D)} for each candidate readout."""
    import torch.nn.functional as F

    usable = (len(frames) // 2) * 2
    frames = frames[:usable]
    cls_all, patch_all = [], []
    for i in range(0, usable, 32):
        chunk = frames[i:i + 32]
        inputs = proc(images=list(chunk), return_tensors="pt")
        inputs = {k: v.to(device) for k, v in dict(inputs).items()}
        out = model(**inputs).last_hidden_state
        n_tok = out.shape[1]
        side = int(n_tok ** 0.5)
        while side > 0 and side * side > n_tok:
            side -= 1
        prefix = n_tok - side * side
        cls_all.append(out[:, 0].float().cpu())          # CLS is token 0
        t = out[:, prefix:].reshape(out.shape[0], side, side, out.shape[-1])
        t = t.permute(0, 3, 1, 2)
        t = F.adaptive_avg_pool2d(t, (grid, grid))
        patch_all.append(t.permute(0, 2, 3, 1).reshape(out.shape[0], -1).float().cpu())

    cls = torch.cat(cls_all).numpy().reshape(usable // 2, 2, -1).mean(1)
    patch = torch.cat(patch_all).numpy().reshape(usable // 2, 2, -1).mean(1)
    return {"patch": patch, "cls": cls,
            "both": np.concatenate([cls, patch], axis=1)}


def ridge_r2(X: np.ndarray, Y: np.ndarray, lam: float = 1.0) -> float:
    """Held-out R^2 of a ridge map from features to actions."""
    n = len(X)
    cut = int(0.8 * n)
    Xtr, Xte, Ytr, Yte = X[:cut], X[cut:], Y[:cut], Y[cut:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ay, asd = Ytr.mean(0), Ytr.std(0) + 1e-6
    Ytr, Yte = (Ytr - ay) / asd, (Yte - ay) / asd
    d = Xtr.shape[1]
    # Solve in whichever space is smaller.
    if d <= len(Xtr):
        W = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(d), Xtr.T @ Ytr)
    else:
        K = Xtr @ Xtr.T + lam * np.eye(len(Xtr))
        W = Xtr.T @ np.linalg.solve(K, Ytr)
    pred = Xte @ W
    ss_res = float(((Yte - pred) ** 2).sum())
    ss_tot = float(((Yte - Yte.mean(0)) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def main() -> int:
    device = get_device("auto")
    ds = EpisodeDataset(Path("data/episodes/r1_push"))
    proc, model = build("vc1", device)

    Xs: dict[str, list] = {}
    Ys: list[np.ndarray] = []
    for i in range(min(4, len(ds))):
        ep = ds[i]
        frames = ep["pixels_r1_ref"]
        acts = ep["action"].astype(np.float32)
        r = readouts(frames, proc, model, device)
        n = min(len(next(iter(r.values()))), len(acts) // 2)
        for k, v in r.items():
            Xs.setdefault(k, []).append(v[:n])
        Ys.append(acts[: 2 * n : 2])

    Y = np.concatenate(Ys)
    print(f"{len(Y)} aligned frames from {min(4, len(ds))} episodes\n")
    print(f"{'readout':8s} {'dim':>7}  held-out R2 (action from features)")
    print("-" * 52)
    scores = {}
    for k in ("patch", "cls", "both"):
        X = np.concatenate(Xs[k])
        scores[k] = ridge_r2(X, Y)
        print(f"{k:8s} {X.shape[1]:>7}  {scores[k]:+.4f}")

    print()
    if scores["cls"] > scores["patch"] + 0.05:
        print("CLS clearly beats patch pooling. E11 reads VC-1 through an")
        print("interface it was not trained to expose, so its last-place")
        print("result is an artefact of the readout and must be re-run.")
    elif scores["patch"] >= scores["cls"] - 0.02:
        print("Patch pooling is not the problem -- CLS does no better. VC-1's")
        print("features genuinely carry little linearly-decodable action")
        print("signal here, and the E11 number stands, with the normalisation")
        print("caveat still open.")
    else:
        print("Inconclusive: CLS is somewhat better but not decisively.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
