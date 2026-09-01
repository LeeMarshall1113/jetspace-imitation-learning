#!/usr/bin/env python3
"""Did VC-1's weights actually load, or is it a random ViT wearing the name?

    python scripts/verify_vc1.py

VC-1 is about to carry the paper's headline -- that a purpose-built robotics
representation cannot be separated from random features under nuisance shift.
Before publishing that, rule out the boring explanation: that the checkpoint
never loaded and "VC-1" has been a randomly initialised ViT all along.

The loader calls load_state_dict(strict=False) and guards only on the count of
MISSING keys. Unexpected keys are discarded in silence, so a checkpoint whose
names are prefixed differently could leave much of timm's random init in place.
And VC-1 sitting between the trained encoders and the random control is exactly
what a partial load would look like.

The test that settles it: build the architecture twice from the same seed, load
the checkpoint into one, and compare tensors. Anything bit-identical to the
fresh init did not come from the checkpoint.
"""

from __future__ import annotations

import sys

import numpy as np
import torch


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "vc1"
    repos = {"vc1": ("facebook/vc1-base", "vit_base_patch16_224"),
             "vc1-large": ("facebook/vc1-large", "vit_large_patch16_224")}
    repo, arch = repos[name]

    import timm
    from huggingface_hub import get_token, hf_hub_download

    ckpt = hf_hub_download(repo, "pytorch_model.bin", token=get_token())
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = state.get("model", state)
    print(f"{repo}  ->  timm {arch}")
    print(f"checkpoint tensors: {len(sd)}")
    print(f"sample keys: {list(sd)[:4]}")

    torch.manual_seed(0)
    net = timm.create_model(arch, pretrained=False, num_classes=0)
    fresh = {k: v.clone() for k, v in net.state_dict().items()}
    missing, unexpected = net.load_state_dict(sd, strict=False)

    print(f"\nmissing keys    {len(missing):4d}   (in the model, not the file)")
    print(f"unexpected keys {len(unexpected):4d}   (in the file, not the model)")
    if missing[:5]:
        print(f"  missing e.g.    {missing[:5]}")
    if unexpected[:5]:
        print(f"  unexpected e.g. {unexpected[:5]}")

    after = net.state_dict()
    same, changed, n_same_params, n_tot = [], [], 0, 0
    for k, v in fresh.items():
        n_tot += v.numel()
        if torch.equal(v, after[k]):
            same.append(k)
            n_same_params += v.numel()
        else:
            changed.append(k)
    print(f"\ntensors unchanged from random init: {len(same)}/{len(fresh)}")
    print(f"parameters still random: {n_same_params:,}/{n_tot:,} "
          f"({100 * n_same_params / max(n_tot, 1):.1f}%)")
    if same[:8]:
        print(f"  e.g. {same[:8]}")

    verdict_ok = n_same_params / max(n_tot, 1) < 0.02
    print("\n" + ("WEIGHTS LOADED: the checkpoint overwrote the initialisation."
                  if verdict_ok else
                  "PROBLEM: a substantial fraction of the network is still at "
                  "random initialisation. Any claim about VC-1 is a claim "
                  "about a partly-random network."))

    # Behavioural confirmation: loaded and unloaded nets should not agree.
    torch.manual_seed(0)
    rnd = timm.create_model(arch, pretrained=False, num_classes=0).eval()
    x = torch.randn(2, 3, 224, 224, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        a = net.eval().forward_features(x).flatten().numpy()
        b = rnd.forward_features(x).flatten().numpy()
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    print(f"\ncosine(VC-1 features, same-seed random ViT) = {cos:+.4f}")
    print("  near 1.0 would mean the two networks compute the same thing")
    print(f"  feature std: loaded {a.std():.4f}   random {b.std():.4f}")
    return 0 if verdict_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
