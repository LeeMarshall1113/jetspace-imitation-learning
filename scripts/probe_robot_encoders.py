#!/usr/bin/env python3
"""Can the robotics-specific encoders be loaded and pooled like the others?

    PYTHONPATH=/workspace/.pydeps python scripts/probe_robot_encoders.py

A manipulation benchmark comparing only general-vision encoders invites the
obvious objection: where are the models trained for manipulation? These are the
three a robotics reviewer expects.

Each is awkward in its own way, so each is probed rather than assumed:

  VC-1    NeurIPS 2023. On the Hub, but as a raw MAE ViT-B/16 state dict plus a
          hydra config -- not a transformers model. Rebuilt here with timm.
  Theia   CoRL 2024, distils several vision foundation models specifically for
          robot learning. Ships custom modelling code, so trust_remote_code.
  R3M     CoRL 2022. Not on the Hub under any resolving name; ships as a GitHub
          package whose weights came from Google Drive. Probed last because it
          is the one most likely to be unavailable.
"""

from __future__ import annotations

import sys

import numpy as np
import torch


def report(name: str, ok: bool, detail: str) -> None:
    print(f"{name:10s} {'OK  ' if ok else 'FAIL'} {detail}")


def probe_vc1() -> None:
    try:
        import timm
        from huggingface_hub import hf_hub_download, get_token

        ckpt = hf_hub_download("facebook/vc1-base", "pytorch_model.bin",
                               token=get_token())
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = state.get("model", state)
        model = timm.create_model("vit_base_patch16_224", pretrained=False,
                                  num_classes=0)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        loaded = len(model.state_dict()) - len(missing)
        with torch.no_grad():
            feats = model.forward_features(torch.zeros(1, 3, 224, 224))
        report("vc1-base", True,
               f"{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M, "
               f"{loaded}/{len(model.state_dict())} keys matched, "
               f"tokens {tuple(feats.shape)}")
        if len(missing) > 20:
            print(f"           WARNING: {len(missing)} missing keys -- the "
                  f"checkpoint may not map cleanly onto this architecture")
    except Exception as e:  # noqa: BLE001
        report("vc1-base", False, f"{type(e).__name__}: {str(e)[:100]}")


def probe_theia() -> None:
    try:
        from transformers import AutoModel

        m = AutoModel.from_pretrained(
            "theaiinstitute/theia-base-patch16-224-cddsv",
            trust_remote_code=True).eval()
        with torch.no_grad():
            out = m.forward_feature(torch.zeros(1, 3, 224, 224))
        shape = tuple(out.shape) if torch.is_tensor(out) else type(out).__name__
        report("theia", True,
               f"{sum(p.numel() for p in m.parameters()) / 1e6:.0f}M, "
               f"features {shape}")
    except Exception as e:  # noqa: BLE001
        report("theia", False, f"{type(e).__name__}: {str(e)[:100]}")


def probe_r3m() -> None:
    try:
        from r3m import load_r3m
        m = load_r3m("resnet50").eval()
        with torch.no_grad():
            out = m(torch.zeros(1, 3, 224, 224) * 255)
        report("r3m", True, f"features {tuple(out.shape)}")
    except Exception as e:  # noqa: BLE001
        report("r3m", False, f"{type(e).__name__}: {str(e)[:100]}")


def main() -> int:
    print(f"torch {torch.__version__}, numpy {np.__version__}\n")
    probe_vc1()
    probe_theia()
    probe_r3m()
    print("\nAny encoder that fails here is reported as unavailable in the "
          "write-up\nrather than quietly omitted from the comparison.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
