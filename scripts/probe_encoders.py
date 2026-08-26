#!/usr/bin/env python3
"""Which modern vision backbones can this box actually load?

    python scripts/probe_encoders.py

E11 compares V-JEPA 2 (June 2025) against DINOv2 (2023) and CLIP (2021). That
is not a fair fight: a win over three-year-old encoders reads as a win over
three-year-old encoders. The comparison needs contemporaries.

Probes each candidate for: gated access, transformers support at the installed
version, and whether the output token layout is one `cache_latents_hf.py` can
pool. Reports rather than assumes -- several of these are gated on the Hub and
will fail without an accepted licence.
"""

from __future__ import annotations

import sys
import traceback

CANDIDATES = [
    # name, HF id, released, kind
    ("dinov3-b", "facebook/dinov3-vitb16-pretrain-lvd1689m", "2025-08", "image SSL"),
    ("dinov3-l", "facebook/dinov3-vitl16-pretrain-lvd1689m", "2025-08", "image SSL"),
    ("dinov2-b", "facebook/dinov2-base", "2023-04", "image SSL"),
    ("siglip2-b", "google/siglip2-base-patch16-224", "2025-02", "image-text"),
    ("siglip-b", "google/siglip-base-patch16-224", "2023-03", "image-text"),
    ("pe-core-b", "facebook/PE-Core-B16-224", "2025-04", "image-text"),
    ("aimv2-l", "apple/aimv2-large-patch14-224", "2024-11", "image autoregressive"),
    ("clip-b16", "openai/clip-vit-base-patch16", "2021-01", "image-text"),
    ("vit-in1k", "google/vit-base-patch16-224", "2020-10", "supervised"),
]


def probe(hf_id: str) -> tuple[bool, str]:
    import torch
    from transformers import AutoImageProcessor, AutoModel

    try:
        proc = AutoImageProcessor.from_pretrained(hf_id)
        model = AutoModel.from_pretrained(hf_id)
        if hasattr(model, "vision_model"):
            model = model.vision_model
        model = model.eval()
        n = sum(p.numel() for p in model.parameters())

        import numpy as np
        dummy = [np.zeros((224, 224, 3), dtype=np.uint8)]
        with torch.no_grad():
            out = model(**proc(images=dummy, return_tensors="pt"))
        if not hasattr(out, "last_hidden_state"):
            return False, "no last_hidden_state on output"
        t = out.last_hidden_state.shape[1]
        # cache_latents_hf.py needs a square patch grid, with or without CLS.
        side_cls = round((t - 1) ** 0.5)
        side_raw = round(t ** 0.5)
        if side_cls * side_cls == t - 1:
            layout = f"{side_cls}x{side_cls} + CLS"
        elif side_raw * side_raw == t:
            layout = f"{side_raw}x{side_raw}, no CLS"
        else:
            return False, f"{t} tokens: not a square grid, pooling unsupported"
        return True, f"{n / 1e6:>4.0f}M params, {layout}"
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).split("\n")[0][:110]
        if "gated" in msg.lower() or "401" in msg or "restricted" in msg.lower():
            return False, f"GATED -- needs an accepted licence: {msg}"
        return False, f"{type(exc).__name__}: {msg}"


def main() -> int:
    import transformers
    print(f"transformers {transformers.__version__}\n")
    print(f"{'name':11s} {'released':9s} {'kind':22s} {'status'}")
    print("-" * 92)
    usable = []
    for name, hf_id, rel, kind in CANDIDATES:
        ok, detail = probe(hf_id)
        print(f"{name:11s} {rel:9s} {kind:22s} {'OK  ' if ok else 'FAIL'} {detail}")
        if ok:
            usable.append((name, hf_id, rel))
    print(f"\n{len(usable)}/{len(CANDIDATES)} usable")
    if usable:
        print("\nusable, newest first:")
        for name, hf_id, rel in sorted(usable, key=lambda x: x[2], reverse=True):
            print(f"  {rel}  {name:11s} {hf_id}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
