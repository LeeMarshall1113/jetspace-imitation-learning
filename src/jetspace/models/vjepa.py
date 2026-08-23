"""Frozen V-JEPA 2 encoder — the M3 backbone.

The encoder is never trained. It is run once over the dataset and its outputs
cached to disk; everything downstream reads latents rather than pixels. That is
what makes the project fit on one 16 GB consumer GPU: a frozen module needs no
gradients and no optimizer state, and its outputs are deterministic, so the
expensive part is paid exactly once.

Model: `facebook/vjepa2-vitl-fpc64-256` — 326M parameters, hidden size 1024,
patch 16, tubelet 2, 64 frames per clip at 256x256. ViT-L rather than ViT-g
(1B+) because the larger variants buy accuracy we cannot yet measure a need for,
at a memory cost we would feel immediately.

**Spatial structure is deliberately preserved.** Mean-pooling V-JEPA's tokens
into one vector per timestep is the obvious compression and it would repeat,
exactly, the defect that cost M2 four attempts: global average pooling is
translation invariant, and these tasks are entirely about *where* things are
(see docs/ledger.md L4). Tokens are pooled on a grid, never to a point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
TUBELET = 2       # frames per temporal latent position
PATCH = 16
CROP = 256

# V-JEPA 2 preprocessing, from the model's video_preprocessor_config.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class LatentSpec:
    """Shape of what the encoder produces, so callers need not guess."""

    hidden: int          # channels per token
    grid: int            # spatial tokens per side after pooling
    frames_per_latent: int

    @property
    def tokens(self) -> int:
        return self.grid * self.grid

    @property
    def dim(self) -> int:
        return self.tokens * self.hidden


class VJEPAEncoder(nn.Module):
    """Frozen V-JEPA 2, wrapped to emit per-timestep spatial token grids."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        *,
        pool_grid: int = 4,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        self.model_id = model_id
        self.device_str = device
        self.dtype = dtype
        self.pool_grid = pool_grid

        self.model = AutoModel.from_pretrained(model_id, torch_dtype=dtype).to(device)
        self.model.eval()
        # Freeze explicitly rather than relying on no_grad at every call site:
        # one forgotten context manager would otherwise start training a 326M
        # parameter backbone by accident.
        for p in self.model.parameters():
            p.requires_grad_(False)

        cfg = self.model.config
        self.hidden = cfg.hidden_size
        self.native_grid = cfg.crop_size // cfg.patch_size
        self.spec = LatentSpec(
            hidden=self.hidden, grid=pool_grid, frames_per_latent=cfg.tubelet_size
        )

    @property
    def out_dim(self) -> int:
        """Flattened latent size, for policies that want a vector."""
        return self.spec.dim

    def _preprocess(self, frames: np.ndarray) -> torch.Tensor:
        """(T, H, W, 3) uint8 -> (1, T, 3, 256, 256) normalised."""
        x = torch.from_numpy(np.ascontiguousarray(frames)).to(self.device_str)
        x = x.permute(0, 3, 1, 2).float() / 255.0
        if x.shape[-1] != CROP:
            x = F.interpolate(x, size=(CROP, CROP), mode="bilinear", align_corners=False)
        mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
        return ((x - mean) / std).unsqueeze(0).to(self.dtype)

    @torch.no_grad()
    def encode(self, frames: np.ndarray, chunk: int = 32) -> torch.Tensor:
        """Encode one episode.

        Args:
            frames: (T, H, W, 3) uint8.
            chunk: frames per forward pass. Bounds peak memory — the token count
                grows linearly in clip length, and a 400-frame episode encoded in
                one pass would not fit.

        Returns:
            (T', grid, grid, hidden) where T' = T // tubelet.
        """
        if frames.shape[0] < TUBELET:
            raise ValueError(f"Need at least {TUBELET} frames, got {frames.shape[0]}")

        outs = []
        # Truncate to a whole number of tubelets; a trailing odd frame has no
        # temporal partner and the model cannot place it.
        usable = (frames.shape[0] // TUBELET) * TUBELET
        for start in range(0, usable, chunk):
            end = min(start + chunk, usable)
            if (end - start) % TUBELET:
                end -= 1
            if end <= start:
                break
            x = self._preprocess(frames[start:end])
            feats = self.model.get_vision_features(x)          # (1, N, hidden)
            n_t = (end - start) // TUBELET
            g = self.native_grid
            feats = feats.reshape(1, n_t, g, g, self.hidden)

            if self.pool_grid != g:
                # Grid pooling, not global pooling. Reduces tokens ~16x while
                # keeping "where" recoverable.
                f = feats.reshape(n_t, g, g, self.hidden).permute(0, 3, 1, 2)
                f = F.adaptive_avg_pool2d(f, (self.pool_grid, self.pool_grid))
                feats = f.permute(0, 2, 3, 1)
            else:
                feats = feats.squeeze(0)
            outs.append(feats.float().cpu())

        return torch.cat(outs, dim=0)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """Batch of single frames -> flattened latents, for use as a BC encoder.

        Each frame is duplicated to fill one tubelet, since V-JEPA is a video
        model and has no single-frame mode. Wasteful, but it makes the frozen
        encoder a drop-in replacement for `SimpleVisualEncoder` so M2 and M3 can
        be compared under identical training code.
        """
        b = pixels.shape[0]
        x = pixels
        if x.shape[-1] != CROP:
            x = F.interpolate(x, size=(CROP, CROP), mode="bilinear", align_corners=False)
        mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        x = x.unsqueeze(1).repeat(1, TUBELET, 1, 1, 1).to(self.dtype)

        with torch.no_grad():
            feats = self.model.get_vision_features(x)
        g = self.native_grid
        feats = feats.reshape(b, g, g, self.hidden).permute(0, 3, 1, 2)
        if self.pool_grid != g:
            feats = F.adaptive_avg_pool2d(feats, (self.pool_grid, self.pool_grid))
        return feats.flatten(1).float()
