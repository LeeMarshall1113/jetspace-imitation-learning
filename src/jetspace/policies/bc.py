"""Behavior cloning: the M2 baseline.

This is deliberately the dumbest thing that could work -- map the current frame
plus proprioception straight to the demonstrated action, with no world model, no
planning and no reinforcement learning. Its whole job is to be the floor that
the full JEPA + RL stack has to beat. If M4 cannot beat this, the project has
produced nothing (see REQUIREMENTS.md).

The visual encoder is injected rather than hard-coded, because M3 replaces this
small CNN with a frozen V-JEPA 2 encoder. Everything else in the policy, the
training loop and the evaluator stays put when that happens.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialSoftmax(nn.Module):
    """Expected 2D image position of each feature channel.

    Turns a (B, C, H, W) feature map into 2C numbers: the softmax-weighted
    centroid of each channel's activation. This is the standard visuomotor
    bottleneck (Levine et al., 2016) and it exists because the obvious
    alternative is actively wrong for control -- see `SimpleVisualEncoder`.
    """

    def __init__(self, height: int, width: int, temperature: float = 1.0) -> None:
        super().__init__()
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1.0, 1.0, width),
            torch.linspace(-1.0, 1.0, height),
            indexing="xy",
        )
        self.register_buffer("pos_x", pos_x.reshape(1, 1, -1))
        self.register_buffer("pos_y", pos_y.reshape(1, 1, -1))
        self.temperature = temperature

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat.shape
        attn = F.softmax(feat.reshape(b, c, h * w) / self.temperature, dim=-1)
        return torch.cat([(attn * self.pos_x).sum(-1), (attn * self.pos_y).sum(-1)], dim=1)


class SimpleVisualEncoder(nn.Module):
    """A small strided CNN with a spatial-softmax head.

    Stand-in for the frozen V-JEPA encoder in M3. Kept small on purpose: ~5k
    training frames is not much data, and a large backbone trained from scratch
    would memorise it.

    The head is the load-bearing part. This encoder originally ended in
    `AdaptiveAvgPool2d(1)`, which is *explicitly translation invariant* -- it
    averages away where anything is in the frame. For a reaching task, where the
    target is IS the signal, that made the encoder structurally incapable of
    supplying it. Measured cost: 3.7% success, with validation loss 0.798 on
    unit-variance targets, i.e. explaining ~20% of the residual's variance.

    Spatial softmax is the opposite choice: it reports *only* position, as the
    expected image coordinates of each channel. The final stride-2 stage is also
    dropped, leaving a 14x14 map rather than 7x7, because the target is roughly
    7 px across at this input size and localising it on a 7x7 grid is hopeless.
    """

    def __init__(self, out_dim: int = 256, in_size: int = 112, channels: int = 128,
                 stages: int = 3):
        """`in_size` and `stages` together set the spatial grid, and the grid is
        what bounds how precisely the policy can localise anything.

        The default 112 with 3 stride-2 stages gives a 14x14 map. Measured on
        reach, that policy's median closest approach was 3.9 cm against a 4.0 cm
        success radius -- landing just inside the boundary half the time. One
        grid cell spans roughly the same 3.6 cm, which is not a coincidence:
        the policy was precision-limited by its own feature resolution, not by
        capacity or by perception.

        Raising `in_size` to 224 or dropping to `stages=2` each double the grid.
        Both cost compute in the early layers only, where the maps are cheap.
        """
        super().__init__()
        self.in_size = in_size
        self.stages = stages
        # Widths must always END at `channels`, because the spatial-softmax
        # head projects from 2*channels. The previous slice-based version
        # produced [3, 32, 64] for stages=4 -- three layers ending at 64, a
        # feature map the head could not consume -- and every stages=4 run died
        # before printing a number.
        widths = {1: [3, channels],
                  2: [3, 64, channels],
                  3: [3, 32, 64, channels],
                  4: [3, 32, 64, 96, channels]}
        if stages not in widths:
            raise ValueError(f"stages must be one of {sorted(widths)}, got {stages}")
        chans = widths[stages]
        layers = []
        for i, (a, b) in enumerate(zip(chans[:-1], chans[1:])):
            k, pad = (5, 2) if i == 0 else (3, 1)
            layers += [nn.Conv2d(a, b, k, stride=2, padding=pad), nn.ReLU()]
        self.backbone = nn.Sequential(*layers)
        feat_size = in_size // (2 ** stages)
        self.feat_size = feat_size
        self.keypoints = SpatialSoftmax(feat_size, feat_size)
        self.proj = nn.Sequential(nn.Linear(2 * channels, out_dim), nn.ReLU())
        self.out_dim = out_dim

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """pixels: (B, 3, H, W) float in [0, 1]."""
        if pixels.shape[-1] != self.in_size:
            pixels = F.interpolate(
                pixels, size=(self.in_size, self.in_size), mode="bilinear", align_corners=False
            )
        return self.proj(self.keypoints(self.backbone(pixels)))


class BCPolicy(nn.Module):
    """Maps (image, proprio) -> action."""

    def __init__(
        self,
        encoder: nn.Module,
        proprio_dim: int,
        action_dim: int,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(encoder.out_dim + proprio_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, pixels: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.encoder(pixels), proprio], dim=-1))

    @torch.no_grad()
    def act(self, pixels, proprio, qpos, norm, device: str = "cpu"):  # noqa: ANN001
        """Single-observation inference. Accepts numpy, returns an absolute command.

        The network predicts a normalised DELTA; the absolute joint target is
        reconstructed here as `qpos + delta`, so callers never have to know
        which action space the policy was trained in.
        """
        import numpy as np

        self.eval()
        px = torch.as_tensor(np.asarray(pixels), dtype=torch.float32, device=device)
        px = px.permute(2, 0, 1).unsqueeze(0) / 255.0
        pr = torch.as_tensor(np.asarray(proprio), dtype=torch.float32, device=device).unsqueeze(0)
        pred = self(px, pr).squeeze(0).cpu().numpy()

        delta = pred * np.asarray(norm["action_std"]) + np.asarray(norm["action_mean"])
        return np.asarray(qpos) + delta
