"""A from-scratch convolutional encoder, as the control arm for E6.

E6 asks whether a frozen 326M-parameter video foundation model is worth its
cost against a ~7M-parameter CNN trained on the task's own data. The question
matters because the answer decides what the paper is about: if the CNN wins,
"frozen video foundation models are not worth it at this data scale" is the
finding, and it contradicts a widely held assumption.

**The comparison has a trap, and it is the reason this file exists separately
from the policy encoder in `policies/bc.py`.**

A *jointly trained* encoder can make its own latent space trivially
predictable. Nothing in a prediction loss forbids collapsing the
representation toward a constant: prediction error goes to zero and the world
model is worthless. A frozen encoder cannot do this, because it cannot move.
So comparing raw validation losses hands the win to the trained encoder by
construction, every time, regardless of which representation is better.

Three arms are therefore required, not two:

    frozen V-JEPA     cannot cheat; the incumbent
    frozen RANDOM CNN cannot cheat either, and isolates what PRETRAINING buys
                      as opposed to what architecture buys
    trained CNN       can cheat; must be checked for it

The random arm is the cheap and most informative one. If random convolutional
features match V-JEPA, then 22M videos of pretraining bought nothing here, and
that is a sharper result than either of the other two arms can give.

Collapse is detected, not assumed absent, by three measures already in the
repository: the gain ratio against the do-nothing baseline (a collapsed space
has a tiny baseline error too, so the *ratio* stays honest where raw loss does
not), the inverse-dynamics probe R^2 (a collapsed space cannot recover
actions), and the shuffled-action test.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ScratchCNNEncoder(nn.Module):
    """Frame-pair CNN producing latents shaped like V-JEPA's pooled output.

    Consumes `frames_per_latent` consecutive frames stacked on the channel
    axis, so the temporal rate matches V-JEPA's tubelet of 2 exactly. A
    comparison at a different temporal resolution would confound representation
    quality with how much time each latent covers.
    """

    def __init__(
        self,
        hidden: int = 1024,
        grid: int = 4,
        frames_per_latent: int = 2,
        width: int = 64,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.grid = grid
        self.frames_per_latent = frames_per_latent

        c_in = 3 * frames_per_latent
        chans = [c_in, width, width * 2, width * 4, width * 8]
        layers: list[nn.Module] = []
        for a, b in zip(chans[:-1], chans[1:]):
            layers += [
                nn.Conv2d(a, b, 4, stride=2, padding=1),
                nn.GroupNorm(min(8, b), b),
                nn.SiLU(),
            ]
        # 224 -> 112 -> 56 -> 28 -> 14, then project and pool to the grid.
        layers += [nn.Conv2d(chans[-1], hidden, 3, stride=1, padding=1)]
        self.net = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(grid)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """(B, fpl*3, H, W) uint8-or-float -> (B, grid, grid, hidden)."""
        if frames.dtype == torch.uint8:
            frames = frames.float() / 255.0
        z = self.pool(self.net(frames))              # (B, hidden, g, g)
        return z.permute(0, 2, 3, 1).contiguous()    # (B, g, g, hidden)

    @torch.no_grad()
    def encode(self, video: torch.Tensor, batch: int = 64) -> torch.Tensor:
        """(T, H, W, 3) -> (T // fpl, grid, grid, hidden).

        Mirrors `VJEPAEncoder.encode`'s contract so the two are drop-in
        substitutes downstream. There is deliberately no windowing here: a CNN
        has no temporal position embedding, so it cannot acquire the period-8
        comb that overlapped-window transformer encoding stamps into simulated
        latents (ledger L6). That difference is itself a result and should not
        be papered over by giving the CNN an artificial window.
        """
        fpl = self.frames_per_latent
        usable = (len(video) // fpl) * fpl
        v = video[:usable]
        if v.dtype == torch.uint8:
            v = v.float() / 255.0
        # (N, fpl, H, W, 3) -> (N, fpl*3, H, W)
        v = v.reshape(-1, fpl, *v.shape[1:])
        v = v.permute(0, 1, 4, 2, 3).reshape(v.shape[0], fpl * 3, *v.shape[2:4])

        out = []
        for i in range(0, len(v), batch):
            out.append(self.forward(v[i : i + batch].to(next(self.parameters()).device)))
        return torch.cat(out, dim=0).cpu()


def build(kind: str, **kw) -> ScratchCNNEncoder:
    """`kind` is 'random' or 'trained'; both are the same architecture.

    They differ only in whether the weights are ever updated, which is exactly
    the axis E6 needs to isolate: architecture versus what training on the task
    buys, held apart from what pretraining on 22M videos buys.
    """
    enc = ScratchCNNEncoder(**kw)
    if kind == "random":
        for p in enc.parameters():
            p.requires_grad_(False)
        enc.eval()
    return enc
