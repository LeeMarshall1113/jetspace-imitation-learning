#!/usr/bin/env python3
"""Smallest latent count across the N1 datasets, for equalising every rung.

Lives in its own file rather than inline in run_n1.sh because a heredoc nested
inside a heredoc terminates at the first matching delimiter, which silently
truncated the script the first time this was written inline.
"""

from __future__ import annotations

import glob

import numpy as np

SETS = [
    "n1_R1_cubes", "n1_R2_penmug_s9", "n1_R3_penmug_s12",
    "n1_R4_blocks_top", "n1_V_blocks_wrist",
    "n1_sim_push_pretty", "n1_sim_pickplace_pretty", "n1_sim_push_pretty_dr",
]

counts = []
for s in SETS:
    files = glob.glob(f"cache/latents/{s}/episode_*.npy")
    if files:
        counts.append(sum(np.load(f).shape[0] for f in files))

print(min(counts) if counts else 500)
