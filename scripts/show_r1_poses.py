#!/usr/bin/env python3
"""Print the R1 camera grid and each pose's displacement from the reference.

Run before collecting: a pose grid is easy to get subtly wrong (azimuth sign,
elevation measured from the wrong plane, distance scaling the wrong quantity),
and every such error still produces a full set of numbers.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jetspace.envs.so101_env import (  # noqa: E402
    R1_POSES, R1_REF, _LOOK_AT, _spherical, r1_displacement,
)

ref = _spherical(R1_REF["azim"], R1_REF["elev"], R1_REF["dist"])
print(f"look-at {_LOOK_AT}")
print(f"reference azim {R1_REF['azim']}  elev {R1_REF['elev']}  dist {R1_REF['dist']}")
print(f"  -> pos {tuple(round(v, 3) for v in ref)}   "
      f"(the existing 'front' camera sits at 0.25 -0.70 0.50)\n")

print(f"{len(R1_POSES)} poses")
print(f"{'name':14s} {'position':26s} {'angle':>7} {'dist':>6}  arm")
print("-" * 68)
for n, pos in R1_POSES.items():
    d = r1_displacement(n)
    p = "(" + ", ".join(f"{v:6.2f}" for v in pos) + ")"
    kind = ("reference" if n == "r1_ref" else
            "azimuth" if n.startswith("r1_az") else
            "elevation" if n.startswith("r1_el") else
            "distance" if n.startswith("r1_d") else "off-axis")
    print(f"{n:14s} {p:26s} {d['angle']:>6.1f}° {d['dist_ratio']:>5.2f}x  {kind}")
