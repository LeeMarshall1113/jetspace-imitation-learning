#!/usr/bin/env python3
"""What visual regime was each dataset collected under?

    python scripts/check_dataset_regime.py

E10 tried to resolve E9's confound by running cross-task transfer on three
simulated tasks that share one action space. Both arms landed above 1.0
normalised MSE -- worse than predicting the mean action -- at every K including
32, so the comparison carries no information.

Before raising K again, check whether the datasets are even in the same regime
as the real laboratories. A real lab bolts its camera down for the whole
session; a domain-randomised simulator resamples viewpoint, lighting and
clutter every episode. If the sim sets were collected wide-camera, then the
same action corresponds to wildly different images, action prediction is a much
harder problem there than on the real labs, and both arms failing is a property
of the data rather than of transfer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SETS = ["push", "pickplace", "reach", "r1_push", "r1_pickplace", "r1_reach",
        "n1b_A_cubes__ego", "n1b_H_penmug1__camera_2"]


def main() -> int:
    for name in SETS:
        info = Path("data/episodes", name, "info.json")
        print(f"--- {name} ---")
        if not info.exists():
            print("    no info.json")
            continue
        try:
            blob = json.loads(info.read_text())
        except Exception as exc:  # noqa: BLE001
            print(f"    unreadable: {exc}")
            continue

        cams = blob.get("cameras")
        ncam = len(cams) if isinstance(cams, (list, tuple)) else cams
        print(f"    task={blob.get('task')}  cameras={ncam}  "
              f"fps={blob.get('fps')}  image_size={blob.get('image_size')}")

        # The randomisation config may be nested under any of several keys
        # depending on when the set was collected.
        rnd = None
        for key in ("randomization", "randomize", "domain_randomization"):
            if isinstance(blob.get(key), dict):
                rnd = blob[key]
                break
        if rnd is None:
            print("    randomisation: not recorded")
            continue
        print(f"    randomisation enabled={rnd.get('enabled')}  "
              f"camera_mode={rnd.get('camera_mode')!r}  "
              f"distractors={rnd.get('n_distractors')}")
        if rnd.get("enabled") and rnd.get("camera_mode") == "wide":
            print("    -> WIDE CAMERA: viewpoint resampled per episode. Action")
            print("       prediction here is a substantially harder problem")
            print("       than on a fixed-camera real lab.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
