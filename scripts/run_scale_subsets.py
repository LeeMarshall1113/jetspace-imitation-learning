#!/usr/bin/env python3
"""Regenerate the encoder-subset analyses from the current committed code.

    python scripts/run_scale_subsets.py

The paper wants to say how much the E12a estimate moved with sample size. The
9- and 15-encoder JSONs in cache/ cannot answer that on their own: they were
produced before the evaluation-leak fix, the parity guard and the switch to
relative degradation, so they conflate "the estimate moved because n grew" with
"the estimate moved because the analysis was corrected". Re-running the CURRENT
analysis at those same subset sizes separates the two.

Driven from Python rather than a shell loop on purpose. The shell version of
this passed its arguments through a `for a in "push n9" ...` construct that
word-split wrongly under the WSL interop; one iteration arrived with no
arguments at all, fell back to the default task, and rewrote the canonical
cache/e12_push.json. It reproduced identical content so nothing was lost, but
argument lists that cannot be mangled are cheaper than being lucky.

Thread-capped: this box is 8 physical / 16 logical cores and shares them with
other agents. Two uncapped analysis processes once took 13.2 of 16 threads and
made the machine unusable.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

THREADS = "3"
JOBS = [("push", "n9"), ("push", "n15"),
        ("pickplace", "n9"), ("pickplace", "n15")]

DOCKER = [
    "docker", "compose", "-f", "docker/compose.yaml", "--profile", "wsl2",
    "run", "--rm", "-T",
    "-e", "PYTHONPATH=/workspace/.pydeps",
    "-e", f"OMP_NUM_THREADS={THREADS}",
    "-e", f"OPENBLAS_NUM_THREADS={THREADS}",
    "-e", f"MKL_NUM_THREADS={THREADS}",
    "-e", f"NUMEXPR_NUM_THREADS={THREADS}",
    "dev-wsl", "python", "scripts/e12_analyze.py",
]


def main() -> int:
    env = dict(os.environ, DOCKER_UID="1000", DOCKER_GID="1000",
               OMP_NUM_THREADS=THREADS, OPENBLAS_NUM_THREADS=THREADS,
               MKL_NUM_THREADS=THREADS, NUMEXPR_NUM_THREADS=THREADS)
    done, failed = [], []
    for task, subset in JOBS:
        out = Path(f"cache/e12_{task}_{subset}_recomputed.json")
        if out.exists():
            print(f"  {task}/{subset}: already present, skipping")
            done.append(str(out))
            continue
        print(f"  {task}/{subset}: running ({THREADS} threads)", flush=True)
        r = subprocess.run(DOCKER + [task, subset], env=env,
                           capture_output=True, text=True)
        if r.returncode != 0 or not out.exists():
            tail = (r.stderr or r.stdout).strip().splitlines()[-2:]
            print(f"    FAILED rc={r.returncode}: {tail}")
            failed.append(f"{task}/{subset}")
            continue
        done.append(str(out))
        print(f"    wrote {out}")

    print(f"\n{len(done)}/{len(JOBS)} subset files present"
          + (f"; FAILED: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
