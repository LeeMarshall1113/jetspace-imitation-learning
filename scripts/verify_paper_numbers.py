#!/usr/bin/env python3
"""Verify every number the paper claims, against the committed data.

    python scripts/verify_paper_numbers.py          # check all, exit 1 on any fail
    python scripts/verify_paper_numbers.py -v       # show every check, not just fails

This is the reproducibility entry point. It reads only `cache/*.json` -- the
committed result artifacts -- and re-derives each headline in
`docs/paper-numbers.md` from scratch, asserting the paper and the data agree.
No GPU, no Docker, no model downloads; numpy and scipy only, a few seconds.

Why it exists. A number copied by hand from a superseded run is the classic
first-paper error, and it is invisible to a reviewer until one of them checks.
This project came close: interim findings at 9 and 15 encoders moved at final
scale, and one headline had to be retracted outright. The retracted claims are
asserted here too (S7 of paper-numbers.md) -- the script fails if any of them
becomes true again, so a future edit cannot quietly resurrect one.

CI runs this on every push. A reviewer can run it in one command.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CACHE = Path("cache")
AXES = ["lighting", "texture", "clutter", "noise",
        "defocus", "compress", "exposure", "lowres"]
TOL = 5e-3          # printed numbers are 3dp; bootstrap CIs vary in the 3rd


class Report:
    def __init__(self, verbose: bool):
        self.verbose, self.passed, self.failed = verbose, 0, []

    def check(self, name: str, got, want, tol: float = TOL):
        if want is None:
            ok = got is None
        elif isinstance(want, (int, np.integer)) and not isinstance(want, bool):
            ok = got == want
        elif isinstance(want, float):
            ok = got is not None and abs(float(got) - want) <= tol
        else:
            ok = got == want
        if ok:
            self.passed += 1
            if self.verbose:
                print(f"  PASS  {name:52s} {got}")
        else:
            self.failed.append((name, got, want))
            print(f"  FAIL  {name:52s} got {got!r}, expected {want!r}")
        return ok

    def section(self, title: str):
        print(f"\n{title}\n{'-' * len(title)}")


def load(name: str):
    p = CACHE / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def spearman(a, b) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 1e-12 else float("nan")


def rel_of(c: dict) -> float:
    return c.get("rel", c["held"] / max(c["ref_mse"], 1e-9))


# --------------------------------------------------------------------------
# S1 + S3: per-axis rho under the paper's convention, and the identity.
#
# Paper convention: POSITIVE rho = better probes degrade MORE. e12_analyze.py
# stores the negated form, so every comparison below flips it once.
# --------------------------------------------------------------------------
RHO = {
    ("push", "lighting"): +0.442, ("push", "texture"): +0.458,
    ("push", "exposure"): +0.322,
    ("pickplace", "lighting"): -0.018, ("pickplace", "texture"): +0.098,
    ("pickplace", "exposure"): +0.029,
}
MEAN_ABS = {"push": 0.407, "pickplace": 0.048}
N_ARMS = {"push": 22, "pickplace": 20}

#: S2. Rank of the untrained control by ABSOLUTE held-out error, as registered.
E12D = {
    "push": {"lighting": 22, "texture": 21, "exposure": 22, "clutter": 10,
             "noise": 13, "defocus": 1, "compress": 1, "lowres": 1},
    "pickplace": {"lighting": 20, "texture": 20, "exposure": 20, "clutter": 10,
                  "noise": 13, "defocus": 8, "compress": 4, "lowres": 4},
}
VALID = {"lighting", "texture", "exposure"}


def verify_e12(r: Report):
    r.section("S1/S2/S3  E12: per-axis rho, the identity, and the control")
    for task in ("push", "pickplace"):
        d = load(f"e12_{task}.json")
        if d is None:
            r.check(f"{task}: cache present", False, True)
            continue
        for ax in AXES:
            c = d.get(ax)
            if not c or "rows" not in c:
                r.check(f"{task}/{ax}: present", False, True)
                continue
            rows = c["rows"]
            r.check(f"{task}/{ax}: n arms", len(rows), N_ARMS[task])

            # The identity: probe R^2 is an exact monotone transform of ref MSE.
            names = sorted(rows)
            ident = spearman([rows[n]["probe"] for n in names],
                             [rows[n]["ref_mse"] for n in names])
            r.check(f"{task}/{ax}: rho(probe, ref_mse) == -1", ident, -1.0, 1e-9)

            # The control, and the exclusion it implies.
            order = sorted(names, key=lambda n: rows[n]["held"])
            pos = order.index("random") + 1
            r.check(f"{task}/{ax}: control rank", pos, E12D[task][ax])
            r.check(f"{task}/{ax}: excluded", c.get("rho") is None,
                    ax not in VALID)

            if ax in VALID:
                got = -c["rho"]
                r.check(f"{task}/{ax}: rho (paper convention)", got,
                        RHO[(task, ax)])
                # Recompute from the rows rather than trusting the stored value.
                mine = spearman([rows[n]["probe"] for n in names],
                                [rel_of(rows[n]) for n in names])
                r.check(f"{task}/{ax}: rho recomputed from rows", mine, got)

        e = d.get("e12a", {})
        r.check(f"{task}: mean |rho|", e.get("mean_abs"), MEAN_ABS[task])
        r.check(f"{task}: valid axis count", e.get("n_valid_axes"), 3)


# --------------------------------------------------------------------------
# S4: the ranking and the head-to-head comparisons.
# --------------------------------------------------------------------------
# All twenty rows of the ranking table in paper/main.tex, not only the
# headline ones: the other fourteen existed nowhere except the scaffold
# until the pre-writing audit regenerated them (scripts/e12_ranking.py,
# seed 0) and found them matching. Now they cannot drift unnoticed.
RANK_MEAN = {"vjepa2": 1.069, "aimv2": 1.093, "dinov3": 1.120,
             "convnext-large": 1.239, "clip-large": 1.311, "siglip2": 1.323,
             "dinov3-large": 1.503, "convnext": 1.601, "swin": 1.744,
             "dinov2-large": 1.767, "vit-in1k": 1.852, "siglip1": 1.978,
             "clip": 2.252, "vc1": 2.564, "dinov2": 2.647, "ijepa": 2.662,
             "vit-large": 2.841, "vc1-large": 2.899, "beit": 3.391,
             "random": 22.517}
HEAD2HEAD = {                       # (a, b): (diff, lo, hi, distinguishable)
    ("vjepa2", "vc1"): (-1.496, -2.916, -0.323, True),
    ("aimv2", "vc1"): (-1.471, -2.779, -0.346, True),
    ("aimv2", "vc1-large"): (-1.810, -3.811, -0.518, True),
    ("vc1", "random"): (-19.914, -47.206, -4.348, True),
    ("vjepa2", "aimv2"): (+0.025, -0.119, +0.219, False),
}


def verify_ranking(r: Report):
    r.section("S4  encoder ranking and head-to-head comparisons")
    cells = {}
    for task in ("push", "pickplace"):
        d = load(f"e12_{task}.json")
        if d is None:
            continue
        for ax in AXES:
            c = d.get(ax)
            if c and c.get("rho") is not None:
                cells[(task, ax)] = {e: rel_of(v) for e, v in c["rows"].items()}
    r.check("valid axis-task cells", len(cells), 6)
    if not cells:
        return
    keys = sorted(cells)
    encs = sorted(set.intersection(*(set(cells[k]) for k in keys)))
    r.check("encoders present in every valid cell", len(encs), 20)
    M = np.array([[cells[k][e] for k in keys] for e in encs])
    means = dict(zip(encs, M.mean(1)))
    for e, want in RANK_MEAN.items():
        r.check(f"mean relative degradation: {e}", means.get(e), want)

    order = sorted(encs, key=lambda e: means[e])
    r.check("rank 1 is vjepa2", order[0], "vjepa2")
    r.check("rank 2 is aimv2", order[1], "aimv2")
    r.check("vc1 rank", order.index("vc1") + 1, 14)
    r.check("random is last", order[-1], "random")

    # Paired bootstrap, same seed as scripts/e12_ranking.py.
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(keys), (20000, len(keys)))
    boot = np.stack([M[:, i].mean(1) for i in idx])
    ix = {e: encs.index(e) for e in encs}
    for (a, b), (diff, lo, hi, sure) in HEAD2HEAD.items():
        d_ = boot[:, ix[a]] - boot[:, ix[b]]
        l_, h_ = np.percentile(d_, [2.5, 97.5])
        r.check(f"{a} vs {b}: diff", float(d_.mean()), diff, 0.05)
        r.check(f"{a} vs {b}: CI lo", float(l_), lo, 0.6)
        r.check(f"{a} vs {b}: CI hi", float(h_), hi, 0.6)
        r.check(f"{a} vs {b}: distinguishable", bool(h_ < 0 or l_ > 0), sure)


# --------------------------------------------------------------------------
# S5 + S6: the CortexBench control, and R2.
# --------------------------------------------------------------------------
def verify_external(r: Report):
    r.section("S5/S6  CortexBench control and the R2 corroboration")
    for task, rank, n in (("pen-v0", 11, 12), ("relocate-v0", 12, 12)):
        d = load(f"cortexbench_{task}.json")
        if d is None:
            r.check(f"cortexbench/{task}: cache present", False, True)
            continue
        r.check(f"cortexbench/{task}: arms", d.get("n_arms"), n)
        r.check(f"cortexbench/{task}: control rank", d.get("random_rank"), rank)
        # The control must PASS here -- that is the scoping result.
        r.check(f"cortexbench/{task}: passes the control",
                d["random_rank"] > 2 * d["n_arms"] / 3.0, True)
        rows = d["rows"]
        order = sorted(rows, key=lambda e: -rows[e]["probe"])
        r.check(f"cortexbench/{task}: vc1 below vit-in1k",
                order.index("vc1") > order.index("vit-in1k"), True)

    d = load("r2_task_success_reach.json")
    if d is None:
        r.check("r2: cache present", False, True)
        return
    g = np.array([p["gap"] for p in d["poses"]], float)
    s = np.array([p["success"] for p in d["poses"]], float)
    r.check("r2: rho(gap, success)", spearman(g, s), -0.516)
    r.check("r2: reference success", d["reference"]["success"], 0.467)
    r.check("r2: n poses", len(g), 22)


# --------------------------------------------------------------------------
# S7: the retracted claims. These MUST stay false.
# --------------------------------------------------------------------------
def verify_retractions(r: Report):
    r.section("S7  retracted claims must not become true again")
    cells = {}
    for task in ("push", "pickplace"):
        d = load(f"e12_{task}.json")
        if d is None:
            continue
        for ax in AXES:
            c = d.get(ax)
            if c and c.get("rho") is not None:
                cells[(task, ax)] = {e: rel_of(v) for e, v in c["rows"].items()}
    if cells:
        keys = sorted(cells)
        encs = sorted(set.intersection(*(set(cells[k]) for k in keys)))
        M = np.array([[cells[k][e] for k in keys] for e in encs])
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(keys), (20000, len(keys)))
        boot = np.stack([M[:, i].mean(1) for i in idx])
        i_vc, i_rn = encs.index("vc1"), encs.index("random")
        d_ = boot[:, i_vc] - boot[:, i_rn]
        lo, hi = np.percentile(d_, [2.5, 97.5])
        # Retraction 1: "VC-1 cannot be separated from random features."
        r.check("VC-1 IS separable from random (retraction 1)",
                bool(hi < 0 or lo > 0), True)

    d = load("e12_pickplace.json")
    if d:
        m = d.get("e12a", {}).get("mean_abs")
        # Retraction 2: pickplace mean |rho| was reported as 0.733 and 0.536
        # at interim scales; the final value must not be near either.
        r.check("pickplace mean |rho| is not the interim 0.733",
                m is not None and abs(m - 0.733) > 0.1, True)
        r.check("pickplace mean |rho| is not the interim 0.536",
                m is not None and abs(m - 0.536) > 0.1, True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    if not CACHE.exists():
        print("cache/ not found -- run from the repository root", file=sys.stderr)
        return 2

    print("Verifying docs/paper-numbers.md against the committed artifacts.")
    r = Report(a.verbose)
    verify_e12(r)
    verify_ranking(r)
    verify_external(r)
    verify_retractions(r)

    total = r.passed + len(r.failed)
    print(f"\n{'=' * 62}")
    if r.failed:
        print(f"{len(r.failed)} of {total} checks FAILED:")
        for name, got, want in r.failed:
            print(f"  {name}: got {got!r}, expected {want!r}")
        print("\nThe paper and the data disagree. Either the data changed and "
              "docs/paper-numbers.md needs updating, or a number is wrong.")
        return 1
    print(f"All {total} checks passed. The paper's numbers match the data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
