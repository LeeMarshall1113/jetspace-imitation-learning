#!/usr/bin/env python3
"""E12: does probe accuracy predict robustness, per nuisance axis?

    python scripts/e12_analyze.py push

Stage 3 (the image-space axes) is registered in docs/prereg-e12-stage3.md.

Stages 1-2 are NOT. This file used to say "Implements docs/prereg-e12.md";
that document has never existed here or in git history, and the thresholds it
cited (0.4, 0.9, 0.10) live only in the constants below. They may have been
chosen in advance, but nothing archives that, and one of them was softened
after it was seen to fail. Do not describe stages 1-2 as pre-registered.

For every (encoder, axis) cell:

  probe R^2     ridge from frozen features to actions at the REFERENCE
                condition, split BY EPISODE -- what the field reports
  robustness    a head trained at the reference, evaluated at the held-out
                displaced conditions, in normalised action MSE -- what the
                field wants to know

Then correlates the two rankings per axis. E12a registers that the mean |rho|
across axes is <= 0.4; a high rho would mean probes DO predict robustness and
E11's viewpoint-only finding was axis-specific, which is reported as the result
rather than treated as a failure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ENCODERS = [
    # The original nine.
    ("vjepa2", "r1"), ("dinov3", "dinov3"), ("siglip2", "siglip2"),
    ("aimv2", "aimv2"), ("dinov2", "dino"), ("clip", "clip"),
    ("vit-in1k", "vitin1k"), ("vc1", "vc1"), ("random", "r1cnn"),
    # Scale-up. Each changes one thing against an arm already present --
    # capacity within a family, or an earlier generation of one objective --
    # rather than adding a near-duplicate row.
    ("dinov2-large", "dinov2l"), ("dinov3-large", "dinov3l"),
    ("siglip1", "siglip1"), ("vit-large", "vitlarge"),
    ("clip-large", "cliplarge"), ("vc1-large", "vc1large"),
]
AXES = ["lighting", "texture", "clutter",
        # Image-space axes, applied at encode time rather than
        # rendered. Same treatment, same controls.
        "noise", "defocus", "compress", "exposure", "lowres"]
FLOOR = 0.9          # stage3 prereg I1: a head above this cannot be compared
MIN_DEGRADE = 0.10   # stage3 prereg I2: an axis this weak cannot rank anything
LAT = Path("cache/latents")


def load(prefix: str, tag: str, task: str):
    d = LAT / f"{prefix}_e12_{task}__{tag}"
    files = sorted(d.glob("episode_*.npy"))
    if not files:
        return None
    return [np.load(f).astype(np.float32).reshape(np.load(f).shape[0], -1)
            for f in files]


IMAGE_AXES = ("noise", "defocus", "compress", "exposure", "lowres")


def actions(tag: str, task: str):
    d = Path("data/episodes") / f"e12_{task}__{tag}"
    if not d.is_dir() and tag.startswith(IMAGE_AXES):
        # Image-space axes are applied at ENCODE time to the reference
        # episodes: run_e12_all.sh passes data/episodes/e12_<task>__ref as the
        # source for every one of them. So they have latents but no episode
        # directory of their own, and the actions are the reference actions by
        # construction -- identical trajectories, corrupted pixels.
        #
        # Without this, actions() returned nothing for every image level, every
        # encoder scored n_levels=0, and the parity guard excluded all five
        # axes as "0 encoders at full coverage". Stage 3 is 600 arms and about
        # twelve hours; all of it would have analysed to nothing.
        #
        # Deliberately restricted to the known image axes rather than a blanket
        # fallback, so a genuinely missing rendered-axis directory still fails
        # loudly instead of silently scoring against the reference.
        d = Path("data/episodes") / f"e12_{task}__ref"
    return [np.load(f)["action"].astype(np.float32)
            for f in sorted(d.glob("episode_*.npz"))]


def pair(feats, acts):
    X, Y = [], []
    for f, a in zip(feats, acts):
        n = min(len(f), len(a) // 2)
        if n >= 4:
            X.append(f[:n])
            Y.append(a[: 2 * n : 2])
    return X, Y


def ridge(Xtr, Ytr, Xte, lam: float = 10.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    d = Xtr.shape[1]
    if d <= len(Xtr):
        W = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(d), Xtr.T @ Ytr)
    else:
        K = Xtr @ Xtr.T + lam * np.eye(len(Xtr))
        W = Xtr.T @ np.linalg.solve(K, Ytr)
    return Xte @ W


def spearman(a, b) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 1e-12 else float("nan")


def cell(prefix: str, axis: str, task: str, levels: list[str]):
    """(probe R^2 at reference, mean normalised MSE at held-out levels)."""
    ref_f = load(prefix, "ref", task)
    if ref_f is None:
        return None
    ra = actions("ref", task)
    Xs, Ys = pair(ref_f, ra)
    if len(Xs) < 5:
        return None

    cut = max(1, int(0.8 * len(Xs)))
    Xtr, Ytr = np.concatenate(Xs[:cut]), np.concatenate(Ys[:cut])
    Xte, Yte = np.concatenate(Xs[cut:]), np.concatenate(Ys[cut:])

    ay, asd = Ytr.mean(0), Ytr.std(0) + 1e-6
    live = Ytr.std(0) > 1e-6
    Ytr_n = ((Ytr - ay) / asd)[:, live]
    Yte_n = ((Yte - ay) / asd)[:, live]

    pred = ridge(Xtr, Ytr_n, Xte)
    ss_res = float(((Yte_n - pred) ** 2).sum())
    ss_tot = float(((Yte_n - Yte_n.mean(0)) ** 2).sum())
    probe = 1.0 - ss_res / max(ss_tot, 1e-12)

    # Reference-fit MSE, the denominator for degradation and the floor check.
    ref_mse = float(((Yte_n - pred) ** 2).mean())

    # Robustness: same ridge map, at the SAME held-out episodes.
    #
    # Episode seeds are shared across conditions by design, so episode i at a
    # displaced condition is the same trajectory as episode i at the reference.
    # Evaluating on all ten would therefore score 80% of the displaced set on
    # trajectories the ridge was fitted to, and `ref_mse` -- which uses only
    # the held-out 20% -- would not be a comparable denominator.
    #
    # That leak is invisible wherever the nuisance is strong enough to swamp
    # it, and dominant wherever it is not: the first run of this analysis
    # reported clutter degrading performance by MINUS 69.6%, i.e. distractors
    # apparently making the task easier, which is what sent me looking.
    mses = []
    for lv in levels:
        f = load(prefix, lv, task)
        if f is None:
            continue
        Xl, Yl = pair(f, actions(lv, task))
        if len(Xl) <= cut:
            continue
        X, Y = np.concatenate(Xl[cut:]), np.concatenate(Yl[cut:])
        Yn = ((Y - ay) / asd)[:, live]
        mses.append(float(((Yn - ridge(Xtr, Ytr_n, X)) ** 2).mean()))
    if not mses:
        return None
    held = float(np.mean(mses))
    # Relative degradation is the registered PRIMARY metric (stage3 prereg S2).
    #
    # probe = 1 - ss_res/ss_tot and ref_mse = ss_res/N, with ss_tot and N equal
    # across encoders because every encoder is scored on the same held-out
    # actions. So probe R^2 is an exact decreasing function of ref_mse --
    # measured at rho = -1.000 on every axis of both tasks, not approximately.
    # Ranking robustness by absolute `held` therefore re-measures baseline fit,
    # and its correlation with probe accuracy is partly that identity showing
    # through. Dividing by ref removes the baseline and leaves the thing the
    # word "robustness" is supposed to name.
    #
    # Both are kept: absolute `held` is what someone deploying the encoder
    # actually experiences, and where the two disagree that disagreement is a
    # result rather than a nuisance.
    return {"probe": probe, "ref_mse": ref_mse, "held": held,
            "rel": held / max(ref_mse, 1e-9), "n_levels": len(mses)}


def main() -> int:
    task = sys.argv[1] if len(sys.argv) > 1 else "push"
    levels_for = {
        "lighting": ["lighting_0p3", "lighting_0p45", "lighting_0p55",
                     "lighting_0p62"],
        "texture": ["texture_0p06", "texture_0p1", "texture_0p16",
                    "texture_0p24"],
        "clutter": ["clutter_1", "clutter_2", "clutter_3", "clutter_4"],
        "noise": ["noise_4p0", "noise_10p0", "noise_20p0", "noise_35p0"],
        "defocus": ["defocus_1", "defocus_2", "defocus_4", "defocus_7"],
        "compress": ["compress_4", "compress_8", "compress_14", "compress_22"],
        "exposure": ["exposure_0p65", "exposure_0p80", "exposure_1p25",
                     "exposure_1p55"],
        "lowres": ["lowres_2", "lowres_3", "lowres_5", "lowres_8"],
    }

    out: dict = {}
    rhos = {}
    for axis in AXES:
        want = levels_for[axis]

        # Invalidation 0: level parity.
        #
        # cell() averages over whatever levels an encoder happens to have, so
        # an encoder cached only at the easy end of an axis is scored against
        # a shorter and gentler set than one cached at all four. That is not a
        # small effect -- texture degrades by +814% from the first level to the
        # last. Mid-scale-up it bit exactly this way: the six newest encoders
        # had two texture levels each while the original nine had four, which
        # flattered the newcomers and made the ranking, and the rho computed
        # from it, meaningless. Rank only encoders carrying the whole axis.
        rows, short = [], []
        for name, prefix in ENCODERS:
            if not (LAT / f"{prefix}_e12_{task}__ref").is_dir():
                continue
            c = cell(prefix, axis, task, want)
            if c is None:
                continue
            if c["n_levels"] != len(want):
                short.append((name, c["n_levels"]))
                continue
            rows.append((name, c))
        if short:
            print(f"\n{axis}: incomplete level coverage, excluded -- "
                  + ", ".join(f"{n} ({k}/{len(want)})" for n, k in short))
        if len(rows) < 5:
            print(f"\n{axis}: only {len(rows)} encoders at full coverage "
                  f"-- skipping")
            continue

        # Invalidation 1: a head that cannot fit the reference is not comparable.
        bad = [n for n, c in rows if c["ref_mse"] >= FLOOR]
        usable = [(n, c) for n, c in rows if c["ref_mse"] < FLOOR]

        print(f"\n=== {axis} ===")
        print(f"  {'encoder':10s} {'probe R2':>9} {'ref MSE':>9} {'held-out':>9}")
        for n, c in sorted(rows, key=lambda r: r[1]["held"]):
            flag = "  EXCLUDED (ref >= floor)" if c["ref_mse"] >= FLOOR else ""
            print(f"  {n:10s} {c['probe']:>9.3f} {c['ref_mse']:>9.3f} "
                  f"{c['held']:>9.3f}{flag}")
        if bad:
            print(f"  excluded by the reference floor: {bad}")

        if len(usable) < 5:
            print("  too few usable encoders to rank; axis reported, not analysed")
            out[axis] = {"rows": {n: c for n, c in rows}, "rho": None,
                         "excluded": bad}
            continue

        # Invalidation 2: an axis that degrades nobody cannot rank anyone.
        degrade = np.mean([c["held"] / max(c["ref_mse"], 1e-9) - 1.0
                           for _, c in usable])
        if degrade < MIN_DEGRADE:
            print(f"  AXIS TOO WEAK: mean degradation {degrade:+.1%} < "
                  f"{MIN_DEGRADE:.0%}. Reported, excluded from E12a/E12b.")
            out[axis] = {"rows": {n: c for n, c in usable}, "rho": None,
                         "too_weak": True, "degradation": degrade}
            continue

        # E12d, the registered control on ourselves. Frozen random features
        # should be near-useless on any axis that genuinely separates
        # encoders. If they are competitive, the axis is not discriminating
        # and ranking encoders on it measures noise -- so its rows are
        # reported and then excluded, exactly as registered. This check was
        # written into the pre-registration and then omitted from the first
        # version of this script.
        order = [n for n, _ in sorted(usable, key=lambda r: r[1]["held"])]
        if "random" in order:
            pos = order.index("random") + 1
            third = len(order) / 3.0
            print(f"  E12d control: random ranks {pos}/{len(order)}")
            if pos <= 2 * third:
                print(f"  AXIS DOES NOT DISCRIMINATE: random features are not "
                      f"in the bottom third.")
                print(f"  Ranking encoders here measures noise. Reported, "
                      f"excluded from E12a/E12b.")
                out[axis] = {"rows": {n: c for n, c in usable}, "rho": None,
                             "e12d_fired": True, "random_rank": pos,
                             "degradation": degrade}
                continue

        # Higher probe = better; lower error = better. Negate the probe so both
        # rank "better" the same way before correlating.
        #
        # rho_rel is the registered PRIMARY (stage3 prereg S2): relative
        # degradation, which does not inherit the probe/ref_mse identity.
        # rho_abs is the secondary, kept because it is what a deployer feels.
        # They disagree in sign on some axes; that disagreement is a result.
        rho_rel = spearman([-c["probe"] for _, c in usable],
                           [c["rel"] for _, c in usable])
        rho_abs = spearman([-c["probe"] for _, c in usable],
                           [c["held"] for _, c in usable])
        rhos[axis] = rho_rel
        print(f"  mean degradation {degrade:+.1%}")
        print(f"  rho PRIMARY (relative degradation) = {rho_rel:+.3f}")
        print(f"  rho secondary (absolute held-out)  = {rho_abs:+.3f}"
              + ("   <-- SIGN DIFFERS from primary"
                 if rho_rel * rho_abs < 0 else ""))
        print("  CIs: scripts/e12_uncertainty.py -- no effect is claimed here "
              "on a point estimate alone")
        out[axis] = {"rows": {n: c for n, c in usable}, "rho": rho_rel,
                     "rho_abs": rho_abs, "degradation": degrade,
                     "excluded": bad}

    if rhos:
        vals = np.array(list(rhos.values()))
        mean_abs = float(np.abs(vals).mean())
        # |rho|, not rho. The registered clause is "rho < 0.5 on >= 3 axes" and
        # it means "no relationship on those axes". A signed test made every
        # negative rho satisfy it for free -- including a hypothetical -0.9,
        # which is a strong relationship being counted as evidence of none.
        # This passed unnoticed while the absolute metric produced positive
        # rho; switching to the registered primary metric turned them negative
        # and exposed it.
        below = int((np.abs(vals) < 0.5).sum())
        print("\n" + "=" * 62)
        print("E12a -- does probe accuracy predict robustness?")
        print("=" * 62)
        for a, r in rhos.items():
            print(f"  {a:10s} rho {r:+.3f}")
        print(f"  mean |rho| {mean_abs:.3f} over {len(vals)} axes; "
              f"{below}/{len(vals)} below 0.5")
        # Stage3 prereg H1: with fewer than three valid axes this is NOT
        # evaluable, and saying so is the registered outcome. The previous
        # `min(3, len(vals))` quietly reinterpreted the rule as "all available
        # axes", which let a two-axis run render a verdict -- and on push that
        # verdict turned on a single axis missing 0.5 by 0.007. A criterion
        # that cannot fail honestly is not a criterion.
        if len(vals) < 3:
            print(f"\n  NOT EVALUABLE: the registered test needs >= 3 valid "
                  f"axes and only {len(vals)} survived the invalidation "
                  f"conditions.")
            print("  Reporting the axes above without a verdict, as registered.")
            holds = None
        else:
            holds = bool(mean_abs <= 0.4 and below >= 3)
            print(f"\n  registered (mean |rho| <= 0.4 and rho < 0.5 on >= 3 "
                  f"axes): {'HOLDS' if holds else 'FAILS'}")
            if not holds:
                print("  Probes DO predict robustness on these axes. E11's")
                print("  viewpoint-only anti-correlation is then axis-specific,")
                print("  and that scoping is the result.")
        print("  Any claim from this needs the bootstrap CI "
              "(scripts/e12_uncertainty.py).")
        out["e12a"] = {"rhos": rhos, "mean_abs": mean_abs, "holds": holds,
                       "n_valid_axes": len(vals),
                       "evaluable": len(vals) >= 3}

    Path("cache").mkdir(exist_ok=True)
    Path(f"cache/e12_{task}.json").write_text(json.dumps(out, indent=2,
                                                         default=float))
    print(f"\nwrote cache/e12_{task}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
