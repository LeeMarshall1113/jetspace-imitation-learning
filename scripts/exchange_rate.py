#!/usr/bin/env python3
"""E1: an exchange rate between nuisances, not a single-axis ruler.

    python scripts/exchange_rate.py --episodes 6 --frames 64

R1 calibrated latent gap against camera rotation and reported session noise as
"21.8 degrees". The standing objection is **non-injectivity**: a monotone scalar
map is many-to-one, so "equals 21.8 degrees" only says the two quantities
project to the same scalar, not that one resembles the other.

The fix is to calibrate against MORE THAN ONE physical nuisance and check
whether they agree. Sweep camera angle, lighting intensity, and material hue
separately; convert the same measured gap through each curve. If the three
conversions rank shifts consistently, the scalar is a genuine exchange rate. If
they disagree, the ruler is axis-dependent and that has to be said out loud
rather than discovered by a reviewer.

**Why this is worth doing beyond answering the objection.** Existing robot
robustness benchmarks rank nuisance factors -- Factor World (arXiv:2307.03659)
and COLOSSEUM (arXiv:2402.08191) both report which factor hurts most. But each
factor is perturbed by an arbitrary, non-commensurate amount, so those rankings
are confounded with how hard the authors chose to push each knob. An exchange
rate dissolves that confound: it is the missing common scale.

The intellectual ancestor is psychophysics' equivalent input noise (Pelli &
Farell), which expresses an internal deficit as the external noise that would
produce it. That tradition has never been invoked in this literature.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from align_simulator import gap_between  # noqa: E402

#: Each axis: name, the RandomizationConfig field it drives, and the sweep.
#: The reference value is first and defines the zero point for every axis.
AXES = {
    "azimuth": ("camera_azimuth_range", [0, 5, 10, 20, 30, 45, 60], "deg"),
    "light": ("light_diffuse_range", [0.7, 0.62, 0.55, 0.45, 0.38, 0.3], "diffuse"),
    "hue": ("material_hue_jitter", [0.0, 0.03, 0.06, 0.10, 0.16, 0.24], "rgba"),
}

#: Reference values for the axes NOT being swept, so each sweep varies one
#: thing. Without this a lighting sweep would silently ride on a different
#: camera pose than the azimuth sweep and the curves would not share a zero.
REF = {"azimuth": 0.0, "elevation": 30.0, "distance": 0.81,
       "light": 0.7, "hue": 0.0}


def build_cfg(axis: str, value: float):
    from jetspace.envs.randomization import RandomizationConfig

    az = value if axis == "azimuth" else REF["azimuth"]
    light = value if axis == "light" else REF["light"]
    hue = value if axis == "hue" else REF["hue"]
    return RandomizationConfig(
        enabled=True,
        camera_mode="fixed",
        camera_azimuth_range=(az, az),
        camera_elevation_range=(REF["elevation"], REF["elevation"]),
        camera_distance_range=(REF["distance"], REF["distance"]),
        camera_lookat=(0.30, 0.0, 0.10),
        camera_lookat_jitter=0.0,
        camera_pos_jitter=0.0,
        light_diffuse_range=(light, light),
        light_pos_jitter=0.0,
        material_hue_jitter=hue,
        n_distractors=(0, 0),
    )


def render_latents(spec, enc, torch, cfg, episodes: int, frames: int, seed: int,
                   env_holder: dict) -> np.ndarray:
    if "env" not in env_holder:
        env_holder["env"] = spec["env"](image_size=224, pretty=True, randomize=cfg)
    else:
        env_holder["env"].randomizer.cfg = cfg
    env = env_holder["env"]
    expert = spec["expert"](env, np.random.default_rng(seed))

    out = []
    for e in range(episodes):
        obs = env.reset(seed=seed * 1000 + e)
        expert.reset(env)
        cam = env.camera_names[0]
        got = [obs.pixels[cam]]
        for _ in range(frames - 1):
            r = env.step(expert.act(obs))
            obs = r.obs
            got.append(obs.pixels[cam])
            if r.terminated or r.truncated:
                break
        v = np.stack(got)
        z = enc.encode(torch.from_numpy(np.ascontiguousarray(v)),
                       chunk=32, margin=15).float().cpu().numpy()
        out.append(z.reshape(z.shape[0], -1, z.shape[-1]).mean(axis=1))
    return np.concatenate(out, axis=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="push")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="cache/e1_exchange.json")
    ap.add_argument("--rungs", default="cache/e2_rungs.json",
                    help="E2 output; must have been measured at the same --dim")
    args = ap.parse_args()

    import torch

    from jetspace.envs.registry import get_task
    from jetspace.models.vjepa import VJEPAEncoder
    from jetspace.utils.device import get_device

    device = get_device("auto")
    enc = VJEPAEncoder(device=device, pool_grid=4)
    spec = get_task(args.task)
    holder: dict = {}

    print("=" * 70)
    print("E1 -- exchange rate between nuisance axes")
    print("=" * 70)
    print(f"task {args.task}, {args.episodes} episodes x {args.frames} frames, "
          f"PCA {args.dim}\n")

    # The reference is the zero point for every axis and is rendered once.
    ref = render_latents(spec, enc, torch, build_cfg("azimuth", REF["azimuth"]),
                         args.episodes, args.frames, args.seed, holder)
    print(f"reference: {len(ref)} latents\n")

    curves = {}
    for axis, (_field, values, unit) in AXES.items():
        print(f"{axis} sweep ({unit})")
        pts = []
        for v in values:
            lat = render_latents(spec, enc, torch, build_cfg(axis, v),
                                 args.episodes, args.frames, args.seed + 1, holder)
            g = gap_between(ref, lat, args.dim, args.seed)
            delta = abs(v - REF[axis])
            pts.append({"value": float(v), "delta": float(delta), "frechet": float(g)})
            print(f"  {axis} {v:>6.2f}  delta {delta:>6.2f}  frechet {g:8.1f}")
        curves[axis] = pts
        print()

    # ---- the conversion, in both directions -----------------------------
    def to_units(axis: str, target: float) -> float | None:
        pts = sorted(curves[axis], key=lambda p: p["delta"])
        xs = [p["delta"] for p in pts]
        ys = [p["frechet"] for p in pts]
        if target <= ys[0]:
            return 0.0
        for i in range(len(ys) - 1):
            if ys[i] <= target <= ys[i + 1]:
                span = ys[i + 1] - ys[i]
                f = 0.0 if span < 1e-9 else (target - ys[i]) / span
                return xs[i] + f * (xs[i + 1] - xs[i])
        return None

    # ---- rungs come from E2, at a MATCHING dim --------------------------
    # Hardcoding rung numbers from an earlier run is precisely the mistake in
    # ledger L7: Frechet has no absolute scale, so a rung measured at one
    # `pca_dim` cannot be looked up on a curve measured at another. Load them
    # from E2 and refuse to convert if the two disagree.
    rung_path = Path(args.rungs)
    if not rung_path.exists():
        print(f"\nNo {rung_path}. Run scripts/measure_rungs.py --dim {args.dim} "
              f"first; the curves above are still valid on their own.")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"task": args.task, "curves": curves, "conversions": None,
             "episodes": args.episodes, "frames": args.frames}, indent=2))
        print(f"wrote {args.out}")
        return 0

    blob = json.loads(rung_path.read_text())
    if blob["dim"] != args.dim:
        raise SystemExit(
            f"{rung_path} was measured at dim {blob['dim']} but these curves "
            f"are at dim {args.dim}. Frechet values from different PCA "
            f"dimensions are not comparable (ledger L7). Re-run one of them.")
    RUNGS = {k: v["mean"] for k, v in blob["rungs"].items() if k != "null"}
    floor = blob["rungs"]["null"]["mean"]
    print(f"rungs from {rung_path} (dim {blob['dim']}), null floor {floor:.1f}\n")

    print("=" * 70)
    print("THE SAME GAP, EXPRESSED IN THREE DIFFERENT NUISANCE UNITS")
    print("=" * 70)
    header = f"{'measured gap':16s} {'frechet':>9}"
    for axis, (_f, _v, unit) in AXES.items():
        header += f" {axis + ' (' + unit + ')':>18}"
    print(header)
    print("-" * len(header))

    table = {}
    for name, val in RUNGS.items():
        row = f"{name:16s} {val:>9.1f}"
        conv = {}
        for axis in AXES:
            u = to_units(axis, val)
            conv[axis] = u
            row += f" {('beyond' if u is None else f'{u:.2f}'):>18}"
        table[name] = conv
        print(row)

    print()
    print("=" * 70)
    print("IS IT AN EXCHANGE RATE, OR AXIS-DEPENDENT?")
    print("=" * 70)
    # If the axes agree, each should place the rungs in the SAME ORDER and the
    # ratios between rungs should be similar. Disagreement means the scalar is
    # not a common currency and the "equals N degrees" phrasing is unsupported.
    inside = {a: [n for n, c in table.items() if c[a] is not None] for a in AXES}
    common = set.intersection(*(set(v) for v in inside.values())) if inside else set()
    if len(common) < 2:
        print("  Too few rungs fall inside every axis's swept range to compare.")
        print("  Widen the sweeps before claiming an exchange rate.")
    else:
        names = sorted(common, key=lambda n: RUNGS[n])
        print(f"  comparable rungs: {names}")
        for axis in AXES:
            vals = [table[n][axis] for n in names]
            ratio = vals[-1] / max(vals[0], 1e-9)
            print(f"  {axis:9s} ratio {names[-1]} / {names[0]} = {ratio:.2f}x")
        print()
        print("  Consistent ratios across axes support a genuine exchange rate.")
        print("  Divergent ratios mean the conversion depends on which nuisance")
        print("  you calibrate against, and 'equals N degrees' is then a")
        print("  statement about the camera axis alone.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"task": args.task, "curves": curves, "conversions": table,
         "rungs": RUNGS, "episodes": args.episodes, "frames": args.frames},
        indent=2, default=float))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
