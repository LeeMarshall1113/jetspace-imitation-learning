#!/usr/bin/env python3
"""A1: tune a simulator to a target real dataset from unlabelled video.

    python scripts/align_simulator.py --target n1b_A_cubes__ego --budget 200

Implements docs/prereg-align.md. Ten simulator visual parameters are optimised
to minimise the Fréchet distance between simulated latents and a target real
dataset's latents, in the frozen V-JEPA space, using the same instrument as
N1b and R1 so every number is comparable to what is already measured.

**The point is that no real robot is involved.** The target is unlabelled
video. The standard way to tune domain randomisation is by downstream policy
success, which requires evaluating on the physical arm -- the expensive step
this replaces.

**Random search runs at the same budget and is the primary falsifier.** If
uniform sampling matches the optimiser, there is no method here, only the
observation that some simulator configurations happen to resemble some labs.
The registration names this as the condition that kills the claim, so it runs
by default rather than on request.

Held-out episodes the optimiser never sees are scored separately. Optimising
appearance against 100 frames and reporting the fit on those same frames would
be curve-fitting.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from measure_domain_gap import frechet  # noqa: E402

#: name, low, high. Ranges span what a person would plausibly set up, not what
#: MuJoCo will accept -- an optimiser given absurd bounds finds absurd optima.
PARAMS = [
    ("azimuth", -60.0, 60.0),        # degrees about the workspace
    ("elevation", 20.0, 70.0),
    ("distance", 0.5, 1.2),          # metres
    ("lookat_x", 0.15, 0.40),
    ("lookat_y", -0.12, 0.12),
    ("lookat_z", 0.02, 0.25),
    ("light_diffuse", 0.3, 1.0),
    ("light_height", 0.8, 2.5),
    ("hue_r", -0.25, 0.25),          # additive rgba shift on scene geoms
    ("hue_g", -0.25, 0.25),
]


@dataclass
class Objective:
    """Fréchet gap between simulation under a parameter vector and a target."""

    target: np.ndarray          # (N, D) pooled real latents
    task: str
    episodes: int
    frames_per_ep: int
    dim: int
    seed: int
    _cache: dict = None

    def __post_init__(self):
        self._cache = {}
        self._reported = False
        self._env = None
        import torch

        from jetspace.envs.registry import get_task
        from jetspace.models.vjepa import VJEPAEncoder
        from jetspace.utils.device import get_device

        self._torch = torch
        self._device = get_device("auto")
        self._enc = VJEPAEncoder(device=self._device, pool_grid=4)
        self._spec = get_task(self.task)

    def render(self, x: np.ndarray) -> np.ndarray:
        """Roll out `episodes` under parameters x and return pooled latents."""
        import numpy as _np

        from jetspace.envs.randomization import RandomizationConfig

        p = dict(zip([n for n, _, _ in PARAMS], x))
        cfg = RandomizationConfig(
            enabled=True,
            camera_mode="fixed",
            camera_azimuth_range=(p["azimuth"], p["azimuth"]),
            camera_elevation_range=(p["elevation"], p["elevation"]),
            camera_distance_range=(p["distance"], p["distance"]),
            camera_lookat=(p["lookat_x"], p["lookat_y"], p["lookat_z"]),
            camera_lookat_jitter=0.0,
            camera_pos_jitter=0.0,
            light_diffuse_range=(p["light_diffuse"], p["light_diffuse"]),
            light_pos_jitter=0.0,
            material_hue_jitter=0.0,
            n_distractors=(0, 0),
        )
        if self._env is None:
            self._env = self._spec["env"](image_size=224, pretty=True, randomize=cfg)
        else:
            # Swap the config rather than rebuilding: model compilation and EGL
            # context creation dominated the 66 s per evaluation, and neither
            # depends on the parameters being optimised.
            self._env.randomizer.cfg = cfg
        env = self._env
        expert = self._spec["expert"](env, _np.random.default_rng(self.seed))

        frames = []
        for e in range(self.episodes):
            obs = env.reset(seed=self.seed * 1000 + e)
            expert.reset(env)
            cam = env.camera_names[0]
            got = [obs.pixels[cam]]
            for _ in range(self.frames_per_ep - 1):
                r = env.step(expert.act(obs))
                obs = r.obs
                got.append(obs.pixels[cam])
                if r.terminated or r.truncated:
                    break
            frames.append(_np.stack(got))

        lat = []
        for v in frames:
            z = self._enc.encode(self._torch.from_numpy(_np.ascontiguousarray(v)),
                                 chunk=32, margin=15).float().cpu().numpy()
            lat.append(z.reshape(z.shape[0], -1, z.shape[-1]).mean(axis=1))
        return _np.concatenate(lat, axis=0)

    def __call__(self, x: np.ndarray) -> float:
        key = tuple(np.round(x, 4))
        if key in self._cache:
            return self._cache[key]
        try:
            sim = self.render(np.asarray(x, dtype=float))
        except Exception as exc:                # noqa: BLE001
            # An unrenderable configuration is a bad configuration, not a crash
            # -- but the FIRST failure gets reported. Silently returning 1e9
            # made the smoke run print "gap 1000000000.0" for the default
            # configuration with no indication why, which is the same
            # swallow-the-error defect this project keeps finding.
            if not self._reported:
                self._reported = True
                import traceback
                print(f"    objective raised {type(exc).__name__}: {exc}")
                traceback.print_exc()
                print("    (further failures scored 1e9 silently)")
            self._cache[key] = 1e9
            return 1e9
        val = gap_between(self.target, sim, self.dim, self.seed)
        self._cache[key] = val
        return val


def gap_between(real: np.ndarray, sim: np.ndarray, dim: int, seed: int) -> float:
    """Fréchet, statistics fit on the REAL side, as in N1b and R1."""
    n = min(len(real), len(sim))
    if n < dim + 2:
        # NOT an exception -- which is why the smoke run reported a gap of 1e9
        # for the DEFAULT configuration with no traceback and no explanation.
        # A covariance in `dim` dimensions needs more than `dim` samples; one
        # episode of 32 frames yields 16 latents against dim=64.
        raise ValueError(
            f"need > {dim + 2} latents per side for a {dim}-dim covariance, "
            f"have {n}. Raise --episodes/--frames or lower --dim."
        )
    rng = np.random.default_rng(seed)
    a = real[rng.choice(len(real), n, replace=False)]
    b = sim[rng.choice(len(sim), n, replace=False)]
    mu, sd = a.mean(0), a.std(0) + 1e-6
    an, bn = (a - mu) / sd, (b - mu) / sd
    w, v = np.linalg.eigh(np.cov(an, rowvar=False))
    basis = v[:, np.argsort(w)[::-1][:dim]]
    return frechet(an @ basis, bn @ basis)


def load_target(name: str, episodes: slice) -> np.ndarray:
    d = Path("cache/latents") / name
    files = sorted(d.glob("episode_*.npy"))[episodes]
    if not files:
        raise SystemExit(f"no latents in {d}")
    out = []
    for f in files:
        z = np.load(f).astype(np.float32)
        out.append(z.reshape(z.shape[0], -1, z.shape[-1]).mean(axis=1))
    return np.concatenate(out, axis=0)


def cem(obj, budget: int, seed: int, pop: int = 20, elite: float = 0.25):
    """Cross-entropy method. Returns (best_x, best_val, history)."""
    rng = np.random.default_rng(seed)
    lo = np.array([l for _, l, _ in PARAMS])
    hi = np.array([h for _, _, h in PARAMS])
    mu = (lo + hi) / 2
    sd = (hi - lo) / 4
    # With budget < pop the loop below never ran and CEM returned inf while
    # random search returned a real number -- which would have read as "random
    # search wins", the registered falsifier, for a reason that has nothing to
    # do with the objective.
    if budget < pop * 2:
        pop = max(4, budget // 3)
        print(f"    (budget {budget} is small; population reduced to {pop})")
    n_elite = max(2, int(pop * elite))
    best_x, best_v, hist = None, float("inf"), []
    used = 0
    while used + pop <= budget:
        xs = np.clip(rng.normal(mu, sd, size=(pop, len(PARAMS))), lo, hi)
        vs = np.array([obj(x) for x in xs])
        used += pop
        order = np.argsort(vs)
        if vs[order[0]] < best_v:
            best_v, best_x = float(vs[order[0]]), xs[order[0]].copy()
        elites = xs[order[:n_elite]]
        mu = elites.mean(0)
        # Floor the spread so the search cannot collapse before the budget is
        # spent -- a premature collapse would make CEM lose to random search
        # for a reason that has nothing to do with the objective.
        sd = np.maximum(elites.std(0), (hi - lo) * 0.02)
        hist.append({"evals": used, "best": best_v, "batch_best": float(vs[order[0]])})
        print(f"    CEM {used:>4}/{budget}  best {best_v:8.1f}")
    return best_x, best_v, hist


def random_search(obj, budget: int, seed: int):
    """The primary falsifier: same budget, no learning."""
    rng = np.random.default_rng(seed + 9973)
    lo = np.array([l for _, l, _ in PARAMS])
    hi = np.array([h for _, _, h in PARAMS])
    best_x, best_v, hist = None, float("inf"), []
    for i in range(budget):
        x = rng.uniform(lo, hi)
        v = obj(x)
        if v < best_v:
            best_v, best_x = float(v), x.copy()
        if (i + 1) % 20 == 0:
            hist.append({"evals": i + 1, "best": best_v})
            print(f"    RAND {i+1:>4}/{budget}  best {best_v:8.1f}")
    return best_x, best_v, hist


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="n1b_A_cubes__ego")
    ap.add_argument("--task", default="push")
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    train = load_target(args.target, slice(0, 5))
    heldout = load_target(args.target, slice(5, 10))
    print(f"target {args.target}: {len(train)} train latents, "
          f"{len(heldout)} held-out\n")

    obj = Objective(train, args.task, args.episodes, args.frames, args.dim, args.seed)

    print("  default configuration")
    lo = np.array([l for _, l, _ in PARAMS])
    hi = np.array([h for _, _, h in PARAMS])
    default = (lo + hi) / 2
    t0 = time.time()
    base = obj(default)
    per = time.time() - t0
    total = per * args.budget * 2 / 60.0
    print(f"    gap {base:.1f}   ({per:.0f}s per evaluation)")
    print(f"    {args.budget} evaluations x 2 optimisers ~ {total:.0f} min\n")

    print("  CEM")
    cem_x, cem_v, cem_hist = cem(obj, args.budget, args.seed)
    print("\n  random search, same budget")
    rnd_x, rnd_v, rnd_hist = random_search(obj, args.budget, args.seed)

    # Held-out: same configurations, episodes the optimiser never scored.
    obj_ho = Objective(heldout, args.task, args.episodes, args.frames,
                       args.dim, args.seed + 1)
    ho_base = obj_ho(default)
    ho_cem = obj_ho(cem_x) if cem_x is not None else float("nan")

    red = 100 * (base - cem_v) / max(base, 1e-9)
    ho_red = 100 * (ho_base - ho_cem) / max(ho_base, 1e-9)
    margin = 100 * (rnd_v - cem_v) / max(rnd_v, 1e-9)

    print("\n" + "=" * 70)
    print("A1 -- simulator alignment")
    print("=" * 70)
    print(f"  default gap            {base:8.1f}")
    print(f"  CEM                    {cem_v:8.1f}   ({red:+.1f}%)")
    print(f"  random search          {rnd_v:8.1f}   (CEM beats it by {margin:.1f}%)")
    print(f"  held-out, default      {ho_base:8.1f}")
    print(f"  held-out, CEM          {ho_cem:8.1f}   ({ho_red:+.1f}%)")
    print()
    print(f"  P1 reduce >=25%:        {'HOLDS' if red >= 25 else 'FAILS'}  ({red:.1f}%)")
    print(f"  P2 beat random >=10%:   {'HOLDS' if margin >= 10 else 'FAILS'}  ({margin:.1f}%)")
    print(f"  P3 half survives:       "
          f"{'HOLDS' if ho_red >= red / 2 else 'FAILS'}  ({ho_red:.1f}% vs {red/2:.1f}%)")
    if margin < 10:
        print()
        print("  P2 is the PRIMARY FALSIFIER. Random search matching CEM means")
        print("  there is no method here -- only that some simulator settings")
        print("  happen to resemble some labs. Registered as the condition that")
        print("  kills the claim.")
    print("=" * 70)

    best = {n: float(v) for (n, _, _), v in zip(PARAMS, cem_x)} if cem_x is not None else {}
    out = Path(args.out or f"cache/a1_align_{args.target}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "target": args.target, "task": args.task, "budget": args.budget,
        "default_gap": base, "cem_gap": cem_v, "random_gap": rnd_v,
        "heldout_default": ho_base, "heldout_cem": ho_cem,
        "reduction_pct": red, "heldout_reduction_pct": ho_red,
        "cem_margin_pct": margin, "best_params": best,
        "cem_history": cem_hist, "random_history": rnd_hist, "seed": args.seed,
    }, indent=2, default=float))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
