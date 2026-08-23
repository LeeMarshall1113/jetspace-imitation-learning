#!/usr/bin/env python3
"""E2 — is latent distance a usable reward?

    python scripts/eval_reward.py --task pickplace

**This is the go/no-go for the headline claim.** The whole reward design rests
on one assumption: that distance in V-JEPA latent space decreases as a
demonstration progresses toward its goal. If it does not, there is no reward
signal, latent RL has nothing to optimise, and claim A is dead — better to learn
that in an afternoon than after building on it.

The proposed reward (after Balaguer & Carpin 2011, ported to latent space) is

    r_t = -|| z_t - z_goal ||        z_goal = final latent of the nearest demo

so the question is whether that quantity is monotone in time along trajectories
that succeed.

**Controls matter more than the headline number here.** The same computation is
run on raw pixels and on proprioception. If pixel distance tracks progress just
as well, the encoder is not earning its place and the architecture argument is
weaker than it looks. A result is only interesting relative to the dumb baseline
— the lesson from ledger L3.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jetspace.data.episode import EpisodeDataset  # noqa: E402


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation, without pulling in scipy."""
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def progress_stats(dists: np.ndarray) -> dict[str, float]:
    """How well does a distance sequence track progress toward the goal?

    A perfect progress signal falls monotonically, so rank correlation with time
    is -1 and every step decreases.
    """
    t = np.arange(len(dists), dtype=float)
    rho = spearman(t, dists)
    steps = np.diff(dists)
    return {
        "spearman_t_vs_dist": rho,
        "frac_steps_decreasing": float((steps < 0).mean()) if len(steps) else float("nan"),
        "total_drop": float(dists[0] - dists[-1]),
        "relative_drop": float((dists[0] - dists[-1]) / max(dists[0], 1e-9)),
    }


def summarise(name: str, per_episode: list[dict[str, float]]) -> dict[str, float]:
    keys = per_episode[0].keys()
    agg = {k: float(np.nanmean([e[k] for e in per_episode])) for k in keys}
    agg["n_episodes"] = len(per_episode)
    print(
        f"  {name:26s} rho {agg['spearman_t_vs_dist']:+.3f}   "
        f"decreasing {agg['frac_steps_decreasing']:.1%}   "
        f"rel.drop {agg['relative_drop']:+.1%}"
    )
    return agg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="pickplace",
                    help="names the default data/cache paths; any string is valid")
    ap.add_argument("--data", default=None)
    ap.add_argument("--latents", default=None)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = Path(args.data or f"data/episodes/{args.task}")
    lat_dir = Path(args.latents or f"cache/latents/{args.task}")
    if not (lat_dir / "info.json").exists():
        print(f"No latent cache at {lat_dir}. Run scripts/cache_latents.py --task {args.task}")
        return 1

    ds = EpisodeDataset(data)
    camera = ds.info["cameras"][0]
    n = min(args.limit, len(ds))
    print(f"{args.task}: {len(ds)} episodes, analysing {n}\n")

    # Load latents and collect each episode's final latent as a candidate goal.
    latents, pixels, proprios, goals = [], [], [], []
    for i in range(n):
        f = lat_dir / f"episode_{ds.records[i]['index']:06d}.npy"
        if not f.exists():
            continue
        z = np.load(f).astype(np.float32).reshape(len(np.load(f)), -1)
        ep = ds[i]
        latents.append(z)
        pixels.append(ep[f"pixels_{camera}"].astype(np.float32).reshape(len(ep["proprio"]), -1) / 255.0)
        proprios.append(ep["proprio"].astype(np.float32))
        goals.append(z[-1])
    if not latents:
        print("No cached latents matched the episodes in the dataset.")
        return 1
    goals_arr = np.stack(goals)
    print(f"loaded {len(latents)} episodes with latents\n")

    results: dict[str, dict[str, float]] = {}

    # --- 1. Same-episode goal. The optimistic case: does distance to a
    #        trajectory's OWN endpoint fall as it progresses? If this fails,
    #        nothing else can work.
    print("distance to the episode's OWN final state:")
    for label, seqs in (("V-JEPA latent", latents), ("raw pixels", pixels), ("proprioception", proprios)):
        per_ep = []
        for s in seqs:
            d = np.linalg.norm(s - s[-1], axis=1)
            per_ep.append(progress_stats(d))
        results[f"self/{label}"] = summarise(label, per_ep)

    # --- 2. Nearest-other-demo goal. This is the reward we actually propose:
    #        the goal comes from a DIFFERENT demonstration, which is what makes
    #        it usable on a new rollout that has no endpoint of its own yet.
    #
    #        ALL THREE representations run here, not just the latent. The first
    #        version compared latents cross-episode against pixels only
    #        within-episode, which is not a comparison: within an episode the
    #        image trivially approaches its own final frame as the arm settles,
    #        so pixel distance looks excellent for a reason that says nothing
    #        about whether it would transfer to an unseen rollout.
    print("\ndistance to the NEAREST OTHER demo's final state (the proposed reward):")
    for label, seqs in (
        ("V-JEPA latent", latents),
        ("raw pixels", pixels),
        ("proprioception", proprios),
    ):
        ends = np.stack([seq[-1] for seq in seqs])
        per_ep = []
        for i, seq in enumerate(seqs):
            others = np.delete(ends, i, axis=0)
            # Nearest by the trajectory's own endpoint, as the reward would do.
            j = int(np.argmin(np.linalg.norm(others - seq[-1], axis=1)))
            d = np.linalg.norm(seq - others[j], axis=1)
            per_ep.append(progress_stats(d))
        results[f"nearest/{label}"] = summarise(label, per_ep)

    # --- Verdict
    self_rho = results["self/V-JEPA latent"]["spearman_t_vs_dist"]
    px_rho = results["self/raw pixels"]["spearman_t_vs_dist"]
    near_rho = results["nearest/V-JEPA latent"]["spearman_t_vs_dist"]
    near_px = results["nearest/raw pixels"]["spearman_t_vs_dist"]

    print("\n" + "=" * 66)
    print(f"WITHIN-episode : latent {self_rho:+.3f}   pixels {px_rho:+.3f}")
    print(f"CROSS-episode  : latent {near_rho:+.3f}   pixels {near_px:+.3f}"
          "   <- the comparison that matters")
    verdict = (
        "GO — latent distance tracks progress; the reward has signal"
        if near_rho < -0.5
        else "WEAK — usable but noisy; expect a hard RL problem"
        if near_rho < -0.2
        else "NO-GO — latent distance does not track progress. Claim A is dead"
    )
    print(f"VERDICT: {verdict}")
    # Judged CROSS-episode. Within an episode, pixel distance falls trivially as
    # the arm settles into its own final frame — that measures visual similarity,
    # not transferable progress, and it is the classic reason the field moved to
    # learned embeddings rather than pixel distance in the first place.
    margin = near_px - near_rho
    print(
        f"Encoder earns its place: {'YES' if margin > 0.05 else 'NOT CLEARLY'} "
        f"(cross-episode, latent beats pixels by {margin:+.3f} rho)"
    )
    print("=" * 66)

    out = Path(args.out or f"cache/e2_reward_{args.task}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"task": args.task, "results": results,
                               "verdict": verdict}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
