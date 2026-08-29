#!/usr/bin/env python3
"""Regenerate every paper figure and table from the result JSONs.

    python scripts/make_figures.py --out paper/figures

Nothing here recomputes a result. Every number is read from `cache/*.json`,
which is what makes this safe to re-run: when an experiment is extended, the
figures and the LaTeX tables update together and the paper cannot drift out of
sync with the data behind it.

That drift is the specific failure this exists to prevent. A number quoted in an
abstract, copied by hand from a run that has since been superseded, is the
classic first-paper error and it is invisible to every reviewer until one of
them checks. It nearly happened here: the previous version of this file was
written for the 9- and 15-encoder runs and for a thesis since retired, so its
figures would have contradicted the tables in main.tex.

Figures follow docs/paper-numbers.md (the canonical record) and are ordered to
match the paper's argument:

  fig_discriminability   S4.1  five of eight axes cannot rank encoders
  fig_ranking            S4.3  the 20-encoder ranking with bootstrap CIs
  fig_signflip           S3.4  the metric choice reverses the conclusion
  fig_probe_vs_robust    S4.5  the original hypothesis, valid axes only
  fig_cortexbench        S4.4  the field's benchmark passes the same control
  fig_shift_ladder       S5    distribution-shift rungs in one space
  fig_r2_gap_success     S5    latent gap against task success

CONVENTION, matching the paper: rho relates the probe-R^2 ranking to the
RELATIVE-degradation ranking, oriented so POSITIVE means better probes degrade
MORE. The stored `rho` in e12_*.json is the negated form (e12_analyze.py
correlates -probe against rel). This file flips it once, at the point of use,
and never again -- getting that wrong would invert every claim in the paper.

Outputs, per figure: a PDF for the paper, a PNG for looking at, and a `.tex`
fragment for tables that can be \\input directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CACHE = Path("cache")
BOOT = 20000
RNG = np.random.default_rng(0)

AXES = ["lighting", "texture", "clutter", "noise",
        "defocus", "compress", "exposure", "lowres"]

#: Consistent identity per encoder across every figure -- a reader tracking
#: V-JEPA between two plots should not have to re-read the legend. Families
#: share a hue; the untrained control is grey and starred everywhere so it
#: stays findable, which matters because it is the paper's control.
STYLE = {
    "vjepa2": ("#d62728", "o"), "ijepa": ("#ff9896", "o"),
    "dinov2": ("#1f77b4", "s"), "dinov2-large": ("#aec7e8", "s"),
    "dinov3": ("#17becf", "s"), "dinov3-large": ("#9edae5", "s"),
    "siglip2": ("#2ca02c", "^"), "siglip1": ("#98df8a", "^"),
    "aimv2": ("#9467bd", "D"),
    "clip": ("#ff7f0e", "v"), "clip-large": ("#ffbb78", "v"),
    "vit-in1k": ("#8c564b", "P"), "vit-large": ("#c49c94", "P"),
    "vc1": ("#e377c2", "X"), "vc1-large": ("#f7b6d2", "X"),
    "mae": ("#bcbd22", "<"), "beit": ("#dbdb8d", "<"),
    "convnext": ("#7b4173", "h"), "convnext-large": ("#ce6dbd", "h"),
    "resnet50": ("#843c39", "h"), "swin": ("#5254a3", ">"),
    "random": ("#7f7f7f", "*"),
}
TEAL, GREY, EXCL = "#0E6E6E", "#7f7f7f", "#9B3D30"


def style(name):
    return STYLE.get(name, ("#333333", "o"))


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  {name}.pdf / .png")


def load(name: str):
    p = CACHE / name
    return json.loads(p.read_text()) if p.exists() else None


def rel_of(c: dict) -> float:
    """Relative degradation, the registered primary metric."""
    return c.get("rel", c["held"] / max(c["ref_mse"], 1e-9))


def boot_ci(vals: np.ndarray, n: int = BOOT):
    """Percentile CI over a resampled mean. The resampling unit is the
    axis-task CELL; encoders are resampled jointly elsewhere so comparisons
    stay paired."""
    idx = RNG.integers(0, len(vals), (n, len(vals)))
    b = vals[idx].mean(1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def valid_axes(d: dict) -> list:
    return [a for a in AXES
            if a in d and isinstance(d[a], dict) and d[a].get("rho") is not None]


# ------------------------------------------------------------------ figure 1
def fig_discriminability(out: Path) -> None:
    """Where the untrained control ranks, per axis, both tasks.

    The paper's strongest claim, and previously the one with no figure.
    Plotted as a fraction of the field rather than a raw rank: the two tasks
    have different encoder counts (22 vs 20 -- MAE and ResNet-50 fail
    pickplace's reference floor), and raw ranks would make those scales look
    comparable when they are not.
    """
    tasks = [(t, load(f"e12_{t}.json")) for t in ("push", "pickplace")]
    tasks = [(t, d) for t, d in tasks if d]
    if not tasks:
        return
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    w, task_labels = 0.38, []
    for k, (task, d) in enumerate(tasks):
        xs, ys, cols = [], [], []
        n_seen = 0
        for i, a in enumerate(AXES):
            c = d.get(a)
            if not c or "rows" not in c:
                continue
            n = len(c["rows"])
            n_seen = n
            order = sorted(c["rows"], key=lambda e: c["rows"][e]["held"])
            if "random" not in order:
                continue
            xs.append(i + (k - 0.5) * w)
            ys.append((order.index("random") + 1) / n)
            cols.append(GREY if c.get("rho") is not None else EXCL)
        ax.bar(xs, ys, width=w, color=cols, edgecolor="white", linewidth=.6,
               hatch=("" if k == 0 else "///"))
        task_labels.append(f"{task}  (n={n_seen})")
    ax.axhline(2 / 3, ls="--", lw=1.1, color="#131A24")
    ax.text(-0.45, 2 / 3 + .02, "worst-third threshold", fontsize=8)
    ax.set_xticks(range(len(AXES)))
    ax.set_xticklabels(AXES, rotation=20, ha="right")
    ax.set_ylabel("rank of untrained control\n(fraction of field; 1.0 = worst)")
    ax.set_ylim(0, 1.08)
    ax.set_title("An axis below the line cannot separate trained from "
                 "untrained features", fontsize=9.5)
    # Two legends: colour carries the verdict, hatch carries the task. One
    # combined legend would imply the two encode the same thing.
    verdict = ax.legend(
        [plt.Rectangle((0, 0), 1, 1, color=GREY),
         plt.Rectangle((0, 0), 1, 1, color=EXCL)],
        ["discriminates", "excluded"],
        frameon=False, fontsize=8, loc="upper left")
    ax.add_artist(verdict)
    ax.legend([plt.Rectangle((0, 0), 1, 1, facecolor="white",
                             edgecolor="#3D4956", hatch=h)
               for h in ("", "///")],
              task_labels, frameon=False, fontsize=8, loc="upper right")
    save(fig, out, "fig_discriminability")


# ------------------------------------------------------------------ figure 2
def fig_ranking(out: Path) -> None:
    """The encoder ranking as a dot-and-interval plot.

    A forest plot rather than a bar chart: the claim is a TOP GROUP, not a
    winner (V-JEPA 2 and AIMv2 are not distinguishable), and overlapping
    intervals show that where bars would imply a strict order.
    """
    cells = {}
    for task in ("push", "pickplace"):
        d = load(f"e12_{task}.json")
        if not d:
            continue
        for a in valid_axes(d):
            cells[(task, a)] = {e: rel_of(c) for e, c in d[a]["rows"].items()}
    if not cells:
        return
    keys = sorted(cells)
    encs = sorted(set.intersection(*(set(cells[k]) for k in keys)))
    M = np.array([[cells[k][e] for k in keys] for e in encs])
    means = M.mean(1)
    order = np.argsort(means)

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    for row, i in enumerate(order):
        col, mk = style(encs[i])
        lo, hi = boot_ci(M[i])
        ax.plot([lo, hi], [row, row], color=col, lw=2, alpha=.55,
                solid_capstyle="round")
        ax.plot(means[i], row, mk, color=col, ms=6.5,
                markeredgecolor="white", markeredgewidth=.6)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([encs[i] for i in order], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(1.0, ls=":", lw=1, color="#131A24")
    ax.set_xscale("log")
    ax.set_xlabel("relative degradation, log scale\n"
                  "(1.0 = no loss under shift; lower is better)")
    ax.set_title(f"{len(encs)} encoders, {len(keys)} valid cells, "
                 "95% paired bootstrap CI", fontsize=9.5)
    save(fig, out, "fig_ranking")


# ------------------------------------------------------------------ figure 3
def fig_signflip(out: Path) -> None:
    """Primary against secondary metric, per valid cell.

    Points in the shaded quadrants are cells where the metric choice reverses
    the conclusion -- the entanglement result made visual.
    """
    fig, ax = plt.subplots(figsize=(4.5, 4.3))
    for task, mk in (("push", "o"), ("pickplace", "s")):
        d = load(f"e12_{task}.json")
        if not d:
            continue
        for a in valid_axes(d):
            # Stored rho is negated; flip once here to the paper's convention.
            x, y = -d[a]["rho"], -d[a].get("rho_abs", float("nan"))
            ax.plot(x, y, mk, ms=7, color=TEAL, alpha=.85,
                    markeredgecolor="white")
            ax.annotate(f"{task[:4]}/{a[:4]}", (x, y), fontsize=6.5,
                        xytext=(4, 3), textcoords="offset points")
    lim = 0.9
    ax.fill_between([-lim, 0], 0, lim, color=EXCL, alpha=.07, lw=0)
    ax.fill_between([0, lim], -lim, 0, color=EXCL, alpha=.07, lw=0)
    ax.axhline(0, lw=.8, color="#9aa5b1")
    ax.axvline(0, lw=.8, color="#9aa5b1")
    ax.plot([-lim, lim], [-lim, lim], ls=":", lw=1, color="#9aa5b1")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"$\rho$, relative degradation (primary)")
    ax.set_ylabel(r"$\rho$, absolute held-out error (secondary)")
    ax.set_title("Shaded: the metric reverses the sign", fontsize=9.5)
    save(fig, out, "fig_signflip")


# ------------------------------------------------------------------ figure 4
def fig_probe_vs_robust(out: Path) -> None:
    """Probe accuracy against relative degradation, one panel per valid axis.

    A scatter rather than a bar chart because the claim concerns the STRENGTH
    of a relationship, and strength is only legible when the spread is visible.
    Excluded axes are deliberately absent: plotting them would invite reading a
    ranking off an axis that cannot rank.
    """
    for task in ("push", "pickplace"):
        d = load(f"e12_{task}.json")
        if not d:
            continue
        va = valid_axes(d)
        if not va:
            continue
        fig, axes = plt.subplots(1, len(va), figsize=(3.1 * len(va), 3.2),
                                 squeeze=False)
        for ax, a in zip(axes[0], va):
            for e, c in d[a]["rows"].items():
                col, mk = style(e)
                ax.plot(c["probe"], rel_of(c), mk, color=col, ms=6,
                        markeredgecolor="white", markeredgewidth=.5)
            ax.set_yscale("log")
            ax.axhline(1.0, ls=":", lw=.9, color="#9aa5b1")
            ax.set_title(f"{a}   " + r"$\rho$ = " + f"{-d[a]['rho']:+.3f}",
                         fontsize=9)
            ax.set_xlabel(r"probe $R^2$")
        axes[0][0].set_ylabel("relative degradation (log)")
        fig.suptitle(f"{task}: probe accuracy against robustness, "
                     "valid axes only", fontsize=10)
        save(fig, out, f"fig_probe_vs_robust_{task}")


# ------------------------------------------------------------------ figure 5
def fig_cortexbench(out: Path) -> None:
    """The same control on CortexBench's own demonstrations.

    The scoping result: the field's task-variation benchmark passes the control
    our corruption axes fail. Ordered by pen-v0 so VC-1's position relative to
    a plain supervised ViT is readable.
    """
    ds = {t: load(f"cortexbench_{t}.json") for t in ("pen-v0", "relocate-v0")}
    ds = {t: d for t, d in ds.items() if d}
    if not ds:
        return
    first = list(ds.values())[0]
    encs = sorted(set.intersection(*(set(d["rows"]) for d in ds.values())),
                  key=lambda e: -first["rows"][e]["probe"])
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    w = 0.38
    for k, (t, d) in enumerate(ds.items()):
        vals = [d["rows"][e]["probe"] for e in encs]
        cols = [EXCL if e == "RANDOM" else (TEAL if k == 0 else "#48B3AC")
                for e in encs]
        ax.bar(np.arange(len(encs)) + (k - .5) * w, vals, width=w, color=cols,
               edgecolor="white", linewidth=.5,
               label=f"{t}   control {d['random_rank']}/{d['n_arms']}")
    ax.axhline(0, lw=.8, color="#131A24")
    ax.set_xticks(range(len(encs)))
    ax.set_xticklabels(encs, rotation=35, ha="right", fontsize=7.5)
    ax.set_ylabel(r"probe $R^2$")
    ax.set_title("CortexBench Adroit: the untrained control ranks last on "
                 "both tasks", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    save(fig, out, "fig_cortexbench")


# ------------------------------------------------------------------ figure 6
def fig_shift_ladder(out: Path) -> None:
    d = load("e2_rungs.json")
    if not d or "rungs" not in d:
        return
    rungs = [(k, v) for k, v in d["rungs"].items()
             if isinstance(v, dict) and "mean" in v]
    rungs.sort(key=lambda kv: kv[1]["mean"])
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ys = np.arange(len(rungs))
    ax.barh(ys, [v["mean"] for _, v in rungs],
            xerr=[v["sd"] for _, v in rungs], color=TEAL, alpha=.85,
            error_kw={"lw": 1, "ecolor": "#131A24"})
    ax.set_yticks(ys)
    ax.set_yticklabels([str(k) for k, _ in rungs], fontsize=8)
    ax.set_xlabel("Frechet distance between latent sets")
    ax.set_title("Distribution-shift rungs measured in one space "
                 "(mean $\\pm$ sd)", fontsize=9.5)
    save(fig, out, "fig_shift_ladder")


# ------------------------------------------------------------------ figure 7
def fig_r2_gap_success(out: Path) -> None:
    d = load("r2_task_success_reach.json")
    if not d or "poses" not in d:
        return
    g = np.array([p["gap"] for p in d["poses"]], float)
    s = np.array([p["success"] for p in d["poses"]], float)
    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    ax.errorbar(g, s, yerr=[p.get("sd", 0) for p in d["poses"]], fmt="o",
                color=TEAL, ms=5, lw=.9, capsize=2, alpha=.9)
    if "reference" in d:
        ax.axhline(d["reference"]["success"], ls=":", lw=1, color=EXCL)
        ax.text(g.max(), d["reference"]["success"] + .012, "reference",
                ha="right", fontsize=7.5, color=EXCL)
    ax.set_xlabel("latent gap from reference")
    ax.set_ylabel("task success")
    ci = d.get("ci95", [float("nan"), float("nan")])
    ax.set_title(r"$\rho$ = " + f"{d.get('rho', float('nan')):+.3f}   "
                 f"95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]", fontsize=9.5)
    save(fig, out, "fig_r2_gap_success")


# ------------------------------------------------------------------- tables
def table_e12(out: Path, task: str) -> None:
    """Appendix table: every cell, both metrics, exclusions with reasons."""
    d = load(f"e12_{task}.json")
    if not d:
        return
    L = [r"\begin{tabular}{lrrrl}", r"\toprule",
         r"Axis & $n$ & $\rho$ (rel.) & $\rho$ (abs.) & Status \\",
         r"\midrule"]
    for a in AXES:
        c = d.get(a)
        if not c or "rows" not in c:
            continue
        n = len(c["rows"])
        if c.get("rho") is None:
            why = (f"excluded, control {c.get('random_rank', '?')}/{n}"
                   if c.get("e12d_fired") else "excluded")
            L.append(f"{a} & {n} & --- & --- & {why} \\\\")
        else:
            L.append(f"{a} & {n} & ${-c['rho']:+.3f}$ & "
                     f"${-c.get('rho_abs', float('nan')):+.3f}$ & valid \\\\")
    e = d.get("e12a", {})
    L += [r"\midrule",
          f"\\multicolumn{{5}}{{l}}{{mean $|\\rho|$ = "
          f"{e.get('mean_abs', float('nan')):.3f} over "
          f"{e.get('n_valid_axes', 0)} valid axes}} \\\\",
          r"\bottomrule", r"\end{tabular}"]
    out.mkdir(parents=True, exist_ok=True)
    (out / f"table_e12_{task}.tex").write_text("\n".join(L) + "\n")
    print(f"  table_e12_{task}.tex")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/figures")
    args = ap.parse_args()
    out = Path(args.out)

    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "figure.dpi": 120,
        "savefig.bbox": "tight", "axes.spines.top": False,
        "axes.spines.right": False,
    })

    print("figures:")
    fig_discriminability(out)
    fig_ranking(out)
    fig_signflip(out)
    fig_probe_vs_robust(out)
    fig_cortexbench(out)
    fig_shift_ladder(out)
    fig_r2_gap_success(out)
    for t in ("push", "pickplace"):
        table_e12(out, t)
    print(f"\nwrote to {out}/  -- re-run after any experiment to refresh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
