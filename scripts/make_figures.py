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
them checks.

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

#: Consistent identity per encoder across every figure -- a reader tracking
#: V-JEPA between two plots should not have to re-read the legend.
STYLE = {
    "vjepa2": ("#d62728", "o"), "dinov2": ("#1f77b4", "s"),
    "dinov3": ("#17becf", "s"), "siglip2": ("#2ca02c", "^"),
    "siglip1": ("#98df8a", "^"), "aimv2": ("#9467bd", "D"),
    "clip": ("#ff7f0e", "v"), "clip-large": ("#ffbb78", "v"),
    "vit-in1k": ("#8c564b", "P"), "vit-large": ("#c49c94", "P"),
    "vc1": ("#e377c2", "X"), "vc1-large": ("#f7b6d2", "X"),
    "dinov2-large": ("#aec7e8", "s"), "dinov3-large": ("#9edae5", "s"),
    "random": ("#7f7f7f", "*"),
}


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


# ---------------------------------------------------------------- figure 1
def fig_probe_vs_robustness(out: Path, task: str = "push") -> None:
    """The headline: probe accuracy against robustness, one panel per axis.

    A scatter rather than a bar chart because the claim is about the ABSENCE of
    a relationship, and absence is only legible when the reader can see the
    spread. Bars would invite reading the ordering as the finding.
    """
    blob = load(f"e12_{task}.json")
    if not blob:
        print("  (no e12 json yet)")
        return
    axes = [a for a in blob if isinstance(blob.get(a), dict)
            and blob[a].get("rows") and blob[a].get("rho") is not None]
    if not axes:
        print("  (no valid axes yet)")
        return

    n = len(axes)
    fig, axs = plt.subplots(1, n, figsize=(3.4 * n, 3.4), squeeze=False)
    for ax_i, axis in enumerate(axes):
        ax = axs[0][ax_i]
        rows = blob[axis]["rows"]
        for name, c in rows.items():
            col, mk = style(name)
            ax.scatter(c["probe"], c["held"], c=col, marker=mk, s=70,
                       edgecolors="black", linewidths=0.4, zorder=3,
                       label=name)
        ax.set_xlabel("probe $R^2$ at reference")
        if ax_i == 0:
            ax.set_ylabel("held-out MSE (lower better)")
        ax.set_title(f"{axis}   $\\rho$ = {blob[axis]['rho']:+.2f}")
        ax.grid(alpha=0.25, zorder=0)
        # 1.0 is "no better than predicting the mean action" -- the line that
        # separates a working head from a broken one.
        ax.axhline(1.0, color="red", ls=":", lw=1, zorder=1)
    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(8, len(labels)),
               frameon=False, bbox_to_anchor=(0.5, -0.14), fontsize=8)
    fig.suptitle("Probe accuracy does not predict robustness", y=1.02)
    save(fig, out, f"fig1_probe_vs_robustness_{task}")


# ---------------------------------------------------------------- figure 2
def fig_rank_heatmap(out: Path, task: str = "push") -> None:
    """Encoder rank per axis. The point is that rows are not flat."""
    blob = load(f"e12_{task}.json")
    if not blob:
        return
    axes = [a for a in blob if isinstance(blob.get(a), dict)
            and blob[a].get("rows")]
    if not axes:
        return
    names = sorted({n for a in axes for n in blob[a]["rows"]})
    # Order encoders by probe R^2 so the left column is the ranking a
    # practitioner would have used, and the rest shows what it bought.
    probe = {n: blob[axes[0]]["rows"][n]["probe"]
             for n in names if n in blob[axes[0]]["rows"]}
    names = sorted(probe, key=lambda n: -probe[n])

    grid = np.full((len(names), len(axes)), np.nan)
    for j, axis in enumerate(axes):
        rows = blob[axis]["rows"]
        order = sorted(rows, key=lambda n: rows[n]["held"])
        for r, n in enumerate(order, start=1):
            if n in names:
                grid[names.index(n), j] = r

    fig, ax = plt.subplots(figsize=(1.5 + 1.1 * len(axes), 0.42 * len(names) + 1.4))
    im = ax.imshow(grid, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(axes)))
    ax.set_xticklabels(axes, rotation=30, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{n}  ({probe[n]:.2f})" for n in names], fontsize=8)
    ax.set_ylabel("encoder  (probe $R^2$)")
    for i in range(len(names)):
        for j in range(len(axes)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, int(grid[i, j]), ha="center", va="center",
                        fontsize=8)
    fig.colorbar(im, ax=ax, label="rank (1 = most robust)", shrink=0.8)
    ax.set_title("Rank by axis; rows ordered by probe accuracy")
    save(fig, out, f"fig2_rank_heatmap_{task}")


# ---------------------------------------------------------------- figure 3
def fig_r2_gap_vs_success(out: Path) -> None:
    """Latent gap against TASK SUCCESS -- the only behavioural measurement."""
    blob = load("r2_task_success_reach.json")
    if not blob:
        print("  (no r2 json)")
        return
    rows = blob if isinstance(blob, list) else blob.get("poses", blob.get("rows"))
    if not isinstance(rows, list):
        print("  (unrecognised r2 layout)")
        return
    g = np.array([r["gap"] for r in rows if "gap" in r and "success" in r])
    s = np.array([r["success"] for r in rows if "gap" in r and "success" in r])
    if len(g) < 4:
        return
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    ax.scatter(g, s * 100, c="#1f77b4", s=60, edgecolors="black", linewidths=0.4,
               zorder=3)
    z = np.polyfit(g, s * 100, 1)
    xs = np.linspace(g.min(), g.max(), 50)
    ax.plot(xs, np.polyval(z, xs), color="#d62728", lw=1.5, zorder=2)
    ax.set_xlabel("latent gap (Fréchet)")
    ax.set_ylabel("task success (%)")
    ax.set_title("Gap vs task success")
    ax.grid(alpha=0.25, zorder=0)
    save(fig, out, "fig3_gap_vs_success")


# ---------------------------------------------------------------- figure 4
def fig_shift_ladder(out: Path) -> None:
    """E2's rungs on one axis, with the null as the reference line."""
    blob = load("e2_rungs.json")
    if not blob:
        print("  (no e2 json)")
        return
    rungs = blob["rungs"]
    order = [k for k in ["null", "session", "sim_camera", "camera", "cross_lab",
                         "sim2real_dr", "sim2real"] if k in rungs]
    means = [rungs[k]["mean"] for k in order]
    los = [rungs[k]["min"] for k in order]
    his = [rungs[k]["max"] for k in order]

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    y = np.arange(len(order))
    ax.barh(y, means, color="#4c72b0", zorder=3)
    ax.hlines(y, los, his, color="black", lw=1.4, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.axvline(rungs["null"]["mean"], color="red", ls=":", lw=1.2,
               label="estimator noise floor", zorder=2)
    ax.set_xlabel("Fréchet distance (one estimator, one pca_dim)")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.25, zorder=0)
    save(fig, out, "fig4_shift_ladder")


# ----------------------------------------------------------------- tables
def table_e12(out: Path, task: str = "push") -> None:
    """The main results table as a LaTeX fragment."""
    blob = load(f"e12_{task}.json")
    if not blob:
        return
    axes = [a for a in blob if isinstance(blob.get(a), dict)
            and blob[a].get("rows")]
    if not axes:
        return
    names = sorted({n for a in axes for n in blob[a]["rows"]})
    probe = {}
    for a in axes:
        for n, c in blob[a]["rows"].items():
            probe[n] = c["probe"]
    names = sorted(probe, key=lambda n: -probe[n])

    lines = [r"\begin{tabular}{l r " + "r " * len(axes) + "}", r"\toprule",
             "encoder & probe $R^2$ & " + " & ".join(axes) + r" \\",
             r"\midrule"]
    for n in names:
        cells = []
        for a in axes:
            c = blob[a]["rows"].get(n)
            cells.append(f"{c['held']:.3f}" if c else "--")
        safe = n.replace("_", r"\_")
        lines.append(f"{safe} & {probe[n]:.3f} & " + " & ".join(cells) + r" \\")
    lines += [r"\midrule",
              r"Spearman $\rho$ & & " + " & ".join(
                  f"{blob[a]['rho']:+.3f}" if blob[a].get("rho") is not None
                  else "excl." for a in axes) + r" \\",
              r"\bottomrule", r"\end{tabular}"]
    out.mkdir(parents=True, exist_ok=True)
    (out / f"table_e12_{task}.tex").write_text("\n".join(lines))
    print(f"  table_e12_{task}.tex")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--tasks", nargs="+", default=["push", "pickplace"])
    args = ap.parse_args()
    out = Path(args.out)

    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "figure.dpi": 120,
        "savefig.bbox": "tight", "axes.spines.top": False,
        "axes.spines.right": False,
    })

    print("figures:")
    for t in args.tasks:
        fig_probe_vs_robustness(out, t)
        fig_rank_heatmap(out, t)
        table_e12(out, t)
    fig_r2_gap_vs_success(out)
    fig_shift_ladder(out)
    print(f"\nwrote to {out}/  -- re-run after any experiment to refresh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
