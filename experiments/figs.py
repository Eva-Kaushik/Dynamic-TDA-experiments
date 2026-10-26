"""Shared plotting helpers. All figures go to figures/<phase>/ at the repo root.

    python experiments/figs.py        # (re)build P1 figures from results/p1/*.csv
"""
import csv
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import dtda  # noqa: E402


def heatmap(M, rows, cols, title, path, center=5.0, vmax=100.0, fmt="{:.0f}"):
    """Red = far above target, white = on target, blue = below."""
    from matplotlib.colors import TwoSlopeNorm
    M = np.asarray(M, float)
    fig, ax = plt.subplots(figsize=(1.3 + 1.35 * len(cols), 0.9 + 0.5 * len(rows)))
    im = ax.imshow(M, cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=center, vmin=0, vmax=vmax), aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8,
                    color="white" if v > 60 else "black")
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def power_curves(rows, conds, series_names, tests, path):
    lock = [l for k, l in conds if k == "lock"]
    amp = [1.0] + [l for k, l in conds if k == "amplitude"]
    fig, axes = plt.subplots(2, len(tests), figsize=(3.6 * len(tests), 6), sharey=True)
    for j, t in enumerate(tests):
        for s in series_names:
            def get(kind, lvl):
                if kind == "amplitude" and lvl == 1.0:
                    kind, lvl = "lock", 0.0
                return next(r["reject"] for r in rows if r["change"] == kind and r["level"] == lvl
                            and r["series"] == s and r["test"] == t)
            ls = "-" if "norm" in s else ("--" if "raw" in s else ":")
            axes[0, j].plot(lock, [get("lock", l) for l in lock], ls, marker="o", ms=3, label=s)
            axes[1, j].plot(amp, [get("amplitude", l) for l in amp], ls, marker="o", ms=3, label=s)
        axes[0, j].set_title(t, fontsize=9)
        axes[0, j].set_xlabel("fraction of channels phase-locked (topology change)", fontsize=8)
        axes[1, j].set_xlabel("amplitude factor (nuisance change)", fontsize=8)
        for a in axes[:, j]:
            a.axhline(0.05, color="grey", lw=.8, ls=":")
            a.set_ylim(-0.02, 1.02)
    axes[0, 0].set_ylabel("rejection rate")
    axes[1, 0].set_ylabel("rejection rate")
    axes[0, -1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def p1_figures():
    d = os.path.join(dtda.RES_DIR, "p1")
    for fname in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if not (fname.startswith("p1_") and fname.endswith(".csv")):
            continue
        with open(os.path.join(d, fname)) as f:
            rows = list(csv.DictReader(f))
        level = fname[3:-4]
        cols = list(dict.fromkeys(f"{r['detrend']} / {r['perm']}" for r in rows))
        nulls = list(dict.fromkeys(r["null"] for r in rows))
        for dim in dict.fromkeys(r["dim"] for r in rows):
            M = [[100 * float(next(r["size"] for r in rows if r["null"] == n and r["dim"] == dim
                                   and f"{r['detrend']} / {r['perm']}" == c)) for c in cols] for n in nulls]
            safe = dim.replace(" ", "").replace("(", "").replace(")", "")
            heatmap(M, nulls, cols, f"P1 ({level}, {dim}): false-alarm rate (%) with no change; target 5%",
                    dtda.fig_path("p1", f"p1_{level}_{safe}.png"))
    print("P1 figures ->", os.path.join(dtda.FIG_DIR, "p1"))


if __name__ == "__main__":
    p1_figures()
