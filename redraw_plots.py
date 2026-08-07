"""Redraw every figure used in the paper from the committed JSON artifacts.

This is a pure-visualisation step: it reads the analysis outputs under
``plots_ceiling/`` and ``plots_temporal/`` and regenerates the corresponding
PNG figures with a cleaner, print-ready style (dpi=300, colour-blind-safe
palette, no in-figure titles, CI bands instead of error bars). No modelling
is run and no number changes, so the figures always match the committed data.
The figures regenerated here are exactly the five ``\includegraphics``
targets of the manuscript (fano, mi-spectrum, learning curve, PCA variance,
temporal ceiling).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

from src.ceiling_analysis import fano_max_accuracy

ROOT = Path(__file__).resolve().parent
CEIL = ROOT / "plots_ceiling"
TEMP = ROOT / "plots_temporal"

BLUE = "#4C72B0"
ORANGE = "#DD8452"
GREEN = "#55A868"
PURPLE = "#8172B2"
RED = "#C44E52"
GRAY = "#777777"

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def load(name: str, directory: Path) -> list | dict:
    return json.loads((directory / name).read_text(encoding="utf-8"))


def plot_fano_curve() -> None:
    summary = load("ceiling_summary.json", CEIL)
    h_y = summary["target_entropy_bits"]
    zoo = summary["model_families"]

    fig, ax = plt.subplots(figsize=(8, 5))
    mi = np.linspace(0.0, 0.28, 200)
    ax.plot(mi, [fano_max_accuracy(m, h_y) for m in mi],
            color=BLUE, linewidth=2, label="Fano accuracy ceiling")

    palette = [BLUE, ORANGE, GREEN, PURPLE, RED, "#55A868", "#8172B2", "#CCB974"]
    for i, (name, res) in enumerate(zoo.items()):
        x = res["achieved_mi_bits"]
        y = res["test_accuracy"]
        ax.scatter(x, y, s=52, zorder=3, color=palette[i % len(palette)],
                   edgecolor="white", linewidth=0.6)
        ax.annotate(name.replace("_", " "), (x, y), fontsize=8,
                    xytext=(5, 4), textcoords="offset points")

    ax.axhline(0.5, color=GRAY, linestyle=":", linewidth=1.2)
    ax.set_xlabel("Achieved mutual information $I(p;Y)$ (bits)")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0.5, 1.0)
    ax.set_xlim(0.0, 0.28)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(CEIL / "fano_curve.png", dpi=300)
    plt.close(fig)


def plot_mi_spectrum() -> None:
    summary = load("ceiling_summary.json", CEIL)
    top = summary["top_features_mi"]

    names = list(top.keys())[::-1]
    values = list(top.values())[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(np.arange(len(names)), values, color=BLUE)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(0.0, max(values) * 1.18)
    ax.set_xlabel("Mutual information with outcome (bits)")
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", ha="left", fontsize=7)
    fig.tight_layout()
    fig.savefig(CEIL / "mi_spectrum.png", dpi=300)
    plt.close(fig)


def plot_achieved_mi() -> None:
    summary = load("ceiling_summary.json", CEIL)
    h_y = summary["target_entropy_bits"]
    zoo = summary["model_families"]

    names = list(zoo.keys())
    mis = [zoo[n]["achieved_mi_bits"] for n in names]
    fano_acc = [zoo[n]["fano_max_accuracy"] for n in names]

    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width / 2, mis, width, label="Achieved MI $I(p;Y)$", color=BLUE)
    ax.bar(x + width / 2, fano_acc, width, label="Fano-implied max accuracy",
           color=ORANGE)
    ax.axhline(1.0 - h_y, color=RED, linestyle=":", linewidth=1.2,
               label="Chance accuracy $1-H(Y)$")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Bits (left bars) / Accuracy (right bars)")
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.legend(loc="lower center", fontsize=8)
    fig.tight_layout()
    fig.savefig(CEIL / "achieved_mi.png", dpi=300)
    plt.close(fig)


def plot_learning_curve() -> None:
    summary = load("ceiling_summary.json", CEIL)
    lc = summary["learning_curve"]
    sizes = np.array(lc["train_sizes"])
    mean = np.array(lc["mean_accuracy"])
    std = np.array(lc["std_accuracy"])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(sizes, mean - std, mean + std, color=BLUE, alpha=0.2)
    ax.plot(sizes, mean, "o-", color=BLUE, linewidth=2, markersize=5,
            label="Mean test accuracy")
    ax.axhline(mean[-1], color=RED, linestyle="--", linewidth=1.3,
               label=f"Full-data accuracy ({mean[-1]:.3f})")
    ax.set_xlabel("Training-set fraction")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0.6, 0.78)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(CEIL / "learning_curve.png", dpi=300)
    plt.close(fig)


def plot_pca_variance() -> None:
    summary = load("ceiling_summary.json", CEIL)
    pts = summary["pca_variance_points"]
    pairs = sorted((int(n), float(frac)) for frac, n in pts.items())
    xs = np.array([p[0] for p in pairs])
    ys = np.array([p[1] for p in pairs])

    interp = PchipInterpolator(xs, ys)
    xgrid = np.linspace(1, 40, 300)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xgrid, interp(xgrid), color=BLUE, linewidth=2)
    ax.plot(xs, ys, "o", color=BLUE, markersize=5, zorder=3)
    ax.axhline(0.8, color=RED, linestyle=":", linewidth=1.3)
    ax.axhline(0.95, color=ORANGE, linestyle=":", linewidth=1.3)
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(0, 40)
    ax.text(23, 0.815, "80%", color=RED, fontsize=9)
    ax.text(23, 0.965, "95%", color=ORANGE, fontsize=9)
    fig.tight_layout()
    fig.savefig(CEIL / "pca_variance.png", dpi=300)
    plt.close(fig)


def plot_temporal_ceiling() -> None:
    summary = load("temporal_summary.json", TEMP)
    cohort = load("temporal_fixed_cohort.json", TEMP)
    lead = load("temporal_lead_padded.json", TEMP)

    frames = np.array([r["frame_minutes"] for r in summary])
    acc = np.array([r["voting_accuracy"] for r in summary])
    fano = np.array([r["voting_fano_max_accuracy"] for r in summary])
    mi = np.array([r["voting_achieved_mi"] for r in summary])
    h_y = np.array([r["target_entropy_bits"] for r in summary])
    acc_lo = np.array([r["acc_ci_lower"] for r in summary])
    acc_hi = np.array([r["acc_ci_upper"] for r in summary])
    mi_lo = np.array([r["mi_ci_lower"] for r in summary])
    mi_hi = np.array([r["mi_ci_upper"] for r in summary])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.fill_between(frames, acc_lo, acc_hi, color=BLUE, alpha=0.15)
    ax.plot(frames, acc, "o-", color=BLUE, linewidth=2, markersize=5,
            label="Soft-voting accuracy (all games)")
    ax.plot(frames, fano, "s--", color=ORANGE, linewidth=2, markersize=5,
            label="Fano-implied ceiling")
    for min_dur, color in ((32, GREEN), (36, PURPLE)):
        cf = [r for r in cohort if r["min_duration_min"] == min_dur]
        if cf:
            cf.sort(key=lambda r: r["frame_minutes"])
            xs = [r["frame_minutes"] for r in cf]
            ys = [r["voting_accuracy"] for r in cf]
            ax.plot(xs, ys, "^", color=color, linewidth=1.5, markersize=6,
                    label=f"Fixed cohort (duration >= {min_dur} min)")
    lead.sort(key=lambda r: r["frame_minutes"])
    ax.plot([r["frame_minutes"] for r in lead],
            [r["voting_accuracy"] for r in lead], "d-", color=RED,
            linewidth=1.5, markersize=5, label="Lead-padded (duration >= t+4)")
    ax.axhline(0.5, color=GRAY, linestyle=":")
    ax.set_xlabel("Game minute (frame)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.45, 1.0)
    ax.legend(loc="lower right", fontsize=7)

    ax = axes[1]
    ax.fill_between(frames, mi_lo, mi_hi, color=BLUE, alpha=0.15)
    ax.plot(frames, mi, "o-", color=BLUE, linewidth=2, markersize=5,
            label="Achieved MI $I(p;Y)$ (all games)")
    for min_dur, color in ((32, GREEN), (36, PURPLE)):
        cf = [r for r in cohort if r["min_duration_min"] == min_dur]
        if cf:
            cf.sort(key=lambda r: r["frame_minutes"])
            ax.plot([r["frame_minutes"] for r in cf],
                    [r["voting_achieved_mi"] for r in cf], "^", color=color,
                    linewidth=1.5, markersize=6,
                    label=f"Fixed cohort (duration >= {min_dur} min)")
    ax.plot([r["frame_minutes"] for r in lead],
            [r["voting_achieved_mi"] for r in lead], "d-", color=RED,
            linewidth=1.5, markersize=5, label="Lead-padded (duration >= t+4)")
    ax.plot(frames, h_y, "s--", color=ORANGE, linewidth=2, markersize=5,
            label="Target entropy $H(Y)$")
    ax.set_xlabel("Game minute (frame)")
    ax.set_ylabel("Bits")
    ax.set_ylim(0.0, 1.1)
    ax.legend(loc="lower right", fontsize=7)

    fig.tight_layout()
    fig.savefig(TEMP / "temporal_ceiling.png", dpi=300)
    plt.close(fig)


def main() -> None:
    plot_fano_curve()
    plot_mi_spectrum()
    plot_achieved_mi()
    plot_learning_curve()
    plot_pca_variance()
    plot_temporal_ceiling()
    print("Figures redrawn from committed JSON artifacts (dpi=300).")


if __name__ == "__main__":
    main()
