"""Generate the two supplementary figures at 600 dpi from the committed JSON.

Figure S1: the label-noise sweep under training-label refitting, with the
           degenerate frozen-model identity shown for contrast.
Figure S2: the duration-stratified minute-ten composition.

Runs under the WSL Python (matplotlib 3.10 + numpy). Output goes to
paper/entropy_submission/figures/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
CEIL = ROOT / "plots_ceiling"
TEMP = ROOT / "plots_temporal"
OUT = ROOT / "paper" / "entropy_submission" / "figures"
DPI = 600

BLUE, ORANGE, GREEN, GREY = "#1F4E79", "#C55A11", "#2E7D32", "#7F7F7F"
FAMILY_STYLE = {
    "logistic_regression": ("Logistic regression", BLUE, "o"),
    "lightgbm": ("LightGBM", ORANGE, "s"),
    "soft_voting": ("Soft voting", GREEN, "^"),
}


def load(name: str, base: Path = CEIL) -> dict:
    with open(base / name, encoding="utf-8") as fh:
        return json.load(fh)


def fig_label_noise() -> None:
    d = load("label_noise_audit.json")
    sweep = d["noise_sweep"]
    rhos = sorted(sweep, key=float)
    deg = d["degenerate_test_only_sweep"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    x = [float(r) for r in rhos]

    for fam, (label, colour, marker) in FAMILY_STYLE.items():
        acc = [100 * sweep[r][fam]["mean_accuracy"] for r in rhos]
        acc_sd = [100 * sweep[r][fam]["std_accuracy"] for r in rhos]
        mi = [sweep[r][fam]["mean_achieved_mi"] for r in rhos]
        mi_sd = [sweep[r][fam]["std_achieved_mi"] for r in rhos]
        ax1.errorbar(x, acc, yerr=acc_sd, color=colour, marker=marker,
                     lw=2, ms=5, capsize=3, label=label)
        ax2.errorbar(x, mi, yerr=mi_sd, color=colour, marker=marker,
                     lw=2, ms=5, capsize=3, label=label)

    deg_acc = [100 * deg[r]["mean_accuracy"] for r in rhos]
    ax1.plot(x, deg_acc, ":", color=GREY, lw=2,
             label="Frozen model (algebraic identity)")

    ax1.set_xlabel("Training-label corruption rate $\\rho$")
    ax1.set_ylabel("Test accuracy (%)")
    ax1.set_title("Accuracy under training-label noise")
    ax1.legend(frameon=False, fontsize=8, loc="lower left")
    ax1.grid(alpha=0.25, lw=0.6)

    ax2.set_xlabel("Training-label corruption rate $\\rho$")
    ax2.set_ylabel("Achieved MI (bits)")
    ax2.set_title("Achieved information under training-label noise")
    ax2.legend(frameon=False, fontsize=8)
    ax2.grid(alpha=0.25, lw=0.6)

    fig.tight_layout()
    fig.savefig(OUT / "S1_label_noise.png", dpi=DPI)
    plt.close(fig)
    print("wrote S1_label_noise.png")


def fig_strata() -> None:
    d = load("temporal_duration_stratified.json", TEMP)
    order = ["<24 min", "24-29 min", ">=30 min"]
    labels = ["$<$24 min", "24--29 min", "$\\geq$30 min"]
    n = np.array([d[k]["n_samples"] for k in order], dtype=float)
    share = 100 * n / n.sum()
    acc = np.array([100 * d[k]["voting_accuracy"] for k in order])
    mi = np.array([d[k]["voting_achieved_mi"] for k in order])

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
    colours = [BLUE, ORANGE, GREEN]
    x = np.arange(len(order))

    for ax, vals, title, ylab, fmt in (
        (axes[0], share, "Population composition", "Share of games (%)", "{:.1f}"),
        (axes[1], acc, "Minute-ten accuracy", "Accuracy (%)", "{:.1f}"),
        (axes[2], mi, "Minute-ten information", "Achieved MI (bits)", "{:.3f}"),
    ):
        bars = ax.bar(x, vals, color=colours, width=0.62, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.grid(axis="y", alpha=0.25, lw=0.6)
        for b, v in zip(bars, vals):
            ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=9)
        if ax is axes[0]:
            ax.set_ylim(0, max(share) * 1.22)
        elif ax is axes[1]:
            ax.set_ylim(0, 100)
        else:
            ax.set_ylim(0, max(mi) * 1.22)

    fig.tight_layout()
    fig.savefig(OUT / "S2_strata_composition.png", dpi=DPI)
    plt.close(fig)
    print("wrote S2_strata_composition.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_label_noise()
    fig_strata()
