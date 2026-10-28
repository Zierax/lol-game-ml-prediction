"""Regenerate every manuscript figure at 600 dpi from the committed artifacts.

Pure visualisation: reads the JSON produced by ``run_ceiling.py``,
``run_bounds_audit.py``, ``run_temporal_ceiling.py`` and
``run_temporal_accrual.py`` and redraws the figures used by
``paper/entropy_submission/main.tex``.  No modelling is rerun, so the figures
cannot disagree with the reported numbers.

Entropy requires a minimum of 600 dpi, so every figure is written at 600 dpi.

Usage:
    python redraw_paper_figures.py
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
OUT = ROOT / "paper" / "entropy_submission" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

DPI = 600

BLUE, ORANGE, GREEN, PURPLE, RED, GRAY = (
    "#4C72B0", "#DD8452", "#55A868", "#8172B2", "#C44E52", "#777777"
)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load(name: str, directory: Path):
    return json.loads((directory / name).read_text(encoding="utf-8"))


def fig_mi_spectrum() -> None:
    s = load("ceiling_summary.json", CEIL)
    top = s["top_features_mi"]
    names = list(top.keys())[::-1]
    vals = list(top.values())[::-1]

    fig, ax = plt.subplots(figsize=(8, 5.2))
    bars = ax.barh(np.arange(len(names)), vals, color=BLUE)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(0, max(vals) * 1.20)
    ax.set_xlabel("Mutual information with outcome (bits)")
    for b, v in zip(bars, vals):
        ax.text(b.get_width() + 0.004, b.get_y() + b.get_height() / 2,
                f"{v:.3f}", va="center", ha="left", fontsize=7)
    null = s["mi_permutation_null"]
    z_null = ((null["observed_max"] - null["null_max_mean"])
              / null["null_max_std"])
    ax.text(0.99, 0.02,
            f"permutation null of the max over {null['n_features']} features: "
            f"{null['null_max_mean']:.3f} $\\pm$ {null['null_max_std']:.3f} bits\n"
            f"observed {null['observed_max']:.3f} bits "
            f"({z_null:.0f} SD, "
            f"{null['observed_over_null95_ratio']:.1f}$\\times$ null 95th pct)\n"
            f"maxT permutation p $\\leq$ {null['p_value_maxT']:.3f} "
            f"(floor $1/(B{chr(43)}1)$, $B={null['n_permutations']}$)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
            color="#333333",
            bbox=dict(facecolor="#F5F5F5", edgecolor="#BBBBBB", pad=4))
    fig.tight_layout()
    fig.savefig(OUT / "mi_spectrum.png", dpi=DPI)
    plt.close(fig)


def fig_achieved_mi() -> None:
    """Two panels, separate axes: the Fano bound is a function of the MI."""
    s = load("ceiling_summary.json", CEIL)
    fam = s["model_families"]
    h_y = s["target_entropy_bits_test"]
    names = list(fam)
    mis = [fam[n]["achieved_mi_bits"] for n in names]
    cis = [fam[n]["bootstrap_ci"]["achieved_mi"] for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    x = np.arange(len(names))

    err = np.array([[m - c["ci_lower"] for m, c in zip(mis, cis)],
                    [c["ci_upper"] - m for m, c in zip(mis, cis)]])
    ax1.bar(x, mis, 0.6, color=BLUE, yerr=err, capsize=3,
            error_kw={"lw": 0.9})
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax1.set_ylabel("Achieved mutual information $I(\\hat{p};Y)$ (bits)")
    ax1.set_title("Achieved information per family (95% bootstrap CI)")

    accs = [fam[n]["test_accuracy"] for n in names]
    amax = [fam[n]["fano_max_accuracy"] for n in names]
    w = 0.38
    ax2.bar(x - w / 2, accs, w, color=BLUE, label="Test accuracy")
    ax2.bar(x + w / 2, amax, w, color=ORANGE,
            label="Fano $A_{\\max}$ for the extracted information")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0.65, 0.82)
    ax2.legend(loc="lower left", fontsize=7.5)
    ax2.set_title("Accuracy against the Fano bound")

    fig.tight_layout()
    fig.savefig(OUT / "achieved_mi.png", dpi=DPI)
    plt.close(fig)


def fig_fano_curve() -> None:
    s = load("ceiling_summary.json", CEIL)
    h_y = s["target_entropy_bits_test"]
    fam = s["model_families"]

    fig, ax = plt.subplots(figsize=(8, 5.2))
    mi = np.linspace(0.0, 0.30, 300)
    ax.plot(mi, [fano_max_accuracy(m, h_y) for m in mi],
            color=BLUE, lw=2, label="Fano bound $A_{\\max}(I)$")
    pal = [BLUE, ORANGE, GREEN, PURPLE, RED, "#55A868", "#8172B2", "#CCB974"]
    for i, (n, r) in enumerate(fam.items()):
        x, y = r["achieved_mi_bits"], r["test_accuracy"]
        ax.scatter(x, y, s=60, zorder=3, color=pal[i % len(pal)],
                   edgecolor="white", lw=0.6)
        ax.annotate(n.replace("_", " "), (x, y), fontsize=7.5,
                    xytext=(5, 4), textcoords="offset points")
    ax.axhline(0.5, color=GRAY, ls=":", lw=1.2, label="Chance")
    ax.set_xlabel("Achieved mutual information $I(\\hat{p};Y)$ (bits)")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0.5, 1.0)
    ax.set_xlim(0.0, 0.30)
    ax.legend(loc="lower right", fontsize=8)
    ax.text(0.02, 0.06,
            "$H(Y) = 1.000$ bit; every family extracts\n"
            "less than a quarter of the outcome entropy",
            transform=ax.transAxes, fontsize=8, color="#333333",
            bbox=dict(facecolor="#F5F5F5", edgecolor="#BBBBBB", pad=4))
    fig.tight_layout()
    fig.savefig(OUT / "fano_curve.png", dpi=DPI)
    plt.close(fig)


def fig_learning_curve() -> None:
    s = load("ceiling_summary.json", CEIL)
    lcs = s["learning_curves"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for (name, d), col in zip(lcs.items(), (BLUE, ORANGE, GREEN)):
        sz = np.array(d["train_sizes"])
        m = np.array(d["mean_accuracy"])
        sd = np.array(d["std_accuracy"])
        ax1.plot(sz * 100, m, "o-", color=col, lw=2, ms=4, label=name)
        ax1.fill_between(sz * 100, m - sd, m + sd, color=col, alpha=0.15)
        mm = np.array(d["mean_achieved_mi"])
        msd = np.array(d["std_achieved_mi"])
        ax2.plot(sz * 100, mm, "o-", color=col, lw=2, ms=4, label=name)
        ax2.fill_between(sz * 100, mm - msd, mm + msd, color=col, alpha=0.15)
    for ax, ylab, title in (
        (ax1, "Test accuracy", "Accuracy vs training-set size"),
        (ax2, "Achieved MI (bits)", "Achieved information vs training-set size"),
    ):
        ax.set_xlabel("Training data (% of the training pool)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=8)
    ax1.set_ylim(0.66, 0.74)
    ax2.set_ylim(0.10, 0.24)
    verdict = "\n".join(
        f"{n.replace('_', ' ')}: "
        + (
            "FLAT (neg. curvature, p="
            + f"{v['mi_curvature']['quadratic_p_value']:.4f})"
            if v["mi_curvature"]["quadratic_p_value"] < 0.05
            and v["mi_curvature"]["quadratic_coefficient"] < 0
            else ("still rising (p="
                  + f"{v['mi_slope_wls_p']:.4f})"
                  if v["mi_slope_wls_p"] < 0.05
                  else "flat (WLS slope n.s.)")
        )
        for n, v in lcs.items())
    ax2.text(0.02, 0.97,
              "Saturation test (information axis):\n" + verdict,
              transform=ax2.transAxes, va="top", fontsize=7.5, color="#1F3864",
              bbox=dict(facecolor="#F0F4FB", edgecolor="#AABBDD", pad=4))
    fig.tight_layout()
    fig.savefig(OUT / "learning_curve.png", dpi=DPI)
    plt.close(fig)


def fig_pca_variance() -> None:
    s = load("ceiling_summary.json", CEIL)
    audit = s["design_matrix_audit"]
    pts = s["pca_variance_points"]
    pairs = sorted((int(n), float(f)) for f, n in pts.items())
    xs = np.array([p[0] for p in pairs])
    ys = np.array([p[1] for p in pairs])
    interp = PchipInterpolator(xs, ys)
    xg = np.linspace(1, 40, 300)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(xg, interp(xg), color=BLUE, lw=2, label="Cumulative variance")
    ax.plot(xs, ys, "o", color=BLUE, ms=5, zorder=3)
    ax.axhline(0.8, color=RED, ls=":", lw=1.3)
    ax.axhline(0.95, color=ORANGE, ls=":", lw=1.3)
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 40)
    ax.text(23, 0.815, "80%", color=RED, fontsize=9)
    ax.text(23, 0.965, "95%", color=ORANGE, fontsize=9)
    ax.text(0.02, 0.95,
            f"design matrix: rank {audit['numerical_rank']} of "
            f"{audit['n_columns']}\n"
            f"({audit['rank_deficiency']} exactly dependent columns)",
            transform=ax.transAxes, va="top", fontsize=8, color="#333333",
            bbox=dict(facecolor="#F5F5F5", edgecolor="#BBBBBB", pad=4))
    fig.tight_layout()
    fig.savefig(OUT / "pca_variance.png", dpi=DPI)
    plt.close(fig)


def fig_temporal() -> None:
    ts = load("temporal_summary.json", TEMP)
    co = load("temporal_fixed_cohort.json", TEMP)
    lp = load("temporal_lead_padded.json", TEMP)
    ac = load("temporal_accrual.json", TEMP)
    ds = load("temporal_duration_stratified.json", TEMP)

    fr = np.array([r["frame_minutes"] for r in ts])
    acc = np.array([r["voting_accuracy"] for r in ts])
    fano = np.array([r["voting_fano_max_accuracy"] for r in ts])
    mi = np.array([r["voting_achieved_mi"] for r in ts])
    al = np.array([r["acc_ci_lower"] for r in ts])
    ah = np.array([r["acc_ci_upper"] for r in ts])
    ml = np.array([r["mi_ci_lower"] for r in ts])
    mh = np.array([r["mi_ci_upper"] for r in ts])
    ns = np.array([r["n_samples"] for r in ts])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axes[0]
    ax.fill_between(fr, al, ah, color=BLUE, alpha=0.18)
    ax.plot(fr, acc, "o-", color=BLUE, lw=2, ms=4,
            label="All survivors (soft vote)")
    ax.plot(fr, fano, "s--", color=ORANGE, lw=2, ms=4,
            label="Fano $A_{\\max}$ for the same information")
    for min_dur, col in ((32, GREEN), (36, PURPLE)):
        cf = sorted((r for r in co if r["min_duration_min"] == min_dur),
                    key=lambda r: r["frame_minutes"])
        if cf:
            ax.plot([r["frame_minutes"] for r in cf],
                    [r["voting_accuracy"] for r in cf], "^", color=col,
                    lw=1.5, ms=5,
                    label=f"Fixed cohort, duration $\\geq${min_dur} min")
    lps = sorted(lp, key=lambda r: r["frame_minutes"])
    ax.plot([r["frame_minutes"] for r in lps],
            [r["voting_accuracy"] for r in lps], "d-", color=RED, lw=1.5,
            ms=4, label="Lead-padded ($t+4$ min)")
    ax.axhline(0.5, color=GRAY, ls=":")
    ax.set_xlabel("Game minute")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.45, 1.0)
    ax.set_title("Accuracy: accrual versus survivorship")
    ax.legend(loc="lower right", fontsize=6.5)

    ax = axes[1]
    ax.fill_between(fr, ml, mh, color=BLUE, alpha=0.18)
    ax.plot(fr, mi, "o-", color=BLUE, lw=2, ms=4,
            label="Achieved $I(\\hat{p};Y)$, all survivors")
    for min_dur, col in ((32, GREEN), (36, PURPLE)):
        cf = sorted((r for r in co if r["min_duration_min"] == min_dur),
                    key=lambda r: r["frame_minutes"])
        if cf:
            ax.plot([r["frame_minutes"] for r in cf],
                    [r["voting_achieved_mi"] for r in cf], "^", color=col,
                    lw=1.5, ms=5,
                    label=f"Fixed cohort $\\geq${min_dur} min")
    ax.plot([r["frame_minutes"] for r in lps],
            [r["voting_achieved_mi"] for r in lps], "d-", color=RED, lw=1.5,
            ms=4, label="Lead-padded ($t+4$ min)")
    a, b = ac["mi_intercept_bits"], ac["mi_slope_bits_per_minute"]
    grid = np.linspace(10, 30, 100)
    ax.plot(grid, np.clip(a + b * grid, 0, ac["target_entropy_bits"]), "-",
            color="#333333", lw=1.4,
            label=f"linear accrual fit: {b:.4f} bits/min")
    ax.set_xlabel("Game minute")
    ax.set_ylabel("Bits")
    ax.set_ylim(0, 0.62)
    ax.set_title("Achieved information over the match")
    ax.legend(loc="lower right", fontsize=6.5)

    ax = axes[2]
    ax.plot(fr, ns / ns[0] * 100, "o-", color=BLUE, lw=2, ms=4,
            label="All survivors (relative to minute 10)")
    for min_dur, col in ((32, GREEN), (36, PURPLE)):
        cf = sorted((r for r in co if r["min_duration_min"] == min_dur),
                    key=lambda r: r["frame_minutes"])
        if cf:
            ax.plot([r["frame_minutes"] for r in cf],
                    [100.0 * r["n_samples"] / cf[0]["n_samples"]
                     for r in cf], "^", color=col, lw=1.5, ms=5,
                    label=f"Fixed cohort $\\geq${min_dur} min")
    ax.plot([r["frame_minutes"] for r in lps],
            [100.0 * r["n_samples"] / lps[0]["n_samples"] for r in lps],
            "d-", color=RED, lw=1.5, ms=4, label="Lead-padded")
    ax.set_xlabel("Game minute")
    ax.set_ylabel("Surviving frames (% of that series at minute 10)")
    ax.set_title("Survivorship: the sample shrinks as the match proceeds")
    ax.legend(loc="upper right", fontsize=6.5)

    fig.tight_layout()
    fig.savefig(OUT / "temporal_ceiling.png", dpi=DPI)
    plt.close(fig)


def fig_minute10_strata() -> None:
    """Minute-10 information content stratified by final game duration."""
    ds = load("temporal_duration_stratified.json", TEMP)
    rows = ds.get("strata", ds)
    items = rows.items() if isinstance(rows, dict) else [
        (r.get("label", "?"), r) for r in rows
    ]
    labels = [str(k) for k, _ in items]
    accs = [v["voting_accuracy"] for _, v in items]
    mis = [v["voting_achieved_mi"] for _, v in items]
    ns = [int(v["n_samples"]) for _, v in items]

    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = np.arange(len(labels))
    w = 0.38
    b1 = ax.bar(x - w / 2, accs, w, color=BLUE, label="Test accuracy")
    ax2 = ax.twinx()
    b2 = ax2.bar(x + w / 2, mis, w, color=ORANGE,
                 label="Achieved MI (bits)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n(n = {n:,})" for l, n in zip(labels, ns)],
                       fontsize=8)
    ax.set_ylabel("Accuracy")
    ax2.set_ylabel("Achieved MI (bits)")
    ax.set_ylim(0.4, 1.0)
    ax2.set_ylim(0, 0.8)
    ax.set_title("The minute-10 snapshot is not uniformly uninformative")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "minute10_strata.png", dpi=DPI)
    plt.close(fig)


def main() -> None:
    fig_mi_spectrum()
    fig_achieved_mi()
    fig_fano_curve()
    fig_learning_curve()
    fig_pca_variance()
    fig_temporal()
    fig_minute10_strata()
    print(f"Figures written to {OUT} at {DPI} dpi:")
    for p in sorted(OUT.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
