"""Information-ceiling analysis for early-game LoL outcome prediction.

The v1/v2 pipelines established that ~8 model families converge to an
accuracy ceiling near 72%.  This module quantifies *why*: it estimates
the decision-relevant information available at the 10-minute mark and
shows that every model family operates at the Fano-implied information
frontier.

Main quantities computed:

* ``H(Y)``: empirical target entropy.
* Per-feature mutual-information spectrum (signal concentration).
* Achieved mutual information ``I(p_hat; Y)`` per model family (in bits).
* Fano-implied minimum error given the achieved information, and the
  information-efficiency ratio of each family.
* PCA cumulative-variance curve (feature redundancy).
* Learning curves showing accuracy saturation with training-set size.

Fano's inequality (binary case): for a classifier whose outputs carry
``I`` bits of information about the binary target ``Y``, its error rate
``P_e`` satisfies

    H(Y|outputs) <= h(P_e)   =>   P_e >= h^{-1}(H(Y) - I)

where ``h`` is the binary entropy function and ``h^{-1}`` is its inverse
restricted to ``[0, 0.5]``.  Comparing each model's measured error to
this lower bound measures how close the model is to converting its
extracted information into optimal decisions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.data_loader import load_engineered_dataset
from src.models import (
    LIGHTGBM,
    LOGISTIC_REGRESSION,
    RANDOM_FOREST,
    XGBOOST,
    build_pipeline,
    make_models,
)
from src.models_v2 import (
    KNN,
    MLP,
    SVM,
    make_nonlinear_models,
    predict_proba_safe,
)

matplotlib.use("Agg")

SEED: int = 42
TEST_SIZE: float = 0.2
MI_N_NEIGHBORS: int = 5
MI_BIN_COUNTS: tuple[int, ...] = (10, 20, 40)
TOP_N_FEATURES: int = 20

OUTPUT_DIR: Path = Path("plots_ceiling")


# ----------------------------------------------------------------------
# Entropy and Fano bound
# ----------------------------------------------------------------------


def empirical_entropy(y: np.ndarray) -> float:
    """Compute the plug-in (empirical) entropy of the target in bits."""
    counts = np.bincount(y.astype(int), minlength=2)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def binary_entropy(p: float) -> float:
    """Binary entropy h(p) in bits."""
    p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    return float(-p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p))


def inverse_binary_entropy(h: float) -> float:
    """Inverse binary entropy h^{-1}(h) returning the value in [0, 0.5].

    Solves h(p) = h for p in [0, 0.5] via root finding.
    """
    h = float(np.clip(h, 1e-12, 1.0 - 1e-12))
    return float(brentq(lambda p: binary_entropy(p) - h, 1e-12, 0.5 - 1e-12))


def fano_min_error(mi_bits: float, h_y: float) -> float:
    """Minimum error rate for a classifier carrying ``mi_bits`` of information.

    From Fano's inequality for binary targets:
        P_e >= h^{-1}(H(Y) - I(outputs; Y)).
    """
    return inverse_binary_entropy(h_y - mi_bits)


def fano_max_accuracy(mi_bits: float, h_y: float) -> float:
    """Maximum accuracy for a classifier carrying ``mi_bits`` of information."""
    return 1.0 - fano_min_error(mi_bits, h_y)


# ----------------------------------------------------------------------
# Mutual information estimation
# ----------------------------------------------------------------------


def achieved_mutual_information(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    n_bins: int,
) -> float:
    """Mutual information I(binned p_hat; Y) in bits.

    The continuous predicted probability is discretised into ``n_bins``
    equal-width bins; MI is then computed from the joint contingency table
    of (bin, Y).  All quantities are plug-in estimates.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bins[0] = -np.inf
    bins[-1] = np.inf
    bin_idx = np.digitize(y_prob, bins[1:-1])
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    n = len(y_true)
    joint = np.zeros((n_bins, 2), dtype=float)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        joint[b, 0] = float((y_true[mask] == 0).sum())
        joint[b, 1] = float((y_true[mask] == 1).sum())

    joint /= n
    p_bin = joint.sum(axis=1)
    p_y = joint.sum(axis=0)

    mi = 0.0
    for b in range(n_bins):
        if p_bin[b] <= 0:
            continue
        for k in range(2):
            if joint[b, k] <= 0:
                continue
            mi += joint[b, k] * np.log2(
                joint[b, k] / (p_bin[b] * p_y[k])
            )
    return float(mi)


def achieved_mi_robust(y_prob: np.ndarray, y_true: np.ndarray) -> float:
    """Achieved MI averaged across several bin counts (bias robustness)."""
    values = [
        achieved_mutual_information(y_prob, y_true, b)
        for b in MI_BIN_COUNTS
    ]
    return float(np.mean(values))


def mutual_information_spectrum(
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.Series:
    """Per-feature mutual information with the target, sorted descending."""
    mi = mutual_info_classif(
        X, y, random_state=SEED, n_neighbors=MI_N_NEIGHBORS
    )
    return pd.Series(mi, index=X.columns).sort_values(ascending=False)


def pca_cumulative_variance(X: pd.DataFrame) -> np.ndarray:
    """Cumulative explained-variance curve of the standardised features."""
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(random_state=SEED)
    pca.fit(X_scaled)
    return np.cumsum(pca.explained_variance_ratio_)


# ----------------------------------------------------------------------
# Learning curve
# ----------------------------------------------------------------------


def learning_curve(
    model_factory: Callable[[], Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    train_sizes: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0),
    n_repeats: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Accuracy as a function of training fraction (with std across repeats).

    A fixed stratified 20% test set is held out.  For each training size a
    stratified subsample of the remaining data is drawn ``n_repeats`` times
    with different seeds and the model is refit from scratch, so the error
    bars reflect sampling variability at each size.

    Returns ``(train_sizes, mean_acc, std_acc)``.
    """
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=SEED
    )
    train_idx, test_idx = next(splitter.split(X, y))
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    X_pool, y_pool = X.iloc[train_idx], y.iloc[train_idx]

    means: list[float] = []
    stds: list[float] = []
    for frac in train_sizes:
        repeat_acc: list[float] = []
        for r in range(n_repeats):
            if frac >= 1.0:
                sub_train_idx = np.arange(len(X_pool))
            else:
                sub_splitter = StratifiedShuffleSplit(
                    n_splits=1, train_size=frac, random_state=SEED + r
                )
                sub_train_idx, _ = next(sub_splitter.split(X_pool, y_pool))
            X_sub = X_pool.iloc[sub_train_idx]
            y_sub = y_pool.iloc[sub_train_idx]

            est = model_factory()
            est.fit(X_sub, y_sub)
            repeat_acc.append(float(accuracy_score(y_test, est.predict(X_test))))
        means.append(float(np.mean(repeat_acc)))
        stds.append(float(np.std(repeat_acc)))

    return np.asarray(train_sizes), np.asarray(means), np.asarray(stds)


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------


def run_ceiling_analysis() -> dict[str, Any]:
    """Run the full ceiling analysis and persist JSON + figures."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_engineered_dataset()
    y_arr = y.values
    h_y = empirical_entropy(y_arr)
    print(f"Samples: {X.shape[0]}, features: {X.shape[1]}")
    print(f"Target entropy H(Y) = {h_y:.4f} bits")

    # ------------------------------------------------------------------
    # 1. Mutual-information spectrum and redundancy
    # ------------------------------------------------------------------
    print("\n--- Mutual-information spectrum ---")
    mi_spectrum = mutual_information_spectrum(X, y)
    top_n = mi_spectrum.head(TOP_N_FEATURES)
    cumulative_var = pca_cumulative_variance(X)

    print(f"Top-{TOP_N_FEATURES} features by MI (bits):")
    for name, value in top_n.items():
        print(f"  {name:>34s}: {value:.4f}")
    for pct in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        n_pcs = int(np.searchsorted(cumulative_var, pct) + 1)
        print(f"  PCA: {n_pcs} components explain {pct:.0%} variance")

    # ------------------------------------------------------------------
    # 2. Train model zoo and measure achieved information
    # ------------------------------------------------------------------
    print("\n--- Model zoo: achieved mutual information ---")
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=SEED
    )
    train_idx, test_idx = next(splitter.split(X, y))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    zoo: dict[str, Pipeline] = {}
    base = make_models()
    for name in (LOGISTIC_REGRESSION, RANDOM_FOREST, XGBOOST, LIGHTGBM):
        zoo[name] = build_pipeline(base[name])
    nonlinear = make_nonlinear_models()
    for name in (SVM, KNN, MLP):
        zoo[name] = build_pipeline(nonlinear[name])

    zoo_results: dict[str, Any] = {}
    all_prob: list[np.ndarray] = []
    for name, est in zoo.items():
        est.fit(X_train, y_train)
        y_prob = predict_proba_safe(est, X_test)
        y_pred = (y_prob >= 0.5).astype(int)
        acc = float(accuracy_score(y_test, y_pred))
        mi = achieved_mi_robust(y_prob, y_test.values)
        fano_min = fano_min_error(mi, h_y)
        efficiency = (1.0 - acc) / fano_min if fano_min > 0 else float("nan")
        zoo_results[name] = {
            "test_accuracy": acc,
            "test_error": 1.0 - acc,
            "achieved_mi_bits": mi,
            "fano_min_error": fano_min,
            "fano_max_accuracy": 1.0 - fano_min,
            "information_efficiency": efficiency,
        }
        all_prob.append(y_prob)
        print(
            f"  {name:>20s}: acc={acc:.4f}  I(p;Y)={mi:.4f} bits  "
            f"Fano min err={fano_min:.4f}  efficiency={efficiency:.3f}"
        )

    # Soft-voting ensemble (mean of calibrated-compatible probabilities)
    voting_prob = np.mean(np.column_stack(all_prob), axis=1)
    voting_pred = (voting_prob >= 0.5).astype(int)
    voting_acc = float(accuracy_score(y_test, voting_pred))
    voting_mi = achieved_mi_robust(voting_prob, y_test.values)
    fano_voting = fano_min_error(voting_mi, h_y)
    zoo_results["soft_voting"] = {
        "test_accuracy": voting_acc,
        "test_error": 1.0 - voting_acc,
        "achieved_mi_bits": voting_mi,
        "fano_min_error": fano_voting,
        "fano_max_accuracy": 1.0 - fano_voting,
        "information_efficiency": (1.0 - voting_acc) / fano_voting,
    }
    print(
        f"  {'soft_voting':>20s}: acc={voting_acc:.4f}  I(p;Y)={voting_mi:.4f} "
        f"bits  Fano min err={fano_voting:.4f}"
    )

    # ------------------------------------------------------------------
    # 3. Learning curves (saturation evidence)
    # ------------------------------------------------------------------
    print("\n--- Learning curve (LGBM) ---")
    def _lgbm_factory() -> Pipeline:
        return build_pipeline(
            LGBMClassifier(
                n_estimators=300,
                learning_rate=0.1,
                num_leaves=31,
                n_jobs=1,
                random_state=SEED,
                verbose=-1,
            )
        )

    lc_sizes, lc_mean, lc_std = learning_curve(_lgbm_factory, X, y)
    for s, m, sd in zip(lc_sizes, lc_mean, lc_std):
        print(f"  train_size={s:.0%}: acc={m:.4f} ± {sd:.4f}")

    # ------------------------------------------------------------------
    # 4. Persist JSON
    # ------------------------------------------------------------------
    summary: dict[str, Any] = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "target_entropy_bits": h_y,
        "top_features_mi": {k: float(v) for k, v in top_n.items()},
        "pca_variance_points": {
            pct: int(np.searchsorted(cumulative_var, pct) + 1)
            for pct in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
        },
        "model_families": zoo_results,
        "learning_curve": {
            "train_sizes": [float(s) for s in lc_sizes],
            "mean_accuracy": [float(m) for m in lc_mean],
            "std_accuracy": [float(s) for s in lc_std],
        },
    }
    json_path = OUTPUT_DIR / "ceiling_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nCeiling summary saved to {json_path}")

    _plot_mi_spectrum(top_n, mi_spectrum)
    _plot_achieved_mi(zoo_results, h_y)
    _plot_fano_curve(zoo_results, h_y)
    _plot_learning_curve(lc_sizes, lc_mean, lc_std)
    _plot_pca_variance(cumulative_var)
    print(f"Figures saved to {OUTPUT_DIR}/")

    return summary


# ----------------------------------------------------------------------
# Plotting helpers
# ----------------------------------------------------------------------


def _plot_mi_spectrum(
    top_n: pd.Series,
    full_spectrum: pd.Series,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(
        np.arange(len(top_n)),
        top_n.values,
        color="#4C72B0",
    )
    ax.set_yticks(np.arange(len(top_n)))
    ax.set_yticklabels(top_n.index)
    ax.invert_yaxis()
    ax.set_xlabel("Mutual information with outcome (bits)")
    ax.set_title("Top-20 feature mutual information (10-min snapshot)")
    # Annotate signal concentration: number of features above small MI
    n_above = int((full_spectrum > 0.02).sum())
    ax.text(
        0.99,
        0.01,
        f"{n_above} of {len(full_spectrum)} features have MI > 0.02 bits",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#333333",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "mi_spectrum.png", dpi=150)
    plt.close(fig)


def _plot_achieved_mi(
    zoo_results: dict[str, Any],
    h_y: float,
) -> None:
    names = list(zoo_results.keys())
    mis = [zoo_results[n]["achieved_mi_bits"] for n in names]
    fano_acc = [zoo_results[n]["fano_max_accuracy"] for n in names]

    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, mis, width, label="Achieved MI I(p;Y)", color="#4C72B0")
    ax.bar(x + width / 2, fano_acc, width, label="Fano-implied max accuracy",
           color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Bits (left bars) / Accuracy (right bars)")
    ax.axhline(1.0 - h_y, color="#C44E52", linestyle=":", linewidth=1.2,
               label="Chance accuracy (1-H(Y))")
    ax.legend(loc="lower center", fontsize=8)
    ax.set_title("Achieved mutual information and Fano-implied ceiling per family")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "achieved_mi.png", dpi=150)
    plt.close(fig)


def _plot_fano_curve(
    zoo_results: dict[str, Any],
    h_y: float,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    mi_range = np.linspace(0.0, 0.4, 200)
    acc_ceiling = np.array([fano_max_accuracy(m, h_y) for m in mi_range])
    ax.plot(mi_range, acc_ceiling, color="#4C72B0", linewidth=2,
            label="Fano accuracy ceiling")

    for name, res in zoo_results.items():
        ax.scatter(
            res["achieved_mi_bits"],
            res["test_accuracy"],
            s=45,
            zorder=3,
        )
        ax.annotate(
            name.replace("_", " "),
            (res["achieved_mi_bits"], res["test_accuracy"]),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )

    ax.axhline(0.5, color="#C44E52", linestyle=":", label="Chance (0.5)")
    ax.set_xlabel("Achieved mutual information I(p;Y) (bits)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Model accuracy vs the Fano-implied information ceiling")
    ax.set_ylim(0.5, 1.0)
    ax.set_xlim(0.0, 0.4)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fano_curve.png", dpi=150)
    plt.close(fig)


def _plot_learning_curve(
    sizes: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(
        sizes,
        mean,
        yerr=std,
        fmt="o-",
        color="#4C72B0",
        capsize=4,
        linewidth=2,
    )
    ax.set_xlabel("Training-set fraction")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Learning curve: accuracy saturates with training data")
    ax.set_ylim(0.6, 0.78)
    ax.axhline(mean[-1], color="#C44E52", linestyle="--",
               label=f"Full-data accuracy ({mean[-1]:.3f})")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "learning_curve.png", dpi=150)
    plt.close(fig)


def _plot_pca_variance(cumulative_var: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        np.arange(1, len(cumulative_var) + 1),
        cumulative_var,
        color="#4C72B0",
        linewidth=2,
    )
    ax.axhline(0.8, color="#C44E52", linestyle=":", label="80% variance")
    ax.axhline(0.95, color="#DD8452", linestyle=":", label="95% variance")
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("Feature redundancy: variance concentrates in few components")
    ax.set_xlim(1, 40)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pca_variance.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run_ceiling_analysis()
