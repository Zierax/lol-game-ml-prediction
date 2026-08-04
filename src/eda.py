"""Exploratory data analysis and diagnostic visualisations.

Produces the correlation matrix, class-balance distribution, and a
feature-importance ranking that are referenced throughout the research
paper and stored inside the ``plots/`` directory.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit

matplotlib.use("Agg")

SEED: int = 42
TOP_N_FEATURES: int = 20

PLOT_THEME_COLORS: tuple[str, str] = ("#4C72B0", "#C44E52")


def _style_axes() -> None:
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.bbox"] = "tight"


def plot_class_distribution(
    y: pd.Series,
    output_dir: Path,
) -> None:
    """Render the blue-side win-rate (class balance) bar chart.

    Args:
        y: Binary target vector.
        output_dir: Directory in which the PNG is written.
    """
    _style_axes()
    counts = y.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        counts.index.astype(str),
        counts.values,
        color=PLOT_THEME_COLORS,
        edgecolor="black",
        linewidth=0.6,
    )
    ax.bar_label(bars, fmt="%d")
    ax.set_xlabel("Match Outcome (blue side)")
    ax.set_ylabel("Number of Matches")
    ax.set_title("Class Distribution of the Target Variable")
    ax.set_ylim(0, counts.max() * 1.12)
    fig.savefig(output_dir / "class_distribution.png")
    plt.close(fig)


def plot_correlation_matrix(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
) -> None:
    """Render the feature-to-target correlation heatmap.

    Args:
        X: Feature matrix.
        y: Target vector.
        output_dir: Directory in which the PNG is written.
    """
    _style_axes()
    data = X.copy()
    data["blueWins"] = y
    correlations = data.corr(method="pearson")
    target_correlations = correlations["blueWins"].drop("blueWins")
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(
        data.corr(),
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        annot=False,
        fmt=".2f",
        linewidths=0.4,
        ax=ax,
    )
    ax.set_title("Pearson Correlation Matrix of Predictive Features")
    fig.savefig(output_dir / "correlation_matrix.png")
    plt.close(fig)

    ranked = target_correlations.abs().sort_values(ascending=False)
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    bars = ax2.barh(
        ranked.index[:TOP_N_FEATURES][::-1],
        ranked.values[:TOP_N_FEATURES][::-1],
        color=PLOT_THEME_COLORS[0],
        edgecolor="black",
        linewidth=0.5,
    )
    ax2.set_xlabel("Absolute Pearson Correlation with blueWins")
    ax2.set_title("Top Features by Absolute Correlation with Match Outcome")
    fig2.savefig(output_dir / "target_correlations.png")
    plt.close(fig2)


def compute_feature_importance(
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.Series:
    """Rank features by out-of-fold random-forest Gini importance.

    Args:
        X: Feature matrix.
        y: Target vector.

    Returns:
        A Series of mean decrease in impurity, sorted descending.
    """
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=0.2, random_state=SEED
    )
    train_index, _ = next(splitter.split(X, y))
    X_train = X.iloc[train_index]
    y_train = y.iloc[train_index]

    forest = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=SEED,
    )
    forest.fit(X_train, y_train)
    importance = pd.Series(
        forest.feature_importances_,
        index=X.columns,
        name="gini_importance",
    ).sort_values(ascending=False)
    return importance


def plot_feature_importance(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
) -> pd.Series:
    """Render the random-forest Gini importance bar chart.

    Args:
        X: Feature matrix.
        y: Target vector.
        output_dir: Directory in which the PNG is written.
    """
    _style_axes()
    importance = compute_feature_importance(X, y)
    top = importance.head(TOP_N_FEATURES)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        top.index[::-1],
        top.values[::-1],
        color=PLOT_THEME_COLORS[0],
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xlabel("Gini Importance (Mean Decrease in Impurity)")
    ax.set_title("Random Forest Feature Importance Ranking")
    fig.savefig(output_dir / "feature_importance.png")
    plt.close(fig)
    return importance


def generate_eda_report(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
) -> pd.Series:
    """Run the full EDA suite and persist every diagnostic figure.

    Args:
        X: Feature matrix.
        y: Target vector.
        output_dir: Directory in which the PNG figures are written.

    Returns:
        The feature-importance Series for downstream reporting.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_class_distribution(y, output_dir)
    plot_correlation_matrix(X, y, output_dir)
    importance = plot_feature_importance(X, y, output_dir)
    return importance
