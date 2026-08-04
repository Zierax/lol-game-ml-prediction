"""Model zoo: training, hyperparameter tuning, and evaluation utilities.

Implements three classifiers used in the comparative study — Logistic
Regression, Random Forest, and XGBoost — together with a shared evaluation
routine that computes Accuracy, Precision, Recall, F1-Score, and ROC-AUC and
persists the ROC curves and confusion matrices into ``plots/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

matplotlib.use("Agg")

SEED: int = 42
CV_FOLDS: int = 5

ModelEstimator = BaseEstimator | ClassifierMixin

MetricDict = dict[str, float]

LOGISTIC_REGRESSION = "logistic_regression"
RANDOM_FOREST = "random_forest"
XGBOOST = "xgboost"

LOGISTIC_GRID: dict[str, list[Any]] = {
    "model__C": [0.01, 0.1, 1.0, 10.0],
    "model__solver": ["liblinear"],
    "model__penalty": ["l2"],
}

RANDOM_FOREST_GRID: dict[str, list[Any]] = {
    "model__n_estimators": [200, 400],
    "model__max_depth": [10, 20, None],
    "model__min_samples_leaf": [1, 2],
}

XGBOOST_GRID: dict[str, list[Any]] = {
    "model__n_estimators": [200, 400],
    "model__max_depth": [3, 6],
    "model__learning_rate": [0.05, 0.1],
    "model__subsample": [0.8],
}


def build_pipeline(
    model: ModelEstimator,
    scale: bool = True,
) -> Pipeline:
    """Wrap an estimator in a reproducible pipeline with optional scaling.

    Args:
        model: The base estimator instance.
        scale: Whether to prepend a ``StandardScaler`` step.

    Returns:
        A scikit-learn ``Pipeline`` ready for grid search.
    """
    steps: list[tuple[str, Any]] = []
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps=steps)


def make_models() -> dict[str, ModelEstimator]:
    """Instantiate the three baseline estimators with fixed seeds.

    Returns:
        A mapping of model name to an untuned estimator instance.
    """
    return {
        LOGISTIC_REGRESSION: LogisticRegression(max_iter=2000, random_state=SEED),
        RANDOM_FOREST: RandomForestClassifier(n_jobs=-1, random_state=SEED),
        XGBOOST: XGBClassifier(
            n_jobs=-1,
            random_state=SEED,
            eval_metric="logloss",
            tree_method="hist",
            verbosity=0,
        ),
    }


def get_hyperparameter_grids() -> dict[str, dict[str, list[Any]]]:
    """Return the search spaces used for tuning each model family.

    Returns:
        A mapping of model name to its ``GridSearchCV`` parameter grid.
    """
    return {
        LOGISTIC_REGRESSION: LOGISTIC_GRID,
        RANDOM_FOREST: RANDOM_FOREST_GRID,
        XGBOOST: XGBOOST_GRID,
    }


def tune_model(
    model_name: str,
    pipeline: Pipeline,
    grid: dict[str, list[Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> GridSearchCV:
    """Tune a model family with stratified cross-validated grid search.

    Args:
        model_name: Identifier used for progress reporting.
        pipeline: The estimator pipeline to tune.
        grid: Parameter grid compatible with ``GridSearchCV``.
        X_train: Training feature matrix.
        y_train: Training target vector.

    Returns:
        The fitted ``GridSearchCV`` object exposing ``best_estimator_``.
    """
    splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=grid,
        scoring="accuracy",
        cv=splitter,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search


def evaluate_model(
    model_name: str,
    estimator: ModelEstimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> MetricDict:
    """Compute the standard classification metrics for one model.

    Args:
        model_name: Model identifier (used only for messages).
        estimator: Fitted estimator supporting ``predict_proba``.
        X_test: Test feature matrix.
        y_test: Ground-truth test target.
        y_pred: Predicted class labels.
        y_prob: Predicted positive-class probabilities.

    Returns:
        A dictionary with accuracy, precision, recall, F1, and ROC-AUC.
    """
    metrics: MetricDict = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }
    return metrics


def predict_estimator(
    estimator: ModelEstimator,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return hard and soft predictions from a fitted estimator.

    Args:
        estimator: Fitted estimator supporting ``predict``/``predict_proba``.
        X_test: Test feature matrix.

    Returns:
        A tuple ``(y_pred, y_prob)`` of predicted labels and probabilities.
    """
    y_pred = estimator.predict(X_test)
    y_prob = estimator.predict_proba(X_test)[:, 1]
    return np.asarray(y_pred), np.asarray(y_prob)


def plot_roc_curves(
    roc_curves: dict[str, RocCurveDisplay],
    output_dir: Path,
) -> None:
    """Overlay ROC curves for every model on a single figure.

    Args:
        roc_curves: Mapping of model name to a fitted ``RocCurveDisplay``.
        output_dir: Directory in which the PNG is written.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, display in roc_curves.items():
        display.plot(ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Chance")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Receiver Operating Characteristic Curves by Model")
    ax.legend(loc="lower right")
    fig.savefig(output_dir / "roc_curve.png")
    plt.close(fig)


def plot_confusion_matrix(
    model_name: str,
    estimator: ModelEstimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
) -> None:
    """Render and persist the normalised confusion matrix for one model.

    Args:
        model_name: Model identifier used in the filename.
        estimator: Fitted estimator.
        X_test: Test feature matrix.
        y_test: Ground-truth test target.
        output_dir: Directory in which the PNG is written.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(
        estimator,
        X_test,
        y_test,
        normalize="true",
        cmap="Blues",
        values_format=".2f",
        colorbar=False,
        ax=ax,
    )
    ax.set_title(f"Normalised Confusion Matrix — {model_name}")
    fig.savefig(output_dir / f"confusion_matrix_{model_name}.png")
    plt.close(fig)


def save_metrics(
    results: dict[str, MetricDict],
    output_dir: Path,
) -> Path:
    """Persist the consolidated metrics table as JSON and CSV.

    Args:
        results: Mapping of model name to its metric dictionary.
        output_dir: Directory in which the artifacts are written.

    Returns:
        Path to the JSON metrics file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(results).T
    table.to_csv(output_dir / "model_metrics.csv")
    json_path = output_dir / "model_metrics.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return json_path


def print_metrics_table(results: dict[str, MetricDict]) -> None:
    """Print a human-readable comparison table of all model metrics."""
    table = pd.DataFrame(results).T
    styled = table.round(4)
    print("\n" + "=" * 78)
    print("COMPARATIVE MODEL PERFORMANCE (TEST SET)")
    print("=" * 78)
    print(styled.to_string())
    print("=" * 78 + "\n")
