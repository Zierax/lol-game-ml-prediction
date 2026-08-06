"""Model zoo: training, hyperparameter tuning, and evaluation utilities.

Implements four classifiers — Logistic Regression, Random Forest, XGBoost,
and LightGBM — together with a stacking ensemble that combines all four.
Evaluation computes Accuracy, Precision, Recall, F1-Score, ROC-AUC, and
Brier score, and persists ROC curves and confusion matrices into ``plots/``.
"""

import json
import os
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

matplotlib.use("Agg")

SEED: int = 42
CV_FOLDS: int = 5

# Parallelism for the CV search harness.  Inner estimators are
# single-threaded and seeded, so parallelising candidate fits does not
# change results (each fit is independent) — it only reduces wall time.
SEARCH_JOBS: int = int(os.environ.get("SEARCH_JOBS", "1"))

ModelEstimator = BaseEstimator | ClassifierMixin

MetricDict = dict[str, float]

LOGISTIC_REGRESSION = "logistic_regression"
RANDOM_FOREST = "random_forest"
XGBOOST = "xgboost"
LIGHTGBM = "lightgbm"
STACKING = "stacking"

# ----------------------------------------------------------------------
# Hyperparameter grids — deliberately wide to allow genuine tuning
# ----------------------------------------------------------------------

LOGISTIC_GRID: dict[str, list[Any]] = {
    "model__C": [0.01, 0.1, 1.0, 10.0],
    "model__penalty": ["l1", "l2"],
    "model__solver": ["liblinear"],
}

RANDOM_FOREST_GRID: dict[str, list[Any]] = {
    "model__n_estimators": [300, 500],
    "model__max_depth": [15, None],
    "model__min_samples_leaf": [1, 2],
}

XGBOOST_GRID: dict[str, list[Any]] = {
    "model__n_estimators": [300, 500],
    "model__max_depth": [5, 7],
    "model__learning_rate": [0.01, 0.1],
    "model__reg_alpha": [0, 0.1],
}

LIGHTGBM_GRID: dict[str, list[Any]] = {
    "model__n_estimators": [300, 500],
    "model__max_depth": [5, -1],
    "model__learning_rate": [0.01, 0.1],
    "model__num_leaves": [31, 63],
}


# ----------------------------------------------------------------------
# Pipeline construction
# ----------------------------------------------------------------------


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
    """Instantiate the base estimators with fixed seeds.

    All estimators are fully deterministic: ``n_jobs=1`` everywhere and
    ``deterministic=True`` for LightGBM, so repeated runs produce
    byte-identical outputs (run-to-run reproducibility).

    Returns:
        A mapping of model name to an untuned estimator instance.
    """
    return {
        LOGISTIC_REGRESSION: LogisticRegression(
            max_iter=5000, random_state=SEED
        ),
        RANDOM_FOREST: RandomForestClassifier(
            n_estimators=500, n_jobs=1, random_state=SEED
        ),
        XGBOOST: XGBClassifier(
            n_jobs=1,
            random_state=SEED,
            eval_metric="logloss",
            tree_method="hist",
            verbosity=0,
        ),
        LIGHTGBM: LGBMClassifier(
            n_jobs=1,
            deterministic=True,
            random_state=SEED,
            verbose=-1,
            importance_type="gain",
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
        LIGHTGBM: LIGHTGBM_GRID,
    }


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------


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
    splitter = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=SEED
    )
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=grid,
        scoring="accuracy",
        cv=splitter,
        n_jobs=SEARCH_JOBS,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search


def build_stacking_ensemble(
    base_estimators: dict[str, Pipeline],
) -> StackingClassifier:
    """Build a stacking ensemble that combines all tuned base estimators.

    The meta-learner is a regularised logistic regression trained on
    out-of-fold predictions from the base estimators.

    Args:
        base_estimators: Mapping of model name to fitted ``Pipeline``.

    Returns:
        A ``StackingClassifier`` ready for cross-validated evaluation.
    """
    estimators = [(name, est) for name, est in base_estimators.items()]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(
            C=1.0, max_iter=5000, random_state=SEED
        ),
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED),
        stack_method="predict_proba",
        n_jobs=1,
    )


# ----------------------------------------------------------------------
# Prediction & evaluation
# ----------------------------------------------------------------------


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
        A dictionary with accuracy, precision, recall, F1, ROC-AUC, and
        Brier score.
    """
    metrics: MetricDict = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "brier": float(brier_score_loss(y_test, y_prob)),
    }
    return metrics


# ----------------------------------------------------------------------
# Visualisation
# ----------------------------------------------------------------------


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


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------


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
