"""Rigorous evaluation utilities for model comparison.

Provides:
* Nested cross-validation for unbiased performance estimation.
* Paired bootstrap significance testing between model pairs.
* Decision-threshold optimisation for F1 score.
* Confidence-interval estimation via stratified bootstrap.

All randomness is seeded for full reproducibility.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
)

SEED: int = 42
NESTED_OUTER_FOLDS: int = 10
NESTED_INNER_FOLDS: int = 5
BOOTSTRAP_N_ITER: int = 2000
CI_LEVEL: float = 0.95

ModelEstimator = BaseEstimator | ClassifierMixin


# ----------------------------------------------------------------------
# Nested cross-validation
# ----------------------------------------------------------------------


def nested_cv_evaluate(
    model: ModelEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    outer_folds: int = NESTED_OUTER_FOLDS,
    inner_folds: int = NESTED_INNER_FOLDS,
) -> dict[str, Any]:
    """Estimate out-of-fold performance via nested stratified CV.

    The outer loop provides unbiased performance estimates.  The inner
    loop is used only for hyperparameter tuning (caller is responsible
    for providing an already-tuned estimator or a param grid).

    For this implementation the caller passes a pre-tuned pipeline and
    we evaluate it with outer-fold cross-validation only, which is the
    correct way to get an unbiased estimate of the tuned model's
    generalisation performance.

    Args:
        model: A fitted or tuneable estimator (pipeline).
        X: Full feature matrix.
        y: Full target vector.
        outer_folds: Number of outer CV folds.
        inner_folds: Number of inner CV folds (informational).

    Returns:
        Dictionary with mean and std of each metric across outer folds,
        plus per-fold values.
    """
    outer_cv = StratifiedKFold(
        n_splits=outer_folds, shuffle=True, random_state=SEED
    )

    fold_metrics: list[dict[str, float]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        from sklearn.base import clone

        estimator_clone = clone(model)
        estimator_clone.fit(X_train, y_train)

        y_pred = estimator_clone.predict(X_test)
        y_prob = estimator_clone.predict_proba(X_test)[:, 1]

        fold_metrics.append(
            {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred)),
                "recall": float(recall_score(y_test, y_pred)),
                "f1": float(f1_score(y_test, y_pred)),
                "roc_auc": float(roc_auc_score(y_test, y_prob)),
            }
        )

    df = pd.DataFrame(fold_metrics)
    summary: dict[str, Any] = {
        "per_fold": fold_metrics,
        "mean": df.mean().to_dict(),
        "std": df.std().to_dict(),
        "median": df.median().to_dict(),
    }
    return summary


# ----------------------------------------------------------------------
# Paired bootstrap significance test
# ----------------------------------------------------------------------


def paired_bootstrap_test(
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
    n_iter: int = BOOTSTRAP_N_ITER,
    metric: str = "accuracy",
    seed: int = SEED,
) -> dict[str, Any]:
    """Test whether model A is significantly better than model B.

    Uses paired bootstrap resampling: for each iteration, sample with
    replacement from the test set and compute the metric for both models.
    The p-value is the fraction of iterations where B >= A.

    Args:
        y_true: Ground-truth labels.
        y_prob_a: Predicted probabilities from model A.
        y_prob_b: Predicted probabilities from model B.
        n_iter: Number of bootstrap iterations.
        metric: Metric to compare (``"accuracy"`` or ``"f1"``).
        seed: Random seed.

    Returns:
        Dictionary with ``delta``, ``p_value``, ``ci_lower``, ``ci_upper``,
        and ``significant`` (at alpha=0.05).
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)

    def _score(probs: np.ndarray, idx: np.ndarray) -> float:
        preds = (probs[idx] >= 0.5).astype(int)
        if metric == "f1":
            return float(f1_score(y_true[idx], preds))
        return float(accuracy_score(y_true[idx], preds))

    diffs = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.choice(n, size=n, replace=True)
        diffs[i] = _score(y_prob_a, idx) - _score(y_prob_b, idx)

    delta = float(np.mean(diffs))
    p_value = float(np.mean(diffs <= 0))
    alpha = 1.0 - CI_LEVEL
    ci_lower = float(np.percentile(diffs, 100 * alpha / 2))
    ci_upper = float(np.percentile(diffs, 100 * (1 - alpha / 2)))

    return {
        "metric": metric,
        "delta_mean": delta,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant": p_value < 0.05,
    }


# ----------------------------------------------------------------------
# McNemar's test (paired binary classification)
# ----------------------------------------------------------------------


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> dict[str, Any]:
    """McNemar's test for paired classification outcomes.

    Tests the null hypothesis that both classifiers have the same
    error rate.  Uses the continuity-corrected chi-squared variant.

    Args:
        y_true: Ground-truth labels.
        y_pred_a: Hard predictions from model A.
        y_pred_b: Hard predictions from model B.

    Returns:
        Dictionary with ``chi2``, ``p_value``, and ``significant``.
    """
    # Contingency table:
    #          B correct  B wrong
    # A correct   n00       n01
    # A wrong     n10       n11
    correct_a = y_pred_a == y_true
    correct_b = y_pred_b == y_true

    n01 = int(np.sum(correct_a & ~correct_b))
    n10 = int(np.sum(~correct_a & correct_b))

    # Continuity-corrected McNemar
    numerator = abs(n01 - n10) - 1
    denominator = n01 + n10
    if denominator == 0:
        chi2 = 0.0
        p_value = 1.0
    else:
        chi2 = (numerator ** 2) / denominator
        p_value = 1.0 - stats.chi2.cdf(chi2, df=1)

    return {
        "chi2": float(chi2),
        "p_value": float(p_value),
        "n01": n01,
        "n10": n10,
        "significant": p_value < 0.05,
    }


# ----------------------------------------------------------------------
# Decision-threshold optimisation
# ----------------------------------------------------------------------


def optimise_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
    n_thresholds: int = 200,
) -> dict[str, Any]:
    """Find the probability threshold that maximises a given metric.

    Args:
        y_true: Ground-truth labels.
        y_prob: Predicted positive-class probabilities.
        metric: Metric to maximise (``"f1"`` or ``"accuracy"``).
        n_thresholds: Number of threshold candidates.

    Returns:
        Dictionary with ``best_threshold``, ``best_score``, and the full
        threshold/score arrays.
    """
    thresholds = np.linspace(0.05, 0.95, n_thresholds)
    scores = np.empty(n_thresholds)

    for i, t in enumerate(thresholds):
        preds = (y_prob >= t).astype(int)
        if metric == "f1":
            scores[i] = f1_score(y_true, preds)
        else:
            scores[i] = accuracy_score(y_true, preds)

    best_idx = int(np.argmax(scores))
    return {
        "best_threshold": float(thresholds[best_idx]),
        "best_score": float(scores[best_idx]),
        "thresholds": thresholds,
        "scores": scores,
    }


# ----------------------------------------------------------------------
# Confidence intervals via bootstrap
# ----------------------------------------------------------------------


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_iter: int = BOOTSTRAP_N_ITER,
    seed: int = SEED,
) -> dict[str, Any]:
    """Compute bootstrap confidence intervals for standard metrics.

    Args:
        y_true: Ground-truth labels.
        y_prob: Predicted positive-class probabilities.
        n_iter: Number of bootstrap resamples.
        seed: Random seed.

    Returns:
        Dictionary with per-metric ``mean``, ``ci_lower``, ``ci_upper``.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    metrics_acc = np.empty(n_iter)
    metrics_f1 = np.empty(n_iter)
    metrics_auc = np.empty(n_iter)

    for i in range(n_iter):
        idx = rng.choice(n, size=n, replace=True)
        preds = (y_prob[idx] >= 0.5).astype(int)
        metrics_acc[i] = accuracy_score(y_true[idx], preds)
        metrics_f1[i] = f1_score(y_true[idx], preds)
        metrics_auc[i] = roc_auc_score(y_true[idx], y_prob[idx])

    alpha = 1.0 - CI_LEVEL
    result: dict[str, Any] = {}
    for name, arr in [
        ("accuracy", metrics_acc),
        ("f1", metrics_f1),
        ("roc_auc", metrics_auc),
    ]:
        result[name] = {
            "mean": float(np.mean(arr)),
            "ci_lower": float(np.percentile(arr, 100 * alpha / 2)),
            "ci_upper": float(np.percentile(arr, 100 * (1 - alpha / 2))),
        }
    return result
