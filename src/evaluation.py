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
    GridSearchCV,
    StratifiedKFold,
)

SEED: int = 42
NESTED_OUTER_FOLDS: int = 5
NESTED_INNER_FOLDS: int = 5
BOOTSTRAP_N_ITER: int = 2000
CI_LEVEL: float = 0.95

ModelEstimator = BaseEstimator | ClassifierMixin


# ----------------------------------------------------------------------
# Nested cross-validation
# ----------------------------------------------------------------------


def nested_cv_evaluate(
    model: ModelEstimator,
    param_grid: dict[str, list[Any]],
    X: pd.DataFrame,
    y: pd.Series,
    outer_folds: int = NESTED_OUTER_FOLDS,
    inner_folds: int = NESTED_INNER_FOLDS,
    scoring: str = "accuracy",
) -> dict[str, Any]:
    """Estimate out-of-fold performance via true nested stratified CV.

    The outer loop provides unbiased performance estimates.  The inner
    loop performs hyperparameter tuning on each outer training fold using
    GridSearchCV, so the reported metric is free from both data leakage
    and tuning-selection bias.

    Args:
        model: An untuned estimator (pipeline) to be cloned each fold.
        param_grid: GridSearchCV parameter grid for the inner loop.
        X: Full feature matrix.
        y: Full target vector.
        outer_folds: Number of outer CV folds.
        inner_folds: Number of inner CV folds for the inner tuning loop.
        scoring: Scoring metric for GridSearchCV in the inner loop.

    Returns:
        Dictionary with mean and std of each metric across outer folds,
        plus per-fold values and per-fold best inner params.
    """
    outer_cv = StratifiedKFold(
        n_splits=outer_folds, shuffle=True, random_state=SEED
    )
    inner_cv = StratifiedKFold(
        n_splits=inner_folds, shuffle=True, random_state=SEED
    )

    fold_metrics: list[dict[str, float]] = []
    fold_params: list[dict[str, Any]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        from sklearn.base import clone

        inner_search = GridSearchCV(
            estimator=clone(model),
            param_grid=param_grid,
            scoring=scoring,
            cv=inner_cv,
            n_jobs=1,
            refit=True,
        )
        inner_search.fit(X_train, y_train)

        best_estimator = inner_search.best_estimator_
        y_pred = best_estimator.predict(X_test)
        y_prob = best_estimator.predict_proba(X_test)[:, 1]

        fold_metrics.append(
            {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred)),
                "recall": float(recall_score(y_test, y_pred)),
                "f1": float(f1_score(y_test, y_pred)),
                "roc_auc": float(roc_auc_score(y_test, y_prob)),
            }
        )
        fold_params.append(inner_search.best_params_)

    df = pd.DataFrame(fold_metrics)
    summary: dict[str, Any] = {
        "per_fold": fold_metrics,
        "per_fold_params": fold_params,
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
    method: str = "exact",
) -> dict[str, Any]:
    """McNemar's test for paired classification outcomes.

    Tests the null hypothesis that both classifiers have the same
    error rate.

    Three variants are provided:

    * ``"exact"`` (default) --- the exact binomial test on the discordant
      pairs, ``2 * BinomCDF(min(n01, n10); n01+n10, 0.5)``.  This is the
      recommended default for paired binary data because the chi-square
      approximation is unreliable in the sparse-cell regime
      \\ (Fagerland, Lydersen and Laake, *BMC Med. Res. Methodol.* **2013**,
      *13*, 219).
    * ``"chi2"`` --- the uncorrected Pearson chi-square statistic.
    * ``"chi2_cc"`` --- the continuity-corrected chi-square.  Retained only
      for sensitivity analysis; it should not be used when the discordant
      count is large, where it needlessly costs power.

    Args:
        y_true: Ground-truth labels.
        y_pred_a: Hard predictions from model A.
        y_pred_b: Hard predictions from model B.
        method: One of ``"exact"``, ``"chi2"``, ``"chi2_cc"``.

    Returns:
        Dictionary with ``chi2``, ``p_value``, ``n01``, ``n10``, ``method``,
        and ``significant``.
    """
    # Contingency table:
    #          B correct  B wrong
    # A correct   n00       n01
    # A wrong     n10       n11
    correct_a = y_pred_a == y_true
    correct_b = y_pred_b == y_true

    n01 = int(np.sum(correct_a & ~correct_b))
    n10 = int(np.sum(~correct_a & correct_b))
    n_discordant = n01 + n10

    if method == "exact":
        # Exact binomial: P(X <= min(n01,n10)) under Binom(n01+n10, 0.5).
        # Doubling the lower tail is valid because the distribution is
        # symmetric about n/2, so the two tails are equal.
        if n_discordant == 0:
            p_value = 1.0
        else:
            k = min(n01, n10)
            p_value = float(min(1.0, 2.0 * stats.binom.cdf(k, n_discordant, 0.5)))
        chi2 = 0.0 if n_discordant == 0 else float((n01 - n10) ** 2 / n_discordant)
    elif method in ("chi2", "chi2_cc"):
        if n_discordant == 0:
            chi2 = 0.0
        elif method == "chi2_cc":
            # The correction term must not make the numerator negative:
            # when |n01 - n10| <= 1 the corrected statistic is 0 by
            # convention, not 1/n.
            numerator = max(0.0, abs(n01 - n10) - 1.0)
            chi2 = float(numerator ** 2 / n_discordant)
        else:
            chi2 = float((n01 - n10) ** 2 / n_discordant)
        # chi2.sf (the survival function) is numerically stable; computing
        # 1 - cdf loses all precision once the CDF approaches one.
        p_value = float(stats.chi2.sf(chi2, df=1))
    else:
        raise ValueError(f"unknown method: {method!r}")

    return {
        "chi2": float(chi2),
        "p_value": float(p_value),
        "n01": n01,
        "n10": n10,
        "n_discordant": n_discordant,
        "method": method,
        "significant": bool(p_value < 0.05),
    }


def minimum_detectable_difference(
    n_test: int,
    n_discordant: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Minimum detectable accuracy difference for a paired binary test.

    For McNemar's test the discriminating statistic is the imbalance
    between the two discordant counts, whose variance under the paired
    design is ``m / n`` with ``m = n01 + n10``.  Setting the detectable
    difference ``d`` equal to the standard error of that imbalance scaled
    by the normal deviate required for two-sided level ``alpha`` at power
    ``1 - beta``, i.e. solving ``d = z' sqrt(m / (n (n + z'^2)))`` for
    ``z' = z_{1-alpha/2} + z_{power}``, gives

    .. math::

        d = \\left(z_{1-\\alpha/2} + z_{\\text{power}}\\right)
            \\sqrt{\\frac{m}{n\\left(n + \\left(z_{1-\\alpha/2} +
            z_{\\text{power}}\\right)^2\\right)}} .

    Note the power term is part of ``z'``, not folded into the radicand.

    Reporting this quantity is essential: without it, "no significant
    difference" is indistinguishable from "insufficient power to detect a
    difference that might exist".

    Args:
        n_test: Number of paired test observations.
        n_discordant: Observed discordant count ``m = n01 + n10``.
        alpha: **Per-comparison** two-sided significance level. This is
            not a family-wise level: when decisions are Holm-corrected
            across many pairs, the effective per-pair level is larger
            than ``alpha`` and the caller must pass that larger value (or
            recompute) to get a matching MDE.
        power: Target power (``1 - beta``).

    Returns:
        The minimum detectable difference in accuracy (same units as
        accuracy, i.e. a fraction), or ``nan`` if ``n_test`` is zero.
    """
    if n_test <= 0:
        return float("nan")
    z = float(stats.norm.ppf(1.0 - alpha / 2.0) + stats.norm.ppf(power))
    m = float(n_discordant)
    return float(z * np.sqrt(m / (n_test * (n_test + z ** 2))))


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
