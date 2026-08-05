"""Probability calibration layer.

Many classifiers output scores that are not well-calibrated probabilities.
A model that predicts P(win) = 0.8 should be correct 80% of the time.
Tree-based models (RF, XGB, LGBM) tend to produce overconfident predictions
(pushed toward 0 and 1). SVM scores are not probabilities at all.

This module provides:
- Isotonic regression calibration (non-parametric)
- Platt scaling (logistic calibration)
- Temperature scaling (for neural networks)
- Calibration assessment: Brier score, Expected Calibration Error (ECE)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


@dataclass(frozen=True)
class CalibrationResult:
    """Result of probability calibration."""

    proba_calibrated: np.ndarray
    brier_before: float
    brier_after: float
    ece_before: float
    ece_after: float
    logloss_before: float
    logloss_after: float


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error.

    ECE = sum_k (|B_k| / n) * |acc(B_k) - conf(B_k)|
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins[1:-1], right=True)

    ece = 0.0
    n = len(y_true)
    for k in range(n_bins):
        mask = bin_indices == k
        if mask.sum() == 0:
            continue
        bin_size = mask.sum() / n
        avg_confidence = y_prob[mask].mean()
        avg_accuracy = y_true[mask].mean()
        ece += bin_size * abs(avg_accuracy - avg_confidence)

    return float(ece)


def calibrate_isotonic(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[np.ndarray, IsotonicRegression]:
    """Calibrate probabilities using isotonic regression (non-parametric)."""
    ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    ir.fit(y_prob, y_true)
    proba_calibrated = ir.predict(y_prob)
    return proba_calibrated, ir


def calibrate_platt(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[np.ndarray, LogisticRegression]:
    """Calibrate probabilities using Platt scaling (logistic regression on scores)."""
    scores = y_prob.reshape(-1, 1)
    platt = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    platt.fit(scores, y_true)
    proba_calibrated = platt.predict_proba(scores)[:, 1]
    return proba_calibrated, platt


def calibrate_temperature(
    y_true: np.ndarray,
    logits: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Temperature scaling for neural network outputs."""
    best_t = 1.0
    best_nll = float("inf")

    for t in np.linspace(0.1, 5.0, 100):
        scaled_logits = logits / t
        proba = 1.0 / (1.0 + np.exp(-scaled_logits))
        proba = np.clip(proba, 1e-7, 1 - 1e-7)
        nll = log_loss(y_true, proba)
        if nll < best_nll:
            best_nll = nll
            best_t = t

    scaled_logits = logits / best_t
    proba_calibrated = 1.0 / (1.0 + np.exp(-scaled_logits))
    return proba_calibrated, float(best_t)


def assess_calibration(
    y_true: np.ndarray,
    y_prob_before: np.ndarray,
    y_prob_after: np.ndarray,
) -> CalibrationResult:
    """Assess calibration quality before and after calibration."""
    return CalibrationResult(
        proba_calibrated=y_prob_after,
        brier_before=float(brier_score_loss(y_true, y_prob_before)),
        brier_after=float(brier_score_loss(y_true, y_prob_after)),
        ece_before=compute_ece(y_true, y_prob_before),
        ece_after=compute_ece(y_true, y_prob_after),
        logloss_before=float(log_loss(y_true, np.clip(y_prob_before, 1e-7, 1 - 1e-7))),
        logloss_after=float(log_loss(y_true, np.clip(y_prob_after, 1e-7, 1 - 1e-7))),
    )


def make_calibrated_classifier(
    estimator: Any,
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    X_val: pd.DataFrame | np.ndarray,
    y_val: pd.Series | np.ndarray,
    method: str = "isotonic",
) -> tuple[Any, CalibrationResult]:
    """Fit a classifier and calibrate its probability outputs."""
    estimator.fit(X_train, y_train)

    y_prob_val = estimator.predict_proba(X_val)[:, 1]

    if method == "isotonic":
        y_prob_cal, _ = calibrate_isotonic(y_val.values, y_prob_val)
    elif method == "platt":
        y_prob_cal, _ = calibrate_platt(y_val.values, y_prob_val)
    else:
        raise ValueError(f"Unknown calibration method: {method}")

    result = assess_calibration(y_val.values, y_prob_val, y_prob_cal)
    return estimator, result
