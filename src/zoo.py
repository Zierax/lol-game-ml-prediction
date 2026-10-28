"""Single shared fitting pipeline for every table in the manuscript.

Three entry points previously fitted the model zoo separately and produced
slightly different numbers for the same family (Table 1 from a calibrated
pipeline fitted on 6,718 rows, Table 3 from an uncalibrated pipeline on
7,903 rows, and the label-noise audit from a third variant).  That made the
tables not like-for-like.  This module defines one fit-and-score routine and
all three entry points call it, so every reported number comes from the same
fitted models on the same split.

The protocol is:

1. stratified 80/20 split (seed 42);
2. a further stratified 15% fold carved from the training side and used only
   to fit an isotonic calibrator per family;
3. every family fitted on the remaining training rows;
4. scores on the held-out test set, after calibration.

All steps are seeded and single-threaded, so the fit is deterministic.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

from src.evaluation import minimum_detectable_difference, mcnemar_test
from src.features import engineer_features
from src.data_loader import load_dataset
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

SEED: int = 42
TEST_SIZE: float = 0.2
CAL_SIZE: float = 0.15

#: Families in the order used for every table and figure.
BASE_FAMILIES: tuple[str, ...] = (
    LOGISTIC_REGRESSION,
    RANDOM_FOREST,
    XGBOOST,
    LIGHTGBM,
    SVM,
    KNN,
    MLP,
)
ALL_FAMILIES: tuple[str, ...] = BASE_FAMILIES + ("soft_voting",)


def load_master_split() -> dict[str, Any]:
    """Load the engineered dataset and create the canonical split."""
    X_raw, y_series = load_dataset()
    X = engineer_features(X_raw)
    y = np.asarray(y_series)

    outer = StratifiedShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=SEED
    )
    train_idx, test_idx = next(outer.split(X, y))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    inner = StratifiedShuffleSplit(
        n_splits=1, test_size=CAL_SIZE, random_state=SEED
    )
    fit_rel, cal_rel = next(inner.split(X_train, y_train))
    X_fit, X_cal = X_train.iloc[fit_rel], X_train.iloc[cal_rel]
    y_fit, y_cal = y_train[fit_rel], y_train[cal_rel]

    return {
        "X_raw": X_raw,
        "X": X,
        "y": y,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "X_fit": X_fit,
        "y_fit": y_fit,
        "fit_rel": fit_rel,
        "X_cal": X_cal,
        "y_cal": y_cal,
        "n_test": int(len(y_test)),
        "n_fit": int(len(y_fit)),
        "n_cal": int(len(y_cal)),
    }


def build_zoo() -> dict[str, Any]:
    """Fresh, seeded instances of the seven base families."""
    zoo: dict[str, Any] = {}
    base = make_models()
    for name in (LOGISTIC_REGRESSION, RANDOM_FOREST, XGBOOST, LIGHTGBM):
        zoo[name] = build_pipeline(base[name])
    nonlinear = make_nonlinear_models()
    for name in (SVM, KNN, MLP):
        zoo[name] = build_pipeline(nonlinear[name])
    return zoo


def fit_and_score(
    split: dict[str, Any],
    fit_labels: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit, calibrate and score every family on the canonical split.

    Args:
        split: Output of :func:`load_master_split`.
        fit_labels: Optional replacement *training* labels, used by the
            label-noise audit to corrupt training labels while keeping the
            calibration fold and the test set clean.  When given, the
            calibration fold labels are left untouched so that the calibrator
            is not itself corrupted.

    Returns:
        Dictionary with ``probs``, ``preds``, ``acc``, ``auc`` and ``n_fit``.
    """
    X_fit, y_fit = split["X_fit"], split["y_fit"]
    X_cal, y_cal = split["X_cal"], split["y_cal"]
    X_test, y_test = split["X_test"], split["y_test"]

    labels = y_fit if fit_labels is None else np.asarray(fit_labels)

    probs: dict[str, np.ndarray] = {}
    preds: dict[str, np.ndarray] = {}
    acc: dict[str, float] = {}
    auc: dict[str, float] = {}

    base_probs: list[np.ndarray] = []
    for name, est in build_zoo().items():
        est.fit(X_fit, labels)
        raw_cal = predict_proba_safe(est, X_cal)
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(raw_cal, y_cal)
        p = np.clip(cal.predict(predict_proba_safe(est, X_test)), 0.0, 1.0)
        probs[name] = p
        preds[name] = (p >= 0.5).astype(int)
        acc[name] = float(accuracy_score(y_test, preds[name]))
        auc[name] = float(roc_auc_score(y_test, p))
        base_probs.append(p)

    # Soft vote over the calibrated base scores.
    v = np.mean(np.column_stack(base_probs), axis=1)
    probs["soft_voting"] = v
    preds["soft_voting"] = (v >= 0.5).astype(int)
    acc["soft_voting"] = float(accuracy_score(y_test, preds["soft_voting"]))
    auc["soft_voting"] = float(roc_auc_score(y_test, v))

    return {
        "probs": probs,
        "preds": preds,
        "acc": acc,
        "auc": auc,
        "n_fit": int(len(labels)),
    }


def pairwise_tests(y_test: np.ndarray, preds: dict[str, np.ndarray]) -> dict:
    """McNemar family comparison over the seven base families.

    Exact binomial McNemar with Holm step-down adjustment, reporting raw and
    adjusted p-values, discordant counts and the minimum detectable
    difference at each pair's observed discordant count.
    """
    import itertools

    names = list(BASE_FAMILIES)
    pairs = list(itertools.combinations(names, 2))
    n_pairs = len(pairs)
    n_test = len(y_test)

    raw: dict[tuple[str, str], dict] = {}
    for a, b in pairs:
        raw[(a, b)] = mcnemar_test(y_test, preds[a], preds[b], method="exact")

    ordered = sorted(pairs, key=lambda p: raw[p]["p_value"])
    adjusted: dict[tuple[str, str], float] = {}
    running = 0.0
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (n_pairs - rank) * raw[key]["p_value"]))
        adjusted[key] = running

    sensitivity: dict[str, dict[str, float]] = {}
    for m in ("chi2", "chi2_cc"):
        sensitivity[m] = {
            f"{a} vs {b}": mcnemar_test(y_test, preds[a], preds[b],
                                        method=m)["p_value"]
            for a, b in pairs
        }

    return {
        "n_pairs": n_pairs,
        "n_test": int(n_test),
        "method": (
            "exact binomial McNemar (primary) with Holm-Bonferroni step-down "
            "at family-wise alpha = 0.05"
        ),
        "n_discordant_min": int(min(r["n_discordant"] for r in raw.values())),
        "pairs": {
            f"{a} vs {b}": {
                "n01": raw[(a, b)]["n01"],
                "n10": raw[(a, b)]["n10"],
                "n_discordant": raw[(a, b)]["n_discordant"],
                "chi2": raw[(a, b)]["chi2"],
                "p_value": raw[(a, b)]["p_value"],
                "p_value_holm": adjusted[(a, b)],
                "significant_after_holm": adjusted[(a, b)] < 0.05,
                "mde_accuracy_points": float(
                    100.0 * minimum_detectable_difference(
                        n_test, raw[(a, b)]["n_discordant"], alpha=0.05
                    )
                ),
            }
            for (a, b) in pairs
        },
        "sensitivity_p_values": sensitivity,
        "n_significant_uncorrected": int(
            sum(1 for r in raw.values() if r["p_value"] < 0.05)
        ),
        "n_significant_after_holm": int(
            sum(1 for v in adjusted.values() if v < 0.05)
        ),
    }