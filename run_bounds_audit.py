"""Bounds audit: tighten the information-ceiling claim beyond Fano-from-MI.

Answers reviewer R1/R9's central objection---``achieved MI is a DPI lower
bound, so ``no model CAN extract more'' is not established''---with two
measurements on the SAME held-out zoo predictions:

1. Score-Bayes ceiling
   The accuracy of the optimal threshold-free decision rule on the held-out
   scores, i.e.  E_S[ max_y P(Y = y | S = s) ] estimated per score decile
   with out-of-sample frequencies.  A model family whose accuracy is far
   below this number is wasting decision-rule slack; a family near it is
   converting essentially all information in its score into decisions.
   Unlike the Fano bound (computed from the same plug-in MI), this is a
   *measure* of achievable accuracy given the current score, not a
   lower-bound-on-error inequality.

2. KSG achieved-MI cross-check
   The binned plug-in MI estimator has a small positive bias that depends
   on bin count.  We recompute achieved MI with the k-nearest-neighbour
   estimator (Ross 2014) on the continuous score, which is binning-free.
   Agreement with the binned estimate demonstrates the plug-in value is not
   a binning artifact.

Persists plots_ceiling/bounds_audit.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

from src.ceiling_analysis import (
    SEED,
    TEST_SIZE,
    achieved_mi_robust,
    achieved_mutual_information,
    empirical_entropy,
    fano_min_error,
    load_engineered_dataset as _load_main,
)
from src.models import (
    LOGISTIC_REGRESSION,
    RANDOM_FOREST,
    XGBOOST,
    LIGHTGBM,
    build_pipeline,
    make_models,
)
from src.models_v2 import (
    SVM,
    KNN,
    MLP,
    make_nonlinear_models,
    predict_proba_safe,
)

OUTPUT_DIR = Path("plots_ceiling")
N_DECILES: int = 10
KSG_NEIGHBORS: int = 3


def score_bayes_ceiling(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = N_DECILES,
) -> dict[str, float]:
    """Accuracy of the optimal decision rule on the score, per decile.

    Estimated with out-of-sample empirical frequencies P(y | decile).
    ``ceiling`` is E[ max_y P(y|S) ] over the score distribution; this is
    the accuracy a perfect (threshold-free) decision rule would achieve on
    the current score.  ``oracle_profit`` is the gap between the score
    ceiling and the 0.5-threshold accuracy of the same score.
    """
    order = np.argsort(y_prob)
    bins = np.array_split(order, n_bins)
    weighted_sum = 0.0
    for idx in bins:
        if len(idx) == 0:
            continue
        frac_win = float(np.mean(y_true[idx] == 1))
        weighted_sum += len(idx) * max(frac_win, 1.0 - frac_win)
    ceiling = weighted_sum / len(y_true)

    acc_at_half = float(accuracy_score(y_true, (y_prob >= 0.5).astype(int)))
    return {
        "ceiling": float(ceiling),
        "ceiling_profit_over_05": float(ceiling - acc_at_half),
        "n_bins": int(n_bins),
    }


def ksg_achieved_mi(y_prob: np.ndarray, y_true: np.ndarray) -> float:
    """Binning-free achieved MI via the kNN estimator (Ross 2014)."""
    return float(
        mutual_info_classif(
            y_prob.reshape(-1, 1),
            y_true,
            n_neighbors=KSG_NEIGHBORS,
            random_state=SEED,
        )[0]
    )


def run_bounds_audit() -> dict[str, Any]:
    """Fit the identical 8-family zoo on the identical split; audit bounds."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X, y = _load_main()
    y_arr = y.values
    h_y = empirical_entropy(y_arr)

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=SEED
    )
    train_idx, test_idx = next(splitter.split(X, y))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    zoo: dict[str, Any] = {}
    base = make_models()
    for name in (LOGISTIC_REGRESSION, RANDOM_FOREST, XGBOOST, LIGHTGBM):
        zoo[name] = build_pipeline(base[name])
    nonlinear = make_nonlinear_models()
    for name in (SVM, KNN, MLP):
        zoo[name] = build_pipeline(nonlinear[name])

    results: dict[str, Any] = {}
    all_prob: list[np.ndarray] = []
    for name, est in zoo.items():
        est.fit(X_train, y_train)
        y_prob = predict_proba_safe(est, X_test)
        all_prob.append(y_prob)
        acc = float(accuracy_score(y_test, (y_prob >= 0.5).astype(int)))
        mi_binned = achieved_mi_robust(y_prob, y_test.values)
        mi_per_bin = {
            str(b): achieved_mutual_information(y_prob, y_test.values, b)
            for b in (10, 20, 40)
        }
        mi_ksg = ksg_achieved_mi(y_prob, y_test.values)
        sb = score_bayes_ceiling(y_prob, y_test.values)
        fano_min = fano_min_error(mi_binned, h_y)
        results[name] = {
            "test_accuracy": acc,
            "achieved_mi_binned": mi_binned,
            "achieved_mi_per_bin": mi_per_bin,
            "achieved_mi_ksg": mi_ksg,
            "score_bayes_ceiling": sb["ceiling"],
            "ceiling_profit_over_05": sb["ceiling_profit_over_05"],
            "fano_max_accuracy_from_binned_mi": 1.0 - fano_min,
        }
        print(
            f"  {name:>20s}: acc={acc:.4f} | MI binned={mi_binned:.4f} "
            f"KSG={mi_ksg:.4f} | score-Bayes ceiling={sb['ceiling']:.4f} "
            f"(+{sb['ceiling_profit_over_05']:+.4f}) | "
            f"Fano max={1.0 - fano_min:.4f}"
        )

    voting_prob = np.mean(np.column_stack(all_prob), axis=1)
    voting_acc = float(accuracy_score(y_test, (voting_prob >= 0.5).astype(int)))
    mi_v = achieved_mi_robust(voting_prob, y_test.values)
    sb_v = score_bayes_ceiling(voting_prob, y_test.values)
    fano_v = fano_min_error(mi_v, h_y)
    results["soft_voting"] = {
        "test_accuracy": voting_acc,
        "achieved_mi_binned": mi_v,
        "achieved_mi_ksg": ksg_achieved_mi(voting_prob, y_test.values),
        "score_bayes_ceiling": sb_v["ceiling"],
        "ceiling_profit_over_05": sb_v["ceiling_profit_over_05"],
        "fano_max_accuracy_from_binned_mi": 1.0 - fano_v,
    }
    print(
        f"  {'soft_voting':>20s}: acc={voting_acc:.4f} | MI binned={mi_v:.4f} "
        f"| score-Bayes ceiling={sb_v['ceiling']:.4f} "
        f"(+{sb_v['ceiling_profit_over_05']:+.4f}) | "
        f"Fano max={1.0 - fano_v:.4f}"
    )

    audit = {
        "n_samples": int(len(y_arr)),
        "n_features": int(X.shape[1]),
        "test_size": TEST_SIZE,
        "seed": SEED,
        "target_entropy_bits": h_y,
        "models": results,
    }
    (OUTPUT_DIR / "bounds_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(f"\nSaved {OUTPUT_DIR / 'bounds_audit.json'}")
    return audit


if __name__ == "__main__":
    run_bounds_audit()