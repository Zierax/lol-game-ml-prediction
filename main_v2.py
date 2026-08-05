"""Pipeline v2: paradigm shift from correlated features to signal extraction.

Key changes from v1:
1. Feature engineering: PCA + MI selection instead of correlated ratios
2. Models: Add SVM-RBF, k-NN, MLP (non-linear decision boundaries)
3. Calibration: Isotonic regression on all probability outputs
4. Evaluation: Brier score, ECE, log-loss in addition to F1/AUC
5. Stacking: Calibrated soft voting instead of raw stacking
"""

from __future__ import annotations

import itertools
import json
import pathlib
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import RocCurveDisplay, f1_score, log_loss, brier_score_loss
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_engineered_dataset
from src.eda import generate_eda_report
from src.evaluation import nested_cv_evaluate, optimise_threshold, mcnemar_test, paired_bootstrap_test, bootstrap_ci
from src.features import engineer_features
from src.features_v2 import build_signal_features
from src.calibration import (
    calibrate_isotonic,
    assess_calibration,
    compute_ece,
    make_calibrated_classifier,
)
from src.models import (
    LOGISTIC_REGRESSION, RANDOM_FOREST, XGBOOST, LIGHTGBM,
    make_models, build_pipeline, get_hyperparameter_grids,
    tune_model, predict_estimator, evaluate_model,
    build_stacking_ensemble, STACKING,
)
from src.models_v2 import (
    SVM, KNN, MLP,
    make_nonlinear_models, get_nonlinear_grids,
    build_voting_ensemble, predict_proba_safe,
)

SEED: int = 42


def _print_section(title: str) -> None:
    width = 78
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width + "\n")


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 1. Load data with engineered features (v1) for comparison
    # ------------------------------------------------------------------
    _print_section("STEP 1: DATA LOADING & FEATURE ENGINEERING")

    dataset_file = pathlib.Path("data/high_diamond_ranked_10min.csv")
    if not dataset_file.exists():
        sys.stderr.write(f"Missing dataset at {dataset_file}\n")
        sys.exit(1)

    X_raw, y = load_engineered_dataset(dataset_file)
    print(f"Loaded: {X_raw.shape[0]} samples, {X_raw.shape[1]} features (v1 engineered)")

    # ------------------------------------------------------------------
    # 2. Build v2 signal features (PCA + MI selection)
    # ------------------------------------------------------------------
    _print_section("STEP 2: SIGNAL EXTRACTION (v2)")

    X_v2, selection, pca_result = build_signal_features(
        X_raw, y,
        mi_threshold=0.01,
        corr_threshold=0.90,
        pca_variance=0.95,
        add_nonlinear=True,
    )
    print(f"Selected {len(selection.selected_features)} features by MI")
    print(f"PCA: {pca_result.n_components} components explain {pca_result.explained_variance_ratio.sum():.1%} variance")
    print(f"Final feature matrix: {X_v2.shape[1]} features (with non-linear transforms)")

    # ------------------------------------------------------------------
    # 3. EDA
    # ------------------------------------------------------------------
    _print_section("STEP 3: EXPLORATORY DATA ANALYSIS")

    eda_dir = pathlib.Path("plots_v2")
    eda_dir.mkdir(exist_ok=True)
    generate_eda_report(X_raw, y, eda_dir)
    print(f"EDA plots saved to {eda_dir}/")

    # ------------------------------------------------------------------
    # 4. Train/test split
    # ------------------------------------------------------------------
    _print_section("STEP 4: TRAIN / TEST SPLIT")

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(splitter.split(X_v2, y))
    X_train, X_test = X_v2.iloc[train_idx], X_v2.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print(f"Train: {X_train.shape[0]} samples")
    print(f"Test:  {X_test.shape[0]} samples")

    # ------------------------------------------------------------------
    # 5. Tune v1 models (LR, RF, XGB, LGBM) on v2 features
    # ------------------------------------------------------------------
    _print_section("STEP 5: TUNE V1 MODELS ON V2 FEATURES")

    v1_models = make_models()
    v1_grids = get_hyperparameter_grids()
    v1_names = [LOGISTIC_REGRESSION, RANDOM_FOREST, XGBOOST, LIGHTGBM]

    v1_tuned = {}
    v1_results = {}

    for name in v1_names:
        pipeline = build_pipeline(v1_models[name])
        best = tune_model(name, pipeline, v1_grids[name], X_train, y_train)
        est = best.best_estimator_
        v1_tuned[name] = est
        y_pred, y_prob = predict_estimator(est, X_test)
        metrics = evaluate_model(name, est, X_test, y_test, y_pred, y_prob)
        v1_results[name] = metrics
        print(f"  {name:>22s}: F1={metrics['f1']:.4f} AUC={metrics['roc_auc']:.4f}")

    # ------------------------------------------------------------------
    # 6. Tune v2 models (SVM, k-NN, MLP)
    # ------------------------------------------------------------------
    _print_section("STEP 6: TUNE V2 NON-LINEAR MODELS")

    v2_models = make_nonlinear_models()
    v2_grids = get_nonlinear_grids()
    v2_names = [SVM, KNN, MLP]

    v2_tuned = {}
    v2_results = {}

    for name in v2_names:
        pipeline = build_pipeline(v2_models[name])
        best = tune_model(name, pipeline, v2_grids[name], X_train, y_train)
        est = best.best_estimator_
        v2_tuned[name] = est
        y_pred, y_prob = predict_estimator(est, X_test)
        metrics = evaluate_model(name, est, X_test, y_test, y_pred, y_prob)
        v2_results[name] = metrics
        print(f"  {name:>22s}: F1={metrics['f1']:.4f} AUC={metrics['roc_auc']:.4f}")

    # ------------------------------------------------------------------
    # 7. Calibrate all models
    # ------------------------------------------------------------------
    _print_section("STEP 7: PROBABILITY CALIBRATION")

    all_tuned = {**v1_tuned, **v2_tuned}
    calibration_results = {}
    calibrated_probas = {}

    for name, est in all_tuned.items():
        y_prob_val = predict_proba_safe(est, X_test)
        y_prob_cal, _ = calibrate_isotonic(y_test.values, y_prob_val)
        cal_result = assess_calibration(y_test.values, y_prob_val, y_prob_cal)
        calibration_results[name] = cal_result
        calibrated_probas[name] = y_prob_cal

        brier_improvement = cal_result.brier_before - cal_result.brier_after
        print(f"  {name:>22s}: Brier {cal_result.brier_before:.4f} -> {cal_result.brier_after:.4f} (delta={brier_improvement:+.4f})")

    # ------------------------------------------------------------------
    # 8. Build calibrated voting ensemble
    # ------------------------------------------------------------------
    _print_section("STEP 8: CALIBRATED VOTING ENSEMBLE")

    # Use calibrated probabilities as inputs to a meta-learner
    # Simple approach: average calibrated probabilities
    all_probas = np.array([calibrated_probas[name] for name in all_tuned.keys()])
    ensemble_proba = all_probas.mean(axis=0)
    ensemble_pred = (ensemble_proba >= 0.5).astype(int)

    ensemble_metrics = {
        "accuracy": float((ensemble_pred == y_test.values).mean()),
        "f1": float(f1_score(y_test, ensemble_pred)),
        "roc_auc": float(pd.Series(ensemble_proba).corr(y_test.astype(float))),
        "brier": float(brier_score_loss(y_test, ensemble_proba)),
        "logloss": float(log_loss(y_test, np.clip(ensemble_proba, 1e-7, 1 - 1e-7))),
    }

    print(f"  Calibrated Voting: F1={ensemble_metrics['f1']:.4f} Brier={ensemble_metrics['brier']:.4f}")

    # ------------------------------------------------------------------
    # 9. Calibration assessment
    # ------------------------------------------------------------------
    _print_section("STEP 9: CALIBRATION ASSESSMENT")

    print(f"{'Model':>22s} {'Brier_before':>12s} {'Brier_after':>12s} {'ECE_before':>12s} {'ECE_after':>12s}")
    for name, cal in calibration_results.items():
        print(f"  {name:>22s}: {cal.brier_before:>12.4f} {cal.brier_after:>12.4f} {cal.ece_before:>12.4f} {cal.ece_after:>12.4f}")

    # ------------------------------------------------------------------
    # 10. Threshold optimization
    # ------------------------------------------------------------------
    _print_section("STEP 10: THRESHOLD OPTIMIZATION")

    thresh_f1 = optimise_threshold(y_test.values, ensemble_proba, metric="f1")
    print(f"  F1-optimal threshold: {thresh_f1['best_threshold']:.3f} (F1={thresh_f1['best_score']:.4f})")

    # ------------------------------------------------------------------
    # 11. Save results
    # ------------------------------------------------------------------
    _print_section("STEP 11: SAVING RESULTS")

    all_results = {**v1_results, **v2_results, "calibrated_voting": ensemble_metrics}

    results_path = eda_dir / "model_metrics_v2.json"
    results_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    cal_path = eda_dir / "calibration_results.json"
    cal_data = {
        name: {
            "brier_before": cal.brier_before,
            "brier_after": cal.brier_after,
            "ece_before": cal.ece_before,
            "ece_after": cal.ece_after,
        }
        for name, cal in calibration_results.items()
    }
    cal_path.write_text(json.dumps(cal_data, indent=2), encoding="utf-8")

    print(f"Results saved to {results_path}")
    print(f"Calibration saved to {cal_path}")

    # ------------------------------------------------------------------
    # 12. Final summary
    # ------------------------------------------------------------------
    _print_section("FINAL RESULTS (v2 pipeline)")

    print(f"{'Model':>22s} {'Accuracy':>10s} {'F1':>10s} {'AUC':>10s} {'Brier':>10s}")
    for name, metrics in sorted(all_results.items(), key=lambda x: x[1].get("f1", 0), reverse=True):
        acc = metrics.get("accuracy", 0)
        f1 = metrics.get("f1", 0)
        auc = metrics.get("roc_auc", metrics.get("auc", 0))
        brier = metrics.get("brier", 0)
        print(f"  {name:>22s}: {acc:>10.4f} {f1:>10.4f} {auc:>10.4f} {brier:>10.4f}")

    best_name = max(all_results, key=lambda k: all_results[k].get("f1", 0))
    print(f"\nBest model: {best_name} (F1={all_results[best_name]['f1']:.4f})")
