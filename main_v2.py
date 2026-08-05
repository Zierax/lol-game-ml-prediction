"""Pipeline v2: signal extraction with a leakage-free train/val/test protocol.

Key changes from the original v2 pipeline (all fixes verified against
the peer-review revision plan):

1. Feature engineering is fitted on the training split ONLY (MI selection
   + PCA are fit on ``X_train`` and *transformed* on validation/test),
   eliminating the previous use of test labels in feature selection.
2. Calibration (isotonic regression) and threshold optimisation are
   performed on a held-out validation split; the untouched test split is
   used only for final evaluation, so ECE/Brier/F1 gains are honest
   out-of-sample quantities.
3. ROC-AUC is computed with ``roc_auc_score`` (the previous pipeline
   computed a Pearson correlation by mistake).
4. All models are deterministic (``n_jobs=1``, fixed seeds).
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit

from src.data_loader import load_engineered_dataset
from src.eda import generate_eda_report
from src.evaluation import optimise_threshold
from src.features_v2 import (
    build_signal_features,
    transform_signal_features,
)
from src.calibration import (
    calibrate_isotonic,
    assess_calibration,
)
from src.models import (
    LOGISTIC_REGRESSION, RANDOM_FOREST, XGBOOST, LIGHTGBM,
    make_models, build_pipeline, get_hyperparameter_grids,
    tune_model, predict_estimator, evaluate_model,
)
from src.models_v2 import (
    SVM, KNN, MLP,
    make_nonlinear_models, get_nonlinear_grids,
    predict_proba_safe,
)

SEED: int = 42
TEST_SIZE: float = 0.2
VAL_SIZE: float = 0.2


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
    # 1b. EDA (descriptive only; no training/test information used here)
    # ------------------------------------------------------------------
    eda_dir = pathlib.Path("plots_v2")
    eda_dir.mkdir(exist_ok=True)
    generate_eda_report(X_raw, y, eda_dir)
    print(f"EDA plots saved to {eda_dir}/")

    # ------------------------------------------------------------------
    # 2. Leakage-free train / validation / test split (stratified, seed 42)
    # ------------------------------------------------------------------
    _print_section("STEP 2: TRAIN / VALIDATION / TEST SPLIT")

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_val_idx, test_idx = next(splitter.split(X_raw, y))
    X_train_val, X_test = X_raw.iloc[train_val_idx], X_raw.iloc[test_idx]
    y_train_val, y_test = y.iloc[train_val_idx], y.iloc[test_idx]

    val_splitter = StratifiedShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=SEED)
    train_idx, val_idx = next(val_splitter.split(X_train_val, y_train_val))
    X_train, X_val = X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
    y_train, y_val = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]

    print(f"Train:      {X_train.shape[0]} samples")
    print(f"Validation: {X_val.shape[0]} samples")
    print(f"Test:       {X_test.shape[0]} samples")

    # ------------------------------------------------------------------
    # 3. Build v2 signal features on TRAINING data only; transform others
    # ------------------------------------------------------------------
    _print_section("STEP 3: SIGNAL EXTRACTION (v2, train-fitted)")

    X_v2_train, selection, pca_result = build_signal_features(
        X_train, y_train,
        mi_threshold=0.01,
        corr_threshold=0.90,
        pca_variance=0.95,
        add_nonlinear=True,
    )
    X_v2_val = transform_signal_features(X_val, selection, pca_result, add_nonlinear=True)
    X_v2_test = transform_signal_features(X_test, selection, pca_result, add_nonlinear=True)

    print(f"Selected {len(selection.selected_features)} features by MI (train-only)")
    print(f"PCA: {pca_result.n_components} components "
          f"({pca_result.explained_variance_ratio.sum():.1%} train variance)")
    print(f"Final feature matrix: {X_v2_train.shape[1]} features")

    # ------------------------------------------------------------------
    # 4. Tune v1 models (LR, RF, XGB, LGBM) on v2 features
    # ------------------------------------------------------------------
    _print_section("STEP 4: TUNE V1 MODELS ON V2 FEATURES")

    v1_models = make_models()
    v1_grids = get_hyperparameter_grids()
    v1_names = [LOGISTIC_REGRESSION, RANDOM_FOREST, XGBOOST, LIGHTGBM]

    v1_tuned = {}
    v1_results = {}

    for name in v1_names:
        pipeline = build_pipeline(v1_models[name])
        best = tune_model(name, pipeline, v1_grids[name], X_v2_train, y_train)
        est = best.best_estimator_
        v1_tuned[name] = est
        y_pred, y_prob = predict_estimator(est, X_v2_test)
        metrics = evaluate_model(name, est, X_v2_test, y_test, y_pred, y_prob)
        v1_results[name] = metrics
        print(f"  {name:>22s}: F1={metrics['f1']:.4f} AUC={metrics['roc_auc']:.4f}")

    # ------------------------------------------------------------------
    # 5. Tune v2 models (SVM, k-NN, MLP)
    # ------------------------------------------------------------------
    _print_section("STEP 5: TUNE V2 NON-LINEAR MODELS")

    v2_models = make_nonlinear_models()
    v2_grids = get_nonlinear_grids()
    v2_names = [SVM, KNN, MLP]

    v2_tuned = {}
    v2_results = {}

    for name in v2_names:
        pipeline = build_pipeline(v2_models[name])
        best = tune_model(name, pipeline, v2_grids[name], X_v2_train, y_train)
        est = best.best_estimator_
        v2_tuned[name] = est
        y_pred, y_prob = predict_estimator(est, X_v2_test)
        metrics = evaluate_model(name, est, X_v2_test, y_test, y_pred, y_prob)
        v2_results[name] = metrics
        print(f"  {name:>22s}: F1={metrics['f1']:.4f} AUC={metrics['roc_auc']:.4f}")

    # ------------------------------------------------------------------
    # 6. Calibrate on VALIDATION, evaluate on TEST (honest OOS)
    # ------------------------------------------------------------------
    _print_section("STEP 6: VALIDATION-BASED PROBABILITY CALIBRATION")

    all_tuned = {**v1_tuned, **v2_tuned}
    calibration_results = {}
    calibrated_probas_test = {}

    for name, est in all_tuned.items():
        y_prob_val = predict_proba_safe(est, X_v2_val)
        _, calibrator = calibrate_isotonic(y_val.values, y_prob_val)

        y_prob_test = predict_proba_safe(est, X_v2_test)
        y_prob_cal_test = calibrator.predict(y_prob_test)

        cal_result = assess_calibration(y_test.values, y_prob_test, y_prob_cal_test)
        calibration_results[name] = cal_result
        calibrated_probas_test[name] = y_prob_cal_test

        brier_improvement = cal_result.brier_before - cal_result.brier_after
        print(f"  {name:>22s}: Brier {cal_result.brier_before:.4f} -> "
              f"{cal_result.brier_after:.4f} (delta={brier_improvement:+.4f}) "
              f"ECE {cal_result.ece_before:.4f} -> {cal_result.ece_after:.4f}")

    # ------------------------------------------------------------------
    # 7. Calibrated voting ensemble (test-set evaluation)
    # ------------------------------------------------------------------
    _print_section("STEP 7: CALIBRATED VOTING ENSEMBLE")

    all_probas = np.array([calibrated_probas_test[name] for name in all_tuned.keys()])
    ensemble_proba = all_probas.mean(axis=0)
    ensemble_pred = (ensemble_proba >= 0.5).astype(int)

    ensemble_metrics = {
        "accuracy": float(accuracy_score(y_test, ensemble_pred)),
        "f1": float(f1_score(y_test, ensemble_pred)),
        "roc_auc": float(roc_auc_score(y_test, ensemble_proba)),
        "brier": float(brier_score_loss(y_test, ensemble_proba)),
        "logloss": float(log_loss(y_test, np.clip(ensemble_proba, 1e-7, 1 - 1e-7))),
    }

    print(f"  Calibrated Voting: F1={ensemble_metrics['f1']:.4f} "
          f"AUC={ensemble_metrics['roc_auc']:.4f}")

    # ------------------------------------------------------------------
    # 8. Threshold optimisation on VALIDATION; evaluate on TEST
    # ------------------------------------------------------------------
    _print_section("STEP 8: VALIDATION-BASED THRESHOLD OPTIMIZATION")

    all_probas_val = np.array([
        predict_proba_safe(est, X_v2_val) for est in all_tuned.values()
    ])
    ensemble_proba_val = all_probas_val.mean(axis=0)

    thresh_f1 = optimise_threshold(y_val.values, ensemble_proba_val, metric="f1")
    thresh_acc = optimise_threshold(y_val.values, ensemble_proba_val, metric="accuracy")

    def _eval_at_threshold(probs: np.ndarray, threshold: float) -> dict[str, float]:
        preds = (probs >= threshold).astype(int)
        return {
            "accuracy": float(accuracy_score(y_test, preds)),
            "f1": float(f1_score(y_test, preds)),
        }

    oos_f1 = _eval_at_threshold(ensemble_proba, thresh_f1["best_threshold"])
    oos_acc = _eval_at_threshold(ensemble_proba, thresh_acc["best_threshold"])

    threshold_results = {
        "f1_optimal": {
            "threshold": float(thresh_f1["best_threshold"]),
            "validation_f1": float(thresh_f1["best_score"]),
            "test_f1": oos_f1["f1"],
            "test_accuracy": oos_f1["accuracy"],
        },
        "accuracy_optimal": {
            "threshold": float(thresh_acc["best_threshold"]),
            "validation_accuracy": float(thresh_acc["best_score"]),
            "test_accuracy": oos_acc["accuracy"],
            "test_f1": oos_acc["f1"],
        },
    }
    print(f"  F1-optimal threshold: {thresh_f1['best_threshold']:.3f} "
          f"(val F1={thresh_f1['best_score']:.4f}, test F1={oos_f1['f1']:.4f}, "
          f"test acc={oos_f1['accuracy']:.4f})")
    print(f"  Acc-optimal threshold: {thresh_acc['best_threshold']:.3f} "
          f"(val acc={thresh_acc['best_score']:.4f}, test acc={oos_acc['accuracy']:.4f})")

    # ------------------------------------------------------------------
    # 9. Save results
    # ------------------------------------------------------------------
    _print_section("STEP 9: SAVING RESULTS")

    eda_dir = pathlib.Path("plots_v2")
    eda_dir.mkdir(exist_ok=True)

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

    threshold_path = eda_dir / "threshold_oos.json"
    threshold_path.write_text(json.dumps(threshold_results, indent=2), encoding="utf-8")

    print(f"Results saved to {results_path}")
    print(f"Calibration saved to {cal_path}")
    print(f"Threshold saved to {threshold_path}")

    # ------------------------------------------------------------------
    # 10. Final summary
    # ------------------------------------------------------------------
    _print_section("FINAL RESULTS (v2 pipeline, leakage-free)")

    print(f"{'Model':>22s} {'Accuracy':>10s} {'F1':>10s} {'AUC':>10s} {'Brier':>10s}")
    for name, metrics in sorted(all_results.items(), key=lambda x: x[1].get("f1", 0), reverse=True):
        acc = metrics.get("accuracy", 0)
        f1 = metrics.get("f1", 0)
        auc = metrics.get("roc_auc", metrics.get("auc", 0))
        brier = metrics.get("brier", 0)
        print(f"  {name:>22s}: {acc:>10.4f} {f1:>10.4f} {auc:>10.4f} {brier:>10.4f}")

    best_name = max(all_results, key=lambda k: all_results[k].get("f1", 0))
    print(f"\nBest model: {best_name} (F1={all_results[best_name]['f1']:.4f})")
