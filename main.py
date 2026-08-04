"""Orchestrator for the LoL 10-minute outcome prediction pipeline.

Steps:
1. Load and engineer features from data/high_diamond_ranked_10min.csv.
2. Generate EDA diagnostics and plots.
3. Split into stratified train / test sets.
4. Train and tune four base classifiers + stacking ensemble.
5. Evaluate on held-out test set with full metrics.
6. Run nested CV for unbiased performance estimation.
7. Perform pairwise significance tests.
8. Optimise decision thresholds.
9. Persist all results and visualisations.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedShuffleSplit

from src.data_loader import load_engineered_dataset
from src.eda import generate_eda_report
from src.evaluation import (
    bootstrap_ci,
    mcnemar_test,
    nested_cv_evaluate,
    optimise_threshold,
    paired_bootstrap_test,
)
from src.features import engineer_features
from src.models import (
    LIGHTGBM,
    LOGISTIC_REGRESSION,
    RANDOM_FOREST,
    STACKING,
    XGBOOST,
    build_pipeline,
    build_stacking_ensemble,
    evaluate_model,
    get_hyperparameter_grids,
    make_models,
    plot_confusion_matrix,
    plot_roc_curves,
    predict_estimator,
    print_metrics_table,
    save_metrics,
    tune_model,
)

SEED: int = 42


def _print_section(title: str) -> None:
    width = 78
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width + "\n")


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 1. Load data with engineered features
    # ------------------------------------------------------------------
    _print_section("STEP 1: DATA LOADING & FEATURE ENGINEERING")

    dataset_file = pathlib.Path("data/high_diamond_ranked_10min.csv")
    if not dataset_file.exists():
        sys.stderr.write(f"Missing dataset at {dataset_file}\n")
        sys.exit(1)

    X, y = load_engineered_dataset(dataset_file)
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class balance: blueWins mean = {y.mean():.4f}")

    # ------------------------------------------------------------------
    # 2. EDA reports and diagnostics
    # ------------------------------------------------------------------
    _print_section("STEP 2: EXPLORATORY DATA ANALYSIS")

    eda_dir = pathlib.Path("plots")
    eda_dir.mkdir(exist_ok=True)
    generate_eda_report(X, y, eda_dir)
    print(f"EDA plots saved to {eda_dir}/")

    # ------------------------------------------------------------------
    # 3. Stratified train / test split
    # ------------------------------------------------------------------
    _print_section("STEP 3: TRAIN / TEST SPLIT")

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=0.2, random_state=SEED
    )
    train_idx, test_idx = next(splitter.split(X, y))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print(f"Train: {X_train.shape[0]} samples")
    print(f"Test:  {X_test.shape[0]} samples")
    print(f"Train blue-win rate: {y_train.mean():.4f}")
    print(f"Test  blue-win rate: {y_test.mean():.4f}")

    # ------------------------------------------------------------------
    # 4. Tune four base classifiers
    # ------------------------------------------------------------------
    _print_section("STEP 4: HYPERPARAMETER TUNING (4 BASE CLASSIFIERS)")

    models = make_models()
    grids = get_hyperparameter_grids()

    tuned_pipelines: dict[str, any] = {}
    results: dict[str, dict[str, float]] = {}
    roc_displays: dict[str, RocCurveDisplay] = {}

    for name in [LOGISTIC_REGRESSION, RANDOM_FOREST, XGBOOST, LIGHTGBM]:
        base = models[name]
        pipeline = build_pipeline(base)
        print(f"\nTuning {name}...")
        best = tune_model(name, pipeline, grids[name], X_train, y_train)
        print(f"  Best params: {best.best_params_}")
        print(f"  Best CV accuracy: {best.best_score_:.4f}")

        est = best.best_estimator_
        tuned_pipelines[name] = est

        y_pred, y_prob = predict_estimator(est, X_test)
        metrics = evaluate_model(name, est, X_test, y_test, y_pred, y_prob)
        results[name] = metrics

        roc_display = RocCurveDisplay.from_estimator(
            est, X_test, y_test, name=name
        )
        roc_displays[name] = roc_display
        plot_confusion_matrix(name, est, X_test, y_test, eda_dir)

    # ------------------------------------------------------------------
    # 5. Build and evaluate stacking ensemble
    # ------------------------------------------------------------------
    _print_section("STEP 5: STACKING ENSEMBLE")

    print("Building stacking ensemble from 4 base estimators...")
    stacking = build_stacking_ensemble(tuned_pipelines)
    stacking.fit(X_train, y_train)

    y_pred_stack, y_prob_stack = predict_estimator(stacking, X_test)
    metrics_stack = evaluate_model(
        STACKING, stacking, X_test, y_test, y_pred_stack, y_prob_stack
    )
    results[STACKING] = metrics_stack

    roc_display_stack = RocCurveDisplay.from_estimator(
        stacking, X_test, y_test, name=STACKING
    )
    roc_displays[STACKING] = roc_display_stack
    plot_confusion_matrix(STACKING, stacking, X_test, y_test, eda_dir)

    print(f"Stacking test metrics: {metrics_stack}")

    # ------------------------------------------------------------------
    # 6. Plot ROC curves
    # ------------------------------------------------------------------
    _print_section("STEP 6: ROC CURVES")
    plot_roc_curves(roc_displays, eda_dir)
    print(f"ROC curves saved to {eda_dir}/roc_curve.png")

    # ------------------------------------------------------------------
    # 7. Nested CV evaluation of the best model
    # ------------------------------------------------------------------
    _print_section("STEP 7: NESTED CROSS-VALIDATION (UNBIASED ESTIMATE)")

    best_model_name = max(results, key=lambda k: results[k]["f1"])
    best_pipeline = tuned_pipelines.get(best_model_name, stacking)
    print(f"Best model: {best_model_name} (F1={results[best_model_name]['f1']:.4f})")
    print("Running 10-fold nested CV (may take a few minutes)...")

    nested_results = nested_cv_evaluate(best_pipeline, X, y)
    print(f"\nNested CV results for {best_model_name}:")
    for metric_name in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        m = nested_results["mean"][metric_name]
        s = nested_results["std"][metric_name]
        print(f"  {metric_name:>10s}: {m:.4f} ± {s:.4f}")

    # ------------------------------------------------------------------
    # 8. Bootstrap confidence intervals for best model
    # ------------------------------------------------------------------
    _print_section("STEP 8: BOOTSTRAP CONFIDENCE INTERVALS")

    best_est = tuned_pipelines.get(best_model_name, stacking)
    y_prob_best = predict_estimator(best_est, X_test)[1]
    ci_results = bootstrap_ci(y_test.values, y_prob_best)
    print(f"95% Confidence Intervals ({best_model_name}):")
    for metric_name, ci in ci_results.items():
        print(
            f"  {metric_name:>10s}: {ci['mean']:.4f} "
            f"[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]"
        )

    # ------------------------------------------------------------------
    # 9. Threshold optimisation for best model
    # ------------------------------------------------------------------
    _print_section("STEP 9: DECISION THRESHOLD OPTIMISATION")

    thresh_f1 = optimise_threshold(y_test.values, y_prob_best, metric="f1")
    thresh_acc = optimise_threshold(y_test.values, y_prob_best, metric="accuracy")
    print(f"F1-optimal threshold:   {thresh_f1['best_threshold']:.3f} "
          f"(F1={thresh_f1['best_score']:.4f})")
    print(f"Acc-optimal threshold:  {thresh_acc['best_threshold']:.3f} "
          f"(Acc={thresh_acc['best_score']:.4f})")

    # Apply optimised threshold
    y_pred_optimised = (y_prob_best >= thresh_f1["best_threshold"]).astype(int)
    from sklearn.metrics import f1_score as _f1
    f1_optimised = float(_f1(y_test, y_pred_optimised))
    print(f"F1 with optimised threshold: {f1_optimised:.4f}")

    # ------------------------------------------------------------------
    # 10. Pairwise significance tests (best vs. runner-up)
    # ------------------------------------------------------------------
    _print_section("STEP 10: PAIRWISE SIGNIFICANCE TESTS")

    sorted_models = sorted(results.keys(), key=lambda k: results[k]["f1"], reverse=True)
    if len(sorted_models) >= 2:
        model_a = sorted_models[0]
        model_b = sorted_models[1]
        print(f"Comparing {model_a} vs {model_b}:")

        y_prob_a = predict_estimator(
            tuned_pipelines.get(model_a, stacking), X_test
        )[1]
        y_prob_b = predict_estimator(
            tuned_pipelines.get(model_b, stacking), X_test
        )[1]

        bootstrap_result = paired_bootstrap_test(
            y_test.values, y_prob_a, y_prob_b, metric="accuracy"
        )
        print(f"\n  Paired bootstrap test (accuracy):")
        print(f"    Delta:    {bootstrap_result['delta_mean']:+.4f}")
        print(f"    p-value:  {bootstrap_result['p_value']:.4f}")
        print(f"    95% CI:   [{bootstrap_result['ci_lower']:.4f}, "
              f"{bootstrap_result['ci_upper']:.4f}]")
        print(f"    Significant at alpha=0.05: {bootstrap_result['significant']}")

        y_pred_a = predict_estimator(
            tuned_pipelines.get(model_a, stacking), X_test
        )[0]
        y_pred_b = predict_estimator(
            tuned_pipelines.get(model_b, stacking), X_test
        )[0]

        mcnemar_result = mcnemar_test(y_test.values, y_pred_a, y_pred_b)
        print(f"\n  McNemar's test:")
        print(f"    chi2:     {mcnemar_result['chi2']:.4f}")
        print(f"    p-value:  {mcnemar_result['p_value']:.4f}")
        print(f"    Significant at alpha=0.05: {mcnemar_result['significant']}")

    # ------------------------------------------------------------------
    # 11. Save all results
    # ------------------------------------------------------------------
    _print_section("STEP 11: SAVING RESULTS")

    metrics_path = save_metrics(results, eda_dir)

    # Save nested CV results
    nested_path = eda_dir / "nested_cv_results.json"
    nested_path.write_text(json.dumps(nested_results, indent=2), encoding="utf-8")

    # Save CI results
    ci_path = eda_dir / "confidence_intervals.json"
    ci_path.write_text(json.dumps(ci_results, indent=2), encoding="utf-8")

    # Save threshold results
    threshold_path = eda_dir / "threshold_optimisation.json"
    threshold_data = {
        "f1_optimal_threshold": thresh_f1["best_threshold"],
        "f1_optimal_score": thresh_f1["best_score"],
        "acc_optimal_threshold": thresh_acc["best_threshold"],
        "acc_optimal_score": thresh_acc["best_score"],
        "f1_with_optimised_threshold": f1_optimised,
    }
    threshold_path.write_text(
        json.dumps(threshold_data, indent=2), encoding="utf-8"
    )

    print(f"All metrics saved to {metrics_path}")
    print(f"Nested CV results saved to {nested_path}")
    print(f"Confidence intervals saved to {ci_path}")
    print(f"Threshold results saved to {threshold_path}")

    # ------------------------------------------------------------------
    # 12. Final summary
    # ------------------------------------------------------------------
    _print_section("FINAL RESULTS")
    print_metrics_table(results)

    print(f"Best model: {best_model_name}")
    print(f"  Test F1:       {results[best_model_name]['f1']:.4f}")
    print(f"  Test ROC-AUC:  {results[best_model_name]['roc_auc']:.4f}")
    print(f"  Test Accuracy: {results[best_model_name]['accuracy']:.4f}")
    print(f"\n  Nested CV F1:  {nested_results['mean']['f1']:.4f} "
          f"± {nested_results['std']['f1']:.4f}")
    print(f"  Optimised F1:  {f1_optimised:.4f} "
          f"(threshold={thresh_f1['best_threshold']:.3f})")
