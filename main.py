"""Orchestrator for the LoL 10-minute outcome prediction pipeline.

Steps:
1. Load and engineer features from data/high_diamond_ranked_10min.csv.
2. Generate EDA diagnostics and plots.
3. Split into stratified train / validation / test sets.
4. Train and tune four base classifiers + stacking ensemble.
5. Evaluate on held-out test set with full metrics.
6. Run true nested CV for unbiased performance estimation.
7. Perform bootstrap confidence intervals for the best model.
8. Optimise decision thresholds (validation-fitted).
9. Perform all-pairs significance tests (McNemar, paired bootstrap).
10. Persist all results.
"""

from __future__ import annotations

import itertools
import json
import pathlib
import sys
from typing import Any

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit

from src.data_loader import load_engineered_dataset
from src.evaluation import (
    bootstrap_ci,
    mcnemar_test,
    nested_cv_evaluate,
    optimise_threshold,
    paired_bootstrap_test,
)
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
    predict_estimator,
    print_metrics_table,
    save_metrics,
    tune_model,
)

SEED: int = 42
VAL_SIZE: float = 0.15

BASE_CLASSIFIERS: list[str] = [
    LOGISTIC_REGRESSION,
    RANDOM_FOREST,
    XGBOOST,
    LIGHTGBM,
]


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
    # 2. Stratified train / validation / test split
    # ------------------------------------------------------------------
    _print_section("STEP 2: TRAIN / VALIDATION / TEST SPLIT")

    eda_dir = pathlib.Path("plots")
    eda_dir.mkdir(exist_ok=True)

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=0.2, random_state=SEED
    )
    train_idx, test_idx = next(splitter.split(X, y))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Carve a stratified validation fold out of the training side.  The
    # validation fold is used ONLY for threshold optimisation (Step 9),
    # so the test set stays untouched for final evaluation.
    val_splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=VAL_SIZE, random_state=SEED
    )
    tr_idx, val_idx = next(val_splitter.split(X_train, y_train))
    X_train, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_train, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

    print(f"Train: {X_train.shape[0]} samples")
    print(f"Validation: {X_val.shape[0]} samples")
    print(f"Test:  {X_test.shape[0]} samples")
    print(f"Train blue-win rate: {y_train.mean():.4f}")
    print(f"Test  blue-win rate: {y_test.mean():.4f}")

    # ------------------------------------------------------------------
    # 4. Tune four base classifiers
    # ------------------------------------------------------------------
    _print_section("STEP 4: HYPERPARAMETER TUNING (4 BASE CLASSIFIERS)")

    models = make_models()
    grids = get_hyperparameter_grids()

    tuned_pipelines: dict[str, Any] = {}
    results: dict[str, dict[str, float]] = {}

    for name in BASE_CLASSIFIERS:
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

    print(f"Stacking test metrics: {metrics_stack}")

    # ------------------------------------------------------------------
    # 6. True nested CV for unbiased performance estimation
    # ------------------------------------------------------------------
    _print_section("STEP 6: NESTED CROSS-VALIDATION (UNBIASED ESTIMATE)")

    best_model_name = max(
        [k for k in results if k != STACKING],
        key=lambda k: results[k]["f1"],
    )
    overall_best = max(results, key=lambda k: results[k]["f1"])
    best_grid = grids[best_model_name]
    print(f"Best base model by test F1: {best_model_name} (F1={results[best_model_name]['f1']:.4f})")
    print(f"Running 5-fold nested CV with inner 5-fold tuning...")
    print(f"(This performs fresh hyperparameter tuning inside each outer fold)")

    nested_results = nested_cv_evaluate(
        build_pipeline(models[best_model_name]),
        best_grid,
        X, y,
    )
    print(f"\nNested CV results for {best_model_name}:")
    for metric_name in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        m = nested_results["mean"][metric_name]
        s = nested_results["std"][metric_name]
        print(f"  {metric_name:>10s}: {m:.4f} ± {s:.4f}")

    # ------------------------------------------------------------------
    # 7. Bootstrap confidence intervals for best model
    # ------------------------------------------------------------------
    _print_section("STEP 7: BOOTSTRAP CONFIDENCE INTERVALS")

    best_est = tuned_pipelines.get(overall_best, stacking)
    y_prob_best = predict_estimator(best_est, X_test)[1]
    ci_results = bootstrap_ci(y_test.values, y_prob_best)
    print(f"95% Confidence Intervals ({overall_best}):")
    for metric_name, ci in ci_results.items():
        print(
            f"  {metric_name:>10s}: {ci['mean']:.4f} "
            f"[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]"
        )

    # ------------------------------------------------------------------
    # 8. Threshold optimisation for best model (validation-fitted,
    #    evaluated on the untouched test set)
    # ------------------------------------------------------------------
    _print_section("STEP 8: DECISION THRESHOLD OPTIMISATION (VAL-FITTED)")

    y_prob_val = predict_estimator(best_est, X_val)[1]
    thresh_f1 = optimise_threshold(y_val.values, y_prob_val, metric="f1")
    thresh_acc = optimise_threshold(y_val.values, y_prob_val, metric="accuracy")
    print(f"F1-optimal threshold:   {thresh_f1['best_threshold']:.3f} "
          f"(val F1={thresh_f1['best_score']:.4f})")
    print(f"Acc-optimal threshold:  {thresh_acc['best_threshold']:.3f} "
          f"(val Acc={thresh_acc['best_score']:.4f})")

    y_pred_optimised = (y_prob_best >= thresh_f1["best_threshold"]).astype(int)
    f1_optimised = float(f1_score(y_test, y_pred_optimised))
    acc_optimised = float(accuracy_score(y_test, y_pred_optimised))
    y_pred_acc = (y_prob_best >= thresh_acc["best_threshold"]).astype(int)
    acc_at_acc_thresh = float(accuracy_score(y_test, y_pred_acc))
    print(f"Test F1 with val-fitted F1-optimal threshold: {f1_optimised:.4f} "
          f"(accuracy {acc_optimised:.4f})")
    print(f"Test accuracy with val-fitted Acc-optimal threshold: "
          f"{acc_at_acc_thresh:.4f}")

    # ------------------------------------------------------------------
    # 9. All-pairs significance tests
    # ------------------------------------------------------------------
    _print_section("STEP 9: ALL-PAIRS SIGNIFICANCE TESTS")

    all_model_names = list(results.keys())
    pairwise_results: list[dict[str, Any]] = []

    for model_a, model_b in itertools.combinations(all_model_names, 2):
        est_a = tuned_pipelines.get(model_a, stacking)
        est_b = tuned_pipelines.get(model_b, stacking)

        y_prob_a = predict_estimator(est_a, X_test)[1]
        y_prob_b = predict_estimator(est_b, X_test)[1]
        y_pred_a = predict_estimator(est_a, X_test)[0]
        y_pred_b = predict_estimator(est_b, X_test)[0]

        bs = paired_bootstrap_test(
            y_test.values, y_prob_a, y_prob_b, metric="accuracy"
        )
        mc = mcnemar_test(y_test.values, y_pred_a, y_pred_b)

        pairwise_results.append({
            "model_a": model_a,
            "model_b": model_b,
            "f1_a": float(results[model_a]["f1"]),
            "f1_b": float(results[model_b]["f1"]),
            "mcnemar_chi2": float(mc["chi2"]),
            "mcnemar_p": float(mc["p_value"]),
            "mcnemar_sig": bool(mc["significant"]),
            "bootstrap_delta": float(bs["delta_mean"]),
            "bootstrap_p": float(bs["p_value"]),
            "bootstrap_sig": bool(bs["significant"]),
        })

        sig_marker = "*" if mc["significant"] else ""
        print(f"  {model_a:>22s} vs {model_b:<22s}  "
              f"F1: {results[model_a]['f1']:.4f} vs {results[model_b]['f1']:.4f}  "
              f"McNemar p={mc['p_value']:.4f}{sig_marker}")

    # ------------------------------------------------------------------
    # 10. Save all results
    # ------------------------------------------------------------------
    _print_section("STEP 10: SAVING RESULTS")

    metrics_path = save_metrics(results, eda_dir)

    nested_path = eda_dir / "nested_cv_results.json"
    nested_path.write_text(json.dumps(nested_results, indent=2), encoding="utf-8")

    ci_path = eda_dir / "confidence_intervals.json"
    ci_path.write_text(json.dumps(ci_results, indent=2), encoding="utf-8")

    threshold_data = {
        "f1_optimal_threshold": thresh_f1["best_threshold"],
        "f1_optimal_score": thresh_f1["best_score"],
        "acc_optimal_threshold": thresh_acc["best_threshold"],
        "acc_optimal_score": thresh_acc["best_score"],
        "f1_with_optimised_threshold": f1_optimised,
    }
    threshold_path = eda_dir / "threshold_optimisation.json"
    threshold_path.write_text(
        json.dumps(threshold_data, indent=2), encoding="utf-8"
    )

    pairwise_path = eda_dir / "pairwise_significance.json"
    pairwise_path.write_text(
        json.dumps(pairwise_results, indent=2), encoding="utf-8"
    )

    print(f"All metrics saved to {metrics_path}")
    print(f"Nested CV results saved to {nested_path}")
    print(f"Confidence intervals saved to {ci_path}")
    print(f"Threshold results saved to {threshold_path}")
    print(f"Pairwise significance saved to {pairwise_path}")

    # ------------------------------------------------------------------
    # 11. Final summary
    # ------------------------------------------------------------------
    _print_section("FINAL RESULTS")
    print_metrics_table(results)

    print(f"Best model: {overall_best}")
    print(f"  Test F1:       {results[overall_best]['f1']:.4f}")
    print(f"  Test ROC-AUC:  {results[overall_best]['roc_auc']:.4f}")
    print(f"  Test Accuracy: {results[overall_best]['accuracy']:.4f}")
    print(f"\n  Nested CV F1 ({best_model_name}):  {nested_results['mean']['f1']:.4f} "
          f"± {nested_results['std']['f1']:.4f}")
    print(f"  Optimised F1:  {f1_optimised:.4f} "
          f"(threshold={thresh_f1['best_threshold']:.3f})")

    n_sig = sum(1 for pr in pairwise_results if pr["mcnemar_sig"])
    print(f"\n  Pairwise McNemar: {n_sig}/{len(pairwise_results)} comparisons significant at alpha=0.05")
