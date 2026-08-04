"""Orchestrator for the LoL 10‑minute outcome prediction pipeline.

Steps:
1. Load data from data/high_diamond_ranked_10min.csv.
2. Generate EDA diagnostics and plots.
3. Split into stratified train / test sets.
4. Train and tune Logistic Regression, Random Forest and XGBoost.
5. Evaluate models, persist ROC curves, confusion matrices and a metrics table.
6. Emit a concise comparison table to stdout for quick inspection.
"""

from __future__ import annotations

import pathlib
import sys

from src.data_loader import load_dataset
from src.eda import generate_eda_report
from src.models import (
    make_models,
    get_hyperparameter_grids,
    build_pipeline,
    tune_model,
    evaluate_model,
    predict_estimator,
    plot_roc_curves,
    plot_confusion_matrix,
    save_metrics,
    print_metrics_table,
)
from sklearn.metrics import RocCurveDisplay

# Constants
# Removed placeholder dataset path

if __name__ == "__main__":
    # Resolve the dataset path reliably
    dataset_file = pathlib.Path("data/high_diamond_ranked_10min.csv")
    if not dataset_file.exists():
        sys.stderr.write(f"Missing dataset at {dataset_file}\n")
        sys.exit(1)
    X, y = load_dataset(dataset_file)

    # EDA reports and diagnostics
    eda_dir = pathlib.Path("plots")
    eda_dir.mkdir(exist_ok=True)
    generate_eda_report(X, y, eda_dir)

    # Train / test split
    from sklearn.model_selection import StratifiedShuffleSplit

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    models = make_models()
    grids = get_hyperparameter_grids()

    results: dict[str, dict[str, float]] = {}
    roc_displays = {}
    for name, base in models.items():
        pipeline = build_pipeline(base)
        print(f"Tuning {name}…")
        best = tune_model(name, pipeline, grids[name], X_train, y_train)
        print(f"Best params for {name}: {best.best_params_}")
        est = best.best_estimator_
        y_pred, y_prob = predict_estimator(est, X_test)
        metrics = evaluate_model(name, est, X_test, y_test, y_pred, y_prob)
        results[name] = metrics
        # ROC curve display
        roc_display = RocCurveDisplay.from_estimator(est, X_test, y_test, name=name)
        roc_displays[name] = roc_display
        # Confusion matrix plot
        plot_confusion_matrix(name, est, X_test, y_test, eda_dir)

    # Plot ROC curves and save metrics
    plot_roc_curves(roc_displays, eda_dir)
    metrics_path = save_metrics(results, eda_dir)

    print_metrics_table(results)
    print(f"Metrics saved to {metrics_path}")
