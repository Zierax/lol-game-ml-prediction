"""Model zoo v2: non-linear classifiers with native probability outputs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

SEED: int = 42

SVM = "svm"
KNN = "knn"
MLP = "mlp"
VOTING_SOFT = "voting_soft"


def make_nonlinear_models() -> dict[str, BaseEstimator]:
    """Instantiate non-linear classifiers."""
    return {
        SVM: SVC(
            kernel="rbf",
            probability=True,
            random_state=SEED,
            cache_size=500,
        ),
        KNN: KNeighborsClassifier(
            weights="distance",
        ),
        MLP: MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            learning_rate="adaptive",
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=SEED,
        ),
    }


def get_nonlinear_grids() -> dict[str, dict[str, list[Any]]]:
    """Return hyperparameter grids for non-linear models."""
    return {
        SVM: {
            "model__C": [0.1, 1.0, 10.0, 100.0],
            "model__gamma": ["scale", "auto", 0.01, 0.001],
        },
        KNN: {
            "model__n_neighbors": [5, 11, 21, 31, 51],
            "model__metric": ["euclidean", "manhattan"],
        },
        MLP: {
            "model__alpha": [0.0001, 0.001, 0.01],
            "model__learning_rate_init": [0.001, 0.01],
        },
    }


def build_voting_ensemble(
    base_estimators: dict[str, BaseEstimator],
) -> VotingClassifier:
    """Build a soft voting ensemble from calibrated base estimators."""
    estimators = [(name, est) for name, est in base_estimators.items()]
    return VotingClassifier(
        estimators=estimators,
        voting="soft",
        n_jobs=1,
    )


def predict_proba_safe(
    estimator: BaseEstimator,
    X: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    """Safely get probability predictions."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    elif hasattr(estimator, "decision_function"):
        decisions = estimator.decision_function(X)
        return 1.0 / (1.0 + np.exp(-decisions))
    else:
        raise ValueError(f"Estimator {type(estimator).__name__} has no probability output")
