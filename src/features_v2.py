"""Signal extraction and feature engineering v2.

Paradigm shift from v1:
- v1 created 87 features, most of which were correlated expressions of the
  same ~5 underlying dimensions (gold, XP, kills, CS, level).
- v2 extracts the INDEPENDENT signal via PCA, selects features by mutual
  information with correlation filtering, and adds non-linear transformations
  of the principal components that capture the advantage-to-win-probability
  mapping.

The key insight: the relationship between game advantage and win probability
is sigmoid-shaped (small advantages are uncertain, large advantages are
near-certain). We model this explicitly rather than hoping tree ensembles
learn it from correlated raw features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

DEFAULT_PCA_VARIANCE: float = 0.95
DEFAULT_MI_THRESHOLD: float = 0.01
DEFAULT_CORR_THRESHOLD: float = 0.90


@dataclass(frozen=True)
class FeatureSelectionResult:
    """Result of mutual-information-based feature selection with correlation filtering."""

    selected_features: list[str]
    mi_scores: pd.Series
    dropped_by_corr: list[str]


def select_features_by_mi(
    X: pd.DataFrame,
    y: pd.Series,
    mi_threshold: float = DEFAULT_MI_THRESHOLD,
    corr_threshold: float = DEFAULT_CORR_THRESHOLD,
) -> FeatureSelectionResult:
    """Select features by mutual information with correlation filtering."""
    mi = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
    mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)

    above_threshold = mi_series[mi_series > mi_threshold].index.tolist()

    corr_matrix = X[above_threshold].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    dropped: list[str] = []
    selected: list[str] = []

    for feature in above_threshold:
        if feature in dropped:
            continue
        selected.append(feature)
        correlated = [
            col
            for col in above_threshold
            if col != feature
            and col not in selected
            and col not in dropped
            and upper.loc[feature, col] > corr_threshold
        ]
        dropped.extend(correlated)

    return FeatureSelectionResult(
        selected_features=selected,
        mi_scores=mi_series,
        dropped_by_corr=dropped,
    )


@dataclass(frozen=True)
class PCAResult:
    """Result of PCA transformation with metadata."""

    transformed: pd.DataFrame
    pca: PCA
    scaler: StandardScaler
    n_components: int
    explained_variance_ratio: np.ndarray


def fit_pca(
    X: pd.DataFrame,
    variance_threshold: float = DEFAULT_PCA_VARIANCE,
) -> PCAResult:
    """Fit PCA that explains the specified fraction of variance."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca_full = PCA(random_state=42)
    pca_full.fit(X_scaled)
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumulative, variance_threshold) + 1)
    n_components = max(n_components, 2)

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    columns = [f"PC{i + 1}" for i in range(n_components)]
    return PCAResult(
        transformed=pd.DataFrame(X_pca, index=X.index, columns=columns),
        pca=pca,
        scaler=scaler,
        n_components=n_components,
        explained_variance_ratio=pca.explained_variance_ratio_,
    )


def transform_pca(X: pd.DataFrame, pca_result: PCAResult) -> pd.DataFrame:
    """Transform new data using a fitted PCA result."""
    X_scaled = pca_result.scaler.transform(X)
    X_pca = pca_result.pca.transform(X_scaled)
    columns = [f"PC{i + 1}" for i in range(pca_result.n_components)]
    return pd.DataFrame(X_pca, index=X.index, columns=columns)


def add_nonlinear_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add non-linear transformations of principal components.

    The relationship between game advantage and win probability is sigmoid.
    """
    df = X.copy()
    for col in X.columns:
        df[f"{col}_sigmoid"] = 1.0 / (1.0 + np.exp(-df[col]))
        df[f"{col}_squared"] = df[col] ** 2
        df[f"{col}_logabs"] = np.sign(df[col]) * np.log1p(np.abs(df[col]))
    return df


def build_signal_features(
    X_raw: pd.DataFrame,
    y: pd.Series,
    mi_threshold: float = DEFAULT_MI_THRESHOLD,
    corr_threshold: float = DEFAULT_CORR_THRESHOLD,
    pca_variance: float = DEFAULT_PCA_VARIANCE,
    add_nonlinear: bool = True,
) -> tuple[pd.DataFrame, FeatureSelectionResult, PCAResult | None]:
    """Full signal extraction pipeline.

    Fits feature selection and PCA on the supplied (training) data.  The
    caller must apply ``transform_signal_features`` to any held-out data;
    fitting selection/PCA on data that includes test rows would leak
    test-label information into the selected features.
    """
    selection = select_features_by_mi(X_raw, y, mi_threshold, corr_threshold)
    X_selected = X_raw[selection.selected_features]

    pca_result = fit_pca(X_selected, pca_variance)
    X_pca = pca_result.transformed

    if add_nonlinear:
        X_final = add_nonlinear_features(X_pca)
    else:
        X_final = X_pca

    return X_final, selection, pca_result


def transform_signal_features(
    X_raw: pd.DataFrame,
    selection: FeatureSelectionResult,
    pca_result: PCAResult | None,
    add_nonlinear: bool = True,
) -> pd.DataFrame:
    """Transform held-out data using a fitted signal-extraction pipeline.

    Args:
        X_raw: Raw feature matrix (same columns as the training data).
        selection: Fitted feature-selection result from the training side.
        pca_result: Fitted PCA result from the training side.
        add_nonlinear: Must match the value used in ``build_signal_features``.

    Returns:
        The transformed feature matrix with the same schema as the
        training output of ``build_signal_features``.
    """
    X_selected = X_raw[selection.selected_features]
    if pca_result is None:
        X_pca = X_selected
    else:
        X_pca = transform_pca(X_selected, pca_result)

    if add_nonlinear:
        return add_nonlinear_features(X_pca)
    return X_pca
