"""Information-ceiling analysis for early-game LoL outcome prediction.

The v1/v2 pipelines established that ~8 model families converge to an
accuracy ceiling near 72%.  This module quantifies *why*: it estimates
the decision-relevant information available at the 10-minute mark and
shows that every model family operates at the Fano-implied information
frontier.

Main quantities computed:

* ``H(Y)``: empirical target entropy.
* Per-feature mutual-information spectrum (signal concentration).
* Achieved mutual information ``I(p_hat; Y)`` per model family (in bits).
* Fano-implied minimum error given the achieved information, and the
  information-efficiency ratio of each family.
* PCA cumulative-variance curve (feature redundancy).
* Learning curves showing accuracy saturation with training-set size.

Fano's inequality (binary case): for a classifier whose outputs carry
``I`` bits of information about the binary target ``Y``, its error rate
``P_e`` satisfies

    H(Y|outputs) <= h(P_e)   =>   P_e >= h^{-1}(H(Y) - I)

where ``h`` is the binary entropy function and ``h^{-1}`` is its inverse
restricted to ``[0, 0.5]``.  Comparing each model's measured error to
this lower bound measures how close the model is to converting its
extracted information into optimal decisions.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_dataset
from src.features import engineer_features, features_audit
from src.zoo import (
    ALL_FAMILIES,
    BASE_FAMILIES,
    fit_and_score,
    load_master_split,
    pairwise_tests,
)
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
from src.evaluation import mcnemar_test, minimum_detectable_difference

matplotlib.use("Agg")

SEED: int = 42
TEST_SIZE: float = 0.2
MI_N_NEIGHBORS: int = 5
MI_BIN_COUNTS: tuple[int, ...] = (10, 20, 40)
TOP_N_FEATURES: int = 20
BOOTSTRAP_N_ITER: int = 2000
CI_LEVEL: float = 0.95
PERM_N: int = 200

OUTPUT_DIR: Path = Path("plots_ceiling")


# ----------------------------------------------------------------------
# Entropy and Fano bound
# ----------------------------------------------------------------------


def empirical_entropy(y: np.ndarray) -> float:
    """Compute the plug-in (empirical) entropy of the target in bits."""
    counts = np.bincount(y.astype(int), minlength=2)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def binary_entropy(p: float) -> float:
    """Binary entropy h(p) in bits."""
    p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    return float(-p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p))


def inverse_binary_entropy(h: float) -> float:
    """Inverse binary entropy ``h^{-1}`` on the range of ``h|_[0, 1/2]``.

    ``h`` is strictly increasing on ``[0, 0.5]`` with ``h(0) = 0`` and
    ``h(0.5) = 1``, so the inverse is defined on the closed interval
    ``[0, 1]`` and returns a value in ``[0, 0.5]``.  Applying Fano requires the
    argument ``H(Y) - I`` to lie in that interval, i.e. ``I <= H(Y)``; the
    caller is responsible for enforcing that, and
    :func:`fano_min_error` raises rather than clipping so a violation is
    visible instead of silently returning a perfect ceiling.
    """
    h = float(h)
    if h <= 0.0:
        return 0.0
    if h >= 1.0:
        return 0.5
    return float(brentq(lambda p: binary_entropy(p) - h, 1e-12, 0.5 - 1e-12))


def fano_min_error(mi_bits: float, h_y: float) -> float:
    """Minimum error rate for a classifier carrying ``mi_bits`` of information.

    From Fano's inequality for binary targets:
        P_e >= h^{-1}(H(Y) - I(outputs; Y)).

    For a binary target the general Fano bound
    ``H(Y|S) <= h(P_e) + P_e log2(|Y|-1)`` collapses to ``H(Y|S) <= h(P_e)``
    exactly, because ``log2(2-1) = 0``.  This is the sharp binary
    error--information relation (Feder and Merhav, *IEEE Trans. Inf. Theory*
    **1994**, *40*, 259--266): there is no slack term attributable to the
    discreteness or continuity of the score.  Equality holds if and only if
    the conditional error ``P(Y != Yhat | S = s)`` is constant in ``s``; any
    excess of the measured error over this bound reflects (i) positive
    finite-sample bias in the plug-in estimate of ``I(S;Y)`` and (ii)
    non-uniformity of the conditional error across the score range.

    Args:
        mi_bits: Achieved mutual information in **bits**.
        h_y: Target entropy in bits.

    Returns:
        The Fano minimum error, in [0, 0.5].
    """
    residual = h_y - mi_bits
    if residual < 0.0:
        raise ValueError(
            f"achieved MI ({mi_bits:.4f} bits) exceeds H(Y) ({h_y:.4f} bits); "
            "Fano's inequality is undefined for negative conditional entropy"
        )
    return inverse_binary_entropy(residual)


def fano_max_accuracy(mi_bits: float, h_y: float) -> float:
    """Maximum accuracy for a classifier carrying ``mi_bits`` of information."""
    return 1.0 - fano_min_error(mi_bits, h_y)


# ----------------------------------------------------------------------
# Mutual information estimation
# ----------------------------------------------------------------------


def achieved_mutual_information(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    n_bins: int,
    scheme: str = "equal_width",
) -> float:
    """Plug-in mutual information I(binned p_hat; Y) in **bits**.

    The continuous predicted probability is discretised into ``n_bins`` bins
    and the mutual information is computed from the joint contingency table
    of (bin, Y).

    The estimator is a plug-in (maximum-likelihood) estimator and therefore
    carries a positive finite-sample bias.  For a ``K x 2`` table estimated
    from ``N`` samples the Miller--Madow bias is
    ``(K - 1) / (2 N ln 2)`` bits, which is why the count of non-empty cells
    matters as much as the number of bins.

    Discretising the score is itself lossy: by the data-processing inequality
    ``I(binned p; Y) <= I(p; Y)``, so every binned estimate is a lower bound
    on the information the continuous score carries, and the loss shrinks as
    the bin count rises.  Consequently, *averaging over bin counts is a
    variance sensitivity check, not a bias correction* -- every term is
    biased low, and the bias differs across terms.

    Args:
        y_prob: Predicted probabilities of the positive class.
        y_true: Ground-truth binary labels.
        n_bins: Number of bins.
        scheme: ``"equal_width"`` for uniform bins on [0, 1], or
            ``"equal_mass"`` for quantile bins (which is well matched to
            score distributions that pile up near 0 and 1).

    Returns:
        The plug-in mutual information in bits (not bias-corrected).
    """
    if scheme == "equal_mass":
        # Quantile bins: cut points are the empirical quantiles, so each bin
        # holds ~N/n_bins points. Empty bins (ties) are collapsed by
        # np.unique, so the effective bin count may be smaller than requested.
        qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
        cuts = np.unique(np.quantile(y_prob, qs))
        bin_idx = np.digitize(y_prob, cuts)
        n_eff = len(cuts) + 1
    else:
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bins[0] = -np.inf
        bins[-1] = np.inf
        bin_idx = np.digitize(y_prob, bins[1:-1])
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        n_eff = n_bins

    n = len(y_true)
    n_cells = 0
    joint = np.zeros((n_eff, 2), dtype=float)
    for b in range(n_eff):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        joint[b, 0] = float((y_true[mask] == 0).sum())
        joint[b, 1] = float((y_true[mask] == 1).sum())
        n_cells += 1

    joint /= n
    p_bin = joint.sum(axis=1)
    p_y = joint.sum(axis=0)

    mi = 0.0
    for b in range(n_eff):
        if p_bin[b] <= 0:
            continue
        for k in range(2):
            if joint[b, k] <= 0:
                continue
            mi += joint[b, k] * np.log2(
                joint[b, k] / (p_bin[b] * p_y[k])
            )
    return float(mi)


def miller_madow_bias(n_bins_used: int, n: int) -> float:
    """Miller--Madow finite-sample bias of a plug-in MI, in bits.

    For a plug-in estimate of the mutual information of a contingency table
    with ``K_S`` score levels and two outcome classes, the standard result is

    .. math::
        \\mathrm{Bias} = \\frac{-(K_Y - 1) - (K_S - 1) + (K_{YS} - 1)}{2N}
                       = \\frac{K_S - 1}{2N} \\quad \\text{nats},

    where ``K_{YS} = 2 K_S`` when both classes occur in every score level.
    Converting to bits multiplies by ``1 / ln 2``.

    Note that only the *number of score levels* enters: the outcome factor
    cancels.  Using ``2 K_S`` here (the raw cell count) would double the
    correction.

    Args:
        n_bins_used: Number of non-empty score levels ``K_S``.
        n: Sample size.

    Returns:
        The bias estimate in bits.
    """
    if n <= 0:
        return float("nan")
    return float((n_bins_used - 1) / (2.0 * n * np.log(2.0)))


def achieved_mi_robust(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    bias_correct: bool = True,
) -> float:
    """Achieved MI averaged over bin counts, with optional bias correction.

    See :func:`achieved_mutual_information` for why the bin-count average is
    a sensitivity check rather than a bias correction.  When
    ``bias_correct`` is set, the Miller--Madow bias is estimated at each bin
    count (using the realised number of non-empty cells) and subtracted
    before averaging, so the reported value is a bias-corrected estimate.

    Args:
        y_prob: Predicted probabilities.
        y_true: Ground-truth labels.
        bias_correct: Whether to subtract the Miller--Madow bias.

    Returns:
        Achieved mutual information in bits.
    """
    values = []
    for b in MI_BIN_COUNTS:
        v = achieved_mutual_information(y_prob, y_true, b)
        if bias_correct:
            n_nonempty = _nonempty_cells(y_prob, b, "equal_width")
            v -= miller_madow_bias(n_nonempty, len(y_true))
        values.append(v)
    return float(np.mean(values))


def _nonempty_cells(y_prob: np.ndarray, n_bins: int, scheme: str) -> int:
    """Number of non-empty bins for a given binning of ``y_prob``."""
    if scheme == "equal_mass":
        qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
        cuts = np.unique(np.quantile(y_prob, qs))
        idx = np.digitize(y_prob, cuts)
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        edges[0] = -np.inf
        edges[-1] = np.inf
        idx = np.clip(np.digitize(y_prob, edges[1:-1]), 0, n_bins - 1)
    return int(len(np.unique(idx)))


def mutual_information_spectrum(
    X: pd.DataFrame,
    y: pd.Series,
    n_neighbors: int | None = None,
    discrete_features: Any = None,
) -> pd.Series:
    """Per-feature mutual information with the target, in **bits**, sorted descending.

    Uses the Ross (2014) continuous--discrete nearest-neighbour estimator
    \\cite{ross2014mutual} as implemented by
    :func:`sklearn.feature_selection.mutual_info_classif` (despite its
    frequent misattribution to Kraskov--Stögbauer--Grassberger, sklearn's
    implementation is Ross's estimator, not KSG's).

    .. important::
       ``mutual_info_classif`` returns **natural logarithms (nats)**.  Every
       value is therefore divided by ``ln 2`` here so that the spectrum is on
       the same log base as ``H(Y)``, the plug-in achieved MI, and the Fano
       bound.  Omitting this conversion silently mixes log bases and makes
       any comparison between per-feature and total information meaningless.
       Negative estimates are floored at zero by scikit-learn.

    Args:
        X: Feature matrix.
        y: Binary target.
        n_neighbors: Neighbourhood size; defaults to :data:`MI_N_NEIGHBORS`.
        discrete_features: Passed through to scikit-learn.  Count-valued
            columns should be declared discrete because Ross's estimator
            assumes continuous inputs.

    Returns:
        Series of per-feature MI in bits, sorted descending.
    """
    k = MI_N_NEIGHBORS if n_neighbors is None else n_neighbors
    kwargs: dict[str, Any] = {}
    if discrete_features is not None:
        kwargs["discrete_features"] = discrete_features
    mi_nats = mutual_info_classif(
        X, y,
        random_state=SEED,
        n_neighbors=k,
        **kwargs,
    )
    return pd.Series(mi_nats / np.log(2.0), index=X.columns).sort_values(
        ascending=False
    )


def knn_achieved_mi_1d(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    n_neighbors: int = 3,
) -> float:
    """Binning-free achieved MI of a continuous score, in **bits**.

    Applies the Ross (2014) continuous--discrete nearest-neighbour estimator
    \\cite{ross2014mutual} to the pair (score, label).  Unlike the binned
    plug-in estimator this does not discard the resolution of the score, so
    it is a useful sensitivity check; it is **not** a certified bound and,
    being computed by scikit-learn, its negative estimates are floored at
    zero.  Because it is applied to an isotonic-calibrated score (a step
    function with many tied values), its estimate can be biased upward where
    the score ties; treat it as a cross-check, not a bound.

    ``sklearn.feature_selection.mutual_info_classif`` is called on the score
    treated as a feature against the binary label.  Its return value is in
    nats and is converted to bits.

    Args:
        y_prob: Predicted probabilities.
        y_true: Ground-truth binary labels.
        n_neighbors: Neighbourhood size.

    Returns:
        Achieved mutual information in bits.
    """
    score = np.asarray(y_prob, dtype=float).reshape(-1, 1)
    mi_nats = mutual_info_classif(
        score, np.asarray(y_true), random_state=SEED, n_neighbors=n_neighbors
    )[0]
    return float(mi_nats / np.log(2.0))


def mi_permutation_null(
    X: pd.DataFrame,
    y: pd.Series,
    n_permutations: int = 200,
    n_neighbors: int | None = None,
    seed: int = SEED,
) -> dict[str, float]:
    """Permutation null for the maximum of the per-feature MI spectrum.

    Reporting the top of 87 estimates requires a reference distribution.  This
    permutes the labels ``n_permutations`` times, recomputes the full
    spectrum, and records the maximum each time; the resulting distribution
    gives a family-wise threshold for the observed argmax.

    Args:
        X: Feature matrix.
        y: Binary target.
        n_permutations: Number of label permutations.
        n_neighbors: Neighbourhood size.
        seed: Random seed.

    Returns:
        Dictionary with ``n_permutations``, ``n_features``, the null mean/std
        and 95th percentile of the maximum, the observed maximum, and the
        permutation ``p`` for the max statistic (Phipson--Smyth).  Because
        the statistic is already the maximum over features, this ``p`` is
        family-wise error calibrated by construction (Westfall--Young
        maxT); no Bonferroni factor is applied on top, and with ``B``
        permutations the smallest attainable value is ``1 / (B + 1)``, so the
        result is a lower bound reported as ``p <=``.
    """
    k = MI_N_NEIGHBORS if n_neighbors is None else n_neighbors
    rng = np.random.RandomState(seed)
    y_arr = np.asarray(y)
    n_features = X.shape[1]
    maxima = np.empty(n_permutations)
    for i in range(n_permutations):
        y_perm = rng.permutation(y_arr)
        mi = mutual_info_classif(
            X, y_perm, random_state=seed, n_neighbors=k
        ) / np.log(2.0)
        maxima[i] = float(np.max(mi))
    observed = float(
        np.max(mutual_info_classif(X, y_arr, random_state=seed, n_neighbors=k)
               / np.log(2.0))
    )
    return {
        "n_permutations": int(n_permutations),
        "n_features": int(n_features),
        "null_max_mean": float(maxima.mean()),
        "null_max_std": float(maxima.std(ddof=1)),
        "null_max_95th_pct": float(np.percentile(maxima, 95.0)),
        "null_max_observed": float(maxima.max()),
        "observed_max": observed,
        "observed_over_null95_ratio": float(
            observed / np.percentile(maxima, 95.0)
        ),
        "p_value_maxT": float(
            (1.0 + np.sum(maxima >= observed)) / (1.0 + n_permutations)
        ),
        "p_is_lower_bound": True,
    }


def pca_cumulative_variance(X: pd.DataFrame) -> np.ndarray:
    """Cumulative explained-variance curve of the standardised features."""
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(random_state=SEED)
    pca.fit(X_scaled)
    return np.cumsum(pca.explained_variance_ratio_)


def score_bayes_ceiling_crossfit(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    n_folds: int = 5,
    n_bins: int = 10,
    seed: int = SEED,
) -> dict[str, Any]:
    """Cross-fitted score-Bayes ceiling for the accuracy attainable from a score.

    For a *calibrated* score the ``0.5``-threshold rule **is** the Bayes rule,
    so ``accuracy = E[max(p, 1-p)] = A_SB`` identically: the two quantities
    cannot differ, and any apparent agreement is a tautology rather than
    evidence.  The only quantity that carries information is a
    **cross-fitted** version: the decile posteriors are estimated on one part
    of the held-out set and the induced decision rule is scored on the
    remaining part, so the two never share rows.

    The result is an estimate of how much accuracy a score-*conditioned*
    decision layer could attain out of sample, together with a bootstrap
    confidence interval on the difference from the deployed ``0.5``-threshold
    accuracy.  Note that the ``0.5`` rule uses the score continuously whereas
    the decile rule uses it coarsely, so the decile rule can be slightly
    *worse*; a negative difference therefore does not falsify anything, it
    simply reflects discretisation loss.

    Args:
        y_prob: Predicted probabilities on the held-out set.
        y_true: Ground-truth labels.
        n_folds: Number of cross-fitting folds.
        n_bins: Number of score bins (deciles by default).
        seed: Random seed for the fold assignment.

    Returns:
        Dictionary with the cross-fitted ceiling, the plug-in ceiling, the
        ``0.5``-threshold accuracy, their difference, and a bootstrap CI on
        the difference.
    """
    y_prob = np.asarray(y_prob, dtype=float)
    y_true = np.asarray(y_true)
    n = len(y_true)
    acc_05 = float(accuracy_score(y_true, (y_prob >= 0.5).astype(int)))

    # In-sample (plug-in) decile rule, retained only to show the optimism
    # that cross-fitting removes.
    plug_in = _decile_rule_accuracy(y_prob, y_true, n_bins)

    rng = np.random.RandomState(seed)
    folds = np.arange(n) % n_folds
    rng.shuffle(folds)
    correct = np.zeros(n, dtype=bool)
    for f in range(n_folds):
        fit_mask = folds != f
        apply_mask = folds == f
        if fit_mask.sum() == 0 or apply_mask.sum() == 0:
            continue
        # Estimate the posterior on the fit folds only...
        edges = np.unique(np.quantile(y_prob[fit_mask],
                                       np.linspace(0, 1, n_bins + 1)[1:-1]))
        idx_fit = np.digitize(y_prob[fit_mask], edges)
        idx_apply = np.digitize(y_prob[apply_mask], edges)
        post = np.full(len(edges) + 1, 0.5)
        for b in range(len(edges) + 1):
            m = idx_fit == b
            if m.sum() > 0:
                post[b] = float((y_true[fit_mask][m] == 1).mean())
        # ...and apply the induced rule to the held-out folds only.
        pred = (post[idx_apply] >= 0.5).astype(int)
        correct[apply_mask] = (pred == y_true[apply_mask])

    crossfit = float(correct.mean())

    # Bootstrap CI on the cross-fitted difference.
    boot = np.empty(BOOTSTRAP_N_ITER)
    for i in range(BOOTSTRAP_N_ITER):
        idx = rng.choice(n, size=n, replace=True)
        boot[i] = correct[idx].mean() - float(
            accuracy_score(y_true[idx], (y_prob[idx] >= 0.5).astype(int))
        )
    alpha = 1.0 - CI_LEVEL
    return {
        "crossfit_score_bayes_ceiling": crossfit,
        "plug_in_score_bayes_ceiling": plug_in,
        "accuracy_at_0.5": acc_05,
        "crossfit_minus_accuracy": crossfit - acc_05,
        "crossfit_minus_accuracy_ci_lower": float(np.percentile(boot, 100 * alpha / 2)),
        "crossfit_minus_accuracy_ci_upper": float(np.percentile(boot, 100 * (1 - alpha / 2))),
        "plug_in_optimism": plug_in - crossfit,
        "n_folds": int(n_folds),
        "n_bins": int(n_bins),
        "note": (
            "For a calibrated score the 0.5-threshold rule is the Bayes rule, "
            "so accuracy == A_SB identically; only the cross-fitted estimate "
            "is informative. A negative difference reflects the loss from "
            "using a decile instead of the continuous score, not slack."
        ),
    }


def _decile_rule_accuracy(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    n_bins: int,
) -> float:
    """Accuracy of the score-binned Bayes rule evaluated in sample."""
    order = np.argsort(y_prob)
    bins = np.array_split(order, n_bins)
    correct = 0
    for idx in bins:
        if len(idx) == 0:
            continue
        frac_win = float((y_true[idx] == 1).mean())
        pred = 1 if frac_win >= 0.5 else 0
        correct += int((y_true[idx] == pred).sum())
    return float(correct / len(y_true))


def bootstrap_ci_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    h_y: float | None = None,
) -> dict[str, dict[str, float]]:
    """Stratified bootstrap 95% percentile CIs for accuracy, MI, and the Fano bound.

    Resampling is stratified on the outcome so that the class balance of each
    resample matches the original, which materially reduces the variance of
    the entropy and MI statistics.  The Fano interval is derived by
    transforming the MI percentiles through ``A_max``, whose argument order is
    reversed because ``A_max`` is *decreasing* in the Fano minimum error and
    *increasing* in MI -- passing the percentiles in the wrong order silently
    inverts the interval.

    Args:
        y_true: Ground-truth labels.
        y_prob: Predicted probabilities.
        h_y: Target entropy; recomputed from ``y_true`` if omitted.

    Returns:
        Dictionary of per-metric confidence intervals.
    """
    rng = np.random.RandomState(SEED)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    classes = np.unique(y_true)
    class_idx = [np.flatnonzero(y_true == c) for c in classes]

    accs = np.empty(BOOTSTRAP_N_ITER)
    mis = np.empty(BOOTSTRAP_N_ITER)
    for i in range(BOOTSTRAP_N_ITER):
        idx = np.concatenate([
            rng.choice(ci, size=len(ci), replace=True) for ci in class_idx
        ])
        accs[i] = float(
            accuracy_score(y_true[idx], (y_prob[idx] >= 0.5).astype(int))
        )
        mis[i] = achieved_mi_robust(y_prob[idx], y_true[idx])

    # H(Y) is recomputed per resample: at 40 bins the plug-in entropy carries
    # a bias of order (1-2)/N, which is not negligible at N = 1976.
    h_per_boot = np.array(
        [empirical_entropy(y_true[i]) for i in _strat_boot_indices(
            rng, class_idx, BOOTSTRAP_N_ITER)]
    ) if h_y is None else np.full(BOOTSTRAP_N_ITER, h_y)

    fano_accs = np.array([
        fano_max_accuracy(m, h) for m, h in zip(mis, h_per_boot)
    ])

    alpha = 1.0 - CI_LEVEL
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        "accuracy": {
            "ci_lower": float(np.percentile(accs, lo)),
            "ci_upper": float(np.percentile(accs, hi)),
        },
        "achieved_mi": {
            "ci_lower": float(np.percentile(mis, lo)),
            "ci_upper": float(np.percentile(mis, hi)),
        },
        "fano_max_accuracy": {
            "ci_lower": float(np.percentile(fano_accs, lo)),
            "ci_upper": float(np.percentile(fano_accs, hi)),
        },
    }


def _strat_boot_indices(
    rng: np.random.RandomState,
    class_idx: list[np.ndarray],
    n_boot: int,
) -> list[np.ndarray]:
    """Pre-generate stratified bootstrap index arrays (used for H(Y))."""
    return [
        np.concatenate([
            rng.choice(ci, size=len(ci), replace=True) for ci in class_idx
        ])
        for _ in range(n_boot)
    ]


def pairwise_mcnemar_all(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    alpha: float = 0.05,
    methods: tuple[str, ...] = ("exact", "chi2", "chi2_cc"),
) -> dict[str, Any]:
    """McNemar tests over all model pairs with Holm correction and power.

    Reports the **Holm-adjusted** p-value for every pair (the raw p-value alone
    does not support a claim about which comparisons survive correction), the
    minimum detectable difference at the observed discordant count, and a
    sensitivity analysis across three test variants.

    Args:
        y_true: Ground-truth labels.
        predictions: Mapping of family name to hard predictions.
        alpha: Family-wise significance level.
        methods: McNemar variants to compute for the sensitivity analysis.

    Returns:
        Dictionary with per-pair statistics and Holm decisions.
    """
    names = list(predictions.keys())
    pairs = list(itertools.combinations(names, 2))
    n_pairs = len(pairs)
    n_test = len(y_true)

    raw: dict[tuple[str, str], dict[str, Any]] = {}
    for a, b in pairs:
        raw[(a, b)] = mcnemar_test(y_true, predictions[a], predictions[b],
                                  method="exact")

    # Holm step-down on the sorted raw p-values.  Adjusted p-values are
    # computed with the cumulative-maximum rule so they are monotone.
    ordered = sorted(pairs, key=lambda p: raw[p]["p_value"])
    adjusted: dict[tuple[str, str], float] = {}
    running = 0.0
    for rank, key in enumerate(ordered):
        val = min(1.0, (n_pairs - rank) * raw[key]["p_value"])
        running = max(running, val)
        adjusted[key] = running

    sensitivity: dict[tuple[str, str], dict[str, Any]] = {}
    for m in methods:
        if m == "exact":
            continue
        sensitivity_pairs = {
            f"{a} vs {b}": mcnemar_test(y_true, predictions[a], predictions[b],
                                        method=m)["p_value"]
            for a, b in pairs
        }
        sensitivity[m] = sensitivity_pairs

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
                "significant_after_holm": adjusted[(a, b)] < alpha,
                "mde_accuracy_points": float(
                    100.0 * minimum_detectable_difference(
                        n_test, raw[(a, b)]["n_discordant"], alpha=alpha
                    )
                ),
            }
            for (a, b) in pairs
        },
        "sensitivity_p_values": {k: v for k, v in sensitivity.items()},
        "n_significant_after_holm": int(
            sum(1 for v in adjusted.values() if v < alpha)
        ),
    }


# ----------------------------------------------------------------------
# Learning curve
# ----------------------------------------------------------------------


def learning_curve(
    model_factory: Callable[[], Pipeline],
    X: pd.DataFrame,
    y: np.ndarray,
    train_sizes: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9),
    n_repeats: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Accuracy and achieved MI as a function of training size.

    A fixed stratified 20% test set is held out.  For each training size a
    stratified subsample of the remaining data is drawn ``n_repeats`` times
    with different seeds and the model is refit from scratch.

    The grid stops at 0.9.  A point at fraction 1.0 would be a single
    deterministic fit whose standard deviation is zero by construction; adding
    it would give a spurious zero-width error bar at the right edge and would
    drive the fitted slope.  Between-repeat standard deviations fall by an
    order of magnitude across the grid, so the unweighted fit is dominated by
    the noisy small-sample end; :func:`_wls_slope_test` is therefore reported
    alongside the unweighted fit.

    Both accuracy and achieved mutual information are returned.  The
    information-ceiling argument is a claim about information, so the
    information curve is the one that must flatten for the claim to hold.

    Returns:
        ``(train_sizes, mean_acc, std_acc, mean_mi, std_mi)``.
    """
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=SEED
    )
    train_idx, test_idx = next(splitter.split(X, y))
    y_all = np.asarray(y)
    y_test = y_all[test_idx]
    X_test = X.iloc[test_idx]
    X_pool, y_pool = X.iloc[train_idx], y_all[train_idx]

    means: list[float] = []
    stds: list[float] = []
    mi_means: list[float] = []
    mi_stds: list[float] = []
    for frac in train_sizes:
        repeat_acc: list[float] = []
        repeat_mi: list[float] = []
        for r in range(n_repeats):
            sub_splitter = StratifiedShuffleSplit(
                n_splits=1, train_size=frac, random_state=SEED + r
            )
            sub_train_idx, _ = next(
                sub_splitter.split(np.zeros((len(X_pool), 1)), y_pool)
            )
            X_sub = X_pool.iloc[sub_train_idx]
            y_sub = y_pool[sub_train_idx]

            est = model_factory()
            est.fit(X_sub, y_sub)
            y_prob = predict_proba_safe(est, X_test)
            repeat_acc.append(
                float(accuracy_score(y_test, (y_prob >= 0.5).astype(int)))
            )
            repeat_mi.append(achieved_mi_robust(y_prob, y_test))
        means.append(float(np.mean(repeat_acc)))
        stds.append(float(np.std(repeat_acc, ddof=1)))
        mi_means.append(float(np.mean(repeat_mi)))
        mi_stds.append(float(np.std(repeat_mi, ddof=1)))

    return (
        np.asarray(train_sizes),
        np.asarray(means),
        np.asarray(stds),
        np.asarray(mi_means),
        np.asarray(mi_stds),
    )


def _wls_slope_test(
    x: np.ndarray,
    y: np.ndarray,
    sd: np.ndarray,
) -> dict[str, float]:
    """Weighted least squares of ``y`` on ``1 + x`` with per-point weights.

    Weights are ``1 / sd^2`` from the between-repeat standard deviations, so
    the noisy small-sample end of a learning curve does not dominate the fit.

    Args:
        x: Predictor (log2 of the training fraction).
        y: Response.
        sd: Per-point standard deviation used for the weights.

    Returns:
        Dictionary with the slope, its standard error, two-sided p-value and
        the weighted coefficient of determination.
    """
    from scipy import stats as st

    w = 1.0 / np.clip(sd, 1e-9, None) ** 2
    n = len(x)
    dof = n - 2
    sw = w.sum()
    xbar = float((w * x).sum() / sw)
    ybar = float((w * y).sum() / sw)
    sxx = float((w * (x - xbar) ** 2).sum())
    slope = float((w * (x - xbar) * (y - ybar)).sum() / sxx)
    intercept = ybar - slope * xbar
    resid = y - (intercept + slope * x)
    s2 = float((w * resid ** 2).sum() / dof)
    se = float(np.sqrt(s2 / sxx))
    t = slope / se if se > 0 else 0.0
    p = float(2 * st.t.sf(abs(t), dof)) if se > 0 else 1.0
    sst = float((w * (y - ybar) ** 2).sum())
    r2 = 1.0 - float((w * resid ** 2).sum()) / sst if sst > 0 else float("nan")
    return {
        "slope": slope,
        "se": se,
        "p_value": p,
        "r2": r2,
        "ci_lower": slope - 1.96 * se,
        "ci_upper": slope + 1.96 * se,
    }


def _quadratic_curvature_test(
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    """OLS of ``y`` on ``1 + x + x^2``; returns the quadratic-term test.

    A significant negative quadratic term is direct evidence of
    saturation---the curve is concave in log sample size---which a
    significant positive linear slope alone does not exclude.

    Args:
        x: Predictor (log2 of the training fraction).
        y: Response.

    Returns:
        Dictionary with the quadratic coefficient, its p-value and the
        adjusted R-squared of the quadratic model.
    """
    from scipy import stats as st

    n = len(x)
    Xd = np.column_stack([np.ones(n), x, x ** 2])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ beta
    dof = n - 3
    s2 = float((resid ** 2).sum() / dof)
    cov = s2 * np.linalg.inv(Xd.T @ Xd)
    se_quad = float(np.sqrt(cov[2, 2]))
    t = beta[2] / se_quad if se_quad > 0 else 0.0
    p = float(2 * st.t.sf(abs(t), dof)) if se_quad > 0 else 1.0
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / sst if sst > 0 else float("nan")
    return {
        "quadratic_coefficient": float(beta[2]),
        "quadratic_p_value": p,
        "r2": r2,
        "dof": dof,
    }


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------


def run_ceiling_analysis() -> dict[str, Any]:
    """Run the full ceiling analysis and persist JSON + figures."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 0. Split FIRST, so that every reported quantity is either computed
    #    on the training partition only or explicitly a held-out estimate.
    #    (Previously the MI spectrum and the PCA were fitted on all 9,879
    #    rows, which put test-set information into two reported quantities.)
    # ------------------------------------------------------------------
    X_raw, y_s = load_dataset()
    X = engineer_features(X_raw)
    y = np.asarray(y_s)
    h_y = empirical_entropy(y)

    master = load_master_split()
    X_fit = master["X_fit"]
    X_test = master["X_test"]
    y_test = master["y_test"]
    h_y_test = empirical_entropy(y_test)
    X_train = master["X_train"]
    y_train = master["y_train"]

    print(f"Samples: {X.shape[0]}, features: {X.shape[1]} "
          f"(35 raw + {X.shape[1] - X_raw.shape[1]} engineered)")
    print(f"Target entropy H(Y) = {h_y:.6f} bits "
          f"(full); test = {h_y_test:.6f} bits")
    print(f"Fit {master['n_fit']}  Calibrate {master['n_cal']}  "
          f"Test {master['n_test']}")

    # ------------------------------------------------------------------
    # 0b. Design-matrix audit (rank deficiency, collinearity)
    # ------------------------------------------------------------------
    print("\n--- Design-matrix audit ---")
    audit = features_audit(X_raw, X)
    print(f"  numerical rank {audit['numerical_rank']} of {audit['n_columns']} "
          f"-> {audit['rank_deficiency']} exactly dependent columns")
    print(f"  PCA components: {audit['pca_components_for_variance']}")

    # ------------------------------------------------------------------
    # 1. Mutual-information spectrum (TRAIN ONLY) + permutation null
    # ------------------------------------------------------------------
    print("\n--- Mutual-information spectrum (train partition only, bits) ---")
    mi_spectrum = mutual_information_spectrum(X_train, y_train)
    top_n = mi_spectrum.head(TOP_N_FEATURES)
    print(f"Top-{TOP_N_FEATURES} features by MI (bits):")
    for name, value in top_n.items():
        print(f"  {name:>34s}: {value:.4f}")
    print("  Running permutation null for the argmax ...")
    mi_null = mi_permutation_null(X_train, y_train, n_permutations=PERM_N)
    print(f"  null max: {mi_null['null_max_mean']:.4f} "
          f"+/- {mi_null['null_max_std']:.4f}; observed "
          f"{mi_null['observed_max']:.4f}; ratio "
          f"{mi_null['observed_over_null95_ratio']:.1f}x "
          f"the null 95th pct; p(maxT) <= "
          f"{mi_null['p_value_maxT']:.4g}")

    cumulative_var = pca_cumulative_variance(X_train)
    for pct in (0.5, 0.8, 0.95):
        n_pcs = int(np.searchsorted(cumulative_var, pct) + 1)
        print(f"  PCA (train only): {n_pcs} components explain {pct:.0%} variance")

    # ------------------------------------------------------------------
    # 2. Model zoo, using the single shared fit used by every table
    # ------------------------------------------------------------------
    print("\n--- Model zoo: achieved mutual information (bits) ---")
    scored = fit_and_score(master)
    probs = scored["probs"]
    all_pred = scored["preds"]

    zoo_results: dict[str, Any] = {}
    for name in ALL_FAMILIES:
        y_prob = probs[name]
        y_pred = all_pred[name]
        acc = scored["acc"][name]

        mi_plugin = achieved_mi_robust(y_prob, y_test)
        mi_knn = knn_achieved_mi_1d(y_prob, y_test, n_neighbors=3)
        ci = bootstrap_ci_metrics(y_test, y_prob, h_y=h_y_test)

        fano_min = fano_min_error(mi_plugin, h_y_test)
        efficiency = (1.0 - acc) / fano_min if fano_min > 0 else float("nan")
        # bias is (corrected - uncorrected), so it is NEGATIVE; adding it
        # back recovers the uncorrected plug-in estimate.
        bias = mi_plugin - achieved_mi_robust(y_prob, y_test,
                                              bias_correct=False)
        mi_unbiased = mi_plugin - bias
        amax_plugin = fano_max_accuracy(mi_unbiased, h_y_test)

        zoo_results[name] = {
            "test_accuracy": acc,
            "test_error": 1.0 - acc,
            "roc_auc": scored["auc"][name],
            "achieved_mi_bits": mi_plugin,
            "achieved_mi_bits_uncorrected": mi_unbiased,
            "achieved_mi_bits_knn": mi_knn,
            "miller_madow_bias_bits": bias,
            "achieved_mi_per_bin": {
                str(b): achieved_mutual_information(y_prob, y_test, b)
                for b in MI_BIN_COUNTS
            },
            "fano_min_error": fano_min,
            "fano_max_accuracy": fano_max_accuracy(mi_plugin, h_y_test),
            "fano_max_accuracy_from_plugin_mi": amax_plugin,
            "information_efficiency": efficiency,
            "information_efficiency_from_plugin_mi": (
                (1.0 - acc) / fano_min_error(mi_unbiased, h_y_test)
                if mi_unbiased > 0 else float("nan")
            ),
            "bootstrap_ci": ci,
            "score_bayes": score_bayes_ceiling_crossfit(y_prob, y_test),
            "eta_note": (
                "eta >= 1 holds by Jensen for the plug-in estimator with the "
                "plug-in mutual information, so it is an arithmetic/log-base "
                "check rather than evidence; only the binning-free estimate "
                "can violate it in principle."
            ),
        }
        print(
            f"  {name:>20s}: acc={acc:.4f}  I(p;Y)={mi_plugin:.4f} bits  "
            f"(kNN {mi_knn:.4f})  A_max={fano_max_accuracy(mi_plugin, h_y_test):.4f}"
            f"  eta={efficiency:.3f}"
        )

    # ------------------------------------------------------------------
    # 2b. Pairwise McNemar over the 7 base families, Holm-corrected.
    #     Soft voting is EXCLUDED because it is a deterministic function of
    #     the base scores and would not constitute an independent test.
    # ------------------------------------------------------------------
    print("\n--- Pairwise McNemar (7 base families, exact, Holm) ---")
    mcnemar_summary = pairwise_tests(y_test, all_pred)
    n_pairs = mcnemar_summary["n_pairs"]
    sig_pairs = [
        pair for pair, res in mcnemar_summary["pairs"].items()
        if res["significant_after_holm"]
    ]
    print(f"  {n_pairs} pairs tested; significant after Holm: {len(sig_pairs)}")
    for pair, res in sorted(
        mcnemar_summary["pairs"].items(), key=lambda kv: kv[1]["p_value"]
    )[:6]:
        print(f"    {pair:>40s}  n01={res['n01']:4d} n10={res['n10']:4d}  "
              f"p={res['p_value']:.5f}  p_holm={res['p_value_holm']:.4f}  "
              f"MDE={res['mde_accuracy_points']:.2f}")

    # Holm-aware minimum detectable differences.
    ordered = sorted(mcnemar_summary["pairs"].items(),
                     key=lambda kv: kv[1]["p_value"])
    holm_mdes: list[float] = []
    for rank, (pair, res) in enumerate(ordered):
        a_eff = 0.05 / max(1, n_pairs - rank)
        holm_mdes.append(100.0 * minimum_detectable_difference(
            mcnemar_summary["n_test"], res["n_discordant"], alpha=a_eff
        ))
    mcnemar_summary["mde_holm_aware_range_pts"] = [
        float(min(holm_mdes)), float(max(holm_mdes))
    ]
    print(f"  MDE at alpha=0.05: "
          f"{min(r['mde_accuracy_points'] for r in mcnemar_summary['pairs'].values()):.2f}"
          f"-{max(r['mde_accuracy_points'] for r in mcnemar_summary['pairs'].values()):.2f} pts; "
          f"at Holm thresholds: {min(holm_mdes):.2f}-{max(holm_mdes):.2f} pts")

    # ------------------------------------------------------------------
    # 3. Learning curves for three families (linear, boosted, instance)
    # ------------------------------------------------------------------
    print("\n--- Learning curves (3 families, accuracy and achieved MI) ---")
    lc_families: dict[str, Any] = {}
    factories = {
        "logistic_regression": lambda: build_pipeline(
            make_models()[LOGISTIC_REGRESSION]
        ),
        "lightgbm": lambda: build_pipeline(make_models()[LIGHTGBM]),
        "knn": lambda: build_pipeline(make_nonlinear_models()[KNN]),
    }
    for fam_name, factory in factories.items():
        sizes, mean_acc, std_acc, mean_mi, std_mi = learning_curve(
            factory, X, y
        )
        x = np.log2(sizes)
        acc_unw = _ols_slope_test(x, mean_acc)
        mi_unw = _ols_slope_test(x, mean_mi)
        acc_wls = _wls_slope_test(x, mean_acc, std_acc)
        mi_wls = _wls_slope_test(x, mean_mi, std_mi)
        acc_quad = _quadratic_curvature_test(x, mean_acc)
        mi_quad = _quadratic_curvature_test(x, mean_mi)

        lc_families[fam_name] = {
            "train_sizes": [float(s) for s in sizes],
            "mean_accuracy": [float(m) for m in mean_acc],
            "std_accuracy": [float(s) for s in std_acc],
            "mean_achieved_mi": [float(m) for m in mean_mi],
            "std_achieved_mi": [float(s) for s in std_mi],
            "accuracy_slope_per_log2": acc_unw[0],
            "accuracy_slope_p": acc_unw[2],
            "mi_slope_per_log2": mi_unw[0],
            "mi_slope_p": mi_unw[2],
            "accuracy_slope_wls_per_log2": acc_wls["slope"],
            "accuracy_slope_wls_p": acc_wls["p_value"],
            "accuracy_slope_wls_ci": [acc_wls["ci_lower"], acc_wls["ci_upper"]],
            "mi_slope_wls_per_log2": mi_wls["slope"],
            "mi_slope_wls_p": mi_wls["p_value"],
            "accuracy_curvature": acc_quad,
            "mi_curvature": mi_quad,
        }
        print(f"  {fam_name:>20s}: acc {mean_acc[0]:.4f} -> {mean_acc[-1]:.4f}")
        print(f"      acc slope/log2 {acc_unw[0]:+.5f} (p={acc_unw[2]:.4f}) | "
              f"WLS {acc_wls['slope']:+.5f} (p={acc_wls['p_value']:.4f}) | "
              f"quadratic {acc_quad['quadratic_coefficient']:+.5f} "
              f"(p={acc_quad['quadratic_p_value']:.4f})")
        print(f"      MI  slope/log2 {mi_unw[0]:+.5f} (p={mi_unw[2]:.4f}) | "
              f"WLS {mi_wls['slope']:+.5f} (p={mi_wls['p_value']:.4f}) | "
              f"quadratic {mi_quad['quadratic_coefficient']:+.5f} "
              f"(p={mi_quad['quadratic_p_value']:.4f})")

    # ------------------------------------------------------------------
    # 4. Persist JSON
    # ------------------------------------------------------------------
    summary: dict[str, Any] = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_raw_features": int(X_raw.shape[1]),
        "n_engineered_features": int(X.shape[1] - X_raw.shape[1]),
        "target_entropy_bits": h_y,
        "target_entropy_bits_test": h_y_test,
        "seed": SEED,
        "design_matrix_audit": audit,
        "mi_permutation_null": mi_null,
        "top_features_mi": {k: float(v) for k, v in top_n.items()},
        "pca_variance_points": {
            pct: int(np.searchsorted(cumulative_var, pct) + 1)
            for pct in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
        },
        "model_families": zoo_results,
        "pairwise_mcnemar": mcnemar_summary,
        "learning_curves": lc_families,
        "learning_curve": {
            "train_sizes": lc_families["lightgbm"]["train_sizes"],
            "mean_accuracy": lc_families["lightgbm"]["mean_accuracy"],
            "std_accuracy": lc_families["lightgbm"]["std_accuracy"],
        },
        "n_fit_rows": int(master["n_fit"]),
        "n_calibration_rows": int(master["n_cal"]),
        "n_test_rows": int(master["n_test"]),
    }
    json_path = OUTPUT_DIR / "ceiling_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nCeiling summary saved to {json_path}")

    _plot_mi_spectrum(top_n, mi_spectrum)
    _plot_achieved_mi(zoo_results, h_y_test)
    _plot_fano_curve(zoo_results, h_y_test)
    _plot_learning_curve(
        np.asarray(lc_families["lightgbm"]["train_sizes"]),
        np.asarray(lc_families["lightgbm"]["mean_accuracy"]),
        np.asarray(lc_families["lightgbm"]["std_accuracy"]),
    )
    _plot_pca_variance(cumulative_var)
    print(f"Figures saved to {OUTPUT_DIR}/")

    return summary


def _ols_slope_test(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float, float]:
    """OLS slope, standard error, and two-sided p-value for ``y ~ 1 + x``."""
    n = len(x)
    xbar, ybar = x.mean(), y.mean()
    sxx = float(((x - xbar) ** 2).sum())
    slope = float(((x - xbar) * (y - ybar)).sum() / sxx)
    intercept = ybar - slope * xbar
    resid = y - (intercept + slope * x)
    dof = n - 2
    s2 = float((resid ** 2).sum() / dof)
    se = float(np.sqrt(s2 / sxx))
    if se == 0:
        return slope, 0.0, 1.0
    from scipy import stats as _st
    t = slope / se
    p = float(2 * _st.t.sf(abs(t), dof))
    return slope, se, p


# ----------------------------------------------------------------------
# Plotting helpers
# ----------------------------------------------------------------------


def _plot_mi_spectrum(
    top_n: pd.Series,
    full_spectrum: pd.Series,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(
        np.arange(len(top_n)),
        top_n.values,
        color="#4C72B0",
    )
    ax.set_yticks(np.arange(len(top_n)))
    ax.set_yticklabels(top_n.index)
    ax.invert_yaxis()
    ax.set_xlabel("Mutual information with outcome (bits)")
    ax.set_title("Top-20 feature mutual information (10-min snapshot)")
    # Annotate signal concentration: number of features above small MI
    n_above = int((full_spectrum > 0.02).sum())
    ax.text(
        0.99,
        0.01,
        f"{n_above} of {len(full_spectrum)} features have MI > 0.02 bits",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#333333",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "mi_spectrum.png", dpi=150)
    plt.close(fig)


def _plot_achieved_mi(
    zoo_results: dict[str, Any],
    h_y: float,
) -> None:
    """Two panels with SEPARATE axes.

    Plotting bits and accuracy on a shared axis makes the two bar pairs look
    proportional, which is guaranteed because ``A_max(I) = 1 - h^-1(H-I)`` is
    a monotone deterministic function of ``I``.  The visual therefore carries
    no information beyond the identity and invites the reading that the two
    quantities are independent empirical findings.
    """
    names = list(zoo_results.keys())
    mis = [zoo_results[n]["achieved_mi_bits"] for n in names]
    fano_acc = [zoo_results[n]["fano_max_accuracy"] for n in names]
    accs = [zoo_results[n]["test_accuracy"] for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(names))
    width = 0.38

    ax1.bar(x, mis, width * 1.6, color="#4C72B0", label="Achieved MI $I(\\hat{p};Y)$")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha="right")
    ax1.set_ylabel("Mutual information (bits)")
    ax1.set_title("Achieved mutual information per family")
    for xi, v in zip(x, mis):
        ax1.text(xi, v + 0.004, f"{v:.3f}", ha="center", fontsize=7)

    ax2.bar(x - width / 2, accs, width, color="#4C72B0", label="Test accuracy")
    ax2.bar(x + width / 2, fano_acc, width, color="#DD8452",
            label="Fano-implied $A_{\\max}$")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=30, ha="right")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0.6, 0.85)
    ax2.set_title("Accuracy vs the Fano bound for the extracted information")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.text(
        0.02, 0.96,
        "$A_{\\max}(I)$ is a deterministic function of $I$;\n"
        "the two bars are not independent measurements.",
        transform=ax2.transAxes, va="top", fontsize=7, color="#444444",
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "achieved_mi.png", dpi=150)
    plt.close(fig)


def _plot_fano_curve(
    zoo_results: dict[str, Any],
    h_y: float,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    mi_range = np.linspace(0.0, 0.4, 200)
    acc_ceiling = np.array([fano_max_accuracy(m, h_y) for m in mi_range])
    ax.plot(mi_range, acc_ceiling, color="#4C72B0", linewidth=2,
            label="Fano accuracy ceiling")

    for name, res in zoo_results.items():
        ax.scatter(
            res["achieved_mi_bits"],
            res["test_accuracy"],
            s=45,
            zorder=3,
        )
        ax.annotate(
            name.replace("_", " "),
            (res["achieved_mi_bits"], res["test_accuracy"]),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )

    ax.axhline(0.5, color="#C44E52", linestyle=":", label="Chance (0.5)")
    ax.set_xlabel("Achieved mutual information I(p;Y) (bits)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Model accuracy vs the Fano-implied information ceiling")
    ax.set_ylim(0.5, 1.0)
    ax.set_xlim(0.0, 0.4)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fano_curve.png", dpi=150)
    plt.close(fig)


def _plot_learning_curve(
    sizes: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(
        sizes,
        mean,
        yerr=std,
        fmt="o-",
        color="#4C72B0",
        capsize=4,
        linewidth=2,
    )
    ax.set_xlabel("Training-set fraction")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Learning curve: accuracy saturates with training data")
    ax.set_ylim(0.6, 0.78)
    ax.axhline(mean[-1], color="#C44E52", linestyle="--",
               label=f"Full-data accuracy ({mean[-1]:.3f})")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "learning_curve.png", dpi=150)
    plt.close(fig)


def _plot_pca_variance(cumulative_var: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        np.arange(1, len(cumulative_var) + 1),
        cumulative_var,
        color="#4C72B0",
        linewidth=2,
    )
    ax.axhline(0.8, color="#C44E52", linestyle=":", label="80% variance")
    ax.axhline(0.95, color="#DD8452", linestyle=":", label="95% variance")
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("Feature redundancy: variance concentrates in few components")
    ax.set_xlim(1, 40)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pca_variance.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run_ceiling_analysis()
