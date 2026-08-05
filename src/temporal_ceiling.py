"""Temporal information-ceiling analysis on an independent per-frame dataset.

The main ceiling analysis (``ceiling_analysis.py``) established that eight
model families converge near 72% accuracy on the 10-minute snapshot and
that every family sits at the Fano-implied information frontier.  This
module tests the *causal* claim behind that result on an independent
dataset: if the plateau is an information limit rather than a model
limitation, then observing the *same* games at later points in time must
raise the information ceiling.

The dataset used here (bobbyscience soloQ ranked games) records one row
per (game, frame) where ``frame`` is the game minute, giving the same
match a snapshot every two minutes from minute 10 onward.  For each frame
we run the same ceiling protocol — a small model zoo, achieved mutual
information, and the Fano bound — and verify two things:

1. **Cross-dataset generalisation**: at frame 10 the ceiling matches the
   main dataset (~72%), so the plateau is not an artifact of one corpus.
2. **Temporal rise**: accuracy, achieved MI, and the Fano ceiling rise
   with game time, confirming that the early-game limit is informational,
   not a model-capacity effect.

Because every later frame contains only games still in progress (a
duration-conditioned survivor subset), the raw per-frame curves mix
information accrual with cohort recomposition.  We therefore also run a
**fixed-cohort analysis**: games with final duration >= a threshold are
predicted at every minute with the *same* held-out test split (drawn once
on the cohort's game set), so cohort composition and labels are held
constant across frames and only the observed information changes.  This
isolates information accrual from sample-composition effects and is the
decisive control for the temporal claim (peer-review revision plan, C1).

All modelling is deterministic (``SEED = 42``, ``n_jobs=1``), all
quantities are computed strictly from the information available at the
given frame (no future-frame leakage within a frame's feature set), and
``paper/`` remains untracked for double-blind review.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.ceiling_analysis import (
    achieved_mi_robust,
    empirical_entropy,
    fano_min_error,
)
from src.models import build_pipeline
from src.models_v2 import predict_proba_safe

matplotlib.use("Agg")

SEED: int = 42
TEST_SIZE: float = 0.2
BOOTSTRAP_N_ITER: int = 2000
CI_LEVEL: float = 0.95

DATA_PATH: Path = Path("data/lol_ranked_games_temporal.csv")
OUTPUT_DIR: Path = Path("plots_temporal")

# Frames (game minutes) used for the temporal sweep.  All retain at least
# ~10k games; earlier frames are dropped since minute 10 is the earliest
# snapshot and the whole point is to measure the ceiling as information
# accumulates.
FRAMES: tuple[int, ...] = (10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30)

# Final-duration thresholds (ms) for the fixed-cohort analyses.
DURATION_MS_PER_MIN: int = 60_000
COHORT_MINUTES: tuple[int, ...] = (32, 36)

# Columns that do not carry per-frame decision information.
META_COLUMNS: tuple[str, ...] = ("gameId", "gameDuration", "frame", "hasWon")


def load_temporal_data() -> pd.DataFrame:
    """Load the per-frame dataset."""
    df = pd.read_csv(DATA_PATH)
    return df


def build_frame_zoo() -> dict[str, Pipeline]:
    """Small, representative model zoo used at every frame.

    Kept deliberately small (linear + three tree families) so that the
    sweep over many frames remains cheap, while still spanning the
    capacity range used in the full analysis.  All estimators are fully
    deterministic (``n_jobs=1``, fixed seeds) so that repeated runs
    produce byte-identical outputs.
    """
    zoo: dict[str, Pipeline] = {
        "logistic_regression": build_pipeline(
            LogisticRegression(max_iter=5000, random_state=SEED)
        ),
        "random_forest": build_pipeline(
            RandomForestClassifier(n_estimators=500, n_jobs=1,
                                   random_state=SEED),
            scale=False,
        ),
        "xgboost": build_pipeline(
            XGBClassifier(
                n_jobs=1,
                random_state=SEED,
                eval_metric="logloss",
                tree_method="hist",
                verbosity=0,
            ),
            scale=False,
        ),
        "lightgbm": build_pipeline(
            LGBMClassifier(
                n_jobs=1,
                deterministic=True,
                random_state=SEED,
                verbose=-1,
                importance_type="gain",
            ),
            scale=False,
        ),
    }
    return zoo


def bootstrap_ci_accuracy(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Bootstrap (95%) CI for accuracy, achieved MI, and Fano ceiling."""
    rng = np.random.RandomState(SEED)
    n = len(y_true)
    accs = np.empty(BOOTSTRAP_N_ITER)
    mis = np.empty(BOOTSTRAP_N_ITER)
    for i in range(BOOTSTRAP_N_ITER):
        idx = rng.choice(n, size=n, replace=True)
        accs[i] = float(accuracy_score(y_true[idx], (y_prob[idx] >= 0.5).astype(int)))
        mis[i] = achieved_mi_robust(y_prob[idx], y_true[idx])
    alpha = 1.0 - CI_LEVEL
    h_y = empirical_entropy(y_true)
    return {
        "accuracy": {
            "ci_lower": float(np.percentile(accs, 100 * alpha / 2)),
            "ci_upper": float(np.percentile(accs, 100 * (1 - alpha / 2))),
        },
        "achieved_mi": {
            "ci_lower": float(np.percentile(mis, 100 * alpha / 2)),
            "ci_upper": float(np.percentile(mis, 100 * (1 - alpha / 2))),
        },
        "fano_max_accuracy": {
            "ci_lower": float(1.0 - fano_min_error(
                np.percentile(mis, 100 * (1 - alpha / 2)), h_y)),
            "ci_upper": float(1.0 - fano_min_error(
                np.percentile(mis, 100 * alpha / 2), h_y)),
        },
    }


def run_frame_analysis(
    frame: int,
    df: pd.DataFrame,
) -> dict[str, Any]:
    """Run the ceiling protocol for a single game minute."""
    frame_df = df[df["frame"] == frame].copy()
    feature_cols = [c for c in frame_df.columns if c not in META_COLUMNS]
    X = frame_df[feature_cols]
    y = frame_df["hasWon"].astype(int).values
    n = len(y)
    h_y = empirical_entropy(y)

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=SEED
    )
    train_idx, test_idx = next(splitter.split(X, y))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    zoo = build_frame_zoo()
    results: dict[str, Any] = {"n_samples": n, "win_rate": float(y.mean()),
                               "target_entropy_bits": h_y}
    all_prob: list[np.ndarray] = []
    for name, est in zoo.items():
        est.fit(X_train, y_train)
        y_prob = predict_proba_safe(est, X_test)
        y_pred = (y_prob >= 0.5).astype(int)
        acc = float(accuracy_score(y_test, y_pred))
        mi = achieved_mi_robust(y_prob, y_test)
        fano_min = fano_min_error(mi, h_y)
        results[name] = {
            "test_accuracy": acc,
            "achieved_mi_bits": mi,
            "fano_max_accuracy": 1.0 - fano_min,
            "information_efficiency": (1.0 - acc) / fano_min
            if fano_min > 0 else float("nan"),
        }
        all_prob.append(y_prob)

    voting_prob = np.mean(np.column_stack(all_prob), axis=1)
    voting_pred = (voting_prob >= 0.5).astype(int)
    voting_acc = float(accuracy_score(y_test, voting_pred))
    voting_mi = achieved_mi_robust(voting_prob, y_test)
    fano_voting = fano_min_error(voting_mi, h_y)
    results["soft_voting"] = {
        "test_accuracy": voting_acc,
        "achieved_mi_bits": voting_mi,
        "fano_max_accuracy": 1.0 - fano_voting,
        "information_efficiency": (1.0 - voting_acc) / fano_voting,
    }
    results["soft_voting"]["bootstrap_ci"] = bootstrap_ci_accuracy(
        y_test, voting_prob
    )

    print(
        f"  frame={frame:>2d} min | n={n:>6d} | win_rate={y.mean():.3f} | "
        f"H(Y)={h_y:.4f} | soft-vote acc={voting_acc:.4f} | "
        f"I(p;Y)={voting_mi:.4f} | Fano max acc={1.0 - fano_voting:.4f}"
    )
    return results


# ----------------------------------------------------------------------
# Fixed-cohort analysis (survivor-bias control)
# ----------------------------------------------------------------------


def run_fixed_cohort_frame(
    frame: int,
    df: pd.DataFrame,
    min_duration_ms: int,
    test_games: np.ndarray,
) -> dict[str, Any]:
    """Ceiling protocol for one frame restricted to a duration cohort.

    The same held-out game set (``test_games``) is used at every frame so
    that cohort composition and labels are constant across frames; only
    the observed information changes.  The 80/20 split is drawn once on
    the cohort's game set and reused at every frame.
    """
    cohort = df[
        (df["frame"] == frame) & (df["gameDuration"] >= min_duration_ms)
    ].copy()
    feature_cols = [c for c in cohort.columns if c not in META_COLUMNS]
    X = cohort[feature_cols]
    y = cohort["hasWon"].astype(int).values
    n = len(y)
    h_y = empirical_entropy(y)

    is_test = cohort["gameId"].isin(test_games).values
    train_idx = np.where(~is_test)[0]
    test_idx = np.where(is_test)[0]
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            f"Frame {frame}: empty train ({len(train_idx)}) or test "
            f"({len(test_idx)}) split for cohort duration >= {min_duration_ms} ms"
        )
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    zoo = build_frame_zoo()
    results: dict[str, Any] = {
        "n_samples": n, "win_rate": float(y.mean()),
        "target_entropy_bits": h_y,
    }
    all_prob: list[np.ndarray] = []
    for name, est in zoo.items():
        est.fit(X_train, y_train)
        y_prob = predict_proba_safe(est, X_test)
        y_pred = (y_prob >= 0.5).astype(int)
        acc = float(accuracy_score(y_test, y_pred))
        mi = achieved_mi_robust(y_prob, y_test)
        fano_min = fano_min_error(mi, h_y)
        results[name] = {
            "test_accuracy": acc,
            "achieved_mi_bits": mi,
            "fano_max_accuracy": 1.0 - fano_min,
            "information_efficiency": (1.0 - acc) / fano_min
            if fano_min > 0 else float("nan"),
        }
        all_prob.append(y_prob)

    voting_prob = np.mean(np.column_stack(all_prob), axis=1)
    voting_pred = (voting_prob >= 0.5).astype(int)
    voting_acc = float(accuracy_score(y_test, voting_pred))
    voting_mi = achieved_mi_robust(voting_prob, y_test)
    fano_voting = fano_min_error(voting_mi, h_y)
    results["soft_voting"] = {
        "test_accuracy": voting_acc,
        "achieved_mi_bits": voting_mi,
        "fano_max_accuracy": 1.0 - fano_voting,
        "information_efficiency": (1.0 - voting_acc) / fano_voting,
    }
    results["soft_voting"]["bootstrap_ci"] = bootstrap_ci_accuracy(
        y_test, voting_prob
    )

    print(
        f"  [cohort>={min_duration_ms // DURATION_MS_PER_MIN:>2d} min] "
        f"frame={frame:>2d} | n={n:>6d} | win={y.mean():.3f} | "
        f"acc={voting_acc:.4f} | I={voting_mi:.4f} | Fano={1.0 - fano_voting:.4f}"
    )
    return results


def run_fixed_cohort_analysis(
    df: pd.DataFrame,
    min_duration_min: int,
) -> dict[str, Any]:
    """Fixed-cohort sweep: same games, same labels, every frame.

    The 20% test split is drawn once from the cohort's game set (seed 42)
    and reused verbatim at every frame, isolating information accrual
    from cohort recomposition.
    """
    min_duration_ms = min_duration_min * DURATION_MS_PER_MIN
    cohort_games = df[df["gameDuration"] >= min_duration_ms]["gameId"].unique()
    if len(cohort_games) < 200:
        print(f"  Cohort >= {min_duration_min} min: only {len(cohort_games)} games, skipping")
        return {}

    # Fixed 80/20 split on the cohort's GAME set (not rows).
    rng = np.random.RandomState(SEED)
    shuffled = rng.permutation(cohort_games)
    n_test = int(round(len(shuffled) * TEST_SIZE))
    test_games = shuffled[:n_test]

    # Every game in the cohort has final duration >= min_duration_min > 30,
    # so ALL cohort games appear in every frame 10..30.  Reuse the same
    # test game set at every frame; only the observed features change.
    per_frame: dict[str, Any] = {}
    for frame in FRAMES:
        per_frame[str(frame)] = run_fixed_cohort_frame(
            frame, df, min_duration_ms, test_games
        )
    return per_frame


def run_lead_padded_analysis(df: pd.DataFrame) -> dict[str, Any]:
    """Lead-padded control: at frame t keep only games with duration >= t+4.

    Removes near-terminal games (which are trivially predictable) at
    every frame, isolating the rise from terminal-triviality effects.
    Per-frame splits are used here because the cohort itself changes with
    the lead.
    """
    per_frame: dict[str, Any] = {}
    for frame in FRAMES:
        min_duration_ms = (frame + 4) * DURATION_MS_PER_MIN
        lead_df = df[
            (df["frame"] == frame) & (df["gameDuration"] >= min_duration_ms)
        ]
        n = len(lead_df)
        if n < 500:
            print(f"  [lead t+4] frame={frame:>2d}: n={n} too small, skipping")
            continue
        res = run_frame_analysis(frame, lead_df)
        res["n_samples"] = n
        per_frame[str(frame)] = res
    return per_frame


def run_temporal_ceiling() -> dict[str, Any]:
    """Run the full temporal sweep and persist JSON + figures."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_temporal_data()
    print(f"Loaded temporal dataset: {len(df)} rows, "
          f"{df['gameId'].nunique()} games")

    per_frame: dict[str, Any] = {}
    for frame in FRAMES:
        per_frame[str(frame)] = run_frame_analysis(frame, df)

    # ------------------------------------------------------------------
    # Survivor-bias controls
    # ------------------------------------------------------------------
    print("\n--- Fixed-cohort analyses (same games, same labels, every frame) ---")
    fixed_cohort: dict[str, Any] = {}
    for min_min in COHORT_MINUTES:
        print(f"\nCohort: games with duration >= {min_min} min")
        fixed_cohort[str(min_min)] = run_fixed_cohort_analysis(df, min_min)

    print("\n--- Lead-padded control (frame t keeps games with duration >= t+4) ---")
    lead_padded = run_lead_padded_analysis(df)

    # ------------------------------------------------------------------
    # Duration-stratified minute-10 table (composition effect)
    # ------------------------------------------------------------------
    frame10 = df[df["frame"] == 10]
    duration_bins = [
        ("<24 min", frame10["gameDuration"] < 24 * DURATION_MS_PER_MIN),
        ("24-29 min",
         (frame10["gameDuration"] >= 24 * DURATION_MS_PER_MIN)
         & (frame10["gameDuration"] < 30 * DURATION_MS_PER_MIN)),
        (">=30 min", frame10["gameDuration"] >= 30 * DURATION_MS_PER_MIN),
    ]
    stratified = {}
    for label, mask in duration_bins:
        sub = frame10[mask]
        if len(sub) < 100:
            continue
        res = run_frame_analysis(10, sub)
        stratified[label] = {
            "n_samples": int(len(sub)),
            "win_rate": res["win_rate"],
            "voting_accuracy": res["soft_voting"]["test_accuracy"],
            "voting_achieved_mi": res["soft_voting"]["achieved_mi_bits"],
            "voting_fano_max_accuracy": res["soft_voting"]["fano_max_accuracy"],
        }
        print(f"  [{label}] n={len(sub):>6d} acc={stratified[label]['voting_accuracy']:.4f}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    rows = []
    for f, res in per_frame.items():
        sv = res["soft_voting"]
        ci = sv.get("bootstrap_ci", {})
        rows.append(
            {
                "frame_minutes": int(f),
                "n_samples": res["n_samples"],
                "win_rate": res["win_rate"],
                "target_entropy_bits": res["target_entropy_bits"],
                "voting_accuracy": sv["test_accuracy"],
                "voting_achieved_mi": sv["achieved_mi_bits"],
                "voting_fano_max_accuracy": sv["fano_max_accuracy"],
                "voting_information_efficiency": sv["information_efficiency"],
                "acc_ci_lower": ci.get("accuracy", {}).get("ci_lower"),
                "acc_ci_upper": ci.get("accuracy", {}).get("ci_upper"),
                "mi_ci_lower": ci.get("achieved_mi", {}).get("ci_lower"),
                "mi_ci_upper": ci.get("achieved_mi", {}).get("ci_upper"),
            }
        )
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(OUTPUT_DIR / "temporal_summary.csv", index=False)
    summary_df.to_json(OUTPUT_DIR / "temporal_summary.json", orient="records",
                       indent=2)
    print("\nTemporal summary table:")
    print(summary_df.to_string(index=False))

    # Persist fixed-cohort + stratified summaries.
    def _cohort_rows(cohort: dict[str, Any], duration_key: int | None = None) -> list[dict[str, Any]]:
        rows_out = []
        for min_min, per in cohort.items():
            for f, res in per.items():
                sv = res["soft_voting"]
                rows_out.append(
                    {
                        "min_duration_min": (int(min_min) if duration_key is None
                                             else duration_key),
                        "frame_minutes": int(f),
                        "n_samples": res["n_samples"],
                        "win_rate": res["win_rate"],
                        "target_entropy_bits": res["target_entropy_bits"],
                        "voting_accuracy": sv["test_accuracy"],
                        "voting_achieved_mi": sv["achieved_mi_bits"],
                        "voting_fano_max_accuracy": sv["fano_max_accuracy"],
                        "voting_information_efficiency": sv["information_efficiency"],
                    }
                )
        return rows_out

    cohort_rows = _cohort_rows(fixed_cohort)
    lead_rows = _cohort_rows({"lead": lead_padded}, duration_key=4)
    pd.DataFrame(cohort_rows).to_csv(OUTPUT_DIR / "temporal_fixed_cohort.csv",
                                     index=False)
    pd.DataFrame(lead_rows).to_csv(OUTPUT_DIR / "temporal_lead_padded.csv",
                                   index=False)
    pd.DataFrame(cohort_rows).to_json(OUTPUT_DIR / "temporal_fixed_cohort.json",
                                      orient="records", indent=2)
    pd.DataFrame(lead_rows).to_json(OUTPUT_DIR / "temporal_lead_padded.json",
                                    orient="records", indent=2)

    stratified_df = pd.DataFrame(stratified).T
    stratified_df.to_csv(OUTPUT_DIR / "temporal_duration_stratified.csv")
    stratified_df.to_json(OUTPUT_DIR / "temporal_duration_stratified.json",
                          orient="index", indent=2)

    _plot_temporal(summary_df, cohort_rows, lead_rows)
    print(f"\nFigures + summary saved to {OUTPUT_DIR}/")
    return per_frame


def plot_temporal_only() -> None:
    """Regenerate the figure from persisted CSVs (no model fitting)."""
    summary_df = pd.read_csv(OUTPUT_DIR / "temporal_summary.csv")
    cohort_rows = pd.read_csv(OUTPUT_DIR / "temporal_fixed_cohort.csv").to_dict(
        "records"
    )
    lead_rows = pd.read_csv(OUTPUT_DIR / "temporal_lead_padded.csv").to_dict(
        "records"
    )
    _plot_temporal(summary_df, cohort_rows, lead_rows)
    print(f"Figure saved to {OUTPUT_DIR}/temporal_ceiling.png")


def _plot_temporal(
    summary_df: pd.DataFrame,
    cohort_rows: list[dict[str, Any]],
    lead_rows: list[dict[str, Any]],
) -> None:
    """Two-panel figure: accuracy and achieved-MI vs game minute."""
    frames = summary_df["frame_minutes"].values
    acc = summary_df["voting_accuracy"].values
    fano = summary_df["voting_fano_max_accuracy"].values
    mi = summary_df["voting_achieved_mi"].values
    h_y = summary_df["target_entropy_bits"].values

    acc_lo = summary_df["acc_ci_lower"].values
    acc_hi = summary_df["acc_ci_upper"].values
    mi_lo = summary_df["mi_ci_lower"].values
    mi_hi = summary_df["mi_ci_upper"].values

    cohort_df = pd.DataFrame(cohort_rows)
    lead_df = pd.DataFrame(lead_rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.errorbar(frames, acc, yerr=[acc - acc_lo, acc_hi - acc],
                fmt="o-", color="#4C72B0", linewidth=2, capsize=3,
                label="Soft-voting accuracy (all games)")
    ax.plot(frames, fano, "s--", color="#DD8452", linewidth=2,
            label="Fano-implied ceiling")
    if len(cohort_df):
        for min_dur, color in ((32, "#55A868"), (36, "#8172B2")):
            cf = cohort_df[cohort_df["min_duration_min"] == min_dur]
            if len(cf):
                ax.plot(cf["frame_minutes"], cf["voting_accuracy"], "^",
                        color=color, linewidth=2, markersize=6,
                        label=f"Fixed cohort (duration >= {min_dur} min)")
    if len(lead_df):
        ax.plot(lead_df["frame_minutes"], lead_df["voting_accuracy"], "d-",
                color="#C44E52", linewidth=1.6, markersize=5,
                label="Lead-padded (duration >= t+4)")
    ax.axhline(0.5, color="#777777", linestyle=":", label="Chance")
    ax.set_xlabel("Game minute (frame)")
    ax.set_ylabel("Accuracy")
    ax.set_title("The information ceiling rises with game time")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0.45, 1.0)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.errorbar(frames, mi, yerr=[mi - mi_lo, mi_hi - mi],
                fmt="o-", color="#4C72B0", linewidth=2, capsize=3,
                label="Achieved MI I(p;Y) (all games)")
    if len(cohort_df):
        for min_dur, color in ((32, "#55A868"), (36, "#8172B2")):
            cf = cohort_df[cohort_df["min_duration_min"] == min_dur]
            if len(cf):
                ax.plot(cf["frame_minutes"], cf["voting_achieved_mi"], "^",
                        color=color, linewidth=2, markersize=6,
                        label=f"Fixed cohort (duration >= {min_dur} min)")
    if len(lead_df):
        ax.plot(lead_df["frame_minutes"], lead_df["voting_achieved_mi"], "d-",
                color="#C44E52", linewidth=1.6, markersize=5,
                label="Lead-padded (duration >= t+4)")
    ax.plot(frames, h_y, "s--", color="#DD8452", linewidth=2,
            label="Target entropy H(Y)")
    ax.set_xlabel("Game minute (frame)")
    ax.set_ylabel("Bits")
    ax.set_title("Achieved information accumulates over time")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "temporal_ceiling.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run_temporal_ceiling()
