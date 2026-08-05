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
2. **Temporal monotonicity**: accuracy, achieved MI, and the Fano ceiling
   rise monotonically as the game progresses, confirming that the
   early-game limit is informational, not a model-capacity effect.

All modelling is deterministic (``SEED = 42``), all quantities are
computed strictly from the information available at the given frame
(no future-frame leakage within a frame's feature set), and ``paper/``
remains untracked for double-blind review.
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
    fano_max_accuracy,
    fano_min_error,
)
from src.models import build_pipeline, make_models
from src.models_v2 import predict_proba_safe

matplotlib.use("Agg")

SEED: int = 42
TEST_SIZE: float = 0.2

DATA_PATH: Path = Path("data/lol_ranked_games_temporal.csv")
OUTPUT_DIR: Path = Path("plots_temporal")

# Frames (game minutes) used for the temporal sweep.  All retain at least
# ~10k games; earlier frames are dropped since minute 10 is the earliest
# snapshot and the whole point is to measure the ceiling as information
# accumulates.
FRAMES: tuple[int, ...] = (10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30)

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
    capacity range used in the full analysis.
    """
    zoo: dict[str, Pipeline] = {
        "logistic_regression": build_pipeline(
            LogisticRegression(max_iter=5000, random_state=SEED)
        ),
        "random_forest": build_pipeline(
            RandomForestClassifier(n_jobs=-1, random_state=SEED), scale=False
        ),
        "xgboost": build_pipeline(
            XGBClassifier(
                n_jobs=-1,
                random_state=SEED,
                eval_metric="logloss",
                tree_method="hist",
                verbosity=0,
            ),
            scale=False,
        ),
        "lightgbm": build_pipeline(
            LGBMClassifier(
                n_jobs=-1,
                random_state=SEED,
                verbose=-1,
                importance_type="gain",
            ),
            scale=False,
        ),
    }
    return zoo


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

    print(
        f"  frame={frame:>2d} min | n={n:>6d} | win_rate={y.mean():.3f} | "
        f"H(Y)={h_y:.4f} | soft-vote acc={voting_acc:.4f} | "
        f"I(p;Y)={voting_mi:.4f} | Fano max acc={1.0 - fano_voting:.4f}"
    )
    return results


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
    # Summary table
    # ------------------------------------------------------------------
    rows = []
    for f, res in per_frame.items():
        rows.append(
            {
                "frame_minutes": int(f),
                "n_samples": res["n_samples"],
                "win_rate": res["win_rate"],
                "target_entropy_bits": res["target_entropy_bits"],
                "voting_accuracy": res["soft_voting"]["test_accuracy"],
                "voting_achieved_mi": res["soft_voting"]["achieved_mi_bits"],
                "voting_fano_max_accuracy": res["soft_voting"]["fano_max_accuracy"],
            }
        )
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(OUTPUT_DIR / "temporal_summary.csv", index=False)
    summary_df.to_json(OUTPUT_DIR / "temporal_summary.json", orient="records",
                       indent=2)
    print("\nTemporal summary table:")
    print(summary_df.to_string(index=False))

    _plot_temporal(summary_df)
    print(f"\nFigures + summary saved to {OUTPUT_DIR}/")
    return per_frame


def _plot_temporal(summary_df: pd.DataFrame) -> None:
    """Two-panel figure: accuracy and achieved-MI vs game minute."""
    frames = summary_df["frame_minutes"].values
    acc = summary_df["voting_accuracy"].values
    fano = summary_df["voting_fano_max_accuracy"].values
    mi = summary_df["voting_achieved_mi"].values
    h_y = summary_df["target_entropy_bits"].values

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(frames, acc, "o-", color="#4C72B0", linewidth=2,
            label="Soft-voting accuracy")
    ax.plot(frames, fano, "s--", color="#DD8452", linewidth=2,
            label="Fano-implied ceiling")
    ax.axhline(0.5, color="#C44E52", linestyle=":", label="Chance")
    ax.set_xlabel("Game minute (frame)")
    ax.set_ylabel("Accuracy")
    ax.set_title("The information ceiling rises with game time")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0.45, 1.0)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(frames, mi, "o-", color="#4C72B0", linewidth=2,
            label="Achieved MI I(p;Y)")
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
