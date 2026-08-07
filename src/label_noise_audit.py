"""Label-noise robustness audit (peer-review revision plan, experiment C4).

The devil's-advocate review showed that a ~15% symmetric label-flip noise
model reproduces every headline number of the ceiling analysis (accuracy
~0.72, achieved MI ~0.24 bits, saturating learning curves).  This module
quantifies how much label noise the reported signature tolerates:

* On the primary dataset's fixed test split, we flip a fraction ``rho``
  of test labels, refit three representative families (LR, LGBM, soft
  voting) on the *clean* training split, and recompute accuracy and
  achieved MI under the corrupted labels.
* We report the (rho, accuracy) and (rho, MI) curves plus the noise level
  at which the clean-data signature (acc ~0.72, MI ~0.24) is reproduced.
* On the temporal corpus we additionally report the minute-10 plateau for
  games lasting >= 25 and >= 30 minutes (short-game contamination check).

All randomness is seeded (SEED = 42) for reproducibility.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from lightgbm import LGBMClassifier

from src.ceiling_analysis import (
    achieved_mi_robust,
    empirical_entropy,
    fano_min_error,
)
from src.data_loader import load_engineered_dataset
from src.models import build_pipeline
from src.models_v2 import predict_proba_safe
from src.temporal_ceiling import DURATION_MS_PER_MIN, load_temporal_data

matplotlib.use("Agg")

SEED: int = 42
TEST_SIZE: float = 0.2
NOISE_LEVELS: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.20)
N_REPEATS: int = 3

OUTPUT_DIR: Path = Path("plots_ceiling")


def _flip_labels(y: np.ndarray, rho: float, seed: int) -> np.ndarray:
    """Symmetrically flip a fraction rho of the labels (seeded)."""
    rng = np.random.RandomState(seed)
    flip = rng.random(len(y)) < rho
    y_noisy = y.copy()
    y_noisy[flip] = 1 - y_noisy[flip]
    return y_noisy


def run_label_noise_audit() -> dict[str, Any]:
    """Run the label-noise sweep and persist JSON + figure."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_engineered_dataset()
    y_arr = y.values
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=SEED
    )
    train_idx, test_idx = next(splitter.split(X, y_arr))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y_arr[train_idx], y_arr[test_idx]

    # Fixed models: LR (deterministic), LGBM (deterministic), soft voting.
    lr = build_pipeline(LogisticRegression(max_iter=5000, random_state=SEED))
    lgbm = build_pipeline(
        LGBMClassifier(
            n_jobs=1, deterministic=True, random_state=SEED, verbose=-1
        ),
        scale=False,
    )

    results: dict[str, Any] = {
        "clean": {},
        "noise_sweep": {},
        "duration_restricted": {},
    }
    for name, est in (("logistic_regression", lr), ("lightgbm", lgbm)):
        est.fit(X_train, y_train)
        y_prob_clean = predict_proba_safe(est, X_test)
        acc_clean = float(accuracy_score(y_test, (y_prob_clean >= 0.5).astype(int)))
        mi_clean = achieved_mi_robust(y_prob_clean, y_test)
        results["clean"][name] = {
            "test_accuracy": acc_clean,
            "achieved_mi_bits": mi_clean,
        }

    # Soft voting over both families.
    lr.fit(X_train, y_train)
    lgbm.fit(X_train, y_train)
    voting_prob = np.mean(
        np.column_stack([
            predict_proba_safe(lr, X_test),
            predict_proba_safe(lgbm, X_test),
        ]),
        axis=1,
    )
    results["clean"]["soft_voting"] = {
        "test_accuracy": float(accuracy_score(
            y_test, (voting_prob >= 0.5).astype(int))),
        "achieved_mi_bits": achieved_mi_robust(voting_prob, y_test),
    }

    # Noise sweep: corrupt TEST labels only (training stays clean), so the
    # signature collapse is attributable to label noise alone.
    for rho in NOISE_LEVELS:
        row: dict[str, Any] = {"rho": rho}
        for name, est in (
            ("logistic_regression", lr),
            ("soft_voting", None),
        ):
            accs: list[float] = []
            mis: list[float] = []
            for r in range(N_REPEATS):
                y_test_noisy = _flip_labels(y_test, rho, SEED + r)
                if name == "soft_voting":
                    y_prob = voting_prob
                else:
                    y_prob = predict_proba_safe(est, X_test)
                accs.append(float(accuracy_score(
                    y_test_noisy, (y_prob >= 0.5).astype(int))))
                mis.append(achieved_mi_robust(y_prob, y_test_noisy))
            row[name] = {
                "mean_accuracy": float(np.mean(accs)),
                "std_accuracy": float(np.std(accs)),
                "mean_achieved_mi": float(np.mean(mis)),
                "std_achieved_mi": float(np.std(mis)),
            }
        results["noise_sweep"][str(rho)] = row
        print(
            f"  rho={rho:.2f}: soft-vote acc={row['soft_voting']['mean_accuracy']:.4f} "
            f"MI={row['soft_voting']['mean_achieved_mi']:.4f}"
        )

    # Duration-restricted minute-10 plateau on the temporal corpus.
    df = load_temporal_data()
    frame10 = df[df["frame"] == 10]
    for min_dur in (25, 30):
        sub = frame10[frame10["gameDuration"] >= min_dur * DURATION_MS_PER_MIN]
        if len(sub) < 100:
            continue
        feature_cols = [c for c in sub.columns if c not in
                        ("gameId", "gameDuration", "frame", "hasWon")]
        X10 = sub[feature_cols]
        y10 = sub["hasWon"].astype(int).values
        h_y10 = empirical_entropy(y10)
        splitter10 = StratifiedShuffleSplit(
            n_splits=1, test_size=TEST_SIZE, random_state=SEED
        )
        tr_idx, te_idx = next(splitter10.split(X10, y10))
        est = build_pipeline(
            LGBMClassifier(
                n_jobs=1, deterministic=True, random_state=SEED, verbose=-1
            ),
            scale=False,
        )
        est.fit(X10.iloc[tr_idx], y10[tr_idx])
        y_prob = predict_proba_safe(est, X10.iloc[te_idx])
        y_test_ = y10[te_idx]
        acc = float(accuracy_score(y_test_, (y_prob >= 0.5).astype(int)))
        mi = achieved_mi_robust(y_prob, y_test_)
        results["duration_restricted"][f"minute10_duration_ge_{min_dur}"] = {
            "n_samples": int(len(sub)),
            "win_rate": float(y10.mean()),
            "target_entropy_bits": h_y10,
            "test_accuracy": acc,
            "achieved_mi_bits": mi,
            "fano_max_accuracy": 1.0 - fano_min_error(mi, h_y10),
        }
        print(f"  minute-10, duration >= {min_dur} min (n={len(sub)}): "
              f"acc={acc:.4f} MI={mi:.4f}")

    json_path = OUTPUT_DIR / "label_noise_audit.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _plot_noise_sweep(results)
    print(f"\nLabel-noise audit saved to {json_path}")
    return results


def _plot_noise_sweep(results: dict[str, Any]) -> None:
    sweep = results["noise_sweep"]
    rhos = [float(k) for k in sweep]
    acc = [sweep[str(r)]["soft_voting"]["mean_accuracy"] for r in rhos]
    mi = [sweep[str(r)]["soft_voting"]["mean_achieved_mi"] for r in rhos]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.plot(rhos, acc, "o-", color="#4C72B0", linewidth=2)
    ax.axhline(results["clean"]["soft_voting"]["test_accuracy"],
               color="#C44E52", linestyle=":", label="Clean labels")
    ax.set_xlabel("Label-flip rate rho")
    ax.set_ylabel("Soft-voting accuracy (noisy labels)")
    ax.set_title("Accuracy vs test-label noise")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(rhos, mi, "o-", color="#4C72B0", linewidth=2)
    ax.axhline(results["clean"]["soft_voting"]["achieved_mi_bits"],
               color="#C44E52", linestyle=":", label="Clean labels")
    ax.set_xlabel("Label-flip rate rho")
    ax.set_ylabel("Achieved MI (bits, noisy labels)")
    ax.set_title("Achieved MI vs test-label noise")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "label_noise_sweep.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run_label_noise_audit()
