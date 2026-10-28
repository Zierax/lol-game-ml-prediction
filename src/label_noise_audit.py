"""Label-noise robustness audit.

Design note (2026-09-30 revision)
---------------------------------
An earlier version of this audit corrupted the **held-out test labels only**
while keeping the fitted model frozen.  That experiment is an algebraic
identity, not a measurement: flipping a fraction ``rho`` of the labels of a
frozen prediction vector maps accuracy by

    acc(rho) = rho + (1 - 2 rho) acc(0),

independently of the model, the dataset, or the score distribution.  It
therefore carries **no** information about label quality and cannot
discriminate any hypothesis about noise.

The only informative design is to corrupt the **training** labels and refit,
so that the learned score itself changes, and then to measure the accuracy and
the achieved mutual information against the *clean* test labels.  That is what
this module does.  A secondary held-out-only sweep is retained, but reported
explicitly as the degenerate identity it is, so that the difference between
the two designs is visible rather than hidden.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier

from src.ceiling_analysis import achieved_mi_robust, fano_min_error
from src.zoo import fit_and_score, load_master_split

matplotlib.use("Agg")

SEED: int = 42
TEST_SIZE: float = 0.2
NOISE_LEVELS: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)
N_REPEATS: int = 3

OUTPUT_DIR = Path("plots_ceiling")


def _flip_labels(y: np.ndarray, rho: float, seed: int) -> np.ndarray:
    """Symmetrically flip a fraction ``rho`` of the labels (seeded)."""
    rng = np.random.RandomState(seed)
    flip = rng.random(len(y)) < rho
    y_noisy = y.copy()
    y_noisy[flip] = 1 - y_noisy[flip]
    return y_noisy


def _new_families() -> dict[str, Any]:
    """Fresh, seeded instances of the two deterministic families."""
    return {
        "logistic_regression": build_pipeline(
            LogisticRegression(max_iter=5000, random_state=SEED)
        ),
        "lightgbm": build_pipeline(
            LGBMClassifier(
                n_jobs=1, deterministic=True, random_state=SEED, verbose=-1
            ),
            scale=False,
        ),
    }


def run_label_noise_audit() -> dict[str, Any]:
    """Run the label-noise sweep and persist JSON + figure."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    FAMILIES = ("logistic_regression", "lightgbm", "soft_voting")

    master = load_master_split()
    y_test = master["y_test"]
    X_train = master["X_train"]
    y_train = master["y_train"]

    results: dict[str, Any] = {
        "design": {
            "primary": (
                "training-label corruption with refit, calibration fold and "
                "test labels left clean"
            ),
            "secondary": (
                "test-label corruption with a frozen model (algebraic "
                "identity; reported for contrast only)"
            ),
            "noise_levels": list(NOISE_LEVELS),
            "n_repeats": N_REPEATS,
            "seed": SEED,
            "source": "src.zoo.load_master_split / fit_and_score",
        },
        "clean": {},
        "noise_sweep": {},
        "degenerate_test_only_sweep": {},
    }

    # ------------------------------------------------------------------
    # Clean reference, from the shared fit (identical to Table 1)
    # ------------------------------------------------------------------
    scored = fit_and_score(master)
    for name in FAMILIES:
        p = scored["probs"][name]
        results["clean"][name] = {
            "test_accuracy": scored["acc"][name],
            "achieved_mi_bits": achieved_mi_robust(p, y_test),
        }
    print("Clean reference (same fit as Table 1):")
    for name, r in results["clean"].items():
        print(f"  {name:>22s}: acc={r['test_accuracy']:.4f} "
              f"MI={r['achieved_mi_bits']:.4f}")

    # ------------------------------------------------------------------
    # Primary design: corrupt TRAINING labels, refit, score CLEAN test
    # ------------------------------------------------------------------
    print("\nTraining-label noise (refit; calibration fold and test clean):")
    FAMILIES = ("logistic_regression", "lightgbm", "soft_voting")

    for rho in NOISE_LEVELS:
        row: dict[str, Any] = {"rho": rho}
        acc_by: dict[str, list[float]] = {n: [] for n in FAMILIES}
        mi_by: dict[str, list[float]] = {n: [] for n in FAMILIES}
        for r in range(N_REPEATS):
            y_train_noisy = _flip_labels(y_train, rho, SEED + 100 * r)
            fit_labels = y_train_noisy[master["fit_rel"]]
            s = fit_and_score(master, fit_labels=fit_labels)
            for name in acc_by:
                acc_by[name].append(s["acc"][name])
                mi_by[name].append(
                    achieved_mi_robust(s["probs"][name], y_test)
                )
        for name in acc_by:
            row[name] = {
                "mean_accuracy": float(np.mean(acc_by[name])),
                "std_accuracy": float(np.std(acc_by[name], ddof=1)),
                "se_accuracy": float(
                    np.std(acc_by[name], ddof=1) / np.sqrt(len(acc_by[name]))
                ),
                "mean_achieved_mi": float(np.mean(mi_by[name])),
                "std_achieved_mi": float(np.std(mi_by[name], ddof=1)),
            }
        results["noise_sweep"][str(rho)] = row
        print(f"  rho={rho:.2f}: " + "  ".join(
            f"{n} acc={row[n]['mean_accuracy']:.4f} "
            f"MI={row[n]['mean_achieved_mi']:.4f}"
            for n in ("logistic_regression", "lightgbm", "soft_voting")))

    # Per-family degradation at the highest noise level.  The confidence
    # intervals use the exact t-critical value for N_REPEATS - 1 degrees of
    # freedom (here df = 2, crit ~= 4.30), not the normal approximation.
    results["degradation_at_rho_max"] = {}
    rho_max = str(max(NOISE_LEVELS))
    crit_t = float(stats.t.ppf(0.975, N_REPEATS - 1))
    for name in results["clean"]:
        clean = results["clean"][name]
        noisy = results["noise_sweep"][rho_max][name]
        se = np.sqrt(
            results["noise_sweep"][rho_max][name]["se_accuracy"] ** 2
            + results["noise_sweep"]["0.0"][name]["se_accuracy"] ** 2
        )
        drop = 100 * (clean["test_accuracy"] - noisy["mean_accuracy"])
        results["degradation_at_rho_max"][name] = {
            "rho": max(NOISE_LEVELS),
            "accuracy_drop_points": float(drop),
            "accuracy_drop_ci_lower": float(drop - crit_t * se * 100.0),
            "accuracy_drop_ci_upper": float(drop + crit_t * se * 100.0),
            "mi_drop_percent": float(
                100 * (clean["achieved_mi_bits"] - noisy["mean_achieved_mi"])
                / clean["achieved_mi_bits"]
            ),
        }
    print(f"\nDegradation at rho = {max(NOISE_LEVELS)}:")
    for name, d in results["degradation_at_rho_max"].items():
        sig = "SIGNIFICANT" if d["accuracy_drop_ci_lower"] > 0 else "n.s."
        print(f"  {name:>22s}: drops {d['accuracy_drop_points']:.2f} pts "
              f"[{d['accuracy_drop_ci_lower']:.2f}, {d['accuracy_drop_ci_upper']:.2f}] "
              f"({sig}); MI -{d['mi_drop_percent']:.1f}%")

    # ------------------------------------------------------------------
    # Secondary (degenerate) design, reported for contrast
    # ------------------------------------------------------------------
    voting_clean = scored["probs"]["soft_voting"]
    print("\nTest-label-only noise with a frozen model (degenerate identity):")
    for rho in NOISE_LEVELS:
        accs, mis = [], []
        for r in range(N_REPEATS):
            y_noisy = _flip_labels(y_test, rho, SEED + r)
            accs.append(float(
                accuracy_score(y_noisy, (voting_clean >= 0.5).astype(int))
            ))
            mis.append(achieved_mi_robust(voting_clean, y_noisy))
        identity = rho + (1 - 2 * rho) * results["clean"]["soft_voting"]["test_accuracy"]
        results["degenerate_test_only_sweep"][str(rho)] = {
            "rho": rho,
            "mean_accuracy": float(np.mean(accs)),
            "algebraic_identity_prediction": float(identity),
            "mean_achieved_mi": float(np.mean(mis)),
            "note": "acc(rho) = rho + (1-2rho) acc(0) holds for ANY frozen score",
        }
        print(f"  rho={rho:.2f}: acc={np.mean(accs):.4f} "
              f"identity={identity:.4f} (residual {np.mean(accs)-identity:+.5f})")

    # ------------------------------------------------------------------
    # NOTE: duration-restricted minute-10 information is NOT recomputed
    # here. It is produced by the canonical temporal pipeline
    # (run_temporal_ceiling.py -> plots_temporal/
    # temporal_duration_stratified.json) and is cited in the paper from
    # there. Duplicating the fit here would risk a second, divergent
    # implementation of the same stratified quantity.

    json_path = OUTPUT_DIR / "label_noise_audit.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _plot_noise_sweep(results)
    print(f"\nLabel-noise audit saved to {json_path}")
    return results


def _plot_noise_sweep(results: dict[str, Any]) -> None:
    BLUE, ORANGE, GREEN, BLACK, GRAY = ("#4C72B0", "#DD8452", "#55A868",
                                        "#333333", "#888888")
    rhos = [float(k) for k in results["noise_sweep"]]
    clean_acc = results["clean"]["soft_voting"]["test_accuracy"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    clean_acc = results["clean"]["soft_voting"]["test_accuracy"]
    clean_mi = results["clean"]["soft_voting"]["achieved_mi_bits"]
    families = ("logistic_regression", "lightgbm", "soft_voting")
    colors = {"logistic_regression": BLUE, "lightgbm": ORANGE,
              "soft_voting": GREEN}

    ax = axes[0]
    for name in families:
        ax.plot(rhos,
                [results["noise_sweep"][str(r)][name]["mean_accuracy"]
                 for r in rhos], "o-", color=colors[name], lw=2, ms=4,
                label=name.replace("_", " "))
    ax.plot(
        rhos,
        [results["degenerate_test_only_sweep"][str(r)]["mean_accuracy"]
         for r in rhos],
        "k--", linewidth=1.8,
        label="frozen model, test-label noise (identity)",
    )
    ax.axhline(clean_acc, color=GRAY, ls=":", lw=1.2)
    ax.set_xlabel("Training-label flip rate $\\rho$")
    ax.set_ylabel("Accuracy on clean test labels")
    ax.set_title("Robustness to training-label noise, by family")
    ax.legend(loc="lower left", fontsize=7)

    ax = axes[1]
    for name in families:
        base = results["clean"][name]["achieved_mi_bits"]
        vals = [100 * (base - results["noise_sweep"][str(r)][name]["mean_achieved_mi"])
                / base for r in rhos]
        ax.plot(rhos, vals, "o-", color=colors[name], lw=2, ms=4,
                label=name.replace("_", " "))
    ax.set_xlabel("Training-label flip rate $\\rho$")
    ax.set_ylabel("Achieved information lost (%)")
    ax.set_title("Information loss under training-label noise")
    ax.legend(loc="upper left", fontsize=7)

    ax = axes[2]
    deg = results["degradation_at_rho_max"]
    names = list(deg)
    drops = [deg[n]["accuracy_drop_points"] for n in names]
    los = [deg[n]["accuracy_drop_ci_lower"] for n in names]
    his = [deg[n]["accuracy_drop_ci_upper"] for n in names]
    x = np.arange(len(names))
    ax.bar(x, drops, 0.5, color=ORANGE)
    ax.errorbar(x, drops, yerr=[np.array(drops) - np.array(los),
                                np.array(his) - np.array(drops)],
                fmt="none", ecolor=BLACK, capsize=4, lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    ax.set_ylabel("Accuracy drop (points)")
    ax.set_title(f"Degradation at $\\rho = {max(NOISE_LEVELS)}$ "
                 f"(95% CI)")
    ax.axhline(0, color=BLACK, lw=0.8)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "label_noise_sweep.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    run_label_noise_audit()
