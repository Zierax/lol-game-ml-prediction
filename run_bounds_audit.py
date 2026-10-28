"""Bounds audit: two independent cross-checks on the Fano-from-MI ceiling.

The Fano bound computed in ``run_ceiling.py`` rests on a plug-in estimate of
the achieved mutual information ``I(S; Y)``.  This script supplies two
independent checks on that number, both on the SAME held-out zoo predictions
so the comparison is like-for-like.

1. Binning-free achieved MI
   The binned plug-in estimator discards score resolution and is therefore
   downward biased by the data-processing inequality.  Here the same quantity
   is estimated with the Kraskov--Stoegbauer--Grassberger nearest-neighbour
   estimator (Kraskov et al., *Phys. Rev. E* **2004**, *69*, 061138) applied
   to the continuous score, which is binning-free.  Note that
   ``mutual_info_classif`` returns **nats**, so the value is divided by
   ``ln 2`` to land on the same log base as every other MI in the project.
   If the two estimators agree, the plug-in value is not a binning artefact.

2. Cross-fitted score-Bayes ceiling
   For a *calibrated* score the 0.5-threshold rule already **is** the Bayes
   rule, so ``accuracy = E[max(p, 1-p)] = A_SB`` identically and any apparent
   agreement is a tautology.  The informative version therefore estimates the
   score-binned posterior on one part of the held-out set and applies the
   induced rule to the remainder.  A negative difference is expected and is
   not a failure: the 0.5 rule uses the score continuously whereas the binned
   rule does not, so binned is slightly worse.

Persists ``plots_ceiling/bounds_audit.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

from src.ceiling_analysis import (
    SEED,
    TEST_SIZE,
    achieved_mi_robust,
    achieved_mutual_information,
    empirical_entropy,
    fano_max_accuracy,
    fano_min_error,
    knn_achieved_mi_1d,
    miller_madow_bias,
    score_bayes_ceiling_crossfit,
)
from src.zoo import ALL_FAMILIES, fit_and_score, load_master_split

OUTPUT_DIR = Path("plots_ceiling")
KSG_NEIGHBORS: int = 3


def run_bounds_audit() -> dict[str, Any]:
    """Audit the bounds using the SAME fitted zoo that backs Table 1.

    Every table in the manuscript is produced from one fit (see
    :mod:`src.zoo`); this script must not refit, or the two estimators
    would be compared across different models.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    master = load_master_split()
    y = master["y"]
    y_test = master["y_test"]
    h_y = empirical_entropy(y)
    scored = fit_and_score(master)
    probs = scored["probs"]

    def _record(name: str, y_prob: np.ndarray) -> dict[str, Any]:
        acc = float(accuracy_score(y_test, (y_prob >= 0.5).astype(int)))
        mi_binned = achieved_mi_robust(y_prob, y_test, bias_correct=False)
        mi_binned_bc = achieved_mi_robust(y_prob, y_test, bias_correct=True)
        mi_ksg = knn_achieved_mi_1d(y_prob, y_test, n_neighbors=KSG_NEIGHBORS)
        sb = score_bayes_ceiling_crossfit(y_prob, y_test)
        fano_min = fano_min_error(mi_binned_bc, h_y)
        rec = {
            "test_accuracy": acc,
            "achieved_mi_binned_bits": mi_binned,
            "achieved_mi_binned_bits_bias_corrected": mi_binned_bc,
            "miller_madow_bias_bits": mi_binned - mi_binned_bc,
            "achieved_mi_per_bin": {
                str(b): achieved_mutual_information(y_prob, y_test, b)
                for b in (10, 20, 40)
            },
            "achieved_mi_ksg_bits": mi_ksg,
            "knn_minus_binned_bits": mi_ksg - mi_binned_bc,
            "score_bayes_crossfit_ceiling": sb["crossfit_score_bayes_ceiling"],
            "score_bayes_plug_in_ceiling": sb["plug_in_score_bayes_ceiling"],
            "score_bayes_crossfit_minus_accuracy": sb["crossfit_minus_accuracy"],
            "score_bayes_ci_lower": sb["crossfit_minus_accuracy_ci_lower"],
            "score_bayes_ci_upper": sb["crossfit_minus_accuracy_ci_upper"],
            "fano_max_accuracy_from_plugin_mi": fano_max_accuracy(
                mi_binned, h_y),
            "fano_max_accuracy_from_biased_mi": fano_max_accuracy(
                mi_binned_bc, h_y),
            "fano_max_accuracy_from_knn_mi": fano_max_accuracy(mi_ksg, h_y),
        }
        print(
            f"  {name:>20s}: acc={acc:.4f} | MI plug-in={mi_binned_bc:.4f} "
            f"(uncorr {mi_binned:.4f}) KSG={mi_ksg:.4f} "
            f"| kNN-plug={mi_ksg - mi_binned_bc:+.4f} | "
            f"xfit score-Bayes={sb['crossfit_score_bayes_ceiling']:.4f} "
            f"({sb['crossfit_minus_accuracy']:+.4f}) | "
            f"Fano max={fano_max_accuracy(mi_binned_bc, h_y):.4f}"
        )
        return rec

    results: dict[str, Any] = {
        name: _record(name, probs[name]) for name in ALL_FAMILIES
    }

    audit = {
        "n_samples": int(len(master["y"])),
        "n_features": int(master["X"].shape[1]),
        "n_fit_rows": int(scored["n_fit"]),
        "test_size": TEST_SIZE,
        "seed": SEED,
        "target_entropy_bits": h_y,
        "source": (
            "Same fitted zoo as plots_ceiling/ceiling_summary.json "
            "(src.zoo.fit_and_score); no refit."
        ),
        "estimator_note": (
            "All mutual information values are in BITS. "
            "sklearn.feature_selection.mutual_info_classif returns nats and is "
            "divided by ln(2) in src.ceiling_analysis.knn_achieved_mi_1d."
        ),
        "fano_consistency": {
            name: {
                "eta_from_plugin_mi": (
                    (1.0 - rec["test_accuracy"])
                    / fano_min_error(
                        rec["achieved_mi_binned_bits"], h_y
                    )
                ),
                "eta_from_biased_mi": (
                    (1.0 - rec["test_accuracy"])
                    / fano_min_error(
                        rec["achieved_mi_binned_bits_bias_corrected"], h_y
                    )
                ),
                "eta_from_knn_mi": (
                    (1.0 - rec["test_accuracy"])
                    / fano_min_error(rec["achieved_mi_ksg_bits"], h_y)
                ),
            }
            for name, rec in results.items()
        },
        "models": results,
    }
    print("\nFano consistency (eta from the binning-free estimator, "
          "the only variant that can violate Fano in principle):")
    for name, e in audit["fano_consistency"].items():
        print(f"  {name:>20s} eta_knn={e['eta_from_knn_mi']:.3f}")
    for name, e in audit["fano_consistency"].items():
        if e["eta_from_knn_mi"] < 1.0:
            raise AssertionError(
                f"Fano's inequality violated for {name}: "
                f"eta={e['eta_from_knn_mi']:.4f} < 1."
            )

    (OUTPUT_DIR / "bounds_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(f"Saved {OUTPUT_DIR / 'bounds_audit.json'}")
    return audit


if __name__ == "__main__":
    run_bounds_audit()
