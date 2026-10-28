"""Rank-correlation diagnostics for the achieved-information column.

The paper reports Spearman correlations between the achieved information
I(S;Y) of each family, its test accuracy, and its ROC-AUC, to show that the
information column is a statement about score geometry rather than a
model-invariant measure of extraction.

Run with::

    python verify_rank_correlations.py

It reads only the committed artifact ``plots_ceiling/ceiling_summary.json``
and writes nothing.
"""

import json
import math
import os

ARTIFACT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "plots_ceiling", "ceiling_summary.json")

# Canonical family order used throughout the paper.
FAMILIES = ["logistic_regression", "mlp", "lightgbm", "svm", "soft_voting",
            "random_forest", "xgboost", "knn"]


def _ranks(values):
    """Average ranks (1-based) for ties, as in the usual Spearman convention."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def _pearson(x, y):
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = math.sqrt(sum((a - mx) ** 2 for a in x)
                    * sum((b - my) ** 2 for b in y))
    return num / den if den else float("nan")


def spearman(x, y):
    return _pearson(_ranks(x), _ranks(y))


def pearson(x, y):
    return _pearson(x, y)


def main():
    with open(ARTIFACT, encoding="utf-8") as fh:
        summary = json.load(fh)
    families = summary["model_families"]

    acc = [families[k]["test_accuracy"] for k in FAMILIES]
    auc = [families[k]["roc_auc"] for k in FAMILIES]
    mi = [families[k]["achieved_mi_bits"] for k in FAMILIES]

    print("Spearman(acc, MI)  = %.3f" % spearman(acc, mi))
    print("Spearman(AUC, MI)  = %.3f" % spearman(auc, mi))
    print("Spearman(acc, AUC) = %.3f" % spearman(acc, auc))
    print("Pearson(acc, MI), all 8 families  = %.3f" % pearson(acc, mi))
    print("Pearson(acc, MI), 7 without k-NN  = %.3f" % pearson(acc[:-1], mi[:-1]))
    print()
    base = [k for k in FAMILIES if k != "soft_voting"]
    mib = [families[k]["achieved_mi_bits"] for k in base]
    print("MI range, all 8 families    : %.3f - %.3f" % (min(mi), max(mi)))
    print("MI range, 7 base families  : %.3f - %.3f" % (min(mib), max(mib)))
    h_y = summary["target_entropy_bits_test"]
    print("MI as %% of H(Y)          : %.1f%% - %.1f%%"
          % (100 * min(mi) / h_y, 100 * max(mi) / h_y))


if __name__ == "__main__":
    main()
