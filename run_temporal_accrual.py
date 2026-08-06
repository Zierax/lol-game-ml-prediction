"""Temporal information-accrual model (novel quantitative contribution).

Answers reviewer W1 (``known phenomenon; lacks a novel quantitative
information-accrual contribution``) by turning the temporal ceiling curve
into a rate statement:

* Fit cumulative achieved MI ``I(m)`` vs frame minute ``m`` over the
  early-game regime ``m in [10, 24]``.
* Convert the slope (bits gained per game-minute) into minutes-to-85% and
  minutes-to-90%: the game-minute at which the Fano-implied *information
  ceiling* tolerates an error probability of 0.15 / 0.10, i.e. the smallest
  I with ``h^{-1}(H(Y) - I) <= P_e*``.

Reading the ceiling rather than one estimator makes the quantity
information-theoretic: how long (on average over games) one must wait for
the accumulating outcome evidence to clear a given error floor.

Persists plots_temporal/temporal_accrual.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.ceiling_analysis import fano_min_error

OUTPUT_DIR = Path("plots_temporal")
SOURCE = OUTPUT_DIR / "temporal_summary.json"

FIT_MIN_M: int = 10
FIT_MAX_M: int = 24
ERROR_TARGETS: tuple[float, float] = (0.15, 0.10)


def _ceiling_at(intercept: float, slope: float, minute: float, h_y: float) -> float:
    """Fano error ceiling from linearly-extrapolated MI, safely clamped."""
    mi = min(intercept + slope * minute, max(0.999999 * h_y, 0.0))
    mi = max(mi, 0.0)
    return fano_min_error(mi, h_y)


def minutes_to_error_ceiling(
    intercept: float,
    slope: float,
    target_error: float,
    h_y: float,
    max_minute: float = 60.0,
) -> float | None:
    """Game minute at which the linear Fano ceiling first permits target_error."""
    if _ceiling_at(intercept, slope, max_minute, h_y) > target_error:
        return None
    lo, hi = 0.0, max_minute
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _ceiling_at(intercept, slope, mid, h_y) <= target_error:
            hi = mid
        else:
            lo = mid
    return float(round(0.5 * (lo + hi), 2))


def run_temporal_accrual() -> dict[str, Any]:
    series = json.loads(SOURCE.read_text(encoding="utf-8"))
    ms = np.array([r["frame_minutes"] for r in series], dtype=float)
    mi = np.array([r["voting_achieved_mi"] for r in series], dtype=float)
    acc = np.array([r["voting_accuracy"] for r in series], dtype=float)
    h_y = float(series[0]["target_entropy_bits"])

    mask = (ms >= FIT_MIN_M) & (ms <= FIT_MAX_M)
    coeffs = np.polyfit(ms[mask], mi[mask], 1)
    slope, intercept = float(coeffs[0]), float(coeffs[1])
    pred = np.polyval(coeffs, ms[mask])
    r2 = 1.0 - float(
        np.sum((mi[mask] - pred) ** 2) / np.sum((mi[mask] - mi[mask].mean()) ** 2)
    )

    targets: dict[str, float | None] = {}
    for err in ERROR_TARGETS:
        acc_pct = int(round((1.0 - err) * 100))
        targets[f"minutes_to_{acc_pct}_pct"] = minutes_to_error_ceiling(
            intercept, slope, err, h_y
        )

    # Observed (non-extrapolated) crossings, from the measured ceiling series.
    obs_ceiling = np.array([r["voting_fano_max_accuracy"] for r in series])
    observed_crossings: dict[str, float | None] = {}
    for err in ERROR_TARGETS:
        o_acc_pct = int(round((1.0 - err) * 100))
        idx = np.flatnonzero(obs_ceiling >= 1.0 - err)
        observed_crossings[f"observed_minutes_to_{o_acc_pct}_pct"] = (
            float(ms[idx[0]]) if len(idx) > 0 else None
        )

    first_ge_85 = int(ms[np.argmax(acc >= 0.85)]) if (acc >= 0.85).any() else None

    result = {
        "fit_window_minutes": [FIT_MIN_M, FIT_MAX_M],
        "mi_slope_bits_per_minute": slope,
        "mi_intercept_bits": intercept,
        "mi_fit_r2": r2,
        "fitted_mi_at_m10": float(intercept + slope * 10),
        "observed_mi_at_m10": float(mi[0]),
        "target_entropy_bits": h_y,
        "minutes_to_error_targets": targets,
        "observed_crossings": observed_crossings,
        "first_measured_acc_ge_85_at_minute": first_ge_85,
        "observed_acc": dict(zip(map(int, ms), np.round(acc, 4).tolist())),
    }
    (OUTPUT_DIR / "temporal_accrual.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run_temporal_accrual()