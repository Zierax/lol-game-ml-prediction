"""Temporal information-accrual model, with the robustness checks it needs.

Turns the temporal ceiling curve into a rate statement: fit cumulative
achieved mutual information ``I(m)`` against frame minute ``m`` and convert
the slope into the game minute at which the Fano-implied information ceiling
for the extracted information first tolerates a given error probability.

Three checks are performed because the linear form is easy to over-trust:

1. **Window sensitivity.**  The slope and the two crossing minutes are
   recomputed for several fit windows.  If the headline numbers move by tens
   of percent under a defensible change of window, they are not a property of
   the data.
2. **Saturating alternative.**  A linear model in ``m`` has no cap: it
   eventually exceeds ``H(Y)``, which is impossible, and its intercept claims
   non-zero information at ``m = 0`` when no features exist.  A saturating
   (exponential-approach) model is fitted alongside and both are reported.
3. **Observed vs.\ extrapolated crossings.**  The fitted crossings are
   extrapolations; the observed crossings are read off the measured series.
   They are reported separately, and a ``null`` observed crossing is reported
   as ``null`` rather than omitted.

Persists ``plots_temporal/temporal_accrual.json``.
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
FIT_WINDOWS: tuple[tuple[int, int], ...] = (
    (10, 20), (10, 24), (10, 26), (10, 30), (12, 28),
)
ERROR_TARGETS: tuple[float, ...] = (0.15, 0.10)


def _ceiling_at(mi: float, h_y: float) -> float:
    """Fano error floor permitted by ``mi`` bits, clipped to the valid domain."""
    return fano_min_error(float(np.clip(mi, 0.0, h_y)), h_y)


def _linear_crossing(a: float, b: float, target_error: float, h_y: float,
                     max_minute: float = 60.0) -> float | None:
    """Minute at which ``1 - h^-1(H - (a + b m))`` first reaches 1 - target_error."""
    if _ceiling_at(min(a + b * max_minute, h_y), h_y) > target_error:
        return None
    lo, hi = 0.0, max_minute
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _ceiling_at(min(a + b * mid, h_y), h_y) <= target_error:
            hi = mid
        else:
            lo = mid
    return float(round(0.5 * (lo + hi), 2))


def _saturating_crossing(k: float, half: float, m_inf: float,
                         target_error: float, h_y: float,
                         max_minute: float = 60.0) -> float | None:
    """Crossing for ``I(m) = m_inf (1 - exp(-k m)) / (1 - exp(-k half))``-style curves.

    Uses the fitted saturating form ``I(m) = m_inf * (1 - exp(-k m))`` and
    inverts it numerically, respecting the ``I <= H(Y)`` cap.
    """
    def _mi(m: float) -> float:
        return min(m_inf * (1.0 - np.exp(-k * m)), h_y)

    if _ceiling_at(_mi(max_minute), h_y) > target_error:
        return None
    lo, hi = 0.0, max_minute
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _ceiling_at(_mi(mid), h_y) <= target_error:
            hi = mid
        else:
            lo = mid
    return float(round(0.5 * (lo + hi), 2))


def _fit_saturating(m: np.ndarray, mi: np.ndarray) -> tuple[float, float, float]:
    """Least-squares fit of ``I(m) = m_inf (1 - exp(-k m))`` by grid + refine.

    Returns ``(k, m_inf, r2)``.
    """
    best = (np.inf, 1e-3, 0.0, 0.0)
    for k in np.geomspace(1e-3, 0.5, 60):
        for m_inf in np.linspace(max(mi.max(), 1e-3), 3.0, 60):
            pred = m_inf * (1.0 - np.exp(-k * m))
            sse = float(np.sum((mi - pred) ** 2))
            if sse < best[0]:
                r2 = 1.0 - sse / float(np.sum((mi - mi.mean()) ** 2))
                best = (sse, k, m_inf, r2)
    return float(best[1]), float(best[2]), float(best[3])


def run_temporal_accrual() -> dict[str, Any]:
    series = json.loads(SOURCE.read_text(encoding="utf-8"))
    ms = np.array([r["frame_minutes"] for r in series], dtype=float)
    mi = np.array([r["voting_achieved_mi"] for r in series], dtype=float)
    acc = np.array([r["voting_accuracy"] for r in series], dtype=float)
    h_y = float(series[0]["target_entropy_bits"])

    # ---------------- primary linear fit on the declared window ----------
    mask = (ms >= FIT_MIN_M) & (ms <= FIT_MAX_M)
    coeffs = np.polyfit(ms[mask], mi[mask], 1)
    slope, intercept = float(coeffs[0]), float(coeffs[1])
    pred = np.polyval(coeffs, ms[mask])
    r2 = 1.0 - float(
        np.sum((mi[mask] - pred) ** 2)
        / np.sum((mi[mask] - mi[mask].mean()) ** 2)
    )
    # OLS standard error of the slope, so the rate carries an interval.
    resid = mi[mask] - pred
    dof = int(mask.sum()) - 2
    sxx = float(np.sum((ms[mask] - ms[mask].mean()) ** 2))
    slope_se = float(np.sqrt((resid ** 2).sum() / dof / sxx))

    targets: dict[str, float | None] = {}
    for err in ERROR_TARGETS:
        pct = int(round((1.0 - err) * 100))
        targets[f"minutes_to_{pct}_pct"] = _linear_crossing(
            intercept, slope, err, h_y
        )

    # ---------------- window sensitivity ---------------------------------
    window_table: dict[str, Any] = {}
    for lo_m, hi_m in FIT_WINDOWS:
        m2 = (ms >= lo_m) & (ms <= hi_m)
        if m2.sum() < 3:
            continue
        c2 = np.polyfit(ms[m2], mi[m2], 1)
        s2, i2 = float(c2[0]), float(c2[1])
        p2 = np.polyval(c2, ms[m2])
        r2b = 1.0 - float(
            np.sum((mi[m2] - p2) ** 2) / np.sum((mi[m2] - mi[m2].mean()) ** 2)
        )
        window_table[f"{lo_m}-{hi_m}"] = {
            "slope_bits_per_minute": s2,
            "intercept_bits": i2,
            "r2": r2b,
            "minutes_to_85_pct": _linear_crossing(i2, s2, 0.15, h_y),
            "minutes_to_90_pct": _linear_crossing(i2, s2, 0.10, h_y),
            "fitted_mi_at_m10": float(i2 + s2 * 10),
        }
    slopes = [v["slope_bits_per_minute"] for v in window_table.values()]
    t85 = [v["minutes_to_85_pct"] for v in window_table.values()
           if v["minutes_to_85_pct"] is not None]
    t90 = [v["minutes_to_90_pct"] for v in window_table.values()
           if v["minutes_to_90_pct"] is not None]

    # ---------------- saturating alternative -----------------------------
    k_sat, m_inf, r2_sat = _fit_saturating(ms, mi)
    sat_targets = {
        f"minutes_to_{int(round((1 - e) * 100))}_pct": _saturating_crossing(
            k_sat, 0.0, m_inf, e, h_y
        )
        for e in ERROR_TARGETS
    }

    # ---------------- observed (non-extrapolated) crossings --------------
    obs_ceiling = np.array([r["voting_fano_max_accuracy"] for r in series])
    observed_crossings: dict[str, float | None] = {}
    for err in ERROR_TARGETS:
        pct = int(round((1.0 - err) * 100))
        idx = np.flatnonzero(obs_ceiling >= 1.0 - err)
        observed_crossings[f"observed_minutes_to_{pct}_pct"] = (
            float(ms[idx[0]]) if len(idx) > 0 else None
        )

    result = {
        "fit_window_minutes": [FIT_MIN_M, FIT_MAX_M],
        "mi_slope_bits_per_minute": slope,
        "mi_slope_standard_error": slope_se,
        "mi_slope_95ci": [slope - 1.96 * slope_se, slope + 1.96 * slope_se],
        "mi_intercept_bits": intercept,
        "mi_fit_r2": r2,
        "fitted_mi_at_m10": float(intercept + slope * 10),
        "observed_mi_at_m10": float(mi[0]),
        "fit_overshoot_at_m10_bits": float(intercept + slope * 10 - mi[0]),
        "target_entropy_bits": h_y,
        "minutes_to_error_targets": targets,
        "window_sensitivity": {
            "table": window_table,
            "slope_range_bits_per_minute": [min(slopes), max(slopes)],
            "slope_relative_spread_pct": float(
                100.0 * (max(slopes) - min(slopes)) / np.mean(slopes)
            ),
            "minutes_to_85_pct_range": [min(t85), max(t85)] if t85 else None,
            "minutes_to_90_pct_range": [min(t90), max(t90)] if t90 else None,
        },
        "saturating_fit": {
            "form": "I(m) = m_inf * (1 - exp(-k m))",
            "k": k_sat,
            "m_inf_bits": m_inf,
            "r2": r2_sat,
            "m_inf_exceeds_h_y": bool(m_inf > h_y),
            "minutes_to_error_targets": sat_targets,
        },
        "observed_crossings": observed_crossings,
        "first_measured_acc_ge_85_at_minute": (
            int(ms[np.argmax(acc >= 0.85)]) if (acc >= 0.85).any() else None
        ),
        "caveats": [
            "The linear fit overshoots the observed MI at minute 10 by "
            f"{1000 * (intercept + slope * 10 - mi[0]):.1f} millibits; the true "
            "MI curve is convex and saturating.",
            "A linear model in m has no cap and eventually exceeds H(Y), which "
            "is impossible; the saturating fit is reported for that reason.",
            "The 90% crossing is an extrapolation and, where the observed "
            "value is null, is never reached within the measured range.",
        ],
        "observed_acc": dict(zip(map(int, ms), np.round(acc, 4).tolist())),
    }
    (OUTPUT_DIR / "temporal_accrual.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run_temporal_accrual()
