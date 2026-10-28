"""Feature engineering for LoL 10-minute match prediction.

Computes differential, ratio, and interaction features from the raw
10-minute game snapshot.  Every feature is derived exclusively from
information available at the 10-minute mark — no future game state
is used.

Invariants
----------
* No target variable (``blueWins``) is referenced during transformation.
* All ratio features guard against division-by-zero with ``max(denominator, 1)``.
* Output DataFrame index is identical to input index (caller controls split).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

EPSILON: float = 1e-8


def _safe_divide(
    numerator: pd.Series | float,
    denominator: pd.Series | float,
    fill: float = 0.0,
) -> pd.Series:
    """Element-wise division that returns *fill* where denominator is zero."""
    denom = denominator.replace(0, np.nan)
    result = numerator / denom
    return result.fillna(fill)


def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """Transform raw 10-minute features into an enriched feature matrix.

    The function computes three categories of derived features:

    1. **Differential** — blue-side minus red-side for each raw stat.
    2. **Ratio / rate** — per-unit efficiency and participation metrics.
    3. **Interaction** — cross-feature products that capture compound
       advantages (e.g. gold advantage times kill advantage).

    No scaling or normalisation is applied — that belongs in the
    downstream pipeline.

    Args:
        X: Raw feature matrix as loaded by ``data_loader.load_dataset``.

    Returns:
        A new DataFrame with the original columns *plus* engineered features.
    """
    df = X.copy()

    # ------------------------------------------------------------------
    # 1. Differential features  (blue − red)
    # ------------------------------------------------------------------
    diff_pairs: list[tuple[str, str, str]] = [
        ("blueKills", "redKills", "killDiff"),
        ("blueDeaths", "redDeaths", "deathDiff"),
        ("blueAssists", "redAssists", "assistDiff"),
        ("blueWardsPlaced", "redWardsPlaced", "wardPlacedDiff"),
        ("blueWardsDestroyed", "redWardsDestroyed", "wardDestroyedDiff"),
        ("blueEliteMonsters", "redEliteMonsters", "eliteMonsterDiff"),
        ("blueDragons", "redDragons", "dragonDiff"),
        ("blueHeralds", "redHeralds", "heraldDiff"),
        ("blueTowersDestroyed", "redTowersDestroyed", "towerDiff"),
        ("blueTotalMinionsKilled", "redTotalMinionsKilled", "minionDiff"),
        (
            "blueTotalJungleMinionsKilled",
            "redTotalJungleMinionsKilled",
            "jungleMinionDiff",
        ),
    ]
    for blue_col, red_col, name in diff_pairs:
        df[name] = df[blue_col] - df[red_col]

    # Level and gold/experience differentials are already present as
    # blueGoldDiff / blueExperienceDiff, but we add level explicitly.
    df["levelDiff"] = df["blueAvgLevel"] - df["redAvgLevel"]

    # ------------------------------------------------------------------
    # 2. Ratio / rate features
    # ------------------------------------------------------------------
    # KDA  — (kills + assists) / max(deaths, 1)
    df["blueKDA"] = _safe_divide(
        df["blueKills"] + df["blueAssists"],
        df["blueDeaths"].clip(lower=1),
    )
    df["redKDA"] = _safe_divide(
        df["redKills"] + df["redAssists"],
        df["redDeaths"].clip(lower=1),
    )
    df["kdaDiff"] = df["blueKDA"] - df["redKDA"]

    # Kill-death ratio
    df["blueKDR"] = _safe_divide(df["blueKills"], df["blueDeaths"].clip(lower=1))
    df["redKDR"] = _safe_divide(df["redKills"], df["redDeaths"].clip(lower=1))
    df["kdrDiff"] = df["blueKDR"] - df["redKDR"]

    # Kill participation  —  (kills + assists) / total team combat involvement
    blue_team_combat = df["blueKills"] + df["blueDeaths"] + df["blueAssists"]
    red_team_combat = df["redKills"] + df["redDeaths"] + df["redAssists"]
    df["blueKillParticipation"] = _safe_divide(
        df["blueKills"] + df["blueAssists"],
        blue_team_combat.clip(lower=1),
    )
    df["redKillParticipation"] = _safe_divide(
        df["redKills"] + df["redAssists"],
        red_team_combat.clip(lower=1),
    )
    df["killParticipationDiff"] = (
        df["blueKillParticipation"] - df["redKillParticipation"]
    )

    # Gold efficiency — gold per total minion killed (incl. jungle)
    blue_total_minions = (
        df["blueTotalMinionsKilled"] + df["blueTotalJungleMinionsKilled"]
    ).clip(lower=1)
    red_total_minions = (
        df["redTotalMinionsKilled"] + df["redTotalJungleMinionsKilled"]
    ).clip(lower=1)
    df["blueGoldPerMinion"] = _safe_divide(df["blueTotalGold"], blue_total_minions)
    df["redGoldPerMinion"] = _safe_divide(df["redTotalGold"], red_total_minions)
    df["goldPerMinionDiff"] = df["blueGoldPerMinion"] - df["redGoldPerMinion"]

    # Gold-to-experience ratio (economic efficiency)
    df["blueGoldExpRatio"] = _safe_divide(
        df["blueTotalGold"], df["blueTotalExperience"].clip(lower=1)
    )
    df["redGoldExpRatio"] = _safe_divide(
        df["redTotalGold"], df["redTotalExperience"].clip(lower=1)
    )
    df["goldExpRatioDiff"] = df["blueGoldExpRatio"] - df["redGoldExpRatio"]

    # Objective control rate — objectives taken / wards placed (proxy for map control)
    blue_objectives = (
        df["blueDragons"] + df["blueHeralds"] + df["blueEliteMonsters"]
    )
    red_objectives = (
        df["redDragons"] + df["redHeralds"] + df["redEliteMonsters"]
    )
    df["blueObjectiveRate"] = _safe_divide(
        blue_objectives, df["blueWardsPlaced"].clip(lower=1)
    )
    df["redObjectiveRate"] = _safe_divide(
        red_objectives, df["redWardsPlaced"].clip(lower=1)
    )
    df["objectiveRateDiff"] = df["blueObjectiveRate"] - df["redObjectiveRate"]

    # Vision efficiency — wards destroyed / wards placed
    df["blueVisionEfficiency"] = _safe_divide(
        df["blueWardsDestroyed"], df["blueWardsPlaced"].clip(lower=1)
    )
    df["redVisionEfficiency"] = _safe_divide(
        df["redWardsDestroyed"], df["redWardsPlaced"].clip(lower=1)
    )
    df["visionEfficiencyDiff"] = (
        df["blueVisionEfficiency"] - df["redVisionEfficiency"]
    )

    # Experience per level — proxy for individual performance
    df["blueExpPerLevel"] = _safe_divide(
        df["blueTotalExperience"], df["blueAvgLevel"].clip(lower=1)
    )
    df["redExpPerLevel"] = _safe_divide(
        df["redTotalExperience"], df["redAvgLevel"].clip(lower=1)
    )
    df["expPerLevelDiff"] = df["blueExpPerLevel"] - df["redExpPerLevel"]

    # ------------------------------------------------------------------
    # 3. Interaction features
    # ------------------------------------------------------------------
    # Gold advantage multiplied by kill advantage — captures compound leads
    df["goldKillInteraction"] = df["blueGoldDiff"] * df["killDiff"]

    # Gold advantage multiplied by level advantage
    df["goldLevelInteraction"] = df["blueGoldDiff"] * df["levelDiff"]

    # Experience advantage multiplied by kill advantage
    df["expKillInteraction"] = df["blueExperienceDiff"] * df["killDiff"]

    # Non-linear gold advantage — captures extreme leads
    df["goldDiffSquared"] = df["blueGoldDiff"] ** 2
    df["expDiffSquared"] = df["blueExperienceDiff"] ** 2

    # Composite advantage score.
    #
    # IMPORTANT (disclosure, 2026-09-30 audit): the four weights below are
    # FIXED A PRIORI and were never tuned against the outcome.  They were
    # chosen to put the four differentials on comparable scales (gold and
    # experience are O(1000) while kills and levels are O(1-5)), and the
    # sign-times-square-root compression is applied to the two large-scale
    # quantities to limit their dominance of the linear sum.  The complete
    # definition is
    #
    #     compositeAdvantage = 0.3 * sign(dGold) * sqrt(|dGold|)
    #                       + 0.3 * sign(dExp)  * sqrt(|dExp|)
    #                       + 0.2 * dKills
    #                       + 0.2 * dLevel
    #
    # Note that this feature is a deterministic function of four columns that
    # are already in the matrix, so its rank-1 mutual information is
    # expected rather than discovered: mutual information is invariant under
    # monotone reparameterisation, so the composite cannot add information
    # beyond its inputs.  See features_audit() for the rank correlation.
    df["compositeAdvantage"] = (
        0.3 * np.sign(df["blueGoldDiff"]) * np.sqrt(np.abs(df["blueGoldDiff"]))
        + 0.3
        * np.sign(df["blueExperienceDiff"])
        * np.sqrt(np.abs(df["blueExperienceDiff"]))
        + 0.2 * df["killDiff"]
        + 0.2 * df["levelDiff"]
    )

    # First blood interaction — does first blood amplify the kill advantage?
    df["firstBloodKillInteraction"] = df["blueFirstBlood"] * df["killDiff"]

    # Tower pressure — towers destroyed relative to kills
    df["blueTowerKillRatio"] = _safe_divide(
        df["blueTowersDestroyed"], df["blueKills"].clip(lower=1)
    )
    df["redTowerKillRatio"] = _safe_divide(
        df["redTowersDestroyed"], df["redKills"].clip(lower=1)
    )
    df["towerKillRatioDiff"] = df["blueTowerKillRatio"] - df["redTowerKillRatio"]

    # NOTE (2026-09-30 audit): goldPerMinDiff and csPerMinDiff were removed.
    # They are EXACT rescalings of already-present columns: because the
    # snapshot is taken at a fixed 10 minutes, blueGoldPerMin ==
    # blueTotalGold/10 and blueCSPerMin == blueTotalMinionsKilled/10 hold to
    # machine precision, so goldPerMinDiff == blueGoldDiff/10 and
    # csPerMinDiff == minionDiff/10 exactly. They contributed zero
    # information while adding exact linear dependencies to the design
    # matrix. See features_audit() for the rank diagnostic.

    # Jungle control — blue jungle minions / total jungle minions
    total_jungle = (
        df["blueTotalJungleMinionsKilled"]
        + df["redTotalJungleMinionsKilled"]
    ).clip(lower=1)
    df["jungleControlRatio"] = _safe_divide(
        df["blueTotalJungleMinionsKilled"], total_jungle
    )

    # Lane minion dominance — blue lane minions / total lane minions
    total_lane = (
        df["blueTotalMinionsKilled"] + df["redTotalMinionsKilled"]
    ).clip(lower=1)
    df["laneMinionDominance"] = _safe_divide(
        df["blueTotalMinionsKilled"], total_lane
    )

    # Total experience share
    total_exp = (df["blueTotalExperience"] + df["redTotalExperience"]).clip(lower=1)
    df["experienceShare"] = _safe_divide(df["blueTotalExperience"], total_exp)

    # Total gold share
    total_gold = (df["blueTotalGold"] + df["redTotalGold"]).clip(lower=1)
    df["goldShare"] = _safe_divide(df["blueTotalGold"], total_gold)

    # Replace any residual NaN/inf with 0
    df = df.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    return df


ENGINEERED_DIFFERENTIALS: tuple[str, ...] = (
    "killDiff", "deathDiff", "assistDiff", "wardPlacedDiff",
    "wardDestroyedDiff", "eliteMonsterDiff", "dragonDiff", "heraldDiff",
    "towerDiff", "minionDiff", "jungleMinionDiff", "levelDiff",
)

#: Exact (machine-precision) linear dependencies retained only for audit.
EXACT_DEPENDENCIES: tuple[tuple[str, str, str], ...] = (
    ("blueGoldDiff", "blueTotalGold", "blueGoldDiff == blueTotalGold - redTotalGold"),
    ("blueExperienceDiff", "blueTotalExperience",
     "blueExperienceDiff == blueTotalExperience - redTotalExperience"),
    ("blueGoldPerMin", "blueTotalGold", "blueGoldPerMin == blueTotalGold / 10"),
    ("blueCSPerMin", "blueTotalMinionsKilled",
     "blueCSPerMin == blueTotalMinionsKilled / 10"),
)


def features_audit(X_raw: pd.DataFrame, X_eng: pd.DataFrame) -> dict:
    """Audit the engineered design matrix for rank deficiency and collinearity.

    Reports the numerical rank of the standardised matrix (so readers can see
    how many columns are exactly dependent) and the rank correlation of the
    top mutual-information features, which quantifies how much of the
    "informative" set is a single latent signal re-expressed.

    Args:
        X_raw: The cleaned 35-column raw matrix.
        X_eng: The engineered 85-column matrix returned by
            :func:`engineer_features`.

    Returns:
        Dictionary with ``n_columns``, ``numerical_rank``, ``rank_deficiency``,
        the per-threshold component counts, and the Spearman correlation
        matrix of the top features.
    """
    from scipy.stats import spearmanr

    scaled = StandardScaler().fit_transform(X_eng)
    rank = int(np.linalg.matrix_rank(scaled))
    pca = PCA(random_state=42).fit(scaled)
    cum = np.cumsum(pca.explained_variance_ratio_)

    n_cols = int(X_eng.shape[1])
    top_cols = [c for c in ("compositeAdvantage", "blueGoldDiff", "goldShare",
                            "blueExperienceDiff", "killDiff", "levelDiff")
                if c in X_eng.columns]
    corr = spearmanr(X_eng[top_cols].to_numpy()).statistic

    return {
        "n_raw_columns": int(X_raw.shape[1]),
        "n_columns": n_cols,
        "numerical_rank": rank,
        "rank_deficiency": n_cols - rank,
        "pca_components_for_variance": {
            f"{int(p * 100)}%": int(np.searchsorted(cum, p) + 1)
            for p in (0.5, 0.8, 0.9, 0.95, 0.99)
        },
        "spearman_top_features": {
            "columns": top_cols,
            "matrix": np.round(np.asarray(corr), 4).tolist(),
        },
    }
