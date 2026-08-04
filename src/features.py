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

    # Composite advantage score — weighted combination
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

    # Gold per minute differential (already in data, but we add the difference)
    df["goldPerMinDiff"] = df["blueGoldPerMin"] - df["redGoldPerMin"]

    # CS per minute differential
    df["csPerMinDiff"] = df["blueCSPerMin"] - df["redCSPerMin"]

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
