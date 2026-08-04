"""Data loading and basic cleaning for the LoL 10-minute dataset.

This module is responsible for locating the raw dataset, validating its
schema, dropping identifier and perfectly-redundant mirrored columns, and
returning a feature matrix :math:`X` together with the target vector :math:`y`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_DATA_PATH: Path = Path("data") / "high_diamond_ranked_10min.csv"
TARGET_COLUMN: str = "blueWins"

REDUNDANT_COLUMNS: tuple[str, ...] = (
    "gameId",
    "redGoldDiff",
    "redExperienceDiff",
    "redFirstBlood",
)

REQUIRED_COLUMNS: tuple[str, ...] = (
    "blueWardsPlaced",
    "blueWardsDestroyed",
    "blueFirstBlood",
    "blueKills",
    "blueDeaths",
    "blueAssists",
    "blueEliteMonsters",
    "blueDragons",
    "blueHeralds",
    "blueTowersDestroyed",
    "blueTotalGold",
    "blueAvgLevel",
    "blueTotalExperience",
    "blueTotalMinionsKilled",
    "blueTotalJungleMinionsKilled",
    "blueGoldDiff",
    "blueExperienceDiff",
    "blueCSPerMin",
    "blueGoldPerMin",
    "redWardsPlaced",
    "redWardsDestroyed",
    "redKills",
    "redDeaths",
    "redAssists",
    "redEliteMonsters",
    "redDragons",
    "redHeralds",
    "redTowersDestroyed",
    "redTotalGold",
    "redAvgLevel",
    "redTotalExperience",
    "redTotalMinionsKilled",
    "redTotalJungleMinionsKilled",
    "redCSPerMin",
    "redGoldPerMin",
)


class DataError(Exception):
    """Raised when the raw dataset cannot be loaded or cleaned."""


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw CSV dataset into a pandas DataFrame.

    Args:
        path: Path to the raw CSV file. Defaults to the packaged location
            ``data/high_diamond_ranked_10min.csv`` relative to the repo root.

    Returns:
        A pandas DataFrame containing the raw match records.

    Raises:
        DataError: If the file does not exist or cannot be parsed.
    """
    csv_path = Path(path) if path is not None else DEFAULT_DATA_PATH
    if not csv_path.exists():
        raise DataError(
            f"Dataset not found at {csv_path}. "
            "Run `kaggle datasets download -d "
            "bobbyscience/league-of-legends-diamond-ranked-games-10-min "
            "-p data --unzip` first."
        )
    try:
        frame = pd.read_csv(csv_path)
    except Exception as exc:
        raise DataError(f"Failed to parse CSV at {csv_path}: {exc}") from exc
    return frame


def validate_schema(frame: pd.DataFrame) -> None:
    """Verify that all expected feature and target columns are present.

    Args:
        frame: The raw DataFrame.

    Raises:
        DataError: If any required column is missing from the schema.
    """
    expected = (*REQUIRED_COLUMNS, TARGET_COLUMN)
    missing = [col for col in expected if col not in frame.columns]
    if missing:
        raise DataError(
            f"Dataset schema mismatch. Missing columns: {missing}"
        )


def clean_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove identifiers, perfectly redundant mirrored columns, and NaNs.

    The columns ``redGoldDiff``/``redExperienceDiff`` are exact negations of
    their blue-side counterparts and ``redFirstBlood`` is the boolean
    complement of ``blueFirstBlood``. Dropping them removes perfect
    collinearity without losing information.

    Args:
        frame: The raw DataFrame.

    Returns:
        A cleaned DataFrame containing only predictive features.
    """
    validate_schema(frame)
    cleaned = frame.drop(columns=list(REDUNDANT_COLUMNS), errors="ignore")
    duplicated_columns = cleaned.columns[cleaned.columns.duplicated()].tolist()
    if duplicated_columns:
        raise DataError(
            f"Unexpected duplicate columns after cleaning: {duplicated_columns}"
        )
    return cleaned.dropna().reset_index(drop=True)


def load_dataset(
    path: str | Path | None = None,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """End-to-end loading and cleaning entry point.

    Args:
        path: Optional path to the raw CSV file.
        target_column: Name of the binary target column.

    Returns:
        A tuple ``(X, y)`` of the feature matrix and target vector.
    """
    raw = load_data(path)
    cleaned = clean_data(raw)
    y: pd.Series = cleaned[target_column].astype(int)
    X: pd.DataFrame = cleaned.drop(columns=[target_column])
    return X, y
