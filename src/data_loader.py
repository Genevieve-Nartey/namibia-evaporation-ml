"""
data_loader.py
--------------
Load and validate the North Namibian monthly soil dataset.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = {"date", "d2m", "t2m", "mer", "mtdwswrf", "mtpr", "stl1", "swvl1"}
FEATURE_COLS = ["d2m", "t2m", "mtdwswrf", "mtpr", "stl1", "swvl1"]
TARGET_COL = "mer"


def load_data(path: str | Path) -> pd.DataFrame:
    """
    Load and validate the North Namibian soil CSV dataset.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Validated DataFrame with a parsed date column.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at the given path.
    ValueError
        If required columns are missing or the DataFrame is empty.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    logger.info("Loading dataset from %s", path)
    df = pd.read_csv(path, index_col=0)

    # Validate columns
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    if df.empty:
        raise ValueError("Loaded DataFrame is empty.")

    # Parse date
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    logger.info("Loaded %d rows, %d columns", *df.shape)
    _validate_data_quality(df)
    return df


def _validate_data_quality(df: pd.DataFrame) -> None:
    """Log data quality statistics and warn on anomalies."""
    n_missing = df.isnull().sum().sum()
    n_dupes = df.duplicated().sum()

    if n_missing > 0:
        logger.warning("Found %d missing values:\n%s", n_missing, df.isnull().sum())
    else:
        logger.info("No missing values found.")

    if n_dupes > 0:
        logger.warning("Found %d duplicate rows.", n_dupes)
    else:
        logger.info("No duplicate rows found.")

    logger.info(
        "Date range: %s to %s",
        df["date"].min().strftime("%Y-%m"),
        df["date"].max().strftime("%Y-%m"),
    )


def get_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split the DataFrame into feature matrix X and target vector y.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset as returned by load_data().

    Returns
    -------
    X : pd.DataFrame
        Feature columns (all predictors).
    y : pd.Series
        Target column (evaporation rate, 'mer').
    """
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    return X, y
