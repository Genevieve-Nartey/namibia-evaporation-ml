"""
preprocessing.py
----------------
Data scaling and train/test splitting utilities.
Supports StandardScaler, MinMaxScaler, and RobustScaler.
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

logger = logging.getLogger(__name__)

ScalerType = Literal["standard", "minmax", "robust"]
_SCALER_MAP = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split features and target into train/test sets.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    test_size : float
        Proportion of data for the test set (default 0.2).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(
        "Train/test split: %d train, %d test (test_size=%.0f%%)",
        len(X_train),
        len(X_test),
        test_size * 100,
    )
    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler_type: ScalerType = "standard",
) -> tuple[np.ndarray, np.ndarray, StandardScaler | MinMaxScaler | RobustScaler]:
    """
    Fit a scaler on X_train and transform both X_train and X_test.

    The scaler is fitted *only* on training data to prevent data leakage.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    X_test : pd.DataFrame
        Test feature matrix.
    scaler_type : {'standard', 'minmax', 'robust'}
        Which scaler to use.
        - 'standard' : zero mean, unit variance (best for normally distributed features)
        - 'minmax'   : scale to [0, 1] range
        - 'robust'   : uses IQR; best when outliers are present

    Returns
    -------
    X_train_scaled : np.ndarray
    X_test_scaled  : np.ndarray
    scaler         : fitted scaler object (for later use in prediction)

    Raises
    ------
    ValueError
        If scaler_type is not one of the supported values.
    """
    if scaler_type not in _SCALER_MAP:
        raise ValueError(
            f"Unknown scaler '{scaler_type}'. Choose from: {list(_SCALER_MAP.keys())}"
        )

    scaler_cls = _SCALER_MAP[scaler_type]
    scaler = scaler_cls()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.info(
        "Applied %s. Train mean: %s | Train std: %s",
        scaler_cls.__name__,
        np.round(X_train_scaled.mean(axis=0), 4),
        np.round(X_train_scaled.std(axis=0), 4),
    )
    return X_train_scaled, X_test_scaled, scaler
