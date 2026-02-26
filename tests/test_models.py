"""
Tests for src/models.py
"""
import numpy as np
import pandas as pd
import pytest

from src.models import ModelResult, train_knn, train_linear_regression


@pytest.fixture
def synthetic_data():
    """
    Generate a perfectly linear dataset for sanity-check testing.
    y = 2*x1 + 3*x2 → a well-fitted linear model should get R² ≈ 1.
    """
    np.random.seed(0)
    n = 200
    X = pd.DataFrame(
        np.random.randn(n, 6),
        columns=["d2m", "t2m", "mtdwswrf", "mtpr", "stl1", "swvl1"],
    )
    y = pd.Series(2 * X["d2m"] + 3 * X["t2m"], name="mer")

    from src.preprocessing import scale_features, split_data
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_s, X_test_s, _ = scale_features(X_train, X_test)
    return X, y, X_train_s, X_test_s, y_train, y_test


def test_linear_regression_r2_near_one(synthetic_data):
    _, _, X_train, X_test, y_train, y_test = synthetic_data
    result = train_linear_regression(X_train, X_test, y_train, y_test, save_plot=False)
    assert result.r2 > 0.99


def test_knn_returns_model_result(synthetic_data):
    _, _, X_train, X_test, y_train, y_test = synthetic_data
    result = train_knn(X_train, X_test, y_train, y_test, n_neighbors=3, save_plot=False)
    assert isinstance(result, ModelResult)


def test_model_result_to_dict_keys(synthetic_data):
    _, _, X_train, X_test, y_train, y_test = synthetic_data
    result = train_linear_regression(X_train, X_test, y_train, y_test, save_plot=False)
    d = result.to_dict()
    assert set(d.keys()) == {"model", "mse", "rmse", "mae", "r2", "explained_variance"}


def test_knn_mse_non_negative(synthetic_data):
    _, _, X_train, X_test, y_train, y_test = synthetic_data
    result = train_knn(X_train, X_test, y_train, y_test, save_plot=False)
    assert result.mse >= 0
    assert result.rmse >= 0
