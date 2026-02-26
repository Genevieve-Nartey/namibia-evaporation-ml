"""
Tests for src/preprocessing.py
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import scale_features, split_data


@pytest.fixture
def sample_Xy():
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.randn(100, 6),
        columns=["d2m", "t2m", "mtdwswrf", "mtpr", "stl1", "swvl1"],
    )
    y = pd.Series(np.random.randn(100), name="mer")
    return X, y


def test_split_data_shapes(sample_Xy):
    X, y = sample_Xy
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20


def test_split_data_reproducible(sample_Xy):
    X, y = sample_Xy
    split1 = split_data(X, y, random_state=42)
    split2 = split_data(X, y, random_state=42)
    pd.testing.assert_frame_equal(split1[0], split2[0])


def test_standard_scaler_zero_mean(sample_Xy):
    X, y = sample_Xy
    X_train, X_test, _, _ = split_data(X, y)
    X_train_s, _, _ = scale_features(X_train, X_test, scaler_type="standard")
    # mean should be ~0 for training data
    np.testing.assert_allclose(X_train_s.mean(axis=0), 0, atol=1e-10)


def test_standard_scaler_unit_std(sample_Xy):
    X, y = sample_Xy
    X_train, X_test, _, _ = split_data(X, y)
    X_train_s, _, _ = scale_features(X_train, X_test, scaler_type="standard")
    np.testing.assert_allclose(X_train_s.std(axis=0), 1, atol=1e-10)


def test_minmax_scaler_range(sample_Xy):
    X, y = sample_Xy
    X_train, X_test, _, _ = split_data(X, y)
    X_train_s, _, _ = scale_features(X_train, X_test, scaler_type="minmax")
    assert X_train_s.min() >= 0.0 - 1e-10
    assert X_train_s.max() <= 1.0 + 1e-10


def test_invalid_scaler_raises(sample_Xy):
    X, y = sample_Xy
    X_train, X_test, _, _ = split_data(X, y)
    with pytest.raises(ValueError, match="Unknown scaler"):
        scale_features(X_train, X_test, scaler_type="bogus")


def test_no_test_leakage(sample_Xy):
    """Scaler fit on train should NOT use test data — test mean may differ from 0."""
    X, y = sample_Xy
    X_train, X_test, _, _ = split_data(X, y)
    _, X_test_s, _ = scale_features(X_train, X_test, scaler_type="standard")
    # Test set mean after transform won't be exactly 0 (that would imply leakage)
    assert not np.allclose(X_test_s.mean(axis=0), 0, atol=1e-10)
