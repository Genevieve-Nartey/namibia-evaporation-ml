"""
Tests for src/data_loader.py
"""
import io
from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import FEATURE_COLS, TARGET_COL, get_features_and_target, load_data

SAMPLE_CSV = """,date,d2m,t2m,mer,mtdwswrf,mtpr,stl1,swvl1
0,1959-01-01,281.45,298.14,-2.018e-05,482.14,2.157e-05,302.49,0.115
1,1959-02-01,284.85,295.80,-3.196e-05,461.00,3.969e-05,298.59,0.185
2,1959-03-01,281.99,298.03,-1.377e-05,418.51,5.638e-06,301.81,0.080
"""


@pytest.fixture
def tmp_csv(tmp_path: Path) -> Path:
    """Write sample CSV to a temp file and return the path."""
    p = tmp_path / "test_soil.csv"
    p.write_text(SAMPLE_CSV)
    return p


def test_load_data_returns_dataframe(tmp_csv):
    df = load_data(tmp_csv)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_load_data_has_expected_columns(tmp_csv):
    df = load_data(tmp_csv)
    expected = {"date", "d2m", "t2m", "mer", "mtdwswrf", "mtpr", "stl1", "swvl1"}
    assert expected.issubset(set(df.columns))


def test_load_data_date_is_datetime(tmp_csv):
    df = load_data(tmp_csv)
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


def test_load_data_sorted_by_date(tmp_csv):
    df = load_data(tmp_csv)
    assert df["date"].is_monotonic_increasing


def test_load_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_data("nonexistent/path.csv")


def test_load_data_missing_column(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text(",date,d2m\n0,1959-01-01,281.45\n")
    with pytest.raises(ValueError, match="Missing expected columns"):
        load_data(p)


def test_get_features_and_target(tmp_csv):
    df = load_data(tmp_csv)
    X, y = get_features_and_target(df)
    assert list(X.columns) == FEATURE_COLS
    assert y.name == TARGET_COL
    assert len(X) == len(y)
