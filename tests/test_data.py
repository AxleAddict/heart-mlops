"""Tests for ``heart.data``."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from heart.data import (
    CATEGORICAL,
    COLUMNS,
    FEATURES,
    NUMERIC,
    TARGET,
    load_processed,
    split_features_target,
)


def test_schema_constants_are_consistent():
    assert TARGET == "target"
    assert len(COLUMNS) == 14
    assert "num" in COLUMNS
    # FEATURES should be exactly the 13 modeling inputs (no num/target).
    assert set(FEATURES) == set(NUMERIC) | set(CATEGORICAL)
    assert "num" not in FEATURES and "target" not in FEATURES
    # No accidental overlap between numeric / categorical buckets.
    assert set(NUMERIC).isdisjoint(set(CATEGORICAL))


def test_load_processed_parses_question_marks_as_nan(processed_data_file):
    frame = load_processed(processed_data_file)
    assert list(frame.columns) == COLUMNS + [TARGET]
    # Two NaNs were injected in conftest (``ca`` row 0, ``thal`` row 1).
    assert frame["ca"].isna().sum() >= 1
    assert frame["thal"].isna().sum() >= 1


def test_load_processed_binarizes_target(processed_data_file):
    frame = load_processed(processed_data_file)
    assert set(frame[TARGET].unique()).issubset({0, 1})
    # target == 1 iff num > 0.
    assert ((frame["num"] > 0).astype(int) == frame[TARGET]).all()


def test_load_processed_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_processed(tmp_path / "does-not-exist.data")


def test_split_features_target_returns_canonical_columns(sample_frame):
    X, y = split_features_target(sample_frame)
    assert list(X.columns) == FEATURES
    assert len(X) == len(y) == len(sample_frame)
    assert y.name == TARGET
    # Should be a copy — mutating X must not touch the source frame.
    X.iloc[0, 0] = -999
    assert sample_frame.iloc[0][FEATURES[0]] != -999


def test_split_features_target_respects_custom_columns(sample_frame):
    cols = ["age", "chol", "thalach"]
    X, y = split_features_target(sample_frame, feature_cols=cols)
    assert list(X.columns) == cols
    assert isinstance(y, pd.Series)
