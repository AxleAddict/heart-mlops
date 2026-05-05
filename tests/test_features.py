"""Tests for ``heart.features.build_preprocessor``."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from heart.data import CATEGORICAL, FEATURES, NUMERIC
from heart.features import build_preprocessor


def test_build_preprocessor_returns_column_transformer():
    pre = build_preprocessor()
    assert isinstance(pre, ColumnTransformer)
    names = [name for name, _, _ in pre.transformers]
    assert names == ["num", "cat"]


def test_preprocessor_fits_and_transforms_dense_array(sample_frame):
    pre = build_preprocessor()
    X = sample_frame[FEATURES]
    out = pre.fit_transform(X)
    # Output must be dense (sparse_output=False) and free of NaNs after impute.
    assert isinstance(out, np.ndarray)
    assert not np.isnan(out).any()
    # Numeric columns are scaled in place (1 column each); categoricals are
    # one-hot expanded so total width >= len(FEATURES).
    assert out.shape[0] == len(X)
    assert out.shape[1] >= len(NUMERIC) + len(CATEGORICAL)


def test_preprocessor_handles_unknown_categorical_at_transform(sample_frame):
    pre = build_preprocessor()
    pre.fit(sample_frame[FEATURES])
    # Inject an out-of-distribution category into ``cp``; must not raise
    # because the OneHotEncoder uses handle_unknown="ignore".
    novel = sample_frame[FEATURES].head(1).copy()
    novel.loc[:, "cp"] = 99
    out = pre.transform(novel)
    assert out.shape[0] == 1
    assert not np.isnan(out).any()


def test_preprocessor_imputes_missing_values():
    pre = build_preprocessor()
    # Two rows so median/most_frequent are well-defined.
    base = {
        "age": [50.0, 60.0], "trestbps": [120.0, 130.0],
        "chol": [200.0, 220.0], "thalach": [150.0, 160.0],
        "oldpeak": [1.0, 1.5], "ca": [0.0, 1.0],
        "sex": [1, 0], "cp": [1, 2], "fbs": [0, 1],
        "restecg": [0, 1], "exang": [0, 1], "slope": [1, 2], "thal": [3.0, 6.0],
    }
    train = pd.DataFrame(base)[FEATURES]
    pre.fit(train)
    bad = train.head(1).copy()
    bad.loc[:, "ca"] = np.nan
    bad.loc[:, "thal"] = np.nan
    out = pre.transform(bad)
    assert not np.isnan(out).any()
