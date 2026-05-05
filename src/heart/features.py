"""Preprocessing pipeline for the heart-disease feature set.

The transformer is intentionally stateless to construct so the same factory
is used in training, evaluation and serving — guaranteeing identical
preprocessing in every environment.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL, NUMERIC


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer matching the modeling notebook.

    - Numeric columns: median impute + standard scaling
    - Categorical columns: most-frequent impute + one-hot encoding
      (``handle_unknown='ignore'`` so out-of-distribution serving values
      do not crash the prediction call)
    """
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC),
        ("cat", categorical_pipe, CATEGORICAL),
    ])
