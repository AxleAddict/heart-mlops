"""Canonical loader for the UCI Heart Disease processed datasets.

The 14-column schema is shared across Cleveland, Hungarian, Switzerland and
Long-Beach VA sites. Missing values are encoded as ``?`` in the source files
and converted to ``NaN`` here so the sklearn pipeline can impute them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

COLUMNS: list[str] = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

CATEGORICAL: list[str] = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
NUMERIC: list[str] = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
FEATURES: list[str] = NUMERIC + CATEGORICAL
TARGET: str = "target"


def load_processed(path: str | Path) -> pd.DataFrame:
    """Load any ``processed.<site>.data`` file with the canonical schema.

    Adds a binarized ``target`` column (``num > 0``) following the standard
    Cleveland convention used in the published baselines.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Heart-disease file not found: {path}")
    frame = pd.read_csv(path, header=None, names=COLUMNS, na_values="?")
    frame[TARGET] = (frame["num"] > 0).astype(int)
    return frame


def split_features_target(
    frame: pd.DataFrame,
    feature_cols: Iterable[str] = FEATURES,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(X, y)`` with the canonical 13 features and the binary target."""
    cols = list(feature_cols)
    return frame[cols].copy(), frame[TARGET].copy()
