"""Shared fixtures for the heart-disease unit tests.

Builds a tiny in-memory dataset matching the canonical 14-column UCI schema
and a trained sklearn pipeline persisted as the same joblib bundle that
``heart.train.main`` produces in production.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from heart.data import COLUMNS, FEATURES
from heart.features import build_preprocessor


def _make_row(seed: int, label: int) -> dict:
    rng = np.random.default_rng(seed)
    # Pull values from valid ranges so pydantic schemas accept them too.
    return {
        "age": float(rng.integers(30, 75)),
        "sex": int(rng.integers(0, 2)),
        "cp": int(rng.integers(1, 5)),
        "trestbps": float(rng.integers(90, 180)),
        "chol": float(rng.integers(150, 350)),
        "fbs": int(rng.integers(0, 2)),
        "restecg": int(rng.integers(0, 3)),
        "thalach": float(rng.integers(90, 200)),
        "exang": int(rng.integers(0, 2)),
        "oldpeak": float(round(rng.uniform(0.0, 4.0), 1)),
        "slope": int(rng.integers(1, 4)),
        "ca": float(rng.integers(0, 4)),
        "thal": float(rng.choice([3.0, 6.0, 7.0])),
        # ``num`` is the raw 0-4 severity column; binary target derives from it.
        "num": int(label * rng.integers(1, 5)),
    }


@pytest.fixture(scope="session")
def sample_frame() -> pd.DataFrame:
    """A small but class-balanced frame with the canonical schema."""
    rows = [_make_row(seed=i, label=i % 2) for i in range(60)]
    frame = pd.DataFrame(rows, columns=COLUMNS)
    # Inject a couple of missing values to exercise the imputers.
    frame.loc[0, "ca"] = np.nan
    frame.loc[1, "thal"] = np.nan
    frame["target"] = (frame["num"] > 0).astype(int)
    return frame


@pytest.fixture(scope="session")
def processed_data_file(tmp_path_factory, sample_frame) -> Path:
    """Write the sample frame in the raw ``processed.<site>.data`` format."""
    out = tmp_path_factory.mktemp("data") / "processed.fake.data"
    raw = sample_frame[COLUMNS].copy()
    # The loader expects ``?`` for missing values, no header row.
    raw_str = raw.astype(object).where(raw.notna(), "?")
    raw_str.to_csv(out, header=False, index=False)
    return out


@pytest.fixture(scope="session")
def trained_pipeline(sample_frame) -> Pipeline:
    """Fit a tiny logistic-regression pipeline on the synthetic data."""
    X = sample_frame[FEATURES].copy()
    y = sample_frame["target"].copy()
    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", LogisticRegression(max_iter=500, solver="liblinear")),
    ])
    pipe.fit(X, y)
    return pipe


@pytest.fixture()
def model_bundle_path(tmp_path, trained_pipeline) -> Path:
    """Persist the pipeline as the joblib bundle ``HeartModel.load`` expects."""
    out = tmp_path / "model.pkl"
    joblib.dump(
        {
            "model": trained_pipeline,
            "feature_columns": FEATURES,
            "family": "logreg",
            "mlflow_run_id": "test-run-id",
            "params": {"clf__C": 1.0},
            "metrics": {"test_roc_auc": 0.9},
        },
        out,
    )
    return out


@pytest.fixture()
def valid_record() -> dict:
    """A canonical valid input record for /predict."""
    return {
        "age": 63.0, "sex": 1, "cp": 1, "trestbps": 145.0, "chol": 233.0,
        "fbs": 1, "restecg": 2, "thalach": 150.0, "exang": 0,
        "oldpeak": 2.3, "slope": 3, "ca": 0.0, "thal": 6.0,
    }
