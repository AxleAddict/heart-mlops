"""Inference helpers used by both the API and ad-hoc scripts.

The training entrypoint persists a bundle::

    {"model": Pipeline, "feature_columns": [...], "family": "...", ...}

so the serving layer is decoupled from the exact estimator family chosen
during training.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from .data import FEATURES

DEFAULT_MODEL_PATH = Path(os.environ.get("HEART_MODEL_PATH", "models/model.pkl"))


@dataclass(frozen=True)
class PredictionResult:
    prediction: int
    confidence: float
    probability_disease: float
    probability_no_disease: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction": int(self.prediction),
            "confidence": float(self.confidence),
            "probability_disease": float(self.probability_disease),
            "probability_no_disease": float(self.probability_no_disease),
        }


@dataclass
class HeartModel:
    """Thin wrapper around the trained sklearn pipeline bundle."""

    model: Any
    feature_columns: Sequence[str]
    family: str
    mlflow_run_id: str | None = None

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL_PATH) -> "HeartModel":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model bundle not found at {path}. "
                "Run `python -m heart.train ...` first or set HEART_MODEL_PATH."
            )
        bundle = joblib.load(path)
        return cls(
            model=bundle["model"],
            feature_columns=bundle.get("feature_columns", FEATURES),
            family=bundle.get("family", "unknown"),
            mlflow_run_id=bundle.get("mlflow_run_id"),
        )

    def _to_frame(self, payload: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> pd.DataFrame:
        rows = [payload] if isinstance(payload, Mapping) else list(payload)
        frame = pd.DataFrame(rows)
        missing = [c for c in self.feature_columns if c not in frame.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")
        return frame[list(self.feature_columns)]

    def predict(self, payload: Mapping[str, Any]) -> PredictionResult:
        frame = self._to_frame(payload)
        proba = self.model.predict_proba(frame)[0]
        proba_no, proba_yes = float(proba[0]), float(proba[1])
        pred = int(proba_yes >= 0.5)
        return PredictionResult(
            prediction=pred,
            confidence=max(proba_no, proba_yes),
            probability_disease=proba_yes,
            probability_no_disease=proba_no,
        )

    def predict_batch(self, payload: Sequence[Mapping[str, Any]]) -> list[PredictionResult]:
        frame = self._to_frame(payload)
        probas = self.model.predict_proba(frame)
        results: list[PredictionResult] = []
        for row in probas:
            p_no, p_yes = float(row[0]), float(row[1])
            results.append(PredictionResult(
                prediction=int(p_yes >= 0.5),
                confidence=max(p_no, p_yes),
                probability_disease=p_yes,
                probability_no_disease=p_no,
            ))
        return results


@lru_cache(maxsize=1)
def get_model(path: str | None = None) -> HeartModel:
    """Process-wide singleton so the API does not reload the pickle per request."""
    return HeartModel.load(Path(path) if path else DEFAULT_MODEL_PATH)
