"""Tests for ``heart.predict``."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from heart import predict as predict_mod
from heart.predict import HeartModel, PredictionResult, get_model


def test_prediction_result_to_dict_roundtrip():
    r = PredictionResult(
        prediction=1, confidence=0.8,
        probability_disease=0.8, probability_no_disease=0.2,
    )
    d = r.to_dict()
    assert d == {
        "prediction": 1, "confidence": 0.8,
        "probability_disease": 0.8, "probability_no_disease": 0.2,
    }
    assert isinstance(d["prediction"], int)
    assert isinstance(d["confidence"], float)


def test_load_missing_bundle_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        HeartModel.load(tmp_path / "nope.pkl")


def test_load_bundle_populates_metadata(model_bundle_path):
    hm = HeartModel.load(model_bundle_path)
    assert hm.family == "logreg"
    assert hm.mlflow_run_id == "test-run-id"
    assert len(hm.feature_columns) == 13


def test_predict_single_record_returns_valid_probabilities(
    model_bundle_path, valid_record,
):
    hm = HeartModel.load(model_bundle_path)
    res = hm.predict(valid_record)
    assert isinstance(res, PredictionResult)
    assert res.prediction in (0, 1)
    assert 0.0 <= res.probability_disease <= 1.0
    assert 0.0 <= res.probability_no_disease <= 1.0
    assert math.isclose(
        res.probability_disease + res.probability_no_disease, 1.0, abs_tol=1e-6,
    )
    assert res.confidence == max(res.probability_disease, res.probability_no_disease)
    # 0.5 threshold contract.
    assert res.prediction == int(res.probability_disease >= 0.5)


def test_predict_batch_matches_single(model_bundle_path, valid_record):
    hm = HeartModel.load(model_bundle_path)
    other = dict(valid_record); other["age"] = 45.0; other["sex"] = 0
    batch = hm.predict_batch([valid_record, other])
    assert len(batch) == 2
    assert batch[0].to_dict() == hm.predict(valid_record).to_dict()
    assert batch[1].to_dict() == hm.predict(other).to_dict()


def test_predict_missing_feature_raises(model_bundle_path, valid_record):
    hm = HeartModel.load(model_bundle_path)
    bad = dict(valid_record); bad.pop("chol")
    with pytest.raises(ValueError, match="Missing required feature columns"):
        hm.predict(bad)


def test_get_model_is_cached(monkeypatch, model_bundle_path):
    # Ensure no stale entry from prior tests.
    get_model.cache_clear()
    monkeypatch.setattr(predict_mod, "DEFAULT_MODEL_PATH", model_bundle_path)
    a = get_model()
    b = get_model()
    assert a is b
    get_model.cache_clear()
