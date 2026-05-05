"""Tests for ``heart.api`` using FastAPI's TestClient."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from heart import api as api_mod
from heart.predict import HeartModel, get_model


@pytest.fixture()
def client_with_model(model_bundle_path):
    """TestClient with the singleton model overridden to our test bundle."""
    get_model.cache_clear()
    model = HeartModel.load(model_bundle_path)
    api_mod.app.state.model = model
    with TestClient(api_mod.app) as c:
        # The lifespan handler runs ``get_model()`` which would look for the
        # production path — overwrite again afterwards to be safe.
        api_mod.app.state.model = model
        yield c
    get_model.cache_clear()


@pytest.fixture()
def client_without_model():
    get_model.cache_clear()
    with TestClient(api_mod.app) as c:
        api_mod.app.state.model = None
        yield c
    get_model.cache_clear()


def test_health_always_ok(client_without_model):
    resp = client_without_model.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ready_503_when_model_missing(client_without_model):
    resp = client_without_model.get("/ready")
    assert resp.status_code == 503


def test_ready_reports_model_metadata(client_with_model):
    resp = client_with_model.get("/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    assert body["model_family"] == "logreg"
    assert body["mlflow_run_id"] == "test-run-id"


def test_predict_returns_well_formed_response(client_with_model, valid_record):
    resp = client_with_model.post("/predict", json=valid_record)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability_disease"] <= 1.0
    assert 0.0 <= body["probability_no_disease"] <= 1.0
    assert body["confidence"] == max(
        body["probability_disease"], body["probability_no_disease"],
    )


def test_predict_validation_error_on_out_of_range(client_with_model, valid_record):
    bad = dict(valid_record); bad["sex"] = 5  # ge=0, le=1 → 422
    resp = client_with_model.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_503_when_model_missing(client_without_model, valid_record):
    resp = client_without_model.post("/predict", json=valid_record)
    assert resp.status_code == 503


def test_predict_batch_returns_count(client_with_model, valid_record):
    payload = [valid_record, {**valid_record, "age": 45.0, "sex": 0}]
    resp = client_with_model.post("/predict/batch", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2
    for p in body["predictions"]:
        assert p["prediction"] in (0, 1)


def test_predict_batch_rejects_empty_list(client_with_model):
    resp = client_with_model.post("/predict/batch", json=[])
    assert resp.status_code == 422


def test_metrics_endpoint_exposed(client_with_model, valid_record):
    # Trigger one prediction so our custom counters are non-zero.
    client_with_model.post("/predict", json=valid_record)
    resp = client_with_model.get("/metrics")
    assert resp.status_code == 200
    text = resp.text
    assert "heart_predictions_total" in text
    assert "heart_prediction_latency_seconds" in text
