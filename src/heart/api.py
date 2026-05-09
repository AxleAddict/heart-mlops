"""FastAPI serving layer for the heart-disease classifier.

Endpoints
---------
- ``GET  /health``      – liveness probe (no model required)
- ``GET  /ready``       – readiness probe (model loaded)
- ``GET  /metrics``     – Prometheus exposition (HTTP latency, request counts,
                          plus our custom prediction counters/histograms)
- ``POST /predict``     – single-record prediction → ``{prediction, confidence}``
- ``POST /predict/batch`` – batched prediction

Logging
-------
Every request emits a structured JSON log line via ``uvicorn``-friendly
``logging``; combined with the Prometheus ``/metrics`` endpoint this is what
Grafana / Cloud Logging will scrape in the deployed environment.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from .predict import HeartModel, PredictionResult, get_model

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
)
log = logging.getLogger("heart.api")


# --------------------------------------------------------------------------- #
# Request / response schemas (mirror the 13 input features in data.py)
# --------------------------------------------------------------------------- #
class HeartRecord(BaseModel):
    """One patient record matching the UCI 13-feature schema."""
    age: float = Field(..., ge=0, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1=male, 0=female")
    cp: int = Field(..., ge=1, le=4, description="Chest pain type 1-4")
    trestbps: float = Field(..., ge=0, description="Resting BP (mm Hg)")
    chol: float = Field(..., ge=0, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG 0-2")
    thalach: float = Field(..., ge=0, description="Max heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina")
    oldpeak: float = Field(..., description="ST depression (exercise vs rest)")
    slope: int = Field(..., ge=1, le=3, description="ST slope 1-3")
    ca: float = Field(..., ge=0, le=3, description="Major vessels (0-3)")
    thal: float = Field(..., description="3=normal, 6=fixed, 7=reversible")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233,
                "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0,
                "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6,
            }
        }
    }


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0=no disease, 1=disease")
    confidence: float = Field(..., ge=0.0, le=1.0)
    probability_disease: float = Field(..., ge=0.0, le=1.0)
    probability_no_disease: float = Field(..., ge=0.0, le=1.0)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int


# --------------------------------------------------------------------------- #
# Custom Prometheus metrics (prometheus_fastapi_instrumentator already
# exposes http_request_duration_seconds, http_requests_total, etc.)
# --------------------------------------------------------------------------- #
PREDICTION_COUNTER = Counter(
    "heart_predictions_total",
    "Number of /predict calls served, labeled by prediction class.",
    ["prediction"],
)
PREDICTION_LATENCY = Histogram(
    "heart_prediction_latency_seconds",
    "Latency of model.predict() (excludes HTTP overhead).",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
PREDICTION_ERRORS = Counter(
    "heart_prediction_errors_total",
    "Number of /predict calls that raised an error.",
    ["error_type"],
)


# --------------------------------------------------------------------------- #
# Lifespan: load the model once at startup so the first request isn't cold.
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.model = get_model()
        log.info(
            "Loaded model family=%s run_id=%s features=%d",
            app.state.model.family,
            app.state.model.mlflow_run_id,
            len(app.state.model.feature_columns),
        )
    except FileNotFoundError as exc:
        # Allow the app to start so /health works in CI even if the artifact
        # has not been built yet; /predict will return 503.
        log.error("Model not loaded at startup: %s", exc)
        app.state.model = None

    # Pre-initialise error labels so the metric appears in Prometheus at 0
    # even before the first error occurs.
    for label in ("value_error", "validation_error", "runtime_error"):
        PREDICTION_ERRORS.labels(error_type=label)

    yield


app = FastAPI(
    title="Heart Disease Classifier",
    description="UCI Heart Disease binary classifier — MLOps Assignment-I.",
    version="1.0.0",
    lifespan=lifespan,
)

# Exposes /metrics with default + per-route HTTP metrics.
Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


# --------------------------------------------------------------------------- #
# Request access log middleware (structured)
# --------------------------------------------------------------------------- #
@app.middleware("http")
async def access_log(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    log.info(
        "request method=%s path=%s status=%d duration_ms=%.2f client=%s",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
        request.client.host if request.client else "-",
    )
    return response


def _require_model(request: Request) -> HeartModel:
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/health", tags=["ops"])
def health() -> dict:
    return {"status": "ok"}


@app.get("/ready", tags=["ops"])
def ready(request: Request) -> dict:
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "ready",
        "model_family": model.family,
        "mlflow_run_id": model.mlflow_run_id,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(record: HeartRecord, request: Request) -> PredictionResponse:
    model = _require_model(request)
    try:
        with PREDICTION_LATENCY.time():
            result: PredictionResult = model.predict(record.model_dump())
    except ValueError as exc:
        PREDICTION_ERRORS.labels(error_type="value_error").inc()
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        PREDICTION_ERRORS.labels(error_type=type(exc).__name__).inc()
        log.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed")

    PREDICTION_COUNTER.labels(prediction=str(result.prediction)).inc()
    log.info(
        "prediction class=%d confidence=%.4f p_disease=%.4f",
        result.prediction, result.confidence, result.probability_disease,
    )
    return PredictionResponse(**result.to_dict())


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["inference"])
def predict_batch(
    records: Annotated[list[HeartRecord], Field(min_length=1, max_length=1000)],
    request: Request,
) -> BatchPredictionResponse:
    model = _require_model(request)
    try:
        with PREDICTION_LATENCY.time():
            results = model.predict_batch([r.model_dump() for r in records])
    except ValueError as exc:
        PREDICTION_ERRORS.labels(error_type="value_error").inc()
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        PREDICTION_ERRORS.labels(error_type=type(exc).__name__).inc()
        log.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed")

    for r in results:
        PREDICTION_COUNTER.labels(prediction=str(r.prediction)).inc()
    log.info("batch_prediction count=%d", len(results))
    return BatchPredictionResponse(
        predictions=[PredictionResponse(**r.to_dict()) for r in results],
        count=len(results),
    )


@app.exception_handler(RequestValidationError)
async def _validation_error(request: Request, exc: RequestValidationError):
    """Count Pydantic/FastAPI 422 validation errors in our error metric."""
    if request.url.path.startswith("/predict"):
        PREDICTION_ERRORS.labels(error_type="validation_error").inc()
        log.warning("Validation error on %s: %s", request.url.path, exc)
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    log.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
