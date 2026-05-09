# Heart Disease Classifier — ML API

A production-grade MLOps pipeline for the **UCI Heart Disease** binary classification task. The project trains a scikit-learn model with full MLflow experiment tracking, serves it as a FastAPI REST API with Prometheus observability, and ships it through a Jenkins CI/CD pipeline to a Kubernetes cluster.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Local Setup](#local-setup)
  - [1. Create virtual environment](#1-create-virtual-environment)
  - [2. Install dependencies](#2-install-dependencies)
- [Training the Model](#training-the-model)
  - [CLI reference](#cli-reference)
  - [With a remote MLflow server](#with-a-remote-mlflow-server)
  - [What gets logged to MLflow](#what-gets-logged-to-mlflow)
- [Running the Serving API](#running-the-serving-api)
  - [Environment variables](#environment-variables)
  - [Start the server](#start-the-server)
  - [API Endpoints](#api-endpoints)
  - [Example request](#example-request)
- [Running Tests](#running-tests)
- [Docker](#docker)
  - [Build the image](#build-the-image)
  - [Run locally with Docker](#run-locally-with-docker)
  - [Run with a custom model path](#run-with-a-custom-model-path)
- [CI/CD Pipeline](#cicd-pipeline)
  - [CI pipeline (Jenkinsfile)](#ci-pipeline-jenkinsfile)
  - [Prod pipeline (Jenkinsfile.prod)](#prod-pipeline-jenkinsfileprod)
- [Kubernetes Manifests](#kubernetes-manifests)
- [Observability](#observability)
  - [Prometheus metrics](#prometheus-metrics)
  - [Grafana dashboard](#grafana-dashboard)

---

## Project Overview

This project implements the full MLOps lifecycle:

| Stage | Tooling |
|---|---|
| Data processing & EDA | pandas, matplotlib, seaborn |
| Feature engineering | scikit-learn `ColumnTransformer` |
| Model training & tracking | scikit-learn, MLflow |
| Model serving | FastAPI + uvicorn |
| Containerisation | Docker (multi-stage) |
| CI/CD | Jenkins |
| Orchestration | Kubernetes (GKE) |
| Monitoring | Prometheus + Grafana |

---

## Dataset

The model is trained on the **UCI Heart Disease** dataset:

- **Primary training set**: `data/processed.cleveland.data` — 303 patients from Cleveland Clinic
- **External validation set**: `data/processed.hungarian.data` — 294 patients from Budapest (used for generalisation testing and model selection)

**Target**: binary label where `0 = no disease` and `1 = disease present` (original target values 1–4 are mapped to 1).

**13 input features**:

| Feature | Description | Type |
|---|---|---|
| `age` | Age in years | Numeric |
| `sex` | 1 = male, 0 = female | Categorical |
| `cp` | Chest pain type (1–4) | Categorical |
| `trestbps` | Resting blood pressure (mm Hg) | Numeric |
| `chol` | Serum cholesterol (mg/dl) | Numeric |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true) | Categorical |
| `restecg` | Resting ECG results (0–2) | Categorical |
| `thalach` | Max heart rate achieved | Numeric |
| `exang` | Exercise-induced angina (1 = yes) | Categorical |
| `oldpeak` | ST depression (exercise vs rest) | Numeric |
| `slope` | Slope of peak exercise ST segment (1–3) | Categorical |
| `ca` | Number of major vessels coloured by fluoroscopy (0–3) | Numeric |
| `thal` | Thalassemia (3 = normal, 6 = fixed defect, 7 = reversable defect) | Categorical |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        CI/CD (Jenkins)                  │
│                                                         │
│  Git push → Train → Unit tests → Docker build/push      │
│           → Deploy to Non-Prod                          │
│           → (manual gate) → Deploy to Prod              │
└──────────────────────────┬──────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │   Kubernetes Cluster    │
              │                         │
              │  ┌──────────────────┐   │
              │  │  FastAPI  :8000  │   │
              │  │  /predict        │   │
              │  │  /metrics  ──────┼───┼──► Prometheus ──► Grafana
              │  └──────────────────┘   │
              └─────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │  MLflow Tracking Server │
              │  Logs: params, metrics, │
              │  artefacts, model       │
              └─────────────────────────┘
```

**Model selection strategy**: two families are trained (Logistic Regression and Random Forest) using 5-fold stratified cross-validation with `GridSearchCV`. The winner is chosen by **external validation ROC-AUC** (Hungarian dataset), which guards against overfitting to the Cleveland split.

---

## Project Structure

```
project/
├── data/
│   ├── processed.cleveland.data    # Primary training data
│   └── processed.hungarian.data   # External validation data
│
├── src/
│   └── heart/                     # Python package
│       ├── __init__.py
│       ├── data.py                # Data loading, feature constants, schema
│       ├── features.py            # ColumnTransformer preprocessor factory
│       ├── train.py               # MLflow-tracked training entrypoint (CLI)
│       ├── predict.py             # HeartModel class + get_model() singleton
│       └── api.py                 # FastAPI application
│
├── tests/                         # pytest test suite
│   └── ...
│
├── docker/
│   └── Dockerfile                 # Multi-stage build (builder + runtime)
│
├── k8s/
│   ├── deployment-nonprod.yaml    # Non-prod Deployment + Service
│   ├── deployment-blue.yaml       # Prod blue slot Deployment
│   ├── deployment-green.yaml      # Prod green slot Deployment
│   ├── service-prod.yaml          # Prod Service (selector toggles blue/green)
│   ├── monitoring.yaml            # Prometheus ServiceMonitor
│   └── grafana-dashboard.yaml     # Grafana dashboard ConfigMap
│
├── notebooks/                     # Exploratory analysis notebooks
├── models/                        # Saved model bundle (git-ignored)
├── artifacts/                     # Training plots/reports (git-ignored)
│
├── Jenkinsfile                    # CI pipeline (auto-triggers on push)
├── Jenkinsfile.prod               # Prod blue/green deploy pipeline (manual)
├── requirements.txt
└── pytest.ini
```

### Key module responsibilities

| Module | Responsibility |
|---|---|
| `data.py` | `load_processed()` reads raw UCI files, maps target to binary, drops rows with missing values. Defines `FEATURES` list and `CATEGORICAL_FEATURES` / `NUMERIC_FEATURES` constants. |
| `features.py` | `build_preprocessor()` returns a `ColumnTransformer` that one-hot encodes categorical features and standard-scales numeric features. This preprocessor is the first step of every sklearn `Pipeline`. |
| `train.py` | CLI entrypoint. Builds pipelines, runs `GridSearchCV`, logs everything to MLflow (params, metrics, artefacts, plots), selects the winner, and writes `models/model.pkl`. |
| `predict.py` | `HeartModel` wraps the saved bundle. `get_model()` is an `@lru_cache` singleton so the pickle is loaded once per process, not per request. |
| `api.py` | FastAPI app. Validates input with Pydantic, calls `get_model().predict()`, increments Prometheus counters, and emits structured JSON access logs. |

---

## Prerequisites

- Python 3.11+
- `pip`
- Docker (for containerised runs)
- MLflow (only needed if you want the UI — the training script falls back to local file storage)

---

## Local Setup

### 1. Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Training the Model

### CLI reference

```bash
PYTHONPATH=src python -m heart.train \
  --data        data/processed.cleveland.data \
  --val-data    data/processed.hungarian.data \
  --tracking-uri file:./mlruns \
  --experiment  heart-disease-modeling \
  --artifacts-dir artifacts \
  --output-model  models/model.pkl
```

All flags have defaults — the only required arguments are `--data` and `--val-data`.

| Flag | Default | Description |
|---|---|---|
| `--data` | *(required)* | Path to Cleveland training data |
| `--val-data` | *(required)* | Path to Hungarian validation data |
| `--tracking-uri` | `file:./mlruns` | MLflow tracking URI |
| `--experiment` | `heart-disease-modeling` | MLflow experiment name |
| `--artifacts-dir` | `artifacts/` | Local output dir for plots and CSV |
| `--output-model` | `models/model.pkl` | Where to write the winning model bundle |
| `--test-size` | `0.20` | Fraction of Cleveland data held out for testing |
| `--random-state` | `42` | Random seed for reproducibility |

After training completes, the local MLflow UI can be launched with:

```bash
mlflow ui --backend-store-uri file:./mlruns
# Open http://localhost:5000
```

### With a remote MLflow server

Set the environment variable instead of using the flag — it takes precedence:

```bash
export MLFLOW_TRACKING_URI=http://<mlflow-server-host>:5000
export MLFLOW_EXPERIMENT_NAME=heart-disease-ci
PYTHONPATH=src python -m heart.train \
  --data data/processed.cleveland.data \
  --val-data data/processed.hungarian.data
```

### What gets logged to MLflow

Training creates a **nested run hierarchy** per model family:

```
Experiment: heart-disease-modeling
└── Run: logreg                          (parent — family-level summary)
    ├── Run: logreg_C=0.01_penalty=l1   (one child per hyperparameter combo)
    ├── Run: logreg_C=0.1_penalty=l2
    ├── ...
    └── Run: logreg_best                 (best hyperparams, full evaluation)
        ├── Params: best_clf__C, best_clf__penalty
        ├── Metrics: cv_*, test_*, val_*, gap_*
        └── Artefacts: model/, plots/, reports/, cv/
└── Run: randomforest                    (same structure)
    └── ...
```

**Metrics logged** per best run:

| Metric | Description |
|---|---|
| `cv_roc_auc_mean` / `_std` | Cross-validation ROC-AUC (mean ± std) |
| `test_accuracy`, `test_f1`, `test_precision`, `test_recall`, `test_roc_auc` | Cleveland hold-out set |
| `val_accuracy`, `val_f1`, `val_precision`, `val_recall`, `val_roc_auc` | Hungarian external validation |
| `gap_roc_auc`, `gap_f1`, `gap_accuracy` | test − val gap (generalisation check) |

**Artefacts logged**: confusion matrices, ROC curves, classification reports (text), CV results CSV, and the sklearn model via `mlflow.sklearn.log_model`.

**Model bundle** written to disk (`models/model.pkl`) is a dict:
```python
{
    "model": Pipeline,          # preprocessor + classifier
    "feature_columns": [...],   # ordered list of 13 feature names
    "family": "randomforest",   # winning family name
    "mlflow_run_id": "abc123",  # MLflow run ID of the winning run
    "params": {...},            # best hyperparameters
    "metrics": {...},           # test + val metrics
}
```

---

## Running the Serving API

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `HEART_MODEL_PATH` | `models/model.pkl` | Path to the model bundle pickle |
| `LOG_LEVEL` | `INFO` | Python logging level |

### Start the server

```bash
# Ensure the model has been trained first
PYTHONPATH=src uvicorn heart.api:app --host 0.0.0.0 --port 8000 --reload
```

The `--reload` flag restarts the server on code changes (development only).

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe — returns `{"status": "ok"}` always |
| `GET` | `/ready` | Readiness probe — returns 503 if model not loaded |
| `POST` | `/predict` | Single-record prediction |
| `POST` | `/predict/batch` | Batch prediction (up to 1000 records) |
| `GET` | `/metrics` | Prometheus metrics exposition |
| `GET` | `/docs` | Auto-generated Swagger UI |
| `GET` | `/redoc` | ReDoc API documentation |

### Example request

**Single prediction:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "cp": 1,
    "trestbps": 145, "chol": 233, "fbs": 1,
    "restecg": 2, "thalach": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6
  }'
```

**Response:**

```json
{
  "prediction": 1,
  "confidence": 0.9312,
  "probability_disease": 0.9312,
  "probability_no_disease": 0.0688
}
```

- `prediction`: `0` = no disease, `1` = disease
- `confidence`: probability of the predicted class (always ≥ 0.5)
- `probability_disease` / `probability_no_disease`: raw model probabilities

**Batch prediction:**

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '[
    {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
     "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6},
    {"age": 45, "sex": 0, "cp": 2, "trestbps": 120, "chol": 200, "fbs": 0,
     "restecg": 0, "thalach": 170, "exang": 0, "oldpeak": 0.5, "slope": 1, "ca": 0, "thal": 3}
  ]'
```

**Readiness check:**

```bash
curl http://localhost:8000/ready
```

---

## Running Tests

```bash
PYTHONPATH=src pytest tests/ -v --tb=short
```

With coverage report:

```bash
PYTHONPATH=src pytest tests/ --cov=heart --cov-report=term-missing
```

---

## Docker

### Build the image

The Dockerfile uses a **two-stage build**: a `builder` stage installs all Python packages into a prefix, and the lean `runtime` stage copies only the installed packages and application code — no build tools in the final image.

```bash
# From the project root (same directory as docker/Dockerfile)
docker build -f docker/Dockerfile -t heart-api:latest .
```

> **Important**: the build context must be the project root (`docker/Dockerfile` copies `src/` and `models/`). Run the command from the project root, not from inside `docker/`.

### Run locally with Docker

```bash
docker run --rm -p 8000:8000 heart-api:latest
```

Test it:
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'
```

### Run with a custom model path

If you want to mount a model from outside the image:

```bash
docker run --rm -p 8000:8000 \
  -v "$(pwd)/models:/app/models" \
  -e HEART_MODEL_PATH=/app/models/model.pkl \
  -e LOG_LEVEL=DEBUG \
  heart-api:latest
```

### Environment variables at runtime

| Variable | Description |
|---|---|
| `HEART_MODEL_PATH` | Absolute path inside the container to the model bundle |
| `LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

---

## CI/CD Pipeline

### CI pipeline (`Jenkinsfile`)

Triggers automatically on every push to `main`. Runs four stages sequentially:

```
Checkout → Train → Unit Tests → Build & Push Docker → Deploy to Non-Prod
```

1. **Train**: creates a Python venv, installs deps, runs `heart.train` against the remote MLflow server. Artefacts (plots, reports) are archived in Jenkins.
2. **Unit Tests**: runs `pytest` with `PYTHONPATH` set.
3. **Build & Push Docker**: authenticates to the container registry using a GCP service account key stored as a Jenkins credential, then builds and pushes the image tagged with the git commit SHA and `latest`.
4. **Deploy to Non-Prod**: authenticates to Kubernetes, runs `kubectl set image` to roll out the new image, and waits for the rollout to complete.

Jenkins credentials used:

| Credential ID | Type | Purpose |
|---|---|---|
| `gcp-service-account-key` | Secret file | Authenticate to container registry and Kubernetes |
| `mlflow-tracking-uri` | Secret text | MLflow server address |

### Prod pipeline (`Jenkinsfile.prod`)

Triggered **manually** with two parameters:

| Parameter | Description |
|---|---|
| `IMAGE_TAG` | The git commit SHA of the image to deploy (from a passing CI build) |
| `SLOT` | `blue` or `green` — which deployment to update |

Stages:
```
Deploy Slot → Smoke Test → ── Manual approval gate ──► Patch Service Selector
```

The manual gate pauses the pipeline until an operator confirms the smoke test results. Once approved, the prod `Service` selector is patched to point at the newly deployed slot — completing the **blue/green cutover** with zero downtime.

To roll back: re-run the prod pipeline with the previous `IMAGE_TAG` and the same `SLOT`, or patch the service selector back to the previous slot.

---

## Kubernetes Manifests

| File | Purpose |
|---|---|
| `k8s/deployment-nonprod.yaml` | Single Deployment + ClusterIP Service in `nonprod` namespace |
| `k8s/deployment-blue.yaml` | Prod blue slot — `slot: blue` label, 2 replicas |
| `k8s/deployment-green.yaml` | Prod green slot — `slot: green` label, 2 replicas |
| `k8s/service-prod.yaml` | Prod Service — its `selector.slot` is patched to point at the live slot |
| `k8s/monitoring.yaml` | `ServiceMonitor` resources that tell Prometheus to scrape `/metrics` from both namespaces |
| `k8s/grafana-dashboard.yaml` | ConfigMap with the Grafana dashboard JSON — auto-loaded by the Grafana sidecar |

---

## Observability

### Prometheus metrics

The API exposes metrics at `GET /metrics` (Prometheus text format).

**HTTP metrics** (from `prometheus-fastapi-instrumentator`):

| Metric | Description |
|---|---|
| `http_requests_total` | Total requests by method, handler, status code |
| `http_request_duration_seconds` | Request latency histogram by handler |

**Custom ML metrics**:

| Metric | Labels | Description |
|---|---|---|
| `heart_predictions_total` | `prediction` (0/1) | Count of predictions served, split by class |
| `heart_prediction_latency_seconds` | — | Histogram of `model.predict()` time, excluding HTTP overhead |
| `heart_prediction_errors_total` | `error_type` | Count of errors during inference (`validation_error`, `value_error`, `runtime_error`, etc.) |

### Grafana dashboard

The `k8s/grafana-dashboard.yaml` ConfigMap is automatically loaded by the Grafana sidecar (label `grafana_dashboard: "1"`). It contains a dashboard with:

- **KPI row**: total predictions, disease detection rate gauge, median inference latency, error rate
- **Prediction charts**: throughput by class (disease vs no-disease), donut chart showing overall class split
- **HTTP charts**: request throughput (success vs error), latency percentiles (p50 / p95 / p99)
- **Model charts**: inference latency (p50 / p95), errors per minute
- **Disease rate trend**: fraction of predictions that are disease-positive over time (useful for detecting data drift)
- **API call count**: cumulative and windowed call counts for `/predict`
- **Infrastructure**: per-pod CPU and memory usage

The dashboard has an **Environment** dropdown variable that filters all panels to `nonprod`, `prod`, or both simultaneously.
