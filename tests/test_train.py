"""Smoke tests for ``heart.train``.

The full ``main()`` entrypoint runs a sizeable GridSearchCV across two
families; we cover its helpers and a tiny ``_train_family`` invocation here
to keep the unit-test runtime small. End-to-end training is exercised by CI.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import mlflow
import pytest
from sklearn.linear_model import LogisticRegression

from heart import train as train_mod
from heart.data import FEATURES, load_processed, split_features_target


@pytest.fixture()
def mlflow_local(tmp_path, monkeypatch):
    """Point MLflow at a throwaway local file store."""
    uri = f"file:{tmp_path / 'mlruns'}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("unit-tests")
    yield tmp_path
    # Reset so other tests don't inherit our experiment context.
    mlflow.set_tracking_uri("file:./mlruns")


def test_compute_metrics_shape(trained_pipeline, sample_frame):
    X = sample_frame[FEATURES]; y = sample_frame["target"]
    metrics = train_mod._compute_metrics(trained_pipeline, X, y, prefix="test")
    expected = {
        "test_accuracy", "test_precision", "test_recall",
        "test_f1", "test_roc_auc",
    }
    assert set(metrics.keys()) == expected
    for v in metrics.values():
        assert 0.0 <= float(v) <= 1.0


def test_train_family_returns_fitted_bundle(
    sample_frame, mlflow_local, tmp_path,
):
    X = sample_frame[FEATURES]; y = sample_frame["target"]
    # Tiny grid → fast.
    grid = {"clf__C": [0.1, 1.0]}
    artifacts = tmp_path / "artifacts"; artifacts.mkdir()

    result = train_mod._train_family(
        family="logreg",
        estimator=LogisticRegression(solver="liblinear", max_iter=200),
        param_grid=grid,
        X_tr=X, y_tr=y, X_te=X, y_te=y, X_va=X, y_va=y,
        artifacts_dir=artifacts,
    )

    assert result.family == "logreg"
    assert result.best_params.get("clf__C") in (0.1, 1.0)
    # Fitted pipeline must be able to predict.
    preds = result.best_model.predict(X.head(3))
    assert len(preds) == 3
    # Some MLflow artifact files should have been emitted locally.
    assert any(artifacts.iterdir())
    assert "test_roc_auc" in result.test_metrics
    assert "val_roc_auc" in result.val_metrics


def test_main_end_to_end(processed_data_file: Path, mlflow_local, tmp_path, monkeypatch):
    """End-to-end run with shrunken hyperparameter grids for speed."""
    # Patch the heavy grids inside main() so the test stays fast.
    real_train_family = train_mod._train_family

    def fast_train_family(family, estimator, param_grid, *args, **kwargs):
        if family == "logreg":
            param_grid = {"clf__C": [1.0]}
        elif family == "randomforest":
            param_grid = {"clf__n_estimators": [25], "clf__max_depth": [3]}
        return real_train_family(family, estimator, param_grid, *args, **kwargs)

    monkeypatch.setattr(train_mod, "_train_family", fast_train_family)

    out_model = tmp_path / "model.pkl"
    artifacts = tmp_path / "artifacts"
    rc = train_mod.main([
        "--data", str(processed_data_file),
        "--val-data", str(processed_data_file),
        "--tracking-uri", f"file:{tmp_path / 'mlruns'}",
        "--experiment", "unit-tests-main",
        "--artifacts-dir", str(artifacts),
        "--output-model", str(out_model),
        "--test-size", "0.3",
    ])
    assert rc == 0
    assert out_model.exists()
    bundle = joblib.load(out_model)
    assert bundle["family"] in {"logreg", "randomforest"}
    assert list(bundle["feature_columns"]) == FEATURES
    assert (artifacts / "training_summary.json").exists()


def test_load_and_split_match_train_inputs(processed_data_file):
    """Sanity check: the loader output is what train.main feeds into sklearn."""
    frame = load_processed(processed_data_file)
    X, y = split_features_target(frame)
    assert list(X.columns) == FEATURES
    assert len(X) == len(y)
