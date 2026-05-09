"""MLflow-tracked training entrypoint for the heart-disease classifier.

Mirrors ``notebooks/02_modeling.ipynb`` but runs headless so it can be
driven by CI (Jenkins / GitHub Actions). Each model family (logistic
regression, random forest) becomes a parent MLflow run with one nested
child per hyperparameter candidate plus a final ``<family>_best`` child
that logs the refitted estimator, evaluation metrics on the Cleveland
hold-out and on the Hungarian external-validation split, plus plots.

Usage
-----
    python -m heart.train \\
        --data project/data/processed.cleveland.data \\
        --val-data project/data/processed.hungarian.data \\
        --tracking-uri file:./project/mlruns \\
        --experiment heart-disease-modeling \\
        --output-model project/models/model.pkl

Environment overrides (useful in CI):

- ``MLFLOW_TRACKING_URI`` — overrides ``--tracking-uri`` (e.g. the GCP MLflow
  server URL once the platform is in place).
- ``MLFLOW_EXPERIMENT_NAME`` — overrides ``--experiment``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # headless / CI safe — must happen before pyplot import
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from .data import FEATURES, load_processed, split_features_target
from .features import build_preprocessor

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("heart.train")

RANDOM_STATE = 42
SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


# --------------------------------------------------------------------------- #
# Metric helpers (kept here, not in features.py, because they're MLflow-bound)
# --------------------------------------------------------------------------- #
def _compute_metrics(model, X_eval: pd.DataFrame, y_eval: pd.Series, prefix: str) -> dict:
    y_pred = model.predict(X_eval)
    y_proba = model.predict_proba(X_eval)[:, 1]
    return {
        f"{prefix}_accuracy": accuracy_score(y_eval, y_pred),
        f"{prefix}_precision": precision_score(y_eval, y_pred),
        f"{prefix}_recall": recall_score(y_eval, y_pred),
        f"{prefix}_f1": f1_score(y_eval, y_pred),
        f"{prefix}_roc_auc": roc_auc_score(y_eval, y_proba),
    }


def _log_confusion_matrix(model, X_eval, y_eval, out_dir: Path, family: str, split: str) -> Path:
    cm = confusion_matrix(y_eval, model.predict(X_eval))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["no_disease", "disease"])
    ax.set_yticks([0, 1], ["no_disease", "disease"])
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix — {family} ({split})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out = out_dir / f"cm_{family}_{split}.png"
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    mlflow.log_artifact(str(out), artifact_path="plots")
    return out


def _log_roc_curve(model, X_eval, y_eval, out_dir: Path, family: str, split: str) -> Path:
    y_proba = model.predict_proba(X_eval)[:, 1]
    fpr, tpr, _ = roc_curve(y_eval, y_proba)
    auc = roc_auc_score(y_eval, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}", color="#D7263D")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC — {family} ({split})"); ax.legend(loc="lower right")
    out = out_dir / f"roc_{family}_{split}.png"
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    mlflow.log_artifact(str(out), artifact_path="plots")
    return out


def _log_classification_report(model, X_eval, y_eval, out_dir: Path, family: str, split: str) -> Path:
    report = classification_report(
        y_eval, model.predict(X_eval),
        target_names=["no_disease", "disease"], digits=3,
    )
    out = out_dir / f"report_{family}_{split}.txt"
    out.write_text(report)
    mlflow.log_artifact(str(out), artifact_path="reports")
    log.info("\n--- %s on %s ---\n%s", family, split, report)
    return out


# --------------------------------------------------------------------------- #
# Per-family training routine
# --------------------------------------------------------------------------- #
@dataclass
class FamilyResult:
    family: str
    best_model: Pipeline
    best_run_id: str
    best_params: dict
    test_metrics: dict
    val_metrics: dict


def _train_family(
    family: str,
    estimator,
    param_grid: dict,
    X_tr, y_tr, X_te, y_te, X_va, y_va,
    artifacts_dir: Path,
) -> FamilyResult:
    pipe = Pipeline([("prep", build_preprocessor()), ("clf", estimator)])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        pipe, param_grid=param_grid,
        scoring=SCORING, refit="roc_auc",
        cv=cv, n_jobs=-1, return_train_score=False,
    )

    with mlflow.start_run(run_name=family) as parent:
        mlflow.set_tag("model_family", family)
        mlflow.log_params({
            "cv_folds": cv.get_n_splits(),
            "n_train": len(X_tr),
            "n_test": len(X_te),
            "n_val_external": len(X_va),
            "val_source": "processed.hungarian.data",
            "random_state": RANDOM_STATE,
            "refit_metric": "roc_auc",
        })

        log.info("[%s] starting GridSearchCV over %d combos", family, len(grid.param_grid) if isinstance(grid.param_grid, list) else int(np.prod([len(v) for v in param_grid.values()])))
        grid.fit(X_tr, y_tr)

        cv_results = pd.DataFrame(grid.cv_results_).sort_values(
            "mean_test_roc_auc", ascending=False,
        )
        for _, row in cv_results.iterrows():
            params = row["params"]
            child_name = f"{family}_" + "_".join(
                f"{k.split('__')[-1]}={v}" for k, v in params.items()
            )
            with mlflow.start_run(run_name=child_name, nested=True):
                mlflow.set_tag("model_family", family)
                mlflow.set_tag("stage", "cv_candidate")
                mlflow.log_params(params)
                for metric in SCORING:
                    mlflow.log_metric(f"cv_{metric}_mean", row[f"mean_test_{metric}"])
                    mlflow.log_metric(f"cv_{metric}_std", row[f"std_test_{metric}"])
                mlflow.log_metric("cv_fit_time_mean", row["mean_fit_time"])

        cv_csv = artifacts_dir / f"cv_results_{family}.csv"
        cv_results.to_csv(cv_csv, index=False)
        mlflow.log_artifact(str(cv_csv), artifact_path="cv")

        best_params = grid.best_params_
        best_model: Pipeline = grid.best_estimator_

        with mlflow.start_run(run_name=f"{family}_best", nested=True) as best_run:
            mlflow.set_tag("model_family", family)
            mlflow.set_tag("stage", "best")
            mlflow.log_params(best_params)
            best_row = cv_results.iloc[0]
            for metric in SCORING:
                mlflow.log_metric(f"cv_{metric}_mean", best_row[f"mean_test_{metric}"])
                mlflow.log_metric(f"cv_{metric}_std", best_row[f"std_test_{metric}"])

            test_metrics = _compute_metrics(best_model, X_te, y_te, prefix="test")
            mlflow.log_metrics(test_metrics)
            _log_confusion_matrix(best_model, X_te, y_te, artifacts_dir, family, "test")
            _log_roc_curve(best_model, X_te, y_te, artifacts_dir, family, "test")
            _log_classification_report(best_model, X_te, y_te, artifacts_dir, family, "test")

            val_metrics = _compute_metrics(best_model, X_va, y_va, prefix="val")
            mlflow.log_metrics(val_metrics)
            _log_confusion_matrix(best_model, X_va, y_va, artifacts_dir, family, "val_hungarian")
            _log_roc_curve(best_model, X_va, y_va, artifacts_dir, family, "val_hungarian")
            _log_classification_report(best_model, X_va, y_va, artifacts_dir, family, "val_hungarian")

            gap_metrics = {
                f"gap_{m}": test_metrics[f"test_{m}"] - val_metrics[f"val_{m}"]
                for m in ("roc_auc", "f1", "accuracy")
            }
            mlflow.log_metrics(gap_metrics)

            mlflow.sklearn.log_model(
                sk_model=best_model,
                name="model",
                input_example=X_tr.head(3),
            )
            best_run_id = best_run.info.run_id

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_cv_roc_auc", grid.best_score_)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"best_{k}", v)
        for k, v in val_metrics.items():
            mlflow.log_metric(f"best_{k}", v)

        log.info("[%s] best params=%s cv_roc_auc=%.4f", family, best_params, grid.best_score_)
        log.info("[%s] test=%s val=%s gap=%s parent=%s best=%s",
                 family, test_metrics, val_metrics, gap_metrics,
                 parent.info.run_id, best_run_id)

    return FamilyResult(
        family=family,
        best_model=best_model,
        best_run_id=best_run_id,
        best_params=best_params,
        test_metrics=test_metrics,
        val_metrics=val_metrics,
    )


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    p.add_argument("--data", type=Path, required=True,
                   help="Path to processed.cleveland.data (training site).")
    p.add_argument("--val-data", type=Path, required=True,
                   help="Path to processed.hungarian.data (external validation).")
    p.add_argument("--tracking-uri", type=str,
                   default=os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"),
                   help="MLflow tracking URI. Env MLFLOW_TRACKING_URI takes precedence.")
    p.add_argument("--experiment", type=str,
                   default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "heart-disease-modeling"),
                   help="MLflow experiment name.")
    p.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"),
                   help="Local directory for plots/reports/CV CSVs.")
    p.add_argument("--output-model", type=Path, default=Path("models/model.pkl"),
                   help="Where to write the winning sklearn pipeline (.pkl).")
    p.add_argument("--test-size", type=float, default=0.20)
    p.add_argument("--random-state", type=int, default=RANDOM_STATE)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    args.output_model.parent.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    log.info("MLflow tracking=%s experiment=%s", mlflow.get_tracking_uri(), args.experiment)

    df_train = load_processed(args.data)
    X, y = split_features_target(df_train)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_state,
    )
    df_val = load_processed(args.val_data)
    X_va, y_va = split_features_target(df_val)
    log.info("train=%s test=%s val_external=%s", X_tr.shape, X_te.shape, X_va.shape)

    lr_grid = {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "clf__penalty": ["l1", "l2"],
    }
    lr_estimator = LogisticRegression(
        solver="liblinear", max_iter=1000, random_state=args.random_state,
    )
    lr = _train_family(
        "logreg", lr_estimator, lr_grid,
        X_tr, y_tr, X_te, y_te, X_va, y_va, args.artifacts_dir,
    )

    rf_grid = {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [4, 6, 8, None],
        "clf__min_samples_leaf": [1, 3, 5],
    }
    rf_estimator = RandomForestClassifier(
        random_state=args.random_state, n_jobs=-1,
    )
    rf = _train_family(
        "randomforest", rf_estimator, rf_grid,
        X_tr, y_tr, X_te, y_te, X_va, y_va, args.artifacts_dir,
    )

    # Pick the winner by external-validation ROC-AUC (drift-aware selection).
    results = [lr, rf]
    winner = max(results, key=lambda r: r.val_metrics["val_roc_auc"])
    log.info("Winner: %s (val_roc_auc=%.4f)", winner.family, winner.val_metrics["val_roc_auc"])

    joblib.dump(
        {
            "model": winner.best_model,
            "feature_columns": FEATURES,
            "family": winner.family,
            "mlflow_run_id": winner.best_run_id,
            "params": winner.best_params,
            "metrics": {**winner.test_metrics, **winner.val_metrics},
        },
        args.output_model,
    )
    log.info("Saved winning model bundle to %s", args.output_model)

    # Register the winning model in the MLflow Model Registry so that
    # deployment pipelines can tag which version is live in each environment.
    MODEL_REGISTRY_NAME = "HeartDiseaseClassifier"
    model_uri = f"runs:/{winner.best_run_id}/model"
    registered = mlflow.register_model(model_uri, MODEL_REGISTRY_NAME)
    log.info(
        "Registered model '%s' version=%s from run_id=%s",
        MODEL_REGISTRY_NAME, registered.version, winner.best_run_id,
    )

    # Write version number to a file so Jenkins can read it without
    # re-querying MLflow or parsing Python stdout.
    version_file = args.artifacts_dir / "model_version.txt"
    version_file.write_text(str(registered.version))
    log.info("Wrote model version %s to %s", registered.version, version_file)

    summary = {
        "winner": winner.family,
        "winner_run_id": winner.best_run_id,
        "results": [
            {
                "family": r.family,
                "run_id": r.best_run_id,
                "params": r.best_params,
                "test_metrics": r.test_metrics,
                "val_metrics": r.val_metrics,
            }
            for r in results
        ],
    }
    summary_path = args.artifacts_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    log.info("Wrote summary %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
