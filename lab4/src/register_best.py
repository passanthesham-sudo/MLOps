"""DVC stage 3 — compare models, register the best one to MLflow, promote to Production."""

import json
import logging
import os
import sys
from pathlib import Path

import mlflow
import yaml
from mlflow.tracking import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MODELS = ["logistic_regression", "random_forest"]


def setup_mlflow(tracking_uri: str) -> None:
    os.environ.setdefault(
        "MLFLOW_TRACKING_USERNAME",
        os.environ.get("DAGSHUB_USERNAME", ""),
    )
    os.environ.setdefault(
        "MLFLOW_TRACKING_PASSWORD",
        os.environ.get("DAGSHUB_TOKEN", ""),
    )
    mlflow.set_tracking_uri(tracking_uri)


def main() -> None:
    with open("params.yaml") as f:
        mlflow_params = yaml.safe_load(f)["mlflow"]

    setup_mlflow(mlflow_params["tracking_uri"])

    # Load metrics + run IDs written by train.py
    model_results: dict = {}
    for name in MODELS:
        metrics_path = Path(f"metrics/{name}.json")
        with open(metrics_path) as f:
            model_results[name] = json.load(f)
        logger.info(
            "%s → accuracy=%.4f  roc_auc=%.4f  run_id=%s",
            name,
            model_results[name]["accuracy"],
            model_results[name]["roc_auc"],
            model_results[name]["mlflow_run_id"],
        )

    # Pick best by ROC-AUC
    best_name = max(model_results, key=lambda k: model_results[k]["roc_auc"])
    best = model_results[best_name]
    logger.info("Best model: %s (ROC-AUC=%.4f)", best_name, best["roc_auc"])

    # Register to MLflow Model Registry
    registry_name = mlflow_params["model_registry_name"]
    model_uri = f"runs:/{best['mlflow_run_id']}/model"

    logger.info("Registering '%s' to registry as '%s' ...", model_uri, registry_name)
    mv = mlflow.register_model(model_uri=model_uri, name=registry_name)
    logger.info("Registered version: %s", mv.version)

    # Transition new version to Production, archive old ones
    client = MlflowClient()
    client.transition_model_version_stage(
        name=registry_name,
        version=mv.version,
        stage="Production",
        archive_existing_versions=True,
    )
    logger.info("'%s' v%s → Production", registry_name, mv.version)

    # Save registration result (DVC metric)
    result = {
        "best_model": best_name,
        "registry_name": registry_name,
        "version": int(mv.version),
        "stage": "Production",
        "accuracy": best["accuracy"],
        "roc_auc": best["roc_auc"],
        "run_id": best["mlflow_run_id"],
    }
    out_path = Path("metrics/registration.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Saved registration result to %s", out_path)


if __name__ == "__main__":
    main()
