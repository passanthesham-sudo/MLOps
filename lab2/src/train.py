"""DVC stage 2 — build sklearn pipeline, train, evaluate, save model + metrics."""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline

# Ensure lab2/ is in path when invoked by DVC
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.feature_engineering import build_preprocessor  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

CLASSIFIERS: dict = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
}


def load_processed_data() -> tuple:
    p = Path("data/processed")
    X_train = pd.read_csv(p / "X_train.csv")
    X_test = pd.read_csv(p / "X_test.csv")
    y_train = pd.read_csv(p / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(p / "y_test.csv").squeeze("columns")
    return X_train, X_test, y_train, y_test


def build_pipeline(
    model_name: str,
    model_params: dict,
    data_params: dict,
) -> Pipeline:
    preprocessor = build_preprocessor(
        data_params["numeric_features"],
        data_params["categorical_features"],
    )
    clf_class = CLASSIFIERS[model_name]
    classifier = clf_class(
        random_state=data_params["random_state"],
        **model_params,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(CLASSIFIERS))
    args = parser.parse_args()

    with open("params.yaml") as f:
        all_params = yaml.safe_load(f)

    model_params = all_params[args.model]
    data_params = all_params["data"]

    X_train, X_test, y_train, y_test = load_processed_data()
    logger.info("Training %s ...", args.model)

    pipeline = build_pipeline(args.model, model_params, data_params)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    logger.info("Accuracy : %.4f", accuracy)
    logger.info("ROC-AUC  : %.4f", roc_auc)
    logger.info(
        "\n%s",
        classification_report(
            y_test, y_pred, target_names=["Not Survived", "Survived"]
        ),
    )

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{args.model}.pkl"
    joblib.dump(pipeline, model_path)
    logger.info("Saved model to %s", model_path)

    metrics_dir = Path("metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"accuracy": round(accuracy, 4), "roc_auc": round(roc_auc, 4)}
    metrics_path = metrics_dir / f"{args.model}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics to %s", metrics_path)


if __name__ == "__main__":
    main()
