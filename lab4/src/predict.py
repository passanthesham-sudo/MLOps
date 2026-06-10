"""Load the Production model from MLflow registry and run sample predictions."""

import logging
import os
import sys
from pathlib import Path

import mlflow.sklearn
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SAMPLE_PASSENGERS = pd.DataFrame(
    [
        # 1st-class female — historically high survival rate
        {
            "Pclass": 1,
            "Sex": "female",
            "Age": 29.0,
            "SibSp": 0,
            "Parch": 0,
            "Fare": 211.3,
            "Embarked": "S",
        },
        # 3rd-class male — historically low survival rate
        {
            "Pclass": 3,
            "Sex": "male",
            "Age": 22.0,
            "SibSp": 1,
            "Parch": 0,
            "Fare": 7.25,
            "Embarked": "S",
        },
        # 2nd-class female
        {
            "Pclass": 2,
            "Sex": "female",
            "Age": 31.0,
            "SibSp": 1,
            "Parch": 1,
            "Fare": 26.25,
            "Embarked": "S",
        },
        # 1st-class male
        {
            "Pclass": 1,
            "Sex": "male",
            "Age": 35.0,
            "SibSp": 0,
            "Parch": 0,
            "Fare": 52.0,
            "Embarked": "C",
        },
        # 3rd-class female with missing age
        {
            "Pclass": 3,
            "Sex": "female",
            "Age": None,
            "SibSp": 0,
            "Parch": 2,
            "Fare": 15.5,
            "Embarked": "Q",
        },
    ]
)


def load_production_model(tracking_uri: str, registry_name: str):
    os.environ.setdefault(
        "MLFLOW_TRACKING_USERNAME",
        os.environ.get("DAGSHUB_USERNAME", ""),
    )
    os.environ.setdefault(
        "MLFLOW_TRACKING_PASSWORD",
        os.environ.get("DAGSHUB_TOKEN", ""),
    )
    mlflow.set_tracking_uri(tracking_uri)

    model_uri = f"models:/{registry_name}/Production"
    logger.info("Loading model from %s ...", model_uri)
    return mlflow.sklearn.load_model(model_uri)


def main() -> None:
    with open("params.yaml") as f:
        mlflow_params = yaml.safe_load(f)["mlflow"]

    model = load_production_model(
        mlflow_params["tracking_uri"],
        mlflow_params["model_registry_name"],
    )
    logger.info("Model loaded: %s", type(model.named_steps["classifier"]).__name__)

    predictions = model.predict(SAMPLE_PASSENGERS)
    probabilities = model.predict_proba(SAMPLE_PASSENGERS)[:, 1]

    results = SAMPLE_PASSENGERS[["Pclass", "Sex", "Age"]].copy()
    results["survived"] = predictions
    results["survival_prob"] = probabilities.round(3)
    results["verdict"] = results["survived"].map({1: "SURVIVED", 0: "PERISHED"})

    print("\n=== Titanic Survival Predictions (Production Model) ===")
    print(results.to_string(index=False))
    print()


if __name__ == "__main__":
    main()
