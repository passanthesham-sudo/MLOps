"""Prefect batch prediction flow.

Steps
-----
1. Extract  — pull titanic_test from MotherDuck
2. Transform — rename columns to match the sklearn pipeline's expected input
3. Load model — fetch the Production model from DagHub MLflow registry
4. Predict  — score every row in the test set
5. Save     — write predictions back to MotherDuck as titanic_predictions
"""

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
from prefect import flow, get_run_logger, task

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

load_dotenv()

# MotherDuck snake_case → sklearn pipeline's expected capitalized column names
_COL_MAP: dict[str, str] = {
    "pclass": "Pclass",
    "sex": "Sex",
    "age": "Age",
    "sib_sp": "SibSp",
    "parch": "Parch",
    "fare": "Fare",
    "embarked": "Embarked",
}

logger = logging.getLogger(__name__)


def _md_conn(db_name: str) -> duckdb.DuckDBPyConnection:
    token = os.environ.get("MOTHERDUCK_TOKEN", "")
    if not token:
        raise OSError("MOTHERDUCK_TOKEN is not set.")
    return duckdb.connect(f"md:{db_name}?motherduck_token={token}")


# ── Tasks ────────────────────────────────────────────────────────────────────


@task(name="extract-test-data", retries=3, retry_delay_seconds=15)
def extract_from_motherduck(db_name: str) -> pd.DataFrame:
    log = get_run_logger()
    log.info("Extracting titanic_test from MotherDuck '%s' ...", db_name)

    conn = _md_conn(db_name)
    try:
        df = conn.execute("SELECT * FROM titanic_test").df()
    finally:
        conn.close()

    log.info("Extracted %d rows.", len(df))
    return df


@task(name="transform-features")
def transform(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    log = get_run_logger()

    passenger_ids: pd.Series = raw_df["passenger_id"].reset_index(drop=True)
    features: pd.DataFrame = (
        raw_df[list(_COL_MAP.keys())].rename(columns=_COL_MAP).reset_index(drop=True)
    )

    log.info(
        "Features shape: %s | missing Age: %d | missing Embarked: %d",
        features.shape,
        features["Age"].isna().sum(),
        features["Embarked"].isna().sum(),
    )
    return features, passenger_ids


@task(name="load-production-model", retries=2, retry_delay_seconds=20)
def load_production_model(tracking_uri: str, registry_name: str):
    log = get_run_logger()

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
    log.info("Loading model from '%s' ...", model_uri)

    model = mlflow.sklearn.load_model(model_uri)
    clf_name = type(model.named_steps["classifier"]).__name__
    log.info("Loaded: %s", clf_name)
    return model


@task(name="generate-predictions")
def predict(
    model,
    features: pd.DataFrame,
    passenger_ids: pd.Series,
) -> pd.DataFrame:
    log = get_run_logger()

    survived = model.predict(features).astype(int)
    probabilities = model.predict_proba(features)[:, 1].round(4)

    predictions = pd.DataFrame(
        {
            "passenger_id": passenger_ids,
            "survived": survived,
            "survival_probability": probabilities,
        }
    )

    survival_rate = predictions["survived"].mean() * 100
    log.info(
        "Predicted %d passengers — %d survived (%.1f%%)",
        len(predictions),
        predictions["survived"].sum(),
        survival_rate,
    )
    return predictions


@task(name="save-predictions")
def save_predictions(
    predictions: pd.DataFrame,
    db_name: str,
    model_registry_name: str,
) -> None:
    log = get_run_logger()

    output = predictions.copy()
    output["model_name"] = model_registry_name
    output["predicted_at"] = datetime.now(tz=timezone.utc).replace(tzinfo=None)

    conn = _md_conn(db_name)
    try:
        conn.register("predictions_df", output)
        conn.execute("""
            CREATE OR REPLACE TABLE titanic_predictions AS
            SELECT
                passenger_id,
                survived,
                survival_probability,
                model_name,
                CAST(predicted_at AS TIMESTAMP) AS predicted_at
            FROM predictions_df
        """)
        conn.unregister("predictions_df")
        count = conn.execute("SELECT COUNT(*) FROM titanic_predictions").fetchone()[0]
    finally:
        conn.close()

    log.info("Saved %d rows to MotherDuck 'titanic_predictions'.", count)


# ── Flow ─────────────────────────────────────────────────────────────────────


@flow(name="titanic-batch-prediction", log_prints=True)
def batch_prediction_flow(
    db_name: str = "titanic_ml",
    mlflow_tracking_uri: str = "https://dagshub.com/passanthesham-sudo/MLOps.mlflow",
    model_registry_name: str = "titanic-best-model",
) -> None:
    """End-to-end batch scoring: MotherDuck → MLflow Production model → MotherDuck."""

    raw_df = extract_from_motherduck(db_name)

    features, passenger_ids = transform(raw_df)

    model = load_production_model(mlflow_tracking_uri, model_registry_name)

    predictions = predict(model, features, passenger_ids)

    save_predictions(predictions, db_name, model_registry_name)


if __name__ == "__main__":
    batch_prediction_flow()
