import logging

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline

from src.config import Config
from src.feature_engineering import build_preprocessor

logger = logging.getLogger(__name__)


def build_pipelines(config: Config) -> dict[str, Pipeline]:
    preprocessor = build_preprocessor(
        config.numeric_features, config.categorical_features
    )

    logistic_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    random_state=config.random_state,
                    max_iter=1000,
                    C=1.0,
                ),
            ),
        ]
    )

    rf_preprocessor = build_preprocessor(
        config.numeric_features, config.categorical_features
    )

    random_forest_pipeline = Pipeline(
        steps=[
            ("preprocessor", rf_preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=config.random_state,
                    n_estimators=100,
                    max_depth=6,
                    min_samples_leaf=2,
                ),
            ),
        ]
    )

    return {
        "logistic_regression": logistic_pipeline,
        "random_forest": random_forest_pipeline,
    }


def evaluate_pipeline(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    report = classification_report(
        y_test, y_pred, target_names=["Not Survived", "Survived"]
    )

    logger.info("--- %s ---", model_name.upper())
    logger.info("Accuracy : %.4f", accuracy)
    logger.info("ROC-AUC  : %.4f", roc_auc)
    logger.info("\n%s", report)

    return {"model": model_name, "accuracy": accuracy, "roc_auc": roc_auc}


def train_and_evaluate(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: Config,
) -> list[dict]:
    pipelines = build_pipelines(config)
    results = []

    for name, pipeline in pipelines.items():
        logger.info("Training %s ...", name)
        pipeline.fit(X_train, y_train)

        metrics = evaluate_pipeline(pipeline, X_test, y_test, name)
        results.append(metrics)

        model_path = config.models_dir / f"{name}.pkl"
        joblib.dump(pipeline, model_path)
        logger.info("Saved pipeline to %s", model_path)

    best = max(results, key=lambda r: r["roc_auc"])
    logger.info("Best model: %s (ROC-AUC=%.4f)", best["model"], best["roc_auc"])
    return results
