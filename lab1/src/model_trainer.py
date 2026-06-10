import logging
from pathlib import Path

import joblib
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline

from src.feature_engineering import build_preprocessor

logger = logging.getLogger(__name__)


def build_pipeline(cfg: DictConfig) -> Pipeline:
    preprocessor = build_preprocessor(
        list(cfg.data.numeric_features),
        list(cfg.data.categorical_features),
    )
    classifier = instantiate(
        cfg.model.classifier,
        random_state=cfg.training.random_state,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


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
    cfg: DictConfig,
) -> dict:
    pipeline = build_pipeline(cfg)
    model_name = cfg.model.name

    logger.info("Training %s ...", model_name)
    pipeline.fit(X_train, y_train)

    metrics = evaluate_pipeline(pipeline, X_test, y_test, model_name)

    models_dir = Path(cfg.paths.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{model_name}.pkl"
    joblib.dump(pipeline, model_path)
    logger.info("Saved pipeline to %s", model_path)

    return metrics
