"""Configurable Titanic training pipeline using Hydra.

Run with defaults:
    python train.py

Switch model:
    python train.py model=random_forest

Override any parameter:
    python train.py training.test_size=0.3 model.classifier.C=0.5

Run all models (multirun):
    python train.py -m model=logistic_regression,random_forest
"""

import logging

import hydra
from omegaconf import DictConfig

from src.data_loader import load_data
from src.model_trainer import train_and_evaluate

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info("=== Titanic Training Pipeline ===")
    log.info("Model      : %s", cfg.model.name)
    log.info("Data path  : %s", cfg.data.path)
    log.info("Test size  : %s", cfg.training.test_size)

    X_train, X_test, y_train, y_test = load_data(
        cfg.data.path,
        test_size=cfg.training.test_size,
        random_state=cfg.training.random_state,
    )

    result = train_and_evaluate(X_train, X_test, y_train, y_test, cfg)

    log.info(
        "Done | model=%-25s accuracy=%.4f roc_auc=%.4f",
        result["model"],
        result["accuracy"],
        result["roc_auc"],
    )


if __name__ == "__main__":
    main()
