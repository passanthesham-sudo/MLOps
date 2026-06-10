"""Entry point for the Titanic survival prediction training pipeline."""

import logging
import sys

from src.config import Config
from src.data_loader import load_data
from src.model_trainer import train_and_evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    config = Config()

    logger.info("=== Titanic Training Pipeline ===")
    logger.info("Data path  : %s", config.data_path)
    logger.info("Models dir : %s", config.models_dir)

    try:
        X_train, X_test, y_train, y_test = load_data(
            config.data_path,
            test_size=config.test_size,
            random_state=config.random_state,
        )
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)

    results = train_and_evaluate(X_train, X_test, y_train, y_test, config)

    logger.info("=== Results Summary ===")
    for r in results:
        logger.info(
            "%-25s | Accuracy=%.4f | ROC-AUC=%.4f",
            r["model"],
            r["accuracy"],
            r["roc_auc"],
        )

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
