import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

FEATURE_COLS = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
TARGET_COL = "Survived"


def load_data(
    data_path: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    if not Path(data_path).exists():
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'. "
            "Download train.csv from https://www.kaggle.com/competitions/titanic "
            "and place it in data/raw/train.csv"
        )

    df = pd.read_csv(data_path)
    logger.info("Loaded %d rows from %s", len(df), data_path)

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info("Train size: %d | Test size: %d", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test
