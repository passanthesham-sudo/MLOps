"""DVC stage 1 — load raw CSV, stratified split, save processed files."""

import logging
import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

FEATURE_COLS = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
TARGET_COL = "Survived"


def main() -> None:
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["data"]

    df = pd.read_csv(params["path"])
    logger.info("Loaded %d rows from %s", len(df), params["path"])

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=y,
    )

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(out_dir / "X_train.csv", index=False)
    X_test.to_csv(out_dir / "X_test.csv", index=False)
    y_train.to_csv(out_dir / "y_train.csv", index=False)
    y_test.to_csv(out_dir / "y_test.csv", index=False)

    logger.info("Train=%d Test=%d → saved to %s/", len(X_train), len(X_test), out_dir)


if __name__ == "__main__":
    main()
