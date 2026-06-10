"""One-time script: create MotherDuck database and load Titanic test.csv into it."""

import logging
import os
import sys
from pathlib import Path

import duckdb
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

TEST_CSV = Path("data/raw/test.csv")
DB_NAME = os.environ.get("MOTHERDUCK_DB", "titanic_ml")


def get_token() -> str:
    token = os.environ.get("MOTHERDUCK_TOKEN", "")
    if not token:
        logger.error("MOTHERDUCK_TOKEN environment variable is not set.")
        sys.exit(1)
    return token


def create_database(token: str) -> None:
    logger.info("Creating MotherDuck database '%s' (if not exists) ...", DB_NAME)
    conn = duckdb.connect(f"md:?motherduck_token={token}")
    conn.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
    conn.close()
    logger.info("Database ready.")


def load_test_data(token: str) -> None:
    if not TEST_CSV.exists():
        logger.error(
            "Test file not found at '%s'. "
            "Download test.csv from https://www.kaggle.com/competitions/titanic "
            "and place it in lab7/data/raw/test.csv",
            TEST_CSV,
        )
        sys.exit(1)

    logger.info("Loading '%s' → MotherDuck '%s'.titanic_test ...", TEST_CSV, DB_NAME)

    conn = duckdb.connect(f"md:{DB_NAME}?motherduck_token={token}")
    try:
        conn.execute(f"""
            CREATE OR REPLACE TABLE titanic_test AS
            SELECT
                PassengerId  AS passenger_id,
                Pclass       AS pclass,
                Name         AS name,
                Sex          AS sex,
                Age          AS age,
                SibSp        AS sib_sp,
                Parch        AS parch,
                Ticket       AS ticket,
                Fare         AS fare,
                Cabin        AS cabin,
                Embarked     AS embarked
            FROM read_csv_auto('{TEST_CSV.as_posix()}')
        """)
        count = conn.execute("SELECT COUNT(*) FROM titanic_test").fetchone()[0]
        logger.info("Loaded %d rows into titanic_test.", count)
    finally:
        conn.close()


def main() -> None:
    token = get_token()
    create_database(token)
    load_test_data(token)
    logger.info("Done. Run the Prefect flow next: python flows/batch_prediction.py")


if __name__ == "__main__":
    main()
