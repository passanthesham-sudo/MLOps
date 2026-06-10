from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    data_path: Path = Path("data/raw/train.csv")
    models_dir: Path = Path("models")
    test_size: float = 0.2
    random_state: int = 42
    numeric_features: list = field(
        default_factory=lambda: ["Age", "Fare", "SibSp", "Parch"]
    )
    categorical_features: list = field(
        default_factory=lambda: ["Sex", "Embarked", "Pclass"]
    )

    def __post_init__(self) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)
