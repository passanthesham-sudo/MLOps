import logging
import os
from pathlib import Path

import joblib

logger = logging.getLogger(__name__)

_model = None


def load_model():
    global _model
    if _model is not None:
        return _model

    model_path = Path(os.environ.get("MODEL_PATH", "models/best_model.pkl"))

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at '{model_path}'. "
            "Copy your best trained model (from lab4/models/) to lab6/models/best_model.pkl "
            "or set the MODEL_PATH environment variable."
        )

    logger.info("Loading model from %s ...", model_path)
    _model = joblib.load(model_path)
    clf_name = type(_model.named_steps["classifier"]).__name__
    logger.info("Model ready: %s", clf_name)
    return _model


def get_model():
    return load_model()
