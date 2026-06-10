import logging

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.model import get_model
from app.schemas import PredictionResult, PredictRequest, PredictResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Titanic Survival Prediction API",
    description=(
        "Online serving endpoint for the Titanic survival prediction model. "
        "Accepts a batch of passenger records and returns survival predictions."
    ),
    version="1.0.0",
)


@app.on_event("startup")
async def startup_event() -> None:
    get_model()


@app.get("/", include_in_schema=False)
def root() -> JSONResponse:
    return JSONResponse(
        {
            "service": "Titanic Survival Prediction API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
        }
    )


@app.get("/health", tags=["ops"])
def health() -> dict:
    try:
        model = get_model()
        clf_name = type(model.named_steps["classifier"]).__name__
        return {"status": "healthy", "model_type": clf_name}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(request: PredictRequest) -> PredictResponse:
    model = get_model()

    df = pd.DataFrame([p.to_model_row() for p in request.passengers])
    logger.info("Scoring batch of %d passenger(s)", len(df))

    predictions = model.predict(df)
    probabilities = model.predict_proba(df)[:, 1]

    results = [
        PredictionResult(
            survived=int(pred),
            survival_probability=round(float(prob), 4),
            verdict="SURVIVED" if pred == 1 else "PERISHED",
        )
        for pred, prob in zip(predictions, probabilities, strict=True)
    ]

    return PredictResponse(
        model_type=type(model.named_steps["classifier"]).__name__,
        count=len(results),
        predictions=results,
    )
