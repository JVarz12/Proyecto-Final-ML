
"""Servicio de inferencia FastAPI para el modelo de deserción estudiantil."""
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

APP_ROOT = Path(__file__).resolve().parent
MODELS_DIR = APP_ROOT.parent / "mlops_pipeline" / "models"
OUTPUTS_DIR = APP_ROOT.parent / "mlops_pipeline" / "outputs"

metadata_path = MODELS_DIR / "model_metadata.json"
if metadata_path.suffix == ".joblib":
    metadata = joblib.load(metadata_path)
else:
    import json

    with metadata_path.open("r", encoding="utf-8") as fp:
        metadata = json.load(fp)

model = joblib.load(MODELS_DIR / f"best_model_{metadata['model_name']}.joblib")
feature_pipeline = joblib.load(OUTPUTS_DIR / "feature_pipeline.joblib")

app = FastAPI(title="Student Dropout Predictor", version="1.0.0")


class StudentRequest(BaseModel):
    records: List[Dict]


class BatchPrediction(BaseModel):
    predictions: List[str]
    probabilities: List[Dict[str, float]]


@app.get("/health", tags=["infra"])
async def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=BatchPrediction, tags=["model"])
async def predict(request: StudentRequest) -> BatchPrediction:
    input_df = pd.DataFrame(request.records)
    transformed = feature_pipeline.transform(input_df)
    probabilities = model.predict_proba(transformed)
    classes = model.classes_
    preds = model.predict(transformed)

    return BatchPrediction(
        predictions=preds.tolist(),
        probabilities=[{cls: float(prob) for cls, prob in zip(classes, row)} for row in probabilities],
    )
