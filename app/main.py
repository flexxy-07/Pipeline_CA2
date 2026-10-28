from pathlib import Path
from typing import Dict, List, Any
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model.joblib"
artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
feature_names = artifact["feature_names"]

app = FastAPI(
    title="Breast Cancer Classification API",
    description="Educational ML demo. Not for clinical use.",
    version="1.0.0",
)

class PredictionRequest(BaseModel):
    features: Dict[str, float] = Field(
        ..., description="Provide all 30 feature names and their numeric values."
    )

class PredictionResponse(BaseModel):
    prediction: str
    prediction_code: int
    probabilities: Dict[str, float]
    model_metrics: Dict[str, Any]

@app.get("/")
def root():
    return {"message": "Breast Cancer Classification API", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/features")
def get_features():
    return {"count": len(feature_names), "features": feature_names}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    missing = [name for name in feature_names if name not in request.features]
    extra = [name for name in request.features if name not in feature_names]
    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={"missing_features": missing, "unexpected_features": extra},
        )
    try:
        row = pd.DataFrame([[request.features[name] for name in feature_names]],
                           columns=feature_names)
        code = int(model.predict(row)[0])
        probabilities = model.predict_proba(row)[0]
        label = "malignant" if code == 0 else "benign"
        return PredictionResponse(
            prediction=label,
            prediction_code=code,
            probabilities={
                "malignant": round(float(probabilities[0]), 6),
                "benign": round(float(probabilities[1]), 6),
            },
            model_metrics=artifact["metrics"],
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc
