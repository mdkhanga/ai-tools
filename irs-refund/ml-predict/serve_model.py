#! /usr/bin/python3
import joblib
import xgboost as xgb
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Load trained model
model = joblib.load("refund_predictor.pkl")

app = FastAPI(title="Refund Prediction API")

# Request schema
class RefundFeatures(BaseModel):
    income: float
    dependents: int
    withholding: float
    filing_status: int

@app.post("/predict")
def predict_refund(features: RefundFeatures):
    data = pd.DataFrame([features.dict()])
    dmatrix = xgb.DMatrix(data)

    prediction = model.predict(dmatrix)[0]
    return {
        "predicted_refund_days": round(float(prediction), 2)
    }
