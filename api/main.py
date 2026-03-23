"""
api/main.py
FastAPI backend serving fraud detection predictions + RAG financial Q&A.
Production-ready with proper error handling, logging, and health checks.
"""

import json
import joblib
import shap
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from loguru import logger
import yaml
import sys

sys.path.insert(0, ".")
from data.pipeline import preprocess
from rag.engine import get_engine

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="FinSight AI — Financial Risk Intelligence API",
    description="ML-powered fraud detection and RAG financial Q&A platform",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ───────────────────────────────────────────────────────────────
_model = None
_explainer = None
_metadata = None
_rag_engine = None
_config = None


def load_config():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


@app.on_event("startup")
async def startup():
    global _model, _explainer, _metadata, _rag_engine, _config
    _config = load_config()
    artifacts = "models/artifacts"

    try:
        _model = joblib.load(f"{artifacts}/model.pkl")
        _explainer = joblib.load(f"{artifacts}/shap_explainer.pkl")
        with open(f"{artifacts}/metadata.json") as f:
            _metadata = json.load(f)
        logger.info(f"Model loaded | Threshold: {_metadata['threshold']} | "
                    f"AUC: {_metadata['test_metrics']['roc_auc']}")
    except FileNotFoundError:
        logger.warning("Model not found — run `python models/train.py` first.")

    try:
        _rag_engine = get_engine(_config)
        logger.info("RAG engine loaded.")
    except Exception as e:
        logger.warning(f"RAG engine failed to load: {e}")


# ── Request/Response schemas ───────────────────────────────────────────────────
class TransactionRequest(BaseModel):
    TransactionAmt: float = Field(..., gt=0, example=250.00)
    ProductCD: str = Field(..., example="W")
    card_type: str = Field(..., example="credit")
    card_bank: str = Field(..., example="chase")
    device_type: str = Field(..., example="mobile")
    email_domain: str = Field(..., example="gmail")
    hour_of_day: int = Field(..., ge=0, le=23, example=2)
    addr_match: int = Field(..., ge=0, le=1, example=0)
    dist_km: float = Field(..., ge=0, example=450.0)
    tx_count_1d: int = Field(..., ge=0, example=7)
    tx_count_7d: int = Field(..., ge=0, example=22)
    velocity_score: float = Field(..., ge=0, le=1, example=0.87)


class PredictionResponse(BaseModel):
    transaction_id: Optional[str]
    fraud_probability: float
    is_fraud: bool
    risk_level: str
    threshold_used: float
    top_risk_factors: list
    model_version: str


class RAGRequest(BaseModel):
    question: str = Field(..., min_length=5, example="What are the key fraud risk indicators?")


class RAGResponse(BaseModel):
    question: str
    answer: str
    sources: list
    model_used: str


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"status": "operational", "service": "FinSight AI Risk Intelligence Platform"}


@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "healthy",
        "model_loaded": _model is not None,
        "rag_loaded": _rag_engine is not None,
        "model_version": _metadata["model_version"] if _metadata else None,
        "test_auc": _metadata["test_metrics"]["roc_auc"] if _metadata else None,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Fraud Detection"])
def predict(tx: TransactionRequest):
    if _model is None:
        raise HTTPException(503, "Model not loaded. Run training first.")

    # Build DataFrame
    row = pd.DataFrame([tx.dict()])
    row["isFraud"] = 0  # placeholder for preprocess schema

    try:
        X, _, _ = preprocess(row, fit=False, artifacts_path="data/processed/")
    except Exception as e:
        raise HTTPException(422, f"Preprocessing failed: {e}")

    # Predict
    proba = float(_model.predict_proba(X)[:, 1][0])
    threshold = _metadata["threshold"]
    is_fraud = proba >= threshold

    risk_level = (
        "CRITICAL" if proba >= 0.85 else
        "HIGH" if proba >= 0.65 else
        "MEDIUM" if proba >= 0.40 else
        "LOW"
    )

    # SHAP explanation
    top_factors = []
    try:
        sv = _explainer.shap_values(X)[0]
        feature_cols = _metadata["feature_cols"]
        shap_pairs = sorted(zip(feature_cols, sv), key=lambda x: abs(x[1]), reverse=True)
        top_factors = [
            {"feature": f, "shap_value": round(float(v), 4), "direction": "increases" if v > 0 else "decreases"}
            for f, v in shap_pairs[:5]
        ]
    except Exception as e:
        logger.warning(f"SHAP failed: {e}")

    return PredictionResponse(
        transaction_id=None,
        fraud_probability=round(proba, 4),
        is_fraud=is_fraud,
        risk_level=risk_level,
        threshold_used=threshold,
        top_risk_factors=top_factors,
        model_version=_metadata["model_version"],
    )


@app.post("/ask", response_model=RAGResponse, tags=["Financial Q&A"])
def ask(req: RAGRequest):
    if _rag_engine is None:
        raise HTTPException(503, "RAG engine not available.")
    result = _rag_engine.query(req.question)
    return RAGResponse(
        question=req.question,
        answer=result["answer"],
        sources=result["sources"],
        model_used=_config["rag"]["ollama_model"] if _config else "unknown",
    )


@app.get("/model/info", tags=["Model"])
def model_info():
    if _metadata is None:
        raise HTTPException(503, "Model not loaded.")
    return {
        "version": _metadata["model_version"],
        "run_id": _metadata["run_id"],
        "threshold": _metadata["threshold"],
        "test_metrics": _metadata["test_metrics"],
        "top_features": _metadata["top_features"],
    }


@app.post("/batch_predict", tags=["Fraud Detection"])
def batch_predict(transactions: list[TransactionRequest]):
    if _model is None:
        raise HTTPException(503, "Model not loaded.")
    if len(transactions) > 100:
        raise HTTPException(400, "Batch size limited to 100 transactions.")

    results = []
    for tx in transactions:
        try:
            row = pd.DataFrame([tx.dict()])
            row["isFraud"] = 0
            X, _, _ = preprocess(row, fit=False, artifacts_path="data/processed/")
            proba = float(_model.predict_proba(X)[:, 1][0])
            threshold = _metadata["threshold"]
            results.append({
                "fraud_probability": round(proba, 4),
                "is_fraud": proba >= threshold,
                "risk_level": (
                    "CRITICAL" if proba >= 0.85 else "HIGH" if proba >= 0.65
                    else "MEDIUM" if proba >= 0.40 else "LOW"
                )
            })
        except Exception as e:
            results.append({"error": str(e)})

    fraud_count = sum(1 for r in results if r.get("is_fraud"))
    return {
        "total": len(results),
        "fraud_detected": fraud_count,
        "fraud_rate": round(fraud_count / len(results), 4),
        "results": results,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
