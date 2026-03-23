# 🏦 FinSight AI — Financial Risk Intelligence Platform

> End-to-end ML system for real-time fraud detection and AI-powered financial risk Q&A. Production-grade architecture from data ingestion through model deployment and drift monitoring.

---

## 📌 What This Is

FinSight AI is a production-ready ML platform built to solve two core financial risk problems:

1. **Real-Time Fraud Detection** — XGBoost model with SHAP explainability, optimized decision threshold, and sub-100ms inference via FastAPI
2. **Financial Risk Q&A** — RAG pipeline over internal financial documents using local LLM (Ollama/llama3) + FAISS vector search — zero external API dependency

Built to mirror real-world ML engineering standards: modular codebase, experiment tracking, drift monitoring, containerized deployment, and full test coverage.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FinSight AI                              │
├──────────────┬──────────────────┬──────────────────────────────┤
│  Data Layer  │   Model Layer    │      Serving Layer           │
│              │                  │                              │
│ • Synthetic  │ • XGBoost        │ • FastAPI REST API          │
│   transaction│ • SHAP explainer │ • Streamlit Dashboard        │
│   generation │ • MLflow tracking│ • Batch prediction endpoint  │
│ • Feature    │ • Threshold opt. │ • RAG Q&A endpoint          │
│   engineering│ • Model registry │                              │
├──────────────┴──────────────────┴──────────────────────────────┤
│                     Supporting Systems                          │
│  • FAISS vector store   • Ollama (local LLM)                   │
│  • Drift monitoring     • Docker Compose                        │
│  • pytest test suite    • MLflow experiment UI                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| ROC-AUC | ~0.97 |
| Average Precision | ~0.88 |
| F1 Score | ~0.81 |
| Precision | ~0.84 |
| Recall | ~0.79 |

> Trained on 50,000 synthetic transactions with 3.5% fraud rate — mirroring real-world class imbalance. Threshold optimized on validation set to maximize F1.

---

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|-------------|
| **ML** | XGBoost, Scikit-learn, SHAP, imbalanced-learn |
| **GenAI / RAG** | LangChain, FAISS, HuggingFace Embeddings, Ollama (llama3) |
| **MLOps** | MLflow (experiment tracking + registry), Evidently (drift) |
| **API** | FastAPI, Uvicorn, Pydantic |
| **Dashboard** | Streamlit, Plotly |
| **Infrastructure** | Docker, Docker Compose |
| **Testing** | pytest |
| **Languages** | Python 3.11 |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.ai) installed and running (for RAG)
- 4GB RAM minimum

### 1. Clone & Install
```bash
git clone https://github.com/YOUR_USERNAME/finsight-ai.git
cd finsight-ai
pip install -r requirements.txt
```

### 2. Pull LLM (for RAG)
```bash
ollama pull llama3
ollama serve
```

### 3. One-Command Setup (train + index + baseline)
```bash
python setup.py
```

### 4. Start Services
```bash
# API (port 8000)
uvicorn api.main:app --reload --port 8000

# Dashboard (port 8501)
streamlit run dashboard/app.py

# MLflow UI (port 5000)
mlflow ui --backend-store-uri mlruns
```

### Or with Docker
```bash
docker-compose up --build
```

---

## 📡 API Reference

Base URL: `http://localhost:8000`

### `POST /predict` — Fraud Detection
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionAmt": 850.00,
    "ProductCD": "C",
    "card_type": "credit",
    "card_bank": "other",
    "device_type": "mobile",
    "email_domain": "protonmail",
    "hour_of_day": 2,
    "addr_match": 0,
    "dist_km": 520.0,
    "tx_count_1d": 8,
    "tx_count_7d": 24,
    "velocity_score": 0.91
  }'
```

**Response:**
```json
{
  "fraud_probability": 0.8934,
  "is_fraud": true,
  "risk_level": "CRITICAL",
  "threshold_used": 0.45,
  "top_risk_factors": [
    {"feature": "velocity_score", "shap_value": 0.412, "direction": "increases"},
    {"feature": "hour_of_day", "shap_value": 0.287, "direction": "increases"},
    {"feature": "addr_match", "shap_value": 0.231, "direction": "increases"}
  ],
  "model_version": "1.0"
}
```

### `POST /ask` — Financial Risk Q&A
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the SAR filing thresholds under BSA?"}'
```

### `GET /health` — Health Check
```bash
curl http://localhost:8000/health
```

### `GET /model/info` — Model Metadata
```bash
curl http://localhost:8000/model/info
```

### `POST /batch_predict` — Batch Inference (up to 100 transactions)

---

## 📁 Project Structure

```
finsight-ai/
├── configs/
│   └── config.yaml              # Central config (model params, paths, thresholds)
├── data/
│   ├── pipeline.py              # Data generation, feature engineering, preprocessing
│   ├── raw/                     # Raw transaction data
│   └── processed/               # Processed features + preprocessing artifacts
├── models/
│   ├── train.py                 # XGBoost training + SHAP + MLflow
│   └── artifacts/               # Saved model, explainer, metadata
├── rag/
│   ├── engine.py                # LangChain + FAISS + Ollama RAG pipeline
│   ├── documents/               # Financial risk documents (knowledge base)
│   └── faiss_index/             # Persisted vector index
├── api/
│   └── main.py                  # FastAPI app (predict, ask, batch, health)
├── dashboard/
│   └── app.py                   # Streamlit dashboard
├── monitoring/
│   ├── drift.py                 # Evidently + KS test drift detection
│   └── reports/                 # Generated drift reports
├── tests/
│   └── test_pipeline.py         # pytest unit tests
├── mlruns/                      # MLflow experiment artifacts
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── setup.py                     # One-command initialization
```

---

## 🔍 Key Design Decisions

**Why XGBoost over deep learning?**
Tree-based models are the industry standard for tabular fraud detection — interpretable, fast inference, and SHAP-compatible. Deep learning offers marginal lift at the cost of explainability, which regulators require.

**Why local Ollama instead of OpenAI API?**
Zero cost, zero data privacy risk, fully reproducible. Financial institutions cannot send customer data to third-party APIs. Local LLM deployment is the production-realistic approach.

**Why FAISS over Pinecone/Weaviate?**
For a self-contained portfolio project, FAISS provides the same semantic search capability without external services. Production migration to managed vector DB is a configuration change.

**Why threshold optimization?**
Default 0.5 threshold ignores the business cost asymmetry in fraud detection — missing fraud is far more costly than a false positive. Validation-set F1 optimization makes this a business decision, not an arbitrary cutoff.

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

```
tests/test_pipeline.py::TestDataPipeline::test_synthetic_generation PASSED
tests/test_pipeline.py::TestDataPipeline::test_class_imbalance PASSED
tests/test_pipeline.py::TestDataPipeline::test_feature_engineering PASSED
tests/test_pipeline.py::TestDataPipeline::test_no_data_leakage PASSED
tests/test_pipeline.py::TestModel::test_model_predicts PASSED
tests/test_pipeline.py::TestModel::test_model_auc PASSED
tests/test_pipeline.py::TestModel::test_shap_values PASSED
tests/test_pipeline.py::TestRAG::test_document_seeding PASSED
tests/test_pipeline.py::TestRAG::test_index_build PASSED
tests/test_pipeline.py::TestRAG::test_retrieval PASSED
tests/test_pipeline.py::TestMonitoring::test_drift_detection_no_drift PASSED
tests/test_pipeline.py::TestMonitoring::test_drift_detection_with_drift PASSED
```

---

## 📈 MLflow Experiment Tracking

MLflow tracks every training run: hyperparameters, metrics, SHAP plots, and model artifacts.

```bash
mlflow ui --backend-store-uri mlruns
# Open: http://localhost:5000
```

---

## 🔄 Drift Monitoring

Run drift detection against the training reference distribution:

```bash
python monitoring/drift.py
```

Generates a JSON summary and (if Evidently installed) an interactive HTML report showing feature-level drift statistics with KS test p-values.

---

## 📄 License

MIT License — free to use, modify, and distribute.
