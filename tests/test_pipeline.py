"""
tests/test_pipeline.py
Unit tests for data pipeline, model inference, and API endpoints.
"""

import pytest
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, ".")


# ── Data pipeline tests ────────────────────────────────────────────────────────
class TestDataPipeline:
    def test_synthetic_generation(self):
        from data.pipeline import generate_synthetic_transactions
        df = generate_synthetic_transactions(n_samples=1000)
        assert len(df) == 1000
        assert "isFraud" in df.columns
        assert "TransactionAmt" in df.columns
        fraud_rate = df["isFraud"].mean()
        assert 0.01 < fraud_rate < 0.10, f"Unexpected fraud rate: {fraud_rate}"

    def test_class_imbalance(self):
        from data.pipeline import generate_synthetic_transactions
        df = generate_synthetic_transactions(n_samples=5000)
        fraud_count = df["isFraud"].sum()
        assert fraud_count > 0, "No fraud samples generated"
        assert fraud_count < len(df) * 0.1, "Fraud rate suspiciously high"

    def test_feature_engineering(self):
        from data.pipeline import generate_synthetic_transactions, preprocess
        import tempfile, os
        df = generate_synthetic_transactions(n_samples=500)
        with tempfile.TemporaryDirectory() as tmpdir:
            X, y, feature_cols = preprocess(df, fit=True, artifacts_path=tmpdir)
            assert X.shape[0] == 500
            assert len(feature_cols) > 0
            assert not X.isnull().any().any(), "NaN values in features"
            assert y is not None
            assert len(y) == 500

    def test_no_data_leakage(self):
        from data.pipeline import generate_synthetic_transactions, preprocess
        import tempfile
        df = generate_synthetic_transactions(n_samples=200)
        with tempfile.TemporaryDirectory() as tmpdir:
            X, y, feature_cols = preprocess(df, fit=True, artifacts_path=tmpdir)
            assert "isFraud" not in X.columns, "Target leaked into features"
            assert "TransactionID" not in X.columns, "ID column leaked into features"


# ── Model tests ────────────────────────────────────────────────────────────────
class TestModel:
    @pytest.fixture(scope="class")
    def trained_model(self, tmp_path_factory):
        from data.pipeline import generate_synthetic_transactions, preprocess
        from sklearn.model_selection import train_test_split
        from xgboost import XGBClassifier

        tmpdir = str(tmp_path_factory.mktemp("artifacts"))
        df = generate_synthetic_transactions(n_samples=2000)
        X, y, feature_cols = preprocess(df, fit=True, artifacts_path=tmpdir)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = XGBClassifier(n_estimators=50, max_depth=4, random_state=42, verbosity=0)
        model.fit(X_train, y_train)
        return model, X_test, y_test, feature_cols, tmpdir

    def test_model_predicts(self, trained_model):
        model, X_test, y_test, _, _ = trained_model
        proba = model.predict_proba(X_test)[:, 1]
        assert proba.shape[0] == len(X_test)
        assert all(0 <= p <= 1 for p in proba), "Probabilities out of [0,1]"

    def test_model_auc(self, trained_model):
        from sklearn.metrics import roc_auc_score
        model, X_test, y_test, _, _ = trained_model
        proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        assert auc > 0.75, f"AUC too low: {auc:.3f}"

    def test_shap_values(self, trained_model):
        import shap
        model, X_test, _, feature_cols, _ = trained_model
        explainer = shap.TreeExplainer(model)
        sample = X_test.head(10)
        sv = explainer.shap_values(sample)
        assert sv.shape == (len(sample), len(feature_cols))


# ── RAG tests ──────────────────────────────────────────────────────────────────
class TestRAG:
    def test_document_seeding(self, tmp_path):
        from rag.engine import seed_documents
        docs_path = str(tmp_path / "docs")
        seed_documents(docs_path)
        doc_files = list((tmp_path / "docs").glob("*.txt"))
        assert len(doc_files) >= 3, "Not enough seed documents created"

    def test_index_build(self, tmp_path):
        from rag.engine import RAGEngine, seed_documents
        docs_path = str(tmp_path / "docs")
        index_path = str(tmp_path / "index")
        seed_documents(docs_path)

        config = {
            "rag": {
                "ollama_model": "llama3",
                "embedding_model": "all-MiniLM-L6-v2",
                "faiss_index_path": index_path,
                "docs_path": docs_path,
                "chunk_size": 300,
                "chunk_overlap": 30,
                "top_k": 2,
            }
        }
        engine = RAGEngine(config)
        engine.build_index()
        assert engine.vectorstore is not None

    def test_retrieval(self, tmp_path):
        from rag.engine import RAGEngine, seed_documents
        docs_path = str(tmp_path / "docs")
        index_path = str(tmp_path / "index")
        seed_documents(docs_path)

        config = {
            "rag": {
                "ollama_model": "llama3",
                "embedding_model": "all-MiniLM-L6-v2",
                "faiss_index_path": index_path,
                "docs_path": docs_path,
                "chunk_size": 300,
                "chunk_overlap": 30,
                "top_k": 2,
            }
        }
        engine = RAGEngine(config)
        engine.build_index()
        result = engine.query("What are fraud indicators?")
        assert "answer" in result
        assert "sources" in result
        assert len(result["context"]) > 0


# ── Monitoring tests ───────────────────────────────────────────────────────────
class TestMonitoring:
    def test_drift_detection_no_drift(self, tmp_path):
        from monitoring.drift import detect_drift_statistical
        np.random.seed(42)
        ref = pd.DataFrame({"amt": np.random.normal(100, 20, 500),
                             "velocity": np.random.uniform(0, 1, 500)})
        cur = pd.DataFrame({"amt": np.random.normal(102, 21, 200),
                             "velocity": np.random.uniform(0.02, 0.98, 200)})
        result = detect_drift_statistical(ref, cur, threshold=0.1)
        assert "dataset_drift_detected" in result
        assert isinstance(result["dataset_drift_detected"], bool)

    def test_drift_detection_with_drift(self):
        from monitoring.drift import detect_drift_statistical
        np.random.seed(42)
        ref = pd.DataFrame({"amt": np.random.normal(100, 10, 500)})
        cur = pd.DataFrame({"amt": np.random.normal(300, 10, 500)})  # big shift
        result = detect_drift_statistical(ref, cur, threshold=0.1)
        assert result["dataset_drift_detected"] is True
